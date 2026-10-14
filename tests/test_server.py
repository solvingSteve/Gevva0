from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from gevva0.server import build_app


def test_server_endpoints_and_switch_config():
    mock_engine = MagicMock()
    mock_engine.model_path = "models/gemma-4-E2B-it-qat-UD-Q4_K_XL/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
    mock_engine.n_ctx = 8192
    mock_engine.has_vision = True
    mock_engine.mmproj_path = "models/gemma-4-E2B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf"
    mock_engine.mmproj_variant = "BF16"
    mock_engine.calibrator = None

    app = build_app(mock_engine)
    client = TestClient(app)

    # Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    # Models list
    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    assert "models" in models_resp.json()

    # Model switch with resource cleanup
    new_mock_engine = MagicMock()
    new_mock_engine.model_path = "models/gemma-4-E4B-it-qat-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
    new_mock_engine.n_ctx = 4096

    from gevva0.config import _LLM_CONFIG_PATH

    orig_config = _LLM_CONFIG_PATH.read_text(encoding="utf-8") if _LLM_CONFIG_PATH.is_file() else None
    try:
        with patch("gevva0.server.GemmaDecisionEngine.from_config", return_value=new_mock_engine):
            resp = client.post(
                "/v1/config/switch",
                json={
                    "model_path": "models/gemma-4-E4B-it-qat-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
                    "n_ctx": 4096,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert mock_engine.close.called
            assert app.state.engine == new_mock_engine
    finally:
        if orig_config is not None:
            _LLM_CONFIG_PATH.write_text(orig_config, encoding="utf-8")


def test_server_decide_endpoint_options_and_images():
    mock_engine = MagicMock()
    mock_engine.decide.return_value = {
        "decision": "A",
        "decision_text": "Option 1",
        "confidence": 0.95,
        "mode": "fast_path",
        "probabilities": {"Option 1": 0.95, "Option 2": 0.05},
        "escalate": False,
    }

    app = build_app(mock_engine)
    client = TestClient(app)

    # 1. Dict options
    r1 = client.post("/v1/decide", json={"context": "Context 1", "options": {"A": "Option 1", "B": "Option 2"}})
    assert r1.status_code == 200
    assert r1.json()["decision"] == "A"

    # 2. List options
    r2 = client.post("/v1/decide", json={"context": "Context 2", "options": ["Option 1", "Option 2"]})
    assert r2.status_code == 200

    # 3. image_path and image_bytes payloads
    r3 = client.post(
        "/v1/decide",
        json={"context": "Context 3", "options": ["Option 1", "Option 2"], "image_path": "test.png"},
    )
    assert r3.status_code == 200


def test_server_config_defaults_and_cot_prompt():
    mock_engine = MagicMock()
    app = build_app(mock_engine)
    client = TestClient(app)

    # Defaults check
    resp = client.get("/v1/config/defaults")
    assert resp.status_code == 200
    data = resp.json()
    assert data["kv_cache"] == "F16"
    assert "cot_prompt" in data
    assert any(k in data["cot_prompt"] for k in ["First calculate", "Analysis:", "Verify each option"])
    assert data["cot_temp"] == 0.0
    assert data["max_cot_tokens"] == 256

    # Decide with cot_prompt
    mock_engine.decide.return_value = {
        "decision": "A",
        "decision_text": "Approved",
        "confidence": 0.96,
        "mode": "cot_verified",
        "probabilities": {"Approved": 0.96, "Rejected": 0.04},
        "cot_scratchpad": "Calculated sliding window cutoff timestamp: 14:15:00",
    }
    custom_prompt = "Analysis: Custom calculation prefix: "
    r = client.post(
        "/v1/decide",
        json={
            "context": "Rate limiter test",
            "options": {"A": "Approved", "B": "Rejected"},
            "cot_prompt": custom_prompt,
        },
    )
    assert r.status_code == 200
    assert mock_engine.decide.call_args[1]["cot_prompt"] == custom_prompt


def test_server_chat_completions_and_generate():
    mock_engine = MagicMock()
    mock_engine.model_path = "models/gemma-4-E2B-it-qat-UD-Q4_K_XL/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
    mock_engine.chat_completion.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567,
        "model": "gemma-4-26B-A4B-it",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"summary": "test response"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }

    def mock_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "Hello"}}]}
        yield {"choices": [{"delta": {"content": " world"}}]}

    mock_engine.stream_chat_completion = mock_stream

    app = build_app(mock_engine)
    client = TestClient(app)

    # 1. Non-streaming chat completions
    r1 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is logit scoring?"}],
            "temperature": 0.5,
            "max_tokens": 256,
        },
    )
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["choices"][0]["message"]["content"] == '{"summary": "test response"}'
    assert mock_engine.chat_completion.called

    # 2. Streaming chat completions
    r2 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Stream me an answer"}],
            "stream": True,
        },
    )
    assert r2.status_code == 200
    assert "text/event-stream" in r2.headers["content-type"]
    assert "Hello" in r2.text
    assert "[DONE]" in r2.text

    # 3. Simple generate endpoint
    r3 = client.post(
        "/api/generate",
        json={
            "prompt": "Extract JSON",
            "json_mode": True,
        },
    )
    assert r3.status_code == 200
    assert r3.json()["text"] == '{"summary": "test response"}'

    # 4. UI chat route aliases
    resp_chat = client.get("/ui/chat.html")
    assert resp_chat.status_code == 200

    resp_web_dash = client.get("/ui/web_dashboard/chat.html")
    assert resp_web_dash.status_code == 200

    resp_redir = client.get("/chat", follow_redirects=False)
    assert resp_redir.status_code in (302, 307)
    assert resp_redir.headers["location"] == "/ui/chat.html"


def test_server_no_model_loaded_behavior():
    app = build_app(None)
    client = TestClient(app)

    # Health check shows no model loaded
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "no_model_loaded"
    assert health_resp.json()["model"] is None

    # Models list shows active_model as None
    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    assert models_resp.json()["active_model"] is None

    # Running decide without model loaded returns 400 with load model message
    decide_resp = client.post(
        "/v1/decide",
        json={"context": "Hello", "options": {"A": "Option 1", "B": "Option 2"}},
    )
    assert decide_resp.status_code == 400
    assert "No model loaded" in decide_resp.json()["detail"]

    # Running chat completions without model loaded returns 400
    chat_resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert chat_resp.status_code == 400
    assert "No model loaded" in chat_resp.json()["detail"]

    # Running generate without model loaded returns 400
    gen_resp = client.post(
        "/api/generate",
        json={"prompt": "Hello"},
    )
    assert gen_resp.status_code == 400
    assert "No model loaded" in gen_resp.json()["detail"]


def test_server_switch_config_vision_checkbox():
    app = build_app(None)
    client = TestClient(app)

    mock_engine = MagicMock()
    mock_engine.model_path = "models/Qwen3.5-4B-Instruct-Pure-GGUF/Qwen3.5-4B-q8_0.gguf"
    mock_engine.n_ctx = 8192
    mock_engine.has_vision = False

    from gevva0.config import _LLM_CONFIG_PATH

    orig_config = _LLM_CONFIG_PATH.read_text(encoding="utf-8") if _LLM_CONFIG_PATH.is_file() else None
    try:
        with patch("gevva0.server.GemmaDecisionEngine.from_config", return_value=mock_engine) as mock_from_config:
            # Switch with vision=False: mmproj_path should be "none"
            resp = client.post(
                "/v1/config/switch",
                json={
                    "model_path": "models/Qwen3.5-4B-Instruct-Pure-GGUF/Qwen3.5-4B-q8_0.gguf",
                    "n_ctx": 8192,
                    "vision": False,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            mock_from_config.assert_called_with(
                model_path=mock_from_config.call_args[1]["model_path"],
                mmproj_path="none",
            )
    finally:
        if orig_config is not None:
            _LLM_CONFIG_PATH.write_text(orig_config, encoding="utf-8")


def test_align_verbatim_quote_removes_hallucinated_comma():
    from gevva0.audit import align_verbatim_quote

    context = (
        "Procurement Standard 4.1: Non-capital software purchase exceeding $1,500 requires formal VP signature and compliance review. "
        "Addendum 9 (Cloud & Tooling Rider): Cloud developer tooling and observability subscriptions approved under annual platform engineering "
        "budget are pre-authorized up to $10,000 and require only Engineering Manager sign-off. Scenario: Lead requests approval for an annual "
        "$4,200 developer profiling subscription fully allocated under platform budget."
    )
    # The hallucinated quote with extra comma $1,0,0,000
    hallucinated_quote = (
        "Cloud developer tooling and observability subscriptions approved under annual platform engineering "
        "budget are pre-authorized up to $1,0,0,000 and require only Engineering Manager sign-off."
    )
    expected_true_quote = (
        "Cloud developer tooling and observability subscriptions approved under annual platform engineering "
        "budget are pre-authorized up to $10,000 and require only Engineering Manager sign-off."
    )

    aligned = align_verbatim_quote(context, hallucinated_quote)
    assert aligned == expected_true_quote
    assert "$10,000" in aligned
    assert "$1,0,0,000" not in aligned


def test_model_agnostic_wire_format_response():
    mock_engine = MagicMock()
    mock_engine.model_path = "models/Qwen3.5-4B-Instruct-Pure-GGUF/Qwen3.5-4B-q8_0.gguf"
    mock_engine.model_name = "Qwen3.5-4B-q8_0.gguf"
    mock_engine.decide.return_value = {
        "candidate_letter": "A",
        "letter_probabilities": {"A": 0.9, "B": 0.1},
        "latency_ms": 12.5,
    }

    app = build_app(mock_engine)
    client = TestClient(app)

    resp = client.post(
        "/decide",
        json={
            "state": "State description",
            "questions": {
                "q1": {
                    "type": "choice",
                    "options": {"A": "First option", "B": "Second option"},
                }
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "Qwen3.5-4B-q8_0.gguf"
    assert "gemma-4-26B-A4B-it" not in data["model"]


def test_streaming_abort_does_not_deadlock_decide():
    from gevva0.engine import ReentrantCrossThreadLock

    mock_engine = MagicMock()
    mock_engine.model_path = "models/Qwen3.5-4B-Instruct-Pure-GGUF/Qwen3.5-4B-q8_0.gguf"
    mock_engine.model_name = "Qwen3.5-4B-q8_0.gguf"
    lock = ReentrantCrossThreadLock()
    mock_engine._lock = lock

    def fake_stream(*args, **kwargs):
        lock.acquire()
        try:
            yield {"choices": [{"delta": {"content": "chunk1"}}]}
            yield {"choices": [{"delta": {"content": "chunk2"}}]}
        finally:
            lock.release()

    mock_engine.stream_chat_completion = fake_stream
    mock_engine.decide.return_value = {
        "candidate_letter": "C",
        "decision": "C",
        "decision_text": "Correct Option",
        "confidence": 0.98,
        "mode": "fast_path",
        "latency_ms": 15.0,
    }

    app = build_app(mock_engine)
    client = TestClient(app)

    # 1. Start streaming, read 1 chunk, then abort/close connection
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        for line in response.iter_lines():
            if line:
                break
        # Exiting 'with' block aborts the stream mid-flight

    # 2. Immediately execute decide endpoint - must NOT hang or deadlock
    decide_resp = client.post(
        "/v1/decide",
        json={"context": "Some context", "options": {"A": "First", "B": "Second", "C": "Third"}},
    )
    assert decide_resp.status_code == 200
    assert decide_resp.json()["decision"] == "C"



