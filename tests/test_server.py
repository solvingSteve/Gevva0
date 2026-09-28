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
    assert "First calculate" in data["cot_prompt"]
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


