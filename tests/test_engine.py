import numpy as np
import pytest
from pathlib import Path

from gevva0.config import resolve_model_path, get_smallest_model
from gevva0.engine import GemmaDecisionEngine, debiased_cyclic_evaluate


def test_prompt_rendering():
    # Test that prompt rendering adheres to ARCHITECTURE.md with turn tokens and thought channel bypass
    engine = GemmaDecisionEngine.__new__(GemmaDecisionEngine)
    context = "System alert: Disk usage at 100% on primary DB."
    options = {"A": "Storage exhaustion", "B": "Network failure"}

    prompt = engine._render_prompt(context, options)
    assert "<|turn>user\n" in prompt
    assert "<turn|>\n<|turn>model\n" in prompt
    assert "<|channel>thought\n<channel|>" in prompt
    assert "A. Storage exhaustion" in prompt
    assert "B. Network failure" in prompt


def test_smallest_model_decision():
    smallest = get_smallest_model()
    if not smallest:
        pytest.skip("No model file available in models/")

    engine = GemmaDecisionEngine(model_path=smallest["full_path"], n_ctx=2048, verbose=False)

    context = (
        "Procurement Standard 4.1: Non-capital software purchase exceeding $1,500 requires formal VP signature. "
        "Addendum 9 (Cloud & Tooling Rider): Cloud developer tooling subscriptions approved under platform engineering "
        "budget are pre-authorized up to $10,000 and require only Engineering Manager sign-off. "
        "Scenario: Lead requests approval for an annual $4,200 developer profiling subscription fully allocated under platform budget."
    )
    options = {
        "A": "Formal VP signature and secondary compliance review",
        "B": "Immediate escalation to CFO and board approval",
        "C": "Engineering Manager approval only under Addendum 9",
        "D": "Automatic rejection as unauthorized capital expenditure",
    }

    res = engine.decide(context=context, options=options)
    assert res.decision in options
    assert res.confidence > 0.50
    assert "probabilities" in res.__dict__ or hasattr(res, "probabilities")


def test_subfolder_model_and_mmproj_resolution():
    from gevva0.config import resolve_mmproj_path, is_mmproj_file, detect_mmproj_variant

    e2b_folder = "models/gemma-4-E2B-it-qat-UD-Q4_K_XL"
    model_path = resolve_model_path(e2b_folder)
    assert model_path.is_file()
    assert not is_mmproj_file(model_path)

    mmproj = resolve_mmproj_path(model_path)
    assert mmproj is not None
    assert mmproj.is_file()
    assert is_mmproj_file(mmproj)
    assert mmproj.parent == model_path.parent
    assert detect_mmproj_variant(mmproj) == "BF16"


def test_mmproj_variant_detection():
    from gevva0.config import detect_mmproj_variant

    assert detect_mmproj_variant("mmproj-BF16.gguf") == "BF16"
    assert detect_mmproj_variant("models/test/mmproj-F16.gguf") == "F16"
    assert detect_mmproj_variant("mmproj-F32.gguf") == "F32"
    assert detect_mmproj_variant("clip-bf16.gguf") == "BF16"
    assert detect_mmproj_variant("mmproj-Q4_0.gguf") == "Q4_0"
    assert detect_mmproj_variant("mmproj-model-f16.gguf") == "F16"


def test_multiple_mmproj_selects_newest(tmp_path, monkeypatch):
    import time
    from gevva0.config import resolve_mmproj_path

    # Create dummy model and two mmproj files with different timestamps
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    model_file = model_dir / "test-model.gguf"
    model_file.write_text("model content")

    older_mmproj = model_dir / "mmproj-F16.gguf"
    older_mmproj.write_text("older mmproj")

    # Ensure a measurable difference in timestamps
    time.sleep(0.05)
    newer_mmproj = model_dir / "mmproj-BF16.gguf"
    newer_mmproj.write_text("newer mmproj")

    resolved = resolve_mmproj_path(model_file)
    assert resolved is not None
    assert resolved.name == "mmproj-BF16.gguf"


def test_kv_branching_min_tokens_guard():
    smallest = get_smallest_model()
    if not smallest:
        pytest.skip("No model file available in models/")

    with GemmaDecisionEngine(model_path=smallest["full_path"], n_ctx=1024, verbose=False) as engine:
        context = "Short context."
        options = {"A": "Option 1", "B": "Option 2"}

        # Context has ~10 tokens, min_tokens is 1500 -> guard keeps it on clean-reset mode
        res = engine.decide(
            context=context,
            options=options,
            cyclic_debias=True,
            kv_branching=True,
            kv_branching_min_tokens=1500,
        )
        assert res.decision in options
        assert res.confidence > 0.0


def test_score_cyclic_branching_direct():
    smallest = get_smallest_model()
    if not smallest:
        pytest.skip("No model file available in models/")

    with GemmaDecisionEngine(model_path=smallest["full_path"], n_ctx=1024, verbose=False) as engine:
        context = "Procurement rule: purchases over $1000 require VP sign-off."
        options = {"A": "VP sign-off", "B": "Manager sign-off"}
        prefix = engine._render_prefix(context)
        prefix_tokens = engine.llm.tokenize(prefix.encode("utf-8"), add_bos=True, special=True)

        suffixes = []
        for shift in range(2):
            texts = ["VP sign-off", "Manager sign-off"]
            perm = texts[shift:] + texts[:shift]
            d = {"A": perm[0], "B": perm[1]}
            s = engine._render_suffix(d)
            st = engine.llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
            suffixes.append(st)

        # Test return_probs=False (list of branch dicts)
        branch_results = engine.score_cyclic_branching(
            prefix_tokens=prefix_tokens,
            option_suffixes=suffixes,
            labels=["A", "B"],
            return_probs=False,
        )
        assert len(branch_results) == 2
        assert "probabilities" in branch_results[0]
        assert "A" in branch_results[0]["probabilities"]

        # Test return_probs=True (marginalized numpy array)
        probs = engine.score_cyclic_branching(
            prefix_tokens=prefix_tokens,
            option_suffixes=suffixes,
            labels=["A", "B"],
            return_probs=True,
        )
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2
        assert pytest.approx(float(np.sum(probs)), abs=1e-4) == 1.0


def test_default_fp16_kv_cache_and_cot_prompt():
    from gevva0.config import resolve_llm_kwargs, resolve_cot_prompt, resolve_decision_kwargs

    llm_kwargs = resolve_llm_kwargs()
    assert llm_kwargs["type_k"] == 1  # Native FP16 (GGML_TYPE_F16 = 1)
    assert llm_kwargs["type_v"] == 1  # Native FP16 (GGML_TYPE_F16 = 1)

    cot_prompt = resolve_cot_prompt()
    assert "First calculate" in cot_prompt
    assert "sliding window" in cot_prompt

    dec_kwargs = resolve_decision_kwargs()
    assert dec_kwargs["cot_prompt"] == cot_prompt
    assert dec_kwargs["cot_temp"] == 0.0
    assert dec_kwargs["cot_max_tokens"] == 256


def test_sliding_window_cot_verification():
    import json
    bench_suite = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_suite.json"
    if not bench_suite.is_file():
        pytest.skip("benchmark_suite.json not found")

    suite_data = json.loads(bench_suite.read_text(encoding="utf-8"))
    t = suite_data["tests"][1]  # test_02_strict_temporal_sliding_window
    assert t["test_id"] == "test_02_strict_temporal_sliding_window"

    engine = GemmaDecisionEngine.from_config()
    res = engine.decide(
        context=t["context"] + "\n\nQuestion: " + t["question"],
        options=t["options"],
        cot_threshold=1.0,  # force CoT verification
        max_cot_tokens=256,
        cot_temp=0.0,
    )
    assert res.decision == "A"
    assert res.mode == "cot_verified"
    assert res.cot_scratchpad is not None
    assert "14:15:00" in res.cot_scratchpad or "725" in res.cot_scratchpad or "690" in res.cot_scratchpad


