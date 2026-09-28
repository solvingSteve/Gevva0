# Gevva0: Calibrated Local Decision Gateway

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![llama.cpp](https://img.shields.io/badge/Backend-llama.cpp%20CUDA-green.svg)](https://github.com/ggerganov/llama.cpp)
[![JevBench](https://img.shields.io/badge/JevBench%20Rank-%231%20Global%20(74.63)-00f2fe)](docs/JEVBENCH_PUBLICATION_EVALUATION_26B.md)
[![ECE < 0.03](https://img.shields.io/badge/Calibration-ECE%20%3C%200.03-success.svg)]()

A high-speed, local decision gateway running on **Gemma 4** (`llama.cpp` / CUDA) that replaces flaky autoregressive JSON generation with **direct token logit scoring**, **mathematical cyclic debiasing**, **asymmetric post-decision quotation**, and **Platt temperature calibration**.

Get deterministic classifications, well-calibrated confidence scores, and verbatim audit trails in **15–45ms fast-path classification** on consumer hardware.

https://github.com/user-attachments/assets/a654e5ff-d425-4299-b5a1-9a3a1399d666

---

## Why Gevva0?

Autoregressive LLM classification pipelines suffer from four critical failure modes:
- **Logit Poisoning & Pre-Decisional Drift**: Generating chain-of-thought (CoT) before categorical output allows early hallucinated tokens to corrupt the final decision probability.
- **Positional Bias**: Standard single-shot logit scoring shows steep label favoritism (e.g., models picking "A" ~70% of the time regardless of option content).
- **Severe Overconfidence (Poor Calibration)**: Raw softmax distributions yield high confidence (95%+) on wrong answers, making automated routing thresholds risky.
- **Latency Tax**: Autoregressive JSON schemas require 100–300 generated tokens, adding 500–2,500ms of latency per request.

```
Standard Autoregressive Classification (500–2,500ms):
Prompt ──> Generated CoT / JSON ──> [Logit Poisoning Risk] ──> Brittle Output

Gevva0 Calibrated Dual-Path (15–45ms):
Prompt ──> Direct Logit Readout ──> Platt Scaled ──> Confidence >= Threshold?
                                                            │
                                            ├── YES ──> Fast-Path Locked Verdict (15-25ms)
                                            └── NO  ──> Bounded Verification (30-85ms) ─┘
                                                            │
                                                  [Verdict Permanently Locked]
                                                            │
                                                            ▼ (Isolated Forward Pass)
                                                  Asymmetric Grounded Evidence Extraction
```

---

## Official JevBench Leaderboard — Global Rank #1 🏆

Evaluated across easy, original, and hard forensic legal batteries from the official JevBench v1.4.2 benchmark suite.

| Rank | System Architecture | JevBench Score | Intelligence | Hard Tier (Forensic) | p50 Latency |
| :---: | :--- | :---: | :---: | :---: | :---: |
| ⭐ **#1** | **Gevva0 (Gemma 4 26B-A4B MoE)** | **74.63** | **87.2** | **82.9%** | **214 ms** |
| #2 | decider-4b v2 (Mapika) | 64.13 | 49.4 | 41.4% | 118 ms |
| #3 | TypeSafe Jev 1.13.0 (Closed API) | 63.29 | 53.1 | 47.7% | 652 ms |
| #4 | JevK5 v0.2.0 | 62.04 | 48.9 | 42.3% | 122 ms |
| #5 | Cygnet (Frozen Gemma-4-12B) | 61.76 | 49.5 | 43.2% | 126 ms |

> See full evaluation report: [`docs/JEVBENCH_PUBLICATION_EVALUATION_26B.md`](docs/JEVBENCH_PUBLICATION_EVALUATION_26B.md)

### Secondary Ablation Study (N = 650, Gemma 4 26B-A4B MoE)

All local conditions were evaluated under strictly isolated weights (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, 14 GB) across 650 balanced multi-class scenarios (including 18.5% out-of-distribution distractor controls).

| Metric | Baseline (Naive Logits) | Gevva0 (Fast-Path) | Gevva0 (Adaptive CoT) | TypeSafe Jev (Cloud API) |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 76.0% [72.8, 79.2] | 89.1% [86.6, 91.4] | **89.8% [87.5, 92.2]** | 88.3% [85.8, 90.6] |
| **ECE (10 Bins, Calibration)** | 0.186 | 0.064 | **0.027** | 0.043 |
| **Brier Score (Lower=Better)** | 0.452 | 0.211 | **0.189** | 0.217 |
| **Label Invariance (Debiased)** | 71.5% | **100.0%** | **100.0%** | 90.3% |
| **OOD False-Positive Rate** | 22.5% | 5.0% | **3.3%** | 8.3% |
| **Latency (p50 / p95)** | **18ms / 23ms** | 22ms / 28ms | 45ms / 86ms | 112ms / 221ms |
| **Significance (McNemar)** | — | $p < 0.0001$ | $p < 0.0001$ | $p = 0.4152$ (par) |

> See full evaluation report: Gemma 4 26B-A4B (MoE): [`BENCHMARK_REPORT_gemma-4-26B-A4B.md`](docs/BENCHMARK_REPORT_gemma-4-26B-A4B.md)

---

## Quickstart

### 1. Installation
Requires Python 3.12 and an NVIDIA GPU (CUDA 12+ / 13+). Prebuilt wheels are bundled via `llama-cpp-python`:

```bash
# Using uv (Recommended)
uv sync

# Or using standard pip
pip install -r requirements.txt
```

Verify your GPU environment and model loading:
```bash
uv run gevva0 check
```

### 2. Python API

```python
from gevva0 import GevvaEngine

# Initialize the engine (auto-discovers model from llm_config.json)
engine = GevvaEngine()

context = "Customer reports unauthorized double charge on order #89211 after payment timeout."
options = [
    "Billing Dispute / Refund",
    "Technical Bug / Gateway Timeout",
    "Account Security Incident",
    "General Inquiry"
]

# Run decision with cyclic debiasing and confidence calibration
result = engine.decide(
    context=context,
    options=options,
    confidence_threshold=0.85,
    cyclic_debias=True
)

print(f"Verdict: {result.selected_option}")
print(f"Confidence: {result.calibrated_p:.4f}")
print(f"Path Taken: {result.path}")  # 'fast_path' or 'adaptive_cot'

# Asymmetric post-verdict quote extraction (zero logit poisoning)
audit = engine.extract_audit(context=context, locked_choice=result.selected_option)
print(f"Grounding Quote: '{audit.verbatim_quote}'")
```

### 3. CLI & Server

```powershell
# Fast-path decision outputting structured JSON
uv run gevva0 decide `
  --context "Customer reports their invoice was charged twice for the same month." `
  --options "A: Billing Inquiry,B: Technical Bug,C: Churn Risk" `
  --json

# Force CoT scratchpad via high confidence gate threshold
uv run gevva0 decide --context "..." --options "A: x,B: y" --threshold 0.99

# Full cyclic label-permutation debiasing (invariance guarantee)
uv run gevva0 decide --context "..." --options "A: x,B: y,C: z" --cyclic

# Launch API service & Web Dashboard UI (http://localhost:8000/ui)
uv run gevva0 serve --host 127.0.0.1 --port 8000
```

---

## Key Architectural Principles

<details>
<summary><b>Click to expand Architecture & Theoretical Foundations</b></summary>

- **Direct Token Logit Scoring & Prefix Marginalization**: Evaluates candidate token log-probabilities directly from `llm.scores[llm.n_tokens - 1]`. Marginalizes over bare and space-prefixed tokens (`logsumexp(logit(" A"), logit("A"))`) to capture true prior distributions without running an autoregressive decoding loop.
- **Bounded System 2 Verification**: If the fast-path calibrated confidence score is below the threshold, the gateway executes a micro-scratchpad (up to 40 tokens at $T=0.2$) focused purely on option exclusion, then immediately rescores the final choice.
- **Guaranteed Label Invariance (Cyclic Debiasing)**: Evaluates cyclic option permutations ($A \to B \to C \to D \to A$) and projects probability mass back to semantic labels, eliminating positional favoritism.
- **Platt-Style Temperature Calibration**: Minimizes multi-class Brier score over validation scenarios to produce a learned temperature parameter ($T$), aligning raw logit softmaxes with true empirical accuracy ($ECE < 0.03$).
- **Asymmetric Audit Trail (Post-Decision Extraction)**: Once the verdict is locked, an isolated, secondary forward pass extracts verbatim quotes and rationales from the source context. The rationale cannot retroactively corrupt or poison the classification decision.

> For mathematical proofs, multi-class Brier score Murphy decompositions, and VRAM sizing charts, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

</details>

<details>
<summary><b>Click to expand Repository Layout & Configuration</b></summary>

### Repository Layout

```
├── src/
│   └── gevva0/               # Core Python package
│       ├── engine.py         # llama.cpp logit extraction, KV caching, calibration
│       ├── debias.py         # Cyclic permutation debiasing
│       ├── audit.py          # Asymmetric quote & rationale extractor
│       ├── calibration.py    # Platt temperature calibration & Brier fitting
│       ├── metrics.py        # Bootstrap CIs, McNemar tests, ECE, Brier decomposition
│       ├── config.py         # Model path, context size, and auto-discovery
│       ├── schema.py         # Pydantic request / response schemas
│       ├── server.py         # FastAPI service (serves /ui and API routes)
│       └── cli.py            # CLI entrypoint ('gevva0')
├── ui/
│   └── web_dashboard/        # Interactive dashboard & audit UI (mounted at / and /ui/)
│       └── index.html        # Single source of truth web interface
├── benchmarks/
│   ├── benchmark_suite_650.json    # Rigorous N=650 4-class balanced evaluation battery
│   ├── benchmark_suite.json        # Standard benchmark suite
│   ├── generate_rigorous_dataset.py# Generator for N=650 dataset with 18.5% OOD controls
│   ├── run_benchmark.py            # Standardized CLI runner with 4 ablation lines
│   ├── generate_fixtures.py        # Generates multimodal test images
│   └── fixtures/                   # Test image assets (AP invoices, 404 UI, CCTV frames)
├── tests/
│   ├── tests.json            # Curated interactive showcase tests for Web UI
│   ├── test_metrics.py       # Unit tests for statistical & calibration formulas
│   └── test_benchmark_runner.py # Integration tests for benchmark runner
├── docs/
│   ├── ARCHITECTURE.md       # In-depth technical write-up on logit scoring & calibration
│   ├── BENCHMARK_REPORT_gemma-4-26B-A4B.md
│   └── JEVBENCH_PUBLICATION_EVALUATION_26B.md
├── models/
│   └── gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/
│       ├── gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
│       └── mmproj-BF16.gguf
├── requirements.txt          # Autogenerated requirements for pip / non-uv environments
├── pyproject.toml            # Project dependencies managed via uv
└── README.md
```

### Configuration (`llm_config.json`)

```json
{
  "n_ctx": 4096,
  "n_batch": 2048,
  "n_seq_max": 8,
  "kv_unified": true,
  "kv_cache": "F16",
  "type_k": 1,
  "type_v": 1,
  "model": {
    "model_path": "models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "mmproj_path": "models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf",
    "n_gpu_layers": -1,
    "verbose": false
  },
  "decision": {
    "cot_threshold": 0.85,
    "cot_max_tokens": 256,
    "cot_temp": 0.0,
    "cot_prompt": "Analysis: First calculate everything: ",
    "cyclic_debias": true,
    "kv_branching": true,
    "kv_branching_min_tokens": 50
  }
}
```

Overrides can also be set via environment variables: `GEVVA0_MODEL`, `GEVVA0_N_CTX`, and `GEVVA0_CALIBRATION`.

</details>

---

## Reproduce the Benchmarks

The benchmark suite includes an automated runner with $B=10,000$ bootstrap resampling and McNemar continuity-corrected significance testing:

```powershell
# Run benchmark on active model
uv run python benchmarks/run_benchmark.py

# Run on specific parameter scale
uv run python benchmarks/run_benchmark.py --model 26b   # Gemma 4 26B-A4B MoE
uv run python benchmarks/run_benchmark.py --model e4b   # Gemma 4 E4B Dense
uv run python benchmarks/run_benchmark.py --model e2b   # Gemma 4 E2B Edge

# Re-generate synthetic N=650 balanced evaluation battery with OOD controls
uv run python benchmarks/generate_rigorous_dataset.py
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
