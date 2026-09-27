# Architecture & Methodology: Gevva0 Local Calibrated Decision Gateway

## Executive Summary

Gevva0 (**Project Gevva0**) is a high-speed local decision engine built on the **Gemma 4 26B-A4B MoE** architecture (and its dense/edge variants E4B and E2B) using `llama.cpp` and CUDA. It replaces brittle, high-latency autoregressive JSON generation with **direct token logit scoring**, **asymmetric audit trails**, **cyclic debiasing**, and **Platt-style temperature calibration**.

---

## 1. Direct Logit Scoring vs. Autoregressive Hazards

Traditional LLM classification pipelines suffer from several structural failure modes:
1. **Autoregressive Drift & Format Non-compliance**: Forcing a model to generate verbose chain-of-thought (CoT) or JSON often produces markdown artifacts, invalid JSON, or syntax errors.
2. **Logit Poisoning (Pre-Decisional CoT Hazard)**: Generating free-form rationale *before* outputting the decision allows hallucinated phrases or sycophancy early in the sequence to irreversibly corrupt subsequent token distributions.
3. **Severe Latency Overhead**: Generating 100–300 tokens takes 500–2,500ms on local GPUs.

```
Traditional Autoregressive Generation:
Prompt ──> CoT Reasoning Tokens ──> [Logit Poisoning Risk] ──> Categorical Verdict (1,500ms+)

Gevva0 Dual-Path Architecture:
Prompt ──> Direct Logit Readout ──> Calibrated Softmax ──> Top Confidence >= Gate ?
                 │                                                │
                 ├── YES ──> Fast-Path Locked Verdict (15-40ms)   │
                 └── NO  ──> Short Bounded Verification (80-150ms)─┘
                                       │
                    [Decision Permanently Locked]
                                       │
                                       ▼ (Isolated Forward Pass)
                    Asymmetric Evidence & Verbatim Quote Extraction
```

In Gevva0, candidate token logits for candidate letters (`" A"`, `" B"`, `" C"`, ...) are evaluated directly from `llm.scores[llm.n_tokens - 1]`. By marginalizing over leading whitespace and bare tokens (`logsumexp(logit(" A"), logit("A"))`), the system captures the model's pure prior without autoregressive drift.

---

## 2. Adaptive Verification & Bounded CoT

When fast-path confidence falls below the configurable threshold (`confidence_threshold`, default 0.85):
- The model context is appended with:
  `"Analysis: Let's systematically verify each option against the context: "`
- The engine generates a bounded scratchpad of up to 40 tokens (sampling temperature 0.2, terminating at newline or EOS).
- Candidate letter logits are re-extracted immediately following the verification step.
- This hybrid System 1 / System 2 approach resolves ambiguous multi-hop clauses without imposing full autoregressive latency on straightforward queries.

---

## 3. Asymmetric Audit Trail (Post-Decision Quotation)

To satisfy enterprise auditability and compliance without risking logit poisoning:
- **Decision Locking**: The classification verdict is irreversibly locked.
- **Isolated Forward Pass**: A secondary inference pass is triggered using an explicit extraction prompt.
- **Strict Verbatim Grounding**: The model locates the exact text span or visual artifact justifying the locked choice and formats it as JSON containing `quote` and `rationale`.
- **Zero Logit Contamination**: Because this pass occurs *after* the decision, explanation hallucinations cannot alter the classification.

---

## 4. Cyclic Permutation Debiasing

LLMs exhibit intrinsic positional and alphabetic biases (e.g. favoring `"A"` or whichever option appears first). Gevva0 implements cyclic label-permutation debiasing:

Given \( N \) options \( \{O_0, O_1, \dots, O_{N-1}\} \):
1. For shift \( s \in \{0, 1, \dots, N-1\} \), permute options cyclically:
   \[
   O^{(s)}_i = O_{(i + s) \pmod N}
   \]
2. Score candidate logits and compute softmax probabilities \( P^{(s)} \).
3. Map probabilities back to their original semantic option index:
   \[
   P_{\text{debiased}}(O_j) = \frac{1}{N} \sum_{s=0}^{N-1} P^{(s)}\left(\text{letter}( (j - s) \pmod N )\right)
   \]
4. Renormalize over all options. This guarantees total mathematical invariance to label assignment order.

---

## 5. Platt Temperature Calibration & Metric Monitoring

Uncalibrated neural network logits are notoriously overconfident. Gevva0 includes a calibration layer:
\[
P(y = c \mid x) = \frac{\exp(z_c / T)}{\sum_k \exp(z_k / T)}
\]

- **Optimization**: Temperature \( T \) is learned by minimizing the **Brier Score** over a validation set:
  \[
  \text{Brier} = \frac{1}{M} \sum_{m=1}^M \sum_{k=1}^K (P_{m,k} - y_{m,k})^2
  \]
- **Monitoring**: Calibration quality is verified via **Expected Calibration Error (ECE)** across confidence bins:
  \[
  \text{ECE} = \sum_{b=1}^B \frac{|B_b|}{M} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|
  \]
The learned temperature file (`calibration.json`) is loaded automatically and applied to both fast-path and CoT logits.

---

## 6. Hardware Sizing & Context Management

Instead of static, rigid device profiles, Gevva0 exposes fine-grained context sizing (`n_ctx`) directly in `llm_config.json` and through dynamic API reload endpoints.

Reference specifications across Gemma 4 model sizes:

| Architecture | Model File | Parameters (Active / Total) | Min VRAM | Context Tested | Recommended KV Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 26B-A4B MoE** | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 3.8B / 25.2B | 16 GB | 8k – 64k tokens | `Q8_0` (or `Q4_0` for 16GB) |
| **Gemma 4 E4B Dense** | `gemma-4-E4B-it-qat-UD-Q4_K_XL` | 4.2B / 4.2B | 8 GB | 8k – 32k tokens | `Q8_0` |
| **Gemma 4 E2B Edge** | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | 2.1B / 2.1B | 4 GB | 4k – 16k tokens | `Q8_0` |
| **Multimodal Projector**| `mmproj-BF16.gguf` | ~400M | +0.8 GB | Matches LLM ctx | F32 |

Memory scaling formula:
\[
\text{VRAM}_{\text{total}} \approx \text{Weights}_{\text{GB}} + \left( \frac{n_{\text{ctx}}}{1024} \times \text{KV}_{\text{per\_1k}} \right) + \text{Overhead}_{\text{CUDA}}
\]
For Gemma 4 26B-A4B, \(\text{KV}_{\text{per\_1k}} \approx 0.125\text{ GB}\) at 8-bit precision, enabling up to 64,000 tokens of dense contract audit on a single 32 GB RTX 5090.


## 7. KV-Cache Branching for Accelerated Cyclic Debiasing

To achieve strict label invariance without incurring an $N\times$ prefill latency penalty, `GemmaDecisionEngine` implements zero-copy KV-cache sequence branching via low-level `llama.cpp` sequence memory primitives.

```
Base Sequence 0: [ System Prompt + Document Context Prefix (K Tokens) ] ── (Ingested Once)
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼ (kv_cache_seq_cp)       ▼ (kv_cache_seq_cp)       ▼ (kv_cache_seq_cp)
    Sequence 0                Sequence 1                Sequence 2 ... N-1
[ Suffix Shift 0 ]        [ Suffix Shift 1 ]        [ Suffix Shift 2 ... ]
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                 Concurrent Micro-Batch Forward Pass
                 Logits Extracted via llama_get_logits_ith
                                   │
                                   ▼
                 kv_cache_seq_rm (Purge & Reset)

```

#### 7.1. Fork $\to$ Forward $\to$ Discard Lifecycle

* **Unified Context Setup:** During engine initialization, context parameters enforce `n_seq_max >= 8` (supporting up to 32 parallel sequences via `llama_max_parallel_sequences()`) and `kv_unified = True`, preventing static context memory fragmentation.
* **Shared Context Prefill (Seq 0):** In `score_cyclic_branching`, the common prompt prefix (system instructions and contextual document) is ingested a single time into sequence 0 up to token offset $K$.
* **Cache Sequence Forking:** Sequence 0's KV state is cloned to sequences $1 \dots N-1$ via zero-copy cache duplication:
```c
kv_cache_seq_cp(0, s, 0, -1); // wraps llama_memory_seq_cp / llama_kv_cache_seq_cp

```


* **Concurrent Micro-Batch Forward Pass:**
* If the combined token length of all $N$ permuted option suffixes fits within `n_batch`, all branches are evaluated concurrently in a single micro-batch. Output logits are extracted per branch via `llama_get_logits_ith`.
* If the combined suffix length exceeds `n_batch`, the engine executes a sequential fallback evaluating each branch starting strictly at token offset $K$.


* **Safe Cleanup & Discard:** In an isolated `finally` block, forked sequences $1 \dots N-1$ and base sequence 0 are purged via `kv_cache_seq_rm`. This prevents residual KV state leakage and rotary position embedding (RoPE) drift across requests.

---

#### 7.2. Configuration & Threshold Guards

Branching behavior is configured in `llm_config.json` under the `decision` block:

```json
{
  "decision": {
    "cot_threshold": 0.85,
    "cot_max_tokens": 256,
    "cot_temp": 0.2,
    "cyclic_debias": false,
    "kv_branching": false,
    "kv_branching_min_tokens": 1500
  }
}

```

* `kv_branching: false` *(Default)*: Executes standard clean-reset context evaluation for bit-exact calibration replication.
* `kv_branching: true`: Enables the sequence-forked branching pipeline for multi-permutation passes.
* `kv_branching_min_tokens: 1500`: Heuristic bypass. For short inputs ($< 1{,}500$ tokens) where initial prefill overhead is negligible, the engine routes through standard reset to bypass sequence management overhead.
* **Multimodal Guardrail:** Vision requests utilizing the `mmproj` projector automatically fall back to clean context resets to protect clip projector tensor alignment.

---

#### 7.3. CLI & REST API Integration

* **CLI Execution:**
```bash
uv run gevva0 decide \
  --context "..." \
  --options "..." \
  --cyclic \
  --kv-branching \
  --kv-branching-min-tokens 1500

```


* **REST API Schema:**
The parameters are exposed on `DecisionRequest` in `schema.py` and handled by `server.py`:
```python
class DecisionRequest(BaseModel):
    context: str
    options: list[str]
    cyclic_debias: bool = True
    kv_branching: bool = False
    kv_branching_min_tokens: int = 1500

```
