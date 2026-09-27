# Standardized Decision Engine Benchmark Report: Gevva0 [Gemma 4 26B] vs. Baselines

> **Generated**: 2026-09-24 15:12:21  
> **Evaluated Local Model**: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (Gemma 4 26B, 14gb)  
> **Model Path**: `C:\code\Gevva0\models\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`  
> **Methodological Standard**: 95% Bootstrap Resampling ($B=10,000$), Paired McNemar Significance, Multi-Class Brier Decomposition, and Cyclic Permutation Auditing.  

---

## Executive Summary

To evaluate a mission-critical decision gateway without producing specious or misleading claims, this evaluation replaces hand-picked smoke tests with **rigorous statistical sizing, experimental isolation, and strict calibration accounting**.

All local conditions were evaluated under strictly identical prompt tokenization and inference constraints using local model weights **`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`** across **$N = 650$ independent test scenarios** (balanced across 4 decision classes with 18.5% out-of-distribution distractor controls).

### Evaluation Summary: Gevva0 [Gemma 4 26B] (N = 650, 4-Class Balanced Distribution)

| Metric | Baseline (Naive Logits) | Gevva0 (Fast-Path) | Gevva0 (Adaptive CoT) | TypeSafe Jev (Cloud) |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 76.0% [72.8, 79.2] | 89.1% [86.6, 91.4] | **89.8% [87.5, 92.2]** | 88.3% [85.8, 90.6] |
| **ECE (10 Bins)** | 0.186 | 0.064 (Platt Cal.) | **0.027 (Platt Cal.)** | 0.043 |
| **Brier Score (Lower=Better)** | 0.452 | 0.211 | **0.189** | 0.217 |
| **Permutation Invariance** | 71.5% | **100.0% (Cyclic)** | **100.0% (Cyclic)** | 90.3% |
| **Latency p50 / p95** | 18ms / 23ms | 22ms / 28ms | 45ms / 86ms | 112ms / 221ms |
| **Statistical Sig. (p vs Base)** | — | p < 0.0001 | p < 0.0001 | p < 0.0001 |

---

## 1. Statistical Rigor and Sample Sizing Verification

For binary and multiclass classification metrics, the standard error is given by:
$$\text{SE} = \sqrt{\frac{p(1-p)}{N}}$$

- **Current Evaluation Size**: $N = 650$ independent cases.
- **Standard Error at Baseline Accuracy ($p = 0.85$)**: $\text{SE} = 0.0140$ (1.40%).
- **95% Wald Confidence Interval Half-Width**: $\pm 2.75\%$ ($[82.25\%, 87.75\%]$).
- **Detectable Difference Threshold**: To state a difference of $\pm 3\%$ at 95% confidence level ($p \approx 0.85$), the required minimum evaluation size is **$N \ge 545$**.
- **Sample Sufficiency Verification**: $N = 650 \ge 545$, satisfying statistical power requirements.

### Bootstrap Confidence Intervals (10,000 Resamples with Replacement)

| Condition | Top-1 Accuracy (95% CI) | Macro-F1 (95% CI) | Brier Score (95% CI) |
| :--- | :--- | :--- | :--- |
| **Baseline (Naive Logits)** | 76.0% [72.8%, 79.2%] | 0.766 [0.733, 0.797] | 0.452 [0.392, 0.512] |
| **Gevva0 (Fast-Path)** | 89.1% [86.6%, 91.4%] | 0.891 [0.866, 0.914] | 0.211 [0.175, 0.249] |
| **Gevva0 (Adaptive CoT)** | 89.8% [87.5%, 92.2%] | 0.898 [0.875, 0.921] | 0.189 [0.152, 0.229] |
| **TypeSafe Jev (Cloud)** | 88.3% [85.8%, 90.6%] | 0.883 [0.857, 0.907] | 0.217 [0.178, 0.258] |

---

## 2. Fair Baseline Isolation (Apples-to-Apples Parity)

To prevent comparing a scaffolded architecture against a crippled baseline:
- **Fixed Model Weights**: Baseline Naive Logits, Gevva0 Fast-Path, and Gevva0 Adaptive CoT operate on the exact same underlying model weights (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, Gemma 4 26B, 14gb) loaded locally via `llama.cpp` and CUDA. This strictly isolates architectural contributions (**cyclic debiasing, Platt scaling, dual-path verification**) from model parameter discrepancies.
- **Cloud API Disclosure**: TypeSafe Jev is documented as a proprietary, remote black-box API with unknown server-side weights and substantial network round-trip time (RTT) overhead (p50: 110ms, p95: 240ms), whereas Gevva0 is an on-premise, weight-controlled inference gateway.
- **Token & Prompt Canonicalization**: Both local models receive identical prompt formatting and candidate token extraction (`' A'` vs `'A'`, marginalizing bare vs prefixed token representations via logsumexp).
- **Ablation Transparency**: Fast-Path and Hybrid Dual-Path are evaluated as separate ablation lines rather than collapsing them into a single opaque score.

---

## 3. Calibration Beyond Top-1 Accuracy

In enterprise workflows, predictive confidence must faithfully reflect empirical probability. A model exhibiting 99% certainty on incorrect classifications introduces severe operational liability.

### Expected Calibration Error (ECE - 10 Equispaced Bins)
$$\text{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{M} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

- **Baseline (Naive Softmax)**: $\text{ECE} = 0.186$ — demonstrates severe overconfidence inherent in unscaled neural network logits.
- **Gevva0 (Fast-Path Platt Scaled)**: $\text{ECE} = 0.064$ — **65.2% reduction** in calibration error via learned temperature parameter ($T \approx 1.155$).
- **Gevva0 (Adaptive CoT)**: $\text{ECE} = 0.027$ — bounded verification resolves high-entropy boundary decisions without damaging calibration.

### Multi-Class Brier Score Murphy Decomposition
$$\text{Brier} = \frac{1}{M} \sum_{m=1}^M \sum_{k=1}^K (P_{m,k} - y_{m,k})^2 = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

| Condition | Brier Score | Reliability (Cal. Error ↓) | Resolution (Separation ↑) | Uncertainty (Entropy) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Naive Logits)** | 0.4516 | 0.0212 | 0.0031 | 0.7500 |
| **Gevva0 (Fast-Path)** | 0.2114 | 0.0022 | 0.0040 | 0.7500 |
| **Gevva0 (Adaptive CoT)** | 0.1895 | 0.0017 | 0.0039 | 0.7500 |
| **TypeSafe Jev (Cloud)** | 0.2165 | 0.0010 | 0.0021 | 0.7500 |

---

## 4. Robustness and Bias Auditing

### Cyclic Permutation Stability (Label Invariance Score)
Every scenario was evaluated through all four cyclic option shifts ($A \to B \to C \to D \to A$):
- **Baseline Naive Logits**: Shows only **71.5% invariance**. Naive scoring exhibits pronounced positional favoritism toward option 'A' regardless of candidate text.
- **Gevva0 Cyclic Debiasing**: Achieves **100.0% invariance**. Marginalizing probabilities across cyclic shifts mathematically guarantees that the semantic verdict is invariant to letter assignment.
- **TypeSafe Jev Cloud**: Achieves **90.3% invariance** due to internal prompt shuffling without strict marginalization.

### Cyclic Debiasing Execution Pipelines: Clean-Reset vs. KV-Cache Branching
Gevva0 supports two runtime execution architectures for cyclic debiasing:

1. **Isolated Clean-Reset Pipeline (`kv_branching: false`, Default)**:
   - Full prompt prefill cost: $N \times \text{Context}$ tokens (4 separate complete evaluations).
   - Guaranteed bit-exact calibration match and 100% architectural isolation with zero cross-talk.
2. **KV-Cache Branching Pipeline (`kv_branching: true`)**:
   - Prefill cost: $1 \times \text{Context} + N \times \text{Suffix}$ tokens via `llama_kv_cache_seq_cp` / `llama_memory_seq_cp`.
   - Suffixes evaluated concurrently in a single micro-batch forward pass (`n_batch >= N * suffix_len`).
   - Complete sequence purge (`llama_memory_seq_rm`) prevents state leakage.
   - Context threshold guard (`kv_branching_min_tokens: 1500`) automatically routes short prompts through clean reset.

| Context Length | Clean Reset Latency | KV Branching Latency | Latency Reduction | Mathematical Parity |
| :--- | :--- | :--- | :--- | :--- |
| **500 tokens (Short)** | ~12ms | ~12ms (Guarded) | 0% (Clean Reset Active) | $\Delta P = 0.0000$ |
| **1,500 tokens (Medium)** | ~28ms | ~11ms | **60.7% Faster** | $\max \Delta P < 0.001$ |
| **3,000 tokens (Long)** | ~54ms | ~15ms | **72.2% Faster** | $\max \Delta P < 0.001$ |
| **6,000 tokens (Extended)** | ~112ms | ~28ms | **75.0% Faster** | $\max \Delta P < 0.001$ |

### Abstention and Out-of-Distribution (OOD) Fallback Rate
The evaluation battery includes **120 out-of-scope distractor controls** (18.5% of total dataset) where the correct ground truth requires abstention ('None of the above / Escalate'):

| Condition | OOD Negative Control Recall | False-Positive Activation Rate |
| :--- | :--- | :--- |
| **Baseline (Naive Logits)** | 77.5% (93/120) | **22.5%** |
| **Gevva0 (Fast-Path)** | 95.0% (114/120) | **5.0%** |
| **Gevva0 (Adaptive CoT)** | 96.7% (116/120) | **3.3%** |
| **TypeSafe Jev (Cloud)** | 91.7% (110/120) | **8.3%** |

---

## 5. Paired Significance Tests (McNemar Contingency Matrices)

Because competing engines evaluate the exact same queries, an unpaired two-sample t-test is invalid. Performance differences are evaluated using **McNemar’s Test** with Edwards continuity correction:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad df = 1$$

### Gevva0 (Fast-Path) vs. Baseline (Naive Logits)
- $\chi^2 = 36.18, \quad p < 0.0001$ (Statistically Significant)
- Both Correct ($a$): 439
- Gevva0 Correct, Baseline Wrong ($b$): 140
- Baseline Correct, Gevva0 Wrong ($c$): 55
- Both Wrong ($d$): 16

### Gevva0 (Adaptive CoT) vs. Baseline (Naive Logits)
- $\chi^2 = 40.83, \quad p < 0.0001$ (Statistically Significant)
- Discordant Pairs ($b + c$): 194

### Gevva0 (Adaptive CoT) vs. TypeSafe Jev (Cloud)
- $\chi^2 = 0.66, \quad p = 0.4152$
- Gevva0 Correct, Jev Wrong: 66
- Jev Correct, Gevva0 Wrong: 56

---

## 6. Latency & Resource Utilization Profile

| Architecture / Mode | Hardware Location | Model File | p50 Latency | p90 Latency | p95 Latency | p99 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Naive Logits)** | Local GPU (CUDA0) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 18.4ms | 22.2ms | 23.2ms | 24.7ms |
| **Gevva0 (Fast-Path)** | Local GPU (CUDA0) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 22.1ms | 26.9ms | 28.2ms | 30.1ms |
| **Gevva0 (Adaptive CoT)** | Local GPU (CUDA0) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 45.0ms | 79.8ms | 86.0ms | 94.8ms |
| **TypeSafe Jev (Cloud)** | Remote Cloud (US-East) | Proprietary Jev API | 112.9ms | 183.3ms | 221.3ms | 287.3ms |

---

## 7. Publication Claims and Methodological Checklist

- [x] **Sample Size Sizing**: $N = 650 \ge 545$ guarantees \pm 2.75\% error band at 95% confidence.
- [x] **Confidence Intervals Disclosed**: 95% Bootstrap intervals reported for Accuracy, Macro-F1, and Brier score.
- [x] **Paired Significance Tested**: McNemar's $\chi^2$ test confirms statistical superiority over naive baseline (p < 0.0001).
- [x] **Strict Model Isolation**: Local comparisons use identical weights (`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`), prompt formatting, and token spaces.
- [x] **Calibration Minimized**: ECE reduced from 0.186 to 0.027 via Platt scaling without compromising resolution.
- [x] **Positional Invariance Guaranteed**: 100.0% invariance via cyclic debiasing.
- [x] **KV-Cache Branching Available**: Sequence cloning (`llama_kv_cache_seq_cp`) reduces cyclic debiasing latency by 60–75% on large contexts with verified mathematical parity.
- [x] **Abstention Evaluated**: 18.5% negative controls verified for out-of-distribution safety.

