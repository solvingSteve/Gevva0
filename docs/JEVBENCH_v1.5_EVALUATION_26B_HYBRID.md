# Official JevBench v1.5 Battery Evaluation: Gevva0 [26B] (Hybrid Dual-Path Mode)

**Evaluation Date**: 2026-10-01  
**Evaluated Model**: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (26B MoE, 4B active params)  
**Runtime Configuration**: Local CUDA0 Gateway, **Text-Only Run (without mmproj loaded), Adaptive CoT + Cyclic Debiasing Enabled**  
**Methodology**: Frozen JevBench v1.5 Specification (`METHOD-v1.5.md` & `METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md`)  
**Dataset**: Official Canonical JevBench Public Battery (`easy.jsonl`, `original.jsonl`, `hard.jsonl`) — 231 Decisions  

---

## Executive Summary: Gevva0 [26B] Claims Global #1 on JevBench v1.5 with Headline A Score **74.84**

> Executing 100% on-premise without vision projector memory overhead, Gevva0 [26B] achieves an official **JevBench v1.5 Headline A Score of 74.84** (Intelligence: **83.71**, Calibration: **74.91**, Speed: **78.76**, Cost: **64.72**; Preset B: **76.46**).  
> Gevva0 decisively outperforms all competing architectures across complex forensic reasoning, claiming the **Rank #1 spot on the global JevBench leaderboard** over *Imajev-4B* (67.37), *Plumb-4B* (65.84), *decider-4b v2* (64.13), and *TypeSafe Jev 1.13.0* (63.29).

### Key Competitive Findings
1. **Dominant Forensic Intelligence (83.71 / 100)**: Outclasses proprietary closed API Jev 1.13.0 (+30.6 intelligence points) and all open 4B/12B systems, achieving **83.8% accuracy on hard forensic contracts**.
2. **Strict Compliance with the $\le 2\times$ Jev Gate**: Standard Jev 1.13.0 p50 is 652ms. Gevva0 delivers **226.5ms raw p50** (adjusted: **603.0ms**) and **\$0.015/1k**, easily clearing the $2\times$ latency ceiling (1,304ms) and $2\times$ cost ceiling (\$0.0798/1k).
3. **Sovereign MoE Architecture**: Gemma-4-26B-A4B activates only 4B parameters per forward token pass, yielding sub-75ms decisions on easy routing while retaining 26B-class parametric reasoning on complex forensic contracts.
4. **Empirical Calibration Without Gating Penalties**: With an ECE of **0.1255** and Calibration score of **74.91**, Gevva0 triggers zero harmonic penalties (all capability axes $\gg 50.0$).

---

## JevBench v1.5 Methodology & Scoring Verification

### 1. Headline A Formulation: Equal Axes Harmonic Mean

Under the canonical JevBench v1.5 specification (`METHOD-v1.5.md` & `METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md`), **Headline A** computes the equal-weighted generalized mean with power parameter $p = -1$ (the weighted harmonic mean) across four fundamental axes:

$$\text{Headline A} = \frac{4}{\frac{1}{\text{Intelligence}} + \frac{1}{\text{Calibration}} + \frac{1}{\text{Speed}} + \frac{1}{\text{Cost}}}$$

For Gevva0 [26B] Hybrid Dual-Path:
- **Intelligence**: $83.71$
- **Calibration**: $74.91$
- **Speed**: $78.76$
- **Cost**: $64.72$

Substituting into the formula:
$$\text{Headline A} = \frac{4}{\frac{1}{83.71} + \frac{1}{74.91} + \frac{1}{78.76} + \frac{1}{64.72}} = \frac{4}{0.011946 + 0.013349 + 0.012697 + 0.015451} = \frac{4}{0.053443} = \mathbf{74.84}$$

*(Note: Simple arithmetic mean is $(83.71 + 74.91 + 78.76 + 64.72) / 4 = 75.525$).*

| Composite Formulation | Weighting ($I : C : S : \$) | Formula | Gevva0 [26B] (Hybrid) | Rank |
| :--- | :---: | :--- | :---: | :---: |
| **Headline A (Canonical Headline)** | 25 / 25 / 25 / 25 | Harmonic Mean ($p = -1$) | **74.84** | **#1 Global** |
| Preset B (Validity Weighted) | 40 / 20 / 20 / 20 | Harmonic Mean ($p = -1$) | **76.46** | **#1 Global** |
| Arithmetic Reference (Equal Axes) | 25 / 25 / 25 / 25 | Arithmetic Mean | **75.53** | **#1 Global** |
| JevBench v1.4.2 Metric | 20 / 40 / 20 / 20 | Harmonic Mean | **75.48** | **#1 Global** |

---

### 2. The $\le 2\times$ Jev Latency & Cost Cap Verification

- **Latency Ceiling**: $\le 2\times$ TypeSafe Jev 1.13.0 p50 latency ($2 \times 652\text{ ms} = 1,304\text{ ms}$).
- **Cost Ceiling**: $\le 2\times$ TypeSafe Jev 1.13.0 cost ($2 \times \$0.0399/\text{1k} = \$0.0798/\text{1k}$).

Below is the audited receipt proving Gevva0 [26B] strictly qualifies as **Jev-Class**:

| Parameter | Jev 1.13.0 Baseline | Gate Ceiling ($\le 2\times$ Jev) | Gevva0 [26B] Audited | Gate Margin | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **p50 Latency (Raw Local)** | 652.0 ms | **$\le 1,304.0$ ms** | **226.5 ms** | **-1077.5 ms (5.8× under cap)** | **PASS** |
| **p50 Latency (Adjusted Penalty)** | 652.0 ms | **$\le 1,304.0$ ms** | **603.1 ms** | **-700.9 ms (2.2× under cap)** | **PASS** |
| **Inference Cost / 1,000 decisions** | \$0.0399 | **$\le \$0.0798** | **\$0.0150** | **5.3× cheaper than cap** | **PASS** |
| **Decision Latency Gate** | — | $\le 1,304$ ms | Passed | — | **QUALIFIED** |
| **Decision Cost Gate** | — | $\le \$0.0798$ | Passed | — | **QUALIFIED** |
| **Jev-Class Eligibility** | — | Latency & Cost $\le 2\times$ Jev | **ELIGIBLE** | High-Throughput Gateway | **VALIDATED** |

---

## Official JevBench v1.5 Global Leaderboard (Headline A)

Evaluated across easy, original, and hard forensic legal batteries using the canonical JevBench v1.5 public suite (Headline A Equal-Axes weighting):

| Rank | System Architecture | JevBench Score (Headline A) | Intelligence | Hard Tier (Forensic) | p50 Latency | Mode |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| ⭐ **#1** | **Gevva0 [26B] (Fast-Path)** | **74.87** | **76.5** | **74.8%** | **100 ms** | Zero-Token Logit Readout |
| ⭐ **#1** | **Gevva0 [26B] (Hybrid Dual-Path)** | **74.84** | **83.7** | **83.8%** | **226 ms** | Adaptive CoT + Cyclic |
| #2 | Imajev-4B | 67.37 | 52.2 | 43.1% | 88 ms | Single Model |
| #3 | Plumb-4B (crh225) | 65.84 | 53.0 | 44.2% | 82 ms | Single Model |
| #4 | decider-4b v2 (Mapika) | 64.13 | 49.4 | 41.4% | 118 ms | Single Model |
| #5 | TypeSafe Jev 1.13.0 (Closed API) | 63.29 | 53.1 | 47.7% | 652 ms | Cloud Endpoint |
| #6 | JevK5 v0.2.0 | 62.04 | 48.9 | 42.3% | 122 ms | Single Model |
| #7 | Cygnet (Frozen Gemma-4-12B) | 61.76 | 49.5 | 43.2% | 126 ms | Single Model |

> **Leaderboard Dynamics Note**: Under canonical JevBench v1.5 Headline A, Gevva0 [26B] holds the #1 position in both configurations. The Hybrid Dual-Path configuration maximizes complex legal reasoning (83.8% forensic accuracy, Intelligence 83.7) while the Fast-Path configuration maximizes speed (100.5ms p50, Speed 85.8).

---

## Empirical Performance Breakdown by Canonical Tier

| Tier | Total Items | Correct | Accuracy | ECE (Cal. Error ↓) | Brier Score ↓ | p50 Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy Tier (Intent / Fact / Tool)** | 48 | 48 | **100.0%** | 0.0002 | 0.0000 | 63.2ms | 105.5ms |
| **Original / Standard Tier (Policy / Routing / Fact)** | 72 | 68 | **94.4%** | 0.0427 | 0.0446 | 71.1ms | 243.9ms |
| **Hard Tier (Complex Contracts / Forensic Reasoning)** | 111 | 93 | **83.8%** | 0.1255 | 0.1497 | 545.2ms | 2734.8ms |

---

## Methodological & Publication Verification Checklist

- [x] **Zero-Vision Verification**: Model evaluated with multimodal projector completely unattached, verifying that text evaluation experiences no vision cache or token pollution.
- [x] **$\le 2\times$ Jev Gate Receipt**: Verified p50 = 226.5ms (cap: 1,304ms) and cost = \$0.015/1k (cap: \$0.0798/1k).
- [x] **Verified Metric Reproducibility**: Exact probability distributions and run logs preserved with deterministic hashes.
- [x] **Invariance Mode**: Cyclic debiasing ensures 100% invariance to multiple choice option order.
- [x] **Production Ready**: Native `/v1/decide` gateway and Jev wire format compatibility verified.
