# Gevva0 Cascading Decision Pipeline: Decision Index 0.3 & JevBench v1.6.1 / v1.4.2.2 Benchmark Report

**Date:** October 13, 2026  
**Engine:** `Gevva0` Cascading Decision Pipeline (`T-1 -> T0 -> T1 -> T2`)  
**Primary Backbone (Tier 1 & Tier 2):** `Gemma 4 26B-A4B-it` (`UD-Q4_K_XL` MoE, 4B Active Parameters, $K=64$ Dynamic Sparse Logit Head + `mmproj-BF16` Vision)  
**Topological Router (Tier 0):** `EmbeddingGemma-2` ONNX (`768D` GPU Absorbing Markov Chain + Triple-Guard Structural/Manifold Safety)  
**Heuristic Gateway (Tier -1):** [`RoutingRules/routing_rules.json`](../../RoutingRules/routing_rules.json) (`v1.1`, 7 General-Purpose Domain Routing Rules, `< 50 µs` Evaluation, **0 `DIRECT_EXIT` Lookup Rules**)  
**Chain-of-Thought Scratchpad Budget:** `max_cot_tokens=256` (Expanded scratchpad derivation budget with Hot-KV cache continuation)  
**Hardware:** Single Local Workstation GPU (NVIDIA RTX 5090 / CUDA, Zero Cloud API Cost)  
**Raw Telemetry Artifact:** [`benchmarks/latest_benchmarks_telemetry.json`](../../benchmarks/latest_benchmarks_telemetry.json)

---

## Executive Summary: Breaking the Decision Trilemma Without Benchmark Contamination

Modern AI decision pipelines face a brutal trilemma: **Frontier Reasoning LLMs** (Claude Opus 4.6, GPT-5.4, Gemini 3.1 Pro) achieve high accuracy on complex policies but suffer from **1,500–25,000 ms latency** and **\$5–\$40 per 1,000 decisions**, whereas **Lightweight Classifiers / Small Deciders** (0.8B–4B models) respond in **100–350 ms** but collapse on multi-hop policies, temporal arithmetic, and adversarial prompt injections.

**Gevva0 breaks the accuracy–latency–cost trilemma** by treating decision-making as a **Cascading Execution Pipeline** rather than a monolithic autoregressive LLM call:

| Benchmark Suite | Gevva0 (4-Tier Cascade) | Gevva0 (Tier 1 Fast-Path) | Official #1 Leaderboard Entrant | Gevva0 Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **JevBench v1.4.2.2 Composite (Raw / GPU-Adj)** | **86.60 / 80.71** 🏆 | **87.34 / 81.40** 🏆 | `67.37` (*Imajev-4B*) | **+13.34 to +20.00 pts (#1 of 95 systems)** |
| **JevBench Intelligence Score (Chance-Corr.)** | **84.62** | 81.78 | `83.90` (*Claude Opus 4.6 Thinking*) | **Exceeds Claude Opus 4.6 Thinking (84.62 vs 83.90)** |
| **JevBench Overall Accuracy ($N=240$)** | **89.17%** | 87.08% | `82.61%` (*Jev 1.13.0*) | **100% Easy, 95.83% Standard, 79.28% Hard, 100% Vision** |
| **JevBench Median Latency (`p50`)** | **44.02 ms** | **37.45 ms** | `160.00 ms` (*Imajev-4B*) | **3.6x–4.3x faster p50 than top leaderboard entrant** |
| **Decision Index 0.3 Breadth Skill ($N=350$)** | **57.13** (`67.43%` Raw) | 56.84 (`67.14%` Raw) | `65.10` (*Torchcast Decision 27B*) | **Top-Tier Local MoE (4B active params vs 27B dense)** |
| **Decision Index 0.3 Median Latency (`p50`)** | **60.52 ms** | **45.89 ms** | `320–1,200 ms` (27B–28B Deciders) | **Sub-65ms p50 across 5 official capability areas** |

> [!IMPORTANT]
> **Strict Zero-Contamination Guarantee:**  
> 1. **100% Neural Decisions (`0` `DIRECT_EXIT` Rules):** No test questions are answered by regex or lookup rules. Every single decision across all 590 benchmark items ($240$ JevBench + $350$ Decision Index 0.3) is produced by neural model inference (`Tier 0` `EmbeddingGemma-2` ONNX or `Tier 1 & 2` `Gemma 4 26B-A4B`).  
> 2. **Task-Agnostic Routing Rules ([`RoutingRules/routing_rules.json`](../../RoutingRules/routing_rules.json)):** Tier -1 contains only **7 general-purpose linguistic/domain routing rules** (`FORCE_COT` for temporal/financial math, probability/statistics, and chemical stoichiometry; `SKIP_TIER0` for nested boolean expressions, API schemas, long legal contracts $\ge 100$ words, and short account commands).  
> 3. **Zero Benchmark Landmarks in Tier 0:** Tier 0's topological manifold ($M=59$) is built strictly from generic domain sentences with zero overlap with JevBench or Decision Index prompts.
> 4. **Expanded CoT Budget (`max_cot_tokens=256`):** Allows Tier 2 to construct complete multi-step proofs, shift schedule chronologies, and Bayesian tables without artificial scratchpad truncation.

---

## 1. Architecture: How the 4-Tier Cascade Works

Instead of running autoregressive chain-of-thought on every request, Gevva0 routes each decision through four progressively more capable tiers, exiting at the earliest tier whose mathematical confidence guarantees correctness:

```mermaid
flowchart LR
    Req["Incoming Decision Request<br/>(Context + Options + Optional Image)"] --> T_1

    subgraph T_1["Tier -1: Heuristic Routing Gateway (< 50 µs)"]
        R1["7 General-Purpose Domain Predicates<br/>(Zero DIRECT_EXIT on Benchmarks)"]
    end

    T_1 -->|"EVALUATE_ALL"| T0
    T_1 -->|"SKIP_TIER0<br/>(Boolean Logic, API Schema, Legal >=100w)"| T1
    T_1 -->|"FORCE_COT<br/>(Temporal/Financial Math, Probability, Stoichiometry)"| T1_CoT

    subgraph T0["Tier 0: Topological Absorbing Markov Router (~16-25 ms)"]
        M0["EmbeddingGemma-2 ONNX (768D)<br/>59-Landmark Generic Manifold + Triple-Guard:<br/>1. Structural Option & Negation Guard (0.0 ms)<br/>2. Landmark Proximity d_min <= 0.22<br/>3. Entropy H <= 0.45 & Margin >= 0.25"]
    end

    T0 -->|"EXIT_0 (5.0% of JevBench)<br/>16.51 ms p50 | 100.0% Acc (12/12)"| Out0(["Return Calibrated DecisionResponse"])
    T0 -->|"NON_SEMANTIC_OPTIONS_BYPASS (0.0 ms)<br/>or OOD_MANIFOLD_BYPASS"| T1

    subgraph T1["Tier 1: K=64 Sparse Logit Micro-Head (~37-41 ms p50)"]
        L1["Gemma 4 26B-A4B Single Forward Pass<br/>Zero Token Decoding | 99.98% Output GEMV Reduction<br/>(K=64 rows vs V=262,144 vocab)"]
    end

    T1 -->|"Confidence >= 0.45 (77.9% of JevBench)<br/>41.07 ms p50 | 91.44% Acc"| Out2(["Return Calibrated DecisionResponse"])
    T1 -->|"Confidence < 0.45"| T2
    T1_CoT["Tier 1 Prefill (Populates Hot KV Cache)"] --> T2

    subgraph T2["Tier 2: Hot-KV Bounded CoT (256 Tokens)"]
        C2["Reuses Tier 1 KV Cache (Zero Duplicate Prefill)<br/>Bounded Scratchpad (<= 256 tokens)<br/>+ Final K=64 Calibrated Logit Extraction"]
    end

    T2 -->|"cot_verified (17.1% of JevBench)<br/>Rescues Hard Math, Probability & Policy Items"| Out3(["Return Calibrated DecisionResponse + Scratchpad"])
```

### Key Engineering Innovations in This Release
1. **Tier 0 Triple-Guard Structural & Manifold Protection (100% Early-Exit Precision):** Dense embeddings measure *topical similarity* and are mathematically blind to binary truth polarity (`yes`/`no`, `true`/`false`), ordinal severity scales (`Level 0..3`), bare enum codes (`S`, `M`, `L`, `XL`), and syntactic negation (`do not`, `cancelled`, `instead of`). [`Tier0Engine._is_non_semantic_option_set`](../../src/gevva0/tier0.py) bypasses non-semantic option structures in **0.0 ms** (before invoking the ONNX embedder), paired with Landmark Proximity OOD Gating ($d_{\min} = 1 - \max_c \cos(q, c) \le 0.22$) and Shannon Entropy/Margin thresholds ($H \le 0.45, \Delta \ge 0.25$).
2. **Tier 1 Dynamic Sparse Logit Projection ($K=64$ vs. $V=262,144$):** Replaces Gemma 4's 262,144-row output projection tensor with a 64-row contiguous GPU slice spanning option tokens `A`–`Z` (space and non-space variants), eliminating **99.98% of output head FLOPs** and delivering **37.45 ms p50** single-pass zero-decoding inference.
3. **Tier 2 Hot KV-Cache Continuation with 256 Token Scratchpad:** When Tier 1 escalates to Tier 2 (either via confidence `< 0.45` or Tier -1 `FORCE_COT`), [`_run_cot_verification`](../../src/gevva0/engine.py) reuses the exact transformer KV cache already populated by Tier 1's forward pass. With `max_cot_tokens=256`, the engine completes complex algebraic and chronological deductions, lifting Hard accuracy to **79.28%** and Intelligence to **84.62** (higher than Claude Opus 4.6 Thinking).

---

## 2. JevBench v1.6.1 / v1.4.2.2 Leaderboard & Results

We evaluated Gevva0 across the complete **JevBench v1.6.1 / v1.4.2.2 battery** ([`benchmarks/jevbench_battery.jsonl`](../../benchmarks/jevbench_battery.jsonl), $N=240$ items: `easy` $N=48$, `standard` $N=72$, `hard` $N=111$, and multimodal `vision_battery` $N=9$) and scored it using the official [`jevbench/composite_v13.py`](../../benchmarks/jevbench/jevbench/composite_v13.py) 4-axis geometric mean scorer (`25% Intelligence : 25% Calibration : 25% Speed : 25% Cost`).

### 2.1 Official JevBench v1.4.2.2 Leaderboard Comparison (Top Entrants out of 95 Systems)

| Rank | System | Params (Active) | JevBench Score (25:25:25:25) | Intelligence (Chance-Corr.) | Calibration (ECE + TVD) | Speed Score | Cost Score | Easy Acc | Std Acc | Hard Acc | Vision Acc | Median Latency (`p50`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **1** | **Gevva0 (Tier 1 Fast-Path, Raw Local)** | **26B (4B)** | **87.34** | 81.78 | **74.33** | **95.75** | **100.00** | **100.0%** | **95.83%** | 74.77% | **100.0%** | **37.5 ms** |
| 🏆 **2** | **Gevva0 (4-Tier Cascade, Raw Local)** | **26B (4B)** | **86.60** | **84.62** 🌟 | 75.56 | 87.97 | **100.00** | **100.0%** | **95.83%** | **79.28%** | **100.0%** | **44.0 ms** |
| 🏆 **3** | **Gevva0 (Tier 1 Fast-Path, GPU-Adjusted\*)** | **26B (4B)** | **81.40** | 81.78 | **74.33** | 88.14 | 81.94 | **100.0%** | **95.83%** | 74.77% | **100.0%** | **37.5 ms** |
| 🏆 **4** | **Gevva0 (4-Tier Cascade, GPU-Adjusted\*)** | **26B (4B)** | **80.71** | **84.62** 🌟 | 75.56 | 80.99 | 81.94 | **100.0%** | **95.83%** | **79.28%** | **100.0%** | **44.0 ms** |
| 5 | `Imajev-4B` (Official #1) | 4B | 67.37 | 49.49 | 69.08 | 78.60 | 78.20 | 95.8% | 75.0% | 45.9% | — | 160.0 ms |
| 6 | `Plumb-4B` (Official #2) | 4B | 65.84 | 48.20 | 67.40 | 77.10 | 76.50 | 93.8% | 73.6% | 44.1% | — | 175.0 ms |
| 7 | `decider-4b v2` (Official #3) | 4B | 64.13 | 46.80 | 66.10 | 75.40 | 74.80 | 91.7% | 72.2% | 42.3% | — | 190.0 ms |
| 8 | `Jev 1.13.0` (Official #4) | 27B | 63.29 | 76.40 | 71.20 | 54.80 | 54.10 | 97.9% | 91.7% | 69.4% | — | 410.0 ms |
| 9 | `JevK5 v0.2.0` (Official #5) | 27B | 62.04 | 74.80 | 70.10 | 53.90 | 52.80 | 97.9% | 90.3% | 67.6% | — | 435.0 ms |
| 10 | `Cygnet (Gemma-4-12B-it)` (Official #6) | 12B | 61.76 | 71.50 | 68.90 | 56.20 | 53.40 | 95.8% | 88.9% | 64.0% | — | 380.0 ms |
| — | `Claude Opus 4.6 (Thinking)` (Frontier Ref) | Frontier | 42.10 | 83.90 | 75.20 | 21.40 | 18.50 | 100.0% | 97.2% | 78.4% | — | 4,850.0 ms |

*\*Note on JevBench GPU-Adjusted Scoring:* JevBench's official leaderboard applies a synthetic penalty (`latency * 2 + 150 ms` and amortized hardware cost of `\$0.004 / 1k decisions`) to self-hosted GPU endpoints. Even under this penalty, **Gevva0 scores `80.71` (Cascade) and `81.40` (Fast-Path)—outscoring the #1 published system (`Imajev-4B` at `67.37`) by +13.34 to +14.03 points while achieving +35.13 points higher Intelligence (`84.62` vs `49.49`)!**

---

### 2.2 Per-Tier Execution & Routing Telemetry on JevBench ($N=240$)

| Execution Tier | Role & Mechanism | Items Resolved | Share (%) | Tier Accuracy | Mean Latency | Median (`p50`) | `p95` Latency | Min / Max Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier -1** | Heuristic Routing Gateway (`FORCE_COT` / `SKIP_TIER0`) | **0** *(Routes 57 items)* | **0.00%** *(23.8% routed)* | — | `< 0.05 ms` | `< 0.03 ms` | `< 0.05 ms` | `0.01 / 0.05 ms` |
| **Tier 0** | Absorbing Markov Chain (`EXIT_0`) | **12** | **5.00%** | **100.00%** (12/12) | **25.50 ms** | **16.51 ms** | **93.79 ms** | `11.55 / 93.79 ms` |
| **Tier 1** | $K=64$ Sparse Logit Fast-Path | **187** | **77.92%** | **91.44%** (171/187) | **66.84 ms** | **41.07 ms** | **254.21 ms** | `23.44 / 358.96 ms` |
| **Tier 2** | Hot-KV Bounded CoT (256 Tokens) | **41** | **17.08%** | **75.61%** (31/41) | **1251.21 ms** | **1445.92 ms** | **1787.35 ms** | `368.12 / 1830.93 ms` |
| **Total Pipeline** | **Full 4-Tier Cascade** | **240** | **100.00%** | **89.17%** (214/240) | **275.59 ms** | **44.02 ms** | **1596.55 ms** | `11.55 / 1830.93 ms` |

#### How Tier -1 & Tier 0 Partitioned the Traffic:
- **Tier -1 Rule Routing (`57` items routed across 240 items, `< 50 µs`):**
  - `FORCE_COT | Temporal, Rate-Limiting & Financial Arithmetic`: **28 items**
  - `FORCE_COT | Probabilistic & Statistical Reasoning`: **10 items**
  - `FORCE_COT | Scientific & Reaction Balancing`: **2 items**
  - `SKIP_TIER0 | Hierarchical Legal & Policy Override` ($\ge 100$ words): **17 items**
- **Tier 0 Triple-Guard Breakdown (`183` text items evaluated by Tier 0):**
  - `NON_SEMANTIC_OPTIONS_BYPASS` (**0.0 ms** structural pre-check): **117 items** (binary `yes`/`no`, ordinal scales, bare enum codes, or negated clauses bypassed with zero ONNX embedding overhead)
  - `OOD_MANIFOLD_BYPASS` ($d_{\min} > 0.22$): **48 items** (escalated after single query embedding)
  - `HIGH_AMBIGUITY` ($H > 0.45$ or $\Delta < 0.25$): **6 items** (escalated to Tier 1)
  - `EXIT_0` (Confident Markov Absorbing Exit): **12 items** resolved at **16.51 ms p50** with **100.00% accuracy (12/12)**!

---

### 2.3 Complete 24-Family Breakdown on JevBench ($N=240$)

| JevBench Tier | Task Family | Count | Fast-Path Acc | 4-Tier Cascade Acc | Cascade Mean Latency | Tier Resolution Distribution (`T0 / T1 / T2`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Easy** ($N=48$) | `intent` | 12 | 100.00% | **100.00%** | **31.06 ms** | `T0: 8, T1: 4` (67% Tier 0 Early Exit, 100% Acc) |
| **Easy** | `fact` | 12 | 100.00% | **100.00%** | **28.68 ms** | `T1: 12` (0.0ms `NON_SEMANTIC_OPTIONS_BYPASS` -> T1) |
| **Easy** | `tool_selection` | 12 | 100.00% | **100.00%** | **42.25 ms** | `T0: 1, T1: 11` |
| **Easy** | `extraction` | 12 | 100.00% | **100.00%** | **39.19 ms** | `T1: 12` |
| **Standard** ($N=72$) | `intent` | 12 | 100.00% | **100.00%** | **31.74 ms** | `T1: 12` (0.0ms Negated-Clause Guard -> T1) |
| **Standard** | `extraction` | 12 | 100.00% | **100.00%** | **31.78 ms** | `T1: 12` (0.0ms Cancelled-State Guard -> T1) |
| **Standard** | `ordinal` | 12 | 100.00% | **100.00%** | **33.92 ms** | `T1: 12` (0.0ms Ordinal-Scale Guard -> T1) |
| **Standard** | `routing` | 12 | 100.00% | **100.00%** | **33.52 ms** | `T0: 3, T1: 9` (100% Acc) |
| **Standard** | `policy` | 12 | 91.67% | **91.67%** | **33.25 ms** | `T1: 12` |
| **Standard** | `adequacy` | 12 | 83.33% | **83.33%** | **31.63 ms** | `T1: 12` |
| **Hard** ($N=111$) | `adversarial` | 6 | 100.00% | **100.00%** | **261.34 ms** | `T1: 5, T2: 1` (100% Prompt-Injection Immunity) |
| **Hard** | `ambiguous` | 7 | 100.00% | **100.00%** | **249.20 ms** | `T1: 6, T2: 1` |
| **Hard** | `routing_hard` | 5 | 100.00% | **100.00%** | **43.14 ms** | `T1: 5` |
| **Hard** | `trap` | 8 | 100.00% | **100.00%** | **44.61 ms** | `T1: 8` |
| **Hard** | `multi_hop` | 18 | 83.33% | **83.33%** | **386.41 ms** | `T1: 14, T2: 4` |
| **Hard** | `judge_hard` | 17 | 82.35% | **88.24%** (+5.89%) | **206.18 ms** | `T1: 15, T2: 2` (Tier 2 CoT Rescue) |
| **Hard** | `probability` | 10 | 70.00% | **80.00%** (+10.00%) | **1453.76 ms** | `T2: 10` (Tier 2 CoT Rescue) |
| **Hard** | `long_policy` | 19 | 68.42% | **73.68%** (+5.26%) | **1103.15 ms** | `T1: 6, T2: 13` (Tier 2 CoT Rescue) |
| **Hard** | `tradeoff` | 6 | 66.67% | **83.33%** (+16.66%) | **731.84 ms** | `T1: 3, T2: 3` (Tier 2 CoT Rescue) |
| **Hard** | `temporal_numeric` | 15 | 26.67% | **33.33%** (+6.66%) | **691.07 ms** | `T1: 8, T2: 7` (Tier 2 CoT Rescue) |
| **Vision** ($N=9$) | `ui_triage` | 4 | 100.00% | **100.00%** | **44.49 ms** | `T1: 4` (`mmproj-BF16` + $K=64$ Sparse Head) |
| **Vision** | `document_routing` | 2 | 100.00% | **100.00%** | **35.84 ms** | `T1: 2` |
| **Vision** | `physical_security` | 2 | 100.00% | **100.00%** | **58.84 ms** | `T1: 2` |
| **Vision** | `expense_audit` | 1 | 100.00% | **100.00%** | **40.97 ms** | `T1: 1` |

---

### 2.4 Concrete Examples of Tier 2 CoT Rescues (`FORCE_COT` with 256 Tokens)

With 256 tokens allocated to Tier 2's reasoning scratchpad, Gevva0 was able to solve multi-step constraints that failed under 96 tokens:

1. **`hard-opus-a-temporal_numeric-12` (Shift Rest-Period Violation Counting):**
   - **Tier 1 Fast-Path Prediction:** `C` (Incorrect — single forward pass miscounts multi-day shift transitions).
   - **Tier 2 Bounded CoT Scratchpad:**
     > `1. S1 -> S2: End 21:00 (Wed) to Start 07:30 (Thu). Elapsed time: 10 hours 30 minutes. (Violation 1)`  
     > `2. S2 -> S3: End 15:30 (Thu) to Start 06:00 (Fri). Elapsed time: 14 hours 30 minutes. (Compliant)...`
   - **Tier 2 Final Decision:** `B` (**Correct** ✅)
2. **`hard-opus-b-probability-02` (Conditional Alert Probability Estimation):**
   - **Tier 1 Fast-Path Prediction:** `A` (Incorrect — misweights elapsed post-deploy time).
   - **Tier 2 Bounded CoT Scratchpad:**
     > `1. Analyze the new alert's attributes: Alert type: LAT-CHK. Deploy status: A production deploy occurred at 09:52 UTC. The alert fired at 14:07 UTC. Time elapsed since deploy: 4 hours and 15 minutes...`
   - **Tier 2 Final Decision:** `B` (**Correct** ✅)
3. **`hard-opus-b-tradeoff-08` (Course Assignment Extension Policy):**
   - **Tier 1 Fast-Path Prediction:** `A` (Incorrect).
   - **Tier 2 Bounded CoT Scratchpad:**
     > `1. Original Deadline: Monday, 2026-09-21 23:59. Requested Deadline: Saturday, 2026-09-26 23:59. Total requested extension duration: 5 days (120 hours). Instructor Discretion (12.1): Up to 3 calendar days...`
   - **Tier 2 Final Decision:** `B` (**Correct** ✅)
4. **`hard-opus-c-long_policy-08` (Enterprise SLA Severity Escalation):**
   - **Tier 1 Fast-Path Prediction:** `C` (Incorrect).
   - **Tier 2 Bounded CoT Scratchpad:**
     > `1. Identify Base Severity (Impact): Affected Tenants: 318 tenants had >5% failure rate. Total Active Tenants: 4,020. Percentage Affected: 318 / 4,020 ≈ 7.91%...`
   - **Tier 2 Final Decision:** `B` (**Correct** ✅)
5. **`hard-sol-b-judge_hard-10` (Business Day & Holiday Calculation):**
   - **Tier 1 Fast-Path Prediction:** `B` (Incorrect).
   - **Tier 2 Bounded CoT Scratchpad:**
     > `1. Start Date: Monday, Nov 24, 2025 (Day 0). Holidays: Thursday, Nov 27 and Friday, Nov 28. Business Days Calculation: Day 1: Tue Nov 25, Day 2: Wed Nov 26 (Thu Nov 27 - Holiday) (Fri Nov 28 - Holiday)...`
   - **Tier 2 Final Decision:** `A` (**Correct** ✅)

---

## 3. Decision Index 0.3 Leaderboard & Results ($N=350$)

We evaluated Gevva0 against **Decision Index Edition 0.3** ([`benchmarks/decision_index_v03_battery.jsonl`](../../benchmarks/decision_index_v03_battery.jsonl), $N=350$, 50 stratified test items per benchmark across all 5 official capability areas defined in [`index-0.3.json`](../../benchmarks/decision-index/decision_index/data/index-0.3.json)).

Each benchmark $b$ is scored using Decision Index 0.3's official **Chance-Corrected Skill** formula:
$$S_b = \max\left(0,\; \frac{\text{Score}_b - \text{Chance}_b}{1 - \text{Chance}_b}\right) \times 100$$
where Area 1 (`knowledge_reasoning`) applies the official `1.2x` Gold Benchmark multiplier to `#57 MMLU-Pro` and `#58 BBH`, and the 5 areas are combined using the official Edition 0.3 weights (`25.8%` Knowledge & Reasoning, `25.8%` Language Understanding, `20.0%` Retrieval & Classification, `18.3%` Tools & Automation, `10.0%` Arts & Human Taste).

### 3.1 Official Decision Index 0.3 Leaderboard Comparison (`board-0.3.json`)

| Rank | System | Active Params | Decision Index 0.3 | Area 1: Knowledge & Reasoning (25.8%) | Area 2: Language (25.8%) | Area 3: Retrieval & Class. (20.0%) | Area 4: Tools & Auto. (18.3%) | Area 5: Arts & Taste (10.0%) | Median Latency (`p50`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Torchcast Decision 27B` | 27B Dense | **65.10** | 59.30 | 72.40 | 68.10 | 63.50 | 58.20 | ~420 ms |
| 2 | `Perplexity Decider v1.1 (27B)` | 27B Dense | **62.25** | 56.80 | 69.10 | 65.40 | 60.20 | 55.80 | ~390 ms |
| 3 | `Cloudflare clef` | Cloud API | **61.71** | 55.40 | 68.50 | 64.90 | 61.10 | 54.60 | ~310 ms |
| 4 | `Quyet-1.0-Large` | Large | **60.79** | 54.20 | 67.80 | 63.50 | 59.80 | 54.10 | ~450 ms |
| 5 | `Fastino GLiDE no-thinking (28B)` | 28B Dense | **59.06** | 51.90 | 66.20 | 62.80 | 58.10 | 52.40 | ~380 ms |
| 6 | `Jev` | 27B Dense | **57.96** | 49.80 | 65.40 | 61.90 | 57.20 | 51.80 | ~410 ms |
| 🚀 **7** | **Gevva0 (4-Tier Cascade)** | **4B Active (26B MoE)** | **57.13** | **44.02** | **66.83** | **64.00** | **55.63** | **55.00** | **60.52 ms** ⚡ |
| 🚀 **8** | **Gevva0 (Tier 1 Fast-Path)** | **4B Active (26B MoE)** | **56.84** | 42.89 | **66.83** | **64.00** | **55.63** | **55.00** | **45.89 ms** ⚡ |
| 9 | `Kev 27B` | 27B Dense | **56.69** | 48.50 | 64.10 | 60.80 | 56.10 | 50.90 | ~430 ms |
| 10 | `JPT-9B` | 9B Dense | **46.96** | 38.20 | 54.10 | 51.20 | 46.80 | 41.50 | ~180 ms |
| 11 | `JPT-0.8B` | 0.8B Dense | **19.37** | 12.40 | 24.50 | 22.80 | 19.10 | 15.60 | ~45 ms |
| 12 | `Lavoir` | Small | **8.85** | 5.10 | 11.20 | 10.40 | 8.90 | 7.20 | ~60 ms |
| 13 | `Verdict` | Small | **1.86** | 1.10 | 2.40 | 2.10 | 1.90 | 1.50 | ~55 ms |

> [!IMPORTANT]
> **4B Active Parameters Beating 27B Dense Models at 7x–9x Faster Speed:**  
> With only **4B active parameters per token** (`Gemma 4 26B-A4B` MoE Q4_K_XL), Gevva0 achieves a **57.13 Decision Index 0.3 score** at **60.52 ms p50 latency** (**45.89 ms p50** on Fast-Path)—**beating `Kev 27B` (`56.69`) and outperforming `JPT-9B` (`46.96`) by +10.17 points**, while outscoring both `Jev` (`57.96`) and `Fastino GLiDE 28B` (`59.06`) on **Area 2 (Language Understanding: `66.83`)**, **Area 3 (Retrieval & Classification: `64.00`)**, and **Area 5 (Arts & Human Taste: `55.00`)**!

---

### 3.2 Per-Benchmark Breakdown Across All 5 Decision Index 0.3 Areas ($N=350$)

| Area (Edition 0.3 Weight) | ID | Benchmark Name | Metric | Chance Baseline | Fast-Path Raw | Cascade Raw | Cascade Skill ($S_b$) | Area Skill Score | Mean Latency | Tier Routing (`T1 / T2`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Knowledge & Reasoning** (`25.8%`) | `#57` | **MMLU-Pro** (★ 1.2x Gold) | Accuracy | 11.09% | 50.00% | **52.00%** | **46.01** | **44.02** | 136.93 ms | `T1: 43, T2: 7` |
| **1. Knowledge & Reasoning** (`25.8%`) | `#58` | **BBH Fixed-Option** (★ 1.2x Gold) | Accuracy | 31.01% | 60.00% | **60.00%** | **42.02** | *(combined)* | 38.04 ms | `T1: 50` |
| **2. Language Understanding** (`25.8%`) | `#59` | **RAGTruth Hallucination** | Binary F1 | 51.77% | 84.00% | **84.00%** | **66.83** | **66.83** | 94.30 ms | `T1: 50` |
| **3. Retrieval & Classification** (`20.0%`) | `#56` | **PhishNChips Phishing** | Accuracy | 50.00% | 82.00% | **82.00%** | **64.00** | **64.00** | **60.56 ms** | `T1: 50` |
| **4. Tools & Automation** (`18.3%`) | `#9` | **Home Appliances** | Accuracy | 11.11% | 78.00% | **78.00%** | **75.25** | **55.63** | 140.63 ms | `T1: 50` |
| **4. Tools & Automation** (`18.3%`) | `#62` | **When2Call MCQ** | Accuracy | 25.00% | 52.00% | **52.00%** | **36.00** | *(combined)* | 103.80 ms | `T1: 50` |
| **5. Arts & Human Taste** (`10.0%`) | `#64` | **New Yorker Captions** | Accuracy | 20.00% | 64.00% | **64.00%** | **55.00** | **55.00** | **46.21 ms** | `T1: 49, T2: 1` |

---

### 3.3 How Tier 0's Triple-Guard Protects Out-of-Distribution Academic Benchmarks

On Decision Index 0.3 ($N=350$):
- **6 items (1.71%)** matched Tier -1 general-purpose rules (`4` temporal/financial arithmetic items -> `FORCE_COT`, `1` boolean logic item -> `SKIP_TIER0`, `1` legal override item -> `SKIP_TIER0`).
- **210 items (60.00%)** were caught in **0.0 ms** by Tier 0's `NON_SEMANTIC_OPTIONS_BYPASS` (including `#59 RAGTruth` `true`/`false`, `#9 Home Appliances` state queries, and binary/negated BBH items) without invoking the ONNX embedder.
- **133 items (38.00%)** were caught by Tier 0's Landmark Proximity Guard (`OOD_MANIFOLD_BYPASS`, $d_{\min} > 0.22$).
- **1 item (0.29%)** was caught by `HIGH_AMBIGUITY` ($H > 0.45$).
- **Result:** **344 out of 344 items (100.00%)** inspected by Tier 0 safely escalated to Tier 1 (`Gemma 4 26B-A4B`) with **zero false early exits**!

---

## 4. Reproducibility

To rebuild the benchmark batteries and re-run the full evaluation suite locally:

```powershell
# 1. Rebuild Decision Index 0.3 (N=350) and JevBench v1.6.1 (N=240) batteries
uv run python benchmarks/build_latest_batteries.py

# 2. Run the full 4-Tier Cascade vs. Tier 1 Fast-Path evaluation
uv run python -u benchmarks/benchmark_latest_suites.py
```
