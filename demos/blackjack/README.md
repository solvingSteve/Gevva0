# Gevva0 Tier 0 Showcase: Vegas Blackjack 21 (`demos/blackjack`)

An interactive Vegas 6-Deck S17 Blackjack table and high-speed Monte Carlo simulation arena showcasing **Tier 0 (`Tier0Engine` + `Tier0DecisionBridge`)** on **EmbeddingGemma-2 ONNX (300M, 768D)** without cascading to Tier -1, Tier 1, or Tier 2.

---

## Why Blackjack is a Premier Showcase for Tier 0

1. **Exact Absorbing Markov Decision Process (Baldwin et al., 1956; Edward O. Thorp, 1962)**:
   - Every Blackjack hand is a finite Absorbing Markov Chain over transient hand states (Hard 4–20, Soft 12–20, Pairs) terminating in absorbing dealer outcome sinks (`17`, `18`, `19`, `20`, `21`, `Bust`).
   - **Four Strategic Actions**:
     - `[A] HIT` — Draw another card when standing EV is inferior to drawing EV.
     - `[B] STAND` — Lock pat totals (17–21), trap weak dealer upcards (2–6) with stiff totals (12–16), or execute Hi-Lo True Count stand warps.
     - `[C] DOUBLE DOWN (2x)` — Double the wager for one card on powerhouse Hard 9/10/11 and flexible Soft 13–18 against weak dealer upcards.
     - `[D] SPLIT PAIR (2x)` — Split matching pairs into twin hands (always split Aces & 8s; never split 10s or 5s).
2. **Why Human Heuristics Bleed EV**:
   - **Timid Tourist ("Never Bust $\ge 12$")**: Stands on Hard 12–16 against strong dealer 7–Ace, surrendering ~3.9% EV.
   - **Dealer Mimic ("Hit $< 17$")**: Busts stiff hands against weak dealer 2–6 and misses all $2\times$ Doubles and Pair Splits, surrendering ~5.5% EV.
   - **Vegas Hunch**: Stands on Soft 18 vs 9/10/A and splits 10s, surrendering crucial edge.
   - **Gevva0 Tier 0**: Resolves all 4 actions + **Hi-Lo Illustrious 18 True Count Index Warps** (e.g., standing on Hard 16 vs 10 when $\text{TC} > 0$) in **~2 ms** with **100% GTO Parity**.
3. **Sub-3ms Tier 0 Execution via `Tier0DecisionBridge`**:
   - Exposes all four internal components of `Tier0Engine` live in the UI:
     - **(1) Direct Cosine Alignment**: $S_{\text{cosine}}(i) = \frac{q \cdot c_i}{\|q\|\|c_i\|}$
     - **(2) Default 39-Landmark Scaffold (`DEFAULT_CORPUS`)**
     - **(3) Blackjack Domain Manifold Scaffold ($M = 116$)** solved in closed form via $B = (I - Q)^{-1} R$
     - **(4) Production Hybrid Ensemble**: $P_{\text{final}} = 0.65 \cdot \text{Cosine} + 0.35 \cdot \text{Markov}$
     - **Tripwire Diagnostics**: Shannon Entropy $H \le 0.65$ and Logit Margin $\Delta \ge 0.25$ $\rightarrow$ `[EXIT TIER 0]`.

---

## Quickstart

```powershell
# 1. Interactive Table Play alongside Gevva0 Tier 0 (with live ONNX GPU inference)
uv run python demos/blackjack/run.py

# 2. Launch directly into High-Speed Simulation Mode
uv run python demos/blackjack/run.py --sim

# 3. Run in Simulated Stand-In Mode (no ONNX/GPU required)
uv run python demos/blackjack/run.py --simulate

# 4. Run Headless High-Speed Tournament Benchmark (400+ hands per heuristic)
uv run python demos/blackjack/headless.py --hands 400
```

### Controls (Mouse + Keyboard)

| Key / Button | Action |
| :--- | :--- |
| `SPACE` / `H` | **Hit (Draw Card)** (in 1v1 Interactive Table mode) |
| `ENTER` / `K` | **Stand (Hold Total)** (in 1v1 Interactive Table mode) |
| `D` | **Double Down (2x Wager, 1 Card)** (on initial 2-card hand) |
| `P` | **Split Pair (2x Wager, Twin Hands)** (on initial matching pair) |
| `M` | Toggle between **1v1 Table Play** and **High-Speed Simulation Mode** |
| `S` | Cycle Simulation Speed (`1x Animated` $\rightarrow$ `10x Fast` $\rightarrow$ `TURBO`) |
| `T` | Instantly simulate **+50 Full Hands** at turbo speed |
| `O` | Cycle Challenger Bot Heuristic (`Timid Tourist`, `Dealer Mimic`, `Vegas Hunch`, `Flat Basic`) |
| `1` / `2` / `3` / `4` | Switch live between `(4) 65/35 Hybrid`, `(3) Pure Domain AMC`, `(2) 39-Landmark Cold`, `(1) Pure Cosine` |
| `N` | Deal Next Hand |
| `ESC` | Exit Demo |
