# Gevva0 Tier 0 Showcase: The Game of Pig (`demos/pig`)

A fast-paced 1v1 interactive dice battle and high-speed Monte Carlo simulation arena showcasing **Tier 0 (`Tier0Engine` + `Tier0DecisionBridge`)** on **EmbeddingGemma-2 ONNX (300M, 768D)** without cascading to Tier -1, Tier 1, or Tier 2.

---

## Why "The Game of Pig" is the Gold Standard for Tier 0

1. **Exact Absorbing Markov Decision Process (Todd & Fisher, 2007)**:
   - Two players take turns rolling a single 6-sided die.
   - **Roll 2–6**: Add the roll to your unbanked turn pot $k$, then choose **ROLL** or **HOLD**.
   - **Roll 1 ("Pig Out")**: Lose all unbanked turn points ($k \to 0$) and end your turn immediately.
   - **First to 100 points wins.**
2. **Why Human Heuristics Fail**:
   - Humans rely on static gut rules like **"Hold at 20"**.
   - The true Game-Theory Optimal (GTO) threshold $k^*(i, j)$ dynamically warps across the 3D state space $(i, j, k)$:
     - **When behind** (e.g. $40$ vs $82$): GTO Pig takes calculated, aggressive risks ($k^* > 32$).
     - **When ahead** (e.g. $82$ vs $40$): GTO Pig chokes the opponent out with conservative holds ($k^* \le 15$).
3. **Sub-3ms Tier 0 Execution via `Tier0DecisionBridge`**:
   - Uses `Tier0DecisionBridge` (`src/gevva0/bridge.py`) to expose all four internal components of `Tier0Engine`:
     - **(1) Direct Cosine Alignment**: $S_{\text{cosine}}(i) = \frac{q \cdot c_i}{\|q\|\|c_i\|}$
     - **(2) Default 39-Landmark Scaffold (`DEFAULT_CORPUS`)**
     - **(3) Pig Domain Manifold Scaffold ($M = 104$)** solved in closed form via the fundamental matrix $B = (I - Q)^{-1} R$
     - **(4) Production Hybrid Ensemble**: $P_{\text{final}} = 0.65 \cdot \text{Cosine} + 0.35 \cdot \text{Markov}$
     - **Tripwire Diagnostics**: Shannon Entropy $H \le 0.65$ and Logit Margin $\Delta \ge 0.25$ $\rightarrow$ `[EXIT TIER 0]`.

---

## Quickstart

```powershell
# 1. Interactive 1v1 Play against Gevva0 Tier 0 (with live ONNX inference)
uv run python demos/pig/run.py

# 2. Launch directly into High-Speed Simulation Mode
uv run python demos/pig/run.py --sim

# 3. Run in Simulated Stand-In Mode (no ONNX/GPU required)
uv run python demos/pig/run.py --simulate

# 4. Run Headless High-Speed Tournament Benchmark (200+ games per opponent)
uv run python demos/pig/headless.py --games 200
```

### Controls (Mouse + Keyboard)

| Key / Button | Action |
| :--- | :--- |
| `SPACE` / `R` | **Roll Die** (in 1v1 Human Play mode) |
| `H` / `ENTER` | **Hold & Bank Points** (in 1v1 Human Play mode) |
| `M` | Toggle between **1v1 Play vs Gevva0** and **High-Speed Simulation Mode** |
| `S` | Cycle Simulation Speed (`1x Animated` $\rightarrow$ `10x Fast` $\rightarrow$ `TURBO`) |
| `T` | Instantly simulate **+50 Full Games** at turbo speed |
| `O` | Cycle Challenger Bot Heuristic (`Hold at 20`, `Hold at 28`, `Hold at 15`, `Score-Pace`) |
| `1` / `2` / `3` / `4` | Switch live between `(4) 65/35 Hybrid`, `(3) Pure Domain AMC`, `(2) 39-Landmark Cold`, `(1) Pure Cosine` |
| `N` | Start New Game |
| `ESC` | Exit Demo |
