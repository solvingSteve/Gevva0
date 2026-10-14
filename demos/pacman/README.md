# Gevva0 Autonomous Arcade Pac-Man (Superhuman Showcase v2)

An autonomous arcade Pac-Man showcase driven by the **Gevva0 Calibrated Decision Gateway** powered by local **Gemma 4 26B-A4B MoE (QAT Q4_K_XL)** via `llama.cpp` CUDA.
---

## Key Technical Features

1. **In-Process Pinned-Prefix Decision Bridge (`brain.py`)**:
   - The full Pac-Man tactical rulebook and scoring governance (~460 tokens) is ingested into KV-cache Sequence 0 **once** at engine launch and permanently pinned in GPU VRAM.
   - Every tactical turn forks Sequence 0 (`kv_cache_seq_cp`) into an isolated worker sequence, evaluates only the dynamic state suffix (<45 tokens), reads logits directly, and discards the fork (`kv_cache_seq_rm`).
   - Achieves raw forward-pass inference latency of **p50 = 28.1 ms** (p95 = 34.3 ms) on RTX 5090!

2. **Predictive Waypoint Dispatch & Turn Latching (`pilot.py`)**:
   - Eliminates the "every-grid-square" query anti-pattern.
   - When Pac-Man commits to a hallway heading toward Junction $J_k$, the pilot dispatches the situation immediately in the background.
   - The model's decision arrives within ~30 ms and is latched into `pac.next_dir`.
   - When Pac-Man intersects $J_k$, the turn executes with **0 ms effective wait time**!

3. **Exact Forward-Simulation Perception (`tactics.py` & `rollout.py`)**:
   - Replaces fragile heuristics with exact ghost trajectory simulations.
   - Evaluates each legal route candidate against ghost AI vectors, measuring exact survival duration, escape room tiles, and ghost encounter gaps.
   - Strict legal move pruning strips solid walls and lethal hazards deterministically.

4. **Calibrated Platt Temperature & Slack-Aware Bounded CoT**:
   - Applies domain Platt temperature calibration:
     $$P(y = c) = \frac{\exp((z_c - b_c) / T)}{\sum_k \exp((z_k - b_k) / T)}$$
   - High confidence decisions execute immediately on the fast path (93%+ of queries).
   - Ambiguous queries trigger bounded System 2 CoT verification only when junction arrival slack permits.

5. **Cyberpunk Neon Telemetry Mission Control (`renderer.py`)**:
   - Dual-viewport 1200x760 presentation with locked 60 FPS Pygame loop.
   - Left Viewport: Classic 28x31 arcade maze with neon glow, ghost eyes returning, energizer pulsing, and particle celebrations.
   - Right Viewport: Live Gevva0 Mission Control HUD displaying probability distribution bars, Platt temperature ($T=4.145$), CoT scratchpad Derivations, and hardware latency meters.

---

## Quickstart

Run with `uv` from the repository root:

```powershell
# 1. Run with live Gemma 4 26B-A4B Decision Engine on GPU
uv run python demos/pacman/run.py

# 2. Run in Simulated Mode (No GPU required, instant review)
uv run python demos/pacman/run.py --simulate

# 3. Headless Benchmarking & Automated Level Validation
uv run python demos/pacman/headless.py --games 3
```

### Keyboard Controls

| Key | Action |
| :--- | :--- |
| `SPACE` | Pause / Resume Game Simulation |
| `R` | Reset Maze & Restart Game |
| `W` / `UP` | Manual Override: Steer UP |
| `A` / `LEFT` | Manual Override: Steer LEFT |
| `S` / `DOWN` | Manual Override: Steer DOWN |
| `D` / `RIGHT` | Manual Override: Steer RIGHT |
| `ESC` | Exit Demo |

---

## File Architecture

```
demos/pacman/
├── maze.py          # Classic 28x31 arcade maze geometry, graph connectivity, warp tunnels
├── game.py          # Arcade-accurate game core, speed tables, ghost behaviors, deep clone
├── tactics.py       # Tactical perception layer, candidate generation, legal pruning
├── rollout.py       # Forward-simulation ghost path predictor
├── brain.py         # Pinned-prefix Gevva0 bridge, forked decoding, Platt scaling, CoT
├── pilot.py         # Predictive waypoint dispatcher, pre-buffered turn latch
├── renderer.py      # Dual-viewport 60 FPS Pygame renderer & Mission Control HUD
├── run.py           # Main executable launcher
├── test_suite.py    # Automated integration test suite
└── upgrades-prd.md  # Production specification & benchmark criteria
```
