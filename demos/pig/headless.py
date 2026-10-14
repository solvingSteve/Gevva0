"""
Headless High-Speed Simulation & Tournament Benchmark for Gevva0 Tier 0 Pig.

Usage:
    uv run python demos/pig/headless.py --games 200
    uv run python demos/pig/headless.py --games 500 --simulate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from brain import PigTier0Brain, SimulatedTier0Brain
from game import HEURISTIC_PROFILES, PigGame
from gto import get_gto_solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless High-Speed Pig Simulation")
    parser.add_argument("--games", type=int, default=200, help="Games to simulate per opponent")
    parser.add_argument("--simulate", action="store_true", help="Use simulated Tier 0 stand-in")
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="hybrid_domain",
        choices=["hybrid_domain", "domain_markov", "default_39", "pure_cosine"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    solver = get_gto_solver()
    print(f"[GTO Solver] 100x100x100 Absorbing MDP solved in {solver.solve_time_ms:.1f} ms.")

    if args.simulate:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
    else:
        brain = PigTier0Brain(engine_mode=args.engine_mode)
    brain.load()
    if not brain.ready:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
        brain.load()

    # Measure uncached live ONNX GPU inference latency across 25 diverse states first
    sample_states = [
        (0, 0, 0),
        (20, 45, 12),
        (40, 82, 22),
        (78, 40, 16),
        (50, 50, 21),
        (85, 70, 15),
        (30, 75, 19),
        (64, 62, 24),
    ]
    for i_s, j_s, k_s in sample_states:
        sit = brain.build_situation(i_s, j_s, k_s)
        brain.decide(sit, use_turbo_cache=False)

    p50, p95 = brain.latency_stats()
    print(
        f"\n[Tier 0 Engine] {brain.model_name} | Mode={brain.engine_mode}\n"
        f"  Uncached Live Latency: p50={p50:.2f}ms, p95={p95:.2f}ms | GTO Parity={brain.gto_match_rate:.1f}%\n"
    )

    print(f"{'Opponent Heuristic':<32} | {'Games':>6} | {'Gevva0 Win %':>12} | {'Aggr Warps':>10} | {'Lead Chokes':>11} | {'Sim Time':>8}")
    print("-" * 92)

    for idx, (prof_key, prof_name, _) in enumerate(HEURISTIC_PROFILES):
        game = PigGame(seed=args.seed + idx * 100)
        game.heuristic_idx = idx
        t0 = time.perf_counter()
        game.run_turbo_games(brain, num_games=args.games)
        elapsed = time.perf_counter() - t0
        st = game.stats
        print(
            f"{prof_name:<32} | {st.games_played:6d} | {st.gevva_win_rate:11.1f}% | "
            f"{st.gevva_aggressive_warps:10d} | {st.gevva_conservative_chokes:11d} | {elapsed:7.2f}s"
        )

    brain.shutdown()


if __name__ == "__main__":
    main()
