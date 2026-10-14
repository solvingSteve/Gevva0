"""
Headless High-Speed Simulation & Tournament Benchmark for Gevva0 Tier 0 Blackjack.

Usage:
    uv run python demos/blackjack/headless.py --hands 500
    uv run python demos/blackjack/headless.py --hands 1000 --simulate
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

from brain import BlackjackTier0Brain, SimulatedTier0Brain
from game import HEURISTIC_PROFILES, BlackjackGame
from gto import get_blackjack_solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless High-Speed Blackjack Simulation")
    parser.add_argument("--hands", type=int, default=400, help="Hands to simulate per challenger heuristic")
    parser.add_argument("--simulate", action="store_true", help="Use simulated Tier 0 stand-in")
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="hybrid_domain",
        choices=["hybrid_domain", "domain_markov", "default_39", "pure_cosine"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    solver = get_blackjack_solver()
    print(f"[GTO Solver] Exact Blackjack Absorbing Chains & EV Tables solved in {solver.solve_time_ms:.1f} ms.")

    if args.simulate:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
    else:
        brain = BlackjackTier0Brain(engine_mode=args.engine_mode)
    brain.load()
    if not brain.ready:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
        brain.load()

    # Measure uncached live ONNX GPU inference latency across diverse Blackjack states first
    sample_states = [
        (11, False, False, 0, 6, True, False, 0.0),   # Hard 11 vs 6 -> DOUBLE
        (16, False, True, 8, 10, True, True, 0.0),    # Pair of 8s vs 10 -> SPLIT
        (12, True, True, 11, 6, True, True, 0.0),     # Pair of Aces vs 6 -> SPLIT
        (16, False, False, 0, 10, True, False, 0.0),  # Hard 16 vs 10 (TC 0) -> HIT
        (16, False, False, 0, 10, True, False, 2.5),  # Hard 16 vs 10 (TC +2.5) -> STAND (Illustrious 18)
        (18, True, False, 0, 6, True, False, 0.0),    # Soft 18 vs 6 -> DOUBLE
        (18, True, False, 0, 10, True, False, 0.0),   # Soft 18 vs 10 -> HIT
        (15, False, False, 0, 5, True, False, 0.0),   # Hard 15 vs 5 -> STAND
        (20, False, True, 10, 6, True, True, 0.0),    # Pair of 10s vs 6 -> STAND
        (12, False, False, 0, 2, True, False, 0.0),   # Hard 12 vs 2 -> HIT
    ]
    for tot, soft, pair, p_rk, up, c_dbl, c_spl, tc in sample_states:
        sit = brain.build_situation(tot, soft, pair, p_rk, up, c_dbl, c_spl, tc)
        brain.decide(sit, use_turbo_cache=False)

    p50, p95 = brain.latency_stats()
    print(
        f"\n[Tier 0 Engine] {brain.model_name} | Mode={brain.engine_mode}\n"
        f"  Uncached Live Latency: p50={p50:.2f}ms, p95={p95:.2f}ms | GTO Parity={brain.gto_match_rate:.1f}%\n"
    )

    print(
        f"{'Challenger Heuristic':<38} | {'Hands':>5} | {'Gevva0 Net':>10} | {'Opp Net':>9} | {'Gevva Edge':>10} | {'2x Wins':>7} | {'Splits':>6} | {'Time':>6}"
    )
    print("-" * 114)

    for idx, (prof_key, prof_name) in enumerate(HEURISTIC_PROFILES):
        game = BlackjackGame(seed=args.seed + idx * 100)
        game.heuristic_idx = idx
        t0 = time.perf_counter()
        game.run_turbo_hands(brain, num_hands=args.hands)
        elapsed = time.perf_counter() - t0
        st = game.stats
        edge = st.gevva_edge_over_challenger_units
        print(
            f"{prof_name:<38} | {st.hands_played:5d} | {st.gevva_net_units:+9.1f}u | "
            f"{st.challenger_net_units:+8.1f}u | {edge:+9.1f}u | {st.gevva_doubles_won:7d} | "
            f"{st.gevva_splits_played:6d} | {elapsed:5.2f}s"
        )

    brain.shutdown()


if __name__ == "__main__":
    main()
