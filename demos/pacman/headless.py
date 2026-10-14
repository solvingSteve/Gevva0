"""
Headless evaluation: run full games at max speed with the simulated or real brain.

    uv run python demos/pacman/headless.py --games 5            # simulated brain
    uv run python demos/pacman/headless.py --games 2 --real     # Gemma 4 via Gevva0 (sync)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from game import FRAME, Game
from pilot import Pilot


def play(brain, seed: int, max_levels: int = 3, max_time: float = 900.0, verbose=False) -> dict:
    g = Game(seed=seed)
    pilot = Pilot(g, brain, sync=True)
    while not g.game_over and g.time < max_time and g.levels_cleared < max_levels:
        pilot.tick()
        g.update(FRAME)
        if verbose:
            for e in g.events[-3:]:
                pass
    deaths = [e.detail for e in g.events if e.kind == "death"]
    return dict(seed=seed, score=g.score, level=g.level, cleared=g.levels_cleared, deaths=g.deaths,
                t=round(g.time, 1), gpe=round(g.ghosts_per_energizer, 2), death_causes=deaths,
                buffered=f"{pilot.stats.prebuffered_hits}/{pilot.stats.junction_arrivals}",
                stale=pilot.stats.stale, decisions=pilot.stats.dispatches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.real:
        from brain import Gevva0Brain
        brain = Gevva0Brain()
        brain.load()
    else:
        from brain import SimulatedBrain
        brain = SimulatedBrain()
        brain.realtime = False
        brain.load()
    res = []
    for i in range(args.games):
        t0 = time.perf_counter()
        r = play(brain, args.seed + i, args.levels)
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        print(r, flush=True)
        res.append(r)
    print("mean score", sum(r["score"] for r in res) / len(res),
          "| mean levels cleared", sum(r["cleared"] for r in res) / len(res),
          "| deaths", sum(r["deaths"] for r in res))
    if args.real:
        p50, p95 = brain.latency_stats()
        print(f"latency p50 {p50:.1f} ms p95 {p95:.1f} ms | modes {brain.mode_counts}")


if __name__ == "__main__":
    main()
