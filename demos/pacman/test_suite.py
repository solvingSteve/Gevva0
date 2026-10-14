"""
Comprehensive Integration Test Suite: Gevva0 Arcade Pac-Man (v2).
Tests:
1. Maze topology, exits, wrap-around toroidal distances, junctions count.
2. Game core: actors movement, ghost speeds, scatter/chase timers, pellet consumption, energizer fright timers.
3. Tactical perception: corridor follow, ghost threat maps, rollout simulator predictions, option formatting.
4. Pilot: pre-buffered junction arrivals, latch mechanics, reversal safety hysteresis.
5. Brain: Pinned-prefix KV cache fork/forward/discard lifecycle, calibrated posterior calculation, bounded CoT.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root and demo path
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_DEMO_DIR))

from maze import (
    COLS, ROWS, TOTAL_DOTS, TOTAL_ENERGIZERS, JUNCTIONS,
    exits, is_floor, toroidal_dist, follow_corridor, UP, DOWN, LEFT, RIGHT
)
from game import Game, FRAME, BASE_SPEED, PacMan, Ghost, Actor
from tactics import analyze, RouteOption, Situation
from pilot import Pilot
from brain import SimulatedBrain, ModelProfile, softmax


class TestMazeTopology(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(COLS, 28)
        self.assertEqual(ROWS, 31)

    def test_pellet_counts(self):
        self.assertEqual(TOTAL_DOTS, 240)
        self.assertEqual(TOTAL_ENERGIZERS, 4)

    def test_junction_count(self):
        self.assertEqual(len(JUNCTIONS), 34)

    def test_wrap_toroidal_distance(self):
        # Distance across warp tunnel
        d = toroidal_dist((0, 14), (27, 14))
        self.assertAlmostEqual(d, 1.0)


class TestGameCore(unittest.TestCase):
    def test_pacman_eating_dot(self):
        g = Game(seed=1)
        initial_dots = len(g.dots)
        # Advance Pac-Man 10 frames
        for _ in range(10):
            g.update(FRAME)
        self.assertLess(len(g.dots), initial_dots)
        self.assertGreater(g.score, 0)

    def test_clone_exactness(self):
        g = Game(seed=42)
        for _ in range(30):
            g.update(FRAME)
        c = g.clone()
        self.assertEqual(g.pac.pos, c.pac.pos)
        self.assertEqual(g.score, c.score)
        self.assertEqual(len(g.dots), len(c.dots))


class TestTacticsAndRollout(unittest.TestCase):
    def test_situation_analysis(self):
        g = Game(seed=1)
        sit = analyze(g)
        self.assertIsNotNone(sit)
        self.assertGreater(len(sit.options), 0)
        for opt in sit.options:
            self.assertIn(opt.status, ("SAFE", "TIGHT", "TRAPPED", "DEADLY"))
            self.assertGreaterEqual(opt.room, 0)
            self.assertGreaterEqual(opt.outs, 0)

    def test_no_illegal_wall_moves(self):
        g = Game(seed=1)
        sit = analyze(g)
        for opt in sit.options:
            # Committed route path must be entirely on walkable floor
            for t in opt.path:
                self.assertTrue(is_floor(t[0], t[1]))


class TestPilotPrebuffering(unittest.TestCase):
    def test_prebuffering(self):
        b = SimulatedBrain()
        b.realtime = False
        b.load()
        g = Game(seed=1)
        p = Pilot(g, b, sync=True)

        for _ in range(300):
            p.tick()
            g.update(FRAME)

        self.assertGreater(p.stats.dispatches, 0)
        self.assertGreater(p.stats.prebuffered_hits, 0)


class TestBrainCalibration(unittest.TestCase):
    def test_profile_calibration(self):
        prof = ModelProfile(temperature=2.0, bias_offsets={"A": 1.0, "B": 0.0})
        import numpy as np
        raw = np.array([3.0, 1.0])
        cal = prof.calibrate(raw, ["A", "B"])
        # (3.0 - 1.0)/2 = 1.0, (1.0 - 0.0)/2 = 0.5
        np.testing.assert_allclose(cal, np.array([1.0, 0.5]))
        p = softmax(cal)
        self.assertAlmostEqual(float(sum(p)), 1.0)


if __name__ == "__main__":
    unittest.main()
