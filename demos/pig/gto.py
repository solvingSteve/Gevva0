"""
Exact Absorbing Markov Decision Process (MDP) Solver for the Game of Pig.
Based on Todd W. Neller & Clifton G.M. Presser (2004) / Todd & Fisher (2007).

Rules:
- Two players take turns rolling a fair 6-sided die.
- Roll 2..6: Add roll r to turn bank k. Player may ROLL again or HOLD.
- Roll 1 ("Pig"): Lose all unbanked turn points (k -> 0), turn ends.
- Hold: Bank k points permanently (i -> i + k), turn ends.
- First player to reach GOAL = 100 points wins.

Why Pig is the Gold Standard for Tier 0:
- Human players rely on static heuristics like "Hold at 20".
- Optimal GTO strategy dynamically warps the hold threshold k*(i, j) based on
  the exact 3D state (i, j, k):
  * Behind (i << j, especially when j >= 75): takes calculated, aggressive risks (holds at 30..100).
  * Ahead (i >> j, close to 100): chokes the opponent out with conservative holds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

GOAL = 100


@dataclass(frozen=True)
class GTOEvaluation:
    i: int  # Active player banked score
    j: int  # Opponent banked score
    k: int  # Active turn unbanked points
    p_win_roll: float  # Exact win probability if rolling
    p_win_hold: float  # Exact win probability if holding
    optimal_action: str  # "ROLL" or "HOLD"
    ev_delta: float  # p_win_roll - p_win_hold
    hold_threshold: int  # Minimum k >= 1 where HOLD is optimal for this (i, j)
    regime: str  # Tactical regime identifier
    regime_description: str  # Natural language tactical summary for Tier 0 embedding


class PigGTOSolver:
    """Solves the exact 100 x 100 x 100 3D Absorbing Markov Decision Process for Pig."""

    def __init__(self, goal: int = GOAL):
        self.goal = goal
        t0 = time.perf_counter()
        cache_file = Path(__file__).resolve().parent / f".gto_mdp_{goal}.npz"
        loaded = False
        if cache_file.is_file():
            try:
                data = np.load(cache_file)
                self.p_win = data["p_win"]
                self.p_roll = data["p_roll"]
                self.p_hold = data["p_hold"]
                self.optimal_roll = data["optimal_roll"]
                self.hold_thresholds = data["hold_thresholds"]
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self.p_win, self.p_roll, self.p_hold, self.optimal_roll, self.hold_thresholds = (
                self._solve_exact_mdp(goal)
            )
            try:
                np.savez_compressed(
                    cache_file,
                    p_win=self.p_win,
                    p_roll=self.p_roll,
                    p_hold=self.p_hold,
                    optimal_roll=self.optimal_roll,
                    hold_thresholds=self.hold_thresholds,
                )
            except Exception:
                pass
        self.solve_time_ms = (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _solve_exact_mdp(
        goal: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized backward induction + coupled value iteration across anti-diagonals S = i + j.

        Returns:
            p_win: [goal, goal, goal] float64 win probability under optimal play
            p_roll: [goal, goal, goal] float64 win probability if choosing ROLL
            p_hold: [goal, goal, goal] float64 win probability if choosing HOLD
            optimal_roll: [goal, goal, goal] bool True where ROLL >= HOLD
            hold_thresholds: [goal, goal] int32 minimum k >= 1 where HOLD > ROLL
        """
        p_win = np.zeros((goal, goal, goal), dtype=np.float64)
        p_roll = np.zeros((goal, goal, goal), dtype=np.float64)
        p_hold = np.zeros((goal, goal, goal), dtype=np.float64)
        optimal_roll = np.ones((goal, goal, goal), dtype=bool)

        for s in range(2 * goal - 2, -1, -1):
            i_min = max(0, s - (goal - 1))
            i_max = min(goal - 1, s)
            i_vals = np.arange(i_min, i_max + 1)
            j_vals = s - i_vals
            n_states = len(i_vals)
            if n_states == 0:
                continue

            hold_slice = np.ones((n_states, goal), dtype=np.float64)
            for k in range(1, goal):
                next_i = i_vals + k
                unfinished = next_i < goal
                if np.any(unfinished):
                    hold_slice[unfinished, k] = (
                        1.0 - p_win[j_vals[unfinished], next_i[unfinished], 0]
                    )

            p_zero = np.full(n_states, 0.5, dtype=np.float64)
            v_slice = np.ones((n_states, goal), dtype=np.float64)

            for _ in range(14):
                p_lose_turn = 1.0 - p_zero[::-1]
                for k in range(goal - 1, -1, -1):
                    active = (i_vals + k) < goal
                    if not np.any(active):
                        continue
                    sum_r = np.zeros(n_states, dtype=np.float64)
                    for r in (2, 3, 4, 5, 6):
                        kr = k + r
                        if kr >= goal:
                            sum_r += 1.0
                        else:
                            sum_r += v_slice[:, kr]
                    r_prob = (p_lose_turn + sum_r) / 6.0
                    if k == 0:
                        v_slice[active, 0] = r_prob[active]
                    else:
                        v_slice[active, k] = np.maximum(r_prob[active], hold_slice[active, k])

                new_p_zero = v_slice[:, 0].copy()
                if np.max(np.abs(new_p_zero - p_zero)) < 1e-9:
                    p_zero = new_p_zero
                    break
                p_zero = new_p_zero

            p_lose_turn = 1.0 - p_zero[::-1]
            hold_slice[:, 0] = p_lose_turn
            for k in range(goal - 1, -1, -1):
                sum_r = np.zeros(n_states, dtype=np.float64)
                for r in (2, 3, 4, 5, 6):
                    kr = k + r
                    if kr >= goal:
                        sum_r += 1.0
                    else:
                        sum_r += v_slice[:, kr]
                r_prob = (p_lose_turn + sum_r) / 6.0
                active = (i_vals + k) < goal
                r_prob = np.where(active, r_prob, 1.0)
                h_prob = np.where(active, hold_slice[:, k], 1.0)

                p_roll[i_vals, j_vals, k] = r_prob
                p_hold[i_vals, j_vals, k] = h_prob
                p_win[i_vals, j_vals, k] = v_slice[:, k]
                if k == 0:
                    optimal_roll[i_vals, j_vals, 0] = True
                else:
                    optimal_roll[i_vals, j_vals, k] = r_prob >= h_prob

        hold_thresholds = np.zeros((goal, goal), dtype=np.int32)
        for i in range(goal):
            max_k = goal - i
            for j in range(goal):
                th = max_k
                for k in range(1, max_k):
                    if not optimal_roll[i, j, k]:
                        th = k
                        break
                hold_thresholds[i, j] = th

        return p_win, p_roll, p_hold, optimal_roll, hold_thresholds

    def evaluate_state(self, i: int, j: int, k: int) -> GTOEvaluation:
        """Returns exact GTO probabilities, hold threshold, and semantic regime for (i, j, k)."""
        i_c = int(np.clip(i, 0, self.goal - 1))
        j_c = int(np.clip(j, 0, self.goal - 1))
        k_c = int(np.clip(k, 0, self.goal - 1))

        if i + k >= self.goal:
            return GTOEvaluation(
                i=i,
                j=j,
                k=k,
                p_win_roll=float(self.p_roll[i_c, j_c, min(k_c, self.goal - 1 - i_c)]),
                p_win_hold=1.0,
                optimal_action="HOLD",
                ev_delta=-0.1667,
                hold_threshold=max(1, self.goal - i),
                regime="INSTANT_WIN_CLINCH",
                regime_description=f"HOLD and bank points now: 100-point goal reached ({i}+{k}={i+k}). Lock win permanently.",
            )

        pr = float(self.p_roll[i_c, j_c, k_c])
        ph = float(self.p_hold[i_c, j_c, k_c])
        th = int(self.hold_thresholds[i_c, j_c])
        opt_act = "ROLL" if (k == 0 or pr >= ph) else "HOLD"
        delta = pr - ph

        regime, desc = self._classify_regime(i, j, k, th, pr, ph, opt_act)
        return GTOEvaluation(
            i=i,
            j=j,
            k=k,
            p_win_roll=pr,
            p_win_hold=ph,
            optimal_action=opt_act,
            ev_delta=delta,
            hold_threshold=th,
            regime=regime,
            regime_description=desc,
        )

    def _classify_regime(
        self,
        i: int,
        j: int,
        k: int,
        th: int,
        pr: float,
        ph: float,
        opt_act: str,
    ) -> tuple[str, str]:
        """Translates exact (i, j, k) stopping geometry into a concise semantic state description
        that maps onto the Tier 0 Absorbing Markov Chain manifold in ~2ms.
        """
        diff = (i + k) - j
        margin_to_th = th - k

        if k == 0:
            return (
                "FRESH_TURN_ZERO_RISK",
                f"ROLL die again: fresh turn k=0 ({i} vs {j}), zero risk, accumulate points.",
            )

        if opt_act == "ROLL":
            if j >= 78 and (i + k) < 92:
                return (
                    "DESPERATE_DEFICIT_ALL_IN",
                    f"ROLL die again: trailing endgame deficit ({i}+{k} vs {j}, target {th}), press aggressive risk.",
                )
            if k < 14:
                return (
                    "LOW_BANK_ACCUMULATE",
                    f"ROLL die again: small turn pot k={k} < {th}, accumulate points and press positive EV.",
                )
            if i < j - 15:
                return (
                    "TRAILING_AGGRESSIVE_PRESS",
                    f"ROLL die again: trailing by {-diff} pts (k={k} < {th}), take calculated risk and accumulate.",
                )
            if margin_to_th <= 3:
                return (
                    "BOUNDARY_MARGINAL_ROLL",
                    f"ROLL die again: turn pot k={k} below target {th}, positive EV to continue rolling.",
                )
            return (
                "OPTIMAL_EV_CONTINUE_ROLL",
                f"ROLL die again: active pot k={k} below target {th}, continue accumulating turn points.",
            )
        else:
            if i + k >= 82 and i > j + 15:
                return (
                    "AHEAD_CONSERVATIVE_CHOKE",
                    f"HOLD and bank points now: commanding lead ({i}+{k} vs {j}), lock score and protect equity.",
                )
            if k >= 28:
                return (
                    "HIGH_POT_BUST_PROTECTION",
                    f"HOLD and bank points now: large pot k={k} >= {th}, lock score and avoid pig-out reset.",
                )
            if abs(margin_to_th) <= 2:
                return (
                    "BOUNDARY_THRESHOLD_HOLD",
                    f"HOLD and bank points now: pot k={k} reached stopping threshold {th}, bank points safely.",
                )
            return (
                "OPTIMAL_STOPPING_BANK_HOLD",
                f"HOLD and bank points now: pot k={k} >= threshold {th}, lock unbanked points permanently.",
            )

    def get_hold_curve_for_score(self, i: int) -> np.ndarray:
        """Returns array of length 100: hold_threshold k*(i, j) for j = 0..99."""
        i_c = int(np.clip(i, 0, self.goal - 1))
        return self.hold_thresholds[i_c, :].copy()


@lru_cache(maxsize=1)
def get_gto_solver() -> PigGTOSolver:
    return PigGTOSolver(goal=GOAL)
