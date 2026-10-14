"""
Game of Pig Core Engine, Opponent Heuristics, and High-Speed Simulation Arena.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from brain import BasePigBrain, PigSituation
from gevva0.bridge import DecisionResult
from gto import GOAL, GTOEvaluation, get_gto_solver


HEURISTIC_PROFILES = [
    ("hold_20", "Human Gut ('Hold at 20')", 20),
    ("hold_28", "Greedy Gambler ('Hold at 28')", 28),
    ("hold_15", "Timid Banker ('Hold at 15')", 15),
    ("pace_22", "Score-Pace ('Hold at 22')", 22),
]


@dataclass
class TurnEvent:
    player: int  # 0 = Challenger (Human/Heuristic), 1 = Gevva0 Tier 0
    action: str  # "ROLL", "HOLD", "PIG_OUT", "WIN"
    die_value: int | None
    turn_bank_before: int
    turn_bank_after: int
    score_after: int
    gto_optimal: str
    gto_threshold: int
    latency_ms: float = 0.0
    note: str = ""


@dataclass
class MatchStats:
    games_played: int = 0
    challenger_wins: int = 0
    gevva_wins: int = 0
    challenger_busts: int = 0
    gevva_busts: int = 0
    challenger_holds: int = 0
    gevva_holds: int = 0
    challenger_banked_pts: int = 0
    gevva_banked_pts: int = 0
    gevva_comeback_wins: int = 0  # Won after trailing by >= 25 points
    gevva_aggressive_warps: int = 0  # Rolled at k >= 20 when trailing (GTO warp > 20)
    gevva_conservative_chokes: int = 0  # Held at 10 <= k < 20 when leading (GTO warp < 20)
    win_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))

    @property
    def gevva_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.gevva_wins / self.games_played) * 100.0

    @property
    def challenger_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return (self.challenger_wins / self.games_played) * 100.0

    def reset(self) -> None:
        self.games_played = 0
        self.challenger_wins = 0
        self.gevva_wins = 0
        self.challenger_busts = 0
        self.gevva_busts = 0
        self.challenger_holds = 0
        self.gevva_holds = 0
        self.challenger_banked_pts = 0
        self.gevva_banked_pts = 0
        self.gevva_comeback_wins = 0
        self.gevva_aggressive_warps = 0
        self.gevva_conservative_chokes = 0
        self.win_rate_history.clear()


class PigGame:
    """State machine for 1v1 Interactive Play and High-Speed Simulation of Pig."""

    def __init__(self, seed: int | None = None, goal: int = GOAL):
        self.goal = goal
        self.rng = random.Random(seed)
        self.solver = get_gto_solver()
        self.stats = MatchStats()

        # Mode configuration
        self.sim_mode: bool = False  # False = Human vs Gevva0, True = Bot vs Gevva0 Sim
        self.sim_speed: int = 1  # 1 = Animated, 2 = Fast, 3 = Turbo (Multi-game per frame)
        self.heuristic_idx: int = 0  # Default: "Hold at 20"

        # Per-game state
        self.scores: list[int] = [0, 0]
        self.turn: int = 0  # 0 = Challenger, 1 = Gevva0 Tier 0
        self.starting_player: int = 0
        self.turn_bank: int = 0
        self.turn_rolls: list[int] = []
        self.last_die: int = 6
        self.winner: int | None = None
        self.max_gevva_deficit: int = 0

        # Animation timers & visual cues
        self.die_anim_timer: float = 0.0
        self.pig_out_timer: float = 0.0
        self.hold_anim_timer: float = 0.0
        self.ai_think_timer: float = 0.0
        self.banner_text: str = ""
        self.banner_color: tuple[int, int, int] = (255, 215, 0)
        self.banner_timer: float = 0.0

        # Live telemetry for current/last turn
        self.events: deque[TurnEvent] = deque(maxlen=14)
        self.turn_trajectory: list[tuple[int, str, int]] = []  # (k, action, k_star)
        self.last_human_mistake: str | None = None

    @property
    def challenger_name(self) -> str:
        if not self.sim_mode:
            return "YOU (Sir Oinks)"
        return HEURISTIC_PROFILES[self.heuristic_idx][1]

    @property
    def heuristic_hold_target(self) -> int:
        return HEURISTIC_PROFILES[self.heuristic_idx][2]

    def cycle_heuristic(self) -> None:
        self.heuristic_idx = (self.heuristic_idx + 1) % len(HEURISTIC_PROFILES)

    def current_gto(self) -> GTOEvaluation:
        i = self.scores[self.turn]
        j = self.scores[1 - self.turn]
        return self.solver.evaluate_state(i, j, self.turn_bank)

    def new_game(self, alternate_starter: bool = True) -> None:
        if alternate_starter and self.stats.games_played > 0:
            self.starting_player = 1 - self.starting_player
        else:
            self.starting_player = 0
        self.scores = [0, 0]
        self.turn = self.starting_player
        self.turn_bank = 0
        self.turn_rolls = []
        self.winner = None
        self.max_gevva_deficit = 0
        self.die_anim_timer = 0.0
        self.pig_out_timer = 0.0
        self.hold_anim_timer = 0.0
        self.ai_think_timer = 0.25
        self.turn_trajectory.clear()
        self.last_human_mistake = None

    def _check_deficit(self) -> None:
        deficit = self.scores[0] - self.scores[1]
        if deficit > self.max_gevva_deficit:
            self.max_gevva_deficit = deficit

    def execute_roll(self, latency_ms: float = 0.0, animate: bool = True) -> int:
        """Rolls the 6-sided die for the active player."""
        if self.winner is not None:
            return self.last_die

        p = self.turn
        i, j, k = self.scores[p], self.scores[1 - p], self.turn_bank
        gto = self.solver.evaluate_state(i, j, k)

        # Track when Gevva0 warps aggressively past "Hold at 20"
        if p == 1 and k >= 20 and gto.hold_threshold > 20:
            self.stats.gevva_aggressive_warps += 1

        # Track if human rolled when GTO said HOLD
        if p == 0 and not self.sim_mode and k > 0 and gto.optimal_action == "HOLD":
            self.last_human_mistake = (
                f"Over-rolled at k={k} (GTO Hold k*={gto.hold_threshold}, EV {gto.ev_delta*100:+.1f}%)"
            )

        self.turn_trajectory.append((k, "ROLL", gto.hold_threshold))

        die = self.rng.randint(1, 6)
        self.last_die = die
        if animate:
            self.die_anim_timer = 0.22

        if die == 1:
            # PIG OUT! Lose unbanked turn points
            lost = self.turn_bank
            self.turn_bank = 0
            self.turn_rolls = []
            if p == 0:
                self.stats.challenger_busts += 1
            else:
                self.stats.gevva_busts += 1

            self.events.append(
                TurnEvent(
                    player=p,
                    action="PIG_OUT",
                    die_value=1,
                    turn_bank_before=lost,
                    turn_bank_after=0,
                    score_after=self.scores[p],
                    gto_optimal=gto.optimal_action,
                    gto_threshold=gto.hold_threshold,
                    latency_ms=latency_ms,
                    note=f"Rolled 1! Lost {lost} unbanked pts",
                )
            )
            if animate:
                self.pig_out_timer = 0.75
                self.banner_text = f"OINK! ROLLED A 1 (-{lost} PTS)"
                self.banner_color = (255, 75, 110)
                self.banner_timer = 0.85

            # Switch turn
            self.turn = 1 - p
            self.ai_think_timer = 0.32 if animate else 0.0
        else:
            # Roll 2..6: Add to turn bank
            prev_k = self.turn_bank
            self.turn_bank += die
            self.turn_rolls.append(die)
            self.events.append(
                TurnEvent(
                    player=p,
                    action="ROLL",
                    die_value=die,
                    turn_bank_before=prev_k,
                    turn_bank_after=self.turn_bank,
                    score_after=self.scores[p],
                    gto_optimal=gto.optimal_action,
                    gto_threshold=gto.hold_threshold,
                    latency_ms=latency_ms,
                    note=f"Rolled {die} -> Turn pot {self.turn_bank}",
                )
            )
            if animate:
                self.ai_think_timer = 0.26

        return die

    def execute_hold(self, latency_ms: float = 0.0, animate: bool = True) -> None:
        """Banks current turn points permanently into active player's score."""
        if self.winner is not None:
            return
        if self.turn_bank == 0:
            # If k=0, rolling is strictly dominant; force a roll instead of empty hold
            self.execute_roll(latency_ms=latency_ms, animate=animate)
            return

        p = self.turn
        i, j, k = self.scores[p], self.scores[1 - p], self.turn_bank
        gto = self.solver.evaluate_state(i, j, k)

        # Track when Gevva0 chokes out opponent conservatively below 20
        if p == 1 and 10 <= k < 20 and gto.hold_threshold < 20 and i + k < self.goal:
            self.stats.gevva_conservative_chokes += 1

        # Track if human held prematurely when GTO said ROLL
        if p == 0 and not self.sim_mode and gto.optimal_action == "ROLL":
            self.last_human_mistake = (
                f"Held early at k={k} (GTO Target k*={gto.hold_threshold}, EV {-gto.ev_delta*100:+.1f}%)"
            )

        self.turn_trajectory.append((k, "HOLD", gto.hold_threshold))

        self.scores[p] += k
        self.turn_bank = 0
        self.turn_rolls = []
        self._check_deficit()

        if p == 0:
            self.stats.challenger_holds += 1
            self.stats.challenger_banked_pts += k
        else:
            self.stats.gevva_holds += 1
            self.stats.gevva_banked_pts += k

        if self.scores[p] >= self.goal:
            self.winner = p
            self.stats.games_played += 1
            if p == 0:
                self.stats.challenger_wins += 1
            else:
                self.stats.gevva_wins += 1
                if self.max_gevva_deficit >= 25:
                    self.stats.gevva_comeback_wins += 1
            self.stats.win_rate_history.append(self.stats.gevva_win_rate)

            self.events.append(
                TurnEvent(
                    player=p,
                    action="WIN",
                    die_value=None,
                    turn_bank_before=k,
                    turn_bank_after=0,
                    score_after=self.scores[p],
                    gto_optimal=gto.optimal_action,
                    gto_threshold=gto.hold_threshold,
                    latency_ms=latency_ms,
                    note=f"Banked +{k} -> {self.scores[p]} PTS! VICTORY!",
                )
            )
            if animate:
                self.hold_anim_timer = 0.9
                w_name = "GEVVA0 TIER 0" if p == 1 else self.challenger_name
                self.banner_text = f"{w_name} WINS! ({self.scores[p]} PTS)"
                self.banner_color = (50, 245, 160) if p == 1 else (255, 215, 50)
                self.banner_timer = 1.6
        else:
            self.events.append(
                TurnEvent(
                    player=p,
                    action="HOLD",
                    die_value=None,
                    turn_bank_before=k,
                    turn_bank_after=0,
                    score_after=self.scores[p],
                    gto_optimal=gto.optimal_action,
                    gto_threshold=gto.hold_threshold,
                    latency_ms=latency_ms,
                    note=f"Held +{k} -> Score {self.scores[p]}",
                )
            )
            if animate:
                self.hold_anim_timer = 0.45
                self.banner_text = f"BANKED +{k} PTS!"
                self.banner_color = (60, 235, 150)
                self.banner_timer = 0.55
            self.turn = 1 - p
            self.ai_think_timer = 0.32 if animate else 0.0

    def decide_challenger_heuristic(self) -> str:
        """Returns 'ROLL' or 'HOLD' for the Challenger bot in Simulation mode."""
        i, j, k = self.scores[0], self.scores[1], self.turn_bank
        if k == 0:
            return "ROLL"
        if i + k >= self.goal:
            return "HOLD"

        prof_key, _, hold_val = HEURISTIC_PROFILES[self.heuristic_idx]
        if prof_key == "pace_22":
            # Simple human rule: Hold at 22, or roll if opponent >= 82
            if j >= 82:
                return "ROLL"
            return "HOLD" if k >= 22 else "ROLL"

        return "HOLD" if k >= hold_val else "ROLL"

    def step_ai_decision(
        self,
        brain: BasePigBrain,
        animate: bool = True,
        use_turbo_cache: bool = False,
    ) -> DecisionResult | None:
        """Executes one decision step for whichever AI is active (Gevva0 on turn 1, or Heuristic on turn 0 in sim_mode)."""
        if self.winner is not None:
            return None

        if self.turn == 0:
            if not self.sim_mode:
                return None
            act = self.decide_challenger_heuristic()
            if act == "ROLL":
                self.execute_roll(latency_ms=0.0, animate=animate)
            else:
                self.execute_hold(latency_ms=0.0, animate=animate)
            return None

        # Turn == 1: Gevva0 Tier 0 decides [Roll] or [Hold]
        i, j, k = self.scores[1], self.scores[0], self.turn_bank
        if k == 0:
            # Even at k=0, query Tier 0 so telemetry shows the fresh-turn decision
            sit = brain.build_situation(i, j, k, roll_count=len(self.turn_rolls) + 1)
            res = brain.decide(sit, use_turbo_cache=use_turbo_cache)
            self.execute_roll(latency_ms=res.latency_ms, animate=animate)
            return res

        sit = brain.build_situation(i, j, k, roll_count=len(self.turn_rolls) + 1)
        res = brain.decide(sit, use_turbo_cache=use_turbo_cache)
        if res.label == "A":
            self.execute_roll(latency_ms=res.latency_ms, animate=animate)
        else:
            self.execute_hold(latency_ms=res.latency_ms, animate=animate)
        return res

    def run_turbo_games(self, brain: BasePigBrain, num_games: int = 25) -> None:
        """Runs `num_games` complete games at high speed using Tier 0 with turbo regime caching."""
        was_sim = self.sim_mode
        self.sim_mode = True
        try:
            for _ in range(num_games):
                if self.winner is not None:
                    self.new_game(alternate_starter=True)
                steps = 0
                while self.winner is None and steps < 500:
                    self.step_ai_decision(brain, animate=False, use_turbo_cache=True)
                    steps += 1
        finally:
            self.sim_mode = was_sim
