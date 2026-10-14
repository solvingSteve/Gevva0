"""
Gevva0 Game of Pig Brain Adapter powered by Tier0DecisionBridge (Tier 0 Only).

Demonstrates the four core pieces of Tier0Engine without cascading to Tier -1, 1, or 2:
  (1) Direct Cosine Alignment: sim(q, c_i)
  (2) Default 39-Landmark Scaffold (DEFAULT_CORPUS)
  (3) Pig Domain Manifold Scaffold (M = 104 stepping stones, B = (I - Q)^(-1) R)
  (4) Production Hybrid Ensemble: P_final = 0.65 * Cosine + 0.35 * Markov
Plus Tripwire diagnostics: Shannon Entropy H <= 0.65, Logit Margin Delta >= 0.25.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gevva0.bridge import DecisionRequest, DecisionResult, Tier0DecisionBridge
from gevva0.tier0 import DEFAULT_CLASS_ANCHORS, DEFAULT_CORPUS
from gto import GTOEvaluation, get_gto_solver

ONNX_MODEL_DIR = PROJECT_ROOT / "models" / "embeddings" / "embeddinggemma-2-onnx"

# Candidate option descriptions for [A] ROLL and [B] HOLD
CANDIDATE_OPTIONS = [
    "ROLL die again: continue accumulating unbanked turn points, press positive expected value, and take calculated risk below the GTO stopping threshold.",
    "HOLD and bank points now: lock unbanked turn total permanently into score, protect accumulated equity against a 1/6 pig-out reset, and stop at or above the GTO threshold.",
]
CANDIDATE_LABELS = ["A", "B"]  # A = ROLL, B = HOLD

# Domain-specific topological manifold landmarks (M = 104) spanning the 3D (i, j, k) Pig state space
PIG_DOMAIN_CORPUS: list[str] = [
    # --- Cluster 1: Fresh Turn & Low Turn Bank (Roll / Accumulate) ---
    "Turn bank k=0 at score (0 vs 0). Zero unbanked points at risk; must roll die to initiate turn accumulation and gain positive expected value.",
    "Active turn bank is zero with no points at risk. Roll the die immediately to start building turn equity.",
    "Small unbanked turn total k=4 is well below GTO threshold k*=21. Bust penalty is tiny while expected gain is high; continue rolling to build turn bank.",
    "Small unbanked turn total k=6 is far below optimal stopping threshold. Roll again to accumulate points.",
    "Small unbanked turn total k=8 with low downside risk; positive expected value to continue rolling the die.",
    "Small unbanked turn total k=10 is below GTO threshold k*=20. Continue rolling to press turn accumulation.",
    "Small unbanked turn total k=12 is below optimal hold boundary; roll again to build a stronger turn pot.",
    "Turn pot is small (under 14 points) and opponent is not threatening; roll die to accumulate more points.",
    "Early game state with modest turn bank; expected value strongly favors rolling again over premature holding.",
    "Low turn bank at risk; rolling a 2 through 6 adds 4.0 average points against minimal loss on a 1. Roll again.",
    "Turn accumulation phase: current unbanked total is well under the Markov stopping boundary. Keep rolling.",
    "Mid-game balanced score with only 9 unbanked points; do not hold prematurely, roll again for higher expected value.",
    "Only 11 points in turn bank and 65 points needed for 100; continue rolling to maximize win probability.",
    "Turn bank k=13 is below GTO threshold k*=22; continue rolling to reach optimal stopping target.",
    "Fresh turn with small pot k=5; holding now wastes turn tempo. Roll die again.",
    "Low risk turn state: unbanked points are small relative to score gap. Roll again to press expected value.",

    # --- Cluster 2: Trailing Deficit & Aggressive Catch-Up (Roll) ---
    "Trailing opponent by a large deficit and below warped GTO hold threshold; must take calculated aggressive risks and roll again.",
    "Opponent at 82 threatens imminent victory while player trails at 48+18=66 (below GTO target 38). Holding surrenders turn; must roll aggressively to catch up.",
    "Opponent at 88 is one turn away from winning; holding a small bank cannot win. Roll aggressively Toward 100.",
    "Desperate deficit chase: opponent has 85 points, player is far behind. GTO threshold warps upward; continue rolling.",
    "Trailing opponent by 30 points (score 35 vs 65) with turn bank k=20 below warped GTO threshold k*=29. Press advantage and roll again.",
    "Trailing opponent by 25 points; human 'Hold at 20' heuristic fails here. Must roll past 20 to reach warped GTO target.",
    "Opponent leads 74 to 42; standard 20-point hold is mathematically suboptimal. Continue rolling to close the score differential.",
    "High opponent score forces aggressive variance seeking; roll again to push total toward 100.",
    "Deficit press required: holding leaves opponent with commanding win probability. Roll die to stay competitive.",
    "Behind in late game (opponent 78, player 54, turn bank 21 < GTO target 34); take aggressive calculated risk and roll.",
    "Score differential is negative and turn total has not reached the elevated catch-up threshold. Roll again.",
    "Chasing a big lead: Markov absorption favors continuing the turn to overcome the point deficit. Roll die.",
    "Opponent is in the 80s; banking a medium pot still loses on the next turn. Keep rolling for a comeback.",
    "Aggressive GTO warp active: because player trails significantly, optimal stopping threshold rises above 28. Roll again.",
    "Trailing by 18 points in mid-game with k=17 below k*=25; continue rolling to narrow the gap.",
    "Catch-up mode: positive expected win-probability delta from rolling again despite moderate turn bank.",

    # --- Cluster 3: Boundary & Positive-EV Continuation (Roll) ---
    "Active turn bank k=16 is below optimal stopping threshold k*=21 at even score. Positive expected value to continue rolling.",
    "Active turn bank k=18 is below GTO threshold k*=23; rolling yields higher win probability than holding.",
    "Turn bank k=15 approaching GTO threshold k*=20; rolling holds a clear Markov EV edge. Take another roll.",
    "Turn bank k=19 is still 3 points shy of the dynamic GTO stopping boundary k*=22; roll one more time.",
    "Expected win probability for ROLL exceeds HOLD; continue rolling to reach the optimal stopping curve.",
    "Markov fundamental matrix shows positive drift for rolling at current score differential; roll again.",
    "Below the Todd & Fisher optimal stopping surface; rolling maximizes long-run absorption into the win state.",
    "Turn bank has not yet crossed the critical risk-reward inflection point; roll the die again.",
    "Score is tied 50 vs 50 with turn bank 16 < GTO target 20; roll again for maximum win equity.",
    "Player at 20 vs 20 with turn bank 17 < GTO target 22; continue rolling to build optimal bank.",
    "Player at 60 vs 68 with turn bank 18 < GTO target 24; roll again to overtake opponent.",
    "Marginal boundary roll: k is just below k*(i, j) where expected gain from 2..6 still outweighs 1/6 bust risk.",
    "Continue rolling: unbanked pot has not reached the risk saturation limit for this score pair.",
    "Optimal stopping threshold not yet met; keep rolling the 6-sided die.",
    "Positive EV action is ROLL: gain additional turn points before banking.",
    "Calculated continuation: roll again while below the dynamic Markov hold boundary.",

    # --- Cluster 4: Instant Win Clinch & Endgame Lock (Hold) ---
    "Game-winning threshold reached (i + k >= 100). Hold immediately to bank 100+ points and secure guaranteed 100% victory.",
    "Turn total plus banked score reaches 100 points! Hold now to clinch immediate victory with zero risk.",
    "Player has enough unbanked points to cross the 100-point finish line. Hold immediately and win the game.",
    "Score 84 plus turn bank 18 equals 102 (>= 100). Never roll when already at 100; hold to lock 100% win.",
    "Score 92 plus turn bank 9 equals 101 (>= 100). Hold immediately to bank winning points.",
    "Clinch victory now: holding banks 100 points and ends the match immediately.",
    "Guaranteed win on the table (i + k >= 100); rolling would recklessly risk a 1/6 bust. Hold now.",
    "Endgame finish line crossed: bank turn points immediately to claim victory.",

    # --- Cluster 5: Ahead Conservative Choke-Out (Hold) ---
    "Commanding lead (82+15=97 vs opponent 45) and turn bank k=15 >= conservative GTO threshold k*=14. Hold now to lock in points and choke out opponent.",
    "Ahead with high score near 100 while opponent is far behind; GTO threshold warps downward. Hold conservatively to protect lead.",
    "Player leads 78 to 38 with turn bank k=16 >= GTO threshold k*=15. Choke out the opponent with a conservative hold.",
    "When ahead late in the game, avoid unnecessary bust risk; bank the moderate turn pot now to push score into the 90s.",
    "Conservative GTO choke strategy: player is well ahead at 75+16=91 vs 50; hold now to force opponent into desperate low-probability rolls.",
    "Leading comfortably in the endgame; holding locks a commanding 88+ score and starves opponent of comeback equity.",
    "Ahead by 30 points (72 vs 42) with turn bank 17 >= warped GTO threshold 16; hold and bank points safely.",
    "Protect commanding lead: downside of rolling a 1 and giving opponent a free tempo far outweighs small point gain. Hold now.",
    "Front-runner risk management: GTO stopping boundary drops below 18 when holding a big lead near 85. Hold and bank.",
    "Lock in the lead and apply scoreboard pressure; hold current turn points permanently.",
    "Ahead 80 to 55 with k=15 >= k*=14: bank points now so player only needs one short turn to win.",
    "Conservative hold optimal: preserve high win probability by banking points and passing turn from a dominant position.",

    # --- Cluster 6: High Turn Pot Bust Protection (Hold) ---
    "Large unbanked turn pot k=28 exceeds GTO stopping threshold k*=21. A 1/6 pig-out roll would wipe out massive unbanked equity; hold immediately to bank points permanently.",
    "Massive turn bank k=32 at risk! Rolling a 1 loses 32 points while rolling 2..6 only gains 4.0 on average; hold now.",
    "High pot bust protection: unbanked turn total k=26 is well above the optimal stopping threshold. Bank points immediately.",
    "Turn bank k=30 exceeds risk tolerance; lock in the huge turn haul before a 1/6 pig-out destroys the turn.",
    "Unbanked pot has grown to 25+ points, surpassing the Markov stopping boundary; hold to secure permanent score gain.",
    "Catastrophic reset risk: risking 27 unbanked points for a marginal roll is negative EV. Hold and bank now.",
    "Turn pot k=24 > GTO threshold k*=20 at even score; stop rolling and bank the 24 points permanently.",
    "Turn pot k=29 is deep in the hold zone; protect accumulated points and end turn safely.",
    "Heavy turn bank accumulated: expected loss from a 1/6 pig-out (-4.5 pts/roll) exceeds expected gain (+3.3 pts/roll). Hold now.",
    "Do not get greedy with a 26-point turn bank; GTO optimal stopping dictates holding immediately.",
    "Bank the large pot now: convert volatile unbanked turn points into permanent scoreboard equity.",
    "Risk ceiling exceeded (k >= 25): hold immediately to prevent a catastrophic 1 roll.",

    # --- Cluster 7: Standard Optimal Stopping Boundary Crossed (Hold) ---
    "Turn bank k=21 has reached optimal stopping threshold k*=21 (score 40->61 vs 40). Downside bust risk outweighs marginal roll gain; hold to bank turn points safely.",
    "Turn bank k=22 just crossed GTO stopping boundary k*=21 at opening score. Holding locks points with higher win probability; bank points now.",
    "Turn bank k=20 meets the exact GTO hold threshold k*=20 at mid-game score (50 vs 48); hold now to bank points.",
    "Turn bank k=23 >= GTO threshold k*=22; optimal stopping criterion satisfied. Hold and bank.",
    "Markov absorption probability favors HOLD over ROLL: current turn pot has reached the optimal stopping surface.",
    "Stopping threshold reached (k >= k*): holding maximizes total match win probability. Bank points now.",
    "Boundary threshold hold: one more roll carries negative expected win-probability delta. Hold immediately.",
    "Turn bank k=19 >= GTO threshold k*=18 at score (62 vs 55); bank the 19 points to reach 81.",
    "Turn bank k=20 >= GTO threshold k*=19 at score (58 vs 52); hold now for optimal expected value.",
    "Turn bank k=22 >= GTO threshold k*=21 at score (30 vs 35); lock in points permanently.",
    "Optimal stopping point achieved: transition to absorbing HOLD sink to bank turn points.",
    "Risk-reward balance tips to HOLD: unbanked points are sufficient for this score differential.",
    "Hold at the mathematically optimal boundary: bank points and pass the die.",
    "GTO stopping condition met: secure the turn bank and protect win equity.",
    "Hold now: expected win rate from banking exceeds expected win rate from rolling again.",
    "Lock in turn points at the Todd & Fisher Markov boundary; choose HOLD.",
    "Bank points safely: k has crossed the dynamic stopping threshold for (i, j).",
    "Stop rolling and hold: preserve turn gains and advance banked score.",
    "Hold to bank points: downside 1/6 reset risk now dominates upside roll potential.",
    " disciplined GTO hold: bank current turn points into permanent score.",
    "Hold and bank: turn accumulation target reached for current score differential.",
    "Select HOLD: optimal stopping boundary crossed with positive safety margin.",
    "Bank turn total now: secure points and transition turn.",
    "Optimal decision is HOLD: lock unbanked points permanently.",
]

PIG_CLASS_ANCHORS: dict[str, list[str]] = {
    "A": [
        "ROLL die again to accumulate more turn points below the GTO stopping threshold",
        "Fresh turn k=0 or small turn bank: positive expected value to continue rolling",
        "Trailing opponent deficit chase: take aggressive calculated risk and roll again",
        "Below optimal stopping boundary k*: roll die to press win probability",
    ],
    "B": [
        "HOLD and bank turn points permanently into score at or above the GTO threshold",
        "High unbanked turn pot bust protection: stop rolling and lock in points safely",
        "Ahead conservative choke-out or 100-point win clinch: hold immediately",
        "Optimal stopping boundary k* reached: bank points now to avoid 1/6 pig-out reset",
    ],
}


@dataclass
class PigSituation:
    """Represents a single decision point (i, j, k) on Gevva0's turn."""
    i: int
    j: int
    k: int
    roll_count: int
    gto: GTOEvaluation
    query_text: str
    serial: int = 0


class BasePigBrain:
    name = "Tier0-Pig"
    simulated = False

    def __init__(self):
        self.ready = False
        self.loading = True
        self.error: str | None = None
        self.model_name = "EmbeddingGemma-2 ONNX (300M, 768D)"
        self.engine_mode = "hybrid_domain"  # hybrid_domain | domain_markov | default_39 | pure_cosine
        self.latencies: deque[float] = deque(maxlen=500)
        self.embed_latencies: deque[float] = deque(maxlen=500)
        self.graph_latencies: deque[float] = deque(maxlen=500)
        self.total_decisions = 0
        self.gto_agreements = 0
        self.exit0_count = 0
        self.last_result: DecisionResult | None = None
        self.last_situation: PigSituation | None = None
        self.history: deque[tuple[PigSituation, DecisionResult]] = deque(maxlen=60)
        self._serial_counter = 0

    def next_serial(self) -> int:
        self._serial_counter += 1
        return self._serial_counter

    def latency_stats(self) -> tuple[float, float]:
        if not self.latencies:
            return 0.0, 0.0
        arr = np.asarray(self.latencies)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))

    @property
    def gto_match_rate(self) -> float:
        if self.total_decisions == 0:
            return 100.0
        return (self.gto_agreements / self.total_decisions) * 100.0

    @property
    def exit0_rate(self) -> float:
        if self.total_decisions == 0:
            return 100.0
        return (self.exit0_count / self.total_decisions) * 100.0

    def build_situation(self, i: int, j: int, k: int, roll_count: int = 1) -> PigSituation:
        solver = get_gto_solver()
        gto_eval = solver.evaluate_state(i, j, k)
        query_text = gto_eval.regime_description
        return PigSituation(
            i=i,
            j=j,
            k=k,
            roll_count=roll_count,
            gto=gto_eval,
            query_text=query_text,
            serial=self.next_serial(),
        )

    def _record(self, sit: PigSituation, res: DecisionResult) -> None:
        self.latencies.append(res.latency_ms)
        self.embed_latencies.append(res.embed_latency_ms)
        self.graph_latencies.append(res.graph_latency_ms)
        self.total_decisions += 1
        chosen_act = "ROLL" if res.label == "A" else "HOLD"
        if chosen_act == sit.gto.optimal_action:
            self.gto_agreements += 1
        if res.tripwire_passed:
            self.exit0_count += 1
        self.last_situation = sit
        self.last_result = res
        self.history.append((sit, res))

    def shutdown(self) -> None:
        pass


class PigTier0Brain(BasePigBrain):
    """Real GPU/CPU ONNX Tier 0 Brain backed by Tier0DecisionBridge (src/gevva0/bridge.py)."""

    name = "Gevva0-Tier0"
    simulated = False

    def __init__(
        self,
        model_path: Path | str | None = None,
        engine_mode: str = "hybrid_domain",
        device: str | None = None,
    ):
        super().__init__()
        self.model_path = Path(model_path) if model_path else ONNX_MODEL_DIR
        self.engine_mode = engine_mode
        self.device = device
        self.bridge: Tier0DecisionBridge | None = None
        self._turbo_cache: dict[tuple[str, str], DecisionResult] = {}
        self._current_poll: Callable[[], DecisionResult | None] | None = None
        self._pending_sit: PigSituation | None = None

    def load(self) -> None:
        try:
            t0 = time.perf_counter()
            corpus = DEFAULT_CORPUS if self.engine_mode == "default_39" else PIG_DOMAIN_CORPUS
            anchors = DEFAULT_CLASS_ANCHORS if self.engine_mode == "default_39" else PIG_CLASS_ANCHORS
            m_weight = self._resolve_markov_weight(self.engine_mode)

            self.bridge = Tier0DecisionBridge(
                model_path=self.model_path,
                device=self.device,
                corpus=corpus,
                class_anchors=anchors,
                markov_weight=m_weight,
                tau_entropy=0.65,
                tau_margin=0.25,
                tau_ood=0.30,
                tau_temp=0.05,
                p_landmarks=6,
                enable_vision=False,
                enable_audio=False,
            )
            self.bridge.warmup(rounds=2)
            self.bridge.start_worker()
            dev_str = str(getattr(self.bridge.tier0, "device", "cuda")).upper()
            backend_str = str(getattr(self.bridge.tier0, "backend", "onnx")).upper()
            self.model_name = f"EmbeddingGemma-2 ({backend_str} on {dev_str}, M={self.bridge.tier0.M})"
            self.ready = True
            print(
                f"[Gevva0 Pig Tier 0] Bridge online in {time.perf_counter() - t0:.2f}s | "
                f"Mode={self.engine_mode} | Landmarks M={self.bridge.tier0.M} | Device={dev_str}"
            )
        except Exception as exc:
            self.error = str(exc)
            print(f"[Gevva0 Pig Tier 0] Init failed ({exc}); falling back to SimulatedTier0Brain")
        finally:
            self.loading = False

    def load_async(self) -> None:
        threading.Thread(target=self.load, daemon=True, name="pig-tier0-loader").start()

    @staticmethod
    def _resolve_markov_weight(mode: str) -> float:
        if mode == "pure_cosine":
            return 0.0
        if mode == "domain_markov":
            return 1.0
        return 0.35  # hybrid_domain (65% Cosine + 35% Markov) or default_39

    def set_engine_mode(self, mode: str) -> None:
        """Switches live between the 4 Tier0Engine formulations:
        - 'hybrid_domain': 0.65 Cosine + 0.35 Domain Manifold AMC (M=104)
        - 'domain_markov': 1.00 Domain Manifold AMC (B = (I - Q)^(-1) R)
        - 'default_39':    0.65 Cosine + 0.35 Default 39-Landmark AMC
        - 'pure_cosine':   1.00 Direct Cosine Alignment
        """
        if mode == self.engine_mode and self.bridge is not None:
            return
        prev_mode = self.engine_mode
        self.engine_mode = mode
        if self.bridge is None:
            return

        need_rebuild = (prev_mode == "default_39") != (mode == "default_39")
        if need_rebuild:
            if mode == "default_39":
                self.bridge.rebuild_scaffold(DEFAULT_CORPUS, DEFAULT_CLASS_ANCHORS)
            else:
                self.bridge.rebuild_scaffold(PIG_DOMAIN_CORPUS, PIG_CLASS_ANCHORS)

        self.bridge.markov_weight = self._resolve_markov_weight(mode)
        dev_str = str(getattr(self.bridge.tier0, "device", "cuda")).upper()
        backend_str = str(getattr(self.bridge.tier0, "backend", "onnx")).upper()
        self.model_name = f"EmbeddingGemma-2 ({backend_str} on {dev_str}, M={self.bridge.tier0.M})"

    def _build_request(self, sit: PigSituation) -> DecisionRequest:
        return DecisionRequest(
            state_text=sit.query_text,
            options=CANDIDATE_OPTIONS,
            custom_labels=CANDIDATE_LABELS,
            allow_cot=False,
            serial=sit.serial,
            metadata={
                "sit": sit,
                "markov_weight": self._resolve_markov_weight(self.engine_mode),
            },
        )

    def decide(self, sit: PigSituation, use_turbo_cache: bool = False) -> DecisionResult:
        """Synchronous Tier 0 decision (~2 ms on GPU, or <0.05 ms when turbo cache is enabled for batch sim)."""
        if self.bridge is None:
            raise RuntimeError("Tier0DecisionBridge is not initialized.")

        # In high-speed batch simulation mode, cache by (engine_mode, canonical_regime_bucket)
        # so the first hit runs real ONNX + AMC inference and subsequent hits in the same
        # tactical bucket execute instantaneously while preserving real Tier 0 probabilities.
        cache_key = (self.engine_mode, f"{sit.gto.regime}_{sit.k // 4}_{(sit.i - sit.j) // 10}")
        if use_turbo_cache and cache_key in self._turbo_cache:
            cached = self._turbo_cache[cache_key]
            res = DecisionResult(
                choice_index=cached.choice_index,
                label=cached.label,
                choice_text=cached.choice_text,
                probs=dict(cached.probs),
                confidence=cached.confidence,
                latency_ms=cached.latency_ms,
                mode=cached.mode,
                fast_letter=cached.fast_letter,
                fast_confidence=cached.fast_confidence,
                metadata={"sit": sit, "cached": True},
                serial=sit.serial,
                tier=0,
                entropy=cached.entropy,
                margin=cached.margin,
                cosine_probs=dict(cached.cosine_probs) if cached.cosine_probs else None,
                markov_probs=dict(cached.markov_probs) if cached.markov_probs else None,
                cosine_sims=dict(cached.cosine_sims) if cached.cosine_sims else None,
                embed_latency_ms=cached.embed_latency_ms,
                graph_latency_ms=cached.graph_latency_ms,
                tripwire_passed=cached.tripwire_passed,
                tripwire_reason=cached.tripwire_reason,
                scaffold_mode=cached.scaffold_mode,
                landmarks_count=cached.landmarks_count,
            )
            self._record(sit, res)
            return res

        req = self._build_request(sit)
        res = self.bridge.decide(req)
        if use_turbo_cache:
            self._turbo_cache[cache_key] = res
        self._record(sit, res)
        return res

    def submit(self, sit: PigSituation) -> None:
        if self.bridge is None:
            return
        req = self._build_request(sit)
        self._pending_sit = sit
        self._current_poll = self.bridge.submit_async(req)

    def poll(self) -> tuple[PigSituation, DecisionResult] | None:
        if self._current_poll is None:
            return None
        res = self._current_poll()
        if res is not None:
            self._current_poll = None
            sit = res.metadata.get("sit", self._pending_sit)
            self._pending_sit = None
            if sit is not None:
                self._record(sit, res)
                return (sit, res)
        return None

    def shutdown(self) -> None:
        super().shutdown()
        if self.bridge is not None:
            self.bridge.close()


class SimulatedTier0Brain(BasePigBrain):
    """GPU-free stand-in that emulates the 4 Tier0Engine components (Cosine, AMC Graph, 65/35 Hybrid, Tripwires)
    using the exact Todd & Fisher (2007) Absorbing Markov Chain values in ~1.8 ms.
    """

    name = "Tier0-Sim"
    simulated = True

    def __init__(self, engine_mode: str = "hybrid_domain", seed: int = 42):
        super().__init__()
        self.engine_mode = engine_mode
        self.rng = random.Random(seed)
        self.model_name = "Tier 0 Simulated (AMC + Cosine Stand-In)"
        self._pending: tuple[PigSituation, DecisionResult] | None = None

    def load(self) -> None:
        get_gto_solver()
        self.ready = True
        self.loading = False

    def load_async(self) -> None:
        self.load()

    def set_engine_mode(self, mode: str) -> None:
        self.engine_mode = mode

    def decide(self, sit: PigSituation, use_turbo_cache: bool = False) -> DecisionResult:
        gto = sit.gto
        # Emulate (1) Direct Cosine and (2)/(3) Markov Absorption
        # Direct cosine has slight softness near boundary; Domain AMC sharpens the boundary
        ev_diff = gto.ev_delta
        if gto.optimal_action == "ROLL":
            base_adv = max(0.04, abs(ev_diff) * 18.0 + 0.14)
        else:
            base_adv = -max(0.04, abs(ev_diff) * 18.0 + 0.14)

        cos_sim_a = float(np.clip(0.62 + 0.14 * np.tanh(base_adv * 2.2), 0.35, 0.92))
        cos_sim_b = float(np.clip(0.62 - 0.14 * np.tanh(base_adv * 2.2), 0.35, 0.92))

        tau_temp = 0.05
        c_logits = np.array([cos_sim_a / tau_temp, cos_sim_b / tau_temp], dtype=np.float64)
        c_exp = np.exp(c_logits - np.max(c_logits))
        c_probs = c_exp / np.sum(c_exp)

        if self.engine_mode == "default_39":
            # Default 39-landmark corpus has weaker domain separation
            m_logits = np.array([0.3 * base_adv, -0.3 * base_adv], dtype=np.float64)
            landmarks_cnt = 39
            scaffold = "default_39_landmarks"
        else:
            # Domain manifold (M=104) aligns closely with exact absorbing Markov chain
            m_logits = np.array([4.2 * base_adv, -4.2 * base_adv], dtype=np.float64)
            landmarks_cnt = len(PIG_DOMAIN_CORPUS)
            scaffold = "domain_manifold" if self.engine_mode != "pure_cosine" else "pure_cosine"

        m_exp = np.exp(m_logits - np.max(m_logits))
        m_probs = m_exp / np.sum(m_exp)

        if self.engine_mode == "pure_cosine":
            probs_arr = c_probs
            mode_str = "tier0_cosine"
        elif self.engine_mode == "domain_markov":
            probs_arr = m_probs
            mode_str = "tier0_markov"
        else:
            c_norm = c_logits - np.mean(c_logits)
            m_norm = m_logits - np.mean(m_logits)
            comb = 0.65 * c_norm + 0.35 * m_norm
            e_comb = np.exp(comb - np.max(comb))
            probs_arr = e_comb / np.sum(e_comb)
            mode_str = "tier0_hybrid"

        top_idx = int(np.argmax(probs_arr))
        top1 = float(probs_arr[top_idx])
        top2 = float(probs_arr[1 - top_idx])
        margin = top1 - top2
        entropy = float(-np.sum(probs_arr * np.log(probs_arr + 1e-9)) / np.log(2.0))

        embed_ms = self.rng.uniform(1.45, 2.05)
        graph_ms = self.rng.uniform(0.08, 0.18)
        total_ms = round(embed_ms + graph_ms, 2)

        tripwire_ok = (entropy <= 0.65) and (margin >= 0.25)
        reason = None
        if not tripwire_ok:
            reason = "HIGH_AMBIGUITY" if entropy > 0.65 else "LOW_MARGIN"

        lbl = CANDIDATE_LABELS[top_idx]
        res = DecisionResult(
            choice_index=top_idx,
            label=lbl,
            choice_text=CANDIDATE_OPTIONS[top_idx],
            probs={"A": round(float(probs_arr[0]), 4), "B": round(float(probs_arr[1]), 4)},
            confidence=round(top1, 4),
            latency_ms=total_ms,
            mode=mode_str,
            fast_letter=lbl,
            fast_confidence=round(top1, 4),
            metadata={"sit": sit},
            serial=sit.serial,
            tier=0,
            entropy=round(entropy, 4),
            margin=round(margin, 4),
            cosine_probs={"A": round(float(c_probs[0]), 4), "B": round(float(c_probs[1]), 4)},
            markov_probs={"A": round(float(m_probs[0]), 4), "B": round(float(m_probs[1]), 4)},
            cosine_sims={"A": round(cos_sim_a, 4), "B": round(cos_sim_b, 4)},
            embed_latency_ms=round(embed_ms, 3),
            graph_latency_ms=round(graph_ms, 3),
            tripwire_passed=tripwire_ok,
            tripwire_reason=reason,
            scaffold_mode=scaffold,
            landmarks_count=landmarks_cnt,
        )
        self._record(sit, res)
        return res

    def submit(self, sit: PigSituation) -> None:
        res = self.decide(sit)
        self._pending = (sit, res)

    def poll(self) -> tuple[PigSituation, DecisionResult] | None:
        p, self._pending = self._pending, None
        return p
