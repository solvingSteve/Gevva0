"""
Gevva0 Game of Blackjack Brain Adapter powered by Tier0DecisionBridge (Tier 0 Only).

Demonstrates the four core pieces of Tier0Engine without cascading to Tier -1, 1, or 2:
  (1) Direct Cosine Alignment: sim(q, c_i)
  (2) Default 39-Landmark Scaffold (DEFAULT_CORPUS)
  (3) Blackjack Domain Manifold Scaffold (M = 116 stepping stones, B = (I - Q)^(-1) R)
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
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gevva0.bridge import DecisionRequest, DecisionResult, Tier0DecisionBridge
from gevva0.tier0 import DEFAULT_CLASS_ANCHORS, DEFAULT_CORPUS
from gto import BlackjackGTOEvaluation, get_blackjack_solver

ONNX_MODEL_DIR = PROJECT_ROOT / "models" / "embeddings" / "embeddinggemma-2-onnx"

# 4 Canonical Blackjack Candidate Actions: [A] HIT, [B] STAND, [C] DOUBLE, [D] SPLIT
CANDIDATE_OPTIONS = [
    "HIT to draw another card: low hand total, stiff hand against strong dealer upcard (7, 8, 9, 10, Ace), or Soft 18 vs strong dealer.",
    "STAND and hold current total: pat hand (17, 18, 19, 20, 21), stiff hand (12-16) against weak dealer bust upcard (2-6), or True Count stand warp.",
    "DOUBLE DOWN 2x wager for one card: powerhouse Hard 9, 10, or 11, or flexible Soft 13-18 against weak dealer upcard to press maximum positive EV.",
    "SPLIT pair into two independent hands: always split Aces and 8s, or split pairs (2s, 3s, 6s, 7s, 9s) against weak dealer upcards.",
]
CANDIDATE_LABELS = ["A", "B", "C", "D"]
LABEL_TO_ACTION = {"A": "HIT", "B": "STAND", "C": "DOUBLE", "D": "SPLIT"}
ACTION_TO_LABEL = {"HIT": "A", "STAND": "B", "DOUBLE": "C", "SPLIT": "D"}

# Domain-specific topological manifold landmarks (M = 116) spanning the Blackjack state space
BLACKJACK_DOMAIN_CORPUS: list[str] = [
    # --- Cluster 1: [A] HIT — Low Totals, Stiff vs Strong Dealer, & Soft Hand Hits (32 Landmarks) ---
    "HIT to draw another card: low total Hard 5, 6, 7, or 8 vs any Dealer upcard. Zero bust risk; build stronger hand total.",
    "HIT to draw another card: low total Hard 8 vs Dealer 6. Cannot bust on next card; draw to reach pat range.",
    "HIT to draw another card: Hard 9 vs Dealer 2, 7, 8, 9, 10, or Ace. Dealer is too strong to double; draw one card safely.",
    "HIT to draw another card: Hard 10 vs Dealer 10 or Ace. High dealer pat probability makes doubling too risky; hit instead.",
    "HIT to draw another card: Hard 11 vs Dealer Ace in neutral shoe. Draw another card to reach 21.",
    "HIT to draw another card: 3-card Hard 10 or 11 where double down is unavailable; draw another card for 20 or 21.",
    "HIT to draw another card: Hard 12 vs Dealer 2. Only 10-value cards bust 12 (4/13 risk); hitting has higher EV than standing.",
    "HIT to draw another card: Hard 12 vs Dealer 3. Dealer 3 is not weak enough for standing on 12; draw a card.",
    "HIT to draw another card: stiff Hard 12 vs strong Dealer 7, 8, 9, 10, or Ace. Dealer likely makes 17-21; must hit to compete.",
    "HIT to draw another card: stiff Hard 13 vs strong Dealer 7, 8, 9, 10, or Ace. Standing loses heavily; draw to improve.",
    "HIT to draw another card: stiff Hard 14 vs strong Dealer 7, 8, 9, 10, or Ace. Must hit defensive stiff hand against strong dealer.",
    "HIT to draw another card: stiff Hard 15 vs strong Dealer 7, 8, 9, 10, or Ace. Hitting minimizes expected loss against pat dealer.",
    "HIT to draw another card: stiff Hard 16 vs strong Dealer 7, 8, or 9. Dealer likely makes pat total; hit to salvage equity.",
    "HIT to draw another card: stiff Hard 16 vs Dealer 10 or Ace in neutral/negative shoe. Hitting beats standing despite bust risk.",
    "HIT to draw another card: Hard 12 vs Dealer 4 in negative True Count (ten-poor shoe). Safe to draw small card.",
    "HIT to draw another card: Soft 13 or Soft 14 vs Dealer 2, 3, 4, 7, 8, 9, 10, Ace. Zero bust risk with soft Ace cushion.",
    "HIT to draw another card: Soft 15 or Soft 16 vs Dealer 2, 3, 7, 8, 9, 10, Ace. Free draw with flexible Ace.",
    "HIT to draw another card: Soft 17 (Ace-6) vs Dealer 2, 7, 8, 9, 10, Ace. Never stand on Soft 17; draw to improve.",
    "HIT to draw another card: Soft 18 vs strong Dealer 9, 10, or Ace. Standing on 18 is negative EV against 19/20; hit with Ace cushion.",
    "HIT to draw another card: multi-card Soft 16 or Soft 17 vs weak dealer when doubling is not allowed; draw freely.",
    "HIT to draw another card: Pair of 2s, 3s, 6s, or 7s vs strong Dealer 8, 9, 10, Ace. Do not split into strong dealer; hit.",
    "HIT to draw another card: Pair of 4s (Hard 8) vs Dealer 2, 3, 4, 7, 8, 9, 10, Ace. Treat as Hard 8 and draw.",
    "HIT to draw another card: low total Hard 7 vs Dealer 10. Zero bust risk; build stronger hand total.",
    "HIT to draw another card: stiff Hard 13 vs Dealer 10. Dealer likely makes 20; must hit to compete.",
    "HIT to draw another card: stiff Hard 14 vs Dealer 9. Standing loses 77% of the time; draw another card.",
    "HIT to draw another card: stiff Hard 15 vs Dealer 10. Defensive hit recovers crucial expected value.",
    "HIT to draw another card: Soft 18 vs Dealer 10. Hitting Soft 18 upgrades hand without single-card bust risk.",
    "HIT to draw another card: Soft 18 vs Dealer Ace. Draw another card to beat dealer pat 19-21.",
    "HIT to draw another card: Hard 12 vs Dealer 8. Low bust probability (30.8%); hit to reach pat 17-21.",
    "HIT to draw another card: Hard 13 vs Dealer 7. Dealer 7 makes 17+ frequently; hit stiff 13.",
    "HIT to draw another card: Hard 16 vs Dealer 7. Hitting rescues equity against dealer 17.",
    "HIT to draw another card: low total Hard 6 vs Dealer 5. Zero bust risk; draw card.",

    # --- Cluster 2: [B] STAND — Pat Hands, Dealer Bust Traps, & True Count Warps (32 Landmarks) ---
    "STAND and hold pat total: strong Hard 17 vs any Dealer upcard (2 through Ace). Bust risk is 69%; never hit Hard 17.",
    "STAND and hold pat total: strong Hard 18 vs any Dealer upcard. Lock in pat 18 and never risk busting.",
    "STAND and hold pat total: commanding Hard 19 vs any Dealer upcard. High positive EV; stand immediately.",
    "STAND and hold pat total: dominant Hard 20 or Pair of 10s vs any Dealer upcard. Never split 10s; lock ~80% win rate.",
    "STAND and hold pat total: maximum total 21 vs any Dealer upcard. Lock unbeatable total.",
    "STAND and hold pat total: Soft 19 (Ace-8) or Soft 20 (Ace-9) vs Dealer upcard. Pat soft total locks high positive EV.",
    "STAND and hold pat total: Soft 18 vs Dealer 2, 7, or 8. When doubling is unavailable or vs 2/7/8, stand on pat 18.",
    "STAND and hold pat total: Pair of 9s (18) vs Dealer 7, 10, or Ace. Against 7, 18 already beats dealer 17; stand.",
    "STAND and let dealer bust: stiff Hard 12 vs weak Dealer 4, 5, or 6 (bust 40-42%). Avoid player bust and trap dealer.",
    "STAND and let dealer bust: stiff Hard 13 vs weak Dealer 2, 3, 4, 5, or 6. Let dealer draw into stiff bust trap.",
    "STAND and let dealer bust: stiff Hard 14 vs weak Dealer 2, 3, 4, 5, or 6. Preserve player hand while dealer busts.",
    "STAND and let dealer bust: stiff Hard 15 vs weak Dealer 2, 3, 4, 5, or 6. Do not bust before the dealer draws.",
    "STAND and let dealer bust: stiff Hard 16 vs weak Dealer 2, 3, 4, 5, or 6 (bust 42%). Classic dealer bust trap; stand.",
    "STAND on high True Count warp (TC +2.0): Hard 16 vs Dealer 10. Rich ten-shoe busts player hit; Illustrious 18 stand.",
    "STAND on high True Count warp (TC +4.5): Hard 15 vs Dealer 10. Elevated ten density makes standing superior to hitting.",
    "STAND on high True Count warp (TC +5.0): Hard 16 vs Dealer 9. High positive True Count warps basic strategy to STAND.",
    "STAND on high True Count warp (TC +2.5): Hard 12 vs Dealer 3. Ten-rich shoe increases both player bust and dealer bust.",
    "STAND on high True Count warp (TC +3.5): Hard 12 vs Dealer 2. Positive True Count index deviation dictates standing.",
    "STAND and let dealer bust: Hard 13 vs Dealer 5 (bust 42%). Never risk busting against dealer 5.",
    "STAND and let dealer bust: Hard 14 vs Dealer 6 (bust 42%). Lock hand and watch dealer draw stiff cards.",
    "STAND and let dealer bust: Hard 15 vs Dealer 4 (bust 40%). Dealer bust probability protects stiff stand.",
    "STAND and let dealer bust: Hard 16 vs Dealer 6 (bust 42%). Standing captures full dealer bust equity.",
    "STAND and let dealer bust: Hard 12 vs Dealer 6. Optimal stopping against weakest dealer upcard.",
    "STAND and hold pat total: Hard 17 vs Dealer 10. Even against 10, hitting Hard 17 is catastrophic (-0.52 EV); stand.",
    "STAND and hold pat total: Hard 18 vs Dealer Ace. Hold pat 18 and protect equity.",
    "STAND and hold pat total: multi-card Soft 18 vs Dealer 4 when doubling is unavailable; stand on 18.",
    "STAND and hold pat total: Hard 19 vs Dealer 10 (EV +0.61). Lock pat winner.",
    "STAND and hold pat total: Hard 20 vs Dealer Ace (EV +0.65). Stand on dominant 20.",
    "STAND and let dealer bust: Hard 14 vs Dealer 3 (bust 38%). Stand and force dealer to draw.",
    "STAND and let dealer bust: Hard 15 vs Dealer 2 (bust 35%). Stand on stiff 15 against dealer 2.",
    "STAND and hold pat total: Soft 20 vs Dealer 6. Never mess with a 20; stand.",
    "STAND on high True Count warp (TC +1.0): Hard 16 vs Dealer 10. Illustrious 18 #2 deviation: stand.",

    # --- Cluster 3: [C] DOUBLE — Power Doubles 9/10/11 & Soft Doubles (28 Landmarks) ---
    "DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 2, 3, 4, 5, 6, 7, 8, 9, or 10. Highest EV play in Blackjack; draw one card for 21.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 10 vs Dealer 2, 3, 4, 5, 6, 7, 8, or 9. Strong ten-draw expectation for 20; double wager.",
    "DOUBLE DOWN 2x wager: Pair of 5s (Hard 10) vs Dealer 2 through 9. Never split 5s; always double down as powerhouse Hard 10.",
    "DOUBLE DOWN 2x wager: Hard 9 vs vulnerable Dealer 3, 4, 5, or 6. Weak dealer bust upcard + strong 19 draw makes 2x wager optimal.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 6 (EV +0.67). Maximum positive expected value double down.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 10 (EV +0.18). Press advantage with 2x bet on 11.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 10 vs Dealer 6 (EV +0.58). Double bet against bust-prone dealer.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 10 vs Dealer 9 (EV +0.19). Double down to maximize expected return.",
    "DOUBLE DOWN 2x wager: flexible Soft 13 or Soft 14 (Ace-2, Ace-3) vs weak Dealer 5 or 6. Zero bust risk; press 2x wager.",
    "DOUBLE DOWN 2x wager: flexible Soft 15 or Soft 16 (Ace-4, Ace-5) vs weak Dealer 4, 5, or 6. Put 2x money on table against dealer bust upcard.",
    "DOUBLE DOWN 2x wager: flexible Soft 17 (Ace-6) vs weak Dealer 3, 4, 5, or 6. Rule of 9 satisfied; double down with zero bust risk.",
    "DOUBLE DOWN 2x wager: flexible Soft 18 (Ace-7) vs weak Dealer 3, 4, 5, or 6. Aggressive GTO soft double beats standing by +0.11 EV.",
    "DOUBLE DOWN 2x wager: Hard 11 vs Dealer Ace in positive True Count (TC +1.5). Illustrious 18 index warp: double 11 vs Ace.",
    "DOUBLE DOWN 2x wager: Hard 10 vs Dealer 10 or Ace in high True Count (TC +4.0). Ten-rich shoe makes 2x double on 10 highly profitable.",
    "DOUBLE DOWN 2x wager: Hard 9 vs Dealer 2 or 7 in positive True Count. Opportunistic count-warped double down.",
    "DOUBLE DOWN 2x wager: Soft 19 (Ace-8) vs Dealer 6 in positive True Count. Press 2x wager against dealer 6.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 5. Draw one card for 21 against 42% dealer bust upcard.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 10 vs Dealer 4. Press 2x bet on Hard 10.",
    "DOUBLE DOWN 2x wager: Hard 9 vs Dealer 5 (bust 42%). Double wager for one card.",
    "DOUBLE DOWN 2x wager: Hard 9 vs Dealer 6 (bust 42%). Positive EV double down.",
    "DOUBLE DOWN 2x wager: flexible Soft 18 vs Dealer 6 (bust 42%). Classic Rule-of-9 soft double down.",
    "DOUBLE DOWN 2x wager: flexible Soft 17 vs Dealer 5 (bust 42%). Double wager with soft Ace cushion.",
    "DOUBLE DOWN 2x wager: flexible Soft 16 vs Dealer 6 (bust 42%). Double down against weak dealer.",
    "DOUBLE DOWN 2x wager: flexible Soft 15 vs Dealer 5 (bust 42%). Double wager for maximum EV.",
    "DOUBLE DOWN 2x wager: flexible Soft 14 vs Dealer 6 (bust 42%). Double down against dealer 6.",
    "DOUBLE DOWN 2x wager: flexible Soft 13 vs Dealer 5 (bust 42%). Double down against dealer 5.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 8. Double bet for 21.",
    "DOUBLE DOWN 2x wager: powerhouse Hard 10 vs Dealer 7. Double bet for 20.",

    # --- Cluster 4: [D] SPLIT — Aces & 8s + Tactical Pair Splits (24 Landmarks) ---
    "SPLIT pair now: Pair of Aces (12) vs any Dealer upcard (2 through Ace). Always split Aces into two powerhouse 11-starts.",
    "SPLIT pair now: Pair of 8s (16) vs any Dealer upcard (2 through Ace). Always split 8s to escape worst Hard 16 into twin 8s.",
    "SPLIT pair now: Pair of 9s (18) vs weak Dealer 2, 3, 4, 5, 6, 8, or 9. Split 9s to press twin 19-potential hands.",
    "SPLIT pair now: Pair of 7s (14) vs weak Dealer 2, 3, 4, 5, 6, or 7. Split 7s to escape stiff 14 against bust-prone dealer.",
    "SPLIT pair now: Pair of 6s (12) vs weak Dealer 2, 3, 4, 5, or 6. Split 6s against weak dealer upcards.",
    "SPLIT pair now: Pair of 4s (8) vs weak Dealer 5 or 6. Split 4s against dealer 5/6 to create two strong doubling hands.",
    "SPLIT pair now: Pair of 3s (6) vs weak Dealer 2, 3, 4, 5, 6, or 7. Split 3s to put 2x money out against weak dealer.",
    "SPLIT pair now: Pair of 2s (4) vs weak Dealer 2, 3, 4, 5, 6, or 7. Split 2s to press twin hands against weak dealer.",
    "SPLIT pair now: Pair of Aces vs Dealer 6 (bust 42%). Massive +0.58 EV; always split Aces.",
    "SPLIT pair now: Pair of Aces vs Dealer 10. Always split Aces even against strong dealer 10.",
    "SPLIT pair now: Pair of 8s vs Dealer 10. Splitting 8s loses less than hitting or standing on Hard 16; always split 8s.",
    "SPLIT pair now: Pair of 8s vs Dealer 6 (bust 42%). Turn a losing 16 into two winning hands; split 8s.",
    "SPLIT pair now: Pair of 9s vs Dealer 6 (bust 42%). Split 9s against weak dealer 6.",
    "SPLIT pair now: Pair of 9s vs Dealer 8 or 9. Split 9s to beat dealer 18/19.",
    "SPLIT pair now: Pair of 7s vs Dealer 6 (bust 42%). Split 7s into twin hands.",
    "SPLIT pair now: Pair of 6s vs Dealer 5 (bust 42%). Split 6s against bust-prone dealer.",
    "SPLIT pair now: Pair of 3s vs Dealer 6 (bust 42%). Split 3s to press twin hands.",
    "SPLIT pair now: Pair of 2s vs Dealer 5 (bust 42%). Split 2s against weak dealer.",
    "SPLIT pair now: Pair of 10s vs Dealer 5 or 6 in extreme high True Count (TC +5.0). Illustrious 18 ten-split index warp.",
    "SPLIT pair now: Pair of 8s vs Dealer 9. Always split 8s into two independent hands.",
    "SPLIT pair now: Pair of Aces vs Dealer Ace. Always split Aces.",
    "SPLIT pair now: Pair of 9s vs Dealer 5 (bust 42%). Split 9s to maximize expected value.",
    "SPLIT pair now: Pair of 7s vs Dealer 5 (bust 42%). Split 7s against weak dealer.",
    "SPLIT pair now: Pair of 6s vs Dealer 6 (bust 42%). Split 6s against weak dealer.",
]

BLACKJACK_CLASS_ANCHORS: dict[str, list[str]] = {
    "A": [
        "HIT to draw another card on low total or stiff hand vs strong dealer 7, 8, 9, 10, Ace",
        "HIT Soft 18 vs strong Dealer 9, 10, Ace or hit Hard 12 vs Dealer 2, 3",
        "Draw another card with zero bust risk to improve hand total",
        "Defensive hit on stiff Hard 13, 14, 15, 16 against strong dealer upcard",
    ],
    "B": [
        "STAND and hold pat total on Hard 17, 18, 19, 20, 21 or Soft 19, 20",
        "STAND and let dealer bust on stiff Hard 12, 13, 14, 15, 16 vs weak Dealer 2, 3, 4, 5, 6",
        "STAND on high True Count index warp (Hard 16 vs 10, Hard 15 vs 10, Hard 12 vs 3)",
        "Lock pat hand and avoid player bust risk",
    ],
    "C": [
        "DOUBLE DOWN 2x wager for one card on powerhouse Hard 9, 10, or 11",
        "DOUBLE DOWN 2x wager on flexible Soft 13, 14, 15, 16, 17, 18 vs weak Dealer 3, 4, 5, 6",
        "Press maximum positive expected value with 2x bet on advantageous two-card total",
        "Power double down on Hard 11 or Hard 10 to draw 20 or 21",
    ],
    "D": [
        "SPLIT pair into two independent hands: always split Aces and 8s",
        "SPLIT pair of 2s, 3s, 6s, 7s, 9s against weak Dealer 2, 3, 4, 5, 6, 7",
        "Split matching pair to escape stiff total and press twin hand expected value",
        "Divide pair into two hands with 2x wager",
    ],
}


@dataclass
class BlackjackSituation:
    """Represents a single Blackjack decision state on Gevva0's hand."""
    player_total: int
    is_soft: bool
    is_pair: bool
    pair_rank: int
    dealer_upcard: int
    can_double: bool
    can_split: bool
    true_count: float
    cards_str: str
    gto: BlackjackGTOEvaluation
    query_text: str
    serial: int = 0


class BaseBlackjackBrain:
    name = "Tier0-Blackjack"
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
        self.last_situation: BlackjackSituation | None = None
        self.history: deque[tuple[BlackjackSituation, DecisionResult]] = deque(maxlen=60)
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

    def build_situation(
        self,
        player_total: int,
        is_soft: bool,
        is_pair: bool,
        pair_rank: int,
        dealer_upcard: int,
        can_double: bool = True,
        can_split: bool = False,
        true_count: float = 0.0,
        cards_str: str = "",
    ) -> BlackjackSituation:
        solver = get_blackjack_solver()
        gto_eval = solver.evaluate_hand(
            player_total=player_total,
            is_soft=is_soft,
            is_pair=is_pair,
            pair_rank=pair_rank,
            dealer_upcard=dealer_upcard,
            can_double=can_double,
            can_split=can_split,
            true_count=true_count,
        )
        return BlackjackSituation(
            player_total=player_total,
            is_soft=is_soft,
            is_pair=is_pair,
            pair_rank=pair_rank,
            dealer_upcard=dealer_upcard,
            can_double=can_double,
            can_split=can_split,
            true_count=true_count,
            cards_str=cards_str,
            gto=gto_eval,
            query_text=gto_eval.regime_description,
            serial=self.next_serial(),
        )

    def resolve_legal_action(self, sit: BlackjackSituation, res: DecisionResult) -> str:
        """Maps Tier 0 label ('A'..'D') to a legal Blackjack action ('HIT', 'STAND', 'DOUBLE', 'SPLIT')."""
        chosen = LABEL_TO_ACTION.get(res.label, "STAND")
        if chosen == "SPLIT" and not (sit.can_split and sit.is_pair):
            # Fall back to next highest probability legal action
            legal_labels = ["A", "B"] + (["C"] if sit.can_double else [])
            best_lbl = max(legal_labels, key=lambda l: res.probs.get(l, 0.0))
            return LABEL_TO_ACTION[best_lbl]
        if chosen == "DOUBLE" and not sit.can_double:
            best_lbl = "A" if res.probs.get("A", 0.5) >= res.probs.get("B", 0.5) else "B"
            return LABEL_TO_ACTION[best_lbl]
        return chosen

    def _record(self, sit: BlackjackSituation, res: DecisionResult) -> None:
        self.latencies.append(res.latency_ms)
        self.embed_latencies.append(res.embed_latency_ms)
        self.graph_latencies.append(res.graph_latency_ms)
        self.total_decisions += 1
        chosen_act = self.resolve_legal_action(sit, res)
        if chosen_act == sit.gto.optimal_action:
            self.gto_agreements += 1
        if res.tripwire_passed:
            self.exit0_count += 1
        self.last_situation = sit
        self.last_result = res
        self.history.append((sit, res))

    def shutdown(self) -> None:
        pass


class BlackjackTier0Brain(BaseBlackjackBrain):
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
        self._pending_sit: BlackjackSituation | None = None

    def load(self) -> None:
        try:
            t0 = time.perf_counter()
            corpus = DEFAULT_CORPUS if self.engine_mode == "default_39" else BLACKJACK_DOMAIN_CORPUS
            anchors = DEFAULT_CLASS_ANCHORS if self.engine_mode == "default_39" else BLACKJACK_CLASS_ANCHORS
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
            # Pre-warm the 4 Blackjack candidates so B = (I - Q)^(-1) R is cached on GPU
            init_sit = self.build_situation(11, False, False, 0, 6, can_double=True, can_split=False)
            self.bridge.decide(self._build_request(init_sit))
            self.bridge.start_worker()

            dev_str = str(getattr(self.bridge.tier0, "device", "cuda")).upper()
            backend_str = str(getattr(self.bridge.tier0, "backend", "onnx")).upper()
            self.model_name = f"EmbeddingGemma-2 ({backend_str} on {dev_str}, M={self.bridge.tier0.M})"
            self.ready = True
            print(
                f"[Gevva0 Blackjack Tier 0] Bridge online in {time.perf_counter() - t0:.2f}s | "
                f"Mode={self.engine_mode} | Landmarks M={self.bridge.tier0.M} | Device={dev_str}"
            )
        except Exception as exc:
            self.error = str(exc)
            print(f"[Gevva0 Blackjack Tier 0] Init failed ({exc}); falling back to SimulatedTier0Brain")
        finally:
            self.loading = False

    def load_async(self) -> None:
        threading.Thread(target=self.load, daemon=True, name="blackjack-tier0-loader").start()

    @staticmethod
    def _resolve_markov_weight(mode: str) -> float:
        if mode == "pure_cosine":
            return 0.0
        if mode == "domain_markov":
            return 1.0
        return 0.35  # hybrid_domain (65% Cosine + 35% Markov) or default_39

    def set_engine_mode(self, mode: str) -> None:
        """Switches live between the 4 Tier0Engine formulations:
        - 'hybrid_domain': 0.65 Cosine + 0.35 Domain Manifold AMC (M=116)
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
                self.bridge.rebuild_scaffold(BLACKJACK_DOMAIN_CORPUS, BLACKJACK_CLASS_ANCHORS)

        self.bridge.markov_weight = self._resolve_markov_weight(mode)
        dev_str = str(getattr(self.bridge.tier0, "device", "cuda")).upper()
        backend_str = str(getattr(self.bridge.tier0, "backend", "onnx")).upper()
        self.model_name = f"EmbeddingGemma-2 ({backend_str} on {dev_str}, M={self.bridge.tier0.M})"

    def _build_request(self, sit: BlackjackSituation) -> DecisionRequest:
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

    def decide(self, sit: BlackjackSituation, use_turbo_cache: bool = False) -> DecisionResult:
        """Synchronous Tier 0 decision (~2 ms on GPU, or <0.05 ms when turbo cache is enabled for batch sim)."""
        if self.bridge is None:
            raise RuntimeError("Tier0DecisionBridge is not initialized.")

        tc_bucket = int(round(np.clip(sit.true_count, -3.0, 5.0)))
        cache_key = (
            self.engine_mode,
            f"{sit.player_total}_{int(sit.is_soft)}_{sit.pair_rank if sit.can_split else 0}_{sit.dealer_upcard}_{int(sit.can_double)}_{tc_bucket}",
        )
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

    def shutdown(self) -> None:
        super().shutdown()
        if self.bridge is not None:
            self.bridge.close()


class SimulatedTier0Brain(BaseBlackjackBrain):
    """GPU-free stand-in that emulates the 4 Tier0Engine components (Cosine, AMC Graph, 65/35 Hybrid, Tripwires)
    using the exact Thorp/Baldwin Absorbing Markov Chain EVs in ~1.8 ms.
    """

    name = "Tier0-Sim"
    simulated = True

    def __init__(self, engine_mode: str = "hybrid_domain", seed: int = 42):
        super().__init__()
        self.engine_mode = engine_mode
        self.rng = random.Random(seed)
        self.model_name = "Tier 0 Simulated (AMC + Cosine Stand-In)"

    def load(self) -> None:
        get_blackjack_solver()
        self.ready = True
        self.loading = False

    def load_async(self) -> None:
        self.load()

    def set_engine_mode(self, mode: str) -> None:
        self.engine_mode = mode

    def decide(self, sit: BlackjackSituation, use_turbo_cache: bool = False) -> DecisionResult:
        gto = sit.gto
        opt_lbl = ACTION_TO_LABEL.get(gto.optimal_action, "A")
        opt_idx = CANDIDATE_LABELS.index(opt_lbl)

        # Build realistic cosine similarities for the 4 candidates [A: HIT, B: STAND, C: DOUBLE, D: SPLIT]
        sims = np.array([0.52, 0.52, 0.46, 0.44], dtype=np.float64)
        evs = {"A": gto.ev_hit, "B": gto.ev_stand, "C": gto.ev_double, "D": gto.ev_split}
        for idx, lbl in enumerate(CANDIDATE_LABELS):
            if idx == opt_idx:
                sims[idx] = float(np.clip(0.76 + 0.08 * min(1.0, gto.ev_margin * 3.0), 0.72, 0.89))
            else:
                ev_val = evs[lbl]
                if ev_val < -10.0:
                    sims[idx] = 0.41
                else:
                    diff = max(0.02, gto.best_ev - ev_val)
                    sims[idx] = float(np.clip(0.63 - 0.18 * min(1.0, diff), 0.43, 0.65))

        tau_temp = 0.05
        c_logits = sims / tau_temp
        c_exp = np.exp(c_logits - np.max(c_logits))
        c_probs = c_exp / np.sum(c_exp)

        m_logits = np.full(4, -2.5, dtype=np.float64)
        if self.engine_mode == "default_39":
            # Cold 39-landmark manifold has weaker domain separation
            m_logits[opt_idx] = 0.55
            m_logits[(opt_idx + 1) % 2] = 0.15
            landmarks_cnt = 39
            scaffold = "default_39_landmarks"
        else:
            m_logits[opt_idx] = 2.85
            second_idx = 0 if opt_idx != 0 else 1
            m_logits[second_idx] = -0.65
            landmarks_cnt = len(BLACKJACK_DOMAIN_CORPUS)
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
        sorted_p = np.sort(probs_arr)[::-1]
        top1 = float(sorted_p[0])
        top2 = float(sorted_p[1])
        margin = top1 - top2
        entropy = float(-np.sum(probs_arr * np.log(probs_arr + 1e-9)) / np.log(4.0))

        embed_ms = self.rng.uniform(1.52, 2.12)
        graph_ms = self.rng.uniform(0.08, 0.17)
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
            probs={k: round(float(probs_arr[i]), 4) for i, k in enumerate(CANDIDATE_LABELS)},
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
            cosine_probs={k: round(float(c_probs[i]), 4) for i, k in enumerate(CANDIDATE_LABELS)},
            markov_probs={k: round(float(m_probs[i]), 4) for i, k in enumerate(CANDIDATE_LABELS)},
            cosine_sims={k: round(float(sims[i]), 4) for i, k in enumerate(CANDIDATE_LABELS)},
            embed_latency_ms=round(embed_ms, 3),
            graph_latency_ms=round(graph_ms, 3),
            tripwire_passed=tripwire_ok,
            tripwire_reason=reason,
            scaffold_mode=scaffold,
            landmarks_count=landmarks_cnt,
        )
        self._record(sit, res)
        return res
