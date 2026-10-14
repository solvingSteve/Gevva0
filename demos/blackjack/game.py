"""
Game of Blackjack Core Engine, 6-Deck Vegas Shoe with Hi-Lo Card Counting,
Challenger Heuristics, and High-Speed Simulation Arena.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from brain import BaseBlackjackBrain, BlackjackSituation
from gevva0.bridge import DecisionResult
from gto import BlackjackGTOEvaluation, get_blackjack_solver

SUITS = ("♠", "♥", "♦", "♣")
RANKS = (
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
    ("7", 7),
    ("8", 8),
    ("9", 9),
    ("10", 10),
    ("J", 10),
    ("Q", 10),
    ("K", 10),
    ("A", 11),
)

HEURISTIC_PROFILES = [
    ("never_bust", "Timid Tourist ('Never Bust >=12')"),
    ("mimic_dealer", "Dealer Mimic ('Hit <17, No 2x/Split')"),
    ("hunch_gambler", "Vegas Hunch ('Stand 16v10, Split 10s')"),
    ("flat_basic", "Flat Basic ('No True-Count Warps')"),
]


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str
    value: int  # 2..10, or 11 for Ace

    @property
    def hi_lo_tag(self) -> int:
        if self.value <= 6:
            return 1
        if self.value <= 9:
            return 0
        return -1

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


@dataclass
class Hand:
    cards: List[Card] = field(default_factory=list)
    bet: float = 100.0
    doubled: bool = False
    is_split_hand: bool = False
    stood: bool = False
    outcome: str = ""  # "WIN", "BLACKJACK", "LOSE", "BUST", "PUSH"
    net_units: float = 0.0

    @property
    def total_and_soft(self) -> Tuple[int, bool]:
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.value == 11)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total, (aces > 0 and total <= 21)

    @property
    def total(self) -> int:
        return self.total_and_soft[0]

    @property
    def is_soft(self) -> bool:
        return self.total_and_soft[1]

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and not self.is_split_hand and self.total == 21

    @property
    def is_busted(self) -> bool:
        return self.total > 21

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    @property
    def pair_rank(self) -> int:
        return self.cards[0].value if self.is_pair else 0

    @property
    def can_double(self) -> bool:
        return len(self.cards) == 2 and not self.doubled

    @property
    def done(self) -> bool:
        return self.stood or self.doubled or self.total >= 21

    def format_cards(self) -> str:
        return " ".join(str(c) for c in self.cards)


@dataclass
class ActionEvent:
    actor: str  # "PLAYER", "GEVVA-T0", "DEALER"
    action: str  # "HIT", "STAND", "DOUBLE", "SPLIT", "BUST", "SETTLE"
    hand_desc: str
    gto_optimal: str
    ev_best: float
    latency_ms: float = 0.0
    note: str = ""


@dataclass
class BlackjackStats:
    hands_played: int = 0
    challenger_bankroll: float = 1000.0
    gevva_bankroll: float = 1000.0
    challenger_net_units: float = 0.0
    gevva_net_units: float = 0.0
    challenger_wins: int = 0
    gevva_wins: int = 0
    challenger_pushes: int = 0
    gevva_pushes: int = 0
    challenger_losses: int = 0
    gevva_losses: int = 0
    gevva_doubles_won: int = 0
    gevva_splits_played: int = 0
    gevva_tc_warps: int = 0
    challenger_ev_blunders: int = 0
    gevva_rounds_ahead: int = 0  # Rounds where Gevva0 outperformed or tied Challenger

    @property
    def gevva_edge_over_challenger_units(self) -> float:
        return self.gevva_net_units - self.challenger_net_units

    @property
    def gevva_win_or_push_rate(self) -> float:
        total = self.gevva_wins + self.gevva_pushes + self.gevva_losses
        if total == 0:
            return 0.0
        return ((self.gevva_wins + self.gevva_pushes) / total) * 100.0

    def reset(self) -> None:
        self.hands_played = 0
        self.challenger_bankroll = 1000.0
        self.gevva_bankroll = 1000.0
        self.challenger_net_units = 0.0
        self.gevva_net_units = 0.0
        self.challenger_wins = 0
        self.gevva_wins = 0
        self.challenger_pushes = 0
        self.gevva_pushes = 0
        self.challenger_losses = 0
        self.gevva_losses = 0
        self.gevva_doubles_won = 0
        self.gevva_splits_played = 0
        self.gevva_tc_warps = 0
        self.challenger_ev_blunders = 0
        self.gevva_rounds_ahead = 0


class BlackjackGame:
    """Vegas 6-Deck S17 Blackjack State Machine with Duplicate-Deal Seat Comparison."""

    NUM_DECKS = 6
    BASE_BET = 25.0

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.solver = get_blackjack_solver()
        self.stats = BlackjackStats()

        # Mode configuration
        self.sim_mode: bool = False  # False = Human vs Gevva0 vs Dealer, True = Bot vs Gevva0 Sim
        self.sim_speed: int = 1      # 1 = Animated, 2 = Fast, 3 = Turbo
        self.heuristic_idx: int = 0

        # 6-Deck Shoe & Hi-Lo Card Counting
        self.shoe: List[Card] = []
        self.running_count: int = 0
        self._build_and_shuffle_shoe()

        # Per-round duplicate draw stream so Seat 0 and Seat 1 face identical card sequences without burning unused shoe cards
        self._round_shared_draws: List[Card] = []
        self._seat_draw_cursor: List[int] = [0, 0]

        # Seat 0 (Human / Challenger Bot), Seat 1 (Gevva0 Tier 0), and Dealer
        self.challenger_hands: List[Hand] = [Hand()]
        self.challenger_active_idx: int = 0
        self.gevva_hands: List[Hand] = [Hand()]
        self.gevva_active_idx: int = 0
        self.dealer_hand: Hand = Hand()
        self.dealer_hole_revealed: bool = False

        # Phase: "CHALLENGER_TURN" -> "GEVVA_TURN" -> "DEALER_TURN" -> "ROUND_OVER"
        self.phase: str = "CHALLENGER_TURN"
        self.ai_think_timer: float = 0.0
        self.card_anim_timer: float = 0.0
        self.banner_text: str = ""
        self.banner_color: Tuple[int, int, int] = (255, 210, 45)
        self.banner_timer: float = 0.0

        self.events: deque[ActionEvent] = deque(maxlen=14)
        self.last_human_blunder: Optional[str] = None

        self.deal_new_round(animate=False)

    def _build_and_shuffle_shoe(self) -> None:
        self.shoe = [
            Card(rank=r_str, suit=s, value=r_val)
            for _ in range(self.NUM_DECKS)
            for s in SUITS
            for r_str, r_val in RANKS
        ]
        self.rng.shuffle(self.shoe)
        self.running_count = 0

    def _pop_shoe_card(self, count_visible: bool = True) -> Card:
        if len(self.shoe) < 52:
            self._build_and_shuffle_shoe()
        c = self.shoe.pop()
        if count_visible:
            self.running_count += c.hi_lo_tag
        return c

    @property
    def decks_remaining(self) -> float:
        return max(0.5, len(self.shoe) / 52.0)

    @property
    def true_count(self) -> float:
        return round(self.running_count / self.decks_remaining, 2)

    @property
    def challenger_name(self) -> str:
        if not self.sim_mode:
            return "YOU (Player Seat 1)"
        return HEURISTIC_PROFILES[self.heuristic_idx][1]

    def cycle_heuristic(self) -> None:
        self.heuristic_idx = (self.heuristic_idx + 1) % len(HEURISTIC_PROFILES)

    @property
    def active_challenger_hand(self) -> Hand:
        idx = min(self.challenger_active_idx, len(self.challenger_hands) - 1)
        return self.challenger_hands[idx]

    @property
    def active_gevva_hand(self) -> Hand:
        idx = min(self.gevva_active_idx, len(self.gevva_hands) - 1)
        return self.gevva_hands[idx]

    @property
    def dealer_upcard_value(self) -> int:
        return self.dealer_hand.cards[0].value if self.dealer_hand.cards else 10

    def current_active_gto(self) -> BlackjackGTOEvaluation:
        """Returns exact GTO evaluation for whichever seat is currently acting."""
        if self.phase == "CHALLENGER_TURN":
            h = self.active_challenger_hand
            can_sp = h.is_pair and len(self.challenger_hands) == 1
        else:
            h = self.active_gevva_hand
            can_sp = h.is_pair and len(self.gevva_hands) == 1
        return self.solver.evaluate_hand(
            player_total=h.total,
            is_soft=h.is_soft,
            is_pair=h.is_pair,
            pair_rank=h.pair_rank,
            dealer_upcard=self.dealer_upcard_value,
            can_double=h.can_double,
            can_split=can_sp,
            true_count=self.true_count,
        )

    def deal_new_round(self, animate: bool = True) -> None:
        """Deals a fresh round where Seat 0 and Seat 1 receive matched starting cards against Dealer."""
        if len(self.shoe) < 65:
            self._build_and_shuffle_shoe()

        c1 = self._pop_shoe_card(count_visible=True)
        c2 = self._pop_shoe_card(count_visible=True)
        d_up = self._pop_shoe_card(count_visible=True)
        d_hole = self._pop_shoe_card(count_visible=False)  # Counted when revealed

        self._round_shared_draws = []
        self._seat_draw_cursor = [0, 0]

        self.challenger_hands = [Hand(cards=[c1, c2], bet=self.BASE_BET)]
        self.challenger_active_idx = 0
        self.gevva_hands = [Hand(cards=[c1, c2], bet=self.BASE_BET)]
        self.gevva_active_idx = 0
        self.dealer_hand = Hand(cards=[d_up, d_hole], bet=0.0)
        self.dealer_hole_revealed = False
        self.last_human_blunder = None

        if animate:
            self.card_anim_timer = 0.22

        # Check for immediate Natural Blackjack (21 on initial 2 cards) or Dealer Blackjack
        if self.challenger_hands[0].is_blackjack or self.dealer_hand.is_blackjack:
            self._reveal_and_settle_round(animate=animate)
            return

        self.phase = "CHALLENGER_TURN"
        self.ai_think_timer = 0.25 if animate else 0.0

    def _draw_for_seat(self, seat: int) -> Card:
        idx = self._seat_draw_cursor[seat]
        self._seat_draw_cursor[seat] += 1
        if idx < len(self._round_shared_draws):
            return self._round_shared_draws[idx]
        c = self._pop_shoe_card(count_visible=True)
        self._round_shared_draws.append(c)
        return c

    # ------------------------------------------------------------------
    # SEAT 0 (HUMAN / CHALLENGER BOT) ACTIONS
    # ------------------------------------------------------------------
    def execute_challenger_action(self, action: str, animate: bool = True) -> None:
        if self.phase != "CHALLENGER_TURN":
            return
        h = self.active_challenger_hand
        can_sp = h.is_pair and len(self.challenger_hands) == 1
        gto = self.solver.evaluate_hand(
            h.total, h.is_soft, h.is_pair, h.pair_rank, self.dealer_upcard_value, h.can_double, can_sp, self.true_count
        )

        # Normalize illegal requests gracefully
        if action == "SPLIT" and not can_sp:
            action = "HIT" if gto.ev_hit >= gto.ev_stand else "STAND"
        if action == "DOUBLE" and not h.can_double:
            action = "HIT"

        # Check if Challenger committed an EV blunder vs GTO
        if action != gto.optimal_action:
            self.stats.challenger_ev_blunders += 1
            if not self.sim_mode:
                self.last_human_blunder = (
                    f"Chose {action} on {h.format_cards()} vs {self.dealer_hand.cards[0].rank} "
                    f"(GTO: {gto.optimal_action}, EV margin {gto.ev_margin:+.2f})"
                )

        self._apply_action_to_seat(
            seat=0,
            hands_list=self.challenger_hands,
            active_idx_attr="challenger_active_idx",
            action=action,
            gto=gto,
            latency_ms=0.0,
            animate=animate,
        )
        self._advance_turn_pointer(animate=animate)

    # ------------------------------------------------------------------
    # SEAT 1 (GEVVA0 TIER 0) ACTIONS
    # ------------------------------------------------------------------
    def step_gevva_decision(
        self,
        brain: BaseBlackjackBrain,
        animate: bool = True,
        use_turbo_cache: bool = False,
    ) -> Optional[DecisionResult]:
        if self.phase != "GEVVA_TURN":
            return None
        h = self.active_gevva_hand
        if h.done:
            self._advance_turn_pointer(animate=animate)
            return None

        can_sp = h.is_pair and len(self.gevva_hands) == 1
        sit = brain.build_situation(
            player_total=h.total,
            is_soft=h.is_soft,
            is_pair=h.is_pair,
            pair_rank=h.pair_rank,
            dealer_upcard=self.dealer_upcard_value,
            can_double=h.can_double,
            can_split=can_sp,
            true_count=self.true_count,
            cards_str=h.format_cards(),
        )
        res = brain.decide(sit, use_turbo_cache=use_turbo_cache)
        action = brain.resolve_legal_action(sit, res)

        if sit.gto.regime == "TRUE_COUNT_INDEX_WARP_STAND" and action == "STAND":
            self.stats.gevva_tc_warps += 1

        self._apply_action_to_seat(
            seat=1,
            hands_list=self.gevva_hands,
            active_idx_attr="gevva_active_idx",
            action=action,
            gto=sit.gto,
            latency_ms=res.latency_ms,
            animate=animate,
        )
        self._advance_turn_pointer(animate=animate)
        return res

    def _apply_action_to_seat(
        self,
        seat: int,
        hands_list: List[Hand],
        active_idx_attr: str,
        action: str,
        gto: BlackjackGTOEvaluation,
        latency_ms: float,
        animate: bool,
    ) -> None:
        idx = getattr(self, active_idx_attr)
        h = hands_list[idx]
        actor = "GEVVA-T0" if seat == 1 else "PLAYER 1"
        desc_before = f"{'Soft' if h.is_soft else 'Hard'} {h.total} ({h.format_cards()})"

        if action == "STAND":
            h.stood = True
            self.events.append(
                ActionEvent(
                    actor=actor,
                    action="STAND",
                    hand_desc=desc_before,
                    gto_optimal=gto.optimal_action,
                    ev_best=gto.best_ev,
                    latency_ms=latency_ms,
                    note=f"Stood on {h.total} vs Dealer {self.dealer_hand.cards[0].rank}",
                )
            )
        elif action == "HIT":
            c = self._draw_for_seat(seat)
            h.cards.append(c)
            if animate:
                self.card_anim_timer = 0.18
            act_tag = "BUST" if h.is_busted else "HIT"
            self.events.append(
                ActionEvent(
                    actor=actor,
                    action=act_tag,
                    hand_desc=desc_before,
                    gto_optimal=gto.optimal_action,
                    ev_best=gto.best_ev,
                    latency_ms=latency_ms,
                    note=f"Hit +{c} -> {h.total}" + (" (BUST!)" if h.is_busted else ""),
                )
            )
        elif action == "DOUBLE":
            c = self._draw_for_seat(seat)
            h.cards.append(c)
            h.bet *= 2.0
            h.doubled = True
            h.stood = True
            if animate:
                self.card_anim_timer = 0.22
                self.banner_text = f"{actor} DOUBLES DOWN (2x) -> {h.total}!"
                self.banner_color = (255, 210, 45)
                self.banner_timer = 0.75
            self.events.append(
                ActionEvent(
                    actor=actor,
                    action="DOUBLE",
                    hand_desc=desc_before,
                    gto_optimal=gto.optimal_action,
                    ev_best=gto.best_ev,
                    latency_ms=latency_ms,
                    note=f"Doubled 2x +{c} -> {h.total}",
                )
            )
        elif action == "SPLIT":
            c_left = h.cards[0]
            c_right = h.cards[1]
            draw1 = self._draw_for_seat(seat)
            draw2 = self._draw_for_seat(seat)
            h1 = Hand(cards=[c_left, draw1], bet=h.bet, is_split_hand=True)
            h2 = Hand(cards=[c_right, draw2], bet=h.bet, is_split_hand=True)
            # Splitting Aces receives 1 card per Ace and stands immediately
            if c_left.value == 11:
                h1.stood = True
                h2.stood = True
            hands_list[idx] = h1
            hands_list.append(h2)
            if seat == 1:
                self.stats.gevva_splits_played += 1
            if animate:
                self.card_anim_timer = 0.24
                self.banner_text = f"{actor} SPLITS PAIR OF {c_left.rank}s!"
                self.banner_color = (175, 115, 255)
                self.banner_timer = 0.75
            self.events.append(
                ActionEvent(
                    actor=actor,
                    action="SPLIT",
                    hand_desc=desc_before,
                    gto_optimal=gto.optimal_action,
                    ev_best=gto.best_ev,
                    latency_ms=latency_ms,
                    note=f"Split {c_left.rank}s -> [{h1.total}] & [{h2.total}]",
                )
            )

    def _advance_turn_pointer(self, animate: bool = True) -> None:
        if self.phase == "CHALLENGER_TURN":
            while self.challenger_active_idx < len(self.challenger_hands) and self.active_challenger_hand.done:
                self.challenger_active_idx += 1
            if self.challenger_active_idx >= len(self.challenger_hands):
                self.challenger_active_idx = len(self.challenger_hands) - 1
                self.phase = "GEVVA_TURN"
                self.ai_think_timer = 0.25 if animate else 0.0
                while self.gevva_active_idx < len(self.gevva_hands) and self.active_gevva_hand.done:
                    self.gevva_active_idx += 1
                if self.gevva_active_idx >= len(self.gevva_hands):
                    self.gevva_active_idx = len(self.gevva_hands) - 1
                    self._reveal_and_settle_round(animate=animate)
        elif self.phase == "GEVVA_TURN":
            while self.gevva_active_idx < len(self.gevva_hands) and self.active_gevva_hand.done:
                self.gevva_active_idx += 1
            if self.gevva_active_idx >= len(self.gevva_hands):
                self.gevva_active_idx = len(self.gevva_hands) - 1
                self._reveal_and_settle_round(animate=animate)

    def _reveal_and_settle_round(self, animate: bool = True) -> None:
        if not self.dealer_hole_revealed and len(self.dealer_hand.cards) >= 2:
            self.dealer_hole_revealed = True
            self.running_count += self.dealer_hand.cards[1].hi_lo_tag

        # Dealer draws to 17 (S17: stands on all 17s) if at least one non-busted, non-natural hand exists
        any_live = any(not h.is_busted and not h.is_blackjack for h in (self.challenger_hands + self.gevva_hands))
        if any_live and not self.dealer_hand.is_blackjack:
            while self.dealer_hand.total < 17:
                self.dealer_hand.cards.append(self._pop_shoe_card(count_visible=True))
        self.dealer_hand.stood = True

        c_units = self._settle_seat_hands(self.challenger_hands, is_gevva=False)
        g_units = self._settle_seat_hands(self.gevva_hands, is_gevva=True)

        self.stats.hands_played += 1
        self.stats.challenger_net_units = round(self.stats.challenger_net_units + c_units, 2)
        self.stats.gevva_net_units = round(self.stats.gevva_net_units + g_units, 2)
        self.stats.challenger_bankroll = round(self.stats.challenger_bankroll + c_units * self.BASE_BET, 2)
        self.stats.gevva_bankroll = round(self.stats.gevva_bankroll + g_units * self.BASE_BET, 2)
        if g_units >= c_units:
            self.stats.gevva_rounds_ahead += 1

        self.phase = "ROUND_OVER"
        self.ai_think_timer = 0.55 if animate else 0.0
        if animate:
            if g_units > c_units:
                self.banner_text = f"GEVVA-T0 GAINS +{g_units - c_units:.1f}u EDGE ON THIS HAND!"
                self.banner_color = (50, 245, 165)
            elif g_units > 0:
                self.banner_text = f"GEVVA-T0 WINS +{g_units:.1f}u (DEALER {self.dealer_hand.total})"
                self.banner_color = (0, 235, 255)
            elif self.dealer_hand.is_busted:
                self.banner_text = f"DEALER BUSTS ({self.dealer_hand.total})!"
                self.banner_color = (255, 210, 45)
            else:
                self.banner_text = f"DEALER HAS {self.dealer_hand.total}"
                self.banner_color = (255, 145, 185)
            self.banner_timer = 1.1

    def _settle_seat_hands(self, hands: List[Hand], is_gevva: bool) -> float:
        d_tot = self.dealer_hand.total
        d_bj = self.dealer_hand.is_blackjack
        d_bust = self.dealer_hand.is_busted
        net_u = 0.0

        for h in hands:
            wager_u = h.bet / self.BASE_BET
            if h.is_busted:
                h.outcome = "BUST"
                h.net_units = -wager_u
            elif d_bj:
                if h.is_blackjack:
                    h.outcome = "PUSH"
                    h.net_units = 0.0
                else:
                    h.outcome = "LOSE"
                    h.net_units = -wager_u
            elif h.is_blackjack:
                h.outcome = "BLACKJACK"
                h.net_units = 1.5 * wager_u
            elif d_bust or h.total > d_tot:
                h.outcome = "WIN"
                h.net_units = wager_u
                if is_gevva and h.doubled:
                    self.stats.gevva_doubles_won += 1
            elif h.total < d_tot:
                h.outcome = "LOSE"
                h.net_units = -wager_u
            else:
                h.outcome = "PUSH"
                h.net_units = 0.0

            net_u += h.net_units
            if h.net_units > 0:
                if is_gevva:
                    self.stats.gevva_wins += 1
                else:
                    self.stats.challenger_wins += 1
            elif h.net_units < 0:
                if is_gevva:
                    self.stats.gevva_losses += 1
                else:
                    self.stats.challenger_losses += 1
            else:
                if is_gevva:
                    self.stats.gevva_pushes += 1
                else:
                    self.stats.challenger_pushes += 1

        return net_u

    # ------------------------------------------------------------------
    # CHALLENGER HEURISTIC BOT FOR SIMULATION MODE
    # ------------------------------------------------------------------
    def decide_challenger_heuristic(self) -> str:
        h = self.active_challenger_hand
        up = self.dealer_upcard_value
        prof_key = HEURISTIC_PROFILES[self.heuristic_idx][0]

        if prof_key == "never_bust":
            # Timid Tourist: stands on any Hard 12+ for fear of busting; never splits or doubles soft
            if h.can_double and h.total == 11 and up <= 6:
                return "DOUBLE"
            if not h.is_soft and h.total >= 12:
                return "STAND"
            if h.is_soft and h.total >= 18:
                return "STAND"
            return "HIT"

        if prof_key == "mimic_dealer":
            # Dealer Mimic: hits until 17, stands on all 17+; never doubles or splits
            return "STAND" if h.total >= 17 else "HIT"

        if prof_key == "hunch_gambler":
            # Vegas Hunch: splits 10s/5s, stands on 15/16 vs 10, hits stiff 13/14 vs 2/3/4, never soft-doubles
            if h.is_pair and len(self.challenger_hands) == 1 and h.pair_rank in (5, 10) and up <= 6:
                return "SPLIT"
            if h.total >= 15 and up >= 7:
                return "STAND"
            if not h.is_soft and h.total in (13, 14) and up in (2, 3, 4):
                return "HIT"
            if h.can_double and not h.is_soft and h.total == 11 and up <= 6:
                return "DOUBLE"
            return "STAND" if h.total >= 16 else "HIT"

        # flat_basic: Plays static strategy without True Count warps, misses soft doubles & 9-doubles
        if h.is_soft:
            return "STAND" if h.total >= 18 else "HIT"
        if h.can_double and h.total == 9:
            return "HIT"
        gto_flat = self.solver.evaluate_hand(
            h.total, h.is_soft, h.is_pair, h.pair_rank, up, h.can_double, h.is_pair and len(self.challenger_hands) == 1, 0.0
        )
        return gto_flat.optimal_action

    def step_simulation(
        self,
        brain: BaseBlackjackBrain,
        animate: bool = True,
        use_turbo_cache: bool = False,
    ) -> Optional[DecisionResult]:
        """Advances one step in either Interactive Play (Gevva0 turn) or Simulation Mode."""
        if self.phase == "ROUND_OVER":
            self.deal_new_round(animate=animate)
            return None

        if self.phase == "CHALLENGER_TURN":
            if not self.sim_mode:
                return None
            act = self.decide_challenger_heuristic()
            self.execute_challenger_action(act, animate=animate)
            return None

        if self.phase == "GEVVA_TURN":
            return self.step_gevva_decision(brain, animate=animate, use_turbo_cache=use_turbo_cache)

        return None

    def run_turbo_hands(self, brain: BaseBlackjackBrain, num_hands: int = 50) -> None:
        """Simulates `num_hands` complete Blackjack rounds at high speed using Tier 0."""
        was_sim = self.sim_mode
        self.sim_mode = True
        try:
            for _ in range(num_hands):
                if self.phase == "ROUND_OVER":
                    self.deal_new_round(animate=False)
                steps = 0
                while self.phase != "ROUND_OVER" and steps < 30:
                    self.step_simulation(brain, animate=False, use_turbo_cache=True)
                    steps += 1
        finally:
            self.sim_mode = was_sim
