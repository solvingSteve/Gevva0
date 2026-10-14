"""
Gevva0 Game of Blackjack: Interactive Vegas Cyber-Table & High-Speed Simulation Renderer (Pygame).

Left Viewport:
- Dealer Stage with 6-Deck Shoe Penetration, Hi-Lo Running/True Count gauge, & Dealer Bust %
- Side-by-side Seat 0 (Human / Challenger Bot) & Seat 1 (Gevva0 Tier 0 AMC) with crisp vector playing cards
- Interactive clickable buttons for HIT, STAND, DOUBLE 2x, SPLIT, Play/Sim mode, Sim speed, and +50 Turbo Hands
- Live GTO Basic Strategy + Illustrious 18 Matrix Heatmap highlighting the active (Hand, Dealer Upcard) cell

Right Viewport:
- Live interactive diagram of the 4 Tier0Engine components across 4 actions ([A] HIT, [B] STAND, [C] DOUBLE, [D] SPLIT):
  (1) Direct Cosine sim(q, c_i)
  (2) Default 39-Landmarks & (3) Blackjack Domain Manifold (M=116) -> B = (I - Q)^(-1) R
  (4) Production Hybrid Ensemble: P_final = 0.65 * Cosine + 0.35 * Markov
  Tripwire Checks: Shannon Entropy H <= 0.65, Logit Margin Delta >= 0.25 -> [EXIT TIER 0]
- High-Speed Simulation Bankroll & Unit EV Edge Scoreboard
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from brain import BaseBlackjackBrain
from game import BlackjackGame, Card, Hand


@dataclass
class ChipParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float


class BlackjackRenderer:
    WIN_WIDTH = 1280
    WIN_HEIGHT = 820

    PALETTE = {
        "bg": (12, 16, 24),
        "panel_bg": (18, 24, 36),
        "felt_bg": (16, 38, 34),
        "felt_dark": (12, 28, 25),
        "card_bg": (25, 31, 46),
        "panel_border": (48, 60, 84),
        "panel_header": (26, 34, 52),
        "cyber_cyan": (0, 235, 255),
        "cyber_mint": (50, 245, 165),
        "cyber_blue": (45, 130, 255),
        "gold": (255, 210, 45),
        "gold_dark": (195, 145, 15),
        "rose": (255, 75, 110),
        "amber": (255, 165, 35),
        "purple": (180, 120, 255),
        "text_white": (245, 247, 252),
        "text_dim": (148, 158, 182),
        "text_muted": (96, 106, 130),
        "card_face": (248, 250, 255),
        "suit_red": (225, 45, 75),
        "suit_black": (24, 28, 42),
    }

    ENGINE_MODES = [
        ("hybrid_domain", "(4) 65/35 Hybrid [Domain]"),
        ("domain_markov", "(3) Pure AMC [B=(I-Q)⁻¹R]"),
        ("default_39", "(2) 39-Landmark Cold"),
        ("pure_cosine", "(1) Pure Direct Cosine"),
    ]

    def __init__(self, fullscreen: bool = False):
        pygame.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT), flags)
        pygame.display.set_caption(
            "BLACKJACK 21 // GEVVA0 TIER 0 OPTIMAL STOPPING & AMC SHOWCASE (ONNX EMBEDDING)"
        )
        self._init_fonts()
        self.particles: List[ChipParticle] = []
        self.time_sec: float = 0.0
        self.buttons: dict[str, pygame.Rect] = {}
        self._prev_gevva_wins = 0

    def _init_fonts(self) -> None:
        font_mono = "Consolas"
        font_sans = "Segoe UI"
        self.f_giant = pygame.font.SysFont(font_sans, 30, bold=True)
        self.f_huge = pygame.font.SysFont(font_sans, 22, bold=True)
        self.f_title = pygame.font.SysFont(font_sans, 16, bold=True)
        self.f_header = pygame.font.SysFont(font_mono, 13, bold=True)
        self.f_bold = pygame.font.SysFont(font_mono, 12, bold=True)
        self.f_regular = pygame.font.SysFont(font_mono, 12)
        self.f_small = pygame.font.SysFont(font_mono, 11)
        self.f_tiny = pygame.font.SysFont(font_mono, 10)
        self.f_card = pygame.font.SysFont(font_sans, 15, bold=True)
        self.f_suit = pygame.font.SysFont(font_sans, 18, bold=True)

    def spawn_chips(self, x: float, y: float, count: int = 14) -> None:
        for _ in range(count):
            ang = random.uniform(-math.pi * 0.85, -math.pi * 0.15)
            spd = random.uniform(75, 210)
            col = self.PALETTE["gold"] if random.random() < 0.6 else self.PALETTE["cyber_mint"]
            self.particles.append(
                ChipParticle(
                    x=x + random.uniform(-16, 16),
                    y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    life=0.0,
                    max_life=random.uniform(0.50, 0.90),
                    color=col,
                    size=random.uniform(5, 8),
                )
            )

    def render_loading(self, title_msg: str, detail: str = "") -> None:
        self.screen.fill(self.PALETTE["bg"])
        cx, cy = self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2
        box = pygame.Rect(cx - 330, cy - 130, 660, 260)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], box, border_radius=14)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_cyan"], box, 2, border_radius=14)

        # Draw sample Ace & King of Spades in loading box
        self._draw_playing_card(cx - 245, cy - 38, Card("A", "♠", 11), scale=1.1)
        self._draw_playing_card(cx + 195, cy - 38, Card("K", "♦", 10), scale=1.1)

        t_surf = self.f_huge.render("GEVVA0 TIER 0 // VEGAS BLACKJACK 21", True, self.PALETTE["gold"])
        self.screen.blit(t_surf, t_surf.get_rect(center=(cx, cy - 65)))

        m_surf = self.f_title.render(title_msg, True, self.PALETTE["text_white"])
        self.screen.blit(m_surf, m_surf.get_rect(center=(cx, cy - 10)))

        if detail:
            d_surf = self.f_regular.render(detail, True, self.PALETTE["cyber_cyan"])
            self.screen.blit(d_surf, d_surf.get_rect(center=(cx, cy + 25)))

        dots = "." * (int(pygame.time.get_ticks() / 300) % 4)
        w_surf = self.f_regular.render(
            f"Solving Exact Dealer Absorbing Chains & Initializing ONNX{dots:<3}",
            True,
            self.PALETTE["text_dim"],
        )
        self.screen.blit(w_surf, w_surf.get_rect(center=(cx, cy + 65)))
        pygame.display.flip()

    def button_at(self, pos: Tuple[int, int]) -> Optional[str]:
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return name
        return None

    def render(self, game: BlackjackGame, brain: BaseBlackjackBrain, dt: float) -> None:
        self.time_sec += dt
        if game.card_anim_timer > 0:
            game.card_anim_timer = max(0.0, game.card_anim_timer - dt)
        if game.banner_timer > 0:
            game.banner_timer = max(0.0, game.banner_timer - dt)

        if game.stats.gevva_wins > self._prev_gevva_wins and (not game.sim_mode or game.sim_speed == 1):
            self.spawn_chips(475, 290, count=12)
        self._prev_gevva_wins = game.stats.gevva_wins

        self.buttons.clear()
        self.screen.fill(self.PALETTE["bg"])

        # Left Panel: Vegas Blackjack Arena & Live Strategy Matrix
        self._draw_casino_panel(game, brain)

        # Right Panel: Tier 0 Engine Architecture & Telemetry
        self._draw_tier0_panel(game, brain)

        # Update & Draw Chip Particles
        self._update_particles(dt)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # PLAYING CARD RENDERER
    # ------------------------------------------------------------------
    def _draw_playing_card(
        self,
        x: int,
        y: int,
        card: Optional[Card],
        hidden: bool = False,
        scale: float = 1.0,
        highlight_border: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        w = int(48 * scale)
        h = int(68 * scale)

        # Drop shadow
        pygame.draw.rect(self.screen, (6, 8, 12), (x + 3, y + 4, w, h), border_radius=7)
        rect = pygame.Rect(x, y, w, h)

        if hidden or card is None:
            pygame.draw.rect(self.screen, (22, 42, 68), rect, border_radius=7)
            pygame.draw.rect(self.screen, self.PALETTE["cyber_cyan"], rect, 2, border_radius=7)
            inner = rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, (16, 30, 50), inner, border_radius=4)
            q_lbl = self.f_bold.render("G0", True, self.PALETTE["cyber_cyan"])
            self.screen.blit(q_lbl, q_lbl.get_rect(center=rect.center))
            return

        pygame.draw.rect(self.screen, self.PALETTE["card_face"], rect, border_radius=7)
        bdr_col = highlight_border or (175, 185, 205)
        pygame.draw.rect(self.screen, bdr_col, rect, 2 if highlight_border else 1, border_radius=7)

        is_red = card.suit in ("♥", "♦")
        col = self.PALETTE["suit_red"] if is_red else self.PALETTE["suit_black"]

        r_surf = self.f_card.render(card.rank, True, col)
        s_surf = self.f_suit.render(card.suit, True, col)
        self.screen.blit(r_surf, (x + 5, y + 2))
        self.screen.blit(s_surf, s_surf.get_rect(center=(x + w // 2, y + int(h * 0.58))))

    def _draw_button(
        self,
        key: str,
        rect: pygame.Rect,
        label: str,
        bg_color: Tuple[int, int, int],
        border_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = (245, 247, 252),
        enabled: bool = True,
        font: Optional[pygame.font.Font] = None,
    ) -> None:
        self.buttons[key] = rect
        f = font or self.f_bold
        mx, my = pygame.mouse.get_pos()
        hovered = enabled and rect.collidepoint((mx, my))
        if not enabled:
            bg = (22, 24, 34)
            bdr = (42, 46, 62)
            txt = (82, 88, 108)
        else:
            bg = tuple(min(255, c + (22 if hovered else 0)) for c in bg_color)
            bdr = border_color
            txt = text_color

        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, bdr, rect, 2 if hovered else 1, border_radius=8)
        surf = f.render(label, True, txt)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    # ------------------------------------------------------------------
    # LEFT VIEWPORT: VEGAS BLACKJACK ARENA
    # ------------------------------------------------------------------
    def _draw_casino_panel(self, game: BlackjackGame, brain: BaseBlackjackBrain) -> None:
        px, py, pw, ph = 16, 16, 624, self.WIN_HEIGHT - 32
        panel = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], panel, border_radius=12)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], panel, 1, border_radius=12)

        # Header
        hdr = pygame.Rect(px, py, pw, 40)
        pygame.draw.rect(self.screen, self.PALETTE["panel_header"], hdr, border_top_left_radius=12, border_top_right_radius=12)
        t_lbl = self.f_title.render("♠ VEGAS BLACKJACK 21 // ABSORBING MARKOV CHAIN TABLE", True, self.PALETTE["gold"])
        self.screen.blit(t_lbl, (px + 14, py + 9))

        # Top Control Bar
        cy = py + 46
        mode_lbl = "MODE: HIGH-SPEED SIM (M)" if game.sim_mode else "MODE: 1v1 TABLE vs GEVVA0 (M)"
        mode_bg = (22, 58, 72) if game.sim_mode else (58, 28, 64)
        mode_bdr = self.PALETTE["cyber_cyan"] if game.sim_mode else self.PALETTE["purple"]
        self._draw_button("toggle_mode", pygame.Rect(px + 12, cy, 220, 28), mode_lbl, mode_bg, mode_bdr, font=self.f_small)

        spd_names = {1: "SPEED: 1x (S)", 2: "SPEED: 10x (S)", 3: "SPEED: TURBO (S)"}
        self._draw_button(
            "cycle_speed",
            pygame.Rect(px + 238, cy, 126, 28),
            spd_names.get(game.sim_speed, "SPEED: 1x"),
            (28, 36, 56),
            self.PALETTE["amber"] if game.sim_speed > 1 else self.PALETTE["panel_border"],
            font=self.f_small,
        )

        self._draw_button(
            "turbo_50",
            pygame.Rect(px + 370, cy, 138, 28),
            "⚡ +50 SIM HANDS (T)",
            (26, 56, 46),
            self.PALETTE["cyber_mint"],
            font=self.f_small,
        )

        self._draw_button(
            "deal_hand",
            pygame.Rect(px + 514, cy, 98, 28),
            "NEXT HAND (N)",
            (36, 38, 56),
            self.PALETTE["panel_border"],
            font=self.f_small,
        )
        cy += 34

        # 1. Dealer Stage & Shoe Hi-Lo True Count Strip
        dealer_h = 98
        dealer_rect = pygame.Rect(px + 12, cy, pw - 24, dealer_h)
        pygame.draw.rect(self.screen, self.PALETTE["felt_bg"], dealer_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], dealer_rect, 1, border_radius=10)

        self.screen.blit(
            self.f_bold.render("HOUSE DEALER (S17 - STANDS ALL 17s)", True, self.PALETTE["gold"]),
            (dealer_rect.x + 14, dealer_rect.y + 8),
        )
        up_val = game.dealer_upcard_value
        d_bust_pct = game.solver.dealer_probs[up_val][5] * 100.0
        if game.dealer_hole_revealed:
            d_status = f"Total: {game.dealer_hand.total}" + (" (BUST!)" if game.dealer_hand.is_busted else "")
        else:
            up_str = game.dealer_hand.cards[0].rank if game.dealer_hand.cards else "?"
            d_status = f"Upcard: {up_str}  |  Exact Bust Prob: {d_bust_pct:.1f}%"
        self.screen.blit(self.f_small.render(d_status, True, self.PALETTE["text_white"]), (dealer_rect.x + 14, dealer_rect.y + 28))

        # Shoe & Hi-Lo Count Telemetry on right side of Dealer Stage
        tc = game.true_count
        tc_col = self.PALETTE["cyber_mint"] if tc >= 1.5 else (self.PALETTE["rose"] if tc <= -1.5 else self.PALETTE["cyber_cyan"])
        shoe_txt = f"6-Deck Shoe: {game.decks_remaining:.1f}d left  |  Hi-Lo RC: {game.running_count:+d}"
        tc_txt = f"TRUE COUNT: {tc:+.2f} " + ("(TEN-RICH WARP)" if tc >= 1.5 else ("(LOW-CARD SHOE)" if tc <= -1.5 else "(NEUTRAL)"))
        self.screen.blit(self.f_tiny.render(shoe_txt, True, self.PALETTE["text_dim"]), (dealer_rect.right - 235, dealer_rect.y + 10))
        self.screen.blit(self.f_bold.render(tc_txt, True, tc_col), (dealer_rect.right - 235, dealer_rect.y + 26))

        # Dealer Cards Centered
        d_cards = game.dealer_hand.cards
        start_x = dealer_rect.x + 195
        for c_i, c in enumerate(d_cards):
            hide = (c_i == 1) and (not game.dealer_hole_revealed)
            self._draw_playing_card(start_x + c_i * 38, dealer_rect.y + 20, c, hidden=hide, scale=0.95)

        cy += dealer_h + 8

        # 2. Twin Player Seats (Seat 0: Human/Challenger vs Seat 1: Gevva0 Tier 0)
        seat_w = (pw - 32) // 2
        seat_h = 158
        s0_rect = pygame.Rect(px + 12, cy, seat_w, seat_h)
        s1_rect = pygame.Rect(px + 20 + seat_w, cy, seat_w, seat_h)

        self._draw_seat_card(
            s0_rect,
            title=game.challenger_name,
            hands=game.challenger_hands,
            active_idx=game.challenger_active_idx,
            is_turn=(game.phase == "CHALLENGER_TURN"),
            is_gevva=False,
            bankroll=game.stats.challenger_bankroll,
            net_units=game.stats.challenger_net_units,
            game=game,
        )
        self._draw_seat_card(
            s1_rect,
            title="GEVVA0 TIER 0 (AMC)",
            hands=game.gevva_hands,
            active_idx=game.gevva_active_idx,
            is_turn=(game.phase == "GEVVA_TURN"),
            is_gevva=True,
            bankroll=game.stats.gevva_bankroll,
            net_units=game.stats.gevva_net_units,
            game=game,
        )
        cy += seat_h + 8

        # 3. Interactive Action Deck & Exact Action EV Strip
        act_h = 98
        act_rect = pygame.Rect(px + 12, cy, pw - 24, act_h)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], act_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], act_rect, 1, border_radius=10)

        gto = game.current_active_gto()
        ev_d_str = f"{gto.ev_double:+.2f}" if gto.can_double else "N/A"
        ev_sp_str = f"{gto.ev_split:+.2f}" if gto.can_split else "N/A"
        ev_line = (
            f"EXACT MDP EV:  HIT={gto.ev_hit:+.2f}u  |  STAND={gto.ev_stand:+.2f}u  |  "
            f"DOUBLE={ev_d_str}u  |  SPLIT={ev_sp_str}u   ▶ OPTIMAL: {gto.optimal_action}"
        )
        self.screen.blit(self.f_bold.render(ev_line[:82], True, self.PALETTE["cyber_cyan"]), (act_rect.x + 12, act_rect.y + 8))

        if game.last_human_blunder and not game.sim_mode:
            b_lbl = self.f_tiny.render(f"EV Blunder Alert: {game.last_human_blunder}", True, self.PALETTE["rose"])
            self.screen.blit(b_lbl, (act_rect.x + 12, act_rect.y + 26))
        elif game.banner_timer > 0 and game.banner_text:
            b_lbl = self.f_bold.render(game.banner_text, True, game.banner_color)
            self.screen.blit(b_lbl, (act_rect.x + 12, act_rect.y + 25))
        else:
            r_lbl = self.f_tiny.render(f"Tactical Regime: {gto.regime} (Dealer Bust: {gto.dealer_bust_prob*100:.1f}%)", True, self.PALETTE["text_dim"])
            self.screen.blit(r_lbl, (act_rect.x + 12, act_rect.y + 26))

        # 4 Action Buttons: HIT, STAND, DOUBLE 2x, SPLIT
        human_turn = (not game.sim_mode) and (game.phase == "CHALLENGER_TURN")
        h_cur = game.active_challenger_hand
        can_dbl = human_turn and h_cur.can_double
        can_spl = human_turn and h_cur.is_pair and len(game.challenger_hands) == 1

        btn_y = act_rect.bottom - 46
        bw = (act_rect.width - 42) // 4
        self._draw_button(
            "hit",
            pygame.Rect(act_rect.x + 12, btn_y, bw, 36),
            "🃏 HIT (SPACE/H)",
            (20, 92, 74),
            self.PALETTE["cyber_mint"],
            enabled=human_turn,
            font=self.f_bold,
        )
        self._draw_button(
            "stand",
            pygame.Rect(act_rect.x + 18 + bw, btn_y, bw, 36),
            "🛑 STAND (ENTER/K)",
            (108, 78, 18),
            self.PALETTE["gold"],
            enabled=human_turn,
            font=self.f_bold,
        )
        self._draw_button(
            "double",
            pygame.Rect(act_rect.x + 24 + bw * 2, btn_y, bw, 36),
            "💰 DOUBLE 2x (D)",
            (20, 74, 108),
            self.PALETTE["cyber_cyan"],
            enabled=can_dbl,
            font=self.f_bold,
        )
        self._draw_button(
            "split",
            pygame.Rect(act_rect.x + 30 + bw * 3, btn_y, bw, 36),
            "✂ SPLIT PAIR (P)",
            (74, 38, 108),
            self.PALETTE["purple"],
            enabled=can_spl,
            font=self.f_bold,
        )
        cy += act_h + 8

        # 4. Live GTO Strategy Matrix Heatmap (Hard / Soft / Pairs vs Dealer 2..A)
        mat_h = 205
        mat_rect = pygame.Rect(px + 12, cy, pw - 24, mat_h)
        self._draw_strategy_matrix(mat_rect, game, gto)
        cy += mat_h + 8

        # 5. Live Action Ticker
        feed_rect = pygame.Rect(px + 12, cy, pw - 24, py + ph - cy - 10)
        self._draw_action_feed(feed_rect, game)

    def _draw_seat_card(
        self,
        rect: pygame.Rect,
        title: str,
        hands: List[Hand],
        active_idx: int,
        is_turn: bool,
        is_gevva: bool,
        bankroll: float,
        net_units: float,
        game: BlackjackGame,
    ) -> None:
        accent = self.PALETTE["cyber_cyan"] if is_gevva else self.PALETTE["purple"]
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(
            self.screen,
            accent if is_turn else self.PALETTE["panel_border"],
            rect,
            2 if is_turn else 1,
            border_radius=10,
        )

        self.screen.blit(self.f_bold.render(title[:26], True, accent), (rect.x + 12, rect.y + 8))
        u_col = self.PALETTE["cyber_mint"] if net_units >= 0 else self.PALETTE["rose"]
        bank_str = f"${bankroll:,.0f} ({net_units:+.1f}u)"
        b_surf = self.f_bold.render(bank_str, True, u_col)
        self.screen.blit(b_surf, (rect.right - b_surf.get_width() - 12, rect.y + 8))

        # Draw cards for primary hand (and split hand if split occurred)
        for h_idx, h in enumerate(hands[:2]):
            base_x = rect.x + 14 + (h_idx * 138 if len(hands) > 1 else 0)
            base_y = rect.y + 34
            c_step = 24 if len(hands) > 1 else 32
            is_curr_sub = is_turn and (h_idx == active_idx)
            for c_i, c in enumerate(h.cards[:6]):
                self._draw_playing_card(
                    base_x + c_i * c_step,
                    base_y,
                    c,
                    scale=0.88 if len(hands) > 1 else 0.95,
                    highlight_border=accent if is_curr_sub else None,
                )

            # Hand total & outcome badge below cards
            tot_lbl = f"{'S' if h.is_soft else 'H'}{h.total}"
            if h.doubled:
                tot_lbl += " [2x]"
            if h.outcome:
                out_col = (
                    self.PALETTE["cyber_mint"]
                    if h.outcome in ("WIN", "BLACKJACK")
                    else (self.PALETTE["rose"] if h.outcome in ("LOSE", "BUST") else self.PALETTE["gold"])
                )
                tot_lbl += f" • {h.outcome} ({h.net_units:+.1f}u)"
            else:
                out_col = self.PALETTE["text_white"]
            self.screen.blit(self.f_small.render(tot_lbl, True, out_col), (base_x, rect.bottom - 38))

        # Bottom status / Bot Rule selector button
        if not is_gevva:
            self._draw_button(
                "cycle_opponent",
                pygame.Rect(rect.right - 96, rect.bottom - 22, 86, 18),
                "BOT RULE (O)",
                (35, 38, 58),
                self.PALETTE["panel_border"],
                font=self.f_tiny,
            )
            w_txt = f"W/P/L: {game.stats.challenger_wins}/{game.stats.challenger_pushes}/{game.stats.challenger_losses}"
            self.screen.blit(self.f_tiny.render(w_txt, True, self.PALETTE["text_dim"]), (rect.x + 12, rect.bottom - 19))
        else:
            w_txt = (
                f"W/P/L: {game.stats.gevva_wins}/{game.stats.gevva_pushes}/{game.stats.gevva_losses}  |  "
                f"2x Wins: {game.stats.gevva_doubles_won}  Splits: {game.stats.gevva_splits_played}"
            )
            self.screen.blit(self.f_tiny.render(w_txt, True, self.PALETTE["text_dim"]), (rect.x + 12, rect.bottom - 19))

    def _draw_strategy_matrix(
        self,
        rect: pygame.Rect,
        game: BlackjackGame,
        gto: BlackjackGTOEvaluation,
    ) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        tc = game.true_count
        hdr_txt = f"LIVE GTO STOPPING MATRIX (VEGAS S17 + HI-LO TC={tc:+.1f})  [H=Hit  S=Stand  D=Double  P=Split]"
        self.screen.blit(self.f_bold.render(hdr_txt, True, self.PALETTE["cyber_cyan"]), (rect.x + 12, rect.y + 7))

        # Display 8 representative rows spanning Hard, Soft, and Pairs across Dealer 2..10, A
        rows = [
            ("Hard 17+", 17, False, False, 0),
            ("Hard 16", 16, False, False, 0),
            ("Hard 13-15", 14, False, False, 0),
            ("Hard 12", 12, False, False, 0),
            ("Hard 10-11", 11, False, False, 0),
            ("Hard 9", 9, False, False, 0),
            ("Soft 18 (A,7)", 18, True, False, 0),
            ("Pairs (8,8/A,A)", 16, False, True, 8),
        ]
        upcards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        gx, gy = rect.x + 118, rect.y + 28
        cell_w = (rect.width - 130) // 10
        cell_h = 19

        # Column headers (Dealer Upcard 2..A)
        for c_idx, up in enumerate(upcards):
            up_lbl = "A" if up == 11 else str(up)
            is_cur_col = up == gto.dealer_upcard
            col = self.PALETTE["gold"] if is_cur_col else self.PALETTE["text_dim"]
            s = self.f_tiny.render(up_lbl, True, col)
            self.screen.blit(s, s.get_rect(center=(gx + c_idx * cell_w + cell_w // 2, gy + 6)))

        gy += 14

        # Determine which row matches current active hand best
        active_row_idx = 0
        if gto.can_split and gto.is_pair:
            active_row_idx = 7
        elif gto.is_soft:
            active_row_idx = 6
        elif gto.player_total >= 17:
            active_row_idx = 0
        elif gto.player_total == 16:
            active_row_idx = 1
        elif 13 <= gto.player_total <= 15:
            active_row_idx = 2
        elif gto.player_total == 12:
            active_row_idx = 3
        elif gto.player_total in (10, 11):
            active_row_idx = 4
        else:
            active_row_idx = 5

        act_colors = {
            "HIT": (22, 85, 68),
            "STAND": (115, 84, 20),
            "DOUBLE": (18, 78, 115),
            "SPLIT": (82, 42, 120),
        }
        act_code = {"HIT": "H", "STAND": "S", "DOUBLE": "D", "SPLIT": "P"}

        for r_idx, (r_label, tot, soft, pair, p_rank) in enumerate(rows):
            ry = gy + r_idx * cell_h
            is_cur_row = r_idx == active_row_idx
            r_surf = self.f_tiny.render(r_label, True, self.PALETTE["gold"] if is_cur_row else self.PALETTE["text_dim"])
            self.screen.blit(r_surf, (rect.x + 12, ry + 2))

            for c_idx, up in enumerate(upcards):
                cx = gx + c_idx * cell_w
                act = game.solver._canonical_basic_strategy(
                    tot, soft, pair, p_rank, up, can_double=True, can_split=pair, tc=tc
                )
                c_rect = pygame.Rect(cx + 1, ry + 1, cell_w - 2, cell_h - 2)
                pygame.draw.rect(self.screen, act_colors.get(act, (30, 35, 50)), c_rect, border_radius=3)

                if is_cur_row and up == gto.dealer_upcard:
                    pygame.draw.rect(self.screen, (255, 255, 255), c_rect, 2, border_radius=3)

                code_s = self.f_tiny.render(act_code.get(act, "H"), True, self.PALETTE["text_white"])
                self.screen.blit(code_s, code_s.get_rect(center=c_rect.center))

    def _draw_action_feed(self, rect: pygame.Rect, game: BlackjackGame) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        self.screen.blit(
            self.f_bold.render("LIVE SHOE DECISION LOG & GTO EV TELEMETRY:", True, self.PALETTE["text_white"]),
            (rect.x + 12, rect.y + 6),
        )
        y = rect.y + 24
        recent = list(game.events)[-3:]
        if not recent:
            self.screen.blit(
                self.f_small.render("Play your hand (SPACE=Hit, ENTER=Stand, D=Double, P=Split) or press M for Sim!", True, self.PALETTE["text_dim"]),
                (rect.x + 12, y + 4),
            )
            return

        for ev in reversed(recent):
            if y + 15 > rect.bottom - 4:
                break
            col = self.PALETTE["cyber_cyan"] if ev.actor == "GEVVA-T0" else self.PALETTE["purple"]
            if ev.action == "BUST":
                col = self.PALETTE["rose"]
            elif ev.action == "DOUBLE":
                col = self.PALETTE["gold"]
            lat_tag = f" [{ev.latency_ms:.1f}ms]" if ev.latency_ms > 0 else ""
            line = f"[{ev.actor}] {ev.note} (GTO: {ev.gto_optimal}, EV {ev.ev_best:+.2f}u){lat_tag}"
            self.screen.blit(self.f_tiny.render(line[:86], True, col), (rect.x + 12, y))
            y += 16

    # ------------------------------------------------------------------
    # RIGHT VIEWPORT: TIER 0 ENGINE ARCHITECTURE & TELEMETRY
    # ------------------------------------------------------------------
    def _draw_tier0_panel(self, game: BlackjackGame, brain: BaseBlackjackBrain) -> None:
        px, py, pw, ph = 652, 16, self.WIN_WIDTH - 668, self.WIN_HEIGHT - 32
        panel = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], panel, border_radius=12)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], panel, 1, border_radius=12)

        # Header
        hdr = pygame.Rect(px, py, pw, 40)
        pygame.draw.rect(self.screen, self.PALETTE["panel_header"], hdr, border_top_left_radius=12, border_top_right_radius=12)
        t_lbl = self.f_title.render(
            "⚡ TIER 0 ENGINE PIPELINE // ZERO-CASCADE EMBEDDING + AMC",
            True,
            self.PALETTE["cyber_cyan"],
        )
        self.screen.blit(t_lbl, (px + 14, py + 9))

        cy = py + 46
        p50, p95 = brain.latency_stats()
        status_str = f"MODEL: {brain.model_name}  |  LATENCY p50={p50:.2f}ms  p95={p95:.2f}ms"
        self.screen.blit(self.f_bold.render(status_str[:76], True, self.PALETTE["gold"]), (px + 14, cy))
        cy += 22

        # 4 Mode Selector Tabs (Keys 1..4)
        tab_w = (pw - 36) // 4
        for idx, (m_key, m_lbl) in enumerate(self.ENGINE_MODES):
            t_rect = pygame.Rect(px + 12 + idx * (tab_w + 4), cy, tab_w, 26)
            active = brain.engine_mode == m_key
            self._draw_button(
                f"mode_{m_key}",
                t_rect,
                f"[{idx+1}] {m_lbl}",
                (28, 65, 82) if active else (26, 28, 42),
                self.PALETTE["cyber_cyan"] if active else self.PALETTE["panel_border"],
                font=self.f_tiny,
            )
        cy += 32

        res = brain.last_result
        sit = brain.last_situation

        # 1. Inbound Query Vector q Box
        q_box = pygame.Rect(px + 12, cy, pw - 24, 54)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], q_box, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_blue"], q_box, 1, border_radius=8)
        self.screen.blit(
            self.f_bold.render("INBOUND QUERY VECTOR q in R^768 (Gemma Embedding):", True, self.PALETTE["cyber_cyan"]),
            (q_box.x + 10, q_box.y + 5),
        )
        q_str = sit.query_text if sit else "Waiting for Gevva0 hand decision..."
        self.screen.blit(self.f_tiny.render(q_str[:84], True, self.PALETTE["text_white"]), (q_box.x + 10, q_box.y + 22))
        if len(q_str) > 84:
            self.screen.blit(self.f_tiny.render(q_str[84:168], True, self.PALETTE["text_dim"]), (q_box.x + 10, q_box.y + 36))
        cy += 60

        # Connector lines
        mid_x = px + pw // 2
        left_cx = px + pw // 4
        right_cx = px + (pw * 3) // 4
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (mid_x, cy - 6), (mid_x, cy), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (right_cx, cy), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (left_cx, cy + 6), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (right_cx, cy), (right_cx, cy + 6), 2)
        cy += 6

        # 2. Split Branches: (1) Direct Cosine vs (2)/(3) AMC Graph B = (I - Q)^(-1) R
        branch_w = (pw - 32) // 2
        branch_h = 154
        cos_box = pygame.Rect(px + 12, cy, branch_w, branch_h)
        amc_box = pygame.Rect(px + 20 + branch_w, cy, branch_w, branch_h)

        # Left Branch: (1) Direct Cosine
        c_active = brain.engine_mode != "domain_markov"
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], cos_box, border_radius=8)
        pygame.draw.rect(
            self.screen,
            self.PALETTE["cyber_cyan"] if c_active else self.PALETTE["panel_border"],
            cos_box,
            2 if brain.engine_mode == "pure_cosine" else 1,
            border_radius=8,
        )
        self.screen.blit(
            self.f_bold.render("(1) DIRECT COSINE ALIGNMENT", True, self.PALETTE["cyber_cyan"]),
            (cos_box.x + 10, cos_box.y + 6),
        )
        self.screen.blit(
            self.f_tiny.render("Pairwise sim(q, c_i) / tau_temp (0.05)", True, self.PALETTE["text_dim"]),
            (cos_box.x + 10, cos_box.y + 22),
        )

        c_probs = (res.cosine_probs if (res and res.cosine_probs) else {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        c_sims = (res.cosine_sims if (res and res.cosine_sims) else {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
        action_specs = [
            ("A", "HIT", self.PALETTE["cyber_mint"]),
            ("B", "STAND", self.PALETTE["gold"]),
            ("C", "DOUBLE", self.PALETTE["cyber_cyan"]),
            ("D", "SPLIT", self.PALETTE["purple"]),
        ]
        for idx, (k_lbl, a_name, a_col) in enumerate(action_specs):
            self._draw_compact_prob_row(
                cos_box.x + 10,
                cos_box.y + 40 + idx * 27,
                cos_box.width - 20,
                f"[{k_lbl}] {a_name:<6} (s={c_sims.get(k_lbl, 0.0):.2f})",
                c_probs.get(k_lbl, 0.25),
                a_col,
            )

        # Right Branch: (2)/(3) AMC Graph B = (I - Q)^(-1) R
        m_active = brain.engine_mode != "pure_cosine"
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], amc_box, border_radius=8)
        pygame.draw.rect(
            self.screen,
            self.PALETTE["purple"] if m_active else self.PALETTE["panel_border"],
            amc_box,
            2 if brain.engine_mode in ("domain_markov", "default_39") else 1,
            border_radius=8,
        )
        self.screen.blit(
            self.f_bold.render("AMC GRAPH: B = (I - Q)⁻¹ R", True, self.PALETTE["purple"]),
            (amc_box.x + 10, amc_box.y + 6),
        )
        m_cnt = res.landmarks_count if res else 116
        scaf_lbl = (
            f"(2) Default 39-Landmarks (M={m_cnt})"
            if brain.engine_mode == "default_39"
            else f"(3) Blackjack Manifold (M={m_cnt})"
        )
        if not m_active:
            scaf_lbl = "Bypassed (Pure Cosine Mode)"
        self.screen.blit(self.f_tiny.render(scaf_lbl, True, self.PALETTE["text_dim"]), (amc_box.x + 10, amc_box.y + 22))

        m_probs = (res.markov_probs if (res and res.markov_probs) else {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        for idx, (k_lbl, a_name, a_col) in enumerate(action_specs):
            self._draw_compact_prob_row(
                amc_box.x + 10,
                amc_box.y + 40 + idx * 27,
                amc_box.width - 20,
                f"[{k_lbl}] Sink {a_name}",
                m_probs.get(k_lbl, 0.25),
                a_col,
            )
        cy += branch_h

        # Merge lines into (4) Production Hybrid Ensemble
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (left_cx, cy + 6), 2)
        pygame.draw.line(self.screen, self.PALETTE["purple"], (right_cx, cy), (right_cx, cy + 6), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_mint"], (left_cx, cy + 6), (right_cx, cy + 6), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_mint"], (mid_x, cy + 6), (mid_x, cy + 12), 2)
        cy += 12

        # 3. (4) Production Hybrid Ensemble Box (4 Actions: HIT, STAND, DOUBLE, SPLIT)
        hyb_h = 152
        hyb_box = pygame.Rect(px + 12, cy, pw - 24, hyb_h)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], hyb_box, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_mint"], hyb_box, 2, border_radius=8)

        blend_formula = {
            "hybrid_domain": "P_final = 0.65 · Cosine + 0.35 · Markov_Domain",
            "domain_markov": "P_final = 1.00 · Markov_Domain (B = (I - Q)⁻¹ R)",
            "default_39": "P_final = 0.65 · Cosine + 0.35 · Markov_39-Landmarks",
            "pure_cosine": "P_final = 1.00 · Direct Cosine Alignment",
        }.get(brain.engine_mode, "P_final = 0.65 · Cosine + 0.35 · Markov")

        self.screen.blit(
            self.f_bold.render(f"(4) PRODUCTION ENSEMBLE // {blend_formula}", True, self.PALETTE["cyber_mint"]),
            (hyb_box.x + 12, hyb_box.y + 6),
        )

        f_probs = res.probs if res else {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        win_lbl = res.label if res else "A"
        full_labels = [
            ("A", "[A] HIT (DRAW CARD)", self.PALETTE["cyber_mint"]),
            ("B", "[B] STAND (HOLD PAT)", self.PALETTE["gold"]),
            ("C", "[C] DOUBLE DOWN (2x)", self.PALETTE["cyber_cyan"]),
            ("D", "[D] SPLIT PAIR (2x)", self.PALETTE["purple"]),
        ]
        for idx, (k_lbl, full_txt, col) in enumerate(full_labels):
            self._draw_ensemble_bar(
                hyb_box.x + 12,
                hyb_box.y + 28 + idx * 29,
                hyb_box.width - 24,
                full_txt,
                f_probs.get(k_lbl, 0.25),
                selected=(win_lbl == k_lbl),
                color=col,
            )
        cy += hyb_h + 8

        # 4. Tripwire Diagnostics & [ EXIT TIER 0 ] Badge
        trip_h = 80
        trip_box = pygame.Rect(px + 12, cy, pw - 24, trip_h)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], trip_box, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], trip_box, 1, border_radius=8)

        ent_val = res.entropy if res else 0.11
        mar_val = res.margin if res else 0.78
        ent_ok = ent_val <= 0.65
        mar_ok = mar_val >= 0.25
        emb_ms = res.embed_latency_ms if res else 1.72
        grp_ms = res.graph_latency_ms if res else 0.11

        self.screen.blit(
            self.f_bold.render("TRIPWIRE DIAGNOSTICS & ZERO-CASCADE EARLY EXIT:", True, self.PALETTE["text_white"]),
            (trip_box.x + 12, trip_box.y + 6),
        )
        ent_col = self.PALETTE["cyber_mint"] if ent_ok else self.PALETTE["amber"]
        mar_col = self.PALETTE["cyber_mint"] if mar_ok else self.PALETTE["amber"]
        self.screen.blit(
            self.f_small.render(f"• Shannon Entropy H = {ent_val:.3f} <= τ₁ (0.65) [{'PASS' if ent_ok else 'NEAR BOUNDARY'}]", True, ent_col),
            (trip_box.x + 12, trip_box.y + 26),
        )
        self.screen.blit(
            self.f_small.render(f"• Logit Margin    Δ = {mar_val:.3f} >= τ₂ (0.25) [{'PASS' if mar_ok else 'TIGHT MARGIN'}]", True, mar_col),
            (trip_box.x + 12, trip_box.y + 44),
        )
        lat_detail = f"Embed: {emb_ms:.2f}ms | AMC Solve: {grp_ms:.3f}ms | GTO Parity: {brain.gto_match_rate:.1f}%"
        self.screen.blit(self.f_tiny.render(lat_detail, True, self.PALETTE["text_dim"]), (trip_box.x + 12, trip_box.y + 62))

        badge_rect = pygame.Rect(trip_box.right - 158, trip_box.y + 15, 144, 50)
        pygame.draw.rect(self.screen, (18, 56, 44), badge_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_mint"], badge_rect, 2, border_radius=8)
        b1 = self.f_bold.render("[ EXIT TIER 0 ]", True, self.PALETTE["cyber_mint"])
        b2 = self.f_tiny.render("No Tier 1/2 Cascade", True, self.PALETTE["text_white"])
        self.screen.blit(b1, b1.get_rect(center=(badge_rect.centerx, badge_rect.centery - 8)))
        self.screen.blit(b2, b2.get_rect(center=(badge_rect.centerx, badge_rect.centery + 10)))

        cy += trip_h + 8

        # 5. High-Speed Simulation Bankroll & EV Edge Scoreboard
        sim_rect = pygame.Rect(px + 12, cy, pw - 24, py + ph - cy - 10)
        self._draw_sim_scoreboard(sim_rect, game, brain)

    def _draw_compact_prob_row(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        prob: float,
        color: Tuple[int, int, int],
    ) -> None:
        self.screen.blit(self.f_tiny.render(label, True, self.PALETTE["text_white"]), (x, y))
        pct_surf = self.f_bold.render(f"{prob*100:5.1f}%", True, color)
        self.screen.blit(pct_surf, (x + w - pct_surf.get_width(), y - 1))
        bar_y = y + 14
        pygame.draw.rect(self.screen, (14, 17, 26), (x, bar_y, w, 8), border_radius=3)
        fill_w = int(w * max(0.0, min(1.0, prob)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (x, bar_y, fill_w, 8), border_radius=3)

    def _draw_ensemble_bar(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        prob: float,
        selected: bool,
        color: Tuple[int, int, int],
    ) -> None:
        row_rect = pygame.Rect(x, y, w, 26)
        pygame.draw.rect(self.screen, (20, 26, 38) if selected else (15, 18, 28), row_rect, border_radius=5)
        pygame.draw.rect(
            self.screen,
            color if selected else self.PALETTE["panel_border"],
            row_rect,
            2 if selected else 1,
            border_radius=5,
        )
        prefix = "▶ " if selected else "  "
        lbl_surf = self.f_bold.render(
            f"{prefix}{label}",
            True,
            self.PALETTE["text_white"] if selected else self.PALETTE["text_dim"],
        )
        self.screen.blit(lbl_surf, (x + 8, y + 5))

        bar_x = x + 218
        bar_w = w - 295
        pygame.draw.rect(self.screen, (11, 14, 22), (bar_x, y + 7, bar_w, 12), border_radius=4)
        fill_w = int(bar_w * max(0.0, min(1.0, prob)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (bar_x, y + 7, fill_w, 12), border_radius=4)

        pct_surf = self.f_bold.render(
            f"{prob*100:5.1f}%",
            True,
            color if selected else self.PALETTE["text_dim"],
        )
        self.screen.blit(pct_surf, (bar_x + bar_w + 8, y + 5))

    def _draw_sim_scoreboard(self, rect: pygame.Rect, game: BlackjackGame, brain: BaseBlackjackBrain) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        self.screen.blit(
            self.f_bold.render("HIGH-SPEED SIMULATION SCOREBOARD & GTO EV ADVANTAGE:", True, self.PALETTE["gold"]),
            (rect.x + 12, rect.y + 7),
        )
        self._draw_button(
            "reset_stats",
            pygame.Rect(rect.right - 102, rect.y + 5, 90, 20),
            "CLEAR STATS",
            (40, 28, 38),
            self.PALETTE["panel_border"],
            font=self.f_tiny,
        )

        st = game.stats
        y = rect.y + 30
        edge_u = st.gevva_edge_over_challenger_units
        line1 = (
            f"HANDS: {st.hands_played:4d}  |  GEVVA-T0 NET: {st.gevva_net_units:+6.1f}u  |  "
            f"CHALLENGER NET: {st.challenger_net_units:+6.1f}u  |  EDGE: {edge_u:+5.1f}u"
        )
        self.screen.blit(self.f_bold.render(line1, True, self.PALETTE["text_white"]), (rect.x + 12, y))
        y += 22

        # Bankroll advantage bar
        bar_w = rect.width - 24
        pygame.draw.rect(self.screen, (35, 25, 42), (rect.x + 12, y, bar_w, 13), border_radius=5)
        ratio = float(min(0.95, max(0.05, 0.5 + edge_u / max(20.0, st.hands_played * 0.25))))
        fill_w = int(bar_w * ratio)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_mint"], (rect.x + 12, y, fill_w, 13), border_radius=5)
        mid_bar = rect.x + 12 + bar_w // 2
        pygame.draw.line(self.screen, self.PALETTE["text_white"], (mid_bar, y - 2), (mid_bar, y + 15), 2)
        y += 20

        line2 = (
            f"2x Doubles Won: {st.gevva_doubles_won:3d}  |  Pair Splits: {st.gevva_splits_played:3d}  |  "
            f"TC Warps: {st.gevva_tc_warps:3d}  |  Opponent EV Blunders: {st.challenger_ev_blunders:3d}"
        )
        self.screen.blit(self.f_tiny.render(line2, True, self.PALETTE["cyber_cyan"]), (rect.x + 12, y))

        hotkeys = "KEYS: [SPACE/H] Hit  [ENTER/K] Stand  [D] Double  [P] Split  [M] Play/Sim  [S] Speed  [T] +50 Hands  [1-4] Mode"
        self.screen.blit(self.f_tiny.render(hotkeys, True, self.PALETTE["text_dim"]), (rect.x + 12, rect.bottom - 16))

    def _update_particles(self, dt: float) -> None:
        for p in self.particles[:]:
            p.life += dt
            if p.life >= p.max_life:
                self.particles.remove(p)
                continue
            p.vy += 320.0 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            r = max(2, int(p.size * (1.0 - 0.5 * (p.life / p.max_life))))
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), r)
            pygame.draw.circle(self.screen, self.PALETTE["gold_dark"], (int(p.x), int(p.y)), r, 1)
