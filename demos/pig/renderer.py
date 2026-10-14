"""
Gevva0 Game of Pig: Cute Interactive Arcade & High-Speed Simulation Renderer (Pygame).

Left Viewport:
- Cute animated Pig mascots ("Sir Oinks-a-Lot" vs "Gevva-T0 Cyber-Pig")
- 3D-styled tumbling die (with cute pink pig snout on '1'), golden coin stack for Turn Pot
- Live Dynamic GTO Stopping Boundary Chart k*(i, j) vs flat "Hold at 20" heuristic
- Interactive clickable buttons for Roll, Hold, Play/Sim mode, Sim speed, and +50 Turbo Games

Right Viewport:
- Live interactive diagram of the 4 Tier0Engine components:
  (1) Direct Cosine sim(q, c_i)
  (2) Default 39-Landmarks & (3) Pig Domain Manifold (M=104) -> B = (I - Q)^(-1) R
  (4) Production Hybrid Ensemble: P_final = 0.65 * Cosine + 0.35 * Markov
  Tripwire Checks: Shannon Entropy H <= 0.65, Logit Margin Delta >= 0.25 -> [EXIT TIER 0]
- High-Speed Simulation Win-Rate Scoreboard & GTO Warp Telemetry
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from brain import BasePigBrain
from game import PigGame


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float
    kind: str = "coin"  # "coin" | "snout" | "spark"


class PigRenderer:
    WIN_WIDTH = 1280
    WIN_HEIGHT = 820

    PALETTE = {
        "bg": (14, 15, 24),
        "panel_bg": (21, 23, 36),
        "card_bg": (28, 31, 48),
        "panel_border": (52, 58, 84),
        "panel_header": (32, 36, 58),
        "pig_pink": (255, 145, 185),
        "pig_pink_dark": (215, 85, 135),
        "pig_snout": (255, 110, 160),
        "cyber_cyan": (0, 235, 255),
        "cyber_mint": (50, 245, 165),
        "cyber_blue": (45, 130, 255),
        "gold": (255, 210, 45),
        "gold_dark": (205, 150, 15),
        "rose": (255, 75, 110),
        "amber": (255, 165, 35),
        "purple": (175, 115, 255),
        "text_white": (245, 247, 252),
        "text_dim": (148, 156, 180),
        "text_muted": (100, 108, 132),
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
            "GAME OF PIG // GEVVA0 TIER 0 OPTIMAL STOPPING SHOWCASE (ONNX + ABSORBING MARKOV CHAIN)"
        )
        self._init_fonts()
        self.particles: list[Particle] = []
        self.time_sec: float = 0.0
        self.buttons: dict[str, pygame.Rect] = {}
        self._prev_holds = (0, 0)
        self._prev_busts = (0, 0)

    def _init_fonts(self) -> None:
        font_mono = "Consolas"
        font_sans = "Segoe UI"
        self.f_giant = pygame.font.SysFont(font_sans, 34, bold=True)
        self.f_huge = pygame.font.SysFont(font_sans, 24, bold=True)
        self.f_title = pygame.font.SysFont(font_sans, 17, bold=True)
        self.f_header = pygame.font.SysFont(font_mono, 14, bold=True)
        self.f_bold = pygame.font.SysFont(font_mono, 12, bold=True)
        self.f_regular = pygame.font.SysFont(font_mono, 12)
        self.f_small = pygame.font.SysFont(font_mono, 11)
        self.f_tiny = pygame.font.SysFont(font_mono, 10)

    def spawn_coins(self, x: float, y: float, count: int = 16) -> None:
        for _ in range(count):
            ang = random.uniform(-math.pi * 0.85, -math.pi * 0.15)
            spd = random.uniform(80, 220)
            self.particles.append(
                Particle(
                    x=x + random.uniform(-15, 15),
                    y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    life=0.0,
                    max_life=random.uniform(0.55, 0.95),
                    color=self.PALETTE["gold"],
                    size=random.uniform(5, 8),
                    kind="coin",
                )
            )

    def spawn_pig_out_burst(self, x: float, y: float, count: int = 18) -> None:
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(60, 190)
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    life=0.0,
                    max_life=random.uniform(0.45, 0.85),
                    color=self.PALETTE["pig_pink"] if random.random() < 0.6 else self.PALETTE["rose"],
                    size=random.uniform(5, 9),
                    kind="snout",
                )
            )

    def render_loading(self, title_msg: str, detail: str = "") -> None:
        self.screen.fill(self.PALETTE["bg"])
        cx, cy = self.WIN_WIDTH // 2, self.WIN_HEIGHT // 2
        box = pygame.Rect(cx - 320, cy - 130, 640, 260)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], box, border_radius=14)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_cyan"], box, 2, border_radius=14)

        # Draw cute pig face in loading box
        self._draw_cute_pig(cx - 210, cy, radius=42, is_cyber=False, mood="happy")
        self._draw_cute_pig(cx + 210, cy, radius=42, is_cyber=True, mood="cyber")

        t_surf = self.f_huge.render("GEVVA0 TIER 0 // GAME OF PIG", True, self.PALETTE["gold"])
        self.screen.blit(t_surf, t_surf.get_rect(center=(cx, cy - 65)))

        m_surf = self.f_title.render(title_msg, True, self.PALETTE["text_white"])
        self.screen.blit(m_surf, m_surf.get_rect(center=(cx, cy - 10)))

        if detail:
            d_surf = self.f_regular.render(detail, True, self.PALETTE["cyber_cyan"])
            self.screen.blit(d_surf, d_surf.get_rect(center=(cx, cy + 25)))

        dots = "." * (int(pygame.time.get_ticks() / 300) % 4)
        w_surf = self.f_regular.render(
            f"Solving 100x100x100 Absorbing MDP & Initializing ONNX{dots:<3}",
            True,
            self.PALETTE["text_dim"],
        )
        self.screen.blit(w_surf, w_surf.get_rect(center=(cx, cy + 65)))
        pygame.display.flip()

    def button_at(self, pos: tuple[int, int]) -> str | None:
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                return name
        return None

    def render(self, game: PigGame, brain: BasePigBrain, dt: float) -> None:
        self.time_sec += dt
        if game.die_anim_timer > 0:
            game.die_anim_timer = max(0.0, game.die_anim_timer - dt)
        if game.pig_out_timer > 0:
            game.pig_out_timer = max(0.0, game.pig_out_timer - dt)
        if game.hold_anim_timer > 0:
            game.hold_anim_timer = max(0.0, game.hold_anim_timer - dt)
        if game.banner_timer > 0:
            game.banner_timer = max(0.0, game.banner_timer - dt)

        # Trigger particles on new holds or busts in interactive / 1x mode
        cur_holds = (game.stats.challenger_holds, game.stats.gevva_holds)
        cur_busts = (game.stats.challenger_busts, game.stats.gevva_busts)
        if cur_holds != self._prev_holds and (not game.sim_mode or game.sim_speed == 1):
            target_x = 180 if cur_holds[0] > self._prev_holds[0] else 470
            self.spawn_coins(target_x, 220, count=14)
        if cur_busts != self._prev_busts and (not game.sim_mode or game.sim_speed == 1):
            self.spawn_pig_out_burst(325, 360, count=16)
        self._prev_holds = cur_holds
        self._prev_busts = cur_busts

        self.buttons.clear()
        self.screen.fill(self.PALETTE["bg"])

        # Left Panel: The Pigpen Arena
        self._draw_arena_panel(game, brain)

        # Right Panel: Tier 0 Engine Architecture & Telemetry
        self._draw_tier0_panel(game, brain)

        # Update & Draw Particles
        self._update_particles(dt)

        pygame.display.flip()

    # ------------------------------------------------------------------
    # CUTE PIG MASCOT & DIE DRAWING
    # ------------------------------------------------------------------
    def _draw_cute_pig(
        self,
        cx: int,
        cy: int,
        radius: int = 34,
        is_cyber: bool = False,
        mood: str = "happy",
        active_glow: bool = False,
    ) -> None:
        bob = int(math.sin(self.time_sec * 4.5 + (1.5 if is_cyber else 0.0)) * 3.0)
        cy = cy + bob

        if active_glow:
            glow_col = self.PALETTE["cyber_cyan"] if is_cyber else self.PALETTE["pig_pink"]
            pygame.draw.circle(self.screen, glow_col, (cx, cy), radius + 7, 2)

        body_col = (210, 235, 250) if is_cyber else self.PALETTE["pig_pink"]
        ear_col = (130, 200, 235) if is_cyber else self.PALETTE["pig_pink_dark"]
        snout_col = (100, 225, 245) if is_cyber else self.PALETTE["pig_snout"]

        # Floppy / perked pig ears
        ear_r = int(radius * 0.42)
        pygame.draw.circle(self.screen, ear_col, (cx - int(radius * 0.68), cy - int(radius * 0.68)), ear_r)
        pygame.draw.circle(self.screen, ear_col, (cx + int(radius * 0.68), cy - int(radius * 0.68)), ear_r)
        pygame.draw.circle(self.screen, body_col, (cx - int(radius * 0.65), cy - int(radius * 0.62)), int(ear_r * 0.6))
        pygame.draw.circle(self.screen, body_col, (cx + int(radius * 0.65), cy - int(radius * 0.62)), int(ear_r * 0.6))

        # Cyber antenna for Gevva-T0 pig
        if is_cyber:
            pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (cx, cy - radius), (cx, cy - radius - 13), 3)
            pulse_r = 4 + int((math.sin(self.time_sec * 8.0) + 1) * 1.5)
            pygame.draw.circle(self.screen, self.PALETTE["cyber_mint"], (cx, cy - radius - 15), pulse_r)

        # Head circle
        pygame.draw.circle(self.screen, body_col, (cx, cy), radius)
        pygame.draw.circle(
            self.screen,
            self.PALETTE["cyber_cyan"] if is_cyber else self.PALETTE["pig_pink_dark"],
            (cx, cy),
            radius,
            2,
        )

        # Rosy cheeks
        cheek_col = (140, 215, 255) if is_cyber else (255, 105, 155)
        pygame.draw.circle(self.screen, cheek_col, (cx - int(radius * 0.55), cy + int(radius * 0.18)), int(radius * 0.18))
        pygame.draw.circle(self.screen, cheek_col, (cx + int(radius * 0.55), cy + int(radius * 0.18)), int(radius * 0.18))

        # Eyes / Cyber Visor
        eye_y = cy - int(radius * 0.22)
        eye_dx = int(radius * 0.36)
        if mood == "dizzy":
            # X eyes on Pig Out!
            for ex in (cx - eye_dx, cx + eye_dx):
                pygame.draw.line(self.screen, (40, 20, 35), (ex - 5, eye_y - 5), (ex + 5, eye_y + 5), 2)
                pygame.draw.line(self.screen, (40, 20, 35), (ex - 5, eye_y + 5), (ex + 5, eye_y - 5), 2)
        elif is_cyber:
            # Cute glowing cyber visor + eyes
            visor_rect = pygame.Rect(cx - eye_dx - 8, eye_y - 7, (eye_dx + 8) * 2, 14)
            pygame.draw.rect(self.screen, (18, 35, 58), visor_rect, border_radius=6)
            pygame.draw.rect(self.screen, self.PALETTE["cyber_cyan"], visor_rect, 1, border_radius=6)
            pygame.draw.circle(self.screen, self.PALETTE["cyber_mint"], (cx - eye_dx, eye_y), 4)
            pygame.draw.circle(self.screen, self.PALETTE["cyber_mint"], (cx + eye_dx, eye_y), 4)
        else:
            # Cute glossy pig eyes
            for ex in (cx - eye_dx, cx + eye_dx):
                pygame.draw.circle(self.screen, (30, 22, 38), (ex, eye_y), 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (ex - 2, eye_y - 2), 2)

        # Pig Snout oval
        snout_w, snout_h = int(radius * 0.72), int(radius * 0.48)
        snout_rect = pygame.Rect(cx - snout_w // 2, cy + int(radius * 0.02), snout_w, snout_h)
        pygame.draw.ellipse(self.screen, snout_col, snout_rect)
        pygame.draw.ellipse(self.screen, (150, 50, 90) if not is_cyber else (30, 130, 160), snout_rect, 1)
        # Nostrils
        nostril_col = (110, 30, 65) if not is_cyber else (15, 75, 95)
        pygame.draw.circle(self.screen, nostril_col, (cx - int(snout_w * 0.22), snout_rect.centery), 3)
        pygame.draw.circle(self.screen, nostril_col, (cx + int(snout_w * 0.22), snout_rect.centery), 3)

        # Smile / Oink mouth below snout
        mouth_y = snout_rect.bottom + 4
        if mood == "dizzy":
            pygame.draw.circle(self.screen, (90, 20, 45), (cx, mouth_y + 2), 4)
        else:
            pygame.draw.arc(
                self.screen,
                (90, 25, 55) if not is_cyber else (25, 95, 120),
                pygame.Rect(cx - 8, mouth_y - 5, 16, 9),
                math.pi,
                math.tau,
                2,
            )

    def _draw_die(self, cx: int, cy: int, value: int, rolling: bool = False, pig_out: bool = False) -> None:
        size = 76
        if rolling:
            jitter_x = int(math.sin(self.time_sec * 45.0) * 5)
            jitter_y = int(math.cos(self.time_sec * 38.0) * 5)
            cx += jitter_x
            cy += jitter_y
            disp_val = (int(self.time_sec * 24) % 6) + 1
        else:
            disp_val = value

        # Shadow
        shadow_rect = pygame.Rect(cx - size // 2 + 4, cy - size // 2 + 5, size, size)
        pygame.draw.rect(self.screen, (8, 9, 15), shadow_rect, border_radius=16)

        # Die body
        die_rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        if disp_val == 1 and not rolling:
            bg_col = (58, 22, 36)
            bdr_col = self.PALETTE["rose"]
        else:
            bg_col = (245, 245, 252)
            bdr_col = self.PALETTE["cyber_cyan"] if rolling else (180, 190, 215)

        pygame.draw.rect(self.screen, bg_col, die_rect, border_radius=16)
        pygame.draw.rect(self.screen, bdr_col, die_rect, 3, border_radius=16)

        if disp_val == 1:
            # Cute Pink Pig Snout on the '1' face!
            sn_rect = pygame.Rect(cx - 20, cy - 14, 40, 28)
            pygame.draw.ellipse(self.screen, self.PALETTE["pig_snout"], sn_rect)
            pygame.draw.ellipse(self.screen, self.PALETTE["pig_pink"], sn_rect, 2)
            pygame.draw.circle(self.screen, (110, 25, 60), (cx - 8, cy), 4)
            pygame.draw.circle(self.screen, (110, 25, 60), (cx + 8, cy), 4)
            lbl = self.f_tiny.render("PIG 1", True, self.PALETTE["pig_pink"])
            self.screen.blit(lbl, lbl.get_rect(center=(cx, cy + 25)))
            return

        # Standard pips 2..6
        pip_col = (28, 34, 56)
        offs = 19
        pip_coords = {
            2: [(-offs, -offs), (offs, offs)],
            3: [(-offs, -offs), (0, 0), (offs, offs)],
            4: [(-offs, -offs), (offs, -offs), (-offs, offs), (offs, offs)],
            5: [(-offs, -offs), (offs, -offs), (0, 0), (-offs, offs), (offs, offs)],
            6: [(-offs, -offs), (offs, -offs), (-offs, 0), (offs, 0), (-offs, offs), (offs, offs)],
        }
        for dx, dy in pip_coords.get(disp_val, [(0, 0)]):
            pygame.draw.circle(self.screen, pip_col, (cx + dx, cy + dy), 6)

    def _draw_button(
        self,
        key: str,
        rect: pygame.Rect,
        label: str,
        bg_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        text_color: tuple[int, int, int] = (245, 247, 252),
        enabled: bool = True,
        font: pygame.font.Font | None = None,
    ) -> None:
        self.buttons[key] = rect
        f = font or self.f_bold
        mx, my = pygame.mouse.get_pos()
        hovered = enabled and rect.collidepoint((mx, my))
        if not enabled:
            bg = (22, 24, 34)
            bdr = (45, 48, 65)
            txt = (85, 90, 110)
        else:
            bg = tuple(min(255, c + (22 if hovered else 0)) for c in bg_color)
            bdr = border_color
            txt = text_color

        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, bdr, rect, 2 if hovered else 1, border_radius=8)
        surf = f.render(label, True, txt)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    # ------------------------------------------------------------------
    # LEFT VIEWPORT: THE PIGPEN ARENA
    # ------------------------------------------------------------------
    def _draw_arena_panel(self, game: PigGame, brain: BasePigBrain) -> None:
        px, py, pw, ph = 16, 16, 624, self.WIN_HEIGHT - 32
        panel = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], panel, border_radius=12)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], panel, 1, border_radius=12)

        # Header
        hdr = pygame.Rect(px, py, pw, 40)
        pygame.draw.rect(self.screen, self.PALETTE["panel_header"], hdr, border_top_left_radius=12, border_top_right_radius=12)
        t_lbl = self.f_title.render("🐖 THE PIGPEN ARENA // OPTIMAL STOPPING BATTLE", True, self.PALETTE["pig_pink"])
        self.screen.blit(t_lbl, (px + 14, py + 9))

        # Top Control Bar (Play vs Sim, Speed, Opponent, Turbo +50, Reset)
        cy = py + 48
        mode_lbl = "MODE: HIGH-SPEED SIM (M)" if game.sim_mode else "MODE: 1v1 PLAY vs GEVVA0 (M)"
        mode_bg = (22, 58, 72) if game.sim_mode else (68, 28, 56)
        mode_bdr = self.PALETTE["cyber_cyan"] if game.sim_mode else self.PALETTE["pig_pink"]
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
            "⚡ +50 SIM GAMES (T)",
            (32, 54, 46),
            self.PALETTE["cyber_mint"],
            font=self.f_small,
        )

        self._draw_button(
            "new_game",
            pygame.Rect(px + 514, cy, 98, 28),
            "NEW GAME (N)",
            (36, 38, 56),
            self.PALETTE["panel_border"],
            font=self.f_small,
        )
        cy += 36

        # Player Cards (Challenger vs Gevva0 Tier 0)
        card_w = (pw - 36) // 2
        card_h = 152
        p0_rect = pygame.Rect(px + 12, cy, card_w, card_h)
        p1_rect = pygame.Rect(px + 24 + card_w, cy, card_w, card_h)

        for idx, rect in enumerate((p0_rect, p1_rect)):
            is_active = (game.turn == idx) and (game.winner is None)
            is_cyber = idx == 1
            accent = self.PALETTE["cyber_cyan"] if is_cyber else self.PALETTE["pig_pink"]
            pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
            pygame.draw.rect(
                self.screen,
                accent if is_active else self.PALETTE["panel_border"],
                rect,
                2 if is_active else 1,
                border_radius=10,
            )

            # Cute Pig Mascot
            mood = "dizzy" if (game.pig_out_timer > 0 and game.turn != idx) else "happy"
            self._draw_cute_pig(
                rect.x + 48,
                rect.y + 56,
                radius=31,
                is_cyber=is_cyber,
                mood=mood,
                active_glow=is_active,
            )

            # Name & Turn Badge
            p_name = "GEVVA0 TIER 0 (AMC)" if is_cyber else game.challenger_name
            name_surf = self.f_bold.render(p_name[:24], True, accent)
            self.screen.blit(name_surf, (rect.x + 92, rect.y + 12))

            if is_active:
                badge_txt = "● ACTIVE TURN"
                b_col = self.PALETTE["cyber_mint"] if is_cyber else self.PALETTE["gold"]
            elif game.winner == idx:
                badge_txt = "🏆 WINNER!"
                b_col = self.PALETTE["gold"]
            else:
                badge_txt = "WAITING..."
                b_col = self.PALETTE["text_muted"]
            self.screen.blit(self.f_tiny.render(badge_txt, True, b_col), (rect.x + 92, rect.y + 30))

            # Score Readout
            sc = game.scores[idx]
            pend = game.turn_bank if is_active else 0
            sc_surf = self.f_giant.render(f"{sc}", True, self.PALETTE["text_white"])
            self.screen.blit(sc_surf, (rect.x + 92, rect.y + 46))

            if pend > 0:
                pend_surf = self.f_title.render(f"+{pend} -> {sc + pend}", True, self.PALETTE["gold"])
                self.screen.blit(pend_surf, (rect.x + 92 + sc_surf.get_width() + 8, rect.y + 56))
            else:
                pts_lbl = self.f_small.render("/ 100 PTS", True, self.PALETTE["text_dim"])
                self.screen.blit(pts_lbl, (rect.x + 92 + sc_surf.get_width() + 6, rect.y + 62))

            # Progress Bar to 100
            bar_x, bar_y = rect.x + 14, rect.bottom - 38
            bar_w, bar_h = rect.width - 28, 14
            pygame.draw.rect(self.screen, (15, 17, 26), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
            bank_w = int(bar_w * min(1.0, sc / 100.0))
            if bank_w > 0:
                pygame.draw.rect(self.screen, accent, (bar_x, bar_y, bank_w, bar_h), border_radius=6)
            if pend > 0:
                tot_w = int(bar_w * min(1.0, (sc + pend) / 100.0))
                pend_w = max(0, tot_w - bank_w)
                if pend_w > 0:
                    pygame.draw.rect(
                        self.screen,
                        self.PALETTE["gold"],
                        (bar_x + bank_w, bar_y, pend_w, bar_h),
                        border_radius=4,
                    )

            # Sub-bar win count
            wins = game.stats.gevva_wins if is_cyber else game.stats.challenger_wins
            w_rate = game.stats.gevva_win_rate if is_cyber else game.stats.challenger_win_rate
            sub_txt = f"Match Wins: {wins} ({w_rate:.1f}%)"
            self.screen.blit(self.f_tiny.render(sub_txt, True, self.PALETTE["text_dim"]), (bar_x, rect.bottom - 19))

            if not is_cyber:
                # Cycle Opponent button in corner of P0 card
                self._draw_button(
                    "cycle_opponent",
                    pygame.Rect(rect.right - 92, rect.bottom - 21, 80, 17),
                    "BOT RULE (O)",
                    (35, 38, 58),
                    self.PALETTE["panel_border"],
                    font=self.f_tiny,
                )

        cy += card_h + 10

        # Center Dice Stage & Turn Bank Box
        stage_h = 185
        stage_rect = pygame.Rect(px + 12, cy, pw - 24, stage_h)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], stage_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], stage_rect, 1, border_radius=10)

        # Draw 3D Tumbling Die on Left of Stage
        die_cx, die_cy = stage_rect.x + 76, stage_rect.y + 68
        self._draw_die(
            die_cx,
            die_cy,
            game.last_die,
            rolling=(game.die_anim_timer > 0),
            pig_out=(game.pig_out_timer > 0),
        )

        # Draw Golden Coin Stack for Turn Pot
        pot_x = stage_rect.x + 155
        pot_base_y = stage_rect.y + 105
        n_coins = min(14, (game.turn_bank + 2) // 3) if game.turn_bank > 0 else 0
        for c_i in range(n_coins):
            cy_coin = pot_base_y - c_i * 5
            c_rect = pygame.Rect(pot_x, cy_coin, 32, 8)
            pygame.draw.ellipse(self.screen, self.PALETTE["gold"], c_rect)
            pygame.draw.ellipse(self.screen, self.PALETTE["gold_dark"], c_rect, 1)

        # Turn Bank & Exact GTO Analysis Readout
        info_x = stage_rect.x + 202
        gto = game.current_gto()
        pot_title = self.f_huge.render(f"TURN POT: {game.turn_bank} PTS", True, self.PALETTE["gold"])
        self.screen.blit(pot_title, (info_x, stage_rect.y + 12))

        rolls_str = ", ".join(str(r) for r in game.turn_rolls[-10:]) if game.turn_rolls else "Fresh turn (no rolls yet)"
        self.screen.blit(
            self.f_small.render(f"Turn Rolls: [{rolls_str}]", True, self.PALETTE["text_dim"]),
            (info_x, stage_rect.y + 42),
        )

        # GTO Threshold vs Human "Hold at 20" comparison
        th_diff = gto.hold_threshold - 20
        if th_diff >= 4:
            warp_tag = f"AGGRESSIVE WARP (+{th_diff} vs 20)"
            warp_col = self.PALETTE["cyber_cyan"]
        elif th_diff <= -3:
            warp_tag = f"CONSERVATIVE CHOKE ({th_diff} vs 20)"
            warp_col = self.PALETTE["cyber_mint"]
        else:
            warp_tag = "BALANCED PACE (~20)"
            warp_col = self.PALETTE["gold"]

        gto_line = f"GTO Hold Target k*(i,j) = {gto.hold_threshold} pts  |  {warp_tag}"
        self.screen.blit(self.f_bold.render(gto_line, True, warp_col), (info_x, stage_rect.y + 62))

        ev_line = (
            f"Exact MDP Win Prob:  ROLL = {gto.p_win_roll*100:5.1f}%   |   "
            f"HOLD = {gto.p_win_hold*100:5.1f}%   |   Optimal: {gto.optimal_action}"
        )
        self.screen.blit(self.f_small.render(ev_line, True, self.PALETTE["text_white"]), (info_x, stage_rect.y + 82))

        if game.last_human_mistake and not game.sim_mode:
            m_lbl = self.f_tiny.render(f"Suboptimal Human Heuristic Alert: {game.last_human_mistake}", True, self.PALETTE["rose"])
            self.screen.blit(m_lbl, (info_x, stage_rect.y + 102))
        else:
            r_lbl = self.f_tiny.render(f"Regime: {gto.regime}", True, self.PALETTE["text_muted"])
            self.screen.blit(r_lbl, (info_x, stage_rect.y + 102))

        # Interactive Roll & Hold Buttons (Bottom of Stage)
        human_turn = (not game.sim_mode) and (game.turn == 0) and (game.winner is None)
        btn_y = stage_rect.bottom - 48
        roll_rect = pygame.Rect(stage_rect.x + 16, btn_y, 275, 38)
        hold_rect = pygame.Rect(stage_rect.x + 305, btn_y, 280, 38)

        self._draw_button(
            "roll",
            roll_rect,
            "🎲 ROLL DIE (SPACE / R)",
            (22, 95, 78),
            self.PALETTE["cyber_mint"],
            enabled=human_turn,
            font=self.f_header,
        )
        hold_label = f"🏦 HOLD & BANK +{game.turn_bank} PTS (H / ENTER)"
        self._draw_button(
            "hold",
            hold_rect,
            hold_label,
            (110, 78, 18),
            self.PALETTE["gold"],
            enabled=(human_turn and game.turn_bank > 0),
            font=self.f_header,
        )

        # Banner overlay if active
        if game.banner_timer > 0 and game.banner_text:
            b_surf = self.f_bold.render(game.banner_text, True, game.banner_color)
            self.screen.blit(b_surf, (stage_rect.x + 16, stage_rect.y + 114))

        cy += stage_h + 10

        # Dynamic GTO Stopping Boundary Chart k*(i, j) vs Opponent Score j
        chart_h = 220
        chart_rect = pygame.Rect(px + 12, cy, pw - 24, chart_h)
        self._draw_gto_boundary_chart(chart_rect, game)
        cy += chart_h + 10

        # Recent Turn Action Ticker
        feed_rect = pygame.Rect(px + 12, cy, pw - 24, py + ph - cy - 10)
        self._draw_action_feed(feed_rect, game)

    def _draw_gto_boundary_chart(self, rect: pygame.Rect, game: PigGame) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        active_p = game.turn
        i_score = game.scores[active_p]
        j_score = game.scores[1 - active_p]
        p_tag = "Gevva0" if active_p == 1 else "Player 1"

        hdr = self.f_bold.render(
            f"DYNAMIC GTO STOPPING BOUNDARY k*(i={i_score}, j) vs HUMAN 'HOLD AT 20' ({p_tag})",
            True,
            self.PALETTE["cyber_cyan"],
        )
        self.screen.blit(hdr, (rect.x + 14, rect.y + 8))

        # Plot area
        gx, gy = rect.x + 46, rect.y + 32
        gw, gh = rect.width - 66, rect.height - 54
        plot_rect = pygame.Rect(gx, gy, gw, gh)
        pygame.draw.rect(self.screen, (14, 16, 25), plot_rect)
        pygame.draw.rect(self.screen, (42, 48, 70), plot_rect, 1)

        max_k_axis = 55.0

        def to_xy(j_val: float, k_val: float) -> tuple[int, int]:
            x = gx + int((j_val / 99.0) * gw)
            y = gy + gh - int((min(max_k_axis, max(0.0, k_val)) / max_k_axis) * gh)
            return x, y

        # Grid lines
        for k_tick in (10, 20, 30, 40, 50):
            _, y_t = to_xy(0, k_tick)
            col = (60, 40, 55) if k_tick == 20 else (26, 30, 44)
            pygame.draw.line(self.screen, col, (gx, y_t), (gx + gw, y_t), 1)
            lbl = self.f_tiny.render(f"{k_tick}", True, self.PALETTE["rose"] if k_tick == 20 else self.PALETTE["text_muted"])
            self.screen.blit(lbl, (gx - 22, y_t - 6))

        # Human "Hold at 20" static line
        _, y20 = to_xy(0, 20)
        for dash_x in range(gx, gx + gw, 12):
            pygame.draw.line(self.screen, self.PALETTE["rose"], (dash_x, y20), (min(gx + gw, dash_x + 6), y20), 2)
        h20_lbl = self.f_tiny.render("Human Gut Heuristic (Hold=20)", True, self.PALETTE["rose"])
        self.screen.blit(h20_lbl, (gx + 8, y20 - 14))

        # Warped GTO Hold Curve k*(i, j) for j = 0..99
        curve = game.solver.get_hold_curve_for_score(i_score)
        pts = [to_xy(j_idx, float(curve[j_idx])) for j_idx in range(100)]

        # Shaded polygon under curve (ROLL zone)
        poly_pts = [(gx, gy + gh)] + pts + [(gx + gw, gy + gh)]
        surf = pygame.Surface(( rect.width, rect.height ), pygame.SRCALPHA)
        shifted_poly = [(x - rect.x, y - rect.y) for x, y in poly_pts]
        pygame.draw.polygon(surf, (0, 210, 255, 28), shifted_poly)
        self.screen.blit(surf, rect.topleft)

        # Draw curve line
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, self.PALETTE["cyber_mint"], False, pts, 3)

        # Zone labels
        self.screen.blit(
            self.f_tiny.render("▲ HOLD ZONE (Bank Points)", True, self.PALETTE["gold"]),
            (gx + gw - 165, gy + 6),
        )
        self.screen.blit(
            self.f_tiny.render("▼ ROLL ZONE (Press EV)", True, self.PALETTE["cyber_mint"]),
            (gx + gw - 165, gy + gh - 16),
        )

        # Current Opponent Score vertical cursor & Active Turn Bank dot
        cur_x, cur_y = to_xy(float(j_score), float(game.turn_bank))
        _, th_y = to_xy(float(j_score), float(curve[min(99, j_score)]))
        pygame.draw.line(self.screen, (80, 95, 130), (cur_x, gy), (cur_x, gy + gh), 1)
        pygame.draw.circle(self.screen, self.PALETTE["cyber_cyan"], (cur_x, th_y), 5, 2)

        pulse_r = 6 + int((math.sin(self.time_sec * 9.0) + 1) * 2)
        pygame.draw.circle(self.screen, self.PALETTE["gold"], (cur_x, cur_y), pulse_r)
        pygame.draw.circle(self.screen, (20, 20, 30), (cur_x, cur_y), 3)

        # X-axis labels
        axis_lbl = self.f_tiny.render(
            f"Opponent Banked Score j (0 -> 99)   |   Current State: (i={i_score}, j={j_score}, k={game.turn_bank})",
            True,
            self.PALETTE["text_dim"],
        )
        self.screen.blit(axis_lbl, (gx + 10, gy + gh + 4))

    def _draw_action_feed(self, rect: pygame.Rect, game: PigGame) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        self.screen.blit(
            self.f_bold.render("LIVE TURN LOG & GTO TELEMETRY:", True, self.PALETTE["text_white"]),
            (rect.x + 12, rect.y + 6),
        )
        y = rect.y + 24
        recent = list(game.events)[-4:]
        if not recent:
            self.screen.blit(
                self.f_small.render("Press SPACE to roll the die or M to launch High-Speed Simulation!", True, self.PALETTE["text_dim"]),
                (rect.x + 12, y + 4),
            )
            return

        for ev in reversed(recent):
            if y + 15 > rect.bottom - 4:
                break
            who = "GEVVA-T0" if ev.player == 1 else "PLAYER 1"
            col = self.PALETTE["cyber_cyan"] if ev.player == 1 else self.PALETTE["pig_pink"]
            if ev.action == "PIG_OUT":
                col = self.PALETTE["rose"]
            elif ev.action in ("HOLD", "WIN"):
                col = self.PALETTE["cyber_mint"]
            lat_tag = f" [{ev.latency_ms:.1f}ms]" if ev.latency_ms > 0 else ""
            line = f"[{who}] {ev.note} (GTO: {ev.gto_optimal}, k*={ev.gto_threshold}){lat_tag}"
            self.screen.blit(self.f_tiny.render(line[:86], True, col), (rect.x + 12, y))
            y += 16

    # ------------------------------------------------------------------
    # RIGHT VIEWPORT: TIER 0 ENGINE ARCHITECTURE & TELEMETRY
    # ------------------------------------------------------------------
    def _draw_tier0_panel(self, game: PigGame, brain: BasePigBrain) -> None:
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

        cy = py + 48
        # Model & Latency Strip
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
        cy += 34

        res = brain.last_result
        sit = brain.last_situation

        # 1. Inbound Query Vector q Box
        q_box = pygame.Rect(px + 12, cy, pw - 24, 58)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], q_box, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_blue"], q_box, 1, border_radius=8)
        self.screen.blit(
            self.f_bold.render("INBOUND QUERY VECTOR q in R^768 (Gemma Embedding):", True, self.PALETTE["cyber_cyan"]),
            (q_box.x + 10, q_box.y + 6),
        )
        q_str = sit.query_text if sit else "Waiting for first Gevva0 turn decision..."
        self.screen.blit(self.f_tiny.render(q_str[:84], True, self.PALETTE["text_white"]), (q_box.x + 10, q_box.y + 24))
        if len(q_str) > 84:
            self.screen.blit(self.f_tiny.render(q_str[84:168], True, self.PALETTE["text_dim"]), (q_box.x + 10, q_box.y + 39))
        cy += 66

        # Connector arrows splitting to (1) Direct Cosine and (2)/(3) AMC Graph
        mid_x = px + pw // 2
        left_cx = px + pw // 4
        right_cx = px + (pw * 3) // 4
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (mid_x, cy - 8), (mid_x, cy), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (right_cx, cy), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (left_cx, cy + 8), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (right_cx, cy), (right_cx, cy + 8), 2)
        cy += 8

        # 2. Split Branches: (1) Direct Cosine vs (2)/(3) AMC Graph Construction
        branch_w = (pw - 32) // 2
        branch_h = 132
        cos_box = pygame.Rect(px + 12, cy, branch_w, branch_h)
        amc_box = pygame.Rect(px + 20 + branch_w, cy, branch_w, branch_h)

        # Branch Left: (1) Direct Cosine
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
            (cos_box.x + 10, cos_box.y + 8),
        )
        self.screen.blit(
            self.f_tiny.render("Pairwise sim(q, c_i) / tau_temp (0.05)", True, self.PALETTE["text_dim"]),
            (cos_box.x + 10, cos_box.y + 25),
        )

        c_probs = (res.cosine_probs if (res and res.cosine_probs) else {"A": 0.5, "B": 0.5})
        c_sims = (res.cosine_sims if (res and res.cosine_sims) else {"A": 0.0, "B": 0.0})
        self._draw_mini_prob_bar(
            cos_box.x + 10,
            cos_box.y + 48,
            cos_box.width - 20,
            f"[A] ROLL (sim={c_sims.get('A', 0.0):.3f})",
            c_probs.get("A", 0.5),
            self.PALETTE["cyber_mint"],
        )
        self._draw_mini_prob_bar(
            cos_box.x + 10,
            cos_box.y + 86,
            cos_box.width - 20,
            f"[B] HOLD (sim={c_sims.get('B', 0.0):.3f})",
            c_probs.get("B", 0.5),
            self.PALETTE["gold"],
        )

        # Branch Right: (2)/(3) AMC Graph Construction B = (I - Q)^(-1) R
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
            (amc_box.x + 10, amc_box.y + 8),
        )
        m_cnt = res.landmarks_count if res else 104
        scaf_lbl = (
            f"(2) Default 39-Landmarks (M={m_cnt})"
            if brain.engine_mode == "default_39"
            else f"(3) Pig Domain Manifold (M={m_cnt})"
        )
        if not m_active:
            scaf_lbl = "Bypassed (Pure Cosine Mode)"
        self.screen.blit(self.f_tiny.render(scaf_lbl, True, self.PALETTE["text_dim"]), (amc_box.x + 10, amc_box.y + 25))

        m_probs = (res.markov_probs if (res and res.markov_probs) else {"A": 0.5, "B": 0.5})
        self._draw_mini_prob_bar(
            amc_box.x + 10,
            amc_box.y + 48,
            amc_box.width - 20,
            "[A] Sink ROLL Absorption",
            m_probs.get("A", 0.5),
            self.PALETTE["purple"],
        )
        self._draw_mini_prob_bar(
            amc_box.x + 10,
            amc_box.y + 86,
            amc_box.width - 20,
            "[B] Sink HOLD Absorption",
            m_probs.get("B", 0.5),
            self.PALETTE["amber"],
        )
        cy += branch_h

        # Connector lines merging into (4) Production Hybrid Ensemble
        pygame.draw.line(self.screen, self.PALETTE["cyber_cyan"], (left_cx, cy), (left_cx, cy + 8), 2)
        pygame.draw.line(self.screen, self.PALETTE["purple"], (right_cx, cy), (right_cx, cy + 8), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_mint"], (left_cx, cy + 8), (right_cx, cy + 8), 2)
        pygame.draw.line(self.screen, self.PALETTE["cyber_mint"], (mid_x, cy + 8), (mid_x, cy + 16), 2)
        cy += 16

        # 3. (4) Production Hybrid Ensemble Box
        hyb_h = 114
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
            (hyb_box.x + 12, hyb_box.y + 8),
        )

        f_probs = res.probs if res else {"A": 0.5, "B": 0.5}
        win_lbl = res.label if res else "A"
        self._draw_large_decision_bar(
            hyb_box.x + 12,
            hyb_box.y + 32,
            hyb_box.width - 24,
            "[A] ROLL DIE AGAIN",
            f_probs.get("A", 0.5),
            selected=(win_lbl == "A"),
            color=self.PALETTE["cyber_mint"],
        )
        self._draw_large_decision_bar(
            hyb_box.x + 12,
            hyb_box.y + 70,
            hyb_box.width - 24,
            "[B] HOLD & BANK POINTS",
            f_probs.get("B", 0.5),
            selected=(win_lbl == "B"),
            color=self.PALETTE["gold"],
        )
        cy += hyb_h + 10

        # 4. Tripwire Checks Box (Shannon Entropy H <= 0.65, Logit Margin Delta >= 0.25)
        trip_h = 84
        trip_box = pygame.Rect(px + 12, cy, pw - 24, trip_h)
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], trip_box, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], trip_box, 1, border_radius=8)

        ent_val = res.entropy if res else 0.12
        mar_val = res.margin if res else 0.76
        ent_ok = ent_val <= 0.65
        mar_ok = mar_val >= 0.25
        emb_ms = res.embed_latency_ms if res else 1.75
        grp_ms = res.graph_latency_ms if res else 0.12

        self.screen.blit(
            self.f_bold.render("TRIPWIRE DIAGNOSTICS & ZERO-CASCADE EARLY EXIT:", True, self.PALETTE["text_white"]),
            (trip_box.x + 12, trip_box.y + 8),
        )
        ent_col = self.PALETTE["cyber_mint"] if ent_ok else self.PALETTE["amber"]
        mar_col = self.PALETTE["cyber_mint"] if mar_ok else self.PALETTE["amber"]
        self.screen.blit(
            self.f_small.render(f"• Shannon Entropy H = {ent_val:.3f} <= τ₁ (0.65) [{'PASS' if ent_ok else 'NEAR BOUNDARY'}]", True, ent_col),
            (trip_box.x + 12, trip_box.y + 30),
        )
        self.screen.blit(
            self.f_small.render(f"• Logit Margin    Δ = {mar_val:.3f} >= τ₂ (0.25) [{'PASS' if mar_ok else 'TIGHT MARGIN'}]", True, mar_col),
            (trip_box.x + 12, trip_box.y + 48),
        )
        lat_detail = f"Embed: {emb_ms:.2f}ms | AMC Lookup: {grp_ms:.3f}ms | GTO Match: {brain.gto_match_rate:.1f}%"
        self.screen.blit(self.f_tiny.render(lat_detail, True, self.PALETTE["text_dim"]), (trip_box.x + 12, trip_box.y + 66))

        # [ EXIT TIER 0 ] Badge on right of Tripwire box
        badge_rect = pygame.Rect(trip_box.right - 158, trip_box.y + 16, 144, 50)
        pygame.draw.rect(self.screen, (18, 56, 44), badge_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.PALETTE["cyber_mint"], badge_rect, 2, border_radius=8)
        b1 = self.f_bold.render("[ EXIT TIER 0 ]", True, self.PALETTE["cyber_mint"])
        b2 = self.f_tiny.render("No Tier 1/2 Cascade", True, self.PALETTE["text_white"])
        self.screen.blit(b1, b1.get_rect(center=(badge_rect.centerx, badge_rect.centery - 8)))
        self.screen.blit(b2, b2.get_rect(center=(badge_rect.centerx, badge_rect.centery + 10)))

        cy += trip_h + 10

        # 5. Simulation Arena Scoreboard & Why GTO Beats "Hold at 20"
        sim_rect = pygame.Rect(px + 12, cy, pw - 24, py + ph - cy - 12)
        self._draw_sim_scoreboard(sim_rect, game, brain)

    def _draw_mini_prob_bar(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        prob: float,
        color: tuple[int, int, int],
    ) -> None:
        self.screen.blit(self.f_tiny.render(label, True, self.PALETTE["text_white"]), (x, y))
        pct_surf = self.f_bold.render(f"{prob*100:5.1f}%", True, color)
        self.screen.blit(pct_surf, (x + w - pct_surf.get_width(), y - 1))
        bar_y = y + 15
        pygame.draw.rect(self.screen, (15, 17, 26), (x, bar_y, w, 10), border_radius=4)
        fill_w = int(w * max(0.0, min(1.0, prob)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (x, bar_y, fill_w, 10), border_radius=4)

    def _draw_large_decision_bar(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        prob: float,
        selected: bool,
        color: tuple[int, int, int],
    ) -> None:
        row_rect = pygame.Rect(x, y, w, 32)
        pygame.draw.rect(self.screen, (20, 26, 38) if selected else (16, 18, 28), row_rect, border_radius=6)
        pygame.draw.rect(
            self.screen,
            color if selected else self.PALETTE["panel_border"],
            row_rect,
            2 if selected else 1,
            border_radius=6,
        )
        prefix = "▶ " if selected else "  "
        lbl_surf = self.f_bold.render(f"{prefix}{label}", True, self.PALETTE["text_white"] if selected else self.PALETTE["text_dim"])
        self.screen.blit(lbl_surf, (x + 8, y + 8))

        bar_x = x + 230
        bar_w = w - 310
        pygame.draw.rect(self.screen, (12, 14, 22), (bar_x, y + 9, bar_w, 14), border_radius=5)
        fill_w = int(bar_w * max(0.0, min(1.0, prob)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, color, (bar_x, y + 9, fill_w, 14), border_radius=5)

        pct_surf = self.f_header.render(f"{prob*100:5.1f}%", True, color if selected else self.PALETTE["text_dim"])
        self.screen.blit(pct_surf, (bar_x + bar_w + 10, y + 7))

    def _draw_sim_scoreboard(self, rect: pygame.Rect, game: PigGame, brain: BasePigBrain) -> None:
        pygame.draw.rect(self.screen, self.PALETTE["card_bg"], rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], rect, 1, border_radius=10)

        self.screen.blit(
            self.f_bold.render("HIGH-SPEED SIMULATION SCOREBOARD & GTO WARP ADVANTAGE:", True, self.PALETTE["gold"]),
            (rect.x + 12, rect.y + 8),
        )
        self._draw_button(
            "reset_stats",
            pygame.Rect(rect.right - 102, rect.y + 6, 90, 20),
            "CLEAR STATS",
            (40, 28, 38),
            self.PALETTE["panel_border"],
            font=self.f_tiny,
        )

        st = game.stats
        y = rect.y + 32
        line1 = (
            f"GAMES: {st.games_played:4d}  |  GEVVA-T0 WINS: {st.gevva_wins:4d} ({st.gevva_win_rate:5.1f}%)  |  "
            f"OPPONENT WINS: {st.challenger_wins:4d} ({st.challenger_win_rate:5.1f}%)"
        )
        self.screen.blit(self.f_bold.render(line1, True, self.PALETTE["text_white"]), (rect.x + 12, y))
        y += 22

        # Win-rate tug-of-war bar
        bar_w = rect.width - 24
        pygame.draw.rect(self.screen, (35, 25, 40), (rect.x + 12, y, bar_w, 14), border_radius=6)
        g_ratio = (st.gevva_wins / st.games_played) if st.games_played > 0 else 0.5
        g_fill = int(bar_w * g_ratio)
        if g_fill > 0:
            pygame.draw.rect(self.screen, self.PALETTE["cyber_mint"], (rect.x + 12, y, g_fill, 14), border_radius=6)
        mid_bar = rect.x + 12 + bar_w // 2
        pygame.draw.line(self.screen, self.PALETTE["text_white"], (mid_bar, y - 2), (mid_bar, y + 16), 2)
        y += 22

        line2 = (
            f"Aggressive Catch-Up Rolls (k>=20): {st.gevva_aggressive_warps:3d}  |  "
            f"Conservative Lead Chokes (k<20): {st.gevva_conservative_chokes:3d}  |  "
            f"Comeback Wins: {st.gevva_comeback_wins:3d}"
        )
        self.screen.blit(self.f_tiny.render(line2, True, self.PALETTE["cyber_cyan"]), (rect.x + 12, y))
        y += 18

        hotkeys = "KEYS: [SPACE] Roll  [H] Hold  [M] Play/Sim  [S] Speed  [T] +50 Sim Games  [1-4] Tier0 Mode  [O] Bot"
        self.screen.blit(self.f_tiny.render(hotkeys, True, self.PALETTE["text_dim"]), (rect.x + 12, rect.bottom - 18))

    def _update_particles(self, dt: float) -> None:
        for p in self.particles[:]:
            p.life += dt
            if p.life >= p.max_life:
                self.particles.remove(p)
                continue
            p.vy += 320.0 * dt  # gravity
            p.x += p.vx * dt
            p.y += p.vy * dt
            r = max(2, int(p.size * (1.0 - 0.5 * (p.life / p.max_life))))
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), r)
            if p.kind == "coin":
                pygame.draw.circle(self.screen, self.PALETTE["gold_dark"], (int(p.x), int(p.y)), r, 1)
