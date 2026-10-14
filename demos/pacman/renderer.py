"""
Gevva0 Arcade Pac-Man: Retro Cyberpunk Visual Renderer & Mission Control HUD.
Renders the arcade canvas alongside Gevva0 Neural Telemetry Mission Control panel:
- Left Viewport: Full 28x31 classic arcade board with neon aesthetic, threat field overlay,
  ghost targeting & eyes returning, fruit, energizer pulsing, particle sparkles, and smooth Pac-Man mouth animation.
- Right Viewport: Live Gevva0 telemetry with Platt temperature T, decision candidates cards,
  probability bars, latency readout (sub-35ms), CoT reasoning traces, and game statistics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import pygame

from game import Game, Actor, PacMan, Ghost, tile_of
from maze import COLS, ROWS, LAYOUT, exits, DIRS, UP, LEFT, DOWN, RIGHT, DIR_NAMES, ARROWS
from tactics import Situation, RouteOption


@dataclass
class Sparkle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: float


class PacmanRenderer:
    TILE_SIZE = 22
    MAZE_ORIGIN_X = 22
    MAZE_ORIGIN_Y = 50

    WIN_WIDTH = 1200
    WIN_HEIGHT = 760

    PALETTE = {
        "bg": (10, 10, 16),
        "panel_bg": (16, 17, 26),
        "panel_border": (40, 45, 65),
        "panel_header": (26, 30, 46),
        "neon_blue": (35, 125, 255),
        "neon_blue_glow": (0, 60, 180),
        "neon_cyan": (0, 240, 255),
        "neon_pink": (255, 60, 160),
        "neon_yellow": (255, 225, 35),
        "neon_red": (255, 45, 55),
        "neon_green": (40, 240, 120),
        "neon_orange": (255, 140, 25),
        "text_white": (245, 245, 250),
        "text_dim": (135, 140, 160),
        "text_gold": (255, 215, 0),
        "dot": (255, 220, 160),
        "gate": (255, 175, 215),
    }

    GHOST_COLORS = {
        "Blinky": (255, 45, 55),
        "Pinky": (255, 140, 200),
        "Inky": (40, 220, 255),
        "Clyde": (255, 160, 30),
    }

    def __init__(self, fullscreen: bool = False):
        pygame.init()
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT), flags)
        pygame.display.set_caption("ARCADE PAC-MAN // GEVVA0 CALIBRATED DECISION GATEWAY (GEMMA 4 26B)")
        self.clock = pygame.time.Clock()
        self.particles: list[Sparkle] = []

        self._init_fonts()
        self.flash_alpha = 0.0
        self.flash_color = (255, 60, 60)
        self.pulse_time = 0.0

    def _init_fonts(self):
        font_name = "Consolas"
        self.f_huge = pygame.font.SysFont(font_name, 34, bold=True)
        self.f_title = pygame.font.SysFont(font_name, 18, bold=True)
        self.f_header = pygame.font.SysFont(font_name, 14, bold=True)
        self.f_bold = pygame.font.SysFont(font_name, 12, bold=True)
        self.f_regular = pygame.font.SysFont(font_name, 12)
        self.f_small = pygame.font.SysFont(font_name, 11)
        self.f_tiny = pygame.font.SysFont(font_name, 10)

    def trigger_flash(self, color=(255, 60, 60), max_alpha: float = 30.0):
        self.flash_alpha = min(max_alpha, 40.0)
        self.flash_color = color

    def emit_sparkles(self, x: float, y: float, color=(255, 230, 40), count=6):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(20, 80)
            self.particles.append(
                Sparkle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=0.0,
                    max_life=random.uniform(0.25, 0.5),
                    color=color,
                    size=random.uniform(2, 4),
                )
            )

    def render_loading(self, message: str, detail: str = ""):
        self.screen.fill(self.PALETTE["bg"])
        cx = self.WIN_WIDTH // 2
        cy = self.WIN_HEIGHT // 2

        # Glowing panel box
        box_w, box_h = 600, 200
        box_rect = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], box_rect)
        pygame.draw.rect(self.screen, self.PALETTE["neon_cyan"], box_rect, 2)

        # Pulse highlight line
        pulse = (math.sin(pygame.time.get_ticks() * 0.006) + 1.0) * 0.5
        line_w = int(box_w * 0.8)
        line_rect = pygame.Rect(cx - line_w // 2, cy - box_h // 2 + 10, line_w, 2)
        pygame.draw.rect(self.screen, (int(30 + pulse * 100), int(150 + pulse * 100), 255), line_rect)

        # Title
        title = self.f_huge.render("GEVVA0 DECISION ENGINE", True, self.PALETTE["neon_yellow"])
        self.screen.blit(title, title.get_rect(center=(cx, cy - 45)))

        # Loading Message
        msg_surf = self.f_title.render(message, True, self.PALETTE["text_white"])
        self.screen.blit(msg_surf, msg_surf.get_rect(center=(cx, cy + 5)))

        # Detail line
        if detail:
            det_surf = self.f_regular.render(detail, True, self.PALETTE["neon_cyan"])
            self.screen.blit(det_surf, det_surf.get_rect(center=(cx, cy + 35)))

        # Spinner animation dots
        dots = "." * (int(pygame.time.get_ticks() / 350) % 4)
        dot_surf = self.f_title.render(f"Please wait{dots:<3}", True, self.PALETTE["text_dim"])
        self.screen.blit(dot_surf, dot_surf.get_rect(center=(cx, cy + 65)))

        pygame.display.flip()

    def render(self, game: Game, situation: Optional[Situation], pilot, brain, dt: float):
        self.pulse_time += dt
        self.screen.fill(self.PALETTE["bg"])

        # 1. Left Arcade Viewport
        self._draw_arcade_header(game)
        self._draw_maze(game, situation)
        self._draw_pellets(game)
        self._draw_ghosts(game)
        self._draw_pacman(game)
        self._draw_arcade_footer(game)

        # Overlays
        if game.game_over:
            self._draw_banner("GAME OVER", self.PALETTE["neon_red"])
        elif game.level_clear_timer > 0:
            self._draw_banner(f"LEVEL {game.level} CLEARED!", self.PALETTE["neon_green"])

        # 2. Right Telemetry Mission Control Panel
        self._draw_telemetry_panel(game, situation, pilot, brain)

        # 3. Particle System
        self._update_particles(dt)

        # 4. Subtle flash alert
        if self.flash_alpha > 0.5:
            flash_surf = pygame.Surface((self.WIN_WIDTH, self.WIN_HEIGHT), pygame.SRCALPHA)
            flash_surf.fill((*self.flash_color, int(self.flash_alpha)))
            self.screen.blit(flash_surf, (0, 0))
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 60.0)

        pygame.display.flip()

    def _draw_arcade_header(self, game: Game):
        # 1UP and HIGH SCORE headers
        lbl_1up = self.f_header.render("1UP", True, self.PALETTE["neon_red"])
        lbl_hi = self.f_header.render("HIGH SCORE", True, self.PALETTE["neon_cyan"])
        val_score = self.f_huge.render(f"{game.score:06d}", True, self.PALETTE["text_white"])
        val_hi = self.f_huge.render("033330", True, self.PALETTE["text_gold"])

        self.screen.blit(lbl_1up, (self.MAZE_ORIGIN_X + 20, 10))
        self.screen.blit(val_score, (self.MAZE_ORIGIN_X + 20, 26))

        self.screen.blit(lbl_hi, (self.MAZE_ORIGIN_X + 380, 10))
        self.screen.blit(val_hi, (self.MAZE_ORIGIN_X + 380, 26))

    def _draw_maze(self, game: Game, situation: Optional[Situation]):
        ox, oy = self.MAZE_ORIGIN_X, self.MAZE_ORIGIN_Y
        ts = self.TILE_SIZE

        # Outer border glow
        border_rect = pygame.Rect(ox - 3, oy - 3, COLS * ts + 6, ROWS * ts + 6)
        pygame.draw.rect(self.screen, (20, 25, 40), border_rect, 1)

        # Draw walls & doors
        for y, row in enumerate(LAYOUT):
            for x, ch in enumerate(row):
                px = ox + x * ts
                py = oy + y * ts
                rect = pygame.Rect(px, py, ts, ts)

                if ch == "#":
                    # Neon wall block with rounded inner bevel
                    pygame.draw.rect(self.screen, (15, 20, 36), rect)
                    pygame.draw.rect(self.screen, self.PALETTE["neon_blue"], rect, 1)
                elif ch == "-":
                    # Ghost door
                    gate_y = py + ts // 2
                    pygame.draw.line(self.screen, self.PALETTE["gate"], (px, gate_y), (px + ts, gate_y), 3)

        # Danger heat-map tiles overlay
        # if situation and situation.danger_tiles:
        #     for dx, dy in situation.danger_tiles:
        #         d_rect = pygame.Rect(ox + dx * ts + 2, oy + dy * ts + 2, ts - 4, ts - 4)
        #         d_surf = pygame.Surface((ts - 4, ts - 4), pygame.SRCALPHA)
        #         d_surf.fill((255, 30, 40, 35))
        #         self.screen.blit(d_surf, d_rect.topleft)

        # Route breadcrumb preview (committed path)
        if situation:
            latched_opt = getattr(situation, "latched_opt", None)
            if latched_opt:
                for tx, ty in latched_opt.path:
                    cx = ox + tx * ts + ts // 2
                    cy = oy + ty * ts + ts // 2
                    pygame.draw.circle(self.screen, (0, 240, 255, 120), (cx, cy), 3)

    def _draw_pellets(self, game: Game):
        ox, oy = self.MAZE_ORIGIN_X, self.MAZE_ORIGIN_Y
        ts = self.TILE_SIZE

        # Dots
        for x, y in game.dots:
            cx = ox + x * ts + ts // 2
            cy = oy + y * ts + ts // 2
            pygame.draw.circle(self.screen, self.PALETTE["dot"], (cx, cy), 2)

        # Energizers (pulsing)
        pulse = (math.sin(self.pulse_time * 8.0) + 1.0) * 0.5
        e_radius = int(5.5 + pulse * 2.5)
        for x, y in game.energizers:
            cx = ox + x * ts + ts // 2
            cy = oy + y * ts + ts // 2
            pygame.draw.circle(self.screen, self.PALETTE["neon_yellow"], (cx, cy), e_radius)
            pygame.draw.circle(self.screen, (255, 255, 200), (cx, cy), max(2, e_radius - 3))

        # Fruit bonus
        if game.fruit_active:
            fx = ox + 13.5 * ts
            fy = oy + 17.0 * ts
            pygame.draw.circle(self.screen, self.PALETTE["neon_red"], (int(fx), int(fy)), 7)
            pygame.draw.circle(self.screen, self.PALETTE["neon_green"], (int(fx) + 3, int(fy) - 6), 3)

    def _draw_ghosts(self, game: Game):
        ox, oy = self.MAZE_ORIGIN_X, self.MAZE_ORIGIN_Y
        ts = self.TILE_SIZE

        for g in game.ghosts:
            gx = ox + g.x * ts + ts // 2
            gy = oy + g.y * ts + ts // 2
            rad = ts // 2 - 1

            if g.state in ("eyes", "entering"):
                # Eyeballs only
                self._draw_eyes(gx, gy, g.dir)
                continue

            # Determine body color
            if g.frightened:
                if game.fright_flashing() and int(self.pulse_time * 8.0) % 2 == 0:
                    color = (240, 240, 255)
                else:
                    color = (30, 80, 255)
            else:
                color = self.GHOST_COLORS.get(g.name, (200, 200, 200))

            # Ghost dome
            rect = pygame.Rect(gx - rad, gy - rad, rad * 2, rad * 2)
            pygame.draw.circle(self.screen, color, (int(gx), int(gy)), rad)
            pygame.draw.rect(self.screen, color, (gx - rad, gy, rad * 2, rad))

            # Skirt waves
            wave_w = rad * 2 // 3
            for i in range(3):
                wx = gx - rad + i * wave_w + wave_w // 2
                wy = gy + rad
                pygame.draw.circle(self.screen, color, (int(wx), int(wy)), wave_w // 2)

            # Eyes
            if not g.frightened:
                self._draw_eyes(gx, gy, g.dir)
            else:
                # Scared blue face
                eye_c = (255, 255, 255) if color != (240, 240, 255) else (255, 40, 40)
                pygame.draw.circle(self.screen, eye_c, (int(gx - 4), int(gy - 2)), 2)
                pygame.draw.circle(self.screen, eye_c, (int(gx + 4), int(gy - 2)), 2)

    def _draw_eyes(self, gx: float, gy: float, direction: tuple[int, int]):
        dx, dy = direction
        pupil_ox = dx * 2
        pupil_oy = dy * 2

        # Left eye
        pygame.draw.circle(self.screen, (255, 255, 255), (int(gx - 4), int(gy - 3)), 4)
        pygame.draw.circle(self.screen, (20, 40, 180), (int(gx - 4 + pupil_ox), int(gy - 3 + pupil_oy)), 2)

        # Right eye
        pygame.draw.circle(self.screen, (255, 255, 255), (int(gx + 4), int(gy - 3)), 4)
        pygame.draw.circle(self.screen, (20, 40, 180), (int(gx + 4 + pupil_ox), int(gy - 3 + pupil_oy)), 2)

    def _draw_pacman(self, game: Game):
        ox, oy = self.MAZE_ORIGIN_X, self.MAZE_ORIGIN_Y
        ts = self.TILE_SIZE
        pac = game.pac

        px = ox + pac.x * ts + ts // 2
        py = oy + pac.y * ts + ts // 2
        rad = ts // 2

        # Mouth angle oscillation
        mouth_angle = abs(math.sin(pac.anim * 14.0)) * 0.7 + 0.05
        heading_angle = math.atan2(pac.dir[1], pac.dir[0])

        start_angle = heading_angle + mouth_angle
        end_angle = heading_angle + math.tau - mouth_angle

        # Draw filled pie slice
        points = [(int(px), int(py))]
        steps = 16
        for i in range(steps + 1):
            a = start_angle + (end_angle - start_angle) * (i / steps)
            points.append((int(px + rad * math.cos(a)), int(py + rad * math.sin(a))))

        pygame.draw.polygon(self.screen, self.PALETTE["neon_yellow"], points)
        pygame.draw.circle(self.screen, (255, 255, 220), (int(px), int(py)), 2)

    def _draw_arcade_footer(self, game: Game):
        ox, oy = self.MAZE_ORIGIN_X, self.MAZE_ORIGIN_Y + ROWS * self.TILE_SIZE + 10

        # Lives
        lives_lbl = self.f_regular.render("LIVES:", True, self.PALETTE["text_dim"])
        self.screen.blit(lives_lbl, (ox, oy))
        for i in range(max(0, game.lives)):
            lx = ox + 55 + i * 18
            ly = oy + 6
            pygame.draw.circle(self.screen, self.PALETTE["neon_yellow"], (lx, ly), 6)

        # Pellets & Ghost counts
        eaten_lbl = self.f_regular.render(
            f"DOTS: {game.dots_left:03d}/244   DEVOURED GHOSTS: {game.ghosts_eaten_total}   LEVEL: {game.level}",
            True,
            self.PALETTE["text_dim"],
        )
        self.screen.blit(eaten_lbl, (ox + 160, oy))

    def _draw_banner(self, text: str, color: tuple[int, int, int]):
        bx = self.MAZE_ORIGIN_X + (COLS * self.TILE_SIZE) // 2
        by = self.MAZE_ORIGIN_Y + (ROWS * self.TILE_SIZE) // 2
        w, h = 320, 60
        rect = pygame.Rect(bx - w // 2, by - h // 2, w, h)
        pygame.draw.rect(self.screen, (15, 18, 30), rect)
        pygame.draw.rect(self.screen, color, rect, 2)

        lbl = self.f_huge.render(text, True, color)
        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_telemetry_panel(self, game: Game, situation: Optional[Situation], pilot, brain):
        px = self.MAZE_ORIGIN_X + COLS * self.TILE_SIZE + 24
        py = 20
        pw = self.WIN_WIDTH - px - 20
        ph = self.WIN_HEIGHT - 40

        panel_rect = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(self.screen, self.PALETTE["panel_bg"], panel_rect)
        pygame.draw.rect(self.screen, self.PALETTE["panel_border"], panel_rect, 1)

        # 1. Header & Engine status
        hdr_rect = pygame.Rect(px, py, pw, 36)
        pygame.draw.rect(self.screen, self.PALETTE["panel_header"], hdr_rect)
        title = self.f_title.render("GEVVA0 TELEMETRY // GEMMA 4 26B-A4B", True, self.PALETTE["neon_cyan"])
        self.screen.blit(title, (px + 14, py + 8))

        cy = py + 48

        # 2. Hardware and Model Profile Specs
        T_val = getattr(brain.profile, "temperature", 4.145) if hasattr(brain, "profile") else 4.145
        cot_th = getattr(brain.profile, "cot_threshold", 0.50) if hasattr(brain, "profile") else 0.50
        status_line = (
            f"ENGINE: {brain.name.upper()}  |  PLATT T: {T_val:.3f}  |  CoT GATE: {cot_th:.2f}  |  VRAM: CUDA0"
        )
        self.screen.blit(self.f_bold.render(status_line, True, self.PALETTE["text_gold"]), (px + 14, cy))
        cy += 24

        # 3. Latency & Junction Buffer Stats
        p50, p95 = brain.latency_stats()
        buf_ratio = (
            f"{pilot.stats.prebuffered_hits}/{pilot.stats.junction_arrivals}"
            if pilot.stats.junction_arrivals > 0
            else "N/A"
        )
        perf_line = (
            f"LATENCY: p50={p50:.1f}ms p95={p95:.1f}ms | BUFFERED: {buf_ratio} | DISPATCHES: {pilot.stats.dispatches}"
        )
        self.screen.blit(self.f_regular.render(perf_line, True, self.PALETTE["text_dim"]), (px + 14, cy))
        cy += 28

        # 4. Tactical Threat Status Box
        pygame.draw.line(self.screen, self.PALETTE["panel_border"], (px + 14, cy), (px + pw - 14, cy), 1)
        cy += 10

        threat_text = "NORMAL HARVEST PATROL"
        threat_color = self.PALETTE["neon_green"]
        if situation and situation.threat_alert:
            threat_text = f"THREAT ADVISORY: {situation.hunters_near} HUNTER(S) CLOSING"
            threat_color = self.PALETTE["neon_red"]
        elif game.fright_timer > 0:
            threat_text = f"PREDATOR HUNT ACTIVE ({game.fright_timer:.1f}s REMAINING)"
            threat_color = self.PALETTE["neon_blue"]

        threat_box = pygame.Rect(px + 14, cy, pw - 28, 28)
        pygame.draw.rect(self.screen, (22, 26, 40), threat_box)
        pygame.draw.rect(self.screen, threat_color, threat_box, 1)
        th_lbl = self.f_bold.render(f"HAZARD STATE: {threat_text}", True, threat_color)
        self.screen.blit(th_lbl, (px + 24, cy + 6))
        cy += 38

        # 5. Junction Dispatch Candidates
        section_lbl = self.f_header.render("JUNCTION ROUTE CANDIDATES (LOGIT SCORING):", True, self.PALETTE["text_white"])
        self.screen.blit(section_lbl, (px + 14, cy))
        cy += 22

        latest_dec = pilot.decision
        active_sit = pilot.decided_sit or situation

        if active_sit and active_sit.options:
            probs = latest_dec.probs if latest_dec else {}
            for opt in active_sit.options:
                is_selected = latest_dec and latest_dec.letter == opt.key
                card_rect = pygame.Rect(px + 14, cy, pw - 28, 48)

                # Card styling
                if is_selected:
                    card_bg = (35, 45, 65)
                    card_bdr = self.PALETTE["neon_yellow"] if opt.status != "DEADLY" else self.PALETTE["neon_red"]
                else:
                    card_bg = (20, 22, 32)
                    card_bdr = self.PALETTE["panel_border"]

                pygame.draw.rect(self.screen, card_bg, card_rect)
                pygame.draw.rect(self.screen, card_bdr, card_rect, 1 if not is_selected else 2)

                # Option label & Arrow
                arrow = ARROWS.get(opt.dir, "•")
                opt_str = f"[{opt.key}] {arrow} {opt.name:<7} | {opt.status:<7} | room {opt.room} | outs {opt.outs} | dots {opt.path_dots}"
                col = self.PALETTE["text_white"] if is_selected else self.PALETTE["text_dim"]
                self.screen.blit(self.f_bold.render(opt_str, True, col), (px + 24, cy + 6))

                # Probability bar
                prob = probs.get(opt.key, 0.0)
                bar_x = px + pw - 180
                bar_y = cy + 8
                bar_w = 110
                bar_h = 10
                pygame.draw.rect(self.screen, (30, 35, 50), (bar_x, bar_y, bar_w, bar_h))
                fill_w = int(bar_w * prob)
                fill_col = self.PALETTE["neon_cyan"] if is_selected else self.PALETTE["neon_blue_glow"]
                if fill_w > 0:
                    pygame.draw.rect(self.screen, fill_col, (bar_x, bar_y, fill_w, bar_h))
                pct_str = f"{prob * 100:4.1f}%"
                self.screen.blit(self.f_bold.render(pct_str, True, self.PALETTE["text_white"]), (bar_x + bar_w + 8, bar_y - 2))

                # Subline extras (power, edible ghosts, gap)
                extra = []
                if opt.power is not None:
                    extra.append(f"Power: {opt.power}t")
                if opt.eat:
                    extra.append(f"Catch {opt.eat[0]} (+{opt.eat[2]})")
                if opt.gap < 50:
                    extra.append(f"Gap: {opt.gap:.1f}t ({opt.gap_ghost})")
                subline = " | ".join(extra) if extra else "Route clear"
                self.screen.blit(self.f_tiny.render(subline, True, self.PALETTE["text_dim"]), (px + 24, cy + 28))

                cy += 54
        else:
            self.screen.blit(self.f_regular.render("Cruising straight corridor... awaiting next waypoint.", True, self.PALETTE["text_dim"]), (px + 24, cy + 10))
            cy += 40

        # 6. Chain-of-Thought / Decision Proof Scratchpad
        cy += 10
        pygame.draw.line(self.screen, self.PALETTE["panel_border"], (px + 14, cy), (px + pw - 14, cy), 1)
        cy += 10

        self.screen.blit(self.f_header.render("LAST GEVVA0 VERDICT & BOUNDED COT SCRATCHPAD:", True, self.PALETTE["text_white"]), (px + 14, cy))
        cy += 20

        if latest_dec:
            mode_tag = f"MODE: {latest_dec.mode.upper()}  |  CONFIDENCE: {latest_dec.confidence*100:.1f}%  |  LATENCY: {latest_dec.latency_ms:.1f}ms"
            self.screen.blit(self.f_bold.render(mode_tag, True, self.PALETTE["neon_yellow"]), (px + 14, cy))
            cy += 20

            cot_box = pygame.Rect(px + 14, cy, pw - 28, 90)
            pygame.draw.rect(self.screen, (12, 14, 22), cot_box)
            pygame.draw.rect(self.screen, self.PALETTE["panel_border"], cot_box, 1)

            if latest_dec.cot_text:
                cot_lines = [
                    "CoT Bounded Verification Scratchpad:",
                    latest_dec.cot_text[:85],
                    latest_dec.cot_text[85:170],
                ]
                for i, ln in enumerate(cot_lines):
                    self.screen.blit(self.f_tiny.render(ln, True, self.PALETTE["neon_cyan"]), (px + 20, cy + 6 + i * 16))
            else:
                fast_txt = (
                    f"Fast-Path Confidence Gate Passed ({latest_dec.confidence*100:.1f}% >= {cot_th*100:.0f}%).\n"
                    f"Selected Option [{latest_dec.letter}] via zero-drift direct token logit scoring.\n"
                    f"Effective execution latency: 0 ms at junction waypoint arrival."
                )
                for i, ln in enumerate(fast_txt.splitlines()):
                    self.screen.blit(self.f_small.render(ln, True, self.PALETTE["text_dim"]), (px + 20, cy + 10 + i * 18))
        cy += 105

        # 7. Hotkey instructions footer
        controls = "SPACE: Pause   R: Reset   1-4: Force Dir   ESC: Exit"
        self.screen.blit(self.f_bold.render(controls, True, self.PALETTE["neon_cyan"]), (px + 14, py + ph - 24))

    def _update_particles(self, dt: float):
        for p in self.particles[:]:
            p.life += dt
            if p.life >= p.max_life:
                self.particles.remove(p)
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            alpha = int(255 * (1.0 - p.life / p.max_life))
            color = (*p.color, alpha) if len(p.color) == 3 else p.color
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), max(1, int(p.size)))
