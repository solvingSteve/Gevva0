"""
Arcade-accurate Pac-Man game core (v2) for the Gevva0 showcase.

Pure simulation, no rendering, deterministic given a seed, deep-copyable (used by
the calibration oracle for rollouts). Implements the arcade rules that matter
for play strength:

  * per-level speed tables (Pac-Man, ghosts, frightened, tunnel, Cruise Elroy)
  * scatter/chase wave schedule with forced reversals
  * Blinky / Pinky / Inky / Clyde targeting (incl. the classic UP overflow bug)
  * red-zone "no UP" tiles, ghost house release by dot counters and idle timer
  * dot-eating stalls (1 frame per dot, 3 per energizer), fruit, extra life
  * 200/400/800/1600 ghost chains, frightened flashing, eyes returning home
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from maze import (
    COLS,
    DIRS,
    DOWN,
    FLOOR_SET,
    FRUIT_POS,
    GHOST_DOOR_EXIT,
    LEFT,
    NO_UP_TILES,
    OPPOSITE,
    PAC_START,
    RIGHT,
    TOTAL_DOTS,
    UP,
    exits,
    initial_pellets,
    is_tunnel,
    step,
    toroidal_dist,
)

BASE_SPEED = 75.75757625 / 8.0  # arcade 100% speed in tiles/second (~9.47)
FRAME = 1.0 / 60.0

GHOST_NAMES = ("Blinky", "Pinky", "Inky", "Clyde")
SCATTER_TARGETS = {
    "Blinky": (25, -3),
    "Pinky": (2, -3),
    "Inky": (27, 31),
    "Clyde": (0, 31),
}
HOUSE_X = {"Blinky": 13.5, "Pinky": 13.5, "Inky": 11.5, "Clyde": 15.5}
FRUIT_POINTS = [100, 300, 500, 500, 700, 700, 1000, 1000, 2000, 2000, 3000, 3000, 5000]
FRUIT_NAMES = ["Cherry", "Strawberry", "Orange", "Orange", "Apple", "Apple", "Melon", "Melon",
               "Galaxian", "Galaxian", "Bell", "Bell", "Key"]
FRIGHT_TIME = [6, 5, 4, 3, 2, 5, 2, 2, 1, 5, 2, 1, 1, 3, 1, 1, 0, 1, 0]


def level_params(level: int) -> dict:
    """Arcade speed/timing table (fractions of BASE_SPEED)."""
    if level == 1:
        p = dict(pac=0.80, pac_fright=0.90, ghost=0.75, ghost_fright=0.50, tunnel=0.40,
                 elroy1_dots=20, elroy1=0.80, elroy2=0.85)
    elif level <= 4:
        p = dict(pac=0.90, pac_fright=0.95, ghost=0.85, ghost_fright=0.55, tunnel=0.45,
                 elroy1_dots=30 if level == 2 else 40, elroy1=0.90, elroy2=0.95)
    elif level <= 20:
        p = dict(pac=1.00, pac_fright=1.00, ghost=0.95, ghost_fright=0.60, tunnel=0.50,
                 elroy1_dots=min(40 + 10 * ((level - 5) // 3), 120), elroy1=1.00, elroy2=1.05)
    else:
        p = dict(pac=0.90, pac_fright=0.90, ghost=0.95, ghost_fright=0.95, tunnel=0.50,
                 elroy1_dots=120, elroy1=1.00, elroy2=1.05)
    p["elroy2_dots"] = p["elroy1_dots"] // 2
    p["fright_time"] = FRIGHT_TIME[min(level, len(FRIGHT_TIME)) - 1]
    if level == 1:
        p["waves"] = [7, 20, 7, 20, 5, 20, 5, math.inf]
    elif level <= 4:
        p["waves"] = [7, 20, 7, 20, 5, 1033, 1 / 60, math.inf]
    else:
        p["waves"] = [5, 20, 5, 20, 5, 1037, 1 / 60, math.inf]
    p["idle_release"] = 4.0 if level <= 4 else 3.0
    p["dot_limits"] = {"Pinky": 0, "Inky": 30 if level == 1 else 0,
                       "Clyde": 60 if level == 1 else (50 if level == 2 else 0)}
    return p


def tile_of(x: float, y: float) -> tuple[int, int]:
    return (int(math.floor(x + 0.5)) % COLS, int(math.floor(y + 0.5)))


@dataclass
class Actor:
    x: float
    y: float
    dir: tuple[int, int]
    moving: bool = True

    @property
    def tile(self) -> tuple[int, int]:
        return tile_of(self.x, self.y)

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    def dist_to_next_center(self) -> float:
        """Distance (tiles) along current direction to the next tile center."""
        dx, dy = self.dir
        c = self.x if dx else self.y
        s = dx or dy
        target = math.floor(c + 1e-6) + 1 if s > 0 else math.ceil(c - 1e-6) - 1
        return abs(target - c)


@dataclass
class PacMan(Actor):
    next_dir: tuple[int, int] | None = None
    turn_at: tuple[int, int] | None = None
    stall: float = 0.0
    anim: float = 0.0


@dataclass
class Ghost(Actor):
    name: str = ""
    state: str = "house"       # house | leaving | active | eyes | entering
    frightened: bool = False
    dot_counter: int = 0
    bob: float = 0.0


@dataclass
class GameEvent:
    t: float
    kind: str
    detail: str = ""
    points: int = 0


class Game:
    def __init__(self, seed: int = 0, level: int = 1, lives: int = 3, speed_scale: float = 1.0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.speed_scale = speed_scale
        self.score = 0
        self.lives = lives
        self.level = level
        self.time = 0.0
        self.events: list[GameEvent] = []
        self.extra_life_awarded = False
        self.game_over = False
        self.ghosts_eaten_total = 0
        self.energizers_eaten_total = 0
        self.deaths = 0
        self.levels_cleared = 0
        self.ghosts_eaten_this_energizer = 0
        self.energizer_ghost_counts: list[int] = []
        self._start_level(fresh=True)

    # ------------------------------------------------------------------ setup
    def _start_level(self, fresh: bool):
        self.params = level_params(self.level)
        if fresh:
            self.dots, self.energizers = initial_pellets()
            self.dots_eaten = 0
            self.fruit_spawned = 0
        self.global_dot_mode = not fresh   # after a death the arcade uses a global counter
        self.global_dot_counter = 0
        self.wave_index = 0
        self.wave_timer = 0.0
        self.fright_timer = 0.0
        self.idle_timer = 0.0
        self.freeze = 0.0
        self.dying = 0.0
        self.ready = 0.0
        self.fruit_timer = 0.0
        self.level_clear_timer = 0.0
        self.pac = PacMan(x=PAC_START[0], y=PAC_START[1], dir=LEFT)
        self.ghosts = [
            Ghost(x=13.5, y=11.0, dir=LEFT, name="Blinky", state="active"),
            Ghost(x=13.5, y=14.0, dir=DOWN, name="Pinky", state="house"),
            Ghost(x=11.5, y=14.0, dir=UP, name="Inky", state="house"),
            Ghost(x=15.5, y=14.0, dir=UP, name="Clyde", state="house"),
        ]

    def ghost(self, name: str) -> Ghost:
        for g in self.ghosts:
            if g.name == name:
                return g
        raise KeyError(name)

    def clone(self) -> "Game":
        """Lightweight copy for forward simulation (events are not carried over)."""
        c = Game.__new__(Game)
        c.__dict__.update(self.__dict__)
        c.rng = random.Random(self.seed * 7919 + int(self.time * 60))
        c.events = []
        c.dots = set(self.dots)
        c.energizers = set(self.energizers)
        c.energizer_ghost_counts = []
        p = self.pac
        c.pac = PacMan(p.x, p.y, p.dir, p.moving, p.next_dir, p.turn_at, p.stall, p.anim)
        c.ghosts = [Ghost(g.x, g.y, g.dir, g.moving, g.name, g.state, g.frightened, g.dot_counter, g.bob)
                    for g in self.ghosts]
        return c

    # ---------------------------------------------------------------- queries
    @property
    def mode(self) -> str:
        return "scatter" if self.wave_index % 2 == 0 else "chase"

    @property
    def dots_left(self) -> int:
        return len(self.dots) + len(self.energizers)

    @property
    def fruit_active(self) -> bool:
        return self.fruit_timer > 0

    @property
    def fruit_points(self) -> int:
        return FRUIT_POINTS[min(self.level, len(FRUIT_POINTS)) - 1]

    @property
    def fruit_name(self) -> str:
        return FRUIT_NAMES[min(self.level, len(FRUIT_NAMES)) - 1]

    def pac_speed(self) -> float:
        frac = self.params["pac_fright"] if self.fright_timer > 0 else self.params["pac"]
        return frac * BASE_SPEED * self.speed_scale

    def elroy_level(self) -> int:
        if self.ghost("Clyde").state == "house":
            return 0  # arcade: Elroy suspended until Clyde has left after a death
        if self.dots_left <= self.params["elroy2_dots"]:
            return 2
        if self.dots_left <= self.params["elroy1_dots"]:
            return 1
        return 0

    def ghost_speed(self, g: Ghost) -> float:
        p = self.params
        if g.state in ("eyes", "entering"):
            frac = 1.6
        elif g.state in ("house", "leaving"):
            frac = 0.45
        elif is_tunnel(*g.tile):
            frac = p["tunnel"]
        elif g.frightened:
            frac = p["ghost_fright"]
        elif g.name == "Blinky" and self.elroy_level() == 2:
            frac = p["elroy2"]
        elif g.name == "Blinky" and self.elroy_level() == 1:
            frac = p["elroy1"]
        else:
            frac = p["ghost"]
        return frac * BASE_SPEED * self.speed_scale

    def is_dangerous(self, g: Ghost) -> bool:
        return g.state in ("active", "leaving") and not g.frightened

    def fright_flashing(self) -> bool:
        return 0 < self.fright_timer < 2.0

    # ---------------------------------------------------------------- control
    def legal_dirs(self, tile=None) -> tuple:
        return exits(tile or self.pac.tile)

    def set_pac_direction(self, d: tuple[int, int]):
        """Joystick input: reversal is instant, other turns latch until the next tile center."""
        pac = self.pac
        if d == OPPOSITE.get(pac.dir) and pac.moving:
            pac.dir = d
            pac.next_dir = None
            return
        pac.next_dir = d
        if not pac.moving:
            t = pac.tile
            if abs(pac.x - round(pac.x)) < 1e-6 and abs(pac.y - round(pac.y)) < 1e-6 and d in exits(t):
                pac.dir = d
                pac.moving = True
                pac.next_dir = None

    # ----------------------------------------------------------------- update
    def update(self, dt: float):
        if self.game_over:
            return
        self.time += dt
        if self.level_clear_timer > 0:
            self.level_clear_timer -= dt
            if self.level_clear_timer <= 0:
                self.level += 1
                self._start_level(fresh=True)
            return
        if self.dying > 0:
            self.dying -= dt
            if self.dying <= 0:
                if self.lives <= 0:
                    self.game_over = True
                    self._event("game_over", f"Final score {self.score}")
                else:
                    self._start_level(fresh=False)
            return
        if self.ready > 0:
            self.ready -= dt
            return
        if self.freeze > 0:
            self.freeze -= dt
            # eyes keep moving during the ghost-eaten freeze in the arcade
            for g in self.ghosts:
                if g.state in ("eyes", "entering"):
                    self._move_ghost(g, self.ghost_speed(g) * dt)
            return

        self._update_timers(dt)
        self._move_pac(dt)
        self._eat()
        self._check_collisions()
        if self.dying > 0 or self.level_clear_timer > 0:
            return
        for g in self.ghosts:
            self._move_ghost(g, self.ghost_speed(g) * dt)
        self._check_collisions()

    def _update_timers(self, dt: float):
        if self.fright_timer > 0:
            self.fright_timer -= dt
            if self.fright_timer <= 0:
                self.fright_timer = 0
                for g in self.ghosts:
                    g.frightened = False
                self.energizer_ghost_counts.append(self.ghosts_eaten_this_energizer)
        else:
            waves = self.params["waves"]
            self.wave_timer += dt
            if self.wave_index < len(waves) and self.wave_timer >= waves[self.wave_index]:
                self.wave_timer = 0.0
                self.wave_index += 1
                self._reverse_ghosts()
        if self.fruit_timer > 0:
            self.fruit_timer -= dt
        self.idle_timer += dt
        if self.idle_timer >= self.params["idle_release"]:
            self.idle_timer = 0.0
            g = self._next_house_ghost()
            if g:
                g.state = "leaving"
        self._check_house_release()

    def _reverse_ghosts(self):
        for g in self.ghosts:
            if g.state == "active":
                g.dir = OPPOSITE[g.dir]

    def _next_house_ghost(self) -> Ghost | None:
        for name in ("Pinky", "Inky", "Clyde"):
            g = self.ghost(name)
            if g.state == "house":
                return g
        return None

    def _check_house_release(self):
        g = self._next_house_ghost()
        if not g:
            return
        if self.global_dot_mode:
            limit = {"Pinky": 7, "Inky": 17, "Clyde": 32}[g.name]
            if self.global_dot_counter >= limit:
                g.state = "leaving"
                if g.name == "Clyde":
                    self.global_dot_mode = False
        elif g.dot_counter >= self.params["dot_limits"][g.name]:
            g.state = "leaving"

    # ------------------------------------------------------------- pac motion
    def _move_pac(self, dt: float):
        pac = self.pac
        pac.anim += dt
        if pac.stall > 0:
            used = min(pac.stall, dt)
            pac.stall -= used
            dt -= used
        dist = self.pac_speed() * dt
        if not pac.moving:
            if pac.next_dir and pac.next_dir in exits(pac.tile):
                pac.dir = pac.next_dir
                pac.next_dir = None
                pac.moving = True
            else:
                return
        self._advance(pac, dist, self._pac_at_center)

    def _pac_at_center(self, pac: PacMan) -> bool:
        t = pac.tile
        legal = exits(t)
        if pac.next_dir and pac.next_dir in legal and pac.next_dir != OPPOSITE[pac.dir]:
            if pac.turn_at is None or pac.turn_at == t:
                pac.dir = pac.next_dir
                pac.next_dir = None
                pac.turn_at = None
        if pac.dir not in legal:
            # Layer-1 motor reflex: auto-corner through single bends
            fwd = [d for d in legal if d != OPPOSITE[pac.dir]]
            if len(fwd) == 1 and len(legal) == 2:
                pac.dir = fwd[0]
                return True
            pac.moving = False
            return False
        return True

    def _advance(self, a: Actor, dist: float, at_center) -> None:
        """Move an actor along the grid, invoking `at_center` at each tile center."""
        guard = 0
        while dist > 1e-9 and guard < 64:
            guard += 1
            dx, dy = a.dir
            to_c = a.dist_to_next_center()
            if to_c > dist + 1e-9:
                a.x += dx * dist
                a.y += dy * dist
                self._wrap(a)
                return
            a.x += dx * to_c
            a.y += dy * to_c
            a.x = round(a.x * 2) / 2 if not dx else round(a.x)
            a.y = round(a.y * 2) / 2 if not dy else round(a.y)
            self._wrap(a)
            dist -= to_c
            if not at_center(a):
                return

    @staticmethod
    def _wrap(a: Actor):
        if a.x < -0.5:
            a.x += COLS
        elif a.x >= COLS - 0.5:
            a.x -= COLS

    # ----------------------------------------------------------- ghost motion
    def _ghost_target(self, g: Ghost) -> tuple[int, int]:
        if g.state == "eyes":
            return (13, 11)
        if self.mode == "scatter" and not (g.name == "Blinky" and self.elroy_level() > 0):
            return SCATTER_TARGETS[g.name]
        pt = self.pac.tile
        pd = self.pac.dir
        if g.name == "Blinky":
            return pt
        if g.name == "Pinky":
            ax, ay = pt[0] + 4 * pd[0], pt[1] + 4 * pd[1]
            if pd == UP:
                ax -= 4  # arcade overflow bug
            return (ax, ay)
        if g.name == "Inky":
            ax, ay = pt[0] + 2 * pd[0], pt[1] + 2 * pd[1]
            if pd == UP:
                ax -= 2
            bt = self.ghost("Blinky").tile
            return (2 * ax - bt[0], 2 * ay - bt[1])
        # Clyde
        if toroidal_dist(g.tile, pt) > 8:
            return pt
        return SCATTER_TARGETS["Clyde"]

    def _ghost_at_center(self, g: Ghost) -> bool:
        t = g.tile
        if g.state == "eyes" and t in ((13, 11), (14, 11)):
            g.x, g.y = GHOST_DOOR_EXIT
            g.state = "entering"
            g.dir = DOWN
            return False
        options = [d for d in exits(t) if d != OPPOSITE[g.dir]]
        if not g.frightened and g.state == "active" and t in NO_UP_TILES and UP in options and len(options) > 1:
            options.remove(UP)
        if not options:
            g.dir = OPPOSITE[g.dir]
            return True
        if g.frightened and g.state == "active":
            g.dir = self.rng.choice(options)
            return True
        tx, ty = self._ghost_target(g)
        best, best_d = None, math.inf
        for d in DIRS:  # tie-break preference UP, LEFT, DOWN, RIGHT
            if d not in options:
                continue
            nx, ny = step(t, d)
            dd = (nx - tx) ** 2 + (ny - ty) ** 2
            if dd < best_d:
                best, best_d = d, dd
        g.dir = best
        return True

    def _move_ghost(self, g: Ghost, dist: float):
        if g.state == "house":
            g.bob += dist
            g.y = 14.0 + 0.35 * math.sin(g.bob * 3.0)
            return
        if g.state == "leaving":
            g.frightened = False
            hx = 13.5
            if abs(g.x - hx) > 1e-3:
                s = 1 if hx > g.x else -1
                mv = min(dist, abs(hx - g.x))
                g.x += s * mv
                g.dir = RIGHT if s > 0 else LEFT
                dist -= mv
            if abs(g.y - 14.0) > 0.01 and g.y > 14.0 and abs(g.x - hx) <= 1e-3:
                pass
            if dist > 0 and abs(g.x - hx) <= 1e-3:
                mv = min(dist, g.y - GHOST_DOOR_EXIT[1])
                g.y -= mv
                g.dir = UP
                if g.y <= GHOST_DOOR_EXIT[1] + 1e-6:
                    g.y = GHOST_DOOR_EXIT[1]
                    g.state = "active"
                    g.dir = LEFT
            return
        if g.state == "entering":
            hx = HOUSE_X[g.name] if g.name != "Blinky" else 13.5
            if g.y < 14.0:
                mv = min(dist, 14.0 - g.y)
                g.y += mv
                dist -= mv
            if dist > 0 and g.y >= 14.0 - 1e-6:
                if abs(g.x - hx) > 1e-3:
                    s = 1 if hx > g.x else -1
                    g.x += s * min(dist, abs(hx - g.x))
                else:
                    g.state = "leaving"
                    g.frightened = False
            return
        self._advance(g, dist, self._ghost_at_center)

    # ----------------------------------------------------------- interactions
    def _eat(self):
        pac = self.pac
        t = pac.tile
        ate = False
        if t in self.dots:
            self.dots.discard(t)
            self.score += 10
            pac.stall += FRAME
            ate = True
        elif t in self.energizers:
            self.energizers.discard(t)
            self.score += 50
            pac.stall += 3 * FRAME
            ate = True
            self._start_fright()
        if ate:
            self.dots_eaten += 1
            self.idle_timer = 0.0
            if self.global_dot_mode:
                self.global_dot_counter += 1
            else:
                g = self._next_house_ghost()
                if g:
                    g.dot_counter += 1
            if self.dots_eaten in (70, 170) and self.fruit_spawned < 2:
                self.fruit_spawned += 1
                self.fruit_timer = 9.5
            if self.dots_left == 0:
                self.levels_cleared += 1
                self.level_clear_timer = 2.0
                self._event("level_clear", f"Level {self.level} cleared")
        if self.fruit_active and toroidal_dist(pac.pos, FRUIT_POS) < 0.6:
            self.fruit_timer = 0
            self.score += self.fruit_points
            self._event("fruit", self.fruit_name, self.fruit_points)
        if not self.extra_life_awarded and self.score >= 10000:
            self.extra_life_awarded = True
            self.lives += 1
            self._event("extra_life", "Bonus Pac-Man at 10,000")

    def _start_fright(self):
        self.energizers_eaten_total += 1
        if self.fright_timer > 0:
            self.energizer_ghost_counts.append(self.ghosts_eaten_this_energizer)
        self.ghosts_eaten_this_energizer = 0
        self._event("energizer", "Power pellet")
        ft = self.params["fright_time"]
        for g in self.ghosts:
            if g.state == "active":
                g.dir = OPPOSITE[g.dir]
        if ft <= 0:
            return
        self.fright_timer = float(ft)
        for g in self.ghosts:
            if g.state in ("active", "house"):
                g.frightened = True

    def _check_collisions(self):
        pac = self.pac
        for g in self.ghosts:
            if g.state not in ("active", "leaving"):
                continue
            if g.tile != pac.tile and toroidal_dist(g.pos, pac.pos) > 0.55:
                continue
            if g.frightened and g.state == "active":
                self.ghosts_eaten_this_energizer += 1
                self.ghosts_eaten_total += 1
                pts = 200 * (2 ** (self.ghosts_eaten_this_energizer - 1))
                self.score += pts
                g.state = "eyes"
                g.frightened = False
                self.freeze = 0.5
                self._event("ghost_eaten", g.name, pts)
            elif not g.frightened:
                self.lives -= 1
                self.deaths += 1
                self.dying = 1.6
                self.fright_timer = 0
                self._event("death", f"Caught by {g.name}")
                return

    def _event(self, kind: str, detail: str = "", points: int = 0):
        self.events.append(GameEvent(self.time, kind, detail, points))

    # ------------------------------------------------------------------ stats
    @property
    def ghosts_per_energizer(self) -> float:
        counts = list(self.energizer_ghost_counts)
        if self.fright_timer > 0:
            counts.append(self.ghosts_eaten_this_energizer)
        return sum(counts) / len(counts) if counts else 0.0

    @property
    def playing(self) -> bool:
        return not (self.game_over or self.dying > 0 or self.level_clear_timer > 0
                    or self.ready > 0 or self.freeze > 0)
