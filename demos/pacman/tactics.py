"""
Tactical perception layer (PRD sections 5 & 6).

Turns the raw game state into a *Situation*: the upcoming junction Pac-Man is
committed to, plus 2-4 route options, each annotated with measured facts:

    status   SAFE | TIGHT | TRAPPED | DEADLY
    room     tiles Pac-Man can still reach ahead of every dangerous ghost via this route
    gap      head start (tiles) over the closest dangerous ghost along the committed path
    path     dots on the route to the next junction
    near     route distance to the closest dot
    ahead    dots within 14 tiles via this route
    power    route distance to a reachable power pellet
    eat      a blue ghost this route can catch before fright expires (+points)

Perception only *measures*; it never picks. The Gevva0 decision model weighs the
options (survival vs. points vs. tempo) via calibrated logit scoring. The only
hard rule applied here is the PRD 6.4 legal-move pruning: walls are never
offered and DEADLY routes are stripped whenever a survivable route exists.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from maze import (
    DIR_NAMES,
    DIRS,
    JUNCTIONS,
    OPPOSITE,
    exits,
    follow_corridor,
    step,
)
from game import BASE_SPEED, Game, Ghost, tile_of

SAFE_MARGIN = 0.75     # tiles of head start required to pass through a tile
FREE_ROOM = 45         # room at/above which a route is considered wide open
TRAP_ROOM = 8          # room below which a route is a closing pocket
NEAR_GHOST = 7         # BFS tiles: a dangerous ghost this close raises a threat alert
ROOM_CAP = 200
LETTERS = "ABCDEFGH"

STATUS_RANK = {"SAFE": 0, "TIGHT": 1, "TRAPPED": 2, "DEADLY": 3}


@dataclass
class RouteOption:
    name: str                      # UP / LEFT / DOWN / RIGHT / REVERSE
    dir: tuple[int, int]           # joystick direction to apply
    reverse: bool
    path: list[tuple[int, int]]    # committed tiles from Pac-Man to the end node
    end: tuple[int, int]
    status: str = "SAFE"
    room: int = 0
    gap: float = 99.0
    gap_ghost: str = ""
    path_dots: int = 0
    near: int | None = None
    ahead: int = 0
    power: int | None = None
    eat: tuple[str, int, int] | None = None   # (ghost, distance, points)
    fruit: int | None = None
    tunnel: bool = False
    key: str = ""
    turn: tuple[int, int] | None = None      # for BACK-x options: turn to take at turn_at
    turn_at: tuple[int, int] | None = None
    outs: int = 0
    survive: float = 0.0
    sim_eaten: int = 0

    def card(self) -> str:
        """Token-minimal option line fed to the model."""
        gap = f"{self.gap:.1f}" if self.gap < 50 else "clear"
        parts = [f"{self.name}", f"{self.status} outs {self.outs} gap {gap} room {self.room}"]
        near = "none" if self.near is None else str(self.near)
        parts.append(f"dots path {self.path_dots} near {near} ahead {self.ahead}")
        extras = []
        if self.power is not None:
            extras.append(f"power {self.power}")
        if self.eat:
            extras.append(f"eat {self.eat[0]} {self.eat[1]}t +{self.eat[2]}")
        if self.fruit is not None:
            extras.append(f"fruit {self.fruit}")
        if self.tunnel:
            extras.append("tunnel")
        if extras:
            parts.append(" ".join(extras))
        return " | ".join(parts)


@dataclass
class Situation:
    junction: tuple[int, int]
    approach: tuple[int, int] | None
    dist_to_junction: float
    eta_ms: float
    options: list[RouteOption]
    pruned: list[RouteOption]
    threat_alert: bool
    power_value: str
    hunters_near: int
    ghost_lines: list[str]
    header: str
    suffix: str
    danger_tiles: set = field(default_factory=set)
    pac_tile: tuple[int, int] = (0, 0)
    created: float = 0.0
    serial: int = 0

    @property
    def labels(self) -> list[str]:
        return [o.key for o in self.options]

    def option(self, key: str) -> RouteOption | None:
        return next((o for o in self.options if o.key == key), None)


# ---------------------------------------------------------------------------
# Ghost timing maps
# ---------------------------------------------------------------------------

def _bfs(seeds: list[tuple[tuple[int, int], float]], blocked: dict | None = None, limit: int = 999):
    """Multi-seed BFS returning tile -> distance (tiles). `blocked` maps tile->dir forbidden on first expansion."""
    dist: dict[tuple[int, int], float] = {}
    q: deque = deque()
    for t, d in sorted(seeds, key=lambda s: s[1]):
        if t not in dist or d < dist[t]:
            dist[t] = d
            q.append(t)
    while q:
        t = q.popleft()
        dt = dist[t]
        if dt >= limit:
            continue
        for d in exits(t):
            if blocked and blocked.get(t) == d:
                continue
            n = step(t, d)
            nd = dt + 1
            if n not in dist or nd < dist[n]:
                dist[n] = nd
                q.append(n)
    return dist


def ghost_cruise_speed(game: Game, g: Ghost) -> float:
    p = game.params
    if g.frightened:
        frac = p["ghost_fright"]
    elif g.name == "Blinky" and game.elroy_level() == 2:
        frac = p["elroy2"]
    elif g.name == "Blinky" and game.elroy_level() == 1:
        frac = p["elroy1"]
    else:
        frac = p["ghost"]
    return frac * BASE_SPEED * game.speed_scale


@dataclass
class GhostThreat:
    ghost: Ghost
    times: dict           # tile -> seconds until this ghost can be there
    harmless_until: float  # seconds; >0 for frightened ghosts
    tile_dist: dict       # tile -> tiles (for display)


def ghost_threats(game: Game) -> list[GhostThreat]:
    threats = []
    door = [((13, 11), 0.0), ((14, 11), 0.0)]
    for g in game.ghosts:
        if g.state in ("eyes", "entering"):
            continue
        spd = ghost_cruise_speed(game, g)
        if g.state == "house":
            base = 2.5
            dmap = _bfs(door)
            times = {t: base + d / spd for t, d in dmap.items()}
        elif g.state == "leaving":
            base = (abs(g.x - 13.5) + abs(g.y - 11.0)) / (0.45 * BASE_SPEED * game.speed_scale)
            dmap = _bfs(door)
            times = {t: base + d / spd for t, d in dmap.items()}
        else:
            dnc = g.dist_to_next_center()
            nc = tile_of(g.x + g.dir[0] * dnc, g.y + g.dir[1] * dnc)
            dmap = _bfs([(nc, dnc), (g.tile, 0.0)], blocked={nc: OPPOSITE[g.dir]})
            times = {t: d / spd for t, d in dmap.items()}
        harmless = game.fright_timer if (g.frightened and g.state != "leaving") else 0.0
        if g.frightened and g.state == "house":
            harmless = game.fright_timer
        threats.append(GhostThreat(g, times, harmless, dmap))
    return threats


def danger_time(threats: list[GhostThreat], tile, pac_t: float) -> tuple[float, str]:
    """Earliest time a ghost that is lethal *at that moment* can occupy `tile`."""
    best, who = math.inf, ""
    for th in threats:
        tg = th.times.get(tile)
        if tg is None:
            continue
        if th.harmless_until > 0:
            if pac_t < th.harmless_until - 0.35:
                continue  # still blue when Pac-Man gets there
            tg = max(tg, th.harmless_until - 0.35)
        if tg < best:
            best, who = tg, th.ghost.name
    return best, who


# ---------------------------------------------------------------------------
# Situation analysis
# ---------------------------------------------------------------------------

def _walk_to_junction(start, d):
    """From tile `start` (arrived travelling in `d`), follow the corridor to a junction."""
    path = [start]
    cur, cd = start, d
    for _ in range(200):
        if cur in JUNCTIONS:
            break
        ex = [e for e in exits(cur) if e != OPPOSITE[cd]]
        if not ex:
            break
        if cd not in ex:
            cd = ex[0]
        cur = step(cur, cd)
        path.append(cur)
    return path, cur, cd


def analyze(game: Game, allow_reverse: bool = True) -> Situation | None:
    pac = game.pac
    if game.game_over or game.dying > 0 or game.level_clear_timer > 0:
        return None
    speed = game.pac_speed()

    # --- where is Pac-Man committed to? -------------------------------------
    if not pac.moving:
        J, dJ, approach = pac.tile, 0.0, None
        to_j: list = []
        to_j_d: list = []
    else:
        dnc = pac.dist_to_next_center()
        nc = tile_of(pac.x + pac.dir[0] * dnc, pac.y + pac.dir[1] * dnc)
        to_j, J, approach = _walk_to_junction(nc, pac.dir)
        to_j_d = [dnc + i for i in range(len(to_j))]
        dJ = to_j_d[-1]

    threats = ghost_threats(game)
    dangerous = [th for th in threats if th.harmless_until <= 0]
    pac_tile = pac.tile
    pac_bfs = _bfs([(pac_tile, 0.0)])
    hunters = sorted(
        (th for th in dangerous if th.ghost.state != "house"),
        key=lambda th: min((pac_bfs.get(t, 999) for t in [th.ghost.tile]), default=999),
    )
    hunters_near = sum(1 for th in hunters if pac_bfs.get(th.ghost.tile, 999) <= NEAR_GHOST)

    dots_only_energizers = len(game.dots) == 0 and len(game.energizers) > 0
    if game.fright_timer > 0:
        power_value = "NONE (already powered)"
    elif dots_only_energizers:
        power_value = "REQUIRED (last pellets)"
    elif hunters_near >= 2:
        power_value = f"HIGH ({hunters_near} hunters within {NEAR_GHOST})"
    elif hunters_near == 1:
        power_value = "MEDIUM (1 hunter close)"
    else:
        power_value = "LOW (save it)"

    # --- candidate routes ----------------------------------------------------
    candidates: list[RouteOption] = []
    j_exits = [e for e in exits(J) if approach is None or e != OPPOSITE[approach]]
    for e in DIRS:
        if e not in j_exits:
            continue
        seg, end, _ = follow_corridor(J, e)
        path = list(to_j) + seg
        if not to_j:
            path = [J] + seg
            dists = [0.0] + [1.0 + i for i in range(len(seg))]
        else:
            dists = list(to_j_d) + [dJ + 1 + i for i in range(len(seg))]
        candidates.append(_score_route(game, threats, DIR_NAMES[e], e, False, path, dists, end, speed))

    if pac.moving and allow_reverse:
        back = OPPOSITE[pac.dir]
        at_center = abs(pac.x - round(pac.x)) < 1e-6 and abs(pac.y - round(pac.y)) < 1e-6
        if at_center:
            bs, d0 = pac.tile, 0.0
        else:
            dnc = pac.dist_to_next_center()
            nc = tile_of(pac.x + pac.dir[0] * dnc, pac.y + pac.dir[1] * dnc)
            bs, d0 = step(nc, back), 1.0 - dnc
        if bs in JUNCTIONS and not at_center:
            # just passed a junction: reverse back onto it, then take one of its other exits
            for e in DIRS:
                if e not in exits(bs) or e == pac.dir:
                    continue
                seg, rend, _ = follow_corridor(bs, e)
                rpath = [bs] + seg
                rd = [d0 + i for i in range(len(rpath))]
                nm = "REVERSE" if e == back else f"BACK-{DIR_NAMES[e]}"
                o = _score_route(game, threats, nm, back, True, rpath, rd, rend, speed)
                o.turn = e
                o.turn_at = bs
                candidates.append(o)
        else:
            if bs in JUNCTIONS:
                seg, rend, _ = follow_corridor(bs, back)
                rpath = [bs] + seg
            else:
                rpath, rend, _ = _walk_to_junction(bs, back)
            rd = [d0 + i for i in range(len(rpath))]
            candidates.append(_score_route(game, threats, "REVERSE", back, True, rpath, rd, rend, speed))

    forward = [c for c in candidates if not c.reverse]
    best_fwd = min((STATUS_RANK[c.status] for c in forward), default=3)
    threat_alert = hunters_near > 0 or best_fwd >= STATUS_RANK["TIGHT"]

    options = list(candidates)
    revs = [c for c in options if c.reverse]
    if revs:
        best_rev = min(STATUS_RANK[c.status] for c in revs)
        # reversal is only offered when it is strictly safer than going forward
        keep_reverse = best_rev < best_fwd
        if not keep_reverse:
            options = [c for c in options if not c.reverse]
        else:
            options = [c for c in options if not c.reverse or STATUS_RANK[c.status] < best_fwd]

    # PRD 6.4: strip DEADLY routes whenever a survivable route exists
    pruned = []
    if any(o.status != "DEADLY" for o in options):
        pruned = [o for o in options if o.status == "DEADLY"]
        options = [o for o in options if o.status != "DEADLY"]

    for i, o in enumerate(options):
        o.key = LETTERS[i]

    # --- threat visualisation: tiles a ghost reaches before Pac-Man ----------
    danger_tiles = set()
    for t, d in pac_bfs.items():
        if d > 18:
            continue
        tg, _ = danger_time(threats, t, d / speed)
        if (tg - d / speed) * speed < SAFE_MARGIN:
            danger_tiles.add(t)

    ghost_lines = []
    for g in game.ghosts:
        if g.state in ("eyes", "entering"):
            st = "eyes"
        elif g.state == "house":
            st = "home"
        elif g.frightened:
            st = f"BLUE {game.fright_timer:.1f}s"
        else:
            st = game.mode.upper() if not (g.name == "Blinky" and game.elroy_level()) else "ELROY"
        d = pac_bfs.get(g.tile)
        ghost_lines.append(f"{g.name} {st} {int(d) if d is not None else '?'}t")

    eta_ms = dJ / max(speed, 1e-6) * 1000.0
    header = (
        f"Level {game.level} | dots left {game.dots_left} | ghosts {game.mode.upper()} | "
        f"power value {power_value}\n"
        f"Pac-Man heading {DIR_NAMES.get(pac.dir, '?')}, {dJ:.0f} tiles to junction {J[0]},{J[1]}\n"
        f"Ghosts: {', '.join(ghost_lines)}"
    )
    suffix = header + "\nOptions:\n" + "\n".join(f"{o.key}. {o.card()}" for o in options)

    return Situation(
        junction=J,
        approach=approach,
        dist_to_junction=dJ,
        eta_ms=eta_ms,
        options=options,
        pruned=pruned,
        threat_alert=threat_alert,
        power_value=power_value,
        hunters_near=hunters_near,
        ghost_lines=ghost_lines,
        header=header,
        suffix=suffix,
        danger_tiles=danger_tiles,
        pac_tile=pac_tile,
        created=game.time,
    )


def _score_route(game: Game, threats, name, d, reverse, path, dists, end, speed) -> RouteOption:
    opt = RouteOption(name=name, dir=d, reverse=reverse, path=path, end=end)
    opt.tunnel = any(t[1] == 14 and (t[0] <= 1 or t[0] >= 26) for t in path)

    # committed path safety
    min_gap, gap_ghost = math.inf, ""
    reached: dict = {}
    ok = True
    for t, pd in zip(path, dists):
        tp = pd / speed
        tg, who = danger_time(threats, t, tp)
        gap = (tg - tp) * speed
        if gap < min_gap:
            min_gap, gap_ghost = gap, who
        if ok and gap >= SAFE_MARGIN:
            reached.setdefault(t, pd)
        else:
            ok = False
    opt.gap = round(min_gap, 1) if min_gap < 99 else 99.0
    opt.gap_ghost = gap_ghost

    # safe expansion beyond the committed path, whole corridors only (junction graph):
    # a corridor counts only if Pac-Man can traverse ALL of it ahead of every ghost,
    # otherwise it is a pocket he can be sealed inside.
    if ok:
        q = deque([(end, reached.get(end, dists[-1]))])
        seen_nodes = {end}
        while q and len(reached) < ROOM_CAP:
            node, nd0 = q.popleft()
            for e in exits(node):
                seg, seg_end, _ = follow_corridor(node, e)
                if not seg or seg_end in seen_nodes and all(t in reached for t in seg):
                    continue
                good = True
                for i, t in enumerate(seg):
                    nd = nd0 + 1 + i
                    tg, _ = danger_time(threats, t, nd / speed)
                    if (tg - nd / speed) * speed < SAFE_MARGIN:
                        good = False
                        break
                if not good:
                    continue
                for i, t in enumerate(seg):
                    reached.setdefault(t, nd0 + 1 + i)
                if seg_end not in seen_nodes:
                    seen_nodes.add(seg_end)
                    q.append((seg_end, nd0 + len(seg)))
    committed = len(set(path))
    opt.room = max(0, len(reached) - committed)

    # unrestricted route distances (dots / blue ghosts)
    free = {t: pd for t, pd in zip(path, dists)}
    q = deque([end])
    while q:
        t = q.popleft()
        for e in exits(t):
            n = step(t, e)
            if n not in free:
                free[n] = free[t] + 1
                q.append(n)

    dots = game.dots
    targets = dots if dots else game.energizers
    opt.path_dots = sum(1 for t in set(path) if t in dots)
    near = min((free[t] for t in targets if t in free), default=None)
    opt.near = None if near is None else int(math.ceil(near))
    opt.ahead = sum(1 for t in targets if free.get(t, 999) <= 14)

    pw = [reached[t] for t in game.energizers if t in reached]
    opt.power = int(math.ceil(min(pw))) if pw else None

    if game.fruit_active and (13, 17) in reached:
        opt.fruit = int(reached[(13, 17)])

    # catchable blue ghosts
    if game.fright_timer > 0:
        best = None
        value = 200 * (2 ** game.ghosts_eaten_this_energizer)
        for th in threats:
            g = th.ghost
            if not (g.frightened and g.state == "active"):
                continue
            dist = free.get(g.tile)
            if dist is None:
                continue
            t_catch = dist / max(speed - ghost_cruise_speed(game, g) * 0.35, 1e-3)
            if t_catch < game.fright_timer - 0.25 and (best is None or dist < best[1]):
                best = (g.name, int(math.ceil(dist)), min(value, 1600))
        opt.eat = best

    # status from exact forward simulation of the ghosts
    from rollout import HORIZON, rollout_route
    rr = rollout_route(game, opt)
    opt.outs = rr.outs
    opt.survive = round(rr.survive_s, 2)
    opt.sim_eaten = rr.eaten
    opt.gap = round(min(rr.min_gap, 99.0), 1)
    opt.gap_ghost = gap_ghost
    if rr.died_on_path:
        opt.status = "DEADLY"
    elif rr.outs == 0:
        opt.status = "TRAPPED"
    elif rr.outs <= 1 or rr.min_gap < 2.5:
        opt.status = "TIGHT"
    else:
        opt.status = "SAFE"
    if opt.status != "DEADLY" and opt.power is None:
        pw = [free[t] for t in game.energizers if t in free and free[t] <= 20]
        opt.power = int(math.ceil(min(pw))) if pw else None
    return opt


# ---------------------------------------------------------------------------
# Reference heuristic (used ONLY by the --simulate brain and calibration rollouts)
# ---------------------------------------------------------------------------

def heuristic_utility(o: RouteOption, sit: Situation) -> float:
    u = {"SAFE": 0.0, "TIGHT": -25.0, "TRAPPED": -200.0, "DEADLY": -1000.0}[o.status]
    if sit.threat_alert:
        u += min(o.outs, 10) * 4.0 + min(o.gap, 10) * 2.0 + min(o.room, 80) * 0.1
    if o.eat:
        u += 90 - 2.5 * o.eat[1]
    if o.power is not None:
        if sit.power_value.startswith(("HIGH", "REQUIRED")):
            u += 70 - 3 * o.power
        elif sit.power_value.startswith("LOW") and o.power <= 3:
            u -= 20
    u += 3.0 * o.path_dots + 0.4 * o.ahead
    if o.near is not None:
        u += max(0, 25 - o.near) * 1.2
    if o.fruit is not None:
        u += max(0, 30 - o.fruit)
    if o.reverse:
        u -= 8
    return u
