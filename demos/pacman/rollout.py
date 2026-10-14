"""
Forward-simulation perception (exact ghost prediction).

Arcade ghosts are deterministic (except while frightened), so the most truthful
way to measure a route's danger is to simulate it. For each candidate route we
clone the game, drive Pac-Man along the route, then branch at every following
junction up to a short horizon. The result is a set of *facts* per route:

    survive_s  longest survival time found (capped at horizon)
    outs       number of distinct continuations that survive the full horizon
    min_gap    closest approach (tiles) of a lethal ghost on the best continuation
    eaten      blue ghosts eaten on the best continuation

The model still chooses; this only measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from game import Game, tile_of
from maze import JUNCTIONS, OPPOSITE, exits, follow_corridor, toroidal_dist

SIM_DT = 2.0 / 60.0
HORIZON = 3.0
MAX_NODES = 160


@dataclass
class RolloutResult:
    survive_s: float
    outs: int
    min_gap: float
    eaten: int
    died_on_path: bool
    dots: int


class _Budget:
    def __init__(self, n):
        self.n = n


def _lethal_gap(g: Game) -> float:
    p = g.pac.pos
    best = 99.0
    for gh in g.ghosts:
        if gh.state in ("active", "leaving") and not gh.frightened:
            d = toroidal_dist(p, gh.pos)
            if d < best:
                best = d
    return best


def _run_until(g: Game, end, t_limit: float, start_tile) -> tuple[str, float]:
    """Advance until Pac-Man enters `end` (after leaving start), dies, or time runs out.
    Returns (status, min_gap) with status in {'arrived', 'dead', 'horizon', 'stuck'}."""
    left_start = g.pac.tile != start_tile
    deaths0 = g.deaths
    min_gap = 99.0
    guard = 0
    while g.time < t_limit:
        guard += 1
        if guard > 400:
            return "stuck", min_gap
        g.update(SIM_DT)
        if g.deaths > deaths0 or g.dying > 0:
            return "dead", 0.0
        if g.level_clear_timer > 0:
            return "horizon", min_gap
        gap = _lethal_gap(g)
        if gap < min_gap:
            min_gap = gap
        t = g.pac.tile
        if t != start_tile:
            left_start = True
        if left_start and t == end:
            return "arrived", min_gap
        if not g.pac.moving and g.pac.tile in JUNCTIONS:
            return "arrived", min_gap
    return "horizon", min_gap


def _explore(g: Game, node, t_end: float, budget: _Budget) -> tuple[float, int, float, int]:
    """At junction `node` (Pac-Man entering its tile), branch over exits.
    Returns (best_survival_time_abs, outs, best_min_gap, best_eaten)."""
    if g.time >= t_end:
        return g.time, 1, 99.0, 0
    best = (g.time, 0, 0.0, 0)
    outs = 0
    arrival = g.pac.dir
    for e in exits(node):
        if e == OPPOSITE[arrival] or budget.n <= 0:
            continue
        budget.n -= 1
        c = g.clone()
        c.pac.next_dir = e
        c.pac.turn_at = node
        if not c.pac.moving:
            c.set_pac_direction(e)
        _, nxt, _ = follow_corridor(node, e)
        eaten0 = c.ghosts_eaten_total
        status, gap = _run_until(c, nxt, t_end, node)
        if status == "dead":
            cand = (c.time, 0, 0.0, 0)
        elif status in ("horizon", "stuck"):
            cand = (t_end, 1, gap, c.ghosts_eaten_total - eaten0)
        else:
            st, o, sub_gap, sub_eat = _explore(c, nxt, t_end, budget)
            cand = (st, o, min(gap, sub_gap), c.ghosts_eaten_total - eaten0 + sub_eat)
        outs += cand[1]
        if (cand[0], cand[2]) > (best[0], best[2]):
            best = cand
    return best[0], outs, best[2], best[3]


def rollout_route(game: Game, opt, horizon: float = HORIZON, budget_nodes: int = MAX_NODES) -> RolloutResult:
    g = game.clone()
    t0 = g.time
    pac = g.pac
    start_tile = pac.tile
    if opt.reverse:
        g.set_pac_direction(opt.dir)
        if opt.turn is not None and opt.turn != opt.dir:
            g.pac.next_dir = opt.turn
            g.pac.turn_at = opt.turn_at
    elif not pac.moving:
        g.set_pac_direction(opt.dir)
    else:
        pac.next_dir = opt.dir
        pac.turn_at = opt.path[0] if False else _junction_of(opt)
    speed = max(game.pac_speed(), 1e-3)
    path_time = len(opt.path) / speed
    t_end = t0 + max(horizon, path_time + 1.2)
    dots0 = len(g.dots)
    eaten0 = g.ghosts_eaten_total
    status, gap = _run_until(g, opt.end, t_end, start_tile)
    if status == "dead":
        return RolloutResult(g.time - t0, 0, 0.0, 0, True, dots0 - len(g.dots))
    if status in ("horizon", "stuck"):
        return RolloutResult(t_end - t0, 1, gap, g.ghosts_eaten_total - eaten0, False, dots0 - len(g.dots))
    budget = _Budget(budget_nodes)
    st, outs, sub_gap, sub_eat = _explore(g, opt.end, t_end, budget)
    return RolloutResult(st - t0, outs, min(gap, sub_gap), g.ghosts_eaten_total - eaten0 + sub_eat,
                         False, dots0 - len(g.dots))


def _junction_of(opt):
    """The junction where a forward option's turn happens (first junction on its path)."""
    for t in opt.path:
        if t in JUNCTIONS:
            return t
    return opt.path[0]
