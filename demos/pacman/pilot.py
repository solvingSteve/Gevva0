"""
Pilot: predictive junction dispatch + latch (PRD section 5.1).

Every time the tactical picture changes (Pac-Man or a ghost enters a new tile,
fright starts/ends...), the pilot snapshots a Situation for the junction Pac-Man
is committed to and dispatches it to the brain. Results are latched into
`pac.next_dir` tagged with the target junction, so the turn executes the instant
Pac-Man reaches the junction center: 0 ms effective decision latency.

A REVERSE verdict is applied immediately (instant 180 is legal in the arcade).
Stale verdicts (Pac-Man already past the junction / changed heading) are dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from game import Game
from tactics import Situation, analyze


@dataclass
class PilotStats:
    dispatches: int = 0
    applied: int = 0
    stale: int = 0
    reversals: int = 0
    junction_arrivals: int = 0
    prebuffered_hits: int = 0
    unbuffered_arrivals: int = 0


class Pilot:
    def __init__(self, game: Game, brain, sync: bool = False, cot_slack_ms: float | None = None):
        self.game = game
        self.brain = brain
        self.sync = sync
        self.stats = PilotStats()
        self.last_key = None
        self.serial = 0
        self.situation: Situation | None = None          # latest analysed situation (for HUD)
        self.pending: Situation | None = None            # situation currently being scored
        self.decided_sit: Situation | None = None        # situation of the last applied verdict
        self.decision = None
        self.latched: tuple | None = None                # (junction, approach, dir)
        self.cot_slack_ms = cot_slack_ms
        self._last_tile = None
        self.dispatch_times: dict[int, float] = {}
        self.rev_tile = None
        self.rev_time = -99.0

    def _reverse_allowed(self) -> bool:
        g = self.game
        if g.time - self.rev_time > 1.0:
            return True
        return g.pac.tile != self.rev_tile and g.time - self.rev_time > 0.25

    def reset(self):
        self.last_key = None
        self.latched = None
        self.pending = None

    def state_key(self):
        g = self.game
        p = g.pac
        return (
            p.tile, p.dir, p.moving, g.level, g.lives,
            tuple((gh.tile, gh.state, gh.frightened) for gh in g.ghosts),
            g.fright_timer > 0, int(g.fright_timer * 2) if g.fright_timer > 0 else 0,
            g.fruit_active, g.mode, g.dots_left,
        )

    def _allow_cot(self, sit: Situation) -> bool:
        slack = self.cot_slack_ms if self.cot_slack_ms is not None else self.brain.cot_latency_ms * 1.25 + 40
        return sit.eta_ms > slack

    def tick(self):
        g = self.game
        if not g.playing:
            self.last_key = None
            return
        self._track_arrivals()

        # 1) collect finished verdicts (async mode)
        if not self.sync:
            res = self.brain.poll()
            if res is not None:
                sit, d = res
                self.pending = None
                self._apply(sit, d)

        # 2) dispatch when the tactical picture changed
        key = self.state_key()
        if key == self.last_key:
            return
        if not self.sync and self.brain.busy:
            return  # GPU busy: the next completion triggers a fresh dispatch with the newest state
        sit = analyze(g, allow_reverse=self._reverse_allowed())
        if sit is None or not sit.options:
            return
        self.serial += 1
        sit.serial = self.serial
        self.situation = sit
        self.last_key = key
        self.stats.dispatches += 1
        allow_cot = self._allow_cot(sit)
        if self.sync:
            d = self.brain.decide(sit, allow_cot)
            self._apply(sit, d)
        else:
            self.pending = sit
            self.dispatch_times[sit.serial] = time.perf_counter()
            self.brain.submit(sit, allow_cot)

    def _apply(self, sit: Situation, d):
        g = self.game
        pac = g.pac
        opt = sit.option(d.letter)
        if opt is None:
            return
        cur_tile_ok = True
        if pac.moving:
            # verdict must still refer to the junction Pac-Man is heading to
            cur = analyze_target(g)
            cur_tile_ok = cur == (sit.junction, sit.approach)
        else:
            cur_tile_ok = (pac.tile == sit.junction)
        if not cur_tile_ok and not opt.reverse:
            self.stats.stale += 1
            return
        self.decision = d
        self.decided_sit = sit
        self.stats.applied += 1
        if opt.reverse:
            if pac.moving and pac.dir == opt.dir:
                return  # already reversed
            if pac.moving and (pac.dir != (-opt.dir[0], -opt.dir[1])):
                self.stats.stale += 1
                return
            g.set_pac_direction(opt.dir)
            self.latched = None
            if opt.turn is not None and opt.turn != opt.dir:
                pac.next_dir = opt.turn
                pac.turn_at = opt.turn_at
                self.latched = (opt.turn_at, opt.dir, opt.turn)
            self.stats.reversals += 1
            self.rev_tile = pac.tile
            self.rev_time = g.time
            self.last_key = None  # re-plan immediately for the new heading
            return
        if not pac.moving:
            g.set_pac_direction(opt.dir)
            self.latched = None
            return
        pac.next_dir = opt.dir
        pac.turn_at = sit.junction
        self.latched = (sit.junction, sit.approach, opt.dir)

    def _track_arrivals(self):
        pac = self.game.pac
        t = pac.tile
        if t == self._last_tile:
            return
        self._last_tile = t
        from maze import JUNCTIONS
        if t in JUNCTIONS:
            self.stats.junction_arrivals += 1
            if self.latched and self.latched[0] == t:
                self.stats.prebuffered_hits += 1
            else:
                self.stats.unbuffered_arrivals += 1


def analyze_target(game: Game):
    """Cheap recomputation of (junction, approach) Pac-Man is currently committed to."""
    from maze import JUNCTIONS, OPPOSITE, exits, step
    from game import tile_of
    pac = game.pac
    dnc = pac.dist_to_next_center()
    cur = tile_of(pac.x + pac.dir[0] * dnc, pac.y + pac.dir[1] * dnc)
    cd = pac.dir
    for _ in range(200):
        if cur in JUNCTIONS:
            return (cur, cd)
        ex = [e for e in exits(cur) if e != OPPOSITE[cd]]
        if not ex:
            return (cur, cd)
        if cd not in ex:
            cd = ex[0]
        cur = step(cur, cd)
    return (cur, cd)
