"""
Maze topology for the Gevva0 Arcade Pac-Man showcase (v2).

Classic 28x31 arcade layout. Tile legend:
    '#' wall          '.' pellet        'o' energizer (power pellet)
    '_' empty floor   ' ' void (outside the maze, rendered black)
    '-' ghost door    'H' ghost house interior

Coordinates are (x, y) tile indices; tile centers sit on integer coordinates.
Row 14 contains the left/right warp tunnel (x wraps around COLS).
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

LAYOUT = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.#####_##_#####.######",
    "     #.#####_##_#####.#     ",
    "     #.##__________##.#     ",
    "     #.##_###--###_##.#     ",
    "######.##_#HHHHHH#_##.######",
    "______.___#HHHHHH#___.______",
    "######.##_#HHHHHH#_##.######",
    "     #.##_########_##.#     ",
    "     #.##__________##.#     ",
    "     #.##_########_##.#     ",
    "######.##_########_##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#o..##.......__.......##..o#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
]

COLS = 28
ROWS = 31
TUNNEL_ROW = 14

# Direction vectors. Order matters: arcade ghost tie-break preference is UP, LEFT, DOWN, RIGHT.
UP = (0, -1)
LEFT = (-1, 0)
DOWN = (0, 1)
RIGHT = (1, 0)
DIRS = (UP, LEFT, DOWN, RIGHT)
DIR_NAMES = {UP: "UP", LEFT: "LEFT", DOWN: "DOWN", RIGHT: "RIGHT"}
NAME_TO_DIR = {v: k for k, v in DIR_NAMES.items()}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
ARROWS = {UP: "\u25b2", LEFT: "\u25c0", DOWN: "\u25bc", RIGHT: "\u25b6"}

PAC_START = (13.5, 23.0)
GHOST_DOOR_EXIT = (13.5, 11.0)   # tile just above the door, where ghosts emerge
GHOST_HOUSE_CENTER = (13.5, 14.0)
FRUIT_POS = (13.5, 17.0)

# Arcade "red zones": ghosts in scatter/chase may not choose UP on these tiles.
NO_UP_TILES = {(12, 11), (15, 11), (12, 23), (15, 23)}

assert len(LAYOUT) == ROWS and all(len(r) == COLS for r in LAYOUT), "maze layout must be 28x31"


def wrap_x(x: int) -> int:
    return x % COLS


def cell(x: int, y: int) -> str:
    if y < 0 or y >= ROWS:
        return " "
    return LAYOUT[y][wrap_x(x)]


def is_floor(x: int, y: int) -> bool:
    """Walkable by Pac-Man (and by ghosts outside the house)."""
    return cell(x, y) in "._o"


def is_tunnel(x: int, y: int) -> bool:
    return y == TUNNEL_ROW and (wrap_x(x) <= 5 or wrap_x(x) >= 22)


FLOOR_TILES: list[tuple[int, int]] = [
    (x, y) for y in range(ROWS) for x in range(COLS) if is_floor(x, y)
]
FLOOR_SET = set(FLOOR_TILES)


def step(tile: tuple[int, int], d: tuple[int, int]) -> tuple[int, int]:
    return (wrap_x(tile[0] + d[0]), tile[1] + d[1])


@lru_cache(maxsize=None)
def exits(tile: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Legal movement directions from a floor tile (UP, LEFT, DOWN, RIGHT order)."""
    return tuple(d for d in DIRS if step(tile, d) in FLOOR_SET)


def is_junction(tile: tuple[int, int]) -> bool:
    return len(exits(tile)) >= 3


JUNCTIONS = {t for t in FLOOR_TILES if is_junction(t)}


def initial_pellets() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    dots, energizers = set(), set()
    for (x, y) in FLOOR_TILES:
        c = LAYOUT[y][x]
        if c == ".":
            dots.add((x, y))
        elif c == "o":
            energizers.add((x, y))
    return dots, energizers


TOTAL_DOTS = len(initial_pellets()[0])
TOTAL_ENERGIZERS = len(initial_pellets()[1])


def follow_corridor(start: tuple[int, int], d: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """
    Walk from `start` in direction `d`, auto-cornering through bends, until reaching
    a junction (>=3 exits). Returns (path_tiles_excluding_start, end_tile, arrival_dir).
    """
    path: list[tuple[int, int]] = []
    cur = start
    cur_d = d
    for _ in range(COLS * ROWS):
        nxt = step(cur, cur_d)
        if nxt not in FLOOR_SET:
            break
        path.append(nxt)
        cur = nxt
        if cur in JUNCTIONS:
            break
        ex = [e for e in exits(cur) if e != OPPOSITE[cur_d]]
        if not ex:
            break
        if cur_d not in ex:
            cur_d = ex[0]
    return path, cur, cur_d


def bfs_distances(
    start: tuple[int, int],
    blocked_first_dir: tuple[int, int] | None = None,
    max_dist: int = 999,
) -> dict[tuple[int, int], int]:
    """Breadth-first tile distances over the floor graph (with tunnel wrap)."""
    dist = {start: 0}
    q = deque([start])
    while q:
        t = q.popleft()
        dt = dist[t]
        if dt >= max_dist:
            continue
        for d in exits(t):
            if t == start and blocked_first_dir is not None and d == blocked_first_dir:
                continue
            n = step(t, d)
            if n not in dist:
                dist[n] = dt + 1
                q.append(n)
    return dist


def toroidal_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = abs(a[0] - b[0])
    dx = min(dx, COLS - dx)
    dy = abs(a[1] - b[1])
    return (dx * dx + dy * dy) ** 0.5
