"""
Arcade Pac-Man Showcase Runner: Gevva0 Calibrated Decision Gateway on Gemma 4 26B MoE.

Usage:
    uv run python demos/pacman/run.py             # Full real-time Gevva0 Decision Gateway (GPU)
    uv run python demos/pacman/run.py --simulate  # Neural simulation mode (no GPU required)
    uv run python demos/pacman/run.py --fps 60    # Locked 60 FPS arcade loop
    uv run python demos/pacman/run.py --fullscreen # Fullscreen cabinet presentation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pygame

# Add project root and demo directory to sys.path
_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_DEMO_DIR))

from game import Game, FRAME
from pilot import Pilot
from renderer import PacmanRenderer


def main():
    parser = argparse.ArgumentParser(description="Gevva0 Arcade Pac-Man Showcase")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated neural gateway (no GPU required)")
    parser.add_argument("--fps", type=int, default=60, help="Target display framerate (default: 60)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--cot", action="store_true", default=True, help="Enable bounded CoT verification on low confidence")
    parser.add_argument("--cot-threshold", type=float, default=None, help="Override CoT gate threshold (default from profile)")
    parser.add_argument("--temperature", type=float, default=None, help="Override Platt temperature T (default from profile)")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for ghost frightened behaviors")
    parser.add_argument("--speed", type=float, default=1.0, help="Game speed multiplier")
    args = parser.parse_args()

    # 1. Initialize Renderer
    renderer = PacmanRenderer(fullscreen=args.fullscreen)

    print("\n=======================================================")
    print("   GEVVA0 CALIBRATED DECISION GATEWAY: PAC-MAN V2      ")
    print("=======================================================")
    print(f"Mode: {'SIMULATION' if args.simulate else 'GPU REAL INFERENCE'}")

    # 2. Initialize Decision Brain & Load Model FIRST before gameplay
    if args.simulate:
        from brain import SimulatedBrain
        brain = SimulatedBrain(latency_ms=(22.0, 34.0), seed=args.seed)
        model_desc = "Neural Simulation Mode (Heuristic Stand-in)"
    else:
        from brain import Gevva0Brain
        brain = Gevva0Brain(
            cot=args.cot,
            cot_threshold=args.cot_threshold,
            temperature=args.temperature,
        )
        model_desc = f"{brain.model_name} on CUDA0"

    renderer.render_loading(
        f"Loading Decision Model: {brain.name}...",
        detail=model_desc
    )
    pygame.event.pump()

    # Start asynchronous background loader
    brain.load_async()

    # Animated loading screen while model weights & KV prefix sequence 0 are loaded
    clock = pygame.time.Clock()
    while not brain.ready:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                brain.shutdown()
                pygame.quit()
                return

        detail_txt = (
            f"Pinning Rulebook Prefix into KV Sequence 0..."
            if not getattr(brain, "error", None)
            else f"Error: {brain.error}"
        )
        renderer.render_loading(
            f"Loading Decision Model: {brain.name}...",
            detail=detail_txt
        )
        clock.tick(30)

    print("[Gevva0 Pac-Man] Decision model fully loaded and ready! Starting game...\n")
    print("Press SPACE to pause, R to reset, ESC to exit.\n")

    # 3. Initialize Game Core & Pilot (Only after model is loaded and ready)
    game = Game(seed=args.seed, speed_scale=args.speed)
    pilot = Pilot(game, brain, sync=False)

    paused = False
    running = True

    # Main Arcade Loop (Locked 60 FPS)
    while running:
        dt = clock.tick(args.fps) / 1000.0

        # Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    game = Game(seed=int(time.time()), speed_scale=args.speed)
                    pilot = Pilot(game, brain, sync=False)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    game.set_pac_direction((0, -1))
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    game.set_pac_direction((-1, 0))
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    game.set_pac_direction((0, 1))
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    game.set_pac_direction((1, 0))

        if not paused:
            # Step pilot: checks waypoint arrivals, polls decisions, and dispatches upcoming junctions
            if brain.ready:
                pilot.tick()

            # Step continuous physics
            game.update(dt)

            # Check game events for visual particle sparks
            if game.events:
                latest = game.events[-1]
                if latest.kind == "ghost_eaten" and latest.t >= game.time - dt * 2:
                    px, py = renderer.MAZE_ORIGIN_X + game.pac.x * renderer.TILE_SIZE, renderer.MAZE_ORIGIN_Y + game.pac.y * renderer.TILE_SIZE
                    renderer.emit_sparkles(px, py, (40, 240, 120), count=15)
                    renderer.trigger_flash((40, 240, 120), max_alpha=20.0)
                elif latest.kind == "death" and latest.t >= game.time - dt * 2:
                    renderer.trigger_flash((255, 40, 40), max_alpha=35.0)

        # Render display
        renderer.render(game, pilot.situation, pilot, brain, dt)

    brain.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
