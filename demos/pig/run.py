"""
Gevva0 Game of Pig Showcase Runner: Tier 0 Optimal Stopping Engine (ONNX + Absorbing Markov Chain).

Usage:
    uv run python demos/pig/run.py              # Interactive 1v1 Play vs Gevva0 Tier 0 (GPU ONNX)
    uv run python demos/pig/run.py --sim        # Launch directly into High-Speed Simulation Arena
    uv run python demos/pig/run.py --simulate   # Neural stand-in mode (no ONNX/GPU required)
    uv run python demos/pig/run.py --fullscreen # Fullscreen presentation mode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pygame

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from brain import PigTier0Brain, SimulatedTier0Brain
from game import PigGame
from gto import get_gto_solver
from renderer import PigRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Gevva0 Tier 0 Game of Pig Showcase")
    parser.add_argument("--sim", action="store_true", help="Start directly in High-Speed Simulation mode")
    parser.add_argument("--simulate", action="store_true", help="Use simulated Tier 0 stand-in (no ONNX required)")
    parser.add_argument(
        "--engine-mode",
        type=str,
        default="hybrid_domain",
        choices=["hybrid_domain", "domain_markov", "default_39", "pure_cosine"],
        help="Initial Tier 0 internal scoring formulation",
    )
    parser.add_argument("--fps", type=int, default=60, help="Target framerate (default: 60)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    renderer = PigRenderer(fullscreen=args.fullscreen)

    print("\n=====================================================================")
    print("   GEVVA0 TIER 0 SHOWCASE: THE GAME OF PIG (OPTIMAL STOPPING MDP)    ")
    print("=====================================================================")
    print("Tier: Tier 0 ONLY (EmbeddingGemma-2 ONNX + Absorbing Markov Chain)")

    renderer.render_loading(
        "Solving Exact 3D Absorbing Markov Decision Process (1,000,000 states)...",
        detail="Todd & Fisher (2007) Optimal Stopping Surface",
    )
    pygame.event.pump()
    solver = get_gto_solver()
    print(f"[GTO Solver] Solved 100x100x100 Pig MDP in {solver.solve_time_ms:.1f} ms.")

    if args.simulate:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
        detail_desc = "Simulated Tier 0 Stand-In"
    else:
        brain = PigTier0Brain(engine_mode=args.engine_mode)
        detail_desc = "Loading EmbeddingGemma-2 ONNX + 104-Landmark Pig Domain Manifold..."

    renderer.render_loading(f"Initializing {brain.name}...", detail=detail_desc)
    pygame.event.pump()
    brain.load_async()

    clock = pygame.time.Clock()
    while not brain.ready and brain.loading:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                brain.shutdown()
                pygame.quit()
                return
        renderer.render_loading(f"Initializing {brain.name}...", detail=detail_desc)
        clock.tick(30)

    if not brain.ready and brain.error:
        print(f"[Notice] Falling back to SimulatedTier0Brain ({brain.error})")
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
        brain.load()

    game = PigGame(seed=args.seed)
    if args.sim:
        game.sim_mode = True
        game.sim_speed = 2

    # Run one initial evaluation so the telemetry panel is populated immediately at launch
    init_sit = brain.build_situation(0, 0, 0, roll_count=1)
    brain.decide(init_sit)

    running = True
    while running:
        dt = clock.tick(args.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_r):
                    if not game.sim_mode and game.turn == 0 and game.winner is None:
                        game.execute_roll(animate=True)
                    elif game.winner is not None:
                        game.new_game(alternate_starter=True)
                elif event.key in (pygame.K_h, pygame.K_RETURN):
                    if not game.sim_mode and game.turn == 0 and game.winner is None and game.turn_bank > 0:
                        game.execute_hold(animate=True)
                elif event.key == pygame.K_m:
                    game.sim_mode = not game.sim_mode
                elif event.key == pygame.K_s:
                    game.sim_speed = (game.sim_speed % 3) + 1
                elif event.key == pygame.K_o:
                    game.cycle_heuristic()
                elif event.key == pygame.K_t:
                    game.run_turbo_games(brain, num_games=50)
                elif event.key == pygame.K_n:
                    game.new_game(alternate_starter=False)
                elif event.key == pygame.K_1:
                    brain.set_engine_mode("hybrid_domain")
                elif event.key == pygame.K_2:
                    brain.set_engine_mode("domain_markov")
                elif event.key == pygame.K_3:
                    brain.set_engine_mode("default_39")
                elif event.key == pygame.K_4:
                    brain.set_engine_mode("pure_cosine")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = renderer.button_at(event.pos)
                if clicked == "roll":
                    if not game.sim_mode and game.turn == 0 and game.winner is None:
                        game.execute_roll(animate=True)
                elif clicked == "hold":
                    if not game.sim_mode and game.turn == 0 and game.winner is None and game.turn_bank > 0:
                        game.execute_hold(animate=True)
                elif clicked == "toggle_mode":
                    game.sim_mode = not game.sim_mode
                elif clicked == "cycle_speed":
                    game.sim_speed = (game.sim_speed % 3) + 1
                elif clicked == "cycle_opponent":
                    game.cycle_heuristic()
                elif clicked == "turbo_50":
                    game.run_turbo_games(brain, num_games=50)
                elif clicked == "new_game":
                    game.new_game(alternate_starter=False)
                elif clicked == "reset_stats":
                    game.stats.reset()
                    game.new_game(alternate_starter=False)
                elif clicked and clicked.startswith("mode_"):
                    m_target = clicked.replace("mode_", "")
                    brain.set_engine_mode(m_target)

        # Step AI / Simulation Loop
        if game.sim_mode:
            if game.sim_speed == 3:
                # Turbo mode: run 4 complete games per frame
                game.run_turbo_games(brain, num_games=4)
            elif game.sim_speed == 2:
                # Fast mode: 3 decisions per frame
                for _ in range(3):
                    if game.winner is not None:
                        game.new_game(alternate_starter=True)
                    game.step_ai_decision(brain, animate=False, use_turbo_cache=True)
            else:
                # 1x Animated Simulation
                if game.winner is not None:
                    game.ai_think_timer -= dt
                    if game.ai_think_timer <= -0.6:
                        game.new_game(alternate_starter=True)
                else:
                    game.ai_think_timer -= dt
                    if game.ai_think_timer <= 0.0:
                        game.step_ai_decision(brain, animate=True, use_turbo_cache=False)
        else:
            # 1v1 Play Mode: Gevva0 steps automatically when it's Player 1's turn
            if game.winner is None and game.turn == 1:
                game.ai_think_timer -= dt
                if game.ai_think_timer <= 0.0:
                    game.step_ai_decision(brain, animate=True, use_turbo_cache=False)

        renderer.render(game, brain, dt)

    brain.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
