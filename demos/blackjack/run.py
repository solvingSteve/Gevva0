"""
Gevva0 Game of Blackjack Showcase Runner: Tier 0 Optimal Stopping Engine (ONNX + Absorbing Markov Chain).

Usage:
    uv run python demos/blackjack/run.py              # Interactive Play vs Gevva0 Tier 0 (GPU ONNX)
    uv run python demos/blackjack/run.py --sim        # Launch directly into High-Speed Simulation Arena
    uv run python demos/blackjack/run.py --simulate   # Neural stand-in mode (no ONNX/GPU required)
    uv run python demos/blackjack/run.py --fullscreen # Fullscreen presentation mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from brain import BlackjackTier0Brain, SimulatedTier0Brain
from game import BlackjackGame
from gto import get_blackjack_solver
from renderer import BlackjackRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Gevva0 Tier 0 Game of Blackjack Showcase")
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

    renderer = BlackjackRenderer(fullscreen=args.fullscreen)

    print("\n=====================================================================")
    print("  GEVVA0 TIER 0 SHOWCASE: VEGAS BLACKJACK 21 (ABSORBING MARKOV MDP)  ")
    print("=====================================================================")
    print("Tier: Tier 0 ONLY (EmbeddingGemma-2 ONNX + Absorbing Markov Chain)")

    renderer.render_loading(
        "Solving Exact Dealer Absorbing Markov Chains & Player EV Surfaces...",
        detail="Thorp (1962) / Baldwin (1956) S17 + Hi-Lo Illustrious 18",
    )
    pygame.event.pump()
    solver = get_blackjack_solver()
    print(f"[GTO Solver] Solved Exact Blackjack Absorbing Chains & EV Tables in {solver.solve_time_ms:.1f} ms.")

    if args.simulate:
        brain = SimulatedTier0Brain(engine_mode=args.engine_mode, seed=args.seed)
        detail_desc = "Simulated Tier 0 Stand-In"
    else:
        brain = BlackjackTier0Brain(engine_mode=args.engine_mode)
        detail_desc = "Loading EmbeddingGemma-2 ONNX + 116-Landmark Blackjack Manifold..."

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

    game = BlackjackGame(seed=args.seed)
    if args.sim:
        game.sim_mode = True
        game.sim_speed = 2

    # Populate telemetry panel immediately at launch with active hand evaluation
    h0 = game.active_gevva_hand
    init_sit = brain.build_situation(
        player_total=h0.total,
        is_soft=h0.is_soft,
        is_pair=h0.is_pair,
        pair_rank=h0.pair_rank,
        dealer_upcard=game.dealer_upcard_value,
        can_double=h0.can_double,
        can_split=h0.is_pair,
        true_count=game.true_count,
        cards_str=h0.format_cards(),
    )
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
                elif event.key in (pygame.K_SPACE, pygame.K_h):
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN":
                        game.execute_challenger_action("HIT", animate=True)
                    elif game.phase == "ROUND_OVER":
                        game.deal_new_round(animate=True)
                elif event.key in (pygame.K_RETURN, pygame.K_k):
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN":
                        game.execute_challenger_action("STAND", animate=True)
                    elif game.phase == "ROUND_OVER":
                        game.deal_new_round(animate=True)
                elif event.key == pygame.K_d:
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN" and game.active_challenger_hand.can_double:
                        game.execute_challenger_action("DOUBLE", animate=True)
                elif event.key == pygame.K_p:
                    if (
                        not game.sim_mode
                        and game.phase == "CHALLENGER_TURN"
                        and game.active_challenger_hand.is_pair
                        and len(game.challenger_hands) == 1
                    ):
                        game.execute_challenger_action("SPLIT", animate=True)
                elif event.key == pygame.K_m:
                    game.sim_mode = not game.sim_mode
                elif event.key == pygame.K_s:
                    game.sim_speed = (game.sim_speed % 3) + 1
                elif event.key == pygame.K_o:
                    game.cycle_heuristic()
                elif event.key == pygame.K_t:
                    game.run_turbo_hands(brain, num_hands=50)
                elif event.key == pygame.K_n:
                    game.deal_new_round(animate=True)
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
                if clicked == "hit":
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN":
                        game.execute_challenger_action("HIT", animate=True)
                elif clicked == "stand":
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN":
                        game.execute_challenger_action("STAND", animate=True)
                elif clicked == "double":
                    if not game.sim_mode and game.phase == "CHALLENGER_TURN" and game.active_challenger_hand.can_double:
                        game.execute_challenger_action("DOUBLE", animate=True)
                elif clicked == "split":
                    if (
                        not game.sim_mode
                        and game.phase == "CHALLENGER_TURN"
                        and game.active_challenger_hand.is_pair
                        and len(game.challenger_hands) == 1
                    ):
                        game.execute_challenger_action("SPLIT", animate=True)
                elif clicked == "toggle_mode":
                    game.sim_mode = not game.sim_mode
                elif clicked == "cycle_speed":
                    game.sim_speed = (game.sim_speed % 3) + 1
                elif clicked == "cycle_opponent":
                    game.cycle_heuristic()
                elif clicked == "turbo_50":
                    game.run_turbo_hands(brain, num_hands=50)
                elif clicked == "deal_hand":
                    game.deal_new_round(animate=True)
                elif clicked == "reset_stats":
                    game.stats.reset()
                    game.deal_new_round(animate=True)
                elif clicked and clicked.startswith("mode_"):
                    m_target = clicked.replace("mode_", "")
                    brain.set_engine_mode(m_target)

        # Step AI / Simulation Loop
        if game.sim_mode:
            if game.sim_speed == 3:
                # Turbo mode: simulate 6 complete hands per frame
                game.run_turbo_hands(brain, num_hands=6)
            elif game.sim_speed == 2:
                # Fast mode: 3 actions per frame
                for _ in range(3):
                    game.step_simulation(brain, animate=False, use_turbo_cache=True)
            else:
                # 1x Animated Simulation
                game.ai_think_timer -= dt
                if game.ai_think_timer <= 0.0:
                    game.step_simulation(brain, animate=True, use_turbo_cache=False)
        else:
            # 1v1 Play Mode: Gevva0 steps automatically when it's Gevva0's turn
            if game.phase == "GEVVA_TURN":
                game.ai_think_timer -= dt
                if game.ai_think_timer <= 0.0:
                    game.step_gevva_decision(brain, animate=True, use_turbo_cache=False)

        renderer.render(game, brain, dt)

    brain.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
