"""
Automated Test Suite for Gevva0 Tier 0 Game of Blackjack & Tier0DecisionBridge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from brain import (
    BLACKJACK_CLASS_ANCHORS,
    BLACKJACK_DOMAIN_CORPUS,
    BlackjackTier0Brain,
    SimulatedTier0Brain,
)
from game import BlackjackGame
from gevva0.bridge import DecisionBridge, DecisionRequest, Tier0DecisionBridge
from gevva0.tier0 import Tier0Engine
from gto import get_blackjack_solver


def test_exact_blackjack_gto_solver_properties():
    solver = get_blackjack_solver()

    # 1. Dealer bust probability for upcard 6 is ~42.08% under S17 infinite-deck chain
    d_bust_6 = solver.dealer_probs[6][5]
    assert 0.415 <= d_bust_6 <= 0.428

    # 2. Dealer bust probability for upcard 10 is ~21.4%
    d_bust_10 = solver.dealer_probs[10][5]
    assert 0.205 <= d_bust_10 <= 0.225

    # 3. Hard 11 vs Dealer 6 -> Optimal is DOUBLE with high positive EV (> +0.60)
    ev_11_v_6 = solver.evaluate_hand(11, False, False, 0, 6, can_double=True, can_split=False)
    assert ev_11_v_6.optimal_action == "DOUBLE"
    assert ev_11_v_6.best_ev > 0.60

    # 4. Pair of 8s vs Dealer 10 -> Optimal is SPLIT
    ev_88_v_10 = solver.evaluate_hand(16, False, True, 8, 10, can_double=True, can_split=True)
    assert ev_88_v_10.optimal_action == "SPLIT"

    # 5. Pair of 10s vs Dealer 6 (neutral count) -> Optimal is STAND (never split 10s)
    ev_tt_v_6 = solver.evaluate_hand(20, False, True, 10, 6, can_double=True, can_split=True)
    assert ev_tt_v_6.optimal_action == "STAND"

    # 6. Soft 18 vs Dealer 6 -> DOUBLE; Soft 18 vs Dealer 10 -> HIT; Soft 18 vs Dealer 7 -> STAND
    assert solver.evaluate_hand(18, True, False, 0, 6, can_double=True).optimal_action == "DOUBLE"
    assert solver.evaluate_hand(18, True, False, 0, 10, can_double=True).optimal_action == "HIT"
    assert solver.evaluate_hand(18, True, False, 0, 7, can_double=True).optimal_action == "STAND"

    # 7. Illustrious 18 True Count Warp: Hard 16 vs Dealer 10 is HIT at TC=0.0, STAND at TC=+2.0
    assert solver.evaluate_hand(16, False, False, 0, 10, true_count=0.0).optimal_action == "HIT"
    assert solver.evaluate_hand(16, False, False, 0, 10, true_count=2.0).optimal_action == "STAND"


def test_tier0_decision_bridge_with_blackjack_manifold():
    M = len(BLACKJACK_DOMAIN_CORPUS)
    assert M >= 100
    assert set(BLACKJACK_CLASS_ANCHORS.keys()) == {"A", "B", "C", "D"}

    engine = Tier0Engine.__new__(Tier0Engine)
    engine.device = "cpu"
    engine.classes = []
    engine._dynamic_cache = {}
    engine._candidate_cache = {}
    engine.use_compile = False
    engine.supports_vision = False
    engine.supports_audio = False

    np.random.seed(11)
    landmarks_np = np.random.randn(M, 8).astype(np.float32)
    landmarks_np /= np.linalg.norm(landmarks_np, axis=1, keepdims=True)
    engine.M = M
    engine.landmarks_np = landmarks_np

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import torch

    engine.landmarks = torch.tensor(landmarks_np, dtype=torch.float32, device="cpu")
    sim_matrix = np.dot(landmarks_np, landmarks_np.T)
    rows, cols, data = [], [], []
    for i in range(M):
        sim_matrix[i, i] = -np.inf
        top_k = np.argsort(sim_matrix[i])[-4:]
        for idx in top_k:
            rows.append(i)
            cols.append(idx)
            data.append(0.15)
    engine.Q_raw = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
    engine.lu_factor = spla.splu(sp.eye(M, format="csc") - engine.Q_raw.tocsc())

    def mock_embed(texts, use_compiled=True):
        vecs = []
        for idx_t, t in enumerate(texts):
            if "double" in t.lower():
                v = landmarks_np[2].copy()
            elif "split" in t.lower():
                v = landmarks_np[3].copy()
            elif "stand" in t.lower():
                v = landmarks_np[1].copy()
            else:
                v = landmarks_np[0].copy()
            vecs.append(v)
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return torch.tensor(arr, dtype=torch.float32)

    engine.embed = mock_embed

    bridge = DecisionBridge.from_tier0(tier0_engine=engine, markov_weight=0.35)
    assert isinstance(bridge, Tier0DecisionBridge)

    req = DecisionRequest(
        state_text="DOUBLE DOWN 2x wager: powerhouse Hard 11 vs Dealer 6.",
        options=["HIT card", "STAND pat", "DOUBLE DOWN 2x", "SPLIT pair"],
        custom_labels=["A", "B", "C", "D"],
    )
    res = bridge.decide(req)
    assert res.tier == 0
    assert res.mode == "tier0_hybrid"
    assert res.label == "C"
    assert res.cosine_probs is not None
    assert res.markov_probs is not None
    assert res.scaffold_mode == "domain_manifold"


def test_blackjack_game_and_simulated_brain_match():
    brain = SimulatedTier0Brain(engine_mode="hybrid_domain", seed=123)
    brain.load()
    game = BlackjackGame(seed=123)
    game.run_turbo_hands(brain, num_hands=100)

    assert game.stats.hands_played == 100
    assert brain.total_decisions >= 100
    assert brain.gto_match_rate >= 95.0
    # Gevva0 Tier 0 should outperform the Timid Tourist ('Never Bust') heuristic
    assert game.stats.gevva_edge_over_challenger_units >= 0.0


def test_blackjack_renderer_headless_smoke(monkeypatch):
    import pygame
    from renderer import BlackjackRenderer

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    renderer = BlackjackRenderer(fullscreen=False)
    brain = SimulatedTier0Brain(engine_mode="hybrid_domain", seed=42)
    brain.load()
    game = BlackjackGame(seed=42)

    sit = brain.build_situation(11, False, False, 0, 6, can_double=True, can_split=False)
    brain.decide(sit)
    renderer.render_loading("Loading Tier 0...", detail="Testing Blackjack loading screen")
    renderer.render(game, brain, dt=0.016)

    game.sim_mode = True
    renderer.spawn_chips(300, 250, count=6)
    for mode in ("hybrid_domain", "domain_markov", "default_39", "pure_cosine"):
        brain.set_engine_mode(mode)
        game.step_simulation(brain, animate=True)
        renderer.render(game, brain, dt=0.016)

    assert len(renderer.buttons) >= 9
    pygame.quit()
