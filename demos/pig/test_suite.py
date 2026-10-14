"""
Automated Test Suite for Gevva0 Tier 0 Game of Pig & Tier0DecisionBridge.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from brain import PIG_CLASS_ANCHORS, PIG_DOMAIN_CORPUS, PigTier0Brain, SimulatedTier0Brain
from game import PigGame
from gevva0.bridge import DecisionBridge, DecisionRequest, Tier0DecisionBridge
from gevva0.tier0 import Tier0Engine
from gto import PigGTOSolver, get_gto_solver


def test_exact_mdp_solver_properties():
    solver = get_gto_solver()
    assert solver.p_win.shape == (100, 100, 100)

    # First-player advantage at (0, 0, 0) in standard Pig is ~53.06% (Todd & Fisher 2007)
    p_start = solver.p_win[0, 0, 0]
    assert 0.525 <= p_start <= 0.536

    # At (0, 0), optimal first-turn hold threshold is around 21..22
    th_00 = solver.hold_thresholds[0, 0]
    assert 19 <= th_00 <= 23

    # When trailing badly against an opponent at 88, GTO warps threshold aggressively (> 30)
    th_trailing = solver.hold_thresholds[40, 88]
    assert th_trailing > 30

    # When leading 82 to 25, GTO warps threshold conservatively (< 18) to choke out opponent
    th_leading = solver.hold_thresholds[82, 25]
    assert th_leading <= 18


def test_tier0_decision_bridge_with_mock_engine():
    engine = Tier0Engine.__new__(Tier0Engine)
    engine.device = "cpu"
    engine.classes = []
    engine._dynamic_cache = {}
    engine._candidate_cache = {}
    engine.use_compile = False
    engine.supports_vision = False
    engine.supports_audio = False

    M = len(PIG_DOMAIN_CORPUS)
    assert M >= 100
    np.random.seed(7)
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

    def mock_embed(texts, prefix=""):
        vecs = []
        for t in texts:
            if "roll" in t.lower() and "hold immediately" not in t.lower():
                v = landmarks_np[0].copy()
            else:
                v = landmarks_np[1].copy()
            vecs.append(v)
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return torch.tensor(arr, dtype=torch.float32)

    engine.embed = mock_embed

    bridge = DecisionBridge.from_tier0(tier0_engine=engine, markov_weight=0.35)
    assert isinstance(bridge, Tier0DecisionBridge)

    req = DecisionRequest(
        state_text="Small unbanked turn total k=6; continue rolling to accumulate points.",
        options=["ROLL die again", "HOLDimmediately and bank"],
        custom_labels=["A", "B"],
    )
    res = bridge.decide(req)
    assert res.tier == 0
    assert res.mode == "tier0_hybrid"
    assert res.label in ("A", "B")
    assert res.cosine_probs is not None
    assert res.markov_probs is not None
    assert res.scaffold_mode == "domain_manifold"


def test_pig_game_and_simulated_brain_match():
    brain = SimulatedTier0Brain(engine_mode="hybrid_domain", seed=123)
    brain.load()
    game = PigGame(seed=123)
    game.run_turbo_games(brain, num_games=40)

    assert game.stats.games_played == 40
    assert game.stats.gevva_wins + game.stats.challenger_wins == 40
    assert brain.total_decisions > 40
    assert brain.gto_match_rate >= 95.0


def test_pig_renderer_headless_smoke(monkeypatch):
    import os
    import pygame
    from renderer import PigRenderer

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    renderer = PigRenderer(fullscreen=False)
    brain = SimulatedTier0Brain(engine_mode="hybrid_domain", seed=42)
    brain.load()
    game = PigGame(seed=42)

    sit = brain.build_situation(24, 62, 18, roll_count=3)
    brain.decide(sit)
    renderer.render_loading("Loading Tier 0...", detail="Testing loading screen")
    renderer.render(game, brain, dt=0.016)

    # Test particles and simulation mode rendering across all 4 engine modes
    game.sim_mode = True
    renderer.spawn_coins(200, 200, count=5)
    renderer.spawn_pig_out_burst(300, 300, count=5)
    for mode in ("hybrid_domain", "domain_markov", "default_39", "pure_cosine"):
        brain.set_engine_mode(mode)
        game.step_ai_decision(brain, animate=True)
        renderer.render(game, brain, dt=0.016)

    assert len(renderer.buttons) >= 8
    pygame.quit()

