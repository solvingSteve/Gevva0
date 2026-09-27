"""Unit tests for statistical rigor and calibration metrics (gevva0.metrics)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from gevva0.metrics import (
    bootstrap_ci,
    calculate_permutation_invariance,
    calculate_sample_size_stats,
    compute_abstention_metrics,
    compute_accuracy,
    compute_brier_decomposition,
    compute_brier_score,
    compute_ece,
    compute_latency_percentiles,
    compute_macro_f1,
    mcnemar_test,
)


def test_sample_size_stats_wald():
    # Test N=20 at 80% accuracy -> Wald CI ~ +/- 17.5% ([62.5%, 97.5%])
    res_20 = calculate_sample_size_stats(n=20, p=0.80, alpha=0.05, target_diff=0.03)
    assert pytest.approx(res_20["standard_error"], rel=1e-2) == 0.0894
    assert pytest.approx(res_20["wald_margin_of_error"], abs=0.01) == 0.175
    assert pytest.approx(res_20["wald_ci_lower"], abs=0.01) == 0.625
    assert pytest.approx(res_20["wald_ci_upper"], abs=0.01) == 0.975

    # Test N=650 at p=0.85 -> min required N >= 540 for +/- 3% at 95% CI
    res_650 = calculate_sample_size_stats(n=650, p=0.85, alpha=0.05, target_diff=0.03)
    assert res_650["min_required_n"] >= 540
    assert res_650["wald_margin_of_error"] < 0.03  # Margin of error is strictly under 3%


def test_bootstrap_ci():
    y_true = np.array([0, 1, 2, 3] * 50)
    y_pred = np.array([0, 1, 2, 0] * 50)  # 75% accuracy

    res = bootstrap_ci(compute_accuracy, y_true, y_pred, n_resamples=1000, ci=0.95, seed=42)
    assert pytest.approx(res["point_estimate"], abs=1e-4) == 0.75
    assert res["ci_lower"] < 0.75 < res["ci_upper"]
    assert 0.65 < res["ci_lower"] < 0.75
    assert 0.75 < res["ci_upper"] < 0.85


def test_macro_f1():
    # Perfect predictions
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    assert compute_macro_f1(y_true, y_pred) == 1.0

    # Imbalanced error
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    f1 = compute_macro_f1(y_true, y_pred)
    assert 0.0 < f1 < 1.0


def test_compute_ece_binning():
    # Perfect calibration: confidence 1.0 on all correct items
    probs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    labels = np.array([0, 1, 2, 3])
    res = compute_ece(probs, labels, n_bins=10)
    assert pytest.approx(res["ece"], abs=1e-5) == 0.0
    assert len(res["bin_data"]) == 10

    # Miscalibrated: 100% confidence, but completely wrong
    miscal_probs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    miscal_labels = np.array([1, 0])
    miscal_ece = compute_ece(miscal_probs, miscal_labels, n_bins=10)
    assert pytest.approx(miscal_ece["ece"], abs=1e-4) == 1.0


def test_brier_decomposition_murphy():
    np.random.seed(42)
    m = 200
    k = 4
    labels = np.random.choice(k, size=m)
    probs = np.random.dirichlet(np.ones(k), size=m)

    decomp = compute_brier_decomposition(probs, labels, n_bins=10)
    assert "brier_score" in decomp
    assert "reliability" in decomp
    assert "resolution" in decomp
    assert "uncertainty" in decomp
    assert decomp["brier_score"] > 0
    assert decomp["uncertainty"] > 0

    # Verify Murphy reconstruction relation: Brier ~= Reliability - Resolution + Uncertainty + WithinBinResidual
    reconstructed = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"] + decomp["within_bin_residual"]
    assert pytest.approx(decomp["brier_score"], rel=1e-3) == reconstructed


def test_mcnemar_test():
    y_true = np.array([0] * 100)
    # Model A correct 90 times, Model B correct 70 times
    pred_a = np.array([0] * 90 + [1] * 10)
    pred_b = np.array([0] * 70 + [1] * 30)

    res = mcnemar_test(y_true, pred_a, pred_b)
    assert res["contingency"]["both_correct_a"] == 70
    assert res["contingency"]["a_correct_b_wrong_b"] == 20
    assert res["contingency"]["a_wrong_b_correct_c"] == 0
    assert res["contingency"]["both_wrong_d"] == 10
    assert res["statistic"] > 0
    assert res["p_value"] < 0.001
    assert res["significant"] is True

    # Identical predictions -> p = 1.0
    res_ident = mcnemar_test(y_true, pred_a, pred_a)
    assert res_ident["p_value"] == 1.0
    assert res_ident["significant"] is False


def test_permutation_invariance():
    # 3 samples, 4 cyclic shifts each
    shifts_perfect = [
        ["A", "A", "A", "A"],
        ["B", "B", "B", "B"],
        ["C", "C", "C", "C"],
    ]
    res_perf = calculate_permutation_invariance(shifts_perfect)
    assert res_perf["label_invariance_score_pct"] == 100.0

    shifts_flawed = [
        ["A", "A", "A", "A"],
        ["B", "A", "B", "B"],  # non-invariant
    ]
    res_flawed = calculate_permutation_invariance(shifts_flawed)
    assert res_flawed["label_invariance_score_pct"] == 50.0


def test_abstention_metrics():
    y_true = ["A", "B", "OOD", "OOD"]
    y_pred = ["A", "B", "OOD", "A"]  # 1 correct abstention, 1 false positive

    res = compute_abstention_metrics(y_true, y_pred, ood_labels={"OOD"})
    assert res["ood_count"] == 2
    assert res["ood_correct"] == 1
    assert res["ood_recall_pct"] == 50.0
    assert res["false_positive_activation_rate_pct"] == 50.0


def test_latency_percentiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    res = compute_latency_percentiles(data)
    assert pytest.approx(res["p50"], abs=1.0) == 55.0
    assert pytest.approx(res["mean"], abs=1.0) == 55.0
    assert res["p95"] > res["p90"] > res["p50"]


def test_kv_branching_parity():
    """Verify mathematical parity between clean-reset isolated cyclic debiasing
    and KV-cache branching cyclic debiasing:
    max_k |P_isolated(O_k) - P_branching(O_k)| < 0.005 (0.5%).
    """
    from gevva0.config import get_smallest_model
    from gevva0.engine import GemmaDecisionEngine

    smallest = get_smallest_model()
    if not smallest:
        pytest.skip("No model file available in models/")

    with GemmaDecisionEngine(model_path=smallest["full_path"], n_ctx=1024, verbose=False) as engine:
        context = (
            "Procurement Standard 4.1: Non-capital software purchase exceeding $1,500 requires formal VP signature. "
            "Addendum 9 (Cloud & Tooling Rider): Cloud developer tooling subscriptions approved under platform engineering "
            "budget are pre-authorized up to $10,000 and require only Engineering Manager sign-off. "
            "Scenario: Lead requests approval for an annual $4,200 developer profiling subscription fully allocated under platform budget."
        )
        options = {
            "A": "Formal VP signature and secondary compliance review",
            "B": "Immediate escalation to CFO and board approval",
            "C": "Engineering Manager approval only under Addendum 9",
            "D": "Automatic rejection as unauthorized capital expenditure",
        }

        # 1. Isolated clean-reset cyclic debiasing
        res_iso = engine.decide(
            context=context,
            options=options,
            cyclic_debias=True,
            kv_branching=False,
        )

        # 2. KV-cache branching cyclic debiasing (forcing min_tokens to 10 to ensure branching is triggered)
        res_bra = engine.decide(
            context=context,
            options=options,
            cyclic_debias=True,
            kv_branching=True,
            kv_branching_min_tokens=10,
        )

        probs_iso = np.array([res_iso.probabilities[options[k]] for k in sorted(options.keys())])
        probs_bra = np.array([res_bra.probabilities[options[k]] for k in sorted(options.keys())])

        max_div = float(np.max(np.abs(probs_iso - probs_bra)))
        assert max_div < 0.005, (
            f"KV state divergence {max_div:.5f} exceeded 0.005 (0.5%) - potential KV state leakage!"
        )
        assert res_iso.decision == res_bra.decision
