"""Statistical and Calibration Metrics Suite for Gevva0 Benchmarking.

Provides rigorous sample size estimation, bootstrap confidence intervals,
paired McNemar tests, multiclass Brier score decomposition (Reliability,
Resolution, Uncertainty), Expected Calibration Error (ECE), permutation
invariance scoring, and abstention/OOD auditing.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy import stats


def calculate_sample_size_stats(
    n: int,
    p: float = 0.85,
    alpha: float = 0.05,
    target_diff: float = 0.03,
) -> Dict[str, float]:
    """Compute standard error, Wald confidence interval half-width, and minimum sample size.

    Args:
        n: Current evaluation sample size.
        p: Baseline accuracy or expected probability (default: 0.85).
        alpha: Significance level (default 0.05 for 95% confidence).
        target_diff: Target detectable difference / margin of error (default: 0.03 for +/-3%).

    Returns:
        Dict with SE, wald_ci_lower, wald_ci_upper, wald_margin_of_error, and min_required_n.
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    se = math.sqrt(p * (1.0 - p) / n)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    margin_of_error = z * se
    ci_lower = max(0.0, p - margin_of_error)
    ci_upper = min(1.0, p + margin_of_error)

    # Minimum sample size to state difference of +/- target_diff at (1 - alpha) confidence
    min_required_n = math.ceil(((z / target_diff) ** 2) * p * (1.0 - p))

    return {
        "n": float(n),
        "p": p,
        "standard_error": round(se, 6),
        "z_critical": round(z, 4),
        "wald_margin_of_error": round(margin_of_error, 4),
        "wald_ci_lower": round(ci_lower, 4),
        "wald_ci_upper": round(ci_upper, 4),
        "target_diff": target_diff,
        "min_required_n": int(min_required_n),
    }


def compute_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute standard classification accuracy (Top-1)."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def compute_macro_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
) -> float:
    """Compute unweighted Macro-F1 across all unique classes."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0

    if labels is None:
        unique_labels = np.unique(np.concatenate([y_t, y_p]))
    else:
        unique_labels = np.asarray(labels)

    if len(unique_labels) == 0:
        return 0.0

    f1_scores: List[float] = []
    for label in unique_labels:
        tp = int(np.sum((y_t == label) & (y_p == label)))
        fp = int(np.sum((y_t != label) & (y_p == label)))
        fn = int(np.sum((y_t == label) & (y_p != label)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence[Any],
    y_pred_or_probs: Sequence[Any],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Calculate 95% Bootstrap Confidence Intervals using empirical percentiles with replacement.

    Args:
        metric_fn: Function metric_fn(y_true_resampled, y_pred_resampled) -> float
        y_true: Ground truth labels.
        y_pred_or_probs: Predictions or probability distributions.
        n_resamples: Number of bootstrap iterations (default 10,000).
        ci: Confidence interval fraction (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, and margin.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred_or_probs)
    n = len(y_t)
    if n == 0:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin": 0.0}

    point_estimate = float(metric_fn(y_t, y_p))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))

    boot_scores = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = indices[b]
        boot_scores[b] = metric_fn(y_t[idx], y_p[idx])

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_scores, lower_pct))
    ci_upper = float(np.percentile(boot_scores, upper_pct))
    margin = (ci_upper - ci_lower) / 2.0

    return {
        "point_estimate": round(point_estimate, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "margin": round(margin, 6),
        "ci_level": ci,
    }


def compute_ece(
    probs: np.ndarray,
    labels: Union[np.ndarray, Sequence[int]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Partition model predictions into B equally spaced confidence bins and calculate ECE.

    Formula:
        ECE = sum_{b=1}^B (|B_b| / M) * |acc(B_b) - conf(B_b)|

    Args:
        probs: Array of prediction probabilities of shape (M, K) or 1D array of top-1 confidences.
        labels: Ground truth labels (1D class indices or one-hot (M, K)).
        n_bins: Number of bins (default: 10).

    Returns:
        Dict with ece, bin_data (list of per-bin statistics), and max_calibration_error.
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels)

    if probs_arr.ndim == 2:
        m, k = probs_arr.shape
        confidences = np.max(probs_arr, axis=1)
        predictions = np.argmax(probs_arr, axis=1)
        if labels_arr.ndim == 2:
            targets = np.argmax(labels_arr, axis=1)
        else:
            targets = labels_arr.astype(int)
        accuracies = (predictions == targets).astype(np.float64)
    else:
        # 1D confidences and binary correct indicator
        confidences = probs_arr
        accuracies = labels_arr.astype(np.float64)
        m = len(confidences)

    if m == 0:
        return {"ece": 0.0, "max_calibration_error": 0.0, "bin_data": []}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    mce_val = 0.0
    bin_data = []

    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            err = abs(bin_acc - bin_conf)
            weight = count / m
            ece_val += weight * err
            mce_val = max(mce_val, err)
        else:
            bin_acc = 0.0
            bin_conf = (lo + hi) / 2.0
            err = 0.0

        bin_data.append({
            "bin": b + 1,
            "range": [round(lo, 2), round(hi, 2)],
            "count": count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "error": round(err, 4),
        })

    return {
        "ece": round(ece_val, 6),
        "max_calibration_error": round(mce_val, 6),
        "n_bins": n_bins,
        "sample_count": m,
        "bin_data": bin_data,
    }


def compute_brier_score(
    probs: np.ndarray,
    labels: Union[np.ndarray, Sequence[int]],
) -> float:
    """Compute the multi-class Brier score:

        Brier = (1 / M) * sum_{m=1}^M sum_{k=1}^K (P_{m,k} - y_{m,k})^2
    """
    p_arr = np.asarray(probs, dtype=np.float64)
    l_arr = np.asarray(labels)

    if p_arr.ndim == 1:
        # Binary case or pre-extracted probabilities for positive class
        y_one_hot = l_arr.astype(np.float64)
        return float(np.mean((p_arr - y_one_hot) ** 2))

    m, k = p_arr.shape
    if l_arr.ndim == 1:
        y_one_hot = np.zeros((m, k), dtype=np.float64)
        y_one_hot[np.arange(m), l_arr.astype(int)] = 1.0
    else:
        y_one_hot = l_arr.astype(np.float64)

    return float(np.mean(np.sum((p_arr - y_one_hot) ** 2, axis=1)))


def compute_brier_decomposition(
    probs: np.ndarray,
    labels: Union[np.ndarray, Sequence[int]],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Decompose Multi-class Brier Score into Reliability, Resolution, and Uncertainty.

    Murphy / Yates canonical decomposition:
        Brier Score = Reliability - Resolution + Uncertainty + WithinBinVariance

    Components:
        - Reliability (Calibration Error): How close average forecast probabilities in
          each bin are to observed relative frequencies. Lower is better (0 = perfect calibration).
        - Resolution: Ability of the forecasting system to discriminate different outcome states
          away from the sample base rate (climatology). Higher is better.
        - Uncertainty: The inherent entropy / variance of the ground-truth outcomes:
          Uncertainty = sum_{k=1}^K y_bar_k * (1 - y_bar_k). Independent of the model.

    Returns:
        Dict with brier, reliability, resolution, uncertainty, and within_bin_var.
    """
    p_arr = np.asarray(probs, dtype=np.float64)
    l_arr = np.asarray(labels)

    if p_arr.ndim == 1:
        p_arr = np.column_stack([1.0 - p_arr, p_arr])
        if l_arr.ndim == 1:
            y_one_hot = np.column_stack([1.0 - l_arr, l_arr])
        else:
            y_one_hot = l_arr
    else:
        m, k = p_arr.shape
        if l_arr.ndim == 1:
            y_one_hot = np.zeros((m, k), dtype=np.float64)
            y_one_hot[np.arange(m), l_arr.astype(int)] = 1.0
        else:
            y_one_hot = l_arr.astype(np.float64)

    m, k = p_arr.shape
    brier = float(np.mean(np.sum((p_arr - y_one_hot) ** 2, axis=1)))

    # Overall base rate for each class (climatology)
    y_bar = np.mean(y_one_hot, axis=0)  # shape (k,)
    uncertainty = float(np.sum(y_bar * (1.0 - y_bar)))

    confidences = np.max(p_arr, axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    reliability = 0.0
    resolution = 0.0
    within_bin_residual = 0.0

    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        if b == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)

        count = int(np.sum(mask))
        if count == 0:
            continue

        weight = count / m
        p_bin_mean = np.mean(p_arr[mask], axis=0)  # (k,)
        y_bin_mean = np.mean(y_one_hot[mask], axis=0)  # (k,)

        reliability += weight * float(np.sum((p_bin_mean - y_bin_mean) ** 2))
        resolution += weight * float(np.sum((y_bin_mean - y_bar) ** 2))

        diff_p = p_arr[mask] - p_bin_mean
        diff_y = y_one_hot[mask] - y_bin_mean
        term_p = float(np.mean(np.sum(diff_p ** 2, axis=1)))
        term_cov = float(np.mean(np.sum(diff_p * diff_y, axis=1)))
        within_bin_residual += weight * (term_p - 2.0 * term_cov)

    return {
        "brier_score": round(brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "within_bin_residual": round(within_bin_residual, 6),
        "murphy_reconstructed": round(reliability - resolution + uncertainty + within_bin_residual, 6),
    }


def mcnemar_test(
    y_true: Sequence[Any],
    y_pred_a: Sequence[Any],
    y_pred_b: Sequence[Any],
) -> Dict[str, Any]:
    """Run McNemar's Paired Significance Test on the contingency matrix of paired predictions.

    Evaluates whether the performance difference between Model A and Model B on the exact
    same queries is statistically significant (p < 0.05).

    Contingency Matrix:
        a: Both A and B correct
        b: Model A correct, Model B incorrect
        c: Model A incorrect, Model B correct
        d: Both A and B incorrect

    With Edwards continuity correction:
        chi2 = (|b - c| - 1)^2 / (b + c)
    Or exact binomial test if discordant pairs (b + c) < 25.

    Returns:
        Dict with statistic, p_value, contingency table, and significant (bool).
    """
    y_t = np.asarray(y_true)
    a_pred = np.asarray(y_pred_a)
    b_pred = np.asarray(y_pred_b)

    if len(y_t) != len(a_pred) or len(y_t) != len(b_pred):
        raise ValueError("y_true, y_pred_a, and y_pred_b must have the exact same length.")

    correct_a = (a_pred == y_t)
    correct_b = (b_pred == y_t)

    table_a = int(np.sum(correct_a & correct_b))
    table_b = int(np.sum(correct_a & (~correct_b)))  # A correct, B wrong
    table_c = int(np.sum((~correct_a) & correct_b))  # A wrong, B correct
    table_d = int(np.sum((~correct_a) & (~correct_b)))

    discordant = table_b + table_c

    if discordant == 0:
        stat = 0.0
        p_val = 1.0
        method = "none"
    elif discordant < 25:
        # Exact two-sided binomial test for small discordant counts,
        # but report Edwards continuity-corrected chi2 as statistic
        chi2_stat = float(((max(0.0, abs(table_b - table_c) - 1.0)) ** 2) / discordant)
        binom_res = stats.binomtest(table_b, discordant, 0.5, alternative="two-sided")
        stat = chi2_stat
        p_val = float(binom_res.pvalue)
        method = "exact_binomial"
    else:
        # Edwards continuity-corrected Chi-Square
        stat = float(((max(0.0, abs(table_b - table_c) - 1.0)) ** 2) / discordant)
        p_val = float(stats.chi2.sf(stat, 1))
        method = "edwards_chi2"

    return {
        "statistic": round(stat, 4),
        "p_value": p_val,
        "significant": p_val < 0.05,
        "method": method,
        "contingency": {
            "both_correct_a": table_a,
            "a_correct_b_wrong_b": table_b,
            "a_wrong_b_correct_c": table_c,
            "both_wrong_d": table_d,
            "discordant_pairs": discordant,
        },
    }


def calculate_permutation_invariance(
    paired_shift_predictions: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    """Calculate cyclic permutation stability (Label Invariance Score).

    Args:
        paired_shift_predictions: List of M items, where each item is a list of K semantic
                                  choice predictions across cyclic shifts (A->B->C->D->A).

    Returns:
        Dict with label_invariance_score (% where all shifts agreed), shift_counts, and sample_count.
    """
    m = len(paired_shift_predictions)
    if m == 0:
        return {"label_invariance_score_pct": 0.0, "total_instances": 0, "invariant_instances": 0}

    invariant_count = 0
    shift_discrepancies: List[int] = []

    for shifts in paired_shift_predictions:
        unique_semantic_decisions = set(shifts)
        if len(unique_semantic_decisions) == 1:
            invariant_count += 1
        shift_discrepancies.append(len(unique_semantic_decisions))

    invariance_pct = (invariant_count / m) * 100.0

    return {
        "label_invariance_score_pct": round(invariance_pct, 2),
        "total_instances": m,
        "invariant_instances": invariant_count,
        "mean_distinct_choices_per_item": round(float(np.mean(shift_discrepancies)), 3),
    }


def compute_abstention_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    ood_labels: Union[Set[Any], Sequence[Any]],
) -> Dict[str, Any]:
    """Measure abstention recall and false-positive activation rates on out-of-scope prompts.

    Args:
        y_true: Ground truth decisions.
        y_pred: Predicted decisions.
        ood_labels: Set or list of labels indicating Out-of-Distribution / Non-matching / Escalate.

    Returns:
        Dict with ood_count, ood_correct, ood_recall_pct, and false_positive_activation_rate_pct.
    """
    ood_set = set(ood_labels)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    is_ood_truth = np.isin(y_t, list(ood_set))
    ood_count = int(np.sum(is_ood_truth))

    if ood_count == 0:
        return {
            "ood_count": 0,
            "ood_correct": 0,
            "ood_recall_pct": 0.0,
            "false_positive_activation_rate_pct": 0.0,
        }

    ood_correct = int(np.sum(is_ood_truth & np.isin(y_p, list(ood_set))))
    # False-positive activation: an OOD prompt that is incorrectly matched to an affirmative in-scope class
    false_positives = ood_count - ood_correct

    recall_pct = (ood_correct / ood_count) * 100.0
    fp_rate_pct = (false_positives / ood_count) * 100.0

    return {
        "ood_count": ood_count,
        "ood_correct": ood_correct,
        "ood_recall_pct": round(recall_pct, 2),
        "false_positive_activation_rate_pct": round(fp_rate_pct, 2),
    }


def compute_latency_percentiles(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Compute standard latency percentiles (p50, p90, p95, p99, mean)."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
    }
