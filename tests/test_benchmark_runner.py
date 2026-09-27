"""Integration tests for benchmark runner and report generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from benchmarks.run_benchmark import (
    BenchmarkSimulator,
    generate_markdown_report,
    load_suite,
    run_rigorous_evaluation,
)


def test_load_suite(tmp_path: Path):
    suite_file = tmp_path / "test_suite.json"
    data = {
        "tests": [
            {"test_id": "t1", "expected_ground_truth": "A", "options": {"A": "Opt A", "B": "Opt B"}},
            {"test_id": "t2", "expected_ground_truth": "B", "options": {"A": "Opt A", "B": "Opt B"}},
        ]
    }
    suite_file.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_suite(suite_file)
    assert len(loaded) == 2
    assert loaded[0]["test_id"] == "t1"


def test_benchmark_simulator():
    sim = BenchmarkSimulator(seed=123)
    test = {
        "test_id": "t1",
        "expected_ground_truth": "B",
        "is_ood": False,
        "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
    }
    res_base = sim.evaluate_test(test, "baseline_naive")
    assert "decision" in res_base
    assert "confidence" in res_base
    assert len(res_base["probabilities"]) == 4

    res_fast = sim.evaluate_test(test, "gevva0_fast_path")
    assert res_fast["perm_invariant"] is True

    res_cot = sim.evaluate_test(test, "gevva0_adaptive_cot")
    assert res_cot["perm_invariant"] is True


def test_rigorous_evaluation_and_report_generation(tmp_path: Path):
    tests = [
        {
            "test_id": f"t_{i}",
            "expected_ground_truth": ["A", "B", "C", "D"][i % 4],
            "is_ood": (i % 5 == 0),
            "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
        }
        for i in range(40)
    ]

    results = run_rigorous_evaluation(tests, n_bootstrap=200, seed=42)
    assert results["n_samples"] == 40
    assert "gevva0_fast_path" in results["condition_metrics"]
    assert "gevva0_adaptive_cot" in results["condition_metrics"]
    assert results["mcnemar_vs_baseline"]["gevva0_fast_path"] is not None

    out_md = tmp_path / "REPORT_TEST.md"
    report_text = generate_markdown_report(results, output_path=out_md)

    assert out_md.is_file()
    assert "# Standardized Decision Engine Benchmark Report" in report_text
    assert "Evaluation Summary" in report_text
    assert "Top-1 Accuracy" in report_text
    assert "McNemar" in report_text
