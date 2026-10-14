"""Generate a deterministic, domain-stratified, mixed-cardinality calibration slice
from Decision Index battery for fast Platt temperature & Brier score minimization.

Composition (125 items):
- 50 rows of 10-option MMLU-Pro questions (stratified across 14 primary domains).
- 50 rows of 4-option When2Call MCQ questions (including 17 out-of-distribution /
  cannot_answer scenarios to preserve flat, low-confidence distributions).
- 25 rows of 2-option binary verification items (BBH boolean_expressions,
  causal_judgement, formal_fallacies, web_of_lies).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATTERY_PATH = PROJECT_ROOT / "benchmarks" / "decision_index_battery.jsonl"
BBH_BOOL_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "decision-index"
    / "work"
    / "artifacts"
    / "benchmark-suite"
    / "raw"
    / "repos"
    / "cand-BIG-Bench-Hard"
    / "bbh"
    / "boolean_expressions.json"
)
OUTPUT_JSON = PROJECT_ROOT / "benchmarks" / "calibration_slice_100.json"
OUTPUT_JSONL = PROJECT_ROOT / "benchmarks" / "calibration_slice_100.jsonl"


def create_calibration_slice(
    battery_path: Path = BATTERY_PATH,
    output_json: Path = OUTPUT_JSON,
    output_jsonl: Path | None = OUTPUT_JSONL,
) -> List[Dict[str, Any]]:
    if not battery_path.is_file():
        raise FileNotFoundError(f"Battery file not found: {battery_path}")

    with open(battery_path, "r", encoding="utf-8") as f:
        battery_rows = [json.loads(line) for line in f if line.strip()]

    # 1. 50 MMLU-Pro questions (stratified across academic and reasoning domains)
    mmlu_rows = [r for r in battery_rows if r.get("benchmark") == "MMLU-Pro"]

    # 2. 50 When2Call questions (including 17 OOD cannot_answer scenarios)
    when2call_rows = [r for r in battery_rows if r.get("benchmark") == "When2Call MCQ"]

    # 3. 2-option binary verification items
    k2_rows = [
        r
        for r in battery_rows
        if len(r.get("questions", {}).get("q", {}).get("criteria", {})) == 2
    ]

    # Supplement K=2 rows to reach exactly 25 items if needed
    if len(k2_rows) < 25 and BBH_BOOL_PATH.is_file():
        with open(BBH_BOOL_PATH, "r", encoding="utf-8") as f:
            bbh_data = json.load(f)
        for idx, ex in enumerate(bbh_data.get("examples", [])):
            row_id = f"58:boolean_expressions:extra_{idx}"
            if any(r["id"] == f"58:boolean_expressions:{idx}" for r in k2_rows):
                continue
            k2_rows.append({
                "id": row_id,
                "benchmark": "BBH fixed-option tasks",
                "family": "BBH fixed-option tasks",
                "split": "evaluation",
                "state": ex["input"],
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Which option is the correct answer?",
                        "criteria": {"True": "True", "False": "False"},
                    }
                },
                "expected": {"q": ex["target"]},
                "metadata": {
                    "source_id": f"boolean_expressions:extra_{idx}",
                    "group_id": row_id,
                    "track": "boolean_expressions",
                    "task_family": "boolean_expressions",
                },
                "scoring": {"type": "candidate_decisions", "primary_field": "q"},
            })
            if len(k2_rows) >= 25:
                break

    # Combine into deterministic calibration slice
    calibration_set = mmlu_rows + when2call_rows + k2_rows[:25]

    # Save as JSON array
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(calibration_set, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Save as JSONL
    if output_jsonl:
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in calibration_set:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"Generated calibration slice with {len(calibration_set)} samples:\n"
        f"  • MMLU-Pro (high cardinality / domain stratified): {len(mmlu_rows)}\n"
        f"  • When2Call (4-option, including OOD cannot_answer): {len(when2call_rows)}\n"
        f"  • BBH Binary (2-option verification): {len(k2_rows[:25])}\n"
        f"Saved to {output_json} and {output_jsonl}"
    )
    return calibration_set


if __name__ == "__main__":
    create_calibration_slice()
