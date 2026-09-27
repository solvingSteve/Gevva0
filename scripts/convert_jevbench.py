"""JevBench to Gevva0 Standardized Benchmark Suite Converter.

Converts JevBench canonical task records (easy.jsonl, hard.jsonl, original.jsonl)
into Gevva0 benchmark suite JSON format (benchmarks/benchmark_suite.json) for
execution via the Gevva0 Web UI dashboard ("Test All") or run_benchmark.py.

Includes backup, restore, tier filtering, and ground truth letter mapping.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize_jevbench_task(task_dict: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Convert a canonical JevBench record into a Gevva0 benchmark test."""
    task_id = task_dict.get("id") or f"jevbench_test_{idx:03d}"
    family = task_dict.get("family", "decision")
    provenance = task_dict.get("provenance") or {}

    title = f"[{family}] {task_id}"
    stressor = (
        provenance.get("why_hard")
        or provenance.get("rationale")
        or provenance.get("label_basis")
        or f"JevBench {family} evaluation"
    )

    question_spec = task_dict.get("question") or {}
    qtype = question_spec.get("type", "choice")
    instructions = question_spec.get("instructions", "")
    crit = question_spec.get("criteria")

    # State -> context
    state = task_dict.get("state", "")
    if isinstance(state, dict):
        context = json.dumps(state, indent=2, ensure_ascii=False)
    else:
        context = str(state)

    # Labels -> options
    raw_labels = task_dict.get("labels") or []
    if not raw_labels:
        return None

    labels = [str(x) for x in raw_labels]
    n = len(labels)
    if n > len(LETTERS):
        labels = labels[: len(LETTERS)]
        n = len(labels)

    options: Dict[str, str] = {}
    for i, lab in enumerate(labels):
        letter = LETTERS[i]
        if qtype == "score" and isinstance(crit, (list, tuple)):
            level_idx = int(lab) if lab.isdigit() else i
            desc = crit[level_idx] if level_idx < len(crit) else lab
            options[letter] = f"Level {lab}: {desc}"
        elif isinstance(crit, dict):
            desc_key = {"yes": "true", "no": "false"}.get(lab, lab) if qtype == "noul" else lab
            desc = crit.get(desc_key) or crit.get(lab)
            if desc and str(desc).strip() != lab.strip():
                options[letter] = f"{lab}: {str(desc).strip()}"
            else:
                options[letter] = lab
        else:
            options[letter] = lab

    # Expected ground truth -> Letter (A, B, C, D...)
    expected_raw = task_dict.get("expected")
    expected_letter = ""
    if expected_raw is not None:
        expected_str = str(expected_raw)
        if expected_str in labels:
            expected_letter = LETTERS[labels.index(expected_str)]
        elif qtype == "noul":
            norm_exp = "yes" if expected_str in ("yes", "true", "1") else "no"
            if norm_exp in labels:
                expected_letter = LETTERS[labels.index(norm_exp)]
        elif qtype == "score":
            try:
                score_idx = int(expected_str)
                if 0 <= score_idx < n:
                    expected_letter = LETTERS[score_idx]
            except ValueError:
                pass

    return {
        "test_id": task_id,
        "title": title,
        "stressor": stressor,
        "primitive": qtype,
        "context": context,
        "question": instructions,
        "options": options,
        "expected_ground_truth": expected_letter,
        "family": family,
        "split": task_dict.get("split", "public"),
    }


def load_tasks_from_file(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_suite_from_datasets(
    dataset_files: List[Path],
    output_file: Path,
    suite_title: str = "JevBench Standardized Decision Battery",
    backup: bool = True,
    limit: Optional[int] = None,
) -> int:
    raw_tasks = []
    for p in dataset_files:
        if p.is_file():
            raw_tasks.extend(load_tasks_from_file(p))

    if limit and limit > 0:
        raw_tasks = raw_tasks[:limit]

    converted_tests = []
    for idx, item in enumerate(raw_tasks, 1):
        norm = normalize_jevbench_task(item, idx)
        if norm:
            converted_tests.append(norm)

    if backup and output_file.is_file():
        backup_path = output_file.with_name(output_file.stem + ".backup" + output_file.suffix)
        shutil.copy2(output_file, backup_path)
        print(f"[BACKUP] Backed up existing suite to: {backup_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    suite_json = {
        "benchmark_suite": suite_title,
        "separator_delimiter": ",",
        "tests": converted_tests,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(suite_json, f, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully exported {len(converted_tests)} standardized tests to: {output_file}")
    return len(converted_tests)


def restore_backup(target_file: Path) -> bool:
    backup_path = target_file.with_name(target_file.stem + ".backup" + target_file.suffix)
    if not backup_path.is_file():
        print(f"[ERROR] Backup file not found: {backup_path}")
        return False
    shutil.copy2(backup_path, target_file)
    print(f"[OK] Restored {target_file} from {backup_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert JevBench datasets to Gevva0 benchmark suite format")
    parser.add_argument(
        "--tier",
        choices=["easy", "hard", "original", "all"],
        default="all",
        help="Which public dataset tier to convert (default: all)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Custom path to .jsonl file or dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/benchmark_suite.json",
        help="Output destination path (default: benchmarks/benchmark_suite.json)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .backup.json file before overwriting",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore original benchmark suite from .backup.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of converted tests",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "jevbench" else Path(__file__).resolve().parent
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    if args.restore:
        success = restore_backup(out_path)
        raise SystemExit(0 if success else 1)

    datasets_dir = project_root / "jevbench" / "datasets" / "public"
    files_to_convert: List[Path] = []

    if args.input:
        in_p = Path(args.input)
        if in_p.is_file():
            files_to_convert = [in_p]
        elif in_p.is_dir():
            files_to_convert = sorted(in_p.glob("*.jsonl"))
        else:
            print(f"❌ Input path not found: {in_p}")
            raise SystemExit(1)
    else:
        if args.tier in ("easy", "all"):
            files_to_convert.append(datasets_dir / "easy.jsonl")
        if args.tier in ("original", "all"):
            files_to_convert.append(datasets_dir / "original.jsonl")
        if args.tier in ("hard", "all"):
            files_to_convert.append(datasets_dir / "hard.jsonl")

    title = f"JevBench {args.tier.capitalize()} Public Battery" if args.tier != "all" else "JevBench Public Decision Battery (Easy + Original + Hard)"

    build_suite_from_datasets(
        dataset_files=files_to_convert,
        output_file=out_path,
        suite_title=title,
        backup=not args.no_backup,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()