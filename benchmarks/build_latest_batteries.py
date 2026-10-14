"""Build the latest Decision Index 0.3 and JevBench v1.6.1/v1.4.2.2 benchmark batteries.

Generates:
1. `benchmarks/decision_index_v03_battery.jsonl` (N=350 across all 5 official Decision Index 0.3 areas)
2. `benchmarks/jevbench_battery.jsonl` (N=240 across JevBench Easy, Standard, Hard, and Vision)
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DI_ROOT = ROOT / "benchmarks" / "decision-index"
JB_ROOT = ROOT / "benchmarks" / "jevbench"

if str(DI_ROOT) not in sys.path:
    sys.path.insert(0, str(DI_ROOT))


def build_decision_index_v03_battery(per_benchmark: int = 50, seed: int = 42) -> Path:
    rng = random.Random(seed)
    added_gz = DI_ROOT / "work" / "artifacts" / "benchmark-suite" / "release-v2-rebuilt" / "added-rows.jsonl.gz"
    test56_path = DI_ROOT / "work" / "artifacts" / "benchmark-suite" / "release-v2-rebuilt" / "test-56.jsonl"

    name_to_id = {
        "MMLU-Pro": 57,
        "BBH fixed-option tasks": 58,
        "When2Call MCQ": 62,
        "New Yorker caption matching": 64,
    }
    by_bench: dict[int, list[dict]] = {57: [], 58: [], 62: [], 64: []}
    with gzip.open(added_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            b_raw = row.get("benchmark")
            b_id = b_raw if isinstance(b_raw, int) else name_to_id.get(str(b_raw))
            if b_id in by_bench and row.get("split") in ("test", "evaluation"):
                by_bench[b_id].append(row)

    # #56 PhishNChips
    by_bench[56] = []
    with open(test56_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                by_bench[56].append(json.loads(line))

    # #59 RAGTruth response-level hallucination (Area 2: Language Understanding)
    from decision_index.suite.build import adapters_added
    from decision_index.suite.build.layout import Layout

    rag_rows = [
        r for r in adapters_added.ragtruth(Layout(DI_ROOT / "work"))
        if r.get("split") in ("test", "evaluation")
    ]
    by_bench[59] = rag_rows

    # #9 Home appliances (Area 4: Tools & Automation)
    from decision_index.suite.build import home_appliance as ha_mod

    raw_ha = [
        ha_mod.make_row(h, i)
        for h in range(len(ha_mod.HOUSEHOLDS))
        for i in range(10)
    ]
    ha_rows = [r for r in raw_ha if r.get("split") in ("test", "evaluation")]
    subset_path = DI_ROOT / "decision_index" / "data" / "release-v2.1" / "home-appliances-subset.json"
    if subset_path.exists():
        keep_ids = set(json.loads(subset_path.read_text(encoding="utf-8")).get("keep_test_ids", []))
        if keep_ids:
            ha_rows = [r for r in ha_rows if r.get("id") in keep_ids]
    for r in ha_rows:
        # Set primary_field to the primary resolution decision question
        roles = (r.get("metadata") or {}).get("question_roles") or {}
        res_qid = next((qid for qid, role in roles.items() if role == "resolution"), None)
        if not res_qid and r.get("questions"):
            res_qid = next(iter(r["questions"].keys()))
        r["primary_field"] = res_qid
    by_bench[9] = ha_rows

    area_map = {
        57: ("knowledge_reasoning", "MMLU-Pro", 0.1109, 1.2, "accuracy"),
        58: ("knowledge_reasoning", "BBH fixed-option tasks", 0.3101, 1.2, "accuracy"),
        59: ("language", "RAGTruth response-level hallucination", 0.5177, 1.0, "f1"),
        56: ("retrieval_classification", "PhishNChips phishing decisions", 0.5000, 1.0, "accuracy"),
        62: ("tools_automation", "When2Call MCQ", 0.2500, 1.0, "accuracy"),
        9: ("tools_automation", "Home appliances", 0.1111, 1.0, "accuracy"),
        64: ("arts_human_taste", "New Yorker caption matching", 0.2000, 1.0, "accuracy"),
    }

    selected: list[dict] = []
    for b_id in [57, 58, 59, 56, 62, 9, 64]:
        pool = list(by_bench[b_id])
        area_key, b_name, chance, gold_mult, metric_kind = area_map[b_id]
        if b_id == 59:
            # Stratified sampling on True/False to preserve realistic hallucination prevalence
            def _is_true(r: dict) -> bool:
                exp = r.get("expected")
                if isinstance(exp, dict):
                    return bool(next(iter(exp.values())))
                return bool(exp)

            pos = [r for r in pool if _is_true(r)]
            neg = [r for r in pool if not _is_true(r)]
            rng.shuffle(pos)
            rng.shuffle(neg)
            n_pos = per_benchmark // 2
            n_neg = per_benchmark - n_pos
            sample = pos[:n_pos] + neg[:n_neg]
            rng.shuffle(sample)
        else:
            rng.shuffle(pool)
            sample = pool[: min(per_benchmark, len(pool))]

        for idx, item in enumerate(sample):
            row_copy = dict(item)
            row_copy["id"] = str(item.get("id") or f"di03_{b_id}_{idx:04d}")
            row_copy["benchmark"] = b_id
            row_copy["benchmark_name"] = b_name
            row_copy["area"] = area_key
            row_copy["chance"] = chance
            row_copy["gold_multiplier"] = gold_mult
            row_copy["metric_kind"] = metric_kind
            selected.append(row_copy)

    out_path = ROOT / "benchmarks" / "decision_index_v03_battery.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Decision Index 0.3] Wrote {len(selected)} items to {out_path}")
    return out_path


def build_jevbench_battery() -> Path:
    tiers = [
        ("easy", JB_ROOT / "datasets" / "public" / "easy.jsonl"),
        ("standard", JB_ROOT / "datasets" / "public" / "original.jsonl"),
        ("hard", JB_ROOT / "datasets" / "public" / "hard.jsonl"),
        ("vision", JB_ROOT / "datasets" / "public" / "vision_battery.jsonl"),
    ]
    rows: list[dict] = []
    for tier_name, path in tiers:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["jevbench_tier"] = tier_name
                rows.append(item)

    out_path = ROOT / "benchmarks" / "jevbench_battery.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[JevBench v1.6.1] Wrote {len(rows)} items to {out_path}")
    return out_path


if __name__ == "__main__":
    build_decision_index_v03_battery(per_benchmark=50, seed=42)
    build_jevbench_battery()
