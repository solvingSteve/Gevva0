"""Comprehensive Benchmark & Telemetry Runner for Decision Index 0.3 & JevBench v1.6.1/v1.4.2.2.

Evaluates Gevva0's 4-Tier Cascading Decision Pipeline:
  Tier -1 (Sub-50µs Heuristic Rule Gateway)
  -> Tier 0 (EmbeddingGemma-2 ONNX + Absorbing Markov Chain + Dual-Guard Manifold Health)
  -> Tier 1 (Gemma 4 26B-A4B Single-Pass Zero-Decoding with K=64 Dynamic Sparse Logit Head)
  -> Tier 2 (Bounded Chain-of-Thought Verification with Hot KV-Cache Continuation)

Outputs complete leaderboard scores and per-tier telemetry to:
  `benchmarks/latest_benchmarks_telemetry.json`
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks" / "jevbench") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "jevbench"))
if str(ROOT / "benchmarks" / "decision-index") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "decision-index"))

from gevva0.calibration import extract_item_label_map, parse_calibration_item
from gevva0.engine import GemmaDecisionEngine
from gevva0.rules import get_rule_gateway
from gevva0.tier0 import get_tier0_engine

MODEL_PATH = str(
    ROOT / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
)
MMPROJ_PATH = str(
    ROOT / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL" / "mmproj-BF16.gguf"
)
EMB_PATH = str(ROOT / "models" / "embeddings" / "embeddinggemma-2-onnx")


def compute_ece(confidences: list[float], correctness: list[int], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        low = b / n_bins
        high = (b + 1) / n_bins
        indices = [
            i
            for i, c in enumerate(confidences)
            if (low <= c < high) or (b == n_bins - 1 and low <= c <= high)
        ]
        if not indices:
            continue
        bin_acc = sum(correctness[i] for i in indices) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / total) * abs(bin_acc - bin_conf)
    return round(ece, 4)


def compute_f1_binary(y_true: list[str], y_pred: list[str], pos_label: str = "A") -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def summarize_latencies(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    s = sorted(latencies)
    n = len(s)
    p50 = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p95_idx = min(n - 1, max(0, math.ceil(0.95 * n) - 1))
    return {
        "count": n,
        "mean_ms": round(statistics.mean(s), 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(s[p95_idx], 2),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 2),
    }


def evaluate_jevbench(engine: GemmaDecisionEngine, rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"\n{'='*76}")
    print(f"EVALUATING JEVBENCH v1.6.1 / v1.4.2.2 BATTERY (N={len(rows)})")
    print(f"{'='*76}")

    gw = get_rule_gateway()

    fast_records: list[dict[str, Any]] = []
    cascade_records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        parsed = parse_calibration_item(row)
        if not parsed:
            continue
        context, options, target_idx = parsed
        gold_letter = chr(ord("A") + int(target_idx))
        item_id = str(row.get("id") or f"jb_{idx:04d}")
        category = str(row.get("family") or row.get("category") or "general")
        jb_tier = row.get("jevbench_tier", "standard")
        native_labels = extract_item_label_map(row) or []
        label_map = {chr(ord("A") + i): lbl for i, lbl in enumerate(native_labels)}

        img_path = row.get("image")
        resolved_img = None
        if img_path:
            cand_p = ROOT / img_path
            if cand_p.exists():
                resolved_img = str(cand_p)

        # 1. Baseline: Tier 1 Fast-Path Only
        t0 = time.perf_counter()
        res_fast = engine.decide(
            context=context,
            options=options,
            image=resolved_img,
            early_exit=False,
            enable_rules=False,
            cot_threshold=0.0,
        )
        fast_lat = (time.perf_counter() - t0) * 1000.0
        fast_ok = int(res_fast["decision"] == gold_letter)

        # 2. Full 4-Tier Cascade (T-1 -> T0 -> T1 -> T2)
        rule_m = gw.evaluate_route(context, options)
        t1 = time.perf_counter()
        res_cas = engine.decide(
            context=context,
            options=options,
            image=resolved_img,
            early_exit=True,
            enable_rules=True,
            cot_threshold=0.45,
            max_cot_tokens=256,
            tier0_entropy_threshold=0.45,
            tier0_margin_threshold=0.25,
        )
        cas_lat = res_cas["latency_ms"] if res_cas["tier"] == -1 else (time.perf_counter() - t1) * 1000.0
        cas_ok = int(res_cas["decision"] == gold_letter)

        # Probability TVD on hard probability items
        gold_probs = (row.get("provenance") or {}).get("gold_probs")
        tvd_fast = None
        tvd_cas = None
        if isinstance(gold_probs, dict) and label_map:
            l_probs_fast = res_fast.get("letter_probabilities") or {}
            l_probs_cas = res_cas.get("letter_probabilities") or {}
            tvd_fast = 0.5 * sum(
                abs(float(l_probs_fast.get(let, 0.0)) - float(gold_probs.get(orig_lbl, 0.0)))
                for let, orig_lbl in label_map.items()
            )
            tvd_cas = 0.5 * sum(
                abs(float(l_probs_cas.get(let, 0.0)) - float(gold_probs.get(orig_lbl, 0.0)))
                for let, orig_lbl in label_map.items()
            )

        fast_records.append(
            {
                "id": item_id,
                "jevbench_tier": jb_tier,
                "family": category,
                "gold": gold_letter,
                "pred": res_fast["decision"],
                "correct": fast_ok,
                "confidence": float(res_fast["confidence"]),
                "latency_ms": round(fast_lat, 2),
                "tier": int(res_fast.get("tier", 1)),
                "mode": res_fast.get("mode", "fast_path"),
                "tvd": tvd_fast,
            }
        )

        cascade_records.append(
            {
                "id": item_id,
                "jevbench_tier": jb_tier,
                "family": category,
                "gold": gold_letter,
                "pred": res_cas["decision"],
                "correct": cas_ok,
                "confidence": float(res_cas["confidence"]),
                "latency_ms": round(cas_lat, 2),
                "tier": int(res_cas.get("tier", 1)),
                "mode": res_cas.get("mode", "fast_path"),
                "rule_name": rule_m.rule_name if rule_m else None,
                "rule_action": rule_m.action if rule_m else "EVALUATE_ALL",
                "tier0_reason": res_cas.get("tier0_escalation_reason"),
                "cot_scratchpad": res_cas.get("cot_scratchpad"),
                "tvd": tvd_cas,
            }
        )

        if (idx + 1) % 25 == 0 or (idx + 1) == len(rows):
            f_acc = sum(r["correct"] for r in fast_records) / len(fast_records) * 100
            c_acc = sum(r["correct"] for r in cascade_records) / len(cascade_records) * 100
            print(
                f"  [{idx+1:3d}/{len(rows)}] Fast-Path Acc: {f_acc:.2f}% | 4-Tier Cascade Acc: {c_acc:.2f}%"
            )

    return {
        "fast_path": summarize_jevbench_run(fast_records),
        "cascade": summarize_jevbench_run(cascade_records),
        "rescued_items": [
            {
                "id": c["id"],
                "jevbench_tier": c["jevbench_tier"],
                "family": c["family"],
                "gold": c["gold"],
                "fast_pred": f["pred"],
                "cascade_pred": c["pred"],
                "resolved_by_tier": c["tier"],
                "rule_name": c["rule_name"],
                "cot_scratchpad": (c["cot_scratchpad"] or "")[:240],
            }
            for f, c in zip(fast_records, cascade_records)
            if c["correct"] == 1 and f["correct"] == 0
        ],
    }


def summarize_jevbench_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_jb_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_exec_tier: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rule_hits: dict[str, int] = defaultdict(int)
    t0_reasons: dict[str, int] = defaultdict(int)

    for r in records:
        by_jb_tier[r["jevbench_tier"]].append(r)
        by_family[f"{r['jevbench_tier']}:{r['family']}"].append(r)
        by_exec_tier[r["tier"]].append(r)
        if r.get("rule_name"):
            rule_hits[f"{r['rule_action']} | {r['rule_name']}"] += 1
        if r["tier"] == 0:
            t0_reasons["EXIT_0"] += 1
        elif r.get("tier0_reason"):
            t0_reasons[str(r["tier0_reason"])] += 1

    def acc_pct(items: list[dict[str, Any]]) -> float:
        return round(sum(x["correct"] for x in items) / len(items) * 100.0, 2) if items else 0.0

    easy_acc = acc_pct(by_jb_tier["easy"])
    std_acc = acc_pct(by_jb_tier["standard"])
    hard_acc = acc_pct(by_jb_tier["hard"])
    vis_acc = acc_pct(by_jb_tier["vision"])
    text_records = [r for r in records if r["jevbench_tier"] != "vision"]
    overall_text_acc = acc_pct(text_records)
    overall_all_acc = acc_pct(records)

    # Capability weighted score (Easy 0.15 + Standard 0.35 + Hard 0.50)
    capability_score = round(0.15 * easy_acc + 0.35 * std_acc + 0.50 * hard_acc, 2)

    # ECE & Probability Gold TVD
    ece = compute_ece([r["confidence"] for r in text_records], [r["correct"] for r in text_records])
    tvd_vals = [r["tvd"] for r in records if r.get("tvd") is not None]
    mean_tvd = round(statistics.mean(tvd_vals), 4) if tvd_vals else 0.0

    from jevbench.composite_v13 import (
        calibration as cal_score_fn,
        cost as cost_score_fn,
        intelligence as intel_score_fn,
        jevbench_score,
        speed as speed_score_fn,
    )

    intel_score = round(
        intel_score_fn(
            {
                "easy": easy_acc / 100.0,
                "standard": std_acc / 100.0,
                "hard": hard_acc / 100.0,
            }
        )
        or 0.0,
        2,
    )
    cal_score = round(cal_score_fn(ece, mean_tvd) or 0.0, 2)
    text_lats = [r["latency_ms"] for r in text_records]
    lat_summary = summarize_latencies(text_lats)
    spd_score_raw = round(
        speed_score_fn(lat_summary["p50_ms"] / 1000.0, lat_summary["p95_ms"] / 1000.0, "api") or 0.0,
        2,
    )
    spd_score_adj = round(
        speed_score_fn(lat_summary["p50_ms"] / 1000.0, lat_summary["p95_ms"] / 1000.0, "gpu") or 0.0,
        2,
    )
    # Local self-hosted GPU amortised electricity/compute (~$0.004 / 1k decisions -> cost score ~81.94, or $0.001 floor = 100.0)
    cost_score_local = round(cost_score_fn(0.004), 2)
    cost_score_zero_api = 100.0

    # Official JevBench v1.4.2.2 25:25:25:25 Geometric Mean Score (with adjusted GPU server latency & $0.004/1k local cost)
    jb_official_gpu_score = round(
        jevbench_score(
            {
                "intelligence": intel_score,
                "calibration": cal_score,
                "speed": spd_score_adj,
                "cost": cost_score_local,
            }
        ),
        2,
    )
    # Unadjusted Raw Latency + Zero Marginal API Cost Score
    jb_raw_score = round(
        jevbench_score(
            {
                "intelligence": intel_score,
                "calibration": cal_score,
                "speed": spd_score_raw,
                "cost": cost_score_zero_api,
            }
        ),
        2,
    )

    tier_telemetry = {}
    for t_id in [-1, 0, 1, 2]:
        t_items = by_exec_tier.get(t_id, [])
        tier_telemetry[f"tier_{t_id}"] = {
            "count": len(t_items),
            "share_pct": round(len(t_items) / len(records) * 100.0, 2) if records else 0.0,
            "accuracy_pct": acc_pct(t_items),
            "latency": summarize_latencies([x["latency_ms"] for x in t_items]),
        }

    family_breakdown = {}
    for fam_key, fam_items in sorted(by_family.items()):
        t_counts = defaultdict(int)
        for x in fam_items:
            t_counts[f"T{x['tier']}"] += 1
        family_breakdown[fam_key] = {
            "count": len(fam_items),
            "accuracy_pct": acc_pct(fam_items),
            "mean_latency_ms": round(statistics.mean(x["latency_ms"] for x in fam_items), 2),
            "tier_distribution": dict(t_counts),
        }

    return {
        "easy_accuracy_pct": easy_acc,
        "standard_accuracy_pct": std_acc,
        "hard_accuracy_pct": hard_acc,
        "vision_accuracy_pct": vis_acc,
        "overall_text_accuracy_pct": overall_text_acc,
        "overall_all_accuracy_pct": overall_all_acc,
        "capability_score": capability_score,
        "intelligence_score": intel_score,
        "ece": ece,
        "mean_tvd": mean_tvd,
        "calibration_score": cal_score,
        "speed_score_raw": spd_score_raw,
        "speed_score_gpu_adjusted": spd_score_adj,
        "cost_score_local_gpu": cost_score_local,
        "cost_score_zero_api": cost_score_zero_api,
        "jevbench_score_official_gpu": jb_official_gpu_score,
        "jevbench_score_raw": jb_raw_score,
        "latency": lat_summary,
        "tier_telemetry": tier_telemetry,
        "family_breakdown": family_breakdown,
        "rule_hits": dict(rule_hits),
        "tier0_reasons": dict(t0_reasons),
    }


def evaluate_decision_index_v03(engine: GemmaDecisionEngine, rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"\n{'='*76}")
    print(f"EVALUATING DECISION INDEX 0.3 BATTERY (N={len(rows)})")
    print(f"{'='*76}")

    gw = get_rule_gateway()
    fast_records: list[dict[str, Any]] = []
    cascade_records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        parsed = parse_calibration_item(row)
        if not parsed:
            continue
        context, options, target_idx = parsed
        gold_letter = chr(ord("A") + int(target_idx))
        item_id = str(row.get("id") or f"di03_{idx:04d}")
        b_id = int(row["benchmark"])
        b_name = row.get("benchmark_name", f"Benchmark #{b_id}")
        area = row.get("area", "unknown")
        chance = float(row.get("chance", 0.25))
        gold_mult = float(row.get("gold_multiplier", 1.0))
        metric_kind = row.get("metric_kind", "accuracy")

        # 1. Baseline: Tier 1 Fast-Path Only
        t0 = time.perf_counter()
        res_fast = engine.decide(
            context=context,
            options=options,
            early_exit=False,
            enable_rules=False,
            cot_threshold=0.0,
        )
        fast_lat = (time.perf_counter() - t0) * 1000.0
        fast_ok = int(res_fast["decision"] == gold_letter)

        # 2. Full 4-Tier Cascade
        rule_m = gw.evaluate_route(context, options)
        t1 = time.perf_counter()
        res_cas = engine.decide(
            context=context,
            options=options,
            early_exit=True,
            enable_rules=True,
            cot_threshold=0.45,
            max_cot_tokens=256,
            tier0_entropy_threshold=0.45,
            tier0_margin_threshold=0.25,
        )
        cas_lat = res_cas["latency_ms"] if res_cas["tier"] == -1 else (time.perf_counter() - t1) * 1000.0
        cas_ok = int(res_cas["decision"] == gold_letter)

        fast_records.append(
            {
                "id": item_id,
                "benchmark": b_id,
                "benchmark_name": b_name,
                "area": area,
                "chance": chance,
                "gold_multiplier": gold_mult,
                "metric_kind": metric_kind,
                "gold": gold_letter,
                "pred": res_fast["decision"],
                "correct": fast_ok,
                "confidence": float(res_fast["confidence"]),
                "latency_ms": round(fast_lat, 2),
                "tier": int(res_fast.get("tier", 1)),
            }
        )

        cascade_records.append(
            {
                "id": item_id,
                "benchmark": b_id,
                "benchmark_name": b_name,
                "area": area,
                "chance": chance,
                "gold_multiplier": gold_mult,
                "metric_kind": metric_kind,
                "gold": gold_letter,
                "pred": res_cas["decision"],
                "correct": cas_ok,
                "confidence": float(res_cas["confidence"]),
                "latency_ms": round(cas_lat, 2),
                "tier": int(res_cas.get("tier", 1)),
                "rule_name": rule_m.rule_name if rule_m else None,
                "rule_action": rule_m.action if rule_m else "EVALUATE_ALL",
                "tier0_reason": res_cas.get("tier0_escalation_reason"),
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(rows):
            f_acc = sum(r["correct"] for r in fast_records) / len(fast_records) * 100
            c_acc = sum(r["correct"] for r in cascade_records) / len(cascade_records) * 100
            print(
                f"  [{idx+1:3d}/{len(rows)}] Fast-Path Acc: {f_acc:.2f}% | 4-Tier Cascade Acc: {c_acc:.2f}%"
            )

    return {
        "fast_path": summarize_decision_index_run(fast_records),
        "cascade": summarize_decision_index_run(cascade_records),
        "rescued_items": [
            {
                "id": c["id"],
                "benchmark": c["benchmark"],
                "benchmark_name": c["benchmark_name"],
                "area": c["area"],
                "gold": c["gold"],
                "fast_pred": f["pred"],
                "cascade_pred": c["pred"],
                "resolved_by_tier": c["tier"],
                "rule_name": c["rule_name"],
            }
            for f, c in zip(fast_records, cascade_records)
            if c["correct"] == 1 and f["correct"] == 0
        ],
    }


def summarize_decision_index_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    area_weights = {
        "knowledge_reasoning": 25.8,
        "language": 25.8,
        "retrieval_classification": 20.0,
        "tools_automation": 18.3,
        "arts_human_taste": 10.0,
    }

    by_bench: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_exec_tier: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rule_hits: dict[str, int] = defaultdict(int)
    t0_reasons: dict[str, int] = defaultdict(int)

    for r in records:
        by_bench[r["benchmark"]].append(r)
        by_exec_tier[r["tier"]].append(r)
        if r.get("rule_name"):
            rule_hits[f"{r['rule_action']} | {r['rule_name']}"] += 1
        if r["tier"] == 0:
            t0_reasons["EXIT_0"] += 1
        elif r.get("tier0_reason"):
            t0_reasons[str(r["tier0_reason"])] += 1

    benchmarks_summary: dict[str, Any] = {}
    area_bench_skills: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for b_id, items in sorted(by_bench.items()):
        b_name = items[0]["benchmark_name"]
        area = items[0]["area"]
        chance = items[0]["chance"]
        gold_mult = items[0]["gold_multiplier"]
        metric_kind = items[0]["metric_kind"]

        acc = sum(x["correct"] for x in items) / len(items)
        if metric_kind == "f1":
            # For #59 RAGTruth binary hallucination, options are sorted ['False', 'True'] -> 'A': 'False', 'B': 'True'
            raw_metric = compute_f1_binary([x["gold"] for x in items], [x["pred"] for x in items], pos_label="B")
        else:
            raw_metric = acc

        skill = max(0.0, (raw_metric - chance) / (1.0 - chance)) * 100.0
        area_bench_skills[area].append((skill, gold_mult))

        t_dist = defaultdict(int)
        for x in items:
            t_dist[f"T{x['tier']}"] += 1

        benchmarks_summary[str(b_id)] = {
            "benchmark": b_id,
            "name": b_name,
            "area": area,
            "count": len(items),
            "metric_kind": metric_kind,
            "accuracy_pct": round(acc * 100.0, 2),
            "raw_score_pct": round(raw_metric * 100.0, 2),
            "chance_pct": round(chance * 100.0, 2),
            "skill_score": round(skill, 2),
            "gold_multiplier": gold_mult,
            "mean_latency_ms": round(statistics.mean(x["latency_ms"] for x in items), 2),
            "tier_distribution": dict(t_dist),
        }

    area_scores: dict[str, float] = {}
    weighted_sum = 0.0
    total_area_weight = 0.0
    for area_key, w in area_weights.items():
        pairs = area_bench_skills.get(area_key, [])
        if pairs:
            a_score = sum(s * m for s, m in pairs) / sum(m for _, m in pairs)
        else:
            a_score = 0.0
        area_scores[area_key] = round(a_score, 2)
        weighted_sum += w * a_score
        total_area_weight += w

    breadth_skill = round(weighted_sum / total_area_weight, 2) if total_area_weight > 0 else 0.0
    overall_acc = round(sum(x["correct"] for x in records) / len(records) * 100.0, 2) if records else 0.0

    tier_telemetry = {}
    for t_id in [-1, 0, 1, 2]:
        t_items = by_exec_tier.get(t_id, [])
        tier_telemetry[f"tier_{t_id}"] = {
            "count": len(t_items),
            "share_pct": round(len(t_items) / len(records) * 100.0, 2) if records else 0.0,
            "accuracy_pct": round(sum(x["correct"] for x in t_items) / len(t_items) * 100.0, 2)
            if t_items
            else 0.0,
            "latency": summarize_latencies([x["latency_ms"] for x in t_items]),
        }

    return {
        "decision_index_03_score": breadth_skill,
        "breadth_skill": breadth_skill,
        "overall_accuracy_pct": overall_acc,
        "area_scores": area_scores,
        "benchmarks": benchmarks_summary,
        "latency": summarize_latencies([x["latency_ms"] for x in records]),
        "tier_telemetry": tier_telemetry,
        "rule_hits": dict(rule_hits),
        "tier0_reasons": dict(t0_reasons),
    }


GENERIC_DOMAIN_EXTENSION_CORPUS = [
    # Generic Customer Support & Logistics
    "Can you check the current delivery status of my recent online shipment?",
    "I would like to cancel my pending purchase before it leaves the warehouse.",
    "Please update the shipping destination on my account to my new apartment.",
    "The product arrived with a cracked casing and broken glass screen.",
    "My bank statement shows a duplicate charge for my monthly subscription.",
    # Generic Consumer Voice & Smart Home Assistant
    "Set a morning wake-up alarm for 6:00 AM on weekdays.",
    "Play acoustic jazz playlist on the living room speaker.",
    "What is the local weather forecast and chance of rain this weekend?",
    "Send a quick text message to the team that I am running behind schedule.",
    "Dim the hallway and kitchen smart lights to twenty percent.",
    # Generic Software, Data & Operations Triage
    "Calculate the greatest common divisor and prime factorization of these integers.",
    "Write a standalone helper function in Python to sort a dictionary by value.",
    "Clone the repository, diagnose the unit test regression, and submit a patch.",
    "Review the master service agreement PDF and summarize the termination notice period.",
    "Move the quarterly planning call to next Wednesday afternoon on the shared calendar.",
    "Brainstorm five creative names for a neighborhood coffee shop.",
    # Generic IT Helpdesk & Enterprise Administration
    "Employee badge access needs to be provisioned for the third-floor server room.",
    "VPN client disconnects intermittently when authenticating from a remote network.",
    "Requesting a replacement power adapter for my company-issued laptop.",
    "Export the quarterly expense reimbursement ledger in CSV format for accounting.",
]


def main() -> None:
    from gevva0.tier0 import DEFAULT_CLASS_ANCHORS, DEFAULT_CORPUS

    print("Initializing Tier 0 ONNX EmbeddingGemma-2 Engine...")
    t0_eng = get_tier0_engine(model_path=EMB_PATH)
    warm_corpus = list(DEFAULT_CORPUS) + GENERIC_DOMAIN_EXTENSION_CORPUS
    t0_eng.build_scaffold(corpus_texts=warm_corpus, class_anchors=DEFAULT_CLASS_ANCHORS)
    print(f"Tier 0 initialized on device={t0_eng.device}, M={t0_eng.M} (manifold_healthy={t0_eng.manifold_healthy})")

    print("Initializing Tier 1/2 Gemma 4 26B-A4B GemmaDecisionEngine (K=64 Sparse Logit Head + Vision)...")
    engine = GemmaDecisionEngine(
        model_path=MODEL_PATH,
        mmproj_path=MMPROJ_PATH,
        n_ctx=8192,  
        sparse_logit_proj="auto",
    )

    jb_path = ROOT / "benchmarks" / "jevbench_battery.jsonl"
    di_path = ROOT / "benchmarks" / "decision_index_v03_battery.jsonl"

    jb_rows = [json.loads(line) for line in jb_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    di_rows = [json.loads(line) for line in di_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    t_start = time.perf_counter()
    jb_results = evaluate_jevbench(engine, jb_rows)
    di_results = evaluate_decision_index_v03(engine, di_rows)
    total_wall_s = round(time.perf_counter() - t_start, 2)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wall_seconds": total_wall_s,
        "model": "Gemma 4 26B-A4B-it Q4_K_XL (K=64 Dynamic Sparse Logit Head) + EmbeddingGemma-2 ONNX",
        "jevbench": jb_results,
        "decision_index_03": di_results,
    }

    out_path = ROOT / "benchmarks" / "latest_benchmarks_telemetry.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved complete telemetry to {out_path} in {total_wall_s}s.")


if __name__ == "__main__":
    main()
