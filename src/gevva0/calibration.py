from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def softmax(logits: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    if axis is None:
        shifted = arr - arr.max()
        exp = np.exp(shifted)
        return exp / exp.sum()
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def brier_score(
    probs: np.ndarray | list[np.ndarray],
    labels: np.ndarray | list[int] | list[np.ndarray],
) -> float:
    if isinstance(probs, np.ndarray) and isinstance(labels, np.ndarray) and probs.ndim == 2 and labels.ndim == 2:
        probs = np.asarray(probs, dtype=np.float64)
        one_hot = np.asarray(labels, dtype=np.float64)
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Variable-length list of samples
    total = 0.0
    count = 0
    for p, y in zip(probs, labels):
        p_arr = np.asarray(p, dtype=np.float64)
        if hasattr(y, "__len__"):
            y_arr = np.asarray(y, dtype=np.float64)
        else:
            y_arr = np.zeros_like(p_arr)
            y_arr[int(y)] = 1.0
        total += float(np.sum((p_arr - y_arr) ** 2))
        count += 1
    return total / max(count, 1)


def ece(
    probs: np.ndarray | list[np.ndarray],
    labels: np.ndarray | list[int] | list[np.ndarray],
    n_bins: int = 15,
) -> float:
    confs: list[float] = []
    corrects: list[float] = []

    if isinstance(probs, np.ndarray) and isinstance(labels, np.ndarray) and probs.ndim == 2:
        if len(probs) == 0:
            return 0.0
        probs = np.asarray(probs, dtype=np.float64)
        confidence = probs.max(axis=1)
        predicted = probs.argmax(axis=1)
        target = labels.argmax(axis=1) if labels.ndim == 2 else labels
        correct = (predicted == target).astype(np.float64)
        confs = list(confidence)
        corrects = list(correct)
    else:
        for p, y in zip(probs, labels):
            p_arr = np.asarray(p, dtype=np.float64)
            c = float(np.max(p_arr))
            pred = int(np.argmax(p_arr))
            target = int(np.argmax(y)) if hasattr(y, "__len__") else int(y)
            confs.append(c)
            corrects.append(1.0 if pred == target else 0.0)

    if not confs:
        return 0.0

    conf_arr = np.array(confs, dtype=np.float64)
    corr_arr = np.array(corrects, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr > lo) & (conf_arr <= hi)
        if not mask.any():
            continue
        total += float(mask.sum()) * abs(float(corr_arr[mask].mean()) - float(conf_arr[mask].mean()))
    return total / len(conf_arr)


class TemperatureCalibrator:
    def __init__(
        self,
        temperature: float = 1.0,
        temperatures: dict[int | str, float] | None = None,
        bias_offsets: dict[str, float] | None = None,
        cot_threshold: float | None = None,
        metadata: dict | None = None,
    ):
        self.temperature = float(temperature)
        self.temperatures: dict[int, float] = {}
        if temperatures:
            for k, v in temperatures.items():
                try:
                    self.temperatures[int(k)] = float(v)
                except (ValueError, TypeError):
                    pass
        self.bias_offsets: dict[str, float] = bias_offsets or {}
        self.cot_threshold: float | None = float(cot_threshold) if cot_threshold is not None else None
        self.metadata: dict = dict(metadata) if metadata else {}

    def get_temperature(self, k: int | None = None) -> float:
        """Return cardinality-specific temperature T_k if available, else global temperature."""
        if k is not None and k in self.temperatures:
            return self.temperatures[k]
        return self.temperature

    def fit_brier_score(
        self,
        logits: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[int] | list[np.ndarray],
        optimize_cardinalities: bool = True,
        verbose: bool = True,
    ) -> "TemperatureCalibrator":
        # Check if 2D numpy array
        is_fixed_2d = (
            isinstance(logits, np.ndarray)
            and isinstance(labels, np.ndarray)
            and logits.ndim == 2
            and labels.ndim == 2
        )

        if is_fixed_2d:
            logits_arr = np.asarray(logits, dtype=np.float64)
            one_hot_labels = np.asarray(labels, dtype=np.float64)

            def brier_objective(t: np.ndarray) -> float:
                return brier_score(softmax(logits_arr / t[0], axis=1), one_hot_labels)

            start = max(self.temperature, 0.06)
            res = minimize(brier_objective, x0=[start], bounds=[(0.05, 10.0)], method="L-BFGS-B")
            self.temperature = float(res.x[0])
            target_indices = list(one_hot_labels.argmax(axis=1))
            before_probs = softmax(logits_arr, axis=1)
            after_probs = softmax(logits_arr / self.temperature, axis=1)

            k_card = logits_arr.shape[1]
            if optimize_cardinalities:
                self.temperatures[k_card] = self.temperature

            cal_brier = brier_score(after_probs, one_hot_labels)
            cal_ece = ece(after_probs, target_indices)
            confs = list(np.max(after_probs, axis=1))
            p20 = float(np.percentile(confs, 20)) if confs else 0.5
            p25 = float(np.percentile(confs, 25)) if confs else 0.5
            median = float(np.percentile(confs, 50)) if confs else 0.5

            self.cot_threshold = self.cot_threshold or 0.5
            self.metadata.update({
                "empirical_T": round(self.temperature, 4),
                "p20": round(p20, 4),
                "p25": round(p25, 4),
                "median": round(median, 4),
                "calibrated_brier": round(cal_brier, 4),
                "calibrated_ece": round(cal_ece, 4),
            })
            if verbose:
                print(f"Optimal Calibrated Temperature T: {self.temperature:.4f}")
                print(
                    "Brier: "
                    f"{brier_score(before_probs, one_hot_labels):.4f} -> "
                    f"{cal_brier:.4f} | "
                    f"ECE: {ece(before_probs, target_indices):.4f} -> {cal_ece:.4f}"
                )
            return self

        # Variable-length list of samples
        logits_list = [np.asarray(l, dtype=np.float64) for l in logits]
        target_indices = [
            int(np.argmax(y)) if hasattr(y, "__len__") else int(y)
            for y in labels
        ]

        def var_brier_objective(t: np.ndarray) -> float:
            scaled_probs = [softmax(l / t[0]) for l in logits_list]
            return brier_score(scaled_probs, target_indices)

        start = max(self.temperature, 0.06)
        res = minimize(var_brier_objective, x0=[start], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])

        # Optimize cardinality-specific temperatures (T_2, T_4, T_10, etc.)
        if optimize_cardinalities and logits_list:
            card_groups: dict[int, tuple[list[np.ndarray], list[int]]] = {}
            for l, y in zip(logits_list, target_indices):
                k = len(l)
                if k not in card_groups:
                    card_groups[k] = ([], [])
                card_groups[k][0].append(l)
                card_groups[k][1].append(y)

            for k, (k_logits, k_targets) in card_groups.items():
                if len(k_logits) >= 2:
                    def k_objective(t: np.ndarray, logs=k_logits, tgts=k_targets) -> float:
                        k_probs = [softmax(l / t[0]) for l in logs]
                        return brier_score(k_probs, tgts)

                    res_k = minimize(k_objective, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
                    self.temperatures[k] = round(float(res_k.x[0]), 4)
                else:
                    self.temperatures[k] = round(self.temperature, 4)

        before_probs = [softmax(l) for l in logits_list]
        after_probs = [softmax(l / self.get_temperature(len(l))) for l in logits_list]
        cal_brier = brier_score(after_probs, target_indices)
        cal_ece = ece(after_probs, target_indices)
        confs = [float(np.max(p)) for p in after_probs]
        p20 = float(np.percentile(confs, 20)) if confs else 0.5
        p25 = float(np.percentile(confs, 25)) if confs else 0.5
        median = float(np.percentile(confs, 50)) if confs else 0.5

        self.cot_threshold = self.cot_threshold or 0.5
        self.metadata.update({
            "empirical_T": round(self.temperature, 4),
            "cardinality_temperatures": {str(k): round(v, 4) for k, v in sorted(self.temperatures.items())},
            "p20": round(p20, 4),
            "p25": round(p25, 4),
            "median": round(median, 4),
            "calibrated_brier": round(cal_brier, 4),
            "calibrated_ece": round(cal_ece, 4),
        })

        if verbose:
            temps_str = f", Cardinality Ts: {self.temperatures}" if self.temperatures else ""
            print(f"Optimal Calibrated Temperature T: {self.temperature:.4f}{temps_str}")
            print(
                "Brier: "
                f"{brier_score(before_probs, target_indices):.4f} -> "
                f"{cal_brier:.4f} | "
                f"ECE: {ece(before_probs, target_indices):.4f} -> {cal_ece:.4f}"
            )
        return self

    def fit_affine(
        self,
        logits: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[int] | list[np.ndarray],
        candidate_keys: list[str] | None = None,
        verbose: bool = True,
    ) -> "TemperatureCalibrator":
        """
        Fits both domain temperature T and zero-mean positional bias offsets b
        by minimizing the multiclass Brier score over validation samples:
            Brier = 1/M sum_m sum_k (P_{m,k} - y_{m,k})^2
            where P_m = softmax((z_m - b) / T)
        """
        logits_list = [np.asarray(l, dtype=np.float64) for l in logits]
        target_indices = [
            int(np.argmax(y)) if hasattr(y, "__len__") else int(y)
            for y in labels
        ]

        if not logits_list:
            return self

        K = len(logits_list[0])
        keys = candidate_keys or [chr(ord('A') + i) for i in range(K)]

        def affine_objective(theta: np.ndarray) -> float:
            T = theta[0]
            if K > 1:
                b_free = theta[1:]
                b_last = -np.sum(b_free)
                b_vec = np.append(b_free, b_last)
            else:
                b_vec = np.zeros(K)

            scaled_probs = []
            for l in logits_list:
                l_len = len(l)
                if l_len == K:
                    cal = (l - b_vec) / T
                else:
                    b_sub = np.array([b_vec[i] if i < K else 0.0 for i in range(l_len)])
                    cal = (l - b_sub) / T
                scaled_probs.append(softmax(cal))

            return brier_score(scaled_probs, target_indices)

        x0 = np.zeros(1 + (K - 1))
        x0[0] = max(self.temperature, 0.1)
        bounds = [(0.05, 10.0)] + [(-5.0, 5.0)] * (K - 1)

        res = minimize(affine_objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        self.temperature = float(res.x[0])
        if K > 1:
            b_free = res.x[1:]
            b_last = -float(np.sum(b_free))
            b_vec = list(b_free) + [b_last]
            self.bias_offsets = {keys[i]: round(float(b_vec[i]), 4) for i in range(K)}
        else:
            self.bias_offsets = {}

        if verbose:
            before_probs = [softmax(l) for l in logits_list]
            after_probs = [self.transform(l, keys[:len(l)]) for l in logits_list]
            print(f"Optimal Calibrated Temperature T: {self.temperature:.4f}, Bias: {self.bias_offsets}")
            print(
                "Brier: "
                f"{brier_score(before_probs, target_indices):.4f} -> "
                f"{brier_score(after_probs, target_indices):.4f} | "
                f"ECE: {ece(before_probs, target_indices):.4f} -> {ece(after_probs, target_indices):.4f}"
            )
        return self

    def transform_logits(
        self,
        logits: np.ndarray,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        """
        Transforms raw logits into calibrated logits:
            z_calibrated = (z_raw - b) / T_k
        where b is positional bias and T_k is the cardinality-specific temperature.
        """
        arr = np.asarray(logits, dtype=np.float64).copy()
        if labels is not None and self.bias_offsets:
            if arr.ndim == 1:
                for i, l in enumerate(labels):
                    if l in self.bias_offsets:
                        arr[i] -= self.bias_offsets[l]
            elif arr.ndim == 2:
                for i, l in enumerate(labels):
                    if l in self.bias_offsets:
                        arr[:, i] -= self.bias_offsets[l]
        k = arr.shape[-1] if arr.ndim in (1, 2) else None
        t = self.get_temperature(k)
        return arr / t

    def transform(
        self,
        logits: np.ndarray,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        cal_logits = self.transform_logits(logits, labels=labels)
        axis = 1 if cal_logits.ndim == 2 else None
        return softmax(cal_logits, axis=axis)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"temperature": float(self.temperature)}
        if self.temperatures:
            d["temperatures"] = {str(k): float(v) for k, v in sorted(self.temperatures.items())}
        if self.cot_threshold is not None:
            d["cot_threshold"] = float(self.cot_threshold)
        if self.bias_offsets:
            d["bias_offsets"] = self.bias_offsets
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        temp = float(data.get("temperature", 1.0))
        temps = data.get("temperatures") or data.get("cardinality_temperatures")
        bias = data.get("bias_offsets", {})
        if bias:
            bias = {str(k): float(v) for k, v in bias.items()}
        cot_th = data.get("cot_threshold")
        meta = data.get("metadata", {})
        return cls(
            temperature=temp,
            temperatures=temps,
            bias_offsets=bias,
            cot_threshold=float(cot_th) if cot_th is not None else None,
            metadata=meta,
        )


# ── Calibration Battery Loader & Automated Calibration ───────────────────────────

def load_calibration_dataset(data_path: str | Path | None = None) -> list[dict]:
    """Load calibration dataset rows from slice or battery."""
    candidate_paths: list[Path] = []
    if data_path:
        candidate_paths.append(Path(data_path))
        candidate_paths.append(PROJECT_ROOT / data_path)

    candidate_paths.extend([
        PROJECT_ROOT / "benchmarks" / "calibration_slice_100.json",
        PROJECT_ROOT / "benchmarks" / "calibration_slice_100.jsonl",
        PROJECT_ROOT / "benchmarks" / "decision_index_battery.jsonl",
        PROJECT_ROOT / "calibration.jsonl",
        PROJECT_ROOT / "benchmarks" / "benchmark_suite.json",
    ])

    resolved_path: Path | None = None
    for p in candidate_paths:
        if p.is_file():
            resolved_path = p
            break

    if not resolved_path:
        raise FileNotFoundError(f"No calibration dataset found (searched {candidate_paths})")

    raw_text = resolved_path.read_text(encoding="utf-8").strip()
    if raw_text.startswith("[") or raw_text.startswith("{"):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "tests" in parsed:
                return parsed["tests"]
        except Exception:
            pass

    rows: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        raise ValueError(f"No valid rows found in calibration file: {resolved_path}")
    return rows


def extract_item_label_map(row: dict) -> list[str] | None:
    """Return the ordered native option labels for a Decision Index, JevBench, or Gevva0 row."""
    if "questions" in row and isinstance(row["questions"], dict) and row["questions"]:
        q_map = row["questions"]
        primary = (row.get("scoring") or {}).get("primary_field")
        if primary and primary in q_map:
            q_key = primary
        elif "q" in q_map:
            q_key = "q"
        elif "verdict" in q_map:
            q_key = "verdict"
        elif "answer" in q_map:
            q_key = "answer"
        else:
            q_key = next(iter(q_map.keys()))
        q_spec = q_map[q_key]
        criteria = q_spec.get("criteria") or {}
        if not criteria and q_spec.get("type") == "noul":
            return ["false", "true"]
        return [str(k) for k in criteria.keys()] if len(criteria) >= 2 else None

    if "question" in row and "labels" in row and isinstance(row["labels"], list):
        return [str(x) for x in row["labels"]] if len(row["labels"]) >= 2 else None

    raw_options = row.get("options")
    if isinstance(raw_options, dict):
        return [str(k).strip().upper() for k in raw_options.keys()]
    if isinstance(raw_options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [letters[i] for i in range(min(len(raw_options), len(letters)))]
    return None


def parse_calibration_item(row: dict) -> tuple[str, dict[str, str], int] | None:
    """Parse a dataset row (supporting Decision Index 0.3, JevBench v1.6, and legacy schemas)

    into (context, options_dict, target_idx).
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Schema 1: Decision Index (state, questions.<primary_field>.criteria, expected.<primary_field>)
    if "questions" in row and isinstance(row["questions"], dict) and row["questions"]:
        q_map = row["questions"]
        primary = (row.get("scoring") or {}).get("primary_field")
        if primary and primary in q_map:
            q_key = primary
        elif "q" in q_map:
            q_key = "q"
        elif "verdict" in q_map:
            q_key = "verdict"
        elif "answer" in q_map:
            q_key = "answer"
        elif "tool" in q_map:
            q_key = "tool"
        else:
            q_key = next(iter(q_map.keys()))

        q_spec = q_map[q_key]
        q_type = q_spec.get("type", "choice")
        criteria = q_spec.get("criteria") or {}
        if not criteria and q_type == "noul":
            criteria = {"false": "False / No", "true": "True / Yes"}
        if not criteria or len(criteria) < 2 or len(criteria) > len(letters):
            return None

        state = row.get("state", "")
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        instructions = q_spec.get("instructions", "")
        full_context = f"{state_str}\n\nQuestion: {instructions}" if instructions else state_str

        keys = [str(k) for k in criteria.keys()]
        descs = [
            f"{k}: {v}" if v is not None and str(v).strip() != str(k).strip() else str(k)
            for k, v in criteria.items()
        ]
        assigned_letters = [letters[i] for i in range(len(keys))]
        options = {assigned_letters[i]: descs[i] for i in range(len(keys))}

        exp = row.get("expected", {})
        target = exp.get(q_key) if isinstance(exp, dict) else exp
        if isinstance(target, bool):
            target_str = "true" if target else "false"
        else:
            target_str = str(target) if target is not None else ""

        keys_lower = [k.lower() for k in keys]
        if target_str in keys:
            target_idx = keys.index(target_str)
        elif target_str.lower() in keys_lower:
            target_idx = keys_lower.index(target_str.lower())
        elif target_str in assigned_letters:
            target_idx = assigned_letters.index(target_str)
        elif target_str in descs:
            target_idx = descs.index(target_str)
        else:
            return None

        return full_context, options, target_idx

    # Schema 2: JevBench (state, question.{type, instructions, criteria}, labels, expected)
    if "question" in row and isinstance(row["question"], dict) and "labels" in row:
        labels = [str(x) for x in (row.get("labels") or [])]
        n = len(labels)
        if n < 2 or n > len(letters):
            return None

        q_spec = row["question"]
        q_type = q_spec.get("type", "choice")
        instructions = q_spec.get("instructions", "")
        crit = q_spec.get("criteria")

        option_texts: list[str] = []
        for i, lab in enumerate(labels):
            if q_type == "score" and isinstance(crit, (list, tuple)):
                idx = int(lab) if lab.isdigit() else i
                desc = crit[idx] if idx < len(crit) else lab
                option_texts.append(f"Level {lab}: {desc}")
            elif isinstance(crit, dict):
                desc_key = {"yes": "true", "no": "false"}.get(lab.lower(), lab) if q_type == "noul" else lab
                desc = crit.get(desc_key) or crit.get(lab)
                if desc and str(desc).strip() != lab.strip():
                    option_texts.append(f"{lab}: {desc}")
                else:
                    option_texts.append(lab)
            else:
                option_texts.append(lab)

        state = row.get("state", "")
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        full_context = f"{state_str}\n\nQuestion: {instructions.strip()}" if instructions and instructions.strip() else state_str

        options = {letters[i]: option_texts[i] for i in range(n)}
        exp = row.get("expected")
        if exp is None:
            return None
        exp_str = str(exp)
        if exp_str in labels:
            target_idx = labels.index(exp_str)
        elif exp_str.lower() in [l.lower() for l in labels]:
            target_idx = [l.lower() for l in labels].index(exp_str.lower())
        else:
            return None

        return full_context, options, target_idx

    # Schema 3: Legacy (context, options, label / expected_ground_truth)
    label = row.get("label") or row.get("expected_ground_truth")
    if not label:
        return None

    ctx = row.get("context", "")
    q = row.get("question", "")
    if isinstance(q, str) and q and q not in ctx:
        ctx = f"{ctx}\n\nQuestion: {q}"

    raw_options = row.get("options")
    if isinstance(raw_options, dict):
        options = {str(k).strip().upper(): str(v).strip() for k, v in raw_options.items()}
    elif isinstance(raw_options, list):
        options = {letters[i]: str(v).strip() for i, v in enumerate(raw_options) if i < len(letters)}
    else:
        # String spec e.g. "A: yes, B: no"
        from .cli import parse_options
        try:
            options = parse_options(str(raw_options))
        except Exception:
            return None

    opt_keys = list(options.keys())
    if label not in opt_keys:
        return None

    target_idx = opt_keys.index(label)
    return ctx, options, target_idx


def calibrate_model(
    engine: Any,
    calibration_data: Path | str | list[dict] | None = None,
    output_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[TemperatureCalibrator, Path]:
    """Run calibration using Decision Index slice and save model-calibration.json."""
    if isinstance(calibration_data, list):
        rows = calibration_data
    else:
        rows = load_calibration_dataset(calibration_data)

    if verbose:
        print(f"[calibrate] Evaluating {len(rows)} samples across Decision Index battery...")

    logits_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for r in rows:
        parsed = parse_calibration_item(r)
        if not parsed:
            continue
        ctx, opts, target_idx = parsed
        try:
            scores = engine.score(ctx, opts)
            logits_list.append(np.asarray(scores, dtype=np.float64))
            labels_list.append(target_idx)
        except Exception as exc:
            if verbose:
                print(f"[calibrate] warning: failed scoring sample ({exc})")
            continue

    if not logits_list:
        raise RuntimeError("No valid labeled rows could be evaluated for calibration.")

    cal = TemperatureCalibrator()
    cal.fit_brier_score(logits_list, labels_list, optimize_cardinalities=True, verbose=verbose)

    model_path_str = getattr(engine, "model_path", None)
    m_p = Path(model_path_str) if model_path_str else PROJECT_ROOT / "models"
    model_dir = m_p.parent if m_p.is_file() else m_p

    out_file = Path(output_path) if output_path else model_dir / "model-calibration.json"
    model_id = model_dir.name if model_dir.name != "models" else m_p.stem

    cal.metadata.update({
        "model_id": model_id,
        "empirical_T": round(cal.temperature, 4),
        "samples": len(logits_list),
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })

    cal.save(out_file)
    if verbose:
        print(f"[calibrate] Saved calibrated profile to {out_file}")
    return cal, out_file
