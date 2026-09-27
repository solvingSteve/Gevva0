from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


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
        bias_offsets: dict[str, float] | None = None,
    ):
        self.temperature = float(temperature)
        self.bias_offsets: dict[str, float] = bias_offsets or {}

    def fit_brier_score(
        self,
        logits: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[int] | list[np.ndarray],
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
            res = minimize(brier_objective, x0=[start], bounds=[(0.05, 5.0)], method="L-BFGS-B")
            self.temperature = float(res.x[0])
            if verbose:
                target_indices = one_hot_labels.argmax(axis=1)
                before_probs = softmax(logits_arr, axis=1)
                after_probs = softmax(logits_arr / self.temperature, axis=1)
                print(f"Optimal Calibrated Temperature T: {self.temperature:.4f}")
                print(
                    "Brier: "
                    f"{brier_score(before_probs, one_hot_labels):.4f} -> "
                    f"{brier_score(after_probs, one_hot_labels):.4f} | "
                    f"ECE: {ece(before_probs, target_indices):.4f} -> {ece(after_probs, target_indices):.4f}"
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
        res = minimize(var_brier_objective, x0=[start], bounds=[(0.05, 5.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        if verbose:
            before_probs = [softmax(l) for l in logits_list]
            after_probs = [softmax(l / self.temperature) for l in logits_list]
            print(f"Optimal Calibrated Temperature T: {self.temperature:.4f}")
            print(
                "Brier: "
                f"{brier_score(before_probs, target_indices):.4f} -> "
                f"{brier_score(after_probs, target_indices):.4f} | "
                f"ECE: {ece(before_probs, target_indices):.4f} -> {ece(after_probs, target_indices):.4f}"
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
        bounds = [(0.05, 5.0)] + [(-5.0, 5.0)] * (K - 1)

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
            z_calibrated = (z_raw - b) / T
        where b is the positional/label bias vector and T is the temperature.
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
        return arr / self.temperature

    def transform(
        self,
        logits: np.ndarray,
        labels: list[str] | None = None,
    ) -> np.ndarray:
        cal_logits = self.transform_logits(logits, labels=labels)
        axis = 1 if cal_logits.ndim == 2 else None
        return softmax(cal_logits, axis=axis)

    def to_dict(self) -> dict:
        d = {"temperature": self.temperature}
        if self.bias_offsets:
            d["bias_offsets"] = self.bias_offsets
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        temp = float(data.get("temperature", 1.0))
        bias = data.get("bias_offsets", {})
        if bias:
            bias = {str(k): float(v) for k, v in bias.items()}
        return cls(temperature=temp, bias_offsets=bias)
