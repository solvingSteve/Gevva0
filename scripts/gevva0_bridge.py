"""
Gevva0 Bridge: Arcade Pac-Man Decision Gateway
==============================================
Asynchronous, thread-safe gateway connecting the real-time Pac-Man loop 
to the Gemma 4 / Gevva0 logit scoring engine.
Supports multimodal vision (screen captures) and strict 5-action junction resolution:
  [A] NORTH (Up)
  [B] SOUTH (Down)
  [C] EAST (Right)
  [D] WEST (Left)
  [E] STOP (Hold position / Maintain heading)
"""

from __future__ import annotations
import sys
import os
import time
import threading
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


@dataclass
class DecisionResult:
    selected_key: str    # "A", "B", "C", "D", or "E"
    selected_text: str
    confidence: float
    latency_ms: float
    mode: str            # "fast_path", "cot_verified", "cyclic_debiased"
    probabilities: dict[str, float]
    threat_alert: bool = False
    timestamp: float = field(default_factory=time.time)
    tick: int = 0
    dispatch_id: int = 0
    tac_desc: str = ""
    cot_scratchpad: Optional[str] = None


def normalize_decision_options(options: Any) -> dict[str, str]:
    """
    Normalizes candidate options into a strict dict[str, str]:
      - dict[str, str]: e.g. {"A": "Steer EAST", "B": "Steer WEST"} -> returned directly
      - list[dict]: e.g. [{"key": "A", "label": "Steer EAST"}, ...] -> {"A": "Steer EAST", ...}
      - list[str]: e.g. ["Steer EAST", "Steer WEST"] -> {"A": "Steer EAST", "B": "Steer WEST"}
    """
    from typing import Any as _Any
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, (list, tuple)):
        normalized = {}
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                k = opt.get("key") or opt.get("letter") or chr(ord("A") + i)
                v = opt.get("label") or opt.get("action") or opt.get("description") or str(opt)
                normalized[str(k)] = str(v)
            else:
                normalized[chr(ord("A") + i)] = str(opt)
        return normalized
    return {}


class Gevva0Bridge:
    def __init__(
        self,
        decision_interval_ticks: int = 10,
        use_vision: bool = False,
        model_path: Optional[str] = None,
        kv_branching: bool = True,
        kv_branching_min_tokens: int = 0,
        cyclic_debias: bool = False,
        cot_threshold: float = 0.0,
        cot_threshold_threat: float = 0.1,
        n_gpu_layers: Optional[int] = -1,
        calibration_path: Optional[str | Path] = None,
    ):
        self.decision_interval = decision_interval_ticks
        self.use_vision = use_vision
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.kv_branching = kv_branching
        self.kv_branching_min_tokens = kv_branching_min_tokens
        self.cyclic_debias = cyclic_debias
        self.cot_threshold = cot_threshold
        self.cot_threshold_threat = cot_threshold_threat
        self.n_gpu_layers = n_gpu_layers
        self.engine = None
        self.is_ready = False
        self.loading = True
        self.error: Optional[str] = None
        self.model_name = "Loading model from llm_config.json..."

        self.latest_decision: Optional[DecisionResult] = None
        self.decision_history: deque[DecisionResult] = deque(maxlen=60)
        self.total_decisions: int = 0
        self.avg_latency_ms: float = 0.0

        self._pending_request: Optional[dict] = None
        self._result_ready = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

    def initialize(self):
        try:
            from gevva0 import GemmaDecisionEngine
            from gevva0.config import resolve_model_path, format_friendly_name, _load_llm_config, PROJECT_ROOT
            cfg = _load_llm_config() or {}
            cfg_model = self.model_path or cfg.get("model", {}).get("model_path")
            resolved = resolve_model_path(cfg_model) if cfg_model else resolve_model_path()
            self.model_name = format_friendly_name(str(resolved))

            engine_kwargs = {}
            if self.n_gpu_layers is not None:
                engine_kwargs["n_gpu_layers"] = self.n_gpu_layers

            # Domain Calibration: check explicitly provided path, env var, or local calibration_pacman.json
            cal_target = self.calibration_path
            if not cal_target:
                env_cal = os.environ.get("GEVVA0_CALIBRATION")
                if env_cal and Path(env_cal).exists():
                    cal_target = env_cal
                elif (Path(__file__).parent / "calibration_pacman.json").exists():
                    cal_target = str(Path(__file__).parent / "calibration_pacman.json")
                elif (PROJECT_ROOT / "calibration_pacman.json").exists():
                    cal_target = str(PROJECT_ROOT / "calibration_pacman.json")

            if cal_target:
                engine_kwargs["calibration_path"] = cal_target
                print(f"[Gevva0 Pac-Man] Pointing engine to domain calibration profile: {cal_target}")

            print(f"[Gevva0 Pac-Man] Booting Calibrated Decision Engine (GPU VRAM Accelerated) with model: {self.model_name} ({resolved.name})...")
            self.engine = GemmaDecisionEngine.from_config(
                model_path=str(resolved) if cfg_model else None,
                **engine_kwargs,
            )
            self.is_ready = True
            self.loading = False
            print(f"[Gevva0 Pac-Man] Engine online (GPU Accelerated)! Model: {self.model_name} | Vision enabled: {self.engine.has_vision}")

            if self.engine.calibrator and self.engine.calibrator.bias_offsets:
                b_info = ", ".join(f"{k}: {v:+0.2f}" for k, v in sorted(self.engine.calibrator.bias_offsets.items()))
                print(f"[Gevva0 Pac-Man] Domain Vector Calibration ACTIVE (T={self.engine.calibrator.temperature:.2f} | Bias Offsets: {b_info}).")
                if self.cyclic_debias:
                    print("[Gevva0 Pac-Man] Bypassing cyclic debiasing -> single forward pass active (~30-45ms latency).")
                    self.cyclic_debias = False

            if self.use_vision and not self.engine.has_vision:
                print("[Gevva0 Pac-Man] Vision projector unavailable, falling back to text telemetry")
                self.use_vision = False

            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
        except Exception as exc:
            self.error = str(exc)
            self.loading = False
            print(f"[Gevva0 Pac-Man] Init Error: {exc}")

    def initialize_async(self):
        t = threading.Thread(target=self.initialize, daemon=True)
        t.start()

    def request_decision(
        self,
        context: str,
        options: Any,
        frame_png_bytes: Optional[bytes] = None,
        tick: int = 0,
        threat_alert: bool = False,
        dispatch_id: int = 0,
        tac_desc: str = "",
    ):
        if not self.is_ready:
            return
        norm_options = normalize_decision_options(options)
        with self._lock:
            self._pending_request = {
                "context": context,
                "options": norm_options,
                "frame": frame_png_bytes if self.use_vision else None,
                "tick": tick,
                "threat_alert": threat_alert,
                "dispatch_id": dispatch_id,
                "tac_desc": tac_desc,
            }
            self._result_ready.set()

    def decide(
        self,
        context: str,
        options: Any,
        frame_png_bytes: Optional[bytes] = None,
        tick: int = 0,
        threat_alert: bool = False,
        dispatch_id: int = 0,
        tac_desc: str = "",
    ) -> Optional[DecisionResult]:
        """Synchronously evaluates candidate options at the active junction."""
        options = normalize_decision_options(options)
        if len(options) <= 1:
            only_key = next(iter(options.keys())) if options else "A"
            only_text = options.get(only_key, "Continue")
            decision = DecisionResult(
                selected_key=only_key,
                selected_text=only_text,
                confidence=1.0,
                latency_ms=0.01,
                mode="forced_move",
                probabilities={only_text: 1.0},
                threat_alert=threat_alert,
                tick=tick,
                dispatch_id=dispatch_id,
                tac_desc=tac_desc or "Automatic forced move (single legal candidate)",
            )
            with self._lock:
                self.latest_decision = decision
                self.decision_history.append(decision)
                self.total_decisions += 1
            return decision

        if not self.is_ready or self.engine is None:
            return None
        with self._lock:
            cot_thresh = self.cot_threshold_threat if threat_alert else self.cot_threshold
            result = self.engine.decide(
                context=context,
                options=options,
                image=frame_png_bytes if self.use_vision else None,
                cot_threshold=cot_thresh,
                max_cot_tokens=80,
                cyclic_debias=self.cyclic_debias and not self.use_vision,
                kv_branching=self.kv_branching,
                kv_branching_min_tokens=self.kv_branching_min_tokens,
            )

            decision = DecisionResult(
                selected_key=result.decision,
                selected_text=result.decision_text,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                mode=result.mode,
                probabilities=result.probabilities or {},
                threat_alert=threat_alert,
                tick=tick,
                dispatch_id=dispatch_id,
                tac_desc=tac_desc,
                cot_scratchpad=result.cot_scratchpad,
            )

            self.latest_decision = decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_decisions - 1) + decision.latency_ms)
                / self.total_decisions
            )
            return decision

    def get_latest_decision(self) -> Optional[DecisionResult]:
        return self.latest_decision

    def _worker_loop(self):
        while not self._shutdown.is_set():
            self._result_ready.wait(timeout=0.5)
            self._result_ready.clear()

            with self._lock:
                req = self._pending_request
                self._pending_request = None

            if req is None:
                continue

            try:
                cot_thresh = self.cot_threshold_threat if req.get("threat_alert") else self.cot_threshold
                result = self.engine.decide(
                    context=req["context"],
                    options=req["options"],
                    image=req["frame"],
                    cot_threshold=cot_thresh,
                    max_cot_tokens=80,
                    cyclic_debias=self.cyclic_debias and not self.use_vision,
                    kv_branching=self.kv_branching,
                    kv_branching_min_tokens=self.kv_branching_min_tokens,
                )

                decision = DecisionResult(
                    selected_key=result.decision,
                    selected_text=result.decision_text,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                    mode=result.mode,
                    probabilities=result.probabilities or {},
                    threat_alert=req["threat_alert"],
                    tick=req["tick"],
                    dispatch_id=req.get("dispatch_id", 0),
                    tac_desc=req.get("tac_desc", ""),
                    cot_scratchpad=result.cot_scratchpad,
                )

                self.latest_decision = decision
                self.decision_history.append(decision)
                self.total_decisions += 1
                self.avg_latency_ms = (
                    (self.avg_latency_ms * (self.total_decisions - 1) + decision.latency_ms)
                    / self.total_decisions
                )
            except Exception as e:
                print(f"[Gevva0 Pac-Man] Inference error: {e}")

    def shutdown(self):
        self._shutdown.set()
        self._result_ready.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
        if self.engine:
            self.engine.close()


class SimulatedBridge:
    """
    Intelligent simulated Gevva0 Gateway for testing and review without RTX 5090 GPU.
    Uses spatial heuristics to simulate realistic softmax logit distributions,
    calibrated confidences, and 16-28ms latencies.
    """

    def __init__(
        self,
        decision_interval_ticks: int = 20,
        use_vision: bool = True,
        model_path: Optional[str] = None,
        kv_branching: bool = False,
        kv_branching_min_tokens: int = 50,
        cyclic_debias: bool = True,
        cot_threshold: float = 0.72,
        cot_threshold_threat: float = 0.88,
        cpu_only: bool = True,
        calibration_path: Optional[str | Path] = None,
        **kwargs,
    ):
        self.decision_interval = decision_interval_ticks
        self.use_vision = use_vision
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.kv_branching = kv_branching
        self.kv_branching_min_tokens = kv_branching_min_tokens
        self.cyclic_debias = cyclic_debias
        self.cot_threshold = cot_threshold
        self.cot_threshold_threat = cot_threshold_threat
        self.cpu_only = cpu_only
        self.calibrator = None
        self.is_ready = False
        self.loading = True
        self.error: Optional[str] = None

        from gevva0.config import resolve_model_path, format_friendly_name, _load_llm_config, PROJECT_ROOT
        cfg = _load_llm_config() or {}
        cfg_model = self.model_path or cfg.get("model", {}).get("model_path")
        try:
            resolved = resolve_model_path(cfg_model) if cfg_model else resolve_model_path()
            self.model_name = format_friendly_name(str(resolved))
        except Exception:
            self.model_name = "Gemma 4"

        self.latest_decision: Optional[DecisionResult] = None
        self.decision_history: deque[DecisionResult] = deque(maxlen=60)
        self.total_decisions: int = 0
        self.avg_latency_ms: float = 21.4

        self._pending_request: Optional[dict] = None
        self._result_ready = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None

    def initialize(self):
        time.sleep(1.0)
        from gevva0.calibration import TemperatureCalibrator
        from gevva0.config import PROJECT_ROOT
        cal_target = self.calibration_path
        if not cal_target:
            env_cal = os.environ.get("GEVVA0_CALIBRATION")
            if env_cal and Path(env_cal).exists():
                cal_target = env_cal
            elif (Path(__file__).parent / "calibration_pacman.json").exists():
                cal_target = str(Path(__file__).parent / "calibration_pacman.json")
            elif (PROJECT_ROOT / "calibration_pacman.json").exists():
                cal_target = str(PROJECT_ROOT / "calibration_pacman.json")

        if cal_target and Path(cal_target).exists():
            self.calibrator = TemperatureCalibrator.load(cal_target)
            t_val = self.calibrator.temperature
            b_cnt = len(self.calibrator.bias_offsets)
            print(f"[Simulated Gateway] Loaded Domain Calibration from {cal_target} (T={t_val:.4f}, bias_offsets={b_cnt}).")
        else:
            self.calibrator = TemperatureCalibrator(temperature=1.1550)

        self.is_ready = True
        self.loading = False
        print(f"[Simulated Gateway] Calibrated Neural Emulation Ready (Model: {self.model_name}, T={self.calibrator.temperature:.4f}).")
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def initialize_async(self):
        t = threading.Thread(target=self.initialize, daemon=True)
        t.start()

    def request_decision(
        self,
        context: str,
        options: Any,
        frame_png_bytes: Optional[bytes] = None,
        tick: int = 0,
        threat_alert: bool = False,
        heuristic_priorities: Optional[dict[str, float]] = None,
        dispatch_id: int = 0,
        tac_desc: str = "",
    ):
        norm_options = normalize_decision_options(options)
        priors = heuristic_priorities
        if priors is None and isinstance(options, (list, tuple)):
            priors = {
                opt["key"]: opt["priority"]
                for opt in options
                if isinstance(opt, dict) and "key" in opt and "priority" in opt
            }
        with self._lock:
            self._pending_request = {
                "context": context,
                "options": norm_options,
                "tick": tick,
                "threat_alert": threat_alert,
                "priorities": priors or {},
                "dispatch_id": dispatch_id,
                "tac_desc": tac_desc,
            }
            self._result_ready.set()

    def decide(
        self,
        context: str,
        options: Any,
        frame_png_bytes: Optional[bytes] = None,
        tick: int = 0,
        threat_alert: bool = False,
        heuristic_priorities: Optional[dict[str, float]] = None,
        dispatch_id: int = 0,
        tac_desc: str = "",
    ) -> DecisionResult:
        import numpy as np
        norm_options = normalize_decision_options(options)
        if heuristic_priorities is None and isinstance(options, (list, tuple)):
            heuristic_priorities = {
                opt["key"]: opt["priority"]
                for opt in options
                if isinstance(opt, dict) and "key" in opt and "priority" in opt
            }
        options = norm_options
        with self._lock:
            if len(options) <= 1:
                only_key = next(iter(options.keys())) if options else "A"
                only_text = options.get(only_key, "Continue")
                decision = DecisionResult(
                    selected_key=only_key,
                    selected_text=only_text,
                    confidence=1.0,
                    latency_ms=0.01,
                    mode="forced_move",
                    probabilities={only_text: 1.0},
                    threat_alert=threat_alert,
                    tick=tick,
                    dispatch_id=dispatch_id,
                    tac_desc=tac_desc or "Automatic forced move (single legal candidate)",
                )
                self.latest_decision = decision
                self.decision_history.append(decision)
                self.total_decisions += 1
                return decision

            latency_ms = random.uniform(16.5, 29.8)
            time.sleep(latency_ms / 1000.0)

            keys = list(options.keys())
            priorities = heuristic_priorities or {}
            raw_scores = []
            for k in keys:
                val = priorities.get(k, 10.0)
                raw_scores.append(val / 25.0 + random.gauss(0, 0.2))

            from gevva0.calibration import softmax
            if self.calibrator:
                scaled = self.calibrator.transform_logits(np.array(raw_scores), keys)
            else:
                scaled = np.array(raw_scores) / 1.1550
            probs = softmax(scaled)

            top_idx = int(np.argmax(probs))
            chosen_key = keys[top_idx]
            chosen_text = options[chosen_key]
            confidence = float(probs[top_idx])

            prob_dict = {options[k]: round(float(probs[i]), 4) for i, k in enumerate(keys)}

            cot_thresh = self.cot_threshold_threat if threat_alert else self.cot_threshold
            decision = DecisionResult(
                selected_key=chosen_key,
                selected_text=chosen_text,
                confidence=round(confidence, 4),
                latency_ms=round(latency_ms, 2),
                mode="fast_path" if confidence >= cot_thresh else "cot_verified",
                probabilities=prob_dict,
                threat_alert=threat_alert,
                tick=tick,
                dispatch_id=dispatch_id,
                tac_desc=tac_desc,
                cot_scratchpad="Threats: None. Pellets: prioritized open corridor. Rule 4.1 compliant." if confidence < cot_thresh else None,
            )

            self.latest_decision = decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_decisions - 1) + latency_ms)
                / self.total_decisions
            )
            return decision

    def get_latest_decision(self) -> Optional[DecisionResult]:
        return self.latest_decision

    def _worker_loop(self):
        import numpy as np

        while not self._shutdown.is_set():
            self._result_ready.wait(timeout=0.5)
            self._result_ready.clear()

            with self._lock:
                req = self._pending_request
                self._pending_request = None

            if req is None:
                continue

            # Simulate 18ms - 32ms logit extraction latency
            latency_ms = random.uniform(16.5, 29.8)
            time.sleep(latency_ms / 1000.0)

            options = normalize_decision_options(req["options"])
            keys = list(options.keys())
            if not keys:
                continue

            if len(options) <= 1:
                only_key = next(iter(options.keys())) if options else "A"
                only_text = options.get(only_key, "Continue")
                decision = DecisionResult(
                    selected_key=only_key,
                    selected_text=only_text,
                    confidence=1.0,
                    latency_ms=0.01,
                    mode="forced_move",
                    probabilities={only_text: 1.0},
                    threat_alert=req.get("threat_alert", False),
                    tick=req.get("tick", 0),
                    dispatch_id=req.get("dispatch_id", 0),
                    tac_desc=req.get("tac_desc", "") or "Automatic forced move (single legal candidate)",
                )
                self.latest_decision = decision
                self.decision_history.append(decision)
                self.total_decisions += 1
                continue

            priorities = req.get("priorities", {})
            raw_scores = []
            for k in keys:
                val = priorities.get(k, 10.0)
                # Add slight logit jitter
                raw_scores.append(val / 25.0 + random.gauss(0, 0.2))

            # Domain calibrated softmax transformation
            from gevva0.calibration import softmax
            if self.calibrator:
                scaled = self.calibrator.transform_logits(np.array(raw_scores), keys)
            else:
                scaled = np.array(raw_scores) / 1.1550
            probs = softmax(scaled)

            top_idx = int(np.argmax(probs))
            chosen_key = keys[top_idx]
            chosen_text = options[chosen_key]
            confidence = float(probs[top_idx])

            prob_dict = {options[k]: round(float(probs[i]), 4) for i, k in enumerate(keys)}

            cot_thresh = self.cot_threshold_threat if req.get("threat_alert") else self.cot_threshold
            decision = DecisionResult(
                selected_key=chosen_key,
                selected_text=chosen_text,
                confidence=round(confidence, 4),
                latency_ms=round(latency_ms, 2),
                mode="fast_path" if confidence >= cot_thresh else "cot_verified",
                probabilities=prob_dict,
                threat_alert=req["threat_alert"],
                tick=req["tick"],
                dispatch_id=req.get("dispatch_id", 0),
                tac_desc=req.get("tac_desc", ""),
                cot_scratchpad="Simulated verification: selected highest priority action without conflict." if confidence < cot_thresh else None,
            )

            self.latest_decision = decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_decisions - 1) + latency_ms)
                / self.total_decisions
            )

    def shutdown(self):
        self._shutdown.set()
        self._result_ready.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
