"""
Gevva0 Pac-Man Brain Adapter powered by the universal DecisionBridge.

Architectural boundary:
- DecisionBridge (src/gevva0/bridge.py) standardizes Gevva0's core strengths:
  pinned KV cache prefixes, multi-sequence cyclic debiasing, logit marginalization,
  and bounded CoT verification into a reusable, calibrated engine.
- brain.py formats Pac-Man route cards (tactics -> DecisionRequest) and translates
  verdicts (DecisionResult) into joystick actions for the Pilot and telemetry for the HUD.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gevva0.bridge import DecisionBridge, DecisionRequest, DecisionResult

# Backwards compatibility alias
Decision = DecisionResult

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
DEFAULT_PROFILE_PATH = DEFAULT_MODEL_DIR / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.json"
RULEBOOK = """You are the tactical decision core of an arcade Pac-Man agent.
At each junction, evaluate the candidate route options (A, B, C, D) and select the single best letter.

Telemetry Fields:
- status: SAFE (uncontested) > TIGHT (active pursuit, limited margin) > TRAPPED (dead-end pocket / closing pincer) > DEADLY (immediate collision).
- room: Free tiles reachable ahead of dangerous ghosts. Higher room guarantees future escape branching.
- gap: Tile head-start over closest dangerous ghost ("clear" = uncontested). Gap < 2 is high-risk.
- tunnel: Route enters warp tunnel (ghosts move at half speed; high evasion value).
- power: Distance to power pellet (HIGH/REQUIRED = must consume now; LOW = conserve for defense).
- eat: Vulnerable blue ghost catchable before recovery (multiplies score).
- dots: Pellets on path to next junction / sector density ahead.
- fruit: Bonus target distance.
- REVERSE: 180-degree turnaround.

Hierarchical Decision Rules (Strict Priority Cascade 1 -> 5):
1. ZERO-COLLISION SURVIVAL (Absolute Constraint):
   - Immediately eliminate DEADLY and TRAPPED routes if any SAFE or TIGHT route exists.
   - If all options are threatened, maximize: (gap * 2 + room). Prioritize tunnel routes.
   - If multiple routes are safe, eliminate REVERSE to maintain forward momentum and avoid loops.
2. BLUE GHOST PREDATION:
   - Select an "eat" route if status is SAFE or TIGHT (gap >= 2) to harvest bonus points.
3. STRATEGIC POWER PELLET GATING:
   - If power is HIGH/REQUIRED, take the power pellet immediately.
   - If power is LOW, avoid the power pellet to preserve it, unless no other safe dots exist.
4. DOT SWEEP & PATH MOMENTUM:
   - Maximize dots path. Break ties using higher forward room.
5. BONUS HARVEST:
   - Take fruit only if route status is SAFE.

Decision: The single best candidate letter (A, B, C, or D) is:"""

COT_PROMPT = "\nVerification (eliminate DEADLY/TRAPPED, compare room/dots): "
COT_CONCLUSION = "\nBest option:"



@dataclass
class ModelProfile:
    temperature: float = 1.1550
    cot_threshold: float = 0.0
    bias_offsets: dict[str, float] = field(default_factory=dict)
    cot_max_tokens: int = 40
    raw: dict = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None) -> "ModelProfile":
        if path is None or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pac = data.get("pacman", {}) if isinstance(data.get("pacman"), dict) else {}
        return cls(
            temperature=float(pac.get("temperature", data.get("temperature", 1.0))),
            cot_threshold=float(pac.get("cot_threshold", data.get("cot_threshold", 0.0))),
            bias_offsets={k: float(v) for k, v in pac.get("bias_offsets", {}).items()},
            cot_max_tokens=int(pac.get("cot_max_tokens", 32)),
            raw=data,
            path=Path(path),
        )

    def calibrate(self, logits: np.ndarray, labels: list[str]) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64).copy()
        for i, l in enumerate(labels):
            z[i] -= self.bias_offsets.get(l, 0.0)
        return z / max(self.temperature, 1e-3)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


class BaseBrain:
    name = "brain"
    simulated = False

    def __init__(self):
        self.ready = False
        self.loading = True
        self.error: str | None = None
        self.model_name = ""
        self.latencies: deque[float] = deque(maxlen=400)
        self.total_decisions = 0
        self.mode_counts: dict[str, int] = {}
        self.history: deque[DecisionResult] = deque(maxlen=80)
        self.cot_latency_ms = 160.0
        self.log_path: Path | None = None
        self._log_fh = None

    def latency_stats(self) -> tuple[float, float]:
        if not self.latencies:
            return 0.0, 0.0
        arr = np.asarray(self.latencies)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))

    def open_log(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = path
        self._log_fh = open(path, "a", encoding="utf-8")

    def _log(self, sit, d: DecisionResult):
        if not self._log_fh:
            return
        rec = {
            "t": round(getattr(sit, "created", 0.0), 3),
            "serial": getattr(sit, "serial", d.serial),
            "labels": getattr(sit, "labels", list(d.probs.keys())),
            "suffix": getattr(sit, "suffix", ""),
            "letter": d.letter,
            "probs": d.probs,
            "mode": d.mode,
            "latency_ms": round(d.latency_ms, 2),
            "raw_logits": d.raw_logits,
            "threat": getattr(sit, "threat_alert", False),
            "cot": d.cot_text,
        }
        self._log_fh.write(json.dumps(rec) + "\n")
        self._log_fh.flush()

    def _record_decision(self, sit, d: DecisionResult):
        self.latencies.append(d.latency_ms)
        if d.mode == "cot_verified":
            self.cot_latency_ms = 0.8 * self.cot_latency_ms + 0.2 * d.latency_ms
        self.total_decisions += 1
        self.mode_counts[d.mode] = self.mode_counts.get(d.mode, 0) + 1
        self.history.append(d)
        self._log(sit, d)

    def shutdown(self):
        if self._log_fh:
            self._log_fh.close()


class Gevva0Brain(BaseBrain):
    """
    Pac-Man tactical brain adapter powered by the universal DecisionBridge.
    Formats route options into DecisionRequest and decodes DecisionResult.
    """
    name = "Gevva0"

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        profile_path: Path = DEFAULT_PROFILE_PATH,
        cyclic_debias: bool = False,
        cot: bool = False,
        cot_threshold: float | None = None,
        temperature: float | None = None,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.profile = ModelProfile.load(profile_path)
        if cot_threshold is not None:
            self.profile.cot_threshold = cot_threshold
        if temperature is not None:
            self.profile.temperature = temperature
        self.cyclic_debias = cyclic_debias
        self.cot_enabled = cot
        self.bridge: DecisionBridge | None = None
        self.model_name = "Gemma 4 26B-A4B (QAT Q4_K_XL)"
        self.prefix_tokens = 0
        self._current_poll: Callable[[], DecisionResult | None] | None = None
        self._pending_sit = None

    def load(self):
        try:
            t0 = time.perf_counter()
            self.bridge = DecisionBridge(
                model_path=self.model_path,
                system_prompt=RULEBOOK,
                temperature=self.profile.temperature,
                cot_threshold=self.profile.cot_threshold if self.cot_enabled else 0.0,
                cot_max_tokens=self.profile.cot_max_tokens,
                bias_offsets=self.profile.bias_offsets,
                cot_prompt=COT_PROMPT,
                cot_conclusion=COT_CONCLUSION,
            )
            self.prefix_tokens = self.bridge.prefix_tokens
            self.bridge.warmup(rounds=2)
            self.bridge.start_worker()
            self.ready = True
            print(
                f"[Gevva0 Pac-Man] universal bridge online in {time.perf_counter() - t0:.1f}s | "
                f"Platt T={self.profile.temperature:.3f} | CoT threshold={self.profile.cot_threshold:.2f} | "
                f"bias={self.profile.bias_offsets or 'none'}"
            )
        except Exception as exc:
            self.error = str(exc)
            print(f"[Gevva0 Pac-Man] universal bridge init failed: {exc}")
        finally:
            self.loading = False

    def load_async(self):
        threading.Thread(target=self.load, daemon=True, name="gevva0-loader").start()

    @property
    def busy(self) -> bool:
        if self.bridge is not None:
            return self.bridge.busy or self._current_poll is not None
        return False

    def _build_request(self, sit, allow_cot: bool) -> DecisionRequest:
        return DecisionRequest(
            state_text=sit.header,
            options=[o.card() for o in sit.options],
            custom_labels=sit.labels,
            allow_cot=self.cot_enabled and allow_cot,
            cyclic_debias=self.cyclic_debias,
            serial=sit.serial,
            metadata={"sit": sit, "threat_alert": sit.threat_alert},
        )

    def submit(self, sit, allow_cot: bool):
        """Non-blocking asynchronous submission to DecisionBridge."""
        if self.bridge is None:
            return
        req = self._build_request(sit, allow_cot)
        self._pending_sit = sit
        self._current_poll = self.bridge.submit_async(req)

    def poll(self) -> tuple[object, DecisionResult] | None:
        """Polls GPU worker status. Returns (Situation, DecisionResult) when complete."""
        if self._current_poll is None:
            return None
        res = self._current_poll()
        if res is not None:
            self._current_poll = None
            sit = res.metadata.get("sit", self._pending_sit)
            self._pending_sit = None
            self._record_decision(sit, res)
            return (sit, res)
        return None

    def decide(self, sit, allow_cot: bool = True) -> DecisionResult:
        """Synchronous decision route."""
        labels = sit.labels
        if len(labels) == 1:
            res = DecisionResult(
                choice_index=0,
                label=labels[0],
                choice_text=sit.options[0].card() if sit.options else labels[0],
                probs={labels[0]: 1.0},
                confidence=1.0,
                latency_ms=0.0,
                mode="forced",
                serial=sit.serial,
                metadata={"sit": sit},
            )
        else:
            req = self._build_request(sit, allow_cot)
            res = self.bridge.decide(req)
        self._record_decision(sit, res)
        return res

    def shutdown(self):
        super().shutdown()
        if self.bridge is not None:
            self.bridge.close()


class SimulatedBrain(BaseBrain):
    """GPU-free stand-in. Scores options with the reference heuristic; clearly labelled SIMULATED."""
    name = "Simulated"
    simulated = True

    def __init__(self, latency_ms: tuple[float, float] = (22.0, 34.0), seed: int = 0):
        super().__init__()
        self.model_name = "SIMULATED (heuristic stand-in, no GPU)"
        self.lat = latency_ms
        self.rng = random.Random(seed)
        self.realtime = True
        self._req = None
        self._res: tuple[object, DecisionResult] | None = None
        self._busy = False
        self._cv = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name="simulated-brain")
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy or self._req is not None

    def load(self):
        self.ready = True
        self.loading = False

    def load_async(self):
        self.load()
        self.start()

    def submit(self, sit, allow_cot: bool):
        with self._cv:
            self._req = (sit, allow_cot)
            self._cv.notify()

    def poll(self) -> tuple[object, DecisionResult] | None:
        with self._cv:
            r, self._res = self._res, None
            return r

    def _loop(self):
        while True:
            with self._cv:
                while self._req is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                sit, allow_cot = self._req
                self._req = None
                self._busy = True
            try:
                d = self.decide(sit, allow_cot)
                with self._cv:
                    self._res = (sit, d)
            except Exception as exc:
                print(f"[SimulatedBrain] inference error: {exc}")
            finally:
                self._busy = False

    def decide(self, sit, allow_cot: bool = True) -> DecisionResult:
        labels = sit.labels
        if len(labels) == 1:
            d = DecisionResult(
                choice_index=0,
                label=labels[0],
                choice_text=sit.options[0].card() if sit.options else labels[0],
                probs={labels[0]: 1.0},
                confidence=1.0,
                latency_ms=0.0,
                mode="forced",
                serial=sit.serial,
                metadata={"sit": sit},
            )
        else:
            from tactics import heuristic_utility
            u = np.array([heuristic_utility(o, sit) for o in sit.options]) / 12.0
            p = softmax(u)
            lat = self.rng.uniform(*self.lat)
            if self.realtime:
                time.sleep(lat / 1000.0)
            top = int(np.argmax(p))
            d = DecisionResult(
                choice_index=top,
                label=labels[top],
                choice_text=sit.options[top].card(),
                probs={l: float(x) for l, x in zip(labels, p)},
                confidence=float(p[top]),
                latency_ms=lat,
                mode="simulated",
                raw_logits=[float(x) for x in u],
                serial=sit.serial,
                metadata={"sit": sit},
            )
        self._record_decision(sit, d)
        return d

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
        super().shutdown()
