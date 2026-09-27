"""Project Gevva0: local calibrated decision engine on Gemma 4 26B-A4B MoE (GGUF)."""

from .calibration import TemperatureCalibrator, brier_score, ece, softmax
from .engine import GemmaDecisionEngine, debiased_cyclic_evaluate
from .audit import DecisionEvidence, explain_decision
from .profiles import HARDWARE_PRESETS, estimate_vram_usage

__version__ = "0.1.0"

__all__ = [
    "GemmaDecisionEngine",
    "debiased_cyclic_evaluate",
    "TemperatureCalibrator",
    "brier_score",
    "ece",
    "softmax",
    "DecisionEvidence",
    "explain_decision",
    "HARDWARE_PRESETS",
    "estimate_vram_usage",
    "__version__",
]
