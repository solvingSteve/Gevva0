"""Project Gevva0: local calibrated decision engine on Gemma 4 26B-A4B MoE (GGUF)."""

from .calibration import TemperatureCalibrator, brier_score, ece, softmax
from .engine import GemmaDecisionEngine, GevvaEngine, debiased_cyclic_evaluate
from .bridge import DecisionBridge, DecisionRequest, DecisionResult, Tier0DecisionBridge
from .audit import DecisionEvidence, explain_decision
from .profiles import HARDWARE_PRESETS, estimate_vram_usage
from .sparse_logits import SparseLogitProjector, is_gemma4_model

__version__ = "0.1.0"

__all__ = [
    "GevvaEngine",
    "GemmaDecisionEngine",
    "debiased_cyclic_evaluate",
    "DecisionBridge",
    "Tier0DecisionBridge",
    "DecisionRequest",
    "DecisionResult",
    "TemperatureCalibrator",
    "brier_score",
    "ece",
    "softmax",
    "DecisionEvidence",
    "explain_decision",
    "HARDWARE_PRESETS",
    "estimate_vram_usage",
    "SparseLogitProjector",
    "is_gemma4_model",
    "__version__",
]
