"""
Pac-Man brain adapter refactored to consume the generic DecisionBridge.
This module demonstrates the universal bridge pattern implemented in demos/pacman/brain.py.
"""

from __future__ import annotations

from brain import (
    COT_CONCLUSION,
    COT_PROMPT,
    RULEBOOK,
    Decision,
    DecisionBridge,
    DecisionRequest,
    DecisionResult,
    Gevva0Brain,
    ModelProfile,
    SimulatedBrain,
    softmax,
)

__all__ = [
    "DecisionBridge",
    "DecisionRequest",
    "DecisionResult",
    "Decision",
    "Gevva0Brain",
    "SimulatedBrain",
    "ModelProfile",
    "softmax",
    "RULEBOOK",
    "COT_PROMPT",
    "COT_CONCLUSION",
]