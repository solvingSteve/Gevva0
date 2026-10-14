from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DecisionEvidence(BaseModel):
    verdict_letter: str
    quote: str = Field(
        description="Verbatim excerpt from the context text supporting the decision."
    )
    rationale: str = Field(
        description="One to two sentences explaining why the evidence supports the verdict."
    )
    generation_latency_ms: float

    @property
    def verbatim_quote(self) -> str:
        return self.quote


class DecisionResponse(BaseModel):
    test_id: Optional[str] = None
    decision: str
    decision_text: str
    confidence: float
    latency_ms: float
    mode: str = "fast_path"
    escalate: bool = False
    evidence: Optional[DecisionEvidence] = None

    # Tier 0 early exit and cascade tracking
    tier: int = 1
    tier0_escalation_reason: Optional[str] = None
    tier0_metrics: Optional[dict[str, Any]] = None

    # Compatibility attributes for playground / existing clients
    duration_ms: Optional[float] = None
    candidate_letter: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    letter_probabilities: Optional[dict[str, float]] = None
    cot_scratchpad: Optional[str] = None
    image_attached: Optional[bool] = None

    @property
    def selected_option(self) -> str:
        return self.decision_text or self.decision

    @property
    def calibrated_p(self) -> float:
        return self.confidence

    @property
    def path(self) -> str:
        return self.mode

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class AuditRequest(BaseModel):
    context: str = ""
    verdict_letter: str
    verdict_text: str
    image: Optional[str] = None
    max_tokens: int = 192
    temperature: float = 0.1


class DecisionRequest(BaseModel):
    context: str = ""
    options: dict[str, str] | list[str] = Field(
        ..., description="Candidate options as a dict {'A': '...', 'B': '...'} or list of strings"
    )
    confidence_threshold: Optional[float] = None
    escalate_threshold: float = 0.60
    enable_rules: bool = True  # Enable Tier -1 Heuristic Rule Gateway
    early_exit: bool = False  # Enable Tier 0 Topological Absorbing Early Exit
    force_tier0: bool = False  # Force Tier 0 evaluation even on cold-start manifold (M < 50)
    tier0_entropy_threshold: Optional[float] = None
    tier0_margin_threshold: Optional[float] = None
    cyclic_debias: bool = True
    kv_branching: bool = False  # Allows per-request override
    kv_branching_min_tokens: int = 50
    image: Optional[str] = None
    images: Optional[list[str]] = None
    image_path: Optional[str] = None
    image_bytes: Optional[str] = None
    max_cot_tokens: Optional[int] = None
    cot_temp: Optional[float] = None
    cot_prompt: Optional[str] = Field(None, description="Custom Chain-of-Thought reasoning prompt prefix.")
    use_calibrator: Optional[bool] = None
    include_evidence: bool = False
    test_id: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Query or prompt text to classify")
    candidates: list[str] = Field(..., description="List of candidate categories or actions")
    early_exit: bool = Field(True, description="Enable Tier 0 early exit")
    force_tier0: bool = Field(False, description="Force Tier 0 evaluation even when manifold has fewer than 50 landmarks")
    tau_entropy: float = Field(0.65, description="Shannon entropy threshold tau_1")
    tau_margin: float = Field(0.25, description="Classification margin threshold tau_2")
    tau_ood: float = Field(0.40, description="Out-of-distribution distance threshold")


class ClassifyResponse(BaseModel):
    tier: int = Field(..., description="Resolved tier: 0 (Early Exit), 1 (Fast-Path), or 2 (CoT)")
    decision: str = Field(..., description="Selected candidate label or category")
    confidence: float = Field(..., description="Decision confidence probability")
    latency_ms: float = Field(..., description="Observed end-to-end latency in milliseconds")
    mode: str = Field("tier0_early_exit", description="Execution mode: tier0_early_exit, fast_path, cot_verified")
    escalated_from_tier0: bool = False
    tier0_reason: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    entropy: Optional[float] = None
    margin: Optional[float] = None

