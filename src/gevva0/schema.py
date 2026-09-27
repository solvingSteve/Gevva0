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


class DecisionResponse(BaseModel):
    test_id: Optional[str] = None
    decision: str
    decision_text: str
    confidence: float
    latency_ms: float
    mode: str = "fast_path"
    escalate: bool = False
    evidence: Optional[DecisionEvidence] = None

    # Compatibility attributes for playground / existing clients
    duration_ms: Optional[float] = None
    candidate_letter: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    letter_probabilities: Optional[dict[str, float]] = None
    cot_scratchpad: Optional[str] = None
    image_attached: Optional[bool] = None

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

