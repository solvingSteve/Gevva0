"""Asymmetric quote and operational rationale extraction.

Key architectural principle:
Inference justification MUST NOT precede or poison the decision logits.
Audit extraction runs strictly AFTER the categorical decision is locked in a
separate inference pass, ensuring zero logit contamination while extracting
verbatim quotes and human-readable operational justifications.
"""

from __future__ import annotations
import base64
import json
import re
import time
from typing import TYPE_CHECKING, List, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .engine import GemmaDecisionEngine


class DecisionEvidence(BaseModel):
    verdict_letter: str
    quote: str = Field(
        description="Verbatim excerpt from the context or document supporting the decision."
    )
    rationale: str = Field(
        description="One to two sentences explaining why the evidence supports the verdict."
    )
    generation_latency_ms: float


def explain_decision(
    engine: "GemmaDecisionEngine",
    context: str,
    verdict_letter: str,
    verdict_text: str,
    image_bytes_list: Optional[List[bytes]] = None,
    max_tokens: int = 192,
    temperature: float = 0.1,
) -> DecisionEvidence:
    """Extract verbatim source quotes and rationale AFTER decision is locked."""
    start_t = time.perf_counter()

    with engine._lock:
        if image_bytes_list and engine.has_vision:
            b64_urls = [
                f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}"
                for b in image_bytes_list
            ]
            audit_prompt = (
                f"A diagnostic system classified this visual input as: [{verdict_letter}] {verdict_text}.\n\n"
            )
            if context and context.strip():
                audit_prompt += f"Context: {context}\n\n"
            audit_prompt += (
                f"Your Task:\n"
                f"1. Locate and extract the exact verbatim text, routing number, error title, or key visual element from the image that justifies this verdict.\n"
                f"2. Provide a 1-2 sentence operational rationale.\n\n"
                f"Respond strictly in JSON format matching this schema:\n"
                f"{{\n"
                f'  "verbatim_quote": "<exact text or visual observation>",\n'
                f'  "rationale": "<1-2 sentence justification>"\n'
                f"}}\n"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": audit_prompt},
                        *[{"type": "image_url", "image_url": {"url": u}} for u in b64_urls],
                    ],
                }
            ]

            response = engine.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw_text = response["choices"][0]["message"]["content"].strip()

        else:
            # Text-only isolated audit prompt
            clean_context = context.strip() if context else "No context text provided."
            audit_prompt = (
                f"<|turn>user\n"
                f"Context:\n{clean_context}\n\n"
                f"The system rendered the following verdict:\n"
                f"Verdict: [{verdict_letter}] {verdict_text}\n\n"
                f"Your Task:\n"
                f"1. Extract the exact, verbatim quote from the Context that directly supports this verdict.\n"
                f"2. Provide a 1-2 sentence rationale explaining why this excerpt justifies the choice.\n\n"
                f"Respond strictly in JSON format matching this schema:\n"
                f"{{\n"
                f'  "verbatim_quote": "<verbatim excerpt from Context>",\n'
                f'  "rationale": "<1-2 sentence explanation>"\n'
                f"}}\n<turn|>\n"
                f"<|turn>model\n"
                f"<|channel>thought\n<channel|>"
                f"```json\n"
            )

            engine.llm.reset()
            engine.llm._ctx.kv_cache_clear()
            tokens = engine.llm.tokenize(audit_prompt.encode("utf-8"), add_bos=True)
            engine.llm.eval(tokens)

            generated = []
            for _ in range(max_tokens):
                tok = engine.llm.sample(temp=temperature)
                if tok in (int(engine.llm.token_eos()), int(engine.llm.tokenize(b"<turn|>")[0])):
                    break
                generated.append(tok)
                engine.llm.eval([tok])

            raw_text = engine.llm.detokenize(generated).decode("utf-8", errors="ignore")

        # Parse JSON
        quote = ""
        rationale = ""
        try:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                quote = parsed.get("verbatim_quote", "")
                rationale = parsed.get("rationale", "")
        except Exception:
            pass

        if not quote and not rationale:
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            rationale = clean[:250]
            quote = f"Derived from verdict [{verdict_letter}]"

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return DecisionEvidence(
            verdict_letter=verdict_letter,
            quote=quote,
            rationale=rationale,
            generation_latency_ms=round(latency_ms, 2),
        )
