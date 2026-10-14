"""Asymmetric quote and operational rationale extraction.

Key architectural principle:
Inference justification MUST NOT precede or poison the decision logits.
Audit extraction runs strictly AFTER the categorical decision is locked in a
separate inference pass, ensuring zero logit contamination while extracting
verbatim quotes and human-readable operational justifications.
"""

from __future__ import annotations
import base64
import difflib
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

    @property
    def verbatim_quote(self) -> str:
        return self.quote


def align_verbatim_quote(context: str, quote: str, threshold: float = 0.70) -> str:
    """Ensure an extracted quote matches the source context verbatim.
    
    Corrects LLM generation quirks, BPE token boundary artifacts, and hallucinated
    punctuation/digits (e.g. restoring '$10,000' if the model output '$1,0,0,000').
    """
    if not context or not quote:
        return quote
    if quote in context:
        return quote

    quote_clean = quote.strip().strip('"\'“”')
    if not quote_clean:
        return quote
    if quote_clean in context:
        return quote_clean

    # Try case-insensitive exact match
    lower_context = context.lower()
    lower_quote = quote_clean.lower()
    idx = lower_context.find(lower_quote)
    if idx != -1:
        return context[idx : idx + len(quote_clean)]

    # Word-level phrase anchoring: locate matching starting and ending word sequences
    words = quote_clean.split()
    if len(words) >= 3:
        s_idx = -1
        for n_start in (6, 5, 4, 3, 2):
            if n_start > len(words):
                continue
            start_phrase = " ".join(words[:n_start])
            s_idx = context.find(start_phrase)
            if s_idx == -1:
                s_idx = lower_context.find(start_phrase.lower())
            if s_idx != -1:
                break

        e_idx = -1
        search_start = s_idx if s_idx != -1 else 0
        for n_end in (6, 5, 4, 3, 2):
            if n_end > len(words):
                continue
            end_phrase = " ".join(words[-n_end:])
            e_idx = context.find(end_phrase, search_start)
            if e_idx == -1:
                e_idx = lower_context.find(end_phrase.lower(), search_start)
            if e_idx != -1:
                e_idx += len(end_phrase)
                break

        if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
            candidate = context[s_idx:e_idx]
            ratio = difflib.SequenceMatcher(None, candidate, quote_clean).ratio()
            if ratio >= threshold:
                return candidate

    return quote


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
            # Text-only isolated audit prompt (model-agnostic for Gemma, Qwen, Llama, etc.)
            clean_context = context.strip() if context else "No context text provided."
            audit_prompt = (
                f"Context:\n{clean_context}\n\n"
                f"The system rendered the following verdict:\n"
                f"Verdict: [{verdict_letter}] {verdict_text}\n\n"
                f"Your Task:\n"
                f"1. Locate and extract the exact, verbatim quote from the Context that directly supports this verdict. Copy the excerpt character-for-character, byte-for-byte from the context text (do not modify numbers, currency symbols, or punctuation; e.g. copy '$10,000' exactly as written in the context).\n"
                f"2. Provide a 1-2 sentence rationale explaining why this excerpt justifies the choice.\n\n"
                f"Respond strictly in JSON format matching this schema:\n"
                f"{{\n"
                f'  "verbatim_quote": "<verbatim excerpt copied exactly from Context>",\n'
                f'  "rationale": "<1-2 sentence explanation>"\n'
                f"}}"
            )

            messages = [{"role": "user", "content": audit_prompt}]
            try:
                response = engine.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_text = response["choices"][0]["message"]["content"].strip()
            except Exception:
                # Fallback to direct raw completion if model chat template fails
                engine.llm.reset()
                engine.llm._ctx.kv_cache_clear()
                res = engine.llm(
                    prompt=f"{audit_prompt}\n\n```json\n",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["```", "<turn|>", "<|im_end|>", "\n\n\n"],
                )
                raw_text = res["choices"][0]["text"].strip()
            finally:
                try:
                    engine.llm.reset()
                    engine.llm._ctx.kv_cache_clear()
                except Exception:
                    pass

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

        # Align quote to exact verbatim substring in context to eliminate token drift / extra commas (e.g. $1,0,0,000 -> $10,000)
        if context and quote:
            quote = align_verbatim_quote(context, quote)

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return DecisionEvidence(
            verdict_letter=verdict_letter,
            quote=quote,
            rationale=rationale,
            generation_latency_ms=round(latency_ms, 2),
        )
