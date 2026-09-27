"""Cyclic permutation debiasing for logit-based decision gateways.

When evaluating multi-choice options via direct logits, models exhibit position
and label bias (e.g., favoring 'A' regardless of content).

This module runs N cyclical permutations of the option order, computes the logit
distribution for each permutation, maps probabilities back to original semantic options,
and computes an unbiased marginalized distribution.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Sequence, Optional
import numpy as np

if TYPE_CHECKING:
    from .engine import GemmaDecisionEngine

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def debiased_cyclic_evaluate(
    engine: "GemmaDecisionEngine",
    context: str,
    option_texts: Sequence[str],
    image_bytes_list: Optional[Sequence[bytes]] = None,
    use_calibrator: Optional[bool] = None,
    kv_branching: bool = False,
    kv_branching_min_tokens: int = 1500,
) -> np.ndarray:
    """Run cyclic label-permutation debiasing across options.

    Args:
        engine: Instantiated GemmaDecisionEngine
        context: Context string / prompt
        option_texts: Sequence of candidate option descriptions
        image_bytes_list: Optional list of raw image bytes for vision models
        use_calibrator: Whether to apply Platt calibration temperature
        kv_branching: Whether to enable KV-cache branching for cyclic branches
        kv_branching_min_tokens: Minimum context token count to trigger branching

    Returns:
        np.ndarray of debiased probabilities corresponding to option_texts
    """
    n = len(option_texts)
    if n < 2 or n > len(LETTERS):
        raise ValueError(f"Number of options must be between 2 and {len(LETTERS)}")

    letters = list(LETTERS[:n])

    # Ensure context does not contain trailing options block before rendering
    clean_context = context.split("\nOptions:\n")[0].rstrip() if "\nOptions:\n" in context else context

    # Check if KV-cache branching should be used
    if kv_branching and not image_bytes_list and hasattr(engine, "score_cyclic_branching"):
        prefix = engine._render_prefix(clean_context)
        prefix_tokens = engine.llm.tokenize(prefix.encode("utf-8"), add_bos=True, special=True)
        if len(prefix_tokens) >= kv_branching_min_tokens:
            suffixes = []
            for shift in range(n):
                permuted_texts = [option_texts[(i + shift) % n] for i in range(n)]
                opts_dict = {letters[i]: permuted_texts[i] for i in range(n)}
                suffix = engine._render_suffix(opts_dict)
                st = engine.llm.tokenize(suffix.encode("utf-8"), add_bos=False, special=True)
                suffixes.append(st)
            return engine.score_cyclic_branching(
                prefix_tokens=prefix_tokens,
                option_suffixes=suffixes,
                labels=letters,
                use_calibrator=use_calibrator,
                return_probs=True,
            )

    aggregated_probs = np.zeros(n, dtype=np.float64)

    for shift in range(n):
        # Strict functional cyclic permutation: shift puts option_texts[(i + shift) % n] at slot i
        permuted_texts = [option_texts[(i + shift) % n] for i in range(n)]
        opts_dict = {letters[i]: permuted_texts[i] for i in range(n)}
        fast_res = engine._fast_path_evaluate(
            context=clean_context,
            options=opts_dict,
            use_calibrator=use_calibrator,
            image_bytes_list=list(image_bytes_list) if image_bytes_list else None,
        )
        # Map letter probability back to original semantic option index
        letter_probs = fast_res.get("letter_probabilities")
        for i, letter in enumerate(letters):
            orig_idx = (i + shift) % n
            if letter_probs is not None:
                prob_val = letter_probs[letter]
            else:
                prob_val = fast_res["probabilities"].get(opts_dict[letter], 0.0)
            aggregated_probs[orig_idx] += prob_val

    aggregated_probs /= n
    norm_sum = np.sum(aggregated_probs)
    if norm_sum > 0:
        aggregated_probs /= norm_sum
    return aggregated_probs
