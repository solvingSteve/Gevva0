from __future__ import annotations
import base64
import ctypes
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler, Jinja2ChatFormatter
import numpy as np

from .calibration import TemperatureCalibrator, softmax
from .debias import debiased_cyclic_evaluate
from .config import (
    DEFAULT_COT_MAX_TOKENS,
    DEFAULT_COT_PROMPT,
    DEFAULT_COT_TEMP,
    DEFAULT_COT_THRESHOLD,
    DEFAULT_KV_BRANCHING,
    DEFAULT_KV_BRANCHING_MIN_TOKENS,
    LETTERS,
    _load_llm_config,
    detect_mmproj_variant,
    PROJECT_ROOT,
    resolve_calibration_path,
    resolve_cot_prompt,
    resolve_llm_kwargs,
    resolve_mmproj_path,
    resolve_model_path,
    resolve_n_ctx,
)
from .schema import DecisionEvidence, DecisionResponse


def load_image_bytes(image: str | bytes | Path) -> bytes:
    """Resolve an image from bytes, Path, file path, data URI, URL, or raw base64."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, Path):
        return image.read_bytes()
    if isinstance(image, str):
        image_str = image.strip()
        if image_str.startswith("data:"):
            # Data URI: data:image/png;base64,....
            parts = image_str.split(",", 1)
            b64_data = parts[1] if len(parts) > 1 else parts[0]
            return base64.b64decode(b64_data)
        if image_str.startswith("http://") or image_str.startswith("https://"):
            req = urllib.request.Request(image_str, headers={"User-Agent": "Gevva0-Vision/1.0"})
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        candidate_paths = [
            Path(image_str),
            PROJECT_ROOT / image_str,
            PROJECT_ROOT / "benchmarks" / image_str,
            PROJECT_ROOT / "tests" / image_str,
            PROJECT_ROOT / "benchmarks" / "fixtures" / Path(image_str).name,
            PROJECT_ROOT / "fixtures" / Path(image_str).name,
        ]
        for p_cand in candidate_paths:
            try:
                if p_cand.is_file():
                    return p_cand.read_bytes()
            except Exception:
                pass
        # Fallback: check if it is raw base64 string
        try:
            decoded = base64.b64decode(image_str)
            if len(decoded) > 8 and (
                decoded.startswith(b"\x89PNG")
                or decoded.startswith(b"\xff\xd8\xff")
                or decoded.startswith(b"GIF8")
                or decoded.startswith(b"RIFF")
                or decoded.startswith(b"BM")
            ):
                return decoded
        except Exception:
            pass
    raise ValueError(f"Unable to load image from input: {str(image)[:80]}")


_UNSET = object()


class GemmaDecisionEngine:
    def __init__(
        self,
        model_path: str | None = None,
        mmproj_path: str | Path | None | object = _UNSET,
        n_ctx: int | None = None,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        verbose: bool = False,
        calibrator: TemperatureCalibrator | None = None,
        n_batch: int | None = None,
        n_ubatch: int | None = None,
        type_k: int | None = None,
        type_v: int | None = None,
    ):
        self.model_path = str(resolve_model_path(model_path))
        if mmproj_path is _UNSET:
            from .config import get_active_profile
            prof = get_active_profile()
            model_sec = prof.get("model", {})
            if "mmproj_path" in model_sec and (
                model_sec["mmproj_path"] is None or str(model_sec["mmproj_path"]).lower() in ("none", "null", "false", "")
            ):
                resolved_mmproj = None
            else:
                resolved_mmproj = resolve_mmproj_path(self.model_path)
        elif mmproj_path is not None and str(mmproj_path).lower() not in ("none", "null", "false", ""):
            resolved_mmproj = resolve_mmproj_path(mmproj_path)
        else:
            resolved_mmproj = None

        self.mmproj_path = str(resolved_mmproj) if resolved_mmproj else None
        self.mmproj_variant = detect_mmproj_variant(self.mmproj_path) if self.mmproj_path else None
        self.n_ctx = n_ctx or resolve_n_ctx()
        self._lock = threading.RLock()
        self.chat_handler: Gemma4ChatHandler | None = None

        # Prompt caching state (Solution A: KV-cache prefix retention)
        self._cached_prefix_tokens: list[int] | None = None
        self._cached_prefix_len: int = 0
        self._prefix_ingested: bool = False

        print(f"[config] context size: n_ctx={self.n_ctx}")
        print(f"Loading Gemma 4 from {self.model_path} ...")

        # Initialize multimodal projector handler if available
        if self.mmproj_path and Path(self.mmproj_path).is_file():
            try:
                variant_tag = f" [{self.mmproj_variant}]" if self.mmproj_variant else ""
                print(f"[vision] attaching Gemma 4 multimodal projector{variant_tag} from {self.mmproj_path}")
                self.chat_handler = Gemma4ChatHandler(clip_model_path=self.mmproj_path)
            except Exception as exc:
                print(f"[vision] warning: failed to initialize Gemma4ChatHandler ({exc}); running text-only")
                self.chat_handler = None
        else:
            print("[vision] multimodal projector disabled; running text-only")

        # Build optional kwargs only when a value was provided
        extra_kwargs: dict = {}
        if n_batch is not None:
            extra_kwargs["n_batch"] = n_batch
        if n_ubatch is not None:
            extra_kwargs["n_ubatch"] = n_ubatch

        # Default to native FP16 (GGML_TYPE_F16 = 1) for KV cache if not explicitly specified
        resolved_type_k = 1 if type_k is None else type_k
        resolved_type_v = 1 if type_v is None else type_v
        extra_kwargs["type_k"] = resolved_type_k
        extra_kwargs["type_v"] = resolved_type_v
        
        import llama_cpp.llama_cpp as _llama_cpp_mod
        orig_default_params = _llama_cpp_mod.llama_context_default_params

        def _custom_default_params():
            p = orig_default_params()
            try:
                max_parallel = int(_llama_cpp_mod.llama_max_parallel_sequences())
                p.n_seq_max = min(32, max_parallel)
            except Exception:
                p.n_seq_max = 8
            p.kv_unified = True
            return p

        _llama_cpp_mod.llama_context_default_params = _custom_default_params
        try:
            try:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    logits_all=True,
                    flash_attn=flash_attn,
                    verbose=verbose,
                    chat_handler=self.chat_handler,
                    **extra_kwargs,
                )
            except Exception as exc:
                if flash_attn:
                    print(f"flash_attn init failed ({exc}); retrying without flash_attn")
                    self.llm = Llama(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        logits_all=True,
                        flash_attn=False,
                        verbose=verbose,
                        chat_handler=self.chat_handler,
                        **extra_kwargs,
                    )
                else:
                    raise
        finally:
            _llama_cpp_mod.llama_context_default_params = orig_default_params

        # Warm up multimodal context if handler is present
        if self.chat_handler is not None:
            try:
                self.chat_handler._init_mtmd_context(self.llm)
                print(f"[vision] multimodal projector context initialized successfully (CUDA0)")
            except Exception as exc:
                print(f"[vision] warning: mtmd projector warmup failed ({exc}) - model may have dimension mismatch")
                self.chat_handler = None

        self.calibrator = calibrator
        self.warmup()

    def warmup(self) -> None:
        """Warm up CUDA kernels and KV cache so the first request does not suffer cold start latency."""
        try:
            with self._lock:
                bos = int(self.llm.token_bos())
                self.llm.reset()
                self.llm._ctx.kv_cache_clear()
                self.llm.eval([bos])
                self.llm.reset()
                self.llm._ctx.kv_cache_clear()
        except Exception as exc:
            print(f"[engine] warmup note: {exc}")

    def _ensure_vision_handler(self) -> bool:
        """Ensure multimodal projector is attached, lazy-loading if available."""
        if self.has_vision:
            return True

        target_mmproj = self.mmproj_path or resolve_mmproj_path(self.model_path)
        if target_mmproj and Path(target_mmproj).is_file():
            try:
                self.mmproj_path = str(target_mmproj)
                self.mmproj_variant = detect_mmproj_variant(self.mmproj_path)
                variant_tag = f" [{self.mmproj_variant}]" if self.mmproj_variant else ""
                print(f"[vision] lazy-attaching Gemma 4 multimodal projector{variant_tag} from {self.mmproj_path}")
                self.chat_handler = Gemma4ChatHandler(clip_model_path=self.mmproj_path)
                self.chat_handler._init_mtmd_context(self.llm)
                print(f"[vision] multimodal projector initialized successfully on-demand")
                return True
            except Exception as exc:
                print(f"[vision] warning: failed to lazy-load multimodal projector ({exc})")
                self.chat_handler = None
                return False
        return False

    def __enter__(self) -> "GemmaDecisionEngine":
        return self

    def __exit__(self, *exc_info) -> None:
        pass

    @property
    def has_vision(self) -> bool:
        return self.chat_handler is not None and getattr(self.chat_handler, "mtmd_ctx", None) is not None

    @classmethod
    def from_config(
        cls,
        model_path: str | None = None,
        mmproj_path: str | Path | None | object = _UNSET,
        load_calibration: bool = True,
        calibration_path: str | Path | None = None,
        **kwargs_override,
    ) -> "GemmaDecisionEngine":
        calibrator = None
        if load_calibration:
            cal_path = Path(calibration_path) if calibration_path else resolve_calibration_path()
            if cal_path.exists():
                calibrator = TemperatureCalibrator.load(cal_path)
                bias_info = f", bias_offsets={len(calibrator.bias_offsets)}" if calibrator.bias_offsets else ""
                print(f"Loaded calibrator (T={calibrator.temperature:.4f}{bias_info}) from {cal_path}")

        kwargs = resolve_llm_kwargs()
        if model_path:
            kwargs["model_path"] = str(resolve_model_path(model_path))
            if mmproj_path is not _UNSET:
                kwargs["mmproj_path"] = str(mmproj_path) if (mmproj_path and str(mmproj_path).lower() not in ("none", "null", "false", "")) else None
            elif kwargs.get("mmproj_path") is None:
                pass
            else:
                resolved = resolve_mmproj_path(kwargs["model_path"])
                kwargs["mmproj_path"] = str(resolved) if resolved else None
        elif mmproj_path is not _UNSET:
            kwargs["mmproj_path"] = str(mmproj_path) if (mmproj_path and str(mmproj_path).lower() not in ("none", "null", "false", "")) else None
        kwargs["calibrator"] = calibrator
        kwargs.update(kwargs_override)
        return cls(**kwargs)

    def close(self) -> None:
        """Release LLM context, vision projector, and GPU memory."""
        with getattr(self, "_lock", threading.RLock()):
            if hasattr(self, "llm") and self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
                self.llm = None
            self.chat_handler = None

    def __enter__(self) -> "GemmaDecisionEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def _last_logits(self) -> np.ndarray:
        return np.asarray(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float64)

    def _candidate_token_ids(self, labels: list[str]) -> list[int]:
        return [
            int(self.llm.tokenize(f" {c}".encode("utf-8"), add_bos=False)[0])
            for c in labels
        ]

    def _extract_candidate_logits_from_scores(self, last_logits: np.ndarray, labels: list[str]) -> np.ndarray:
        """Extract candidate letter logits from a 1D logit array, marginalizing over leading space and no-space tokens."""
        logits = []
        for c in labels:
            tid_sp = int(self.llm.tokenize(f" {c}".encode("utf-8"), add_bos=False)[0])
            tid_no = int(self.llm.tokenize(c.encode("utf-8"), add_bos=False)[0])
            l_sp = last_logits[tid_sp]
            l_no = last_logits[tid_no]
            # logsumexp over space and non-space token representations
            max_l = max(l_sp, l_no)
            combined = max_l + np.log(np.exp(l_sp - max_l) + np.exp(l_no - max_l))
            logits.append(combined)
        return np.asarray(logits, dtype=np.float64)

    def _extract_candidate_logits(self, labels: list[str]) -> np.ndarray:
        """Extract candidate letter logits, marginalizing over leading space and no-space tokens."""
        return self._extract_candidate_logits_from_scores(self._last_logits, labels)

    def _render_prefix(self, context: str) -> str:
        clean_context = context.split("\nOptions:\n")[0].rstrip() if "\nOptions:\n" in context else context
        return (
            f"<|turn>user\n"
            f"You are an expert classification and diagnostic decision engine.\n"
            f"- Identify the true primary intent or root cause rather than past history or secondary symptoms.\n"
            f"- Base your answer strictly on the facts provided in the context.\n\n"
            f"Context:\n{clean_context}\n\n"
        )

    def _render_suffix(self, options: dict[str, str]) -> str:
        rendered_opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        return (
            f"Options:\n{rendered_opts}\n\n"
            f"Which option is correct? Reply with ONLY the letter of the correct option.<turn|>\n"
            f"<|turn>model\n"
            f"<|channel>thought\n<channel|>"
        )

    def _render_prompt(self, context: str, options: dict[str, str]) -> str:
        return self._render_prefix(context) + self._render_suffix(options)

    # ------------------------------------------------------------------
    # Solution A: KV-cache prefix retention (prompt caching)
    # ------------------------------------------------------------------

    def cache_prefix(self, static_text: str) -> int:
        """Pre-cache a static context prefix into the KV cache.

        Tokenizes the static portion of the prompt (system instruction + policy
        directive) and evaluates it once.  Subsequent ``decide_cached()`` calls
        only need to evaluate the short dynamic suffix (~100 tokens per tick).

        Args:
            static_text: The unchanging policy/directive text (e.g. the
                governance rules from ``optimal-play-prompt-full.txt``).

        Returns:
            The number of cached prefix tokens (*K*).
        """
        prefix_str = (
            f"<|turn>user\n"
            f"You are an expert classification and diagnostic decision engine.\n"
            f"- Identify the true primary intent or root cause rather than past history or secondary symptoms.\n"
            f"- Base your answer strictly on the facts provided in the context.\n\n"
            f"Context:\n{static_text}\n"
        )
        with self._lock:
            tokens = self.llm.tokenize(
                prefix_str.encode("utf-8"), add_bos=True, special=True
            )
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            self.llm.eval(tokens)

            self._cached_prefix_tokens = tokens
            self._cached_prefix_len = len(tokens)
            self._prefix_ingested = True

            print(f"[cache] static prefix cached: {self._cached_prefix_len} tokens in KV slot 0")
            return self._cached_prefix_len

    def invalidate_prefix_cache(self) -> None:
        """Mark the cached prefix KV state as stale (e.g. after model switch)."""
        self._prefix_ingested = False

    def _ensure_prefix_ingested(self) -> None:
        """Re-ingest the cached prefix tokens if the KV cache was invalidated."""
        if self._cached_prefix_tokens is not None and not self._prefix_ingested:
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            self.llm.eval(self._cached_prefix_tokens)
            self._prefix_ingested = True

    def score_cyclic_branching(
        self,
        prefix_tokens: list[int],
        option_suffixes: list[list[int]],
        labels: list[str] | None = None,
        use_calibrator: bool | None = None,
        return_probs: bool = True,
    ) -> np.ndarray | list[dict]:
        """
        Evaluates cyclic option shifts by sharing the ingested prefix KV cache
        across distinct sequence slots (seq_id 1..N-1).

        Lifecycle: Fork -> Forward -> Discard
        1. Ingest shared context once on base sequence (seq_id = 0)
        2. Deep-copy the cached KV cells to distinct isolated sequence IDs
        3. Evaluate each permuted suffix on its isolated sequence branch (single micro-batch if fits)
        4. Purge forked sequences completely (never leave trailing tokens)
        """
        K = len(prefix_tokens)
        N = len(option_suffixes)
        if labels is None:
            labels = list(LETTERS[:N])

        with self._lock:
            # 1. Ingest shared context once on base sequence (seq_id = 0)
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            for s in range(N):
                self.llm._ctx.kv_cache_seq_rm(s, 0, -1)
            self.llm.eval(prefix_tokens)

            try:
                # 2. Deep-copy the cached KV cells to distinct isolated sequence IDs
                for s in range(1, N):
                    self.llm._ctx.kv_cache_seq_cp(0, s, 0, -1)

                branch_logits_list: list[np.ndarray] = []
                total_suffix_tokens = sum(len(st) for st in option_suffixes)

                # 3. Evaluate each permuted suffix on its isolated sequence branch
                if total_suffix_tokens <= self.llm.n_batch:
                    # Batch Forward Pass: evaluate concurrently in a single micro-batch
                    print(f"[KV-Branching] Sequence 0 prefilled. Forked to sequences 1..{N-1} via kv_cache_seq_cp. Micro-batch forward evaluated.")
                    self.llm._batch.reset()
                    self.llm._batch.batch.n_tokens = total_suffix_tokens

                    idx = 0
                    last_indices: list[int] = []
                    for s in range(N):
                        st = option_suffixes[s]
                        n_s = len(st)
                        for i, tok in enumerate(st):
                            self.llm._batch.batch.token[idx] = tok
                            self.llm._batch.batch.pos[idx] = K + i
                            self.llm._batch.batch.seq_id[idx][0] = s
                            self.llm._batch.batch.n_seq_id[idx] = 1
                            is_last = (i == n_s - 1)
                            self.llm._batch.batch.logits[idx] = is_last
                            if is_last:
                                last_indices.append(idx)
                            idx += 1

                    self.llm._ctx.decode(self.llm._batch)

                    for s in range(N):
                        logits_ptr = self.llm._ctx.get_logits_ith(last_indices[s])
                        last_logits = np.ctypeslib.as_array(
                            logits_ptr, shape=(self.llm.n_vocab(),)
                        ).astype(np.float64)
                        branch_c_logits = self._extract_candidate_logits_from_scores(last_logits, labels)
                        branch_logits_list.append(branch_c_logits)
                else:
                    # Suffixes exceed single micro-batch: evaluate branch-by-branch
                    print(f"[KV-Branching] Suffix tokens ({total_suffix_tokens}) exceeded n_batch ({self.llm.n_batch}). Fallback to branch-by-branch evaluation.")
                    for s in range(N):
                        st = option_suffixes[s]
                        for chunk_start in range(0, len(st), self.llm.n_batch):
                            chunk = st[chunk_start : min(len(st), chunk_start + self.llm.n_batch)]
                            self.llm._batch.reset()
                            self.llm._batch.batch.n_tokens = len(chunk)
                            for i, tok in enumerate(chunk):
                                abs_pos = chunk_start + i
                                self.llm._batch.batch.token[i] = tok
                                self.llm._batch.batch.pos[i] = K + abs_pos
                                self.llm._batch.batch.seq_id[i][0] = s
                                self.llm._batch.batch.n_seq_id[i] = 1
                                self.llm._batch.batch.logits[i] = (abs_pos == len(st) - 1)
                            self.llm._ctx.decode(self.llm._batch)
                        logits_ptr = self.llm._ctx.get_logits_ith(len(chunk) - 1)
                        last_logits = np.ctypeslib.as_array(
                            logits_ptr, shape=(self.llm.n_vocab(),)
                        ).astype(np.float64)
                        branch_c_logits = self._extract_candidate_logits_from_scores(last_logits, labels)
                        branch_logits_list.append(branch_c_logits)

            finally:
                # 4. CRITICAL: Purge forked sequences and base sequence completely
                for s in range(N):
                    self.llm._ctx.kv_cache_seq_rm(s, 0, -1)
                self.llm._ctx.kv_cache_clear()
                self.llm.reset()

            apply_calibrator = self.calibrator is not None and use_calibrator is not False

            if not return_probs:
                results = []
                for s in range(N):
                    c_logits = branch_logits_list[s]
                    if apply_calibrator:
                        c_logits = self.calibrator.transform_logits(c_logits, labels)
                    p = softmax(c_logits)
                    results.append({
                        "branch": s,
                        "logits": c_logits,
                        "probabilities": {labels[i]: float(p[i]) for i in range(len(labels))},
                    })
                return results

            # Compute marginalized probabilities mapped back to original options
            aggregated_probs = np.zeros(N, dtype=np.float64)
            for s in range(N):
                c_logits = branch_logits_list[s]
                if apply_calibrator:
                    c_logits = self.calibrator.transform_logits(c_logits, labels)
                p = softmax(c_logits)
                for orig_idx in range(N):
                    perm_idx = (orig_idx - s) % N
                    aggregated_probs[orig_idx] += p[perm_idx]

            aggregated_probs /= N
            norm_sum = np.sum(aggregated_probs)
            if norm_sum > 0:
                aggregated_probs /= norm_sum
            return aggregated_probs

    def _candidate_logprobs(self, prompt: str, labels: list[str]) -> np.ndarray:
        self.llm.reset()
        self.llm._ctx.kv_cache_clear()
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        self.llm.eval(tokens)
        return self._extract_candidate_logits(labels)

    def _eval_multimodal(
        self,
        context: str,
        options: dict[str, str],
        image_bytes_list: list[bytes],
    ) -> None:
        if not self.has_vision:
            self._ensure_vision_handler()
        if not self.has_vision:
            raise RuntimeError(
                "Multimodal vision projector is not available. "
                "Ensure Gemma 4 26B is active and models/mmproj-BF16.gguf exists."
            )

        handler = self.chat_handler
        assert handler is not None
        handler._init_mtmd_context(self.llm)

        rendered_opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt_text = (
            "You are an expert visual classification and diagnostic decision engine.\n"
            "- Inspect the image thoroughly to determine the correct classification, defect, or action.\n"
            "- Base your answer strictly on the visual information in the image and any context provided.\n\n"
        )
        if context and context.strip():
            prompt_text += f"Context:\n{context.strip()}\n\n"
        prompt_text += (
            f"Options:\n{rendered_opts}\n\n"
            f"Which option is correct? Reply with ONLY the letter of the correct option."
        )

        user_content: list[dict] = []
        for img_bytes in image_bytes_list:
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}})
        user_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": user_content}]

        image_urls = handler.get_image_urls(messages)
        media_marker = handler._mtmd_cpp.mtmd_default_marker().decode("utf-8")

        template_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[
                Jinja2ChatFormatter.IgnoreGenerationTags,
                jinja2.ext.loopcontrols,
            ],
        )
        template_env.filters["tojson"] = Jinja2ChatFormatter.tojson
        template = template_env.from_string(handler._get_chat_template(self.llm))

        text = template.render(
            messages=handler._get_template_messages(messages, media_marker),
            add_generation_prompt=True,
            eos_token=handler._decode_token_piece(self.llm.detokenize([self.llm.token_eos()])),
            bos_token=handler._decode_token_piece(self.llm.detokenize([self.llm.token_bos()])),
        )
        text = handler._postprocess_template_text(text, image_urls, media_marker)

        bitmaps = []
        bitmap_cleanup = []
        try:
            for url in image_urls:
                b = handler.load_image(url)
                bmp = handler._create_bitmap_from_bytes(b)
                bitmaps.append(bmp)
                bitmap_cleanup.append(bmp)

            input_text = handler._mtmd_cpp.mtmd_input_text()
            input_text_bytes = text.encode("utf-8")
            input_text.text = input_text_bytes
            input_text.text_len = len(input_text_bytes)
            input_text.add_special = True
            input_text.parse_special = True

            chunks = handler._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to initialize mtmd input chunks")

            try:
                bitmap_array = (handler._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                res = handler._mtmd_cpp.mtmd_tokenize(
                    handler.mtmd_ctx,
                    chunks,
                    ctypes.byref(input_text),
                    bitmap_array,
                    len(bitmaps),
                )
                if res != 0:
                    raise ValueError(f"mtmd_tokenize failed with code {res}")

                self.llm.reset()
                self.llm._ctx.kv_cache_clear()

                n_chunks = handler._mtmd_cpp.mtmd_input_chunks_size(chunks)
                for i in range(n_chunks):
                    chunk = handler._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None:
                        continue
                    ctype = handler._mtmd_cpp.mtmd_input_chunk_get_type(chunk)
                    if ctype == handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        n_out = ctypes.c_size_t()
                        toks_ptr = handler._mtmd_cpp.mtmd_input_chunk_get_tokens_text(chunk, ctypes.byref(n_out))
                        if toks_ptr and n_out.value > 0:
                            tokens = [toks_ptr[j] for j in range(n_out.value)]
                            if self.llm.n_tokens + len(tokens) > self.llm.n_ctx():
                                raise ValueError(f"Prompt exceeds n_ctx: {self.llm.n_tokens + len(tokens)} > {self.llm.n_ctx()}")
                            self.llm.eval(tokens)
                    elif ctype in [handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE, handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                        chunk_n_tokens = handler._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                        if self.llm.n_tokens + chunk_n_tokens > self.llm.n_ctx():
                            raise ValueError(f"Image chunks exceed n_ctx: {self.llm.n_tokens + chunk_n_tokens} > {self.llm.n_ctx()}")
                        new_n_past = llama_cpp.llama_pos()
                        res = handler._mtmd_cpp.mtmd_helper_eval_chunk_single(
                            handler.mtmd_ctx,
                            self.llm._ctx.ctx,
                            chunk,
                            llama_cpp.llama_pos(self.llm.n_tokens),
                            llama_cpp.llama_seq_id(0),
                            self.llm.n_batch,
                            False,
                            ctypes.byref(new_n_past),
                        )
                        if res != 0:
                            raise ValueError(f"mtmd_helper_eval_chunk_single failed with code {res}")
                        self.llm.n_tokens = new_n_past.value
            finally:
                handler._mtmd_cpp.mtmd_input_chunks_free(chunks)
        finally:
            for bmp in bitmap_cleanup:
                handler._mtmd_cpp.mtmd_bitmap_free(bmp)

    def score(self, context: str, options: dict[str, str]) -> np.ndarray:
        with self._lock:
            return self._candidate_logprobs(self._render_prompt(context, options), list(options.keys()))

    def _fast_path_evaluate(
        self,
        context: str,
        options: dict[str, str],
        use_calibrator: bool | None = None,
        image_bytes_list: list[bytes] | None = None,
    ) -> dict:
        labels = list(options.keys())
        if not (2 <= len(labels) <= len(LETTERS)):
            raise ValueError(f"options count must be between 2 and {len(LETTERS)}")
        if len(set(options.values())) != len(options):
            raise ValueError("option texts must be unique")

        start_t = time.perf_counter()
        if image_bytes_list:
            self._eval_multimodal(context, options, image_bytes_list)
            raw_logits = self._extract_candidate_logits(labels)
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
        else:
            prompt = self._render_prompt(context, options)
            raw_logits = self._candidate_logprobs(prompt, labels)

        apply_calibrator = self.calibrator is not None and use_calibrator is not False
        if apply_calibrator:
            raw_logits = self.calibrator.transform_logits(raw_logits, labels)
        probs = softmax(raw_logits)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_letter = labels[top_idx]

        return {
            "letter": top_letter,
            "decision": options[top_letter],
            "confidence": round(top_conf, 4),
            "latency_ms": round(latency_ms, 2),
            "probabilities": {options[k]: round(float(p), 4) for k, p in zip(labels, probs)},
            "letter_probabilities": {k: float(p) for k, p in zip(labels, probs)},
        }

    def explain_decision(
        self,
        context: str,
        verdict_letter: str,
        verdict_text: str,
        image_bytes_list: list[bytes] | None = None,
        max_tokens: int = 192,
        temperature: float = 0.1,  # Keep low to prevent hallucinating quotes
    ) -> DecisionEvidence:
        """
        Extracts verbatim source quotes and justification AFTER the decision is locked.
        Runs as an isolated prompt pass to guarantee zero logit contamination.
        """
        start_t = time.perf_counter()

        with self._lock:
            if image_bytes_list and not self.has_vision:
                self._ensure_vision_handler()
            if image_bytes_list and self.has_vision:
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
                    f'  "quote": "verbatim text or specific visual element from image",\n'
                    f'  "rationale": "one or two sentence explanation"\n'
                    f"}}\n"
                    f"Output ONLY valid JSON."
                )
                msg_content: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in b64_urls]
                msg_content.append({"type": "text", "text": audit_prompt})
                chat_res = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": msg_content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw_output = chat_res["choices"][0]["message"].get("content", "").strip()
            else:
                explanation_prompt = (
                    f"user\n"
                    f"You are an evidence auditing engine. A prior diagnostic system determined that "
                    f"Option [{verdict_letter}] is the correct classification for the context below.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Selected Decision: [{verdict_letter}] {verdict_text}\n\n"
                    f"Your Task:\n"
                    f"1. Locate and extract the exact, verbatim quote from the context that justifies this decision.\n"
                    f"2. Provide a 1-2 sentence operational rationale.\n\n"
                    f"Respond strictly in JSON format matching this schema:\n"
                    f"{{\n"
                    f'  "quote": "verbatim sentence or phrase from the context",\n'
                    f'  "rationale": "one or two sentence explanation"\n'
                    f"}}\n"
                    f"Output ONLY valid JSON.\n"
                    f"model\n"
                )

                tokens = self.llm.tokenize(explanation_prompt.encode("utf-8"), add_bos=True)
                self.llm.reset()
                self.llm.eval(tokens)

                generated_tokens: list[int] = []
                eos = int(self.llm.token_eos())

                for _ in range(max_tokens):
                    tok = self.llm.sample(temp=temperature)
                    if tok == eos:
                        break
                    generated_tokens.append(tok)
                    self.llm.eval([tok])

                raw_output = self.llm.detokenize(generated_tokens).decode("utf-8", errors="ignore").strip()

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        # Parse JSON extraction with robust fallback
        try:
            cleaned_json = raw_output
            if "```" in cleaned_json:
                cleaned_json = cleaned_json.split("```")[1]
                if cleaned_json.startswith("json"):
                    cleaned_json = cleaned_json[4:]
                cleaned_json = cleaned_json.strip()

            json_match = re.search(r"\{.*\}", cleaned_json, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group(0)

            parsed = json.loads(cleaned_json)
            quote = parsed.get("quote", "Visual feature or verbatim quote extracted.")
            rationale = parsed.get("rationale", raw_output)
        except Exception:
            quote = "Extracted from input context/image (see rationale)."
            rationale = raw_output

        return DecisionEvidence(
            verdict_letter=verdict_letter,
            quote=quote,
            rationale=rationale,
            generation_latency_ms=round(latency_ms, 2),
        )

    def _run_cot_verification(
        self,
        context: str,
        options: dict[str, str],
        labels: list[str],
        use_calibrator: bool | None = None,
        cot_temp: float = DEFAULT_COT_TEMP,
        max_cot_tokens: int = DEFAULT_COT_MAX_TOKENS,
        cot_prompt: str | None = None,
    ) -> tuple[str, str, np.ndarray, str]:
        """Runs bounded CoT verification on ambiguous or emergency decisions."""
        apply_calibrator = self.calibrator is not None and use_calibrator is not False
        prompt = self._render_prompt(context, options)
        self.llm.reset()
        self.llm._ctx.kv_cache_clear()
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        self.llm.eval(tokens)

        prefix = cot_prompt if cot_prompt is not None else resolve_cot_prompt()
        analysis = prefix if prefix.startswith("\n") else f"\n{prefix}"
        self.llm.eval(self.llm.tokenize(analysis.encode("utf-8"), add_bos=False, special=True))

        generated: list[int] = []
        eos = int(self.llm.token_eos())
        stop_tokens = {eos}
        for special_str in ["<turn|>", "<end_of_turn>", "<|turn>", "<channel|>", "<|channel>"]:
            st = self.llm.tokenize(special_str.encode("utf-8"), add_bos=False, special=True)
            if st:
                stop_tokens.add(int(st[0]))
        for _ in range(max_cot_tokens):
            next_tok = self.llm.sample(temp=cot_temp)
            if next_tok in stop_tokens:
                break
            generated.append(next_tok)
            self.llm.eval([next_tok])
            if len(generated) >= 10:
                # Terminate on double newline (\n\n) or EOS/stop tokens, allowing multi-line scratchpad derivations
                recent = self.llm.detokenize(generated[-6:]).decode("utf-8", errors="ignore")
                if "\n\n" in recent or "\r\n\r\n" in recent:
                    break

        cot_reasoning = self.llm.detokenize(generated).decode("utf-8", errors="ignore")
        conclusion_cue = "\nTherefore, the correct option letter is: "
        self.llm.eval(self.llm.tokenize(conclusion_cue.encode("utf-8"), add_bos=False, special=True))
        final_logits = self._extract_candidate_logits(labels)
        if apply_calibrator:
            final_logits = self.calibrator.transform_logits(final_logits, labels)
        post_probs = softmax(final_logits)
        new_top = int(np.argmax(post_probs))
        new_letter = labels[new_top]
        self.llm.reset()
        self.llm._ctx.kv_cache_clear()
        return new_letter, options[new_letter], post_probs, cot_reasoning.strip()

    def decide(
        self,
        context: str = "",
        options: dict[str, str] = ...,
        image: str | bytes | Path | list[str | bytes | Path] | None = None,
        cot_threshold: float | None = None,
        max_cot_tokens: int = DEFAULT_COT_MAX_TOKENS,
        cot_temp: float = DEFAULT_COT_TEMP,
        cot_prompt: str | None = None,
        use_calibrator: bool | None = None,
        cyclic_debias: bool = False,
        kv_branching: bool | None = None,
        kv_branching_min_tokens: int | None = None,
        include_evidence: bool = False,
        test_id: str | None = None,
        **kwargs,
    ) -> DecisionResponse:
        """
        Main entrypoint. Evaluates options against context and/or images.
        Always executes fast-path logit evaluation first.
        Optionally triggers on-demand evidence extraction or CoT verification.
        """
        if cot_threshold is None:
            cot_threshold = DEFAULT_COT_THRESHOLD
        if kv_branching is None or kv_branching_min_tokens is None:
            decision_cfg = (_load_llm_config() or {}).get("decision", {})
            if kv_branching is None:
                kv_branching = decision_cfg.get("kv_branching", DEFAULT_KV_BRANCHING)
            if kv_branching_min_tokens is None:
                kv_branching_min_tokens = decision_cfg.get("kv_branching_min_tokens", DEFAULT_KV_BRANCHING_MIN_TOKENS)

        # Parse image(s) if provided
        image_bytes_list: list[bytes] = []
        if image is not None:
            if isinstance(image, (list, tuple)):
                image_bytes_list = [load_image_bytes(img) for img in image if img is not None]
            else:
                image_bytes_list = [load_image_bytes(image)]

        # Deterministic short-circuit: if only 1 option is present, no inference is required
        if len(options) == 1:
            only_key = next(iter(options.keys()))
            only_text = options[only_key]
            return DecisionResponse(
                test_id=test_id,
                decision=only_key,
                decision_text=only_text,
                confidence=1.0,
                latency_ms=0.01,
                duration_ms=0.01,
                candidate_letter=only_key,
                probabilities={only_text: 1.0},
                letter_probabilities={only_key: 1.0},
                mode="forced_move",
                escalate=False,
                evidence=None,
                cot_scratchpad="Single candidate option provided; executed deterministically.",
                image_attached=bool(image_bytes_list),
            )

        start_t = time.perf_counter()
        with self._lock:
            # 1. Fast-path logit evaluation (15-22ms)
            fast_result = self._fast_path_evaluate(
                context,
                options,
                use_calibrator=use_calibrator,
                image_bytes_list=image_bytes_list if image_bytes_list else None,
            )

            # 2. Check CoT verification escalation
            # Optimization: Bypass cyclic debiasing when CoT is active.
            # Once the model generates a reasoning trace explicitly naming the direction,
            # positional token bias drops close to zero. Running a single forward pass
            # drops CoT latency from ~900ms down to ~120-150ms.
            if cot_threshold > 0.0 and fast_result["confidence"] < cot_threshold and max_cot_tokens > 0 and not image_bytes_list:
                labels = list(options.keys())
                new_letter, new_text, post_probs, cot_reasoning = self._run_cot_verification(
                    context=context,
                    options=options,
                    labels=labels,
                    use_calibrator=use_calibrator,
                    cot_temp=cot_temp,
                    max_cot_tokens=max_cot_tokens,
                    cot_prompt=cot_prompt,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                evidence = None
                if include_evidence:
                    evidence = self.explain_decision(
                        context=context,
                        verdict_letter=new_letter,
                        verdict_text=new_text,
                        image_bytes_list=image_bytes_list if image_bytes_list else None,
                    )
                return DecisionResponse(
                    test_id=test_id,
                    decision=new_letter,
                    decision_text=new_text,
                    confidence=round(float(np.max(post_probs)), 4),
                    latency_ms=round(latency_ms, 2),
                    duration_ms=round(latency_ms, 2),
                    candidate_letter=new_letter,
                    probabilities={options[k]: round(float(p), 4) for k, p in zip(labels, post_probs)},
                    letter_probabilities={k: round(float(p), 4) for k, p in zip(labels, post_probs)},
                    mode="cot_verified",
                    escalate=float(np.max(post_probs)) < 0.60,
                    evidence=evidence,
                    cot_scratchpad=cot_reasoning,
                    image_attached=bool(image_bytes_list),
                )

            # 3. CoT not triggered: fast-path confident.
            # If cyclic_debias is requested, run debiasing on the fast-path.
            if cyclic_debias:
                labels = list(options.keys())
                probs = debiased_cyclic_evaluate(
                    self,
                    context,
                    [options[k] for k in labels],
                    image_bytes_list=image_bytes_list if image_bytes_list else None,
                    use_calibrator=use_calibrator,
                    kv_branching=bool(kv_branching),
                    kv_branching_min_tokens=kv_branching_min_tokens or DEFAULT_KV_BRANCHING_MIN_TOKENS,
                )
                latency_ms = (time.perf_counter() - start_t) * 1000.0
                top_idx = int(np.argmax(probs))
                top_letter = labels[top_idx]
                top_conf = float(probs[top_idx])

                evidence = None
                if include_evidence:
                    evidence = self.explain_decision(
                        context=context,
                        verdict_letter=top_letter,
                        verdict_text=options[top_letter],
                        image_bytes_list=image_bytes_list if image_bytes_list else None,
                    )
                return DecisionResponse(
                    test_id=test_id,
                    decision=top_letter,
                    decision_text=options[top_letter],
                    confidence=round(top_conf, 4),
                    latency_ms=round(latency_ms, 2),
                    duration_ms=round(latency_ms, 2),
                    candidate_letter=top_letter,
                    probabilities={options[k]: round(float(p), 4) for k, p in zip(labels, probs)},
                    letter_probabilities={k: round(float(p), 4) for k, p in zip(labels, probs)},
                    mode="cyclic_debiased",
                    escalate=top_conf < 0.60,
                    evidence=evidence,
                    cot_scratchpad=None,
                    image_attached=bool(image_bytes_list),
                )

            # 4. Standard fast-path without cyclic debiasing
            evidence = None
            if include_evidence:
                evidence = self.explain_decision(
                    context=context,
                    verdict_letter=fast_result["letter"],
                    verdict_text=fast_result["decision"],
                    image_bytes_list=image_bytes_list if image_bytes_list else None,
                )

            return DecisionResponse(
                test_id=test_id,
                decision=fast_result["letter"],
                decision_text=fast_result["decision"],
                confidence=fast_result["confidence"],
                latency_ms=fast_result["latency_ms"],
                duration_ms=fast_result["latency_ms"],
                candidate_letter=fast_result["letter"],
                probabilities=fast_result["probabilities"],
                letter_probabilities=fast_result.get("letter_probabilities"),
                mode="fast_path",
                escalate=fast_result["confidence"] < 0.60,
                evidence=evidence,
                cot_scratchpad=None,
                image_attached=bool(image_bytes_list),
            )
