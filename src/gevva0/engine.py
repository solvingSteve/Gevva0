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

from .calibration import TemperatureCalibrator, calibrate_model, softmax
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
    find_model_calibration_path,
    PROJECT_ROOT,
    resolve_calibration_path,
    resolve_cot_prompt,
    resolve_llm_kwargs,
    resolve_mmproj_path,
    resolve_model_path,
    resolve_n_ctx,
    update_config_calibration,
)
from .schema import DecisionEvidence, DecisionResponse
from .sparse_logits import (
    SparseLogitProjector,
    is_gemma4_model,
    resolve_sparse_logit_proj_flag,
)


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


class ReentrantCrossThreadLock:
    """Thread-safe re-entrant lock permitting cross-thread release during generator finalization / GC.
    
    Standard threading.RLock raises `RuntimeError: cannot release un-acquired lock`
    if a generator yields mid-stream, is abandoned (e.g. client disconnect / page navigation),
    and is subsequently garbage collected on a different worker/main thread.
    This lock allows re-entrancy on the calling thread while safely permitting cleanup
    from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._count: int = 0
        self._state_lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        with self._state_lock:
            if self._owner == me:
                self._count += 1
                return True

        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            with self._state_lock:
                self._owner = me
                self._count = 1
        return acquired

    def release(self) -> None:
        with self._state_lock:
            if self._count <= 0:
                return
            self._count -= 1
            if self._count == 0:
                self._owner = None
                try:
                    self._lock.release()
                except RuntimeError:
                    pass

    def __enter__(self) -> "ReentrantCrossThreadLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


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
        sparse_logit_proj: bool | str | None = "auto",
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
        self.model_name = Path(self.model_path).name if self.model_path else "model"
        self.n_ctx = n_ctx or resolve_n_ctx()
        self._lock = ReentrantCrossThreadLock()
        self.chat_handler: Gemma4ChatHandler | None = None
        self.tier0 = None
        self._label_tok_cache: dict[str, tuple[int, int]] = {}
        self.sparse_projector: SparseLogitProjector | None = None
        self.is_gemma4: bool = False

        # Prompt caching state (Solution A: KV-cache prefix retention)
        self._cached_prefix_tokens: list[int] | None = None
        self._cached_prefix_len: int = 0
        self._prefix_ingested: bool = False

        print(f"[config] context size: n_ctx={self.n_ctx}")
        print(f"Loading model from {self.model_path} ...")

        # Initialize multimodal projector handler if available
        if self.mmproj_path and Path(self.mmproj_path).is_file():
            try:
                variant_tag = f" [{self.mmproj_variant}]" if self.mmproj_variant else ""
                print(f"[vision] attaching multimodal projector{variant_tag} from {self.mmproj_path}")
                self.chat_handler = Gemma4ChatHandler(clip_model_path=self.mmproj_path)
            except Exception as exc:
                print(f"[vision] warning: failed to initialize multimodal chat handler ({exc}); running text-only")
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
                    logits_all=False,
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
                        logits_all=False,
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

        # Auto-detect Gemma 4 architecture and configure Sparse Logit Projection ("Micro-Head Scoring")
        # Note: Must be activated AFTER warmup() so ggml_backend_sched reserves full-vocabulary
        # compute buffers before shrinking model.output to K=64.
        self.is_gemma4 = is_gemma4_model(self.llm, self.model_path)
        want_sparse = resolve_sparse_logit_proj_flag(
            sparse_logit_proj, llm=self.llm, model_path=self.model_path
        )
        if want_sparse:
            try:
                proj = SparseLogitProjector(self.llm)
                if proj.supported:
                    self.sparse_projector = proj
                    print(
                        f"[sparse-logits] Gemma 4 Micro-Head Scoring enabled "
                        f"(K={proj.k_rows} vs V={proj.n_vocab}, 99.98% output projection reduction)"
                    )
            except Exception as exc:
                print(f"[sparse-logits] warning: failed to initialize Micro-Head projector ({exc}); using full vocab")

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

                # If multimodal vision projector is active, warm up vision encoder & mtmd chunks
                if self.has_vision and self.chat_handler is not None:
                    try:
                        import io
                        from PIL import Image
                        t_vw = time.perf_counter()
                        buf = io.BytesIO()
                        Image.new("RGB", (64, 64), color="gray").save(buf, format="PNG")
                        dummy_bytes = buf.getvalue()
                        self._eval_multimodal(
                            context="warmup",
                            options={"A": "Warmup Option A", "B": "Warmup Option B"},
                            image_bytes_list=[dummy_bytes],
                        )
                        self.llm.reset()
                        self.llm._ctx.kv_cache_clear()
                        vw_ms = (time.perf_counter() - t_vw) * 1000.0
                        print(f"[vision] warmup multimodal forward pass completed ({vw_ms:.1f} ms, CUDA kernels & buffers pre-allocated)")
                    except Exception as v_exc:
                        print(f"[vision] warmup note: {v_exc}")
        except Exception as exc:
            print(f"[engine] warmup note: {exc}")

    def _ensure_vision_handler(self) -> bool:
        """Ensure multimodal projector is attached, lazy-loading if available."""
        if self.has_vision:
            return True

        if not self.mmproj_path or str(self.mmproj_path).lower() in ("none", "null", "false", ""):
            return False

        target_mmproj = self.mmproj_path
        if target_mmproj and Path(target_mmproj).is_file():
            try:
                self.mmproj_path = str(target_mmproj)
                self.mmproj_variant = detect_mmproj_variant(self.mmproj_path)
                variant_tag = f" [{self.mmproj_variant}]" if self.mmproj_variant else ""
                print(f"[vision] lazy-attaching multimodal projector{variant_tag} from {self.mmproj_path}")
                self.chat_handler = Gemma4ChatHandler(clip_model_path=self.mmproj_path)
                self.chat_handler._init_mtmd_context(self.llm)
                print(f"[vision] multimodal projector initialized successfully on-demand")
                return True
            except Exception as exc:
                print(f"[vision] warning: failed to lazy-load multimodal projector ({exc})")
                self.chat_handler = None
                return False
        return False

    @property
    def has_vision(self) -> bool:
        return self.chat_handler is not None and getattr(self.chat_handler, "mtmd_ctx", None) is not None

    def get_tier0(self, auto_load: bool = False):
        """Returns the Tier 0 EmbeddingGemma engine, or None if not loaded."""
        from .tier0 import get_tier0_engine, is_tier0_loaded

        if getattr(self, "tier0", None) is not None:
            if not is_tier0_loaded():
                self.tier0 = None

        if getattr(self, "tier0", None) is None:
            self.tier0 = get_tier0_engine(auto_load=auto_load)
        return self.tier0


    @classmethod
    def from_config(
        cls,
        model_path: str | None = None,
        mmproj_path: str | Path | None | object = _UNSET,
        load_calibration: bool = True,
        calibration_path: str | Path | None = None,
        auto_calibrate_if_missing: bool = True,
        **kwargs_override,
    ) -> "GemmaDecisionEngine":
        resolved_model_path = resolve_model_path(model_path)
        calibrator = None
        cal_path = None
        if load_calibration:
            cal_path = Path(calibration_path) if calibration_path else find_model_calibration_path(resolved_model_path)
            if cal_path and cal_path.is_file():
                try:
                    calibrator = TemperatureCalibrator.load(cal_path)
                    update_config_calibration(calibrator)
                    bias_info = f", bias_offsets={len(calibrator.bias_offsets)}" if calibrator.bias_offsets else ""
                    temps_info = f", cardinality_Ts={calibrator.temperatures}" if calibrator.temperatures else ""
                    print(f"Loaded calibrator (T={calibrator.temperature:.4f}{temps_info}{bias_info}) from {cal_path}")
                except Exception as exc:
                    print(f"[config] warning: failed loading calibrator from {cal_path}: {exc}")
                    calibrator = None

        kwargs = resolve_llm_kwargs()
        if model_path:
            kwargs["model_path"] = str(resolved_model_path)
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
        engine = cls(**kwargs)

        if load_calibration and calibrator is None and auto_calibrate_if_missing:
            target_out = resolve_calibration_path(resolved_model_path)
            print(f"[engine] Calibration JSON not found for {engine.model_name}. Running calibration on Decision Index slice...")
            try:
                cal, out_file = calibrate_model(engine, output_path=target_out, verbose=True)
                engine.calibrator = cal
                update_config_calibration(cal)
                print(f"[engine] Calibrated profile saved to {out_file} (T={cal.temperature:.4f}) and loaded into config.")
            except Exception as exc:
                print(f"[engine] warning: auto-calibration failed ({exc})")

        return engine

    def close(self) -> None:
        """Release LLM context, vision projector, and GPU memory."""
        with getattr(self, "_lock", threading.RLock()):
            if getattr(self, "sparse_projector", None) is not None:
                try:
                    self.sparse_projector.deactivate()
                except Exception:
                    pass
                self.sparse_projector = None
            if hasattr(self, "chat_handler") and self.chat_handler is not None:
                if getattr(self.chat_handler, "mtmd_ctx", None) is not None:
                    try:
                        self.chat_handler._mtmd_cpp.mtmd_free(self.chat_handler.mtmd_ctx)
                    except Exception:
                        pass
                    self.chat_handler.mtmd_ctx = None
                self.chat_handler = None
            if hasattr(self, "llm") and self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
                self.llm = None

    @property
    def sparse_logit_proj_enabled(self) -> bool:
        proj = getattr(self, "sparse_projector", None)
        return proj is not None and proj.active

    def _full_vocab_scope(self):
        proj = getattr(self, "sparse_projector", None)
        if proj is not None:
            return proj.full_vocab_scope()
        import contextlib
        return contextlib.nullcontext()

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
        if getattr(self.llm, "_logits_all", False):
            return np.asarray(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float64)
        ptr = self.llm._ctx.get_logits()
        proj = getattr(self, "sparse_projector", None)
        n = proj.k_rows if (proj is not None and proj.active) else self.llm.n_vocab()
        return np.ctypeslib.as_array(ptr, shape=(n,)).astype(np.float64)

    def _get_label_token_pair(self, label: str) -> tuple[int, int]:
        cache = getattr(self, "_label_tok_cache", None)
        if cache is None:
            cache = {}
            self._label_tok_cache = cache
        if label not in cache:
            tid_sp = int(self.llm.tokenize(f" {label}".encode("utf-8"), add_bos=False)[0])
            tid_no = int(self.llm.tokenize(label.encode("utf-8"), add_bos=False)[0])
            cache[label] = (tid_sp, tid_no)
        return cache[label]

    def _candidate_token_ids(self, labels: list[str]) -> list[int]:
        return [self._get_label_token_pair(c)[0] for c in labels]

    def _extract_candidate_logits_from_scores(self, last_logits: np.ndarray, labels: list[str]) -> np.ndarray:
        """Extract candidate letter logits from a 1D logit array, marginalizing over leading space and no-space tokens."""
        logits = []
        for c in labels:
            tid_sp, tid_no = self._get_label_token_pair(c)
            l_sp = float(last_logits[tid_sp])
            l_no = float(last_logits[tid_no])
            # logsumexp over space and non-space token representations
            max_l = max(l_sp, l_no)
            combined = max_l + np.log(np.exp(l_sp - max_l) + np.exp(l_no - max_l))
            logits.append(combined)
        return np.asarray(logits, dtype=np.float64)

    def _extract_candidate_logits_from_ptr(self, logits_ptr: Any, labels: list[str]) -> np.ndarray:
        """Extract candidate letter logits directly from a C float* logits pointer without full-vocab copy."""
        logits = []
        for c in labels:
            tid_sp, tid_no = self._get_label_token_pair(c)
            l_sp = float(logits_ptr[tid_sp])
            l_no = float(logits_ptr[tid_no])
            max_l = max(l_sp, l_no)
            combined = max_l + np.log(np.exp(l_sp - max_l) + np.exp(l_no - max_l))
            logits.append(combined)
        return np.asarray(logits, dtype=np.float64)

    def _extract_candidate_logits(self, labels: list[str]) -> np.ndarray:
        """Extract candidate letter logits, marginalizing over leading space and no-space tokens."""
        proj = getattr(self, "sparse_projector", None)
        if proj is not None and proj.can_score_labels(labels):
            return proj.extract_marginalized_logits(labels, branch_out_id=0)
        if getattr(self.llm, "_ctx", None) is not None and not getattr(self.llm, "_logits_all", False):
            return self._extract_candidate_logits_from_ptr(self.llm._ctx.get_logits(), labels)
        return self._extract_candidate_logits_from_scores(self._last_logits, labels)

    def _eval_prefix_no_logits(self, tokens: list[int], seq_id: int = 0) -> None:
        """Decode prefix tokens into KV cache without computing output logits on any token."""
        nb = self.llm.n_batch
        batch = self.llm._batch
        for c0 in range(0, len(tokens), nb):
            chunk = tokens[c0 : c0 + nb]
            batch.reset()
            batch.batch.n_tokens = len(chunk)
            for i, tok in enumerate(chunk):
                batch.batch.token[i] = tok
                batch.batch.pos[i] = c0 + i
                batch.batch.seq_id[i][0] = seq_id
                batch.batch.n_seq_id[i] = 1
                batch.batch.logits[i] = 0
            self.llm._ctx.decode(batch)
        self.llm.n_tokens = len(tokens)

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
            self._eval_prefix_no_logits(tokens, seq_id=0)

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
            self._eval_prefix_no_logits(self._cached_prefix_tokens, seq_id=0)
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
            # 1. Ingest shared context once on base sequence (seq_id = 0) without output projection
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            for s in range(N):
                self.llm._ctx.kv_cache_seq_rm(s, 0, -1)
            self._eval_prefix_no_logits(prefix_tokens, seq_id=0)

            try:
                # 2. Deep-copy the cached KV cells to distinct isolated sequence IDs
                for s in range(1, N):
                    self.llm._ctx.kv_cache_seq_cp(0, s, 0, -1)

                branch_logits_list: list[np.ndarray] = []
                total_suffix_tokens = sum(len(st) for st in option_suffixes)
                proj = getattr(self, "sparse_projector", None)
                use_sparse = proj is not None and proj.can_score_labels(labels)

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
                        if use_sparse and proj is not None:
                            out_id = proj.get_branch_output_id(last_indices[s])
                            branch_c_logits = proj.extract_marginalized_logits(labels, branch_out_id=out_id)
                        else:
                            logits_ptr = self.llm._ctx.get_logits_ith(last_indices[s])
                            branch_c_logits = self._extract_candidate_logits_from_ptr(logits_ptr, labels)
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
                        if use_sparse and proj is not None:
                            branch_c_logits = proj.extract_marginalized_logits(labels, branch_out_id=0)
                        else:
                            logits_ptr = self.llm._ctx.get_logits_ith(len(chunk) - 1)
                            branch_c_logits = self._extract_candidate_logits_from_ptr(logits_ptr, labels)
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
                "Ensure a compatible vision model and mmproj are loaded."
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
            enable_thinking=False,
        )
        text = handler._postprocess_template_text(text, image_urls, media_marker)

        # Ensure any thinking block is closed so candidate logits directly evaluate the decision choice
        if text.rstrip().endswith("<think>"):
            text = text.rstrip() + "\n\n</think>\n\n"
        elif "<think>" in text and "</think>" not in text:
            text = text.rstrip() + "\n</think>\n\n"

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
                print(f"[vision-tokens] mtmd tokenized into {n_chunks} chunk(s):")
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
                            head_toks = tokens[:6]
                            tail_toks = tokens[-6:] if len(tokens) > 6 else tokens
                            print(f"  [chunk {i}] TEXT: {len(tokens)} tokens | head={head_toks} tail={tail_toks}")
                            if self.llm.n_tokens + len(tokens) > self.llm.n_ctx():
                                raise ValueError(f"Prompt exceeds n_ctx: {self.llm.n_tokens + len(tokens)} > {self.llm.n_ctx()}")
                            self.llm.eval(tokens)
                    elif ctype in [handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE, handler._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                        chunk_n_tokens = handler._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                        print(f"  [chunk {i}] IMAGE embeddings: {chunk_n_tokens} visual tokens (directly preceding question context)")
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

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Executes a traditional autoregressive chat completion using the shared llama.cpp model.
        Acquires the engine lock for thread safety and cleanly resets the KV cache afterwards.
        """
        with self._lock:
            has_image = any(
                isinstance(m.get("content"), list)
                and any(isinstance(item, dict) and item.get("type") == "image_url" for item in m["content"])
                for m in messages
            )
            if has_image and not self.has_vision:
                self._ensure_vision_handler()

            try:
                call_kwargs: dict[str, Any] = {
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if response_format:
                    call_kwargs["response_format"] = response_format
                if stop:
                    call_kwargs["stop"] = stop
                call_kwargs.update(kwargs)

                with self._full_vocab_scope():
                    return self.llm.create_chat_completion(**call_kwargs)
            finally:
                try:
                    self.llm.reset()
                    self.llm._ctx.kv_cache_clear()
                except Exception:
                    pass

    def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        **kwargs: Any,
    ):
        """
        Streams autoregressive chat tokens chunk by chunk.
        Ensures lock safety and resets KV cache upon stream completion or disconnection.
        """
        has_image = any(
            isinstance(m.get("content"), list)
            and any(isinstance(item, dict) and item.get("type") == "image_url" for item in m["content"])
            for m in messages
        )
        if has_image and not self.has_vision:
            self._ensure_vision_handler()

        self._lock.acquire()
        proj = getattr(self, "sparse_projector", None)
        was_sparse_active = bool(proj is not None and proj.active)
        if was_sparse_active and proj is not None:
            proj.deactivate()
        try:
            call_kwargs: dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if response_format:
                call_kwargs["response_format"] = response_format
            if stop:
                call_kwargs["stop"] = stop
            call_kwargs.update(kwargs)

            for chunk in self.llm.create_chat_completion(**call_kwargs):
                yield chunk
        finally:
            if was_sparse_active and proj is not None:
                try:
                    proj.activate()
                except Exception:
                    pass
            try:
                self.llm.reset()
                self.llm._ctx.kv_cache_clear()
            except Exception:
                pass
            try:
                self._lock.release()
            except Exception:
                pass

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
            with self._full_vocab_scope():
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
        reuse_kv: bool = False,
    ) -> tuple[str, str, np.ndarray, str]:
        """Runs bounded CoT verification on ambiguous or emergency decisions."""
        apply_calibrator = self.calibrator is not None and use_calibrator is not False
        if not reuse_kv or getattr(self.llm, "n_tokens", 0) == 0:
            prompt = self._render_prompt(context, options)
            self.llm.reset()
            self.llm._ctx.kv_cache_clear()
            tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
            self.llm.eval(tokens)

        prefix = cot_prompt if cot_prompt is not None else resolve_cot_prompt()
        analysis = prefix if prefix.startswith("\n") else f"\n{prefix}"

        generated: list[int] = []
        eos = int(self.llm.token_eos())
        stop_tokens = {eos}
        for special_str in ["<turn|>", "<end_of_turn>", "<|turn>", "<channel|>", "<|channel>"]:
            st = self.llm.tokenize(special_str.encode("utf-8"), add_bos=False, special=True)
            if st:
                stop_tokens.add(int(st[0]))

        with self._full_vocab_scope():
            analysis_tokens = self.llm.tokenize(analysis.encode("utf-8"), add_bos=False, special=True)
            if self.llm.n_tokens + len(analysis_tokens) + 16 > self.llm.n_ctx():
                # Near context boundary; truncate or avoid overflowing KV cache
                pass
            else:
                self.llm.eval(analysis_tokens)
                for _ in range(max_cot_tokens):
                    if self.llm.n_tokens + 16 >= self.llm.n_ctx():
                        break
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
        cue_tokens = self.llm.tokenize(conclusion_cue.encode("utf-8"), add_bos=False, special=True)
        if self.llm.n_tokens + len(cue_tokens) < self.llm.n_ctx():
            self.llm.eval(cue_tokens)
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
        options: dict[str, str] | list[str] = ...,
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
        confidence_threshold: float | None = None,
        enable_rules: bool = True,
        early_exit: bool = False,
        force_tier0: bool = False,
        tier0_entropy_threshold: float | None = None,
        tier0_margin_threshold: float | None = None,
        tier0_ood_threshold: float | None = None,
        tier0_landmark_distance_threshold: float | None = None,
        **kwargs,
    ) -> DecisionResponse:
        """
        Main entrypoint. Evaluates options against context and/or images.
        Supports Tier 0 Topological Absorbing Early Exit before LLM inference.
        Tier 0 (Early Exit) -> Tier 1 (Fast-Path zero-decoding) -> Tier 2 (CoT Verification).
        """
        if confidence_threshold is not None and cot_threshold is None:
            cot_threshold = confidence_threshold
        if cot_threshold is None:
            cot_threshold = DEFAULT_COT_THRESHOLD
        if isinstance(options, (list, tuple)):
            alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            options = {alphabet[i]: opt for i, opt in enumerate(options)}
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
                tier=0,
                escalate=False,
                evidence=None,
                cot_scratchpad="Single candidate option provided; executed deterministically.",
                image_attached=bool(image_bytes_list),
            )

        start_t = time.perf_counter()

        # ── Tier -1: Heuristic Rule Gateway (No model required, < 0.05ms) ──────
        rule_match = None
        rule_action = None
        if enable_rules:
            from .rules import get_rule_gateway
            rule_gw = get_rule_gateway()
            rule_match = rule_gw.evaluate_route(context, options)
            rule_action = rule_match.action if rule_match else None

            if rule_action == "DIRECT_EXIT" and rule_match and rule_match.decision:
                win_letter = rule_match.decision
                win_text = rule_match.decision_text or options.get(win_letter, win_letter)
                dur = (time.perf_counter() - start_t) * 1000.0
                full_probs = {opt_txt: (1.0 if k == win_letter else 0.0) for k, opt_txt in options.items()}
                if win_text not in full_probs:
                    full_probs[win_text] = 1.0
                full_letter_probs = {k: (1.0 if k == win_letter else 0.0) for k in options.keys()}
                if win_letter not in full_letter_probs:
                    full_letter_probs[win_letter] = 1.0
                return DecisionResponse(
                    test_id=test_id,
                    decision=win_letter,
                    decision_text=win_text,
                    confidence=1.0,
                    latency_ms=round(dur, 2),
                    duration_ms=round(dur, 2),
                    candidate_letter=win_letter,
                    probabilities=full_probs,
                    letter_probabilities=full_letter_probs,
                    mode="tier-1_rule_exit",
                    tier=-1,
                    escalate=False,
                    cot_scratchpad=f"Tier -1 Heuristic Exit: [{rule_match.rule_name}] -> DIRECT_EXIT ({rule_match.description})",
                )

            if rule_action == "FORCE_COT":
                cot_threshold = 1.0

        # ── Tier 0: Topological Absorbing Markov Early Exit ────────────────────
        escalated_from_tier0 = False
        tier0_reason: str | None = None
        tier0_metrics: dict | None = None

        has_images = bool(image_bytes_list)
        t0_eng = self.get_tier0(auto_load=False)
        t0_supports_vision = getattr(t0_eng, "supports_vision", False) if t0_eng else False

        if early_exit and (not has_images or t0_supports_vision) and (context.strip() or has_images) and (rule_action not in ("SKIP_TIER0", "FORCE_COT")):
            if t0_eng is None:
                # No Embedding model loaded; skip Tier 0
                pass
            else:
                try:
                    t0_start = time.perf_counter()
                    from .tier0 import (
                        DEFAULT_TIER0_ENTROPY_THRESHOLD,
                        DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD,
                        DEFAULT_TIER0_MARGIN_THRESHOLD,
                        DEFAULT_TIER0_OOD_THRESHOLD,
                    )
                    t_ent = tier0_entropy_threshold if tier0_entropy_threshold is not None else DEFAULT_TIER0_ENTROPY_THRESHOLD
                    t_mar = tier0_margin_threshold if tier0_margin_threshold is not None else DEFAULT_TIER0_MARGIN_THRESHOLD
                    t_ood = tier0_ood_threshold if tier0_ood_threshold is not None else DEFAULT_TIER0_OOD_THRESHOLD
                    d_ood = (
                        tier0_landmark_distance_threshold
                        if tier0_landmark_distance_threshold is not None
                        else DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD
                    )

                    t0_res = t0_eng.evaluate(
                        query=context.strip(),
                        candidates=options,
                        images=image_bytes_list if t0_supports_vision else None,
                        tau_entropy=t_ent,
                        tau_margin=t_mar,
                        tau_ood=t_ood,
                        delta_ood=d_ood,
                        force_tier0=force_tier0,
                        enforce_manifold_health=True,
                    )
                    t0_duration = (time.perf_counter() - t0_start) * 1000.0
                    tier0_metrics = t0_res

                    if t0_res.get("action") == "EXIT_0":
                        win_letter = t0_res["decision"]
                        win_text = t0_res.get("decision_text") or options.get(win_letter, "")
                        conf = t0_res["confidence"]
                        h_val = t0_res.get("entropy", 0.0)
                        m_val = t0_res.get("margin", 0.0)

                        evidence = None
                        if include_evidence:
                            evidence = self.explain_decision(
                                context=context,
                                verdict_letter=win_letter,
                                verdict_text=win_text,
                                image_bytes_list=image_bytes_list if image_bytes_list else None,
                            )

                        return DecisionResponse(
                            test_id=test_id,
                            decision=win_letter,
                            decision_text=win_text,
                            confidence=conf,
                            latency_ms=round(t0_duration, 2),
                            duration_ms=round(t0_duration, 2),
                            candidate_letter=win_letter,
                            probabilities=t0_res.get("probabilities"),
                            letter_probabilities=t0_res.get("letter_probabilities"),
                            mode="tier0_early_exit",
                            tier=0,
                            escalate=False,
                            evidence=evidence,
                            cot_scratchpad=f"Tier 0 Early Exit (Markov Absorbing Chain): H={h_val:.3f} < {t_ent}, Margin Δ={m_val:.3f} > {t_mar}. Resolved without LLM autoregression.",
                            tier0_metrics=tier0_metrics,
                            image_attached=bool(image_bytes_list),
                        )
                    else:
                        escalated_from_tier0 = True
                        tier0_reason = t0_res.get("reason", "AMBIGUITY")
                except Exception as exc:
                    print(f"[Tier 0] warning: early exit evaluation failed ({exc}); escalating to Tier 1")
                    escalated_from_tier0 = True
                    tier0_reason = f"ERROR: {exc}"


        with self._lock:
            # 1. Fast-path logit evaluation (15-22ms)
            fast_result = self._fast_path_evaluate(
                context,
                options,
                use_calibrator=use_calibrator,
                image_bytes_list=image_bytes_list if image_bytes_list else None,
            )

            # 2. Check CoT verification escalation
            # Optimization: Bypass cyclic debiasing when CoT is active, and reuse hot KV cache
            # from _fast_path_evaluate() so prompt prefill is never duplicated.
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
                    reuse_kv=True,
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
                    tier=2,
                    escalate=float(np.max(post_probs)) < 0.60,
                    evidence=evidence,
                    cot_scratchpad=cot_reasoning,
                    image_attached=bool(image_bytes_list),
                    tier0_escalation_reason=tier0_reason,
                    tier0_metrics=tier0_metrics,
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
                    tier=1,
                    escalate=top_conf < 0.60,
                    evidence=evidence,
                    cot_scratchpad=None,
                    image_attached=bool(image_bytes_list),
                    tier0_escalation_reason=tier0_reason,
                    tier0_metrics=tier0_metrics,
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
                tier=1,
                escalate=fast_result["confidence"] < 0.60,
                evidence=evidence,
                cot_scratchpad=None,
                image_attached=bool(image_bytes_list),
                tier0_escalation_reason=tier0_reason,
                tier0_metrics=tier0_metrics,
            )

    def classify(
        self,
        text: str,
        candidates: list[str],
        early_exit: bool = True,
        force_tier0: bool = False,
        tau_entropy: float = 0.65,
        tau_margin: float = 0.25,
        tau_ood: float = 0.40,
    ) -> dict[str, Any]:
        """Classifies text against candidates via Tier 0 Early Exit or Tier 1 escalation."""
        t0 = time.perf_counter()
        if early_exit:
            t0_eng = self.get_tier0(auto_load=False)
            if t0_eng is None:
                tier0_reason = "NO_EMBEDDING_LOADED"
            else:
                try:
                    t0_res = t0_eng.evaluate(
                        query=text,
                        candidates=candidates,
                        tau_entropy=tau_entropy,
                        tau_margin=tau_margin,
                        tau_ood=tau_ood,
                        force_tier0=force_tier0,
                        enforce_manifold_health=True,
                    )
                    if t0_res.get("action") == "EXIT_0":
                        return {
                            "tier": 0,
                            "decision": t0_res.get("decision_text") or t0_res.get("decision"),
                            "confidence": t0_res["confidence"],
                            "latency_ms": t0_res["latency_ms"],
                            "mode": "tier0_early_exit",
                            "escalated_from_tier0": False,
                            "tier0_reason": None,
                            "probabilities": t0_res.get("probabilities"),
                            "entropy": t0_res.get("entropy"),
                            "margin": t0_res.get("margin"),
                        }
                    tier0_reason = t0_res.get("reason", "AMBIGUITY")
                except Exception as exc:
                    tier0_reason = f"ERROR: {exc}"
        else:
            tier0_reason = "BYPASS"


        # Tier 1 Fallback: Gevva0 zero-decoding token scoring
        alphabet = [chr(ord("A") + i) for i in range(len(candidates))]
        options = {alphabet[i]: candidates[i] for i in range(len(candidates))}
        res = self.decide(context=text, options=options, early_exit=False)
        total_lat = (time.perf_counter() - t0) * 1000.0
        return {
            "tier": 2 if res.mode == "cot_verified" else 1,
            "decision": res.decision_text or res.decision,
            "confidence": res.confidence,
            "latency_ms": round(total_lat, 2),
            "mode": res.mode,
            "escalated_from_tier0": True if early_exit else False,
            "tier0_reason": tier0_reason,
            "probabilities": res.probabilities,
        }

    def extract_audit(
        self,
        context: str,
        locked_choice: str,
        image: str | bytes | Path | list[str | bytes | Path] | None = None,
        max_tokens: int = 192,
        temperature: float = 0.1,
    ) -> DecisionEvidence:
        """
        Asymmetric post-verdict quote extraction ensuring zero logit poisoning.
        Extracts verbatim source quotes and operational justification AFTER decision is locked.
        """
        verdict_letter = locked_choice if len(locked_choice) == 1 and locked_choice.isalpha() else "A"
        verdict_text = locked_choice
        image_bytes_list: list[bytes] = []
        if image is not None:
            if isinstance(image, (list, tuple)):
                image_bytes_list = [load_image_bytes(img) for img in image if img is not None]
            else:
                image_bytes_list = [load_image_bytes(image)]
        return self.explain_decision(
            context=context,
            verdict_letter=verdict_letter,
            verdict_text=verdict_text,
            image_bytes_list=image_bytes_list or None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def create_session(
        self,
        system_prompt: str,
        temperature: float = 1.0,
        cot_threshold: float = 0.85,
        cot_max_tokens: int = 32,
        cot_prompt: str | None = None,
        cot_conclusion: str | None = None,
        bias_offsets: dict[str, float] | None = None,
        prompt_formatter: Any | None = None,
        start_worker: bool = True,
        sparse_logit_proj: bool | str | None = None,
    ):
        """
        Creates a pinned agent session wrapping DecisionBridge.
        Pins an invariant rulebook into KV cache sequence 0 and returns an asynchronous, tick-ready worker.
        """
        from .bridge import DecisionBridge

        bridge = DecisionBridge(
            system_prompt=system_prompt,
            temperature=temperature,
            cot_threshold=cot_threshold,
            cot_max_tokens=cot_max_tokens,
            cot_prompt=cot_prompt,
            cot_conclusion=cot_conclusion,
            bias_offsets=bias_offsets,
            prompt_formatter=prompt_formatter,
            engine=self,
            sparse_logit_proj=sparse_logit_proj,
        )
        if start_worker:
            bridge.start_worker()
        return bridge


# Public unified client alias
GevvaEngine = GemmaDecisionEngine

