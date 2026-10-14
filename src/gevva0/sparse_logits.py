"""
Gemma 4 Sparse Logit Projection ("Micro-Head Scoring").

Eliminates the 262,144-token full-vocabulary output projection tax during
Tier 1 Fast-Path and Cyclic KV-Branching inference by projecting the final
hidden state against only a compact K=64 micro-head containing the 52
candidate option tokens (' A'..' Z' and 'A'..'Z').

When Tier 2 Chain-of-Thought (CoT), post-verdict audit extraction, or
autoregressive chat completion runs, the full 262,144-token output head is
transparently restored via `full_vocab_scope()`.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
from pathlib import Path
import sys
import threading
from typing import Any

import llama_cpp.llama_cpp as lc
import numpy as np

from .config import LETTERS


class _ggml_tensor(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("buffer", ctypes.c_void_p),
        ("ne", ctypes.c_int64 * 4),
        ("nb", ctypes.c_size_t * 4),
        ("op", ctypes.c_int32),
        ("op_params", ctypes.c_int32 * 16),
        ("flags", ctypes.c_int32),
        ("src", ctypes.c_void_p * 10),
        ("view_src", ctypes.c_void_p),
        ("view_offs", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("name", ctypes.c_char * 64),
        ("extra", ctypes.c_void_p),
        ("padding", ctypes.c_char * 8),
    ]


_PATCH_LOCK = threading.Lock()
_DECODE_PATCHED = False
_GGML_FNS: tuple[Any, Any] | None = None

# Offsets in llama_model (llama.dll x86_64 build)
_MODEL_TOK_EMBD_OFFSET = 0x7268
_MODEL_OUTPUT_OFFSET = 0x72A8
_DECODE_NVOCAB_LOAD_OFFSET = 0x919
_ORIG_DECODE_INSN = b"\x4c\x8b\x5d\xb8"  # mov r11, [rbp - 0x48] (n_vocab)
_PATCH_DECODE_INSN = b"\x4c\x8b\x5b\x10"  # mov r11, [rbx + 0x10] (t_logits->ne[0])


def _get_ggml_tensor_io() -> tuple[Any, Any]:
    global _GGML_FNS
    if _GGML_FNS is not None:
        return _GGML_FNS

    lib_dir = Path(lc.__file__).parent / "lib"
    if sys.platform == "win32":
        ggml_base = ctypes.CDLL(str(lib_dir / "ggml-base.dll"))
    elif sys.platform == "darwin":
        ggml_base = ctypes.CDLL(str(lib_dir / "libggml-base.dylib"))
    else:
        ggml_base = ctypes.CDLL(str(lib_dir / "libggml-base.so"))

    tensor_set = ggml_base.ggml_backend_tensor_set
    tensor_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    tensor_set.restype = None

    tensor_get = ggml_base.ggml_backend_tensor_get
    tensor_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    tensor_get.restype = None

    _GGML_FNS = (tensor_get, tensor_set)
    return _GGML_FNS


def _ensure_decode_patched() -> bool:
    """Patch llama_context::decode + 0x919 so async logit copy uses t_logits->ne[0]."""
    global _DECODE_PATCHED
    with _PATCH_LOCK:
        if _DECODE_PATCHED:
            return True
        if sys.platform != "win32":
            return False
        try:
            lib_dir = Path(lc.__file__).parent / "lib"
            llama_dll = ctypes.CDLL(str(lib_dir / "llama.dll"))
            fn_decode = ctypes.cast(
                getattr(llama_dll, "?decode@llama_context@@QEAAHAEBUllama_batch@@@Z"),
                ctypes.c_void_p,
            ).value
            if not fn_decode:
                return False

            patch_addr = fn_decode + _DECODE_NVOCAB_LOAD_OFFSET
            cur = bytes((ctypes.c_uint8 * 4).from_address(patch_addr))
            if cur == _PATCH_DECODE_INSN:
                _DECODE_PATCHED = True
                return True
            if cur != _ORIG_DECODE_INSN:
                return False

            kernel32 = ctypes.windll.kernel32
            old_prot = ctypes.c_ulong()
            if not kernel32.VirtualProtect(ctypes.c_void_p(patch_addr), 4, 0x40, ctypes.byref(old_prot)):
                return False
            ctypes.memmove(patch_addr, _PATCH_DECODE_INSN, 4)
            kernel32.VirtualProtect(ctypes.c_void_p(patch_addr), 4, old_prot.value, ctypes.byref(old_prot))
            _DECODE_PATCHED = True
            return True
        except Exception:
            return False


def is_gemma4_model(llm: Any = None, model_path: str | Path | None = None) -> bool:
    """Auto-detect whether the loaded LLM or model path is a Gemma 4 architecture model."""
    if llm is not None:
        meta = getattr(llm, "metadata", None)
        if isinstance(meta, dict):
            arch = str(meta.get("general.architecture", "")).strip().lower()
            if arch == "gemma4":
                return True
            gen_name = str(meta.get("general.name", "")).strip().lower()
            if "gemma-4" in gen_name or "gemma4" in gen_name or "gemma 4" in gen_name:
                return True

    if model_path is not None:
        s = str(model_path).lower()
        if "gemma-4" in s or "gemma4" in s or "gemma_4" in s:
            return True

    return False


def resolve_sparse_logit_proj_flag(
    sparse_logit_proj: bool | str | None,
    llm: Any = None,
    model_path: str | Path | None = None,
) -> bool:
    """Resolve sparse_logit_proj setting ('auto', True, False) into a boolean."""
    if isinstance(sparse_logit_proj, bool):
        return sparse_logit_proj
    if sparse_logit_proj is None:
        return is_gemma4_model(llm=llm, model_path=model_path)
    val = str(sparse_logit_proj).strip().lower()
    if val in ("true", "1", "yes", "on", "enabled", "enable"):
        return True
    if val in ("false", "0", "no", "off", "disabled", "disable"):
        return False
    return is_gemma4_model(llm=llm, model_path=model_path)


class SparseLogitProjector:
    """Manages the K=64 Micro-Head output projection for Gemma 4 in llama.cpp."""

    MICRO_HEAD_K = 64

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.active = False
        self.supported = False
        self.n_vocab: int = int(llm.n_vocab())
        self.k_rows: int = self.MICRO_HEAD_K
        self.label_to_micro_idx: dict[str, tuple[int, int]] = {}
        self.candidate_token_ids: list[int] = []

        self._t_out_ptr: int = 0
        self._t_out: _ggml_tensor | None = None
        self._row_bytes: int = 0
        self._orig_ne1: int = 0
        self._orig_nb2: int = 0
        self._orig_nb3: int = 0
        self._orig_head_buf: Any = None
        self._compact_buf: Any = None

        self._initialize()

    def _initialize(self) -> None:
        model_obj = getattr(self.llm, "_model", None)
        ctx_obj = getattr(self.llm, "_ctx", None)
        model_ptr = getattr(model_obj, "model", None)
        ctx_ptr = getattr(ctx_obj, "ctx", None)
        if not isinstance(model_ptr, int) or not isinstance(ctx_ptr, int):
            return

        if not _ensure_decode_patched():
            return

        p_embd = ctypes.c_void_p.from_address(model_ptr + _MODEL_TOK_EMBD_OFFSET).value
        p_out = ctypes.c_void_p.from_address(model_ptr + _MODEL_OUTPUT_OFFSET).value
        if not p_embd or not p_out:
            return

        t_embd = _ggml_tensor.from_address(p_embd)
        t_out = _ggml_tensor.from_address(p_out)

        # Verify output tensor matches vocabulary dimension and does not alias input tok_embd data
        if t_out.ne[1] != self.n_vocab or self.n_vocab <= self.k_rows:
            return
        if p_embd == p_out or t_embd.data == t_out.data:
            return

        row_bytes = int(t_out.nb[1])
        if row_bytes <= 0:
            return

        cand_ids: list[int] = []
        label_map: dict[str, tuple[int, int]] = {}
        for i, ch in enumerate(LETTERS):
            tok_sp = int(self.llm.tokenize(f" {ch}".encode("utf-8"), add_bos=False)[0])
            tok_no = int(self.llm.tokenize(ch.encode("utf-8"), add_bos=False)[0])
            idx_sp = 2 * i
            idx_no = 2 * i + 1
            cand_ids.append(tok_sp)
            cand_ids.append(tok_no)
            label_map[ch] = (idx_sp, idx_no)

        padded_ids = cand_ids + [cand_ids[0]] * (self.k_rows - len(cand_ids))
        tensor_get, _ = _get_ggml_tensor_io()

        head_nbytes = self.k_rows * row_bytes
        orig_head_buf = (ctypes.c_uint8 * head_nbytes)()
        tensor_get(p_out, orig_head_buf, 0, head_nbytes)

        compact_buf = (ctypes.c_uint8 * head_nbytes)()
        compact_addr = ctypes.addressof(compact_buf)
        for idx, tid in enumerate(padded_ids):
            tensor_get(
                p_out,
                ctypes.c_void_p(compact_addr + idx * row_bytes),
                tid * row_bytes,
                row_bytes,
            )

        # Ensure ggml_backend_sched has allocated full-vocabulary compute buffers before shrinking t_out.ne[1]
        if not lc.llama_get_logits(ctx_ptr):
            try:
                bos = int(self.llm.token_bos())
                self.llm.reset()
                self.llm._ctx.kv_cache_clear()
                self.llm.eval([bos])
                self.llm.reset()
                self.llm._ctx.kv_cache_clear()
            except Exception:
                pass

        self._t_out_ptr = p_out
        self._t_out = t_out
        self._row_bytes = row_bytes
        self._orig_ne1 = int(t_out.ne[1])
        self._orig_nb2 = int(t_out.nb[2])
        self._orig_nb3 = int(t_out.nb[3])
        self._orig_head_buf = orig_head_buf
        self._compact_buf = compact_buf
        self.candidate_token_ids = cand_ids
        self.label_to_micro_idx = label_map
        self.supported = True

        self.activate()

    def _invalidate_graph_cache(self) -> None:
        """Invalidate ctx->gf_res_prev->params so the next decode rebuilds the output projection node."""
        try:
            ctx_ptr = getattr(getattr(self.llm, "_ctx", None), "ctx", None)
            if isinstance(ctx_ptr, int) and ctx_ptr:
                gf_res_prev = ctypes.c_void_p.from_address(ctx_ptr + 0x308).value
                if gf_res_prev:
                    ctypes.c_int64.from_address(gf_res_prev + 0x110).value = -1
        except Exception:
            pass

    def activate(self) -> None:
        """Switch model.output on GPU into compact K=64 Micro-Head mode."""
        if not self.supported or self.active or self._t_out is None:
            return
        _, tensor_set = _get_ggml_tensor_io()
        head_nbytes = self.k_rows * self._row_bytes
        tensor_set(self._t_out_ptr, self._compact_buf, 0, head_nbytes)
        self._t_out.ne[1] = self.k_rows
        self._t_out.nb[2] = head_nbytes
        self._t_out.nb[3] = head_nbytes
        self._invalidate_graph_cache()
        self.active = True

    def deactivate(self) -> None:
        """Restore model.output on GPU to the full 262,144-token vocabulary."""
        if not self.supported or not self.active or self._t_out is None:
            return
        _, tensor_set = _get_ggml_tensor_io()
        head_nbytes = self.k_rows * self._row_bytes
        self._t_out.ne[1] = self._orig_ne1
        self._t_out.nb[2] = self._orig_nb2
        self._t_out.nb[3] = self._orig_nb3
        tensor_set(self._t_out_ptr, self._orig_head_buf, 0, head_nbytes)
        self._invalidate_graph_cache()
        self.active = False

    @contextlib.contextmanager
    def full_vocab_scope(self):
        """Temporarily restore full-vocabulary output projection for token generation (CoT / chat)."""
        was_active = self.active
        if was_active:
            self.deactivate()
        try:
            yield
        finally:
            if was_active:
                self.activate()

    def can_score_labels(self, labels: list[str]) -> bool:
        return self.supported and self.active and all(lbl in self.label_to_micro_idx for lbl in labels)

    def get_branch_output_id(self, tok_idx: int) -> int:
        """Return the compact output row index (0..n_outputs-1) for batch token `tok_idx`."""
        ctx_ptr = self.llm._ctx.ctx
        base_ptr = lc.llama_get_logits(ctx_ptr)
        ith_ptr = lc.llama_get_logits_ith(ctx_ptr, tok_idx)
        diff_bytes = ctypes.cast(ith_ptr, ctypes.c_void_p).value - ctypes.cast(base_ptr, ctypes.c_void_p).value
        return int(diff_bytes // (self.n_vocab * 4))

    def get_micro_logits(self, branch_out_id: int = 0) -> np.ndarray:
        """Return the K=64 float logits for output `branch_out_id`."""
        ctx_ptr = self.llm._ctx.ctx
        base_ptr = lc.llama_get_logits(ctx_ptr)
        branch_addr = ctypes.cast(base_ptr, ctypes.c_void_p).value + branch_out_id * self.k_rows * 4
        branch_ptr = ctypes.cast(branch_addr, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(branch_ptr, shape=(self.k_rows,)).astype(np.float64)

    def extract_marginalized_logits(self, labels: list[str], branch_out_id: int = 0) -> np.ndarray:
        """Compute logsumexp(logit(' X'), logit('X')) for each label in `labels` from the K=64 micro-head."""
        micro = self.get_micro_logits(branch_out_id=branch_out_id)
        out = np.empty(len(labels), dtype=np.float64)
        for i, lbl in enumerate(labels):
            idx_sp, idx_no = self.label_to_micro_idx[lbl]
            l_sp = float(micro[idx_sp])
            l_no = float(micro[idx_no])
            m = l_sp if l_sp >= l_no else l_no
            out[i] = m + math.log(math.exp(l_sp - m) + math.exp(l_no - m))
        return out
