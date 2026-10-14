"""
gevva0.bridge - Universal calibrated decision gateway for interactive agents.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class DecisionRequest:
    state_text: str
    options: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_cot: bool = True
    cyclic_debias: bool = False
    custom_labels: list[str] | None = None
    serial: int = 0
    cot_prompt: str | None = None
    cot_conclusion: str | None = None


@dataclass
class DecisionResult:
    choice_index: int
    label: str
    choice_text: str
    probs: dict[str, float]
    confidence: float
    latency_ms: float
    mode: str  # tier0_hybrid | tier0_markov | tier0_cosine | fast_path | cot_verified | cyclic_debiased | forced | simulated | error_fallback
    cot_text: str | None = None
    raw_logits: list[float] | None = None
    fast_letter: str | None = None
    fast_confidence: float | None = None
    prompt_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    serial: int = 0
    ts: float = field(default_factory=time.perf_counter)
    tier: int = 1
    entropy: float = 0.0
    margin: float = 0.0
    cosine_probs: dict[str, float] | None = None
    markov_probs: dict[str, float] | None = None
    cosine_sims: dict[str, float] | None = None
    embed_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    tripwire_passed: bool = True
    tripwire_reason: str | None = None
    scaffold_mode: str | None = None
    landmarks_count: int = 0

    @property
    def letter(self) -> str:
        """Alias for label for backwards compatibility with Pac-Man and other client code."""
        return self.label


class DecisionBridge:
    """Universal high-speed decision gateway backed by GemmaDecisionEngine."""

    ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    BASE_SEQ = 0
    WORK_SEQ = 1
    DEFAULT_COT_PROMPT = "\nAnalysis: compare status and room first, then targets: "
    DEFAULT_COT_CONCLUSION = "\nTherefore, the best option letter is:"

    def __init__(
        self,
        model_path: str | Path | None = None,
        system_prompt: str = "",
        n_ctx: int = 4096,
        temperature: float = 1.0,
        cot_threshold: float = 0.85,
        cot_max_tokens: int = 32,
        cot_prompt: str | None = None,
        cot_conclusion: str | None = None,
        bias_offsets: dict[str, float] | None = None,
        prompt_formatter: Callable[[str, list[str], list[str]], str] | None = None,
        verbose: bool = False,
        engine: Any | None = None,
        sparse_logit_proj: bool | str | None = "auto",
    ):
        if engine is not None:
            self.engine = engine
            self.model_path = Path(engine.model_path)
        else:
            from gevva0 import GemmaDecisionEngine
            from gevva0.config import resolve_model_path

            self.model_path = Path(resolve_model_path(model_path))
            self.engine = GemmaDecisionEngine.from_config(
                model_path=str(self.model_path),
                mmproj_path=None,
                load_calibration=False,
                n_ctx=n_ctx,
                verbose=verbose,
                sparse_logit_proj=sparse_logit_proj,
            )

        self.system_prompt = system_prompt
        self.temperature = max(temperature, 1e-3)
        self.cot_threshold = cot_threshold
        self.cot_max_tokens = cot_max_tokens
        self.cot_prompt = cot_prompt or self.DEFAULT_COT_PROMPT
        self.cot_conclusion = cot_conclusion or self.DEFAULT_COT_CONCLUSION
        self.bias_offsets = bias_offsets or {}
        self.prompt_formatter = prompt_formatter
        self.verbose = verbose
        self.llm = self.engine.llm
        self.n_vocab = self.llm.n_vocab()
        self._tok_cache: dict[str, tuple[int, int]] = {}
        self.stop_tokens = self._init_stop_tokens()

        from .sparse_logits import SparseLogitProjector, resolve_sparse_logit_proj_flag

        eng_proj = getattr(self.engine, "sparse_projector", None)
        if isinstance(eng_proj, SparseLogitProjector):
            if sparse_logit_proj is not None and not resolve_sparse_logit_proj_flag(
                sparse_logit_proj, llm=self.llm, model_path=self.model_path
            ):
                eng_proj.deactivate()
                self.sparse_projector: SparseLogitProjector | None = None
            else:
                self.sparse_projector = eng_proj
        else:
            self.sparse_projector = None
            if resolve_sparse_logit_proj_flag(
                sparse_logit_proj, llm=self.llm, model_path=self.model_path
            ):
                try:
                    proj = SparseLogitProjector(self.llm)
                    if proj.supported:
                        self.sparse_projector = proj
                except Exception:
                    self.sparse_projector = None

        # Pin system prefix
        self.prefix_tokens = self._pin_prefix(system_prompt)

        # Non-blocking async queue
        self._in_queue: queue.Queue[tuple[DecisionRequest, threading.Event, list[DecisionResult]]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_processing = False

    @property
    def sparse_logit_proj_enabled(self) -> bool:
        return self.sparse_projector is not None and self.sparse_projector.active

    def _full_vocab_scope(self):
        if self.sparse_projector is not None:
            return self.sparse_projector.full_vocab_scope()
        import contextlib
        return contextlib.nullcontext()

    def _init_stop_tokens(self) -> set[int]:
        stops = {int(self.llm.token_eos())}
        for s in ["<turn|>", "<end_of_turn>", "<|turn>", "<channel|>", "<|channel>"]:
            t = self.llm.tokenize(s.encode(), add_bos=False, special=True)
            if len(t) == 1:
                stops.add(int(t[0]))
        return stops

    def _pin_prefix(self, prompt: str) -> int:
        formatted = f"<|turn>user\n{prompt}\n\n"
        tokens = self.llm.tokenize(formatted.encode("utf-8"), add_bos=True, special=True)
        ctx = self.llm._ctx
        self.llm.reset()
        ctx.kv_cache_clear()
        self._decode(tokens, seq=self.BASE_SEQ, start_pos=0, want_last=False)
        return len(tokens)

    def _decode(self, tokens: list[int], seq: int, start_pos: int, want_last: bool = True) -> np.ndarray | None:
        batch = self.llm._batch
        nb = self.llm.n_batch
        out = None
        for c0 in range(0, len(tokens), nb):
            chunk = tokens[c0:c0 + nb]
            batch.reset()
            batch.batch.n_tokens = len(chunk)
            for i, tok in enumerate(chunk):
                batch.batch.token[i] = tok
                batch.batch.pos[i] = start_pos + c0 + i
                batch.batch.seq_id[i][0] = seq
                batch.batch.n_seq_id[i] = 1
                batch.batch.logits[i] = want_last and (c0 + i == len(tokens) - 1)
            self.llm._ctx.decode(batch)
            if want_last and (c0 + len(chunk) == len(tokens)):
                ptr = self.llm._ctx.get_logits_ith(len(chunk) - 1)
                n_cols = (
                    self.sparse_projector.k_rows
                    if (self.sparse_projector is not None and self.sparse_projector.active)
                    else self.n_vocab
                )
                out = np.ctypeslib.as_array(ptr, shape=(n_cols,)).astype(np.float64)
        return out

    def _decode_for_candidates(
        self,
        tokens: list[int],
        seq: int,
        start_pos: int,
        labels: list[str],
    ) -> np.ndarray:
        """Decode tokens on `seq` and extract marginalized candidate logits without full-vocab host copy."""
        if self.sparse_projector is not None and self.sparse_projector.can_score_labels(labels):
            batch = self.llm._batch
            nb = self.llm.n_batch
            for c0 in range(0, len(tokens), nb):
                chunk = tokens[c0 : c0 + nb]
                batch.reset()
                batch.batch.n_tokens = len(chunk)
                for i, tok in enumerate(chunk):
                    batch.batch.token[i] = tok
                    batch.batch.pos[i] = start_pos + c0 + i
                    batch.batch.seq_id[i][0] = seq
                    batch.batch.n_seq_id[i] = 1
                    batch.batch.logits[i] = (c0 + i == len(tokens) - 1)
                self.llm._ctx.decode(batch)
            return self.sparse_projector.extract_marginalized_logits(labels, branch_out_id=0)

        v = self._decode(tokens, seq=seq, start_pos=start_pos, want_last=True)
        return self._marginalized_letter_logits(v, labels)

    def _letter_ids(self, label: str) -> tuple[int, int]:
        if label not in self._tok_cache:
            sp = int(self.llm.tokenize(f" {label}".encode(), add_bos=False)[0])
            no = int(self.llm.tokenize(label.encode(), add_bos=False)[0])
            self._tok_cache[label] = (sp, no)
        return self._tok_cache[label]

    def _marginalized_letter_logits(self, vocab_logits: np.ndarray, labels: list[str]) -> np.ndarray:
        out = []
        for l in labels:
            a, b = self._letter_ids(l)
            m = max(vocab_logits[a], vocab_logits[b])
            out.append(m + math.log(math.exp(vocab_logits[a] - m) + math.exp(vocab_logits[b] - m)))
        return np.asarray(out)

    def _render_prompt(self, state_text: str, options: list[str], labels: list[str]) -> str:
        if self.prompt_formatter is not None:
            return self.prompt_formatter(state_text, options, labels)
        rendered_options = "\n".join(f"{lbl}. {opt}" for lbl, opt in zip(labels, options))
        return (
            f"State:\n{state_text}\n\n"
            f"Options:\n{rendered_options}\n\n"
            f"Best option?<turn|>\n"
            f"<|turn>model\n<|channel>thought\n<channel|>"
        )

    def _calibrate_and_softmax(self, raw_logits: np.ndarray, labels: list[str]) -> np.ndarray:
        z = np.asarray(raw_logits, dtype=np.float64).copy()
        for i, l in enumerate(labels):
            z[i] -= self.bias_offsets.get(l, 0.0)
        scaled = z / self.temperature
        exp_z = np.exp(scaled - np.max(scaled))
        return exp_z / exp_z.sum()

    # --- Core Synchronous Execution ---
    def decide(self, request: DecisionRequest) -> DecisionResult:
        n_opts = len(request.options)
        if n_opts == 0:
            raise ValueError("DecisionRequest must provide at least one option.")

        labels = request.custom_labels or self.ALPHABET[:n_opts]
        if n_opts == 1:
            return DecisionResult(
                choice_index=0,
                label=labels[0],
                choice_text=request.options[0],
                probs={labels[0]: 1.0},
                confidence=1.0,
                latency_ms=0.0,
                mode="forced",
                metadata=request.metadata,
                serial=request.serial,
            )

        t0 = time.perf_counter()
        ctx = self.llm._ctx
        prompt_tokens = 0

        if request.cyclic_debias:
            raw = self._score_cyclic(request.state_text, request.options, labels)
            mode = "cyclic_debiased"
            cot_text = None
        else:
            prompt = self._render_prompt(request.state_text, request.options, labels)
            toks = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            prompt_tokens = len(toks)
            ctx.kv_cache_seq_rm(self.WORK_SEQ, 0, -1)
            ctx.kv_cache_seq_cp(self.BASE_SEQ, self.WORK_SEQ, 0, -1)
            try:
                raw = self._decode_for_candidates(toks, seq=self.WORK_SEQ, start_pos=self.prefix_tokens, labels=labels)
            finally:
                ctx.kv_cache_seq_rm(self.WORK_SEQ, 0, -1)
            mode = "fast_path"
            cot_text = None

        probs = self._calibrate_and_softmax(raw, labels)
        top_idx = int(np.argmax(probs))
        fast_letter = labels[top_idx]
        fast_conf = float(probs[top_idx])

        # Slack/Confidence-Aware Bounded CoT Verification
        if (
            not request.cyclic_debias
            and request.allow_cot
            and self.cot_threshold > 0
            and fast_conf < self.cot_threshold
        ):
            raw, cot_text = self._score_cot(
                request.state_text,
                request.options,
                labels,
                cot_prompt=request.cot_prompt,
                cot_conclusion=request.cot_conclusion,
            )
            probs = self._calibrate_and_softmax(raw, labels)
            top_idx = int(np.argmax(probs))
            mode = "cot_verified"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DecisionResult(
            choice_index=top_idx,
            label=labels[top_idx],
            choice_text=request.options[top_idx],
            probs={lbl: float(p) for lbl, p in zip(labels, probs)},
            confidence=float(probs[top_idx]),
            latency_ms=latency_ms,
            mode=mode,
            cot_text=cot_text,
            raw_logits=[float(x) for x in raw],
            fast_letter=fast_letter,
            fast_confidence=fast_conf,
            prompt_tokens=prompt_tokens,
            metadata=request.metadata,
            serial=request.serial,
        )

    def _score_cyclic(self, state: str, options: list[str], labels: list[str]) -> np.ndarray:
        n = len(labels)
        ctx = self.llm._ctx
        suffixes = []
        for s in range(n):
            rot = [options[(i + s) % n] for i in range(n)]
            suffixes.append(
                self.llm.tokenize(
                    self._render_prompt(state, rot, labels).encode("utf-8"),
                    add_bos=False,
                    special=True,
                )
            )

        seqs = [self.WORK_SEQ + s for s in range(n)]
        for sq in seqs:
            ctx.kv_cache_seq_rm(sq, 0, -1)
            ctx.kv_cache_seq_cp(self.BASE_SEQ, sq, 0, -1)

        per_branch = []
        use_sparse = self.sparse_projector is not None and self.sparse_projector.can_score_labels(labels)
        try:
            total = sum(len(t) for t in suffixes)
            batch = self.llm._batch
            if total <= self.llm.n_batch:
                batch.reset()
                batch.batch.n_tokens = total
                idx, last = 0, []
                for s, toks in enumerate(suffixes):
                    for i, tok in enumerate(toks):
                        batch.batch.token[idx] = tok
                        batch.batch.pos[idx] = self.prefix_tokens + i
                        batch.batch.seq_id[idx][0] = seqs[s]
                        batch.batch.n_seq_id[idx] = 1
                        batch.batch.logits[idx] = i == len(toks) - 1
                        if i == len(toks) - 1:
                            last.append(idx)
                        idx += 1
                ctx.decode(batch)
                for s in range(n):
                    if use_sparse and self.sparse_projector is not None:
                        out_id = self.sparse_projector.get_branch_output_id(last[s])
                        per_branch.append(self.sparse_projector.extract_marginalized_logits(labels, branch_out_id=out_id))
                    else:
                        ptr = ctx.get_logits_ith(last[s])
                        v = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float64)
                        per_branch.append(self._marginalized_letter_logits(v, labels))
            else:
                for s, toks in enumerate(suffixes):
                    per_branch.append(
                        self._decode_for_candidates(toks, seq=seqs[s], start_pos=self.prefix_tokens, labels=labels)
                    )
        finally:
            for sq in seqs:
                ctx.kv_cache_seq_rm(sq, 0, -1)

        agg = np.zeros(n)
        for s in range(n):
            for orig in range(n):
                agg[orig] += per_branch[s][(orig - s) % n]
        return agg / n

    def _score_cot(
        self,
        state: str,
        options: list[str],
        labels: list[str],
        cot_prompt: str | None = None,
        cot_conclusion: str | None = None,
    ) -> tuple[np.ndarray, str]:
        ctx = self.llm._ctx
        prompt_text = cot_prompt or self.cot_prompt
        conclusion_text = cot_conclusion or self.cot_conclusion

        base_prompt = self._render_prompt(state, options, labels) + prompt_text
        base_toks = self.llm.tokenize(base_prompt.encode("utf-8"), add_bos=False, special=True)
        ctx.kv_cache_seq_rm(self.WORK_SEQ, 0, -1)
        ctx.kv_cache_seq_cp(self.BASE_SEQ, self.WORK_SEQ, 0, -1)
        try:
            with self._full_vocab_scope():
                v = self._decode(base_toks, seq=self.WORK_SEQ, start_pos=self.prefix_tokens)
                pos = self.prefix_tokens + len(base_toks)
                gen: list[int] = []
                for _ in range(self.cot_max_tokens):
                    tok = int(np.argmax(v))
                    if tok in self.stop_tokens:
                        break
                    gen.append(tok)
                    v = self._decode([tok], seq=self.WORK_SEQ, start_pos=pos)
                    pos += 1
                    if len(gen) >= 4 and "\n" in self.llm.detokenize(gen[-4:]).decode("utf-8", errors="ignore"):
                        break
            text = self.llm.detokenize(gen).decode("utf-8", errors="ignore").strip()
            concl = self.llm.tokenize(conclusion_text.encode("utf-8"), add_bos=False, special=True)
            cand_logits = self._decode_for_candidates(concl, seq=self.WORK_SEQ, start_pos=pos, labels=labels)
            return cand_logits, text
        finally:
            ctx.kv_cache_seq_rm(self.WORK_SEQ, 0, -1)

    def warmup(self, rounds: int = 2):
        dummy_req = DecisionRequest(
            state_text="warmup\nOptions:\nA. UP\nB. DOWN",
            options=["UP", "DOWN"],
            custom_labels=["A", "B"],
            allow_cot=False,
        )
        for _ in range(rounds):
            self.decide(dummy_req)

    @property
    def busy(self) -> bool:
        return self._is_processing or not self._in_queue.empty()

    # --- Asynchronous Worker Interface for Game & Real-Time Loops ---
    def start_worker(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="gevva0-bridge-worker")
        self._worker_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                req, ev, container = self._in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._is_processing = True
            try:
                res = self.decide(req)
                container.append(res)
            except Exception as exc:
                fallback_label = req.custom_labels[0] if req.custom_labels else "A"
                container.append(
                    DecisionResult(
                        choice_index=0,
                        label=fallback_label,
                        choice_text=req.options[0] if req.options else "",
                        probs={fallback_label: 1.0},
                        confidence=0.0,
                        latency_ms=0.0,
                        mode="error_fallback",
                        metadata=req.metadata,
                        serial=req.serial,
                    )
                )
            finally:
                ev.set()
                self._is_processing = False
                self._in_queue.task_done()

    def submit_async(self, request: DecisionRequest) -> Callable[[], DecisionResult | None]:
        """Submits a request to the GPU worker thread without blocking the calling thread."""
        ev = threading.Event()
        container: list[DecisionResult] = []
        self._in_queue.put((request, ev, container))

        def poll() -> DecisionResult | None:
            return container[0] if ev.is_set() else None

        return poll

    def close(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        try:
            self.engine.close()
        except Exception:
            pass

    @classmethod
    def from_tier0(
        cls,
        model_path: str | Path | None = None,
        device: str | None = None,
        corpus: list[str] | None = None,
        class_anchors: dict[str, list[str]] | None = None,
        markov_weight: float = 0.35,
        tau_entropy: float = 0.65,
        tau_margin: float = 0.25,
        tau_ood: float = 0.40,
        tau_temp: float = 0.05,
        p_landmarks: int = 4,
        tier0_engine: Any | None = None,
    ) -> "Tier0DecisionBridge":
        """Factory returning a Tier 0-only game bridge backed by Tier0Engine (ONNX + Absorbing Markov Chain)."""
        return Tier0DecisionBridge(
            model_path=model_path,
            device=device,
            corpus=corpus,
            class_anchors=class_anchors,
            markov_weight=markov_weight,
            tau_entropy=tau_entropy,
            tau_margin=tau_margin,
            tau_ood=tau_ood,
            tau_temp=tau_temp,
            p_landmarks=p_landmarks,
            tier0_engine=tier0_engine,
        )


class Tier0DecisionBridge:
    """Universal ultra-low-latency game & agent decision bridge backed solely by Tier0Engine.

    Exposes the four core components of Tier0Engine to real-time game loops:
      (1) Direct Cosine Alignment: sim(q, c_i)
      (2) Default 39-Landmark Scaffold (DEFAULT_CORPUS)
      (3) Domain Manifold Scaffold: Absorbing Markov Chain B = (I - Q)^(-1) R
      (4) Production Hybrid Ensemble: P_final = (1 - w) * Cosine + w * Markov (default 65/35)
    Along with Shannon Entropy (H <= tau_1) and Logit Margin (Delta >= tau_2) tripwire diagnostics.
    """

    ALPHABET = DecisionBridge.ALPHABET

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        corpus: list[str] | None = None,
        class_anchors: dict[str, list[str]] | None = None,
        markov_weight: float = 0.35,
        tau_entropy: float = 0.65,
        tau_margin: float = 0.25,
        tau_ood: float = 0.40,
        tau_temp: float = 0.05,
        tau_proj: float = 0.04,
        p_landmarks: int = 4,
        enable_vision: bool = False,
        enable_audio: bool = False,
        tier0_engine: Any | None = None,
    ):
        if tier0_engine is not None:
            self.tier0 = tier0_engine
            self.model_path = Path(str(getattr(tier0_engine, "model_path", model_path or "tier0")))
            if corpus is not None or class_anchors is not None:
                from gevva0.tier0 import DEFAULT_CORPUS, DEFAULT_CLASS_ANCHORS
                self.tier0.build_scaffold(
                    corpus or DEFAULT_CORPUS,
                    class_anchors or DEFAULT_CLASS_ANCHORS,
                )
                self.tier0._dynamic_cache.clear()
        else:
            from gevva0.tier0 import Tier0Engine
            self.tier0 = Tier0Engine(
                model_path=str(model_path) if model_path is not None else None,
                device=device,
                corpus=corpus,
                class_anchors=class_anchors,
                enable_vision=enable_vision,
                enable_audio=enable_audio,
            )
            self.model_path = Path(str(self.tier0.model_path))

        self.markov_weight = float(markov_weight)
        self.tau_entropy = float(tau_entropy)
        self.tau_margin = float(tau_margin)
        self.tau_ood = float(tau_ood)
        self.tau_temp = float(tau_temp)
        self.tau_proj = float(tau_proj)
        self.p_landmarks = int(p_landmarks)

        # Async worker state (matches DecisionBridge)
        self._in_queue: queue.Queue[tuple[DecisionRequest, threading.Event, list[DecisionResult]]] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_processing = False
        self._lock = threading.Lock()

    def rebuild_scaffold(
        self,
        corpus: list[str] | None = None,
        class_anchors: dict[str, list[str]] | None = None,
    ) -> None:
        """Rebuilds the topological Markov scaffold (e.g. toggling between Default 39-Landmarks and Domain Manifold)."""
        from gevva0.tier0 import DEFAULT_CORPUS, DEFAULT_CLASS_ANCHORS
        with self._lock:
            self.tier0._dynamic_cache.clear()
            self.tier0.build_scaffold(
                corpus or DEFAULT_CORPUS,
                class_anchors or DEFAULT_CLASS_ANCHORS,
            )

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Synchronously evaluates a DecisionRequest using Tier0Engine in ~2ms."""
        n_opts = len(request.options)
        if n_opts == 0:
            raise ValueError("DecisionRequest must provide at least one option.")

        labels = request.custom_labels or self.ALPHABET[:n_opts]
        if n_opts == 1:
            return DecisionResult(
                choice_index=0,
                label=labels[0],
                choice_text=request.options[0],
                probs={labels[0]: 1.0},
                confidence=1.0,
                latency_ms=0.0,
                mode="forced",
                metadata=request.metadata,
                serial=request.serial,
                tier=0,
                tripwire_passed=True,
                scaffold_mode="forced",
                landmarks_count=int(getattr(self.tier0, "M", 0)),
            )

        candidates = {lbl: opt for lbl, opt in zip(labels, request.options)}
        m_weight = float(request.metadata.get("markov_weight", self.markov_weight))

        with self._lock:
            res = self.tier0.evaluate(
                query=request.state_text,
                candidates=candidates,
                p_landmarks=self.p_landmarks,
                tau_entropy=self.tau_entropy,
                tau_margin=self.tau_margin,
                tau_ood=self.tau_ood,
                tau_proj=self.tau_proj,
                tau_temp=self.tau_temp,
                force_tier0=True,
                enforce_manifold_health=False,
                markov_weight=m_weight,
            )

        decision_label = res.get("decision", labels[0])
        choice_index = labels.index(decision_label) if decision_label in labels else 0
        letter_probs = res.get("letter_probabilities") or {lbl: 1.0 / n_opts for lbl in labels}
        probs = {lbl: float(letter_probs.get(lbl, 0.0)) for lbl in labels}

        if m_weight <= 0.0:
            mode_str = "tier0_cosine"
        elif m_weight >= 1.0:
            mode_str = "tier0_markov"
        else:
            mode_str = "tier0_hybrid"

        tripwire_passed = res.get("action") == "EXIT_0"
        tripwire_reason = res.get("reason") if not tripwire_passed else None

        return DecisionResult(
            choice_index=choice_index,
            label=decision_label,
            choice_text=request.options[choice_index],
            probs=probs,
            confidence=float(res.get("confidence", probs.get(decision_label, 0.5))),
            latency_ms=float(res.get("latency_ms", 0.0)),
            mode=mode_str,
            fast_letter=decision_label,
            fast_confidence=float(res.get("confidence", 0.5)),
            metadata=request.metadata,
            serial=request.serial,
            tier=0,
            entropy=float(res.get("entropy", 0.0)),
            margin=float(res.get("margin", 0.0)),
            cosine_probs=res.get("cosine_probabilities"),
            markov_probs=res.get("markov_probabilities"),
            cosine_sims=res.get("cosine_similarities"),
            embed_latency_ms=float(res.get("embed_latency_ms", 0.0)),
            graph_latency_ms=float(res.get("graph_latency_ms", 0.0)),
            tripwire_passed=tripwire_passed,
            tripwire_reason=tripwire_reason,
            scaffold_mode=res.get("scaffold_mode", "domain_manifold"),
            landmarks_count=int(res.get("landmarks_count", getattr(self.tier0, "M", 0))),
        )

    def warmup(self, rounds: int = 2) -> None:
        dummy_req = DecisionRequest(
            state_text="Warmup Pig turn state: player score 40, opponent score 45, turn bank 10.",
            options=[
                "ROLL again to accumulate more turn points and press expected value advantage",
                "HOLD now to bank current turn points safely and protect score",
            ],
            custom_labels=["A", "B"],
            allow_cot=False,
        )
        for _ in range(rounds):
            self.decide(dummy_req)

    @property
    def busy(self) -> bool:
        return self._is_processing or not self._in_queue.empty()

    def start_worker(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="gevva0-tier0-bridge-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                req, ev, container = self._in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._is_processing = True
            try:
                res = self.decide(req)
                container.append(res)
            except Exception:
                fallback_label = req.custom_labels[0] if req.custom_labels else "A"
                container.append(
                    DecisionResult(
                        choice_index=0,
                        label=fallback_label,
                        choice_text=req.options[0] if req.options else "",
                        probs={fallback_label: 1.0},
                        confidence=0.0,
                        latency_ms=0.0,
                        mode="error_fallback",
                        metadata=req.metadata,
                        serial=req.serial,
                        tier=0,
                    )
                )
            finally:
                ev.set()
                self._is_processing = False
                self._in_queue.task_done()

    def submit_async(self, request: DecisionRequest) -> Callable[[], DecisionResult | None]:
        ev = threading.Event()
        container: list[DecisionResult] = []
        self._in_queue.put((request, ev, container))

        def poll() -> DecisionResult | None:
            return container[0] if ev.is_set() else None

        return poll

    def close(self) -> None:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)