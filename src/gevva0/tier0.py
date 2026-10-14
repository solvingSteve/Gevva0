"""Tier 0 Semantic Absorbing Early-Exit Engine (Gevva-T0).

Zero-token early-exit classification gateway based on topological absorbing
Markov chains constructed over EmbeddingGemma 768D vector space.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

from .config import PROJECT_ROOT, resolve_tier0_model_path

logger = logging.getLogger("gevva0.tier0")

DEFAULT_TIER0_ENTROPY_THRESHOLD = 0.65
DEFAULT_TIER0_MARGIN_THRESHOLD = 0.25
DEFAULT_TIER0_OOD_THRESHOLD = 0.40
DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD = 0.22
DEFAULT_MIN_MANIFOLD_LANDMARKS = 50

# Built-in representative domain corpus covering core classification clusters
DEFAULT_CORPUS = [
    # Billing & Financial
    "Where is my monthly invoice?",
    "Can I get a refund for my order?",
    "Charge on my credit card is incorrect",
    "I need a billing receipt for tax purposes",
    "Dispute fraudulent credit card transaction",
    "Why was my account billed twice this month?",
    "Subscription invoice and payment receipt request",
    "Refund status for cancelled service renewal",
    "Account collections and trailing subscription fee statement",
    "Credit card chargeback and payment discrepancy",
    # Technical & Infrastructure
    "My screen went black and won't turn on",
    "The server is overheating and CPU throttled",
    "Application crashed with segmentation fault",
    "Hardware buttons are stuck on the physical gateway",
    "Network connection timed out with 504 Gateway Timeout error",
    "Device firmware update failed and corrupted flash memory",
    "Unresolved technical outage and server error log",
    "Kernel panic crash dump analysis and driver failure",
    "Kubernetes pod crash loop backoff memory saturation",
    "Database disk utilization reached 100% read-only lock",
    # Sales & Commercial
    "How much does the enterprise plan cost?",
    "I want to talk to sales about pricing tiers",
    "Requesting a formal enterprise demo for our team",
    "Need contract upgrade quote for 500 enterprise seats",
    "Volume licensing discount schedule and commercial contract",
    "Sales representative consultation for corporate onboarding",
    "Commercial sales routing to complete Enterprise cluster upgrade",
    "Annual enterprise license pricing agreement review",
    "Talk to sales executive regarding corporate package tier",
    "Quote for upgrading from Starter subscription to Enterprise",
    # Security & Incident Response
    "Hostile indirect injection attempt seeking credential exfiltration",
    "System security maintenance override penetration testing token",
    "Adversarial intent masking prompt injection attack detected",
    "Account security breach unauthorized access credentials leaked",
    "Malicious payload attempting API key exfiltration",
    # Legal & Compliance
    "Legal compliance audit report SOC 2 Type II counter-signature",
    "Mutual NDA agreement legal counsel review before contract",
    "Uncapped contractual liability gross negligence data disclosure",
    "Regulatory GDPR compliance inquiry and legal terms",
]

DEFAULT_CLASS_ANCHORS = {
    "Billing": [
        "monthly invoice statement",
        "credit card refund transaction",
        "charge dispute billing receipt payment",
        "subscription fees and accounting invoice",
    ],
    "Technical": [
        "hardware server crash",
        "screen black failure error",
        "overheating kernel panic crash bug",
        "504 gateway timeout database outage",
    ],
    "Sales": [
        "enterprise pricing demo",
        "contract quote upgrade sales commercial",
        "subscription volume licensing account executive",
        "corporate sales agreement onboarding",
    ],
    "Security": [
        "prompt injection credential exfiltration",
        "adversarial penetration breach attack payload",
        "unauthorized security access token leak",
    ],
    "Legal": [
        "legal compliance SOC 2 NDA contract",
        "regulatory terms contractual liability carve-out",
        "confidential data disclosure legal counsel",
    ],
}


class Tier0Engine:
    """Tier 0 Ultra-Low-Latency Intelligent Router and Calibrated Classification Engine.
    
    Operates directly on 768D dense representations from EmbeddingGemma-2.
    Projects query to nearest topological landmarks, evaluates Markov absorption
    probabilities via precomputed or dynamically factored fundamental matrix B = (I - Q)^(-1) R,
    and applies tripwire gating (Shannon Entropy, Margin, OOD distance).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        corpus: list[str] | None = None,
        class_anchors: dict[str, list[str]] | None = None,
        # Default use_compile to False: eager PyTorch on CUDA executes in <10ms on RTX 5090 without
        # TorchInductor/Triton compilation pauses (which freeze for 15-30s on dynamic sequence lengths on Windows).
        use_compile: bool = False,
        enable_vision: bool = True,
        enable_audio: bool = True,
    ):
        self.enable_vision = bool(enable_vision)
        self.enable_audio = bool(enable_audio)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cpu":
            import os
            try:
                torch.set_num_threads(min(8, os.cpu_count() or 4))
            except Exception:
                pass

        self.model_path = resolve_tier0_model_path(model_path)
        logger.info(f"Initializing Tier 0 on {self.device} with model: {self.model_path}")
        print(f"[Tier 0] Loading {self.model_path} onto {self.device}...")

        p_model = Path(self.model_path)
        self.is_onnx = (
            (p_model.is_dir() and ((p_model / "model.onnx").is_file() or list(p_model.glob("*.onnx"))))
            or (p_model.is_file() and p_model.suffix.lower() == ".onnx")
            or "onnx" in str(self.model_path).lower()
        )
        self.is_gguf = not self.is_onnx and (
            str(self.model_path).lower().endswith(".gguf") or (
                p_model.is_file() and p_model.suffix.lower() == ".gguf"
            )
        )
        if self.is_onnx:
            self.backend = "onnx"
        elif self.is_gguf:
            self.backend = "gguf"
        else:
            self.backend = "transformers"

        self.model_name = p_model.stem if p_model.exists() else str(self.model_path)
        self.dim = 768  # Initial default, updated dynamically during warmup
        self.supports_vision = False
        self.supports_audio = False
        self.tokenizer = None
        self.processor = None
        self.onnx_text_session = None
        self.onnx_vision_session = None
        self.onnx_audio_session = None

        if self.is_onnx:
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                model_onnx_path = (p_model / "model.onnx") if p_model.is_dir() else p_model
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3
                self.onnx_text_session = ort.InferenceSession(str(model_onnx_path), sess_options=sess_opts, providers=providers)

                base_dir = p_model.parent if p_model.is_file() else p_model
                v_path = base_dir / "vision_encoder.onnx"
                if self.enable_vision and v_path.is_file():
                    self.onnx_vision_session = ort.InferenceSession(str(v_path), sess_options=sess_opts, providers=providers)
                    self.supports_vision = True
                else:
                    self.supports_vision = False
                    if not self.enable_vision and v_path.is_file():
                        print("[Tier 0] Vision encoder disabled by configuration (saving ~671 MB VRAM).")

                a_path = base_dir / "audio_encoder.onnx"
                if self.enable_audio and a_path.is_file():
                    self.onnx_audio_session = ort.InferenceSession(str(a_path), sess_options=sess_opts, providers=providers)
                    self.supports_audio = True
                else:
                    self.supports_audio = False
                    if not self.enable_audio and a_path.is_file():
                        print("[Tier 0] Audio encoder disabled by configuration (saving ~1.17 GB VRAM).")

                # Locate local processor / tokenizer
                search_dirs = [
                    base_dir,
                    PROJECT_ROOT / "models" / "embeddings" / "embeddinggemma-2",
                    PROJECT_ROOT / "models" / "embeddinggemma-2-repo",
                    PROJECT_ROOT / "embeddinggemma-2-repo",
                    Path("C:/code/embeddinggemma-2"),
                ]
                for sd in search_dirs:
                    if sd.is_dir() and ((sd / "processor_config.json").is_file() or (sd / "tokenizer.json").is_file()):
                        try:
                            self.processor = AutoProcessor.from_pretrained(str(sd))
                            self.tokenizer = getattr(self.processor, "tokenizer", None)
                            if self.tokenizer is not None:
                                break
                        except Exception:
                            try:
                                self.tokenizer = AutoTokenizer.from_pretrained(str(sd))
                                if self.tokenizer is not None:
                                    break
                            except Exception:
                                pass

                if self.tokenizer is None:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")
                    except Exception:
                        pass

                if self.tokenizer is None:
                    raise RuntimeError(
                        f"Tokenizer files not found in '{base_dir}'. "
                        f"ONNX models require the lightweight tokenizer files (~37 MB total: config.json, "
                        f"processor_config.json, preprocessor_config.json, tokenizer.json, "
                        f"tokenizer_config.json, tokenizer.model) in the model directory. "
                        f"(The 1.5 GB model.safetensors file is NOT required)."
                    )

                self.raw_embedder = None
                self.embedder = None
                self.llama_model = None
                self.use_compile = False
            except Exception as exc:
                raise RuntimeError(f"Failed to load ONNX embedding model '{self.model_path}': {exc}") from exc
        elif self.is_gguf:
            try:
                from llama_cpp import Llama
                n_gpu = -1 if self.device == "cuda" else 0
                self.llama_model = Llama(
                    model_path=str(self.model_path),
                    embedding=True,
                    n_gpu_layers=n_gpu,
                    verbose=False,
                )
                self.tokenizer = None
                self.raw_embedder = None
                self.embedder = None
                self.use_compile = False
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load GGUF embedding model '{self.model_path}' with llama.cpp: {exc}. "
                    f"Ensure this architecture is supported by llama.cpp."
                ) from exc
        else:
            self.llama_model = None
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load text-only configuration to minimize memory (~650 MB) and maximize speed
            cfg = AutoConfig.from_pretrained(self.model_path)
            if hasattr(cfg, "audio_config"):
                cfg.audio_config = None
            if hasattr(cfg, "vision_config"):
                cfg.vision_config = None

            self.raw_embedder = AutoModel.from_pretrained(
                self.model_path,
                config=cfg,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()

            self.use_compile = bool(use_compile)
            if self.use_compile and self.device == "cuda":
                try:
                    print(f"[Tier 0] Compiling embedder with torch.compile for online query fast-path...")
                    self.embedder = torch.compile(self.raw_embedder)
                except Exception as exc:
                    print(f"[Tier 0] torch.compile notice: {exc}; running in eager mode")
                    self.embedder = self.raw_embedder
                    self.use_compile = False
            else:
                self.embedder = self.raw_embedder

        self.landmarks: torch.Tensor | None = None
        self.landmarks_np: np.ndarray | None = None
        self.B_matrix: torch.Tensor | None = None  # [M, K] on device
        self.classes: list[str] = []
        self.M: int = 0
        self.K: int = 0

        # Sparse internal graph components for dynamic candidates
        self.Q_raw: sp.csr_matrix | None = None
        self.lu_factor: spla.SuperLU | None = None
        self.I_sparse: sp.csc_matrix | None = None
        self._dynamic_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # Candidate Cache keyed by tuple of candidate strings (keeps tensors directly on GPU)
        self._candidate_cache: dict[tuple[str, ...], torch.Tensor] = {}

        # Warm up embedder, detect vector dimension, and trigger JIT compilation
        self._warmup()

        # Build initial scaffold
        initial_corpus = corpus or DEFAULT_CORPUS
        initial_anchors = class_anchors or DEFAULT_CLASS_ANCHORS
        self.build_scaffold(initial_corpus, initial_anchors)

        # Run a complete synthetic warmup decision pass through the full pipeline
        self._warmup_evaluate()

    def _warmup(self) -> None:
        """Warm up embedding execution, detect vector dimension, and trigger compile if enabled."""
        try:
            print("[Tier 0] Warming up embedder and detecting vector dimension...")
            t_w = time.perf_counter()
            with torch.inference_mode():
                dummy = self.embed(["task: classification | query: warmup classification query"])
                if self.device == "cuda":
                    torch.cuda.synchronize()
            self.dim = int(dummy.shape[-1])
            elapsed_w = (time.perf_counter() - t_w) * 1000.0
            print(f"[Tier 0] Warmup complete ({elapsed_w:.1f} ms, {self.dim}D vector space). Ready for inference.")
        except Exception as e:
            logger.warning(f"[Tier 0] Warmup warning: {e}")

    def _warmup_evaluate(self) -> None:
        """Runs a synthetic end-to-end evaluation pass to pre-compile CUDA kernels and candidate caches."""
        try:
            dummy_images = None
            if getattr(self, "supports_vision", False) and getattr(self, "onnx_vision_session", None) is not None:
                from PIL import Image
                dummy_images = [Image.new("RGB", (64, 64), color="gray")]
            t_s = time.perf_counter()
            with torch.inference_mode():
                _ = self.evaluate(
                    query="warmup system health diagnostic",
                    candidates={"A": "Warmup Alpha", "B": "Warmup Beta"},
                    images=dummy_images,
                )
                if self.device == "cuda":
                    torch.cuda.synchronize()
            w_ms = (time.perf_counter() - t_s) * 1000.0
            print(f"[Tier 0] Synthetic pipeline warmup complete ({w_ms:.1f} ms). Sub-millisecond inference ready.")
        except Exception as exc:
            logger.warning(f"[Tier 0] Pipeline warmup note: {exc}")

    def get_candidate_embeddings(self, candidates: dict[str, str] | list[str]) -> torch.Tensor:
        """Returns cached candidate embeddings on GPU, embedding once on first access."""
        if isinstance(candidates, dict):
            # Create a stable key from the option texts sorted by option letters
            cache_key = tuple(candidates[k] for k in sorted(candidates.keys()))
        else:
            cache_key = tuple(candidates)

        if cache_key not in self._candidate_cache:
            option_texts = list(cache_key)
            # Embed candidates once and keep directly on GPU
            self._candidate_cache[cache_key] = self.embed(option_texts)

        return self._candidate_cache[cache_key]

    @torch.inference_mode()
    def embed(self, texts: list[str], use_compiled: bool = True) -> torch.Tensor:
        """Extracts L2-normalized embeddings from the active model (ONNX, GGUF, or HuggingFace)."""
        if getattr(self, "is_onnx", False) and getattr(self, "onnx_text_session", None) is not None:
            max_batch = 16
            if len(texts) > max_batch:
                chunks = [
                    self.embed(texts[i : i + max_batch], use_compiled=use_compiled)
                    for i in range(0, len(texts), max_batch)
                ]
                return torch.cat(chunks, dim=0)

            formatted = [f"task: classification | query: {t}" for t in texts]
            tok = self.tokenizer or getattr(self.processor, "tokenizer", None)
            inputs = tok(
                formatted,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            m_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "image_features": np.zeros((0, 512), dtype=np.float32),
                "video_features": np.zeros((0, 512), dtype=np.float32),
                "audio_features": np.zeros((0, 512), dtype=np.float32),
            }
            out = self.onnx_text_session.run(None, m_inputs)
            tensor = torch.tensor(out[1], dtype=torch.float32, device=self.device)
            return F.normalize(tensor, p=2, dim=1)

        if getattr(self, "is_gguf", False) and getattr(self, "llama_model", None) is not None:
            formatted = [f"task: classification | query: {t}" for t in texts]
            raw = self.llama_model.embed(formatted, normalize=True)
            if len(texts) == 1 and len(raw) > 0 and isinstance(raw[0], (float, int)):
                raw = [raw]
            tensor = torch.tensor(raw, dtype=torch.float32, device=self.device)
            return F.normalize(tensor, p=2, dim=1)

        formatted = [f"task: classification | query: {t}" for t in texts]
        pad_mode = False if len(texts) == 1 else True
        inputs = self.tokenizer(
            formatted,
            padding=pad_mode,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        model_to_use = (
            self.embedder
            if (self.use_compile and use_compiled and len(texts) == 1)
            else self.raw_embedder
        )
        outputs = model_to_use(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    @torch.inference_mode()
    def embed_multimodal(
        self,
        query: str,
        images: list[Any] | None = None,
        audio: Any | None = None,
    ) -> torch.Tensor:
        """Embeds query together with image or audio modalities into the unified embedding space."""
        if not getattr(self, "is_onnx", False) or (not images and not audio):
            return self.embed([query])

        from PIL import Image
        import io
        import base64

        pil_images = []
        if images:
            for item in images:
                try:
                    if isinstance(item, Image.Image):
                        pil_images.append(item.convert("RGB"))
                    elif isinstance(item, (bytes, bytearray)):
                        pil_images.append(Image.open(io.BytesIO(item)).convert("RGB"))
                    elif isinstance(item, str):
                        clean_str = item.strip()
                        if clean_str.startswith("data:") or len(clean_str) > 500:
                            b64_data = clean_str.split(",")[-1] if "," in clean_str else clean_str
                            raw_b = base64.b64decode(b64_data)
                            pil_images.append(Image.open(io.BytesIO(raw_b)).convert("RGB"))
                        else:
                            p_img = Path(clean_str)
                            if not p_img.is_absolute():
                                if (PROJECT_ROOT / clean_str).exists():
                                    p_img = PROJECT_ROOT / clean_str
                                elif (PROJECT_ROOT / "benchmarks" / clean_str).exists():
                                    p_img = PROJECT_ROOT / "benchmarks" / clean_str
                            if p_img.exists():
                                pil_images.append(Image.open(str(p_img)).convert("RGB"))
                except Exception as e:
                    logger.warning(f"[Tier 0] Failed to load image for multimodal embedding: {e}")

        if pil_images and getattr(self, "supports_vision", False) and self.onnx_vision_session is not None and self.processor is not None:
            num_imgs = len(pil_images)
            img_prefix = "<|image|>" * num_imgs
            q_clean = query.strip()
            prompt = f"{img_prefix}task: classification | query: {q_clean}" if q_clean else img_prefix
            proc_inputs = self.processor(
                text=prompt,
                images=pil_images if num_imgs > 1 else pil_images[0],
                return_tensors="np",
            )
            v_inputs = {
                "pixel_values": proc_inputs["pixel_values"].astype(np.float32),
                "pixel_position_ids": proc_inputs["image_position_ids"].astype(np.int64),
            }
            v_out = self.onnx_vision_session.run(None, v_inputs)
            img_features = v_out[0].astype(np.float32)

            m_inputs = {
                "input_ids": proc_inputs["input_ids"].astype(np.int64),
                "attention_mask": proc_inputs["attention_mask"].astype(np.int64),
                "image_features": img_features,
                "video_features": np.zeros((0, 512), dtype=np.float32),
                "audio_features": np.zeros((0, 512), dtype=np.float32),
            }
            out = self.onnx_text_session.run(None, m_inputs)
            tensor = torch.tensor(out[1], dtype=torch.float32, device=self.device)
            return F.normalize(tensor, p=2, dim=1)

        return self.embed([query])

    def build_scaffold(
        self,
        corpus_texts: list[str],
        class_anchors: dict[str, list[str]] | None = None,
        k_neighbors: int = 6,
        tau_graph: float = 0.05,
        tau_sink: float = 0.04,
        q_internal_weight: float = 0.50,
    ) -> None:
        """Builds landmark mesh, connects k-NN graph Q, factorizes (I - Q), and wires absorbing sinks."""
        print(f"[Tier 0] Building topological scaffold across {len(corpus_texts)} corpus landmarks...")
        corpus_vecs = self.embed(corpus_texts).float().cpu().numpy()
        self.M = len(corpus_texts)
        self.landmarks_np = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        self.dim = int(self.landmarks_np.shape[1])
        self.landmarks = torch.tensor(
            self.landmarks_np,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )

        # 1. Build Sparse k-NN Transition Matrix Q
        sim_matrix = np.dot(self.landmarks_np, self.landmarks_np.T)
        rows, cols, data = [], [], []

        for i in range(self.M):
            sim_matrix[i, i] = -np.inf
            k_eff = min(k_neighbors, self.M - 1)
            top_k_idx = np.argsort(sim_matrix[i])[-k_eff:]
            top_k_sims = sim_matrix[i, top_k_idx]

            weights = np.exp((top_k_sims - np.max(top_k_sims)) / tau_graph)
            weights = weights / np.sum(weights)

            for idx, w in zip(top_k_idx, weights):
                rows.append(i)
                cols.append(idx)
                data.append(w * q_internal_weight)

        self.Q_raw = sp.csr_matrix((data, (rows, cols)), shape=(self.M, self.M))

        # 2. Pre-factorize A = (I - Q) for lightning-fast dynamic solves
        self.I_sparse = sp.eye(self.M, format="csc")
        A_csc = self.I_sparse - self.Q_raw.tocsc()
        self.lu_factor = spla.splu(A_csc)

        # 3. Wire absorbing boundary R if class anchors provided
        if class_anchors:
            self.classes = list(class_anchors.keys())
            self.K = len(self.classes)
            R_raw = np.zeros((self.M, self.K))

            for class_idx, (class_name, anchor_examples) in enumerate(class_anchors.items()):
                anchor_vecs = self.embed(anchor_examples).float().cpu().numpy()
                centroid = np.mean(anchor_vecs, axis=0, keepdims=True)
                centroid = centroid / np.linalg.norm(centroid)

                affinities = np.dot(self.landmarks_np, centroid.T).flatten()
                num_anchors = max(1, int(self.M * 0.25))
                top_anchors = np.argsort(affinities)[-num_anchors:]
                top_affs = affinities[top_anchors]
                w_sink = np.exp((top_affs - np.max(top_affs)) / tau_sink)
                w_sink = (w_sink / np.max(w_sink)) * 0.50
                R_raw[top_anchors, class_idx] = w_sink

            # Row normalize transition probabilities
            row_sums = self.Q_raw.sum(axis=1).A1 + R_raw.sum(axis=1)
            row_sums = np.maximum(row_sums, 1e-9)
            for i in range(self.M):
                R_raw[i, :] /= row_sums[i]

            R_sparse = sp.csr_matrix(R_raw)
            B_solved = self.lu_factor.solve(R_sparse.toarray())
            b_sums = np.maximum(B_solved.sum(axis=1, keepdims=True), 1e-9)
            B_solved = B_solved / b_sums

            self.B_matrix = torch.tensor(B_solved, dtype=torch.float32, device=self.device)
            print(f"[Tier 0] Fundamental matrix B ({self.M}x{self.K}) ready on {self.device}.")

    def _clean_query(self, query: str) -> str:
        """Strips accidental prompt contamination or option lists from query before embedding."""
        import re
        q = query.strip()
        split_patterns = [
            r"\n\s*(?:Options|Candidates|Choices):\s*\n",
            r"\n\s*\[?A\]?[:\.\s]",
        ]
        for pat in split_patterns:
            parts = re.split(pat, q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1 and len(parts[0].strip()) >= 15:
                q = parts[0].strip()
                break
        return q

    @staticmethod
    def _is_non_semantic_option_set(
        query: str | None,
        candidates: dict[str, str] | list[str] | None,
    ) -> bool:
        """Returns True if candidates or query structure cannot be resolved by bi-encoder cosine similarity.

        Specifically guards against:
        1. Binary polarity / truth-value options (yes/no, true/false, valid/invalid)
        2. Ordinal scales (Level 0, Level 1, ...) or bare 1-2 character enum codes (S, M, L, XL)
        3. Syntactic negation or superseded state transitions in the query text
        """
        import re

        if query:
            q_head = query.split("\n\nQuestion:")[0]
            if re.search(
                r"\b(do not|don't|not asking|cancelled|superseded|instead of|ignore|except|unless|unknown if)\b",
                 query,
                re.IGNORECASE,
            ) or re.search(r"\bno\s+\w+\s+(needed|requested|selected|confirmed)\b", q_head, re.IGNORECASE):
                return True

        if not candidates:
            return False

        texts = list(candidates.values()) if isinstance(candidates, dict) else [str(x) for x in candidates]
        if not texts:
            return False

        heads = [t.split(":", 1)[0].strip().lower() for t in texts]
        polarity_tokens = {"yes", "no", "true", "false", "valid", "invalid", "unknown", "other", "none"}
        if any(h in polarity_tokens for h in heads):
            return True

        if all(h.startswith("level ") or h.isdigit() or len(h) <= 2 for h in heads):
            return True

        return False

    def _resolve_absorption_for_candidates(
        self,
        candidate_labels: list[str],
        candidate_texts: list[str],
        tau_sink: float = 0.04,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes or retrieves absorbing fundamental matrix B and candidate embedding vectors."""
        cache_key = tuple(candidate_texts)
        if cache_key in self._dynamic_cache:
            return self._dynamic_cache[cache_key]

        K = len(candidate_texts)
        cand_tensors = self.embed(candidate_texts)
        cand_vecs = cand_tensors.float().cpu().numpy()
        affinities = np.dot(self.landmarks_np, cand_vecs.T)  # [M, K]

        R_raw = np.zeros((self.M, K))
        num_anchors = max(1, int(self.M * 0.10))
        for k in range(K):
            top_anchors = np.argsort(affinities[:, k])[-num_anchors:]
            top_affs = affinities[top_anchors, k]
            w_sink = np.exp((top_affs - np.max(top_affs)) / tau_sink)
            w_sink = (w_sink / np.max(w_sink)) * 0.50
            R_raw[top_anchors, k] = w_sink

        row_sums = np.maximum(self.Q_raw.sum(axis=1).A1 + R_raw.sum(axis=1), 1e-9)
        for i in range(self.M):
            R_raw[i, :] /= row_sums[i]

        B_solved = self.lu_factor.solve(R_raw)
        b_sums = np.maximum(B_solved.sum(axis=1, keepdims=True), 1e-9)
        B_solved = B_solved / b_sums

        B_tensor = torch.tensor(B_solved, dtype=torch.float32, device=self.device)
        self._dynamic_cache[cache_key] = (B_tensor, cand_tensors)
        return B_tensor, cand_tensors

    @property
    def modalities(self) -> list[str]:
        """Returns supported modalities for this Tier 0 embedding engine."""
        mods = ["text"]
        if getattr(self, "supports_vision", False):
            mods.append("vision")
        if getattr(self, "supports_audio", False):
            mods.append("audio")
        return mods

    @property
    def manifold_healthy(self) -> bool:
        """Returns True if the loaded landmark manifold is large enough (M >= 50) for reliable early exit."""
        m_count = int(getattr(self, "M", 0))
        if hasattr(self, "landmarks") and self.landmarks is not None:
            m_count = max(m_count, int(self.landmarks.shape[0]))
        return m_count >= DEFAULT_MIN_MANIFOLD_LANDMARKS

    @torch.inference_mode()
    def evaluate(
        self,
        query: str,
        candidates: dict[str, str] | list[str] | None = None,
        image: Any | None = None,
        images: list[Any] | None = None,
        audio: Any | None = None,
        p_landmarks: int = 4,
        tau_entropy: float = DEFAULT_TIER0_ENTROPY_THRESHOLD,
        tau_margin: float = DEFAULT_TIER0_MARGIN_THRESHOLD,
        tau_ood: float = DEFAULT_TIER0_OOD_THRESHOLD,
        delta_ood: float = DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD,
        tau_proj: float = 0.04,
        tau_temp: float = 0.05,
        force_tier0: bool = False,
        enforce_manifold_health: bool = False,
        min_landmarks: int = DEFAULT_MIN_MANIFOLD_LANDMARKS,
        markov_weight: float = 0.35,
        vision_markov_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Runs online batch-1 embedding, GPU landmark projection, and tripwire gating.
        
        Returns:
            Dictionary with 'action' ('EXIT_0' or 'ESCALATE'), confidence, margin,
            normalized entropy, candidate probabilities, component breakdowns
            (Direct Cosine, Markov Absorption, 65/35 Hybrid), and latency breakdown.
        """
        t0 = time.perf_counter()

        all_images = images if images is not None else ([image] if image is not None else None)
        is_multimodal = bool(
            (all_images or audio)
            and (getattr(self, "supports_vision", False) or getattr(self, "supports_audio", False))
        )

        # 0. Manifold Health Tripwire: Zero-cost bypass before embedding when M < min_landmarks
        # Note: Multimodal/vision queries using 100% Direct Cosine (vision_markov_weight == 0.0)
        # match image embeddings directly against candidate text labels (CLIP/SigLIP style)
        # and do not depend on the text-only Markov manifold Q.
        m_count = int(getattr(self, "M", 0))
        if hasattr(self, "landmarks") and self.landmarks is not None:
            m_count = max(m_count, int(self.landmarks.shape[0]))
        if (
            enforce_manifold_health
            and not force_tier0
            and m_count < min_landmarks
            and not (is_multimodal and vision_markov_weight <= 0.0)
        ):
            return {
                "action": "ESCALATE",
                "tier": 0,
                "reason": "COLD_MANIFOLD_BYPASS",
                "landmarks_count": m_count,
                "min_required_landmarks": min_landmarks,
                "latency_ms": 0.0,
                "embed_latency_ms": 0.0,
            }

        # 0b. Structural Semantic-Suitability Guard: Bi-encoder vector similarity cannot resolve
        # binary polarity (yes/no, true/false), ordinal severity levels, bare enum codes, or negated clauses.
        if (
            enforce_manifold_health
            and not force_tier0
            and not is_multimodal
            and self._is_non_semantic_option_set(query, candidates)
        ):
            return {
                "action": "ESCALATE",
                "tier": 0,
                "reason": "NON_SEMANTIC_OPTIONS_BYPASS",
                "landmarks_count": m_count,
                "latency_ms": 0.0,
                "embed_latency_ms": 0.0,
            }

        # 1. Clean query to prevent prompt contamination and extract L2-normalized embedding
        clean_query = self._clean_query(query) if query else ""
        if is_multimodal:
            q = self.embed_multimodal(clean_query, images=all_images, audio=audio)
        else:
            q = self.embed([clean_query])
        t_embed = (time.perf_counter() - t0) * 1000.0

        # 2. GPU landmark projection (< 0.1 ms) & Landmark Proximity / OOD Distance Tripwire
        t_proj_start = time.perf_counter()
        similarities = torch.matmul(q.to(self.landmarks.dtype), self.landmarks.T).squeeze(0)
        p_eff = min(p_landmarks, self.M)
        top_sims, top_indices = torch.topk(similarities, p_eff)
        max_sim = top_sims[0].item()
        d_min = 1.0 - max_sim

        if (
            enforce_manifold_health
            and not force_tier0
            and not (is_multimodal and vision_markov_weight <= 0.0)
            and d_min > delta_ood
        ):
            total_lat = (time.perf_counter() - t0) * 1000.0
            return {
                "action": "ESCALATE",
                "tier": 0,
                "reason": "OOD_MANIFOLD_BYPASS",
                "d_min": round(d_min, 4),
                "delta_ood": round(delta_ood, 4),
                "max_similarity": round(max_sim, 4),
                "landmarks_count": m_count,
                "latency_ms": round(total_lat, 2),
                "embed_latency_ms": round(t_embed, 3),
            }

        # 3. Softmax weighting over nearest landmarks
        top_sims_f = top_sims.float()
        weights = F.softmax((top_sims_f - top_sims_f.max()) / tau_proj, dim=0).unsqueeze(1)

        cosine_probs_dict: dict[str, float] | None = None
        markov_probs_dict: dict[str, float] | None = None
        cosine_sims_dict: dict[str, float] | None = None
        scaffold_mode = "pure_cosine"

        # 4. Resolve candidate distribution with sharp temperature scaling
        if candidates is not None:
            if isinstance(candidates, dict):
                labels = sorted(candidates.keys())
                texts = [candidates[k] for k in labels]
            else:
                labels = [f"Option_{i}" for i in range(len(candidates))]
                texts = list(candidates)
            class_names = labels
            text_names = texts
            K = len(labels)

            # For vision/multimodal queries, default to 100% Direct Cosine Alignment
            # (vision_markov_weight = 0.0) so text-only landmark diffusion does not corrupt
            # direct visual-to-label alignment.
            m_weight = float(vision_markov_weight) if is_multimodal else float(markov_weight)
            use_markov_hybrid = (
                m_weight > 0.0
                and (m_count >= min_landmarks or not enforce_manifold_health or force_tier0)
                and getattr(self, "Q_raw", None) is not None
                and getattr(self, "lu_factor", None) is not None
                and getattr(self, "landmarks_np", None) is not None
            )
            if use_markov_hybrid:
                try:
                    B_tensor, cand_tensors = self._resolve_absorption_for_candidates(
                        labels, texts, tau_sink=tau_proj
                    )
                except Exception:
                    # (3) Ablation / Fallback Mode: If graph inversion fails, fall back to Pure Cosine
                    use_markov_hybrid = False
                    B_tensor = None
                    cand_tensors = self.get_candidate_embeddings(candidates)
            else:
                B_tensor = None
                cand_tensors = self.get_candidate_embeddings(candidates)

            # Direct semantic alignment with sharp temperature tau_temp (0.05)
            # Expands Gemma's narrow ~0.20 cosine delta into a sharp, calibrated dynamic range
            cand_sims = torch.matmul(q.float(), cand_tensors.float().T).squeeze(0)
            max_cand_sim = cand_sims.max().item()

            if max(max_sim, max_cand_sim) < tau_ood and not force_tier0:
                total_lat = (time.perf_counter() - t0) * 1000.0
                return {
                    "action": "ESCALATE",
                    "tier": 0,
                    "reason": "OUT_OF_DISTRIBUTION",
                    "d_min": round(d_min, 4),
                    "delta_ood": round(delta_ood, 4),
                    "max_similarity": round(max(max_sim, max_cand_sim), 4),
                    "threshold": tau_ood,
                    "latency_ms": round(total_lat, 2),
                    "embed_latency_ms": round(t_embed, 2),
                }

            cosine_logits = cand_sims / tau_temp
            cosine_probs_t = F.softmax(cosine_logits, dim=-1)
            cosine_probs_dict = {class_names[i]: round(cosine_probs_t[i].item(), 4) for i in range(K)}
            cosine_sims_dict = {class_names[i]: round(cand_sims[i].item(), 4) for i in range(K)}

            if use_markov_hybrid and B_tensor is not None:
                gathered_B = B_tensor[top_indices]
                raw_absorbed = (weights * gathered_B).sum(dim=0)
                markov_logits = torch.log(raw_absorbed + 1e-9) / (tau_temp * 4.0)
                markov_probs_t = F.softmax(markov_logits, dim=-1)
                markov_probs_dict = {class_names[i]: round(markov_probs_t[i].item(), 4) for i in range(K)}
                scaffold_mode = "domain_manifold" if m_count >= min_landmarks else "default_39_landmarks"

                if m_weight >= 1.0:
                    probs = markov_probs_t
                else:
                    m_norm = markov_logits - markov_logits.mean()
                    c_norm = cosine_logits - cosine_logits.mean()
                    c_weight = 1.0 - m_weight
                    probs = F.softmax(m_weight * m_norm + c_weight * c_norm, dim=-1)
            else:
                probs = cosine_probs_t
                scaffold_mode = "pure_cosine"
        else:
            if max_sim < tau_ood and not force_tier0:
                total_lat = (time.perf_counter() - t0) * 1000.0
                return {
                    "action": "ESCALATE",
                    "tier": 0,
                    "reason": "OUT_OF_DISTRIBUTION",
                    "d_min": round(d_min, 4),
                    "delta_ood": round(delta_ood, 4),
                    "max_similarity": round(max_sim, 4),
                    "threshold": tau_ood,
                    "latency_ms": round(total_lat, 2),
                    "embed_latency_ms": round(t_embed, 2),
                }

            B = self.B_matrix
            class_names = self.classes
            text_names = self.classes
            K = self.K

            # Lookup absorption probabilities (< 0.01 ms)
            gathered_B = B[top_indices]
            raw_absorbed = (weights * gathered_B).sum(dim=0)
            probs = F.softmax(torch.log(raw_absorbed + 1e-9) / tau_temp, dim=-1)
            markov_probs_dict = {class_names[i]: round(probs[i].item(), 4) for i in range(K)}
            scaffold_mode = "domain_manifold" if m_count >= min_landmarks else "default_39_landmarks"

        t_graph = (time.perf_counter() - t_proj_start) * 1000.0

        # 5. Tripwire Gating Metrics
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        top1_prob = sorted_probs[0].item()
        top2_prob = sorted_probs[1].item() if K > 1 else 0.0
        margin = top1_prob - top2_prob

        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        max_entropy = np.log(K) if K > 1 else 1.0
        norm_entropy = entropy / max_entropy

        total_lat = (time.perf_counter() - t0) * 1000.0

        top_idx = sorted_indices[0].item()
        decision_label = class_names[top_idx]
        decision_text = text_names[top_idx]

        probs_dict = {text_names[i]: round(probs[i].item(), 4) for i in range(K)}
        letter_probs = {class_names[i]: round(probs[i].item(), 4) for i in range(K)}

        # Gating Tripwires
        is_ambiguous = norm_entropy > tau_entropy
        is_low_margin = margin < tau_margin

        if is_ambiguous or is_low_margin:
            escalate_reason = "HIGH_AMBIGUITY" if is_ambiguous else "LOW_MARGIN"
            return {
                "action": "ESCALATE",
                "tier": 0,
                "reason": escalate_reason,
                "decision": decision_label,
                "decision_text": decision_text,
                "confidence": round(top1_prob, 4),
                "margin": round(margin, 4),
                "entropy": round(norm_entropy, 4),
                "tau_entropy": tau_entropy,
                "tau_margin": tau_margin,
                "probabilities": probs_dict,
                "letter_probabilities": letter_probs,
                "cosine_probabilities": cosine_probs_dict,
                "markov_probabilities": markov_probs_dict,
                "cosine_similarities": cosine_sims_dict,
                "scaffold_mode": scaffold_mode,
                "landmarks_count": m_count,
                "d_min": round(d_min, 4),
                "delta_ood": round(delta_ood, 4),
                "max_similarity": round(max_sim, 4),
                "latency_ms": round(total_lat, 2),
                "embed_latency_ms": round(t_embed, 3),
                "graph_latency_ms": round(t_graph, 3),
            }

        return {
            "action": "EXIT_0",
            "tier": 0,
            "decision": decision_label,
            "decision_text": decision_text,
            "confidence": round(top1_prob, 4),
            "margin": round(margin, 4),
            "entropy": round(norm_entropy, 4),
            "tau_entropy": tau_entropy,
            "tau_margin": tau_margin,
            "probabilities": probs_dict,
            "letter_probabilities": letter_probs,
            "cosine_probabilities": cosine_probs_dict,
            "markov_probabilities": markov_probs_dict,
            "cosine_similarities": cosine_sims_dict,
            "scaffold_mode": scaffold_mode,
            "landmarks_count": m_count,
            "d_min": round(d_min, 4),
            "delta_ood": round(delta_ood, 4),
            "max_similarity": round(max_sim, 4),
            "latency_ms": round(total_lat, 2),
            "embed_latency_ms": round(t_embed, 3),
            "graph_latency_ms": round(t_graph, 3),
        }


_GLOBAL_TIER0_ENGINE: Tier0Engine | None = None


def is_tier0_loaded() -> bool:
    """Returns True if the Tier 0 Embedding model is currently loaded in memory."""
    global _GLOBAL_TIER0_ENGINE
    return _GLOBAL_TIER0_ENGINE is not None


def unload_tier0_engine() -> None:
    """Unloads Tier 0 Embedding model and releases GPU/CPU memory."""
    global _GLOBAL_TIER0_ENGINE
    if _GLOBAL_TIER0_ENGINE is not None:
        try:
            if hasattr(_GLOBAL_TIER0_ENGINE, "llama_model") and _GLOBAL_TIER0_ENGINE.llama_model is not None:
                del _GLOBAL_TIER0_ENGINE.llama_model
            if hasattr(_GLOBAL_TIER0_ENGINE, "onnx_text_session"):
                del _GLOBAL_TIER0_ENGINE.onnx_text_session
            if hasattr(_GLOBAL_TIER0_ENGINE, "onnx_vision_session"):
                del _GLOBAL_TIER0_ENGINE.onnx_vision_session
            if hasattr(_GLOBAL_TIER0_ENGINE, "onnx_audio_session"):
                del _GLOBAL_TIER0_ENGINE.onnx_audio_session
            if hasattr(_GLOBAL_TIER0_ENGINE, "processor"):
                del _GLOBAL_TIER0_ENGINE.processor
            if hasattr(_GLOBAL_TIER0_ENGINE, "raw_embedder"):
                del _GLOBAL_TIER0_ENGINE.raw_embedder
            if hasattr(_GLOBAL_TIER0_ENGINE, "embedder"):
                del _GLOBAL_TIER0_ENGINE.embedder
            if hasattr(_GLOBAL_TIER0_ENGINE, "tokenizer"):
                del _GLOBAL_TIER0_ENGINE.tokenizer
            if hasattr(_GLOBAL_TIER0_ENGINE, "landmarks"):
                del _GLOBAL_TIER0_ENGINE.landmarks
            if hasattr(_GLOBAL_TIER0_ENGINE, "B_matrix"):
                del _GLOBAL_TIER0_ENGINE.B_matrix
        except Exception:
            pass
        _GLOBAL_TIER0_ENGINE = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Tier 0] Embedding model successfully unloaded and memory reclaimed.")


def load_tier0_engine(
    model_path: str | None = None,
    device: str | None = None,
    use_compile: bool = False,
    enable_vision: bool = True,
    enable_audio: bool = True,
) -> Tier0Engine:
    """Explicitly initializes and loads Tier 0 engine."""
    global _GLOBAL_TIER0_ENGINE
    unload_tier0_engine()
    _GLOBAL_TIER0_ENGINE = Tier0Engine(
        model_path=model_path,
        device=device,
        use_compile=use_compile,
        enable_vision=enable_vision,
        enable_audio=enable_audio,
    )
    return _GLOBAL_TIER0_ENGINE


def get_tier0_engine(
    model_path: str | None = None,
    device: str | None = None,
    use_compile: bool = False,
    auto_load: bool = True,
    enable_vision: bool = True,
    enable_audio: bool = True,
) -> Tier0Engine | None:
    """Returns global singleton Tier0Engine.
    
    If auto_load is False and engine is not already loaded, returns None.
    """
    global _GLOBAL_TIER0_ENGINE
    if _GLOBAL_TIER0_ENGINE is None:
        if not auto_load:
            return None
        _GLOBAL_TIER0_ENGINE = Tier0Engine(
            model_path=model_path,
            device=device,
            use_compile=use_compile,
            enable_vision=enable_vision,
            enable_audio=enable_audio,
        )
    return _GLOBAL_TIER0_ENGINE


