import base64
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


from . import __version__
from .config import (
    DEFAULT_COT_MAX_TOKENS,
    DEFAULT_COT_PROMPT,
    DEFAULT_COT_TEMP,
    DEFAULT_COT_THRESHOLD,
    DEFAULT_ESCALATE_THRESHOLD,
    PROJECT_ROOT,
    _LLM_CONFIG_PATH,
    _load_llm_config,
    get_active_profile_name,
    get_smallest_model,
    resolve_llm_kwargs,
    resolve_mmproj_path,
    resolve_model_path,
    scan_mmproj_models,
    scan_models,
)
from .engine import GemmaDecisionEngine, load_image_bytes
from .schema import AuditRequest, DecisionEvidence, DecisionResponse

_UI_DASHBOARD_DIR = PROJECT_ROOT / "ui" / "web_dashboard"
_BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
_TESTS_DIR = PROJECT_ROOT / "tests"
if not _TESTS_DIR.is_dir() and _BENCHMARKS_DIR.is_dir():
    _TESTS_DIR = _BENCHMARKS_DIR


class DecisionRequest(BaseModel):
    context: str = Field("", json_schema_extra={"example": "User prompt or document context to classify"})
    options: dict[str, str] | list[str] = Field(
        ..., json_schema_extra={"example": {"A": "Billing Inquiry", "B": "Technical Bug", "C": "Churn Risk"}}
    )
    image: str | None = Field(None, description="Base64 data URL, raw base64, or local file path")
    images: list[str] | None = Field(None, description="List of base64 images or file paths")
    image_path: str | None = Field(None, description="Local image path or URL")
    image_bytes: str | None = Field(None, description="Base64 encoded image string")
    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0)
    escalate_threshold: float = Field(DEFAULT_ESCALATE_THRESHOLD, ge=0.0, le=1.0)
    cyclic_debias: bool = Field(False, description="Run cyclic label-permutation debiasing.")
    kv_branching: bool | None = Field(None, description="Enable KV-cache branching for cyclic debiasing.")
    kv_branching_min_tokens: int | None = Field(None, ge=1, description="Minimum prefix tokens required to trigger KV-cache branching.")
    max_cot_tokens: int | None = Field(None, ge=1, le=2048)
    cot_temp: float | None = Field(None, ge=0.0, le=2.0)
    cot_prompt: str | None = Field(None, description="Optional custom CoT reasoning prompt prefix.")
    use_calibrator: bool | None = Field(None)
    include_evidence: bool = Field(False, description="Extract source quotes and operational rationale post-decision.")
    test_id: str | None = Field(None)


class SwitchProfileRequest(BaseModel):
    profile: str = Field(..., json_schema_extra={"example": "16gb_preserve_vram"})


class SwitchConfigRequest(BaseModel):
    model_path: str | None = Field(None, json_schema_extra={"example": "models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"})
    n_ctx: int | None = Field(None, json_schema_extra={"example": 8192})
    profile: str | None = Field(None, json_schema_extra={"example": "5090_throughput"})
    kv_cache: str | None = Field(None, json_schema_extra={"example": "F16"})
    cot_prompt: str | None = Field(None, json_schema_extra={"example": "Analysis: First calculate..."})
    cot_temp: float | None = Field(None, ge=0.0, le=2.0)
    max_cot_tokens: int | None = Field(None, ge=16, le=2048)


class ChatMessage(BaseModel):
    role: str = Field("user", description="Role: system, user, or assistant")
    content: Any = Field(..., description="Message text or multimodal content items")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="List of messages in conversation")
    model: str | None = Field(None, description="Optional model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=16384)
    stream: bool = Field(False, description="Stream Server-Sent Events (SSE)")
    response_format: dict[str, Any] | None = Field(None, description="e.g. {'type': 'json_object'}")
    stop: list[str] | str | None = Field(None)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or instructions")
    system: str | None = Field(None, description="Optional system instruction")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=16384)
    json_mode: bool = Field(False, description="Enforce JSON object response format")
    stream: bool = Field(False, description="Stream Server-Sent Events (SSE)")
    image: str | None = Field(None, description="Base64 data URL, raw base64, or local file path")


def build_app(engine: GemmaDecisionEngine) -> FastAPI:

    app = FastAPI(title="Project Gevva0 Gateway", version=__version__)
    app.state.engine = engine
    _reload_lock = threading.Lock()

    def _engine() -> GemmaDecisionEngine:
        return app.state.engine

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui/")

    @app.get("/health")
    def health() -> dict[str, Any]:
        eng = _engine()
        cal = eng.calibrator
        return {
            "status": "ok",
            "model": eng.model_path,
            "has_vision": eng.has_vision,
            "mmproj": eng.mmproj_path,
            "mmproj_variant": getattr(eng, "mmproj_variant", None),
            "n_ctx": eng.n_ctx,
            "profile": get_active_profile_name(),
            "temperature": cal.temperature if cal else None,
        }

    # ── Model & Config management ─────────────────────────────

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        """Scan models directory and return available models and current active config."""
        eng = _engine()
        scanned = scan_models()
        active_model = eng.model_path
        try:
            rel_active = Path(active_model).relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            rel_active = active_model

        smallest = get_smallest_model()

        return {
            "models": scanned,
            "active_model": rel_active,
            "active_n_ctx": eng.n_ctx,
            "smallest_model": smallest["path"] if smallest else None,
            "has_vision": eng.has_vision,
            "active_mmproj": eng.mmproj_path,
            "active_mmproj_variant": getattr(eng, "mmproj_variant", None),
            "mmproj_models": scan_mmproj_models(),
        }

    @app.post("/v1/config/switch")
    def switch_config(req: SwitchConfigRequest) -> dict[str, Any]:
        """Switch model path and/or context size, persist to llm_config.json, and reload engine."""
        cfg = _load_llm_config()
        if cfg is None:
            raise HTTPException(status_code=404, detail="llm_config.json not found")

        reload_needed = False
        if req.model_path:
            resolved_p = resolve_model_path(req.model_path)
            if not resolved_p.is_file():
                raise HTTPException(status_code=400, detail=f"Model file not found: {req.model_path}")

            try:
                rel_model_path = resolved_p.relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel_model_path = str(resolved_p)

            if "model" not in cfg:
                cfg["model"] = {}
            cfg["model"]["model_path"] = rel_model_path

            matching_mmproj = resolve_mmproj_path(resolved_p)
            if matching_mmproj and matching_mmproj.is_file():
                try:
                    rel_mmproj = matching_mmproj.relative_to(PROJECT_ROOT).as_posix()
                except Exception:
                    rel_mmproj = str(matching_mmproj)
                cfg["model"]["mmproj_path"] = rel_mmproj
            else:
                cfg["model"]["mmproj_path"] = None
            reload_needed = True

        if req.n_ctx is not None:
            if req.n_ctx < 512 or req.n_ctx > 131072:
                raise HTTPException(status_code=400, detail="n_ctx must be between 512 and 131072")
            cfg["n_ctx"] = req.n_ctx
            reload_needed = True

        if req.kv_cache is not None:
            norm_kv = req.kv_cache.strip().upper()
            if norm_kv in ("F16", "FP16"):
                cfg["kv_cache"] = "F16"
                cfg["type_k"] = 1
                cfg["type_v"] = 1
            elif norm_kv == "Q8_0":
                cfg["kv_cache"] = "Q8_0"
                cfg["type_k"] = 8
                cfg["type_v"] = 8
            elif norm_kv == "Q4_0":
                cfg["kv_cache"] = "Q4_0"
                cfg["type_k"] = 2
                cfg["type_v"] = 2
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported kv_cache precision: {req.kv_cache}")
            reload_needed = True

        if "decision" not in cfg:
            cfg["decision"] = {}
        if req.cot_prompt is not None:
            cfg["decision"]["cot_prompt"] = req.cot_prompt
        if req.cot_temp is not None:
            cfg["decision"]["cot_temp"] = req.cot_temp
        if req.max_cot_tokens is not None:
            cfg["decision"]["cot_max_tokens"] = req.max_cot_tokens

        import gc
        import json

        with open(_LLM_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

        if reload_needed:
            with _reload_lock:
                old_engine = getattr(app.state, "engine", None)
                if old_engine is not None and hasattr(old_engine, "close"):
                    try:
                        old_engine.close()
                    except Exception:
                        pass
                    app.state.engine = None
                    del old_engine
                    gc.collect()

                try:
                    if req.model_path:
                        new_engine = GemmaDecisionEngine.from_config(
                            model_path=str(resolved_p),
                            mmproj_path=str(matching_mmproj) if matching_mmproj else None,
                        )
                    else:
                        new_engine = GemmaDecisionEngine.from_config()
                    app.state.engine = new_engine
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Failed to reload model: {exc}")
        else:
            new_engine = _engine()

        try:
            rel_model = Path(new_engine.model_path).relative_to(PROJECT_ROOT).as_posix()
        except Exception:
            rel_model = new_engine.model_path

        return {
            "status": "ok",
            "active_model": rel_model,
            "n_ctx": new_engine.n_ctx,
            "profile": None,
        }

    # ── Profile management (Deprecated) ─────────────────────────

    @app.get("/v1/profiles")
    def list_profiles() -> dict[str, Any]:
        """Return deprecated profiles info."""
        return {
            "active_profile": None,
            "profiles": {},
            "deprecated": True,
            "message": "Profiles have been deprecated in favor of explicit n_ctx settings."
        }

    @app.post("/v1/profiles/switch")
    def switch_profile(req: SwitchProfileRequest) -> dict[str, Any]:
        """Deprecated profile switch endpoint."""
        raise HTTPException(
            status_code=400,
            detail="Profiles are no longer used. Please set context size directly via /v1/config/switch with n_ctx.",
        )

    # ── Decision endpoint ─────────────────────────────────────

    @app.post("/v1/decide", response_model=DecisionResponse)
    @app.post("/api/decide", response_model=DecisionResponse)
    def make_decision(req: DecisionRequest) -> DecisionResponse:
        eng = _engine()
        options_dict: dict[str, str]
        if isinstance(req.options, list):
            from .debias import LETTERS
            options_dict = {LETTERS[i]: str(v).strip() for i, v in enumerate(req.options) if i < len(LETTERS)}
        else:
            options_dict = dict(req.options)

        if len(options_dict) < 2 or len(options_dict) > 26:
            raise HTTPException(status_code=400, detail="Options count must be between 2 and 26.")
        img_path = getattr(req, "image_path", None)
        img_bytes = getattr(req, "image_bytes", None)
        has_image = bool(req.image or req.images or img_path or img_bytes)
        if not req.context.strip() and not has_image:
            raise HTTPException(status_code=400, detail="Either context or image must be provided.")

        try:
            t0 = time.perf_counter()
            image_input = req.images or req.image or img_path or img_bytes
            cot_thresh = req.confidence_threshold
            if cot_thresh is None:
                decision_cfg = (_load_llm_config() or {}).get("decision", {})
                cot_thresh = decision_cfg.get("cot_threshold", DEFAULT_COT_THRESHOLD)

            decide_kwargs: dict[str, Any] = {
                "context": req.context,
                "options": options_dict,
                "image": image_input,
                "cot_threshold": cot_thresh,
                "cyclic_debias": req.cyclic_debias,
                "include_evidence": req.include_evidence,
                "test_id": req.test_id,
            }
            if req.kv_branching is not None:
                decide_kwargs["kv_branching"] = req.kv_branching
            if req.kv_branching_min_tokens is not None:
                decide_kwargs["kv_branching_min_tokens"] = req.kv_branching_min_tokens
            if req.max_cot_tokens is not None:
                decide_kwargs["max_cot_tokens"] = req.max_cot_tokens
            if req.cot_temp is not None:
                decide_kwargs["cot_temp"] = req.cot_temp
            if req.cot_prompt is not None:
                decide_kwargs["cot_prompt"] = req.cot_prompt
            if req.use_calibrator is not None:
                decide_kwargs["use_calibrator"] = req.use_calibrator

            result = eng.decide(**decide_kwargs)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            if isinstance(result, DecisionResponse):
                result.duration_ms = round(duration_ms, 2)
                if req.escalate_threshold > 0:
                    result.escalate = result.confidence < req.escalate_threshold
                return result

            letter = result.get("candidate_letter") or result.get("decision", "")
            text = result.get("decision_text") or (result.get("decision", "") if letter != result.get("decision") else "")
            return DecisionResponse(
                test_id=req.test_id,
                decision=letter,
                decision_text=text,
                confidence=result.get("confidence", 0.0),
                latency_ms=result.get("latency_ms", round(duration_ms, 2)),
                duration_ms=round(duration_ms, 2),
                candidate_letter=letter,
                probabilities=result.get("probabilities"),
                letter_probabilities=result.get("letter_probabilities"),
                mode=result.get("mode", "fast_path"),
                escalate=(result["confidence"] < req.escalate_threshold) if req.escalate_threshold > 0 else False,
                evidence=result.get("evidence"),
                cot_scratchpad=result.get("cot_scratchpad"),
                image_attached=bool(image_input),
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ── JevBench / TypeSafe wire format endpoints ─────────────

    @app.post("/decide")
    @app.post("/v1/systemone")
    def handle_jev_wire(body: dict[str, Any]) -> Any:
        eng = _engine()
        # If this is already a Gevva0 DecisionRequest with 'options' and 'context':
        if "options" in body and "questions" not in body:
            return make_decision(DecisionRequest(**body))

        state = str(body.get("state", ""))
        questions = body.get("questions")
        is_dict_questions = isinstance(questions, dict)

        q_items = []
        if is_dict_questions:
            for q_id, q_spec in questions.items():
                q_items.append((q_id, q_spec))
        elif isinstance(questions, list):
            for q_spec in questions:
                q_items.append((q_spec.get("id", "decision"), q_spec))
        else:
            raise HTTPException(status_code=400, detail="Missing or invalid 'questions' in request.")

        from .debias import LETTERS

        answers_list = []
        answers_dict = {}
        last_latency_ms = 0.0

        for q_id, q_spec in q_items:
            q_type = q_spec.get("type", "choice")
            instructions = q_spec.get("instructions", "")
            raw_opts = q_spec.get("options") or q_spec.get("levels") or q_spec.get("criteria") or {}

            if isinstance(raw_opts, dict):
                labels = list(raw_opts.keys())
                descs = [f"{k}: {v}" if v and str(v).strip() != str(k).strip() else str(k) for k, v in raw_opts.items()]
            elif isinstance(raw_opts, (list, tuple)):
                labels = [str(i) for i in range(len(raw_opts))]
                descs = [f"Level {i}: {v}" for i, v in enumerate(raw_opts)]
            else:
                labels = []
                descs = []

            if not labels:
                raise HTTPException(status_code=400, detail=f"Question '{q_id}' has no options/levels/criteria.")

            letters = [LETTERS[i] for i in range(min(len(labels), len(LETTERS)))]
            opt_dict = {letters[i]: descs[i] for i in range(len(letters))}
            full_context = f"{state}\n\nQuestion: {instructions}" if instructions else state

            res = eng.decide(context=full_context, options=opt_dict)
            let_probs = getattr(res, "letter_probabilities", None) or {}
            last_latency_ms = getattr(res, "latency_ms", 0.0)

            # Map letter probabilities back to exact labels
            probs = {labels[i]: float(let_probs.get(letters[i], 0.0)) for i in range(len(letters))}
            tot = sum(probs.values())
            if tot > 0:
                probs = {k: v / tot for k, v in probs.items()}
            best_label = max(probs, key=probs.get)

            ans_entry = {
                "id": q_id,
                "type": q_type,
                "probabilities": probs,
                "choice": best_label,
            }
            if q_type == "noul":
                ans_entry["noul"] = probs.get("yes", 0.0)

            answers_list.append(ans_entry)
            answers_dict[q_id] = ans_entry

        if is_dict_questions:
            return {
                "model": "gemma-4-26B-A4B-it",
                "answers": answers_dict,
                "usage": {"input_tokens": 0, "output_tokens": 1},
            }
        else:
            return {
                "model": "gemma-4-26B-A4B-it",
                "answers": answers_list,
                "latency_ms": last_latency_ms,
            }

    # ── Audit Evidence endpoint ───────────────────────────────

    @app.post("/api/audit-evidence", response_model=DecisionEvidence)
    @app.post("/v1/audit-evidence", response_model=DecisionEvidence)
    def get_audit_evidence(req: AuditRequest) -> DecisionEvidence:
        """
        Called when the user clicks 'Inspect Audit Trail' / 'View Evidence'.
        Extracts verbatim source quotes and operational rationale post-decision.
        """
        eng = _engine()
        try:
            image_bytes_list = [load_image_bytes(req.image)] if req.image else None
            return eng.explain_decision(
                context=req.context,
                verdict_letter=req.verdict_letter,
                verdict_text=req.verdict_text,
                image_bytes_list=image_bytes_list,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Traditional Autoregressive Generation & Chat API ──────

    @app.post("/v1/chat/completions")
    @app.post("/api/chat")
    def chat_completions(req: ChatCompletionRequest):
        """
        OpenAI-compatible chat completion endpoint.
        Uses the shared Gemma 4 model in VRAM for autoregressive text/JSON generation.
        """
        eng = _engine()
        msg_dicts = [m.model_dump() for m in req.messages]

        if req.stream:
            def event_generator():
                try:
                    for chunk in eng.stream_chat_completion(
                        messages=msg_dicts,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        max_tokens=req.max_tokens,
                        response_format=req.response_format,
                        stop=req.stop,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    err_chunk = {"error": {"message": str(exc), "type": "server_error"}}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            try:
                return eng.chat_completion(
                    messages=msg_dicts,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    response_format=req.response_format,
                    stop=req.stop,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/generate")
    def generate_completion(req: GenerateRequest):
        """
        Convenience endpoint for single-turn prompt text/JSON generation.
        """
        eng = _engine()
        messages: list[dict[str, Any]] = []
        if req.system and req.system.strip():
            messages.append({"role": "system", "content": req.system.strip()})

        if req.image:
            try:
                b = load_image_bytes(req.image)
                b64_url = f"data:image/png;base64,{base64.b64encode(b).decode('utf-8')}"
                user_content = [
                    {"type": "text", "text": req.prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ]
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}")
        else:
            user_content = req.prompt

        messages.append({"role": "user", "content": user_content})
        response_format = {"type": "json_object"} if req.json_mode else None

        if req.stream:
            def event_generator():
                try:
                    for chunk in eng.stream_chat_completion(
                        messages=messages,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        max_tokens=req.max_tokens,
                        response_format=response_format,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    err_chunk = {"error": {"message": str(exc), "type": "server_error"}}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            try:
                res = eng.chat_completion(
                    messages=messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    response_format=response_format,
                )
                choice = res.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                return {
                    "text": text,
                    "usage": res.get("usage", {}),
                    "model": res.get("model", getattr(eng, "model_path", "")),
                }
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/v1/config/defaults")

    def get_config_defaults() -> dict[str, Any]:
        """Return runtime decision settings and defaults from config."""
        cfg = _load_llm_config() or {}
        decision_cfg = cfg.get("decision", {})

        thresh = float(decision_cfg.get("cot_threshold", 0.0))
        if thresh <= 0.0:
            policy = "fast_path"
        elif thresh >= 1.0:
            policy = "always_verify"
        else:
            policy = "balanced"

        kv_cache_str = cfg.get("kv_cache")
        if not kv_cache_str:
            tk = cfg.get("type_k")
            if tk == 1:
                kv_cache_str = "F16"
            elif tk == 8:
                kv_cache_str = "Q8_0"
            elif tk == 2:
                kv_cache_str = "Q4_0"
            else:
                kv_cache_str = "F16"

        return {
            "routing_policy": policy,
            "balanced_threshold": 0.50,
            "cot_threshold": thresh,
            "cyclic_debias": bool(decision_cfg.get("cyclic_debias", False)),
            "escalate_enabled": False,
            "escalate_threshold": 0.70,
            "cot_temp": float(decision_cfg.get("cot_temp", DEFAULT_COT_TEMP)),
            "max_cot_tokens": int(decision_cfg.get("cot_max_tokens", DEFAULT_COT_MAX_TOKENS)),
            "cot_prompt": str(decision_cfg.get("cot_prompt", DEFAULT_COT_PROMPT)),
            "kv_cache": kv_cache_str,
            "use_calibrator": True,
        }

    # ── Benchmark / Test Suite endpoint ───────────────────────

    @app.get("/benchmarks/benchmark_suite.json")
    @app.get("/tests/tests.json")
    def get_tests():
        json_file = _BENCHMARKS_DIR / "benchmark_suite.json"
        if not json_file.is_file():
            json_file = _TESTS_DIR / "tests.json"
        if not json_file.is_file():
            raise HTTPException(status_code=404, detail="benchmark suite json not found")
        import json

        with open(json_file, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Mount static files, benchmarks, fixtures, and UI dashboards
    fixtures_dir = _BENCHMARKS_DIR / "fixtures"
    if fixtures_dir.is_dir():
        app.mount("/benchmarks/fixtures", StaticFiles(directory=str(fixtures_dir)), name="benchmarks_fixtures")
        app.mount("/tests/fixtures", StaticFiles(directory=str(fixtures_dir)), name="tests_fixtures")
        app.mount("/fixtures", StaticFiles(directory=str(fixtures_dir)), name="fixtures")

    if _BENCHMARKS_DIR.is_dir():
        app.mount("/benchmarks", StaticFiles(directory=str(_BENCHMARKS_DIR)), name="benchmarks")

    if _TESTS_DIR.is_dir():
        app.mount("/tests", StaticFiles(directory=str(_TESTS_DIR)), name="tests")

    if _UI_DASHBOARD_DIR.is_dir():
        app.mount("/ui/web_dashboard", StaticFiles(directory=str(_UI_DASHBOARD_DIR), html=True), name="ui_web_dashboard")
        app.mount("/ui", StaticFiles(directory=str(_UI_DASHBOARD_DIR), html=True), name="ui")

    @app.get("/chat", include_in_schema=False)
    @app.get("/chat.html", include_in_schema=False)
    def chat_redirect():
        return RedirectResponse(url="/ui/chat.html")


    return app


class _LazyApp:
    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _resolve(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app

    async def __call__(self, scope, receive, send) -> None:
        await self._resolve()(scope, receive, send)


def create_app(engine: GemmaDecisionEngine | None = None) -> FastAPI:
    if engine is None:
        engine = GemmaDecisionEngine.from_config()
    return build_app(engine)


app = _LazyApp()
