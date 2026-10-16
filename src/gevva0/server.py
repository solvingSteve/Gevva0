import base64
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


from . import __version__
from .calibration import TemperatureCalibrator, calibrate_model
from .config import (
    DEFAULT_COT_MAX_TOKENS,
    DEFAULT_COT_PROMPT,
    DEFAULT_COT_TEMP,
    DEFAULT_COT_THRESHOLD,
    DEFAULT_ESCALATE_THRESHOLD,
    PROJECT_ROOT,
    _LLM_CONFIG_PATH,
    _load_llm_config,
    find_model_calibration_path,
    get_active_profile_name,
    get_smallest_model,
    resolve_calibration_path,
    resolve_llm_kwargs,
    resolve_mmproj_path,
    resolve_model_path,
    scan_mmproj_models,
    scan_models,
    scan_embedding_models,
    update_config_calibration,
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
    enable_rules: bool = Field(True, description="Enable Tier -1 Heuristic Rule Gateway.")
    early_exit: bool = Field(False, description="Enable Tier 0 Topological Absorbing Early Exit.")
    force_tier0: bool = Field(False, description="Force Tier 0 evaluation even when manifold has fewer than 50 landmarks.")
    tier0_entropy_threshold: float | None = Field(None, ge=0.0, le=1.0)
    tier0_margin_threshold: float | None = Field(None, ge=0.0, le=1.0)
    tier0_landmark_distance_threshold: float | None = Field(None, ge=0.0, le=2.0)
    cyclic_debias: bool = Field(False, description="Run cyclic label-permutation debiasing.")
    kv_branching: bool | None = Field(None, description="Enable KV-cache branching for cyclic debiasing.")
    kv_branching_min_tokens: int | None = Field(None, ge=1, description="Minimum prefix tokens required to trigger KV-cache branching.")
    max_cot_tokens: int | None = Field(None, ge=1, le=2048)
    cot_temp: float | None = Field(None, ge=0.0, le=2.0)
    cot_prompt: str | None = Field(None, description="Optional custom CoT reasoning prompt prefix.")
    use_calibrator: bool | None = Field(None)
    include_evidence: bool = Field(False, description="Extract source quotes and operational rationale post-decision.")
    test_id: str | None = Field(None)


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Query or prompt text to classify")
    candidates: list[str] = Field(..., description="List of candidate categories or actions")
    early_exit: bool = Field(True, description="Enable Tier 0 early exit")
    force_tier0: bool = Field(False, description="Force Tier 0 evaluation even when manifold has fewer than 50 landmarks")
    tau_entropy: float = Field(0.65, ge=0.0, le=1.0, description="Shannon entropy threshold tau_1")
    tau_margin: float = Field(0.25, ge=0.0, le=1.0, description="Classification margin threshold tau_2")
    tau_ood: float = Field(0.40, ge=0.0, le=1.0, description="Out-of-distribution candidate similarity threshold")
    delta_ood: float = Field(0.22, ge=0.0, le=2.0, description="Landmark proximity / OOD cosine distance threshold delta_OOD")


class ClassifyResponse(BaseModel):
    tier: int = Field(..., description="Resolved tier: 0 (Early Exit), 1 (Fast-Path), or 2 (CoT)")
    decision: str = Field(..., description="Selected candidate label or category")
    confidence: float = Field(..., description="Decision confidence probability")
    latency_ms: float = Field(..., description="Observed end-to-end latency in milliseconds")
    mode: str = Field("tier0_early_exit", description="Execution mode")
    escalated_from_tier0: bool = False
    tier0_reason: str | None = None
    probabilities: dict[str, float] | None = None
    entropy: float | None = None
    margin: float | None = None


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
    cot_threshold: float | None = Field(None, ge=0.0, le=1.0)
    temperature: float | None = Field(None, ge=0.01, le=50.0)
    vision: bool | None = Field(None, description="Enable or disable multimodal projector (Vision)")
    load_vision: bool | None = Field(None, description="Alias for vision")


class LoadTier0Request(BaseModel):
    model_path: str | None = Field(None, description="Path or identifier of embedding model to load")
    device: str | None = Field(None, description="Target device ('cuda' or 'cpu')")
    vision: bool | None = Field(True, description="Enable vision encoder if supported by model")
    audio: bool | None = Field(True, description="Enable audio encoder if supported by model")


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


class ReloadRulesRequest(BaseModel):
    filename: str | None = None


class SelectRuleFileRequest(BaseModel):
    filename: str


class SaveRulesRequest(BaseModel):
    filename: str | None = None
    data: dict[str, Any]



def build_app(engine: GemmaDecisionEngine | None = None) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Initialize Tier -1 Rule Gateway from RoutingRules JSON configuration
        print("[startup] Initializing Tier -1 Rule-Based Gateway...")
        try:
            from .rules import get_rule_gateway
            gw = get_rule_gateway()
            print(f"[startup] Tier -1 Gateway ready: {len(gw.rules)} rules loaded from {gw.active_file}")
        except Exception as exc:
            print(f"[startup] Tier -1 Gateway note: {exc}")

        # Models (Tier 0 Embedding & Backbone LLM) are loaded on-demand via UI or API.
        # Web UI and endpoints are available immediately without waiting for model loads.
        print("[startup] Gateway ready. Web UI and endpoints are available immediately.")

        yield

        if getattr(app.state, "engine", None) is not None:
            try:
                app.state.engine.close()
            except Exception:
                pass

    app = FastAPI(title="Project Gevva0 Gateway", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    _reload_lock = threading.Lock()

    def _engine() -> GemmaDecisionEngine | None:
        return getattr(app.state, "engine", None)

    def _require_engine() -> GemmaDecisionEngine:
        eng = _engine()
        if eng is None:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Please load a model first.",
            )
        return eng


    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui/")

    @app.get("/health")
    def health() -> dict[str, Any]:
        eng = _engine()
        from .tier0 import is_tier0_loaded, get_tier0_engine
        emb_loaded = is_tier0_loaded()
        t0 = get_tier0_engine() if emb_loaded else None
        m_count = int(getattr(t0, "M", 0)) if t0 else 0

        if eng is None:
            return {
                "status": "ok" if emb_loaded else "no_model_loaded",
                "llm_loaded": False,
                "embedding_loaded": emb_loaded,
                "landmarks_count": m_count,
                "model": None,
                "has_vision": False,
                "mmproj": None,
                "mmproj_variant": None,
                "n_ctx": None,
                "profile": None,
                "temperature": None,
            }
        cal = eng.calibrator
        rel_cal_path = None
        if cal and cal.metadata.get("calibration_path"):
            try:
                rel_cal_path = Path(cal.metadata["calibration_path"]).relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel_cal_path = str(cal.metadata["calibration_path"])
        elif eng.model_path:
            p = find_model_calibration_path(eng.model_path)
            if p:
                try:
                    rel_cal_path = p.relative_to(PROJECT_ROOT).as_posix()
                except Exception:
                    rel_cal_path = str(p)

        return {
            "status": "ok",
            "llm_loaded": True,
            "embedding_loaded": emb_loaded,
            "landmarks_count": m_count,
            "model": eng.model_path,
            "has_vision": eng.has_vision,
            "mmproj": eng.mmproj_path,
            "mmproj_variant": getattr(eng, "mmproj_variant", None),
            "n_ctx": eng.n_ctx,
            "profile": get_active_profile_name(),
            "temperature": cal.temperature if cal else None,
            "temperatures": cal.temperatures if cal else {},
            "calibrated": cal is not None,
            "cot_threshold": cal.cot_threshold if cal else None,
            "calibration_path": rel_cal_path,
            "calibration_metadata": cal.metadata if cal else None,
        }


    # ── Model & Config management ─────────────────────────────

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        """Scan models directory and return available models and current active config."""
        eng = _engine()
        scanned = scan_models()
        active_model = eng.model_path if eng else None
        rel_active = None
        if active_model:
            try:
                rel_active = Path(active_model).relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel_active = active_model

        smallest = get_smallest_model()

        return {
            "models": scanned,
            "active_model": rel_active,
            "active_n_ctx": eng.n_ctx if eng else (_load_llm_config() or {}).get("n_ctx", 8192),
            "smallest_model": smallest["path"] if smallest else None,
            "has_vision": eng.has_vision if eng else False,
            "active_mmproj": eng.mmproj_path if eng else None,
            "active_mmproj_variant": getattr(eng, "mmproj_variant", None) if eng else None,
            "mmproj_models": scan_mmproj_models(),
        }

    @app.post("/v1/config/switch")
    def switch_config(req: SwitchConfigRequest) -> dict[str, Any]:
        """Switch model path and/or context size, persist to llm_config.json, and reload engine."""
        cfg = _load_llm_config()
        if cfg is None:
            cfg = {}

        reload_needed = False
        resolved_p = None
        matching_mmproj = None
        vision_opt = req.vision if req.vision is not None else req.load_vision

        target_model = req.model_path
        if not target_model and getattr(app.state, "engine", None) is None:
            target_model = cfg.get("model", {}).get("model_path")

        if target_model:
            resolved_p = resolve_model_path(target_model)
            if not resolved_p.is_file():
                raise HTTPException(status_code=400, detail=f"Model file not found: {target_model}")

            try:
                rel_model_path = resolved_p.relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel_model_path = str(resolved_p)

            if "model" not in cfg:
                cfg["model"] = {}
            cfg["model"]["model_path"] = rel_model_path

            if vision_opt is False:
                matching_mmproj = None
                cfg["model"]["mmproj_path"] = None
            elif vision_opt is True:
                matching_mmproj = resolve_mmproj_path(resolved_p)
                if matching_mmproj and matching_mmproj.is_file():
                    try:
                        rel_mmproj = matching_mmproj.relative_to(PROJECT_ROOT).as_posix()
                    except Exception:
                        rel_mmproj = str(matching_mmproj)
                    cfg["model"]["mmproj_path"] = rel_mmproj
                else:
                    cfg["model"]["mmproj_path"] = None
                    matching_mmproj = None
            else:
                matching_mmproj = resolve_mmproj_path(resolved_p)
                if matching_mmproj and matching_mmproj.is_file():
                    try:
                        rel_mmproj = matching_mmproj.relative_to(PROJECT_ROOT).as_posix()
                    except Exception:
                        rel_mmproj = str(matching_mmproj)
                    cfg["model"]["mmproj_path"] = rel_mmproj
                else:
                    cfg["model"]["mmproj_path"] = None
                    matching_mmproj = None
            reload_needed = True
        elif vision_opt is not None:
            curr_model = getattr(getattr(app.state, "engine", None), "model_path", None) or cfg.get("model", {}).get("model_path")
            if curr_model:
                resolved_p = resolve_model_path(curr_model)
                if vision_opt:
                    matching_mmproj = resolve_mmproj_path(resolved_p)
                    if matching_mmproj and matching_mmproj.is_file():
                        try:
                            rel_mmproj = matching_mmproj.relative_to(PROJECT_ROOT).as_posix()
                        except Exception:
                            rel_mmproj = str(matching_mmproj)
                        cfg.setdefault("model", {})["mmproj_path"] = rel_mmproj
                    else:
                        cfg.setdefault("model", {})["mmproj_path"] = None
                        matching_mmproj = None
                else:
                    cfg.setdefault("model", {})["mmproj_path"] = None
                    matching_mmproj = None
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
                    mmproj_arg = str(matching_mmproj) if matching_mmproj else ("none" if vision_opt is False else None)
                    if resolved_p:
                        new_engine = GemmaDecisionEngine.from_config(
                            model_path=str(resolved_p),
                            mmproj_path=mmproj_arg,
                        )
                    else:
                        new_engine = GemmaDecisionEngine.from_config(
                            mmproj_path=mmproj_arg,
                        )
                    app.state.engine = new_engine
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Failed to reload model: {exc}")
        else:
            new_engine = _engine()

        if new_engine is not None:
            if req.cot_threshold is not None or req.temperature is not None:
                cal = new_engine.calibrator
                if cal is not None:
                    if req.cot_threshold is not None:
                        cal.cot_threshold = float(req.cot_threshold)
                    if req.temperature is not None:
                        cal.temperature = float(req.temperature)
                    cal_p = resolve_calibration_path(new_engine.model_path)
                    try:
                        cal.save(cal_p)
                    except Exception:
                        pass
            try:
                rel_model = Path(new_engine.model_path).relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel_model = new_engine.model_path
            n_ctx_val = new_engine.n_ctx
            has_vis = new_engine.has_vision
            cal = new_engine.calibrator
        else:
            rel_model = None
            n_ctx_val = cfg.get("n_ctx", 8192)
            has_vis = False
            cal = None

        return {
            "status": "ok",
            "active_model": rel_model,
            "n_ctx": n_ctx_val,
            "has_vision": has_vis,
            "calibrated": cal is not None,
            "temperature": cal.temperature if cal else None,
            "temperatures": cal.temperatures if cal else {},
            "cot_threshold": cal.cot_threshold if cal else None,
            "profile": None,
        }

    @app.post("/v1/models/unload")
    @app.post("/v1/llm/unload")
    def unload_llm_model() -> dict[str, Any]:
        """Explicitly unload the backbone LLM from VRAM and free memory."""
        with _reload_lock:
            old_engine = getattr(app.state, "engine", None)
            if old_engine is not None and hasattr(old_engine, "close"):
                try:
                    old_engine.close()
                except Exception:
                    pass
            app.state.engine = None
            import gc
            gc.collect()
            print("[LLM] Backbone LLM successfully unloaded from memory.")
        return {"status": "ok", "loaded": False, "active_model": None}

    # ── Calibration endpoint ───────────────────────────────────


    class CalibrateModelRequest(BaseModel):
        model_path: str | None = None
        data_path: str | None = None

    @app.post("/v1/models/calibrate")
    def trigger_model_calibration(req: CalibrateModelRequest | None = None) -> dict[str, Any]:
        """Explicitly run Decision Index calibration on a model and save model-calibration.json."""
        eng = _engine()
        target_model = req.model_path if req else None
        if not eng and not target_model:
            raise HTTPException(status_code=400, detail="No model loaded and no model_path specified")

        if target_model:
            resolved_p = resolve_model_path(target_model)
            if not eng or Path(eng.model_path).resolve() != resolved_p.resolve():
                temp_eng = GemmaDecisionEngine.from_config(model_path=str(resolved_p), load_calibration=False)
                active_eng = temp_eng
            else:
                active_eng = eng
        else:
            active_eng = eng

        out_path = resolve_calibration_path(active_eng.model_path)
        data_file = req.data_path if req else None
        cal, saved_file = calibrate_model(active_eng, calibration_data=data_file, output_path=out_path, verbose=True)
        active_eng.calibrator = cal
        update_config_calibration(cal)

        return {
            "status": "ok",
            "model": active_eng.model_path,
            "calibrated_at": saved_file.as_posix(),
            "temperature": cal.temperature,
            "temperatures": cal.temperatures,
            "cot_threshold": cal.cot_threshold,
            "metadata": cal.metadata,
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

    # ── Tier -1 Heuristic Rules Endpoints ─────────────────────

    @app.get("/v1/rules")
    def list_rules() -> dict[str, Any]:
        """List available rule JSON files in RoutingRules and return compiled active rules."""
        from .rules import get_rule_gateway
        gw = get_rule_gateway()
        files = gw.list_rule_files()
        return {
            "status": "ok",
            "active_file": gw.active_file,
            "version": gw.version,
            "default_action": gw.default_action,
            "loaded_rules_count": len(gw.rules),
            "rules": [r.to_dict() for r in gw.rules],
            "files": files,
        }

    @app.post("/v1/rules/select")
    def select_rule_file(req: SelectRuleFileRequest) -> dict[str, Any]:
        """Select and hot-load a different rule JSON file from RoutingRules/."""
        from .rules import get_rule_gateway
        gw = get_rule_gateway()
        count = gw.load_rules(req.filename)
        return {
            "status": "ok",
            "active_file": gw.active_file,
            "loaded_rules_count": count,
            "rules": [r.to_dict() for r in gw.rules],
        }

    @app.post("/v1/config/rules/reload")
    @app.post("/v1/rules/reload")
    def reload_rules(req: ReloadRulesRequest | None = None) -> dict[str, Any]:
        """Hot-reload rules from disk without restarting server or models."""
        from .rules import get_rule_gateway
        gw = get_rule_gateway()
        fname = req.filename if req else None
        count = gw.load_rules(fname)
        return {
            "status": "ok",
            "active_file": gw.active_file,
            "loaded_rules": count,
            "rules": [r.to_dict() for r in gw.rules],
        }

    @app.get("/v1/rules/current")
    def get_current_rules() -> dict[str, Any]:
        from .rules import get_rule_gateway
        gw = get_rule_gateway()
        content = {}
        if gw.config_path.exists():
            try:
                content = json.loads(gw.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "filename": gw.active_file,
            "data": content,
            "compiled_count": len(gw.rules),
        }

    @app.post("/v1/rules/save")
    def save_rules(req: SaveRulesRequest) -> dict[str, Any]:
        from .rules import get_rule_gateway
        gw = get_rule_gateway()
        fname = req.filename or gw.active_file
        target_path = gw.rules_dir / fname
        target_path.write_text(json.dumps(req.data, indent=2), encoding="utf-8")
        count = gw.load_rules(fname)
        return {"status": "ok", "saved_file": fname, "loaded_rules": count}

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

        t0_start_all = time.perf_counter()

        # ── 1. Tier -1: Heuristic Rule Gateway (No model required, < 0.05ms) ───
        rule_match = None
        rule_action = None
        if req.enable_rules:
            from .rules import get_rule_gateway
            rule_gw = get_rule_gateway()
            rule_match = rule_gw.evaluate_route(req.context, options_dict)
            rule_action = rule_match.action if rule_match else None

            if rule_action == "DIRECT_EXIT" and rule_match and rule_match.decision:
                win_letter = rule_match.decision
                win_text = rule_match.decision_text or options_dict.get(win_letter, win_letter)
                duration_ms = (time.perf_counter() - t0_start_all) * 1000.0
                return DecisionResponse(
                    test_id=req.test_id,
                    decision=win_letter,
                    decision_text=win_text,
                    confidence=1.0,
                    latency_ms=round(duration_ms, 2),
                    duration_ms=round(duration_ms, 2),
                    candidate_letter=win_letter,
                    probabilities={win_text: 1.0},
                    letter_probabilities={win_letter: 1.0},
                    mode="tier-1_rule_exit",
                    tier=-1,
                    escalate=False,
                    cot_scratchpad=f"Tier -1 Heuristic Exit: [{rule_match.rule_name}] -> DIRECT_EXIT ({rule_match.description})",
                )

        # ── 2. Handle Case Where LLM Backbone is NOT Loaded ───────────────────
        if eng is None:
            from .tier0 import get_tier0_engine, is_tier0_loaded
            t0_eng = get_tier0_engine(auto_load=False) if is_tier0_loaded() else None
            t0_supports_vision = getattr(t0_eng, "supports_vision", False) if t0_eng else False
            should_run_tier0 = (
                req.early_exit
                and (not has_image or t0_supports_vision)
                and (req.context.strip() or has_image)
                and (rule_action != "SKIP_TIER0")
            )
            if should_run_tier0 and t0_eng is not None:
                try:
                    t0_s = time.perf_counter()
                    from .tier0 import (
                        DEFAULT_TIER0_ENTROPY_THRESHOLD,
                        DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD,
                        DEFAULT_TIER0_MARGIN_THRESHOLD,
                        DEFAULT_TIER0_OOD_THRESHOLD,
                    )
                    t_ent = req.tier0_entropy_threshold if req.tier0_entropy_threshold is not None else DEFAULT_TIER0_ENTROPY_THRESHOLD
                    t_mar = req.tier0_margin_threshold if req.tier0_margin_threshold is not None else DEFAULT_TIER0_MARGIN_THRESHOLD
                    t_ood = DEFAULT_TIER0_OOD_THRESHOLD
                    t_dist = (
                        req.tier0_landmark_distance_threshold
                        if req.tier0_landmark_distance_threshold is not None
                        else DEFAULT_TIER0_LANDMARK_DISTANCE_THRESHOLD
                    )

                    image_input = req.images or req.image or img_path or img_bytes
                    images_for_t0 = None
                    if has_image and t0_supports_vision:
                        if isinstance(image_input, (list, tuple)):
                            images_for_t0 = list(image_input)
                        else:
                            images_for_t0 = [image_input]

                    t0_res = t0_eng.evaluate(
                        query=req.context.strip(),
                        candidates=options_dict,
                        images=images_for_t0,
                        tau_entropy=t_ent,
                        tau_margin=t_mar,
                        tau_ood=t_ood,
                        delta_ood=t_dist,
                    )
                    t0_duration = (time.perf_counter() - t0_s) * 1000.0

                    if t0_res.get("action") == "EXIT_0":
                        win_letter = t0_res["decision"]
                        win_text = t0_res.get("decision_text") or options_dict.get(win_letter, "")
                        conf = t0_res["confidence"]
                        h_val = t0_res.get("entropy", 0.0)
                        m_val = t0_res.get("margin", 0.0)
                        return DecisionResponse(
                            test_id=req.test_id,
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
                            cot_scratchpad=f"Tier 0 Early Exit (Markov Absorbing Chain): H={h_val:.3f} < {t_ent}, Margin Δ={m_val:.3f} > {t_mar}. Resolved without LLM autoregression.",
                            tier0_metrics=t0_res,
                            image_attached=has_image,
                        )
                    else:
                        esc_reason = t0_res.get("reason", "AMBIGUITY")
                        print(f"[Tier 0 Escalation] Request escalated from Tier 0 ({esc_reason}), but LLM backbone is not loaded; skipping model query.")
                        win_letter = t0_res.get("decision", "")
                        win_text = t0_res.get("decision_text") or options_dict.get(win_letter, "")
                        total_ms = (time.perf_counter() - t0_start_all) * 1000.0
                        return DecisionResponse(
                            test_id=req.test_id,
                            decision=win_letter,
                            decision_text=win_text,
                            confidence=t0_res.get("confidence", 0.0),
                            latency_ms=round(t0_duration, 2),
                            duration_ms=round(total_ms, 2),
                            candidate_letter=win_letter,
                            probabilities=t0_res.get("probabilities"),
                            letter_probabilities=t0_res.get("letter_probabilities"),
                            mode="tier0_escalated_no_model",
                            tier=0,
                            escalate=True,
                            tier0_escalation_reason=esc_reason,
                            tier0_metrics=t0_res,
                            cot_scratchpad=f"[ESCALATE] Tier 0 escalated ({esc_reason}). LLM model is not loaded; model query was not executed.",
                            image_attached=has_image,
                        )
                except Exception as exc:
                    print(f"[Tier 0] Evaluation error: {exc}")

            raise HTTPException(
                status_code=400,
                detail="No model loaded (Tier 0 embedding model skipped or unavailable). Please load a model first.",
            )


        # ── 4. Backbone LLM Query (Tier 1 Fast-Path & Tier 2 CoT) ──────────────
        try:
            image_input = req.images or req.image or img_path or img_bytes
            cot_thresh = req.confidence_threshold
            if cot_thresh is None:
                decision_cfg = (_load_llm_config() or {}).get("decision", {})
                cot_thresh = decision_cfg.get("cot_threshold", DEFAULT_COT_THRESHOLD)

            if rule_action == "FORCE_COT":
                cot_thresh = 1.0

            decide_kwargs: dict[str, Any] = {
                "context": req.context,
                "options": options_dict,
                "image": image_input,
                "cot_threshold": cot_thresh,
                "cyclic_debias": req.cyclic_debias,
                "enable_rules": req.enable_rules,
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
            if req.early_exit:
                decide_kwargs["early_exit"] = True
            if req.force_tier0:
                decide_kwargs["force_tier0"] = True
            if req.tier0_entropy_threshold is not None:
                decide_kwargs["tier0_entropy_threshold"] = req.tier0_entropy_threshold
            if req.tier0_margin_threshold is not None:
                decide_kwargs["tier0_margin_threshold"] = req.tier0_margin_threshold
            if req.tier0_landmark_distance_threshold is not None:
                decide_kwargs["tier0_landmark_distance_threshold"] = req.tier0_landmark_distance_threshold

            result = eng.decide(**decide_kwargs)
            duration_ms = (time.perf_counter() - t0_start_all) * 1000.0

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
                tier=result.get("tier", 1),
                escalate=(result["confidence"] < req.escalate_threshold) if req.escalate_threshold > 0 else False,
                evidence=result.get("evidence"),
                cot_scratchpad=result.get("cot_scratchpad"),
                image_attached=bool(image_input),
                tier0_escalation_reason=result.get("tier0_escalation_reason"),
                tier0_metrics=result.get("tier0_metrics"),
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ── Tier 0 Router & Classification Endpoint ───────────────

    @app.post("/v1/classify", response_model=ClassifyResponse)
    @app.post("/api/classify", response_model=ClassifyResponse)
    def classify_query(req: ClassifyRequest) -> ClassifyResponse:
        """Classify a query against candidate categories via Tier 0 Early Exit or Tier 1 escalation."""
        from .tier0 import get_tier0_engine, is_tier0_loaded
        eng = _engine()
        t0 = time.perf_counter()

        if req.early_exit and is_tier0_loaded():
            try:
                t0_eng = get_tier0_engine(auto_load=False)
                if t0_eng:
                    t0_res = t0_eng.evaluate(
                        query=req.text,
                        candidates=req.candidates,
                        tau_entropy=req.tau_entropy,
                        tau_margin=req.tau_margin,
                        tau_ood=req.tau_ood,
                        delta_ood=req.delta_ood,
                        force_tier0=req.force_tier0,
                        enforce_manifold_health=(eng is not None),
                    )
                    if t0_res.get("action") == "EXIT_0":
                        return ClassifyResponse(
                            tier=0,
                            decision=t0_res.get("decision_text") or t0_res.get("decision"),
                            confidence=t0_res["confidence"],
                            latency_ms=t0_res["latency_ms"],
                            mode="tier0_early_exit",
                            escalated_from_tier0=False,
                            tier0_reason=None,
                            probabilities=t0_res.get("probabilities"),
                            entropy=t0_res.get("entropy"),
                            margin=t0_res.get("margin"),
                        )
                    if eng is None:
                        esc_reason = t0_res.get("reason", "AMBIGUITY")
                        print(f"[Tier 0 Escalation] Classify request escalated ({esc_reason}), but LLM backbone is not loaded; skipping model query.")
                        tot_ms = (time.perf_counter() - t0) * 1000.0
                        return ClassifyResponse(
                            tier=0,
                            decision=t0_res.get("decision_text") or t0_res.get("decision") or "",
                            confidence=t0_res.get("confidence", 0.0),
                            latency_ms=round(tot_ms, 2),
                            mode="tier0_escalated_no_model",
                            escalated_from_tier0=True,
                            tier0_reason=esc_reason,
                            probabilities=t0_res.get("probabilities"),
                            entropy=t0_res.get("entropy"),
                            margin=t0_res.get("margin"),
                        )
            except Exception as exc:
                print(f"[Tier 0] Evaluation error: {exc}")

        eng_req = _require_engine()
        try:
            classify_kwargs: dict[str, Any] = {
                "text": req.text,
                "candidates": req.candidates,
                "early_exit": req.early_exit,
                "tau_entropy": req.tau_entropy,
                "tau_margin": req.tau_margin,
                "tau_ood": req.tau_ood,
            }
            if req.force_tier0:
                classify_kwargs["force_tier0"] = True
            res = eng_req.classify(**classify_kwargs)
            return ClassifyResponse(**res)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/v1/tier0/models")
    def list_tier0_models() -> dict[str, Any]:
        """List available embedding models discovered across models/embeddings and models directories."""
        models = scan_embedding_models()
        return {
            "models": models,
            "count": len(models),
        }

    @app.get("/v1/tier0/status")
    def tier0_status() -> dict[str, Any]:
        """Return Tier 0 Early Exit engine status, dimension, and topological landmark counts."""
        from .tier0 import DEFAULT_MIN_MANIFOLD_LANDMARKS, get_tier0_engine, is_tier0_loaded
        eng = _engine()
        t0 = None
        if eng is not None and hasattr(eng, "get_tier0"):
            try:
                t0 = eng.get_tier0(auto_load=False)
            except Exception:
                t0 = None

        if t0 is None and is_tier0_loaded():
            try:
                t0 = get_tier0_engine(auto_load=False)
            except Exception:
                t0 = None

        if t0 is None:
            return {
                "status": "not_loaded",
                "available": False,
                "loaded": False,
                "dimension": 768,
                "landmarks_count": 0,
                "manifold_healthy": False,
                "min_required_landmarks": DEFAULT_MIN_MANIFOLD_LANDMARKS,
            }
        try:
            m_count = int(getattr(t0, "M", 0))
            return {
                "status": "ready",
                "available": True,
                "loaded": True,
                "device": getattr(t0, "device", "cpu"),
                "model_path": getattr(t0, "model_path", None),
                "model_name": getattr(t0, "model_name", None),
                "dimension": getattr(t0, "dim", 768),
                "backend": getattr(t0, "backend", "transformers"),
                "enable_vision": getattr(t0, "enable_vision", True),
                "enable_audio": getattr(t0, "enable_audio", True),
                "supports_vision": getattr(t0, "supports_vision", False),
                "supports_audio": getattr(t0, "supports_audio", False),
                "modalities": getattr(t0, "modalities", ["text"]),
                "landmarks_count": m_count,
                "manifold_healthy": m_count >= DEFAULT_MIN_MANIFOLD_LANDMARKS,
                "min_required_landmarks": DEFAULT_MIN_MANIFOLD_LANDMARKS,
                "classes": getattr(t0, "classes", []),
                "tau_entropy_default": 0.65,
                "tau_margin_default": 0.25,
            }
        except Exception as exc:
            return {"status": "error", "available": False, "loaded": False, "error": str(exc)}

    @app.post("/v1/tier0/load")
    def load_tier0(req: LoadTier0Request | None = None) -> dict[str, Any]:
        """Explicitly load or switch the Tier 0 Embedding model."""
        from .tier0 import DEFAULT_MIN_MANIFOLD_LANDMARKS, load_tier0_engine
        try:
            m_path = req.model_path if req else None
            dev = req.device if req else None
            v = req.vision if req and req.vision is not None else True
            a = req.audio if req and req.audio is not None else True
            t0 = load_tier0_engine(model_path=m_path, device=dev, enable_vision=v, enable_audio=a)
            eng = _engine()
            if eng is not None:
                eng.tier0 = t0
            m_count = int(getattr(t0, "M", 0))
            return {
                "status": "ok",
                "loaded": True,
                "device": t0.device,
                "model_path": getattr(t0, "model_path", None),
                "model_name": getattr(t0, "model_name", None),
                "dimension": getattr(t0, "dim", 768),
                "backend": getattr(t0, "backend", "transformers"),
                "enable_vision": getattr(t0, "enable_vision", True),
                "enable_audio": getattr(t0, "enable_audio", True),
                "supports_vision": getattr(t0, "supports_vision", False),
                "supports_audio": getattr(t0, "supports_audio", False),
                "modalities": getattr(t0, "modalities", ["text"]),
                "landmarks_count": m_count,
                "manifold_healthy": m_count >= DEFAULT_MIN_MANIFOLD_LANDMARKS,
                "min_required_landmarks": DEFAULT_MIN_MANIFOLD_LANDMARKS,
                "classes": t0.classes,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load Tier 0 Embedding model: {exc}")

    @app.post("/v1/tier0/unload")
    def unload_tier0() -> dict[str, Any]:
        """Unload the Tier 0 Embedding model from memory."""
        from .tier0 import unload_tier0_engine
        try:
            unload_tier0_engine()
            eng = _engine()
            if eng is not None:
                eng.tier0 = None
            return {"status": "ok", "loaded": False}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to unload Tier 0: {exc}")


    # ── JevBench / TypeSafe wire format endpoints ─────────────

    @app.post("/decide")
    @app.post("/v1/systemone")
    def handle_jev_wire(body: dict[str, Any]) -> Any:
        eng = _require_engine()
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

        m_name = getattr(eng, "model_name", None) or (Path(eng.model_path).name if getattr(eng, "model_path", None) else "model")
        if is_dict_questions:
            return {
                "model": m_name,
                "answers": answers_dict,
                "usage": {"input_tokens": 0, "output_tokens": 1},
            }
        else:
            return {
                "model": m_name,
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
        eng = _require_engine()
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
        Uses the shared LLM in VRAM for autoregressive text/JSON generation.
        """
        eng = _require_engine()
        msg_dicts = [m.model_dump() for m in req.messages]

        if req.stream:
            def event_generator():
                gen = eng.stream_chat_completion(
                    messages=msg_dicts,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    response_format=req.response_format,
                    stop=req.stop,
                )
                try:
                    for chunk in gen:
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    err_chunk = {"error": {"message": str(exc), "type": "server_error"}}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    try:
                        gen.close()
                    except Exception:
                        pass

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
        eng = _require_engine()
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
                gen = eng.stream_chat_completion(
                    messages=messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    response_format=response_format,
                )
                try:
                    for chunk in gen:
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield f"data: {json.dumps({'text': delta})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    err_chunk = {"error": {"message": str(exc), "type": "server_error"}}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    try:
                        gen.close()
                    except Exception:
                        pass

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
        """Return runtime decision settings and defaults from active calibrator and config."""
        eng = _engine()
        cal = eng.calibrator if eng else None
        cfg = _load_llm_config() or {}
        decision_cfg = cfg.get("decision", {})

        cot_thresh = 0.50
        cal_temp = None
        cal_temps = {}
        cal_meta = None
        cal_path = None

        if cal is not None:
            cal_temp = cal.temperature
            cal_temps = cal.temperatures
            cal_meta = cal.metadata
            if cal.cot_threshold is not None:
                cot_thresh = float(cal.cot_threshold)
            if cal.metadata.get("calibration_path"):
                cal_path = cal.metadata.get("calibration_path")
        elif eng and eng.model_path:
            p = find_model_calibration_path(eng.model_path)
            if p and p.is_file():
                try:
                    cal_data = json.loads(p.read_text(encoding="utf-8"))
                    cal_temp = cal_data.get("temperature")
                    cal_temps = cal_data.get("temperatures") or cal_data.get("cardinality_temperatures") or {}
                    cal_meta = cal_data.get("metadata", {})
                    if cal_data.get("cot_threshold") is not None:
                        cot_thresh = float(cal_data.get("cot_threshold"))
                    cal_path = p.relative_to(PROJECT_ROOT).as_posix()
                except Exception:
                    pass
        else:
            if "cot_threshold" in decision_cfg:
                cot_thresh = float(decision_cfg["cot_threshold"])

        if cot_thresh <= 0.0:
            policy = "fast_path"
        elif cot_thresh >= 1.0:
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
            "balanced_threshold": cot_thresh,
            "cot_threshold": cot_thresh,
            "early_exit": bool(decision_cfg.get("early_exit", True)),
            "tier0_entropy_threshold": float(decision_cfg.get("tier0_entropy_threshold", 0.65)),
            "tier0_margin_threshold": float(decision_cfg.get("tier0_margin_threshold", 0.25)),
            "tier0_landmark_distance_threshold": float(decision_cfg.get("tier0_landmark_distance_threshold", 0.22)),
            "cyclic_debias": bool(decision_cfg.get("cyclic_debias", False)),
            "escalate_enabled": False,
            "escalate_threshold": 0.70,
            "cot_temp": float(decision_cfg.get("cot_temp", DEFAULT_COT_TEMP)),
            "max_cot_tokens": int(decision_cfg.get("cot_max_tokens", DEFAULT_COT_MAX_TOKENS)),
            "cot_prompt": str(decision_cfg.get("cot_prompt", DEFAULT_COT_PROMPT)),
            "kv_cache": kv_cache_str,
            "use_calibrator": True,
            "calibrated": cal is not None or cal_temp is not None,
            "calibrated_temperature": cal_temp,
            "calibrated_temperatures": cal_temps,
            "calibration_metadata": cal_meta,
            "calibration_path": cal_path,
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
            self._app = create_app(None)
        return self._app

    async def __call__(self, scope, receive, send) -> None:
        await self._resolve()(scope, receive, send)


def create_app(engine: GemmaDecisionEngine | None = None) -> FastAPI:
    return build_app(engine)


app = _LazyApp()
