from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "calibration.json"

MODEL_PATH_ENV = "GEVVA0_MODEL"
N_CTX_ENV = "GEVVA0_N_CTX"
CALIBRATION_PATH_ENV = "GEVVA0_CALIBRATION"
MMPROJ_PATH_ENV = "GEVVA0_MMPROJ"

# Legacy hard-coded defaults (used as fallbacks when llm_config.json is absent)
DEFAULT_N_CTX = 8192
DEFAULT_COT_THRESHOLD = 0.85
DEFAULT_COT_MAX_TOKENS = 256
DEFAULT_COT_TEMP = 0.0
DEFAULT_COT_PROMPT = "Analysis: First calculate the exact sliding window cutoff timestamp and evaluate which log entries fall strictly inside it: "
DEFAULT_ESCALATE_THRESHOLD = 0.60
DEFAULT_KV_BRANCHING = False
DEFAULT_KV_BRANCHING_MIN_TOKENS = 1500
MAX_OPTIONS = 26
MIN_OPTIONS = 2

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------------------------------------------------------------------------
# llm_config.json configuration resolution (profiles removed)
# ---------------------------------------------------------------------------

_LLM_CONFIG_PATH = PROJECT_ROOT / "llm_config.json"


def _load_llm_config() -> dict | None:
    """Load and return the llm_config.json dict, or None if the file is missing."""
    if _LLM_CONFIG_PATH.is_file():
        with open(_LLM_CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def get_active_profile_name() -> str | None:
    """Legacy helper: profiles are removed in favor of direct ctx size."""
    return None


def get_active_profile() -> dict:
    """Return top-level configuration settings.
    
    Profiles are no longer used; n_ctx is configured directly.
    """
    cfg = _load_llm_config()
    if cfg is None:
        return {}

    merged: dict = dict(cfg)
    merged["model"] = cfg.get("model", {})
    merged["decision"] = cfg.get("decision", {})
    return merged



# ---------------------------------------------------------------------------
# Model discovery and disk size helpers
# ---------------------------------------------------------------------------

import re


def format_size_disk(size_bytes: int) -> str:
    """Format on-disk byte size to human readable string (e.g. 14gb, 8.7gb, 2.6gb)."""
    gb = size_bytes / (1000 ** 3)
    if gb >= 10:
        return f"{round(gb)}gb"
    elif gb >= 1:
        val = round(gb, 1)
        return f"{int(val) if val.is_integer() else val}gb"
    else:
        return f"{round(gb * 1024)}mb"


def format_friendly_name(filename: str) -> str:
    """Extract a clean, friendly name from a GGUF filename (e.g. Gemma 4 26B)."""
    name = Path(filename).stem
    m = re.match(r"(?i)gemma[-_ ]*4[-_ ]*([a-z0-9]+)", name)
    if m:
        param = m.group(1).upper()
        return f"Gemma 4 {param}"
    parts = name.split("-")
    if len(parts) >= 2:
        return " ".join(p.capitalize() for p in parts[:3])
    return name


def is_mmproj_file(path: str | Path) -> bool:
    """Return True if the file name indicates a multimodal projector / CLIP model."""
    name = Path(path).name.lower()
    return "mmproj" in name or "clip" in name


def detect_mmproj_variant(path: str | Path) -> str | None:
    """Auto-detect the multimodal projector variant (e.g. BF16, F16, F32, Q4_0) from filename."""
    name = Path(path).stem
    m = re.search(r"(?i)(?:mmproj|clip)(?:[-_]model)?[-_]([a-zA-Z0-9_]+)", name)
    if m:
        return m.group(1).upper()
    for var in ["BF16", "F16", "F32", "Q8_0", "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q5_0", "Q5_1"]:
        if var.lower() in name.lower():
            return var
    return None


def get_file_creation_timestamp(p: Path) -> float:
    """Return creation / birth time of a file, falling back to mtime or ctime."""
    try:
        st = p.stat()
        birthtime = getattr(st, "st_birthtime", None)
        ctime = getattr(st, "st_ctime", 0.0)
        mtime = getattr(st, "st_mtime", 0.0)
        created = birthtime if birthtime is not None else ctime
        return max(created, mtime)
    except OSError:
        return 0.0


def scan_models() -> list[dict]:
    """Scan the models directory for LLM .gguf files (excluding mmproj), sorted by size ascending.
    
    Discovers models located in subfolders (e.g. models/<name>/<name>.gguf) or in the root.
    Associates each model with any matching mmproj in the same subfolder (newest created if multiple).
    """
    candidates = []
    models_dirs = [PROJECT_ROOT / "models", PROJECT_ROOT / "model"]
    seen_paths = set()

    for m_dir in models_dirs:
        if not m_dir.is_dir():
            continue
        for p in m_dir.rglob("*.gguf"):
            if is_mmproj_file(p):
                continue
            resolved = p.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                size_bytes = p.stat().st_size
            except OSError:
                continue
            try:
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_path = str(p)

            # Discover matching mmproj in the same subfolder (newest created file if multiple)
            matching_mmproj: str | None = None
            matching_variant: str | None = None
            mmproj_siblings = [s for s in p.parent.glob("*.gguf") if is_mmproj_file(s)]
            if mmproj_siblings:
                mmproj_siblings.sort(key=get_file_creation_timestamp, reverse=True)
                newest = mmproj_siblings[0]
                try:
                    matching_mmproj = newest.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    matching_mmproj = str(newest)
                matching_variant = detect_mmproj_variant(newest)

            size_str = format_size_disk(size_bytes)
            friendly = format_friendly_name(p.name)
            label = f"{friendly} ({size_str})"

            candidates.append({
                "path": rel_path,
                "full_path": str(resolved),
                "filename": p.name,
                "folder": p.parent.name,
                "friendly_name": friendly,
                "label": label,
                "size_bytes": size_bytes,
                "size_gb": size_str,
                "mmproj_path": matching_mmproj,
                "mmproj_variant": matching_variant,
            })

    candidates.sort(key=lambda m: m["size_bytes"])
    return candidates


def scan_mmproj_models() -> list[dict]:
    """Scan the models directory for multimodal projector (.gguf) files, sorted newest created first."""
    candidates = []
    models_dirs = [PROJECT_ROOT / "models", PROJECT_ROOT / "model"]
    seen_paths = set()

    for m_dir in models_dirs:
        if not m_dir.is_dir():
            continue
        for p in m_dir.rglob("*.gguf"):
            if not is_mmproj_file(p):
                continue
            resolved = p.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                size_bytes = p.stat().st_size
            except OSError:
                continue
            try:
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_path = str(p)

            size_str = format_size_disk(size_bytes)
            variant = detect_mmproj_variant(p)
            created_ts = get_file_creation_timestamp(p)
            candidates.append({
                "path": rel_path,
                "full_path": str(resolved),
                "filename": p.name,
                "folder": p.parent.name,
                "variant": variant,
                "created_ts": created_ts,
                "size_bytes": size_bytes,
                "size_gb": size_str,
            })

    # Sort newest created first
    candidates.sort(key=lambda m: (m.get("created_ts", 0), m["size_bytes"]), reverse=True)
    return candidates


def resolve_mmproj_path(model_path: str | Path | None = None) -> Path | None:
    """Resolve multimodal projector path for a model from its subfolder, env var, or llm_config.
    
    If multiple mmproj files exist in the model's subfolder, selects the newest created file.
    Auto-detects variants (e.g. BF16, F16, F32).
    """
    env = os.environ.get(MMPROJ_PATH_ENV)
    if env:
        if env.lower() in ("none", "null", "false", "off", "0", ""):
            return None
        p = Path(env)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p

    if model_path is None:
        profile = get_active_profile()
        model_sec = profile.get("model", {})
        if "mmproj_path" in model_sec and (
            model_sec["mmproj_path"] is None or str(model_sec["mmproj_path"]).lower() in ("none", "null", "false", "")
        ):
            return None
    elif isinstance(model_path, str) and model_path.lower() in ("none", "null", "false", "off", ""):
        return None

    # Inspect the model's subfolder
    target_path = model_path
    if not target_path:
        profile = get_active_profile()
        target_path = profile.get("model", {}).get("model_path")

    if target_path:
        # If target_path is already a direct path to an mmproj file, return it
        p_check = Path(target_path)
        if not p_check.is_absolute():
            cand_check = PROJECT_ROOT / p_check
            if cand_check.is_file():
                p_check = cand_check
            elif (PROJECT_ROOT / "models" / p_check).is_file():
                p_check = PROJECT_ROOT / "models" / p_check
        if p_check.is_file() and is_mmproj_file(p_check):
            return p_check

        search_dir: Path | None = None
        try:
            resolved_m = resolve_model_path(target_path)
            search_dir = resolved_m.parent
        except Exception:
            p = Path(target_path)
            if not p.is_absolute():
                cand_p = PROJECT_ROOT / p
                if cand_p.exists():
                    p = cand_p
                elif (PROJECT_ROOT / "models" / p).exists():
                    p = PROJECT_ROOT / "models" / p
            search_dir = p if p.is_dir() else p.parent

        if search_dir and search_dir.is_dir():
            siblings = [s for s in search_dir.glob("*.gguf") if is_mmproj_file(s)]
            if siblings:
                siblings.sort(key=get_file_creation_timestamp, reverse=True)
                return siblings[0]

        cand_str = str(target_path).replace("\\", "/").rstrip("/")
        cand_name = Path(target_path).name
        for m in scan_models():
            if m["path"] == cand_str or m["filename"] == cand_name or m.get("folder") == cand_name:
                if m.get("mmproj_path"):
                    cand_mmproj = PROJECT_ROOT / m["mmproj_path"]
                    if cand_mmproj.is_file():
                        return cand_mmproj

    # Check llm_config.json
    profile = get_active_profile()
    model_section = profile.get("model", {})
    if model_section.get("mmproj_path"):
        cand = PROJECT_ROOT / model_section["mmproj_path"]
        if cand.is_file():
            return cand

    return None


def get_smallest_model() -> dict | None:
    """Return the smallest .gguf model on disk, or None if none found."""
    models = scan_models()
    return models[0] if models else None


# ---------------------------------------------------------------------------
# Public resolvers (backward-compatible)
# ---------------------------------------------------------------------------

def resolve_model_path(model_input: str | Path | None = None) -> Path:
    """Resolve an LLM .gguf model path from a filepath, subfolder, filename, or config."""
    cand_input = model_input or os.environ.get(MODEL_PATH_ENV)
    if not cand_input:
        profile = get_active_profile()
        cand_input = profile.get("model", {}).get("model_path")

    if cand_input:
        p = Path(cand_input)
        if not p.is_absolute():
            cand_p = PROJECT_ROOT / p
            if cand_p.exists():
                p = cand_p

        # Direct file check
        if p.is_file() and not is_mmproj_file(p):
            return p

        # Direct folder check: pick the primary non-mmproj .gguf in this subfolder
        if p.is_dir():
            for sub in p.glob("*.gguf"):
                if not is_mmproj_file(sub):
                    return sub

        # Flexible matching against scanned models
        cand_str = str(cand_input).replace("\\", "/").rstrip("/")
        cand_name = Path(cand_input).name
        cand_stem = Path(cand_input).stem
        for m in scan_models():
            if (
                m["path"] == cand_str
                or m["filename"] == cand_name
                or m.get("folder") == cand_name
                or m.get("folder") == cand_stem
                or Path(m["filename"]).stem == cand_stem
            ):
                return Path(m["full_path"])

        # Flexible shorthand / substring matching (e.g. '26b', 'e4b', 'e2b')
        cand_lower = cand_str.lower()
        for m in scan_models():
            if cand_lower in m["filename"].lower() or cand_lower in m.get("folder", "").lower():
                return Path(m["full_path"])

    # Fallback to smallest available model on disk, else DEFAULT_MODEL_PATH
    smallest = get_smallest_model()
    if smallest:
        return Path(smallest["full_path"])

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    models = scan_models()
    if models:
        return Path(models[0]["full_path"])

    return DEFAULT_MODEL_PATH


def resolve_n_ctx() -> int:
    env = os.environ.get(N_CTX_ENV)
    if env:
        return int(env)
    profile = get_active_profile()
    if "n_ctx" in profile:
        return int(profile["n_ctx"])
    return DEFAULT_N_CTX


def resolve_calibration_path() -> Path:
    env = os.environ.get(CALIBRATION_PATH_ENV)
    return Path(env) if env else DEFAULT_CALIBRATION_PATH


def resolve_cot_prompt() -> str:
    """Return the CoT reasoning prefix prompt from active profile or default."""
    profile = get_active_profile()
    decision = profile.get("decision", {})
    return str(decision.get("cot_prompt", DEFAULT_COT_PROMPT))


def resolve_llm_kwargs() -> dict:
    """Return the full set of Llama() constructor kwargs resolved from the active profile.

    Keys returned: model_path, mmproj_path, n_ctx, n_gpu_layers, flash_attn, verbose,
    n_batch, n_ubatch, type_k, type_v.
    Defaults to native FP16 KV cache (type_k=1, type_v=1).
    """
    profile = get_active_profile()
    model_section = profile.get("model", {})
    resolved_model = resolve_model_path()
    if "mmproj_path" in model_section:
        cfg_mmproj = model_section["mmproj_path"]
        if cfg_mmproj is None or str(cfg_mmproj).lower() in ("none", "null", "false", ""):
            mmproj = None
        else:
            cand = PROJECT_ROOT / cfg_mmproj if not Path(cfg_mmproj).is_absolute() else Path(cfg_mmproj)
            mmproj = cand if cand.is_file() else resolve_mmproj_path(resolved_model)
    else:
        mmproj = resolve_mmproj_path(resolved_model)

    # Resolve KV cache precision: default to FP16 (GGML_TYPE_F16 = 1)
    kv_cache_str = profile.get("kv_cache")
    type_k = profile.get("type_k")
    type_v = profile.get("type_v")
    if kv_cache_str is not None:
        kv_str = str(kv_cache_str).strip().upper()
        if kv_str in ("F16", "FP16"):
            type_k = type_v = 1
        elif kv_str == "Q8_0":
            type_k = type_v = 8
        elif kv_str == "Q4_0":
            type_k = type_v = 2
        elif kv_str in ("F32", "FP32"):
            type_k = type_v = 0
    if type_k is None:
        type_k = 1  # GGML_TYPE_F16 = 1
    if type_v is None:
        type_v = 1  # GGML_TYPE_F16 = 1

    return {
        "model_path": str(resolved_model),
        "mmproj_path": str(mmproj) if mmproj else None,
        "n_ctx": resolve_n_ctx(),
        "n_gpu_layers": model_section.get("n_gpu_layers", -1),
        "flash_attn": profile.get("flash_attn", True),
        "verbose": model_section.get("verbose", False),
        "n_batch": profile.get("n_batch"),
        "n_ubatch": profile.get("n_ubatch"),
        "type_k": type_k,
        "type_v": type_v,
    }


def resolve_decision_kwargs() -> dict:
    """Return decision-tuning parameters from the active profile."""
    profile = get_active_profile()
    decision = profile.get("decision", {})

    return {
        "cot_threshold": decision.get("cot_threshold", DEFAULT_COT_THRESHOLD),
        "cot_max_tokens": decision.get("cot_max_tokens", DEFAULT_COT_MAX_TOKENS),
        "cot_temp": decision.get("cot_temp", DEFAULT_COT_TEMP),
        "cot_prompt": decision.get("cot_prompt", DEFAULT_COT_PROMPT),
        "cyclic_debias": decision.get("cyclic_debias", False),
        "kv_branching": decision.get("kv_branching", DEFAULT_KV_BRANCHING),
        "kv_branching_min_tokens": decision.get("kv_branching_min_tokens", DEFAULT_KV_BRANCHING_MIN_TOKENS),
    }
