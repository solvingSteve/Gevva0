from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .calibration import TemperatureCalibrator
from .config import (
    DEFAULT_COT_THRESHOLD,
    DEFAULT_ESCALATE_THRESHOLD,
    PROJECT_ROOT,
    resolve_calibration_path,
)
from .debias import LETTERS
from .engine import GemmaDecisionEngine


def parse_options(spec: str) -> dict[str, str]:
    import ast
    import json

    spec_clean = spec.strip()
    if (spec_clean.startswith("[") and spec_clean.endswith("]")) or (spec_clean.startswith("{") and spec_clean.endswith("}")):
        parsed = None
        try:
            parsed = json.loads(spec_clean)
        except Exception:
            try:
                parsed = ast.literal_eval(spec_clean)
            except Exception:
                pass
        if isinstance(parsed, list):
            return {LETTERS[i]: str(v).strip() for i, v in enumerate(parsed) if i < len(LETTERS)}
        elif isinstance(parsed, dict):
            return {str(k).strip().upper(): str(v).strip() for k, v in parsed.items()}

    options: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"invalid option {part!r}; expected 'LETTER: text'")
        key, value = part.split(":", 1)
        key = key.strip().upper()
        if len(key) != 1 or not key.isalpha():
            raise SystemExit(f"invalid option letter {key!r}")
        options[key] = value.strip()
    if len(options) < 2:
        raise SystemExit("at least two options are required, e.g. 'A: yes,B: no'")
    return dict(sorted(options.items()))


def load_calibrator() -> TemperatureCalibrator | None:
    path = resolve_calibration_path()
    if path.exists():
        cal = TemperatureCalibrator.load(path)
        print(f"Loaded calibrator (T={cal.temperature:.4f}) from {path}", file=sys.stderr)
        return cal
    return None


def cmd_check(args: argparse.Namespace) -> int:
    engine = GemmaDecisionEngine(model_path=args.model, verbose=args.verbose)
    llm = engine.llm
    print(f"model: {engine.model_path}")
    print(f"n_ctx: {engine.n_ctx} | n_vocab: {llm._n_vocab}")
    variant_tag = f" [{engine.mmproj_variant}]" if getattr(engine, "mmproj_variant", None) else ""
    print(f"vision: {'enabled (mmproj=' + str(engine.mmproj_path) + variant_tag + ')' if engine.has_vision else 'disabled'}")
    print(f"token_eos: {int(llm.token_eos())}")
    for letter in "ABCDEF":
        tid_sp = int(llm.tokenize(f" {letter}".encode("utf-8"), add_bos=False)[0])
        tid_no = int(llm.tokenize(letter.encode("utf-8"), add_bos=False)[0])
        text_sp = llm.detokenize([tid_sp]).decode("utf-8", errors="ignore")
        text_no = llm.detokenize([tid_no]).decode("utf-8", errors="ignore")
        print(f"candidate {letter!r}: space_id={tid_sp} ({text_sp!r}) | nospace_id={tid_no} ({text_no!r})")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    if not args.context.strip() and not args.image:
        raise SystemExit("either --context or --image must be provided")
    mmproj_override = None if getattr(args, "no_vision", False) else getattr(args, "mmproj", None)
    engine = GemmaDecisionEngine.from_config(
        model_path=args.model,
        mmproj_path=mmproj_override,
        load_calibration=True,
    )
    options = parse_options(args.options)
    result = engine.decide(
        context=args.context,
        options=options,
        image=args.image,
        cot_threshold=args.threshold,
        cyclic_debias=args.cyclic,
        kv_branching=args.kv_branching,
        kv_branching_min_tokens=args.kv_branching_min_tokens,
    )
    result["escalate"] = result["confidence"] < args.escalate_threshold
    if args.json:
        if hasattr(result, "model_dump_json"):
            print(result.model_dump_json(indent=2))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"decision:   {result['decision']}")
        print(f"confidence: {result['confidence']:.4f}")
        print(f"mode:       {result['mode']}")
        print(f"escalate:   {result['escalate']}")
        if result.get("image_attached"):
            print(f"image:      attached")
        for text, prob in result["probabilities"].items():
            print(f"  {prob:.4f}  {text}")
        if result["cot_scratchpad"]:
            print(f"scratchpad: {result['cot_scratchpad']}")
    return 0


def _dict_spec(options: dict[str, str]) -> str:
    return ",".join(f"{k}: {v}" for k, v in options.items())


def cmd_calibrate(args: argparse.Namespace) -> int:
    data_path = Path(args.data) if args.data else Path("calibration.jsonl")
    if not data_path.is_file():
        if (PROJECT_ROOT / data_path).is_file():
            data_path = PROJECT_ROOT / data_path
        elif (PROJECT_ROOT / "calibration.jsonl").is_file():
            data_path = PROJECT_ROOT / "calibration.jsonl"
        elif (PROJECT_ROOT / "benchmarks" / "benchmark_suite.json").is_file():
            data_path = PROJECT_ROOT / "benchmarks" / "benchmark_suite.json"
        elif (PROJECT_ROOT / "tests" / "tests.json").is_file():
            data_path = PROJECT_ROOT / "tests" / "tests.json"
        else:
            raise SystemExit(f"no calibration rows found in {args.data}")

    rows: list[dict] = []
    raw_text = data_path.read_text(encoding="utf-8").strip()
    if raw_text.startswith("{") or raw_text.startswith("["):
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "tests" in parsed:
                rows = parsed["tests"]
            elif isinstance(parsed, list):
                rows = parsed
        except Exception:
            pass

    if not rows:
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    if not rows:
        raise SystemExit(f"no calibration rows found in {data_path}")

    engine = GemmaDecisionEngine(model_path=args.model)
    logits_list: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        label = row.get("label") or row.get("expected_ground_truth")
        if not label:
            continue
        ctx = row.get("context", "")
        q = row.get("question", "")
        if q and q not in ctx:
            ctx = f"{ctx}\n\nQuestion: {q}"

        if isinstance(row["options"], dict):
            options = {str(k).strip().upper(): str(v).strip() for k, v in row["options"].items()}
        else:
            options = parse_options(row["options"])

        keys = list(options.keys())
        if label not in keys:
            continue

        logits_list.append(engine.score(ctx, options))
        labels.append(keys.index(label))

    if not logits_list:
        raise SystemExit("no valid labeled rows found for calibration")

    cal = load_calibrator() or TemperatureCalibrator()
    cal.fit_brier_score(logits_list, labels)
    out = resolve_calibration_path() if args.out is None else Path(args.out)
    cal.save(out)
    print(f"saved calibrator to {out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    mmproj_override = None if getattr(args, "no_vision", False) else getattr(args, "mmproj", None)
    engine = GemmaDecisionEngine.from_config(
        model_path=args.model,
        mmproj_path=mmproj_override,
        load_calibration=True,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gevva0", description="Project Gevva0: local calibrated decision engine")
    parser.add_argument("--version", action="version", version=f"gevva0 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Load the model and print tokenizer sanity info")
    p_check.add_argument("--model", default=None)
    p_check.add_argument("--verbose", action="store_true")
    p_check.set_defaults(func=cmd_check)

    p_decide = sub.add_parser("decide", help="Classify a context or image against a set of options")
    p_decide.add_argument("--context", default="", help="Context text or prompt")
    p_decide.add_argument("--image", default=None, help="Path to image file, URL, or data URI")
    p_decide.add_argument("--options", required=True, help="e.g. 'A: Billing Inquiry,B: Technical Bug,C: Churn Risk'")
    p_decide.add_argument("--threshold", type=float, default=DEFAULT_COT_THRESHOLD)
    p_decide.add_argument("--escalate-threshold", type=float, default=DEFAULT_ESCALATE_THRESHOLD)
    p_decide.add_argument("--cyclic", action="store_true", help="run cyclic label-permutation debiasing")
    p_decide.add_argument("--kv-branching", action="store_true", help="enable KV-cache branching for cyclic debiasing")
    p_decide.add_argument("--kv-branching-min-tokens", type=int, default=50, help="min context tokens to activate KV-cache branching")
    p_decide.add_argument("--no-vision", action="store_true", help="force disable vision projector")
    p_decide.add_argument("--mmproj", default=None, help="path to multimodal projector")
    p_decide.add_argument("--json", action="store_true")
    p_decide.add_argument("--model", default=None)
    p_decide.set_defaults(func=cmd_decide)

    p_cal = sub.add_parser("calibrate", help="Fit Platt temperature on a JSONL calibration set")
    p_cal.add_argument("--data", default="calibration.jsonl", help="JSONL or JSON calibration dataset with context, options, and label")
    p_cal.add_argument("--out", default=None)
    p_cal.add_argument("--model", default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--model", default=None)
    p_serve.add_argument("--no-vision", action="store_true", help="force disable vision projector")
    p_serve.add_argument("--mmproj", default=None, help="path to multimodal projector")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
