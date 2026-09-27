from pathlib import Path
import pytest
from gevva0.cli import build_parser


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["calibrate"])
    assert args.command == "calibrate"
    assert args.data == "calibration.jsonl"

    decide_args = parser.parse_args(["decide", "--options", "A: Yes,B: No"])
    assert decide_args.command == "decide"
    assert decide_args.threshold == 0.85
