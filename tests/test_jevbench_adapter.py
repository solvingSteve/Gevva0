import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "jevbench") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "jevbench"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


from jevbench.adapters.gevva0 import Gevva0Adapter
from convert_jevbench import normalize_jevbench_task
from jevbench.tasks import Task


def test_gevva0_adapter_build_request():
    adapter = Gevva0Adapter(endpoint="http://127.0.0.1:8000")
    task = Task(
        id="test-item-01",
        family="intent",
        state="Where is my shipment?",
        question={"instructions": "Determine intent", "type": "choice", "criteria": {"track": "Track package", "cancel": "Cancel package"}},
        labels=["track", "cancel"],
        expected="track",
        split="public",
    )
    payload, letters = adapter.build_request(task)
    assert letters == ["A", "B"]
    assert "Where is my shipment?" in payload["context"]
    assert "Determine intent" in payload["context"]
    assert "A" in payload["options"]
    assert "B" in payload["options"]
    assert payload["test_id"] == "test-item-01"


def test_convert_jevbench_normalizer():
    raw_task = {
        "id": "easy-intent-00",
        "family": "intent",
        "state": "I need help with my invoice.",
        "question": {
            "instructions": "Classify user intent",
            "type": "choice",
            "criteria": {
                "billing": "Invoice or payment question",
                "support": "General technical help",
            },
        },
        "labels": ["billing", "support"],
        "expected": "billing",
        "split": "public",
        "provenance": {"why_hard": "Distractor keywords"},
    }
    res = normalize_jevbench_task(raw_task, 1)
    assert res is not None
    assert res["test_id"] == "easy-intent-00"
    assert res["expected_ground_truth"] == "A"  # "billing" is index 0 -> "A"
    assert res["options"]["A"] == "billing: Invoice or payment question"
    assert res["options"]["B"] == "support: General technical help"
