import tempfile
from pathlib import Path
import numpy as np
import pytest

from gevva0.calibration import TemperatureCalibrator, brier_score, ece, softmax


def test_softmax_properties():
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]

    # 2D batch
    batch_logits = np.array([[2.0, 1.0], [0.0, 3.0]])
    batch_probs = softmax(batch_logits, axis=1)
    assert np.allclose(np.sum(batch_probs, axis=1), [1.0, 1.0])


def test_brier_score_and_ece():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    labels = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    score = brier_score(probs, labels)
    assert 0.0 <= score <= 1.0

    err = ece(probs, labels)
    assert 0.0 <= err <= 1.0


def test_variable_length_brier_score_and_ece():
    # Variable number of options per scenario
    probs = [np.array([0.7, 0.2, 0.1]), np.array([0.9, 0.1])]
    labels = [0, 0]

    score = brier_score(probs, labels)
    assert 0.0 <= score <= 1.0

    err = ece(probs, labels)
    assert 0.0 <= err <= 1.0


def test_calibrator_fit_and_persistence():
    logits = [
        np.array([12.0, 8.0, 2.0, 1.0]),
        np.array([9.0, 15.0, 3.0]),
        np.array([2.0, 10.0]),
    ]
    labels = [0, 1, 1]

    cal = TemperatureCalibrator()
    cal.fit_brier_score(logits, labels, verbose=False)
    assert 0.05 <= cal.temperature <= 5.0

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cal.save(tmp_path)
        loaded = TemperatureCalibrator.load(tmp_path)
        assert np.isclose(cal.temperature, loaded.temperature)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_vector_calibration_and_affine_fit():
    # Test vector bias subtraction: z_cal = (z_raw - b) / T
    cal = TemperatureCalibrator(
        temperature=1.08,
        bias_offsets={"A": 0.42, "B": -0.15, "C": -0.08, "D": -0.11, "E": -0.08},
    )
    raw_logits = np.array([2.0, 1.0, 0.5])
    labels = ["A", "B", "C"]
    cal_logits = cal.transform_logits(raw_logits, labels)

    expected = np.array([(2.0 - 0.42) / 1.08, (1.0 - (-0.15)) / 1.08, (0.5 - (-0.08)) / 1.08])
    assert np.allclose(cal_logits, expected)

    cal_probs = cal.transform(raw_logits, labels)
    assert np.isclose(np.sum(cal_probs), 1.0)

    # Test JSON persistence with bias offsets
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cal.save(tmp_path)
        loaded = TemperatureCalibrator.load(tmp_path)
        assert np.isclose(loaded.temperature, 1.08)
        assert loaded.bias_offsets == cal.bias_offsets
        assert np.allclose(loaded.transform(raw_logits, labels), cal_probs)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Test fit_affine optimization
    synth_logits = [
        np.array([5.0, 3.0, 1.0]),
        np.array([4.8, 4.0, 1.2]),
        np.array([2.0, 5.5, 1.0]),
        np.array([2.2, 5.0, 1.1]),
    ]
    synth_labels = [0, 0, 1, 1]
    fitted = TemperatureCalibrator()
    fitted.fit_affine(synth_logits, synth_labels, candidate_keys=["A", "B", "C"], verbose=False)
    assert 0.05 <= fitted.temperature <= 5.0
    assert "A" in fitted.bias_offsets and "B" in fitted.bias_offsets and "C" in fitted.bias_offsets
    # Sum of zero-mean offsets should be close to 0
    assert np.isclose(sum(fitted.bias_offsets.values()), 0.0, atol=1e-3)
