from pathlib import Path
import numpy as np
from reality_stone.clarus.fault_ood_calibration import _ece, _threshold, run_fault_ood_gate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "fault_ood_calibration_v7.json"


def test_ece_is_zero_for_exact_constant_frequency() -> None:
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    probability = np.full(4, 0.5)
    assert _ece(probability, labels) == 0.0


def test_calibration_threshold_keeps_registered_empirical_risk() -> None:
    probability = np.linspace(0.01, 0.99, 100)
    labels = np.zeros(100)
    labels[-10:] = 1.0
    threshold = _threshold(probability, labels, 0.01)
    assert labels[probability < threshold].mean() <= 0.01


def test_gate_is_offline_and_retains_useful_coverage() -> None:
    report = run_fault_ood_gate(CONFIG)
    assert report["resource_usage"]["external_download_bytes"] == 0
    assert report["summary"]["candidate"]["coverage"] >= 0.25
