from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts" / "agi" / "episodic_ltm_dream_factorial_integrity_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_g7m_v1_failed_validation_is_byte_locked_and_test_is_unopened() -> None:
    ledger_raw = LEDGER.read_bytes()
    ledger = json.loads(ledger_raw)

    assert ledger_raw.endswith(b"\n")
    assert b"\r\n" not in ledger_raw
    assert ledger["status"] == "validation_FAIL_test_unopened"
    assert ledger["validation"] == {
        "split": "validation",
        "seeds": "78100-78139",
        "passed": False,
        "checks_total": 73,
        "checks_passed": 67,
        "checks_failed": 6,
        "failed_checks": [
            "L_main.hidden_ci",
            "L_main.hidden_reduction",
            "M10.convergence",
            "M10.extra_stability",
            "M11.convergence",
            "M11.extra_stability",
        ],
        "resource_passed": True,
    }
    for relative, expected in ledger["files"].items():
        assert _sha256(ROOT / relative) == expected

    validation_path = (
        ROOT / "artifacts" / "agi" / "episodic_ltm_dream_factorial_validation_v1.json"
    )
    calibration_path = (
        ROOT
        / "artifacts"
        / "agi"
        / "episodic_ltm_dream_factorial_train_calibration_v1.json"
    )
    validation = json.loads(validation_path.read_bytes())
    assert validation["split"] == "validation"
    assert validation["passed"] is False
    assert validation["performance_passed"] is False
    assert validation["resource_passed"] is True
    assert validation["registration_sha256"] == ledger["files"][
        "experiments/preregistration/episodic_ltm_dream_factorial_v1.json"
    ]
    assert validation["implementation_sha256"] == {
        "examples/agi/episodic_ltm_dream_bridge_gate.py": ledger["files"][
            "examples/agi/episodic_ltm_dream_bridge_gate.py"
        ],
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py": ledger[
            "files"
        ]["reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py"],
    }
    assert validation["train_calibration_sha256"] == _sha256(calibration_path)
    failed = sorted(name for name, passed in validation["checks"].items() if not passed)
    assert failed == sorted(ledger["validation"]["failed_checks"])
    assert sum(validation["checks"].values()) == ledger["validation"]["checks_passed"]
    assert len(validation["checks"]) == ledger["validation"]["checks_total"]

    test_artifact = ROOT / ledger["test_lock"]["artifact"]
    assert ledger["test_lock"]["opened"] is False
    assert ledger["test_lock"]["artifact_must_be_absent"] is True
    assert not test_artifact.exists()
