from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    ROOT / "artifacts" / "agi" / "episodic_ltm_dream_factorial_integrity_v2.json"
)


def _canonical_utf8_lf_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    assert raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    raw.decode("utf-8")
    return raw


def _json(path: Path) -> dict[str, Any]:
    return json.loads(_canonical_utf8_lf_bytes(path))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_commit(value: str) -> None:
    assert len(value) == 40
    assert set(value) <= set("0123456789abcdef")


def test_g7m_v2_pass_is_byte_locked_with_an_unbroken_test_chain() -> None:
    ledger = _json(LEDGER)

    assert ledger["status"] == "validation_PASS_test_PASS_byte_locked"
    assert ledger["creation_provenance"] == {
        "branch": "research/agi-g7m-v2-validation",
        "head": "a2700770ec42e330ae0bf3bda919867e6932f532",
        "head_parent": "9d11f86d053368302f2b7f3d8510d689ca036aaf",
        "head_subject": "test(agi): verify V2 post-unlock artifact chain",
        "scientific_artifacts_tracked_at_head": True,
        "all_locked_files_tracked_at_head": True,
        "content_basis": (
            "Raw bytes at the stated head, including the post-test audit-only "
            "test-source transition recorded below; no registered seed is generated "
            "or evaluated by this ledger."
        ),
    }
    for commit in ledger["commits"].values():
        _assert_commit(commit)
    _assert_commit(ledger["creation_provenance"]["head"])
    _assert_commit(ledger["creation_provenance"]["head_parent"])
    assert ledger["creation_provenance"]["head"] == ledger["commits"][
        "post_test_audit"
    ]
    assert ledger["creation_provenance"]["head_parent"] == ledger["commits"][
        "locked_test"
    ]
    assert ledger["post_test_audit_transition"] == {
        "path": "tests/test_episodic_ltm_dream_bridge_v2.py",
        "base_commit": "d74a56bdbc70a2adb5cb6149a2c3a584c0529960",
        "transition_commit": "a2700770ec42e330ae0bf3bda919867e6932f532",
        "base_raw_sha256": (
            "11ef3c176045e2a5c51e347411a9353a5fa417777694b580c25039e99aea4ff7"
        ),
        "current_raw_sha256": (
            "ca9d7366e652e89235cc3853048d67cdda2129850b4a5e657eb82d871f865f5c"
        ),
        "scope": (
            "Replace the pre-test-only artifact-absence assertion with a read-only "
            "byte-chain audit when the locked test artifact exists."
        ),
        "model_changed": False,
        "gate_changed": False,
        "registered_seed_generated": False,
        "artifact_regenerated": False,
    }

    expected_paths = {
        "experiments/preregistration/episodic_ltm_dream_factorial_v2.json",
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py",
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py",
        "tests/test_episodic_ltm_dream_bridge_v2.py",
        "artifacts/agi/episodic_ltm_dream_factorial_implementation_lock_v2.json",
        "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v2.json",
        "artifacts/agi/episodic_ltm_dream_factorial_validation_v2.json",
        "artifacts/agi/episodic_ltm_dream_factorial_test_v2.json",
    }
    assert set(ledger["files"]) == expected_paths
    for relative, contract in ledger["files"].items():
        raw = _canonical_utf8_lf_bytes(ROOT / relative)
        assert _sha256(raw) == contract["raw_sha256"]
        assert len(raw) == contract["size_bytes"]
        assert contract["eol"] == "LF"
        assert contract["terminal_lf_count"] == 1
        assert contract["utf8_bom"] is False
        _assert_commit(contract["first_locked_commit"])

    prereg_path = (
        ROOT
        / "experiments"
        / "preregistration"
        / "episodic_ltm_dream_factorial_v2.json"
    )
    module_path = (
        ROOT
        / "reality_stone"
        / "python"
        / "reality_stone"
        / "clarus"
        / "episodic_ltm_dream_bridge_v2.py"
    )
    cli_path = ROOT / "examples" / "agi" / "episodic_ltm_dream_bridge_v2_gate.py"
    implementation_lock_path = (
        ROOT
        / "artifacts"
        / "agi"
        / "episodic_ltm_dream_factorial_implementation_lock_v2.json"
    )
    calibration_path = (
        ROOT
        / "artifacts"
        / "agi"
        / "episodic_ltm_dream_factorial_train_calibration_v2.json"
    )
    validation_path = (
        ROOT
        / "artifacts"
        / "agi"
        / "episodic_ltm_dream_factorial_validation_v2.json"
    )
    test_path = (
        ROOT / "artifacts" / "agi" / "episodic_ltm_dream_factorial_test_v2.json"
    )

    prereg = _json(prereg_path)
    implementation_lock = _json(implementation_lock_path)
    calibration = _json(calibration_path)
    validation = _json(validation_path)
    test = _json(test_path)

    chain = ledger["lock_chain"]
    file_contracts = ledger["files"]
    registration_sha256 = file_contracts[
        "experiments/preregistration/episodic_ltm_dream_factorial_v2.json"
    ]["raw_sha256"]
    implementation_sha256 = {
        "examples/agi/episodic_ltm_dream_bridge_v2_gate.py": _sha256(
            _canonical_utf8_lf_bytes(cli_path)
        ),
        "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py": (
            _sha256(_canonical_utf8_lf_bytes(module_path))
        ),
    }
    implementation_lock_sha256 = file_contracts[
        "artifacts/agi/episodic_ltm_dream_factorial_implementation_lock_v2.json"
    ]["raw_sha256"]
    calibration_sha256 = file_contracts[
        "artifacts/agi/episodic_ltm_dream_factorial_train_calibration_v2.json"
    ]["raw_sha256"]
    validation_sha256 = file_contracts[
        "artifacts/agi/episodic_ltm_dream_factorial_validation_v2.json"
    ]["raw_sha256"]
    test_sha256 = file_contracts[
        "artifacts/agi/episodic_ltm_dream_factorial_test_v2.json"
    ]["raw_sha256"]

    assert chain == {
        "registration_sha256": registration_sha256,
        "implementation_lock_artifact_sha256": implementation_lock_sha256,
        "train_calibration_sha256": calibration_sha256,
        "validation_artifact_sha256": validation_sha256,
        "test_artifact_sha256": test_sha256,
    }
    assert implementation_lock["registration_sha256"] == registration_sha256
    assert implementation_lock["implementation_sha256"] == implementation_sha256
    assert implementation_lock["registered_seed_used_for_prelock_equivalence"] == 0
    assert implementation_lock["off_range_shared_equivalence_report"][
        "registered_seed_count"
    ] == 0
    assert implementation_lock["off_range_shared_equivalence_report"][
        "all_passed"
    ] is True

    immutable_v1_sha256 = implementation_lock["immutable_v1_dependency_sha256"]
    frozen_v1_equivalence_sha256 = implementation_lock[
        "frozen_v1_comparator_equivalence_sha256"
    ]
    for artifact in (calibration, validation, test):
        assert artifact["registration_sha256"] == registration_sha256
        assert artifact["implementation_sha256"] == implementation_sha256
        assert artifact["implementation_lock_artifact_sha256"] == (
            implementation_lock_sha256
        )
        assert artifact["immutable_v1_dependency_sha256"] == immutable_v1_sha256
        assert artifact["frozen_v1_comparator_equivalence_sha256"] == (
            frozen_v1_equivalence_sha256
        )

    assert calibration["source_split"] == "train_only"
    for result in (validation, test):
        assert result["train_calibration_sha256"] == calibration_sha256
        assert result["passed"] is True
        assert result["performance_passed"] is True
        assert result["resource_passed"] is True
        assert len(result["checks"]) == 84
        assert all(result["checks"].values())

    assert validation["split"] == "validation"
    assert len(validation["seed_results"]) == 40
    assert validation["resource_usage"]["evaluation_seeds"] == 40
    assert validation["resource_usage"]["train_seeds"] == 40
    assert validation["test_lock"] == {
        "test_opened_after_validation_pass": False,
        "validation_artifact_sha256": None,
    }

    assert test["split"] == "test"
    assert len(test["seed_results"]) == 60
    assert test["resource_usage"]["evaluation_seeds"] == 60
    assert test["resource_usage"]["train_seeds"] == 0
    assert test["test_lock"] == {
        "test_opened_after_validation_pass": True,
        "validation_artifact_sha256": validation_sha256,
    }

    assert prereg["data_roles"]["train"]["seeds"] == list(range(80100, 80140))
    assert prereg["data_roles"]["validation"]["seeds"] == list(
        range(81100, 81140)
    )
    assert prereg["data_roles"]["test"]["seeds"] == list(range(82100, 82160))
    assert prereg["test_lock"]["open_only_after_validation_all_of_pass"] is True
    assert prereg["test_lock"]["v1_test_79100_79159_remains_forbidden"] is True

    assert ledger["outcomes"]["validation"] == {
        "split": "validation",
        "seeds": "81100-81139",
        "seed_count": 40,
        "passed": True,
        "performance_passed": True,
        "resource_passed": True,
        "checks_total": 84,
        "checks_passed": 84,
        "test_opened_in_this_artifact": False,
    }
    assert ledger["outcomes"]["test"] == {
        "split": "test",
        "seeds": "82100-82159",
        "seed_count": 60,
        "passed": True,
        "performance_passed": True,
        "resource_passed": True,
        "checks_total": 84,
        "checks_passed": 84,
        "opened_only_after_validation_pass": True,
    }
