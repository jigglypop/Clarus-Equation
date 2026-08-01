from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from reality_stone.clarus.local_memory_verifier import verify_confirmation


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "artifacts/agi/local_memory_aml32_preregistration.json"
RESULT_PATHS = {
    1: ROOT / "artifacts/agi/local_memory_aml32_h1_confirmatory.json",
    6: ROOT / "artifacts/agi/local_memory_aml32_h6_confirmatory.json",
}
IMPLEMENTATION = (
    ROOT / "reality_stone/python/reality_stone/clarus/local_memory.py"
)


def _inputs() -> tuple[dict[str, object], dict[int, dict[str, object]], str]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    results = {
        horizon: json.loads(path.read_text(encoding="utf-8"))
        for horizon, path in RESULT_PATHS.items()
    }
    implementation_hash = hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest()
    return preregistration, results, implementation_hash


def test_locked_confirmatory_artifacts_recompute_to_pass() -> None:
    preregistration, results, implementation_hash = _inputs()

    proof = verify_confirmation(
        preregistration,
        results,
        implementation_sha256=implementation_hash,
    )

    assert proof["proof_passed"]
    assert proof["errors"] == []


def test_tampered_threshold_is_rejected() -> None:
    preregistration, results, implementation_hash = _inputs()
    tampered = copy.deepcopy(results)
    tampered[1]["result"]["recordings"][0]["criteria"]["min_memory_delta"] = 0.0

    proof = verify_confirmation(
        preregistration,
        tampered,
        implementation_sha256=implementation_hash,
    )

    assert not proof["proof_passed"]
    assert any("criteria changed" in error for error in proof["errors"])


def test_tampered_implementation_hash_is_rejected() -> None:
    preregistration, results, _ = _inputs()

    proof = verify_confirmation(
        preregistration,
        results,
        implementation_sha256="0" * 64,
    )

    assert not proof["proof_passed"]
    assert "implementation hash differs from preregistration" in proof["errors"]
