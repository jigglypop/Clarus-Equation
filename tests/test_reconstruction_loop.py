from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import reality_stone.clarus.reconstruction_loop as reconstruction_loop_module
from reality_stone.clarus.reconstruction_loop import (
    build_reconstruction_loop_status,
    main,
    validate_reconstruction_loop_status,
)


@pytest.fixture(scope="module")
def status() -> dict[str, object]:
    return build_reconstruction_loop_status()


def test_loop_advances_subgates_but_does_not_skip_the_protocell_stage(
    status: dict[str, object],
) -> None:
    results = {row["loop_id"]: row for row in status["loop_results"]}

    assert results["P0.0"]["status"] == "pass"
    assert results["P0.1"]["status"] == "pass"
    assert results["P0.2"]["status"] == "pass"
    assert results["P0.2"]["evidence"][0]["independently_verified"]
    assert results["P0.3"]["status"] == "ready"
    assert status["last_completed_loop"] == "P0.2"
    assert status["active_loop"] == "P0.3"
    assert not status["current_stage_promoted"]
    assert not status["next_organism_unlocked"]


def test_stage_ladder_keeps_every_later_organism_locked(
    status: dict[str, object],
) -> None:
    ladder = status["stage_ladder"]

    assert ladder[0]["stage_id"] == "P0"
    assert ladder[0]["status"] == "in_progress"
    assert ladder[-1]["stage_id"] == "P10"
    assert ladder[-1]["target"] == "human-level developmental agent"
    assert all(stage["status"] == "locked" for stage in ladder[1:])
    assert not status["claim_scope"]["human_reconstruction_complete"]


def test_loop_status_validation_fails_closed_on_premature_promotion(
    status: dict[str, object],
) -> None:
    assert validate_reconstruction_loop_status(status)

    changed = deepcopy(status)
    changed["current_stage_promoted"] = True
    changed["next_organism_unlocked"] = True
    changed["stage_ladder"][0]["status"] = "engineering_pass"
    changed["stage_ladder"][1]["status"] = "ready"

    assert not validate_reconstruction_loop_status(changed)


def test_failed_first_gate_locks_later_gates_and_clears_completed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reconstruction_loop_module,
        "build_finite_resource_certificate",
        lambda: {"all_engineering_gates_passed": True},
    )
    monkeypatch.setattr(
        reconstruction_loop_module,
        "verify_finite_resource_certificate",
        lambda _certificate: SimpleNamespace(
            verified=True,
            checks=("forced local pass",),
            errors=(),
        ),
    )
    monkeypatch.setattr(
        reconstruction_loop_module,
        "verify_existence_certificate",
        lambda _certificate: SimpleNamespace(
            verified=False,
            checks=(),
            errors=("forced prerequisite failure",),
        ),
    )

    blocked = build_reconstruction_loop_status()
    results = {row["loop_id"]: row for row in blocked["loop_results"]}

    assert results["P0.0"]["status"] == "fail"
    assert results["P0.1"]["status"] == "locked"
    assert results["P0.1"]["evidence"][0]["passed"]
    assert results["P0.2"]["status"] == "locked"
    assert blocked["last_completed_loop"] is None
    assert blocked["active_loop"] == "P0.0"
    assert not blocked["current_stage_promoted"]


def test_failed_second_gate_preserves_only_the_continuous_passed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reconstruction_loop_module,
        "build_finite_resource_certificate",
        lambda: {"all_engineering_gates_passed": True},
    )
    monkeypatch.setattr(
        reconstruction_loop_module,
        "verify_finite_resource_certificate",
        lambda _certificate: SimpleNamespace(
            verified=False,
            checks=(),
            errors=("forced current-gate failure",),
        ),
    )

    blocked = build_reconstruction_loop_status()
    results = {row["loop_id"]: row for row in blocked["loop_results"]}

    assert results["P0.0"]["status"] == "pass"
    assert results["P0.1"]["status"] == "fail"
    assert results["P0.2"]["status"] == "locked"
    assert blocked["last_completed_loop"] == "P0.0"
    assert blocked["active_loop"] == "P0.1"
    assert not blocked["current_stage_promoted"]


def test_failed_stoichiometric_gate_locks_robustness_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reconstruction_loop_module,
        "verify_stoichiometric_certificate",
        lambda _certificate: SimpleNamespace(
            verified=False,
            checks=(),
            errors=("forced stoichiometric failure",),
        ),
    )

    blocked = build_reconstruction_loop_status()
    results = {row["loop_id"]: row for row in blocked["loop_results"]}

    assert results["P0.0"]["status"] == "pass"
    assert results["P0.1"]["status"] == "pass"
    assert results["P0.2"]["status"] == "fail"
    assert results["P0.3"]["status"] == "locked"
    assert blocked["last_completed_loop"] == "P0.1"
    assert blocked["active_loop"] == "P0.2"
    assert not blocked["current_stage_promoted"]


def test_require_stage_promotion_fails_until_all_p0_loops_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "status.json"

    assert main(["--output", str(output)]) == 0
    capsys.readouterr()
    assert main(
        ["--output", str(output), "--require-stage-promotion"]
    ) == 1
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(["--output", str(output), "--require-current-loop-pass"])
    assert exc_info.value.code == 2


def test_committed_loop_status_matches_fresh_recomputation(
    status: dict[str, object],
) -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "biology"
        / "origin_life_reconstruction_loop_status.json"
    )
    observed = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert observed == status
