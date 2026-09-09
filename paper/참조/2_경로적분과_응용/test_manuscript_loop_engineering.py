"""Mutation and CLI regressions for the manuscript audit ledgers."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import improvement_loop_engineering as improvement  # noqa: E402
import rejection_loop_engineering as rejection  # noqa: E402


def failed_names(checks) -> set[str]:
    return {check.name for check in checks if not check.passed}


def test_regression_witness_digest_float_canonicalization() -> None:
    report = rejection.build_report()
    assert rejection.rejected_literal_occurrence_count() == 0
    assert rejection.rejected_inventory() == []
    assert report.source_rejected_occurrences == 0
    assert report.occurrence_routes == ()
    assert (
        report.regression_witness_registry_sha256
        == rejection.EXPECTED_REGRESSION_WITNESS_REGISTRY_SHA256
    )
    assert rejection.canonical_digest_value(0.6666666666666669) == (
        rejection.canonical_digest_value(0.6666666666666667)
    )
    assert rejection.canonical_digest_value(-0.0) == {"$float15": "0"}
    assert rejection.canonical_digest_value(1.0) != (
        rejection.canonical_digest_value("1")
    )
    for nonfinite in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="non-finite"):
            rejection.canonical_digest_value(nonfinite)


def test_rejection_report_mutations_fail_closed_without_exceptions() -> None:
    report = rejection.build_report()
    baseline_failures = failed_names(rejection.validate_report(report))
    assert "canonical-report-rebuild" not in baseline_failures
    assert "loop-identity-and-aggregate-consistency" not in baseline_failures

    first = report.loops[0]
    mutations = (
        replace(report, loops=report.loops[:-1]),
        replace(
            report,
            loops=(
                replace(
                    first,
                    parent_claim="ALL CLAIMS ARE TRUE",
                    maximum_supported_stage="ARBITRARY_PASS",
                    ce_specific_physical_claim_closed=True,
                ),
            )
            + report.loops[1:],
        ),
    )
    for mutated in mutations:
        failures = failed_names(rejection.validate_report(mutated))
        assert "canonical-report-rebuild" in failures
        assert "loop-identity-and-aggregate-consistency" in failures
        assert failures > baseline_failures

    first_witness = report.deleted_parent_regression_witnesses[0]
    mutated_witness_report = replace(
        report,
        deleted_parent_regression_witnesses=(
            replace(first_witness, passed=False),
        )
        + report.deleted_parent_regression_witnesses[1:],
    )
    witness_failures = failed_names(
        rejection.validate_report(mutated_witness_report)
    )
    assert "canonical-report-rebuild" in witness_failures
    assert "deleted-parent-regression-witness-registry" in witness_failures
    assert witness_failures > baseline_failures


def test_improvement_report_mutations_fail_closed_without_exceptions() -> None:
    report = improvement.build_report()
    baseline_failures = failed_names(improvement.validate_report(report))
    assert "canonical-report-rebuild" not in baseline_failures
    assert "source-pin-integrity" not in baseline_failures
    assert "branch-identity-stage-and-aggregate-consistency" not in baseline_failures

    bad_pins = deepcopy(report.source_pins)
    bad_pins["desi_upstream"]["mean_sha256"] = "bad-hash"
    first = report.branches[0]
    mutations = (
        replace(
            report,
            source_pins=bad_pins,
            branches=(
                replace(
                    first,
                    source_loop_id="missing-loop",
                    maximum_supported_stage="ARBITRARY_PASS",
                    original_claim_promoted=True,
                    ce_specific_physical_claim_closed=True,
                ),
            )
            + report.branches[1:],
        ),
        replace(report, branches=report.branches[:-1]),
    )
    for index, mutated in enumerate(mutations):
        failures = failed_names(improvement.validate_report(mutated))
        assert "canonical-report-rebuild" in failures
        assert "branch-identity-stage-and-aggregate-consistency" in failures
        if index == 0:
            assert "source-pin-integrity" in failures
        assert failures > baseline_failures


def test_improvement_text_json_exit_parity(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        improvement,
        "validate_report",
        lambda report: (
            improvement.SelfCheck("synthetic-cli-check", True, "pass"),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["improvement_loop_engineering.py", "--self-test"])
    text_exit = improvement.main()
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["improvement_loop_engineering.py", "--json"])
    json_exit = improvement.main()
    payload = json.loads(capsys.readouterr().out)

    assert text_exit == json_exit
    assert payload["passed"] is (json_exit == 0)
    assert payload["live_artifact_verification"]["status"] == "NOT_RUN"


def test_release_mode_requires_all_live_artifacts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        improvement,
        "validate_report",
        lambda report: (
            improvement.SelfCheck("synthetic-cli-check", True, "pass"),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "improvement_loop_engineering.py",
            "--json",
            "--require-live-artifacts",
        ],
    )
    exit_code = improvement.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["passed"] is False
    assert payload["live_artifact_verification"]["status"] == "FAIL"
    assert set(payload["live_artifact_verification"]["missing"]) == {
        "portal_pdf",
        "portal_image",
        "desi_mean",
        "desi_covariance",
    }
