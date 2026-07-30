from __future__ import annotations

import copy
import json
import math

import pytest

from reality_stone.clarus.bootstrap_solver import BootstrapSolver
from reality_stone.clarus.core_model_selection import (
    NOT_TESTABLE,
    RECURSION_SCOPE,
    UNDERIDENTIFIED,
    candidate_by_id,
    candidate_specs,
    evaluate_manifest,
    load_manifest,
    validate_manifest,
)


CE_CANDIDATE_ID = "exponential__linear__ce_delta"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def report(manifest):
    return evaluate_manifest(manifest)


def test_manifest_declares_scalar_equal_row_sum_scope(manifest):
    assert manifest["recursion_scope"] == RECURSION_SCOPE
    assert manifest["model_spec"]["recursion_scope"] == RECURSION_SCOPE
    assert "not the full vector A recursion" in manifest["description"]


def test_wrong_or_missing_recursion_scope_is_rejected(manifest):
    wrong = copy.deepcopy(manifest)
    wrong["recursion_scope"] = "full_vector"
    with pytest.raises(ValueError, match="recursion_scope"):
        validate_manifest(wrong)

    missing_model_scope = copy.deepcopy(manifest)
    del missing_model_scope["model_spec"]["recursion_scope"]
    with pytest.raises(ValueError, match="model_spec recursion_scope"):
        validate_manifest(missing_model_scope)


def test_manifest_expands_to_preregistered_27_model_grid(manifest, report):
    specs = candidate_specs(manifest)
    assert len(specs) == 3 * 3 * 3 == 27
    assert report.candidate_count == 27
    assert all(spec.recursion_scope == RECURSION_SCOPE for spec in specs)


def test_exponential_is_factorization_compatible_but_controls_are_not(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    rational_q1 = candidate_by_id(report, "rational_q1__linear__ce_delta")
    rational_q2 = candidate_by_id(report, "rational_q2__linear__ce_delta")

    assert ce.algebraic.factorization_compatible
    assert ce.algebraic.factorization_defect_max < 1.0e-12
    assert not rational_q1.algebraic.factorization_compatible
    assert not rational_q2.algebraic.factorization_compatible
    assert rational_q1.algebraic_status == "PASS"
    assert rational_q2.algebraic_status == "PASS"


def test_all_preregistered_feedback_and_survival_controls_pass_basic_algebra(report):
    assert report.algebraic_status == "PASS"
    assert all(candidate.algebraic_status == "PASS" for candidate in report.candidates)
    for candidate in report.candidates:
        audit = candidate.algebraic
        assert audit.survival_positive_and_bounded
        assert audit.survival_monotone
        assert audit.feedback_bounded
        assert audit.feedback_monotone
        assert audit.root_count >= 1
        assert audit.max_root_residual <= 1.0e-12


def test_ce_scan_returns_legacy_nontrivial_and_trivial_roots(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    roots = {branch.root.branch_label: branch.root for branch in ce.branches}

    assert set(roots) == {"nontrivial_1", "trivial"}
    assert math.isclose(
        roots["nontrivial_1"].value,
        0.0486466333,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    )
    assert math.isclose(roots["trivial"].value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert roots["nontrivial_1"].residual <= 1.0e-12
    assert roots["trivial"].residual <= 1.0e-12


def test_ce_branch_stability_is_separate_from_root_existence(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    roots = {branch.root.branch_label: branch.root for branch in ce.branches}
    physical = roots["nontrivial_1"]
    trivial = roots["trivial"]

    assert physical.stable
    assert physical.eligible_for_selection
    assert physical.stability_radius < 1.0
    assert not trivial.stable
    assert not trivial.eligible_for_selection
    assert trivial.stability_radius > 1.0


def test_generalized_ce_candidate_regresses_to_legacy_bootstrap_solver(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    generalized = next(
        branch.root.value
        for branch in ce.branches
        if branch.root.branch_label == "nontrivial_1"
    )
    legacy = BootstrapSolver().solve(method="brent")

    assert math.isclose(generalized, legacy, rel_tol=0.0, abs_tol=1.0e-10)


def test_one_selection_observable_is_explicitly_underidentified(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    physical = next(
        branch
        for branch in ce.branches
        if branch.root.branch_label == "nontrivial_1"
    )

    assert report.n_selection_observations == 1
    assert report.n_independent_selection_observations == 1
    assert report.selection_status == UNDERIDENTIFIED
    assert ce.selection_status == UNDERIDENTIFIED
    assert physical.selection.status == UNDERIDENTIFIED
    assert physical.selection.n_observations == 1
    assert physical.selection.n_independent_observations == 1
    assert physical.selection.chi2 is not None


def test_inputs_never_enter_the_selection_denominator(report):
    ce = candidate_by_id(report, CE_CANDIDATE_ID)
    physical = next(
        branch
        for branch in ce.branches
        if branch.root.branch_label == "nontrivial_1"
    )
    residual_keys = {item.key for item in physical.selection.residuals}

    assert residual_keys == {"omega_b_fraction"}
    assert "spatial_dimension" not in residual_keys
    assert "delta_legacy" not in residual_keys


def test_stable_trivial_only_candidate_is_not_eligible_for_selection(report):
    candidate = candidate_by_id(
        report,
        "rational_q1__residual_square__ce_delta",
    )

    assert {branch.root.branch_label for branch in candidate.branches} == {"trivial"}
    assert candidate.branches[0].root.stable
    assert not candidate.branches[0].root.eligible_for_selection
    assert candidate.branches[0].selection.status == NOT_TESTABLE
    assert candidate.selection_status == NOT_TESTABLE


def test_report_never_conflates_algebraic_and_selection_status(report):
    payload = report.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["algebraic_status"] == "PASS"
    assert payload["selection_status"] == UNDERIDENTIFIED
    assert all(
        "algebraic_status" in candidate and "selection_status" in candidate
        for candidate in payload["candidates"]
    )
    assert "VALIDATED" not in encoded.upper()
