"""원래 사전 제약의 전달에서 밀도와 실제 끝점 항을 검사한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_quantum_constraint_transfer.py"
SPEC = importlib.util.spec_from_file_location("ce_quantum_constraint_checks", SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_original_gram_support_does_not_require_coarse_completion():
    q = checks.geometry_lengths(.5)
    q[checks.moves.FINAL.edge_index[0, 2]] = .7
    q[checks.E] = 2/3
    assert checks.moves.FINAL.evaluate(q)["minimum_gram_eigenvalue"] > 0
    with pytest.raises(ValueError):
        checks.full.COARSE.evaluate(q[checks.full.BOUNDARY_IDS])
    lower, upper = checks.fine_interval(q)
    assert lower < 2/3 < upper
    for fraction in (.2, .8):
        sample = q.copy()
        sample[checks.E] = lower+fraction*(upper-lower)
        assert checks.moves.FINAL.evaluate(sample)["minimum_gram_eigenvalue"] > 0


def test_original_action_derivatives_and_geometric_interval(report):
    for row in report["geometry"]:
        assert row["interval_error"] < 1e-12
        assert row["action_h_error"] < 1e-10
    for row in report["operators"]:
        assert row["operator_error"] < 1e-6
        assert row["opposite_phase_formula_error"] < 1e-6
        assert row["opposite_phase_residual"] > 1
        if row["kind"] == "squared":
            assert row["omitted_connection"] > .5


def test_periodic_transfer_extracts_zero_mode_and_projects_excited_mode(report):
    for row in report["transfers"]:
        zero, excited, _ = row["rows"]
        assert complex(*zero["amplitude"]) == pytest.approx(1, abs=1e-12)
        assert abs(complex(*excited["amplitude"])) < 1e-12
        for mode in row["rows"]:
            assert mode["norm"] == pytest.approx(1, abs=1e-12)
            assert mode["boundary_error"] < 1e-10


def test_antiperiodic_domain_has_nonzero_constraint_endpoint_amplitude(report):
    for row in report["transfers"]:
        twisted = row["rows"][2]
        width = checks.full.limit(row["h"])
        assert complex(*twisted["constraint_amplitude"]) == pytest.approx(2j/width, abs=1e-10)
        assert twisted["boundary_magnitude"] > .8


def test_omitting_kernel_density_connection_leaves_bulk_term(report):
    for row in report["transfers"]:
        assert row["wrong_kernel_identity_error"] < 1e-6
        if row["kind"] == "squared":
            assert abs(complex(*row["wrong_kernel_force"])) > .4
        else:
            assert abs(complex(*row["wrong_kernel_force"])) < 1e-10


def test_full_interval_constraint_and_moving_boundary_identities(report):
    for row in report["identities"]:
        assert row["ward_error"] < 1e-6
        assert row["refined_ward_error"] < 1e-6
        assert row["derivative_error"] < 1e-6
        assert row["refined_derivative_error"] < 1e-6
        assert row["order_difference"] < 1e-6
        expected = row["h"]/(row["h"]**2+1/9)*(1 if row["kind"] == "length" else 2)
        assert row["omitted_boundary_error"] == pytest.approx(expected, abs=1e-6)
        if row["beta"] == 0:
            assert complex(*row["kernel"]) == pytest.approx(1, abs=1e-12)
            assert abs(complex(*row["derivative_h"])) < 1e-9


def test_density_covariance_preserves_transfers_but_different_priors_change_kernel(report):
    for h in (.5, .9, 1., 1.1):
        paired = [row for row in report["transfers"] if row["h"] == h]
        for left, right in zip(paired[0]["rows"], paired[1]["rows"]):
            assert complex(*left["amplitude"]) == pytest.approx(complex(*right["amplitude"]), abs=1e-12)
    kernels = [complex(*row["kernel"]) for row in report["identities"] if row["h"] == 1 and row["beta"] == 20]
    assert abs(kernels[0]-kernels[1]) > .1


@pytest.mark.parametrize("order", [0, True, 16])
def test_invalid_quadrature_order_is_rejected(order):
    with pytest.raises(ValueError):
        checks.sample_rule((0, 1), order)


def test_saved_result_matches_current_source_and_dependency_hashes():
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected

