"""사후 제약 전달의 기하·밀도·누설·정준 가지를 독립 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_postconstraint_projection.py"
SPEC = importlib.util.spec_from_file_location("ce_postconstraint_checks", SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_closed_gradient_matches_actual_four_simplex_action(report):
    for row in report["geometry"]:
        assert row["gradient_error"] < 1e-9
        assert row["minimum_gram"] > 0
        assert row["interval_error"] < 1e-10
        assert row["flat_mixed_formula_error"] < 1e-10


def test_full_gram_interval_has_distinct_one_sided_boundary_derivatives(report):
    for row in report["geometry"]:
        assert abs(row["left_upper_derivative"]) < 1e-7
        assert row["right_upper_derivative"] < -.2
        assert row["right_derivative_error"] < 1e-7


def test_moving_fiber_operator_matches_actual_phase_difference_in_both_densities(report):
    for row in report["operators"]:
        assert row["operator_error"] < 1e-6
        assert row["original_c_error"] < 1e-6
        assert row["original_c_lift_relation_error"] < 1e-6
    assert {row["beta"] for row in report["operators"]} == {0., 5.}


def test_nonzero_lower_endpoint_exposes_missing_density_connection(report):
    rows = [row for row in report["operators"] if row["asymmetric"]]
    assert all(row["lower"] > .03 for row in rows)
    squared = next(row for row in rows if row["kind"] == "squared")
    assert squared["omitted_connection_residual"] > 5e-4
    assert squared["omitted_connection_residual"] == pytest.approx(
        squared["omitted_connection_size"], abs=1e-6)


def test_whole_fiber_leakage_is_positive_even_at_rank_drop(report):
    for row in report["distributions"]:
        assert row["variance"] > .09
        assert row["quadrature_difference"] < 1e-8
        assert row["beta_zero_leakage"] == 0
        assert row["leakage_squared"] == pytest.approx(25*row["variance"], abs=1e-12)
    critical = next(row for row in report["distributions"] if row["h"] == 1.)
    assert critical["variance"] == pytest.approx(.1702520758063384, abs=1e-11)
    assert critical["leakage_squared"] > 4


def test_matching_endpoint_values_do_not_imply_zero_leakage(report):
    row = report["distributions"][-1]
    assert row["h"] == pytest.approx(math.sqrt(5/3))
    assert abs(row["endpoint_jump"]) < 1e-12
    assert row["leakage_squared"] > 2


def test_fourier_leakage_converges_but_derivative_sum_grows(report):
    fourier = report["fourier"]
    target_variance = report["distributions"][2]["variance"]
    last_variance, last_sum, last_rate_error = 0., 0., math.inf
    for row in fourier["rows"]:
        assert last_variance < row["partial_variance"] < target_variance
        total = row["cutoff"]*row["derivative_sum_over_cutoff"]
        assert total > last_sum
        rate_error = abs(row["derivative_sum_over_cutoff"]-fourier["asymptotic_rate"])
        assert rate_error < last_rate_error
        last_variance, last_sum, last_rate_error = row["partial_variance"], total, rate_error
    assert report["fourier_grid_difference"] < 1e-3
    assert fourier["asymptotic_rate"] == pytest.approx(.5931834714399288, abs=1e-12)
    assert complex(*fourier["asymptotic_n_coefficient"]).imag < 0


def test_local_darboux_inverse_has_two_branches_at_h1(report):
    monotone, fold = report["darboux"]
    assert monotone["critical_edge"] is None
    assert fold["critical_edge"] == 2
    assert abs(fold["critical_gradient"]) < 1e-12
    assert fold["curvature"] == pytest.approx(6*math.sqrt(2), abs=1e-12)
    assert fold["curvature_error"] < 1e-6
    error = math.inf
    for row in fold["branches"]:
        assert row["left"] < 2 < row["right"]
        assert row["left_gradient"] < 0 < row["right_gradient"]
        assert row["inverse_residual"] < 1e-10
        assert abs(row["separation_ratio"]-1) < error
        error = abs(row["separation_ratio"]-1)


@pytest.mark.parametrize("edge", [0., checks.full.limit(1.)])
def test_derivative_does_not_extend_to_degenerate_endpoints(edge):
    with pytest.raises(ValueError):
        checks.symmetric_g_e(edge, 1.)


def test_saved_result_matches_current_source_and_transitive_dependencies():
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected
