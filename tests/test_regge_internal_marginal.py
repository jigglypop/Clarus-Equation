"""실제 내부 변 적분에서 고전 경계 제거·측도·끝점 효과를 구분한다."""

import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/regge_internal_marginal.py"
SPEC = importlib.util.spec_from_file_location("ce_internal_marginal_checks", SOURCE)
model = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(model)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return model.run()


def test_four_to_two_move_preserves_boundary_tetrahedra():
    def boundary(complex_):
        counts = {}
        for cell in complex_.cells:
            for facet in combinations(cell, 4):
                facet = tuple(sorted(facet))
                counts[facet] = counts.get(facet, 0)+1
        return {facet for facet, count in counts.items() if count == 1}
    assert boundary(model.FINAL) == boundary(model.COARSE)
    assert set(model.FINAL.edges)-set(model.COARSE.edges) == {(0, 1)}


def test_exact_gram_domain_and_stationary_coefficient(report):
    import sympy as sp
    e = sp.Symbol("e", positive=True)
    exact = report["exact_certificate"]
    determinant = sp.sympify(exact["gram_determinant"], locals={"e": e})
    assert sp.simplify(determinant-4*e**2*(13-9*e**2)/27) == 0
    assert determinant.subs(e, sp.sqrt(13)/3) == 0
    assert exact["stationary_gradient"] == "0"
    assert exact["internal_hessian"] == "-9*sqrt(3)/2"
    assert report["interval"][1] == pytest.approx(math.sqrt(13)/3)


def test_closed_action_and_gradient_match_actual_regge_complex(report):
    assert max(report["closed_formula_errors"]) < 1e-10
    assert model.symmetric_action(0) == pytest.approx(8*math.pi*math.sqrt(3)/3, abs=1e-12)
    assert model.symmetric_action(model.LIMIT) == pytest.approx(
        2*math.pi*math.sqrt(14)-4*math.pi*math.sqrt(26)/9, abs=1e-12)
    assert model.symmetric_gradient(model.LIMIT) == pytest.approx(-19*math.pi/(12*math.sqrt(2)), abs=1e-12)


def test_stationary_point_is_maximum_with_finite_endpoint_action():
    for e in (.01, .2, .7, .99):
        assert model.symmetric_gradient(e) > 0
    for e in (1.01, 1.1, model.LIMIT-.0001):
        assert model.symmetric_gradient(e) < 0
    assert model.symmetric_action(0) < model.symmetric_action(model.LIMIT) < model.symmetric_action(1)


def test_classical_elimination_matches_all_boundary_momenta_and_hessian(report):
    assert len(report["classical_cases"]) == 3
    for row in report["classical_cases"]:
        assert row["interval"][0] < row["internal_length"] < row["interval"][1]
        assert row["minimum_crossing_barycentric"] > .2
        for field in ("action_residual", "gradient_residual", "internal_equation_residual", "flatness_residual"):
            assert row[field] < 1e-9
        assert row["boundary_schur_residual"] < 1e-6
        assert row["internal_hessian"] < 0
    assert report["classical_cases"][0]["internal_hessian"] == pytest.approx(model.INTERNAL_HESSIAN, abs=1e-8)


def test_crossing_outside_shared_tetrahedron_is_rejected():
    points = model.reference_points()
    points[:2, 1:] += 10
    boundary = model.COARSE.lengths(points)
    with pytest.raises(ValueError, match="내부를 통과"):
        model.flat_completion(boundary)


def test_flat_cospherical_geometry_does_not_allow_gaussian_elimination(report):
    control = report["cospherical_control"]
    np.testing.assert_allclose(np.linalg.norm(control["points"], axis=1), 1, atol=1e-15)
    assert control["minimum_gram_eigenvalue"] > .01
    assert control["internal_length"] == pytest.approx(2)
    assert abs(control["internal_hessian_numeric"]) < 1e-10
    assert report["exact_certificate"]["cospherical_area_derivative"] == "0"
    assert control["schur_rejected"]
    boundary = model.COARSE.lengths(control["points"])
    with pytest.raises(ValueError, match="내부 헤시안이 특이"):
        model.classical_case(boundary)


def test_both_measures_normalize_and_independent_quadratures_agree(report):
    for row in report["integrals"]:
        assert row["quadrature_difference"] < 1e-8
        if row["beta"] == 0:
            assert complex(*row["kernel"]) == pytest.approx(1, abs=1e-12)


def test_normalized_measures_give_distinct_kernels_but_passive_change_preserves_one(report):
    rows = [row for row in report["integrals"] if row["beta"] == 10]
    assert abs(complex(*rows[0]["kernel"])-complex(*rows[1]["kernel"])) > .1
    assert report["passive_coordinate_change_residual"] < 1e-8
    assert report["stationary_density_ratio"] == pytest.approx(6/math.sqrt(13))


def test_endpoint_terms_improve_the_full_oscillatory_approximation(report):
    for row in report["integrals"]:
        if row["beta"] >= 40:
            assert row["saddle_and_endpoints_error"] < .35*row["saddle_only_error"]
            assert row["saddle_and_endpoints_error"]*row["beta"]**1.5 < 1.2
    saddle, _ = model.stationary_terms(40, "length")
    removed_phase = saddle*np.exp(-40j*model.symmetric_action(1))
    assert np.angle(removed_phase) == pytest.approx(-math.pi/4, abs=1e-12)


def test_finite_decaying_integral_is_controlled_by_left_endpoint(report):
    for row in report["integrals"]:
        if row["beta"] == 640:
            assert row["scaled_decaying_kernel"] > 0
            assert row["decaying_ratio_to_leading"] == pytest.approx(1, rel=3e-6)
    assert model.INTERNAL_HESSIAN < 0


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_invalid_integral_coefficient_is_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        model.integral(beta, "length")


def test_artifact_matches_current_source_dependencies_and_results(report):
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert set(saved["dependencies"]) == {"regge_pachner_constraints.py", "regge_pachner_creation.py",
                                         "regge_pachner_transport.py", "regge_tent_transfer.py"}
    for name, digest in saved["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
    for previous, current in zip(saved["integrals"], report["integrals"], strict=True):
        np.testing.assert_allclose(previous["kernel"], current["kernel"], atol=1e-11, rtol=0)
    assert "공통 계량" in " ".join(saved["unfinished"])
