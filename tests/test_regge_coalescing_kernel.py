"""실제 기하·유한 구적·곡률 정상해로 합류 근사의 범위를 검사한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/regge_coalescing_kernel.py"
SPEC = importlib.util.spec_from_file_location("ce_coalescing_checks", SOURCE)
model = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(model)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))


def test_current_artifact_and_exact_critical_certificate(report):
    assert report["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, digest in report["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
    assert report["exact_certificate"] == model.exact_certificate()
    assert report["exact_certificate"]["critical_derivatives_2_3_4"] == [
        "0", "48*sqrt(3)", "720*sqrt(3)"]
    assert "공통 계량" in " ".join(report["unfinished"])


@pytest.mark.parametrize("h", [.5, .9, 1, 1.1])
def test_actual_geometry_matches_variable_domain_action_and_equation(h):
    points = model.reference_points()
    points[:2, 0] = [-h, h]
    lengths = model.FINAL.lengths(points)
    interval = model.internal_interval(lengths[model.BOUNDARY_IDS])
    np.testing.assert_allclose(interval, [0, model.limit(h)], atol=1e-13)
    for e in (model.limit(h)*.31, model.limit(h)*.94, 2*h):
        lengths[model.E_ID] = e
        actual = model.FINAL.evaluate(lengths)
        assert model.action(e, h) == pytest.approx(actual["action"], abs=1e-11)
        assert model.gradient(e, h) == pytest.approx(actual["gradient"][model.E_ID], abs=1e-11)
    assert model.hessian(2*h, h) == pytest.approx(12*math.sqrt(3)*h*(h*h-1), abs=1e-12)


def test_curved_stationary_branch_is_an_actual_nonflat_solution(report):
    for row in report["branches"]:
        assert row["minimum_gram_eigenvalue"] > 0
        assert row["stationary_residual"] < 1e-10
        assert row["hessian_difference"] < 1e-7
        if abs(row["edge"]-2*row["h"]) > 1e-8:
            assert row["internal_deficit"] > .09
        else:
            assert row["internal_deficit"] < 1e-10
    for h in (.5, math.sqrt(7)/3):
        assert len(model.stationary_points(h)) == 1
    assert len(model.stationary_points(.9)) == 2
    assert model.stationary_points(1) == [2]
    assert len(model.stationary_points(1.1)) == 2


def test_stationary_classification_and_degenerate_inflection():
    for h in (.9, .99, 1.01, 1.1):
        left, right = model.stationary_points(h)
        assert model.hessian(left, h) < 0 < model.hessian(right, h)
        assert model.action(left, h) > model.action(right, h)
    assert model.gradient(1.99, 1) > 0
    assert model.gradient(2.01, 1) > 0
    with pytest.raises(ValueError, match="합류점"):
        model.gaussian_term(20, 1, "length")


@pytest.mark.parametrize("kind", ["length", "squared"])
def test_full_integrals_normalize_and_remain_bounded_with_moving_boundary(kind, report):
    for h in (.9, 1, 1.1):
        value, _ = model.integral(0, h, kind, order=128)
        assert value == pytest.approx(1, abs=1e-12)
    for row in report["integrals"]:
        assert abs(complex(*row["kernel"])) <= 1+1e-12
        assert row["quadrature_difference"] < 1e-8
    current, _ = model.integral(20, 1, kind)
    independent, _ = model.integral(20, 1, kind, order=1024)
    saved = next(row for row in report["integrals"]
                 if row["h"] == 1 and row["beta"] == 20 and row["measure"] == kind)
    assert current == pytest.approx(independent, abs=1e-8)
    assert current == pytest.approx(complex(*saved["kernel"]), abs=1e-10)


def test_two_normalized_measures_remain_distinct_at_coalescence(report):
    rows = [row for row in report["integrals"] if row["h"] == 1 and row["beta"] == 20]
    assert abs(complex(*rows[0]["kernel"])-complex(*rows[1]["kernel"])) > .1
    ratio = model.density(2, 1, "squared")/model.density(2, 1, "length")
    assert ratio == pytest.approx(6/math.sqrt(10))


def test_full_kernel_continuity_is_distinct_from_gaussian_divergence(report):
    values = report["continuity"]
    assert all(right["maximum_kernel_difference"] < left["maximum_kernel_difference"]
               for left, right in zip(values, values[1:]))
    assert values[-1]["maximum_kernel_difference"] < 4e-5
    assert values[-1]["maximum_gaussian_modulus"] > 50
    assert values[-1]["maximum_kernel_difference"]/values[-1]["offset"] < 36


def test_uniform_coefficients_have_the_critical_limit_and_squared_control():
    for kind in ("length", "squared"):
        critical = model.airy_parameters(1, kind)
        for h in (1-.0001, 1+.0001):
            nearby = model.airy_parameters(h, kind)
            assert nearby["a0"] == pytest.approx(critical["a0"], rel=.001)
            assert nearby["a1"] == pytest.approx(critical["a1"], rel=.002)
            assert nearby["delta"]/(h-1)**2 == pytest.approx(model.KAPPA**2/4, rel=.001)


def test_airy_correction_matches_full_quadrature_near_merging_saddles(report):
    for row in report["integrals"]:
        if row["beta"] == 320 and row["h"] in (.99, 1, 1.01):
            assert row["airy_and_endpoints_error"] < .0005
            if row["h"] != 1:
                assert row["airy_and_endpoints_error"] < .01*row["gaussian_and_endpoints_error"]
    for row in report["critical_asymptotic"]:
        assert row["quadrature_difference"] < 1e-8
        if row["beta"] == 1280:
            assert row["airy_and_endpoints_error"] < 4e-5
            assert row["leading_error"] < .02


def test_fixed_geometry_regression(report):
    assert report["old_action_regression"] < 1e-12
    for edge in np.linspace(0, model.limit(.5), 31):
        assert model.action(edge, .5) == pytest.approx(model.symmetric_action(edge), abs=1e-12)


@pytest.mark.parametrize("h", [0, -1, float("nan"), float("inf")])
def test_invalid_boundary_height_rejected(h):
    with pytest.raises(ValueError):
        model.integral(1, h, "length")


@pytest.mark.parametrize("beta", [-1, float("nan"), float("inf")])
def test_invalid_phase_coefficient_rejected(beta):
    with pytest.raises(ValueError):
        model.integral(beta, 1, "length")


def test_approximations_reject_unsupported_domains():
    with pytest.raises(ValueError, match="확인한"):
        model.airy_term(20, .8, "length")
    with pytest.raises(ValueError, match="정상 끝점"):
        model.endpoint_term(20, math.sqrt(7)/3, "length")
    with pytest.raises(ValueError, match="측도"):
        model.integral(1, 1, "unknown")
    with pytest.raises(ValueError, match="차수"):
        model.integral(1, 1, "length", order=True)


def test_unresolved_distinct_saddles_are_rejected_without_breaking_exact_coalescence():
    with pytest.raises(ValueError, match="분리할 수 없다"):
        model.airy_parameters(float(np.nextafter(1, 0)), "length")
    assert np.isfinite(model.airy_term(20, 1, "length"))
