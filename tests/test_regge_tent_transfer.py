"""천막 이동의 작용·제약·합성을 독립 기하와 직접 12단체 계산으로 검증한다."""

import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"verify/Q-0020/regge_tent_transfer.py"
SPEC = importlib.util.spec_from_file_location("ce_regge_tent_checks", SOURCE)
tent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tent)


@pytest.fixture(scope="module")
def pair():
    return tent.two_step()


def test_triangle_areas_and_angles_from_embedded_normals():
    points = tent.reference_points()
    complex_ = tent.ReggeComplex(tent.tent_cells())
    evaluation = complex_.evaluate(complex_.lengths(points))
    for triangle, area in zip(complex_.triangles, evaluation["areas"]):
        p, q, r = points[list(triangle)]
        vectors = np.array([q-p, r-p])
        assert area == pytest.approx(math.sqrt(np.linalg.det(vectors @ vectors.T))/2, abs=1e-12)
    for cell, angles in zip(complex_.cells, evaluation["dihedrals"]):
        local = points[list(cell)]
        outward = []
        for excluded in range(5):
            face = np.delete(local, excluded, axis=0)
            _, _, vh = np.linalg.svd(face[1:]-face[0], full_matrices=True)
            normal = vh[-1]
            if np.dot(normal, local[excluded]-face[0]) > 0:
                normal = -normal
            outward.append(normal)
        for triangle, angle in zip(combinations(range(5), 3), angles):
            i, j = [index for index in range(5) if index not in triangle]
            independent = math.acos(-np.dot(outward[i], outward[j]))
            assert angle == pytest.approx(independent, abs=1e-12)


def test_schlafli_gradient_against_direct_action_difference():
    complex_ = tent.ReggeComplex(tent.tent_cells())
    lengths = complex_.lengths(tent.reference_points())
    # 정상 배경에서 떨어진 경우도 검사하여 결손각 미분 누락을 잡는다.
    lengths[-1] += .001
    result = complex_.evaluate(lengths)
    delta = 2e-6
    for index in range(len(lengths)):
        direction = np.eye(len(lengths))[index]*delta
        derivative = (complex_.evaluate(lengths+direction)["action"]-complex_.evaluate(lengths-direction)["action"])/(2*delta)
        assert result["gradient"][index] == pytest.approx(derivative, abs=3e-8)


def test_cell_order_and_vertex_order_do_not_change_action():
    first = tent.ReggeComplex(tent.tent_cells())
    second = tent.ReggeComplex([tuple(reversed(c)) for c in reversed(tent.tent_cells())])
    lengths = first.lengths(tent.reference_points())
    a, b = first.evaluate(lengths), second.evaluate(lengths)
    assert a["action"] == pytest.approx(b["action"], abs=1e-12)
    np.testing.assert_allclose(a["gradient"], b["gradient"], atol=1e-12)


@pytest.mark.parametrize("index", ["first", "second"])
def test_flat_gauge_and_one_physical_direction(pair, index):
    result = pair[index]
    assert result["bulk_deficit_residual"] < 1e-12
    assert abs(result["pole_gradient"]) < 1e-12
    assert result["mixed_singular_values"][0] > 1
    assert result["mixed_singular_values"][1] < 1e-8
    assert min(values[-1] for values in result["gauge_singular_values"]) > .1
    assert result["mixed_gauge_residual"] < 1e-8
    assert result["hessian_skew"] < 1e-8
    assert np.linalg.det(result["map"]) == pytest.approx(1, abs=1e-12)
    # 경계 자체를 움직이면 외재 곡률 항이 남으므로 이 값은 소거하지 않는다.
    boundary_momentum = result["evaluation"]["gradient"][result["ids"][:5]]
    assert np.linalg.norm(result["yin"].T @ boundary_momentum) > .01


def test_pre_post_constraints_include_momentum_shift(pair):
    result = pair["first"]
    a, b, c = result["effective"][:5, :5], result["effective"][:5, 5:], result["effective"][5:, 5:]
    qin, qout = result["ein"]*.3, result["eout"]*(-.2)
    pin, pout = -a @ qin-b @ qout, b.T @ qin+c @ qout
    assert np.linalg.norm(result["yin"].T @ (pin+a @ qin)) < 1e-8
    assert np.linalg.norm(result["yout"].T @ (pout-c @ qout)) < 1e-8
    np.testing.assert_allclose(result["map"] @ [.3, result["ein"] @ pin], [-.2, result["eout"] @ pout], atol=1e-11)
    shift = result["yin"] @ np.array([.1, .2, -.1, .05])
    shifted_pin = -a @ (qin+shift)-b @ qout
    np.testing.assert_allclose(shifted_pin-pin, -a @ shift, atol=1e-12)
    # 좌표의 게이지 이동에는 운동량의 이동도 함께 따른다.
    np.testing.assert_allclose(b.T @ (qin+shift)+c @ qout, pout, atol=1e-8)


def test_length_and_action_coefficient_scaling(pair):
    original = pair["first"]
    enlarged = tent.one_step(tent.reference_points()*3, step=3e-4)
    weighted = tent.one_step(beta=2)
    assert enlarged["evaluation"]["action"] == pytest.approx(9*original["evaluation"]["action"], rel=1e-12)
    np.testing.assert_allclose(enlarged["evaluation"]["gradient"], 3*original["evaluation"]["gradient"], atol=1e-11)
    np.testing.assert_allclose(enlarged["coefficients"], original["coefficients"], atol=1e-7)
    np.testing.assert_allclose(weighted["coefficients"], 2*original["coefficients"], atol=1e-9)


def test_hessian_coefficients_converge_with_difference_step(pair):
    finer = tent.one_step(step=5e-5)
    np.testing.assert_allclose(finer["hessian"], pair["first"]["hessian"], atol=1e-7)
    np.testing.assert_allclose(finer["coefficients"], pair["first"]["coefficients"], atol=1e-7)


def test_direct_global_action_requires_fixed_corner(pair):
    assert len(pair["complex"].cells) == 12
    assert len(pair["complex"].edges) == 26
    assert pair["full_hessian"].shape == (17, 17)
    assert pair["residuals"]["action_with_corner"] < 1e-12
    assert pair["corner"] > 18
    for key in ("assembled_hessian", "internal_stationarity", "global_bulk_deficits"):
        assert pair["residuals"][key] < 1e-8


def test_full_internal_quotient_matches_sequential_elimination(pair):
    assert pair["internal_gauge"].shape == (7, 4)
    assert np.linalg.matrix_rank(pair["internal_gauge"]) == 4
    assert len(pair["internal_physical_eigenvalues"]) == 3
    np.testing.assert_allclose(pair["direct_outer"], pair["sequential_outer"], atol=1e-8)
    for key in ("internal_gauge", "middle_gauge", "coefficients", "canonical_composition"):
        assert pair["residuals"][key] < 1e-8
    assert sum(abs(pair["middle_eigenvalues"]) < 1e-8) == 4


def test_oscillatory_composition_keeps_maslov_phase(pair):
    assert pair["middle_denominator"] < 0
    assert pair["maslov_phase"] == pytest.approx(np.exp(-1j*math.pi/4))
    assert pair["residuals"]["gaussian_with_maslov"] < 1e-12
    assert pair["residuals"]["gaussian_norm"] < 1e-12
    w, amplitude = .8+.3j, (.8/math.pi)**.25
    w1, amp1 = tent.gaussian_transfer(pair["first"]["coefficients"], w, amplitude)
    _, amp2 = tent.gaussian_transfer(pair["second"]["coefficients"], w1, amp1)
    _, direct = tent.gaussian_transfer(pair["coefficients"], w, amplitude)
    assert abs(amp2-direct) > .01
    # 음의 중간 계수에서는 실수 가우스 가중치가 증가한다.
    assert math.exp(-pair["middle_denominator"]/2) > 1e4


def test_planar_control_is_nondegenerate_but_pole_elimination_fails():
    points = tent.reference_points(0)
    complex_ = tent.ReggeComplex(tent.tent_cells())
    evaluation = complex_.evaluate(complex_.lengths(points))
    assert evaluation["minimum_gram_eigenvalue"] > .2
    with pytest.raises(ValueError, match="천막변"):
        tent.one_step(points)


def test_offshell_and_invalid_branches_are_rejected(pair):
    lengths = pair["first"]["lengths"].copy()
    lengths[pair["first"]["ids"][-1]] += .001
    with pytest.raises(ValueError, match="정상 조건"):
        tent.one_step(lengths=lengths)
    with pytest.raises(ValueError):
        tent.scalar_map([1, 0, 1])
    with pytest.raises(ValueError):
        tent.compose_coefficients([1, 1, 2], [-2, 1, 1])
    with pytest.raises(ValueError):
        tent.gaussian_transfer([1, 1, 1], -1, 1)
    with pytest.raises(ValueError):
        pair["complex"].evaluate(pair["lengths"], beta=float("nan"))


def test_saved_source_hash_and_scope():
    report = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["source_sha256"][SOURCE.name] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert report["scope"]["two_step_global_action_and_quotient_composition"]
    for key in ("euclidean_middle_gaussian_converges", "existing_split_V_derived", "physical_clock_or_mass_derived", "common_metric_selected", "lorentzian_einstein_limit_derived"):
        assert report["scope"][key] is False
