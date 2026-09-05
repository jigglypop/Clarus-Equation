"""전체 길이 차트와 이동 구간의 운동량 전달·좌표 동등성을 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"verify/Q-0020/regge_pachner_transport.py"
SPEC = importlib.util.spec_from_file_location("ce_pachner_transport_checks", SOURCE)
transport = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(transport)
    from regge_pachner_creation import admissible_interval
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return transport.run()


@pytest.mark.parametrize("seed", [11, 23, 41])
def test_fourteen_coordinate_chart_roundtrip_and_independent_length_derivative(seed):
    u = transport.reference_coordinates()+np.random.default_rng(seed).normal(0, .17, 14)
    data = transport.chart(u)
    np.testing.assert_allclose(transport.inverse_chart(data["lengths"]), u, atol=3e-14)
    assert np.linalg.matrix_rank(data["jacobian"]) == 14
    assert np.linalg.svd(data["jacobian"], compute_uv=False)[-1] > .04
    for i in range(14):
        perturbation = np.eye(14)[i]*2e-6
        finite = (transport.chart(u+perturbation)["lengths"]-transport.chart(u-perturbation)["lengths"])/(4e-6)
        np.testing.assert_allclose(finite, data["jacobian"][:, i], atol=3e-9)


@pytest.mark.parametrize("seed", [0, 19])
def test_squared_bounds_match_general_gram_formula_and_derivatives(seed):
    u = transport.reference_coordinates()+np.random.default_rng(seed).normal(0, .15, 14)
    data = transport.squared_bounds(u)
    low, high = admissible_interval(data["lengths"][transport.MOVE.local_old_ids])
    assert data["A"] == pytest.approx(low**2, abs=4e-14)
    assert data["B"] == pytest.approx(high**2, abs=4e-14)
    assert data["D"] == pytest.approx(data["B"]-data["A"], abs=4e-14)
    for i in range(14):
        delta = np.eye(14)[i]*1e-6
        plus, minus = transport.squared_bounds(u+delta), transport.squared_bounds(u-delta)
        for key in ("A", "B", "D"):
            assert (plus[key]-minus[key])/(2e-6) == pytest.approx(data["d"+key][i], abs=4e-9)


def test_cusp_is_not_replaced_by_a_smooth_length_endpoint(report):
    assert report["cusp"]["left_derivative"] == pytest.approx(-1, abs=1e-9)
    assert report["cusp"]["right_derivative"] == pytest.approx(1, abs=1e-9)
    assert abs(report["cusp"]["squared_endpoint_central_derivative"]) < 1e-10
    u = transport.reference_coordinates()
    bounds = transport.squared_bounds(u)
    y = math.sqrt(bounds["A"]+.43*bounds["D"])
    with pytest.raises(ValueError, match="아래끝"):
        transport.fields(u, y, "length")
    assert np.all(np.isfinite(transport.fields(u, y)["horizontal"]))


def test_all_old_momenta_and_new_mode_intertwine_on_actual_waves(report):
    assert report["chart_rank"] == 14
    assert {(item["coordinate"], item["excited"]) for item in report["cases"]} == {
        ("squared", False), ("squared", True), ("length", True)}
    for item in report["cases"]:
        assert item["old_momentum_residual"] < 2e-8
        assert item["new_momentum_residual"] < 2e-8
        assert item["old_vector_commutator"] < 1e-7
        assert item["cross_vector_commutator"] < 2e-13


def test_half_divergence_is_required_for_the_actual_wave(report):
    for item in report["cases"]:
        assert item["omitted_half_divergence_old_error"] > .2
        if item["coordinate"] == "squared":
            assert item["omitted_half_divergence_new_error"] > .3


def test_curvature_is_not_a_dirac_observable_of_supplied_new_constraint(report):
    for item in report["cases"]:
        assert abs(item["curvature_constraint_commutator_coefficient"]) > 1
        if item["coordinate"] == "squared":
            expected = -1/math.sqrt(item["t"]*(1-item["t"]))
            assert item["curvature_constraint_commutator_coefficient"] == pytest.approx(expected, abs=2e-8)


def test_preparation_choice_and_passive_coordinate_change_are_distinct(report):
    data = report["preparation_comparison"]
    assert data["length_norm"] == pytest.approx(1, abs=1e-12)
    assert data["squared_norm"] == pytest.approx(1, abs=1e-12)
    assert data["length_curvature_mean"] == pytest.approx(2-math.pi/2, abs=2e-11)
    assert data["squared_curvature_mean"] == pytest.approx(0, abs=2e-11)
    assert data["overlap"] == pytest.approx(2*math.sqrt(2)/3, abs=2e-11)
    assert data["overlap"]**2 == pytest.approx(8/9, abs=2e-11)
    assert data["passive_norm"] == pytest.approx(1, abs=2e-12)
    assert data["passive_curvature_residual"] < 2e-12
    assert report["quadrature_convergence"] < 2e-11


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_boundary_or_nonfinite_new_lengths_are_rejected(value):
    with pytest.raises(ValueError):
        transport.fields(transport.reference_coordinates(), value)


def test_invalid_chart_and_difference_step_are_rejected():
    for bad in (np.zeros(13), np.full(14, np.nan), np.full(14, 1000)):
        with pytest.raises(ValueError):
            transport.chart(bad)
    with pytest.raises(ValueError):
        transport.transport_check(transport.reference_coordinates(), step=0)


def test_saved_artifact_matches_execution_and_preserves_physical_ceiling(report):
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, digest in saved["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
    for key in report["preparation_comparison"]:
        assert saved["preparation_comparison"][key] == pytest.approx(report["preparation_comparison"][key], abs=2e-12)
    for old, current in zip(saved["cases"], report["cases"]):
        assert old["old_momentum_residual"] == pytest.approx(current["old_momentum_residual"], abs=1e-12)
    assert "du dy" in " ".join(saved["assumptions"])
    assert "인위적으로" in " ".join(saved["assumptions"])
    assert "디랙 관측량이 아니다" in " ".join(saved["negative_controls"])
    assert "전체 중력 제약" in " ".join(saved["unfinished"])
