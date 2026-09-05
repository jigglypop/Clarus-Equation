"""기존 분할 공분산·양자 물리성·환경 점유식을 독립 경로로 검사한다."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("split_quantum_source_under_test", HERE / "split_quantum_source.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_quantum_source_matches_stored_classical_covariance(k):
    result = module.source_check(k)
    assert max(result["residuals"].values()) < 1e-12
    assert result["position_only_cp_min_eigenvalue"] > -1e-12
    assert result["naive_both_quadrature_cp_min_eigenvalue"] == pytest.approx(-(k - 1) / 2)
    # 임의의 부모 상태에 대한 평균 전달과 가역성도 확인한다.
    dilation = module.source_dilation(k)
    original = np.arange(2 * k, dtype=float) / 3 - 1
    np.testing.assert_allclose(np.linalg.solve(dilation, dilation @ original), original, atol=1e-13)


def test_entangled_reference_exposes_invalid_simultaneous_copy():
    k = 3
    c, s = np.cosh(2.), np.sinh(2.)
    parent = reference = .5 * c * np.eye(2)
    cross = .5 * s * np.diag([1., -1.])
    stored_noise = module.classical_split().split_C([-1, 0, 0, 0])[1:, 1:]
    dilation = module.source_dilation(k)
    valid_x = dilation[:, :2]
    valid_noise = .5 * dilation[:, 2:] @ dilation[:, 2:].T
    minima = []
    for x, noise in (
        (np.tile(np.eye(2), (k, 1)), np.kron(stored_noise, np.eye(2))),
        (valid_x, valid_noise),
    ):
        output = np.block([[x @ parent @ x.T + noise, x @ cross],
                           [cross.T @ x.T, reference]])
        uncertainty = output + .5j * np.kron(np.eye(k + 1), module.J)
        minima.append(np.linalg.eigvalsh(uncertainty)[0])
    assert minima[0] < -.1
    assert minima[1] > -1e-12


@pytest.mark.parametrize("coupling_squared", [1., 3., 4.])
def test_full_covariance_dynamics_matches_number_amplitude_identity(coupling_squared):
    rows = module.bath_number_check(3, coupling_squared, sites=32)
    assert max(row["number_identity_residual"] for row in rows) < 1e-12
    assert max(row["total_number_residual"] for row in rows) < 1e-11
    assert rows[0]["boundary_number_from_covariance"] == pytest.approx(1 / 3)
    assert max(row["q_variance_identity_residual"] for row in rows) < 1e-12
    # 압축 상태는 순간 위치 분산이 진공 분산보다 작을 수 있다.
    assert rows[1]["q_variance_from_covariance"] < .5


def test_populated_mode_retention_and_preparation_energy():
    assert module.source_check(2)["vacuum_input_output_number"] == pytest.approx(11 / 16)
    assert module.source_check(3)["vacuum_input_output_number"] == pytest.approx(1)
    assert module.source_check(4)["vacuum_input_output_number"] == pytest.approx(43 / 32)
    assert module.asymptotic_number(2, 2) == 0
    assert module.asymptotic_number(3, 3) == pytest.approx(1 / 12)
    assert module.asymptotic_number(4, 4) == pytest.approx(25 / 144)
    assert module.asymptotic_number(3, 1) == 0


def test_result_retains_scope_of_supplied_protocol():
    result = module.run()
    assert result["infinite_bath_limit_precedes_time_average"]
    for key in (
        "autonomous_source_action_derived_from_CE",
        "microscopic_coupling_derived_from_CE",
        "common_metric_selection_proved",
        "all_historical_labels_are_simultaneous_canonical_modes",
        "q0016_physical_card_revived",
        "q0017_merge_rule_combined",
        "energy_is_universal_minimum_work",
    ):
        assert result[key] is False


@pytest.mark.parametrize("k", [1, True, 2.5])
def test_non_splits_are_rejected(k):
    with pytest.raises(ValueError, match="children"):
        module.source_dilation(k)


@pytest.mark.parametrize("k,depth,root_variance", [
    (2, 0, .5), (10000, 0, .5), (2, 2, .5), (3, 3, .5), (4, 2, .5), (2, 2, 1.),
])
def test_repeated_actual_split_attains_fixed_covariance_energy_bound(k, depth, root_variance):
    q = module.frontier_position_covariance(k, depth, root_variance)
    q_map, p_map = module.recursive_source_maps(k, depth, root_variance)
    np.testing.assert_allclose(.5 * q_map @ q_map.T, q, atol=1e-12, rtol=0)
    np.testing.assert_allclose(q_map @ p_map.T, np.eye(len(q)), atol=1e-12, rtol=0)
    p = .5 * p_map @ p_map.T
    attained = (np.trace(q) + np.trace(p) - len(q)) / 2
    assert attained == pytest.approx(module.minimum_number_at_position_covariance(q))
    _, _, _, exact = module.frontier_resource_spectrum(k, depth, root_variance)
    assert attained == pytest.approx(float(exact))


def test_hierarchical_traces_and_finite_mean_energy_obstruction():
    from fractions import Fraction

    for depth in (0, 1, 2, 3, 16, 32):
        spectrum, trace_q, trace_inverse, _ = module.frontier_resource_spectrum(2, depth)
        assert sum(count for _, count in spectrum) == 2**depth
        assert sum(value * count for value, count in spectrum) == trace_q
        assert sum(count / value for value, count in spectrum) == trace_inverse
    assert module.frontier_resource_spectrum(2, 2)[3] == Fraction(103, 32)
    assert module.frontier_resource_spectrum(2, 3)[3] == Fraction(663, 64)
    assert module.frontier_resource_spectrum(2, 2)[3] < 10
    assert module.frontier_resource_spectrum(2, 3)[3] > 10


def test_position_momentum_correlation_cannot_lower_fixed_q_energy():
    q = np.array([[1.5, .2], [.2, .8]])
    pure_p = np.linalg.inv(q) / 4
    # p -> p + Bq는 B 대칭일 때 물리적인 정준 변환이다.
    shear = np.array([[.4, .2], [.2, -.3]])
    mixed_p = pure_p + shear @ q @ shear.T
    cross = q @ shear.T
    covariance = np.block([[q, cross], [cross.T, mixed_p]])
    omega = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    assert np.linalg.eigvalsh(covariance + .5j * omega)[0] > -1e-12
    excess = (np.trace(q) + np.trace(mixed_p) - 2) / 2 - module.minimum_number_at_position_covariance(q)
    assert excess == pytest.approx(np.trace(shear @ q @ shear.T) / 2)
    assert excess > 0


@pytest.mark.parametrize("covariance", [
    [[0.]], [[float("nan")]], [[1., .2], [0., 1.]], [1., 2.],
])
def test_invalid_position_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="position covariance"):
        module.minimum_number_at_position_covariance(covariance)
