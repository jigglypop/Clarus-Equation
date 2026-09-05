"""기록과 확률 선택의 구분, 에너지 보존 이완과 환경 재사용을 검산한다."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("selection_mechanisms_under_test", HERE / "selection_mechanisms.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_qnd_record_preserves_populations_for_a_nonuniform_coherent_state():
    amplitudes = np.sqrt([0.2, 0.3, 0.5]) * np.exp(1j * np.array([0., 0.7, -0.2]))
    state = np.outer(amplitudes, amplitudes.conj())
    operators = module.soft_record_operators(np.array([0., 1., 2.]), 0.5)
    result = module.apply_channel(operators, state)
    np.testing.assert_allclose(np.diag(result), np.diag(state), atol=1e-15)
    assert abs(result[0, 2]) < abs(state[0, 2])


def test_conditioning_and_rejection_together_restore_the_original_population():
    state = np.ones((3, 3)) / 3
    operators = module.soft_record_operators([0., 1., 2.], 0.5)
    branches = [operator @ state @ operator.T for operator in operators]
    probabilities = [np.trace(branch) for branch in branches]
    assert sum(probabilities) == pytest.approx(1)
    conditioned = branches[0] / probabilities[0]
    assert conditioned[0, 0] > 0.8
    np.testing.assert_allclose(np.diag(sum(branches)), [1/3] * 3)


def test_relaxation_targets_are_supplied_and_energy_can_be_stored_in_environment():
    result = module.finite_controls()
    assert result["relaxed_mismatch"] == pytest.approx(2/3)
    assert result["relaxation_energy_intertwining_residual"] < 1e-14
    assert result["collision_energy_commutator_residual"] < 1e-14
    assert result["collision_unitarity_residual"] < 1e-14


def test_reusing_one_environment_does_not_iterate_the_reset_channel():
    result = module.finite_controls()
    fresh = result["fresh_environment_mismatch_steps_1_to_4"]
    reused = result["same_environment_mismatch_steps_1_to_4"]
    np.testing.assert_allclose(fresh, [(5/3) * 0.4**n for n in range(1, 5)])
    angle = math.asin(math.sqrt(0.6))
    np.testing.assert_allclose(reused, [(5/3) * math.cos(n * angle)**2 for n in range(1, 5)])
    assert reused[2] > reused[1]


def test_gaussian_selection_has_a_cost_and_extensive_mismatch():
    one = module.gaussian_mismatch(np.array([2.]), 0.5)
    many = module.gaussian_mismatch(np.full(100, 2.), 0.5)
    assert one["posterior_total_squared_mismatch"] == pytest.approx(2/9)
    assert one["log_acceptance"] == pytest.approx(-0.5 * math.log(9))
    assert many["posterior_total_squared_mismatch"] == pytest.approx(100 * 2/9)
    assert many["log_acceptance"] == pytest.approx(-50 * math.log(9))


def test_chain_spectral_formula_matches_independent_dense_hamiltonian():
    sites = 7
    matrix = 3*np.eye(sites) + np.diag(np.ones(sites-1), 1) + np.diag(np.ones(sites-1), -1)
    energy, modes = np.linalg.eigh(matrix)
    for time in (0., 0.3, 2., 10.):
        expected = np.sum(modes[0]**2 * np.exp(-1j * energy * time))
        assert module.chain_boundary_amplitude(sites, time) == pytest.approx(expected, abs=1e-14)


def test_bound_state_survives_an_infinite_environment():
    result = module.autonomous_bath_control()
    assert result["4096_site_probability_difference"] < 1e-12
    negative = result["detuned_negative_control"]
    assert negative["finite_truncation_eigenvector_residual"] < 1e-14
    assert negative["infinite_chain_boundary_survival_limit"] == pytest.approx(9/16)
