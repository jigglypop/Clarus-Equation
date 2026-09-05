"""공통 모드 보존과 환경 결합상태를 독립 선형대수·조합 수치로 검산한다."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020" / "interface_bath.py"
spec = importlib.util.spec_from_file_location("interface_bath_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_differences_preserve_common_mode_without_setting_differences_to_zero():
    incidence = module.pair_incidence(3)
    np.testing.assert_array_equal(incidence @ np.ones(3), np.zeros(3))
    assert np.linalg.norm(incidence @ np.array([1., -1., 0.])) > 0
    np.testing.assert_allclose(np.linalg.eigvalsh(incidence.T @ incidence), [0., 3., 3.], atol=1e-14)


def test_actual_refinement_owner_counts_and_degrees_of_freedom():
    assert module.owner_histogram(1) == {3: 10, 4: 5}
    histogram = module.owner_histogram(2)
    assert histogram == {4: 25, 9: 10, 12: 5}
    assert sum(count * owners for owners, count in histogram.items()) == 250
    assert sum(count * (owners - 1) for owners, count in histogram.items()) == 210


def test_band_edge_threshold_is_not_a_normalizable_bound_state():
    assert module.bound_pair(2)["bound_states"] == 0
    assert module.bound_pair(2.01)["bound_states"] == 2
    energy, _ = module.finite_chain_check(2)
    assert len(energy) == 0


def test_nonzero_long_time_mean_in_two_small_interfaces():
    assert module.bound_pair(3)["long_time_mean_survival"] == pytest.approx(1/8)
    assert module.bound_pair(4)["long_time_mean_survival"] == pytest.approx(2/9)


def test_bound_energies_and_overlap_match_independent_dense_diagonalization():
    result = module.bound_pair(5)
    energy, weights = module.finite_chain_check(5)
    np.testing.assert_allclose(energy, [-2.5, 2.5], atol=1e-13)
    np.testing.assert_allclose(weights, [3/8, 3/8], atol=1e-13)
    assert sum(weights**2) == pytest.approx(result["long_time_mean_survival"])


def test_owner_normalization_removes_collective_enhancement_as_an_extra_choice():
    for owners in (3, 4, 12):
        incidence = module.pair_incidence(owners) / np.sqrt(owners)
        spectrum = np.linalg.eigvalsh(incidence.T @ incidence)
        np.testing.assert_allclose(spectrum[1:], np.ones(owners - 1), atol=1e-14)
        assert module.bound_pair(float(spectrum[-1]))["bound_states"] == 0

@pytest.mark.parametrize("sites", [0, 1, -1, True, 2.5])
def test_finite_chain_rejects_invalid_size(sites):
    with pytest.raises(ValueError, match="sites"):
        module.finite_chain_check(3, sites)


@pytest.mark.parametrize("coupling", [0, -1, float("nan"), float("inf")])
def test_finite_chain_rejects_invalid_coupling(coupling):
    with pytest.raises(ValueError, match="coupling"):
        module.finite_chain_check(coupling)


@pytest.mark.parametrize("depth", [True, 1.0, 0, 3])
def test_owner_histogram_requires_supported_integer_depth(depth):
    with pytest.raises(ValueError, match="depths"):
        module.owner_histogram(depth)

