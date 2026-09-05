"""두 입자 계산을 독립 대칭 텐서 기저와 준비 상태의 정확한 값으로 검산한다."""

import importlib.util
from itertools import combinations_with_replacement
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("nonlinear_bath_under_test", HERE / "nonlinear_bath.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original_path


def _independent_two_particle(q, u, sites=5):
    h = 10 * np.eye(sites) + np.diag(np.ones(sites-1), 1) + np.diag(np.ones(sites-1), -1)
    h[0, 1] = h[1, 0] = math.sqrt(q)
    states = list(combinations_with_replacement(range(sites), 2))
    embedding = np.zeros((sites*sites, len(states)))
    for col, (i, j) in enumerate(states):
        if i == j:
            embedding[i*sites+j, col] = 1
        else:
            embedding[i*sites+j, col] = embedding[j*sites+i, col] = 1 / math.sqrt(2)
    h2 = np.kron(h, np.eye(sites)) + np.kron(np.eye(sites), h)
    h2[0, 0] += u
    return embedding.T @ h2 @ embedding


@pytest.mark.parametrize("q,u", [(3., 0.), (3., .1), (4., 1.), (1., 8.)])
def test_resolvent_matches_independent_two_boson_matrix(q, u):
    matrix = _independent_two_particle(q, u)
    assert matrix[0, 0] == pytest.approx(20 + u)
    assert matrix[1, 0] == pytest.approx(math.sqrt(2*q))
    energies, vectors = np.linalg.eigh(matrix)
    result = module.finite_pair(q, u, sites=5)
    assert result["energy_over_g"] == pytest.approx(energies[-1], abs=1e-11)
    assert result["local_pair_weight"] == pytest.approx(vectors[0, -1]**2, abs=1e-11)


def test_squeezed_source_includes_interaction_preparation_energy():
    source = module.source_budget(3, 8.)
    assert source["mean_number"] == pytest.approx(1 / 3)
    assert source["two_particle_probability"] == pytest.approx(math.sqrt(3) / 16)
    assert source["factorial_second_moment"] == pytest.approx(2 / 3)
    assert source["initial_energy_over_g"] == pytest.approx(6)
    assert module.source_budget(3, 0.)["initial_energy_over_g"] == pytest.approx(10 / 3)


def test_raw_coupling_bound_survives_but_is_not_an_exact_occupation():
    expected = math.sqrt(3) / 2048
    for u in (0., .1, 1., 8.):
        bound = module.infinite_retention_bound(3, 3, u)
        assert bound["time_mean_liminf_number_lower_bound"] == pytest.approx(expected)
        assert module.finite_pair(3, u)["local_pair_weight"] >= 1 / 16 - 1e-12


def test_normalized_linear_coupling_can_bind_an_interacting_pair():
    bound = module.infinite_retention_bound(3, 1, 8.)
    assert bound["local_pair_weight_lower_bound"] == .5
    assert bound["time_mean_liminf_number_lower_bound"] == pytest.approx(math.sqrt(3) / 32)
    assert module.finite_pair(1, 8.)["energy_over_g"] > 28
    for u in (0., 1., 4.):
        undecided = module.infinite_retention_bound(3, 1, u)
        assert not undecided["positive_bound_established"]
        assert undecided["time_mean_liminf_number_lower_bound"] is None


@pytest.mark.parametrize("q,u,sites", [
    (0., 1., 5), (float("nan"), 1., 5), (100., 1., 5),
    (1., -1., 5), (1., float("inf"), 5), (1., 1., True), (1., 1., 257),
])
def test_invalid_or_unstable_inputs_are_rejected(q, u, sites):
    with pytest.raises(ValueError):
        module.finite_pair(q, u, sites)


def test_output_preserves_conditional_scope():
    result = module.run()
    assert len(result["cases"]) == 9
    for flag in ("finite_chain_proves_infinite_time_limit", "full_time_mean_existence_asserted",
                 "bound_is_predicted_retained_number", "initial_energy_fixed_while_varying_u",
                 "all_nonlinear_environments_excluded", "autonomous_source_action_derived_from_CE",
                 "common_metric_selection_proved"):
        assert result[flag] is False
