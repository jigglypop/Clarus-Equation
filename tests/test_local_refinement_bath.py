"""실제 단체 분할, 양의 국소 작용, 공분산 전달과 감쇠 경계를 독립 검산한다."""

import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("local_refinement_bath_under_test", HERE / "local_refinement_bath.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(HERE / "F-01"))
    reference_spec = importlib.util.spec_from_file_location("local_bath_refinement_reference", HERE / "F-01" / "predict_fold_budget.py")
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
finally:
    sys.path[:] = original_path


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_leaf_order_matches_existing_geometric_refinement(depth):
    coordinates = np.vstack((np.zeros(4), np.eye(4)))
    points = dict(enumerate(coordinates))
    expected = [tuple(range(5))]
    for _ in range(depth):
        expected = reference.refine(expected, points)
    cells, words = module.refined_cells(depth)
    assert cells == expected
    assert len(set(words)) == 5**depth
    assert words == sorted(words)


def test_facet_adjacency_matches_independent_pairwise_intersection():
    graph = module.dual_graph(2)
    expected = [
        (i, j)
        for i, j in combinations(range(25), 2)
        if len(set(graph["cells"][i]) & set(graph["cells"][j])) == 4
    ]
    assert graph["edges"] == expected
    assert len(expected) == 60
    assert graph["boundary_facets"] == 5
    assert np.all(np.count_nonzero(graph["incidence"], axis=1) == 2)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_cut_rayleigh_bound_and_source_occupation(depth):
    graph, values, _ = module.spectrum(depth)
    count = len(values)
    indicator = np.zeros(count)
    indicator[:count // 5] = 1.0
    indicator -= indicator.mean()
    rayleigh = indicator @ graph["laplacian"] @ indicator / (indicator @ indicator)
    assert rayleigh == pytest.approx(25 / count, rel=1e-12)
    assert values[1] <= rayleigh + 1e-12
    assert values[-1] / values[1] >= (count - 1) / 5 - 1e-12
    covariance, number = module.source_state(depth)
    q, p = covariance[0::2, 0::2], covariance[1::2, 1::2]
    np.testing.assert_allclose(q @ p, np.eye(count) / 4, atol=1e-11)
    assert np.linalg.eigvalsh(number).min() >= 9 / 40 - 1e-11


def test_edge_hamiltonian_matches_independent_positive_square():
    epsilon, kappa = 2.0, 0.7
    graph, energies, weights, h = module.finite_edge_hamiltonian(
        epsilon=epsilon, kappa=kappa, nodes=8
    )
    count = len(graph["cells"])
    rng = np.random.default_rng(20260905)
    state = rng.normal(size=len(h)) + 1j * rng.normal(size=len(h))
    q = state[:count]
    bath = state[count:].reshape(len(graph["edges"]), len(energies))
    square = np.sqrt(energies)[None, :] * bath
    square += math.sqrt(kappa) * (graph["incidence"] @ q)[:, None] * np.sqrt(weights)[None, :]
    expected = epsilon * np.vdot(q, q).real + np.vdot(square, square).real
    assert np.vdot(state, h @ state).real == pytest.approx(expected, rel=1e-13)
    coupling = h[count:, :count]
    schur = h[:count, :count] - coupling.T @ (
        coupling / np.tile(energies, len(graph["edges"]))[:, None]
    )
    np.testing.assert_allclose(schur, epsilon * np.eye(count), atol=1e-13)


@pytest.mark.parametrize("time", [0.0, 0.7, 3.0])
def test_modal_channel_matches_full_unitary_covariance(time):
    result = module.finite_channel_check(time=time, nodes=8)
    assert result["hamiltonian_minimum_eigenvalue"] > 0
    assert result["amplitude_decomposition_residual"] < 1e-12
    assert result["noise_decomposition_residual"] < 1e-12
    assert result["channel_cp_minimum_eigenvalue"] > -1e-12
    common = np.ones(5) / math.sqrt(5)
    np.testing.assert_allclose(
        result["system_amplitude"] @ common,
        np.exp(-2j * time) * common, atol=1e-12,
    )
    _, _, _, h = module.finite_edge_hamiltonian(nodes=8)
    energy, vectors = np.linalg.eigh(h)
    u = (vectors * np.exp(-1j * energy * time)) @ vectors.T
    system_covariance, _ = module.source_state(1)
    full_covariance = np.eye(2 * len(h)) / 2
    full_covariance[:10, :10] = system_covariance
    # 이 독립 조립은 q 전체, p 전체 순서를 만든 다음 교차 순서로 재배열한다.
    real_map = np.block([[u.real, -u.imag], [u.imag, u.real]])
    order = np.ravel(np.column_stack((np.arange(len(h)), np.arange(len(h)) + len(h))))
    real_map = real_map[np.ix_(order, order)]
    direct = (real_map @ full_covariance @ real_map.T)[:10, :10]
    reduced = result["x"] @ system_covariance @ result["x"].T + result["y"]
    np.testing.assert_allclose(reduced, direct, atol=1e-11)


def test_all_scalar_coupling_floor_against_independent_finite_stars():
    nodes, weights = np.polynomial.laguerre.laggauss(32)
    for ratio in (1e2, 1e4):
        floor = module.uniform_survival_floor(ratio)["all_scalar_kappa_worst_mode_floor"]
        for coupling in (1e-5, 1e-3, 0.1, 1.0, 100.0):
            survival = []
            for strength in (coupling, coupling * ratio):
                h = np.diag(np.r_[2 + strength, nodes])
                h[0, 1:] = h[1:, 0] = np.sqrt(strength * nodes * weights)
                energy, basis = np.linalg.eigh(h)
                survival.append(abs(np.sum(basis[0]**2 * np.exp(-1j * energy)))**2)
            assert max(survival) >= floor - 1e-9
    assert module.refinement_floor(20)["all_scalar_kappa_worst_mode_floor"] > 0.99999


def test_fixed_graph_long_time_uses_continuum_and_matches_covariance_number():
    result = module.continuum_case(3, 200.0)
    assert 0 < result["remaining_contrast_number"] < 1e-6
    assert result["trace_distance_upper_using_numeric_number"] < 0.002
    assert result["covariance_number_residual"] < 1e-9
    assert result["slowest_graph_mode_initial_number"] > 1.0
    assert not result["quadrature_error_is_rigorous_bound"]


def test_small_positive_coupling_is_not_rounded_to_zero(monkeypatch):
    seen = []
    def response(epsilon, strength, time):
        seen.append(strength)
        return {
            "amplitude": [1.0, 0.0], "survival_probability": 1.0,
            "quadrature_error_estimate": 0.0,
        }
    monkeypatch.setattr(module, "_continuum_response", response)
    module.continuum_case(1, 0.0, kappa=1e-14)
    assert all(value > 0 for value in seen)
    assert seen == pytest.approx([5e-14] * 4, rel=1e-11, abs=0)


@pytest.mark.parametrize("depth", [0, True, 1.5, 5])
def test_graph_rejects_unsupported_depth_before_allocation(depth):
    with pytest.raises(ValueError):
        module.dual_graph(depth)


def test_recorded_artifact_matches_current_dependencies_and_scope():
    report = json.loads((HERE / "local_refinement_bath.json").read_text(encoding="utf-8"))
    for relative, expected in report["source_sha256"].items():
        assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected
    result = report["conditional_results"]
    assert result["positive_time_independent_edge_hamiltonian_constructed"]
    assert result["one_scalar_kappa_uniform_fixed_time_attenuation_excluded"]
    assert not result["common_metric_tensor_selected"]
    assert not result["continuum_general_relativity_derived"]


@pytest.mark.parametrize("phase", [-1.0, 0.0, 1.0])
def test_local_phase_preserves_position_and_common_state_but_changes_energy(phase):
    result = module.phase_preparation_case(depth=2, phase=phase)
    for key in (
        "initial_position_covariance_residual", "common_marginal_covariance_residual",
        "canonical_residual", "purity_residual", "mode_number_gain_formula_residual",
        "energy_gain_formula_residual",
    ):
        assert result[key] < 1e-10
    if phase:
        assert result["phase_preparation_energy_gain"] > 0
        assert result["additional_emitted_bath_energy"] > 0
        assert result["phase_prepared_remaining_contrast_number"] > result["reference_remaining_contrast_number"]
    else:
        assert result["phase_preparation_energy_gain"] == 0


def test_phase_covariance_matches_independent_wavefunction_derivatives():
    initial, _ = module.source_state(1)
    phase = 0.7
    graph = module.dual_graph(1)
    q = initial[0::2, 0::2]
    laplacian = graph["laplacian"]
    inverse = np.linalg.inv(q)
    # log ψ=-q^T Q^-1 q/4+i b q^T Lq/2의 p 작용을 직접 적분한 이차 모멘트.
    derivative = inverse / 2 - 1j * phase * laplacian
    momentum = (derivative.conj() @ q @ derivative.T).real
    mixed = (1j * q @ derivative.T).real
    prepared, _ = module.phase_prepared_state(1, phase)
    np.testing.assert_allclose(prepared[1::2, 1::2], momentum, atol=1e-12)
    np.testing.assert_allclose(prepared[0::2, 1::2], mixed, atol=1e-12)


def test_minimum_energy_condition_selects_zero_phase_in_the_local_family():
    zero = module.phase_preparation_case(1, 0.0)
    positive = module.phase_preparation_case(1, 0.5)
    negative = module.phase_preparation_case(1, -0.5)
    doubled = module.phase_preparation_case(1, 1.0)
    assert positive["phase_prepared_initial_energy"] > zero["phase_prepared_initial_energy"]
    assert positive["phase_prepared_initial_energy"] == pytest.approx(negative["phase_prepared_initial_energy"])
    assert doubled["phase_preparation_energy_gain"] == pytest.approx(4 * positive["phase_preparation_energy_gain"])
