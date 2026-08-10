from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.causal_world_simulator import (
    LinearWorldModel,
    chart_transition,
    cycle_holonomy,
    fit_controlled_linear_model,
    harmonic_anchor_extension,
    holonomy_frustration,
    multistep_error_bound,
    quadratic_cost,
    reconstruct_latent,
    run_synthetic_gate,
)


def test_full_rank_sensor_atlas_reconstructs_latent_exactly() -> None:
    operator = np.array([[1.0, 0.2], [0.1, 1.0], [1.0, -0.4]])
    states = np.array([[0.4, -0.2], [1.1, 0.7], [-0.3, 0.9]])
    observations = states @ operator.T
    reconstructed = reconstruct_latent(operator, observations)
    assert np.allclose(reconstructed, states, atol=1e-12)


def test_rank_deficient_sensor_cannot_identify_null_direction() -> None:
    operator = np.array([[1.0, 0.0]])
    states = np.array([[0.4, 3.0], [0.4, -2.0]])
    reconstructed = reconstruct_latent(operator, states @ operator.T)
    assert np.array_equal(reconstructed[0], reconstructed[1])
    assert not np.allclose(reconstructed, states)


def test_persistent_excitation_recovers_noiseless_controlled_law() -> None:
    transition = np.array([[0.8, 0.1], [-0.2, 0.7]])
    control = np.array([[0.3], [0.15]])
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(200, 1))
    states = np.empty((201, 2))
    states[0] = np.array([0.2, -0.4])
    for index, action in enumerate(actions):
        states[index + 1] = transition @ states[index] + control @ action
    fitted = fit_controlled_linear_model(states, actions)
    assert np.allclose(fitted.transition, transition, atol=1e-12)
    assert np.allclose(fitted.control, control, atol=1e-12)


def test_harmonic_extension_is_unique_energy_minimizer() -> None:
    adjacency = np.zeros((5, 5))
    for index in range(4):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    harmonic = harmonic_anchor_extension(laplacian, (0, 4), np.array([0.0, 1.0]))
    assert np.allclose(harmonic, np.linspace(0.0, 1.0, 5))
    perturbation = harmonic.copy()
    perturbation[2] += 0.1
    assert harmonic @ laplacian @ harmonic < perturbation @ laplacian @ perturbation
    with pytest.raises(ValueError, match="component"):
        harmonic_anchor_extension(np.zeros((2, 2)), (0,), np.array([1.0]))


def test_chart_cocycle_has_identity_holonomy_and_detects_corruption() -> None:
    charts = {
        "a": np.eye(2),
        "b": np.array([[1.0, 0.3], [0.0, 1.0]]),
        "c": np.array([[0.8, 0.0], [0.2, 1.0]]),
    }
    transitions = {
        (source, target): chart_transition(charts[source], charts[target])
        for source, target in (("a", "b"), ("b", "c"), ("c", "a"))
    }
    exact = cycle_holonomy(transitions, ("a", "b", "c", "a"))
    assert holonomy_frustration(exact) < 1e-28
    transitions[("b", "c")][0, 1] += 0.2
    corrupted = cycle_holonomy(transitions, ("a", "b", "c", "a"))
    assert holonomy_frustration(corrupted) > 1e-3


def test_exact_one_step_planner_is_no_worse_than_zero_action() -> None:
    model = LinearWorldModel(
        transition=np.array([[0.9, 0.1], [0.0, 0.8]]),
        control=np.array([[0.4], [0.2]]),
    )
    state = np.array([1.0, -0.3])
    target = np.zeros(2)
    q_matrix = np.eye(2)
    r_matrix = np.array([[0.1]])
    action = model.optimal_one_step_action(state, target, q_matrix, r_matrix)
    planned = model.predict(state, action)
    zero = model.predict(state, np.zeros(1))
    assert quadratic_cost(planned, action, target, q_matrix, r_matrix) <= quadratic_cost(
        zero, np.zeros(1), target, q_matrix, r_matrix
    )


def test_geometric_rollout_bound_and_end_to_end_gate() -> None:
    assert np.isclose(multistep_error_bound(0.5, 1.0, 0.1, 3), 0.3)
    report, data = run_synthetic_gate()
    assert report.passed
    assert report.test_r2_model > report.test_r2_persistence
    assert report.planned_mean_cost < report.zero_action_mean_cost
    assert len(data.states) == len(data.actions) + 1
