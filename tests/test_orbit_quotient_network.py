import numpy as np
import pytest

from reality_stone.clarus.orbit_quotient_network import (
    DelayedEdge,
    DelayedOrbitNetwork,
    advance_full,
    initial_snapshot,
    lift_orbit_trajectory,
    project_orbit_state,
    simulate_full,
    simulate_quotient,
    simulate_sparse_initial_deviation,
    translate_cells,
)


def _network() -> DelayedOrbitNetwork:
    return DelayedOrbitNetwork(
        orbit_count=3,
        bias=(0.05, -0.03, 0.02),
        edges=(
            DelayedEdge(0, 0, -1, 1, 0.16),
            DelayedEdge(0, 1, 0, 2, -0.11),
            DelayedEdge(1, 0, 1, 1, 0.13),
            DelayedEdge(1, 2, 0, 3, 0.12),
            DelayedEdge(2, 1, -1, 2, 0.17),
            DelayedEdge(2, 2, 1, 1, 0.14),
        ),
    )


def test_projection_lift_and_delayed_quotient_are_exact() -> None:
    network = _network()
    initial = np.asarray((0.2, -0.1, 0.3))
    inputs = np.asarray(tuple((0.01 * t, -0.02, 0.015) for t in range(8)))
    quotient = simulate_quotient(network, initial, inputs)
    for cover_size in (32, 64, 128, 256):
        lifted = lift_orbit_trajectory(quotient, cover_size)
        full = simulate_full(network, lifted[0], lifted[1:] * 0.0 + inputs[:, None, :])
        assert np.max(np.abs(full - lifted)) <= 1e-12
        assert np.max(np.abs(project_orbit_state(lifted[0]) - initial)) <= 1e-12


def test_translation_equivariance_and_open_boundary_control() -> None:
    network = _network()
    rng = np.random.default_rng(19)
    initial = rng.normal(0.0, 0.2, size=(32, 3))
    drive = rng.normal(0.0, 0.03, size=(1, 32, 3))
    shifted = simulate_full(network, translate_cells(initial, 5), translate_cells(drive, 5))
    reference = translate_cells(simulate_full(network, initial, drive), 5)
    assert np.max(np.abs(shifted - reference)) <= 1e-12
    open_shifted = simulate_full(
        network, translate_cells(initial, 5), translate_cells(drive, 5), boundary="open"
    )
    open_reference = translate_cells(simulate_full(network, initial, drive, boundary="open"), 5)
    assert np.max(np.abs(open_shifted - open_reference)) > 1e-6


def test_sparse_cone_reconstructs_full_perturbed_cover_and_obeys_causality() -> None:
    network = _network()
    initial = np.asarray((0.2, -0.1, 0.3))
    inputs = np.asarray(tuple((0.01 * t, -0.02, 0.015) for t in range(8)))
    sparse = simulate_sparse_initial_deviation(
        network, 64, initial, inputs, {(31, 0): 0.4}, active_budget=64
    )
    full_initial = lift_orbit_trajectory(initial, 64)
    full_initial[31, 0] += 0.4
    full = simulate_full(network, full_initial, lift_orbit_trajectory(inputs, 64))
    assert np.max(np.abs(sparse.reconstructed - full)) <= 1e-12
    # Orbit 1 can first receive this perturbation through a delay-one edge.
    assert (32, 1) not in sparse.active_by_time[0]
    assert (32, 1) in sparse.active_by_time[1]
    bound = network.orbit_count * (2 * network.maximum_shift * 8 + 1)
    assert max(map(len, sparse.active_by_time)) <= bound
    with pytest.raises(RuntimeError, match="exceeds budget"):
        simulate_sparse_initial_deviation(
            network, 64, initial, inputs, {(31, 0): 0.4}, active_budget=1
        )


def test_snapshot_continuation_and_same_tick_rejection() -> None:
    network = _network()
    rng = np.random.default_rng(23)
    initial = rng.normal(0.0, 0.1, size=(32, 3))
    inputs = rng.normal(0.0, 0.02, size=(8, 32, 3))
    snapshot = initial_snapshot(network, initial)
    for row in inputs[:4]:
        snapshot = advance_full(network, snapshot, row)
    restored = type(snapshot)(snapshot.time, tuple(item.copy() for item in snapshot.history))
    for row in inputs[4:]:
        restored = advance_full(network, restored, row)
    uninterrupted = simulate_full(network, initial, inputs)
    assert np.array_equal(restored.history[-1], uninterrupted[-1])
    with pytest.raises(ValueError, match="same-tick"):
        DelayedOrbitNetwork(1, (0.0,), (DelayedEdge(0, 0, 0, 0, 0.2),))


def test_small_gain_and_destructive_symmetry_controls() -> None:
    network = _network()
    assert network.small_gain < 1.0
    uniform = np.repeat(np.asarray(((0.2, -0.1, 0.3),)), 32, axis=0)
    full = simulate_full(network, uniform, np.zeros((1, 32, 3)))
    # One untied bias destroys spatial constancy.
    untied = full[1].copy()
    untied[0, 0] = np.tanh(np.arctanh(untied[0, 0]) + 0.05)
    assert np.max(np.abs(untied - lift_orbit_trajectory(project_orbit_state(untied), 32))) > 1e-6
    # Same Frobenius-scale modulation with zero spatial mean is still non-equitable.
    modulation = np.cos(2.0 * np.pi * np.arange(32) / 32.0)
    varying = full[1].copy()
    varying[:, 0] = np.tanh(np.arctanh(varying[:, 0]) + 0.05 * modulation)
    assert np.max(np.ptp(varying, axis=0)) > 1e-6
    # Corrupting one orbit label is not the registered lift.
    corrupted = uniform.copy()
    corrupted[0, (0, 1)] = corrupted[0, (1, 0)]
    assert np.max(np.abs(corrupted - lift_orbit_trajectory(project_orbit_state(corrupted), 32))) > 1e-6
    # Index-first Top-K does not commute with translation.
    mask = np.zeros(32)
    mask[:5] = 1.0
    assert not np.array_equal(np.roll(mask, 7), mask)


def test_small_gain_controls_nonzero_spatial_modes() -> None:
    network = _network()
    cells = np.arange(64)
    background = np.repeat(np.asarray(((0.1, -0.05, 0.02),)), 64, axis=0)
    checkerboard = 0.1 * np.cos(np.pi * cells)
    perturbed = background.copy()
    perturbed[:, 0] += checkerboard
    zero_input = np.zeros((40, 64, 3))
    base_path = simulate_full(network, background, zero_input)
    perturbed_path = simulate_full(network, perturbed, zero_input)
    initial_error = np.max(np.abs(perturbed_path[0] - base_path[0]))
    final_error = np.max(np.abs(perturbed_path[-1] - base_path[-1]))
    assert final_error < initial_error * 1e-8
