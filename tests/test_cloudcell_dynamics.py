from __future__ import annotations

import numpy as np

from reality_stone.clarus.cloudcell_dynamics import (
    NeuralRecording,
    evaluate_dynamics_panel,
    evaluate_latent_closure,
    evaluate_recording_dynamics,
)


def _coupled_recording(seed: int = 7, n_units: int = 12, n_time: int = 520) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    latent = np.zeros(n_time)
    innovation = rng.normal(scale=0.7, size=n_time)
    for index in range(1, n_time):
        latent[index] = 0.75 * latent[index - 1] + innovation[index]

    activity = np.zeros((n_units, n_time))
    loadings = np.linspace(0.5, 1.3, n_units)
    noise = rng.normal(scale=0.35, size=(n_units, n_time))
    for index in range(2, n_time):
        activity[:, index] = (
            0.58 * activity[:, index - 1]
            - 0.12 * activity[:, index - 2]
            + loadings * latent[index - 1]
            + noise[:, index]
        )
    return NeuralRecording(f"coupled-{seed}", np.arange(n_time, dtype=float), activity)


def _independent_recording(seed: int = 11, n_units: int = 12, n_time: int = 520) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    activity = np.zeros((n_units, n_time))
    noise = rng.normal(scale=0.5, size=(n_units, n_time))
    for index in range(2, n_time):
        activity[:, index] = (
            0.72 * activity[:, index - 1]
            - 0.08 * activity[:, index - 2]
            + noise[:, index]
        )
    return NeuralRecording(f"independent-{seed}", np.arange(n_time, dtype=float), activity)


def test_coupled_dynamics_require_local_history_and_population_cloud() -> None:
    gate = evaluate_recording_dynamics(
        _coupled_recording(),
        n_components=3,
        max_targets=8,
        min_cloud_delta=0.0001,
        min_positive_fraction=0.5,
    )

    assert len(gate.scores) == 8
    assert gate.median("delta_current_state") > 0.5
    assert gate.median("delta_memory") > 0.0
    assert gate.median("delta_cloud_given_local") > 0.0
    assert gate.median("delta_local_given_cloud") > 0.0
    assert gate.median("delta_time_alignment") > 0.0
    assert gate.passed


def test_independent_units_do_not_pass_population_cloud_gate() -> None:
    gate = evaluate_recording_dynamics(
        _independent_recording(),
        n_components=3,
        max_targets=8,
        min_cloud_delta=0.01,
        min_positive_fraction=0.75,
    )

    assert gate.median("delta_memory") > 0.0
    assert not gate.passed


def test_panel_counts_recordings_not_target_neurons() -> None:
    recordings = [_coupled_recording(seed) for seed in (1, 2, 3)]
    panel = evaluate_dynamics_panel(
        recordings,
        min_recordings_passed=3,
        n_components=3,
        max_targets=6,
        min_cloud_delta=0.0001,
        min_positive_fraction=0.5,
    )

    assert len(panel.recordings) == 3
    assert panel.pass_count == 3
    assert panel.passed
    assert panel.to_dict(include_targets=False)["replicate_unit"] == (
        "independently recorded animal"
    )


def test_gap_crossing_samples_are_excluded() -> None:
    recording = _coupled_recording(n_time=420)
    time = recording.time.copy()
    time[300:] += 50.0
    gate = evaluate_recording_dynamics(
        NeuralRecording(recording.recording_id, time, recording.activity),
        n_components=2,
        max_targets=1,
    )

    score = gate.scores[0]
    assert score.n_test < 420 - int(0.8 * 420)


def test_population_latent_closure_detects_cross_axis_transition() -> None:
    rng = np.random.default_rng(31)
    n_time = 700
    latent = np.zeros((n_time, 2))
    transition = np.asarray([[0.78, 0.42], [-0.36, 0.71]])
    for index in range(1, n_time):
        latent[index] = transition @ latent[index - 1]
        latent[index] += rng.normal(scale=0.22, size=2)
    loadings = rng.normal(size=(18, 2))
    activity = loadings @ latent.T
    activity += rng.normal(scale=0.04, size=activity.shape)
    recording = NeuralRecording("rotating-latent", np.arange(n_time), activity)

    score = evaluate_latent_closure(
        recording,
        n_components=2,
        state_order=1,
        horizon_steps=4,
        min_transition_delta=0.001,
        max_composition_gap=0.15,
    )

    assert score.r2_direct > score.r2_diagonal
    assert score.r2_composed > score.r2_persistence
    assert score.passed


def test_population_latent_closure_accepts_uncoupled_markov_state() -> None:
    rng = np.random.default_rng(43)
    n_time = 700
    latent = np.zeros((n_time, 2))
    for index in range(1, n_time):
        latent[index, 0] = 0.82 * latent[index - 1, 0] + rng.normal(scale=0.2)
        latent[index, 1] = 0.55 * latent[index - 1, 1] + rng.normal(scale=0.2)
    activity = np.vstack(
        [latent[:, 0] + rng.normal(scale=0.03, size=n_time) for _ in range(8)]
        + [latent[:, 1] + rng.normal(scale=0.03, size=n_time) for _ in range(8)]
    )
    recording = NeuralRecording("diagonal-latent", np.arange(n_time), activity)

    score = evaluate_latent_closure(
        recording,
        n_components=2,
        state_order=1,
        horizon_steps=4,
        min_transition_delta=0.01,
    )

    assert score.r2_direct > score.r2_persistence
    assert score.r2_composed > score.r2_persistence
    assert score.passed
