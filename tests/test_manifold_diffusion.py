from __future__ import annotations

import numpy as np

from reality_stone.clarus.cloudcell_dynamics import NeuralRecording
from reality_stone.clarus.manifold_diffusion import evaluate_diffusion_recording


def _nonlinear_manifold(
    seed: int = 5,
    *,
    n_units: int = 16,
    n_time: int = 1200,
) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    phase = np.zeros(n_time, dtype=float)
    for index in range(1, n_time):
        phase[index] = (
            phase[index - 1]
            + 0.075
            + 0.055 * np.sin(phase[index - 1])
            + 0.018 * np.sin(3.0 * phase[index - 1])
        )
    activity = np.empty((n_units, n_time), dtype=float)
    offsets = np.linspace(0.0, 2.0 * np.pi, n_units, endpoint=False)
    for unit, offset in enumerate(offsets):
        harmonic = 1 + unit % 4
        activity[unit] = np.sin(harmonic * phase + offset)
        activity[unit] += 0.25 * np.cos((harmonic + 1) * phase - offset)
    activity += rng.normal(scale=0.035, size=activity.shape)
    return NeuralRecording(
        f"nonlinear-manifold-{seed}",
        np.arange(n_time, dtype=float),
        activity,
    )


def _white_noise(
    seed: int = 17,
    *,
    n_units: int = 16,
    n_time: int = 1000,
) -> NeuralRecording:
    rng = np.random.default_rng(seed)
    return NeuralRecording(
        f"white-noise-{seed}",
        np.arange(n_time, dtype=float),
        rng.normal(size=(n_units, n_time)),
    )


def test_diffusion_analog_beats_linear_map_on_nonlinear_manifold() -> None:
    gate = evaluate_diffusion_recording(
        _nonlinear_manifold(),
        horizon_steps=12,
        n_components=8,
        delay_count=3,
        delay_stride=2,
        n_shifts=19,
        min_diffusion_delta=0.01,
        min_positive_fraction=0.6,
    )

    assert gate.r2_diffusion > 0.80
    assert gate.delta_diffusion_over_best_baseline > 0.01
    assert gate.positive_unit_fraction_over_linear >= 0.6
    assert gate.shift_p_value <= 0.05
    assert gate.passed


def test_white_noise_does_not_pass_diffusion_gate() -> None:
    gate = evaluate_diffusion_recording(
        _white_noise(),
        horizon_steps=6,
        n_components=6,
        delay_count=2,
        delay_stride=2,
        n_shifts=19,
    )

    assert not gate.passed
    assert gate.r2_diffusion < 0.1


def test_increment_forecast_preserves_current_state_and_transports_displacement() -> None:
    gate = evaluate_diffusion_recording(
        _nonlinear_manifold(seed=19),
        horizon_steps=12,
        forecast_target="increment",
        n_shifts=19,
        min_diffusion_delta=0.01,
        min_positive_fraction=0.6,
    )

    assert gate.forecast_target == "increment"
    assert gate.delta_diffusion_over_best_baseline > 0.01
    assert gate.passed


def test_latent_output_tests_population_state_instead_of_noisy_unit_rows() -> None:
    gate = evaluate_diffusion_recording(
        _nonlinear_manifold(seed=21),
        horizon_steps=12,
        forecast_target="increment",
        output_space="latent",
        n_shifts=19,
        min_diffusion_delta=0.01,
        min_positive_fraction=0.6,
    )

    assert gate.output_space == "latent"
    assert gate.n_output_dimensions == gate.n_components
    assert gate.n_observed_units > gate.n_output_dimensions
    assert gate.passed


def test_test_block_cannot_change_diffusion_state_or_selection() -> None:
    recording = _nonlinear_manifold(seed=23)
    changed = recording.activity.copy()
    changed[:, int(0.81 * changed.shape[1]) :] += 100.0
    altered = NeuralRecording(recording.recording_id, recording.time, changed)

    original = evaluate_diffusion_recording(
        recording,
        horizon_steps=10,
        n_shifts=3,
    )
    test_changed = evaluate_diffusion_recording(
        altered,
        horizon_steps=10,
        n_shifts=3,
    )

    assert original.selected_neighbors == test_changed.selected_neighbors
    assert original.selected_ridge == test_changed.selected_ridge
    assert original.state_model_sha256 == test_changed.state_model_sha256
