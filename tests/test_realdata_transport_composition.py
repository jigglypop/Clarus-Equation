from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.realdata_transport_composition import (
    E17Block,
    TransportConfig,
    evaluate_transport_block,
    phase_states,
)


def _trials_from_states(
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: TransportConfig,
) -> np.ndarray:
    count, dimension = states[0].shape
    trials = np.zeros((count, 180, dimension), dtype=float)
    time = np.linspace(config.trial_start_seconds, config.trial_stop_seconds, 180)
    half = 0.5 * config.phase_window_seconds
    for center, state in zip(config.phase_centers_seconds, states, strict=True):
        mask = (time >= center - half) & (time <= center + half)
        trials[:, mask, :] = state[:, None, :]
    return trials


def test_phase_states_requires_equal_lag_and_uses_whole_windows() -> None:
    bad = TransportConfig(phase_centers_seconds=(-1.5, -1.0, -0.3))
    with pytest.raises(ValueError, match="equally spaced"):
        bad.validate()

    config = TransportConfig()
    rng = np.random.default_rng(7)
    states = tuple(rng.normal(size=(20, 4)) for _ in range(3))
    recovered = phase_states(_trials_from_states(states, config), config)
    for actual, expected in zip(recovered, states, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_exact_linear_semigroup_survives_heldout_trials_and_controls() -> None:
    config = TransportConfig(latent_rank=5, ridge=1e-6, folds=5)
    rng = np.random.default_rng(19)
    x0 = rng.normal(size=(80, 5))
    raw = rng.normal(size=(5, 5))
    q, _ = np.linalg.qr(raw)
    coefficient = 0.72 * q
    intercept = np.linspace(-0.1, 0.1, 5)
    x1 = x0 @ coefficient + intercept
    x2 = x1 @ coefficient + intercept
    block = E17Block(
        session_id="synthetic",
        animal="test",
        condition="exact",
        source_path="synthetic",
        source_sha256="0" * 64,
        trials=_trials_from_states((x0, x1, x2), config),
    )
    result = evaluate_transport_block(block, config=config)
    assert result["near_direct"]
    assert result["beats_persistence"]
    assert result["beats_deranged"]
    assert result["core_consistent"]
    assert result["g_composition_excess_over_direct"] <= 1e-5
    assert result["composition_advantage_over_permuted_interface"] > 0.0


def test_independent_phases_do_not_fake_predictive_composition() -> None:
    config = TransportConfig(latent_rank=4, ridge=1.0, folds=5)
    rng = np.random.default_rng(23)
    states = tuple(rng.normal(size=(100, 4)) for _ in range(3))
    block = E17Block(
        session_id="synthetic-null",
        animal="test",
        condition="null",
        source_path="synthetic",
        source_sha256="1" * 64,
        trials=_trials_from_states(states, config),
    )
    result = evaluate_transport_block(block, config=config)
    assert not result["beats_mean"]
    assert not result["core_consistent"]
