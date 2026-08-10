from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus.nonlinear_object_world import (
    LocalChartModel,
    ObjectWorldConfig,
    canonical_observation,
    generate_object_episode,
    physics_step,
    rollout,
    run_object_permanence_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "nonlinear_object_permanence_v1.json"


def test_episode_is_seed_deterministic_and_first_frame_is_visible() -> None:
    first = generate_object_episode(17, objects=3)
    second = generate_object_episode(17, objects=3)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.actions, second.actions)
    assert np.array_equal(first.visibility, second.visibility)
    assert np.array_equal(first.observations, second.observations, equal_nan=True)
    _, visible = canonical_observation(first.observations[0], 3)
    assert np.all(visible)


def test_observation_hides_state_and_shuffles_slots() -> None:
    episode = generate_object_episode(23, objects=3, occlusion=(16, 20))
    hidden = ~episode.visibility
    assert np.any(hidden)
    assert np.isnan(episode.observations[..., :7]).any()
    assert not np.array_equal(
        episode.observations[0, :, 0], episode.observations[1, :, 0]
    )


def test_physics_rejects_bad_shapes_and_stays_finite() -> None:
    episode = generate_object_episode(31, objects=2)
    next_state, _ = physics_step(episode.states[0], np.zeros(2), ObjectWorldConfig())
    assert np.all(np.isfinite(next_state))
    assert np.all(np.abs(next_state[:, :2]) <= 1.0)
    with pytest.raises(ValueError, match="state"):
        physics_step(np.zeros((2, 4)), np.zeros(2), ObjectWorldConfig())


def test_local_chart_learns_positive_force_coefficients() -> None:
    config = ObjectWorldConfig()
    episodes = [generate_object_episode(seed, objects=2 + seed % 2) for seed in range(10, 16)]
    model = LocalChartModel(config)
    model.fit(episodes)
    assert model.nonlinear_strength > 0.0
    assert model.drag > 0.0
    assert model.action_gain > 0.0
    predicted = rollout(model, episodes[0].states[0], episodes[0].actions[:5])
    assert predicted.shape == (6, len(episodes[0].states[0]), 7)
    assert np.all(np.isfinite(predicted))


def test_end_to_end_gate_records_zero_download_policy() -> None:
    report = run_object_permanence_gate(CONFIG)
    assert report["resource_policy"]["external_download_bytes"] == 0
    assert report["resource_policy"]["trajectory_files_written"] == 0
    assert len(report["config_sha256"]) == 64
    assert report["g2_passed"]
    assert report["g3_passed"]
