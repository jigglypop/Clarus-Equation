from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus.active_perception import (
    covariance_observe,
    covariance_predict,
    information_gain,
    run_active_perception_gate,
)
from reality_stone.clarus.nonlinear_object_world import generate_object_episode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "active_perception_v1.json"
CONFIG_V3 = ROOT / "experiments" / "preregistration" / "active_perception_v3.json"


def test_noisy_episode_remains_seed_deterministic() -> None:
    first = generate_object_episode(89, objects=4, velocity_process_noise_std=0.008)
    second = generate_object_episode(89, objects=4, velocity_process_noise_std=0.008)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.observations, second.observations, equal_nan=True)


def test_covariance_and_information_gain_equations() -> None:
    covariance = np.array([0.1, 0.4])
    predicted = covariance_predict(covariance, 0.02)
    observed = covariance_observe(predicted, 0.01)
    gain = information_gain(predicted, 0.01)
    assert np.all(predicted > covariance)
    assert np.all(observed < predicted)
    assert gain[1] > gain[0] > 0.0
    with pytest.raises(ValueError, match="positive"):
        covariance_observe(covariance, 0.0)


def test_validation_loop_respects_resource_ceiling() -> None:
    report = run_active_perception_gate(CONFIG, split="validation")
    assert report["resource_usage"]["external_download_bytes"] == 0
    assert report["resource_usage"]["trajectory_files_written"] == 0
    assert report["resource_usage"]["evaluation_episodes"] == 5
    assert report["resource_passed"]


def test_locked_v3_gate_passes_without_external_data() -> None:
    report = run_active_perception_gate(CONFIG_V3, split="test")
    assert report["passed"]
    assert report["paired_improvement"]["ci95_lower"] > 0.0
    assert report["policies"]["max_information_gain"]["mean_query_rate"] <= 0.25
    assert report["resource_usage"]["external_download_bytes"] == 0
