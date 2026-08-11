import pytest

from reality_stone.clarus.local_cloud_ood_benchmark import (
    LearnedOODConfig,
    _panel_configs,
    _raw_sequences,
    _train_model,
    evaluate_ood_development,
)
from reality_stone.clarus.local_cloud_benchmark import generate_episodes


def _small():
    return LearnedOODConfig(train_episodes=32, evaluation_episodes=32, epochs=2)


def test_ood_panels_lock_noise_and_horizon_only() -> None:
    panels = _panel_configs(_small())
    assert (panels["id"].episode_steps, panels["id"].noise_sigma) == (4, 0.04)
    assert (panels["noise"].episode_steps, panels["noise"].noise_sigma) == (4, 0.08)
    assert (panels["horizon"].episode_steps, panels["horizon"].noise_sigma) == (8, 0.04)
    assert (panels["combined"].episode_steps, panels["combined"].noise_sigma) == (8, 0.08)


def test_raw_sequence_contains_same_twenty_observation_scalars() -> None:
    config = _small()
    episodes = generate_episodes(3001, 32, _panel_configs(config)["id"], split="train")
    raw = _raw_sequences(episodes)
    assert raw.shape == (32, 4, 20)


def test_learned_training_is_deterministic_for_fixed_seed() -> None:
    config = _small()
    episodes = generate_episodes(3002, 32, _panel_configs(config)["id"], split="train")
    raw = _raw_sequences(episodes)
    targets = tuple(row.target for row in episodes)
    _, first = _train_model("elman", 3, raw, targets, seed=99, config=config)
    _, second = _train_model("elman", 3, raw, targets, seed=99, config=config)
    assert first.state_hash == second.state_hash
    assert first.parameter_count == second.parameter_count
    assert first.multiply_estimate_per_tick == second.multiply_estimate_per_tick == 72


def test_small_ood_development_is_deterministic_except_wall_time() -> None:
    config = _small()
    first = evaluate_ood_development((3101, 3102), config, bootstrap_samples=100, bootstrap_seed=44)
    second = evaluate_ood_development(
        (3101, 3102), config, bootstrap_samples=100, bootstrap_seed=44
    )
    assert first.panel_means == second.panel_means
    assert first.strong_contrasts == second.strong_contrasts
    assert first.compute_matched_contrasts == second.compute_matched_contrasts
    assert first.gates == second.gates
    assert first.integrity == second.integrity


@pytest.mark.parametrize(
    "kwargs",
    (
        {"train_episodes": 31},
        {"evaluation_episodes": True},
        {"epochs": 0},
        {"learning_rate": "0.01"},
        {"gradient_clip": float("nan")},
    ),
)
def test_invalid_ood_config_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        LearnedOODConfig(**kwargs)
