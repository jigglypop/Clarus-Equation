import numpy as np
import pytest

from reality_stone.clarus.local_cloud_benchmark import (
    ARMS,
    LocalCloudBenchmarkConfig,
    accuracy,
    episode_features,
    evaluate_development_seed,
    evaluate_registered_development,
    feature_matrix,
    fit_frozen_ridge,
    generate_episodes,
    kernel_for_arm,
)


def _small_config():
    return LocalCloudBenchmarkConfig(train_episodes=32, evaluation_episodes=32)


def test_factorial_generation_is_balanced_and_split_separated() -> None:
    config = _small_config()
    train = generate_episodes(91, 32, config, split="train")
    evaluation = generate_episodes(91, 32, config, split="evaluation")
    conditions = {(row.context_index, row.local_bits[:3]) for row in train}
    assert len(conditions) == 32
    assert train != evaluation
    assert {row.target for row in train} == {-1, 1}


def test_all_factorial_arms_have_equal_state_and_feature_width() -> None:
    episode = generate_episodes(17, 32, _small_config(), split="train")[0]
    for arm in ARMS:
        kernel = kernel_for_arm(arm)
        assert len(episode_features(episode, kernel)) == 20
        assert kernel.config.local_count * kernel.config.width + kernel.config.width == 20


def test_feature_extraction_is_label_blind() -> None:
    episode = generate_episodes(8, 32, _small_config(), split="train")[0]
    forged = type(episode)(
        observations=episode.observations,
        target=-episode.target,
        context_index=episode.context_index,
        local_bits=episode.local_bits,
    )
    kernel = kernel_for_arm("full")
    assert episode_features(episode, kernel) == episode_features(forged, kernel)


def test_true_lesions_change_full_transition_features() -> None:
    episode = generate_episodes(18, 32, _small_config(), split="train")[0]
    kernel = kernel_for_arm("full")
    intact = episode_features(episode, kernel)
    cut = episode_features(episode, kernel, lesion="cross_cut")
    local_reset = episode_features(
        episode, kernel, lesion="local_reset", lesion_at_decision_only=True
    )
    cloud_reset = episode_features(
        episode, kernel, lesion="cloud_reset", lesion_at_decision_only=True
    )
    assert intact != cut
    assert intact != local_reset
    assert intact != cloud_reset


def test_ridge_is_fit_from_train_rows_and_is_frozen_for_evaluation() -> None:
    config = _small_config()
    train = generate_episodes(31, 32, config, split="train")
    evaluation = generate_episodes(31, 32, config, split="evaluation")
    kernel = kernel_for_arm("full")
    train_x = feature_matrix(train, kernel)
    readout = fit_frozen_ridge(
        train_x, (row.target for row in train), ridge_lambda=config.ridge_lambda
    )
    before = readout
    score = accuracy(
        readout, feature_matrix(evaluation, kernel), (row.target for row in evaluation)
    )
    assert readout == before
    assert 0.0 <= score <= 1.0
    assert len(readout.coefficients) == 20
    assert 1.0 <= readout.effective_degrees_of_freedom <= 21.0


def test_fixed_readout_is_reused_for_lesion_not_refit() -> None:
    config = _small_config()
    train = generate_episodes(71, 32, config, split="train")
    evaluation = generate_episodes(72, 32, config, split="evaluation")
    kernel = kernel_for_arm("full")
    readout = fit_frozen_ridge(
        feature_matrix(train, kernel),
        (row.target for row in train),
        ridge_lambda=config.ridge_lambda,
    )
    intact = readout.predict_scores(feature_matrix(evaluation, kernel))
    cut = readout.predict_scores(feature_matrix(evaluation, kernel, lesion="cross_cut"))
    assert not np.array_equal(intact, cut)


def test_development_seed_has_all_arms_and_true_lesions() -> None:
    row = evaluate_development_seed(45001, _small_config())
    assert set(row.accuracies) == set(ARMS)
    assert set(row.lesion_accuracies) == {"cross_cut", "local_reset", "cloud_reset"}
    assert row.nonfinite_count == 0
    assert row.label_state_bypass_count == 0


def test_registered_development_is_seed_paired_and_deterministic() -> None:
    config = _small_config()
    first = evaluate_registered_development(
        (45011, 45012), config, bootstrap_samples=100, bootstrap_seed=7788
    )
    second = evaluate_registered_development(
        (45011, 45012), config, bootstrap_samples=100, bootstrap_seed=7788
    )
    assert first == second
    assert first.seed_count == 2
    assert set(first.gates) == {
        "full_accuracy",
        "paired_improvement_lcb",
        "factorial_interaction_lcb",
        "cross_cut_loss_lcb_positive",
        "local_reset_loss_lcb_positive",
        "cloud_reset_loss_lcb_positive",
        "integrity",
    }
    assert all(value == 20 for value in first.state_scalar_count.values())


def test_registered_development_rejects_duplicate_seed() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_registered_development(
            (9, 9), _small_config(), bootstrap_samples=100, bootstrap_seed=1
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"train_episodes": 15},
        {"evaluation_episodes": True},
        {"episode_steps": 3},
        {"noise_sigma": 0.0},
        {"ridge_lambda": "0.001"},
    ),
)
def test_invalid_benchmark_config_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        LocalCloudBenchmarkConfig(**kwargs)
