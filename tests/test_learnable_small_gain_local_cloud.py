import numpy as np
import pytest
import torch

from reality_stone.clarus.learnable_small_gain_local_cloud import (
    LearnableSmallGainConfig,
    LearnableSmallGainLocalCloud,
    StableBilinearLocalCloud,
    model_hash,
    train_learnable_small_gain,
    train_stable_bilinear,
)


def test_default_certificate_and_live_operator_budgets() -> None:
    config = LearnableSmallGainConfig()
    certificate = config.certificate()
    model = LearnableSmallGainLocalCloud(config)
    assert certificate["certified"]
    assert certificate["q"] < 0.99
    row_sums = model.operator_row_sums()
    expected = (0.82, 0.82, 0.06, 0.04, 1.0, 1.0, 0.08)
    assert all(actual <= bound + 5e-7 for actual, bound in zip(row_sums, expected))


def test_forward_has_one_score_per_sequence_and_finite_gradients() -> None:
    model = LearnableSmallGainLocalCloud()
    sequence = torch.linspace(-1.0, 1.0, 5 * 8 * 20).reshape(5, 8, 20)
    scores = model(sequence)
    scores.sum().backward()
    assert scores.shape == (5,)
    assert torch.all(torch.isfinite(scores))
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())


def test_training_is_deterministic() -> None:
    rng = np.random.default_rng(81)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    first = train_learnable_small_gain(features, labels, seed=9, epochs=2)
    second = train_learnable_small_gain(features, labels, seed=9, epochs=2)
    assert model_hash(first) == model_hash(second)


def test_invalid_or_unstable_config_fails_closed() -> None:
    with pytest.raises(ValueError):
        LearnableSmallGainConfig(local_budget=True)
    with pytest.raises(ValueError, match="not contractive"):
        LearnableSmallGainConfig(
            local_budget=0.9,
            cloud_budget=0.9,
            cloud_to_local_budget=0.2,
            local_to_cloud_budget=0.2,
            interaction_budget=0.2,
        )


def test_stable_bilinear_is_small_and_strictly_contractive() -> None:
    model = StableBilinearLocalCloud()
    assert model.parameter_count() == 53
    assert model.contraction_factor() < 0.995
    sequence = torch.zeros(7, 8, 20)
    assert model(sequence).shape == (7,)


def test_salience_gate_preserves_a_finite_bounded_state_path() -> None:
    model = StableBilinearLocalCloud()
    sequence = torch.zeros(1, 8, 20)
    sequence[:, 0, :16] = 1.0
    sequence[:, 1, 16] = 1.0
    sequence[:, 2:, :] = 0.01
    with torch.no_grad():
        score = model(sequence)
        local_retention, cloud_retention = model.retentions()
    assert torch.isfinite(score).all()
    assert float(torch.max(local_retention)) < 0.995
    assert float(torch.max(cloud_retention)) < 0.995


def test_stable_bilinear_training_is_deterministic() -> None:
    rng = np.random.default_rng(82)
    features = rng.normal(size=(32, 4, 20)).astype(np.float32)
    labels = tuple(1 if value > 0 else -1 for value in rng.normal(size=32))
    first = train_stable_bilinear(features, labels, seed=10, epochs=2)
    second = train_stable_bilinear(features, labels, seed=10, epochs=2)
    assert model_hash(first) == model_hash(second)
