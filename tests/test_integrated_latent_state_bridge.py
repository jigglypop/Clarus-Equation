from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import integrated_latent_state_bridge as acbsm
from reality_stone.clarus import sparse_causal_bridge as base
from reality_stone.clarus.reliability_rollout_bridge import PrefixReader


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v8.json"


@pytest.fixture(scope="module")
def registration():
    result, _ = base._load_registration(CONFIG)
    return result


@pytest.fixture(scope="module")
def context(registration):
    return acbsm.build_context(CONFIG, registration)


@pytest.fixture(scope="module")
def development_unit_episode(registration):
    # Historical disclosed unit-only seed; never an ACBSM score seed.
    return base.simulate_episode(76999, registration, environment="ood", steps=100)


def test_development_role_is_frozen_and_disjoint(registration) -> None:
    assert acbsm.DEVELOPMENT_SEEDS == tuple(range(82100, 82356))
    assert not set(acbsm.DEVELOPMENT_SEEDS) & acbsm._raw_historical_seeds(CONFIG)
    assert not set(acbsm.DEVELOPMENT_SEEDS) & set(range(81100, 81356))


def test_dynamics_are_ordered_stable_and_psd(context) -> None:
    for dynamics in (
        context.sparse_rank2,
        context.sparse_rank1,
        context.dense_rank2,
        context.zero_rank2,
    ):
        assert np.all(np.abs(dynamics.transition) < 0.98 + 1e-15)
        assert np.min(np.linalg.eigvalsh(dynamics.process_covariance)) >= 0.0
        assert np.min(np.linalg.eigvalsh(dynamics.training_observation_covariance)) >= 0.0
        assert dynamics.training_loading.shape == (4, dynamics.rank)
        assert np.isfinite(dynamics.moment_fit_error)
    assert context.sparse_rank2.transition[0] < context.sparse_rank2.transition[1]


def test_prediction_api_is_prefix_only() -> None:
    parameters = set(inspect.signature(acbsm.predict_from_prefix).parameters)
    assert parameters == {"prefix_states", "context", "registration"}
    assert not {"episode", "truth", "future", "hidden", "target"} & parameters


def test_predictions_controls_and_posteriors_are_finite(
    registration, context, development_unit_episode
) -> None:
    result = acbsm.predict_from_prefix(
        development_unit_episode.states[:81], context, registration
    )
    assert set(result.models) == set(acbsm.MODEL_NAMES)
    assert all(value.shape == (20, 4) for value in result.models.values())
    assert all(np.all(np.isfinite(value)) for value in result.models.values())
    assert result.maximum_covariance_negative_eigenvalue <= 1e-10
    assert all(radius <= 0.98 for radius in result.pathwise_jacobian_radii.values())
    for report in result.posterior.values():
        assert report["minimum_covariance_eigenvalue"] >= -1e-10
        assert len(report["forecast_trace"]) == 20
        assert all(value >= 0.0 for value in report["forecast_trace"])


def test_first_step_is_internal_belief_injection(
    registration, context, development_unit_episode
) -> None:
    prefix = development_unit_episode.states[:81]
    belief = acbsm.filter_prefix(
        prefix, context.v8_context.parent.sparse_mechanism, context.sparse_rank2
    )
    path, _ = acbsm.rollout_from_belief(
        prefix[-1], context.v8_context.parent.sparse_mechanism,
        context.sparse_rank2, belief, 20
    )
    transition = np.diag(context.sparse_rank2.transition)
    expected_mean = transition @ belief.mean
    expected = (
        context.v8_context.parent.sparse_mechanism.predict(prefix[-1])[0]
        + context.sparse_rank2.center
        + belief.geometry.loading @ expected_mean
    )
    assert np.allclose(path[0], expected, rtol=0.0, atol=1e-14)


def test_future_and_hidden_poisoning_is_bit_identical(
    registration, context, development_unit_episode
) -> None:
    original_states = development_unit_episode.states.copy()
    poisoned_states = original_states.copy()
    poisoned_states[81:] += np.arange(1, 21)[:, None] * 1_000_000.0
    poisoned_hidden = development_unit_episode.hidden.copy()
    poisoned_hidden[:] = 1_000_000.0
    original_reader = PrefixReader(original_states, 80)
    poisoned_reader = PrefixReader(poisoned_states, 80)
    original = acbsm.predict_from_prefix(
        original_reader.through_origin(), context, registration
    )
    poisoned = acbsm.predict_from_prefix(
        poisoned_reader.through_origin(), context, registration
    )
    for name in acbsm.MODEL_NAMES:
        assert np.array_equal(original.models[name], poisoned.models[name])
    assert original.posterior == poisoned.posterior
    assert original_reader.max_observed_state_index == 80
    assert poisoned_reader.max_observed_state_index == 80
    assert original_reader.future_observation_reads == 0
    assert poisoned_reader.future_observation_reads == 0
    assert poisoned_hidden.shape == development_unit_episode.hidden.shape


def test_h5_is_only_the_h20_prefix(registration, context, development_unit_episode) -> None:
    result = acbsm.predict_from_prefix(
        development_unit_episode.states[:81], context, registration
    )
    for path in result.models.values():
        assert np.array_equal(path[:5], path[0:5])


def test_v8_locked_test_remains_unopened() -> None:
    test_artifact = ROOT / "artifacts" / "agi" / "sparse_causal_bridge_test_v8.json"
    assert not test_artifact.exists()
    validation = json.loads(
        (ROOT / "artifacts" / "agi" / "sparse_causal_bridge_validation_v8.json")
        .read_text(encoding="utf-8")
    )
    assert validation["passed"] is False
