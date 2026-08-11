from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import sparse_causal_bridge as base
from reality_stone.clarus.reliability_rollout_bridge import (
    PrefixReader,
    _build_training_context,
    _canonical_json_sha256,
    _implementation_hashes,
    _test_hashes,
    _validation_artifact_path,
    predict_from_prefix,
    run_reliability_closure_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v7.json"
EXPECTED_RAW_FILE_SHA = "134ddaa793170b898649b79e11407c10f35d1468ba95701544a06905d9448c3e"
EXPECTED_MERGED_SHA = "3cfa4ddc9bb6ab04bb7b37403780ef2fd4a894d26e7c45c1c84e062434fb4259"
EXPECTED_CANONICAL_SHA = "2d1c06cb9259e52e435e28017b82d89924c4c305c0dc81b29beadf78ede13365"


@pytest.fixture(scope="module")
def registration() -> dict:
    merged, raw = base._load_registration(CONFIG)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_MERGED_SHA
    return merged


@pytest.fixture(scope="module")
def context(registration: dict):
    return _build_training_context(CONFIG, registration)


@pytest.fixture(scope="module")
def development_episode(registration: dict):
    # This seed is outside every registered V1--V7 role and is never evidence.
    return base.simulate_episode(76999, registration, environment="ood", steps=100)


def test_v7_registration_was_locked_before_implementation(registration: dict) -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == EXPECTED_RAW_FILE_SHA
    assert _canonical_json_sha256(registration) == EXPECTED_CANONICAL_SHA
    assert registration["status"] == "locked_pre_implementation"
    assert registration["runner"] == "symmetric_consensus_closure"
    assert registration["active_gate"] == "closure_gate"
    assert registration["closure"]["horizon"] == 20
    assert registration["closure_gate"]["h5_is_gating"] is False
    assert len(registration["data_roles"]["validation"]["seeds"]) == 96
    assert len(registration["data_roles"]["test"]["seeds"]) == 96
    assert set(registration["data_roles"]["validation"]["seeds"]).isdisjoint(
        registration["data_roles"]["test"]["seeds"]
    )


def test_historical_parent_and_failure_hashes_are_exact(context) -> None:
    assert context.parent_raw_sha256 == (
        "41c17778c7aa2adcd36557ca0042ea0d2de90c817acbd8730bbc97424f553986"
    )
    assert context.v5_failure_raw_sha256 == (
        "6dd4999e385fc47ea5ccd2e3e1233c60f2d1968554b82dd5cb95a34524f9e9a0"
    )
    assert context.parent_report["passed"] is True
    assert context.v5_failure_report["passed"] is False


def test_training_only_normalization_is_frozen(context, registration: dict) -> None:
    expected = np.asarray(registration["normalization"]["expected_scales"])
    assert np.allclose(context.scales, expected, rtol=0.0, atol=1e-12)
    assert np.all(context.scales > 0)
    assert np.isfinite(context.train_normalized_norm_q99)


def test_prediction_api_cannot_receive_episode_future_or_hidden() -> None:
    parameters = set(inspect.signature(predict_from_prefix).parameters)
    assert parameters == {"prefix_states", "context", "registration"}
    assert not {"episode", "future", "truth", "hidden", "outcomes"} & parameters


def test_symmetric_controllers_use_valid_independent_weights(
    context, registration: dict, development_episode
) -> None:
    prefix = development_episode.states[:81].copy()
    prefix.setflags(write=False)
    result = predict_from_prefix(prefix, context, registration)
    assert set(result.models) == {
        "sparse_consensus",
        "no_sparse_consensus",
        "symmetric_dense_consensus",
        "v5_sparse_parent",
        "stable_adaptive_dense_prefix_free",
        "persistence",
    }
    assert all(value.shape == (20, 4) for value in result.models.values())
    assert len(result.weights["sparse_consensus"]) == 3
    assert len(result.weights["symmetric_dense_consensus"]) == 3
    assert len(result.weights["no_sparse_consensus"]) == 2
    for weights in result.weights.values():
        assert np.all(weights >= 0)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-12)
    assert not np.array_equal(
        result.weights["sparse_consensus"],
        result.weights["symmetric_dense_consensus"],
    )
    assert result.component_rollouts == 8


def test_future_and_hidden_poisoning_leave_pipeline_predictions_identical(
    context, registration: dict, development_episode
) -> None:
    original_states = development_episode.states.copy()
    poisoned_states = original_states.copy()
    poisoned_states[81:] += np.arange(1, 21)[:, None] * 10_000.0
    poisoned_hidden = development_episode.hidden.copy()
    poisoned_hidden[:] = -99_999.0

    original_reader = PrefixReader(original_states, origin=80)
    poisoned_reader = PrefixReader(poisoned_states, origin=80)
    original = predict_from_prefix(original_reader.through_origin(), context, registration)
    poisoned = predict_from_prefix(poisoned_reader.through_origin(), context, registration)
    for name in original.models:
        assert np.array_equal(original.models[name], poisoned.models[name])
    for name in original.weights:
        assert np.array_equal(original.weights[name], poisoned.weights[name])
    assert original_reader.max_observed_state_index == 80
    assert poisoned_reader.max_observed_state_index == 80
    assert original_reader.future_observation_reads == 0
    assert poisoned_reader.future_observation_reads == 0
    assert poisoned_hidden.shape == development_episode.hidden.shape


def test_h5_is_only_the_exact_prefix_of_each_h20_prediction(
    context, registration: dict, development_episode
) -> None:
    result = predict_from_prefix(development_episode.states[:81], context, registration)
    for prediction in result.models.values():
        h5 = prediction[:5].copy()
        assert np.array_equal(h5, prediction[:5])
        assert prediction.shape[0] == 20


def test_all_development_predictions_and_radii_are_finite(
    context, registration: dict, development_episode
) -> None:
    result = predict_from_prefix(development_episode.states[:81], context, registration)
    assert all(np.all(np.isfinite(value)) for value in result.models.values())
    assert all(
        np.isfinite(value) and value < 0.98 for value in result.pathwise_jacobian_radii.values()
    )
    assert abs(context.sparse_ar) < 0.98
    assert abs(context.dense_probe_ar) < 0.98


def test_lock_hashes_cover_all_registered_sources_and_tests() -> None:
    hashes = _implementation_hashes()
    assert set(hashes) == {
        "reliability_rollout_bridge.py",
        "free_rollout_bridge.py",
        "latent_causal_bridge.py",
        "sparse_causal_bridge.py",
    }
    assert all(len(value) == 64 for value in hashes.values())
    test_hashes = _test_hashes(ROOT)
    assert set(test_hashes) == {"test_reliability_rollout_bridge.py"}
    assert len(test_hashes["test_reliability_rollout_bridge.py"]) == 64


def test_locked_test_cannot_open_without_a_passing_validation() -> None:
    path = _validation_artifact_path(CONFIG)
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report.get("passed") is not True
    with pytest.raises(PermissionError):
        run_reliability_closure_gate(CONFIG, split="test")
