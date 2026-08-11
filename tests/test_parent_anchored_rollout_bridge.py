from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import sparse_causal_bridge as base
from reality_stone.clarus.parent_anchored_rollout_bridge import (
    MODEL_NAMES,
    _assert_historical_seed_disjoint,
    _atomic_write_once,
    _build_training_context,
    _canonical_json_sha256,
    _implementation_hashes,
    _test_hashes,
    predict_from_prefix,
)
from reality_stone.clarus.reliability_rollout_bridge import PrefixReader


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v8.json"
EXPECTED_RAW_SHA = "a175a3d722f031e4878741a3c7136c75b1af229287e8e90442cecbc9591cdafc"
EXPECTED_CHAIN_SHA = "bf507cd9f02330297634e09deb1e1c6fb16239e62756a5b4cd1d316022d681e9"
EXPECTED_CANONICAL_SHA = "cea8124e0545b2c01ec278004dffe0a38eedb00fb43751c283a37c822d1fcf43"


@pytest.fixture(scope="module")
def registration() -> dict:
    merged, raw = base._load_registration(CONFIG)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_CHAIN_SHA
    return merged


@pytest.fixture(scope="module")
def context(registration: dict):
    return _build_training_context(CONFIG, registration)


@pytest.fixture(scope="module")
def development_episode(registration: dict):
    # Disclosed non-evidence seed; never use either V8 evidence block in unit tests.
    return base.simulate_episode(76999, registration, environment="ood", steps=100)


def test_v8_registration_is_exact_and_preimplementation(registration: dict) -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == EXPECTED_RAW_SHA
    assert _canonical_json_sha256(registration) == EXPECTED_CANONICAL_SHA
    assert registration["status"] == "locked_pre_implementation"
    assert registration["baseline_commit"] == "7c11c04db061d8ef7fc40be68c5a201766c7bd22"
    assert registration["runner"] == "parent_anchored_shrinkage_confirmation"
    assert registration["active_gate"] == "parent_anchor_gate"
    assert registration["parent_anchor"]["critical_value_n256_df255"] == 1.9693105698498752
    validation = registration["data_roles"]["validation"]["seeds"]
    test = registration["data_roles"]["test"]["seeds"]
    assert validation == list(range(80100, 80356))
    assert test == list(range(81100, 81356))
    assert not set(validation) & set(test)
    assert not any(_assert_historical_seed_disjoint(CONFIG, registration).values())


def test_training_lock_reproduces_independent_gains(context, registration: dict) -> None:
    assert context.gain_fit_windows == 176
    assert context.sparse_gain == pytest.approx(0.7868543064870357, abs=1e-15)
    assert context.dense_gain == pytest.approx(0.7835668486813699, abs=1e-15)
    assert context.zero_bridge_gain == pytest.approx(0.882857758971467, abs=1e-15)
    assert len({context.sparse_gain, context.dense_gain, context.zero_bridge_gain}) == 3
    assert np.count_nonzero(context.zero_bridge_mechanism.bridge) == 0
    assert context.zero_bridge_mechanism.declared_edges == ()
    assert np.allclose(
        context.parent.scales,
        np.asarray(registration["normalization"]["expected_scales"]),
        rtol=0.0,
        atol=1e-12,
    )
    assert context.parent.parent_report["passed"] is True
    assert context.parent.v5_failure_report["passed"] is False


def test_prediction_api_has_no_future_bearing_parameter() -> None:
    parameters = set(inspect.signature(predict_from_prefix).parameters)
    assert parameters == {"prefix_states", "context", "registration"}
    assert not {"episode", "future", "truth", "hidden", "outcome"} & parameters


def test_exact_formula_controls_shapes_and_envelopes(
    context, registration: dict, development_episode
) -> None:
    prefix = development_episode.states[:81].copy()
    prefix.setflags(write=False)
    result = predict_from_prefix(prefix, context, registration)
    assert tuple(result.models) == MODEL_NAMES
    assert all(value.shape == (20, 4) for value in result.models.values())
    persistence = result.models["persistence"]
    sparse = result.models["v5_sparse_parent"]
    expected = persistence + context.sparse_gain * (sparse - persistence)
    assert np.array_equal(result.models["parent_anchored_sparse"], expected)
    assert max(result.convex_envelope_violations.values()) <= 1e-12
    for prediction in result.models.values():
        assert np.array_equal(prediction[:5], prediction[0:5])


def test_future_and_hidden_poisoning_is_bit_identical(
    context, registration: dict, development_episode
) -> None:
    original_states = development_episode.states.copy()
    poisoned_states = original_states.copy()
    poisoned_states[81:] += np.arange(1, 21)[:, None] * 100_000.0
    poisoned_hidden = development_episode.hidden.copy()
    poisoned_hidden[:] = -999_999.0
    original_reader = PrefixReader(original_states, 80)
    poisoned_reader = PrefixReader(poisoned_states, 80)
    original = predict_from_prefix(original_reader.through_origin(), context, registration)
    poisoned = predict_from_prefix(poisoned_reader.through_origin(), context, registration)
    for name in MODEL_NAMES:
        assert np.array_equal(original.models[name], poisoned.models[name])
    assert original_reader.max_observed_state_index == 80
    assert poisoned_reader.max_observed_state_index == 80
    assert original_reader.future_observation_reads == 0
    assert poisoned_reader.future_observation_reads == 0
    assert poisoned_hidden.shape == development_episode.hidden.shape


def test_development_components_are_finite_and_retained_stable(
    context, registration: dict, development_episode
) -> None:
    result = predict_from_prefix(development_episode.states[:81], context, registration)
    assert all(np.all(np.isfinite(value)) for value in result.models.values())
    for name in ("sparse", "symmetric_dense", "zero_bridge"):
        assert result.component_pathwise_jacobian_radii[name] <= 0.98
    assert max(abs(context.parent.sparse_ar), abs(context.parent.dense_probe_ar),
               abs(context.zero_bridge_ar)) <= 0.98
    assert registration["parent_anchor"]["sparse_augmented_common_norm_bound"] <= 0.98


def test_hash_surface_is_complete() -> None:
    assert set(_implementation_hashes()) == {
        "parent_anchored_rollout_bridge.py",
        "reliability_rollout_bridge.py",
        "free_rollout_bridge.py",
        "latent_causal_bridge.py",
        "sparse_causal_bridge.py",
    }
    assert set(_test_hashes(ROOT)) == {"test_parent_anchored_rollout_bridge.py"}
    assert all(len(value) == 64 for value in _implementation_hashes().values())


def test_atomic_artifact_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    _atomic_write_once(path, {"passed": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"passed": False}
    with pytest.raises(FileExistsError):
        _atomic_write_once(path, {"passed": True})
