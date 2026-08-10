from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from reality_stone.clarus import sparse_causal_bridge as base
from reality_stone.clarus.free_rollout_bridge import (
    _implementation_hashes,
    _load_frozen_parent,
    _mechanisms_from_parent,
    fit_prefix_residual_filter,
    fit_stable_observational_model,
    free_rollout,
    run_free_rollout_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v5.json"
EXPECTED_REGISTRATION_SHA = (
    "31e029705a372b622bf5d7109784b89bb0b42d959c4960ef2bdedb3a8c07b78a"
)


def _registration() -> tuple[dict, bytes]:
    return base._load_registration(CONFIG)


def _frozen_mechanism() -> tuple[dict, base.BridgeModel]:
    registration, _ = _registration()
    parent, _ = _load_frozen_parent(CONFIG, registration)
    mechanism, _ = _mechanisms_from_parent(registration, parent)
    return registration, mechanism


def test_v5_registration_is_single_origin_and_freshly_scored() -> None:
    registration, raw = _registration()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_REGISTRATION_SHA
    assert registration["runner"] == "single_origin_free_rollout"
    assert registration["active_gate"] == "rollout_gate"
    assert registration["rollout"]["mode"] == "single_origin_calibrated_free"
    assert registration["rollout"]["horizons"] == [5, 20]
    assert registration["data_roles"]["validation"]["seeds"][0] == 57100
    assert registration["data_roles"]["test"]["seeds"][0] == 58100
    assert registration["data_roles"]["validation"]["steps_per_seed"] == 100
    assert registration["data_roles"]["validation"][
        "intervention_pairs_per_source_per_seed"
    ] == 0


def test_rollout_api_cannot_receive_episode_future_or_hidden() -> None:
    parameters = set(inspect.signature(free_rollout).parameters)
    assert parameters == {
        "mechanism",
        "x_previous",
        "x_anchor",
        "horizon",
        "residual_filter",
    }
    assert not {"episode", "states", "outcomes", "hidden"} & parameters


def test_h5_is_exact_prefix_of_one_h20_free_rollout() -> None:
    registration, mechanism = _frozen_mechanism()
    parent, _ = _load_frozen_parent(CONFIG, registration)
    episode = base.simulate_episode(
        57199, registration, environment="ood", steps=100
    )
    prefix = episode.states[:81]
    residual_filter = fit_prefix_residual_filter(
        prefix,
        mechanism,
        float(parent["latent_filter"]["shared_train_scalar_ar"]),
    )
    rollout20 = free_rollout(
        mechanism,
        x_previous=prefix[-2],
        x_anchor=prefix[-1],
        horizon=20,
        residual_filter=residual_filter,
    )
    rollout5 = free_rollout(
        mechanism,
        x_previous=prefix[-2],
        x_anchor=prefix[-1],
        horizon=5,
        residual_filter=residual_filter,
    )
    assert np.array_equal(rollout5, rollout20[:5])


def test_future_state_and_hidden_poisoning_cannot_change_predictions() -> None:
    registration, mechanism = _frozen_mechanism()
    parent, _ = _load_frozen_parent(CONFIG, registration)
    episode = base.simulate_episode(
        57200, registration, environment="ood", steps=100
    )
    poisoned_states = episode.states.copy()
    poisoned_states[81:] += np.arange(1, 21)[:, None] * 1000.0
    poisoned_hidden = episode.hidden.copy()
    poisoned_hidden[:] = -9999.0
    prefix = episode.states[:81]
    poisoned_prefix = poisoned_states[:81]
    assert np.array_equal(prefix, poisoned_prefix)
    residual_filter = fit_prefix_residual_filter(
        prefix,
        mechanism,
        float(parent["latent_filter"]["shared_train_scalar_ar"]),
    )
    original = free_rollout(
        mechanism,
        x_previous=prefix[-2],
        x_anchor=prefix[-1],
        horizon=20,
        residual_filter=residual_filter,
    )
    poisoned = free_rollout(
        mechanism,
        x_previous=poisoned_prefix[-2],
        x_anchor=poisoned_prefix[-1],
        horizon=20,
        residual_filter=residual_filter,
    )
    assert np.array_equal(original, poisoned)
    original_score = np.mean((episode.states[81:] - original) ** 2)
    poisoned_score = np.mean((poisoned_states[81:] - poisoned) ** 2)
    assert original_score != poisoned_score
    assert poisoned_hidden.shape == episode.hidden.shape


def test_stable_observational_fit_has_no_cubic_term() -> None:
    registration, _ = _registration()
    role = registration["data_roles"]["observational_train"]
    episodes = [
        base.simulate_episode(
            int(seed),
            registration,
            environment=role["environment"],
            steps=120,
        )
        for seed in role["seeds"][:2]
    ]
    all_edges = tuple(
        (source, target)
        for source in range(4)
        for target in range(4)
        if source != target
    )
    model = fit_stable_observational_model(
        "stable", episodes, all_edges, registration["learning"]["ridge"]
    )
    assert np.array_equal(model.local_coefficients[:, 2], np.zeros(4))
    assert np.all(np.isfinite(model.local_coefficients))
    assert np.all(np.isfinite(model.bridge))


def test_equal_probe_dense_uses_every_off_diagonal_estimate() -> None:
    registration, _ = _registration()
    parent, _ = _load_frozen_parent(CONFIG, registration)
    sparse, dense = _mechanisms_from_parent(registration, parent)
    assert np.count_nonzero(sparse.bridge) == 2
    assert np.count_nonzero(dense.bridge) == 12
    assert np.array_equal(
        sparse.local_coefficients[:, 1], dense.local_coefficients[:, 1]
    )


def test_v5_hash_lock_covers_rollout_filter_and_generator() -> None:
    hashes = _implementation_hashes()
    assert set(hashes) == {
        "free_rollout_bridge.py",
        "latent_causal_bridge.py",
        "sparse_causal_bridge.py",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_v5_validation_failure_is_reproducible_without_opening_test() -> None:
    report = run_free_rollout_gate(CONFIG, split="validation")
    assert not report["passed"]
    assert {name for name, passed in report["checks"].items() if not passed} == {
        "h5_seed_wins_persistence",
        "h5_ci_persistence",
        "h20_vs_stable_adaptive_dense",
        "h20_ci_persistence",
    }
    assert report["stability"]["future_observation_reads_by_predictor"] == 0
    assert report["resource_usage"]["forecast_origins_per_seed"] == 1
    assert report["resource_usage"]["free_rollout_steps_per_seed"] == 20
