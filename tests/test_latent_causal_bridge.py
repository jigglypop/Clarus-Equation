from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus import sparse_causal_bridge as base
from reality_stone.clarus.latent_causal_bridge import (
    _assert_v3_test_unlocked,
    _implementation_hashes,
    estimate_full_mechanism,
    fit_pooled_residual_autoregression,
    fit_residual_filter,
    mechanism_model,
    run_latent_causal_bridge_gate,
    sequential_filter_prediction,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v3.json"


def _registration() -> dict:
    return base._load_registration(CONFIG)[0]


def test_v3_restores_direct_target_confounding_with_fresh_seeds() -> None:
    registration = _registration()
    assert registration["runner"] == "latent_residual_filter"
    assert registration["scm"]["train_hidden_loadings"][2] == 0.55
    assert registration["scm"]["ood_hidden_loadings"][2] == -0.55
    assert registration["data_roles"]["validation"]["seeds"][0] == 37100
    assert registration["data_roles"]["test"]["seeds"][0] == 38100


def test_full_paired_estimator_recovers_diagonal_and_bridges_noiselessly() -> None:
    registration = copy.deepcopy(_registration())
    registration["intervention"]["independent_sensor_noise_std"] = 0.0
    probe = base.generate_probe(
        49101,
        registration,
        environment="train",
        stationary_steps=160,
        pairs_per_source=12,
    )
    estimates = estimate_full_mechanism(
        probe, registration["intervention"]["amplitude"]
    )
    diagonal = np.asarray([estimates[(index, index)]["estimate"] for index in range(4)])
    assert np.allclose(diagonal, registration["scm"]["self_coefficients"])
    assert np.isclose(estimates[(0, 2)]["estimate"], 0.52)
    assert np.isclose(estimates[(2, 3)]["estimate"], -0.48)
    assert abs(estimates[(0, 1)]["estimate"]) < 1e-12


def test_rank_one_filter_recovers_ood_loading_subspace() -> None:
    registration = _registration()
    truth_bridge = base._true_bridge(registration)
    truth_self = np.asarray(registration["scm"]["self_coefficients"])
    mechanism = mechanism_model(
        "truth",
        truth_self,
        truth_bridge,
        ((0, 2), (2, 3)),
    )
    episode = base.simulate_episode(
        49102,
        registration,
        environment="ood",
        steps=240,
    )
    residual_filter = fit_residual_filter(episode, mechanism, calibration_steps=100)
    loading = np.asarray(registration["scm"]["ood_hidden_loadings"])
    cosine = abs(float(residual_filter.direction @ (loading / np.linalg.norm(loading))))
    assert cosine > 0.9
    assert residual_filter.variance_fraction > 0.75
    assert abs(residual_filter.autoregression - registration["scm"]["latent_ar"]) < 0.1


def test_sequential_prediction_does_not_read_the_current_outcome() -> None:
    registration = _registration()
    mechanism = mechanism_model(
        "truth",
        np.asarray(registration["scm"]["self_coefficients"]),
        base._true_bridge(registration),
        ((0, 2), (2, 3)),
    )
    episode = base.simulate_episode(
        49103,
        registration,
        environment="ood",
        steps=180,
    )
    calibration = 80
    residual_filter = fit_residual_filter(episode, mechanism, calibration)
    original = sequential_filter_prediction(
        episode, mechanism, residual_filter, calibration
    )
    changed_states = episode.states.copy()
    changed_states[calibration + 1] += 100.0
    changed = base.Episode(changed_states, episode.hidden.copy())
    altered = sequential_filter_prediction(changed, mechanism, residual_filter, calibration)
    assert np.array_equal(original[0], altered[0])
    assert not np.array_equal(original[1], altered[1])


def test_filter_rejects_calibration_that_consumes_the_evaluation() -> None:
    registration = _registration()
    mechanism = mechanism_model(
        "truth",
        np.asarray(registration["scm"]["self_coefficients"]),
        base._true_bridge(registration),
        ((0, 2), (2, 3)),
    )
    episode = base.simulate_episode(
        49104,
        registration,
        environment="ood",
        steps=20,
    )
    with pytest.raises(ValueError, match="leave evaluation rows"):
        fit_residual_filter(episode, mechanism, calibration_steps=20)


def test_pooled_train_ar_is_invariant_and_override_is_honored() -> None:
    registration = _registration()
    mechanism = mechanism_model(
        "truth",
        np.asarray(registration["scm"]["self_coefficients"]),
        base._true_bridge(registration),
        ((0, 2), (2, 3)),
    )
    episodes = [
        base.simulate_episode(
            seed,
            registration,
            environment="train",
            steps=400,
        )
        for seed in (49201, 49202, 49203, 49204)
    ]
    autoregression = fit_pooled_residual_autoregression(episodes, mechanism)
    assert abs(autoregression - registration["scm"]["latent_ar"]) < 0.05
    ood = base.simulate_episode(
        49205,
        registration,
        environment="ood",
        steps=180,
    )
    residual_filter = fit_residual_filter(
        ood,
        mechanism,
        calibration_steps=80,
        autoregression_override=autoregression,
    )
    assert residual_filter.autoregression == autoregression


def test_v3_implementation_hashes_cover_generator_and_filter() -> None:
    hashes = _implementation_hashes()
    assert set(hashes) == {"latent_causal_bridge.py", "sparse_causal_bridge.py"}
    assert all(len(value) == 64 for value in hashes.values())


def test_v4_validation_gate_and_code_bound_lock_pass() -> None:
    config = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v4.json"
    registration, raw = base._load_registration(config)
    report = run_latent_causal_bridge_gate(config, split="validation")
    assert report["passed"]
    assert report["selection"]["causal_edges"] == ["A->C", "C->D"]
    assert report["implementation_sha256"] == _implementation_hashes()
    import hashlib

    _assert_v3_test_unlocked(config, registration, hashlib.sha256(raw).hexdigest())
