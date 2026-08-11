from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from reality_stone.clarus.sparse_causal_bridge import (
    BridgeModel,
    _load_registration,
    _one_step,
    _true_bridge,
    _validate_registration,
    combine_probes,
    estimate_intervention_edges,
    generate_probe,
    laplace_beltrami_proposal,
    observational_edge_diagnostics,
    permute_probe_signs,
    run_sparse_causal_bridge_gate,
    select_causal_edges,
    simulate_episode,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "preregistration" / "sparse_causal_bridge_v2.json"


def _registration() -> dict:
    return _load_registration(CONFIG)[0]


def test_v2_recursively_merges_v1_without_reusing_validation_seeds() -> None:
    registration = _registration()
    assert registration["scm"]["self_coefficients"] == [0.14, 0.12, 0.78, 0.76]
    assert registration["scm"]["train_hidden_loadings"][2] == 0.0
    assert registration["data_roles"]["validation"]["seeds"][0] == 17100
    assert registration["experiment"] == "sparse_causal_bridge_v2"


def test_matrix_orientation_is_target_by_source() -> None:
    registration = _registration()
    state = np.array([0.6, 0.0, 0.0, 0.0])
    outcome = _one_step(
        state,
        0.0,
        np.zeros(4),
        registration,
        "train",
    )
    assert np.isclose(outcome[2], 0.52 * np.tanh(0.6))
    assert np.isclose(outcome[0], 0.14 * 0.6)


def test_lb_geometry_is_finite_and_covers_both_true_pairs() -> None:
    registration = _registration()
    geometry = laplace_beltrami_proposal(registration)
    assert np.min(geometry["eigenvalues"]) >= -1e-12
    assert np.isfinite(geometry["heat_kernel"]).all()
    assert {(0, 2), (2, 3)} <= set(geometry["pairs"])
    assert (0, 1) in geometry["pairs"]


def test_paired_do_cancels_hidden_and_process_noise() -> None:
    registration = copy.deepcopy(_registration())
    registration["intervention"]["independent_sensor_noise_std"] = 0.0
    probe = generate_probe(
        9123,
        registration,
        environment="train",
        stationary_steps=128,
        pairs_per_source=8,
    )
    amplitude = registration["intervention"]["amplitude"]
    rows = probe.source == 0
    effect = probe.y_plus[rows] - probe.y_minus[rows]
    assert np.allclose(effect[:, 2], 2.0 * 0.52 * np.tanh(amplitude))
    assert np.allclose(effect[:, 1], 0.0)


def test_intervention_estimator_recovers_direction_and_rejects_confounder() -> None:
    registration = copy.deepcopy(_registration())
    registration["intervention"]["independent_sensor_noise_std"] = 0.0
    probe = generate_probe(
        9124,
        registration,
        environment="train",
        stationary_steps=160,
        pairs_per_source=12,
    )
    estimates = estimate_intervention_edges(
        probe, registration["intervention"]["amplitude"]
    )
    assert np.isclose(estimates[(0, 2)]["estimate"], 0.52)
    assert np.isclose(estimates[(2, 3)]["estimate"], -0.48)
    assert abs(estimates[(0, 1)]["estimate"]) < 1e-12
    assert abs(estimates[(1, 0)]["estimate"]) < 1e-12


def test_selector_fixture_uses_no_truth_labels() -> None:
    registration = _registration()
    train_role = registration["data_roles"]["observational_train"]
    holdout_role = registration["data_roles"]["observational_selector_holdout"]
    train = [
        simulate_episode(
            seed,
            registration,
            environment="train",
            steps=160,
        )
        for seed in train_role["seeds"][:2]
    ]
    holdout = [
        simulate_episode(
            seed,
            registration,
            environment="train",
            steps=160,
        )
        for seed in holdout_role["seeds"][:2]
    ]
    diagnostics = observational_edge_diagnostics(
        train, holdout, registration["learning"]["ridge"]
    )
    probes = [
        generate_probe(
            seed,
            registration,
            environment="train",
            stationary_steps=160,
            pairs_per_source=16,
        )
        for seed in registration["data_roles"]["topology_intervention_probe"]["seeds"][:2]
    ]
    effects = estimate_intervention_edges(
        combine_probes(probes), registration["intervention"]["amplitude"]
    )
    selected = select_causal_edges(
        effects,
        diagnostics,
        laplace_beltrami_proposal(registration),
        registration["learning"],
    )
    assert set(selected) == {(0, 2), (2, 3)}


def test_permuted_label_and_no_bridge_controls_select_nothing() -> None:
    registration = _registration()
    role = registration["data_roles"]["topology_intervention_probe"]
    probe = combine_probes(
        [
            generate_probe(
                seed,
                registration,
                environment="train",
                stationary_steps=role["stationary_steps_per_seed"],
                pairs_per_source=role["pairs_per_source_per_seed"],
            )
            for seed in role["seeds"]
        ]
    )
    permuted = permute_probe_signs(
        probe, registration["negative_controls"]["permuted_intervention_seed"]
    )
    effects = estimate_intervention_edges(
        permuted, registration["intervention"]["amplitude"]
    )
    assert max(abs(float(item["estimate"])) for item in effects.values()) < 0.15

    zero = np.zeros_like(_true_bridge(registration))
    no_bridge = generate_probe(
        registration["negative_controls"]["no_bridge_seed"],
        registration,
        environment="train",
        stationary_steps=160,
        pairs_per_source=16,
        bridge_override=zero,
    )
    null_effects = estimate_intervention_edges(
        no_bridge, registration["intervention"]["amplitude"]
    )
    assert max(abs(float(item["estimate"])) for item in null_effects.values()) < 0.15


def test_hidden_loading_shift_flips_confounded_association_only() -> None:
    registration = _registration()
    train = simulate_episode(9911, registration, environment="train", steps=800)
    ood = simulate_episode(9911, registration, environment="ood", steps=800)
    assert np.corrcoef(train.states[:, 0], train.states[:, 1])[0, 1] > 0.7
    assert np.corrcoef(ood.states[:, 0], ood.states[:, 1])[0, 1] < -0.7
    plus = np.array([0.75, 0.0, 0.1, -0.1])
    minus = plus.copy()
    minus[0] = -0.75
    train_effect = _one_step(plus, 0.3, np.zeros(4), registration, "train") - _one_step(
        minus, 0.3, np.zeros(4), registration, "train"
    )
    ood_effect = _one_step(plus, 0.3, np.zeros(4), registration, "ood") - _one_step(
        minus, 0.3, np.zeros(4), registration, "ood"
    )
    assert np.array_equal(train_effect, ood_effect)


def test_seed_is_deterministic() -> None:
    registration = _registration()
    first = simulate_episode(9921, registration, environment="ood", steps=32)
    second = simulate_episode(9921, registration, environment="ood", steps=32)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.hidden, second.hidden)


def test_lesion_changes_only_its_direct_prediction_row() -> None:
    bridge = np.zeros((4, 4))
    bridge[2, 0] = 0.52
    local = np.zeros((4, 3))
    intact = BridgeModel("intact", local, bridge, ((0, 2),))
    lesion = BridgeModel("lesion", local, np.zeros((4, 4)), ())
    states = np.array([[0.5, -0.1, 0.2, 0.3], [-0.4, 0.2, 0.1, -0.2]])
    difference = intact.predict(states) - lesion.predict(states)
    assert np.max(np.abs(difference[:, [0, 1, 3]])) == 0.0
    assert np.max(np.abs(difference[:, 2])) > 0.0


def test_invalid_duplicate_seed_is_rejected() -> None:
    registration = copy.deepcopy(_registration())
    registration["negative_controls"]["no_bridge_seed"] = registration["data_roles"][
        "validation"
    ]["seeds"][0]
    with pytest.raises(ValueError, match="seeds must be disjoint"):
        _validate_registration(registration)


def test_validation_gate_passes_without_download_or_trajectory_dump() -> None:
    report = run_sparse_causal_bridge_gate(CONFIG, split="validation")
    assert report["passed"]
    assert report["selection"]["causal_edges"] == ["A->C", "C->D"]
    assert report["negative_controls"]["no_bridge_selected_edges"] == []
    assert report["negative_controls"]["permuted_intervention_selected_edges"] == []
    assert report["resource_usage"]["external_download_bytes"] == 0
    assert report["resource_usage"]["trajectory_files_written"] == 0


def test_evaluation_seed_change_cannot_change_selection(tmp_path: Path) -> None:
    baseline = run_sparse_causal_bridge_gate(CONFIG, split="validation")
    registration = copy.deepcopy(_registration())
    registration.pop("extends", None)
    registration.pop("overrides", None)
    registration["data_roles"]["validation"]["seeds"] = list(range(27100, 27120))
    alternate_path = tmp_path / "selection_leakage_guard.json"
    alternate_path.write_text(json.dumps(registration), encoding="utf-8")
    alternate = run_sparse_causal_bridge_gate(alternate_path, split="validation")
    assert baseline["selection"]["causal_edges"] == alternate["selection"]["causal_edges"]
    assert (
        baseline["selection"]["intervention_diagnostics"]
        == alternate["selection"]["intervention_diagnostics"]
    )
    assert baseline["models"]["causal_bridge"]["seed_global_rmse"] != alternate[
        "models"
    ]["causal_bridge"]["seed_global_rmse"]
