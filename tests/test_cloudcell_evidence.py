from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.cloudcell_evidence import (
    CloudCellGateConfig,
    TrialPopulation,
    block_shift_null,
    build_cloudcell_artifact,
    coding_comparison,
    evaluate_panel,
    maintenance_persistence_gate,
    unit_dropout_gate,
)


def test_trial_population_requires_aligned_causal_windows():
    data = make_cloudcell_trials(seed=3)

    with pytest.raises(ValueError, match="same trials x units shape"):
        TrialPopulation(
            subject_id=data.subject_id,
            encoding=data.encoding,
            maintenance_early=data.maintenance_early,
            maintenance_late=data.maintenance_late[:, :-1],
            probe=data.probe,
            memory_load=data.memory_load,
            probe_in_out=data.probe_in_out,
            trial_ids=data.trial_ids,
        )


def test_distributed_code_beats_train_selected_best_single_and_block_shift():
    data = make_cloudcell_trials(seed=11)

    comparison = coding_comparison(data.encoding, data.memory_load)
    dropout = unit_dropout_gate(data.encoding, data.memory_load)
    null = block_shift_null(
        data.encoding,
        data.memory_load,
        n_shifts=19,
        block_size=7,
    )

    assert comparison.population_balanced_accuracy > 0.70
    assert comparison.population_gain_over_single > 0.10
    assert comparison.population_gain_over_baseline > 0.30
    assert dropout.minimum_population_gain > 0.0
    assert null.observed_gain == pytest.approx(comparison.population_gain_over_single)
    assert null.p_value <= 0.05


def test_maintenance_persistence_separates_local_cloud_and_full_models():
    data = make_cloudcell_trials(seed=17)

    persistence = maintenance_persistence_gate(
        data.maintenance_early,
        data.maintenance_late,
    )

    assert persistence.valid_units == data.n_units
    assert persistence.local_gain_over_baseline > 0.25
    assert persistence.full_gain_over_local > 0.01
    assert persistence.full_gain_over_cloud > 0.01
    assert persistence.full_gain_over_best_partial > 0.01


def test_task_covariate_baseline_is_shared_by_all_persistence_models():
    data = make_cloudcell_trials(seed=19)
    load_classes = np.unique(data.memory_load)
    task_load = np.column_stack(
        [data.memory_load == label for label in load_classes[1:]]
    ).astype(float)

    persistence = maintenance_persistence_gate(
        data.maintenance_early,
        data.maintenance_late,
        covariates=task_load,
    )

    assert persistence.valid_units == data.n_units
    assert persistence.local_gain_over_baseline > 0.25
    assert persistence.full_gain_over_best_partial > 0.01


def test_probe_innovation_and_task_baseline_are_explicit_exploratory_variants():
    data = make_cloudcell_trials(seed=21)
    config = CloudCellGateConfig(
        n_shifts=9,
        block_size=11,
        max_null_p=0.10,
        probe_feature_variant="innovation",
        persistence_baseline="task_load",
    )

    subject = evaluate_panel([data], config)["subjects"][0]

    assert subject["window_policy"]["probe_feature_variant"] == "innovation"
    assert subject["window_policy"]["persistence_baseline"] == "task_load"


def test_panel_can_pass_operational_signature_but_never_identifies_literal_monad():
    datasets = [
        make_cloudcell_trials(seed=23, subject_id="synthetic-a"),
        make_cloudcell_trials(seed=29, subject_id="synthetic-b"),
    ]
    config = CloudCellGateConfig(
        n_shifts=9,
        block_size=11,
        min_population_gain=0.05,
        min_dropout_gain=0.0,
        min_local_gain=0.10,
        min_full_over_best_gain=0.005,
        max_null_p=0.10,
        min_subject_fraction=1.0,
    )

    panel = evaluate_panel(datasets, config)
    artifact = build_cloudcell_artifact(
        panel,
        config=config,
        provenance=[{"subject_id": item.subject_id, "sha256": "a" * 64} for item in datasets],
    )

    assert panel["operational_gate_passed"] is True
    assert panel["literal_coded_monad_claim"]["decision"] == "withhold_identity_claim"
    assert "not a subject-independent" in panel["literal_coded_monad_claim"]["scope"]
    assert artifact["gate_passed"] is True
    assert artifact["claim_not_identified"] == (
        "a biological neuron is literally a mathematical monad"
    )


def test_registered_kill_rule_blocks_noise_only_panel():
    rng = np.random.default_rng(101)
    n_trials, n_units = 150, 7
    noise = TrialPopulation(
        subject_id="noise",
        encoding=rng.normal(size=(n_trials, n_units)),
        maintenance_early=rng.normal(size=(n_trials, n_units)),
        maintenance_late=rng.normal(size=(n_trials, n_units)),
        probe=rng.normal(size=(n_trials, n_units)),
        memory_load=np.tile(np.arange(3), n_trials // 3),
        probe_in_out=np.tile(np.arange(2), n_trials // 2),
        trial_ids=np.arange(n_trials),
    )
    config = CloudCellGateConfig(
        n_shifts=9,
        block_size=10,
        min_population_gain=0.10,
        min_dropout_gain=0.05,
        min_local_gain=0.10,
        min_full_over_best_gain=0.05,
        max_null_p=0.10,
        min_subject_fraction=1.0,
    )

    panel = evaluate_panel([noise], config)

    assert panel["operational_gate_passed"] is False
    assert panel["subject_pass_count"] == 0
    assert panel["literal_coded_monad_claim"]["decision"] == "withhold_identity_claim"
    assert "did not pass" in panel["literal_coded_monad_claim"]["reason"]


def make_cloudcell_trials(
    *,
    seed: int,
    subject_id: str = "synthetic",
    n_trials: int = 210,
    n_units: int = 8,
) -> TrialPopulation:
    rng = np.random.default_rng(seed)
    memory_load = np.repeat(np.arange(1, 4), n_trials // 3)
    rng.shuffle(memory_load)
    probe = np.tile(np.arange(2), n_trials // 2)
    rng.shuffle(probe)

    load_code = rng.normal(size=(3, n_units))
    load_code -= np.mean(load_code, axis=1, keepdims=True)
    encoding = 0.55 * load_code[memory_load - 1] + rng.normal(
        scale=0.85,
        size=(n_trials, n_units),
    )

    local_state = rng.normal(size=(n_trials, n_units))
    shared_state = rng.normal(size=(n_trials, 2)) @ rng.normal(size=(2, n_units))
    probe_code = rng.normal(size=n_units)
    early = (
        local_state
        + 0.35 * shared_state
        + 0.50 * (2.0 * probe[:, None] - 1.0) * probe_code[None, :]
    )
    other_mean = (np.sum(early, axis=1, keepdims=True) - early) / (n_units - 1)
    late = 0.72 * early + 0.55 * other_mean + rng.normal(
        scale=0.22,
        size=(n_trials, n_units),
    )
    probe_direction = rng.choice((-1.0, 1.0), size=n_units)
    probe_epoch = (
        0.48 * (2.0 * probe[:, None] - 1.0) * probe_direction[None, :]
        + rng.normal(scale=0.90, size=(n_trials, n_units))
    )
    return TrialPopulation(
        subject_id=subject_id,
        encoding=encoding,
        maintenance_early=early,
        maintenance_late=late,
        probe=probe_epoch,
        memory_load=memory_load,
        probe_in_out=probe,
        trial_ids=np.arange(n_trials),
    )
