from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reality_stone.clarus.tafazoli_session_operator_probe import (
    NO,
    PENDING,
    TEST_UNAVAILABLE,
    ProbeClaimLocks,
    ProbeConfig,
    SESSION_NEURON_COUNTS,
    VERDICT_BRAIN_PROGRAMMING_LANGUAGE,
    VERDICT_SESSION_LOCAL_SHORT_MEMORY,
    VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR,
    VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR,
    VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR,
    extract_tafazoli_train_dimensions,
    make_whole_trial_folds,
    prepare_session_latent_fold,
    recovered_session_specs,
    run_tafazoli_session_operator_probe_from_arrays,
    transition_start_indices,
    validate_claim_locks,
    validate_probe_report,
    verify_official_classifier_checksum,
)


def _synthetic_dynamics(
    operators: tuple[np.ndarray, ...],
    *,
    seed: int,
    trials: int = 12,
    timepoints: int = 41,
    lag_bins: int = 10,
) -> np.ndarray:
    """Return non-negative count-like data with known interleaved dynamics."""

    rng = np.random.default_rng(seed)
    session_values = []
    for operator in operators:
        neuron_count = operator.shape[0]
        state = np.zeros((trials, timepoints, neuron_count), dtype=np.float64)
        state[:, :lag_bins, :] = rng.normal(
            scale=0.35,
            size=(trials, lag_bins, neuron_count),
        )
        for time_index in range(lag_bins, timepoints):
            state[:, time_index, :] = (
                state[:, time_index - lag_bins, :] @ operator.T
                + rng.normal(scale=0.015, size=(trials, neuron_count))
            )
        anscombe_domain = 6.0 + state
        counts = np.maximum(np.square(anscombe_domain) - 3.0 / 8.0, 0.0)
        session_values.append(np.transpose(counts, (0, 2, 1)))
    return np.concatenate(session_values, axis=1)


@pytest.fixture(scope="module")
def synthetic_report():
    forward = np.asarray(
        [
            [0.72, -0.32, 0.00],
            [0.32, 0.72, 0.00],
            [0.00, 0.00, 0.55],
        ]
    )
    opposite = np.asarray(
        [
            [0.72, 0.32, 0.00],
            [-0.32, 0.72, 0.00],
            [0.00, 0.00, 0.35],
        ]
    )
    dim1 = _synthetic_dynamics((forward, forward), seed=11)
    dim3 = _synthetic_dynamics((opposite, opposite), seed=29)
    config = ProbeConfig(
        fold_count=3,
        lag_bins=10,
        transition_stride_bins=1,
        rank_cap=3,
        successor_shuffle_count=5,
        session_neuron_counts=(3, 3),
        session_animals=("animal_a", "animal_b"),
        rank_stability_caps=(1, 3),
    )
    return run_tafazoli_session_operator_probe_from_arrays(
        dim1,
        dim3,
        config=config,
    )


def test_recovered_27_session_boundaries_are_exact_and_contiguous() -> None:
    specs = recovered_session_specs()

    assert len(specs) == 27
    assert tuple(spec.neuron_count for spec in specs) == SESSION_NEURON_COUNTS
    assert specs[0].column_start_zero_based == 0
    assert specs[-1].column_stop_exclusive == 403
    assert all(
        left.column_stop_exclusive == right.column_start_zero_based
        for left, right in zip(specs, specs[1:])
    )
    assert tuple(spec.animal for spec in specs[:9]) == ("Chico",) * 9
    assert tuple(spec.animal for spec in specs[9:]) == ("Silas",) * 18


def test_whole_trial_folds_and_lag_pairs_are_deterministic() -> None:
    folds = make_whole_trial_folds(36, fold_count=6, seed=20260730)

    assert folds[0].test_indices == (1, 2, 16, 21, 27, 34)
    assert len(folds) == 6
    all_test = []
    for fold in folds:
        assert not set(fold.train_indices) & set(fold.test_indices)
        assert set(fold.train_indices) | set(fold.test_indices) == set(
            range(36)
        )
        all_test.extend(fold.test_indices)
    assert sorted(all_test) == list(range(36))

    starts = transition_start_indices(81, lag_bins=10, stride_bins=1)
    assert starts.shape == (71,)
    assert starts[0] == 0
    assert starts[-1] == 70
    assert np.all(starts + 10 < 81)
    assert transition_start_indices(
        81,
        lag_bins=10,
        stride_bins=10,
    ).tolist() == [0, 10, 20, 30, 40, 50, 60, 70]


def test_extractor_never_reads_dim2_labels_factors_or_saved_test() -> None:
    dim1 = np.ones((4, 2, 12))
    dim3 = np.full((4, 2, 12), 3.0)

    class TrainOnlyCell:
        def __init__(self, train: np.ndarray):
            self.train = train

        def __getitem__(self, index: int) -> np.ndarray:
            if index != 0:
                raise AssertionError("saved test set was accessed")
            return self.train

    class DimensionCells:
        def __getitem__(self, index: int) -> TrainOnlyCell:
            if index == 0:
                return TrainOnlyCell(dim1)
            if index == 2:
                return TrainOnlyCell(dim3)
            raise AssertionError("dimension 2 was accessed")

    class ClassifierOptions(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            if key != "Dimpredictors":
                raise AssertionError(f"oracle field was accessed: {key}")
            return DimensionCells()

    observed1, observed3 = extract_tafazoli_train_dimensions(
        ClassifierOptions()
    )

    np.testing.assert_array_equal(observed1, dim1)
    np.testing.assert_array_equal(observed3, dim3)


def test_official_runner_checksum_guard_precedes_fixed_slicing() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_official_classifier_checksum(Path(__file__))


def test_latent_transform_is_fit_on_training_trials_only() -> None:
    operator = np.diag([0.7, 0.5, 0.3])
    source = _synthetic_dynamics((operator,), seed=41)
    target = source.copy()
    fold = make_whole_trial_folds(12, fold_count=3, seed=20260730)[0]

    prepared = prepare_session_latent_fold(
        source,
        target,
        fold,
        rank_cap=3,
        event_mean_removed=True,
    )
    mutated = source.copy()
    mutated[np.asarray(fold.test_indices)] *= 100.0
    prepared_mutated = prepare_session_latent_fold(
        mutated,
        target,
        fold,
        rank_cap=3,
        event_mean_removed=True,
    )

    np.testing.assert_array_equal(
        prepared.transform.neuron_mean,
        prepared_mutated.transform.neuron_mean,
    )
    np.testing.assert_array_equal(
        prepared.transform.neuron_scale,
        prepared_mutated.transform.neuron_scale,
    )
    np.testing.assert_array_equal(
        prepared.transform.event_time_mean,
        prepared_mutated.transform.event_time_mean,
    )
    np.testing.assert_array_equal(
        prepared.transform.components,
        prepared_mutated.transform.components,
    )


def test_report_is_session_local_deterministic_and_animal_weighted(
    synthetic_report,
) -> None:
    report = synthetic_report
    validate_probe_report(report)

    assert len(report.session_specs) == 2
    assert len(report.within_results) == 4
    assert len(report.transfer_results) == 4
    assert len(report.event_mean_removed_within_results) == 4
    assert len(report.event_mean_removed_transfer_results) == 4
    assert {
        item.session_index_one_based for item in report.within_results
    } == {1, 2}
    assert {item.dimension for item in report.within_results} == {1, 3}
    assert {
        (item.source_dimension, item.target_dimension)
        for item in report.transfer_results
    } == {(1, 3), (3, 1)}
    assert all(
        item.representation_fit_on_source_only
        for item in report.transfer_results
    )
    assert {
        item.animal for item in report.aggregates
    } == {"ALL", "animal_a", "animal_b"}
    assert all(
        item.session_count == 2
        for item in report.aggregates
        if item.animal == "ALL"
    )
    assert all(
        item.session_count == 1
        for item in report.aggregates
        if item.animal != "ALL"
    )
    assert report.blind_fields_used == ()
    assert report.train_only_preprocessing
    assert report.saved_test_role == "not_used"

    rerun = run_tafazoli_session_operator_probe_from_arrays(
        _synthetic_dynamics(
            (
                np.asarray(
                    [
                        [0.72, -0.32, 0.00],
                        [0.32, 0.72, 0.00],
                        [0.00, 0.00, 0.55],
                    ]
                ),
            )
            * 2,
            seed=11,
        ),
        _synthetic_dynamics(
            (
                np.asarray(
                    [
                        [0.72, 0.32, 0.00],
                        [-0.32, 0.72, 0.00],
                        [0.00, 0.00, 0.35],
                    ]
                ),
            )
            * 2,
            seed=29,
        ),
        config=report.config,
    )
    assert json.dumps(
        report.to_dict(),
        sort_keys=True,
        allow_nan=False,
    ) == json.dumps(
        rerun.to_dict(),
        sort_keys=True,
        allow_nan=False,
    )


def test_frozen_transfer_is_separate_from_target_refit(
    synthetic_report,
) -> None:
    report = synthetic_report

    assert all(
        np.isfinite(item.metrics.source_grand_mean_r2)
        for item in report.transfer_results
    )
    assert all(
        item.target_refit_source_grand_mean_r2
        > item.metrics.source_grand_mean_r2
        for item in report.transfer_results
    )
    assert all(
        item.frozen_vs_target_refit_skill < 0.0
        for item in report.transfer_results
    )
    assert {
        item.rank_cap for item in report.rank_stability
    } == {1, 3}


def test_verdict_keys_and_overclaim_locks_are_stable(
    synthetic_report,
) -> None:
    report = synthetic_report

    assert report.verdict(VERDICT_SESSION_LOCAL_SHORT_MEMORY).answer in {
        "YES",
        "NO",
    }
    assert (
        report.verdict(VERDICT_SHARED_STATIONARY_DIRECTED_OPERATOR).answer
        == NO
    )
    assert (
        report.verdict(VERDICT_STATE_DEPENDENT_SWITCHING_OPERATOR).answer
        == PENDING
    )
    assert (
        report.verdict(
            VERDICT_SHARED_CALL_GRAPH_HIERARCHICAL_OPERATOR
        ).answer
        == TEST_UNAVAILABLE
    )
    assert report.verdict(VERDICT_BRAIN_PROGRAMMING_LANGUAGE).answer == NO
    assert not report.claim_locks.shared_call_graph_refuted
    assert not report.claim_locks.hierarchical_operator_refuted
    assert not report.claim_locks.brain_programming_language_identified
    assert {
        item.key for item in report.next_tests
    } == {
        "label_free_switching_operator_k2_k3",
        "event_time_convergence_bottleneck",
        "parent_operator_plus_session_task_residual",
        "common_successor_state",
    }

    with pytest.raises(ValueError, match="claim locks"):
        validate_claim_locks(
            replace(
                ProbeClaimLocks(),
                shared_call_graph_refuted=True,
            )
        )


@pytest.mark.parametrize(
    "bad",
    [
        np.full((6, 2, 21), np.nan),
        np.full((6, 2, 21), -1.0),
        np.zeros((6, 2)),
    ],
)
def test_probe_rejects_invalid_population_tensors(bad: np.ndarray) -> None:
    config = ProbeConfig(
        fold_count=3,
        lag_bins=10,
        successor_shuffle_count=2,
        session_neuron_counts=(2,),
        session_animals=("animal",),
        rank_stability_caps=(1,),
        run_event_mean_removed_sensitivity=False,
    )
    with pytest.raises(ValueError):
        run_tafazoli_session_operator_probe_from_arrays(
            bad,
            np.ones((6, 2, 21)),
            config=config,
        )


def test_stride_ten_sensitivity_uses_fewer_within_trial_pairs() -> None:
    operator = np.diag([0.7, 0.5, 0.3])
    dim1 = _synthetic_dynamics((operator,), seed=71)
    dim3 = _synthetic_dynamics((operator,), seed=83)
    common = dict(
        fold_count=3,
        lag_bins=10,
        rank_cap=3,
        successor_shuffle_count=2,
        session_neuron_counts=(3,),
        session_animals=("animal",),
        rank_stability_caps=(3,),
        run_event_mean_removed_sensitivity=False,
    )
    dense = run_tafazoli_session_operator_probe_from_arrays(
        dim1,
        dim3,
        config=ProbeConfig(transition_stride_bins=1, **common),
    )
    sparse = run_tafazoli_session_operator_probe_from_arrays(
        dim1,
        dim3,
        config=ProbeConfig(transition_stride_bins=10, **common),
    )

    assert dense.within_results[0].metrics.transition_count == 12 * 31
    assert sparse.within_results[0].metrics.transition_count == 12 * 4
    assert (
        dense.verdict(VERDICT_BRAIN_PROGRAMMING_LANGUAGE).answer
        == sparse.verdict(VERDICT_BRAIN_PROGRAMMING_LANGUAGE).answer
        == NO
    )
