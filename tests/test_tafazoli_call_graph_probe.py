from __future__ import annotations

from dataclasses import replace
import inspect
import json
from typing import Any

import numpy as np
import pytest

import reality_stone.clarus.tafazoli_call_graph_probe as call_graph
from reality_stone.clarus.tafazoli_call_graph_probe import (
    NO,
    PENDING,
    TEST_UNAVAILABLE,
    YES,
    AggregateResult,
    CallGraphClaimLocks,
    CallGraphProbeConfig,
    FrozenTransferResult,
    PastOnlyGate,
    SessionModelResult,
    TrajectoryDesign,
    aggregate_session_models,
    apply_switching_predictor,
    apply_var_predictor,
    assign_past_only_states,
    build_trajectory_design,
    centroid_parameter_count,
    evaluate_hub_fold,
    fit_past_only_gate,
    fit_state_parent_rank1_predictor,
    fit_switching_predictor,
    fit_var_predictor,
    run_tafazoli_call_graph_probe_from_arrays,
    score_heldout_gaussian_codelength,
    switching_parameter_count,
    validate_call_graph_claim_locks,
    validate_call_graph_probe_report,
    var_parameter_count,
)
from reality_stone.clarus.tafazoli_session_operator_probe import (
    SessionSpec,
    extract_tafazoli_train_dimensions,
    make_whole_trial_folds,
    recovered_session_specs,
)


def _small_array_report():
    rng = np.random.default_rng(481)
    dim1 = rng.poisson(3.0, size=(8, 4, 9)).astype(np.float64)
    dim3 = rng.poisson(3.0, size=(8, 4, 9)).astype(np.float64)
    specs = (
        SessionSpec(1, "Chico", 0, 2),
        SessionSpec(2, "Silas", 2, 4),
    )
    config = CallGraphProbeConfig(
        states=(2,),
        history_depths=(1,),
        rank_cap=2,
        lag_bins=2,
        primary_stride_bins=2,
        fold_count=2,
        kmeans_restarts=2,
        kmeans_max_iterations=20,
        run_event_mean_removed_sensitivity=False,
        run_reverse_descriptive_control=False,
    )
    return run_tafazoli_call_graph_probe_from_arrays(
        dim1,
        dim3,
        config=config,
        session_specs=specs,
    )


@pytest.fixture(scope="module")
def small_array_report():
    return _small_array_report()


def test_trajectory_design_never_crosses_trials_and_keeps_targets_separate() -> None:
    latent = np.arange(2 * 20, dtype=np.float64).reshape(2, 20, 1)

    design = build_trajectory_design(
        latent,
        history_depth=2,
        anchor_history_depth=3,
        lag_bins=3,
        stride_bins=3,
    )

    assert design.anchor_indices.tolist() == [6, 9, 12, 15]
    np.testing.assert_array_equal(design.current, latent[:, [6, 9, 12, 15], :])
    np.testing.assert_array_equal(
        design.history[..., :1],
        latent[:, [6, 9, 12, 15], :],
    )
    np.testing.assert_array_equal(
        design.history[..., 1:],
        latent[:, [3, 6, 9, 12], :],
    )
    np.testing.assert_array_equal(
        design.successor,
        latent[:, [9, 12, 15, 18], :],
    )
    assert np.all(design.current[0] < latent[1, 0, 0])
    assert np.all(design.successor[0] < latent[1, 0, 0])


def test_whole_trial_folds_are_disjoint_complete_and_deterministic() -> None:
    first = make_whole_trial_folds(11, fold_count=3, seed=20260730)
    second = make_whole_trial_folds(11, fold_count=3, seed=20260730)

    assert first == second
    held_out = []
    for fold in first:
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
        assert set(fold.train_indices) | set(fold.test_indices) == set(range(11))
        held_out.extend(fold.test_indices)
    assert sorted(held_out) == list(range(11))


def test_extractor_blocks_dimension_two_labels_factors_and_saved_test() -> None:
    dim1 = np.ones((4, 2, 12))
    dim3 = np.full((4, 2, 12), 3.0)

    class TrainOnlyCell:
        def __init__(self, train: np.ndarray):
            self.train = train

        def __getitem__(self, index: int) -> np.ndarray:
            if index != 0:
                raise AssertionError("saved classifier test set was accessed")
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
                raise AssertionError(f"label or factor field was accessed: {key}")
            return DimensionCells()

    observed1, observed3 = extract_tafazoli_train_dimensions(ClassifierOptions())

    np.testing.assert_array_equal(observed1, dim1)
    np.testing.assert_array_equal(observed3, dim3)


def test_state_count_and_history_depth_are_independent_axes() -> None:
    rng = np.random.default_rng(71)
    latent = rng.normal(size=(5, 25, 3))

    short = build_trajectory_design(
        latent,
        history_depth=1,
        anchor_history_depth=3,
        lag_bins=4,
        stride_bins=4,
    )
    deep = build_trajectory_design(
        latent,
        history_depth=3,
        anchor_history_depth=3,
        lag_bins=4,
        stride_bins=4,
    )

    np.testing.assert_array_equal(short.anchor_indices, deep.anchor_indices)
    np.testing.assert_array_equal(short.current, deep.current)
    np.testing.assert_array_equal(short.successor, deep.successor)
    assert short.history.shape[-1] == 3
    assert deep.history.shape[-1] == 9
    assert switching_parameter_count(2, 3) == var_parameter_count(2, 3)
    assert switching_parameter_count(3, 3) == var_parameter_count(3, 3)
    assert centroid_parameter_count(2, 1, 3) == 6
    assert centroid_parameter_count(2, 3, 3) == 18


def test_gate_api_is_past_only_and_future_poison_cannot_change_assignments() -> None:
    rng = np.random.default_rng(902)
    history = rng.normal(size=(9, 7, 4))
    future = rng.normal(size=(9, 7, 2))
    poisoned_future = future + 1e9

    gate = fit_past_only_gate(
        history,
        state_count=3,
        history_depth=2,
        latent_rank=2,
        seed=18,
        restarts=3,
        max_iterations=30,
    )
    before = assign_past_only_states(gate, history)
    _ = poisoned_future
    after = assign_past_only_states(gate, history)

    assert "successor" not in inspect.signature(fit_past_only_gate).parameters
    assert "target" not in inspect.signature(fit_past_only_gate).parameters
    assert "successor" not in inspect.signature(assign_past_only_states).parameters
    assert "target" not in inspect.signature(assign_past_only_states).parameters
    np.testing.assert_array_equal(before, after)


def test_gate_and_full_small_report_are_deterministic(small_array_report) -> None:
    rng = np.random.default_rng(119)
    history = rng.normal(size=(8, 5, 2))
    left = fit_past_only_gate(
        history,
        state_count=2,
        history_depth=1,
        latent_rank=2,
        seed=17,
        restarts=4,
    )
    right = fit_past_only_gate(
        history,
        state_count=2,
        history_depth=1,
        latent_rank=2,
        seed=17,
        restarts=4,
    )

    np.testing.assert_array_equal(left.centroids, right.centroids)
    assert left.training_inertia == right.training_inertia
    rerun = _small_array_report()
    assert json.dumps(
        small_array_report.to_dict(),
        sort_keys=True,
        allow_nan=False,
    ) == json.dumps(rerun.to_dict(), sort_keys=True, allow_nan=False)


def _switching_tensors(
    rng: np.random.Generator,
    *,
    trials: int,
    anchors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = rng.integers(0, 2, size=(trials, anchors))
    centers = np.asarray(((-2.5, 0.5), (2.5, -0.5)))
    current = centers[states] + rng.normal(scale=0.35, size=(trials, anchors, 2))
    past = rng.normal(size=(trials, anchors, 2))
    operators = np.asarray(
        (
            ((0.8, 0.7), (-0.5, 0.6)),
            ((-0.7, 0.5), (0.8, 0.4)),
        )
    )
    successor = np.einsum("...i,...ij->...j", current, operators[states])
    successor += rng.normal(scale=0.04, size=successor.shape)
    return current, past, successor, states


def test_true_switching_beats_parameter_matched_stationary_var() -> None:
    train_rng = np.random.default_rng(411)
    test_rng = np.random.default_rng(877)
    train_current, train_past, train_y, _ = _switching_tensors(
        train_rng,
        trials=70,
        anchors=8,
    )
    test_current, test_past, test_y, _ = _switching_tensors(
        test_rng,
        trials=30,
        anchors=8,
    )
    gate = fit_past_only_gate(
        train_current,
        state_count=2,
        history_depth=1,
        latent_rank=2,
        seed=1337,
        restarts=6,
    )
    train_states = assign_past_only_states(gate, train_current)
    test_states = assign_past_only_states(gate, test_current)
    switching = fit_switching_predictor(
        train_current,
        train_y,
        train_states,
        state_count=2,
        ridge_alpha=0.01,
    )
    train_history = np.concatenate((train_current, train_past), axis=2)
    test_history = np.concatenate((test_current, test_past), axis=2)
    stationary = fit_var_predictor(
        train_history,
        train_y,
        history_order=2,
        latent_rank=2,
        ridge_alpha=0.01,
    )
    switching_score = score_heldout_gaussian_codelength(
        family="synthetic_switching",
        train_observed=train_y,
        train_predicted=apply_switching_predictor(
            switching,
            train_current,
            train_states,
        ),
        test_observed=test_y,
        test_predicted=apply_switching_predictor(
            switching,
            test_current,
            test_states,
        ),
        dynamic_parameter_count=switching.parameter_count,
        gate_parameter_count=centroid_parameter_count(2, 1, 2),
        model_selection_cost_bits=0.0,
    )
    var_score = score_heldout_gaussian_codelength(
        family="synthetic_var",
        train_observed=train_y,
        train_predicted=apply_var_predictor(stationary, train_history),
        test_observed=test_y,
        test_predicted=apply_var_predictor(stationary, test_history),
        dynamic_parameter_count=stationary.parameter_count,
        gate_parameter_count=0,
        model_selection_cost_bits=0.0,
    )

    assert switching.parameter_count == stationary.parameter_count
    assert switching_score.test_sse < 0.05 * var_score.test_sse
    assert switching_score.total_codelength_bits < var_score.total_codelength_bits


def _hub_fixture(
    *,
    seed: int,
    trials: int,
    time_funnel: bool,
) -> tuple[TrajectoryDesign, np.ndarray]:
    rng = np.random.default_rng(seed)
    anchors = np.arange(6, dtype=np.int64)
    states = np.empty((trials, 6), dtype=np.int64)
    states[:, 0] = np.arange(trials) % 2 * 2
    states[:, 1] = 1
    states[:, 2] = 2
    states[:, 3] = (np.arange(trials) + 1) % 2 * 2
    states[:, 4] = 1
    states[:, 5] = 2
    current = rng.normal(size=(trials, 6, 2))
    if time_funnel:
        time_values = np.asarray(
            (
                (-7.0, 7.0),
                (-5.0, 5.0),
                (-3.0, 3.0),
                (3.0, -3.0),
                (5.0, -5.0),
                (7.0, -7.0),
            )
        )
        successor = np.broadcast_to(time_values, current.shape).copy()
        successor += rng.normal(scale=0.01, size=successor.shape)
    else:
        operator = np.asarray(((0.8, -0.4), (0.5, 0.7)))
        successor = current @ operator
        successor += rng.normal(scale=0.03, size=successor.shape)
    return (
        TrajectoryDesign(
            history=current.copy(),
            current=current,
            successor=successor,
            anchor_indices=anchors,
        ),
        states,
    )


def _hub_config() -> CallGraphProbeConfig:
    return CallGraphProbeConfig(
        states=(3,),
        history_depths=(1,),
        rank_cap=2,
        lag_bins=1,
        primary_stride_bins=1,
        fold_count=2,
        ridge_alpha=0.01,
        kmeans_restarts=1,
        run_event_mean_removed_sensitivity=False,
        run_reverse_descriptive_control=False,
    )


def _dummy_hub_gate() -> PastOnlyGate:
    return PastOnlyGate(
        state_count=3,
        history_depth=1,
        latent_rank=2,
        centroids=np.zeros((3, 2)),
        training_inertia=0.0,
    )


def test_event_time_funnel_does_not_pass_as_shared_hub_dynamics() -> None:
    train_design, train_states = _hub_fixture(
        seed=110,
        trials=80,
        time_funnel=True,
    )
    test_design, test_states = _hub_fixture(
        seed=210,
        trials=40,
        time_funnel=True,
    )

    result = evaluate_hub_fold(
        train_design=train_design,
        test_design=test_design,
        train_states=train_states,
        test_states=test_states,
        gate=_dummy_hub_gate(),
        config=_hub_config(),
    )

    assert result.available
    assert result.hub_state == 1
    assert result.distinct_train_callers == 2
    assert result.shared is not None
    assert result.time_locked is not None
    assert result.time_locked.test_sse < 0.01 * result.shared.test_sse
    assert (
        result.time_locked.total_codelength_bits
        < result.shared.total_codelength_bits
    )


def test_true_shared_hub_dynamics_beats_time_locked_funnel() -> None:
    train_design, train_states = _hub_fixture(
        seed=310,
        trials=100,
        time_funnel=False,
    )
    test_design, test_states = _hub_fixture(
        seed=410,
        trials=50,
        time_funnel=False,
    )

    result = evaluate_hub_fold(
        train_design=train_design,
        test_design=test_design,
        train_states=train_states,
        test_states=test_states,
        gate=_dummy_hub_gate(),
        config=_hub_config(),
    )

    assert result.available
    assert result.shared is not None
    assert result.time_locked is not None
    assert result.shared.test_sse < 0.05 * result.time_locked.test_sse
    assert (
        result.shared.total_codelength_bits
        < result.time_locked.total_codelength_bits
    )


def _aggregate_fixture(
    *,
    animal: str,
    event_mean_removed: bool,
    complete_hubs: bool = True,
) -> AggregateResult:
    return AggregateResult(
        state_count=2,
        history_depth=2,
        event_mean_removed=event_mean_removed,
        animal=animal,
        unit_count=54 if animal == "all" else 18,
        median_switching_vs_var_skill=0.2,
        median_switching_codelength_advantage_bits_per_scalar=0.2,
        median_forward_vs_reverse_skill=0.2,
        median_forward_codelength_advantage_over_reverse_bits_per_scalar=0.2,
        median_state_parent_rank1_vs_var_skill=0.2,
        median_state_parent_rank1_codelength_advantage_bits_per_scalar=0.2,
        all_units_have_complete_hub_folds=complete_hubs,
        median_hub_shared_vs_time_skill=0.2,
        median_hub_shared_codelength_advantage_over_time_bits_per_scalar=0.2,
        median_hub_shared_codelength_advantage_over_caller_bits_per_scalar=0.2,
        median_hub_forward_vs_reverse_skill=0.2,
    )


def _transfer_fixtures(*, positive: bool) -> tuple[FrozenTransferResult, ...]:
    value = 0.2 if positive else -0.2
    results = []
    for event_mean_removed in (False, True):
        for session_index, animal in ((1, "Chico"), (2, "Silas")):
            for source, target in ((1, 3), (3, 1)):
                results.append(
                    FrozenTransferResult(
                        analysis_key=f"frozen_dim{source}_to_dim{target}_states2",
                        session_index_one_based=session_index,
                        animal=animal,
                        neuron_count=3,
                        source_dimension=source,
                        target_dimension=target,
                        state_count=2,
                        history_depth=2,
                        event_mean_removed=event_mean_removed,
                        frozen_test_sse=0.8 if positive else 1.2,
                        target_refit_test_sse=1.0,
                        frozen_vs_target_refit_skill=value,
                        frozen_codelength_advantage_over_target_refit_bits_per_scalar=value,
                        source_representation_and_gate_frozen=True,
                        target_rows_paired_to_source_rows=False,
                    )
                )
    return tuple(results)


def _complete_aggregate_fixtures() -> tuple[AggregateResult, ...]:
    return tuple(
        _aggregate_fixture(
            animal=animal,
            event_mean_removed=event_mean_removed,
        )
        for event_mean_removed in (False, True)
        for animal in ("all", "Chico", "Silas")
    )


@pytest.mark.parametrize(
    ("transfer_positive", "expected_candidate"),
    ((False, NO), (True, YES)),
)
def test_frontend_common_callee_candidate_requires_hub_and_frozen_transfer(
    transfer_positive: bool,
    expected_candidate: str,
) -> None:
    verdicts = call_graph._build_verdicts(
        _complete_aggregate_fixtures(),
        _transfer_fixtures(positive=transfer_positive),
        config=CallGraphProbeConfig(
            states=(2,),
            history_depths=(2,),
        ),
    )
    answers = {item.key: item.answer for item in verdicts}

    assert answers["latent_common_successor_proxy_passed"] == YES
    assert (
        answers["frontend_to_common_callee_observational_candidate_supported"]
        == expected_candidate
    )
    assert answers["biological_common_callee_assembly_identified"] == NO
    assert (
        answers["common_callee_architecture_exists_or_is_absent"]
        == TEST_UNAVAILABLE
    )


def test_frontend_common_callee_candidate_cannot_mix_different_state_counts() -> None:
    hub_only_k2 = _complete_aggregate_fixtures()
    transfer_only_k3_aggregates = tuple(
        replace(
            item,
            state_count=3,
            history_depth=3,
            median_hub_shared_vs_time_skill=-0.2,
            median_hub_shared_codelength_advantage_over_time_bits_per_scalar=-0.2,
        )
        for item in _complete_aggregate_fixtures()
    )
    failed_k2_transfer = _transfer_fixtures(positive=False)
    passing_k3_transfer = tuple(
        replace(item, state_count=3, history_depth=3)
        for item in _transfer_fixtures(positive=True)
    )

    verdicts = call_graph._build_verdicts(
        (*hub_only_k2, *transfer_only_k3_aggregates),
        (*failed_k2_transfer, *passing_k3_transfer),
        config=CallGraphProbeConfig(
            states=(2, 3),
            history_depths=(2, 3),
        ),
    )
    answers = {item.key: item.answer for item in verdicts}

    assert answers["latent_common_successor_proxy_passed"] == YES
    assert answers["frozen_cross_dimension_switching_transfer_passed"] == YES
    assert (
        answers["frontend_to_common_callee_observational_candidate_supported"]
        == NO
    )


def test_unavailable_hub_fold_cannot_be_silently_dropped_from_pass() -> None:
    aggregates = list(_complete_aggregate_fixtures())
    aggregates[0] = replace(
        aggregates[0],
        all_units_have_complete_hub_folds=False,
    )

    verdicts = call_graph._build_verdicts(
        aggregates,
        _transfer_fixtures(positive=True),
        config=CallGraphProbeConfig(
            states=(2,),
            history_depths=(2,),
        ),
    )
    answers = {item.key: item.answer for item in verdicts}

    assert answers["latent_common_successor_proxy_passed"] == NO
    assert (
        answers["frontend_to_common_callee_observational_candidate_supported"]
        == NO
    )


def _empty_session_result(
    *,
    session: SessionSpec,
    dimension: int,
    state_count: int,
    history_depth: int,
) -> SessionModelResult:
    return SessionModelResult(
        analysis_key=(
            f"within_dim{dimension}_states{state_count}_history{history_depth}"
        ),
        session_index_one_based=session.index_one_based,
        animal=session.animal,
        neuron_count=session.neuron_count,
        dimension=dimension,
        state_count=state_count,
        history_depth=history_depth,
        var_order=state_count,
        parameter_matched_dynamic_block=True,
        event_mean_removed=False,
        stride_bins=10,
        fold_results=(),
        switching_vs_var_skill=0.0,
        switching_codelength_advantage_bits_per_scalar=0.0,
        forward_vs_reverse_skill=None,
        forward_codelength_advantage_over_reverse_bits_per_scalar=None,
        state_parent_rank1_vs_var_skill=0.0,
        state_parent_rank1_codelength_advantage_bits_per_scalar=0.0,
        hub_available_fold_count=0,
        hub_total_fold_count=6,
        hub_all_folds_available=False,
        hub_shared_vs_time_skill=None,
        hub_shared_codelength_advantage_over_time_bits_per_scalar=None,
        hub_shared_codelength_advantage_over_caller_bits_per_scalar=None,
        hub_forward_vs_reverse_skill=None,
        hub_forward_codelength_advantage_over_reverse_bits_per_scalar=None,
    )


def test_each_official_state_history_aggregate_has_54_session_dimension_units() -> None:
    specs = recovered_session_specs()
    results = tuple(
        _empty_session_result(
            session=session,
            dimension=dimension,
            state_count=state_count,
            history_depth=history_depth,
        )
        for session in specs
        for dimension in (1, 3)
        for state_count in (2, 3)
        for history_depth in (1, 2, 3)
    )

    aggregates = aggregate_session_models(results)
    all_animal = tuple(item for item in aggregates if item.animal == "all")

    assert len(specs) * 2 == 54
    assert len(all_animal) == 6
    assert {
        (item.state_count, item.history_depth, item.unit_count)
        for item in all_animal
    } == {
        (state_count, history_depth, 54)
        for state_count in (2, 3)
        for history_depth in (1, 2, 3)
    }


def test_state_parent_is_not_task_inheritance_and_claim_locks_hold(
    small_array_report,
) -> None:
    rng = np.random.default_rng(144)
    current = rng.normal(size=(8, 5, 2))
    successor = rng.normal(size=(8, 5, 2))
    states = rng.integers(0, 2, size=(8, 5))
    predictor = fit_state_parent_rank1_predictor(
        current,
        successor,
        states,
        state_count=2,
        ridge_alpha=1.0,
    )

    assert predictor.family == "state_parent_plus_rank1_residual"
    assert "not" in (fit_state_parent_rank1_predictor.__doc__ or "")
    assert "task inheritance" in (fit_state_parent_rank1_predictor.__doc__ or "")
    assert (
        small_array_report.verdict("task_inheritance_tree_identified").answer
        == NO
    )
    assert (
        small_array_report.verdict(
            "task_inheritance_architecture_exists_or_is_absent"
        ).answer
        == TEST_UNAVAILABLE
    )
    with pytest.raises(ValueError, match="claim lock"):
        validate_call_graph_claim_locks(
            replace(CallGraphClaimLocks(), task_inheritance_identified=True)
        )


def test_array_report_is_session_local_row_unpaired_and_claim_locked(
    small_array_report,
) -> None:
    report = small_array_report
    validate_call_graph_probe_report(report)

    assert len(report.session_specs) == 2
    assert len(report.model_results) == 4
    assert {item.dimension for item in report.model_results} == {1, 3}
    assert {
        item.session_index_one_based for item in report.model_results
    } == {1, 2}
    assert all(
        item.neuron_count
        == report.session_specs[item.session_index_one_based - 1].neuron_count
        for item in report.model_results
    )
    assert report.blind_fields_used == ()
    assert report.saved_test_role == "not_used"
    assert not report.claim_locks.d1_d3_rows_treated_as_paired_trials
    assert all(
        item.source_representation_and_gate_frozen
        and not item.target_rows_paired_to_source_rows
        for item in report.frozen_transfer_results
    )
    assert all(
        fold.gate_uses_current_and_past_only
        and not fold.test_target_passed_to_gate
        for item in report.model_results
        for fold in item.fold_results
    )
    assert (
        report.verdict("brain_programming_language_identified").answer == NO
    )
    assert (
        report.verdict(
            "frontend_to_common_callee_observational_candidate_supported"
        ).answer
        == PENDING
    )
