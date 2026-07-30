from __future__ import annotations

from dataclasses import asdict, replace
import inspect
import json

import numpy as np
import pytest

import reality_stone.clarus.tafazoli_tree_tournament as tournament
from reality_stone.clarus.tafazoli_call_graph_probe import TrajectoryDesign
from reality_stone.clarus.tafazoli_session_operator_probe import SessionSpec
from reality_stone.clarus.tafazoli_tree_tournament import (
    FAMILY_AXIS_TREE,
    FAMILY_FLAT_SWITCHING,
    FAMILY_MATCHED_VAR,
    FAMILY_OBLIQUE_TREE,
    FAMILY_PARENT_TREE,
    IMPLEMENTED_FAMILIES,
    NO,
    PENDING,
    TREE_FAMILIES,
    CandidateSpec,
    TreeTournamentClaimLocks,
    TreeTournamentConfig,
    enumerate_candidate_specs,
    run_tafazoli_tree_tournament_from_arrays,
    validate_tree_tournament_claim_locks,
    validate_tree_tournament_report,
)


def _small_config() -> TreeTournamentConfig:
    return TreeTournamentConfig(
        families=(
            FAMILY_MATCHED_VAR,
            FAMILY_FLAT_SWITCHING,
            FAMILY_AXIS_TREE,
        ),
        leaf_counts=(2,),
        history_depths=(1,),
        global_anchor_depth=2,
        rank_cap=2,
        lag_bins=2,
        primary_stride_bins=2,
        outer_fold_count=3,
        inner_fold_count=2,
        kmeans_restarts=1,
        kmeans_max_iterations=10,
        split_quantiles=(0.35, 0.5, 0.65),
        minimum_leaf_samples=2,
        minimum_leaf_trials=2,
        run_event_mean_removed_sensitivity=False,
        run_reverse_descriptive_control=False,
    )


def _small_array_report():
    rng = np.random.default_rng(807)
    dim1 = rng.poisson(3.0, size=(12, 4, 13)).astype(np.float64)
    dim3 = rng.poisson(3.0, size=(12, 4, 13)).astype(np.float64)
    specs = (
        SessionSpec(1, "Chico", 0, 2),
        SessionSpec(2, "Silas", 2, 4),
    )
    return run_tafazoli_tree_tournament_from_arrays(
        dim1,
        dim3,
        config=_small_config(),
        session_specs=specs,
    )


@pytest.fixture(scope="module")
def small_array_report():
    return _small_array_report()


def _tree_fixture(
    kind: str,
    *,
    seed: int,
    trial_count: int = 20,
    anchor_count: int = 8,
) -> TrajectoryDesign:
    rng = np.random.default_rng(seed)
    current = rng.uniform(-2.0, 2.0, size=(trial_count, anchor_count, 2))
    noise = rng.normal(0.0, 0.02, size=current.shape)
    if kind == "axis":
        gate = current[..., 0] <= 0.0
        left = np.array(((2.5, 0.0), (0.0, 0.3)))
        right = np.array(((-2.5, 0.0), (0.0, 0.3)))
    elif kind == "oblique":
        gate = current[..., 0] + current[..., 1] <= 0.0
        left = np.array(((2.5, 1.5), (-0.5, 0.3)))
        right = np.array(((-2.5, -1.5), (0.5, 0.3)))
    elif kind == "parent":
        gate = current[..., 0] <= 0.0
        root = np.array(((0.8, 0.2), (-0.1, 0.7)))
        rank_one = np.array(((1.5,), (0.4,))) @ np.array(((0.8, -0.6),))
        left = root + rank_one
        right = root - rank_one
    else:
        raise ValueError(kind)
    left_successor = np.einsum("...i,ij->...j", current, left)
    right_successor = np.einsum("...i,ij->...j", current, right)
    successor = np.where(
        gate[..., None],
        left_successor,
        right_successor,
    ) + noise
    return TrajectoryDesign(
        history=current,
        current=current,
        successor=successor,
        anchor_indices=np.arange(anchor_count, dtype=np.int64),
    )


def _synthetic_config() -> TreeTournamentConfig:
    return TreeTournamentConfig(
        families=IMPLEMENTED_FAMILIES,
        leaf_counts=(2,),
        history_depths=(1,),
        global_anchor_depth=2,
        rank_cap=2,
        lag_bins=1,
        primary_stride_bins=1,
        outer_fold_count=2,
        inner_fold_count=2,
        kmeans_restarts=1,
        kmeans_max_iterations=10,
        minimum_leaf_samples=20,
        minimum_leaf_trials=5,
    )


def _global_affine_test_mse(
    train: TrajectoryDesign,
    test: TrajectoryDesign,
) -> float:
    train_current = train.current.reshape(-1, train.current.shape[-1])
    train_successor = train.successor.reshape(-1, train.successor.shape[-1])
    design = np.column_stack((np.ones(train_current.shape[0]), train_current))
    coefficients = np.linalg.lstsq(design, train_successor, rcond=None)[0]
    test_current = test.current.reshape(-1, test.current.shape[-1])
    test_design = np.column_stack((np.ones(test_current.shape[0]), test_current))
    prediction = test_design @ coefficients
    return float(
        np.mean(
            np.square(
                prediction - test.successor.reshape(prediction.shape),
            )
        )
    )


@pytest.mark.parametrize(
    ("kind", "family", "expected_kind", "maximum_mse_ratio"),
    (
        ("axis", FAMILY_AXIS_TREE, "axis", 0.01),
        (
            "oblique",
            FAMILY_OBLIQUE_TREE,
            "fixed_two_sparse_oblique",
            0.25,
        ),
        ("parent", FAMILY_PARENT_TREE, "axis", 0.70),
    ),
)
def test_each_restricted_tree_family_recovers_its_synthetic_fingerprint(
    kind: str,
    family: str,
    expected_kind: str,
    maximum_mse_ratio: float,
) -> None:
    train = _tree_fixture(kind, seed=1)
    test = _tree_fixture(kind, seed=2)
    predictor = tournament._fit_tree(
        family,
        train,
        leaf_count=2,
        history_depth=1,
        config=_synthetic_config(),
    )
    prediction = tournament._tree_predictions(
        predictor,
        test.history.reshape(-1, 2),
        test.current.reshape(-1, 2),
    ).reshape(test.successor.shape)
    tree_mse = float(np.mean(np.square(prediction - test.successor)))

    assert predictor.nodes[0].split is not None
    assert predictor.nodes[0].split.kind == expected_kind
    assert predictor.discrete_search_bits > 0.0
    assert tree_mse / _global_affine_test_mse(train, test) < maximum_mse_ratio


def _continuous_fixture(
    *,
    seed: int,
    trial_count: int = 40,
    anchor_count: int = 8,
) -> tuple[TrajectoryDesign, TrajectoryDesign]:
    rng = np.random.default_rng(seed)
    current = rng.normal(0.0, 1.0, size=(trial_count, anchor_count, 2))
    older = rng.normal(0.0, 1.0, size=current.shape)
    operator = np.array(((0.8, 0.2), (-0.1, 0.7)))
    successor = np.einsum("...i,ij->...j", current, operator)
    successor += rng.normal(0.0, 0.2, size=current.shape)
    anchors = np.arange(anchor_count, dtype=np.int64)
    tree_design = TrajectoryDesign(
        history=current,
        current=current,
        successor=successor,
        anchor_indices=anchors,
    )
    var_design = TrajectoryDesign(
        history=np.concatenate((current, older), axis=2),
        current=current,
        successor=successor,
        anchor_indices=anchors,
    )
    return tree_design, var_design


def test_continuous_var_null_does_not_reward_tree_search() -> None:
    config = _synthetic_config()
    tree_train, var_train = _continuous_fixture(seed=3)
    tree_test, var_test = _continuous_fixture(seed=4)
    var = tournament._score_var(
        var_train,
        var_test,
        spec=CandidateSpec(FAMILY_MATCHED_VAR, 2, 2),
        config=config,
    )

    for family in TREE_FAMILIES:
        tree = tournament._score_tree(
            tree_train,
            tree_test,
            spec=CandidateSpec(family, 2, 1),
            config=config,
        )
        assert tree.score.total_codelength_bits > var.score.total_codelength_bits


def test_declared_empty_tree_leaf_is_rejected_instead_of_changing_state_count() -> None:
    design = _tree_fixture("axis", seed=9)
    history = design.history.reshape(-1, 2)
    current = design.current.reshape(-1, 2)
    successor = design.successor.reshape(-1, 2)
    split = tournament.SplitRule(
        kind="axis",
        feature_indices=(0,),
        weights=(1.0,),
        threshold=-100.0,
        search_cost_bits=1.0,
    )
    nodes = (
        tournament.TreeNode(0, 0, None, 1, 2, None, split),
        tournament.TreeNode(1, 1, 0, None, None, 0, None),
        tournament.TreeNode(2, 1, 0, None, None, 1, None),
    )

    with pytest.raises(
        tournament._CandidateUnavailable,
        match="no outer-training support",
    ):
        tournament._fit_hard_experts_given_tree(
            FAMILY_AXIS_TREE,
            nodes,
            history,
            current,
            successor,
            history_depth=1,
            ridge_alpha=1.0,
            discrete_search_bits=1.0,
        )


def test_candidate_universe_and_nested_grid_are_finite() -> None:
    config = TreeTournamentConfig()

    assert config.families == IMPLEMENTED_FAMILIES
    assert len(enumerate_candidate_specs(config, FAMILY_MATCHED_VAR)) == 2
    for family in IMPLEMENTED_FAMILIES[1:]:
        assert len(enumerate_candidate_specs(config, family)) == 6


def test_config_requires_independent_baselines_and_nonoverlap() -> None:
    with pytest.raises(ValueError, match="baselines are required"):
        TreeTournamentConfig(
            families=(FAMILY_MATCHED_VAR, FAMILY_AXIS_TREE),
        )
    with pytest.raises(ValueError, match="must not reuse overlapping"):
        TreeTournamentConfig(primary_stride_bins=9)


def test_outer_model_selection_api_cannot_receive_outer_test() -> None:
    parameters = inspect.signature(
        tournament._select_family_inside_outer_train
    ).parameters

    assert tuple(parameters) == (
        "inner_prepared",
        "family",
        "config",
        "seed_tokens",
    )


def test_small_report_is_session_local_nested_blind_and_serializable(
    small_array_report,
) -> None:
    report = small_array_report

    validate_tree_tournament_report(report)
    assert len(report.results) == 12
    assert len(report.aggregates) == 9
    assert {item.dimension for item in report.results} == {1, 3}
    assert {item.session_index_one_based for item in report.results} == {1, 2}
    assert report.blind_fields_used == ()
    assert report.saved_test_role == "not_used"
    assert report.screening_survivors == ()
    assert report.model_relative_winner is None
    assert (
        report.verdict("tested_tree_family_outperformed_flat_baselines").answer
        == PENDING
    )
    assert report.verdict("unique_model_relative_tree_winner").answer == PENDING
    assert report.verdict("brain_executes_winning_tree_algorithm").answer == NO
    assert all(value is False for value in asdict(report.claim_locks).values())
    for result in report.results:
        assert len(result.fold_results) == report.config.outer_fold_count
        assert result.complete_outer_folds
        for fold in result.fold_results:
            assert fold.baselines_independently_nested_selected
            assert not fold.selection.outer_test_used_for_selection
            assert not fold.outer_test_target_used_for_selection_or_gate
            assert not fold.d1_d3_rows_treated_as_paired_trials
    json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )


def test_small_report_is_deterministic(small_array_report) -> None:
    rerun = _small_array_report()

    assert rerun.to_dict() == small_array_report.to_dict()


def test_claim_lock_tampering_is_rejected() -> None:
    validate_tree_tournament_claim_locks(TreeTournamentClaimLocks())

    with pytest.raises(ValueError, match="must remain false"):
        validate_tree_tournament_claim_locks(
            replace(
                TreeTournamentClaimLocks(),
                biological_tree_algorithm_identified=True,
            )
        )
