"""Mouse IBL/OpenAlyx temporal unit-coupling gate.

The block-regularized unit gate showed that unit detail survives after
task/history and hybrid region bins when the unit block gets its own ridge
penalty.  This gate asks the next question: is the surviving unit signal only a
static same-window decoder, or does a lagged unit block also carry held-out
behavior information?

For choice and first-movement speed:

    source window: pre-stimulus [-300, 0] ms
    target window: stimulus [20, 320] ms

For wheel direction:

    source window: pre-movement [-400, -100] ms
    target window: first movement [-100, 200] ms

The comparison is nested and fold-safe:

    M_XR:       y ~ [X_task, R_target]
    M_XRU0:     y ~ [X_task, R_target, U_target]
    M_XRUL:     y ~ [X_task, R_target, U_lag]
    M_XRU0UL:   y ~ [X_task, R_target, U_target, U_lag]

The key residuals are:

    Delta_{U_lag | X,R}       = BA(M_XRUL) - BA(M_XR)
    Delta_{U_lag | X,R,U0}    = BA(M_XRU0UL) - BA(M_XRU0)

This is a temporal GLM-style regularized decoder, not a causal connectivity
claim.  A positive replicated result means the next stronger biological model
should estimate directed trial-time coupling rather than only static unit
identity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_all_unit_nested_regularization_gate import all_unit_feature_block
from mouse_ibl_channel_region_rescue_gate import (
    CANDIDATES,
    load_probe,
    make_models as make_region_models,
    probe_feature_block,
    summarize_probe,
)
from mouse_ibl_lagged_coupling_proxy_gate import PRE_MOVEMENT_WINDOW, PRE_STIMULUS_WINDOW
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    WindowSpec,
    auc_score,
    balanced_accuracy,
    choice_target,
    class_counts,
    first_movement_speed_target,
    stratified_folds,
    wheel_action_direction_target,
    window_bounds,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates


RESULT_JSON = Path(__file__).with_name("mouse_ibl_temporal_glm_coupling_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_temporal_glm_coupling_report.md")


def finite_rows(blocks: dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for value in blocks.values():
        current = np.all(np.isfinite(value), axis=1)
        mask = current if mask is None else mask & current
    if mask is None:
        raise ValueError("no feature blocks supplied")
    return mask


def zscore_block(
    train_block: np.ndarray,
    test_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_block, axis=0)
    scale = np.nanstd(train_block, axis=0)
    scale[scale < 1e-9] = 1.0
    return (train_block - mean) / scale, (test_block - mean) / scale


def block_ridge_fold_scores(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
) -> np.ndarray:
    values = np.unique(y)
    if len(values) != 2:
        raise ValueError("block ridge expects a binary target")
    y_signed = np.where(y == values[1], 1.0, -1.0)
    train_parts = []
    test_parts = []
    penalty_values = []
    for name in block_names:
        train_block, test_block = zscore_block(blocks[name][train], blocks[name][test])
        train_parts.append(train_block)
        test_parts.append(test_block)
        penalty_values.extend([float(penalties[name])] * train_block.shape[1])
    x_train = np.column_stack([np.ones(len(train)), hstack(train_parts)])
    x_test = np.column_stack([np.ones(len(test)), hstack(test_parts)])
    penalty = np.diag([0.0, *penalty_values])
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_signed[train]
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_test @ weights


def choose_unit_penalty(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    outer_train: np.ndarray,
    block_names: list[str],
    base_penalties: dict[str, float],
    unit_block_names: list[str],
    unit_penalties: list[float],
    inner_folds: int,
    seed: int,
) -> float:
    y_train = y[outer_train]
    inner = stratified_folds(y_train, inner_folds, seed)
    best_penalty = float(unit_penalties[0])
    best_ba = -1.0
    for unit_penalty in unit_penalties:
        scores = np.zeros(len(outer_train), dtype=float)
        penalties = dict(base_penalties)
        for name in unit_block_names:
            penalties[name] = float(unit_penalty)
        for inner_test_local in inner:
            inner_train_local = np.setdiff1d(
                np.arange(len(outer_train)),
                inner_test_local,
            )
            inner_train = outer_train[inner_train_local]
            inner_test = outer_train[inner_test_local]
            scores[inner_test_local] = block_ridge_fold_scores(
                blocks,
                y,
                inner_train,
                inner_test,
                block_names,
                penalties,
            )
        predicted = (scores >= 0).astype(int)
        current_ba = balanced_accuracy(y_train, predicted)
        if current_ba > best_ba + 1e-12 or (
            abs(current_ba - best_ba) <= 1e-12 and unit_penalty > best_penalty
        ):
            best_ba = current_ba
            best_penalty = float(unit_penalty)
    return best_penalty


def scores_for_model(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    block_names: list[str],
    base_penalties: dict[str, float],
    unit_block_names: list[str],
    unit_penalties: list[float],
    folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    scores = np.zeros(len(y), dtype=float)
    selected: list[float] = []
    for outer_idx, test in enumerate(stratified_folds(y, folds, seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        penalties = dict(base_penalties)
        if unit_block_names:
            unit_penalty = choose_unit_penalty(
                blocks,
                y,
                train,
                block_names,
                base_penalties,
                unit_block_names,
                unit_penalties,
                inner_folds,
                seed + 1009 + outer_idx,
            )
            selected.append(unit_penalty)
            for name in unit_block_names:
                penalties[name] = unit_penalty
        scores[test] = block_ridge_fold_scores(
            blocks,
            y,
            train,
            test,
            block_names,
            penalties,
        )
    return scores, selected


def metrics_from_scores(
    target_name: str,
    window_name: str,
    model_name: str,
    feature_count: int,
    y: np.ndarray,
    scores: np.ndarray,
    selected_unit_penalties: list[float],
) -> dict[str, object]:
    predicted = (scores >= 0).astype(int)
    counts = class_counts(y)
    majority_accuracy = max(float(np.mean(y == value)) for value in np.unique(y))
    penalty_counts = Counter(f"{value:g}" for value in selected_unit_penalties)
    return {
        "target": target_name,
        "window": window_name,
        "model": model_name,
        "feature_count": int(feature_count),
        "n_trials": int(len(y)),
        "class_counts": counts,
        "majority_accuracy": majority_accuracy,
        "balanced_accuracy": balanced_accuracy(y, predicted),
        "auc": auc_score(y, scores),
        "selected_unit_penalties": selected_unit_penalties,
        "selected_unit_penalty_counts": dict(sorted(penalty_counts.items())),
        "selected_unit_penalty_median": (
            float(np.median(selected_unit_penalties))
            if selected_unit_penalties
            else None
        ),
    }


def evaluate_temporal_target(
    *,
    target_name: str,
    window_name: str,
    x_block: np.ndarray,
    r_target: np.ndarray,
    u_target: np.ndarray,
    u_lag: np.ndarray,
    y_all: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    blocks_all = {
        "X": x_block,
        "R": r_target,
        "U0": u_target,
        "UL": u_lag,
    }
    valid = valid & finite_rows(blocks_all)
    y = np.asarray(y_all[valid], dtype=int)
    blocks = {name: value[valid] for name, value in blocks_all.items()}
    unit_grid = [float(value) for value in args.unit_penalties]

    model_specs = [
        (
            "task_history_plus_hybrid_region",
            ["X", "R"],
            {"X": args.task_penalty, "R": args.region_penalty},
            [],
        ),
        (
            "task_history_plus_hybrid_region_plus_current_unit",
            ["X", "R", "U0"],
            {"X": args.task_penalty, "R": args.region_penalty},
            ["U0"],
        ),
        (
            "task_history_plus_hybrid_region_plus_lagged_unit",
            ["X", "R", "UL"],
            {"X": args.task_penalty, "R": args.region_penalty},
            ["UL"],
        ),
        (
            "task_history_plus_hybrid_region_plus_current_and_lagged_unit",
            ["X", "R", "U0", "UL"],
            {"X": args.task_penalty, "R": args.region_penalty},
            ["U0", "UL"],
        ),
    ]

    rows = []
    for model_name, block_names, penalties, unit_block_names in model_specs:
        scores, selected = scores_for_model(
            blocks,
            y,
            block_names,
            penalties,
            unit_block_names,
            unit_grid,
            args.folds,
            args.inner_folds,
            args.seed,
        )
        rows.append(
            metrics_from_scores(
                target_name,
                window_name,
                model_name,
                sum(blocks[name].shape[1] for name in block_names),
                y,
                scores,
                selected,
            )
        )
    return annotate_target({"target": target_name, "window": window_name, "rows": rows}, args.min_delta)


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    xr = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    xru0 = by_model["task_history_plus_hybrid_region_plus_current_unit"]["balanced_accuracy"]
    xrul = by_model["task_history_plus_hybrid_region_plus_lagged_unit"]["balanced_accuracy"]
    xru0ul = by_model[
        "task_history_plus_hybrid_region_plus_current_and_lagged_unit"
    ]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_task_region"] = float(row["balanced_accuracy"] - xr)
        row["delta_vs_task_region_current_unit"] = float(row["balanced_accuracy"] - xru0)
    target["temporal_comparison"] = {
        "task_history_plus_hybrid_region_balanced_accuracy": xr,
        "task_history_plus_hybrid_region_plus_current_unit_balanced_accuracy": xru0,
        "task_history_plus_hybrid_region_plus_lagged_unit_balanced_accuracy": xrul,
        "task_history_plus_hybrid_region_plus_current_and_lagged_unit_balanced_accuracy": xru0ul,
        "current_unit_increment_after_task_region": float(xru0 - xr),
        "lagged_unit_increment_after_task_region": float(xrul - xr),
        "lagged_unit_increment_after_task_region_current_unit": float(xru0ul - xru0),
        "current_unit_increment_after_task_region_lagged_unit": float(xru0ul - xrul),
        "lagged_unit_after_task_region": bool(xrul > xr + min_delta),
        "lagged_unit_after_task_region_current_unit": bool(xru0ul > xru0 + min_delta),
        "selected_unit_penalty_counts_current": by_model[
            "task_history_plus_hybrid_region_plus_current_unit"
        ]["selected_unit_penalty_counts"],
        "selected_unit_penalty_counts_lagged": by_model[
            "task_history_plus_hybrid_region_plus_lagged_unit"
        ]["selected_unit_penalty_counts"],
        "selected_unit_penalty_counts_current_lagged": by_model[
            "task_history_plus_hybrid_region_plus_current_and_lagged_unit"
        ]["selected_unit_penalty_counts"],
        "selected_unit_penalty_median_current": by_model[
            "task_history_plus_hybrid_region_plus_current_unit"
        ]["selected_unit_penalty_median"],
        "selected_unit_penalty_median_lagged": by_model[
            "task_history_plus_hybrid_region_plus_lagged_unit"
        ]["selected_unit_penalty_median"],
        "selected_unit_penalty_median_current_lagged": by_model[
            "task_history_plus_hybrid_region_plus_current_and_lagged_unit"
        ]["selected_unit_penalty_median"],
    }
    return target


def region_features_for_window(
    probes: list[dict[str, object]],
    trials,
    spec: WindowSpec,
    min_label_spikes: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    starts, ends, valid = window_bounds(trials, spec)
    blocks = [
        probe_feature_block(probe, starts, ends, valid, min_label_spikes)
        for probe in probes
    ]
    models = make_region_models(blocks)
    return models["hybrid_acronym_channel_id_by_probe"], blocks, valid


def unit_features_for_window(
    probes: list[dict[str, object]],
    trials,
    spec: WindowSpec,
    min_unit_spikes: int,
    max_units_per_probe: int,
) -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    starts, ends, valid = window_bounds(trials, spec)
    blocks = [
        all_unit_feature_block(
            probe,
            starts,
            ends,
            valid,
            min_unit_spikes,
            max_units_per_probe,
        )
        for probe in probes
    ]
    unit = hstack([block["all_unit_features"] for block in blocks])
    return unit, blocks, valid


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    task_history, task_history_names = task_history_covariates(trials)

    stim_region, stim_region_blocks, stim_region_valid = region_features_for_window(
        probes,
        trials,
        STIMULUS_WINDOW,
        args.min_label_spikes,
    )
    move_region, move_region_blocks, move_region_valid = region_features_for_window(
        probes,
        trials,
        MOVEMENT_WINDOW,
        args.min_label_spikes,
    )
    stim_unit, stim_unit_blocks, stim_unit_valid = unit_features_for_window(
        probes,
        trials,
        STIMULUS_WINDOW,
        args.min_unit_spikes,
        args.max_units_per_probe,
    )
    move_unit, move_unit_blocks, move_unit_valid = unit_features_for_window(
        probes,
        trials,
        MOVEMENT_WINDOW,
        args.min_unit_spikes,
        args.max_units_per_probe,
    )
    pre_stim_unit, pre_stim_unit_blocks, pre_stim_unit_valid = unit_features_for_window(
        probes,
        trials,
        PRE_STIMULUS_WINDOW,
        args.min_unit_spikes,
        args.max_units_per_probe,
    )
    pre_move_unit, pre_move_unit_blocks, pre_move_unit_valid = unit_features_for_window(
        probes,
        trials,
        PRE_MOVEMENT_WINDOW,
        args.min_unit_spikes,
        args.max_units_per_probe,
    )

    choice, choice_valid, choice_meta = choice_target(trials)
    speed, speed_valid, speed_meta = first_movement_speed_target(trials)
    wheel_direction, wheel_valid, wheel_meta = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )

    targets = [
        evaluate_temporal_target(
            target_name="choice_sign",
            window_name=f"{PRE_STIMULUS_WINDOW.name} -> {STIMULUS_WINDOW.name}",
            x_block=task_history,
            r_target=stim_region,
            u_target=stim_unit,
            u_lag=pre_stim_unit,
            y_all=choice,
            valid=choice_valid & stim_region_valid & stim_unit_valid & pre_stim_unit_valid,
            args=args,
        ),
        evaluate_temporal_target(
            target_name="first_movement_speed",
            window_name=f"{PRE_STIMULUS_WINDOW.name} -> {STIMULUS_WINDOW.name}",
            x_block=task_history,
            r_target=stim_region,
            u_target=stim_unit,
            u_lag=pre_stim_unit,
            y_all=speed,
            valid=speed_valid & stim_region_valid & stim_unit_valid & pre_stim_unit_valid,
            args=args,
        ),
        evaluate_temporal_target(
            target_name="wheel_action_direction",
            window_name=f"{PRE_MOVEMENT_WINDOW.name} -> {MOVEMENT_WINDOW.name}",
            x_block=task_history,
            r_target=move_region,
            u_target=move_unit,
            u_lag=pre_move_unit,
            y_all=wheel_direction,
            valid=wheel_valid & move_region_valid & move_unit_valid & pre_move_unit_valid,
            args=args,
        ),
    ]

    unit_summaries = [
        {
            "collection": probe["collection"],
            "stimulus_selected_unit_count": stim_block["selected_unit_count"],
            "movement_selected_unit_count": move_block["selected_unit_count"],
            "pre_stimulus_selected_unit_count": pre_stim_block["selected_unit_count"],
            "pre_movement_selected_unit_count": pre_move_block["selected_unit_count"],
            "stimulus_unit_feature_count": stim_block["feature_count"],
            "movement_unit_feature_count": move_block["feature_count"],
            "pre_stimulus_unit_feature_count": pre_stim_block["feature_count"],
            "pre_movement_unit_feature_count": pre_move_block["feature_count"],
        }
        for probe, stim_block, move_block, pre_stim_block, pre_move_block in zip(
            probes,
            stim_unit_blocks,
            move_unit_blocks,
            pre_stim_unit_blocks,
            pre_move_unit_blocks,
        )
    ]
    return {
        "openalyx_url": OPENALYX_URL,
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "wheel_sample_count": int(len(wheel_timestamps)),
        "probe_summaries": [
            summarize_probe(probe, block)
            for probe, block in zip(probes, stim_region_blocks)
        ],
        "unit_summaries": unit_summaries,
        "feature_counts": {
            "task_history": int(task_history.shape[1]),
            "stimulus_region": int(stim_region.shape[1]),
            "movement_region": int(move_region.shape[1]),
            "stimulus_unit": int(stim_unit.shape[1]),
            "movement_unit": int(move_unit.shape[1]),
            "pre_stimulus_unit": int(pre_stim_unit.shape[1]),
            "pre_movement_unit": int(pre_move_unit.shape[1]),
        },
        "target_metadata": {
            "choice_sign": choice_meta,
            "first_movement_speed": speed_meta,
            "wheel_action_direction": wheel_meta,
        },
        "task_history_feature_names": task_history_names,
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "task_penalty": float(args.task_penalty),
        "region_penalty": float(args.region_penalty),
        "unit_penalties": [float(value) for value in args.unit_penalties],
        "seed": int(args.seed),
        "min_delta": float(args.min_delta),
        "min_unit_spikes": int(args.min_unit_spikes),
        "max_units_per_probe": int(args.max_units_per_probe),
        "targets": targets,
    }


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    if candidate["kind"] == "single":
        collections = [candidate["collection"]]
    else:
        collections = candidate["collections"]
    return SimpleNamespace(
        eid=candidate["eid"],
        session_ref=candidate["session_ref"],
        collections=collections,
        folds=args.folds,
        inner_folds=args.inner_folds,
        task_penalty=args.task_penalty,
        region_penalty=args.region_penalty,
        unit_penalties=args.unit_penalties,
        seed=args.seed,
        min_label_spikes=args.min_label_spikes,
        min_unit_spikes=args.min_unit_spikes,
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
        min_delta=args.min_delta,
        max_units_per_probe=args.max_units_per_probe,
    )


def target_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        comparison = target["temporal_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["task_history_plus_hybrid_region"]["n_trials"],
                "class_counts": by_model["task_history_plus_hybrid_region"][
                    "class_counts"
                ],
                **comparison,
            }
        )
    return rows


def candidate_summary(candidate: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    rows = target_summary_rows(result)
    return {
        "name": candidate["name"],
        "kind": candidate["kind"],
        "eid": candidate["eid"],
        "session_ref": candidate["session_ref"],
        "collections": result["collections"],
        "reason": candidate["reason"],
        "trial_count": result["trial_count"],
        "feature_counts": result["feature_counts"],
        "target_rows": rows,
        "lagged_unit_after_task_region_count": sum(
            row["lagged_unit_after_task_region"] for row in rows
        ),
        "lagged_unit_after_task_region_current_unit_count": sum(
            row["lagged_unit_after_task_region_current_unit"] for row in rows
        ),
    }


def strip_for_panel(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key != "target_metadata"}


def aggregate(candidates: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    results = []
    summaries = []
    for candidate in candidates:
        result = evaluate_session(candidate_namespace(candidate, args))
        results.append({"candidate": candidate, "result": strip_for_panel(result)})
        summaries.append(candidate_summary(candidate, result))

    target_names = sorted({row["target"] for item in summaries for row in item["target_rows"]})
    target_replication = {}
    for target in target_names:
        rows = [
            row for item in summaries for row in item["target_rows"] if row["target"] == target
        ]
        target_replication[target] = {
            "candidate_count": len(rows),
            "lagged_unit_after_task_region_count": sum(
                row["lagged_unit_after_task_region"] for row in rows
            ),
            "lagged_unit_after_task_region_current_unit_count": sum(
                row["lagged_unit_after_task_region_current_unit"] for row in rows
            ),
            "mean_task_region_balanced_accuracy": sum(
                row["task_history_plus_hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_region_current_unit_balanced_accuracy": sum(
                row[
                    "task_history_plus_hybrid_region_plus_current_unit_balanced_accuracy"
                ]
                for row in rows
            )
            / len(rows),
            "mean_task_region_lagged_unit_balanced_accuracy": sum(
                row[
                    "task_history_plus_hybrid_region_plus_lagged_unit_balanced_accuracy"
                ]
                for row in rows
            )
            / len(rows),
            "mean_task_region_current_lagged_unit_balanced_accuracy": sum(
                row[
                    "task_history_plus_hybrid_region_plus_current_and_lagged_unit_balanced_accuracy"
                ]
                for row in rows
            )
            / len(rows),
            "mean_current_unit_increment_after_task_region": sum(
                row["current_unit_increment_after_task_region"] for row in rows
            )
            / len(rows),
            "mean_lagged_unit_increment_after_task_region": sum(
                row["lagged_unit_increment_after_task_region"] for row in rows
            )
            / len(rows),
            "mean_lagged_unit_increment_after_task_region_current_unit": sum(
                row["lagged_unit_increment_after_task_region_current_unit"] for row in rows
            )
            / len(rows),
        }

    def replicated_positive(target: str, count_key: str, mean_key: str) -> bool:
        row = target_replication.get(target, {})
        return bool(row.get(count_key, 0) >= 3 and row.get(mean_key, 0.0) > 0.0)

    lagged_after_region_supported = any(
        replicated_positive(
            target,
            "lagged_unit_after_task_region_count",
            "mean_lagged_unit_increment_after_task_region",
        )
        for target in target_replication
    )
    lagged_after_current_supported = any(
        replicated_positive(
            target,
            "lagged_unit_after_task_region_current_unit_count",
            "mean_lagged_unit_increment_after_task_region_current_unit",
        )
        for target in target_replication
    )
    return {
        "candidate_count": len(candidates),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "task_penalty": args.task_penalty,
        "region_penalty": args.region_penalty,
        "unit_penalties": [float(value) for value in args.unit_penalties],
        "seed": args.seed,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "min_delta": args.min_delta,
        "max_units_per_probe": args.max_units_per_probe,
        "lagged_unit_after_task_region_supported": bool(lagged_after_region_supported),
        "lagged_unit_after_task_region_current_unit_supported": bool(
            lagged_after_current_supported
        ),
        "temporal_glm_coupling_gate_passed": bool(lagged_after_current_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx temporal GLM coupling gate",
        "",
        "This gate asks whether lagged unit activity survives after task/history, target-window hybrid region bins, and current-window unit detail.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- task penalty: {output['task_penalty']}",
        f"- region penalty: {output['region_penalty']}",
        f"- unit penalties: `{output['unit_penalties']}`",
        f"- min unit spikes: {output['min_unit_spikes']}",
        f"- max units per probe: {output['max_units_per_probe']}",
        f"- lagged unit after task+region supported: `{output['lagged_unit_after_task_region_supported']}`",
        f"- lagged unit after task+region+current-unit supported: `{output['lagged_unit_after_task_region_current_unit_supported']}`",
        f"- temporal GLM coupling gate passed: `{output['temporal_glm_coupling_gate_passed']}`",
        "",
        "## nested comparison",
        "",
        "$$",
        "M_{XR}: y_i\\sim[X_i,R_i],",
        "\\qquad",
        "M_{XRU_0}: y_i\\sim[X_i,R_i,U_{0,i}],",
        "$$",
        "",
        "$$",
        "M_{XRU_L}: y_i\\sim[X_i,R_i,U_{L,i}],",
        "\\qquad",
        "M_{XRU_0U_L}: y_i\\sim[X_i,R_i,U_{0,i},U_{L,i}].",
        "$$",
        "",
        "The strict residual is",
        "",
        "$$",
        "\\Delta_{U_L\\mid X,R,U_0}",
        "=",
        "\\mathrm{BA}(M_{XRU_0U_L})",
        "-",
        "\\mathrm{BA}(M_{XRU_0}).",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | lag U after XR | lag U after XR+U0 | mean XR BA | mean XR+U0 BA | mean XR+UL BA | mean XR+U0+UL BA | mean UL|XR | mean UL|XR,U0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["lagged_unit_after_task_region_count"]),
                    str(row["lagged_unit_after_task_region_current_unit_count"]),
                    f"{row['mean_task_region_balanced_accuracy']:.6f}",
                    f"{row['mean_task_region_current_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_task_region_lagged_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_task_region_current_lagged_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_lagged_unit_increment_after_task_region']:.6f}",
                    f"{row['mean_lagged_unit_increment_after_task_region_current_unit']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | lag U after XR | lag U after XR+U0 | choice UL|XR,U0 | speed UL|XR,U0 | wheel UL|XR,U0 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in output["summaries"]:
        by_target = {row["target"]: row for row in summary["target_rows"]}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['name']}`",
                    str(summary["trial_count"]),
                    str(summary["lagged_unit_after_task_region_count"]),
                    str(summary["lagged_unit_after_task_region_current_unit_count"]),
                    f"{by_target['choice_sign']['lagged_unit_increment_after_task_region_current_unit']:.6f}",
                    f"{by_target['first_movement_speed']['lagged_unit_increment_after_task_region_current_unit']:.6f}",
                    f"{by_target['wheel_action_direction']['lagged_unit_increment_after_task_region_current_unit']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## verdict", ""])
    lines.extend(
        [
            f"- lagged unit after task+region supported: `{output['lagged_unit_after_task_region_supported']}`",
            f"- lagged unit after task+region+current-unit supported: `{output['lagged_unit_after_task_region_current_unit_supported']}`",
            f"- temporal GLM coupling gate passed: `{output['temporal_glm_coupling_gate_passed']}`",
            "",
            "해석:",
            "",
            "- 양성이면 static unit identity 뒤에도 lagged unit activity가 남으므로 temporal coupling 후보로 승격한다.",
            "- 음성이면 block-regularized unit detail은 주로 target-window static readout으로 남고, causal/effective coupling claim은 보류한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def selected_candidates(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.single_session:
        return [
            {
                "name": "nyu30_motor_striatal_multi_probe",
                "kind": "multi",
                "eid": DEFAULT_EID,
                "session_ref": DEFAULT_SESSION_REF,
                "collections": args.collections,
                "reason": "same-session motor cortex plus striatal/septal multi-probe bridge",
            }
        ]
    return CANDIDATES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument(
        "--unit-penalties",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--min-label-spikes", type=int, default=100_000)
    parser.add_argument("--min-unit-spikes", type=int, default=1_000)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=1e-3)
    parser.add_argument("--max-units-per-probe", type=int, default=192)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx temporal GLM coupling gate")
    print(f"  candidates={output['candidate_count']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} UL|XR="
            + f"{row['lagged_unit_after_task_region_count']}/"
            + f"{row['candidate_count']} UL|XR,U0="
            + f"{row['lagged_unit_after_task_region_current_unit_count']}/"
            + f"{row['candidate_count']} mean_UL|XR,U0="
            + f"{row['mean_lagged_unit_increment_after_task_region_current_unit']:.6f}"
        )
    print(
        "  temporal_glm_coupling_gate_passed="
        + str(output["temporal_glm_coupling_gate_passed"])
    )
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
