"""Mouse IBL/OpenAlyx block-regularized unit residual gate.

The common-ridge all-unit nested gate did not promote an independent unit
residual after task/history and hybrid anatomical region bins.  One possible
counterexample is regularization mismatch: the unit block is high-dimensional,
so a single ridge penalty may overfit units while treating the compact region
block too generously.

This gate keeps task and region penalties fixed, then selects the unit penalty
inside each training fold by inner cross-validation:

    M_XR:      y ~ [X_task, R_hybrid]
    M_XU:      y ~ [X_task, U_unit]                 with tuned lambda_U
    M_XRU:     y ~ [X_task, R_hybrid, U_unit]       with tuned lambda_U

The key residual remains

    Delta_{U | X,R}^{block} = BA(M_XRU) - BA(M_XR).

If this becomes positive and replicated, the previous negative unit residual
was mostly a common-penalty artifact.  If it stays non-positive, the current
mouse term should remain task/history plus hybrid region compression, with
unit identity treated as a stronger flat-decoder caveat rather than a promoted
additive residual.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_all_unit_nested_regularization_gate import (
    all_unit_feature_block,
    make_all_unit_models,
)
from mouse_ibl_channel_region_rescue_gate import (
    CANDIDATES,
    load_probe,
    make_models as make_region_models,
    probe_feature_block,
    summarize_probe,
)
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
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


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_block_regularized_unit_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_block_regularized_unit_report.md"
)


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


def scores_for_fixed_model(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    scores = np.zeros(len(y), dtype=float)
    for test in stratified_folds(y, folds, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        scores[test] = block_ridge_fold_scores(
            blocks,
            y,
            train,
            test,
            block_names,
            penalties,
        )
    return scores, []


def choose_unit_penalty(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    outer_train: np.ndarray,
    block_names: list[str],
    base_penalties: dict[str, float],
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
        penalties = {**base_penalties, "U": float(unit_penalty)}
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


def scores_for_unit_tuned_model(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    block_names: list[str],
    base_penalties: dict[str, float],
    unit_penalties: list[float],
    folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    scores = np.zeros(len(y), dtype=float)
    selected = []
    for outer_idx, test in enumerate(stratified_folds(y, folds, seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        unit_penalty = choose_unit_penalty(
            blocks,
            y,
            train,
            block_names,
            base_penalties,
            unit_penalties,
            inner_folds,
            seed + 1009 + outer_idx,
        )
        selected.append(unit_penalty)
        penalties = {**base_penalties, "U": unit_penalty}
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


def evaluate_block_target(
    *,
    target_name: str,
    window_name: str,
    x_block: np.ndarray,
    r_block: np.ndarray,
    u_block: np.ndarray,
    global_rate: np.ndarray,
    y_all: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    blocks_all = {
        "X": x_block,
        "R": r_block,
        "U": u_block,
        "G": global_rate,
    }
    valid = valid & finite_rows(blocks_all)
    y = np.asarray(y_all[valid], dtype=int)
    blocks = {name: value[valid] for name, value in blocks_all.items()}
    base = {"X": args.task_penalty, "R": args.region_penalty}
    unit_grid = [float(value) for value in args.unit_penalties]

    model_specs = [
        (
            "global_rate",
            ["G"],
            {"G": args.region_penalty},
            False,
        ),
        (
            "task_history",
            ["X"],
            {"X": args.task_penalty},
            False,
        ),
        (
            "task_history_plus_hybrid_region",
            ["X", "R"],
            base,
            False,
        ),
        (
            "block_task_history_plus_all_unit",
            ["X", "U"],
            {"X": args.task_penalty},
            True,
        ),
        (
            "block_task_history_plus_hybrid_region_plus_all_unit",
            ["X", "R", "U"],
            base,
            True,
        ),
    ]
    rows = []
    for model_name, block_names, penalties, tune_unit in model_specs:
        if tune_unit:
            scores, selected = scores_for_unit_tuned_model(
                blocks,
                y,
                block_names,
                penalties,
                unit_grid,
                args.folds,
                args.inner_folds,
                args.seed,
            )
        else:
            scores, selected = scores_for_fixed_model(
                blocks,
                y,
                block_names,
                penalties,
                args.folds,
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
    task = by_model["task_history"]["balanced_accuracy"]
    task_region = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    task_unit = by_model["block_task_history_plus_all_unit"]["balanced_accuracy"]
    task_region_unit = by_model[
        "block_task_history_plus_hybrid_region_plus_all_unit"
    ]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_task_history"] = float(row["balanced_accuracy"] - task)
        row["delta_vs_task_region"] = float(row["balanced_accuracy"] - task_region)
        row["delta_vs_task_unit"] = float(row["balanced_accuracy"] - task_unit)
        row["delta_vs_task_region_unit"] = float(
            row["balanced_accuracy"] - task_region_unit
        )
    target["block_comparison"] = {
        "task_history_balanced_accuracy": task,
        "task_history_plus_hybrid_region_balanced_accuracy": task_region,
        "block_task_history_plus_all_unit_balanced_accuracy": task_unit,
        "block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": task_region_unit,
        "region_increment_after_task": float(task_region - task),
        "block_unit_increment_after_task": float(task_unit - task),
        "block_unit_increment_after_task_region": float(task_region_unit - task_region),
        "region_increment_after_block_task_unit": float(task_region_unit - task_unit),
        "block_unit_beats_region_after_task": bool(task_unit > task_region + min_delta),
        "block_unit_residual_after_task_region": bool(
            task_region_unit > task_region + min_delta
        ),
        "region_residual_after_block_task_unit": bool(
            task_region_unit > task_unit + min_delta
        ),
        "selected_unit_penalty_counts_xu": by_model[
            "block_task_history_plus_all_unit"
        ]["selected_unit_penalty_counts"],
        "selected_unit_penalty_counts_xru": by_model[
            "block_task_history_plus_hybrid_region_plus_all_unit"
        ]["selected_unit_penalty_counts"],
        "selected_unit_penalty_median_xu": by_model[
            "block_task_history_plus_all_unit"
        ]["selected_unit_penalty_median"],
        "selected_unit_penalty_median_xru": by_model[
            "block_task_history_plus_hybrid_region_plus_all_unit"
        ]["selected_unit_penalty_median"],
    }
    return target


def feature_blocks_for_window(
    probes: list[dict[str, object]],
    trials,
    spec,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    starts, ends, valid = window_bounds(trials, spec)
    region_blocks = [
        probe_feature_block(
            probe,
            starts,
            ends,
            valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]
    unit_blocks = [
        all_unit_feature_block(
            probe,
            starts,
            ends,
            valid,
            args.min_unit_spikes,
            args.max_units_per_probe,
        )
        for probe in probes
    ]
    region_models = make_region_models(region_blocks)
    unit_models = make_all_unit_models(unit_blocks)
    return (
        {
            "R": region_models["hybrid_acronym_channel_id_by_probe"],
            "U": unit_models["all_unit_by_probe"],
            "G": region_models["global_rate"],
        },
        region_blocks,
        unit_blocks,
        valid,
    )


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    task_history, task_history_names = task_history_covariates(trials)

    stimulus_features, stimulus_region_blocks, stimulus_unit_blocks, stim_valid = (
        feature_blocks_for_window(probes, trials, STIMULUS_WINDOW, args)
    )
    movement_features, movement_region_blocks, movement_unit_blocks, move_valid = (
        feature_blocks_for_window(probes, trials, MOVEMENT_WINDOW, args)
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
        evaluate_block_target(
            target_name="choice_sign",
            window_name=STIMULUS_WINDOW.name,
            x_block=task_history,
            r_block=stimulus_features["R"],
            u_block=stimulus_features["U"],
            global_rate=stimulus_features["G"],
            y_all=choice,
            valid=choice_valid & stim_valid,
            args=args,
        ),
        evaluate_block_target(
            target_name="first_movement_speed",
            window_name=STIMULUS_WINDOW.name,
            x_block=task_history,
            r_block=stimulus_features["R"],
            u_block=stimulus_features["U"],
            global_rate=stimulus_features["G"],
            y_all=speed,
            valid=speed_valid & stim_valid,
            args=args,
        ),
        evaluate_block_target(
            target_name="wheel_action_direction",
            window_name=MOVEMENT_WINDOW.name,
            x_block=task_history,
            r_block=movement_features["R"],
            u_block=movement_features["U"],
            global_rate=movement_features["G"],
            y_all=wheel_direction,
            valid=wheel_valid & move_valid,
            args=args,
        ),
    ]

    unit_summaries = [
        {
            "collection": probe["collection"],
            "stimulus_selected_unit_count": stim_block["selected_unit_count"],
            "movement_selected_unit_count": move_block["selected_unit_count"],
            "stimulus_unit_feature_count": stim_block["feature_count"],
            "movement_unit_feature_count": move_block["feature_count"],
        }
        for probe, stim_block, move_block in zip(
            probes,
            stimulus_unit_blocks,
            movement_unit_blocks,
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
            summarize_probe(probe, stimulus_block)
            for probe, stimulus_block in zip(probes, stimulus_region_blocks)
        ],
        "unit_summaries": unit_summaries,
        "feature_counts": {
            "task_history": int(task_history.shape[1]),
            "stimulus_region": int(stimulus_features["R"].shape[1]),
            "stimulus_unit": int(stimulus_features["U"].shape[1]),
            "movement_region": int(movement_features["R"].shape[1]),
            "movement_unit": int(movement_features["U"].shape[1]),
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
        comparison = target["block_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["task_history_plus_hybrid_region"]["n_trials"],
                "class_counts": by_model["task_history_plus_hybrid_region"][
                    "class_counts"
                ],
                "task_history_balanced_accuracy": comparison[
                    "task_history_balanced_accuracy"
                ],
                "task_history_plus_hybrid_region_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_region_balanced_accuracy"
                ],
                "block_task_history_plus_all_unit_balanced_accuracy": comparison[
                    "block_task_history_plus_all_unit_balanced_accuracy"
                ],
                "block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": comparison[
                    "block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy"
                ],
                "region_increment_after_task": comparison[
                    "region_increment_after_task"
                ],
                "block_unit_increment_after_task": comparison[
                    "block_unit_increment_after_task"
                ],
                "block_unit_increment_after_task_region": comparison[
                    "block_unit_increment_after_task_region"
                ],
                "region_increment_after_block_task_unit": comparison[
                    "region_increment_after_block_task_unit"
                ],
                "block_unit_beats_region_after_task": comparison[
                    "block_unit_beats_region_after_task"
                ],
                "block_unit_residual_after_task_region": comparison[
                    "block_unit_residual_after_task_region"
                ],
                "region_residual_after_block_task_unit": comparison[
                    "region_residual_after_block_task_unit"
                ],
                "selected_unit_penalty_median_xu": comparison[
                    "selected_unit_penalty_median_xu"
                ],
                "selected_unit_penalty_median_xru": comparison[
                    "selected_unit_penalty_median_xru"
                ],
                "selected_unit_penalty_counts_xu": comparison[
                    "selected_unit_penalty_counts_xu"
                ],
                "selected_unit_penalty_counts_xru": comparison[
                    "selected_unit_penalty_counts_xru"
                ],
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
        "block_unit_residual_after_task_region_count": sum(
            row["block_unit_residual_after_task_region"] for row in rows
        ),
        "region_residual_after_block_task_unit_count": sum(
            row["region_residual_after_block_task_unit"] for row in rows
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
            "block_unit_residual_after_task_region_count": sum(
                row["block_unit_residual_after_task_region"] for row in rows
            ),
            "region_residual_after_block_task_unit_count": sum(
                row["region_residual_after_block_task_unit"] for row in rows
            ),
            "block_unit_beats_region_after_task_count": sum(
                row["block_unit_beats_region_after_task"] for row in rows
            ),
            "mean_task_history_balanced_accuracy": sum(
                row["task_history_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_region_balanced_accuracy": sum(
                row["task_history_plus_hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_block_task_history_plus_all_unit_balanced_accuracy": sum(
                row["block_task_history_plus_all_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": sum(
                row[
                    "block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy"
                ]
                for row in rows
            )
            / len(rows),
            "mean_region_increment_after_task": sum(
                row["region_increment_after_task"] for row in rows
            )
            / len(rows),
            "mean_block_unit_increment_after_task": sum(
                row["block_unit_increment_after_task"] for row in rows
            )
            / len(rows),
            "mean_block_unit_increment_after_task_region": sum(
                row["block_unit_increment_after_task_region"] for row in rows
            )
            / len(rows),
            "mean_region_increment_after_block_task_unit": sum(
                row["region_increment_after_block_task_unit"] for row in rows
            )
            / len(rows),
            "median_selected_unit_penalty_xru": float(
                np.median([row["selected_unit_penalty_median_xru"] for row in rows])
            ),
        }

    def replicated_positive(target: str, count_key: str, mean_key: str) -> bool:
        row = target_replication.get(target, {})
        return bool(row.get(count_key, 0) >= 3 and row.get(mean_key, 0.0) > 0.0)

    unit_residual_supported = any(
        replicated_positive(
            target,
            "block_unit_residual_after_task_region_count",
            "mean_block_unit_increment_after_task_region",
        )
        for target in target_replication
    )
    region_residual_supported = any(
        replicated_positive(
            target,
            "region_residual_after_block_task_unit_count",
            "mean_region_increment_after_block_task_unit",
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
        "block_unit_residual_after_task_region_supported": bool(unit_residual_supported),
        "region_residual_after_block_task_unit_supported": bool(region_residual_supported),
        "block_regularized_unit_gate_passed": bool(unit_residual_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx block-regularized unit residual gate",
        "",
        "Common-ridge nested gate 다음 반례는 unit block penalty mismatch다.",
        "이 gate는 outer fold의 train split 내부에서만 \\(\\lambda_U\\)를 고르고, held-out outer fold에서 block-regularized residual을 평가한다.",
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
        f"- block unit residual after task+region supported: `{output['block_unit_residual_after_task_region_supported']}`",
        f"- region residual after block task+unit supported: `{output['region_residual_after_block_task_unit_supported']}`",
        f"- block-regularized unit gate passed: `{output['block_regularized_unit_gate_passed']}`",
        "",
        "## block equation",
        "",
        "$$",
        "\\hat\\beta_{\\lambda_U}^{(-k)}",
        "=",
        "\\arg\\min_\\beta",
        "\\left\\|y-Z\\beta\\right\\|_2^2",
        "+",
        "\\lambda_X\\|\\beta_X\\|_2^2",
        "+",
        "\\lambda_R\\|\\beta_R\\|_2^2",
        "+",
        "\\lambda_U\\|\\beta_U\\|_2^2.",
        "$$",
        "",
        "The unit penalty is chosen only on the outer-train split:",
        "",
        "$$",
        "\\lambda_U^{*(-k)}",
        "=",
        "\\arg\\max_{\\lambda_U\\in\\Lambda_U}",
        "\\mathrm{BA}_{\\mathrm{inner}}(\\lambda_U).",
        "$$",
        "",
        "The main residual is",
        "",
        "$$",
        "\\Delta_{U\\mid X,R}^{\\mathrm{block}}",
        "=",
        "\\mathrm{BA}(M_{XRU}^{\\mathrm{block}})",
        "-",
        "\\mathrm{BA}(M_{XR}).",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | block unit residual count | region residual count | unit>region after task | mean task BA | mean task+region BA | mean block task+unit BA | mean block task+region+unit BA | mean block unit residual | mean region residual | median lambda_U XRU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["block_unit_residual_after_task_region_count"]),
                    str(row["region_residual_after_block_task_unit_count"]),
                    str(row["block_unit_beats_region_after_task_count"]),
                    f"{row['mean_task_history_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_block_task_history_plus_all_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_block_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_block_unit_increment_after_task_region']:.6f}",
                    f"{row['mean_region_increment_after_block_task_unit']:.6f}",
                    f"{row['median_selected_unit_penalty_xru']:.6g}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | unit residuals | region residuals | choice U_block_given_XR | speed U_block_given_XR | wheel U_block_given_XR | choice R_given_XU_block | speed R_given_XU_block | wheel R_given_XU_block |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    str(summary["block_unit_residual_after_task_region_count"]),
                    str(summary["region_residual_after_block_task_unit_count"]),
                    f"{by_target['choice_sign']['block_unit_increment_after_task_region']:.6f}",
                    f"{by_target['first_movement_speed']['block_unit_increment_after_task_region']:.6f}",
                    f"{by_target['wheel_action_direction']['block_unit_increment_after_task_region']:.6f}",
                    f"{by_target['choice_sign']['region_increment_after_block_task_unit']:.6f}",
                    f"{by_target['first_movement_speed']['region_increment_after_block_task_unit']:.6f}",
                    f"{by_target['wheel_action_direction']['region_increment_after_block_task_unit']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## verdict", ""])
    lines.extend(
        [
            f"- block unit residual after task+region supported: `{output['block_unit_residual_after_task_region_supported']}`",
            f"- region residual after block task+unit supported: `{output['region_residual_after_block_task_unit_supported']}`",
            f"- block-regularized unit gate passed: `{output['block_regularized_unit_gate_passed']}`",
            "",
            "해석:",
            "",
            "- 이 gate가 양성이면 common-ridge all-unit 음성 결과는 penalty mismatch였다고 볼 수 있다.",
            "- 이 gate도 음성이면 unit identity는 flat comparison에서는 강하지만, \\([X,R]\\) 뒤의 독립 additive 항으로는 아직 승격되지 않는다.",
            "- 다음 단계는 single-trial behavior target에 대한 temporal GLM coupling이다.",
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

    print("Mouse IBL/OpenAlyx block-regularized unit residual gate")
    print(f"  candidates={output['candidate_count']}")
    print(f"  unit_penalties={output['unit_penalties']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} U_block|XR="
            + f"{row['block_unit_residual_after_task_region_count']}/"
            + f"{row['candidate_count']} R|XU_block="
            + f"{row['region_residual_after_block_task_unit_count']}/"
            + f"{row['candidate_count']} mean_U_block|XR="
            + f"{row['mean_block_unit_increment_after_task_region']:.6f}"
        )
    print(
        "  block_regularized_unit_gate_passed="
        + str(output["block_regularized_unit_gate_passed"])
    )
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
