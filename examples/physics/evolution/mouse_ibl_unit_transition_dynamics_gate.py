"""Mouse IBL/OpenAlyx unit transition dynamics gate.

The strict temporal GLM failed because it asked whether lagged unit activity
adds behavioral decoding after the current unit window is already present.  A
more direct coupling question is neural, not behavioral:

    U_0 ~ X_task + R_target + U_lag

This gate asks whether lagged unit activity improves held-out prediction of
the current-window unit state after task/history and target-window hybrid
region bins.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_channel_region_rescue_gate import (
    CANDIDATES,
    load_probe,
    probe_feature_block,
    summarize_probe,
)
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    WindowSpec,
    window_bounds,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_MOVEMENT_WINDOW,
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
    zscore_block,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_unit_transition_dynamics_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_unit_transition_dynamics_report.md")


def contiguous_folds(n: int, folds: int) -> list[np.ndarray]:
    indices = np.arange(n)
    return [fold for fold in np.array_split(indices, folds) if len(fold)]


def finite_mask(blocks: dict[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    mask = np.all(np.isfinite(y), axis=1)
    for block in blocks.values():
        mask &= np.all(np.isfinite(block), axis=1)
    return mask


def standardize_y(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0)
    scale = np.nanstd(y_train, axis=0)
    scale[scale < 1e-9] = 1.0
    return (y_train - mean) / scale, (y_test - mean) / scale, mean, scale


def ridge_predict_multioutput(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
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
    y_train_z, y_test_z, _, _ = standardize_y(y[train], y[test])
    penalty = np.diag([0.0, *penalty_values])
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train_z
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_test @ weights, y_test_z


def r2_scores(y_true_z: np.ndarray, y_pred_z: np.ndarray) -> dict[str, float]:
    residual = y_true_z - y_pred_z
    sse = np.sum(residual * residual, axis=0)
    sst = np.sum((y_true_z - np.mean(y_true_z, axis=0)) ** 2, axis=0)
    valid = sst > 1e-12
    feature_r2 = np.full(y_true_z.shape[1], np.nan, dtype=float)
    feature_r2[valid] = 1.0 - sse[valid] / sst[valid]
    pooled_sse = float(np.sum(sse[valid]))
    pooled_sst = float(np.sum(sst[valid]))
    pooled = float(1.0 - pooled_sse / pooled_sst) if pooled_sst > 1e-12 else float("nan")
    return {
        "pooled_r2": pooled,
        "mean_feature_r2": float(np.nanmean(feature_r2)),
        "median_feature_r2": float(np.nanmedian(feature_r2)),
        "positive_feature_fraction": float(np.nanmean(feature_r2 > 0.0)),
        "valid_feature_count": int(np.sum(valid)),
    }


def scores_for_model(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
    folds: int,
) -> dict[str, float]:
    y_pred = np.zeros_like(y, dtype=float)
    y_true = np.zeros_like(y, dtype=float)
    for test in contiguous_folds(len(y), folds):
        train = np.setdiff1d(np.arange(len(y)), test)
        pred, true = ridge_predict_multioutput(
            blocks,
            y,
            train,
            test,
            block_names,
            penalties,
        )
        y_pred[test] = pred
        y_true[test] = true
    return r2_scores(y_true, y_pred)


def evaluate_transition(
    name: str,
    source_window: WindowSpec,
    target_window: WindowSpec,
    x_block: np.ndarray,
    r_target: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    blocks_all = {
        "X": x_block,
        "R": r_target,
        "UL": u_lag,
    }
    keep = finite_mask(blocks_all, u_target) & valid
    blocks = {key: value[keep] for key, value in blocks_all.items()}
    y = u_target[keep]
    model_specs = [
        (
            "task_history",
            ["X"],
            {"X": args.task_penalty},
        ),
        (
            "task_history_lagged_unit",
            ["X", "UL"],
            {"X": args.task_penalty, "UL": args.lag_unit_penalty},
        ),
        (
            "task_region",
            ["X", "R"],
            {"X": args.task_penalty, "R": args.region_penalty},
        ),
        (
            "task_region_lagged_unit",
            ["X", "R", "UL"],
            {"X": args.task_penalty, "R": args.region_penalty, "UL": args.lag_unit_penalty},
        ),
    ]
    rows = []
    for model, block_names, penalties in model_specs:
        row = {
            "model": model,
            "block_names": block_names,
            "penalties": penalties,
            **scores_for_model(blocks, y, block_names, penalties, args.folds),
        }
        rows.append(row)
    by_model = {row["model"]: row for row in rows}
    task = by_model["task_history"]
    task_lag = by_model["task_history_lagged_unit"]
    base = by_model["task_region"]
    lag = by_model["task_region_lagged_unit"]
    delta_task_pooled = float(task_lag["pooled_r2"] - task["pooled_r2"])
    delta_task_mean = float(task_lag["mean_feature_r2"] - task["mean_feature_r2"])
    delta_pooled = float(lag["pooled_r2"] - base["pooled_r2"])
    delta_mean = float(lag["mean_feature_r2"] - base["mean_feature_r2"])
    return {
        "transition": name,
        "source_window": source_window.name,
        "target_window": target_window.name,
        "trial_count": int(len(y)),
        "unit_feature_count": int(y.shape[1]),
        "rows": rows,
        "nested_comparison": {
            "task_pooled_r2": task["pooled_r2"],
            "task_lagged_unit_pooled_r2": task_lag["pooled_r2"],
            "lagged_unit_increment_after_task_pooled_r2": delta_task_pooled,
            "task_mean_feature_r2": task["mean_feature_r2"],
            "task_lagged_unit_mean_feature_r2": task_lag["mean_feature_r2"],
            "lagged_unit_increment_after_task_mean_feature_r2": delta_task_mean,
            "lagged_unit_transition_after_task": bool(
                delta_task_pooled > args.min_delta and delta_task_mean > args.min_delta_mean
            ),
            "task_region_pooled_r2": base["pooled_r2"],
            "task_region_lagged_unit_pooled_r2": lag["pooled_r2"],
            "lagged_unit_increment_pooled_r2": delta_pooled,
            "task_region_mean_feature_r2": base["mean_feature_r2"],
            "task_region_lagged_unit_mean_feature_r2": lag["mean_feature_r2"],
            "lagged_unit_increment_mean_feature_r2": delta_mean,
            "lagged_unit_transition_after_task_region": bool(
                delta_pooled > args.min_delta and delta_mean > args.min_delta_mean
            ),
        },
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
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

    transitions = [
        evaluate_transition(
            name="pre_stimulus_to_stimulus_unit",
            source_window=PRE_STIMULUS_WINDOW,
            target_window=STIMULUS_WINDOW,
            x_block=task_history,
            r_target=stim_region,
            u_lag=pre_stim_unit,
            u_target=stim_unit,
            valid=stim_region_valid & stim_unit_valid & pre_stim_unit_valid,
            args=args,
        ),
        evaluate_transition(
            name="pre_movement_to_movement_unit",
            source_window=PRE_MOVEMENT_WINDOW,
            target_window=MOVEMENT_WINDOW,
            x_block=task_history,
            r_target=move_region,
            u_lag=pre_move_unit,
            u_target=move_unit,
            valid=move_region_valid & move_unit_valid & pre_move_unit_valid,
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
        "probe_summaries": [
            summarize_probe(probe, block)
            for probe, block in zip(probes, stim_region_blocks)
        ],
        "unit_summaries": unit_summaries,
        "task_history_feature_names": task_history_names,
        "feature_counts": {
            "task_history": int(task_history.shape[1]),
            "stimulus_region": int(stim_region.shape[1]),
            "movement_region": int(move_region.shape[1]),
            "stimulus_unit": int(stim_unit.shape[1]),
            "movement_unit": int(move_unit.shape[1]),
            "pre_stimulus_unit": int(pre_stim_unit.shape[1]),
            "pre_movement_unit": int(pre_move_unit.shape[1]),
        },
        "transitions": transitions,
    }


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eid=candidate["eid"],
        session_ref=candidate["session_ref"],
        collections=candidate.get("collections", [candidate.get("collection")]),
        folds=args.folds,
        task_penalty=args.task_penalty,
        region_penalty=args.region_penalty,
        lag_unit_penalty=args.lag_unit_penalty,
        min_delta=args.min_delta,
        min_delta_mean=args.min_delta_mean,
        min_label_spikes=args.min_label_spikes,
        min_unit_spikes=args.min_unit_spikes,
        max_units_per_probe=args.max_units_per_probe,
    )


def strip_session(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"probe_summaries", "task_history_feature_names"}
    }


def summarize_panel(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    names = [transition["transition"] for transition in results[0]["transitions"]]
    summary = []
    for name in names:
        rows = [
            transition
            for result in results
            for transition in result["transitions"]
            if transition["transition"] == name
        ]
        nested = [row["nested_comparison"] for row in rows]
        positive = [
            item["lagged_unit_transition_after_task_region"]
            for item in nested
        ]
        positive_after_task = [
            item["lagged_unit_transition_after_task"]
            for item in nested
        ]
        mean_delta_pooled = float(np.mean([item["lagged_unit_increment_pooled_r2"] for item in nested]))
        mean_delta_mean = float(np.mean([item["lagged_unit_increment_mean_feature_r2"] for item in nested]))
        mean_delta_after_task_pooled = float(
            np.mean([item["lagged_unit_increment_after_task_pooled_r2"] for item in nested])
        )
        mean_delta_after_task_mean = float(
            np.mean([item["lagged_unit_increment_after_task_mean_feature_r2"] for item in nested])
        )
        summary.append(
            {
                "transition": name,
                "candidates": len(rows),
                "positive_after_task_replications": int(sum(positive_after_task)),
                "positive_replications": int(sum(positive)),
                "mean_lagged_unit_increment_after_task_pooled_r2": mean_delta_after_task_pooled,
                "mean_lagged_unit_increment_after_task_mean_feature_r2": mean_delta_after_task_mean,
                "mean_lagged_unit_increment_pooled_r2": mean_delta_pooled,
                "mean_lagged_unit_increment_mean_feature_r2": mean_delta_mean,
                "supported": bool(
                    sum(positive) >= args.min_replications
                    and mean_delta_pooled > args.min_delta
                    and mean_delta_mean > args.min_delta_mean
                ),
            }
        )
    return {
        "transition_summary": summary,
        "unit_transition_dynamics_gate_passed": bool(any(row["supported"] for row in summary)),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx unit transition dynamics gate",
        "",
        "This gate asks a neural transition question before returning to behavior.",
        "",
        "$$",
        "U_t = A U_{t-\\ell}+B X_t+C R_t+\\epsilon_t",
        "$$",
        "",
        "The tested residual is:",
        "",
        "$$",
        "\\Delta_{\\mathrm{transition}\\mid X,R_0}",
        "=",
        "R^2[U_t\\mid X_t,R_t,U_{t-\\ell}]",
        "-",
        "R^2[U_t\\mid X_t,R_t].",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- task penalty: {output['task_penalty']}",
        f"- region penalty: {output['region_penalty']}",
        f"- lag unit penalty: {output['lag_unit_penalty']}",
        f"- min pooled delta: {output['min_delta']}",
        f"- min mean-feature delta: {output['min_delta_mean']}",
        f"- unit transition dynamics gate passed: `{output['unit_transition_dynamics_gate_passed']}`",
        "",
        "## transition replication",
        "",
        "| transition | candidates | pos after X | pos after X,R0 | mean dR2 after X | mean dR2 after X,R0 | supported |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["transition_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['transition']}`",
                    str(row["candidates"]),
                    str(row["positive_after_task_replications"]),
                    str(row["positive_replications"]),
                    f"{row['mean_lagged_unit_increment_after_task_pooled_r2']:.6f}",
                    f"{row['mean_lagged_unit_increment_pooled_r2']:.6f}",
                    f"`{row['supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | pre-stim dR2 after X | pre-stim dR2 after X,R0 | pre-move dR2 after X | pre-move dR2 after X,R0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in output["candidate_results"]:
        by_name = {row["transition"]: row["nested_comparison"] for row in result["transitions"]}
        pre_stim = by_name["pre_stimulus_to_stimulus_unit"]
        pre_move = by_name["pre_movement_to_movement_unit"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result['name']}`",
                    str(result["trial_count"]),
                    f"{pre_stim['lagged_unit_increment_after_task_pooled_r2']:.6f}",
                    f"{pre_stim['lagged_unit_increment_pooled_r2']:.6f}",
                    f"{pre_move['lagged_unit_increment_after_task_pooled_r2']:.6f}",
                    f"{pre_move['lagged_unit_increment_pooled_r2']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Positive result means lagged unit activity predicts the next unit state after task/history and region compression.",
            "- Negative result means the previous behavioral temporal-GLM failure is not rescued by a simple linear unit-transition model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
    )
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--lag-unit-penalty", type=float, default=100.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-delta-mean", type=float, default=0.0005)
    parser.add_argument("--min-replications", type=int, default=4)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=192)
    args = parser.parse_args()

    if args.single_session:
        candidates = [
            {
                "name": "nyu30_motor_striatal_multi_probe",
                "eid": args.eid,
                "session_ref": args.session_ref,
                "collections": args.collections,
            }
        ]
    else:
        candidates = CANDIDATES

    candidate_results = []
    for candidate in candidates:
        result = evaluate_session(candidate_namespace(candidate, args))
        result["name"] = candidate["name"]
        candidate_results.append(strip_session(result))

    output = {
        "openalyx_url": OPENALYX_URL,
        "candidate_count": len(candidate_results),
        "folds": args.folds,
        "task_penalty": args.task_penalty,
        "region_penalty": args.region_penalty,
        "lag_unit_penalty": args.lag_unit_penalty,
        "min_delta": args.min_delta,
        "min_delta_mean": args.min_delta_mean,
        "min_replications": args.min_replications,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": candidate_results,
    }
    output.update(summarize_panel(candidate_results, args))

    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(make_report(output))

    print("Mouse IBL/OpenAlyx unit transition dynamics gate")
    print(f"  candidates              = {len(candidate_results)}")
    for row in output["transition_summary"]:
        print(
            f"  {row['transition']}: positive={row['positive_replications']}/"
            f"{row['candidates']} mean pooled delta="
            f"{row['mean_lagged_unit_increment_pooled_r2']:.6f}"
        )
    print(f"  gate_passed             = {output['unit_transition_dynamics_gate_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
