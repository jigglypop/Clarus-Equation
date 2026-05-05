"""Fold-local top-unit sufficiency for the Mouse IBL action innovation subspace.

Top units are selected inside each outer fold from the train-fitted target-window
PCA loading basis and the train-selected action innovation axes.  Cluster labels
are session-local, so this script does not aggregate cluster ids across sessions
as if they were the same biological unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mouse_ibl_action_subspace_mechanism_map import (
    fit_transition_predict_with_basis,
    unit_metadata,
)
from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import load_probe, load_ranked_candidates, ridge_classifier_scores
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, load_common
from mouse_ibl_nested_innovation_subspace_gate import select_axes_inside_train
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    balanced_accuracy,
    first_movement_speed_target,
    stratified_folds,
    wheel_action_direction_target,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_MOVEMENT_WINDOW,
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
    zscore_block,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_action_top_unit_sufficiency_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_action_top_unit_sufficiency_report.md")


def top_unit_names_from_basis(
    basis: np.ndarray,
    selected_axes: list[int],
    metadata: list[dict[str, Any]],
    probe_label: str,
    top_unit_count: int,
) -> list[str]:
    selected_indices = [axis - 1 for axis in selected_axes if 0 < axis <= basis.shape[1]]
    if not selected_indices:
        return []
    weights = np.sum(np.abs(basis[:, selected_indices]), axis=1)
    eligible = np.asarray(
        [
            str(row.get("probe", "")) == probe_label
            and not bool(row.get("is_other", False))
            for row in metadata
        ],
        dtype=bool,
    )
    if not np.any(eligible):
        return []
    eligible_indices = np.where(eligible)[0]
    order = eligible_indices[np.argsort(weights[eligible_indices])[::-1]]
    top_indices = order[:top_unit_count]
    return [str(metadata[index]["feature"]) for index in top_indices if weights[index] > 0]


def masks_for_feature_names(
    lag_metadata: list[dict[str, Any]],
    target_metadata: list[dict[str, Any]],
    top_names: list[str],
    condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    top = set(top_names)
    lag_mask = np.asarray([str(row.get("feature", "")) in top for row in lag_metadata], dtype=bool)
    target_mask = np.asarray(
        [str(row.get("feature", "")) in top for row in target_metadata],
        dtype=bool,
    )
    if condition == "only_top_units":
        return lag_mask, target_mask
    if condition == "drop_top_units":
        return ~lag_mask, ~target_mask
    if condition == "full":
        return np.ones(len(lag_metadata), dtype=bool), np.ones(len(target_metadata), dtype=bool)
    raise ValueError(f"unknown condition: {condition}")


def fit_scores_for_condition(
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    lag_metadata: list[dict[str, Any]],
    target_metadata: list[dict[str, Any]],
    condition: str,
    top_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    lag_mask, target_mask = masks_for_feature_names(
        lag_metadata,
        target_metadata,
        top_names,
        condition,
    )
    info = {
        "condition": condition,
        "lag_units": int(np.sum(lag_mask)),
        "target_units": int(np.sum(target_mask)),
    }
    if info["lag_units"] < args.min_units or info["target_units"] < args.min_units:
        info["skipped"] = True
        return None, None, info

    hhat_train, hhat_test, eps_train, eps_test, _, _ = fit_transition_predict_with_basis(
        x,
        r,
        u_lag[:, lag_mask],
        u_target[:, target_mask],
        train,
        test,
        args,
    )
    selected_axes, _ = select_axes_inside_train(
        x[train],
        r[train],
        hhat_train,
        eps_train,
        y[train],
        args,
    )
    selected_indices = [axis - 1 for axis in selected_axes if 0 < axis <= eps_train.shape[1]]
    x_train, x_test = zscore_block(x[train], x[test])
    r_train, r_test = zscore_block(r[train], r[test])
    train_blocks = {"X": x_train, "R": r_train, "HHAT": hhat_train}
    test_blocks = {"X": x_test, "R": r_test, "HHAT": hhat_test}
    penalties = {
        "X": args.task_penalty,
        "R": args.region_penalty,
        "HHAT": args.latent_penalty,
    }
    baseline = ridge_classifier_scores(
        train_blocks,
        test_blocks,
        y[train],
        ["X", "R", "HHAT"],
        penalties,
    )
    subspace = baseline.copy()
    if selected_indices:
        train_blocks = {**train_blocks, "EPS_SUB": eps_train[:, selected_indices]}
        test_blocks = {**test_blocks, "EPS_SUB": eps_test[:, selected_indices]}
        penalties = {**penalties, "EPS_SUB": args.innovation_penalty}
        subspace = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X", "R", "HHAT", "EPS_SUB"],
            penalties,
        )
    info["selected_axes"] = selected_axes
    info["skipped"] = False
    return baseline, subspace, info


def evaluate_target(
    target: str,
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    lag_metadata: list[dict[str, Any]],
    target_metadata: list[dict[str, Any]],
    y_all: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    finite = (
        valid
        & np.isfinite(y_all)
        & np.all(np.isfinite(x), axis=1)
        & np.all(np.isfinite(r), axis=1)
        & np.all(np.isfinite(u_lag), axis=1)
        & np.all(np.isfinite(u_target), axis=1)
    )
    x = x[finite]
    r = r[finite]
    u_lag = u_lag[finite]
    u_target = u_target[finite]
    y = y_all[finite].astype(int)
    score_rows = {
        condition: {
            "baseline": np.zeros(len(y), dtype=float),
            "subspace": np.zeros(len(y), dtype=float),
            "filled": np.zeros(len(y), dtype=bool),
        }
        for condition in args.conditions
    }
    fold_rows = []
    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        hhat_train, _, eps_train, _, basis, explained = fit_transition_predict_with_basis(
            x,
            r,
            u_lag,
            u_target,
            train,
            test,
            args,
        )
        selected_axes, _ = select_axes_inside_train(
            x[train],
            r[train],
            hhat_train,
            eps_train,
            y[train],
            args,
        )
        top_names = top_unit_names_from_basis(
            basis,
            selected_axes,
            target_metadata,
            args.probe_label,
            args.top_unit_count,
        )
        fold_info = {
            "outer_fold": outer_index,
            "top_unit_names": top_names,
            "selected_axes_for_top_units": selected_axes,
            "selected_explained_fraction": [
                float(explained[axis - 1]) for axis in selected_axes if 0 < axis <= len(explained)
            ],
            "conditions": [],
        }
        for condition in args.conditions:
            baseline, subspace, info = fit_scores_for_condition(
                x,
                r,
                u_lag,
                u_target,
                y,
                train,
                test,
                lag_metadata,
                target_metadata,
                condition,
                top_names,
                args,
            )
            fold_info["conditions"].append(info)
            if baseline is None or subspace is None:
                continue
            score_rows[condition]["baseline"][test] = baseline
            score_rows[condition]["subspace"][test] = subspace
            score_rows[condition]["filled"][test] = True
        fold_rows.append(fold_info)

    condition_results = []
    for condition, rows in score_rows.items():
        filled = rows["filled"]
        if not np.any(filled):
            condition_results.append(
                {
                    "target": target,
                    "condition": condition,
                    "trial_count": 0,
                    "baseline_balanced_accuracy": None,
                    "subspace_balanced_accuracy": None,
                    "subspace_increment": None,
                    "subspace_supported": False,
                    "skipped": True,
                }
            )
            continue
        baseline_ba = balanced_accuracy(y[filled], (rows["baseline"][filled] >= 0).astype(int))
        subspace_ba = balanced_accuracy(y[filled], (rows["subspace"][filled] >= 0).astype(int))
        increment = float(subspace_ba - baseline_ba)
        condition_results.append(
            {
                "target": target,
                "condition": condition,
                "trial_count": int(np.sum(filled)),
                "baseline_balanced_accuracy": baseline_ba,
                "subspace_balanced_accuracy": subspace_ba,
                "subspace_increment": increment,
                "subspace_supported": bool(increment > args.min_delta),
                "skipped": False,
            }
        )
    return {"target": target, "conditions": condition_results, "fold_rows": fold_rows}


def evaluate_session(args: argparse.Namespace) -> dict[str, Any]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    task_history, _ = task_history_covariates(trials)

    stim_region, _, stim_region_valid = region_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_label_spikes
    )
    move_region, _, move_region_valid = region_features_for_window(
        probes, trials, MOVEMENT_WINDOW, args.min_label_spikes
    )
    stim_unit, stim_unit_blocks, stim_unit_valid = unit_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    move_unit, move_unit_blocks, move_unit_valid = unit_features_for_window(
        probes, trials, MOVEMENT_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_stim_unit, pre_stim_unit_blocks, pre_stim_valid = unit_features_for_window(
        probes, trials, PRE_STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_move_unit, pre_move_unit_blocks, pre_move_valid = unit_features_for_window(
        probes, trials, PRE_MOVEMENT_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    speed, speed_valid, _ = first_movement_speed_target(trials)
    wheel, wheel_valid, _ = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )

    targets = [
        evaluate_target(
            "first_movement_speed",
            task_history,
            stim_region,
            pre_stim_unit,
            stim_unit,
            unit_metadata(probes, pre_stim_unit_blocks),
            unit_metadata(probes, stim_unit_blocks),
            speed,
            speed_valid & stim_region_valid & stim_unit_valid & pre_stim_valid,
            args,
        ),
        evaluate_target(
            "wheel_action_direction",
            task_history,
            move_region,
            pre_move_unit,
            move_unit,
            unit_metadata(probes, pre_move_unit_blocks),
            unit_metadata(probes, move_unit_blocks),
            wheel,
            wheel_valid & move_region_valid & move_unit_valid & pre_move_valid,
            args,
        ),
    ]
    return {
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "targets": targets,
    }


def candidate_namespace(candidate: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eid=candidate["eid"],
        session_ref=candidate["session_ref"],
        collections=candidate.get("collections", [candidate.get("collection")]),
        folds=args.folds,
        inner_folds=args.inner_folds,
        components=args.components,
        seed=args.seed,
        task_penalty=args.task_penalty,
        region_penalty=args.region_penalty,
        latent_penalty=args.latent_penalty,
        innovation_penalty=args.innovation_penalty,
        min_delta=args.min_delta,
        subspace_size=args.subspace_size,
        min_label_spikes=args.min_label_spikes,
        min_unit_spikes=args.min_unit_spikes,
        max_units_per_probe=args.max_units_per_probe,
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
        conditions=args.conditions,
        probe_label=args.probe_label,
        top_unit_count=args.top_unit_count,
        min_units=args.min_units,
    )


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    target_names = sorted(
        {target["target"] for result in results for target in result["targets"]}
    )
    for target_name in target_names:
        full_mean = None
        for condition in args.conditions:
            condition_rows = [
                condition_row
                for result in results
                for target in result["targets"]
                if target["target"] == target_name
                for condition_row in target["conditions"]
                if condition_row["condition"] == condition
            ]
            increments = [
                float(row["subspace_increment"])
                for row in condition_rows
                if row.get("subspace_increment") is not None
            ]
            supported_count = int(sum(value > args.min_delta for value in increments))
            mean_increment = float(np.mean(increments)) if increments else None
            median_increment = float(np.median(increments)) if increments else None
            if condition == "full":
                full_mean = mean_increment
            rows.append(
                {
                    "target": target_name,
                    "condition": condition,
                    "candidates": len(condition_rows),
                    "evaluated": len(increments),
                    "supported_count": supported_count,
                    "mean_increment": mean_increment,
                    "median_increment": median_increment,
                    "mean_delta_vs_full": (
                        None
                        if mean_increment is None or full_mean is None
                        else float(mean_increment - full_mean)
                    ),
                    "passed": bool(
                        len(increments) > 0
                        and supported_count >= args.min_replications
                        and mean_increment is not None
                        and mean_increment > args.min_delta
                    ),
                }
            )
    return rows


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def make_report(output: dict[str, Any]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx action top-unit sufficiency",
        "",
        "Top units are selected inside each outer fold from train-fitted loading mass.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- probe label: `{output['probe_label']}`",
        f"- top units per fold: {output['top_unit_count']}",
        "",
        "## summary",
        "",
        "| target | condition | evaluated | supported | mean dBA | median dBA | mean delta vs full | passed |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    f"`{row['condition']}`",
                    f"{row['evaluated']}/{row['candidates']}",
                    f"{row['supported_count']}/{row['evaluated']}",
                    fmt(row["mean_increment"]),
                    fmt(row["median_increment"]),
                    fmt(row["mean_delta_vs_full"]),
                    f"`{row['passed']}`",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## verdict",
        "",
        "- `only_top_units` asks whether fold-local top probe units are sufficient.",
        "- `drop_top_units` asks whether the remaining unit ensemble can compensate.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
    )
    parser.add_argument("--output-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--subspace-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--innovation-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=7)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=64)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=0.02)
    parser.add_argument("--conditions", nargs="+", default=["full", "drop_top_units", "only_top_units"])
    parser.add_argument("--probe-label", default="probe00")
    parser.add_argument("--top-unit-count", type=int, default=16)
    parser.add_argument("--min-units", type=int, default=4)
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
        candidates = load_ranked_candidates(str(args.candidates_json), args.candidate_limit)

    results = []
    for candidate in candidates:
        result = evaluate_session(candidate_namespace(candidate, args))
        result["name"] = candidate["name"]
        results.append(result)

    output = {
        "openalyx_url": OPENALYX_URL,
        "candidate_count": len(results),
        "conditions": args.conditions,
        "probe_label": args.probe_label,
        "top_unit_count": args.top_unit_count,
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "components": args.components,
        "subspace_size": args.subspace_size,
        "min_delta": args.min_delta,
        "min_replications": args.min_replications,
        "candidate_results": results,
    }
    output["summary"] = summarize(results, args)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx action top-unit sufficiency")
    for row in output["summary"]:
        print(
            f"  {row['target']} {row['condition']}: "
            f"supported={row['supported_count']}/{row['evaluated']} "
            f"mean={fmt(row['mean_increment'])} "
            f"delta_vs_full={fmt(row['mean_delta_vs_full'])}"
        )
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
