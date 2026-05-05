"""Mouse IBL/OpenAlyx nested innovation-subspace gate.

The directed-axis gate showed that target-specific best innovation axes are
informative, but the stability gate showed that the axis identity is not stable
enough to name a shared axis.  This gate prevents post-hoc test-set axis
selection: for each outer fold it selects a small innovation subspace using
only the outer-train trials, then tests that subspace on the held-out trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import (
    fit_transition_predict,
    load_probe,
    load_ranked_candidates,
    ridge_classifier_scores,
)
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    balanced_accuracy,
    choice_target,
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
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_nested_innovation_subspace_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_nested_innovation_subspace_report.md")


def select_axes_inside_train(
    x_train: np.ndarray,
    r_train: np.ndarray,
    hhat_train: np.ndarray,
    eps_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict[str, object]]]:
    n_axes = eps_train.shape[1]
    baseline_scores = np.zeros(len(y_train), dtype=float)
    axis_scores = [np.zeros(len(y_train), dtype=float) for _ in range(n_axes)]
    for inner_test in stratified_folds(y_train, args.inner_folds, args.seed + 101):
        inner_train = np.setdiff1d(np.arange(len(y_train)), inner_test)
        train_blocks = {
            "X": x_train[inner_train],
            "R": r_train[inner_train],
            "HHAT": hhat_train[inner_train],
        }
        test_blocks = {
            "X": x_train[inner_test],
            "R": r_train[inner_test],
            "HHAT": hhat_train[inner_test],
        }
        penalties = {
            "X": args.task_penalty,
            "R": args.region_penalty,
            "HHAT": args.latent_penalty,
        }
        baseline_scores[inner_test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y_train[inner_train],
            ["X", "R", "HHAT"],
            penalties,
        )
        for axis in range(n_axes):
            axis_name = f"EPS{axis + 1}"
            axis_train_blocks = {
                **train_blocks,
                axis_name: eps_train[inner_train, axis : axis + 1],
            }
            axis_test_blocks = {
                **test_blocks,
                axis_name: eps_train[inner_test, axis : axis + 1],
            }
            axis_penalties = {**penalties, axis_name: args.innovation_penalty}
            axis_scores[axis][inner_test] = ridge_classifier_scores(
                axis_train_blocks,
                axis_test_blocks,
                y_train[inner_train],
                ["X", "R", "HHAT", axis_name],
                axis_penalties,
            )

    baseline_ba = balanced_accuracy(y_train, (baseline_scores >= 0).astype(int))
    axis_rows = []
    for axis, scores in enumerate(axis_scores, start=1):
        ba = balanced_accuracy(y_train, (scores >= 0).astype(int))
        axis_rows.append(
            {
                "axis": axis,
                "inner_balanced_accuracy": ba,
                "inner_increment": float(ba - baseline_ba),
            }
        )
    axis_rows.sort(key=lambda row: row["inner_increment"], reverse=True)
    selected = [int(row["axis"]) for row in axis_rows[: args.subspace_size]]
    return selected, axis_rows


def evaluate_target(
    target: str,
    window: str,
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    y_all: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
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
    baseline_scores = np.zeros(len(y), dtype=float)
    subspace_scores = np.zeros(len(y), dtype=float)
    selected_by_fold = []

    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        hhat_train, hhat_test, eps_train, eps_test = fit_transition_predict(
            x,
            r,
            u_lag,
            u_target,
            train,
            test,
            args,
        )
        selected_axes, inner_axis_rows = select_axes_inside_train(
            x[train],
            r[train],
            hhat_train,
            eps_train,
            y[train],
            args,
        )
        selected_indices = [axis - 1 for axis in selected_axes]
        train_blocks = {"X": x[train], "R": r[train], "HHAT": hhat_train}
        test_blocks = {"X": x[test], "R": r[test], "HHAT": hhat_test}
        penalties = {
            "X": args.task_penalty,
            "R": args.region_penalty,
            "HHAT": args.latent_penalty,
        }
        baseline_scores[test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X", "R", "HHAT"],
            penalties,
        )
        subspace_train_blocks = {
            **train_blocks,
            "EPS_SUB": eps_train[:, selected_indices],
        }
        subspace_test_blocks = {
            **test_blocks,
            "EPS_SUB": eps_test[:, selected_indices],
        }
        subspace_penalties = {**penalties, "EPS_SUB": args.innovation_penalty}
        subspace_scores[test] = ridge_classifier_scores(
            subspace_train_blocks,
            subspace_test_blocks,
            y[train],
            ["X", "R", "HHAT", "EPS_SUB"],
            subspace_penalties,
        )
        selected_by_fold.append(
            {
                "outer_fold": outer_index,
                "selected_axes": selected_axes,
                "inner_top_axis_rows": inner_axis_rows[: args.report_top_axes],
            }
        )

    baseline_ba = balanced_accuracy(y, (baseline_scores >= 0).astype(int))
    subspace_ba = balanced_accuracy(y, (subspace_scores >= 0).astype(int))
    increment = float(subspace_ba - baseline_ba)
    return {
        "target": target,
        "window": window,
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "baseline_model": "task_region_predicted_latent",
        "subspace_model": f"nested_top_{args.subspace_size}_innovation_axes",
        "baseline_balanced_accuracy": baseline_ba,
        "subspace_balanced_accuracy": subspace_ba,
        "subspace_increment": increment,
        "subspace_supported": bool(increment > args.min_delta),
        "selected_axes_by_fold": selected_by_fold,
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
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
    stim_unit, _, stim_unit_valid = unit_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    move_unit, _, move_unit_valid = unit_features_for_window(
        probes, trials, MOVEMENT_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_stim_unit, _, pre_stim_valid = unit_features_for_window(
        probes, trials, PRE_STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_move_unit, _, pre_move_valid = unit_features_for_window(
        probes, trials, PRE_MOVEMENT_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    choice, choice_valid, _ = choice_target(trials)
    speed, speed_valid, _ = first_movement_speed_target(trials)
    wheel, wheel_valid, _ = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )
    targets = [
        evaluate_target(
            "choice_sign",
            "pre_stimulus_to_stimulus",
            task_history,
            stim_region,
            pre_stim_unit,
            stim_unit,
            choice,
            choice_valid & stim_region_valid & stim_unit_valid & pre_stim_valid,
            args,
        ),
        evaluate_target(
            "first_movement_speed",
            "pre_stimulus_to_stimulus",
            task_history,
            stim_region,
            pre_stim_unit,
            stim_unit,
            speed,
            speed_valid & stim_region_valid & stim_unit_valid & pre_stim_valid,
            args,
        ),
        evaluate_target(
            "wheel_action_direction",
            "pre_movement_to_movement",
            task_history,
            move_region,
            pre_move_unit,
            move_unit,
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


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
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
        report_top_axes=args.report_top_axes,
        min_label_spikes=args.min_label_spikes,
        min_unit_spikes=args.min_unit_spikes,
        max_units_per_probe=args.max_units_per_probe,
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
    )


def target_by_name(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {target["target"]: target for target in result["targets"]}


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    target_names = [target["target"] for target in results[0]["targets"]]
    target_summary = []
    for target_name in target_names:
        rows = [
            target
            for result in results
            for target in result["targets"]
            if target["target"] == target_name
        ]
        increments = [row["subspace_increment"] for row in rows]
        supported_count = int(sum(row["subspace_supported"] for row in rows))
        target_summary.append(
            {
                "target": target_name,
                "candidates": len(rows),
                "subspace_supported_count": supported_count,
                "mean_subspace_increment": float(np.mean(increments)),
                "median_subspace_increment": float(np.median(increments)),
                "supported": bool(
                    supported_count >= args.min_replications
                    and float(np.mean(increments)) > args.min_delta
                ),
            }
        )

    session_split_rows = []
    for result in results:
        targets = target_by_name(result)
        choice = targets["choice_sign"]["subspace_increment"]
        speed = targets["first_movement_speed"]["subspace_increment"]
        wheel = targets["wheel_action_direction"]["subspace_increment"]
        action_mean = float(np.mean([speed, wheel]))
        session_split_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "choice_subspace_increment": choice,
                "speed_subspace_increment": speed,
                "wheel_subspace_increment": wheel,
                "action_mean_subspace_increment": action_mean,
                "action_minus_choice_subspace_increment": action_mean - choice,
                "split_supported": bool(action_mean - choice > args.min_split_delta),
            }
        )
    split_values = np.asarray(
        [row["action_minus_choice_subspace_increment"] for row in session_split_rows],
        dtype=float,
    )
    split_supported_count = int(sum(row["split_supported"] for row in session_split_rows))
    return {
        "target_summary": target_summary,
        "session_split_rows": session_split_rows,
        "split_summary": {
            "mean_action_minus_choice_subspace_increment": float(np.mean(split_values)),
            "median_action_minus_choice_subspace_increment": float(np.median(split_values)),
            "split_supported_count": split_supported_count,
            "split_supported": bool(
                split_supported_count >= args.min_split_replications
                and float(np.mean(split_values)) > args.min_split_delta
            ),
        },
        "nested_subspace_gate_passed": bool(any(row["supported"] for row in target_summary)),
    }


def make_report(output: dict[str, object]) -> str:
    split = output["split_summary"]
    lines = [
        "# Mouse IBL/OpenAlyx nested innovation-subspace gate",
        "",
        "$$",
        "y_t=g(X_t,R_t,\\hat H_t,\\epsilon_{t,S_{train}})",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- components: {output['components']}",
        f"- outer folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- subspace size: {output['subspace_size']}",
        f"- nested subspace gate passed: `{output['nested_subspace_gate_passed']}`",
        "",
        "## target summary",
        "",
        "| target | candidates | subspace supported | mean dBA | median dBA | supported |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in output["target_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    str(row["candidates"]),
                    str(row["subspace_supported_count"]),
                    f"{row['mean_subspace_increment']:.6f}",
                    f"{row['median_subspace_increment']:.6f}",
                    f"`{row['supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## choice/action split",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| mean action - choice subspace dBA | {split['mean_action_minus_choice_subspace_increment']:.6f} |",
            f"| median action - choice subspace dBA | {split['median_action_minus_choice_subspace_increment']:.6f} |",
            f"| split supported | {split['split_supported_count']}/{output['candidate_count']} |",
            "",
            "## per-session split",
            "",
            "| candidate | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in output["session_split_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['choice_subspace_increment']:.6f}",
                    f"{row['speed_subspace_increment']:.6f}",
                    f"{row['wheel_subspace_increment']:.6f}",
                    f"{row['action_minus_choice_subspace_increment']:.6f}",
                    f"`{row['split_supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Axes are selected inside the outer train fold, so outer test trials do not choose the subspace.",
            "- A positive result supports a reproducible train-selected innovation subspace rather than pure post-hoc best-axis selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--report-top-axes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--innovation-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=7)
    parser.add_argument("--min-split-delta", type=float, default=0.002)
    parser.add_argument("--min-split-replications", type=int, default=7)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=64)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=0.02)
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
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "components": args.components,
        "subspace_size": args.subspace_size,
        "task_penalty": args.task_penalty,
        "region_penalty": args.region_penalty,
        "latent_penalty": args.latent_penalty,
        "innovation_penalty": args.innovation_penalty,
        "min_delta": args.min_delta,
        "min_replications": args.min_replications,
        "min_split_delta": args.min_split_delta,
        "min_split_replications": args.min_split_replications,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": results,
    }
    output.update(summarize(results, args))
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    args.report_md.write_text(make_report(output))
    print("Mouse IBL/OpenAlyx nested innovation-subspace gate")
    for row in output["target_summary"]:
        print(
            f"  {row['target']}: subspace={row['subspace_supported_count']}/"
            f"{row['candidates']} mean={row['mean_subspace_increment']:.6f}"
        )
    print(
        "  action-choice="
        f"{output['split_summary']['mean_action_minus_choice_subspace_increment']:.6f} "
        f"split={output['split_summary']['split_supported_count']}/{output['candidate_count']}"
    )
    print(f"  nested_subspace_gate_passed={output['nested_subspace_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
