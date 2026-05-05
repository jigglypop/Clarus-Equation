"""Mouse IBL/OpenAlyx directed latent-axis split gate.

The previous 12-session gate showed that the full innovation vector helps
action targets more than choice.  This gate splits the innovation vector into
PCA/transition components and asks whether single innovation axes carry action
readout after task, region, and the predictable latent trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


DEFAULT_CANDIDATES_JSON = Path(__file__).with_name(
    "mouse_ibl_channel_fallback_registered_panel_ranker_results.json"
)
RESULT_JSON = Path(__file__).with_name("mouse_ibl_directed_latent_axis_split_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_directed_latent_axis_split_report.md")


def axis_scores_for_target(
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
    n_axes = args.components
    baseline_scores = np.zeros(len(y), dtype=float)
    axis_scores = [np.zeros(len(y), dtype=float) for _ in range(n_axes)]

    for test in stratified_folds(y, args.folds, args.seed):
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
        for axis in range(n_axes):
            axis_name = f"EPS{axis + 1}"
            axis_train_blocks = {
                **train_blocks,
                axis_name: eps_train[:, axis : axis + 1],
            }
            axis_test_blocks = {
                **test_blocks,
                axis_name: eps_test[:, axis : axis + 1],
            }
            axis_penalties = {**penalties, axis_name: args.innovation_penalty}
            axis_scores[axis][test] = ridge_classifier_scores(
                axis_train_blocks,
                axis_test_blocks,
                y[train],
                ["X", "R", "HHAT", axis_name],
                axis_penalties,
            )

    baseline_pred = (baseline_scores >= 0).astype(int)
    baseline_ba = balanced_accuracy(y, baseline_pred)
    axes = []
    for axis, scores in enumerate(axis_scores, start=1):
        pred = (scores >= 0).astype(int)
        ba = balanced_accuracy(y, pred)
        increment = float(ba - baseline_ba)
        axes.append(
            {
                "axis": axis,
                "balanced_accuracy": ba,
                "increment_after_task_region_hhat": increment,
                "supported": bool(increment > args.min_delta),
            }
        )
    axes.sort(key=lambda row: row["increment_after_task_region_hhat"], reverse=True)
    best = axes[0]
    return {
        "target": target,
        "window": window,
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "baseline_model": "task_region_predicted_latent",
        "baseline_balanced_accuracy": baseline_ba,
        "best_axis": best["axis"],
        "best_axis_increment": best["increment_after_task_region_hhat"],
        "best_axis_supported": best["supported"],
        "positive_axis_count": int(sum(row["supported"] for row in axes)),
        "axis_rows": axes,
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
        axis_scores_for_target(
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
        axis_scores_for_target(
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
        axis_scores_for_target(
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
        components=args.components,
        seed=args.seed,
        task_penalty=args.task_penalty,
        region_penalty=args.region_penalty,
        latent_penalty=args.latent_penalty,
        innovation_penalty=args.innovation_penalty,
        min_delta=args.min_delta,
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
        best_increments = [row["best_axis_increment"] for row in rows]
        target_summary.append(
            {
                "target": target_name,
                "candidates": len(rows),
                "best_axis_supported_count": int(
                    sum(row["best_axis_supported"] for row in rows)
                ),
                "mean_best_axis_increment": float(np.mean(best_increments)),
                "median_best_axis_increment": float(np.median(best_increments)),
                "mean_positive_axis_count": float(
                    np.mean([row["positive_axis_count"] for row in rows])
                ),
                "supported": bool(
                    sum(row["best_axis_supported"] for row in rows)
                    >= args.min_replications
                    and float(np.mean(best_increments)) > args.min_delta
                ),
            }
        )

    split_rows = []
    for result in results:
        targets = target_by_name(result)
        choice = targets["choice_sign"]["best_axis_increment"]
        speed = targets["first_movement_speed"]["best_axis_increment"]
        wheel = targets["wheel_action_direction"]["best_axis_increment"]
        action_mean = float(np.mean([speed, wheel]))
        split_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "choice_best_axis": targets["choice_sign"]["best_axis"],
                "speed_best_axis": targets["first_movement_speed"]["best_axis"],
                "wheel_best_axis": targets["wheel_action_direction"]["best_axis"],
                "choice_best_increment": choice,
                "speed_best_increment": speed,
                "wheel_best_increment": wheel,
                "action_mean_best_increment": action_mean,
                "action_minus_choice_best_increment": action_mean - choice,
                "split_supported": bool(action_mean - choice > args.min_split_delta),
            }
        )
    split_values = np.asarray(
        [row["action_minus_choice_best_increment"] for row in split_rows], dtype=float
    )
    split_supported_count = int(sum(row["split_supported"] for row in split_rows))
    return {
        "target_summary": target_summary,
        "split_summary": {
            "mean_action_minus_choice_best_axis_increment": float(np.mean(split_values)),
            "median_action_minus_choice_best_axis_increment": float(np.median(split_values)),
            "split_supported_count": split_supported_count,
            "split_supported": bool(
                split_supported_count >= args.min_split_replications
                and float(np.mean(split_values)) > args.min_split_delta
            ),
        },
        "session_split_rows": split_rows,
        "directed_axis_gate_passed": bool(
            any(row["supported"] for row in target_summary)
            and split_supported_count >= args.min_split_replications
        ),
    }


def make_report(output: dict[str, object]) -> str:
    split = output["split_summary"]
    lines = [
        "# Mouse IBL/OpenAlyx directed latent-axis split gate",
        "",
        "$$",
        "y_{action}=g_a(X_t,R_t,\\hat H_t,\\epsilon_{t,k})",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- components: {output['components']}",
        f"- folds: {output['folds']}",
        f"- directed axis gate passed: `{output['directed_axis_gate_passed']}`",
        "",
        "## target summary",
        "",
        "| target | candidates | best axis supported | mean best dBA | median best dBA | mean positive axes | supported |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["target_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    str(row["candidates"]),
                    str(row["best_axis_supported_count"]),
                    f"{row['mean_best_axis_increment']:.6f}",
                    f"{row['median_best_axis_increment']:.6f}",
                    f"{row['mean_positive_axis_count']:.3f}",
                    f"`{row['supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## choice/action split on best axes",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| mean action - choice best-axis dBA | {split['mean_action_minus_choice_best_axis_increment']:.6f} |",
            f"| median action - choice best-axis dBA | {split['median_action_minus_choice_best_axis_increment']:.6f} |",
            f"| split supported | {split['split_supported_count']}/{output['candidate_count']} |",
            "",
            "## per-session best axes",
            "",
            "| candidate | choice axis | speed axis | wheel axis | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in output["session_split_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    str(row["choice_best_axis"]),
                    str(row["speed_best_axis"]),
                    str(row["wheel_best_axis"]),
                    f"{row['choice_best_increment']:.6f}",
                    f"{row['speed_best_increment']:.6f}",
                    f"{row['wheel_best_increment']:.6f}",
                    f"{row['action_minus_choice_best_increment']:.6f}",
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
            "- This gate asks whether a single innovation component can carry target-specific readout.",
            "- If action best-axis increments exceed choice best-axis increments, the next model should split directed action axes from policy/choice readout.",
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
    parser.add_argument("--components", type=int, default=12)
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
        "components": args.components,
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
    print("Mouse IBL/OpenAlyx directed latent-axis split gate")
    for row in output["target_summary"]:
        print(
            f"  {row['target']}: best_axis={row['best_axis_supported_count']}/"
            f"{row['candidates']} mean={row['mean_best_axis_increment']:.6f}"
        )
    print(
        "  action-choice="
        f"{output['split_summary']['mean_action_minus_choice_best_axis_increment']:.6f} "
        f"split={output['split_summary']['split_supported_count']}/{output['candidate_count']}"
    )
    print(f"  directed_axis_gate_passed={output['directed_axis_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
