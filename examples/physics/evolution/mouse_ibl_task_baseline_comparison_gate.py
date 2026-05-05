"""Mouse IBL/OpenAlyx task-baseline comparison gate.

The channel-region rescue gate reduced the anatomical ``unknown`` caveat.  The
next counterexample is behavioral: perhaps the region decoder only recovers
trial timing, stimulus table covariates, or slow session drift.  This script
reuses the rescued hybrid region features and compares them with trial-table
baselines that avoid current-choice/current-movement leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_channel_region_rescue_gate import (
    CANDIDATES,
    REPORT_MD as RESCUE_REPORT_MD,
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
    choice_target,
    evaluate_target,
    first_movement_speed_target,
    safe_float_array,
    wheel_action_direction_target,
    window_bounds,
)


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_task_baseline_comparison_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_task_baseline_comparison_report.md"
)


def finite_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    if np.all(missing):
        filled = np.zeros_like(values, dtype=float)
    else:
        filled = values.copy()
        filled[missing] = float(np.nanmedian(values))
    return np.column_stack([filled, missing.astype(float)])


def lag(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[0] = fill
    result[1:] = values[:-1]
    return result


def column(trials, name: str, default: float = np.nan) -> np.ndarray:
    if name not in trials:
        return np.full(len(trials), default, dtype=float)
    return safe_float_array(trials[name])


def add_feature(
    matrices: list[np.ndarray],
    names: list[str],
    name: str,
    values: np.ndarray,
    *,
    with_missing_indicator: bool = True,
) -> None:
    block = finite_column(values)
    matrices.append(block[:, :1])
    names.append(name)
    if with_missing_indicator and np.any(block[:, 1] > 0):
        matrices.append(block[:, 1:2])
        names.append(f"{name}_missing")


def timing_covariates(trials) -> tuple[np.ndarray, list[str]]:
    n_trials = len(trials)
    matrices: list[np.ndarray] = []
    names: list[str] = []
    trial_index = np.linspace(-1.0, 1.0, n_trials) if n_trials > 1 else np.zeros(1)
    stim_on = column(trials, "stimOn_times")
    go_cue = column(trials, "goCue_times")
    intervals_0 = column(trials, "intervals_0")

    if np.any(np.isfinite(stim_on)):
        session_time = stim_on - float(np.nanmin(stim_on))
    else:
        session_time = np.zeros(n_trials, dtype=float)

    add_feature(matrices, names, "trial_index", trial_index, with_missing_indicator=False)
    add_feature(matrices, names, "session_time_from_first_stim", session_time)
    add_feature(matrices, names, "stim_delay_from_interval_start", stim_on - intervals_0)
    add_feature(matrices, names, "go_cue_delay_from_stim", go_cue - stim_on)
    return hstack(matrices), names


def task_history_covariates(trials) -> tuple[np.ndarray, list[str]]:
    timing, timing_names = timing_covariates(trials)
    matrices = [timing]
    names = list(timing_names)

    left = column(trials, "contrastLeft")
    right = column(trials, "contrastRight")
    left0 = np.where(np.isfinite(left), left, 0.0)
    right0 = np.where(np.isfinite(right), right, 0.0)
    signed_contrast = right0 - left0
    absolute_contrast = np.maximum(np.abs(left0), np.abs(right0))
    stimulus_side = np.where(np.isfinite(right), 1.0, np.where(np.isfinite(left), -1.0, 0.0))
    probability_left = column(trials, "probabilityLeft")

    add_feature(matrices, names, "signed_contrast", signed_contrast)
    add_feature(matrices, names, "absolute_contrast", absolute_contrast)
    add_feature(matrices, names, "stimulus_side", stimulus_side)
    add_feature(matrices, names, "probability_left", probability_left)

    choice = column(trials, "choice")
    feedback = column(trials, "feedbackType")
    reward = column(trials, "rewardVolume")
    stim_on = column(trials, "stimOn_times")
    first_movement = column(trials, "firstMovement_times")
    response = column(trials, "response_times")
    reaction_latency = first_movement - stim_on
    response_latency = response - stim_on

    add_feature(matrices, names, "previous_choice", lag(np.sign(choice)))
    add_feature(matrices, names, "previous_feedback", lag(feedback))
    add_feature(matrices, names, "previous_reward", lag(reward))
    add_feature(matrices, names, "previous_signed_contrast", lag(signed_contrast))
    add_feature(matrices, names, "previous_absolute_contrast", lag(absolute_contrast))
    add_feature(matrices, names, "previous_reaction_latency", lag(reaction_latency))
    add_feature(matrices, names, "previous_response_latency", lag(response_latency))
    return hstack(matrices), names


def build_models(
    trials,
    region_models: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    timing, timing_names = timing_covariates(trials)
    task_history, task_history_names = task_history_covariates(trials)
    hybrid = region_models["hybrid_acronym_channel_id_by_probe"]
    global_rate = region_models["global_rate"]
    return (
        {
            "timing_only": timing,
            "task_history": task_history,
            "hybrid_region_by_probe": hybrid,
            "task_history_plus_hybrid_region": hstack([task_history, hybrid]),
            "global_rate": global_rate,
        },
        {
            "timing_feature_names": timing_names,
            "task_history_feature_names": task_history_names,
            "timing_feature_count": int(timing.shape[1]),
            "task_history_feature_count": int(task_history.shape[1]),
            "hybrid_region_feature_count": int(hybrid.shape[1]),
            "task_history_plus_hybrid_feature_count": int(task_history.shape[1] + hybrid.shape[1]),
        },
    )


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    timing = by_model["timing_only"]["balanced_accuracy"]
    task = by_model["task_history"]["balanced_accuracy"]
    hybrid = by_model["hybrid_region_by_probe"]["balanced_accuracy"]
    task_plus_hybrid = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_timing_only"] = float(row["balanced_accuracy"] - timing)
        row["delta_vs_task_history"] = float(row["balanced_accuracy"] - task)
    target["baseline_comparison"] = {
        "timing_only_balanced_accuracy": timing,
        "task_history_balanced_accuracy": task,
        "hybrid_region_balanced_accuracy": hybrid,
        "task_history_plus_hybrid_balanced_accuracy": task_plus_hybrid,
        "hybrid_delta_vs_timing_only": float(hybrid - timing),
        "hybrid_delta_vs_task_history": float(hybrid - task),
        "task_plus_hybrid_delta_vs_task_history": float(task_plus_hybrid - task),
        "hybrid_beats_timing": bool(hybrid > timing + min_delta),
        "task_plus_hybrid_beats_task_history": bool(
            task_plus_hybrid > task + min_delta
        ),
    }
    return target


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]

    stim_start, stim_end, stim_valid = window_bounds(trials, STIMULUS_WINDOW)
    move_start, move_end, move_valid = window_bounds(trials, MOVEMENT_WINDOW)
    stimulus_blocks = [
        probe_feature_block(
            probe,
            stim_start,
            stim_end,
            stim_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]
    movement_blocks = [
        probe_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]

    choice, choice_valid, choice_meta = choice_target(trials)
    speed, speed_valid, speed_meta = first_movement_speed_target(trials)
    wheel_direction, wheel_valid, wheel_meta = wheel_action_direction_target(
        trials,
        wheel_timestamps,
        wheel_position,
        args.min_abs_wheel_displacement,
    )

    stimulus_models, stimulus_model_meta = build_models(
        trials,
        make_region_models(stimulus_blocks),
    )
    movement_models, movement_model_meta = build_models(
        trials,
        make_region_models(movement_blocks),
    )
    targets = [
        annotate_target(
            evaluate_target(
                target_name="choice_sign",
                window_name=STIMULUS_WINDOW.name,
                x_models=stimulus_models,
                y_all=choice,
                valid=choice_valid & stim_valid,
                folds=args.folds,
                ridge=args.ridge,
                permutations=args.permutations,
                seed=args.seed,
            ),
            args.min_delta,
        ),
        annotate_target(
            evaluate_target(
                target_name="first_movement_speed",
                window_name=STIMULUS_WINDOW.name,
                x_models=stimulus_models,
                y_all=speed,
                valid=speed_valid & stim_valid,
                folds=args.folds,
                ridge=args.ridge,
                permutations=args.permutations,
                seed=args.seed,
            ),
            args.min_delta,
        ),
        annotate_target(
            evaluate_target(
                target_name="wheel_action_direction",
                window_name=MOVEMENT_WINDOW.name,
                x_models=movement_models,
                y_all=wheel_direction,
                valid=wheel_valid & move_valid,
                folds=args.folds,
                ridge=args.ridge,
                permutations=args.permutations,
                seed=args.seed,
            ),
            args.min_delta,
        ),
    ]

    probe_summaries = [
        summarize_probe(probe, stimulus_block)
        for probe, stimulus_block in zip(probes, stimulus_blocks)
    ]
    return {
        "openalyx_url": OPENALYX_URL,
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "wheel_sample_count": int(len(wheel_timestamps)),
        "probe_summaries": probe_summaries,
        "stimulus_model_meta": stimulus_model_meta,
        "movement_model_meta": movement_model_meta,
        "target_metadata": {
            "choice_sign": choice_meta,
            "first_movement_speed": speed_meta,
            "wheel_action_direction": wheel_meta,
        },
        "folds": int(args.folds),
        "ridge": float(args.ridge),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "min_delta": float(args.min_delta),
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
        ridge=args.ridge,
        permutations=args.permutations,
        seed=args.seed,
        min_label_spikes=args.min_label_spikes,
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
        min_delta=args.min_delta,
    )


def target_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        comparison = target["baseline_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["hybrid_region_by_probe"]["n_trials"],
                "class_counts": by_model["hybrid_region_by_probe"]["class_counts"],
                "timing_only_balanced_accuracy": comparison[
                    "timing_only_balanced_accuracy"
                ],
                "task_history_balanced_accuracy": comparison[
                    "task_history_balanced_accuracy"
                ],
                "hybrid_region_balanced_accuracy": comparison[
                    "hybrid_region_balanced_accuracy"
                ],
                "task_history_plus_hybrid_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_balanced_accuracy"
                ],
                "hybrid_delta_vs_timing_only": comparison[
                    "hybrid_delta_vs_timing_only"
                ],
                "hybrid_delta_vs_task_history": comparison[
                    "hybrid_delta_vs_task_history"
                ],
                "task_plus_hybrid_delta_vs_task_history": comparison[
                    "task_plus_hybrid_delta_vs_task_history"
                ],
                "hybrid_beats_timing": comparison["hybrid_beats_timing"],
                "task_plus_hybrid_beats_task_history": comparison[
                    "task_plus_hybrid_beats_task_history"
                ],
                "hybrid_passed_global_gate": bool(
                    by_model["hybrid_region_by_probe"]["passed"]
                ),
                "task_plus_hybrid_passed_global_gate": bool(
                    by_model["task_history_plus_hybrid_region"]["passed"]
                ),
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
        "target_rows": rows,
        "hybrid_beats_timing_count": sum(row["hybrid_beats_timing"] for row in rows),
        "task_plus_hybrid_beats_task_count": sum(
            row["task_plus_hybrid_beats_task_history"] for row in rows
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
            "hybrid_beats_timing_count": sum(row["hybrid_beats_timing"] for row in rows),
            "task_plus_hybrid_beats_task_count": sum(
                row["task_plus_hybrid_beats_task_history"] for row in rows
            ),
            "hybrid_global_gate_count": sum(
                row["hybrid_passed_global_gate"] for row in rows
            ),
            "task_plus_hybrid_global_gate_count": sum(
                row["task_plus_hybrid_passed_global_gate"] for row in rows
            ),
            "mean_timing_only_balanced_accuracy": sum(
                row["timing_only_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_balanced_accuracy": sum(
                row["task_history_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_hybrid_region_balanced_accuracy": sum(
                row["hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_balanced_accuracy": sum(
                row["task_history_plus_hybrid_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_hybrid_delta_vs_timing_only": sum(
                row["hybrid_delta_vs_timing_only"] for row in rows
            )
            / len(rows),
            "mean_hybrid_delta_vs_task_history": sum(
                row["hybrid_delta_vs_task_history"] for row in rows
            )
            / len(rows),
            "mean_task_plus_hybrid_delta_vs_task_history": sum(
                row["task_plus_hybrid_delta_vs_task_history"] for row in rows
            )
            / len(rows),
        }

    timing_rejected = (
        target_replication.get("choice_sign", {}).get("hybrid_beats_timing_count", 0) >= 3
        and target_replication.get("wheel_action_direction", {}).get(
            "hybrid_beats_timing_count",
            0,
        )
        >= 3
        and target_replication.get("first_movement_speed", {}).get(
            "hybrid_beats_timing_count",
            0,
        )
        >= 3
    )
    task_increment_supported = (
        target_replication.get("choice_sign", {}).get(
            "task_plus_hybrid_beats_task_count",
            0,
        )
        >= 3
        and target_replication.get("wheel_action_direction", {}).get(
            "task_plus_hybrid_beats_task_count",
            0,
        )
        >= 3
    )
    return {
        "candidate_count": len(candidates),
        "folds": args.folds,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "seed": args.seed,
        "min_label_spikes": args.min_label_spikes,
        "min_delta": args.min_delta,
        "rescue_report": str(RESCUE_REPORT_MD),
        "timing_counterexample_rejected": bool(timing_rejected),
        "task_increment_supported": bool(task_increment_supported),
        "baseline_gate_passed": bool(timing_rejected and task_increment_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx task-baseline comparison gate",
        "",
        "Channel-region rescue 뒤의 다음 반례는 region decoder가 단순 trial timing, stimulus table, previous-trial history만 읽는 경우다.",
        "이 gate는 current choice, current first movement, current response, current feedback을 baseline feature에서 제외하고 hybrid region feature와 비교한다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- min delta for baseline win: {output['min_delta']}",
        f"- timing counterexample rejected: `{output['timing_counterexample_rejected']}`",
        f"- task increment supported: `{output['task_increment_supported']}`",
        f"- baseline gate passed: `{output['baseline_gate_passed']}`",
        "",
        "## models",
        "",
        "| model | feature definition | leakage rule |",
        "|---|---|---|",
        "| `timing_only` | trial index, session time, stim delay, go-cue delay | no choice/movement/outcome |",
        "| `task_history` | timing + current stimulus/probability + previous trial choice/reward/latency | no current choice/movement/outcome |",
        "| `hybrid_region_by_probe` | channel-rescued probe-region spike rates | neural window feature |",
        "| `task_history_plus_hybrid_region` | task history baseline plus hybrid region feature | incremental neural contribution |",
        "| `global_rate` | one scalar total hybrid firing rate | flat firing-rate baseline |",
        "",
        "The main comparison is",
        "",
        "$$",
        "\\Delta_{\\mathrm{timing}}=\\mathrm{BA}(R^{\\mathrm{hybrid}})-\\mathrm{BA}(X^{\\mathrm{timing}}),",
        "\\qquad",
        "\\Delta_{\\mathrm{task}}=\\mathrm{BA}([X^{\\mathrm{task}},R^{\\mathrm{hybrid}}])-\\mathrm{BA}(X^{\\mathrm{task}}).",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | hybrid beats timing | task+hybrid beats task | mean timing BA | mean task BA | mean hybrid BA | mean task+hybrid BA | mean delta timing | mean delta task |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["hybrid_beats_timing_count"]),
                    str(row["task_plus_hybrid_beats_task_count"]),
                    f"{row['mean_timing_only_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_balanced_accuracy']:.6f}",
                    f"{row['mean_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_hybrid_balanced_accuracy']:.6f}",
                    f"{row['mean_hybrid_delta_vs_timing_only']:.6f}",
                    f"{row['mean_task_plus_hybrid_delta_vs_task_history']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | hybrid beats timing | task+hybrid beats task | choice delta timing | choice delta task | speed delta timing | speed delta task | wheel delta timing | wheel delta task |",
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
                    str(summary["hybrid_beats_timing_count"]),
                    str(summary["task_plus_hybrid_beats_task_count"]),
                    f"{by_target['choice_sign']['hybrid_delta_vs_timing_only']:.6f}",
                    f"{by_target['choice_sign']['task_plus_hybrid_delta_vs_task_history']:.6f}",
                    f"{by_target['first_movement_speed']['hybrid_delta_vs_timing_only']:.6f}",
                    f"{by_target['first_movement_speed']['task_plus_hybrid_delta_vs_task_history']:.6f}",
                    f"{by_target['wheel_action_direction']['hybrid_delta_vs_timing_only']:.6f}",
                    f"{by_target['wheel_action_direction']['task_plus_hybrid_delta_vs_task_history']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## per-candidate target details", ""])
    for summary in output["summaries"]:
        lines.extend(
            [
                f"### {summary['name']}",
                "",
                f"- eid: `{summary['eid']}`",
                f"- session: `{summary['session_ref']}`",
                f"- reason: {summary['reason']}",
                "",
                "| target | n | timing BA | task BA | hybrid BA | task+hybrid BA | hybrid-timing | hybrid-task | task+hybrid-task |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["target_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['target']}`",
                        str(row["n_trials"]),
                        f"{row['timing_only_balanced_accuracy']:.6f}",
                        f"{row['task_history_balanced_accuracy']:.6f}",
                        f"{row['hybrid_region_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_hybrid_balanced_accuracy']:.6f}",
                        f"{row['hybrid_delta_vs_timing_only']:.6f}",
                        f"{row['hybrid_delta_vs_task_history']:.6f}",
                        f"{row['task_plus_hybrid_delta_vs_task_history']:.6f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- timing counterexample rejected: `{output['timing_counterexample_rejected']}`",
            f"- task increment supported: `{output['task_increment_supported']}`",
            f"- baseline gate passed: `{output['baseline_gate_passed']}`",
            "",
            "해석:",
            "",
            "- Hybrid region feature가 timing-only baseline을 반복적으로 넘으면, 단순 session drift 또는 trial clock 반례가 약해진다.",
            "- `task_history_plus_hybrid_region`이 `task_history`를 넘으면, current stimulus와 previous-trial history만으로는 남는 neural increment가 있다는 뜻이다.",
            "- 이 gate는 causal proof가 아니다. 다만 mouse region/probe readout을 task-table artifact보다 더 강한 형태로 걸러낸다.",
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
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--min-label-spikes", type=int, default=100_000)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=1e-3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx task-baseline comparison gate")
    print(f"  candidates={output['candidate_count']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} hybrid>timing={row['hybrid_beats_timing_count']}/"
            + f"{row['candidate_count']} task+hybrid>task="
            + f"{row['task_plus_hybrid_beats_task_count']}/"
            + f"{row['candidate_count']} mean_delta_task="
            + f"{row['mean_task_plus_hybrid_delta_vs_task_history']:.6f}"
        )
    print(f"  baseline_gate_passed={output['baseline_gate_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
