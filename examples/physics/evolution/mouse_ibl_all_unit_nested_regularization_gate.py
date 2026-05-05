"""Mouse IBL/OpenAlyx all-unit nested-regularization gate.

The flat-unit comparison showed that a bounded top-unit readout can dominate
hybrid anatomical region bins for choice and wheel direction.  This stronger
gate asks a more precise question: after task/history covariates and rescued
region bins have already been included, does high-spike unit identity still add
cross-validated decoding information?

The nested comparison is

    M_X:      y ~ X_task
    M_XR:     y ~ [X_task, R_hybrid]
    M_XU:     y ~ [X_task, U_all]
    M_XRU:    y ~ [X_task, R_hybrid, U_all]

and the main residual is

    Delta_{U | X,R} = BA(M_XRU) - BA(M_XR).

This is not yet a GLM coupling estimate.  It is a regularized nested decoder
that decides whether the mouse term should keep explicit unit identity before
moving on to unit-to-unit or unit-to-region coupling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_channel_region_rescue_gate import (
    CANDIDATES,
    load_probe,
    make_models as make_region_models,
    probe_feature_block,
    summarize_probe,
)
from mouse_ibl_flat_unit_region_comparison_gate import OTHER_UNITS_LABEL
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    choice_target,
    evaluate_target,
    first_movement_speed_target,
    wheel_action_direction_target,
    window_bounds,
    window_features,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_all_unit_nested_regularization_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_all_unit_nested_regularization_report.md"
)


def active_unit_group(
    spike_clusters: np.ndarray,
    min_unit_spikes: int,
    max_units: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    max_cluster_id = int(np.max(spike_clusters))
    counts = np.bincount(spike_clusters, minlength=max_cluster_id + 1)
    active = np.where(counts >= min_unit_spikes)[0]
    ranked = sorted(
        ((int(cluster_id), int(counts[cluster_id])) for cluster_id in active),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ranked if max_units <= 0 else ranked[:max_units]
    group_names = [f"cluster:{cluster_id}" for cluster_id, _ in selected]
    group_names.append(OTHER_UNITS_LABEL)
    other_idx = len(group_names) - 1
    cluster_group = np.full(max_cluster_id + 1, other_idx, dtype=np.int32)
    for group_idx, (cluster_id, _) in enumerate(selected):
        cluster_group[cluster_id] = group_idx
    spike_counts = {f"cluster:{cluster_id}": count for cluster_id, count in ranked}
    selected_ids = {cluster_id for cluster_id, _ in selected}
    spike_counts[OTHER_UNITS_LABEL] = int(
        sum(
            int(counts[cluster_id])
            for cluster_id in range(len(counts))
            if cluster_id not in selected_ids
        )
    )
    return group_names, cluster_group, spike_counts


def all_unit_feature_block(
    probe: dict[str, object],
    starts: np.ndarray,
    ends: np.ndarray,
    valid_window: np.ndarray,
    min_unit_spikes: int,
    max_units_per_probe: int,
) -> dict[str, object]:
    spike_times = probe["spike_times"]
    spike_clusters = probe["spike_clusters"]
    unit_names, unit_group, unit_counts = active_unit_group(
        spike_clusters,
        min_unit_spikes,
        max_units_per_probe,
    )
    return {
        "all_unit_features": window_features(
            spike_times,
            spike_clusters,
            unit_group,
            starts,
            ends,
            valid_window,
            len(unit_names),
        ),
        "all_unit_feature_names": [f"{probe['label']}:{name}" for name in unit_names],
        "all_unit_spike_counts": {
            f"{probe['label']}:{name}": int(count)
            for name, count in unit_counts.items()
        },
        "selected_unit_count": int(len(unit_names) - 1),
        "feature_count": int(len(unit_names)),
    }


def make_all_unit_models(unit_blocks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    unit = hstack([block["all_unit_features"] for block in unit_blocks])
    return {
        "all_unit_by_probe": unit,
        "all_unit_global_rate": np.nansum(unit, axis=1, keepdims=True),
    }


def build_models(
    trials,
    region_models: dict[str, np.ndarray],
    unit_models: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    task_history, task_history_names = task_history_covariates(trials)
    region = region_models["hybrid_acronym_channel_id_by_probe"]
    unit = unit_models["all_unit_by_probe"]
    global_rate = region_models["global_rate"]
    return (
        {
            "task_history": task_history,
            "task_history_plus_hybrid_region": hstack([task_history, region]),
            "task_history_plus_all_unit": hstack([task_history, unit]),
            "task_history_plus_hybrid_region_plus_all_unit": hstack(
                [task_history, region, unit]
            ),
            "global_rate": global_rate,
        },
        {
            "task_history_feature_names": task_history_names,
            "task_history_feature_count": int(task_history.shape[1]),
            "hybrid_region_feature_count": int(region.shape[1]),
            "all_unit_feature_count": int(unit.shape[1]),
            "task_region_feature_count": int(task_history.shape[1] + region.shape[1]),
            "task_unit_feature_count": int(task_history.shape[1] + unit.shape[1]),
            "task_region_unit_feature_count": int(
                task_history.shape[1] + region.shape[1] + unit.shape[1]
            ),
        },
    )


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    task = by_model["task_history"]["balanced_accuracy"]
    task_region = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    task_unit = by_model["task_history_plus_all_unit"]["balanced_accuracy"]
    task_region_unit = by_model[
        "task_history_plus_hybrid_region_plus_all_unit"
    ]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_task_history"] = float(row["balanced_accuracy"] - task)
        row["delta_vs_task_region"] = float(row["balanced_accuracy"] - task_region)
        row["delta_vs_task_unit"] = float(row["balanced_accuracy"] - task_unit)
        row["delta_vs_task_region_unit"] = float(
            row["balanced_accuracy"] - task_region_unit
        )
    target["nested_comparison"] = {
        "task_history_balanced_accuracy": task,
        "task_history_plus_hybrid_region_balanced_accuracy": task_region,
        "task_history_plus_all_unit_balanced_accuracy": task_unit,
        "task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": task_region_unit,
        "region_increment_after_task": float(task_region - task),
        "unit_increment_after_task": float(task_unit - task),
        "unit_increment_after_task_region": float(task_region_unit - task_region),
        "region_increment_after_task_unit": float(task_region_unit - task_unit),
        "unit_beats_region_after_task": bool(task_unit > task_region + min_delta),
        "unit_residual_after_task_region": bool(
            task_region_unit > task_region + min_delta
        ),
        "region_residual_after_task_unit": bool(
            task_region_unit > task_unit + min_delta
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
    stimulus_region_blocks = [
        probe_feature_block(
            probe,
            stim_start,
            stim_end,
            stim_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]
    movement_region_blocks = [
        probe_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
            args.min_label_spikes,
        )
        for probe in probes
    ]
    stimulus_unit_blocks = [
        all_unit_feature_block(
            probe,
            stim_start,
            stim_end,
            stim_valid,
            args.min_unit_spikes,
            args.max_units_per_probe,
        )
        for probe in probes
    ]
    movement_unit_blocks = [
        all_unit_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
            args.min_unit_spikes,
            args.max_units_per_probe,
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
        make_region_models(stimulus_region_blocks),
        make_all_unit_models(stimulus_unit_blocks),
    )
    movement_models, movement_model_meta = build_models(
        trials,
        make_region_models(movement_region_blocks),
        make_all_unit_models(movement_unit_blocks),
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
        for probe, stimulus_block in zip(probes, stimulus_region_blocks)
    ]
    unit_summaries = [
        {
            "collection": probe["collection"],
            "selected_unit_count": block["selected_unit_count"],
            "unit_feature_count": block["feature_count"],
            "top_unit_spike_counts": sorted(
                (
                    (name, count)
                    for name, count in block["all_unit_spike_counts"].items()
                    if not name.endswith(OTHER_UNITS_LABEL)
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:12],
            "other_units_spike_count": int(
                block["all_unit_spike_counts"].get(
                    f"{probe['label']}:{OTHER_UNITS_LABEL}",
                    0,
                )
            ),
        }
        for probe, block in zip(probes, stimulus_unit_blocks)
    ]
    return {
        "openalyx_url": OPENALYX_URL,
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "wheel_sample_count": int(len(wheel_timestamps)),
        "probe_summaries": probe_summaries,
        "unit_summaries": unit_summaries,
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
        ridge=args.ridge,
        permutations=args.permutations,
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
        comparison = target["nested_comparison"]
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
                "task_history_plus_all_unit_balanced_accuracy": comparison[
                    "task_history_plus_all_unit_balanced_accuracy"
                ],
                "task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy"
                ],
                "region_increment_after_task": comparison[
                    "region_increment_after_task"
                ],
                "unit_increment_after_task": comparison["unit_increment_after_task"],
                "unit_increment_after_task_region": comparison[
                    "unit_increment_after_task_region"
                ],
                "region_increment_after_task_unit": comparison[
                    "region_increment_after_task_unit"
                ],
                "unit_beats_region_after_task": comparison[
                    "unit_beats_region_after_task"
                ],
                "unit_residual_after_task_region": comparison[
                    "unit_residual_after_task_region"
                ],
                "region_residual_after_task_unit": comparison[
                    "region_residual_after_task_unit"
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
        "target_rows": rows,
        "unit_residual_after_task_region_count": sum(
            row["unit_residual_after_task_region"] for row in rows
        ),
        "region_residual_after_task_unit_count": sum(
            row["region_residual_after_task_unit"] for row in rows
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
            "unit_residual_after_task_region_count": sum(
                row["unit_residual_after_task_region"] for row in rows
            ),
            "region_residual_after_task_unit_count": sum(
                row["region_residual_after_task_unit"] for row in rows
            ),
            "unit_beats_region_after_task_count": sum(
                row["unit_beats_region_after_task"] for row in rows
            ),
            "mean_task_history_balanced_accuracy": sum(
                row["task_history_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_region_balanced_accuracy": sum(
                row["task_history_plus_hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_all_unit_balanced_accuracy": sum(
                row["task_history_plus_all_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy": sum(
                row["task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy"]
                for row in rows
            )
            / len(rows),
            "mean_region_increment_after_task": sum(
                row["region_increment_after_task"] for row in rows
            )
            / len(rows),
            "mean_unit_increment_after_task": sum(
                row["unit_increment_after_task"] for row in rows
            )
            / len(rows),
            "mean_unit_increment_after_task_region": sum(
                row["unit_increment_after_task_region"] for row in rows
            )
            / len(rows),
            "mean_region_increment_after_task_unit": sum(
                row["region_increment_after_task_unit"] for row in rows
            )
            / len(rows),
        }

    def replicated_positive(target: str, count_key: str, mean_key: str) -> bool:
        row = target_replication.get(target, {})
        return bool(row.get(count_key, 0) >= 3 and row.get(mean_key, 0.0) > 0.0)

    unit_residual_supported = any(
        replicated_positive(
            target,
            "unit_residual_after_task_region_count",
            "mean_unit_increment_after_task_region",
        )
        for target in target_replication
    )
    region_residual_supported = any(
        replicated_positive(
            target,
            "region_residual_after_task_unit_count",
            "mean_region_increment_after_task_unit",
        )
        for target in target_replication
    )
    return {
        "candidate_count": len(candidates),
        "folds": args.folds,
        "ridge": args.ridge,
        "permutations": args.permutations,
        "seed": args.seed,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "min_delta": args.min_delta,
        "max_units_per_probe": args.max_units_per_probe,
        "unit_residual_after_task_region_supported": bool(unit_residual_supported),
        "region_residual_after_task_unit_supported": bool(region_residual_supported),
        "all_unit_nested_regularization_passed": bool(unit_residual_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx all-unit nested-regularization gate",
        "",
        "Flat-unit gate 다음 반례는 단순 unit-vs-region 비교가 아니라 nested residual이다.",
        "이 gate는 task/history와 channel-rescued region을 먼저 넣은 뒤, high-spike unit identity가 cross-validated BA를 추가로 올리는지 검사한다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- min unit spikes: {output['min_unit_spikes']}",
        f"- max units per probe: {output['max_units_per_probe']}",
        f"- unit residual after task+region supported: `{output['unit_residual_after_task_region_supported']}`",
        f"- region residual after task+unit supported: `{output['region_residual_after_task_unit_supported']}`",
        f"- all-unit nested regularization passed: `{output['all_unit_nested_regularization_passed']}`",
        "",
        "## nested equations",
        "",
        "$$",
        "M_X:y_i\\sim X_i,\\quad",
        "M_{XR}:y_i\\sim[X_i,R_i],\\quad",
        "M_{XU}:y_i\\sim[X_i,U_i],\\quad",
        "M_{XRU}:y_i\\sim[X_i,R_i,U_i].",
        "$$",
        "",
        "The main residuals are",
        "",
        "$$",
        "\\Delta_{U\\mid X,R}=\\mathrm{BA}(M_{XRU})-\\mathrm{BA}(M_{XR}),",
        "\\qquad",
        "\\Delta_{R\\mid X,U}=\\mathrm{BA}(M_{XRU})-\\mathrm{BA}(M_{XU}).",
        "$$",
        "",
        "`U_i`는 probe별 high-spike unit identity다. `max_units_per_probe<=0`이면 threshold를 넘는 모든 unit을 쓰고, 양수이면 computational guard로 상위 unit만 남긴다.",
        "",
        "## target replication",
        "",
        "| target | candidates | unit residual count | region residual count | unit>region after task | mean task BA | mean task+region BA | mean task+unit BA | mean task+region+unit BA | mean unit residual | mean region residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["unit_residual_after_task_region_count"]),
                    str(row["region_residual_after_task_unit_count"]),
                    str(row["unit_beats_region_after_task_count"]),
                    f"{row['mean_task_history_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_all_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_unit_increment_after_task_region']:.6f}",
                    f"{row['mean_region_increment_after_task_unit']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | unit residuals | region residuals | choice U_given_XR | speed U_given_XR | wheel U_given_XR | choice R_given_XU | speed R_given_XU | wheel R_given_XU |",
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
                    str(summary["unit_residual_after_task_region_count"]),
                    str(summary["region_residual_after_task_unit_count"]),
                    f"{by_target['choice_sign']['unit_increment_after_task_region']:.6f}",
                    f"{by_target['first_movement_speed']['unit_increment_after_task_region']:.6f}",
                    f"{by_target['wheel_action_direction']['unit_increment_after_task_region']:.6f}",
                    f"{by_target['choice_sign']['region_increment_after_task_unit']:.6f}",
                    f"{by_target['first_movement_speed']['region_increment_after_task_unit']:.6f}",
                    f"{by_target['wheel_action_direction']['region_increment_after_task_unit']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## per-candidate details", ""])
    for summary in output["summaries"]:
        lines.extend(
            [
                f"### {summary['name']}",
                "",
                f"- eid: `{summary['eid']}`",
                f"- session: `{summary['session_ref']}`",
                f"- reason: {summary['reason']}",
                "",
                "| target | n | task BA | task+region BA | task+unit BA | task+region+unit BA | U_given_XR | R_given_XU |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["target_rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['target']}`",
                        str(row["n_trials"]),
                        f"{row['task_history_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_hybrid_region_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_all_unit_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_hybrid_region_plus_all_unit_balanced_accuracy']:.6f}",
                        f"{row['unit_increment_after_task_region']:.6f}",
                        f"{row['region_increment_after_task_unit']:.6f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- unit residual after task+region supported: `{output['unit_residual_after_task_region_supported']}`",
            f"- region residual after task+unit supported: `{output['region_residual_after_task_unit_supported']}`",
            f"- all-unit nested regularization passed: `{output['all_unit_nested_regularization_passed']}`",
            "",
            "해석:",
            "",
            "- \\(\\Delta_{U\\mid X,R}>0\\)가 반복되면 mouse 단계 방정식에 explicit unit-detail residual을 남겨야 한다.",
            "- \\(\\Delta_{R\\mid X,U}>0\\)가 반복되면 anatomical compression도 unit decoder 위에서 독립 항으로 남는다.",
            "- 이 gate는 coupling이 아니라 nested decoding이다. 다음 강한 버전은 unit-to-unit GLM coupling 또는 trial-split lag selection이다.",
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
    parser.add_argument("--permutations", type=int, default=80)
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

    print("Mouse IBL/OpenAlyx all-unit nested-regularization gate")
    print(f"  candidates={output['candidate_count']}")
    print(f"  min_unit_spikes={output['min_unit_spikes']}")
    print(f"  max_units_per_probe={output['max_units_per_probe']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} U|XR="
            + f"{row['unit_residual_after_task_region_count']}/"
            + f"{row['candidate_count']} R|XU="
            + f"{row['region_residual_after_task_unit_count']}/"
            + f"{row['candidate_count']} mean_U|XR="
            + f"{row['mean_unit_increment_after_task_region']:.6f}"
        )
    print(
        "  all_unit_nested_regularization_passed="
        + str(output["all_unit_nested_regularization_passed"])
    )
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
