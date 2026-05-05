"""Mouse IBL/OpenAlyx flat-unit versus hybrid-region comparison gate.

The task-baseline gate showed that rescued hybrid region features are not just
trial timing, while choice remains strongly task-covariate dominated.  The next
counterexample is representational: perhaps anatomical region/probe bins add no
useful abstraction because a flat readout over individual units is enough.

This gate compares channel-rescued hybrid region bins with a probe-indexed
top-unit feature set.  The unit baseline is intentionally bounded: for each
probe it keeps the highest-spike clusters and aggregates the remaining clusters
into an ``other_units`` bin, which makes the panel feasible while still testing
whether fine unit identity dominates region identity.
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
    "mouse_ibl_flat_unit_region_comparison_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_flat_unit_region_comparison_report.md"
)
OTHER_UNITS_LABEL = "other_units"


def top_unit_group(
    spike_clusters: np.ndarray,
    max_units: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    max_cluster_id = int(np.max(spike_clusters))
    counts = np.bincount(spike_clusters, minlength=max_cluster_id + 1)
    positive = np.where(counts > 0)[0]
    ranked = sorted(
        ((int(cluster_id), int(counts[cluster_id])) for cluster_id in positive),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ranked[:max_units]
    group_names = [f"cluster:{cluster_id}" for cluster_id, _ in selected]
    group_names.append(OTHER_UNITS_LABEL)
    other_idx = len(group_names) - 1
    cluster_group = np.full(max_cluster_id + 1, other_idx, dtype=np.int16)
    for group_idx, (cluster_id, _) in enumerate(selected):
        cluster_group[cluster_id] = group_idx
    spike_counts = {f"cluster:{cluster_id}": count for cluster_id, count in ranked}
    spike_counts[OTHER_UNITS_LABEL] = int(
        sum(count for _, count in ranked[max_units:])
    )
    return group_names, cluster_group, spike_counts


def unit_feature_block(
    probe: dict[str, object],
    starts: np.ndarray,
    ends: np.ndarray,
    valid_window: np.ndarray,
    max_units_per_probe: int,
) -> dict[str, object]:
    spike_times = probe["spike_times"]
    spike_clusters = probe["spike_clusters"]
    unit_names, unit_group, unit_counts = top_unit_group(
        spike_clusters,
        max_units_per_probe,
    )
    return {
        "unit_features": window_features(
            spike_times,
            spike_clusters,
            unit_group,
            starts,
            ends,
            valid_window,
            len(unit_names),
        ),
        "unit_feature_names": [f"{probe['label']}:{name}" for name in unit_names],
        "unit_spike_counts": {
            f"{probe['label']}:{name}": int(count)
            for name, count in unit_counts.items()
        },
        "selected_unit_count": int(len(unit_names) - 1),
        "feature_count": int(len(unit_names)),
    }


def make_unit_models(unit_blocks: list[dict[str, object]]) -> dict[str, np.ndarray]:
    unit = hstack([block["unit_features"] for block in unit_blocks])
    return {
        "top_unit_by_probe": unit,
        "flat_unit_global_rate": np.nansum(unit, axis=1, keepdims=True),
    }


def build_models(
    trials,
    region_models: dict[str, np.ndarray],
    unit_models: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    task_history, task_history_names = task_history_covariates(trials)
    hybrid = region_models["hybrid_acronym_channel_id_by_probe"]
    unit = unit_models["top_unit_by_probe"]
    global_rate = region_models["global_rate"]
    return (
        {
            "hybrid_region_by_probe": hybrid,
            "top_unit_by_probe": unit,
            "task_history_plus_hybrid_region": hstack([task_history, hybrid]),
            "task_history_plus_top_unit": hstack([task_history, unit]),
            "global_rate": global_rate,
        },
        {
            "task_history_feature_names": task_history_names,
            "task_history_feature_count": int(task_history.shape[1]),
            "hybrid_region_feature_count": int(hybrid.shape[1]),
            "top_unit_feature_count": int(unit.shape[1]),
            "task_history_plus_hybrid_feature_count": int(task_history.shape[1] + hybrid.shape[1]),
            "task_history_plus_top_unit_feature_count": int(task_history.shape[1] + unit.shape[1]),
        },
    )


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    hybrid = by_model["hybrid_region_by_probe"]["balanced_accuracy"]
    unit = by_model["top_unit_by_probe"]["balanced_accuracy"]
    task_hybrid = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    task_unit = by_model["task_history_plus_top_unit"]["balanced_accuracy"]
    global_rate = by_model["global_rate"]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_hybrid_region"] = float(row["balanced_accuracy"] - hybrid)
        row["delta_vs_top_unit"] = float(row["balanced_accuracy"] - unit)
    target["flat_unit_comparison"] = {
        "hybrid_region_balanced_accuracy": hybrid,
        "top_unit_balanced_accuracy": unit,
        "task_history_plus_hybrid_balanced_accuracy": task_hybrid,
        "task_history_plus_top_unit_balanced_accuracy": task_unit,
        "global_rate_balanced_accuracy": global_rate,
        "top_unit_delta_vs_hybrid_region": float(unit - hybrid),
        "task_top_unit_delta_vs_task_hybrid": float(task_unit - task_hybrid),
        "hybrid_region_beats_top_unit": bool(hybrid > unit + min_delta),
        "top_unit_beats_hybrid_region": bool(unit > hybrid + min_delta),
        "task_hybrid_beats_task_top_unit": bool(task_hybrid > task_unit + min_delta),
        "task_top_unit_beats_task_hybrid": bool(task_unit > task_hybrid + min_delta),
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
        unit_feature_block(
            probe,
            stim_start,
            stim_end,
            stim_valid,
            args.max_units_per_probe,
        )
        for probe in probes
    ]
    movement_unit_blocks = [
        unit_feature_block(
            probe,
            move_start,
            move_end,
            move_valid,
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
        make_unit_models(stimulus_unit_blocks),
    )
    movement_models, movement_model_meta = build_models(
        trials,
        make_region_models(movement_region_blocks),
        make_unit_models(movement_unit_blocks),
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
                    for name, count in block["unit_spike_counts"].items()
                    if not name.endswith(OTHER_UNITS_LABEL)
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:12],
            "other_units_spike_count": int(
                block["unit_spike_counts"].get(f"{probe['label']}:{OTHER_UNITS_LABEL}", 0)
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
        min_abs_wheel_displacement=args.min_abs_wheel_displacement,
        min_delta=args.min_delta,
        max_units_per_probe=args.max_units_per_probe,
    )


def target_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        comparison = target["flat_unit_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["hybrid_region_by_probe"]["n_trials"],
                "class_counts": by_model["hybrid_region_by_probe"]["class_counts"],
                "hybrid_region_balanced_accuracy": comparison[
                    "hybrid_region_balanced_accuracy"
                ],
                "top_unit_balanced_accuracy": comparison[
                    "top_unit_balanced_accuracy"
                ],
                "task_history_plus_hybrid_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_balanced_accuracy"
                ],
                "task_history_plus_top_unit_balanced_accuracy": comparison[
                    "task_history_plus_top_unit_balanced_accuracy"
                ],
                "global_rate_balanced_accuracy": comparison[
                    "global_rate_balanced_accuracy"
                ],
                "top_unit_delta_vs_hybrid_region": comparison[
                    "top_unit_delta_vs_hybrid_region"
                ],
                "task_top_unit_delta_vs_task_hybrid": comparison[
                    "task_top_unit_delta_vs_task_hybrid"
                ],
                "hybrid_region_beats_top_unit": comparison[
                    "hybrid_region_beats_top_unit"
                ],
                "top_unit_beats_hybrid_region": comparison[
                    "top_unit_beats_hybrid_region"
                ],
                "task_hybrid_beats_task_top_unit": comparison[
                    "task_hybrid_beats_task_top_unit"
                ],
                "task_top_unit_beats_task_hybrid": comparison[
                    "task_top_unit_beats_task_hybrid"
                ],
                "hybrid_global_gate_passed": bool(
                    by_model["hybrid_region_by_probe"]["passed"]
                ),
                "top_unit_global_gate_passed": bool(
                    by_model["top_unit_by_probe"]["passed"]
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
        "hybrid_region_beats_top_unit_count": sum(
            row["hybrid_region_beats_top_unit"] for row in rows
        ),
        "top_unit_beats_hybrid_region_count": sum(
            row["top_unit_beats_hybrid_region"] for row in rows
        ),
        "task_hybrid_beats_task_top_unit_count": sum(
            row["task_hybrid_beats_task_top_unit"] for row in rows
        ),
        "task_top_unit_beats_task_hybrid_count": sum(
            row["task_top_unit_beats_task_hybrid"] for row in rows
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
            "hybrid_region_beats_top_unit_count": sum(
                row["hybrid_region_beats_top_unit"] for row in rows
            ),
            "top_unit_beats_hybrid_region_count": sum(
                row["top_unit_beats_hybrid_region"] for row in rows
            ),
            "task_hybrid_beats_task_top_unit_count": sum(
                row["task_hybrid_beats_task_top_unit"] for row in rows
            ),
            "task_top_unit_beats_task_hybrid_count": sum(
                row["task_top_unit_beats_task_hybrid"] for row in rows
            ),
            "hybrid_global_gate_count": sum(
                row["hybrid_global_gate_passed"] for row in rows
            ),
            "top_unit_global_gate_count": sum(
                row["top_unit_global_gate_passed"] for row in rows
            ),
            "mean_hybrid_region_balanced_accuracy": sum(
                row["hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_top_unit_balanced_accuracy": sum(
                row["top_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_balanced_accuracy": sum(
                row["task_history_plus_hybrid_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_top_unit_balanced_accuracy": sum(
                row["task_history_plus_top_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_top_unit_delta_vs_hybrid_region": sum(
                row["top_unit_delta_vs_hybrid_region"] for row in rows
            )
            / len(rows),
            "mean_task_top_unit_delta_vs_task_hybrid": sum(
                row["task_top_unit_delta_vs_task_hybrid"] for row in rows
            )
            / len(rows),
        }

    top_unit_dominates = (
        target_replication.get("choice_sign", {}).get(
            "top_unit_beats_hybrid_region_count",
            0,
        )
        >= 3
        and target_replication.get("wheel_action_direction", {}).get(
            "top_unit_beats_hybrid_region_count",
            0,
        )
        >= 3
    )
    region_survives = (
        target_replication.get("first_movement_speed", {}).get(
            "hybrid_region_beats_top_unit_count",
            0,
        )
        >= 3
        or target_replication.get("wheel_action_direction", {}).get(
            "hybrid_region_beats_top_unit_count",
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
        "max_units_per_probe": args.max_units_per_probe,
        "top_unit_dominates_region": bool(top_unit_dominates),
        "region_survives_flat_unit_comparison": bool(region_survives),
        "flat_unit_gate_passed": bool(region_survives and not top_unit_dominates),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx flat-unit versus hybrid-region comparison gate",
        "",
        "Task-baseline gate 다음 반례는 anatomical region/probe bin이 아니라 개별 unit identity만으로 충분한 경우다.",
        "이 gate는 probe별 spike count 상위 unit을 flat feature로 만들고, channel-rescued hybrid region feature와 같은 target에서 비교한다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- max units per probe: {output['max_units_per_probe']}",
        f"- top unit dominates region: `{output['top_unit_dominates_region']}`",
        f"- region survives flat-unit comparison: `{output['region_survives_flat_unit_comparison']}`",
        f"- flat-unit gate passed: `{output['flat_unit_gate_passed']}`",
        "",
        "## models",
        "",
        "| model | feature definition |",
        "|---|---|",
        "| `hybrid_region_by_probe` | channel-rescued anatomical acronym/CCF-id bins by probe |",
        "| `top_unit_by_probe` | highest-spike clusters by probe plus `other_units` bin |",
        "| `task_history_plus_hybrid_region` | task-history baseline plus hybrid region bins |",
        "| `task_history_plus_top_unit` | task-history baseline plus top-unit bins |",
        "| `global_rate` | one scalar total hybrid firing rate |",
        "",
        "The comparison is",
        "",
        "$$",
        "\\Delta_{\\mathrm{unit-region}}=\\mathrm{BA}(U^{\\mathrm{top}})-\\mathrm{BA}(R^{\\mathrm{hybrid}}),",
        "\\qquad",
        "\\Delta_{\\mathrm{task+unit}}=\\mathrm{BA}([X^{\\mathrm{task}},U^{\\mathrm{top}}])-\\mathrm{BA}([X^{\\mathrm{task}},R^{\\mathrm{hybrid}}]).",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | region beats unit | unit beats region | task+region beats task+unit | task+unit beats task+region | mean region BA | mean unit BA | mean task+region BA | mean task+unit BA | mean unit-region delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["hybrid_region_beats_top_unit_count"]),
                    str(row["top_unit_beats_hybrid_region_count"]),
                    str(row["task_hybrid_beats_task_top_unit_count"]),
                    str(row["task_top_unit_beats_task_hybrid_count"]),
                    f"{row['mean_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_top_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_hybrid_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_top_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_top_unit_delta_vs_hybrid_region']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | region>unit | unit>region | task+region>task+unit | task+unit>task+region | choice unit-region | speed unit-region | wheel unit-region |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    str(summary["hybrid_region_beats_top_unit_count"]),
                    str(summary["top_unit_beats_hybrid_region_count"]),
                    str(summary["task_hybrid_beats_task_top_unit_count"]),
                    str(summary["task_top_unit_beats_task_hybrid_count"]),
                    f"{by_target['choice_sign']['top_unit_delta_vs_hybrid_region']:.6f}",
                    f"{by_target['first_movement_speed']['top_unit_delta_vs_hybrid_region']:.6f}",
                    f"{by_target['wheel_action_direction']['top_unit_delta_vs_hybrid_region']:.6f}",
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
                "| target | n | region BA | unit BA | task+region BA | task+unit BA | unit-region | task unit-region |",
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
                        f"{row['hybrid_region_balanced_accuracy']:.6f}",
                        f"{row['top_unit_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_hybrid_balanced_accuracy']:.6f}",
                        f"{row['task_history_plus_top_unit_balanced_accuracy']:.6f}",
                        f"{row['top_unit_delta_vs_hybrid_region']:.6f}",
                        f"{row['task_top_unit_delta_vs_task_hybrid']:.6f}",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## verdict",
            "",
            f"- top unit dominates region: `{output['top_unit_dominates_region']}`",
            f"- region survives flat-unit comparison: `{output['region_survives_flat_unit_comparison']}`",
            f"- flat-unit gate passed: `{output['flat_unit_gate_passed']}`",
            "",
            "해석:",
            "",
            "- Top-unit readout이 hybrid region을 반복적으로 이기면, region/probe 항은 maximal decoder가 아니라 compressed anatomical readout으로 내려간다.",
            "- Hybrid region이 일부 target에서 top-unit과 같거나 더 좋으면, anatomical binning이 단순 정보 손실만은 아니라는 뜻이다.",
            "- 이 gate는 all-unit decoder가 아니라 top-unit bounded decoder다. 따라서 flat-neuron 반례의 강한 버전은 아직 더 큰 unit set 또는 nested regularization으로 남는다.",
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
    parser.add_argument("--max-units-per-probe", type=int, default=96)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx flat-unit versus hybrid-region comparison gate")
    print(f"  candidates={output['candidate_count']}")
    print(f"  max_units_per_probe={output['max_units_per_probe']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} region>unit="
            + f"{row['hybrid_region_beats_top_unit_count']}/"
            + f"{row['candidate_count']} unit>region="
            + f"{row['top_unit_beats_hybrid_region_count']}/"
            + f"{row['candidate_count']} mean_unit-region="
            + f"{row['mean_top_unit_delta_vs_hybrid_region']:.6f}"
        )
    print(f"  flat_unit_gate_passed={output['flat_unit_gate_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
