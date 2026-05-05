"""Mouse IBL/OpenAlyx lagged region-coupling proxy gate.

The same-window interaction proxy failed: multiplying region rates from the
same decoding window did not improve mean balanced accuracy over additive
hybrid region bins.  This script tests the next stronger but still tractable
proxy: source-window region rates multiplied by current target-window region
rates.

This is not a causal connectivity estimate.  It is a lagged coupling feature
that preserves temporal order:

    z_{iab}^{lag} = r_{ia}^{source} r_{ib}^{target}.

For choice and first-movement speed the source window is pre-stimulus
[-300, 0] ms.  For wheel direction the source window is pre-movement
[-400, -100] ms, while the target window remains movement-aligned
[-100, +200] ms.
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
from mouse_ibl_flat_unit_region_comparison_gate import make_unit_models, unit_feature_block
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    WindowSpec,
    choice_target,
    evaluate_target,
    first_movement_speed_target,
    wheel_action_direction_target,
    window_bounds,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates


RESULT_JSON = Path(__file__).with_name("mouse_ibl_lagged_coupling_proxy_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_lagged_coupling_proxy_report.md")

PRE_STIMULUS_WINDOW = WindowSpec(
    name="pre_stimulus_-300_0ms",
    start_column="stimOn_times",
    start_offset=-0.300,
    end_column="stimOn_times",
    end_offset=0.0,
    meaning="pre-stimulus lag source for decision/action preparation",
)

PRE_MOVEMENT_WINDOW = WindowSpec(
    name="pre_movement_-400_-100ms",
    start_column="firstMovement_times",
    start_offset=-0.400,
    end_column="firstMovement_times",
    end_offset=-0.100,
    meaning="pre-movement lag source ending before the movement decoding window",
)


def lagged_interactions(
    source: np.ndarray,
    target: np.ndarray,
    max_pairs: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, object]]:
    source_count = int(source.shape[1])
    target_count = int(target.shape[1])
    if source_count == 0 or target_count == 0:
        return np.empty((target.shape[0], 0), dtype=float), [], {
            "candidate_pair_count": 0,
            "selected_pair_count": 0,
        }

    pairs = [(source_idx, target_idx) for source_idx in range(source_count) for target_idx in range(target_count)]
    if len(pairs) > max_pairs:
        source_var = np.nanvar(source, axis=0)
        target_var = np.nanvar(target, axis=0)
        pairs = sorted(
            pairs,
            key=lambda item: source_var[item[0]] * target_var[item[1]],
            reverse=True,
        )[:max_pairs]

    interaction = np.empty((target.shape[0], len(pairs)), dtype=float)
    for idx, (source_idx, target_idx) in enumerate(pairs):
        interaction[:, idx] = source[:, source_idx] * target[:, target_idx]
    return interaction, pairs, {
        "candidate_pair_count": source_count * target_count,
        "selected_pair_count": len(pairs),
    }


def build_models(
    trials,
    target_region_models: dict[str, np.ndarray],
    source_region_models: dict[str, np.ndarray],
    unit_models: dict[str, np.ndarray],
    max_lagged_pairs: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    task_history, task_history_names = task_history_covariates(trials)
    target_region = target_region_models["hybrid_acronym_channel_id_by_probe"]
    source_region = source_region_models["hybrid_acronym_channel_id_by_probe"]
    unit = unit_models["top_unit_by_probe"]
    global_rate = target_region_models["global_rate"]
    lagged, pairs, pair_meta = lagged_interactions(
        source_region,
        target_region,
        max_lagged_pairs,
    )
    region_plus_lagged = hstack([target_region, lagged])
    return (
        {
            "hybrid_region_by_probe": target_region,
            "lag_source_region_by_probe": source_region,
            "lagged_region_coupling_proxy": lagged,
            "hybrid_region_plus_lagged_coupling": region_plus_lagged,
            "top_unit_by_probe": unit,
            "task_history_plus_hybrid_region": hstack([task_history, target_region]),
            "task_history_plus_lagged_coupling": hstack([task_history, region_plus_lagged]),
            "task_history_plus_top_unit": hstack([task_history, unit]),
            "global_rate": global_rate,
        },
        {
            "task_history_feature_names": task_history_names,
            "task_history_feature_count": int(task_history.shape[1]),
            "source_region_feature_count": int(source_region.shape[1]),
            "target_region_feature_count": int(target_region.shape[1]),
            "lagged_feature_count": int(lagged.shape[1]),
            "top_unit_feature_count": int(unit.shape[1]),
            "lagged_pairs": [(int(left), int(right)) for left, right in pairs],
            **pair_meta,
        },
    )


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    region = by_model["hybrid_region_by_probe"]["balanced_accuracy"]
    source = by_model["lag_source_region_by_probe"]["balanced_accuracy"]
    lagged_only = by_model["lagged_region_coupling_proxy"]["balanced_accuracy"]
    lagged = by_model["hybrid_region_plus_lagged_coupling"]["balanced_accuracy"]
    unit = by_model["top_unit_by_probe"]["balanced_accuracy"]
    task_region = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    task_lagged = by_model["task_history_plus_lagged_coupling"]["balanced_accuracy"]
    task_unit = by_model["task_history_plus_top_unit"]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_hybrid_region"] = float(row["balanced_accuracy"] - region)
        row["delta_vs_lagged_coupling"] = float(row["balanced_accuracy"] - lagged)
    target["lagged_comparison"] = {
        "source_region_balanced_accuracy": source,
        "hybrid_region_balanced_accuracy": region,
        "lagged_only_balanced_accuracy": lagged_only,
        "hybrid_region_plus_lagged_balanced_accuracy": lagged,
        "top_unit_balanced_accuracy": unit,
        "task_history_plus_hybrid_region_balanced_accuracy": task_region,
        "task_history_plus_lagged_balanced_accuracy": task_lagged,
        "task_history_plus_top_unit_balanced_accuracy": task_unit,
        "lagged_delta_vs_region": float(lagged - region),
        "lagged_delta_vs_top_unit": float(lagged - unit),
        "task_lagged_delta_vs_task_region": float(task_lagged - task_region),
        "task_lagged_delta_vs_task_unit": float(task_lagged - task_unit),
        "lagged_beats_region": bool(lagged > region + min_delta),
        "lagged_beats_top_unit": bool(lagged > unit + min_delta),
        "task_lagged_beats_task_region": bool(task_lagged > task_region + min_delta),
        "task_lagged_beats_task_unit": bool(task_lagged > task_unit + min_delta),
    }
    return target


def region_blocks_for_window(
    probes: list[dict[str, object]],
    trials,
    spec: WindowSpec,
    min_label_spikes: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    starts, ends, valid = window_bounds(trials, spec)
    blocks = [
        probe_feature_block(
            probe,
            starts,
            ends,
            valid,
            min_label_spikes,
        )
        for probe in probes
    ]
    return blocks, valid


def unit_blocks_for_window(
    probes: list[dict[str, object]],
    trials,
    spec: WindowSpec,
    max_units_per_probe: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    starts, ends, valid = window_bounds(trials, spec)
    blocks = [
        unit_feature_block(
            probe,
            starts,
            ends,
            valid,
            max_units_per_probe,
        )
        for probe in probes
    ]
    return blocks, valid


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    wheel_timestamps = np.asarray(common["wheel_timestamps"], dtype=float)
    wheel_position = np.asarray(common["wheel_position"], dtype=float)
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]

    pre_stim_region_blocks, pre_stim_valid = region_blocks_for_window(
        probes,
        trials,
        PRE_STIMULUS_WINDOW,
        args.min_label_spikes,
    )
    stim_region_blocks, stim_valid = region_blocks_for_window(
        probes,
        trials,
        STIMULUS_WINDOW,
        args.min_label_spikes,
    )
    pre_move_region_blocks, pre_move_valid = region_blocks_for_window(
        probes,
        trials,
        PRE_MOVEMENT_WINDOW,
        args.min_label_spikes,
    )
    move_region_blocks, move_valid = region_blocks_for_window(
        probes,
        trials,
        MOVEMENT_WINDOW,
        args.min_label_spikes,
    )
    stim_unit_blocks, _ = unit_blocks_for_window(
        probes,
        trials,
        STIMULUS_WINDOW,
        args.max_units_per_probe,
    )
    move_unit_blocks, _ = unit_blocks_for_window(
        probes,
        trials,
        MOVEMENT_WINDOW,
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

    stimulus_models, stimulus_meta = build_models(
        trials,
        make_region_models(stim_region_blocks),
        make_region_models(pre_stim_region_blocks),
        make_unit_models(stim_unit_blocks),
        args.max_lagged_pairs,
    )
    movement_models, movement_meta = build_models(
        trials,
        make_region_models(move_region_blocks),
        make_region_models(pre_move_region_blocks),
        make_unit_models(move_unit_blocks),
        args.max_lagged_pairs,
    )

    targets = [
        annotate_target(
            evaluate_target(
                target_name="choice_sign",
                window_name=f"{PRE_STIMULUS_WINDOW.name}->{STIMULUS_WINDOW.name}",
                x_models=stimulus_models,
                y_all=choice,
                valid=choice_valid & stim_valid & pre_stim_valid,
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
                window_name=f"{PRE_STIMULUS_WINDOW.name}->{STIMULUS_WINDOW.name}",
                x_models=stimulus_models,
                y_all=speed,
                valid=speed_valid & stim_valid & pre_stim_valid,
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
                window_name=f"{PRE_MOVEMENT_WINDOW.name}->{MOVEMENT_WINDOW.name}",
                x_models=movement_models,
                y_all=wheel_direction,
                valid=wheel_valid & move_valid & pre_move_valid,
                folds=args.folds,
                ridge=args.ridge,
                permutations=args.permutations,
                seed=args.seed,
            ),
            args.min_delta,
        ),
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
            for probe, stimulus_block in zip(probes, stim_region_blocks)
        ],
        "stimulus_model_meta": stimulus_meta,
        "movement_model_meta": movement_meta,
        "target_metadata": {
            "choice_sign": choice_meta,
            "first_movement_speed": speed_meta,
            "wheel_action_direction": wheel_meta,
        },
        "source_windows": {
            "choice_sign": PRE_STIMULUS_WINDOW.name,
            "first_movement_speed": PRE_STIMULUS_WINDOW.name,
            "wheel_action_direction": PRE_MOVEMENT_WINDOW.name,
        },
        "target_windows": {
            "choice_sign": STIMULUS_WINDOW.name,
            "first_movement_speed": STIMULUS_WINDOW.name,
            "wheel_action_direction": MOVEMENT_WINDOW.name,
        },
        "folds": int(args.folds),
        "ridge": float(args.ridge),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "min_delta": float(args.min_delta),
        "max_units_per_probe": int(args.max_units_per_probe),
        "max_lagged_pairs": int(args.max_lagged_pairs),
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
        max_lagged_pairs=args.max_lagged_pairs,
    )


def target_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        comparison = target["lagged_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["hybrid_region_by_probe"]["n_trials"],
                "class_counts": by_model["hybrid_region_by_probe"]["class_counts"],
                "source_region_balanced_accuracy": comparison[
                    "source_region_balanced_accuracy"
                ],
                "hybrid_region_balanced_accuracy": comparison[
                    "hybrid_region_balanced_accuracy"
                ],
                "lagged_only_balanced_accuracy": comparison[
                    "lagged_only_balanced_accuracy"
                ],
                "hybrid_region_plus_lagged_balanced_accuracy": comparison[
                    "hybrid_region_plus_lagged_balanced_accuracy"
                ],
                "top_unit_balanced_accuracy": comparison["top_unit_balanced_accuracy"],
                "task_history_plus_hybrid_region_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_region_balanced_accuracy"
                ],
                "task_history_plus_lagged_balanced_accuracy": comparison[
                    "task_history_plus_lagged_balanced_accuracy"
                ],
                "task_history_plus_top_unit_balanced_accuracy": comparison[
                    "task_history_plus_top_unit_balanced_accuracy"
                ],
                "lagged_delta_vs_region": comparison["lagged_delta_vs_region"],
                "lagged_delta_vs_top_unit": comparison["lagged_delta_vs_top_unit"],
                "task_lagged_delta_vs_task_region": comparison[
                    "task_lagged_delta_vs_task_region"
                ],
                "task_lagged_delta_vs_task_unit": comparison[
                    "task_lagged_delta_vs_task_unit"
                ],
                "lagged_beats_region": comparison["lagged_beats_region"],
                "lagged_beats_top_unit": comparison["lagged_beats_top_unit"],
                "task_lagged_beats_task_region": comparison[
                    "task_lagged_beats_task_region"
                ],
                "task_lagged_beats_task_unit": comparison[
                    "task_lagged_beats_task_unit"
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
        "lagged_beats_region_count": sum(row["lagged_beats_region"] for row in rows),
        "lagged_beats_top_unit_count": sum(row["lagged_beats_top_unit"] for row in rows),
        "task_lagged_beats_task_region_count": sum(
            row["task_lagged_beats_task_region"] for row in rows
        ),
        "task_lagged_beats_task_unit_count": sum(
            row["task_lagged_beats_task_unit"] for row in rows
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
            "lagged_beats_region_count": sum(row["lagged_beats_region"] for row in rows),
            "lagged_beats_top_unit_count": sum(row["lagged_beats_top_unit"] for row in rows),
            "task_lagged_beats_task_region_count": sum(
                row["task_lagged_beats_task_region"] for row in rows
            ),
            "task_lagged_beats_task_unit_count": sum(
                row["task_lagged_beats_task_unit"] for row in rows
            ),
            "mean_source_region_balanced_accuracy": sum(
                row["source_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_hybrid_region_balanced_accuracy": sum(
                row["hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_lagged_balanced_accuracy": sum(
                row["hybrid_region_plus_lagged_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_top_unit_balanced_accuracy": sum(
                row["top_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_hybrid_region_balanced_accuracy": sum(
                row["task_history_plus_hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_lagged_balanced_accuracy": sum(
                row["task_history_plus_lagged_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_top_unit_balanced_accuracy": sum(
                row["task_history_plus_top_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_lagged_delta_vs_region": sum(
                row["lagged_delta_vs_region"] for row in rows
            )
            / len(rows),
            "mean_lagged_delta_vs_top_unit": sum(
                row["lagged_delta_vs_top_unit"] for row in rows
            )
            / len(rows),
            "mean_task_lagged_delta_vs_task_region": sum(
                row["task_lagged_delta_vs_task_region"] for row in rows
            )
            / len(rows),
        }

    def replicated_positive(target: str, count_key: str, mean_key: str) -> bool:
        row = target_replication.get(target, {})
        return bool(row.get(count_key, 0) >= 3 and row.get(mean_key, 0.0) > 0.0)

    lagged_supported = (
        replicated_positive(
            "first_movement_speed",
            "lagged_beats_region_count",
            "mean_lagged_delta_vs_region",
        )
        or replicated_positive(
            "wheel_action_direction",
            "lagged_beats_region_count",
            "mean_lagged_delta_vs_region",
        )
    )
    lagged_beats_unit = (
        replicated_positive(
            "choice_sign",
            "lagged_beats_top_unit_count",
            "mean_lagged_delta_vs_top_unit",
        )
        or replicated_positive(
            "wheel_action_direction",
            "lagged_beats_top_unit_count",
            "mean_lagged_delta_vs_top_unit",
        )
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
        "max_lagged_pairs": args.max_lagged_pairs,
        "lagged_supported_over_region": bool(lagged_supported),
        "lagged_beats_top_unit": bool(lagged_beats_unit),
        "lagged_coupling_proxy_passed": bool(lagged_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx lagged region-coupling proxy gate",
        "",
        "Same-window interaction이 실패했기 때문에 source window와 target window를 분리한 lagged coupling proxy를 검사했다.",
        "이 feature는 causal connectivity가 아니라 temporal ordering을 보존한 region-rate product다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- max units per probe: {output['max_units_per_probe']}",
        f"- max lagged pairs: {output['max_lagged_pairs']}",
        f"- lagged supported over region: `{output['lagged_supported_over_region']}`",
        f"- lagged beats top unit: `{output['lagged_beats_top_unit']}`",
        f"- lagged coupling proxy passed: `{output['lagged_coupling_proxy_passed']}`",
        "",
        "## model equation",
        "",
        "$$",
        "z_{iab}^{\\mathrm{lag}}=r_{ia}^{\\mathrm{source}}r_{ib}^{\\mathrm{target}}.",
        "$$",
        "",
        "The tested model is",
        "",
        "$$",
        "R_i^{\\mathrm{lag}}=[R_i^{\\mathrm{target}},Z_i^{\\mathrm{lag}}].",
        "$$",
        "",
        "Source windows are pre-stimulus for choice/speed and pre-movement for wheel direction.",
        "",
        "## target replication",
        "",
        "| target | candidates | lagged beats region | lagged beats unit | task+lagged beats task+region | task+lagged beats task+unit | mean source BA | mean region BA | mean lagged BA | mean unit BA | mean lagged-region delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["lagged_beats_region_count"]),
                    str(row["lagged_beats_top_unit_count"]),
                    str(row["task_lagged_beats_task_region_count"]),
                    str(row["task_lagged_beats_task_unit_count"]),
                    f"{row['mean_source_region_balanced_accuracy']:.6f}",
                    f"{row['mean_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_lagged_balanced_accuracy']:.6f}",
                    f"{row['mean_top_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_lagged_delta_vs_region']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | lagged>region | lagged>unit | task+lagged>task+region | task+lagged>task+unit | choice lag-region | speed lag-region | wheel lag-region |",
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
                    str(summary["lagged_beats_region_count"]),
                    str(summary["lagged_beats_top_unit_count"]),
                    str(summary["task_lagged_beats_task_region_count"]),
                    str(summary["task_lagged_beats_task_unit_count"]),
                    f"{by_target['choice_sign']['lagged_delta_vs_region']:.6f}",
                    f"{by_target['first_movement_speed']['lagged_delta_vs_region']:.6f}",
                    f"{by_target['wheel_action_direction']['lagged_delta_vs_region']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## verdict",
            "",
            f"- lagged supported over region: `{output['lagged_supported_over_region']}`",
            f"- lagged beats top unit: `{output['lagged_beats_top_unit']}`",
            f"- lagged coupling proxy passed: `{output['lagged_coupling_proxy_passed']}`",
            "",
            "해석:",
            "",
            "- Lagged coupling이 additive region보다 반복적으로 높으면 same-window product 실패가 시간 방향성 문제였다는 뜻이다.",
            "- Lagged coupling이 top-unit을 이기지 못하면 unit-detail residual은 계속 남는다.",
            "- 이 gate도 causal connectivity가 아니다. 더 강한 버전은 trial-split lag selection, all-unit nested regularization, 혹은 GLM coupling이다.",
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
    parser.add_argument("--max-lagged-pairs", type=int, default=750)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx lagged region-coupling proxy gate")
    print(f"  candidates={output['candidate_count']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} lagged>region="
            + f"{row['lagged_beats_region_count']}/"
            + f"{row['candidate_count']} lagged>unit="
            + f"{row['lagged_beats_top_unit_count']}/"
            + f"{row['candidate_count']} mean_lag-region="
            + f"{row['mean_lagged_delta_vs_region']:.6f}"
        )
    print(f"  lagged_coupling_proxy_passed={output['lagged_coupling_proxy_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
