"""Mouse IBL/OpenAlyx region-interaction effective-connectivity proxy gate.

The flat-unit gate showed that top-unit identity outperforms anatomical region
compression for choice and wheel direction.  This script asks the next, more
structured question: does a weighted interaction term between rescued
probe-region rates recover useful signal beyond additive region bins?

The interaction feature is not a causal connectivity estimate.  It is a
trial-window proxy for effective coupling:

    z_{iab} = r_{ia} r_{ib},  a < b

where r is the channel-rescued hybrid region rate.  If this term improves
action decoding, the mammalian equation should keep a weighted interaction
operator instead of stopping at additive region identity.
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
    choice_target,
    evaluate_target,
    first_movement_speed_target,
    wheel_action_direction_target,
    window_bounds,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates


RESULT_JSON = Path(__file__).with_name(
    "mouse_ibl_effective_connectivity_proxy_results.json"
)
REPORT_MD = Path(__file__).with_name(
    "mouse_ibl_effective_connectivity_proxy_report.md"
)


def pairwise_interactions(
    x: np.ndarray,
    max_pairs: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, object]]:
    n_features = int(x.shape[1])
    if n_features < 2:
        return np.empty((x.shape[0], 0), dtype=float), [], {
            "candidate_pair_count": 0,
            "selected_pair_count": 0,
        }

    pairs = [(left, right) for left in range(n_features) for right in range(left + 1, n_features)]
    if len(pairs) > max_pairs:
        variances = np.nanvar(x, axis=0)
        ranked_pairs = sorted(
            pairs,
            key=lambda item: variances[item[0]] * variances[item[1]],
            reverse=True,
        )
        pairs = ranked_pairs[:max_pairs]
    interaction = np.empty((x.shape[0], len(pairs)), dtype=float)
    for idx, (left, right) in enumerate(pairs):
        interaction[:, idx] = x[:, left] * x[:, right]
    return interaction, pairs, {
        "candidate_pair_count": n_features * (n_features - 1) // 2,
        "selected_pair_count": len(pairs),
    }


def build_models(
    trials,
    region_models: dict[str, np.ndarray],
    unit_models: dict[str, np.ndarray],
    max_interaction_pairs: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    task_history, task_history_names = task_history_covariates(trials)
    hybrid = region_models["hybrid_acronym_channel_id_by_probe"]
    global_rate = region_models["global_rate"]
    unit = unit_models["top_unit_by_probe"]
    interaction, pairs, pair_meta = pairwise_interactions(hybrid, max_interaction_pairs)
    hybrid_plus_interaction = hstack([hybrid, interaction])
    return (
        {
            "hybrid_region_by_probe": hybrid,
            "region_interaction_proxy": interaction,
            "hybrid_region_plus_interaction": hybrid_plus_interaction,
            "top_unit_by_probe": unit,
            "task_history_plus_hybrid_region": hstack([task_history, hybrid]),
            "task_history_plus_region_interaction": hstack(
                [task_history, hybrid_plus_interaction]
            ),
            "task_history_plus_top_unit": hstack([task_history, unit]),
            "global_rate": global_rate,
        },
        {
            "task_history_feature_names": task_history_names,
            "task_history_feature_count": int(task_history.shape[1]),
            "hybrid_region_feature_count": int(hybrid.shape[1]),
            "interaction_feature_count": int(interaction.shape[1]),
            "top_unit_feature_count": int(unit.shape[1]),
            "interaction_pairs": [(int(left), int(right)) for left, right in pairs],
            **pair_meta,
        },
    )


def annotate_target(target: dict[str, object], min_delta: float) -> dict[str, object]:
    by_model = {row["model"]: row for row in target["rows"]}
    region = by_model["hybrid_region_by_probe"]["balanced_accuracy"]
    interaction = by_model["region_interaction_proxy"]["balanced_accuracy"]
    region_interaction = by_model["hybrid_region_plus_interaction"]["balanced_accuracy"]
    unit = by_model["top_unit_by_probe"]["balanced_accuracy"]
    task_region = by_model["task_history_plus_hybrid_region"]["balanced_accuracy"]
    task_interaction = by_model["task_history_plus_region_interaction"]["balanced_accuracy"]
    task_unit = by_model["task_history_plus_top_unit"]["balanced_accuracy"]
    for row in target["rows"]:
        row["delta_vs_hybrid_region"] = float(row["balanced_accuracy"] - region)
        row["delta_vs_region_interaction"] = float(
            row["balanced_accuracy"] - region_interaction
        )
    target["interaction_comparison"] = {
        "hybrid_region_balanced_accuracy": region,
        "interaction_only_balanced_accuracy": interaction,
        "hybrid_region_plus_interaction_balanced_accuracy": region_interaction,
        "top_unit_balanced_accuracy": unit,
        "task_history_plus_hybrid_region_balanced_accuracy": task_region,
        "task_history_plus_region_interaction_balanced_accuracy": task_interaction,
        "task_history_plus_top_unit_balanced_accuracy": task_unit,
        "interaction_delta_vs_region": float(region_interaction - region),
        "interaction_delta_vs_top_unit": float(region_interaction - unit),
        "task_interaction_delta_vs_task_region": float(task_interaction - task_region),
        "task_interaction_delta_vs_task_unit": float(task_interaction - task_unit),
        "interaction_beats_region": bool(region_interaction > region + min_delta),
        "interaction_beats_top_unit": bool(region_interaction > unit + min_delta),
        "task_interaction_beats_task_region": bool(
            task_interaction > task_region + min_delta
        ),
        "task_interaction_beats_task_unit": bool(
            task_interaction > task_unit + min_delta
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
        args.max_interaction_pairs,
    )
    movement_models, movement_model_meta = build_models(
        trials,
        make_region_models(movement_region_blocks),
        make_unit_models(movement_unit_blocks),
        args.max_interaction_pairs,
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
        "max_interaction_pairs": int(args.max_interaction_pairs),
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
        max_interaction_pairs=args.max_interaction_pairs,
    )


def target_summary_rows(result: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for target in result["targets"]:
        by_model = {row["model"]: row for row in target["rows"]}
        comparison = target["interaction_comparison"]
        rows.append(
            {
                "target": target["target"],
                "window": target["window"],
                "n_trials": by_model["hybrid_region_by_probe"]["n_trials"],
                "class_counts": by_model["hybrid_region_by_probe"]["class_counts"],
                "hybrid_region_balanced_accuracy": comparison[
                    "hybrid_region_balanced_accuracy"
                ],
                "interaction_only_balanced_accuracy": comparison[
                    "interaction_only_balanced_accuracy"
                ],
                "hybrid_region_plus_interaction_balanced_accuracy": comparison[
                    "hybrid_region_plus_interaction_balanced_accuracy"
                ],
                "top_unit_balanced_accuracy": comparison[
                    "top_unit_balanced_accuracy"
                ],
                "task_history_plus_hybrid_region_balanced_accuracy": comparison[
                    "task_history_plus_hybrid_region_balanced_accuracy"
                ],
                "task_history_plus_region_interaction_balanced_accuracy": comparison[
                    "task_history_plus_region_interaction_balanced_accuracy"
                ],
                "task_history_plus_top_unit_balanced_accuracy": comparison[
                    "task_history_plus_top_unit_balanced_accuracy"
                ],
                "interaction_delta_vs_region": comparison[
                    "interaction_delta_vs_region"
                ],
                "interaction_delta_vs_top_unit": comparison[
                    "interaction_delta_vs_top_unit"
                ],
                "task_interaction_delta_vs_task_region": comparison[
                    "task_interaction_delta_vs_task_region"
                ],
                "task_interaction_delta_vs_task_unit": comparison[
                    "task_interaction_delta_vs_task_unit"
                ],
                "interaction_beats_region": comparison["interaction_beats_region"],
                "interaction_beats_top_unit": comparison["interaction_beats_top_unit"],
                "task_interaction_beats_task_region": comparison[
                    "task_interaction_beats_task_region"
                ],
                "task_interaction_beats_task_unit": comparison[
                    "task_interaction_beats_task_unit"
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
        "interaction_beats_region_count": sum(row["interaction_beats_region"] for row in rows),
        "interaction_beats_top_unit_count": sum(row["interaction_beats_top_unit"] for row in rows),
        "task_interaction_beats_task_region_count": sum(
            row["task_interaction_beats_task_region"] for row in rows
        ),
        "task_interaction_beats_task_unit_count": sum(
            row["task_interaction_beats_task_unit"] for row in rows
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
            "interaction_beats_region_count": sum(row["interaction_beats_region"] for row in rows),
            "interaction_beats_top_unit_count": sum(row["interaction_beats_top_unit"] for row in rows),
            "task_interaction_beats_task_region_count": sum(
                row["task_interaction_beats_task_region"] for row in rows
            ),
            "task_interaction_beats_task_unit_count": sum(
                row["task_interaction_beats_task_unit"] for row in rows
            ),
            "mean_hybrid_region_balanced_accuracy": sum(
                row["hybrid_region_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_region_interaction_balanced_accuracy": sum(
                row["hybrid_region_plus_interaction_balanced_accuracy"] for row in rows
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
            "mean_task_history_plus_region_interaction_balanced_accuracy": sum(
                row["task_history_plus_region_interaction_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_task_history_plus_top_unit_balanced_accuracy": sum(
                row["task_history_plus_top_unit_balanced_accuracy"] for row in rows
            )
            / len(rows),
            "mean_interaction_delta_vs_region": sum(
                row["interaction_delta_vs_region"] for row in rows
            )
            / len(rows),
            "mean_interaction_delta_vs_top_unit": sum(
                row["interaction_delta_vs_top_unit"] for row in rows
            )
            / len(rows),
            "mean_task_interaction_delta_vs_task_region": sum(
                row["task_interaction_delta_vs_task_region"] for row in rows
            )
            / len(rows),
        }

    def replicated_positive(target: str, count_key: str, mean_key: str) -> bool:
        row = target_replication.get(target, {})
        return bool(row.get(count_key, 0) >= 3 and row.get(mean_key, 0.0) > 0.0)

    interaction_supported = (
        replicated_positive(
            "first_movement_speed",
            "interaction_beats_region_count",
            "mean_interaction_delta_vs_region",
        )
        or replicated_positive(
            "wheel_action_direction",
            "interaction_beats_region_count",
            "mean_interaction_delta_vs_region",
        )
    )
    interaction_beats_unit = (
        replicated_positive(
            "choice_sign",
            "interaction_beats_top_unit_count",
            "mean_interaction_delta_vs_top_unit",
        )
        or replicated_positive(
            "wheel_action_direction",
            "interaction_beats_top_unit_count",
            "mean_interaction_delta_vs_top_unit",
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
        "max_interaction_pairs": args.max_interaction_pairs,
        "interaction_supported_over_region": bool(interaction_supported),
        "interaction_beats_top_unit": bool(interaction_beats_unit),
        "effective_connectivity_proxy_passed": bool(interaction_supported),
        "target_replication": target_replication,
        "summaries": summaries,
        "results": results,
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx region-interaction effective-connectivity proxy gate",
        "",
        "Flat-unit gate 다음 단계로, channel-rescued region rates의 pairwise interaction이 additive region bins보다 나은지 확인한다.",
        "이 interaction은 causal connectivity가 아니라 trial-window effective coupling proxy다.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- folds: {output['folds']}",
        f"- ridge: {output['ridge']}",
        f"- permutations: {output['permutations']}",
        f"- max units per probe: {output['max_units_per_probe']}",
        f"- max interaction pairs: {output['max_interaction_pairs']}",
        f"- interaction supported over region: `{output['interaction_supported_over_region']}`",
        f"- interaction beats top unit: `{output['interaction_beats_top_unit']}`",
        f"- effective-connectivity proxy passed: `{output['effective_connectivity_proxy_passed']}`",
        "",
        "## model equation",
        "",
        "$$",
        "z_{iab}=r_{ia}r_{ib},\\qquad a<b.",
        "$$",
        "",
        "The tested interaction model is",
        "",
        "$$",
        "R_i^{\\mathrm{int}}=[R_i^{\\mathrm{hybrid}},Z_i],",
        "$$",
        "",
        "and the main increments are",
        "",
        "$$",
        "\\Delta_{\\mathrm{int-region}}=\\mathrm{BA}(R^{\\mathrm{int}})-\\mathrm{BA}(R^{\\mathrm{hybrid}}),",
        "\\qquad",
        "\\Delta_{\\mathrm{task+int}}=\\mathrm{BA}([X^{\\mathrm{task}},R^{\\mathrm{int}}])-\\mathrm{BA}([X^{\\mathrm{task}},R^{\\mathrm{hybrid}}]).",
        "$$",
        "",
        "## target replication",
        "",
        "| target | candidates | int beats region | int beats unit | task+int beats task+region | task+int beats task+unit | mean region BA | mean int BA | mean unit BA | mean task+int BA | mean int-region delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, row in output["target_replication"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{target}`",
                    str(row["candidate_count"]),
                    str(row["interaction_beats_region_count"]),
                    str(row["interaction_beats_top_unit_count"]),
                    str(row["task_interaction_beats_task_region_count"]),
                    str(row["task_interaction_beats_task_unit_count"]),
                    f"{row['mean_hybrid_region_balanced_accuracy']:.6f}",
                    f"{row['mean_region_interaction_balanced_accuracy']:.6f}",
                    f"{row['mean_top_unit_balanced_accuracy']:.6f}",
                    f"{row['mean_task_history_plus_region_interaction_balanced_accuracy']:.6f}",
                    f"{row['mean_interaction_delta_vs_region']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## candidate summaries",
            "",
            "| candidate | trials | int>region | int>unit | task+int>task+region | task+int>task+unit | choice int-region | speed int-region | wheel int-region |",
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
                    str(summary["interaction_beats_region_count"]),
                    str(summary["interaction_beats_top_unit_count"]),
                    str(summary["task_interaction_beats_task_region_count"]),
                    str(summary["task_interaction_beats_task_unit_count"]),
                    f"{by_target['choice_sign']['interaction_delta_vs_region']:.6f}",
                    f"{by_target['first_movement_speed']['interaction_delta_vs_region']:.6f}",
                    f"{by_target['wheel_action_direction']['interaction_delta_vs_region']:.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## verdict", ""])
    lines.extend(
        [
            f"- interaction supported over region: `{output['interaction_supported_over_region']}`",
            f"- interaction beats top unit: `{output['interaction_beats_top_unit']}`",
            f"- effective-connectivity proxy passed: `{output['effective_connectivity_proxy_passed']}`",
            "",
            "해석:",
            "",
            "- Region interaction이 additive region보다 반복적으로 높으면, mouse 항에는 additive region identity보다 weighted interaction 항이 필요하다.",
            "- Region interaction이 top-unit을 이기지 못하면, unit-detail residual은 여전히 남는다.",
            "- 이 결과는 causal effective connectivity가 아니라 windowed interaction proxy다. 다음 강한 버전은 lagged coupling 또는 trial-split nested regularization이다.",
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
    parser.add_argument("--max-interaction-pairs", type=int, default=750)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    output = aggregate(selected_candidates(args), args)
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(make_report(output), encoding="utf-8")

    print("Mouse IBL/OpenAlyx region-interaction effective-connectivity proxy gate")
    print(f"  candidates={output['candidate_count']}")
    for target, row in output["target_replication"].items():
        print(
            "  "
            + f"{target} int>region="
            + f"{row['interaction_beats_region_count']}/"
            + f"{row['candidate_count']} int>unit="
            + f"{row['interaction_beats_top_unit_count']}/"
            + f"{row['candidate_count']} mean_int-region="
            + f"{row['mean_interaction_delta_vs_region']:.6f}"
        )
    print(
        "  effective_connectivity_proxy_passed="
        + f"{output['effective_connectivity_proxy_passed']}"
    )
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
