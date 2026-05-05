"""Ablate top action-map CCF/probe blocks in the nested action subspace gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mouse_ibl_action_subspace_mechanism_map import (
    RESULT_JSON as MECHANISM_JSON,
    candidate_namespace as mechanism_candidate_namespace,
    evaluate_target,
    unit_metadata,
)
from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import load_probe, load_ranked_candidates
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    first_movement_speed_target,
    wheel_action_direction_target,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_MOVEMENT_WINDOW,
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_action_subspace_region_ablation_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_action_subspace_region_ablation_report.md")


def parse_ccf_id(region: str) -> int | None:
    prefix = "ccf_id:"
    if not region.startswith(prefix):
        return None
    try:
        return int(region[len(prefix) :])
    except ValueError:
        return None


def metadata_mask(
    metadata: list[dict[str, Any]],
    condition: str,
    top_ccf_ids: set[int],
    top_acronyms: set[str],
    probe_label: str,
) -> np.ndarray:
    mask = []
    for row in metadata:
        region = str(row.get("region", ""))
        ccf_id = parse_ccf_id(region)
        probe = str(row.get("probe", ""))
        is_top_ccf = (ccf_id in top_ccf_ids if ccf_id is not None else False) or (
            region in top_acronyms
        )
        is_probe = probe == probe_label
        if condition == "full":
            keep = True
        elif condition == "drop_top_ccf":
            keep = not is_top_ccf
        elif condition == "only_top_ccf":
            keep = is_top_ccf
        elif condition == "drop_probe":
            keep = not is_probe
        elif condition == "only_probe":
            keep = is_probe
        else:
            raise ValueError(f"unknown condition: {condition}")
        mask.append(keep)
    return np.asarray(mask, dtype=bool)


def subset_metadata(metadata: list[dict[str, Any]], mask: np.ndarray) -> list[dict[str, Any]]:
    return [row for row, keep in zip(metadata, mask) if bool(keep)]


def evaluate_target_condition(
    target: str,
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    lag_metadata: list[dict[str, Any]],
    target_metadata: list[dict[str, Any]],
    y_all: np.ndarray,
    valid: np.ndarray,
    condition: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    top_ccf_ids = set(args.top_ccf_ids)
    top_acronyms = set(args.top_acronyms)
    lag_mask = metadata_mask(lag_metadata, condition, top_ccf_ids, top_acronyms, args.probe_label)
    target_mask = metadata_mask(
        target_metadata,
        condition,
        top_ccf_ids,
        top_acronyms,
        args.probe_label,
    )
    if int(np.sum(lag_mask)) < args.min_ablation_units or int(np.sum(target_mask)) < args.min_ablation_units:
        return {
            "target": target,
            "condition": condition,
            "lag_units": int(np.sum(lag_mask)),
            "target_units": int(np.sum(target_mask)),
            "trial_count": 0,
            "baseline_balanced_accuracy": None,
            "subspace_balanced_accuracy": None,
            "subspace_increment": None,
            "subspace_supported": False,
            "fold_rows": [],
            "skipped": True,
        }
    result = evaluate_target(
        target,
        x,
        r,
        u_lag[:, lag_mask],
        u_target[:, target_mask],
        subset_metadata(target_metadata, target_mask),
        y_all,
        valid,
        args,
    )
    result["condition"] = condition
    result["lag_units"] = int(np.sum(lag_mask))
    result["target_units"] = int(np.sum(target_mask))
    result["skipped"] = False
    return result


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

    stim_metadata = unit_metadata(probes, stim_unit_blocks)
    move_metadata = unit_metadata(probes, move_unit_blocks)
    pre_stim_metadata = unit_metadata(probes, pre_stim_unit_blocks)
    pre_move_metadata = unit_metadata(probes, pre_move_unit_blocks)

    targets = []
    for condition in args.conditions:
        targets.append(
            evaluate_target_condition(
                "first_movement_speed",
                task_history,
                stim_region,
                pre_stim_unit,
                stim_unit,
                pre_stim_metadata,
                stim_metadata,
                speed,
                speed_valid & stim_region_valid & stim_unit_valid & pre_stim_valid,
                condition,
                args,
            )
        )
        targets.append(
            evaluate_target_condition(
                "wheel_action_direction",
                task_history,
                move_region,
                pre_move_unit,
                move_unit,
                pre_move_metadata,
                move_metadata,
                wheel,
                wheel_valid & move_region_valid & move_unit_valid & pre_move_valid,
                condition,
                args,
            )
        )
    return {
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "targets": targets,
    }


def candidate_namespace(candidate: dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    base = mechanism_candidate_namespace(candidate, args)
    base.conditions = args.conditions
    base.top_ccf_ids = args.top_ccf_ids
    base.top_acronyms = args.top_acronyms
    base.probe_label = args.probe_label
    base.min_ablation_units = args.min_ablation_units
    return base


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    target_names = sorted({target["target"] for result in results for target in result["targets"]})
    for target_name in target_names:
        full_mean = None
        for condition in args.conditions:
            target_rows = [
                target
                for result in results
                for target in result["targets"]
                if target["target"] == target_name and target["condition"] == condition
            ]
            increments = [
                float(row["subspace_increment"])
                for row in target_rows
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
                    "candidates": len(target_rows),
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
        "# Mouse IBL/OpenAlyx action subspace region/probe ablation",
        "",
        "Ablation is applied to the unit matrices before the low-rank transition and nested action-subspace readout.",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- top CCF ids: `{output['top_ccf_ids']}`",
        f"- top acronyms: `{output['top_acronyms']}`",
        f"- probe label: `{output['probe_label']}`",
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
        "- `drop_top_ccf` asks whether the action increment survives without the top anatomical block.",
        "- `only_top_ccf` asks whether the top anatomical block is sufficient by itself.",
        "- `drop_probe` and `only_probe` run the same check for the dominant probe block.",
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
    parser.add_argument("--report-top-axes", type=int, default=5)
    parser.add_argument("--report-top-regions", type=int, default=12)
    parser.add_argument("--report-top-features", type=int, default=16)
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
    parser.add_argument("--top-ccf-ids", nargs="+", type=int, default=[215, 1020, 946, 128, 313])
    parser.add_argument("--top-acronyms", nargs="+", default=["APN", "PO", "PH", "MRN", "MB"])
    parser.add_argument("--probe-label", default="probe00")
    parser.add_argument("--min-ablation-units", type=int, default=4)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["full", "drop_top_ccf", "only_top_ccf", "drop_probe", "only_probe"],
    )
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
        "mechanism_json": str(MECHANISM_JSON),
        "openalyx_url": OPENALYX_URL,
        "candidate_count": len(results),
        "conditions": args.conditions,
        "top_ccf_ids": args.top_ccf_ids,
        "top_acronyms": args.top_acronyms,
        "probe_label": args.probe_label,
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

    print("Mouse IBL/OpenAlyx action subspace region/probe ablation")
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
