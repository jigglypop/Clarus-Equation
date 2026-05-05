"""Mouse IBL/OpenAlyx action innovation-subspace mechanism map.

The nested innovation-subspace gate supported action targets.  This script
maps the train-selected action innovation axes back onto target-window unit
PCA loadings, then aggregates absolute loading mass by probe and anatomical
label.  It is a mechanism map, not a causal localization claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import load_probe, load_ranked_candidates
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
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


RESULT_JSON = Path(__file__).with_name("mouse_ibl_action_subspace_mechanism_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_action_subspace_mechanism_report.md")
UNKNOWN_LABEL = "UNKNOWN"


def pca_scores_with_basis(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_matrix, axis=0)
    scale = np.nanstd(train_matrix, axis=0)
    scale[scale < 1e-9] = 1.0
    train_z = (train_matrix - mean) / scale
    test_z = (test_matrix - mean) / scale
    _, singular_values, vt = np.linalg.svd(train_z, full_matrices=False)
    k = max(1, min(components, vt.shape[0]))
    basis = vt[:k].T
    explained = singular_values[:k] ** 2
    total = float(np.sum(singular_values**2))
    explained_fraction = explained / total if total > 1e-12 else np.zeros(k)
    return train_z @ basis, test_z @ basis, basis, mean, scale, explained_fraction


def fit_transition_predict_with_basis(
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0_train, h0_test, basis, _, _, explained = pca_scores_with_basis(
        u_target[train],
        u_target[test],
        args.components,
    )
    hl_train, hl_test, _, _, _, _ = pca_scores_with_basis(
        u_lag[train],
        u_lag[test],
        args.components,
    )
    x_train, x_test = zscore_block(x[train], x[test])
    r_train, r_test = zscore_block(r[train], r[test])
    design_train = np.column_stack([np.ones(len(train)), x_train, r_train, hl_train])
    design_test = np.column_stack([np.ones(len(test)), x_test, r_test, hl_test])
    penalty_values = (
        [0.0]
        + [args.task_penalty] * x_train.shape[1]
        + [args.region_penalty] * r_train.shape[1]
        + [args.latent_penalty] * hl_train.shape[1]
    )
    penalty = np.diag(penalty_values)
    lhs = design_train.T @ design_train + penalty
    rhs = design_train.T @ h0_train
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    hhat_train = design_train @ weights
    hhat_test = design_test @ weights
    eps_train = h0_train - hhat_train
    eps_test = h0_test - hhat_test
    return hhat_train, hhat_test, eps_train, eps_test, basis, explained


def parse_feature_name(name: str) -> tuple[str, int | None, bool]:
    probe_label, _, unit_name = name.partition(":")
    if unit_name.startswith("cluster:"):
        try:
            return probe_label, int(unit_name.split(":", 1)[1]), False
        except ValueError:
            return probe_label, None, False
    return probe_label, None, True


def unit_metadata(
    probes: list[dict[str, object]],
    unit_blocks: list[dict[str, object]],
) -> list[dict[str, object]]:
    probe_by_label = {str(probe["label"]): probe for probe in probes}
    rows = []
    for block in unit_blocks:
        for name in block["all_unit_feature_names"]:
            probe_label, cluster_id, is_other = parse_feature_name(str(name))
            probe = probe_by_label.get(probe_label)
            acronym = UNKNOWN_LABEL
            if probe is not None and cluster_id is not None:
                acronyms = np.asarray(probe.get("cluster_acronyms", []), dtype=object)
                if 0 <= cluster_id < len(acronyms):
                    value = str(acronyms[cluster_id])
                    if value and value.lower() != "nan":
                        acronym = value
                if acronym == UNKNOWN_LABEL:
                    cluster_channels = np.asarray(probe.get("cluster_channels", []), dtype=np.int64)
                    channel_region_ids = np.asarray(
                        probe.get("channel_region_ids", []),
                        dtype=np.int64,
                    )
                    if 0 <= cluster_id < len(cluster_channels):
                        channel = int(cluster_channels[cluster_id])
                        if 0 <= channel < len(channel_region_ids):
                            region_id = int(channel_region_ids[channel])
                            if region_id > 0:
                                acronym = f"ccf_id:{region_id}"
            if is_other:
                acronym = "OTHER_UNITS"
            rows.append(
                {
                    "feature": str(name),
                    "probe": probe_label,
                    "cluster_id": cluster_id,
                    "region": acronym,
                    "is_other": is_other,
                }
            )
    return rows


def aggregate_loadings(
    basis: np.ndarray,
    selected_axes: list[int],
    metadata: list[dict[str, object]],
    target: str,
) -> dict[str, object]:
    selected_indices = [axis - 1 for axis in selected_axes if 0 < axis <= basis.shape[1]]
    if not selected_indices:
        weights = np.zeros(basis.shape[0], dtype=float)
    else:
        weights = np.sum(np.abs(basis[:, selected_indices]), axis=1)
    total = float(np.sum(weights))
    if total <= 1e-12:
        total = 1.0
    region_mass: defaultdict[str, float] = defaultdict(float)
    probe_mass: defaultdict[str, float] = defaultdict(float)
    feature_rows = []
    for weight, meta in zip(weights, metadata):
        region = str(meta["region"])
        probe = str(meta["probe"])
        mass = float(weight / total)
        region_mass[region] += mass
        probe_mass[probe] += mass
        feature_rows.append(
            {
                "feature": meta["feature"],
                "probe": probe,
                "region": region,
                "cluster_id": meta["cluster_id"],
                "mass": mass,
                "target": target,
            }
        )
    feature_rows.sort(key=lambda row: row["mass"], reverse=True)
    return {
        "region_mass": dict(sorted(region_mass.items(), key=lambda item: item[1], reverse=True)),
        "probe_mass": dict(sorted(probe_mass.items(), key=lambda item: item[1], reverse=True)),
        "top_features": feature_rows[:20],
    }


def evaluate_target(
    target: str,
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    metadata: list[dict[str, object]],
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
    fold_rows = []
    baseline_scores = np.zeros(len(y), dtype=float)
    subspace_scores = np.zeros(len(y), dtype=float)
    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        hhat_train, hhat_test, eps_train, eps_test, basis, explained = (
            fit_transition_predict_with_basis(x, r, u_lag, u_target, train, test, args)
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
        from mouse_ibl_innovation_behavior_gate import ridge_classifier_scores

        baseline_scores[test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X", "R", "HHAT"],
            penalties,
        )
        sub_train_blocks = {**train_blocks, "EPS_SUB": eps_train[:, selected_indices]}
        sub_test_blocks = {**test_blocks, "EPS_SUB": eps_test[:, selected_indices]}
        sub_penalties = {**penalties, "EPS_SUB": args.innovation_penalty}
        subspace_scores[test] = ridge_classifier_scores(
            sub_train_blocks,
            sub_test_blocks,
            y[train],
            ["X", "R", "HHAT", "EPS_SUB"],
            sub_penalties,
        )
        loading_map = aggregate_loadings(basis, selected_axes, metadata, target)
        fold_rows.append(
            {
                "outer_fold": outer_index,
                "selected_axes": selected_axes,
                "selected_explained_fraction": [
                    float(explained[axis - 1]) for axis in selected_axes if 0 < axis <= len(explained)
                ],
                "inner_top_axis_rows": inner_axis_rows[: args.report_top_axes],
                **loading_map,
            }
        )
    baseline_ba = balanced_accuracy(y, (baseline_scores >= 0).astype(int))
    subspace_ba = balanced_accuracy(y, (subspace_scores >= 0).astype(int))
    return {
        "target": target,
        "trial_count": int(len(y)),
        "baseline_balanced_accuracy": baseline_ba,
        "subspace_balanced_accuracy": subspace_ba,
        "subspace_increment": float(subspace_ba - baseline_ba),
        "subspace_supported": bool(subspace_ba - baseline_ba > args.min_delta),
        "fold_rows": fold_rows,
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
    stim_unit, stim_unit_blocks, stim_unit_valid = unit_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    move_unit, move_unit_blocks, move_unit_valid = unit_features_for_window(
        probes, trials, MOVEMENT_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_stim_unit, _, pre_stim_valid = unit_features_for_window(
        probes, trials, PRE_STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_move_unit, _, pre_move_valid = unit_features_for_window(
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


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    target_names = [target["target"] for target in results[0]["targets"]]
    target_summary = []
    target_region_summary = []
    target_probe_summary = []
    target_feature_summary = []
    for target_name in target_names:
        target_rows = [
            target
            for result in results
            for target in result["targets"]
            if target["target"] == target_name
        ]
        increments = np.asarray([row["subspace_increment"] for row in target_rows], dtype=float)
        supported_count = int(np.sum(increments > args.min_delta))
        target_summary.append(
            {
                "target": target_name,
                "candidates": len(target_rows),
                "supported_count": supported_count,
                "mean_increment": float(np.mean(increments)),
                "median_increment": float(np.median(increments)),
                "supported": bool(
                    supported_count >= args.min_replications
                    and float(np.mean(increments)) > args.min_delta
                ),
            }
        )
        region_mass: defaultdict[str, float] = defaultdict(float)
        probe_mass: defaultdict[str, float] = defaultdict(float)
        feature_mass: defaultdict[str, float] = defaultdict(float)
        feature_meta: dict[str, dict[str, object]] = {}
        fold_count = 0
        for row in target_rows:
            for fold in row["fold_rows"]:
                fold_count += 1
                for region, mass in fold["region_mass"].items():
                    region_mass[region] += float(mass)
                for probe, mass in fold["probe_mass"].items():
                    probe_mass[probe] += float(mass)
                for feature in fold["top_features"]:
                    feature_mass[feature["feature"]] += float(feature["mass"])
                    feature_meta[feature["feature"]] = {
                        "probe": feature["probe"],
                        "region": feature["region"],
                        "cluster_id": feature["cluster_id"],
                    }
        denom = float(max(1, fold_count))
        region_rows = [
            {"target": target_name, "region": key, "mean_mass": value / denom}
            for key, value in region_mass.items()
        ]
        probe_rows = [
            {"target": target_name, "probe": key, "mean_mass": value / denom}
            for key, value in probe_mass.items()
        ]
        feature_rows = [
            {
                "target": target_name,
                "feature": key,
                "mean_top_feature_mass": value / denom,
                **feature_meta.get(key, {}),
            }
            for key, value in feature_mass.items()
        ]
        region_rows.sort(key=lambda row: row["mean_mass"], reverse=True)
        probe_rows.sort(key=lambda row: row["mean_mass"], reverse=True)
        feature_rows.sort(key=lambda row: row["mean_top_feature_mass"], reverse=True)
        target_region_summary.extend(region_rows[: args.report_top_regions])
        target_probe_summary.extend(probe_rows)
        target_feature_summary.extend(feature_rows[: args.report_top_features])
    return {
        "target_summary": target_summary,
        "target_region_summary": target_region_summary,
        "target_probe_summary": target_probe_summary,
        "target_feature_summary": target_feature_summary,
        "action_mechanism_map_passed": bool(any(row["supported"] for row in target_summary)),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx action innovation-subspace mechanism map",
        "",
        "$$",
        "\\epsilon_{S_{train}}\\rightarrow\\mathrm{unit/PCA\\ loading\\ mass}",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- outer folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- components: {output['components']}",
        f"- subspace size: {output['subspace_size']}",
        f"- action mechanism map passed: `{output['action_mechanism_map_passed']}`",
        "",
        "## target summary",
        "",
        "| target | candidates | supported | mean dBA | median dBA | passed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in output["target_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    str(row["candidates"]),
                    f"{row['supported_count']}/{row['candidates']}",
                    f"{row['mean_increment']:.6f}",
                    f"{row['median_increment']:.6f}",
                    f"`{row['supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## top region loading mass",
            "",
            "| target | region | mean mass |",
            "|---|---|---:|",
        ]
    )
    for row in output["target_region_summary"]:
        lines.append(f"| `{row['target']}` | `{row['region']}` | {row['mean_mass']:.6f} |")
    lines.extend(
        [
            "",
            "## probe loading mass",
            "",
            "| target | probe | mean mass |",
            "|---|---|---:|",
        ]
    )
    for row in output["target_probe_summary"]:
        lines.append(f"| `{row['target']}` | `{row['probe']}` | {row['mean_mass']:.6f} |")
    lines.extend(
        [
            "",
            "## top feature loading mass",
            "",
            "| target | feature | probe | region | mean top-feature mass |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in output["target_feature_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    f"`{row['feature']}`",
                    f"`{row.get('probe', UNKNOWN_LABEL)}`",
                    f"`{row.get('region', UNKNOWN_LABEL)}`",
                    f"{row['mean_top_feature_mass']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Loading mass maps selected innovation axes back to target-window unit PCA loadings.",
            "- This is a descriptive mechanism map. It does not prove causal localization.",
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
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": results,
    }
    output.update(summarize(results, args))
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    args.report_md.write_text(make_report(output))
    print("Mouse IBL/OpenAlyx action innovation-subspace mechanism map")
    for row in output["target_summary"]:
        print(
            f"  {row['target']}: supported={row['supported_count']}/"
            f"{row['candidates']} mean={row['mean_increment']:.6f}"
        )
    print(f"  action_mechanism_map_passed={output['action_mechanism_map_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
