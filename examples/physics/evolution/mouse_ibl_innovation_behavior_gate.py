"""Mouse IBL/OpenAlyx innovation-to-behavior gate.

The low-rank transition gate supported:

    H_t = A H_{t-lag} + B X_t + C R_t + eps_t

This gate asks where behavior attaches:

    y_t ~ X_t + R_t + Hhat_t + eps_t

For each outer behavior fold, the PCA basis and transition model are fit only
on the training trials.  The test trials receive a predicted latent state
Hhat_t and an innovation eps_t, then the behavior decoder is evaluated on held
out trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_channel_region_rescue_gate import CANDIDATES
from mouse_ibl_low_rank_unit_transition_gate import pca_scores
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
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
    zscore_block,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_innovation_behavior_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_innovation_behavior_report.md")


def output_paths(stem: str) -> tuple[Path, Path]:
    if Path(stem).name != stem:
        raise ValueError("--output-stem must be a file stem, not a path")
    base = Path(__file__).with_name(stem)
    return base.with_name(f"{base.name}_results.json"), base.with_name(f"{base.name}_report.md")


def probe_label(collection: str) -> str:
    for part in collection.split("/"):
        if part.startswith("probe"):
            return part
    return collection


def load_probe(one, eid: str, collection: str) -> dict[str, object]:
    try:
        cluster_acronyms = np.asarray(
            one.load_dataset(
                eid,
                "clusters.brainLocationAcronyms_ccf_2017.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=object,
        )
    except Exception:
        cluster_acronyms = np.asarray([], dtype=object)
    return {
        "collection": collection,
        "label": probe_label(collection),
        "spike_times": np.asarray(
            one.load_dataset(
                eid,
                "spikes.times.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=float,
        ),
        "spike_clusters": np.asarray(
            one.load_dataset(
                eid,
                "spikes.clusters.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
        "cluster_acronyms": cluster_acronyms,
        "cluster_channels": np.asarray(
            one.load_dataset(
                eid,
                "clusters.channels.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
        "channel_region_ids": np.asarray(
            one.load_dataset(
                eid,
                "channels.brainLocationIds_ccf_2017.npy",
                collection=collection,
                query_type="remote",
            ),
            dtype=np.int64,
        ),
    }


def load_ranked_candidates(path: str, limit: int | None = None) -> list[dict[str, object]]:
    data = json.loads(Path(path).read_text())
    candidates = []
    for item in data["selected_candidates"]:
        candidates.append(
            {
                "name": item["session_ref"].replace("/", "_"),
                "eid": item["eid"],
                "session_ref": item["session_ref"],
                "collections": item["channel_probe_collections"],
            }
        )
    return candidates if limit is None else candidates[:limit]


def fit_transition_predict(
    x: np.ndarray,
    r: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h0_train, h0_test, _ = pca_scores(u_target[train], u_target[test], args.components)
    hl_train, hl_test, _ = pca_scores(u_lag[train], u_lag[test], args.components)
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
    return hhat_train, hhat_test, eps_train, eps_test


def ridge_classifier_scores(
    train_blocks: dict[str, np.ndarray],
    test_blocks: dict[str, np.ndarray],
    y_train: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
) -> np.ndarray:
    values = np.unique(y_train)
    if len(values) != 2:
        raise ValueError("binary target required")
    y_signed = np.where(y_train == values[1], 1.0, -1.0)
    train_parts = []
    test_parts = []
    penalty_values = []
    for name in block_names:
        train_block, test_block = zscore_block(train_blocks[name], test_blocks[name])
        train_parts.append(train_block)
        test_parts.append(test_block)
        penalty_values.extend([penalties[name]] * train_block.shape[1])
    x_train = np.column_stack([np.ones(len(y_train)), hstack(train_parts)])
    x_test = np.column_stack([np.ones(test_parts[0].shape[0]), hstack(test_parts)])
    penalty = np.diag([0.0, *penalty_values])
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_signed
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_test @ weights


def evaluate_behavior_target(
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
    models = {
        "task_region": (
            ["X", "R"],
            {"X": args.task_penalty, "R": args.region_penalty},
        ),
        "task_region_predicted_latent": (
            ["X", "R", "HHAT"],
            {"X": args.task_penalty, "R": args.region_penalty, "HHAT": args.latent_penalty},
        ),
        "task_region_innovation": (
            ["X", "R", "EPS"],
            {"X": args.task_penalty, "R": args.region_penalty, "EPS": args.innovation_penalty},
        ),
        "task_region_predicted_latent_innovation": (
            ["X", "R", "HHAT", "EPS"],
            {
                "X": args.task_penalty,
                "R": args.region_penalty,
                "HHAT": args.latent_penalty,
                "EPS": args.innovation_penalty,
            },
        ),
    }
    scores = {name: np.zeros(len(y), dtype=float) for name in models}
    for outer_idx, test in enumerate(stratified_folds(y, args.folds, args.seed)):
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
        train_blocks = {"X": x[train], "R": r[train], "HHAT": hhat_train, "EPS": eps_train}
        test_blocks = {"X": x[test], "R": r[test], "HHAT": hhat_test, "EPS": eps_test}
        for name, (block_names, penalties) in models.items():
            scores[name][test] = ridge_classifier_scores(
                train_blocks,
                test_blocks,
                y[train],
                block_names,
                penalties,
            )
    rows = []
    for name in models:
        predicted = (scores[name] >= 0).astype(int)
        rows.append(
            {
                "model": name,
                "balanced_accuracy": balanced_accuracy(y, predicted),
            }
        )
    by = {row["model"]: row for row in rows}
    xr = by["task_region"]["balanced_accuracy"]
    xrh = by["task_region_predicted_latent"]["balanced_accuracy"]
    xre = by["task_region_innovation"]["balanced_accuracy"]
    xrhe = by["task_region_predicted_latent_innovation"]["balanced_accuracy"]
    return {
        "target": target,
        "window": window,
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "rows": rows,
        "nested_comparison": {
            "predicted_latent_increment_after_task_region": float(xrh - xr),
            "innovation_increment_after_task_region": float(xre - xr),
            "innovation_increment_after_task_region_predicted_latent": float(xrhe - xrh),
            "predicted_latent_increment_after_task_region_innovation": float(xrhe - xre),
            "predicted_latent_supported": bool(xrh > xr + args.min_delta),
            "innovation_supported": bool(xre > xr + args.min_delta),
            "innovation_after_predicted_latent_supported": bool(xrhe > xrh + args.min_delta),
        },
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
        evaluate_behavior_target(
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
        evaluate_behavior_target(
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
        evaluate_behavior_target(
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


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    target_names = [target["target"] for target in results[0]["targets"]]
    summary = []
    for target_name in target_names:
        rows = [
            target
            for result in results
            for target in result["targets"]
            if target["target"] == target_name
        ]
        nested = [row["nested_comparison"] for row in rows]
        h_pos = [item["predicted_latent_supported"] for item in nested]
        eps_pos = [item["innovation_supported"] for item in nested]
        eps_after_h_pos = [
            item["innovation_after_predicted_latent_supported"] for item in nested
        ]
        mean_h = float(
            np.mean([item["predicted_latent_increment_after_task_region"] for item in nested])
        )
        mean_eps = float(
            np.mean([item["innovation_increment_after_task_region"] for item in nested])
        )
        mean_eps_after_h = float(
            np.mean(
                [
                    item["innovation_increment_after_task_region_predicted_latent"]
                    for item in nested
                ]
            )
        )
        summary.append(
            {
                "target": target_name,
                "candidates": len(rows),
                "predicted_latent_positive": int(sum(h_pos)),
                "innovation_positive": int(sum(eps_pos)),
                "innovation_after_predicted_latent_positive": int(sum(eps_after_h_pos)),
                "mean_predicted_latent_increment": mean_h,
                "mean_innovation_increment": mean_eps,
                "mean_innovation_after_predicted_latent_increment": mean_eps_after_h,
                "innovation_supported": bool(
                    sum(eps_after_h_pos) >= args.min_replications
                    and mean_eps_after_h > args.min_delta
                ),
            }
        )
    return {
        "target_summary": summary,
        "innovation_behavior_gate_passed": bool(
            any(row["innovation_supported"] for row in summary)
        ),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx innovation-to-behavior gate",
        "",
        "$$",
        "y_t=g(X_t,R_t,\\hat H_t,\\epsilon_t)",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- components: {output['components']}",
        f"- folds: {output['folds']}",
        f"- innovation behavior gate passed: `{output['innovation_behavior_gate_passed']}`",
        "",
        "## target replication",
        "",
        "| target | candidates | Hhat positive | eps positive | eps after Hhat positive | mean dHhat | mean deps | mean deps after Hhat | innovation supported |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["target_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['target']}`",
                    str(row["candidates"]),
                    str(row["predicted_latent_positive"]),
                    str(row["innovation_positive"]),
                    str(row["innovation_after_predicted_latent_positive"]),
                    f"{row['mean_predicted_latent_increment']:.6f}",
                    f"{row['mean_innovation_increment']:.6f}",
                    f"{row['mean_innovation_after_predicted_latent_increment']:.6f}",
                    f"`{row['innovation_supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- `Hhat` asks whether the predictable latent trajectory carries behavior.",
            "- `eps after Hhat` asks whether the unpredictable latent innovation carries extra behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument(
        "--candidates-json",
        help="Read selected_candidates from a panel ranker JSON file.",
    )
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument(
        "--output-stem",
        default="mouse_ibl_innovation_behavior",
        help="Output stem for result/report files in this script directory.",
    )
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--innovation-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=4)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=64)
    parser.add_argument("--min-abs-wheel-displacement", type=float, default=0.02)
    args = parser.parse_args()

    if args.candidates_json:
        candidates = load_ranked_candidates(args.candidates_json, args.candidate_limit)
    elif args.single_session:
        candidates = [
            {
                "name": "nyu30_motor_striatal_multi_probe",
                "eid": args.eid,
                "session_ref": args.session_ref,
                "collections": args.collections,
            }
        ]
    else:
        candidates = CANDIDATES
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
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": results,
    }
    output.update(summarize(results, args))
    result_json, report_md = output_paths(args.output_stem)
    result_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    report_md.write_text(make_report(output))
    print("Mouse IBL/OpenAlyx innovation-to-behavior gate")
    for row in output["target_summary"]:
        print(
            f"  {row['target']}: eps_after_H={row['innovation_after_predicted_latent_positive']}/"
            f"{row['candidates']} mean={row['mean_innovation_after_predicted_latent_increment']:.6f}"
        )
    print(f"  gate_passed={output['innovation_behavior_gate_passed']}")
    print(f"Saved: {result_json}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
