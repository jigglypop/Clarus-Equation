"""Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate.

The choice neural-block synergy gate narrowed the choice readout to
``P_rich + eps`` rather than a three-way neural synergy.  This gate selects
choice innovation axes using only outer-train trials and a policy-only
baseline, then asks whether the selected axes/subspace replicate and whether
their identities concentrate across sessions and folds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_choice_policy_history_gate import candidate_namespace as base_candidate_namespace
from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import (
    fit_transition_predict,
    load_probe,
    ridge_classifier_scores,
)
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, load_common
from mouse_ibl_region_decision_action_gate import (
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    balanced_accuracy,
    choice_target,
    stratified_folds,
)
from mouse_ibl_richer_choice_policy_gate import richer_policy_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_choice_innovation_reproducibility_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_choice_innovation_reproducibility_report.md")


def load_panel_candidates(
    path: Path,
    candidate_key: str,
    limit: int | None = None,
) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    candidates = []
    for item in data[candidate_key]:
        candidates.append(
            {
                "name": item["session_ref"].replace("/", "_"),
                "eid": item["eid"],
                "session_ref": item["session_ref"],
                "collections": item["channel_probe_collections"],
            }
        )
    return candidates if limit is None else candidates[:limit]


def normalized_entropy(counts: Counter[int], components: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    probs = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(components))


def monte_carlo_top_count_pvalue(
    observed_top_count: int,
    draws: int,
    total_draws: int,
    components: int,
    rng: np.random.Generator,
) -> float:
    exceed = 0
    for _ in range(draws):
        values = rng.integers(1, components + 1, size=total_draws)
        top = Counter(int(value) for value in values).most_common(1)[0][1]
        if top >= observed_top_count:
            exceed += 1
    return float(exceed / draws)


def select_choice_axes_inside_train(
    p_train: np.ndarray,
    eps_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict[str, object]]]:
    n_axes = eps_train.shape[1]
    baseline_scores = np.zeros(len(y_train), dtype=float)
    axis_scores = [np.zeros(len(y_train), dtype=float) for _ in range(n_axes)]
    for inner_test in stratified_folds(y_train, args.inner_folds, args.seed + 303):
        inner_train = np.setdiff1d(np.arange(len(y_train)), inner_test)
        train_blocks = {"P": p_train[inner_train]}
        test_blocks = {"P": p_train[inner_test]}
        penalties = {"P": args.policy_penalty}
        baseline_scores[inner_test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y_train[inner_train],
            ["P"],
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
                ["P", axis_name],
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


def evaluate_choice(
    p_rich_all: np.ndarray,
    r_all: np.ndarray,
    u_lag_all: np.ndarray,
    u_target_all: np.ndarray,
    y_all: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    finite = (
        valid
        & np.isfinite(y_all)
        & np.all(np.isfinite(p_rich_all), axis=1)
        & np.all(np.isfinite(r_all), axis=1)
        & np.all(np.isfinite(u_lag_all), axis=1)
        & np.all(np.isfinite(u_target_all), axis=1)
    )
    p_rich = p_rich_all[finite]
    r = r_all[finite]
    u_lag = u_lag_all[finite]
    u_target = u_target_all[finite]
    y = y_all[finite].astype(int)

    policy_scores = np.zeros(len(y), dtype=float)
    subspace_scores = np.zeros(len(y), dtype=float)
    selected_by_fold = []

    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        _, _, eps_train, eps_test = fit_transition_predict(
            p_rich,
            r,
            u_lag,
            u_target,
            train,
            test,
            args,
        )
        selected_axes, inner_axis_rows = select_choice_axes_inside_train(
            p_rich[train],
            eps_train,
            y[train],
            args,
        )
        selected_indices = [axis - 1 for axis in selected_axes]
        train_blocks = {"P": p_rich[train]}
        test_blocks = {"P": p_rich[test]}
        penalties = {"P": args.policy_penalty}
        policy_scores[test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["P"],
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
            ["P", "EPS_SUB"],
            subspace_penalties,
        )
        selected_by_fold.append(
            {
                "outer_fold": outer_index,
                "selected_axes": selected_axes,
                "inner_top_axis_rows": inner_axis_rows[: args.report_top_axes],
            }
        )

    policy_ba = balanced_accuracy(y, (policy_scores >= 0).astype(int))
    subspace_ba = balanced_accuracy(y, (subspace_scores >= 0).astype(int))
    increment = float(subspace_ba - policy_ba)
    selected_axes_all = [
        axis for fold in selected_by_fold for axis in fold["selected_axes"]
    ]
    counts = Counter(int(axis) for axis in selected_axes_all)
    most_common = counts.most_common()
    top1_axis, top1_count = most_common[0]
    top3_axes = [axis for axis, _ in most_common[:3]]
    top3_count = sum(count for _, count in most_common[:3])
    folds_with_top3 = sum(
        bool(set(fold["selected_axes"]) & set(top3_axes)) for fold in selected_by_fold
    )
    return {
        "target": "choice_sign",
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "policy_balanced_accuracy": policy_ba,
        "subspace_balanced_accuracy": subspace_ba,
        "subspace_increment": increment,
        "subspace_supported": bool(increment > args.min_delta),
        "selected_axes_by_fold": selected_by_fold,
        "axis_counts": dict(sorted(counts.items())),
        "top1_axis": top1_axis,
        "top1_count": top1_count,
        "top3_axes": top3_axes,
        "top3_count": top3_count,
        "folds_with_top3_axis": folds_with_top3,
        "normalized_entropy": normalized_entropy(counts, args.components),
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    richer_policy, richer_names = richer_policy_history_covariates(trials, args.max_policy_lag)
    stim_region, _, stim_region_valid = region_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_label_spikes
    )
    stim_unit, _, stim_unit_valid = unit_features_for_window(
        probes, trials, STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    pre_stim_unit, _, pre_stim_valid = unit_features_for_window(
        probes, trials, PRE_STIMULUS_WINDOW, args.min_unit_spikes, args.max_units_per_probe
    )
    choice, choice_valid, choice_meta = choice_target(trials)
    choice_result = evaluate_choice(
        richer_policy,
        stim_region,
        pre_stim_unit,
        stim_unit,
        choice,
        choice_valid & stim_region_valid & stim_unit_valid & pre_stim_valid,
        args,
    )
    return {
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "richer_policy_feature_count": int(richer_policy.shape[1]),
        "richer_policy_feature_names": richer_names,
        "choice_meta": choice_meta,
        "choice": choice_result,
    }


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    ns = base_candidate_namespace(candidate, args)
    ns.policy_penalty = args.policy_penalty
    ns.max_policy_lag = args.max_policy_lag
    return ns


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    increments = np.asarray([result["choice"]["subspace_increment"] for result in results], dtype=float)
    supported_count = int(np.sum(increments > args.min_delta))
    global_counts: Counter[int] = Counter()
    session_rows = []
    for result in results:
        choice = result["choice"]
        global_counts.update({int(axis): int(count) for axis, count in choice["axis_counts"].items()})
        session_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "policy_ba": choice["policy_balanced_accuracy"],
                "subspace_ba": choice["subspace_balanced_accuracy"],
                "subspace_increment": choice["subspace_increment"],
                "subspace_supported": choice["subspace_supported"],
                "top1_axis": choice["top1_axis"],
                "top1_count": choice["top1_count"],
                "top3_axes": choice["top3_axes"],
                "top3_count": choice["top3_count"],
                "folds_with_top3_axis": choice["folds_with_top3_axis"],
                "normalized_entropy": choice["normalized_entropy"],
            }
        )

    total_axis_draws = sum(global_counts.values())
    most_common = global_counts.most_common()
    global_top1_axis, global_top1_count = most_common[0]
    global_top3_axes = [axis for axis, _ in most_common[:3]]
    global_top3_count = sum(count for _, count in most_common[:3])
    rng = np.random.default_rng(args.seed + 909)
    global_entropy = normalized_entropy(global_counts, args.components)
    top1_p = monte_carlo_top_count_pvalue(
        global_top1_count,
        args.null_draws,
        total_axis_draws,
        args.components,
        rng,
    )
    return {
        "increment_summary": {
            "supported_count": supported_count,
            "mean_increment": float(np.mean(increments)),
            "median_increment": float(np.median(increments)),
            "supported": bool(
                supported_count >= args.min_replications
                and float(np.mean(increments)) > args.min_delta
            ),
        },
        "global_axis_summary": {
            "axis_counts": dict(sorted(global_counts.items())),
            "total_axis_draws": total_axis_draws,
            "top1_axis": global_top1_axis,
            "top1_count": global_top1_count,
            "top1_share": float(global_top1_count / total_axis_draws),
            "top3_axes": global_top3_axes,
            "top3_count": global_top3_count,
            "top3_share": float(global_top3_count / total_axis_draws),
            "normalized_entropy": global_entropy,
            "top1_uniform_null_p": top1_p,
            "axis_identity_stable": bool(
                global_top1_count >= args.min_global_top1_count
                and global_entropy <= args.max_entropy
            ),
            "subspace_concentrated": bool(global_top3_count >= args.min_global_top3_count),
        },
        "session_rows": session_rows,
        "choice_innovation_reproducibility_gate_passed": bool(
            supported_count >= args.min_replications
            and float(np.mean(increments)) > args.min_delta
            and (
                global_top1_count >= args.min_global_top1_count
                or global_top3_count >= args.min_global_top3_count
            )
        ),
    }


def make_report(output: dict[str, object]) -> str:
    inc = output["increment_summary"]
    axis = output["global_axis_summary"]
    lines = [
        "# Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate",
        "",
        "$$",
        "y_{choice}=g(P_{rich},\\epsilon_{S_{choice,train}})",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- outer folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- components: {output['components']}",
        f"- subspace size: {output['subspace_size']}",
        f"- choice innovation reproducibility gate passed: `{output['choice_innovation_reproducibility_gate_passed']}`",
        "",
        "## increment summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| supported | {inc['supported_count']}/{output['candidate_count']} |",
        f"| mean dBA | {inc['mean_increment']:.6f} |",
        f"| median dBA | {inc['median_increment']:.6f} |",
        f"| replicated | `{inc['supported']}` |",
        "",
        "## global axis summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| top1 axis | {axis['top1_axis']} |",
        f"| top1 count | {axis['top1_count']}/{axis['total_axis_draws']} |",
        f"| top1 share | {axis['top1_share']:.6f} |",
        f"| top3 axes | `{axis['top3_axes']}` |",
        f"| top3 count | {axis['top3_count']}/{axis['total_axis_draws']} |",
        f"| top3 share | {axis['top3_share']:.6f} |",
        f"| entropy | {axis['normalized_entropy']:.6f} |",
        f"| top1 null p | {axis['top1_uniform_null_p']:.6f} |",
        f"| stable identity | `{axis['axis_identity_stable']}` |",
        f"| concentrated subspace | `{axis['subspace_concentrated']}` |",
        "",
        "## per-session",
        "",
        "| candidate | policy BA | subspace BA | dBA | supported | top1 axis | top1 | top3 axes | top3 | entropy |",
        "|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in output["session_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['policy_ba']:.6f}",
                    f"{row['subspace_ba']:.6f}",
                    f"{row['subspace_increment']:.6f}",
                    f"`{row['subspace_supported']}`",
                    str(row["top1_axis"]),
                    str(row["top1_count"]),
                    f"`{row['top3_axes']}`",
                    str(row["top3_count"]),
                    f"{row['normalized_entropy']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- Positive dBA means a choice-selected innovation subspace survives after richer policy.",
            "- Axis concentration asks whether selected axes repeat across sessions and outer folds.",
            "- If dBA replicates but axes do not concentrate, the term remains a session-adaptive innovation readout rather than a named stable subspace.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--candidates-json", type=Path, default=DEFAULT_CANDIDATES_JSON)
    parser.add_argument(
        "--candidate-key",
        choices=["selected_candidates", "top_candidates"],
        default="selected_candidates",
    )
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
    parser.add_argument("--max-policy-lag", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--policy-penalty", type=float, default=5.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--innovation-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=7)
    parser.add_argument("--min-global-top1-count", type=int, default=20)
    parser.add_argument("--min-global-top3-count", type=int, default=32)
    parser.add_argument("--max-entropy", type=float, default=0.75)
    parser.add_argument("--null-draws", type=int, default=20000)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=64)
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
        candidates = load_panel_candidates(
            args.candidates_json,
            args.candidate_key,
            args.candidate_limit,
        )

    results = []
    for candidate in candidates:
        result = evaluate_session(candidate_namespace(candidate, args))
        result["name"] = candidate["name"]
        results.append(result)

    output = {
        "openalyx_url": OPENALYX_URL,
        "candidate_count": len(results),
        "candidate_key": args.candidate_key,
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "components": args.components,
        "subspace_size": args.subspace_size,
        "max_policy_lag": args.max_policy_lag,
        "task_penalty": args.task_penalty,
        "policy_penalty": args.policy_penalty,
        "region_penalty": args.region_penalty,
        "latent_penalty": args.latent_penalty,
        "innovation_penalty": args.innovation_penalty,
        "min_delta": args.min_delta,
        "min_replications": args.min_replications,
        "min_global_top1_count": args.min_global_top1_count,
        "min_global_top3_count": args.min_global_top3_count,
        "max_entropy": args.max_entropy,
        "null_draws": args.null_draws,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": results,
    }
    output.update(summarize(results, args))
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    args.report_md.write_text(make_report(output))
    inc = output["increment_summary"]
    axis = output["global_axis_summary"]
    print("Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate")
    print(
        f"  choice_eps_after_policy: supported={inc['supported_count']}/"
        f"{output['candidate_count']} mean={inc['mean_increment']:.6f}"
    )
    print(
        f"  top1_axis={axis['top1_axis']} top1={axis['top1_count']}/"
        f"{axis['total_axis_draws']} top3={axis['top3_count']}/{axis['total_axis_draws']}"
    )
    print(
        "  choice_innovation_reproducibility_gate_passed="
        f"{output['choice_innovation_reproducibility_gate_passed']}"
    )
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
