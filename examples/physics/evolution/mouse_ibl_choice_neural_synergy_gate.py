"""Mouse IBL/OpenAlyx choice neural-block synergy gate.

The richer policy gate found that hand-built policy features do not reliably
beat the linear task/history block, while the combined neural block
``R + Hhat + nested eps`` survives after richer policy.  This gate decomposes
that residual by evaluating all single, pair, and triple neural-block
combinations after the richer policy block on the same outer folds.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_choice_policy_history_gate import candidate_namespace as base_candidate_namespace
from mouse_ibl_directed_latent_axis_split_gate import DEFAULT_CANDIDATES_JSON
from mouse_ibl_innovation_behavior_gate import (
    fit_transition_predict,
    load_probe,
    load_ranked_candidates,
    ridge_classifier_scores,
)
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, load_common
from mouse_ibl_nested_innovation_subspace_gate import select_axes_inside_train
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


RESULT_JSON = Path(__file__).with_name("mouse_ibl_choice_neural_synergy_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_choice_neural_synergy_report.md")
NEURAL_BLOCKS = ("R", "HHAT", "EPS_SUB")


def model_name(blocks: tuple[str, ...]) -> str:
    if not blocks:
        return "P"
    return "P_" + "_".join(blocks)


def neural_combinations() -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for size in range(1, len(NEURAL_BLOCKS) + 1):
        combos.extend(combinations(NEURAL_BLOCKS, size))
    return combos


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

    combos = neural_combinations()
    scores = {model_name(combo): np.zeros(len(y), dtype=float) for combo in combos}
    selected_by_fold = []

    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        hhat_train, hhat_test, eps_train, eps_test = fit_transition_predict(
            p_rich,
            r,
            u_lag,
            u_target,
            train,
            test,
            args,
        )
        selected_axes, inner_axis_rows = select_axes_inside_train(
            p_rich[train],
            r[train],
            hhat_train,
            eps_train,
            y[train],
            args,
        )
        selected_indices = [axis - 1 for axis in selected_axes]
        blocks_train = {
            "P": p_rich[train],
            "R": r[train],
            "HHAT": hhat_train,
            "EPS_SUB": eps_train[:, selected_indices],
        }
        blocks_test = {
            "P": p_rich[test],
            "R": r[test],
            "HHAT": hhat_test,
            "EPS_SUB": eps_test[:, selected_indices],
        }
        penalties = {
            "P": args.policy_penalty,
            "R": args.region_penalty,
            "HHAT": args.latent_penalty,
            "EPS_SUB": args.innovation_penalty,
        }
        for combo in combos:
            name = model_name(combo)
            block_names = ["P", *combo]
            scores[name][test] = ridge_classifier_scores(
                blocks_train,
                blocks_test,
                y[train],
                block_names,
                penalties,
            )
        selected_by_fold.append(
            {
                "outer_fold": outer_index,
                "selected_axes": selected_axes,
                "inner_top_axis_rows": inner_axis_rows[: args.report_top_axes],
            }
        )

    model_rows = []
    for combo in combos:
        name = model_name(combo)
        ba = balanced_accuracy(y, (scores[name] >= 0).astype(int))
        model_rows.append(
            {
                "model": name,
                "neural_blocks": list(combo),
                "block_count": len(combo),
                "balanced_accuracy": ba,
            }
        )
    by_model = {row["model"]: row["balanced_accuracy"] for row in model_rows}
    p_ba = by_model["P"]
    full_name = model_name(NEURAL_BLOCKS)
    full_ba = by_model[full_name]
    singles = [row for row in model_rows if row["block_count"] == 1]
    pairs = [row for row in model_rows if row["block_count"] == 2]
    best_single = max(singles, key=lambda row: row["balanced_accuracy"])
    best_pair = max(pairs, key=lambda row: row["balanced_accuracy"])
    increments = {
        "full_after_policy": float(full_ba - p_ba),
        "full_minus_best_single": float(full_ba - best_single["balanced_accuracy"]),
        "full_minus_best_pair": float(full_ba - best_pair["balanced_accuracy"]),
        "best_single_after_policy": float(best_single["balanced_accuracy"] - p_ba),
        "best_pair_after_policy": float(best_pair["balanced_accuracy"] - p_ba),
    }
    return {
        "target": "choice_sign",
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "model_rows": model_rows,
        "best_single_model": best_single["model"],
        "best_pair_model": best_pair["model"],
        "increments": increments,
        "selected_axes_by_fold": selected_by_fold,
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
    model_names = [row["model"] for row in results[0]["choice"]["model_rows"]]
    model_summary = []
    for model in model_names:
        rows = [
            row
            for result in results
            for row in result["choice"]["model_rows"]
            if row["model"] == model
        ]
        values = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        model_summary.append(
            {
                "model": model,
                "neural_blocks": rows[0]["neural_blocks"],
                "block_count": rows[0]["block_count"],
                "mean_balanced_accuracy": float(np.mean(values)),
                "median_balanced_accuracy": float(np.median(values)),
            }
        )
    model_summary.sort(key=lambda row: (row["block_count"], row["model"]))

    increment_names = list(results[0]["choice"]["increments"].keys())
    increment_summary = []
    for name in increment_names:
        values = np.asarray([result["choice"]["increments"][name] for result in results], dtype=float)
        positive_count = int(np.sum(values > args.min_delta))
        increment_summary.append(
            {
                "increment": name,
                "positive_count": positive_count,
                "mean_increment": float(np.mean(values)),
                "median_increment": float(np.median(values)),
                "supported": bool(
                    positive_count >= args.min_replications
                    and float(np.mean(values)) > args.min_delta
                ),
            }
        )

    best_single_counts: dict[str, int] = {}
    best_pair_counts: dict[str, int] = {}
    session_rows = []
    for result in results:
        choice = result["choice"]
        best_single = choice["best_single_model"]
        best_pair = choice["best_pair_model"]
        best_single_counts[best_single] = best_single_counts.get(best_single, 0) + 1
        best_pair_counts[best_pair] = best_pair_counts.get(best_pair, 0) + 1
        model_ba = {row["model"]: row["balanced_accuracy"] for row in choice["model_rows"]}
        session_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "policy_ba": model_ba["P"],
                "full_ba": model_ba[model_name(NEURAL_BLOCKS)],
                "best_single_model": best_single,
                "best_pair_model": best_pair,
                **choice["increments"],
            }
        )

    summary_by_increment = {row["increment"]: row for row in increment_summary}
    enough_replications = len(results) >= args.min_replications
    return {
        "model_summary": model_summary,
        "increment_summary": increment_summary,
        "session_rows": session_rows,
        "best_single_counts": best_single_counts,
        "best_pair_counts": best_pair_counts,
        "full_after_policy_supported": bool(
            enough_replications and summary_by_increment["full_after_policy"]["supported"]
        ),
        "synergy_over_best_single_supported": bool(
            enough_replications and summary_by_increment["full_minus_best_single"]["supported"]
        ),
        "synergy_over_best_pair_supported": bool(
            enough_replications and summary_by_increment["full_minus_best_pair"]["supported"]
        ),
        "choice_neural_synergy_gate_passed": bool(
            enough_replications
            and summary_by_increment["full_after_policy"]["supported"]
            and (
                summary_by_increment["full_minus_best_single"]["supported"]
                or summary_by_increment["full_minus_best_pair"]["supported"]
            )
        ),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx choice neural-block synergy gate",
        "",
        "$$",
        "y_{choice}=g(P_{rich},R,\\hat H,\\epsilon_{S_{train}})",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- outer folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- components: {output['components']}",
        f"- subspace size: {output['subspace_size']}",
        f"- max policy lag: {output['max_policy_lag']}",
        f"- full after policy supported: `{output['full_after_policy_supported']}`",
        f"- synergy over best single supported: `{output['synergy_over_best_single_supported']}`",
        f"- synergy over best pair supported: `{output['synergy_over_best_pair_supported']}`",
        f"- choice neural synergy gate passed: `{output['choice_neural_synergy_gate_passed']}`",
        "",
        "## model summary",
        "",
        "| model | blocks | mean BA | median BA |",
        "|---|---|---:|---:|",
    ]
    for row in output["model_summary"]:
        block_text = ",".join(row["neural_blocks"]) if row["neural_blocks"] else "none"
        lines.append(
            f"| `{row['model']}` | `{block_text}` | "
            f"{row['mean_balanced_accuracy']:.6f} | {row['median_balanced_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## increment summary",
            "",
            "| increment | positive | mean dBA | median dBA | supported |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in output["increment_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['increment']}`",
                    f"{row['positive_count']}/{output['candidate_count']}",
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
            "## best model counts",
            "",
            f"- best single counts: `{output['best_single_counts']}`",
            f"- best pair counts: `{output['best_pair_counts']}`",
            "",
            "## per-session",
            "",
            "| candidate | policy BA | full BA | best single | best pair | full-policy | full-best-single | full-best-pair |",
            "|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in output["session_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['policy_ba']:.6f}",
                    f"{row['full_ba']:.6f}",
                    f"`{row['best_single_model']}`",
                    f"`{row['best_pair_model']}`",
                    f"{row['full_after_policy']:.6f}",
                    f"{row['full_minus_best_single']:.6f}",
                    f"{row['full_minus_best_pair']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- `full after policy` tests whether the combined neural block remains after richer policy.",
            "- `full minus best single` asks whether the full neural block beats any one of R, Hhat, or EPS alone.",
            "- `full minus best pair` is the stricter three-way synergy test.",
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
    parser.add_argument("--max-policy-lag", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--policy-penalty", type=float, default=5.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--innovation-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=7)
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
        "max_policy_lag": args.max_policy_lag,
        "task_penalty": args.task_penalty,
        "policy_penalty": args.policy_penalty,
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
    print("Mouse IBL/OpenAlyx choice neural-block synergy gate")
    for row in output["increment_summary"]:
        print(
            f"  {row['increment']}: positive={row['positive_count']}/"
            f"{output['candidate_count']} mean={row['mean_increment']:.6f}"
        )
    print(f"  choice_neural_synergy_gate_passed={output['choice_neural_synergy_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
