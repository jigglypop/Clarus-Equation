"""Mouse IBL/OpenAlyx choice policy/history gate.

The nested innovation-subspace gate supported train-selected innovation
subspaces for action, but choice stayed weak.  This gate tests the next
candidate explanation: choice is dominated by task/history or policy-like
covariates, with little repeated neural innovation residual after those
covariates, hybrid region, and predicted low-rank state are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_choice_policy_history_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_choice_policy_history_report.md")


def evaluate_choice(
    x_all: np.ndarray,
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
        & np.all(np.isfinite(x_all), axis=1)
        & np.all(np.isfinite(r_all), axis=1)
        & np.all(np.isfinite(u_lag_all), axis=1)
        & np.all(np.isfinite(u_target_all), axis=1)
    )
    x = x_all[finite]
    r = r_all[finite]
    u_lag = u_lag_all[finite]
    u_target = u_target_all[finite]
    y = y_all[finite].astype(int)

    model_names = [
        "policy_history",
        "policy_history_region",
        "policy_history_region_predicted_latent",
        "policy_history_region_predicted_latent_nested_eps",
    ]
    scores = {name: np.zeros(len(y), dtype=float) for name in model_names}
    selected_by_fold = []

    for outer_index, test in enumerate(stratified_folds(y, args.folds, args.seed)):
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
        scores["policy_history"][test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X"],
            penalties,
        )
        scores["policy_history_region"][test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X", "R"],
            penalties,
        )
        scores["policy_history_region_predicted_latent"][test] = ridge_classifier_scores(
            train_blocks,
            test_blocks,
            y[train],
            ["X", "R", "HHAT"],
            penalties,
        )

        eps_train_sub = eps_train[:, selected_indices]
        eps_test_sub = eps_test[:, selected_indices]
        eps_train_blocks = {**train_blocks, "EPS_SUB": eps_train_sub}
        eps_test_blocks = {**test_blocks, "EPS_SUB": eps_test_sub}
        eps_penalties = {**penalties, "EPS_SUB": args.innovation_penalty}
        scores["policy_history_region_predicted_latent_nested_eps"][test] = (
            ridge_classifier_scores(
                eps_train_blocks,
                eps_test_blocks,
                y[train],
                ["X", "R", "HHAT", "EPS_SUB"],
                eps_penalties,
            )
        )
        selected_by_fold.append(
            {
                "outer_fold": outer_index,
                "selected_axes": selected_axes,
                "inner_top_axis_rows": inner_axis_rows[: args.report_top_axes],
            }
        )

    model_rows = []
    for name in model_names:
        ba = balanced_accuracy(y, (scores[name] >= 0).astype(int))
        model_rows.append({"model": name, "balanced_accuracy": ba})
    by_model = {row["model"]: row["balanced_accuracy"] for row in model_rows}
    increments = {
        "region_after_policy_history": float(
            by_model["policy_history_region"] - by_model["policy_history"]
        ),
        "predicted_latent_after_policy_history_region": float(
            by_model["policy_history_region_predicted_latent"]
            - by_model["policy_history_region"]
        ),
        "nested_eps_after_policy_history_region_predicted_latent": float(
            by_model["policy_history_region_predicted_latent_nested_eps"]
            - by_model["policy_history_region_predicted_latent"]
        ),
        "all_neural_after_policy_history": float(
            by_model["policy_history_region_predicted_latent_nested_eps"]
            - by_model["policy_history"]
        ),
    }
    return {
        "target": "choice_sign",
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "model_rows": model_rows,
        "increments": increments,
        "region_supported": bool(increments["region_after_policy_history"] > args.min_delta),
        "predicted_latent_supported": bool(
            increments["predicted_latent_after_policy_history_region"] > args.min_delta
        ),
        "nested_eps_supported": bool(
            increments["nested_eps_after_policy_history_region_predicted_latent"]
            > args.min_delta
        ),
        "all_neural_supported": bool(increments["all_neural_after_policy_history"] > args.min_delta),
        "selected_axes_by_fold": selected_by_fold,
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    task_history, task_history_names = task_history_covariates(trials)
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
        task_history,
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
        "task_history_feature_names": task_history_names,
        "task_history_feature_count": int(task_history.shape[1]),
        "choice_meta": choice_meta,
        "choice": choice_result,
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
    )


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    model_names = [row["model"] for row in results[0]["choice"]["model_rows"]]
    model_summary = []
    for model in model_names:
        values = [
            row["balanced_accuracy"]
            for result in results
            for row in result["choice"]["model_rows"]
            if row["model"] == model
        ]
        model_summary.append(
            {
                "model": model,
                "mean_balanced_accuracy": float(np.mean(values)),
                "median_balanced_accuracy": float(np.median(values)),
            }
        )

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

    session_rows = []
    for result in results:
        increments = result["choice"]["increments"]
        model_ba = {
            row["model"]: row["balanced_accuracy"] for row in result["choice"]["model_rows"]
        }
        session_rows.append(
            {
                "name": result["name"],
                "eid": result["eid"],
                "session_ref": result["session_ref"],
                "policy_history_ba": model_ba["policy_history"],
                "full_choice_model_ba": model_ba[
                    "policy_history_region_predicted_latent_nested_eps"
                ],
                **increments,
            }
        )

    summary_by_increment = {row["increment"]: row for row in increment_summary}
    enough_replications = len(results) >= args.min_replications
    nested_eps_supported = summary_by_increment[
        "nested_eps_after_policy_history_region_predicted_latent"
    ]["supported"]
    all_neural_supported = summary_by_increment["all_neural_after_policy_history"]["supported"]
    all_neural_mean = summary_by_increment["all_neural_after_policy_history"]["mean_increment"]
    return {
        "model_summary": model_summary,
        "increment_summary": increment_summary,
        "session_rows": session_rows,
        "choice_innovation_residual_supported": bool(
            enough_replications and nested_eps_supported
        ),
        "strict_policy_dominance_supported": bool(
            enough_replications
            and not all_neural_supported
            and all_neural_mean <= args.min_delta
        ),
        "choice_policy_history_gate_passed": bool(
            enough_replications and not nested_eps_supported
        ),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx choice policy/history gate",
        "",
        "$$",
        "y_{choice}=g(X_{policy/history},R,\\hat H,\\epsilon_{S_{train}})",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- outer folds: {output['folds']}",
        f"- inner folds: {output['inner_folds']}",
        f"- components: {output['components']}",
        f"- subspace size: {output['subspace_size']}",
        f"- choice innovation residual supported: `{output['choice_innovation_residual_supported']}`",
        f"- strict policy dominance supported: `{output['strict_policy_dominance_supported']}`",
        f"- choice policy/history gate passed: `{output['choice_policy_history_gate_passed']}`",
        "",
        "## model summary",
        "",
        "| model | mean BA | median BA |",
        "|---|---:|---:|",
    ]
    for row in output["model_summary"]:
        lines.append(
            f"| `{row['model']}` | {row['mean_balanced_accuracy']:.6f} | "
            f"{row['median_balanced_accuracy']:.6f} |"
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
            "## per-session",
            "",
            "| candidate | policy BA | full BA | R after policy | Hhat after X,R | nested eps after X,R,Hhat | all neural after policy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in output["session_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['policy_history_ba']:.6f}",
                    f"{row['full_choice_model_ba']:.6f}",
                    f"{row['region_after_policy_history']:.6f}",
                    f"{row['predicted_latent_after_policy_history_region']:.6f}",
                    f"{row['nested_eps_after_policy_history_region_predicted_latent']:.6f}",
                    f"{row['all_neural_after_policy_history']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- If `nested eps after X,R,Hhat` stays weak, the choice failure is not fixed by reusing the action innovation subspace.",
            "- If `all neural after policy` is also weak, the next choice term should be a richer policy/history latent rather than a neural innovation term.",
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
    print("Mouse IBL/OpenAlyx choice policy/history gate")
    for row in output["increment_summary"]:
        print(
            f"  {row['increment']}: positive={row['positive_count']}/"
            f"{output['candidate_count']} mean={row['mean_increment']:.6f}"
        )
    print(f"  choice_policy_history_gate_passed={output['choice_policy_history_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
