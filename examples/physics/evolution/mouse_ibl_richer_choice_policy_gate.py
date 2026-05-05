"""Mouse IBL/OpenAlyx richer choice policy/history gate.

The linear policy/history gate showed that choice is already strongly decoded
from task/history covariates, while the train-selected innovation subspace does
not replicate as a choice residual.  This gate expands the policy/history block
with lagged choice/outcome traces and task-history interactions, then asks
whether neural residuals remain after that richer policy block.
"""

from __future__ import annotations

import argparse
import json
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
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_nested_innovation_subspace_gate import select_axes_inside_train
from mouse_ibl_region_decision_action_gate import (
    OPENALYX_URL,
    RNG_SEED,
    STIMULUS_WINDOW,
    balanced_accuracy,
    choice_target,
    stratified_folds,
)
from mouse_ibl_task_baseline_comparison_gate import (
    add_feature,
    column,
    lag,
    task_history_covariates,
)
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
)


RESULT_JSON = Path(__file__).with_name("mouse_ibl_richer_choice_policy_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_richer_choice_policy_report.md")


def lag_n(values: np.ndarray, n: int, fill: float = 0.0) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    for _ in range(n):
        result = lag(result, fill)
    return result


def decayed_trace(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    trace = np.zeros_like(values, dtype=float)
    state = 0.0
    for i, value in enumerate(values):
        trace[i] = state
        if np.isfinite(value):
            state = alpha * state + (1.0 - alpha) * value
        else:
            state = alpha * state
    return trace


def richer_policy_history_covariates(trials, max_lag: int = 5) -> tuple[np.ndarray, list[str]]:
    base, names = task_history_covariates(trials)
    matrices = [base]
    rich_names = list(names)

    left = column(trials, "contrastLeft")
    right = column(trials, "contrastRight")
    left0 = np.where(np.isfinite(left), left, 0.0)
    right0 = np.where(np.isfinite(right), right, 0.0)
    signed_contrast = right0 - left0
    absolute_contrast = np.maximum(np.abs(left0), np.abs(right0))
    stimulus_side = np.where(np.isfinite(right), 1.0, np.where(np.isfinite(left), -1.0, 0.0))
    probability_left = column(trials, "probabilityLeft")
    prior_right = np.where(np.isfinite(probability_left), 1.0 - probability_left, 0.5)
    prior_signed = prior_right - np.where(np.isfinite(probability_left), probability_left, 0.5)

    choice_raw = column(trials, "choice")
    choice_signed = np.sign(choice_raw)
    feedback = column(trials, "feedbackType")
    reward = column(trials, "rewardVolume")
    correct = (feedback > 0).astype(float)
    previous_choice = lag(choice_signed)
    previous_feedback = lag(feedback)
    previous_reward = lag(reward)
    previous_correct = lag(correct)
    win_stay = previous_choice * np.where(previous_feedback > 0, 1.0, 0.0)
    lose_switch = -previous_choice * np.where(previous_feedback < 0, 1.0, 0.0)

    interaction_features = {
        "signed_contrast_x_prior": signed_contrast * prior_signed,
        "abs_contrast_x_prior": absolute_contrast * prior_signed,
        "stimulus_side_x_prior": stimulus_side * prior_signed,
        "signed_contrast_x_previous_choice": signed_contrast * previous_choice,
        "prior_x_previous_choice": prior_signed * previous_choice,
        "prior_x_previous_feedback": prior_signed * previous_feedback,
        "previous_choice_x_previous_feedback": previous_choice * previous_feedback,
        "win_stay": win_stay,
        "lose_switch": lose_switch,
        "previous_reward_x_previous_choice": previous_reward * previous_choice,
        "previous_correct_x_previous_choice": previous_correct * previous_choice,
    }
    for name, values in interaction_features.items():
        add_feature(matrices, rich_names, name, values)

    for k in range(2, max_lag + 1):
        add_feature(matrices, rich_names, f"choice_lag_{k}", lag_n(choice_signed, k))
        add_feature(matrices, rich_names, f"feedback_lag_{k}", lag_n(feedback, k))
        add_feature(matrices, rich_names, f"reward_lag_{k}", lag_n(reward, k))
        add_feature(matrices, rich_names, f"signed_contrast_lag_{k}", lag_n(signed_contrast, k))
        add_feature(
            matrices,
            rich_names,
            f"choice_x_feedback_lag_{k}",
            lag_n(choice_signed * feedback, k),
        )

    for alpha in (0.5, 0.75, 0.9):
        label = str(alpha).replace(".", "p")
        add_feature(matrices, rich_names, f"choice_trace_alpha_{label}", decayed_trace(choice_signed, alpha))
        add_feature(matrices, rich_names, f"feedback_trace_alpha_{label}", decayed_trace(feedback, alpha))
        add_feature(matrices, rich_names, f"reward_trace_alpha_{label}", decayed_trace(reward, alpha))
        add_feature(
            matrices,
            rich_names,
            f"choice_feedback_trace_alpha_{label}",
            decayed_trace(choice_signed * feedback, alpha),
        )

    add_feature(matrices, rich_names, "prior_signed", prior_signed)
    add_feature(matrices, rich_names, "prior_signed_squared", prior_signed**2)
    add_feature(matrices, rich_names, "signed_contrast_squared", signed_contrast**2)
    return hstack(matrices), rich_names


def evaluate_choice(
    x_linear_all: np.ndarray,
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
        & np.all(np.isfinite(x_linear_all), axis=1)
        & np.all(np.isfinite(p_rich_all), axis=1)
        & np.all(np.isfinite(r_all), axis=1)
        & np.all(np.isfinite(u_lag_all), axis=1)
        & np.all(np.isfinite(u_target_all), axis=1)
    )
    x_linear = x_linear_all[finite]
    p_rich = p_rich_all[finite]
    r = r_all[finite]
    u_lag = u_lag_all[finite]
    u_target = u_target_all[finite]
    y = y_all[finite].astype(int)

    model_names = [
        "linear_policy_history",
        "richer_policy_history",
        "richer_policy_history_region",
        "richer_policy_history_region_predicted_latent",
        "richer_policy_history_region_predicted_latent_nested_eps",
    ]
    scores = {name: np.zeros(len(y), dtype=float) for name in model_names}
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
            "X": x_linear[train],
            "P": p_rich[train],
            "R": r[train],
            "HHAT": hhat_train,
        }
        blocks_test = {
            "X": x_linear[test],
            "P": p_rich[test],
            "R": r[test],
            "HHAT": hhat_test,
        }
        penalties = {
            "X": args.task_penalty,
            "P": args.policy_penalty,
            "R": args.region_penalty,
            "HHAT": args.latent_penalty,
        }
        scores["linear_policy_history"][test] = ridge_classifier_scores(
            blocks_train, blocks_test, y[train], ["X"], penalties
        )
        scores["richer_policy_history"][test] = ridge_classifier_scores(
            blocks_train, blocks_test, y[train], ["P"], penalties
        )
        scores["richer_policy_history_region"][test] = ridge_classifier_scores(
            blocks_train, blocks_test, y[train], ["P", "R"], penalties
        )
        scores["richer_policy_history_region_predicted_latent"][test] = (
            ridge_classifier_scores(blocks_train, blocks_test, y[train], ["P", "R", "HHAT"], penalties)
        )
        eps_train_blocks = {
            **blocks_train,
            "EPS_SUB": eps_train[:, selected_indices],
        }
        eps_test_blocks = {
            **blocks_test,
            "EPS_SUB": eps_test[:, selected_indices],
        }
        eps_penalties = {**penalties, "EPS_SUB": args.innovation_penalty}
        scores["richer_policy_history_region_predicted_latent_nested_eps"][test] = (
            ridge_classifier_scores(
                eps_train_blocks,
                eps_test_blocks,
                y[train],
                ["P", "R", "HHAT", "EPS_SUB"],
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
        "richer_policy_after_linear_policy": float(
            by_model["richer_policy_history"] - by_model["linear_policy_history"]
        ),
        "region_after_richer_policy": float(
            by_model["richer_policy_history_region"] - by_model["richer_policy_history"]
        ),
        "predicted_latent_after_richer_policy_region": float(
            by_model["richer_policy_history_region_predicted_latent"]
            - by_model["richer_policy_history_region"]
        ),
        "nested_eps_after_richer_policy_region_predicted_latent": float(
            by_model["richer_policy_history_region_predicted_latent_nested_eps"]
            - by_model["richer_policy_history_region_predicted_latent"]
        ),
        "all_neural_after_richer_policy": float(
            by_model["richer_policy_history_region_predicted_latent_nested_eps"]
            - by_model["richer_policy_history"]
        ),
    }
    return {
        "target": "choice_sign",
        "trial_count": int(len(y)),
        "class_counts": {str(int(value)): int(np.sum(y == value)) for value in np.unique(y)},
        "model_rows": model_rows,
        "increments": increments,
        "selected_axes_by_fold": selected_by_fold,
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
    probes = [load_probe(one, args.eid, collection) for collection in args.collections]
    linear_policy, linear_names = task_history_covariates(trials)
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
        linear_policy,
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
        "linear_policy_feature_count": int(linear_policy.shape[1]),
        "richer_policy_feature_count": int(richer_policy.shape[1]),
        "linear_policy_feature_names": linear_names,
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
                "linear_policy_ba": model_ba["linear_policy_history"],
                "richer_policy_ba": model_ba["richer_policy_history"],
                "full_choice_model_ba": model_ba[
                    "richer_policy_history_region_predicted_latent_nested_eps"
                ],
                **increments,
            }
        )

    summary_by_increment = {row["increment"]: row for row in increment_summary}
    enough_replications = len(results) >= args.min_replications
    return {
        "model_summary": model_summary,
        "increment_summary": increment_summary,
        "session_rows": session_rows,
        "richer_policy_supported": bool(
            enough_replications
            and summary_by_increment["richer_policy_after_linear_policy"]["supported"]
        ),
        "choice_neural_residual_after_richer_policy_supported": bool(
            enough_replications
            and summary_by_increment["all_neural_after_richer_policy"]["supported"]
        ),
        "choice_nested_eps_after_richer_policy_supported": bool(
            enough_replications
            and summary_by_increment[
                "nested_eps_after_richer_policy_region_predicted_latent"
            ]["supported"]
        ),
        "richer_choice_policy_gate_passed": bool(
            enough_replications
            and summary_by_increment["richer_policy_after_linear_policy"]["supported"]
            and not summary_by_increment[
                "nested_eps_after_richer_policy_region_predicted_latent"
            ]["supported"]
        ),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx richer choice policy/history gate",
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
        f"- richer policy supported: `{output['richer_policy_supported']}`",
        f"- choice neural residual after richer policy supported: `{output['choice_neural_residual_after_richer_policy_supported']}`",
        f"- choice nested eps after richer policy supported: `{output['choice_nested_eps_after_richer_policy_supported']}`",
        f"- richer choice policy gate passed: `{output['richer_choice_policy_gate_passed']}`",
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
            "| candidate | linear BA | richer BA | full BA | rich-linear | R after rich | Hhat after rich,R | nested eps after rich,R,Hhat | all neural after rich |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in output["session_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"{row['linear_policy_ba']:.6f}",
                    f"{row['richer_policy_ba']:.6f}",
                    f"{row['full_choice_model_ba']:.6f}",
                    f"{row['richer_policy_after_linear_policy']:.6f}",
                    f"{row['region_after_richer_policy']:.6f}",
                    f"{row['predicted_latent_after_richer_policy_region']:.6f}",
                    f"{row['nested_eps_after_richer_policy_region_predicted_latent']:.6f}",
                    f"{row['all_neural_after_richer_policy']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## verdict",
            "",
            "- `richer policy after linear policy` tests whether multi-lag and interaction policy features improve choice beyond the previous linear task/history block.",
            "- `nested eps after rich,R,Hhat` tests whether the action-style innovation subspace still replicates as a choice residual after the richer policy block.",
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
    print("Mouse IBL/OpenAlyx richer choice policy/history gate")
    for row in output["increment_summary"]:
        print(
            f"  {row['increment']}: positive={row['positive_count']}/"
            f"{output['candidate_count']} mean={row['mean_increment']:.6f}"
        )
    print(f"  richer_choice_policy_gate_passed={output['richer_choice_policy_gate_passed']}")
    print(f"Saved: {args.output_json}")
    print(f"Saved: {args.report_md}")


if __name__ == "__main__":
    main()
