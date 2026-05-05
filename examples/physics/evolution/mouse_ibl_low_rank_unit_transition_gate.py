"""Mouse IBL/OpenAlyx low-rank unit transition gate.

The all-unit transition scout showed a split result: lagged units improved
pooled R2 but harmed mean per-feature R2.  That means the temporal signal is
likely concentrated in a few population axes rather than repeated uniformly
across individual unit features.

This gate tests the next equation candidate:

    H_t = A H_{t-lag} + B X_t + C R_t + eps_t

where H is a train-fold PCA state of the unit matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mouse_ibl_channel_region_rescue_gate import CANDIDATES, load_probe
from mouse_ibl_multi_probe_region_gate import DEFAULT_EID, DEFAULT_SESSION_REF, hstack, load_common
from mouse_ibl_region_decision_action_gate import (
    MOVEMENT_WINDOW,
    OPENALYX_URL,
    STIMULUS_WINDOW,
)
from mouse_ibl_task_baseline_comparison_gate import task_history_covariates
from mouse_ibl_temporal_glm_coupling_gate import (
    PRE_MOVEMENT_WINDOW,
    PRE_STIMULUS_WINDOW,
    region_features_for_window,
    unit_features_for_window,
    zscore_block,
)
from mouse_ibl_unit_transition_dynamics_gate import contiguous_folds, finite_mask


RESULT_JSON = Path(__file__).with_name("mouse_ibl_low_rank_unit_transition_results.json")
REPORT_MD = Path(__file__).with_name("mouse_ibl_low_rank_unit_transition_report.md")


def pca_scores(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return train_z @ basis, test_z @ basis, explained_fraction


def ridge_predict(
    blocks: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    block_names: list[str],
    penalties: dict[str, float],
) -> np.ndarray:
    train_parts = []
    test_parts = []
    penalty_values = []
    for name in block_names:
        train_block, test_block = zscore_block(blocks[name][train], blocks[name][test])
        train_parts.append(train_block)
        test_parts.append(test_block)
        penalty_values.extend([float(penalties[name])] * train_block.shape[1])
    x_train = np.column_stack([np.ones(len(train)), hstack(train_parts)])
    x_test = np.column_stack([np.ones(len(test)), hstack(test_parts)])
    y_mean = np.nanmean(y_train, axis=0)
    y_scale = np.nanstd(y_train, axis=0)
    y_scale[y_scale < 1e-9] = 1.0
    y_train_z = (y_train - y_mean) / y_scale
    penalty = np.diag([0.0, *penalty_values])
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train_z
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return x_test @ weights


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    return float(1.0 - sse / sst) if sst > 1e-12 else float("nan")


def crossval_latent_transition(
    x_block: np.ndarray,
    r_block: np.ndarray,
    u_lag: np.ndarray,
    u_target: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    blocks_all = {"X": x_block, "R": r_block, "UL_RAW": u_lag}
    keep = finite_mask(blocks_all, u_target) & valid
    x = x_block[keep]
    r = r_block[keep]
    u_l = u_lag[keep]
    u_0 = u_target[keep]
    folds = contiguous_folds(len(u_0), args.folds)
    models = {
        "task": (["X"], {"X": args.task_penalty}),
        "task_lag": (["X", "HL"], {"X": args.task_penalty, "HL": args.latent_penalty}),
        "task_region": (["X", "R"], {"X": args.task_penalty, "R": args.region_penalty}),
        "task_region_lag": (
            ["X", "R", "HL"],
            {"X": args.task_penalty, "R": args.region_penalty, "HL": args.latent_penalty},
        ),
    }
    preds = {name: [] for name in models}
    truth = []
    explained = []
    for test in folds:
        train = np.setdiff1d(np.arange(len(u_0)), test)
        h0_train, h0_test, h0_exp = pca_scores(u_0[train], u_0[test], args.components)
        hl_train, hl_test, _ = pca_scores(u_l[train], u_l[test], args.components)
        explained.append(float(np.sum(h0_exp)))
        fold_blocks = {
            "X": x,
            "R": r,
            "HL": np.zeros((len(u_0), hl_train.shape[1]), dtype=float),
        }
        fold_blocks["HL"][train] = hl_train
        fold_blocks["HL"][test] = hl_test
        truth.append(h0_test)
        for name, (block_names, penalties) in models.items():
            preds[name].append(
                ridge_predict(
                    fold_blocks,
                    h0_train,
                    h0_test,
                    train,
                    test,
                    block_names,
                    penalties,
                )
            )
    y_true = np.vstack(truth)
    rows = []
    for name in models:
        y_pred = np.vstack(preds[name])
        rows.append({"model": name, "latent_r2": r2(y_true, y_pred)})
    by = {row["model"]: row for row in rows}
    return {
        "trial_count": int(len(u_0)),
        "target_unit_feature_count": int(u_0.shape[1]),
        "components": int(args.components),
        "mean_target_explained_variance": float(np.mean(explained)),
        "rows": rows,
        "nested_comparison": {
            "lag_increment_after_task": float(by["task_lag"]["latent_r2"] - by["task"]["latent_r2"]),
            "lag_increment_after_task_region": float(
                by["task_region_lag"]["latent_r2"] - by["task_region"]["latent_r2"]
            ),
            "task_latent_r2": by["task"]["latent_r2"],
            "task_lag_latent_r2": by["task_lag"]["latent_r2"],
            "task_region_latent_r2": by["task_region"]["latent_r2"],
            "task_region_lag_latent_r2": by["task_region_lag"]["latent_r2"],
        },
    }


def evaluate_session(args: argparse.Namespace) -> dict[str, object]:
    from one.api import ONE

    one = ONE(base_url=OPENALYX_URL, password="international")
    common = load_common(one, args.eid)
    trials = common["trials"]
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
    transitions = [
        {
            "transition": "pre_stimulus_to_stimulus_latent",
            **crossval_latent_transition(
                task_history,
                stim_region,
                pre_stim_unit,
                stim_unit,
                stim_region_valid & stim_unit_valid & pre_stim_valid,
                args,
            ),
        },
        {
            "transition": "pre_movement_to_movement_latent",
            **crossval_latent_transition(
                task_history,
                move_region,
                pre_move_unit,
                move_unit,
                move_region_valid & move_unit_valid & pre_move_valid,
                args,
            ),
        },
    ]
    return {
        "eid": args.eid,
        "session_ref": args.session_ref,
        "collections": args.collections,
        "trial_count": int(len(trials)),
        "transitions": transitions,
    }


def candidate_namespace(candidate: dict[str, object], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        eid=candidate["eid"],
        session_ref=candidate["session_ref"],
        collections=candidate.get("collections", [candidate.get("collection")]),
        folds=args.folds,
        components=args.components,
        task_penalty=args.task_penalty,
        region_penalty=args.region_penalty,
        latent_penalty=args.latent_penalty,
        min_delta=args.min_delta,
        min_replications=args.min_replications,
        min_label_spikes=args.min_label_spikes,
        min_unit_spikes=args.min_unit_spikes,
        max_units_per_probe=args.max_units_per_probe,
    )


def summarize(results: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    names = [row["transition"] for row in results[0]["transitions"]]
    summary = []
    for name in names:
        rows = [t for r in results for t in r["transitions"] if t["transition"] == name]
        nested = [row["nested_comparison"] for row in rows]
        pos_x = [item["lag_increment_after_task"] > args.min_delta for item in nested]
        pos_xr = [item["lag_increment_after_task_region"] > args.min_delta for item in nested]
        mean_x = float(np.mean([item["lag_increment_after_task"] for item in nested]))
        mean_xr = float(np.mean([item["lag_increment_after_task_region"] for item in nested]))
        summary.append(
            {
                "transition": name,
                "candidates": len(rows),
                "positive_after_task": int(sum(pos_x)),
                "positive_after_task_region": int(sum(pos_xr)),
                "mean_lag_increment_after_task": mean_x,
                "mean_lag_increment_after_task_region": mean_xr,
                "supported": bool(
                    sum(pos_xr) >= args.min_replications and mean_xr > args.min_delta
                ),
            }
        )
    return {
        "transition_summary": summary,
        "low_rank_unit_transition_gate_passed": bool(any(row["supported"] for row in summary)),
    }


def make_report(output: dict[str, object]) -> str:
    lines = [
        "# Mouse IBL/OpenAlyx low-rank unit transition gate",
        "",
        "$$",
        "H_t=A H_{t-\\ell}+B X_t+C R_t+\\epsilon_t",
        "$$",
        "",
        "## setup",
        "",
        f"- candidates: {output['candidate_count']}",
        f"- components: {output['components']}",
        f"- folds: {output['folds']}",
        f"- low-rank unit transition gate passed: `{output['low_rank_unit_transition_gate_passed']}`",
        "",
        "## replication",
        "",
        "| transition | candidates | pos after X | pos after X,R0 | mean dR2 after X | mean dR2 after X,R0 | supported |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output["transition_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['transition']}`",
                    str(row["candidates"]),
                    str(row["positive_after_task"]),
                    str(row["positive_after_task_region"]),
                    f"{row['mean_lag_increment_after_task']:.6f}",
                    f"{row['mean_lag_increment_after_task_region']:.6f}",
                    f"`{row['supported']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## candidate summaries", "", "| candidate | trials | pre-stim dR2 XR | pre-move dR2 XR |", "|---|---:|---:|---:|"])
    for result in output["candidate_results"]:
        by = {row["transition"]: row["nested_comparison"] for row in result["transitions"]}
        lines.append(
            f"| `{result['name']}` | {result['trial_count']} | "
            f"{by['pre_stimulus_to_stimulus_latent']['lag_increment_after_task_region']:.6f} | "
            f"{by['pre_movement_to_movement_latent']['lag_increment_after_task_region']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-session", action="store_true")
    parser.add_argument("--eid", default=DEFAULT_EID)
    parser.add_argument("--session-ref", default=DEFAULT_SESSION_REF)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["alf/probe00/pykilosort", "alf/probe01/pykilosort"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--task-penalty", type=float, default=1.0)
    parser.add_argument("--region-penalty", type=float, default=1.0)
    parser.add_argument("--latent-penalty", type=float, default=10.0)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--min-replications", type=int, default=4)
    parser.add_argument("--min-label-spikes", type=int, default=500)
    parser.add_argument("--min-unit-spikes", type=int, default=1000)
    parser.add_argument("--max-units-per-probe", type=int, default=96)
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
        "min_delta": args.min_delta,
        "min_replications": args.min_replications,
        "min_label_spikes": args.min_label_spikes,
        "min_unit_spikes": args.min_unit_spikes,
        "max_units_per_probe": args.max_units_per_probe,
        "candidate_results": results,
    }
    output.update(summarize(results, args))
    RESULT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    REPORT_MD.write_text(make_report(output))
    print("Mouse IBL/OpenAlyx low-rank unit transition gate")
    for row in output["transition_summary"]:
        print(
            f"  {row['transition']}: positive={row['positive_after_task_region']}/"
            f"{row['candidates']} mean dR2 XR={row['mean_lag_increment_after_task_region']:.6f}"
        )
    print(f"  gate_passed={output['low_rank_unit_transition_gate_passed']}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {REPORT_MD}")


if __name__ == "__main__":
    main()
