"""Development pilot for parent-anchored sparse residual shrinkage.

All coefficients are fitted from inherited training data or from the observed
prefix.  The disclosed V7 validation split is used only to compare development
routes.  The V7 test split is never read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from reality_stone.clarus import reliability_rollout_bridge as rel
from reality_stone.clarus import sparse_causal_bridge as base


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/preregistration/sparse_causal_bridge_v7.json"


def _fit_gain(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], scales: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for truth, sparse, persistence in rows:
        direction = (sparse - persistence) / scales
        target = (truth - persistence) / scales
        numerator += float(np.sum(direction * target))
        denominator += float(np.sum(direction * direction))
    if denominator <= 0 or not math.isfinite(denominator):
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _calibration_rows(
    states: np.ndarray,
    origins: list[int],
    context: rel.TrainingContext,
    horizon: int,
    *,
    dense: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = []
    for origin in origins:
        prefix = np.asarray(states[: origin + 1], dtype=float).copy()
        prefix.setflags(write=False)
        truth = np.asarray(states[origin + 1 : origin + horizon + 1], dtype=float)
        if len(truth) != horizon:
            raise ValueError("calibration origin lacks a full H20 target")
        mechanism = context.dense_probe_mechanism if dense else context.sparse_mechanism
        autoregression = context.dense_probe_ar if dense else context.sparse_ar
        learned = rel._latent_rollout(
            mechanism, autoregression, prefix, horizon
        )
        persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
        rows.append((truth, learned, persistence))
    return rows


def _paired_lower(baseline: list[float], candidate: list[float], critical: float) -> float:
    difference = np.asarray(baseline, dtype=float) - np.asarray(candidate, dtype=float)
    return float(
        np.mean(difference)
        - critical * np.std(difference, ddof=1) / math.sqrt(len(difference))
    )


def main() -> None:
    registration, _ = base._load_registration(CONFIG)
    context = rel._build_training_context(CONFIG, registration)
    horizon = int(registration["closure"]["horizon"])
    origin = int(registration["closure"]["origin"])
    critical = float(registration["closure"]["critical_value_n96_df95"])

    global_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    dense_global_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    rows_by_train_seed: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    per_train_seed_gains = []
    train_role = registration["data_roles"]["observational_train"]
    train_origins = list(range(80, int(train_role["steps_per_seed"]) - horizon + 1, 20))
    for raw_seed in train_role["seeds"]:
        episode = base.simulate_episode(
            int(raw_seed),
            registration,
            environment=train_role["environment"],
            steps=int(train_role["steps_per_seed"]),
        )
        seed_rows = _calibration_rows(episode.states, train_origins, context, horizon)
        rows_by_train_seed.append(seed_rows)
        global_rows.extend(seed_rows)
        dense_global_rows.extend(
            _calibration_rows(
                episode.states, train_origins, context, horizon, dense=True
            )
        )
        per_train_seed_gains.append(_fit_gain(seed_rows, context.scales))
    global_gain = _fit_gain(global_rows, context.scales)
    dense_global_gain = _fit_gain(dense_global_rows, context.scales)
    leave_one_seed_out_gains = [
        _fit_gain(
            [
                row
                for other_index, seed_rows in enumerate(rows_by_train_seed)
                if other_index != held_out_index
                for row in seed_rows
            ],
            context.scales,
        )
        for held_out_index in range(len(rows_by_train_seed))
    ]

    selector_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    selector_role = registration["data_roles"]["observational_selector_holdout"]
    selector_origins = list(
        range(80, int(selector_role["steps_per_seed"]) - horizon + 1, 20)
    )
    for raw_seed in selector_role["seeds"]:
        episode = base.simulate_episode(
            int(raw_seed),
            registration,
            environment=selector_role["environment"],
            steps=int(selector_role["steps_per_seed"]),
        )
        selector_rows.extend(
            _calibration_rows(episode.states, selector_origins, context, horizon)
        )
    selector_gain = _fit_gain(selector_rows, context.scales)

    methods = {
        "sparse_parent": [],
        "persistence": [],
        "train_global_gain": [],
        "symmetric_dense_train_global_gain": [],
        "selector_holdout_gain": [],
        "prefix_single_origin_gain": [],
        "prefix_multi_origin_gain": [],
    }
    gains = {"prefix_single_origin_gain": [], "prefix_multi_origin_gain": []}
    role = registration["data_roles"]["validation"]
    for raw_seed in role["seeds"]:
        episode = base.simulate_episode(
            int(raw_seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        prefix = episode.states[: origin + 1].copy()
        prefix.setflags(write=False)
        truth = episode.states[origin + 1 : origin + horizon + 1]
        sparse = rel._latent_rollout(
            context.sparse_mechanism, context.sparse_ar, prefix, horizon
        )
        dense = rel._latent_rollout(
            context.dense_probe_mechanism, context.dense_probe_ar, prefix, horizon
        )
        persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)

        single_gain = _fit_gain(
            _calibration_rows(prefix, [60], context, horizon), context.scales
        )
        multi_gain = _fit_gain(
            _calibration_rows(prefix, [20, 30, 40, 50, 60], context, horizon),
            context.scales,
        )
        gains["prefix_single_origin_gain"].append(single_gain)
        gains["prefix_multi_origin_gain"].append(multi_gain)
        predictions = {
            "sparse_parent": sparse,
            "persistence": persistence,
            "train_global_gain": persistence + global_gain * (sparse - persistence),
            "symmetric_dense_train_global_gain": persistence
            + dense_global_gain * (dense - persistence),
            "selector_holdout_gain": persistence + selector_gain * (sparse - persistence),
            "prefix_single_origin_gain": persistence + single_gain * (sparse - persistence),
            "prefix_multi_origin_gain": persistence + multi_gain * (sparse - persistence),
        }
        for name, prediction in predictions.items():
            methods[name].append(rel._normalized_rmse(truth, prediction, context.scales))

    parent = methods["sparse_parent"]
    persistence = methods["persistence"]
    report = {
        "status": "development_only_v7_validation_pilot",
        "algorithm": "prediction = persistence + gain * (sparse_parent - persistence)",
        "global_training_origins": train_origins,
        "global_training_rows": len(global_rows),
        "train_global_gain": global_gain,
        "dense_train_global_gain": dense_global_gain,
        "selector_holdout_origins": selector_origins,
        "selector_holdout_rows": len(selector_rows),
        "selector_holdout_gain": selector_gain,
        "per_train_seed_sparse_gain": {
            "values": per_train_seed_gains,
            "mean": float(np.mean(per_train_seed_gains)),
            "sd": float(np.std(per_train_seed_gains, ddof=1)),
            "minimum": float(np.min(per_train_seed_gains)),
            "maximum": float(np.max(per_train_seed_gains)),
        },
        "leave_one_train_seed_out_sparse_gain": {
            "values": leave_one_seed_out_gains,
            "mean": float(np.mean(leave_one_seed_out_gains)),
            "sd": float(np.std(leave_one_seed_out_gains, ddof=1)),
            "minimum": float(np.min(leave_one_seed_out_gains)),
            "maximum": float(np.max(leave_one_seed_out_gains)),
        },
        "prefix_gain_summary": {
            name: {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "fraction_zero": float(np.mean(np.asarray(values) == 0.0)),
                "fraction_one": float(np.mean(np.asarray(values) == 1.0)),
            }
            for name, values in gains.items()
        },
        "methods": {},
    }
    for name, values in methods.items():
        report["methods"][name] = {
            "mean_h20_normalized_rmse": float(np.mean(values)),
            "paired_ci95_lower_improvement_vs_parent": _paired_lower(
                parent, values, critical
            ),
            "paired_ci95_lower_improvement_vs_persistence": _paired_lower(
                persistence, values, critical
            ),
            "seed_win_fraction_vs_parent": float(
                np.mean(np.asarray(values) < np.asarray(parent))
            ),
            "seed_win_fraction_vs_persistence": float(
                np.mean(np.asarray(values) < np.asarray(persistence))
            ),
        }
    sparse_candidate = np.asarray(methods["train_global_gain"], dtype=float)
    dense_candidate = np.asarray(
        methods["symmetric_dense_train_global_gain"], dtype=float
    )
    log_ratio = np.log(sparse_candidate / dense_candidate)
    report["symmetric_sparse_vs_dense"] = {
        "geometric_mean_error_ratio": float(np.exp(np.mean(log_ratio))),
        "paired_log_ratio_ci95_lower": float(
            np.mean(log_ratio)
            - critical * np.std(log_ratio, ddof=1) / math.sqrt(len(log_ratio))
        ),
        "paired_log_ratio_ci95_upper": float(
            np.mean(log_ratio)
            + critical * np.std(log_ratio, ddof=1) / math.sqrt(len(log_ratio))
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
