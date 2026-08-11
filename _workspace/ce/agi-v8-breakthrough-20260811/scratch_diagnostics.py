"""Development-only diagnostics over the disclosed V7 validation split.

This script must never read the locked V7 test split.  It diagnoses why the
registered prefix controller failed and reports target-aware ceilings only;
none of its outputs are confirmatory evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reality_stone.clarus import reliability_rollout_bridge as rel
from reality_stone.clarus import sparse_causal_bridge as base


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/preregistration/sparse_causal_bridge_v7.json"


def _corr(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if np.std(av) == 0 or np.std(bv) == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> None:
    registration, _ = base._load_registration(CONFIG)
    if registration["experiment"] != "sparse_causal_bridge_v7":
        raise RuntimeError("V7 registration required")
    context = rel._build_training_context(CONFIG, registration)
    role = registration["data_roles"]["validation"]
    origin = int(registration["closure"]["origin"])
    pseudo_origin = int(registration["closure"]["pseudo_origin"])
    horizon = int(registration["closure"]["horizon"])

    names = ("sparse", "adaptive", "persistence")
    inner_rmse = {name: [] for name in names}
    outer_rmse = {name: [] for name in names}
    outer_predictions = {name: [] for name in names}
    truths: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    radii = {"sparse": [], "dense_probe": [], "adaptive": []}

    for raw_seed in role["seeds"]:
        episode = base.simulate_episode(
            int(raw_seed),
            registration,
            environment=role["environment"],
            steps=int(role["steps_per_seed"]),
        )
        prefix = episode.states[: origin + 1].copy()
        prefix.setflags(write=False)
        inner = prefix[: pseudo_origin + 1]
        inner_truth = prefix[pseudo_origin + 1 : origin + 1]
        truth = episode.states[origin + 1 : origin + horizon + 1]

        sparse_inner = rel._latent_rollout(
            context.sparse_mechanism, context.sparse_ar, inner, horizon
        )
        adaptive_inner, _ = rel._adaptive_rollout(inner, context, horizon)
        persistence_inner = np.repeat(inner[-1][None, :], horizon, axis=0)
        inner_predictions = (sparse_inner, adaptive_inner, persistence_inner)
        for name, prediction in zip(names, inner_predictions):
            inner_rmse[name].append(
                rel._normalized_rmse(inner_truth, prediction, context.scales)
            )

        sparse_outer = rel._latent_rollout(
            context.sparse_mechanism, context.sparse_ar, prefix, horizon
        )
        dense_outer = rel._latent_rollout(
            context.dense_probe_mechanism, context.dense_probe_ar, prefix, horizon
        )
        adaptive_outer, adaptive_model = rel._adaptive_rollout(prefix, context, horizon)
        persistence_outer = np.repeat(prefix[-1][None, :], horizon, axis=0)
        outer = (sparse_outer, adaptive_outer, persistence_outer)
        for name, prediction in zip(names, outer):
            outer_predictions[name].append(prediction)
            outer_rmse[name].append(rel._normalized_rmse(truth, prediction, context.scales))
        truths.append(truth)

        weights.append(
            rel._inverse_root_weights(
                [value[-1] ** 2 for value in inner_rmse.values()],
                float(registration["closure"]["weight_epsilon_dimensionless"]),
            )
        )
        radii["sparse"].append(
            rel.free._maximum_jacobian_radius(context.sparse_mechanism, sparse_outer)
        )
        radii["dense_probe"].append(
            rel.free._maximum_jacobian_radius(context.dense_probe_mechanism, dense_outer)
        )
        radii["adaptive"].append(
            rel.free._maximum_jacobian_radius(adaptive_model, adaptive_outer)
        )

    inner_advantage = np.asarray(inner_rmse["adaptive"]) - np.asarray(inner_rmse["sparse"])
    outer_advantage = np.asarray(outer_rmse["adaptive"]) - np.asarray(outer_rmse["sparse"])
    inner_persistence_advantage = np.asarray(inner_rmse["persistence"]) - np.asarray(
        inner_rmse["sparse"]
    )
    outer_persistence_advantage = np.asarray(outer_rmse["persistence"]) - np.asarray(
        outer_rmse["sparse"]
    )

    # Target-aware fixed-weight grid: diagnostic ceiling, never a candidate fit.
    best = {"mean_rmse": float("inf"), "weights": None}
    best_anchor = {"mean_rmse": float("inf"), "weights": None}
    for sparse_i in range(51):
        for adaptive_i in range(51 - sparse_i):
            persistence_i = 50 - sparse_i - adaptive_i
            w = np.asarray([sparse_i, adaptive_i, persistence_i], dtype=float) / 50.0
            seed_errors = []
            for index, truth in enumerate(truths):
                prediction = (
                    w[0] * outer_predictions["sparse"][index]
                    + w[1] * outer_predictions["adaptive"][index]
                    + w[2] * outer_predictions["persistence"][index]
                )
                seed_errors.append(rel._normalized_rmse(truth, prediction, context.scales))
            mean_error = float(np.mean(seed_errors))
            if mean_error < best["mean_rmse"]:
                best = {"mean_rmse": mean_error, "weights": w.tolist()}
            if w[0] >= 0.5 and mean_error < best_anchor["mean_rmse"]:
                best_anchor = {"mean_rmse": mean_error, "weights": w.tolist()}

    oracle_errors = np.min(np.column_stack([outer_rmse[name] for name in names]), axis=1)
    report = {
        "status": "development_only_v7_validation_diagnostic",
        "seed_count": len(role["seeds"]),
        "mean_inner_rmse": {name: float(np.mean(values)) for name, values in inner_rmse.items()},
        "mean_outer_rmse": {name: float(np.mean(values)) for name, values in outer_rmse.items()},
        "mean_registered_weights": np.mean(np.asarray(weights), axis=0).tolist(),
        "inner_to_outer_error_correlation": {
            name: _corr(inner_rmse[name], outer_rmse[name]) for name in names
        },
        "inner_to_outer_relative_advantage_correlation": {
            "sparse_vs_adaptive": _corr(inner_advantage.tolist(), outer_advantage.tolist()),
            "sparse_vs_persistence": _corr(
                inner_persistence_advantage.tolist(), outer_persistence_advantage.tolist()
            ),
        },
        "winner_agreement": {
            "sparse_vs_adaptive": float(np.mean((inner_advantage > 0) == (outer_advantage > 0))),
            "sparse_vs_persistence": float(
                np.mean((inner_persistence_advantage > 0) == (outer_persistence_advantage > 0))
            ),
        },
        "pathwise_radii": {
            name: {
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "fraction_above_0_98": float(np.mean(np.asarray(values) > 0.98)),
            }
            for name, values in radii.items()
        },
        "target_aware_diagnostic_ceiling": {
            "best_fixed_grid_step_0_02": best,
            "best_fixed_grid_sparse_weight_at_least_0_5": best_anchor,
            "per_seed_oracle_expert_mean_rmse": float(np.mean(oracle_errors)),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

