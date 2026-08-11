"""Post-V8-failure development search; the unopened V8 test is never read."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from reality_stone.clarus import parent_anchored_rollout_bridge as v8
from reality_stone.clarus import reliability_rollout_bridge as rel
from reality_stone.clarus import sparse_causal_bridge as base


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/preregistration/sparse_causal_bridge_v8.json"
CRITICAL = 1.9693105698498752


def rows_for_episode(episode, context, origins, horizon):
    rows = []
    for origin in origins:
        prefix = episode.states[: origin + 1].copy()
        prefix.setflags(write=False)
        truth = episode.states[origin + 1 : origin + horizon + 1]
        sparse = rel._latent_rollout(
            context.parent.sparse_mechanism, context.parent.sparse_ar, prefix, horizon
        )
        persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
        rows.append((truth, sparse, persistence))
    return rows


def sufficient(rows, scales):
    numerator = np.zeros((20, 4))
    denominator = np.zeros((20, 4))
    for truth, sparse, persistence in rows:
        direction = (sparse - persistence) / scales
        target = (truth - persistence) / scales
        numerator += direction * target
        denominator += direction * direction
    return numerator, denominator


def gains_from_sufficient(numerator, denominator, mode, ridge=0.0, anchor=0.0):
    if mode == "scalar":
        return float(np.clip(np.sum(numerator) / np.sum(denominator), 0.0, 1.0))
    if mode == "lead":
        num, den = np.sum(numerator, axis=1), np.sum(denominator, axis=1)
    elif mode == "chart":
        num, den = np.sum(numerator, axis=0), np.sum(denominator, axis=0)
    elif mode == "lead_chart":
        num, den = numerator, denominator
    else:
        raise ValueError(mode)
    scale = float(np.mean(den))
    return np.clip((num + ridge * scale * anchor) / (den + ridge * scale), 0.0, 1.0)


def prediction(sparse, persistence, gain):
    value = np.asarray(gain)
    if value.ndim == 1 and len(value) == 20:
        value = value[:, None]
    return persistence + value * (sparse - persistence)


def mean_path_rmse(rows, scales, gain):
    values = [rel._normalized_rmse(y, prediction(s, p, gain), scales) for y, s, p in rows]
    return float(np.mean(values))


def golden_scalar(rows, scales):
    left, right = 0.0, 1.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = mean_path_rmse(rows, scales, x1), mean_path_rmse(rows, scales, x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = mean_path_rmse(rows, scales, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = mean_path_rmse(rows, scales, x2)
    return (left + right) / 2.0


def paired_lower(baseline, candidate):
    delta = np.asarray(baseline) - np.asarray(candidate)
    mean, sd = float(np.mean(delta)), float(np.std(delta, ddof=1))
    return {"mean": mean, "lower": mean - CRITICAL * sd / math.sqrt(len(delta)),
            "upper": mean + CRITICAL * sd / math.sqrt(len(delta)),
            "win_fraction": float(np.mean(delta > 0.0))}


def main():
    registration, _ = base._load_registration(CONFIG)
    context = v8._build_training_context(CONFIG, registration)
    spec = registration["parent_anchor"]
    origins, horizon = list(map(int, spec["gain_fit_origins"])), int(spec["horizon"])
    role = registration["data_roles"]["observational_train"]
    rows_by_seed = []
    for seed in role["seeds"]:
        episode = base.simulate_episode(int(seed), registration, environment=role["environment"],
                                        steps=int(role["steps_per_seed"]))
        rows_by_seed.append(rows_for_episode(episode, context, origins, horizon))
    all_rows = [row for seed_rows in rows_by_seed for row in seed_rows]
    num, den = sufficient(all_rows, context.parent.scales)
    scalar = gains_from_sufficient(num, den, "scalar")
    metric_scalar = golden_scalar(all_rows, context.parent.scales)

    ridge_grid = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    cv = {}
    for ridge in ridge_grid:
        heldout = []
        for index, seed_rows in enumerate(rows_by_seed):
            fit_rows = [row for other, values in enumerate(rows_by_seed) if other != index for row in values]
            n, d = sufficient(fit_rows, context.parent.scales)
            gain = gains_from_sufficient(n, d, "lead", ridge, scalar)
            heldout.extend(rel._normalized_rmse(y, prediction(s, p, gain), context.parent.scales)
                           for y, s, p in seed_rows)
        cv[ridge] = float(np.mean(heldout))
    selected_ridge = min(ridge_grid, key=cv.get)
    candidates = {
        "registered_scalar_sse": scalar,
        "metric_aligned_scalar": metric_scalar,
        "leadwise_unregularized": gains_from_sufficient(num, den, "lead"),
        "leadwise_cv_regularized": gains_from_sufficient(num, den, "lead", selected_ridge, scalar),
        "chartwise_unregularized": gains_from_sufficient(num, den, "chart"),
        "lead_chart_unregularized": gains_from_sufficient(num, den, "lead_chart"),
    }

    evaluation = {name: [] for name in candidates}
    baselines = {name: [] for name in ("v5", "persistence", "zero", "dense", "v7")}
    validation = registration["data_roles"]["validation"]
    for seed in validation["seeds"]:
        episode = base.simulate_episode(int(seed), registration, environment=validation["environment"],
                                        steps=int(validation["steps_per_seed"]))
        prefix = episode.states[:81].copy()
        prefix.setflags(write=False)
        truth = episode.states[81:101]
        frozen = v8.predict_from_prefix(prefix, context, registration).models
        sparse, persistence = frozen["v5_sparse_parent"], frozen["persistence"]
        for name, gain in candidates.items():
            evaluation[name].append(rel._normalized_rmse(
                truth, prediction(sparse, persistence, gain), context.parent.scales))
        mapping = {"v5": "v5_sparse_parent", "persistence": "persistence",
                   "zero": "zero_bridge_shrinkage", "dense": "symmetric_dense_shrinkage",
                   "v7": "frozen_v7_consensus"}
        for short, full in mapping.items():
            baselines[short].append(rel._normalized_rmse(truth, frozen[full], context.parent.scales))

    report = {
        "status": "post_v8_failure_development_only",
        "v8_test_opened": False,
        "training": {"windows": len(all_rows), "metric_aligned_scalar": metric_scalar,
                     "leadwise_cv_ridge_grid": cv, "selected_ridge": selected_ridge},
        "candidates": {},
    }
    for name, errors in evaluation.items():
        report["candidates"][name] = {
            "gain": np.asarray(candidates[name]).tolist(),
            "mean_h20_rmse": float(np.mean(errors)),
            "vs_v5": paired_lower(baselines["v5"], errors),
            "vs_persistence": paired_lower(baselines["persistence"], errors),
            "vs_zero": paired_lower(baselines["zero"], errors),
            "vs_v7": paired_lower(baselines["v7"], errors),
        }
        log_dense = np.log(np.asarray(errors) / np.asarray(baselines["dense"]))
        report["candidates"][name]["dense_log_ratio_upper"] = float(
            np.mean(log_dense) + CRITICAL * np.std(log_dense, ddof=1) / math.sqrt(len(log_dense)))
    output = Path(__file__).with_name("post_failure_route_search.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(output)
    print(json.dumps({name: {k: value[k] for k in ("mean_h20_rmse", "vs_v5", "vs_zero",
                                                    "dense_log_ratio_upper")}
                      for name, value in report["candidates"].items()}, indent=2))


if __name__ == "__main__":
    main()
