"""Training-only temporal multi-origin sparse ensemble after V8 failure."""

from __future__ import annotations

import itertools
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
OFFSETS = (0, 10, 20)


def components(states, target_origin, context):
    result = []
    for offset in OFFSETS:
        origin = target_origin - offset
        prefix = states[: origin + 1].copy()
        prefix.setflags(write=False)
        full = rel._latent_rollout(
            context.parent.sparse_mechanism, context.parent.sparse_ar,
            prefix, 20 + offset
        )
        result.append(full[offset: offset + 20])
    result.append(np.repeat(states[target_origin][None, :], 20, axis=0))
    return result


def fit_simplex(rows, scales):
    xs, ys = [], []
    for truth, paths in rows:
        xs.append(np.stack([path / scales for path in paths], axis=-1).reshape(-1, len(paths)))
        ys.append((truth / scales).reshape(-1))
    x, y = np.concatenate(xs), np.concatenate(ys)
    best = None
    size = x.shape[1]
    for count in range(1, size + 1):
        for subset in itertools.combinations(range(size), count):
            indices = list(subset)
            design = x[:, indices]
            gram = design.T @ design
            system = np.block([[gram, np.ones((count, 1))],
                               [np.ones((1, count)), np.zeros((1, 1))]])
            target = np.concatenate([design.T @ y, [1.0]])
            solution = np.linalg.lstsq(system, target, rcond=None)[0][:-1]
            if np.min(solution) < -1e-10:
                continue
            weights = np.zeros(size)
            weights[indices] = np.maximum(solution, 0.0)
            weights /= np.sum(weights)
            loss = float(np.sum((x @ weights - y) ** 2))
            if best is None or loss < best[0]:
                best = (loss, weights)
    if best is None:
        raise RuntimeError("simplex fit failed")
    return best[1]


def combine(paths, weights):
    return sum(float(weight) * path for weight, path in zip(weights, paths))


def paired_lower(baseline, candidate):
    delta = np.asarray(baseline) - np.asarray(candidate)
    mean, sd = float(np.mean(delta)), float(np.std(delta, ddof=1))
    return {"mean": mean, "lower": mean - CRITICAL * sd / math.sqrt(len(delta)),
            "upper": mean + CRITICAL * sd / math.sqrt(len(delta)),
            "wins": float(np.mean(delta > 0.0))}


def main():
    registration, _ = base._load_registration(CONFIG)
    context = v8._build_training_context(CONFIG, registration)
    role = registration["data_roles"]["observational_train"]
    episodes = [base.simulate_episode(int(seed), registration, environment=role["environment"],
                                      steps=int(role["steps_per_seed"])) for seed in role["seeds"]]
    rows_by_seed = []
    for episode in episodes:
        rows_by_seed.append([
            (episode.states[int(origin) + 1:int(origin) + 21],
             components(episode.states, int(origin), context))
            for origin in registration["parent_anchor"]["gain_fit_origins"]
        ])
    weights = fit_simplex([row for rows in rows_by_seed for row in rows], context.parent.scales)
    loo_weights, loo_errors = [], []
    for heldout, heldout_rows in enumerate(rows_by_seed):
        fit_rows = [row for index, rows in enumerate(rows_by_seed) if index != heldout for row in rows]
        current = fit_simplex(fit_rows, context.parent.scales)
        loo_weights.append(current.tolist())
        loo_errors.extend(rel._normalized_rmse(y, combine(paths, current), context.parent.scales)
                          for y, paths in heldout_rows)

    errors = {name: [] for name in ("candidate", "v5", "persistence", "zero", "dense", "v7")}
    role = registration["data_roles"]["validation"]
    for seed in role["seeds"]:
        episode = base.simulate_episode(int(seed), registration, environment=role["environment"],
                                        steps=int(role["steps_per_seed"]))
        prefix = episode.states[:81].copy()
        prefix.setflags(write=False)
        truth = episode.states[81:101]
        paths = v8.predict_from_prefix(prefix, context, registration).models
        candidate = combine(components(episode.states[:81], 80, context), weights)
        metric = lambda path: rel._normalized_rmse(truth, path, context.parent.scales)
        errors["candidate"].append(metric(candidate))
        for short, full in (("v5", "v5_sparse_parent"), ("persistence", "persistence"),
                            ("zero", "zero_bridge_shrinkage"),
                            ("dense", "symmetric_dense_shrinkage"),
                            ("v7", "frozen_v7_consensus")):
            errors[short].append(metric(paths[full]))
    candidate = errors["candidate"]
    logs = np.log(np.asarray(candidate) / np.asarray(errors["dense"]))
    report = {
        "status": "post_v8_failure_development_only", "v8_test_opened": False,
        "algorithm": "simplex ensemble of sparse rollouts launched at offsets 0, 10, 20 plus persistence",
        "component_order": ["sparse_origin_80", "sparse_origin_70_tail", "sparse_origin_60_tail", "persistence_80"],
        "training_weights": weights.tolist(), "leave_one_episode_out_weights": loo_weights,
        "leave_one_episode_out_training_mean_rmse": float(np.mean(loo_errors)),
        "mean_h20_rmse": float(np.mean(candidate)),
        "comparisons": {name: paired_lower(errors[name], candidate)
                        for name in ("v5", "persistence", "zero", "v7")},
        "dense_log_ratio_upper": float(np.mean(logs) + CRITICAL * np.std(logs, ddof=1) /
                                       math.sqrt(len(logs))),
    }
    output = Path(__file__).with_name("multi_origin_route_search.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
