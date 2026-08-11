"""Prefix-only local reliability search after the preserved V8 failure."""

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


def rollout_row(states, origin, context, horizon=20):
    prefix = states[: origin + 1].copy()
    prefix.setflags(write=False)
    sparse = rel._latent_rollout(
        context.parent.sparse_mechanism, context.parent.sparse_ar, prefix, horizon
    )
    persistence = np.repeat(prefix[-1][None, :], horizon, axis=0)
    truth = states[origin + 1 : origin + horizon + 1]
    direction = (sparse - persistence) / context.parent.scales
    target = (truth - persistence) / context.parent.scales
    return truth, sparse, persistence, float(np.sum(direction * target)), float(np.sum(direction ** 2))


def fit_global(rows):
    return float(np.clip(sum(row[3] for row in rows) / sum(row[4] for row in rows), 0.0, 1.0))


def local_gain(states, target_origin, context):
    origins = [target_origin - offset for offset in (60, 50, 40, 30, 20)]
    rows = [rollout_row(states, origin, context) for origin in origins]
    return fit_global(rows), origins


def choose_gain(local, prior, strength, one_sided):
    blended = (5.0 * local + strength * prior) / (5.0 + strength)
    return max(prior, blended) if one_sided else blended


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
    rows_by_seed = [[rollout_row(ep.states, origin, context)
                    for origin in registration["parent_anchor"]["gain_fit_origins"]]
                    for ep in episodes]
    strengths = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    cv = {}
    for one_sided in (False, True):
        for strength in strengths:
            errors = []
            for heldout, episode in enumerate(episodes):
                other_rows = [row for index, rows in enumerate(rows_by_seed)
                              if index != heldout for row in rows]
                prior = fit_global(other_rows)
                for origin in registration["parent_anchor"]["gain_fit_origins"]:
                    local, _ = local_gain(episode.states, int(origin), context)
                    gain = choose_gain(local, prior, strength, one_sided)
                    truth, sparse, persistence, _, _ = rollout_row(
                        episode.states, int(origin), context
                    )
                    prediction = persistence + gain * (sparse - persistence)
                    errors.append(rel._normalized_rmse(
                        truth, prediction, context.parent.scales))
            cv[(one_sided, strength)] = float(np.mean(errors))
    selected = min(cv, key=cv.get)
    prior = fit_global([row for rows in rows_by_seed for row in rows])

    candidate, fixed, v5, persistence_errors, zero, dense, v7_errors, gains = ([] for _ in range(8))
    validation = registration["data_roles"]["validation"]
    for seed in validation["seeds"]:
        episode = base.simulate_episode(int(seed), registration, environment=validation["environment"],
                                        steps=int(validation["steps_per_seed"]))
        prefix = episode.states[:81].copy()
        prefix.setflags(write=False)
        truth = episode.states[81:101]
        paths = v8.predict_from_prefix(prefix, context, registration).models
        local, origins = local_gain(episode.states[:81], 80, context)
        gain = choose_gain(local, prior, selected[1], selected[0])
        gains.append(gain)
        pred = paths["persistence"] + gain * (paths["v5_sparse_parent"] - paths["persistence"])
        metric = lambda value: rel._normalized_rmse(truth, value, context.parent.scales)
        candidate.append(metric(pred))
        fixed.append(metric(paths["parent_anchored_sparse"]))
        v5.append(metric(paths["v5_sparse_parent"]))
        persistence_errors.append(metric(paths["persistence"]))
        zero.append(metric(paths["zero_bridge_shrinkage"]))
        dense.append(metric(paths["symmetric_dense_shrinkage"]))
        v7_errors.append(metric(paths["frozen_v7_consensus"]))
    logs = np.log(np.asarray(candidate) / np.asarray(dense))
    report = {
        "status": "post_v8_failure_development_only",
        "v8_test_opened": False,
        "algorithm": "prefix-local gain from five completed H20 backtests, shrunk to training-only global gain",
        "local_origins_at_evaluation": origins,
        "selected_by_leave_one_training_episode_out_cv": {
            "one_sided": selected[0], "prior_strength_in_window_units": selected[1],
            "cv_mean_rmse": cv[selected],
            "grid": {f"one_sided={key[0]},strength={key[1]}": value for key, value in cv.items()},
        },
        "global_prior_gain": prior,
        "evaluation_gain": {"mean": float(np.mean(gains)), "sd": float(np.std(gains, ddof=1)),
                            "min": float(np.min(gains)), "max": float(np.max(gains)),
                            "fraction_equal_prior": float(np.mean(np.asarray(gains) == prior))},
        "mean_h20_rmse": float(np.mean(candidate)),
        "fixed_v8_mean_h20_rmse": float(np.mean(fixed)),
        "comparisons": {"vs_fixed_v8": paired_lower(fixed, candidate),
                        "vs_v5": paired_lower(v5, candidate),
                        "vs_persistence": paired_lower(persistence_errors, candidate),
                        "vs_zero": paired_lower(zero, candidate),
                        "vs_v7": paired_lower(v7_errors, candidate)},
        "dense_log_ratio_upper": float(np.mean(logs) + CRITICAL * np.std(logs, ddof=1) /
                                       math.sqrt(len(logs))),
    }
    output = Path(__file__).with_name("prefix_adaptive_route_search.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
