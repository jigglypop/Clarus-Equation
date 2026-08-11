"""Eight-fold training-only ACBSM screen; no development seed is simulated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reality_stone.clarus import integrated_latent_state_bridge as acbsm
from reality_stone.clarus import latent_causal_bridge as latent
from reality_stone.clarus import parent_anchored_rollout_bridge as v8
from reality_stone.clarus import reliability_rollout_bridge as rel
from reality_stone.clarus import sparse_causal_bridge as base


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/preregistration/sparse_causal_bridge_v8.json"


def main() -> None:
    registration, _ = base._load_registration(CONFIG)
    parent_context = v8._build_training_context(CONFIG, registration).parent
    role = registration["data_roles"]["observational_train"]
    episodes = [
        base.simulate_episode(int(seed), registration, environment=role["environment"],
                              steps=int(role["steps_per_seed"]))
        for seed in role["seeds"]
    ]
    fold_rows = []
    for held_index, held in enumerate(episodes):
        fit = [episode for index, episode in enumerate(episodes) if index != held_index]
        sparse2 = acbsm.fit_stable_rank_two_dynamics(fit, parent_context.sparse_mechanism)
        sparse1 = acbsm.fit_residual_dynamics(fit, parent_context.sparse_mechanism, rank=1)
        dense2 = acbsm.fit_stable_rank_two_dynamics(fit, parent_context.dense_probe_mechanism)
        legacy_ar = latent.fit_pooled_residual_autoregression(fit, parent_context.sparse_mechanism)
        errors = {name: [] for name in ("core", "rank1", "legacy", "dense")}
        radii = []
        for origin in registration["parent_anchor"]["gain_fit_origins"]:
            origin = int(origin)
            prefix = held.states[origin - 80 : origin + 1].copy()
            prefix.setflags(write=False)
            truth = held.states[origin + 1 : origin + 21]
            core, _, _ = acbsm._one_acbsm_prediction(
                prefix, parent_context.sparse_mechanism, sparse2, 20
            )
            rank1, _, _ = acbsm._one_acbsm_prediction(
                prefix, parent_context.sparse_mechanism, sparse1, 20
            )
            dense, _, _ = acbsm._one_acbsm_prediction(
                prefix, parent_context.dense_probe_mechanism, dense2, 20
            )
            legacy = rel._latent_rollout(
                parent_context.sparse_mechanism, legacy_ar, prefix, 20
            )
            for name, prediction in (("core", core), ("rank1", rank1),
                                     ("legacy", legacy), ("dense", dense)):
                errors[name].append(rel._normalized_rmse(
                    truth, prediction, parent_context.scales))
            radii.append(max(
                rel.free._maximum_jacobian_radius(parent_context.sparse_mechanism, core),
                float(np.max(np.abs(sparse2.transition))),
            ))
        means = {name: float(np.mean(values)) for name, values in errors.items()}
        fold_rows.append({
            "held_seed": int(role["seeds"][held_index]),
            "means": means,
            "sparse_rank2_transition": sparse2.transition.tolist(),
            "sparse_rank2_fast_active": sparse2.fast_active,
            "sparse_rank2_fast_signal_fraction": sparse2.fast_signal_fraction,
            "maximum_radius": max(radii),
        })
    arrays = {
        name: np.asarray([row["means"][name] for row in fold_rows])
        for name in ("core", "rank1", "legacy", "dense")
    }
    legacy_mean = float(np.mean(arrays["legacy"]))
    core_improvement = (legacy_mean - float(np.mean(arrays["core"]))) / legacy_mean
    rank1_improvement = (legacy_mean - float(np.mean(arrays["rank1"]))) / legacy_mean
    rank2_over_rank1 = (
        float(np.mean(arrays["rank1"])) - float(np.mean(arrays["core"]))
    ) / float(np.mean(arrays["rank1"]))
    dense_ratio = float(np.mean(arrays["core"]) / np.mean(arrays["dense"]))
    points = {
        "core_vs_legacy": 40.0 * float(np.clip(core_improvement / 0.05, 0.0, 1.0)),
        "rank1_kalman_vs_legacy": 20.0 * float(np.clip(rank1_improvement / 0.03, 0.0, 1.0)),
        "rank2_vs_rank1": 15.0 * float(np.clip(rank2_over_rank1 / 0.02, 0.0, 1.0)),
        "positive_fold_fraction": 10.0 * float(np.mean(arrays["core"] < arrays["legacy"])),
        "dense_noninferiority": 10.0 * float(np.clip((1.05 - dense_ratio) / 0.03, 0.0, 1.0)),
        "identification": 5.0 if sum(row["sparse_rank2_fast_active"] for row in fold_rows) >= 6 else 0.0,
    }
    hard_ok = (
        all(np.isfinite(list(arrays.values())).flat)
        and max(row["maximum_radius"] for row in fold_rows) <= 0.98
    )
    score = float(sum(points.values())) if hard_ok else 0.0
    report = {
        "status": "training_only_leave_one_episode_out_screen",
        "development_seeds_simulated": False,
        "score": score,
        "classification": "ADVANCE" if score >= 75.0 else ("HOLD" if score >= 65.0 else "STOP"),
        "points": points,
        "relative_improvements": {
            "core_vs_legacy": core_improvement,
            "rank1_vs_legacy": rank1_improvement,
            "rank2_vs_rank1": rank2_over_rank1,
            "core_to_dense_ratio": dense_ratio,
        },
        "folds": fold_rows,
    }
    output = Path(__file__).with_name("training_screen.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("score", "classification", "points",
                                                   "relative_improvements")}, indent=2))


if __name__ == "__main__":
    main()
