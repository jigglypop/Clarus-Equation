"""Fresh development pilot for training-only parent-anchored shrinkage.

This runner uses fresh OOD seeds 79100..79355.  It never reads any historical
locked test split and never writes a canonical validation artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from reality_stone.clarus import latent_causal_bridge as latent
from reality_stone.clarus import reliability_rollout_bridge as rel
from reality_stone.clarus import sparse_causal_bridge as base


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/preregistration/sparse_causal_bridge_v7.json"
PILOT_SEEDS = tuple(range(79100, 79356))
TRAIN_ORIGINS = tuple(range(80, 501, 20))
HORIZON = 20
CRITICAL_T_DF255 = 1.9693105698498752


def _fit_gain(
    episodes: list[base.Episode],
    mechanism: base.BridgeModel,
    autoregression: float,
    scales: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for episode in episodes:
        for origin in TRAIN_ORIGINS:
            prefix = episode.states[: origin + 1].copy()
            prefix.setflags(write=False)
            truth = episode.states[origin + 1 : origin + HORIZON + 1]
            learned = rel._latent_rollout(mechanism, autoregression, prefix, HORIZON)
            persistence = np.repeat(prefix[-1][None, :], HORIZON, axis=0)
            direction = (learned - persistence) / scales
            target = (truth - persistence) / scales
            numerator += float(np.sum(direction * target))
            denominator += float(np.sum(direction * direction))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise FloatingPointError("gain denominator must be positive and finite")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _paired_lower(baseline: list[float], candidate: list[float]) -> dict[str, float]:
    difference = np.asarray(baseline, dtype=float) - np.asarray(candidate, dtype=float)
    mean = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    return {
        "mean_improvement": mean,
        "sample_sd": sd,
        "ci95_lower": float(mean - CRITICAL_T_DF255 * sd / math.sqrt(len(difference))),
        "seed_win_fraction": float(np.mean(difference > 0.0)),
    }


def _paired_log_ratio(candidate: list[float], control: list[float]) -> dict[str, float]:
    values = np.log(np.asarray(candidate, dtype=float) / np.asarray(control, dtype=float))
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = CRITICAL_T_DF255 * sd / math.sqrt(len(values))
    return {
        "geometric_mean_ratio": float(np.exp(mean)),
        "mean_log_ratio": mean,
        "sample_sd": sd,
        "ci95_lower": float(mean - margin),
        "ci95_upper": float(mean + margin),
    }


def main() -> None:
    registration, _ = base._load_registration(CONFIG)
    if registration["experiment"] != "sparse_causal_bridge_v7":
        raise RuntimeError("the frozen V7 family is required")
    used = set()
    for path in sorted((ROOT / "experiments/preregistration").glob("sparse_causal_bridge_v*.json")):
        historical, _ = base._load_registration(path)
        for role in historical.get("data_roles", {}).values():
            if isinstance(role, dict):
                used.update(int(seed) for seed in role.get("seeds", []))
    overlap = sorted(used.intersection(PILOT_SEEDS))
    if overlap:
        raise PermissionError(f"fresh pilot seeds overlap historical roles: {overlap}")

    context = rel._build_training_context(CONFIG, registration)
    train_role = registration["data_roles"]["observational_train"]
    train_episodes = [
        base.simulate_episode(
            int(seed),
            registration,
            environment=train_role["environment"],
            steps=int(train_role["steps_per_seed"]),
        )
        for seed in train_role["seeds"]
    ]
    sparse_gain = _fit_gain(
        train_episodes, context.sparse_mechanism, context.sparse_ar, context.scales
    )
    dense_gain = _fit_gain(
        train_episodes,
        context.dense_probe_mechanism,
        context.dense_probe_ar,
        context.scales,
    )
    zero_mechanism = latent.mechanism_model(
        "zero_bridge_parent",
        context.sparse_mechanism.local_coefficients[:, 1],
        np.zeros_like(context.sparse_mechanism.bridge),
        (),
    )
    zero_ar = latent.fit_pooled_residual_autoregression(train_episodes, zero_mechanism)
    zero_gain = _fit_gain(train_episodes, zero_mechanism, zero_ar, context.scales)

    expected_sparse_gain = 0.7868543064870357
    expected_dense_gain = 0.7835668486813699
    if not math.isclose(sparse_gain, expected_sparse_gain, rel_tol=0.0, abs_tol=1e-15):
        raise PermissionError("sparse training gain changed")
    if not math.isclose(dense_gain, expected_dense_gain, rel_tol=0.0, abs_tol=1e-15):
        raise PermissionError("dense training gain changed")

    names = (
        "parent_anchored_sparse",
        "v5_sparse_parent",
        "persistence",
        "symmetric_dense_shrinkage",
        "zero_bridge_shrinkage",
        "stable_adaptive_dense",
        "frozen_v7_consensus",
        "frozen_v7_no_sparse_consensus",
    )
    errors = {name: [] for name in names}
    radii = {"sparse": [], "dense": [], "zero_bridge": [], "adaptive": []}
    maximum_observed_index = -1
    future_reads = 0
    nonfinite_count = 0

    for seed in PILOT_SEEDS:
        episode = base.simulate_episode(
            seed, registration, environment="ood", steps=100
        )
        reader = rel.PrefixReader(episode.states, 80)
        prefix = reader.through_origin()
        maximum_observed_index = max(maximum_observed_index, reader.max_observed_state_index)
        future_reads += reader.future_observation_reads
        truth = episode.states[81:101]

        sparse = rel._latent_rollout(
            context.sparse_mechanism, context.sparse_ar, prefix, HORIZON
        )
        dense = rel._latent_rollout(
            context.dense_probe_mechanism, context.dense_probe_ar, prefix, HORIZON
        )
        zero = rel._latent_rollout(zero_mechanism, zero_ar, prefix, HORIZON)
        adaptive, adaptive_model = rel._adaptive_rollout(prefix, context, HORIZON)
        persistence = np.repeat(prefix[-1][None, :], HORIZON, axis=0)
        frozen = rel.predict_from_prefix(prefix, context, registration)

        predictions = {
            "parent_anchored_sparse": persistence + sparse_gain * (sparse - persistence),
            "v5_sparse_parent": sparse,
            "persistence": persistence,
            "symmetric_dense_shrinkage": persistence + dense_gain * (dense - persistence),
            "zero_bridge_shrinkage": persistence + zero_gain * (zero - persistence),
            "stable_adaptive_dense": adaptive,
            "frozen_v7_consensus": frozen.models["sparse_consensus"],
            "frozen_v7_no_sparse_consensus": frozen.models["no_sparse_consensus"],
        }
        for name, prediction in predictions.items():
            nonfinite_count += int(prediction.size - np.count_nonzero(np.isfinite(prediction)))
            errors[name].append(rel._normalized_rmse(truth, prediction, context.scales))
        radii["sparse"].append(
            rel.free._maximum_jacobian_radius(context.sparse_mechanism, sparse)
        )
        radii["dense"].append(
            rel.free._maximum_jacobian_radius(context.dense_probe_mechanism, dense)
        )
        radii["zero_bridge"].append(
            rel.free._maximum_jacobian_radius(zero_mechanism, zero)
        )
        radii["adaptive"].append(
            rel.free._maximum_jacobian_radius(adaptive_model, adaptive)
        )

    candidate = errors["parent_anchored_sparse"]
    report = {
        "status": "fresh_development_checkpoint",
        "claim_boundary": "four-chart synthetic H20 forecast-controller development only",
        "seeds": {"first": PILOT_SEEDS[0], "last": PILOT_SEEDS[-1], "count": len(PILOT_SEEDS)},
        "historical_seed_overlap": overlap,
        "gains": {
            "sparse": sparse_gain,
            "symmetric_dense": dense_gain,
            "zero_bridge": zero_gain,
        },
        "models": {name: _summary(values) for name, values in errors.items()},
        "comparisons": {
            "vs_v5_parent": _paired_lower(errors["v5_sparse_parent"], candidate),
            "vs_persistence": _paired_lower(errors["persistence"], candidate),
            "vs_zero_bridge": _paired_lower(errors["zero_bridge_shrinkage"], candidate),
            "vs_frozen_v7_consensus": _paired_lower(errors["frozen_v7_consensus"], candidate),
            "vs_frozen_v7_no_sparse": _paired_lower(
                errors["frozen_v7_no_sparse_consensus"], candidate
            ),
            "log_ratio_vs_symmetric_dense": _paired_log_ratio(
                candidate, errors["symmetric_dense_shrinkage"]
            ),
            "log_ratio_vs_stable_adaptive_dense": _paired_log_ratio(
                candidate, errors["stable_adaptive_dense"]
            ),
        },
        "stability": {
            name: {
                "maximum_pathwise_radius": float(np.max(values)),
                "fraction_above_0_98": float(np.mean(np.asarray(values) > 0.98)),
            }
            for name, values in radii.items()
        },
        "integrity": {
            "maximum_observed_state_index": maximum_observed_index,
            "future_observation_reads": future_reads,
            "nonfinite_prediction_count": nonfinite_count,
            "locked_historical_test_opened": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

