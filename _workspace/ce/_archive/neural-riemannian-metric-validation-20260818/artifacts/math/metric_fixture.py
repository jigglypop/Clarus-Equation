"""Deterministic, synthetic-only sanity checks for 11-math.md.

It demonstrates: a known J change changes the operational metric; an exact
null remains null; gain/noise changes can change the metric with W held fixed;
and isotropic ridge regularization is not invariant to a non-orthogonal chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def covariance(j: np.ndarray, q: np.ndarray, horizon: int = 40) -> np.ndarray:
    out = np.zeros_like(q, dtype=float)
    power = np.eye(j.shape[0])
    for _ in range(horizon):
        out += power @ q @ power.T
        power = j @ power
    return out


def length(g: np.ndarray, displacement: np.ndarray) -> float:
    return float(np.sqrt(displacement @ g @ displacement))


def metric(j: np.ndarray, q: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.inv(covariance(j, q) + ridge * np.eye(j.shape[0]))


def main() -> None:
    q = np.diag([0.2, 0.5])
    w_pre = np.diag([0.60, 0.35])
    w_post = np.diag([0.25, 0.35])
    fixed_w = np.diag([0.60, 0.35])
    ridge = 0.1
    d = np.array([1.0, -0.4])

    g_pre, g_post = metric(w_pre, q, ridge), metric(w_post, q, ridge)
    g_null = metric(w_pre, q, ridge)
    # W is fixed; only gain and noise change.
    gain = 0.55
    q_gain_noise = np.diag([0.8, 0.10])
    g_confounded = metric(gain * fixed_w, q_gain_noise, ridge)

    # Congruence is exact without ridge.  Isotropic ridge breaks it in a
    # non-orthogonal coordinate change; a covariant ridge restores it.
    c = covariance(w_pre, q)
    p = np.diag([2.0, 0.5])
    cp, dp = p @ c @ p.T, p @ d
    g0, g0p = np.linalg.inv(c), np.linalg.inv(cp)
    gr, grp = np.linalg.inv(c + ridge * np.eye(2)), np.linalg.inv(cp + ridge * np.eye(2))
    g_covariant = np.linalg.inv(cp + ridge * (p @ p.T))

    result = {
        "positive_metric_change_norm": float(np.linalg.norm(g_post - g_pre)),
        "null_metric_change_norm": float(np.linalg.norm(g_null - g_pre)),
        "fixed_W_gain_noise_metric_change_norm": float(np.linalg.norm(g_confounded - g_pre)),
        "unregularized_length_original": length(g0, d),
        "unregularized_length_transformed": length(g0p, dp),
        "isotropic_ridge_length_original": length(gr, d),
        "isotropic_ridge_length_transformed": length(grp, dp),
        "covariant_ridge_length_transformed": length(g_covariant, dp),
    }
    result["unregularized_invariance_error"] = abs(
        result["unregularized_length_original"] - result["unregularized_length_transformed"]
    )
    result["isotropic_ridge_noninvariance_error"] = abs(
        result["isotropic_ridge_length_original"] - result["isotropic_ridge_length_transformed"]
    )
    result["covariant_ridge_invariance_error"] = abs(
        result["isotropic_ridge_length_original"] - result["covariant_ridge_length_transformed"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    Path(__file__).with_name("metric_fixture_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
