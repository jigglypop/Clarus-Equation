"""Infer rho_B and gamma from graph-mode recovery rates.

The single-map inverse problem cannot uniquely identify gamma without a prior.
Time-resolved recovery can. In the graph eigenbasis, the linear recovery model

    e_{n+1} = (rho_B I - gamma L_G) e_n

has modal decay factors

    mu_k = rho_B - gamma lambda_k.

Given two or more observed modal decay factors, rho_B and gamma can be inferred
directly. This script demonstrates the inverse step on the canonical graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


GRAPH_RESULT_PATH = Path(__file__).with_name("sleep_graph_spread_results.json")
RHO_B_TRUE = 0.155
GAMMA_TRUE = 0.03


def load_lambdas() -> np.ndarray:
    payload = json.loads(GRAPH_RESULT_PATH.read_text(encoding="utf-8"))
    return np.asarray(payload["laplacian_eigenvalues"], dtype=np.float64)


def infer_from_modes(lambda_i: float, mu_i: float, lambda_j: float, mu_j: float) -> tuple[float, float]:
    gamma = (mu_i - mu_j) / (lambda_j - lambda_i)
    rho_b = mu_i + gamma * lambda_i
    return float(rho_b), float(gamma)


def main() -> None:
    lambdas = load_lambdas()
    mu = RHO_B_TRUE - GAMMA_TRUE * lambdas
    stable = np.abs(mu) < 1.0

    pairs = []
    for i in range(1, len(lambdas) - 1):
        for j in range(i + 1, len(lambdas)):
            rho_hat, gamma_hat = infer_from_modes(lambdas[i], mu[i], lambdas[j], mu[j])
            pairs.append(
                {
                    "mode_i": i,
                    "mode_j": j,
                    "lambda_i": float(lambdas[i]),
                    "lambda_j": float(lambdas[j]),
                    "mu_i": float(mu[i]),
                    "mu_j": float(mu[j]),
                    "rho_b_hat": rho_hat,
                    "gamma_hat": gamma_hat,
                }
            )

    # Sensitivity: one percent modal-decay measurement noise.
    rng = np.random.default_rng(42)
    noisy_trials = []
    chosen = (1, len(lambdas) - 1)
    i, j = chosen
    for _ in range(200):
        mu_i = float(mu[i] * (1.0 + rng.normal(0.0, 0.01)))
        mu_j = float(mu[j] * (1.0 + rng.normal(0.0, 0.01)))
        rho_hat, gamma_hat = infer_from_modes(lambdas[i], mu_i, lambdas[j], mu_j)
        noisy_trials.append((rho_hat, gamma_hat))
    noisy = np.asarray(noisy_trials, dtype=np.float64)

    out = {
        "model": "e_{n+1} = (rho_B I - gamma L_G) e_n; mu_k = rho_B - gamma lambda_k",
        "rho_b_true": RHO_B_TRUE,
        "gamma_true": GAMMA_TRUE,
        "lambdas": [float(x) for x in lambdas],
        "mu": [float(x) for x in mu],
        "stable_modes": [bool(x) for x in stable],
        "pairwise_inverse": pairs,
        "noise_test": {
            "chosen_modes": [int(i), int(j)],
            "noise_fraction": 0.01,
            "rho_b_mean": float(noisy[:, 0].mean()),
            "rho_b_std": float(noisy[:, 0].std()),
            "gamma_mean": float(noisy[:, 1].mean()),
            "gamma_std": float(noisy[:, 1].std()),
        },
        "interpretation": {
            "lambda_0": "global uniform mode; estimates rho_B but not gamma by itself",
            "nonzero_modes": "spatial graph modes; differences between their decay rates identify gamma",
        },
    }

    out_path = Path(__file__).with_name("sleep_temporal_modes_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Temporal graph-mode inverse pilot")
    print(f"  true rho_B = {RHO_B_TRUE:.6f}")
    print(f"  true gamma = {GAMMA_TRUE:.6f}")
    print("  mode  lambda    mu")
    for index, (lam, decay) in enumerate(zip(lambdas, mu)):
        print(f"  {index:4d}  {lam:7.4f}  {decay:8.5f}")
    noise = out["noise_test"]
    print(
        "  1% noise inverse: "
        f"rho_B={noise['rho_b_mean']:.6f}+/-{noise['rho_b_std']:.6f}, "
        f"gamma={noise['gamma_mean']:.6f}+/-{noise['gamma_std']:.6f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
