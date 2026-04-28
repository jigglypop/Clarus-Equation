"""Canonical graph-coupled sleep forcing pilot.

This is not a cohort fit. It is the analytic next step after the scalar
q_sleep pilot: given a sleep burden z_sleep and a canonical brain control graph,
compute how local sleep sensitivity spreads through graph coupling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


Q_RESULT_PATH = Path(__file__).with_name("sleep_q_pilot_results.json")
RHO_B = 0.155

NODES = ["ctx", "thal", "hip", "sal", "hyp", "stem", "aut"]
NODE_LABELS = {
    "ctx": "cortex",
    "thal": "thalamus",
    "hip": "hippocampus",
    "sal": "salience hub",
    "hyp": "hypothalamus",
    "stem": "brainstem",
    "aut": "autonomic output",
}

# Canonical anatomical-control graph used for the analytic pilot.
EDGES = {
    ("ctx", "thal"): 0.90,
    ("ctx", "hip"): 0.55,
    ("ctx", "sal"): 0.45,
    ("thal", "sal"): 0.40,
    ("hip", "sal"): 0.35,
    ("hyp", "stem"): 0.85,
    ("stem", "aut"): 0.75,
    ("hyp", "sal"): 0.45,
    ("hyp", "thal"): 0.30,
    ("stem", "thal"): 0.30,
}

# Prior local sleep sensitivity. These are model priors, not fitted data.
ALPHA_PRIOR = {
    "ctx": 0.10,
    "thal": 0.12,
    "hip": 0.08,
    "sal": 0.09,
    "hyp": 0.15,
    "stem": 0.13,
    "aut": 0.06,
}


def adjacency() -> np.ndarray:
    n = len(NODES)
    index = {node: i for i, node in enumerate(NODES)}
    a = np.zeros((n, n), dtype=np.float64)
    for (left, right), weight in EDGES.items():
        i, j = index[left], index[right]
        a[i, j] = weight
        a[j, i] = weight
    return a


def graph_laplacian(a: np.ndarray) -> np.ndarray:
    return np.diag(a.sum(axis=1)) - a


def fixed_response(z_sleep: float, gamma: float, laplacian: np.ndarray) -> np.ndarray:
    forcing = z_sleep * np.array([ALPHA_PRIOR[node] for node in NODES], dtype=np.float64)
    system = (1.0 - RHO_B) * np.eye(len(NODES)) + gamma * laplacian
    return np.linalg.solve(system, forcing)


def main() -> None:
    q_payload = json.loads(Q_RESULT_PATH.read_text(encoding="utf-8"))
    z_mean = float(q_payload["q_sleep"]["z_mean"])
    a = adjacency()
    lap = graph_laplacian(a)
    eigvals = np.linalg.eigvalsh(lap)
    lambda_max = float(eigvals.max())
    gamma_stable_upper = float((1.0 + RHO_B) / lambda_max)
    gamma_grid = [0.0, 0.05, 0.10, 0.20, 0.30]

    responses = {}
    for gamma in gamma_grid:
        response = fixed_response(z_mean, gamma, lap)
        responses[f"{gamma:.2f}"] = {
            "stable": bool(max(abs(RHO_B - gamma * eigvals)) < 1.0),
            "delta_x_a_by_node": {
                node: float(response[index])
                for index, node in enumerate(NODES)
            },
            "delta_x_b_by_node": {
                node: float(-response[index])
                for index, node in enumerate(NODES)
            },
            "max_delta_x_a": float(response.max()),
            "min_delta_x_a": float(response.min()),
            "spread_ratio_max_over_min": float(response.max() / response.min()),
        }

    payload = {
        "model": "e* = ((1-rho_B)I + gamma L_G)^-1 z_sleep alpha",
        "rho_b": RHO_B,
        "z_sleep": z_mean,
        "nodes": NODE_LABELS,
        "edges": {f"{left}-{right}": weight for (left, right), weight in EDGES.items()},
        "alpha_prior": ALPHA_PRIOR,
        "laplacian_eigenvalues": [float(x) for x in eigvals],
        "lambda_max": lambda_max,
        "gamma_stable_upper": gamma_stable_upper,
        "responses": responses,
    }

    out_path = Path(__file__).with_name("sleep_graph_spread_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Graph-coupled sleep forcing pilot")
    print(f"  z_sleep = {z_mean:.6f}")
    print(f"  lambda_max = {lambda_max:.6f}")
    print(f"  gamma stable upper = {gamma_stable_upper:.6f}")
    print("  gamma  stable  ctx     thal    hip     sal     hyp     stem    aut")
    for gamma in gamma_grid:
        row = responses[f"{gamma:.2f}"]
        values = row["delta_x_a_by_node"]
        print(
            f"  {gamma:5.2f}  {str(row['stable']):>6s}"
            f"  {values['ctx']:.4f}"
            f"  {values['thal']:.4f}"
            f"  {values['hip']:.4f}"
            f"  {values['sal']:.4f}"
            f"  {values['hyp']:.4f}"
            f"  {values['stem']:.4f}"
            f"  {values['aut']:.4f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
