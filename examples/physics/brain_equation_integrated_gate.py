"""Integrated prediction gate for the brain state equation.

This pilot combines the components derived in the brain docs:

    p_{r,n+1} = Pi_Delta((1-rho) p* + rho p_{r,n}
                         + gamma Delta_G p_{r,n}
                         + H_r(q_n-q*))

It compares the integrated model against ablations that remove graph coupling
or homeostatic forcing. The point is not that these synthetic numbers are final
evidence, but that the full equation has a concrete holdout gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGIONS = ["cortex", "hippocampus", "hypothalamus", "brainstem", "default_mode"]

EDGES = [
    ("cortex", "hippocampus", 0.55),
    ("cortex", "default_mode", 0.45),
    ("cortex", "hypothalamus", 0.25),
    ("hippocampus", "hypothalamus", 0.30),
    ("hypothalamus", "brainstem", 0.85),
    ("brainstem", "default_mode", 0.35),
]

P_STAR = np.asarray([0.30, 0.25, 0.45], dtype=np.float64)
RHO = 0.155
GAMMA = 0.12

Q_DELTA = {
    "sleep": 0.3445,
    "arousal": 0.1800,
    "metabolic": 0.1200,
}

D_SENSITIVITY = {
    "cortex": {"sleep": 0.1040, "arousal": 0.1497, "metabolic": 0.0766},
    "hippocampus": {"sleep": 0.1273, "arousal": 0.1193, "metabolic": 0.0875},
    "hypothalamus": {"sleep": 0.1865, "arousal": 0.1390, "metabolic": 0.0965},
    "brainstem": {"sleep": 0.1369, "arousal": 0.1596, "metabolic": 0.1051},
    "default_mode": {"sleep": 0.1127, "arousal": 0.0773, "metabolic": 0.1324},
}

CURRENT_P = np.asarray(
    [
        [0.332, 0.250, 0.418],
        [0.318, 0.260, 0.422],
        [0.342, 0.255, 0.403],
        [0.336, 0.258, 0.406],
        [0.310, 0.270, 0.420],
    ],
    dtype=np.float64,
)

PERTURBATION = np.asarray(
    [
        [0.0015, -0.0008, -0.0007],
        [-0.0009, 0.0007, 0.0002],
        [0.0012, -0.0006, -0.0006],
        [-0.0007, 0.0006, 0.0001],
        [0.0008, -0.0004, -0.0004],
    ],
    dtype=np.float64,
)


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 1e-12)
    return values / values.sum()


def laplacian() -> np.ndarray:
    index = {name: i for i, name in enumerate(REGIONS)}
    adjacency = np.zeros((len(REGIONS), len(REGIONS)), dtype=np.float64)
    for left, right, weight in EDGES:
        i = index[left]
        j = index[right]
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def graph_delta(lap: np.ndarray, state: np.ndarray) -> np.ndarray:
    return -lap @ state


def homeostatic_forcing() -> np.ndarray:
    forcing = []
    for region in REGIONS:
        sensitivity = D_SENSITIVITY[region]
        burden = sum(sensitivity[axis] * Q_DELTA[axis] for axis in Q_DELTA)
        # Burden shifts reserve mass into active and structural channels.
        forcing.append([0.70 * burden, 0.20 * burden, -0.90 * burden])
    return np.asarray(forcing, dtype=np.float64)


def predict(
    current_p: np.ndarray,
    *,
    include_graph: bool,
    include_homeostasis: bool,
) -> np.ndarray:
    lap = laplacian()
    delta_g = graph_delta(lap, current_p) if include_graph else np.zeros_like(current_p)
    forcing = homeostatic_forcing() if include_homeostasis else np.zeros_like(current_p)

    predicted = []
    for region_index, p in enumerate(current_p):
        raw = (
            (1.0 - RHO) * P_STAR
            + RHO * p
            + GAMMA * delta_g[region_index]
            + forcing[region_index]
        )
        predicted.append(project_simplex(raw))
    return np.asarray(predicted, dtype=np.float64)


def squared_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((observed - predicted) ** 2))


def main() -> None:
    full = predict(CURRENT_P, include_graph=True, include_homeostasis=True)
    no_graph = predict(CURRENT_P, include_graph=False, include_homeostasis=True)
    no_homeostasis = predict(CURRENT_P, include_graph=True, include_homeostasis=False)
    inertial = predict(CURRENT_P, include_graph=False, include_homeostasis=False)
    observed = np.asarray([project_simplex(row) for row in full + PERTURBATION])

    losses = {
        "full": squared_loss(observed, full),
        "no_graph": squared_loss(observed, no_graph),
        "no_homeostasis": squared_loss(observed, no_homeostasis),
        "inertial": squared_loss(observed, inertial),
    }
    best_ablation = min(losses["no_graph"], losses["no_homeostasis"], losses["inertial"])

    rows = []
    for index, region in enumerate(REGIONS):
        rows.append(
            {
                "region": region,
                "current_p": CURRENT_P[index].tolist(),
                "observed_next_p": observed[index].tolist(),
                "full_prediction": full[index].tolist(),
                "no_graph_prediction": no_graph[index].tolist(),
                "no_homeostasis_prediction": no_homeostasis[index].tolist(),
                "homeostatic_forcing": homeostatic_forcing()[index].tolist(),
            }
        )

    out = {
        "criterion": "L_full < min(L_no_graph, L_no_homeostasis, L_inertial)",
        "rho": RHO,
        "gamma": GAMMA,
        "q_delta": Q_DELTA,
        "losses": losses,
        "full_over_best_ablation": losses["full"] / best_ablation,
        "passed_integrated_gate": losses["full"] < best_ablation,
        "regions": rows,
        "interpretation": {
            "full": "inertia, graph coupling, and homeostatic forcing",
            "no_graph": "removes Delta_G",
            "no_homeostasis": "removes H_r(q_n-q*)",
            "inertial": "keeps only relaxation toward p* and previous state",
        },
    }

    out_path = Path(__file__).with_name("brain_equation_integrated_gate_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Integrated brain-equation gate")
    for name, value in losses.items():
        print(f"  L_{name:14s} = {value:.8f}")
    print(f"  full/best_ablation = {out['full_over_best_ablation']:.6f}")
    print(f"  passed = {out['passed_integrated_gate']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
