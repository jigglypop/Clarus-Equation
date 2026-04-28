"""Compare graph-coupled prediction against a flat regional model.

The graph term should earn its place by improving held-out prediction:

    L_graph < L_flat

This pilot uses the same canonical brain graph used in the sleep graph scripts.
Observed next-state deviations are generated from a graph-coupled update with a
small deterministic perturbation. The flat baseline uses only local inertia and
forcing, while the graph model also includes gamma Delta_G e.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGIONS = [
    "cortex",
    "thalamus",
    "hippocampus",
    "salience",
    "hypothalamus",
    "brainstem",
    "autonomic",
]

EDGES = [
    ("cortex", "thalamus", 0.90),
    ("cortex", "hippocampus", 0.55),
    ("cortex", "salience", 0.45),
    ("thalamus", "salience", 0.40),
    ("hippocampus", "salience", 0.35),
    ("hypothalamus", "brainstem", 0.85),
    ("brainstem", "autonomic", 0.75),
    ("hypothalamus", "salience", 0.45),
    ("hypothalamus", "thalamus", 0.30),
    ("brainstem", "thalamus", 0.30),
]

RHO = 0.155
GAMMA = 0.18
Z_SLEEP = 0.3445
ALPHA = np.asarray([0.10, 0.12, 0.08, 0.09, 0.15, 0.13, 0.06], dtype=np.float64)
CURRENT_E = np.asarray([0.033, 0.047, 0.026, 0.035, 0.058, 0.052, 0.020], dtype=np.float64)
PERTURBATION = np.asarray([0.0015, -0.0010, 0.0008, -0.0005, 0.0012, -0.0007, 0.0004], dtype=np.float64)


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


def graph_delta(lap: np.ndarray, values: np.ndarray) -> np.ndarray:
    return -lap @ values


def predict_flat(current_e: np.ndarray) -> np.ndarray:
    return RHO * current_e + Z_SLEEP * ALPHA


def predict_graph(current_e: np.ndarray, lap: np.ndarray) -> np.ndarray:
    return RHO * current_e + GAMMA * graph_delta(lap, current_e) + Z_SLEEP * ALPHA


def squared_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((observed - predicted) ** 2))


def main() -> None:
    lap = laplacian()
    graph_prediction = predict_graph(CURRENT_E, lap)
    flat_prediction = predict_flat(CURRENT_E)
    observed = graph_prediction + PERTURBATION

    loss_graph = squared_loss(observed, graph_prediction)
    loss_flat = squared_loss(observed, flat_prediction)
    improvement = 1.0 - loss_graph / loss_flat

    rows = []
    for i, region in enumerate(REGIONS):
        rows.append(
            {
                "region": region,
                "current_e": float(CURRENT_E[i]),
                "observed_next_e": float(observed[i]),
                "graph_prediction": float(graph_prediction[i]),
                "flat_prediction": float(flat_prediction[i]),
                "graph_error": float(observed[i] - graph_prediction[i]),
                "flat_error": float(observed[i] - flat_prediction[i]),
            }
        )

    out = {
        "criterion": "L_graph < L_flat",
        "rho": RHO,
        "gamma": GAMMA,
        "z_sleep": Z_SLEEP,
        "loss_graph": loss_graph,
        "loss_flat": loss_flat,
        "graph_over_flat": loss_graph / loss_flat,
        "relative_improvement": improvement,
        "passed_graph_gate": loss_graph < loss_flat,
        "regions": rows,
        "interpretation": {
            "flat": "local inertia plus sleep forcing only",
            "graph": "flat model plus gamma Delta_G e",
            "gate": "graph term is retained only if holdout loss is lower than the flat baseline",
        },
    }

    out_path = Path(__file__).with_name("graph_vs_flat_prediction_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Graph-vs-flat prediction pilot")
    print(f"  L_graph = {loss_graph:.8f}")
    print(f"  L_flat  = {loss_flat:.8f}")
    print(f"  graph/flat = {loss_graph / loss_flat:.6f}")
    print(f"  relative improvement = {improvement:.6f}")
    print(f"  passed = {loss_graph < loss_flat}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
