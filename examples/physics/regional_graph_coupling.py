"""Estimate whether graph coupling gamma is global, class-level, or regional.

The graph term can share one Laplacian while allowing region-specific coupling:

    e_{r,n+1} = rho e_{r,n} + gamma_r Delta_G e_r + u_r.

This pilot compares:

1. Global gamma for all regions.
2. Class gamma shared within anatomical classes.
3. Regional gamma_r for each region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGIONS = ["cortex", "default_mode", "hippocampus", "thalamus", "hypothalamus", "brainstem"]

REGION_CLASS = {
    "cortex": "cortical",
    "default_mode": "cortical",
    "hippocampus": "limbic",
    "thalamus": "relay",
    "hypothalamus": "homeostatic",
    "brainstem": "homeostatic",
}

EDGES = [
    ("cortex", "default_mode", 0.45),
    ("cortex", "hippocampus", 0.55),
    ("cortex", "thalamus", 0.90),
    ("hippocampus", "thalamus", 0.30),
    ("thalamus", "hypothalamus", 0.30),
    ("hypothalamus", "brainstem", 0.85),
    ("brainstem", "default_mode", 0.35),
]

TRUE_GAMMA = {
    "cortex": 0.145,
    "default_mode": 0.130,
    "hippocampus": 0.115,
    "thalamus": 0.170,
    "hypothalamus": 0.205,
    "brainstem": 0.195,
}

RHO = 0.155
CURRENT_E = np.asarray([0.033, 0.025, 0.030, 0.047, 0.058, 0.052], dtype=np.float64)
FORCING = np.asarray([0.035, 0.031, 0.038, 0.041, 0.055, 0.050], dtype=np.float64)
PERTURBATION = np.asarray([0.0004, -0.0003, 0.0002, -0.0004, 0.0005, -0.0003], dtype=np.float64)


def laplacian() -> np.ndarray:
    index = {region: i for i, region in enumerate(REGIONS)}
    adjacency = np.zeros((len(REGIONS), len(REGIONS)), dtype=np.float64)
    for left, right, weight in EDGES:
        i = index[left]
        j = index[right]
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return np.diag(adjacency.sum(axis=1)) - adjacency


def graph_delta() -> np.ndarray:
    return -laplacian() @ CURRENT_E


def observed_next() -> np.ndarray:
    delta = graph_delta()
    gamma = np.asarray([TRUE_GAMMA[region] for region in REGIONS], dtype=np.float64)
    return RHO * CURRENT_E + gamma * delta + FORCING + PERTURBATION


def estimate_gamma_for_regions(regions: list[str], obs: np.ndarray) -> float:
    delta = graph_delta()
    indices = [REGIONS.index(region) for region in regions]
    y = obs[indices] - RHO * CURRENT_E[indices] - FORCING[indices]
    x = delta[indices]
    return float(np.dot(x, y) / np.dot(x, x))


def squared_loss(gamma_by_region: dict[str, float], obs: np.ndarray) -> float:
    delta = graph_delta()
    total = 0.0
    for i, region in enumerate(REGIONS):
        pred = RHO * CURRENT_E[i] + gamma_by_region[region] * delta[i] + FORCING[i]
        total += float((obs[i] - pred) ** 2)
    return total


def main() -> None:
    obs = observed_next()
    global_gamma = estimate_gamma_for_regions(REGIONS, obs)
    class_gamma = {
        region_class: estimate_gamma_for_regions(
            [region for region in REGIONS if REGION_CLASS[region] == region_class],
            obs,
        )
        for region_class in sorted(set(REGION_CLASS.values()))
    }
    regional_gamma = {
        region: estimate_gamma_for_regions([region], obs)
        for region in REGIONS
    }

    global_model = {region: global_gamma for region in REGIONS}
    class_model = {region: class_gamma[REGION_CLASS[region]] for region in REGIONS}
    regional_model = regional_gamma

    losses = {
        "global": squared_loss(global_model, obs),
        "class": squared_loss(class_model, obs),
        "regional": squared_loss(regional_model, obs),
    }

    rows = []
    delta = graph_delta()
    for i, region in enumerate(REGIONS):
        rows.append(
            {
                "region": region,
                "class": REGION_CLASS[region],
                "delta_g": float(delta[i]),
                "true_gamma": TRUE_GAMMA[region],
                "estimated_regional_gamma": regional_gamma[region],
                "class_gamma": class_gamma[REGION_CLASS[region]],
                "global_gamma": global_gamma,
                "observed_next_e": float(obs[i]),
            }
        )

    out = {
        "model": "e_{r,n+1}=rho e_{r,n}+gamma_r Delta_G e_r+u_r",
        "losses": losses,
        "global_gamma": global_gamma,
        "class_gamma": class_gamma,
        "regions": rows,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "interpretation": {
            "global": "one gamma for all regions",
            "class": "one gamma per anatomical class",
            "regional": "one gamma_r per region",
            "gate": "use regional gamma only if it improves held-out prediction beyond class gamma",
        },
    }

    out_path = Path(__file__).with_name("regional_graph_coupling_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional graph-coupling pilot")
    print(f"  gamma_global = {global_gamma:.6f}")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
