"""Estimate whether homeostatic forcing H is global, class-level, or regional.

The common regional equation uses a forcing term:

    p_{r,n+1} = ... + H_r(q_n - q*)

After d_r gives the scalar burden ell_r, H_r determines how that burden moves
mass between active, structural, and background components. This pilot compares
global, anatomical-class, and regional forcing vectors.
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

TRUE_H = {
    "cortex": [0.56, 0.14, -0.70],
    "default_mode": [0.50, 0.16, -0.66],
    "hippocampus": [0.58, 0.17, -0.75],
    "thalamus": [0.52, 0.13, -0.65],
    "hypothalamus": [0.68, 0.18, -0.86],
    "brainstem": [0.62, 0.20, -0.82],
}

BURDEN_LEVELS = np.asarray([0.10, 0.20, 0.32], dtype=np.float64)
PERTURBATION = {
    "cortex": [[0.0003, -0.0002, -0.0001], [-0.0002, 0.0001, 0.0001], [0.0002, -0.0001, -0.0001]],
    "default_mode": [[-0.0002, 0.0001, 0.0001], [0.0002, -0.0001, -0.0001], [-0.0001, 0.0001, 0.0000]],
    "hippocampus": [[0.0002, -0.0001, -0.0001], [-0.0001, 0.0001, 0.0000], [0.0001, -0.0001, 0.0000]],
    "thalamus": [[-0.0002, 0.0001, 0.0001], [0.0001, -0.0001, 0.0000], [-0.0001, 0.0001, 0.0000]],
    "hypothalamus": [[0.0004, -0.0002, -0.0002], [-0.0003, 0.0002, 0.0001], [0.0002, -0.0001, -0.0001]],
    "brainstem": [[-0.0003, 0.0002, 0.0001], [0.0002, -0.0001, -0.0001], [-0.0002, 0.0001, 0.0001]],
}


def observed_forcing(region: str) -> np.ndarray:
    h = np.asarray(TRUE_H[region], dtype=np.float64)
    perturbation = np.asarray(PERTURBATION[region], dtype=np.float64)
    return BURDEN_LEVELS[:, None] * h[None, :] + perturbation


def estimate_h(regions: list[str], observations: dict[str, np.ndarray]) -> np.ndarray:
    y = np.vstack([observations[region] for region in regions])
    x = np.concatenate([BURDEN_LEVELS for _ in regions])
    h = []
    for component in range(3):
        h.append(float(np.dot(x, y[:, component]) / np.dot(x, x)))
    h = np.asarray(h, dtype=np.float64)
    # Preserve simplex mass by removing tiny numerical sum drift.
    return h - h.mean()


def squared_loss(h_by_region: dict[str, np.ndarray], observations: dict[str, np.ndarray]) -> float:
    total = 0.0
    for region, obs in observations.items():
        pred = BURDEN_LEVELS[:, None] * h_by_region[region][None, :]
        total += float(np.sum((obs - pred) ** 2))
    return total


def main() -> None:
    observations = {region: observed_forcing(region) for region in REGIONS}

    global_h = estimate_h(REGIONS, observations)
    class_h = {
        region_class: estimate_h(
            [region for region in REGIONS if REGION_CLASS[region] == region_class],
            observations,
        )
        for region_class in sorted(set(REGION_CLASS.values()))
    }
    regional_h = {region: estimate_h([region], observations) for region in REGIONS}

    global_model = {region: global_h for region in REGIONS}
    class_model = {region: class_h[REGION_CLASS[region]] for region in REGIONS}
    regional_model = regional_h

    losses = {
        "global": squared_loss(global_model, observations),
        "class": squared_loss(class_model, observations),
        "regional": squared_loss(regional_model, observations),
    }

    rows = []
    for region in REGIONS:
        rows.append(
            {
                "region": region,
                "class": REGION_CLASS[region],
                "true_h": TRUE_H[region],
                "estimated_regional_h": regional_h[region].tolist(),
                "class_h": class_h[REGION_CLASS[region]].tolist(),
                "global_h": global_h.tolist(),
            }
        )

    out = {
        "model": "Delta p_r^(q)=ell_r H_r, 1^T H_r=0",
        "losses": losses,
        "global_h": global_h.tolist(),
        "class_h": {key: value.tolist() for key, value in class_h.items()},
        "regions": rows,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "interpretation": {
            "global": "one forcing direction for all regions",
            "class": "one forcing direction per anatomical class",
            "regional": "one forcing direction H_r per region",
            "gate": "use regional H_r only if repeated holdout beats class H_c",
        },
    }

    out_path = Path(__file__).with_name("regional_homeostatic_forcing_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional homeostatic-forcing pilot")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
