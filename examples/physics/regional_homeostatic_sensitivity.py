"""Compare global, class-level, and regional homeostatic sensitivity d_r.

The scalar burden is

    ell_r = d_r^T (q_n - q*)_+.

Previous pilots showed how to estimate d_r from multiple interventions. This
script asks whether d should be shared globally, shared by anatomical class, or
estimated per region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


AXES = ["sleep", "arousal", "metabolic"]

REGIONS = ["cortex", "default_mode", "hippocampus", "thalamus", "hypothalamus", "brainstem"]

REGION_CLASS = {
    "cortex": "cortical",
    "default_mode": "cortical",
    "hippocampus": "limbic",
    "thalamus": "relay",
    "hypothalamus": "homeostatic",
    "brainstem": "homeostatic",
}

Q_DESIGN = np.asarray(
    [
        [0.34, 0.16, 0.10],
        [0.04, 0.38, 0.08],
        [0.08, 0.10, 0.36],
        [0.20, 0.22, 0.18],
    ],
    dtype=np.float64,
)

TRUE_D = {
    "cortex": [0.1040, 0.1497, 0.0766],
    "default_mode": [0.1127, 0.0773, 0.1324],
    "hippocampus": [0.1273, 0.1193, 0.0875],
    "thalamus": [0.1200, 0.1100, 0.0900],
    "hypothalamus": [0.1865, 0.1390, 0.0965],
    "brainstem": [0.1369, 0.1596, 0.1051],
}

PERTURBATION = {
    "cortex": [0.0004, -0.0003, 0.0002, -0.0001],
    "default_mode": [-0.0003, 0.0002, -0.0001, 0.0001],
    "hippocampus": [0.0003, -0.0002, 0.0002, -0.0001],
    "thalamus": [-0.0002, 0.0002, -0.0001, 0.0001],
    "hypothalamus": [0.0005, -0.0003, 0.0002, -0.0002],
    "brainstem": [-0.0004, 0.0003, -0.0002, 0.0001],
}


def observed_burden(region: str) -> np.ndarray:
    d = np.asarray(TRUE_D[region], dtype=np.float64)
    return Q_DESIGN @ d + np.asarray(PERTURBATION[region], dtype=np.float64)


def estimate_d(regions: list[str], observations: dict[str, np.ndarray]) -> np.ndarray:
    y = np.concatenate([observations[region] for region in regions])
    design = np.vstack([Q_DESIGN for _ in regions])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.maximum(solution, 0.0)


def squared_loss(d_by_region: dict[str, np.ndarray], observations: dict[str, np.ndarray]) -> float:
    total = 0.0
    for region, obs in observations.items():
        pred = Q_DESIGN @ d_by_region[region]
        total += float(np.sum((obs - pred) ** 2))
    return total


def main() -> None:
    observations = {region: observed_burden(region) for region in REGIONS}

    global_d = estimate_d(REGIONS, observations)
    class_d = {
        region_class: estimate_d(
            [region for region in REGIONS if REGION_CLASS[region] == region_class],
            observations,
        )
        for region_class in sorted(set(REGION_CLASS.values()))
    }
    regional_d = {region: estimate_d([region], observations) for region in REGIONS}

    global_model = {region: global_d for region in REGIONS}
    class_model = {region: class_d[REGION_CLASS[region]] for region in REGIONS}
    regional_model = regional_d

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
                "true_d": {axis: value for axis, value in zip(AXES, TRUE_D[region])},
                "estimated_regional_d": {
                    axis: float(value) for axis, value in zip(AXES, regional_d[region])
                },
                "class_d": {
                    axis: float(value) for axis, value in zip(AXES, class_d[REGION_CLASS[region]])
                },
                "global_d": {axis: float(value) for axis, value in zip(AXES, global_d)},
            }
        )

    out = {
        "model": "ell_r=d_r^T(q_n-q*)_+",
        "axes": AXES,
        "losses": losses,
        "global_d": {axis: float(value) for axis, value in zip(AXES, global_d)},
        "class_d": {
            region_class: {axis: float(value) for axis, value in zip(AXES, values)}
            for region_class, values in class_d.items()
        },
        "regions": rows,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "interpretation": {
            "global": "one d vector for all regions",
            "class": "one d vector per anatomical class",
            "regional": "one d_r vector per region",
            "gate": "use regional d_r only if repeated holdout beats class d_c",
        },
    }

    out_path = Path(__file__).with_name("regional_homeostatic_sensitivity_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional homeostatic-sensitivity pilot")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
