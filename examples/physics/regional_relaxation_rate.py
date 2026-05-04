"""Estimate whether relaxation rate rho is global, class-level, or regional.

Once p_r^* is set, the next regional parameter is rho_r:

    e_{r,n+1} = rho_r e_{r,n}

where e_r is the deviation from the local reference state. This pilot compares
three hypotheses:

1. Global rho: one relaxation rate for every region.
2. Class rho: one rate per anatomical class.
3. Regional rho: one rate per region.

The model is intentionally one-dimensional in residual norm space so rho_r is
interpretable as a recovery/inertia factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGION_CLASS = {
    "cortex": "cortical",
    "default_mode": "cortical",
    "hippocampus": "limbic",
    "thalamus": "relay",
    "hypothalamus": "homeostatic",
    "brainstem": "homeostatic",
}

TRUE_RHO = {
    "cortex": 0.160,
    "default_mode": 0.150,
    "hippocampus": 0.180,
    "thalamus": 0.140,
    "hypothalamus": 0.205,
    "brainstem": 0.190,
}

INITIAL_RESIDUALS = np.asarray([0.090, 0.070, 0.055, 0.043], dtype=np.float64)
PERTURBATION = {
    "cortex": [0.0005, -0.0003, 0.0002, -0.0001],
    "default_mode": [-0.0004, 0.0002, -0.0001, 0.0001],
    "hippocampus": [0.0003, -0.0002, 0.0001, -0.0001],
    "thalamus": [-0.0003, 0.0002, -0.0001, 0.0001],
    "hypothalamus": [0.0006, -0.0004, 0.0002, -0.0001],
    "brainstem": [-0.0005, 0.0003, -0.0002, 0.0001],
}


def observed_next(region: str) -> np.ndarray:
    rho = TRUE_RHO[region]
    return rho * INITIAL_RESIDUALS + np.asarray(PERTURBATION[region], dtype=np.float64)


def estimate_rho(initial: np.ndarray, next_residual: np.ndarray) -> float:
    return float(np.dot(initial, next_residual) / np.dot(initial, initial))


def average_rho(regions: list[str]) -> float:
    observations = np.concatenate([observed_next(region) for region in regions])
    design = np.concatenate([INITIAL_RESIDUALS for _ in regions])
    return estimate_rho(design, observations)


def squared_loss(rho_by_region: dict[str, float], observations: dict[str, np.ndarray]) -> float:
    total = 0.0
    for region, next_residual in observations.items():
        prediction = rho_by_region[region] * INITIAL_RESIDUALS
        total += float(np.sum((next_residual - prediction) ** 2))
    return total


def main() -> None:
    observations = {region: observed_next(region) for region in REGION_CLASS}

    global_rho = average_rho(list(REGION_CLASS))
    class_rho = {
        region_class: average_rho(
            [region for region, current_class in REGION_CLASS.items() if current_class == region_class]
        )
        for region_class in sorted(set(REGION_CLASS.values()))
    }
    regional_rho = {region: estimate_rho(INITIAL_RESIDUALS, obs) for region, obs in observations.items()}

    global_model = {region: global_rho for region in REGION_CLASS}
    class_model = {region: class_rho[REGION_CLASS[region]] for region in REGION_CLASS}
    regional_model = regional_rho

    losses = {
        "global": squared_loss(global_model, observations),
        "class": squared_loss(class_model, observations),
        "regional": squared_loss(regional_model, observations),
    }

    rows = []
    for region in REGION_CLASS:
        rows.append(
            {
                "region": region,
                "class": REGION_CLASS[region],
                "true_rho": TRUE_RHO[region],
                "estimated_regional_rho": regional_rho[region],
                "class_rho": class_rho[REGION_CLASS[region]],
                "global_rho": global_rho,
                "observed_next_residuals": observations[region].tolist(),
            }
        )

    out = {
        "model": "e_{r,n+1}=rho_r e_{r,n}",
        "losses": losses,
        "global_rho": global_rho,
        "class_rho": class_rho,
        "regions": rows,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "interpretation": {
            "global": "one rho for all regions",
            "class": "one rho per anatomical class",
            "regional": "one rho_r per region",
            "gate": "use regional rho only if it improves holdout beyond class rho",
        },
    }

    out_path = Path(__file__).with_name("regional_relaxation_rate_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional relaxation-rate pilot")
    print(f"  rho_global = {global_rho:.6f}")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
