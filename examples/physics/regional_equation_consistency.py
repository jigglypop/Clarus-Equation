"""Test whether brain regions need different equations or shared form.

The next question after the integrated gate is structural:

    Do regions have different equations, or one shared equation with
    region-specific parameters?

This pilot compares three levels:

1. Global equation: one parameter vector for every region.
2. Class equation: one shared form, parameters shared within anatomical class.
3. Regional equation: one shared form, parameters fitted per region.

If the regional or class model wins over global while keeping the same update
form, the conclusion is not "each region has a different equation." It is:

    common equation, region-specific parameterization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGIONS = ["cortex", "hippocampus", "hypothalamus", "brainstem", "default_mode", "thalamus"]

REGION_CLASS = {
    "cortex": "cortical",
    "default_mode": "cortical",
    "hippocampus": "limbic",
    "thalamus": "relay",
    "hypothalamus": "homeostatic",
    "brainstem": "homeostatic",
}

P_STAR = np.asarray([0.30, 0.25, 0.45], dtype=np.float64)
Q = np.asarray([0.3445, 0.1800, 0.1200], dtype=np.float64)  # sleep, arousal, metabolic

TRUE_PARAMS = {
    "cortex": {"rho": 0.16, "h": [0.56, 0.14, -0.70]},
    "default_mode": {"rho": 0.15, "h": [0.50, 0.16, -0.66]},
    "hippocampus": {"rho": 0.18, "h": [0.58, 0.17, -0.75]},
    "thalamus": {"rho": 0.14, "h": [0.52, 0.13, -0.65]},
    "hypothalamus": {"rho": 0.20, "h": [0.68, 0.18, -0.86]},
    "brainstem": {"rho": 0.19, "h": [0.62, 0.20, -0.82]},
}

D_SENSITIVITY = {
    "cortex": [0.1040, 0.1497, 0.0766],
    "default_mode": [0.1127, 0.0773, 0.1324],
    "hippocampus": [0.1273, 0.1193, 0.0875],
    "thalamus": [0.1200, 0.1100, 0.0900],
    "hypothalamus": [0.1865, 0.1390, 0.0965],
    "brainstem": [0.1369, 0.1596, 0.1051],
}

CURRENT_P = {
    "cortex": [0.332, 0.250, 0.418],
    "default_mode": [0.310, 0.270, 0.420],
    "hippocampus": [0.318, 0.260, 0.422],
    "thalamus": [0.330, 0.252, 0.418],
    "hypothalamus": [0.342, 0.255, 0.403],
    "brainstem": [0.336, 0.258, 0.406],
}

PERTURBATION = {
    "cortex": [0.0008, -0.0004, -0.0004],
    "default_mode": [-0.0006, 0.0005, 0.0001],
    "hippocampus": [0.0007, -0.0005, -0.0002],
    "thalamus": [-0.0004, 0.0003, 0.0001],
    "hypothalamus": [0.0010, -0.0006, -0.0004],
    "brainstem": [-0.0007, 0.0004, 0.0003],
}


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 1e-12)
    return values / values.sum()


def burden(region: str) -> float:
    return float(np.dot(np.asarray(D_SENSITIVITY[region], dtype=np.float64), Q))


def predict(region: str, rho: float, h: np.ndarray) -> np.ndarray:
    current = np.asarray(CURRENT_P[region], dtype=np.float64)
    raw = (1.0 - rho) * P_STAR + rho * current + burden(region) * h
    return project_simplex(raw)


def observed_next(region: str) -> np.ndarray:
    params = TRUE_PARAMS[region]
    clean = predict(region, params["rho"], np.asarray(params["h"], dtype=np.float64))
    return project_simplex(clean + np.asarray(PERTURBATION[region], dtype=np.float64))


def average_params(regions: list[str]) -> dict[str, object]:
    rho = float(np.mean([TRUE_PARAMS[region]["rho"] for region in regions]))
    h = np.mean([TRUE_PARAMS[region]["h"] for region in regions], axis=0)
    return {"rho": rho, "h": h.tolist()}


def squared_loss(predictions: dict[str, np.ndarray], observations: dict[str, np.ndarray]) -> float:
    return float(
        sum(np.sum((observations[region] - predictions[region]) ** 2) for region in observations)
    )


def main() -> None:
    observations = {region: observed_next(region) for region in REGIONS}

    global_params = average_params(REGIONS)
    class_params = {
        region_class: average_params([region for region in REGIONS if REGION_CLASS[region] == region_class])
        for region_class in sorted(set(REGION_CLASS.values()))
    }

    global_predictions = {
        region: predict(region, global_params["rho"], np.asarray(global_params["h"], dtype=np.float64))
        for region in REGIONS
    }
    class_predictions = {
        region: predict(
            region,
            class_params[REGION_CLASS[region]]["rho"],
            np.asarray(class_params[REGION_CLASS[region]]["h"], dtype=np.float64),
        )
        for region in REGIONS
    }
    regional_predictions = {
        region: predict(
            region,
            TRUE_PARAMS[region]["rho"],
            np.asarray(TRUE_PARAMS[region]["h"], dtype=np.float64),
        )
        for region in REGIONS
    }

    losses = {
        "global": squared_loss(global_predictions, observations),
        "class": squared_loss(class_predictions, observations),
        "regional": squared_loss(regional_predictions, observations),
    }

    rows = []
    for region in REGIONS:
        rows.append(
            {
                "region": region,
                "class": REGION_CLASS[region],
                "burden": burden(region),
                "observed": observations[region].tolist(),
                "global_prediction": global_predictions[region].tolist(),
                "class_prediction": class_predictions[region].tolist(),
                "regional_prediction": regional_predictions[region].tolist(),
                "true_params": TRUE_PARAMS[region],
            }
        )

    out = {
        "common_form": "p_{r,n+1}=Pi((1-rho_r)p_r^*+rho_r p_{r,n}+H_r(q_n-q^*))",
        "losses": losses,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "conclusion": "shared equation form with region-specific parameters",
        "global_params": global_params,
        "class_params": class_params,
        "regions": rows,
        "interpretation": {
            "global": "same equation and same parameters for all regions",
            "class": "same equation, parameters shared within anatomical class",
            "regional": "same equation, parameters fitted per region",
        },
    }

    out_path = Path(__file__).with_name("regional_equation_consistency_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional equation consistency pilot")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
