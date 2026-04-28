"""Infer regional sensitivities from multi-axis homeostatic interventions.

The homeostatic control document defines regional burden as

    ell_r = d_r^T (q_n - q*)_+.

Sleep deprivation alone can identify only the sleep component of d_r. Multiple
interventions are needed to separate sleep, arousal, and metabolic sensitivity.
This pilot demonstrates the inverse step with a small design matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


AXES = ["sleep", "arousal", "autonomic", "endocrine", "immune", "metabolic"]

INTERVENTIONS = {
    "normal": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "sleep_deprivation": [0.34, 0.16, 0.08, 0.06, 0.03, 0.10],
    "high_arousal_task": [0.04, 0.38, 0.14, 0.10, 0.02, 0.08],
    "metabolic_stress": [0.08, 0.10, 0.12, 0.08, 0.06, 0.36],
    "recovery_sleep": [-0.18, -0.06, -0.04, -0.03, 0.00, -0.05],
}

TRUE_SENSITIVITY = {
    "cortex": [0.10, 0.16, 0.05, 0.04, 0.02, 0.08],
    "hippocampus": [0.13, 0.12, 0.04, 0.06, 0.03, 0.09],
    "hypothalamus": [0.18, 0.10, 0.12, 0.14, 0.05, 0.07],
    "brainstem": [0.14, 0.13, 0.16, 0.06, 0.04, 0.08],
    "default_mode": [0.11, 0.08, 0.05, 0.05, 0.03, 0.13],
}

PERTURBATION = {
    "cortex": [0.0000, 0.0012, -0.0008, 0.0007, -0.0005],
    "hippocampus": [0.0000, -0.0009, 0.0005, -0.0006, 0.0004],
    "hypothalamus": [0.0000, 0.0010, 0.0008, -0.0007, -0.0005],
    "brainstem": [0.0000, -0.0011, 0.0009, 0.0006, -0.0004],
    "default_mode": [0.0000, 0.0007, -0.0006, 0.0008, -0.0003],
}


def positive_part(values: list[float]) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def design_matrix() -> tuple[list[str], np.ndarray]:
    names = list(INTERVENTIONS)
    q = np.vstack([positive_part(INTERVENTIONS[name]) for name in names])
    return names, q


def infer_sensitivity(q: np.ndarray, burden: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(q, burden, rcond=None)
    return np.maximum(solution, 0.0)


def main() -> None:
    intervention_names, q = design_matrix()
    regions = {}

    for region, sensitivity in TRUE_SENSITIVITY.items():
        d_true = np.asarray(sensitivity, dtype=np.float64)
        burden = q @ d_true + np.asarray(PERTURBATION[region], dtype=np.float64)
        d_hat = infer_sensitivity(q, burden)
        reconstructed = q @ d_hat
        regions[region] = {
            "true_sensitivity": {axis: float(value) for axis, value in zip(AXES, d_true)},
            "observed_burden": {
                name: float(value) for name, value in zip(intervention_names, burden)
            },
            "estimated_sensitivity": {axis: float(value) for axis, value in zip(AXES, d_hat)},
            "reconstruction_rmse": float(np.sqrt(np.mean((burden - reconstructed) ** 2))),
        }

    axis_rankings = {}
    for axis_index, axis in enumerate(AXES):
        axis_rankings[axis] = sorted(
            (
                {
                    "region": region,
                    "estimated_sensitivity": payload["estimated_sensitivity"][axis],
                }
                for region, payload in regions.items()
            ),
            key=lambda item: item["estimated_sensitivity"],
            reverse=True,
        )

    out = {
        "model": "ell_r = d_r^T (q_n - q*)_+",
        "axes": AXES,
        "interventions": INTERVENTIONS,
        "regions": regions,
        "axis_rankings": axis_rankings,
        "interpretation": {
            "sleep": "identified mainly by sleep deprivation and recovery contrast",
            "arousal": "identified by high-arousal task contrast",
            "metabolic": "identified by metabolic-stress contrast",
            "inverse": "multiple interventions are required because a single q axis cannot identify all components of d_r",
        },
    }

    out_path = Path(__file__).with_name("homeostatic_sensitivity_map_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Homeostatic sensitivity-map pilot")
    print("  region        sleep   arousal  metabolic  rmse")
    for region, payload in regions.items():
        estimated = payload["estimated_sensitivity"]
        print(
            f"  {region:12s}  "
            f"{estimated['sleep']:.4f}  "
            f"{estimated['arousal']:.4f}   "
            f"{estimated['metabolic']:.4f}     "
            f"{payload['reconstruction_rmse']:.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
