"""Compare global, class-level, and regional task sensitivity beta_r.

Task active-state pilots estimate

    A_r(u)=A_{r,0}+beta_r u_task.

This script asks whether beta should be shared globally, shared by anatomical
class, or estimated per region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REGIONS = ["visual", "motor", "prefrontal", "default_mode", "hippocampus", "thalamus"]

REGION_CLASS = {
    "visual": "cortical",
    "motor": "cortical",
    "prefrontal": "cortical",
    "default_mode": "cortical",
    "hippocampus": "limbic",
    "thalamus": "relay",
}

TASK_LOADS = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

TRUE_BETA = {
    "visual": 0.420,
    "motor": 0.340,
    "prefrontal": 0.260,
    "default_mode": 0.120,
    "hippocampus": 0.180,
    "thalamus": 0.220,
}

BASE_ACTIVE_SCORE = {
    "visual": 0.18,
    "motor": 0.16,
    "prefrontal": 0.14,
    "default_mode": 0.10,
    "hippocampus": 0.12,
    "thalamus": 0.15,
}

PERTURBATION = {
    "visual": [0.0000, 0.0010, -0.0007, 0.0004],
    "motor": [0.0000, -0.0008, 0.0005, -0.0003],
    "prefrontal": [0.0000, 0.0006, -0.0004, 0.0002],
    "default_mode": [0.0000, -0.0005, 0.0003, -0.0002],
    "hippocampus": [0.0000, 0.0005, -0.0003, 0.0002],
    "thalamus": [0.0000, -0.0004, 0.0003, -0.0001],
}


def observed_active_score(region: str) -> np.ndarray:
    return (
        BASE_ACTIVE_SCORE[region]
        + TRUE_BETA[region] * TASK_LOADS
        + np.asarray(PERTURBATION[region], dtype=np.float64)
    )


def estimate_beta(regions: list[str], observations: dict[str, np.ndarray]) -> float:
    y = np.concatenate(
        [observations[region] - BASE_ACTIVE_SCORE[region] for region in regions]
    )
    x = np.concatenate([TASK_LOADS for _ in regions])
    return float(np.dot(x, y) / np.dot(x, x))


def squared_loss(beta_by_region: dict[str, float], observations: dict[str, np.ndarray]) -> float:
    total = 0.0
    for region, obs in observations.items():
        pred = BASE_ACTIVE_SCORE[region] + beta_by_region[region] * TASK_LOADS
        total += float(np.sum((obs - pred) ** 2))
    return total


def main() -> None:
    observations = {region: observed_active_score(region) for region in REGIONS}

    global_beta = estimate_beta(REGIONS, observations)
    class_beta = {
        region_class: estimate_beta(
            [region for region in REGIONS if REGION_CLASS[region] == region_class],
            observations,
        )
        for region_class in sorted(set(REGION_CLASS.values()))
    }
    regional_beta = {region: estimate_beta([region], observations) for region in REGIONS}

    global_model = {region: global_beta for region in REGIONS}
    class_model = {region: class_beta[REGION_CLASS[region]] for region in REGIONS}
    regional_model = regional_beta

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
                "true_beta": TRUE_BETA[region],
                "estimated_regional_beta": regional_beta[region],
                "class_beta": class_beta[REGION_CLASS[region]],
                "global_beta": global_beta,
                "observed_active_score": observations[region].tolist(),
            }
        )

    out = {
        "model": "A_r(u)=A_{r,0}+beta_r u_task",
        "losses": losses,
        "global_beta": global_beta,
        "class_beta": class_beta,
        "regions": rows,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "interpretation": {
            "global": "one beta for all regions",
            "class": "one beta per anatomical class",
            "regional": "one beta_r per region",
            "gate": "use regional beta_r only if repeated holdout beats class beta_c",
        },
    }

    out_path = Path(__file__).with_name("regional_task_sensitivity_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional task-sensitivity pilot")
    print(f"  beta_global = {global_beta:.6f}")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
