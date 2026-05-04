"""Estimate whether the reference state p_r^* is global, class-level, or regional.

After deciding that brain regions share an update form with region-specific
parameters, the first parameter to close is the reference state p_r^*.

This pilot compares three hypotheses:

1. Global p*: one reference state for every region.
2. Class p*: one reference state per anatomical class.
3. Regional p*: one reference state per region.

The state remains on the simplex in all cases.
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

REFERENCE_SAMPLES = {
    "cortex": [
        [0.300, 0.248, 0.452],
        [0.306, 0.251, 0.443],
        [0.297, 0.253, 0.450],
    ],
    "default_mode": [
        [0.282, 0.256, 0.462],
        [0.286, 0.259, 0.455],
        [0.279, 0.261, 0.460],
    ],
    "hippocampus": [
        [0.292, 0.275, 0.433],
        [0.297, 0.279, 0.424],
        [0.289, 0.281, 0.430],
    ],
    "thalamus": [
        [0.304, 0.242, 0.454],
        [0.309, 0.244, 0.447],
        [0.301, 0.246, 0.453],
    ],
    "hypothalamus": [
        [0.318, 0.263, 0.419],
        [0.324, 0.266, 0.410],
        [0.315, 0.268, 0.417],
    ],
    "brainstem": [
        [0.310, 0.270, 0.420],
        [0.316, 0.272, 0.412],
        [0.307, 0.275, 0.418],
    ],
}


def normalize(rows: list[list[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    return array / array.sum(axis=1, keepdims=True)


def mean_state(rows: np.ndarray) -> np.ndarray:
    mean = rows.mean(axis=0)
    return mean / mean.sum()


def squared_loss(rows_by_region: dict[str, np.ndarray], references: dict[str, np.ndarray]) -> float:
    total = 0.0
    for region, rows in rows_by_region.items():
        ref = references[region]
        total += float(np.sum((rows - ref) ** 2))
    return total


def main() -> None:
    rows_by_region = {region: normalize(samples) for region, samples in REFERENCE_SAMPLES.items()}
    all_rows = np.vstack(list(rows_by_region.values()))

    global_ref = mean_state(all_rows)
    regional_refs = {region: mean_state(rows) for region, rows in rows_by_region.items()}

    class_refs = {}
    for region_class in sorted(set(REGION_CLASS.values())):
        class_rows = np.vstack(
            [rows for region, rows in rows_by_region.items() if REGION_CLASS[region] == region_class]
        )
        class_refs[region_class] = mean_state(class_rows)

    global_predictions = {region: global_ref for region in rows_by_region}
    class_predictions = {region: class_refs[REGION_CLASS[region]] for region in rows_by_region}
    regional_predictions = regional_refs

    losses = {
        "global": squared_loss(rows_by_region, global_predictions),
        "class": squared_loss(rows_by_region, class_predictions),
        "regional": squared_loss(rows_by_region, regional_predictions),
    }

    refs = {
        region: {
            "class": REGION_CLASS[region],
            "regional_p_star": regional_refs[region].tolist(),
            "class_p_star": class_refs[REGION_CLASS[region]].tolist(),
            "global_p_star": global_ref.tolist(),
        }
        for region in rows_by_region
    }

    out = {
        "criterion": "compare global, class, and regional reference states p_r^*",
        "losses": losses,
        "class_over_global": losses["class"] / losses["global"],
        "regional_over_global": losses["regional"] / losses["global"],
        "regional_over_class": losses["regional"] / losses["class"],
        "global_p_star": global_ref.tolist(),
        "class_p_star": {name: value.tolist() for name, value in class_refs.items()},
        "regions": refs,
        "interpretation": {
            "global": "one p^* for all regions",
            "class": "one p^* per anatomical class",
            "regional": "one p_r^* per region",
            "gate": "use the simplest level that remains predictive under holdout",
        },
    }

    out_path = Path(__file__).with_name("regional_reference_state_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Regional reference-state pilot")
    print(f"  L_global   = {losses['global']:.8f}")
    print(f"  L_class    = {losses['class']:.8f}")
    print(f"  L_regional = {losses['regional']:.8f}")
    print(f"  class/global = {out['class_over_global']:.6f}")
    print(f"  regional/global = {out['regional_over_global']:.6f}")
    print(f"  regional/class = {out['regional_over_class']:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
