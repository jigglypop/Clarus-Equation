"""Prediction-gate pilot for the brain state p_r.

The documentation defines p_r = (x_a, x_s, x_b), but that definition should not
be accepted unless it predicts held-out states better than simpler baselines.
This pilot encodes the comparison:

    L_state < min(L_mean, L_single)

where the state model predicts all three components, the mean baseline predicts
the training mean, and the single-proxy baseline only tracks x_a while assigning
the remaining mass to fixed structural/background proportions.
"""

from __future__ import annotations

import json
from pathlib import Path


HOLDOUT = [
    {
        "region": "visual",
        "observed": [0.502, 0.185, 0.313],
        "state_model": [0.493, 0.191, 0.316],
        "single_active": 0.500,
    },
    {
        "region": "default_mode",
        "observed": [0.365, 0.324, 0.311],
        "state_model": [0.358, 0.318, 0.324],
        "single_active": 0.360,
    },
    {
        "region": "hippocampus",
        "observed": [0.377, 0.257, 0.366],
        "state_model": [0.371, 0.264, 0.365],
        "single_active": 0.380,
    },
    {
        "region": "thalamus",
        "observed": [0.362, 0.311, 0.327],
        "state_model": [0.354, 0.304, 0.342],
        "single_active": 0.360,
    },
]

TRAIN_MEAN = [0.332, 0.258, 0.410]
SINGLE_STRUCTURAL_FRACTION = 0.42


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def squared_loss(observed: list[float], predicted: list[float]) -> float:
    return sum((obs - pred) ** 2 for obs, pred in zip(observed, predicted))


def single_proxy_prediction(active: float) -> list[float]:
    remaining = 1.0 - active
    structural = remaining * SINGLE_STRUCTURAL_FRACTION
    background = remaining - structural
    return [active, structural, background]


def main() -> None:
    rows = []
    losses = {"state": 0.0, "mean": 0.0, "single": 0.0}

    for item in HOLDOUT:
        observed = normalize(item["observed"])
        state_model = normalize(item["state_model"])
        mean_model = TRAIN_MEAN
        single_model = single_proxy_prediction(item["single_active"])

        state_loss = squared_loss(observed, state_model)
        mean_loss = squared_loss(observed, mean_model)
        single_loss = squared_loss(observed, single_model)

        losses["state"] += state_loss
        losses["mean"] += mean_loss
        losses["single"] += single_loss

        rows.append(
            {
                "region": item["region"],
                "observed": observed,
                "state_model": state_model,
                "mean_model": mean_model,
                "single_model": single_model,
                "loss_state": state_loss,
                "loss_mean": mean_loss,
                "loss_single": single_loss,
            }
        )

    best_baseline = min(losses["mean"], losses["single"])
    out = {
        "criterion": "L_state < min(L_mean, L_single)",
        "losses": losses,
        "loss_ratios": {
            "state_over_mean": losses["state"] / losses["mean"],
            "state_over_single": losses["state"] / losses["single"],
            "state_over_best_baseline": losses["state"] / best_baseline,
        },
        "passed_prediction_gate": losses["state"] < best_baseline,
        "holdout_rows": rows,
        "interpretation": {
            "state": "predicts x_a, x_s, and x_b jointly",
            "mean": "uses the training mean p_r",
            "single": "uses only active-state information and fixed residual proportions",
        },
    }

    out_path = Path(__file__).with_name("brain_state_prediction_gate_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Brain state prediction-gate pilot")
    print(f"  L_state  = {losses['state']:.6f}")
    print(f"  L_mean   = {losses['mean']:.6f}")
    print(f"  L_single = {losses['single']:.6f}")
    print(f"  state/best_baseline = {out['loss_ratios']['state_over_best_baseline']:.6f}")
    print(f"  passed = {out['passed_prediction_gate']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
