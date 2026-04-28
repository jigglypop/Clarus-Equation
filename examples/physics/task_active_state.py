"""Infer regional task sensitivity from active-state shifts.

This pilot closes the x_a route in the brain docs:

    u_task -> A_r -> p_r -> x_a,r

The state is normalized with the same softmax used in the observation
definition. Given two task loads and the observed active component, the
regional task sensitivity beta_r can be recovered in score space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


TASK_LOADS = [0.0, 1.0, 2.0]

REGIONS = {
    "visual": {"a0": 0.18, "s": 0.02, "b": 0.55, "beta": 0.42},
    "motor": {"a0": 0.16, "s": 0.03, "b": 0.52, "beta": 0.34},
    "prefrontal": {"a0": 0.14, "s": 0.04, "b": 0.50, "beta": 0.26},
    "hippocampus": {"a0": 0.12, "s": 0.05, "b": 0.48, "beta": 0.18},
}


def softmax(scores: list[float]) -> list[float]:
    maximum = max(scores)
    exp_scores = [math.exp(score - maximum) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def active_state(region: dict[str, float], task_load: float) -> dict[str, float]:
    active_score = region["a0"] + region["beta"] * task_load
    structural_score = region["s"]
    background_score = region["b"]
    x_a, x_s, x_b = softmax([active_score, structural_score, background_score])
    return {
        "task_load": task_load,
        "active_score": active_score,
        "structural_score": structural_score,
        "background_score": background_score,
        "x_a": x_a,
        "x_s": x_s,
        "x_b": x_b,
    }


def active_score_from_state(x_a: float, structural_score: float, background_score: float) -> float:
    rest = math.exp(structural_score) + math.exp(background_score)
    return math.log(x_a / (1.0 - x_a)) + math.log(rest)


def infer_beta(
    x_a_low: float,
    x_a_high: float,
    structural_score: float,
    background_score: float,
    task_low: float,
    task_high: float,
) -> float:
    active_low = active_score_from_state(x_a_low, structural_score, background_score)
    active_high = active_score_from_state(x_a_high, structural_score, background_score)
    return (active_high - active_low) / (task_high - task_low)


def main() -> None:
    regions = {}
    for name, region in REGIONS.items():
        trajectory = [active_state(region, task_load) for task_load in TASK_LOADS]
        beta_hat = infer_beta(
            trajectory[0]["x_a"],
            trajectory[-1]["x_a"],
            region["s"],
            region["b"],
            TASK_LOADS[0],
            TASK_LOADS[-1],
        )
        regions[name] = {
            "input": region,
            "trajectory": trajectory,
            "delta_x_a": trajectory[-1]["x_a"] - trajectory[0]["x_a"],
            "delta_x_b": trajectory[-1]["x_b"] - trajectory[0]["x_b"],
            "beta_hat": beta_hat,
        }

    ranking = sorted(
        (
            {
                "region": name,
                "delta_x_a": payload["delta_x_a"],
                "beta_hat": payload["beta_hat"],
            }
            for name, payload in regions.items()
        ),
        key=lambda item: item["delta_x_a"],
        reverse=True,
    )

    out = {
        "model": "A_r(u)=A_r0+beta_r u_task; p_r=softmax(A_r,S_r,B_r)",
        "task_loads": TASK_LOADS,
        "regions": regions,
        "active_shift_ranking": ranking,
        "interpretation": {
            "beta_r": "regional task sensitivity in active-score space",
            "x_a": "softmax-normalized active-state component",
            "inverse": "observed x_a at two task loads recovers beta_r when S_r and B_r are known or estimated",
        },
    }

    out_path = Path(__file__).with_name("task_active_state_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Task active-state pilot")
    print("  region       delta_x_a  delta_x_b  beta_hat")
    for item in ranking:
        name = item["region"]
        payload = regions[name]
        print(
            f"  {name:11s}  "
            f"{payload['delta_x_a']:+.6f}  "
            f"{payload['delta_x_b']:+.6f}  "
            f"{payload['beta_hat']:.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
