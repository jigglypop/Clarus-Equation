"""Infer regional background-reserve sensitivity from resting-state shifts.

This pilot closes the x_b route:

    u_rest -> B_r -> p_r -> x_b,r

The burden variable u_rest can represent sleep deprivation, unstable resting
baseline, or metabolic stress. A larger burden lowers the background-reserve
score. Given two burden levels and the observed background component, the
regional reserve sensitivity chi_r can be recovered in score space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


BURDEN_LEVELS = [0.0, 1.0, 2.0]

REGIONS = {
    "default_mode": {"a": 0.16, "s": 0.04, "b0": 0.72, "chi": 0.36},
    "thalamus": {"a": 0.18, "s": 0.03, "b0": 0.68, "chi": 0.30},
    "brainstem": {"a": 0.14, "s": 0.05, "b0": 0.66, "chi": 0.24},
    "prefrontal": {"a": 0.20, "s": 0.06, "b0": 0.64, "chi": 0.20},
}


def softmax(scores: list[float]) -> list[float]:
    maximum = max(scores)
    exp_scores = [math.exp(score - maximum) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def background_state(region: dict[str, float], burden: float) -> dict[str, float]:
    active_score = region["a"]
    structural_score = region["s"]
    background_score = region["b0"] - region["chi"] * burden
    x_a, x_s, x_b = softmax([active_score, structural_score, background_score])
    return {
        "burden": burden,
        "active_score": active_score,
        "structural_score": structural_score,
        "background_score": background_score,
        "x_a": x_a,
        "x_s": x_s,
        "x_b": x_b,
    }


def background_score_from_state(x_b: float, active_score: float, structural_score: float) -> float:
    rest = math.exp(active_score) + math.exp(structural_score)
    return math.log(x_b / (1.0 - x_b)) + math.log(rest)


def infer_chi(
    x_b_low: float,
    x_b_high: float,
    active_score: float,
    structural_score: float,
    burden_low: float,
    burden_high: float,
) -> float:
    background_low = background_score_from_state(x_b_low, active_score, structural_score)
    background_high = background_score_from_state(x_b_high, active_score, structural_score)
    return (background_low - background_high) / (burden_high - burden_low)


def main() -> None:
    regions = {}
    for name, region in REGIONS.items():
        trajectory = [background_state(region, burden) for burden in BURDEN_LEVELS]
        chi_hat = infer_chi(
            trajectory[0]["x_b"],
            trajectory[-1]["x_b"],
            region["a"],
            region["s"],
            BURDEN_LEVELS[0],
            BURDEN_LEVELS[-1],
        )
        regions[name] = {
            "input": region,
            "trajectory": trajectory,
            "delta_x_b": trajectory[-1]["x_b"] - trajectory[0]["x_b"],
            "delta_x_a": trajectory[-1]["x_a"] - trajectory[0]["x_a"],
            "delta_x_s": trajectory[-1]["x_s"] - trajectory[0]["x_s"],
            "chi_hat": chi_hat,
        }

    ranking = sorted(
        (
            {
                "region": name,
                "delta_x_b": payload["delta_x_b"],
                "chi_hat": payload["chi_hat"],
            }
            for name, payload in regions.items()
        ),
        key=lambda item: item["chi_hat"],
        reverse=True,
    )

    out = {
        "model": "B_r(u)=B_r0-chi_r u_rest; p_r=softmax(A_r,S_r,B_r)",
        "burden_levels": BURDEN_LEVELS,
        "regions": regions,
        "reserve_sensitivity_ranking": ranking,
        "interpretation": {
            "chi_r": "regional sensitivity of background reserve to resting burden",
            "x_b": "softmax-normalized background-reserve component",
            "inverse": "observed x_b at two burden levels recovers chi_r when A_r and S_r are known or estimated",
        },
    }

    out_path = Path(__file__).with_name("background_reserve_state_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Background reserve-state pilot")
    print("  region        delta_x_b  delta_x_a  chi_hat")
    for item in ranking:
        name = item["region"]
        payload = regions[name]
        print(
            f"  {name:12s}  "
            f"{payload['delta_x_b']:+.6f}  "
            f"{payload['delta_x_a']:+.6f}  "
            f"{payload['chi_hat']:.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
