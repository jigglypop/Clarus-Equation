"""Map synaptic plasticity events to the macroscopic structural state x_s.

This pilot keeps the biology minimal:

1. STDP determines the sign and size of local weight changes.
2. A third factor gates whether eligible changes are expressed.
3. Regional structural burden is the normalized sum of expressed weight change
   plus a maintenance cost for existing weights.
4. A softmax maps active, structural, and background scores to p_r.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ETA = 0.02
A_PLUS = 1.0
A_MINUS = 1.05
TAU_PLUS_MS = 20.0
TAU_MINUS_MS = 20.0
LAMBDA_STRUCT = 0.01


REGIONS = {
    "cortex": {
        "active_score": 0.35,
        "background_score": 0.55,
        "synapses": [
            {"delta_t_ms": 10.0, "third_factor": 1.0, "weight": 0.62},
            {"delta_t_ms": 15.0, "third_factor": 0.8, "weight": 0.58},
            {"delta_t_ms": -12.0, "third_factor": 0.7, "weight": 0.51},
            {"delta_t_ms": 35.0, "third_factor": 0.4, "weight": 0.47},
        ],
    },
    "hippocampus": {
        "active_score": 0.42,
        "background_score": 0.45,
        "synapses": [
            {"delta_t_ms": 8.0, "third_factor": 1.2, "weight": 0.71},
            {"delta_t_ms": 18.0, "third_factor": 1.0, "weight": 0.68},
            {"delta_t_ms": -10.0, "third_factor": 0.9, "weight": 0.55},
            {"delta_t_ms": 22.0, "third_factor": 0.7, "weight": 0.60},
        ],
    },
    "thalamus": {
        "active_score": 0.28,
        "background_score": 0.62,
        "synapses": [
            {"delta_t_ms": 12.0, "third_factor": 0.5, "weight": 0.50},
            {"delta_t_ms": -16.0, "third_factor": 0.5, "weight": 0.49},
            {"delta_t_ms": 28.0, "third_factor": 0.3, "weight": 0.45},
            {"delta_t_ms": -30.0, "third_factor": 0.2, "weight": 0.44},
        ],
    },
}


def stdp_kernel(delta_t_ms: float) -> float:
    if delta_t_ms > 0.0:
        return A_PLUS * math.exp(-delta_t_ms / TAU_PLUS_MS)
    if delta_t_ms < 0.0:
        return -A_MINUS * math.exp(delta_t_ms / TAU_MINUS_MS)
    return 0.0


def softmax(scores: list[float]) -> list[float]:
    maximum = max(scores)
    exp_scores = [math.exp(score - maximum) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]


def region_summary(region: dict[str, object]) -> dict[str, object]:
    synapses = region["synapses"]
    events = []
    plasticity_load = 0.0
    signed_change = 0.0
    maintenance_load = 0.0

    for synapse in synapses:
        delta_t_ms = float(synapse["delta_t_ms"])
        third_factor = float(synapse["third_factor"])
        weight = float(synapse["weight"])
        eligibility = stdp_kernel(delta_t_ms)
        delta_w = ETA * third_factor * eligibility
        events.append(
            {
                "delta_t_ms": delta_t_ms,
                "third_factor": third_factor,
                "weight": weight,
                "eligibility": eligibility,
                "delta_w": delta_w,
            }
        )
        plasticity_load += abs(delta_w)
        signed_change += delta_w
        maintenance_load += abs(weight)

    structural_score = plasticity_load + LAMBDA_STRUCT * maintenance_load
    p = softmax(
        [
            float(region["active_score"]),
            structural_score,
            float(region["background_score"]),
        ]
    )

    return {
        "events": events,
        "plasticity_load": plasticity_load,
        "signed_change": signed_change,
        "maintenance_load": maintenance_load,
        "structural_score": structural_score,
        "p": {
            "x_a": p[0],
            "x_s": p[1],
            "x_b": p[2],
        },
    }


def main() -> None:
    summaries = {name: region_summary(region) for name, region in REGIONS.items()}
    ranking = sorted(
        (
            {
                "region": name,
                "structural_score": summary["structural_score"],
                "x_s": summary["p"]["x_s"],
            }
            for name, summary in summaries.items()
        ),
        key=lambda item: item["x_s"],
        reverse=True,
    )

    out = {
        "model": "Delta w_ij = eta m e_ij; S_r = sum |Delta w_ij| + lambda_struct sum |w_ij|; p_r = softmax(A_r,S_r,B_r)",
        "parameters": {
            "eta": ETA,
            "a_plus": A_PLUS,
            "a_minus": A_MINUS,
            "tau_plus_ms": TAU_PLUS_MS,
            "tau_minus_ms": TAU_MINUS_MS,
            "lambda_struct": LAMBDA_STRUCT,
        },
        "regions": summaries,
        "structural_ranking": ranking,
        "interpretation": {
            "positive_delta_t": "pre before post produces positive eligibility",
            "negative_delta_t": "post before pre produces negative eligibility",
            "third_factor": "scales expressed plasticity without changing the STDP sign when positive",
            "x_s": "macro structural burden after softmax normalization",
        },
    }

    out_path = Path(__file__).with_name("synapse_structural_burden_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Synapse structural burden pilot")
    print("  region       plasticity_load  structural_score  x_s")
    for item in ranking:
        name = item["region"]
        summary = summaries[name]
        print(
            f"  {name:11s}  "
            f"{summary['plasticity_load']:.6f}         "
            f"{summary['structural_score']:.6f}          "
            f"{summary['p']['x_s']:.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
