"""Exact affine PMNS interval screen; no Gaussian likelihood reconstruction.

The input intervals are 1D profiles, not independent observations. Their box
intersection has no assigned joint confidence level. The four variants overlap
in data and must not be added. No dimension-to-flavor dynamics is supplied here.
"""
from fractions import Fraction as F
import json
from pathlib import Path


OFFSET = (F(1, 3), F(1, 2), F(0))
SLOPE = (-F(1, 8), F(7, 16), F(1, 8))


def predict(delta):
    d = F(str(delta))
    if not 0 <= d <= F(1, 4):
        raise ValueError("delta must be in [0,1/4]")
    return tuple(a + b*d for a, b in zip(OFFSET, SLOPE))


def intersect_intervals(intervals):
    """Invert each affine constraint with rational arithmetic, including sign."""
    if len(intervals) != 3:
        raise ValueError("three ordered angle intervals required")
    constraints = [(F(0), F(1, 4))]
    for (low, high), a, b in zip(intervals, OFFSET, SLOPE):
        low, high = F(str(low)), F(str(high))
        if not 0 <= low <= high <= 1:
            raise ValueError("ordered probability intervals required")
        constraints.append(tuple(sorted(((low-a)/b, (high-a)/b))))
    low, high = max(x[0] for x in constraints), min(x[1] for x in constraints)
    feasible = low <= high
    return {
        "nonempty": feasible,
        "delta_lower": float(low), "delta_upper": float(high),
        "exact_bounds": [str(low), str(high)],
        "constraint_intervals": [[str(a), str(b)] for a, b in constraints],
        "witness_delta": float((low+high)/2) if feasible else None,
        "joint_confidence_level": None,
    }


def build_report():
    source = json.loads(Path(__file__).with_name("nufit60_pmns_intervals.json").read_text())
    variants = {}
    for name, data in source["variants"].items():
        variants[name] = {
            "delta_from_each_central_angle": [
                float((F(str(y))-a)/b)
                for y, a, b in zip(data["central"], OFFSET, SLOPE)],
            "one_sigma_box": intersect_intervals(data["one_sigma"]),
            "three_sigma_box": intersect_intervals(data["three_sigma"]),
        }
    return {"source": source, "variants": variants,
            "historical_delta": 0.17776,
            "historical_prediction": [float(x) for x in predict("0.17776")],
            "joint_rmse": None, "scientific_success": False,
            "status": "exploratory_interval_screen_only"}


if __name__ == "__main__":
    report = build_report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    for name, row in report["variants"].items():
        print(name, row["delta_from_each_central_angle"],
              row["one_sigma_box"]["nonempty"], row["three_sigma_box"]["exact_bounds"])
