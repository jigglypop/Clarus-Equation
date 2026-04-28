"""Separate acute and chronic sleep-recovery modes from fitted PVT curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


RESULT_PATH = Path(__file__).with_name("sleep_rho_results.json")
CE_RHO_B = 0.155
NIGHTS_PER_B = 1.6


def to_b_scale(rho_night: float) -> float:
    return float(rho_night ** NIGHTS_PER_B)


def recovery_series(rho_night: float, nights: int = 7) -> list[float]:
    return [float(rho_night ** n) for n in range(1, nights + 1)]


def main() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = payload["results"]

    acute_names = {
        "Belenky2003_3h",
        "Banks2010_recovery",
        "VanDongen2003_4h_proxy",
    }
    mild_name = "Belenky2003_5h"
    chronic_name = "Kitamura2016_proxy"

    acute_rhos = np.array(
        [float(row["rho_per_night"]) for row in rows if row["name"] in acute_names],
        dtype=np.float64,
    )
    mild_rho = next(float(row["rho_per_night"]) for row in rows if row["name"] == mild_name)
    chronic_rho = next(float(row["rho_per_night"]) for row in rows if row["name"] == chronic_name)

    acute_mean = float(acute_rhos.mean())
    acute_std = float(acute_rhos.std())
    acute_b = np.array([to_b_scale(x) for x in acute_rhos], dtype=np.float64)
    chronic_b = to_b_scale(chronic_rho)

    # Minimal two-mode decomposition. Acute mode is the CE-compatible recovery;
    # slow mode captures accumulated debt that does not clear in the fast channel.
    modes = {
        "fast": {
            "rho_night_mean": acute_mean,
            "rho_night_std": acute_std,
            "rho_night_values": [float(x) for x in acute_rhos],
            "rho_b_values": [float(x) for x in acute_b],
            "rho_b_min": float(acute_b.min()),
            "rho_b_max": float(acute_b.max()),
            "rho_b_mean": float(acute_b.mean()),
            "ce_rho_b": CE_RHO_B,
        },
        "mild": {
            "rho_night": mild_rho,
            "rho_b": to_b_scale(mild_rho),
        },
        "slow": {
            "rho_night": chronic_rho,
            "rho_b": chronic_b,
        },
    }

    predictions = {
        "fast_residual_by_night": recovery_series(acute_mean),
        "mild_residual_by_night": recovery_series(mild_rho),
        "slow_residual_by_night": recovery_series(chronic_rho),
    }

    out = {
        "source": str(RESULT_PATH),
        "nights_per_b_application": NIGHTS_PER_B,
        "modes": modes,
        "predictions": predictions,
        "interpretation": {
            "fast": "acute sleep-deprivation recovery; CE-compatible B-scale residual",
            "mild": "weaker restriction group; intermediate recovery",
            "slow": "longer accumulated debt proxy; requires an additional slow state",
        },
    }

    out_path = Path(__file__).with_name("sleep_recovery_modes_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Sleep recovery modes")
    print(f"  fast rho_night mean: {acute_mean:.6f} +/- {acute_std:.6f}")
    print(f"  fast rho_B range:    {acute_b.min():.6f} .. {acute_b.max():.6f}")
    print(f"  CE rho_B:            {CE_RHO_B:.6f}")
    print(f"  mild rho_night:      {mild_rho:.6f}, rho_B={to_b_scale(mild_rho):.6f}")
    print(f"  slow rho_night:      {chronic_rho:.6f}, rho_B={chronic_b:.6f}")
    print("  residual after 1/2/3 nights:")
    print(
        "    fast: "
        + ", ".join(f"{x:.4f}" for x in predictions["fast_residual_by_night"][:3])
    )
    print(
        "    slow: "
        + ", ".join(f"{x:.4f}" for x in predictions["slow_residual_by_night"][:3])
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
