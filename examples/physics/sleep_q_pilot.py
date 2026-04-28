"""Estimate a minimal q_sleep coordinate from ds004902 pilot summary values.

The raw ds004902 EEG data are not stored in this repository. The values below
are the previously computed pilot summaries from brain_cosmos.py:

- posterior alpha reactivity ratio: NS 2.1224, SD 1.2490
- Stanford Sleepiness Scale: NS 2.0, SD 5.0
- PVT mean RT proxy: NS 320 ms, SD 359 ms

This script keeps the coordinate transform explicit and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path


PILOT = {
    "posterior_alpha_reactivity_ratio": {"normal_sleep": 2.1224, "sleep_deprived": 1.2490},
    "pvt_mean_rt_ms": {"normal_sleep": 320.0, "sleep_deprived": 359.0},
    "stanford_sleepiness_scale": {"normal_sleep": 2.0, "sleep_deprived": 5.0},
}


def q_sleep_coordinates() -> dict[str, float]:
    alpha = PILOT["posterior_alpha_reactivity_ratio"]
    pvt = PILOT["pvt_mean_rt_ms"]
    sss = PILOT["stanford_sleepiness_scale"]

    z_alpha = (alpha["normal_sleep"] - alpha["sleep_deprived"]) / alpha["normal_sleep"]
    z_pvt = (pvt["sleep_deprived"] - pvt["normal_sleep"]) / pvt["normal_sleep"]
    z_sss = (sss["sleep_deprived"] - sss["normal_sleep"]) / 6.0
    z_mean = (z_alpha + z_pvt + z_sss) / 3.0
    z_geom = (z_alpha * z_pvt * z_sss) ** (1.0 / 3.0)

    return {
        "z_alpha": z_alpha,
        "z_pvt": z_pvt,
        "z_sss": z_sss,
        "z_mean": z_mean,
        "z_geometric_mean": z_geom,
        "alpha_ratio_sd_over_ns": alpha["sleep_deprived"] / alpha["normal_sleep"],
    }


def main() -> None:
    result = {
        "dataset": "ds004902 pilot summary",
        "input": PILOT,
        "q_sleep": q_sleep_coordinates(),
        "interpretation": {
            "z_alpha": "fractional loss of posterior alpha reactivity",
            "z_pvt": "fractional slowing of PVT mean reaction time",
            "z_sss": "Stanford Sleepiness Scale increase normalized by the 1-7 range",
            "z_mean": "simple three-proxy burden coordinate",
            "z_geometric_mean": "agreement-weighted three-proxy burden coordinate",
        },
    }
    out_path = Path(__file__).with_name("sleep_q_pilot_results.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("q_sleep pilot from ds004902 summary")
    for key, value in result["q_sleep"].items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
