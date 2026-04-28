"""Infer sleep-forcing sensitivity bounds from observable state shifts.

Given the ds004902 pilot burden coordinate z_sleep and the minimal forcing

    Delta x_a = alpha_r * z_sleep,
    Delta x_b = -alpha_r * z_sleep,

this script computes which regional sensitivity alpha_r is implied by an
observed stage-1 shift. It also reports the alpha_r needed to make small,
medium, and large shifts observable.
"""

from __future__ import annotations

import json
from pathlib import Path


Q_RESULT_PATH = Path(__file__).with_name("sleep_q_pilot_results.json")


def infer_alpha(delta_x: float, z_sleep: float) -> float:
    if z_sleep <= 0:
        raise ValueError("z_sleep must be positive.")
    return delta_x / z_sleep


def main() -> None:
    q_payload = json.loads(Q_RESULT_PATH.read_text(encoding="utf-8"))
    z_values = {
        "z_mean": float(q_payload["q_sleep"]["z_mean"]),
        "z_geometric_mean": float(q_payload["q_sleep"]["z_geometric_mean"]),
    }
    target_shifts = [0.005, 0.010, 0.020, 0.030, 0.050]

    payload = {
        "model": "Delta x_a = alpha_r * z_sleep, Delta x_b = -alpha_r * z_sleep",
        "z_sleep": z_values,
        "target_shifts": {
            f"{shift:.3f}": {
                name: infer_alpha(shift, z_sleep)
                for name, z_sleep in z_values.items()
            }
            for shift in target_shifts
        },
        "interpretation": {
            "0.005": "very small but detectable in clean within-subject data",
            "0.010": "small stage-1 state movement",
            "0.020": "clear state movement",
            "0.030": "large sleep-deprivation movement",
            "0.050": "very large movement, comparable to the CE active prior scale",
        },
    }

    out_path = Path(__file__).with_name("sleep_sensitivity_bounds_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Sleep forcing sensitivity bounds")
    print("  target_delta_x  alpha_from_z_mean  alpha_from_z_geom")
    for shift in target_shifts:
        row = payload["target_shifts"][f"{shift:.3f}"]
        print(
            f"  {shift:14.3f}"
            f"  {row['z_mean']:17.6f}"
            f"  {row['z_geometric_mean']:17.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
