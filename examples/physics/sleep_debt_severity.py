"""Relate residual recovery floor to sleep-restriction severity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FLOOR_PATH = Path(__file__).with_name("sleep_recovery_floor_results.json")

# Approximate restriction metadata from the study labels used in sleep_rho_fit.py.
# The proxy severity is cumulative lost time in bed relative to an 8h reference.
RESTRICTION = {
    "Belenky2003_3h": {"tib_hours": 3.0, "restriction_days": 7.0},
    "Belenky2003_5h": {"tib_hours": 5.0, "restriction_days": 7.0},
    "Banks2010_recovery": {"tib_hours": 4.0, "restriction_days": 5.0},
    "VanDongen2003_4h_proxy": {"tib_hours": 4.0, "restriction_days": 14.0},
    "Kitamura2016_proxy": {"tib_hours": 5.0, "restriction_days": 5.0},
}


def debt_hours(tib_hours: float, restriction_days: float, reference_hours: float = 8.0) -> float:
    return max(0.0, reference_hours - tib_hours) * restriction_days


def main() -> None:
    floor_payload = json.loads(FLOOR_PATH.read_text(encoding="utf-8"))
    rows = []
    for row in floor_payload["rows"]:
        name = row["name"]
        meta = RESTRICTION[name]
        debt = debt_hours(meta["tib_hours"], meta["restriction_days"])
        floor_fraction = max(0.0, float(row["floor_fraction"]))
        rows.append(
            {
                "name": name,
                "tib_hours": meta["tib_hours"],
                "restriction_days": meta["restriction_days"],
                "debt_hours": debt,
                "floor_fraction": floor_fraction,
                "residual_floor": max(0.0, float(row["residual_floor"])),
                "rho_per_night": float(row["rho_per_night"]),
            }
        )

    # Robust minimal rule: fit only rows with nonnegative floors and comparable PVT lapse metrics.
    fit_names = {"Belenky2003_3h", "Belenky2003_5h", "Banks2010_recovery"}
    fit_rows = [row for row in rows if row["name"] in fit_names]
    x = np.asarray([row["debt_hours"] for row in fit_rows], dtype=np.float64)
    y = np.asarray([row["floor_fraction"] for row in fit_rows], dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.asarray([intercept, slope])
    residual = y - predicted
    r2 = 1.0 - float(np.sum(residual**2) / np.sum((y - y.mean()) ** 2))

    threshold_10 = (0.10 - intercept) / slope
    threshold_25 = (0.25 - intercept) / slope
    threshold_50 = (0.50 - intercept) / slope

    out = {
        "model": "floor_fraction ~= intercept + slope * cumulative_sleep_loss_hours",
        "rows": rows,
        "fit_subset": sorted(fit_names),
        "fit": {
            "intercept": float(intercept),
            "slope_per_hour": float(slope),
            "r2": r2,
            "debt_hours_for_floor_0_10": float(threshold_10),
            "debt_hours_for_floor_0_25": float(threshold_25),
            "debt_hours_for_floor_0_50": float(threshold_50),
        },
        "caveat": "This is a small pilot over heterogeneous literature summaries, not a final dose-response fit.",
    }

    out_path = Path(__file__).with_name("sleep_debt_severity_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Sleep debt severity pilot")
    print("  dataset                         debt_h  floor_frac  rho_night")
    for row in rows:
        print(
            f"  {row['name']:30s}"
            f"  {row['debt_hours']:6.1f}"
            f"  {row['floor_fraction']:10.4f}"
            f"  {row['rho_per_night']:9.4f}"
        )
    print(f"  fit: floor = {intercept:.4f} + {slope:.5f} * debt_hours, R2={r2:.4f}")
    print(f"  floor 0.10 threshold: {threshold_10:.2f} h")
    print(f"  floor 0.25 threshold: {threshold_25:.2f} h")
    print(f"  floor 0.50 threshold: {threshold_50:.2f} h")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
