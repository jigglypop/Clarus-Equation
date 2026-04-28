"""Estimate residual sleep-debt floors from fitted recovery curves."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_PATH = Path(__file__).with_name("sleep_rho_results.json")


def main() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = []

    for row in payload["results"]:
        baseline = float(row["baseline"])
        c_fit = float(row["C_fit"])
        amplitude = float(row["A"])
        floor = c_fit - baseline
        initial_deficit = amplitude + floor
        floor_fraction = floor / initial_deficit if initial_deficit > 0 else 0.0
        recoverable_fraction = amplitude / initial_deficit if initial_deficit > 0 else 1.0
        rows.append(
            {
                "name": row["name"],
                "baseline": baseline,
                "c_fit": c_fit,
                "recoverable_amplitude": amplitude,
                "residual_floor": floor,
                "initial_deficit": initial_deficit,
                "floor_fraction": floor_fraction,
                "recoverable_fraction": recoverable_fraction,
                "rho_per_night": float(row["rho_per_night"]),
                "r2": float(row["r2"]),
            }
        )

    out = {
        "source": str(RESULT_PATH),
        "model": "y(n) = baseline + floor + A rho^n",
        "rows": rows,
        "interpretation": {
            "residual_floor": "C_fit - baseline; deficit not removed by the fast exponential channel",
            "floor_fraction": "residual_floor / initial_deficit",
            "recoverable_fraction": "A / initial_deficit",
        },
    }

    out_path = Path(__file__).with_name("sleep_recovery_floor_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Sleep recovery residual floors")
    print("  dataset                         floor      floor_frac  recoverable  rho_night")
    for row in rows:
        print(
            f"  {row['name']:30s}"
            f"  {row['residual_floor']:9.4f}"
            f"  {row['floor_fraction']:10.4f}"
            f"  {row['recoverable_fraction']:11.4f}"
            f"  {row['rho_per_night']:9.4f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
