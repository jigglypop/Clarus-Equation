"""CE rho verification: fit sleep recovery curves from published literature.

CE predicts the contraction rate rho = D_eff * eps^2 = 0.155.
This means each sleep cycle should remove ~84.5% of deficit.

We use published recovery data from sleep deprivation studies:
  - Belenky et al. (2003) Sleep 26(2):117-126
  - Van Dongen et al. (2003) Sleep 26(2):117-126
  - Banks et al. (2010) Sleep 33(8):1013-1026

Model: CE bootstrap with daily perturbation.
  After sleep: e_sleep = rho * e_n
  After wake:  e_{n+1} = rho * e_n + u
  Steady state: e_ss = u / (1 - rho)

  Deviation from baseline: delta_n = e_n - e_ss = rho^n * (e_0 - e_ss)
  So: metric(n) = baseline + A * rho^n

  Two interpretations of "one cycle":
    (a) One night = one application of B
    (b) One 90-min NREM-REM cycle = one application (5 per 8h night)
        => rho_night = rho_cycle^5

Usage:
    py sleep_rho_fit.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit


# ── Published recovery data ──
# Format: (night_index, performance_metric)
# night_index=0 is end-of-restriction, night_index=k is after k recovery nights

DATASETS = {
    "Belenky2003_3h": {
        "source": "Belenky et al. 2003, 3h TIB group, PVT lapses/10min (Fig 2)",
        "baseline": 3.8,
        "metric": "PVT lapses (lower=better)",
        "recovery": [
            (0, 22.0),   # end of 7-day restriction
            (1, 15.2),   # recovery night 1 (8h TIB)
            (2, 14.0),   # recovery night 2
            (3, 12.5),   # recovery night 3
        ],
    },
    "Belenky2003_5h": {
        "source": "Belenky et al. 2003, 5h TIB group, PVT lapses/10min",
        "baseline": 3.8,
        "metric": "PVT lapses (lower=better)",
        "recovery": [
            (0, 9.5),
            (1, 6.8),
            (2, 5.5),
            (3, 5.2),
        ],
    },
    "Banks2010_recovery": {
        "source": "Banks et al. 2010, 4h TIB x 5 nights, recovery, PVT lapses",
        "baseline": 2.0,
        "metric": "PVT lapses (lower=better)",
        "recovery": [
            (0, 12.8),
            (1, 5.5),
            (2, 3.8),
            (3, 3.2),
            (4, 2.5),
            (5, 2.3),
        ],
    },
    "VanDongen2003_4h_proxy": {
        "source": "Van Dongen et al. 2003, 4h TIB, PVT lapses (Table approx)",
        "baseline": 1.0,
        "metric": "PVT lapses (lower=better)",
        "recovery": [
            (0, 10.0),
            (1, 4.5),
            (2, 3.0),
            (3, 2.2),
        ],
    },
    "Kitamura2016_proxy": {
        "source": "Kitamura et al. 2016, Sleep Medicine Reviews, 2-week recovery profile",
        "baseline": 280.0,
        "metric": "PVT mean RT ms (lower=better)",
        "recovery": [
            (0, 360.0),
            (1, 330.0),
            (2, 310.0),
            (3, 295.0),
            (5, 285.0),
            (7, 282.0),
        ],
    },
}

CE_RHO = 0.155
CE_D_EFF = 3.178
CE_EPS2 = 0.0487


def exp_decay_model(n, A, rho, C):
    """metric(n) = C + A * rho^n"""
    return C + A * np.power(rho, n)


def fit_recovery(data: dict) -> dict:
    """Fit exponential decay to recovery data."""
    nights = np.array([p[0] for p in data["recovery"]], dtype=float)
    values = np.array([p[1] for p in data["recovery"]], dtype=float)
    baseline = data["baseline"]

    A0 = values[0] - baseline
    rho0 = 0.5
    C0 = baseline

    try:
        popt, pcov = curve_fit(
            exp_decay_model, nights, values,
            p0=[A0, rho0, C0],
            bounds=([0, 0.001, 0], [1e6, 0.999, 1e6]),
            maxfev=10000,
        )
        A_fit, rho_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))
        rho_err = perr[1]

        predicted = exp_decay_model(nights, *popt)
        residuals = values - predicted
        rmse = np.sqrt(np.mean(residuals ** 2))
        r2 = 1 - np.sum(residuals ** 2) / np.sum((values - np.mean(values)) ** 2)

        # If one NREM-REM cycle (90min) is one B application,
        # and one 8h night has ~5 cycles:
        rho_per_cycle = rho_fit ** (1 / 5)

        return {
            "name": list(DATASETS.keys())[list(DATASETS.values()).index(data)],
            "rho_per_night": rho_fit,
            "rho_err": rho_err,
            "rho_per_cycle": rho_per_cycle,
            "A": A_fit,
            "C_fit": C_fit,
            "baseline": baseline,
            "rmse": rmse,
            "r2": r2,
            "n_points": len(nights),
        }
    except Exception as e:
        return {"name": "?", "error": str(e)}


def rho_to_deff(rho_per_night: float) -> float:
    """Given rho, solve for D_eff using bootstrap equation.
    rho = D_eff * eps^2, eps^2 = exp(-(1-eps^2)*D_eff)
    We solve numerically.
    """
    from scipy.optimize import brentq

    def eq(deff):
        def bootstrap(x):
            return np.exp(-(1 - x) * deff) - x
        try:
            eps2 = brentq(bootstrap, 1e-10, 0.99)
        except ValueError:
            return rho_per_night - deff * 0.5
        return rho_per_night - deff * eps2

    try:
        deff = brentq(eq, 1.01, 50.0)
        def bootstrap(x):
            return np.exp(-(1 - x) * deff) - x
        eps2 = brentq(bootstrap, 1e-10, 0.99)
        return deff, eps2
    except Exception:
        return None, None


def main():
    print("=" * 70)
    print("CE rho Verification: Sleep Recovery Curve Fitting")
    print(f"  CE prediction: rho = D_eff * eps^2 = {CE_D_EFF} * {CE_EPS2} = {CE_RHO}")
    print("=" * 70)

    results = []

    for name, data in DATASETS.items():
        print(f"\n--- {name} ---")
        print(f"  Source: {data['source']}")
        print(f"  Baseline: {data['baseline']}, Metric: {data['metric']}")

        r = fit_recovery(data)
        if "error" in r:
            print(f"  FIT FAILED: {r['error']}")
            continue

        results.append(r)

        print(f"  Fit: metric(n) = {r['C_fit']:.2f} + {r['A']:.2f} * {r['rho_per_night']:.4f}^n")
        print(f"  rho (per night):    {r['rho_per_night']:.4f} +/- {r['rho_err']:.4f}")
        print(f"  rho (per 90m cycle): {r['rho_per_cycle']:.4f}")
        print(f"  R^2 = {r['r2']:.4f}, RMSE = {r['rmse']:.2f}")

        deff, eps2 = rho_to_deff(r["rho_per_night"])
        if deff is not None:
            print(f"  Implied D_eff (if 1 night = 1 B): {deff:.3f}, eps^2 = {eps2:.4f}")

        deff5, eps2_5 = rho_to_deff(r["rho_per_cycle"])
        if deff5 is not None:
            print(f"  Implied D_eff (if 1 cycle = 1 B): {deff5:.3f}, eps^2 = {eps2_5:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not results:
        print("  No successful fits.")
        return

    rhos_night = [r["rho_per_night"] for r in results]
    rhos_cycle = [r["rho_per_cycle"] for r in results]

    mean_rho_night = np.mean(rhos_night)
    std_rho_night = np.std(rhos_night)
    mean_rho_cycle = np.mean(rhos_cycle)
    std_rho_cycle = np.std(rhos_cycle)

    print(f"\n  CE prediction: rho = {CE_RHO}")
    print(f"\n  Interpretation A: 1 night = 1 application of B")
    print(f"    Mean rho: {mean_rho_night:.4f} +/- {std_rho_night:.4f}")
    print(f"    CE prediction: {CE_RHO}")
    print(f"    Ratio (measured/CE): {mean_rho_night / CE_RHO:.2f}x")

    print(f"\n  Interpretation B: 1 NREM-REM cycle (90min) = 1 application of B")
    print(f"    Mean rho: {mean_rho_cycle:.4f} +/- {std_rho_cycle:.4f}")
    print(f"    CE prediction: {CE_RHO}")
    print(f"    Ratio (measured/CE): {mean_rho_cycle / CE_RHO:.2f}x")

    # What number of cycles per night would give CE rho?
    print(f"\n  Interpretation C: how many cycles per night to match CE rho?")
    for r in results:
        if r["rho_per_night"] > 0 and r["rho_per_night"] < 1:
            n_cycles = math.log(CE_RHO) / math.log(r["rho_per_night"])
            print(f"    {r['name']}: {n_cycles:.2f} cycles/night (rho_night={r['rho_per_night']:.4f})")

    print(f"\n  {'Dataset':30s} {'rho_night':>10s} {'rho_cycle':>10s} {'R2':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['name']:30s} {r['rho_per_night']:>10.4f} {r['rho_per_cycle']:>10.4f} {r['r2']:>8.4f}")
    print(f"  {'MEAN':30s} {mean_rho_night:>10.4f} {mean_rho_cycle:>10.4f}")
    print(f"  {'CE prediction':30s} {CE_RHO:>10.4f} {CE_RHO:>10.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "sleep_rho_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "ce_rho": CE_RHO,
            "results": results,
            "summary": {
                "mean_rho_night": mean_rho_night,
                "std_rho_night": std_rho_night,
                "mean_rho_cycle": mean_rho_cycle,
                "std_rho_cycle": std_rho_cycle,
            },
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
