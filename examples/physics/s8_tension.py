"""
S_8 tension: consistency check of CE Hubble-tension solution against LSS growth.

CE Hubble tension fit (examples/physics/hubble_tension.py) requires:
  delta_eps_0 = -0.055, xi = 5.0  -->  Omega_m_today = (1 - eps_today)/2 = 0.343

This script tests whether that Omega_m_today is compatible with:
  (a) S_8 = sigma_8 * sqrt(Omega_m/0.3) measurements (Planck CMB vs lensing)
  (b) f sigma_8(z) growth-rate data (BOSS/eBOSS/WiggleZ/6dF/...)

CE clarus field has v_EW-scale VEV << M_Pl, so its perturbation contribution
to mu(a, k) (effective Newton constant) is negligible at LSS scales. Hence
LSS deviation from LCDM is dominated by *background* shift in Omega_m.

ASCII-only output.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cosmology import (
    Background,
    linspace,
    logspace,
    interp_linear,
    solve_growth,
    parse_fsigma8_triplets,
)


# Standard fsigma8 compilation (z, fs8, sigma).  Sources cited in commit msg.
FSIG8_DATA = [
    (0.067, 0.423, 0.055),  # 6dFGS
    (0.10,  0.376, 0.038),  # SDSS veloc
    (0.17,  0.510, 0.060),  # 2dFGRS
    (0.18,  0.360, 0.090),  # GAMA
    (0.25,  0.351, 0.058),  # SDSS LRG
    (0.32,  0.427, 0.056),  # BOSS LOWZ
    (0.37,  0.460, 0.038),  # SDSS LRG
    (0.38,  0.497, 0.045),  # BOSS DR12
    (0.44,  0.413, 0.080),  # WiggleZ
    (0.51,  0.458, 0.038),  # BOSS DR12
    (0.59,  0.488, 0.060),  # BOSS LOWZ+CMASS
    (0.60,  0.390, 0.063),  # WiggleZ
    (0.61,  0.436, 0.034),  # BOSS DR12
    (0.73,  0.437, 0.072),  # WiggleZ
    (0.86,  0.400, 0.110),  # VIPERS PDR-2
    (0.978, 0.379, 0.176),  # eBOSS DR14 quasars
    (1.40,  0.482, 0.116),  # FastSound
    (1.48,  0.300, 0.130),  # eBOSS quasars
]


def s8_amplitude(sigma8_0: float, omega_m0: float) -> float:
    return sigma8_0 * math.sqrt(omega_m0 / 0.3)


def predict_fsig8_curve(
    bg: Background,
    sigma8_0: float,
    z_grid: list[float],
    a_min: float = 1.0e-3,
    n_pts: int = 2001,
    eps_grav: float = 0.0,
) -> list[float]:
    a_grid = logspace(a_min, 1.0, n_pts)
    ln_a_grid = [math.log(a) for a in a_grid]
    mu_grid = [1.0 - eps_grav * (1.0 - bg.omega_m_of_a(a)) for a in a_grid]
    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)
    out = []
    for z in z_grid:
        a = 1.0 / (1.0 + z)
        ln_a = math.log(a)
        d = interp_linear(ln_a_grid, d_norm, ln_a)
        fz = interp_linear(ln_a_grid, f_ln, ln_a)
        out.append(fz * (sigma8_0 * d))
    return out


def chi2_against(data: list[tuple[float, float, float]],
                 pred_at_z: dict) -> tuple[float, int]:
    chi2 = 0.0
    n = 0
    for (z, fs8_obs, sig) in data:
        if sig <= 0.0:
            continue
        if z not in pred_at_z:
            continue
        r = (pred_at_z[z] - fs8_obs) / sig
        chi2 += r * r
        n += 1
    return chi2, n


def report_model(label: str, omega_m0: float, sigma8_0: float,
                  data: list[tuple[float, float, float]],
                  eps_grav: float = 0.0) -> dict:
    bg = Background(omega_m0=omega_m0, omega_l0=1.0 - omega_m0)
    s8 = s8_amplitude(sigma8_0, omega_m0)

    z_data = sorted({z for (z, _, _) in data})
    fs8_pred = predict_fsig8_curve(bg, sigma8_0, z_data, eps_grav=eps_grav)
    pred_at_z = dict(zip(z_data, fs8_pred))

    chi2, n = chi2_against(data, pred_at_z)
    dof = max(n - 1, 1)

    print(f"--- {label} ---")
    print(f"  Omega_m0          = {omega_m0:.4f}")
    print(f"  sigma8_0          = {sigma8_0:.4f}")
    print(f"  S_8 = sigma8*sqrt(Om/0.3) = {s8:.4f}")
    print(f"    Planck S_8     = 0.832 +- 0.013   --> tension = {(s8-0.832)/0.013:+.2f} sigma")
    print(f"    KiDS-1000 S_8  = 0.766 +- 0.020   --> tension = {(s8-0.766)/0.020:+.2f} sigma")
    print(f"    DES-Y3 S_8     = 0.776 +- 0.017   --> tension = {(s8-0.776)/0.017:+.2f} sigma")
    print(f"  fsigma8 chi2/N   = {chi2:.3f}/{n}  (chi2_red = {chi2/dof:.3f})")
    print(f"  eps_grav          = {eps_grav:+.4f}")
    print()
    return {"s8": s8, "chi2": chi2, "n": n, "fs8_pred": pred_at_z}


def calibrate_sigma8_to_fsig8(
    bg: Background,
    data: list[tuple[float, float, float]],
    z_pivot: float = 0.51,
    eps_grav: float = 0.0,
) -> float:
    """
    Find sigma8_0 that matches fsigma8(z_pivot) target from data.
    """
    target = None
    for (z, fs8, sig) in data:
        if abs(z - z_pivot) < 1.0e-3 and sig > 0:
            target = fs8
            break
    if target is None:
        return 0.811
    a_grid = logspace(1.0e-3, 1.0, 2001)
    ln_a_grid = [math.log(a) for a in a_grid]
    mu_grid = [1.0 - eps_grav * (1.0 - bg.omega_m_of_a(a)) for a in a_grid]
    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)
    a = 1.0 / (1.0 + z_pivot)
    ln_a = math.log(a)
    d = interp_linear(ln_a_grid, d_norm, ln_a)
    fz = interp_linear(ln_a_grid, f_ln, ln_a)
    if d * fz <= 0.0:
        return 0.811
    return target / (fz * d)


def calibrate_eps_grav_to_chi2(
    omega_m0: float, sigma8_0: float,
    data: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """
    Brute-force search eps_grav in [-0.5, +0.5] minimizing chi2 (sigma8_0 fixed).
    """
    best = None
    bg = Background(omega_m0=omega_m0, omega_l0=1.0 - omega_m0)
    z_data = sorted({z for (z, _, _) in data})
    for k in range(-50, 51):
        eg = 0.01 * k
        fs8_pred = predict_fsig8_curve(bg, sigma8_0, z_data, eps_grav=eg)
        pred_at_z = dict(zip(z_data, fs8_pred))
        chi2, n = chi2_against(data, pred_at_z)
        if best is None or chi2 < best[1]:
            best = (eg, chi2, n)
    return best[0], best[1]


def main() -> int:
    p = argparse.ArgumentParser(prog="s8_tension")
    p.add_argument("--data", type=str, default="",
                   help="z:fs8:sigma triplets, defaults to built-in compilation")
    p.add_argument("--planck-sigma8", type=float, default=0.811)
    p.add_argument("--ce-sigma8", type=float, default=-1.0,
                   help="if <0, calibrate sigma8 to BOSS DR12 z=0.51 anchor")
    p.add_argument("--cal-eps-grav", action="store_true",
                   help="brute-force fit eps_grav for CE to minimize chi2")
    args = p.parse_args()

    if args.data:
        data = parse_fsigma8_triplets(args.data)
    else:
        data = FSIG8_DATA

    print("=" * 72)
    print("S_8 / fsigma8 consistency check")
    print(f"  data points: {len(data)}")
    print("=" * 72)
    print()

    print("=" * 72)
    print("Model 1: LCDM (Planck baseline)")
    print("=" * 72)
    res_lcdm = report_model("LCDM, Planck", 0.315, args.planck_sigma8, data)

    print("=" * 72)
    print("Model 2: CE Hubble-tension solution (Omega_m = 0.343)")
    print("=" * 72)
    if args.ce_sigma8 < 0.0:
        bg_ce = Background(omega_m0=0.343, omega_l0=0.657)
        s8_ce_in = calibrate_sigma8_to_fsig8(bg_ce, data, z_pivot=0.51)
        print(f"  (sigma8_0 calibrated to BOSS z=0.51: {s8_ce_in:.4f})")
    else:
        s8_ce_in = args.ce_sigma8
    res_ce = report_model("CE, no modified gravity", 0.343, s8_ce_in, data)

    if args.cal_eps_grav:
        print("=" * 72)
        print("Model 3: CE + free eps_grav (brute fit)")
        print("=" * 72)
        eg_best, chi2_best = calibrate_eps_grav_to_chi2(
            0.343, s8_ce_in, data
        )
        print(f"  best eps_grav = {eg_best:+.3f}, chi2 = {chi2_best:.3f}")
        report_model("CE + best eps_grav", 0.343, s8_ce_in, data,
                     eps_grav=eg_best)

    print("=" * 72)
    print("Per-z prediction comparison (fsig8)")
    print("=" * 72)
    print(f"  {'z':>7} {'obs':>10} {'sigma':>8} "
          f"{'LCDM':>10} {'(o-LCDM)/s':>12} "
          f"{'CE':>10} {'(o-CE)/s':>10}")
    for (z, fs8, sig) in data:
        lcdm_p = res_lcdm["fs8_pred"].get(z, float("nan"))
        ce_p = res_ce["fs8_pred"].get(z, float("nan"))
        r_l = (fs8 - lcdm_p) / sig if sig > 0 else float("nan")
        r_c = (fs8 - ce_p) / sig if sig > 0 else float("nan")
        print(f"  {z:>7.3f} {fs8:>10.4f} {sig:>8.4f} "
              f"{lcdm_p:>10.4f} {r_l:>+12.2f} "
              f"{ce_p:>10.4f} {r_c:>+10.2f}")
    print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    delta_chi2 = res_ce["chi2"] - res_lcdm["chi2"]
    print(f"  Delta chi2 (CE - LCDM)  = {delta_chi2:+.3f}")
    if delta_chi2 > 0:
        print(f"  -> CE worse than LCDM by {delta_chi2:.2f} chi2 units on growth data")
    else:
        print(f"  -> CE better than LCDM by {-delta_chi2:.2f} chi2 units")
    print(f"  CE S_8 vs Planck   = {(res_ce['s8']-0.832)/0.013:+.2f} sigma")
    print(f"  CE S_8 vs KiDS     = {(res_ce['s8']-0.766)/0.020:+.2f} sigma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
