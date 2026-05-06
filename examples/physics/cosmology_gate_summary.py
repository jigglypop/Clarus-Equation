"""Integrated CE cosmology gate summary.

This script collects the current A-E gate results into one numerical card:
primitive spectrum, dark matter identity, structure growth, initial entropy,
and CMB large-angle anomalies.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from s8_tension import (  # noqa: E402
    FSIG8_DATA,
    calibrate_sigma8_to_fsig8,
    chi2_against,
    predict_fsig8_curve,
    s8_amplitude,
)
from cosmology import Background  # noqa: E402


ALPHA_S = 0.11789
D = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def dx_dD(x: float, d_eff: float) -> float:
    return -x * (1.0 - x) / (1.0 - d_eff * x)


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def rel_error(pred: float, obs: float) -> float:
    return 100.0 * (pred / obs - 1.0)


def growth_card(omega_m: float, sigma8: float) -> dict[str, float]:
    bg = Background(omega_m0=omega_m, omega_l0=1.0 - omega_m)
    z_data = sorted({z for (z, _, _) in FSIG8_DATA})
    fs8_pred = predict_fsig8_curve(bg, sigma8, z_data)
    pred_at_z = dict(zip(z_data, fs8_pred))
    chi2, n = chi2_against(FSIG8_DATA, pred_at_z)
    return {
        "omega_m": omega_m,
        "sigma8": sigma8,
        "s8": s8_amplitude(sigma8, omega_m),
        "chi2": chi2,
        "n": float(n),
    }


def main() -> int:
    sin2 = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2 * (1.0 - sin2)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)

    r_split = 0.38062659567873197
    omega_l = sigma / (1.0 + r_split)
    omega_dm = sigma * r_split / (1.0 + r_split)
    omega_m_static = x + omega_dm

    n_e = (D / 2.0) * d_eff * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)

    q_total = abs(dx_dD(x, d_eff))
    q_a3c = (2.0 / math.pi) * sigma**gamma_eff * x * sigma
    as_raw = compute_a_s(x, sigma, n_e, q_total)
    as_a3c = compute_a_s(x, sigma, n_e, q_a3c)

    s_recursive = -math.log(x)
    contraction = d_eff * x
    n60 = math.log(1.0e-60) / math.log(contraction)

    bg_static = Background(omega_m0=omega_m_static, omega_l0=1.0 - omega_m_static)
    sigma8_static = calibrate_sigma8_to_fsig8(bg_static, FSIG8_DATA, z_pivot=0.51)
    static_growth = growth_card(omega_m_static, sigma8_static)

    bg_h0 = Background(omega_m0=0.343, omega_l0=0.657)
    sigma8_h0 = calibrate_sigma8_to_fsig8(bg_h0, FSIG8_DATA, z_pivot=0.51)
    h0_growth = growth_card(0.343, sigma8_h0)
    lcdm_growth = growth_card(0.315, 0.811)

    s_q_phase = (2.0 / math.pi) ** 2
    s_q_geom = ((2.0 / math.pi) * sigma**gamma_eff) ** 2
    a_hemi = 2.0 * q_a3c / sigma

    print("# Integrated CE Cosmology Gate Summary")
    print()
    print("## Core")
    print()
    print(f"sin2(theta_W) = 4 alpha_s^(4/3) = {sin2:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"sigma = 1 - x = {sigma:.8f}")
    print()

    print("## A-E gate table")
    print()
    print("| gate | key equation | result | verdict |")
    print("|---|---|---:|---|")
    print(f"| A primitive spectrum | `n_s=1-2/N_e` | {n_s:.8f} | scored/closed |")
    print(f"| A scalar amplitude raw | `A_s[|dx/dD|]` | {as_raw:.8e} | rejected raw, {rel_error(as_raw, A_S_REF):+.1f}% |")
    print(f"| A scalar amplitude A3c | `A_s[Q_A3c]` | {as_a3c:.8e} | Open candidate, {rel_error(as_a3c, A_S_REF):+.2f}% |")
    print(f"| A running | `-2/N_e^2` | {alpha_spec:.8e} | Open test |")
    print(f"| A tensor | `12/N_e^2` | {r_tensor:.8f} | Open test |")
    print(f"| B dark matter | `sigma R/(1+R)` | {omega_dm:.8f} | density closed |")
    print(f"| B DM/DE | `R` | {r_split:.8f} | precision ratio gate |")
    print(f"| C static growth | `chi2(fsigma8)` | {static_growth['chi2']:.3f}/{int(static_growth['n'])} | improves LSS, S8 not closed |")
    print(f"| C H0-branch growth | `chi2(fsigma8)` | {h0_growth['chi2']:.3f}/{int(h0_growth['n'])} | partial S8 relief |")
    print(f"| D recursive entropy | `-log(x)=D(1-x)` | {s_recursive:.8f} | dimensionless boundary closed |")
    print(f"| D contraction | `D_eff x` | {contraction:.8f} | stable; n60={n60:.2f} |")
    print(f"| E quadrupole suppression | `[(2/pi)sigma^gamma]^2` | {s_q_geom:.8f} | Open test |")
    print(f"| E hemispherical contrast | `2 Q_A3c/sigma` | {a_hemi:.8f} | Open test |")
    print()

    print("## Growth comparison")
    print()
    print(f"LCDM fsigma8 chi2 = {lcdm_growth['chi2']:.3f}/{int(lcdm_growth['n'])}, S8 = {lcdm_growth['s8']:.8f}")
    print(f"CE static fsigma8 chi2 = {static_growth['chi2']:.3f}/{int(static_growth['n'])}, S8 = {static_growth['s8']:.8f}")
    print(f"CE H0-branch fsigma8 chi2 = {h0_growth['chi2']:.3f}/{int(h0_growth['n'])}, S8 = {h0_growth['s8']:.8f}")
    print(f"Delta chi2 static-LCDM = {static_growth['chi2'] - lcdm_growth['chi2']:+.3f}")
    print(f"Delta chi2 H0branch-LCDM = {h0_growth['chi2'] - lcdm_growth['chi2']:+.3f}")
    print()

    print("## Current bottom line")
    print()
    print("Closed/scored: density split, n_s, H0t0/T_CMB/eta scorecard package.")
    print("Strong candidates: A_s A3c readout, DM collective identity, d=0 entropy boundary.")
    print("Open tests: running, tensor ratio, S8 residual, halo/JWST growth, CMB large-angle axis/likelihood.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
