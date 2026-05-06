"""Falsify the early-late boundary invariant against independent H0 datasets.

The CE boundary-invariant prediction is:
    H0 = 67.247245 km/s/Mpc

This gate compares that value with representative CMB/BAO/local distance-ladder
measurements and also inverts each H0 into an implied N_gauge.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44


DATASETS = [
    {
        "name": "Planck 2018 base LCDM",
        "h0": 67.4,
        "sigma": 0.5,
        "family": "CMB model-dependent",
    },
    {
        "name": "DESI DR2 BAO no-CMB calibration",
        "h0": 68.51,
        "sigma": 0.58,
        "family": "BAO inverse ladder",
    },
    {
        "name": "CCHP 2025 TRGB HST+JWST",
        "h0": 70.39,
        "sigma": math.sqrt(1.22**2 + 1.33**2 + 0.70**2),
        "family": "local ladder",
    },
    {
        "name": "CCHP 2025 JWST-only TRGB",
        "h0": 68.81,
        "sigma": math.hypot(1.79, 1.32),
        "family": "local ladder",
    },
    {
        "name": "CCHP 2025 JWST-only JAGB",
        "h0": 67.80,
        "sigma": math.hypot(2.17, 1.64),
        "family": "local ladder",
    },
    {
        "name": "SH0ES HST Cepheids/SNe",
        "h0": 73.04,
        "sigma": 1.04,
        "family": "local ladder",
    },
    {
        "name": "SH0ES JWST update",
        "h0": 73.17,
        "sigma": 0.86,
        "family": "local ladder",
    },
]


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(1000):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def log_s_from_h0(h0_km_s_mpc: float) -> float:
    h0_s = h0_km_s_mpc / MPC_KM
    return math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)


def h0_from_log_s(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    phase_area = 0.5 * math.pi * math.pi
    boundary = math.pi * delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    log_s_pred = phase_area * n_e - boundary
    h0_pred = h0_from_log_s(log_s_pred)

    print("# H0 Dataset Falsification Gate")
    print()
    print("## CE boundary-invariant prediction")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"H0_pred = {h0_pred:.6f} km/s/Mpc")
    print()

    print("## Dataset comparison")
    print()
    print("| dataset | family | H0 obs | sigma | pull CE-obs | inferred N_gauge | Delta N_gauge | verdict |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for row in DATASETS:
        h0 = float(row["h0"])
        sigma_h0 = float(row["sigma"])
        pull = (h0_pred - h0) / sigma_h0
        log_s = log_s_from_h0(h0)
        invariant_obs = log_s + boundary
        n_e_h0 = invariant_obs / phase_area
        n_gauge_h0 = 2.0 * n_e_h0 / (D_SPATIAL * d_eff)
        delta_ng = n_gauge_h0 - N_GAUGE
        verdict = "supports" if abs(pull) < 2.0 else ("tension" if abs(pull) < 5.0 else "rejects")
        print(
            f"| {row['name']} | {row['family']} | {h0:.3f} | {sigma_h0:.3f} | "
            f"{pull:+.2f} | {n_gauge_h0:.6f} | {delta_ng:+.6f} | {verdict} |"
        )
    print()

    print("## Verdict")
    print()
    print("The invariant is consistent with Planck-like CMB and lower/intermediate JWST CCHP ladders.")
    print("It is mildly tense with the DESI-like BAO+BBN inverse-ladder value used here.")
    print("It is in strong tension with high Cepheid/SN SH0ES values near 73 km/s/Mpc.")
    print("Thus the boundary invariant does not solve the Hubble tension; it picks the low-H0 branch.")
    print("If the local high-H0 Cepheid branch wins, this horizon lift candidate is falsified or needs a late-time branch correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
