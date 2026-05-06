"""Lightweight likelihood proxy for A3c/GER CMB large-angle amplitudes.

This is not a full CMB likelihood.  It is a pre-likelihood consistency check:
given representative hemispherical-power-asymmetry amplitude rows, compare

    H0/null: A = 0
    CE:      A = A_H = 2 Q_A3c / sigma
    fit:     A = inverse-variance weighted common amplitude

The goal is to check whether the CE fixed amplitude is near the common
large-angle amplitude scale before building a map/covariance pipeline.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0


OBS_ROWS = [
    {
        "name": "Planck/PR3 temperature HPA representative",
        "amplitude": 0.070,
        "sigma": 0.021,
        "axis_l_deg": 205.0,
        "axis_b_deg": -20.0,
    },
    {
        "name": "Planck PR4 Sevem E-mode proxy",
        "amplitude": 0.090,
        "sigma": 0.035,
        "axis_l_deg": 234.0,
        "axis_b_deg": -14.0,
    },
]


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def chi2_for(amplitude: float) -> float:
    return sum(((amplitude - row["amplitude"]) / row["sigma"]) ** 2 for row in OBS_ROWS)


def weighted_fit() -> tuple[float, float]:
    w_sum = sum(1.0 / (row["sigma"] ** 2) for row in OBS_ROWS)
    mean = sum(row["amplitude"] / (row["sigma"] ** 2) for row in OBS_ROWS) / w_sum
    sigma = math.sqrt(1.0 / w_sum)
    return mean, sigma


def galactic_to_unit(l_deg: float, b_deg: float) -> tuple[float, float, float]:
    l_rad = math.radians(l_deg)
    b_rad = math.radians(b_deg)
    cb = math.cos(b_rad)
    return cb * math.cos(l_rad), cb * math.sin(l_rad), math.sin(b_rad)


def angular_sep_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = min(1.0, max(-1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)
    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_a3c = p_ger * x * sigma
    a_ce = 2.0 * q_a3c / sigma

    a_fit, sigma_fit = weighted_fit()
    chi2_null = chi2_for(0.0)
    chi2_ce = chi2_for(a_ce)
    chi2_fit = chi2_for(a_fit)
    delta_chi2_ce_null = chi2_ce - chi2_null
    delta_chi2_ce_fit = chi2_ce - chi2_fit
    ce_fit_pull = (a_ce - a_fit) / sigma_fit

    axes = [galactic_to_unit(row["axis_l_deg"], row["axis_b_deg"]) for row in OBS_ROWS]
    axis_sep = angular_sep_deg(axes[0], axes[1])

    print("# A3c CMB Amplitude Likelihood Proxy Gate")
    print()
    print("## CE fixed amplitude")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"P_GER = {p_ger:.8f}")
    print(f"Q_A3c = {q_a3c:.8f}")
    print(f"A_CE = 2 Q_A3c/sigma = {a_ce:.8f}")
    print()

    print("## Input amplitude rows")
    print()
    print("| row | A_obs | sigma | axis (l,b) | CE pull |")
    print("|---|---:|---:|---:|---:|")
    for row in OBS_ROWS:
        row_pull = (a_ce - row["amplitude"]) / row["sigma"]
        print(
            f"| {row['name']} | {row['amplitude']:.5f} | {row['sigma']:.5f} | "
            f"({row['axis_l_deg']:.1f}, {row['axis_b_deg']:.1f}) | {row_pull:+.2f} |"
        )
    print()

    print("## Common-amplitude proxy likelihood")
    print()
    print(f"weighted best-fit A = {a_fit:.8f} +/- {sigma_fit:.8f}")
    print(f"CE vs best-fit pull = {ce_fit_pull:+.2f} sigma")
    print(f"chi2(null A=0) = {chi2_null:.4f} for {len(OBS_ROWS)} rows")
    print(f"chi2(CE fixed A) = {chi2_ce:.4f} for {len(OBS_ROWS)} rows")
    print(f"chi2(best-fit A) = {chi2_fit:.4f} for {len(OBS_ROWS)} rows")
    print(f"Delta chi2 CE-null = {delta_chi2_ce_null:+.4f}")
    print(f"Delta chi2 CE-fit = {delta_chi2_ce_fit:+.4f}")
    print(f"representative axis separation = {axis_sep:.2f} deg")
    print()

    print("## Verdict")
    print()
    print("The CE fixed amplitude is close to the weighted common HPA amplitude proxy.")
    print("It strongly improves over the no-asymmetry A=0 proxy, but is not a full CMB likelihood.")
    print("The next real closure requires a map/covariance pipeline with A fixed to A_CE.")

    if abs(ce_fit_pull) > 2.0:
        raise SystemExit("CE amplitude is too far from the common-amplitude proxy")
    if chi2_ce >= chi2_null:
        raise SystemExit("CE amplitude should improve over null in this proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
