"""Candidate scale maps from recursive entropy to FLRW quantities.

The recursive entropy S_R is dimensionless.  This gate checks which FLRW
quantities can be connected without adding a new dimensional calibration, and
which require an external physical scale such as H0, M_pl, or reheating.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9

MPL_REDUCED_GEV = 2.435e18
H0_KM_S_MPC = 67.4
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(1000):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    s_recursive = -math.log(x)
    contraction = d_eff * x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)

    q_source = x * sigma
    gamma_eff = d_eff / (d_eff + 1.0)
    q_a3c = (2.0 / math.pi) * sigma**gamma_eff * q_source
    a_s_a3c = compute_a_s(x, sigma, n_e, q_a3c)

    curvature_efold = math.exp(-2.0 * n_e)
    curvature_recursive = contraction**n_e
    n_for_1e5_curvature = math.log(1.0e-5) / math.log(contraction)
    n_for_1e60_residual = math.log(1.0e-60) / math.log(contraction)

    h_inf_over_mpl = math.pi * math.sqrt(A_S_REF * r_tensor / 2.0)
    h_inf_gev = h_inf_over_mpl * MPL_REDUCED_GEV
    v_quarter_over_mpl = (1.5 * math.pi * math.pi * A_S_REF * r_tensor) ** 0.25
    v_quarter_gev = v_quarter_over_mpl * MPL_REDUCED_GEV
    h_inf_over_mpl_a3c = math.pi * math.sqrt(a_s_a3c * r_tensor / 2.0)
    h_inf_gev_a3c = h_inf_over_mpl_a3c * MPL_REDUCED_GEV

    h0_s = H0_KM_S_MPC / MPC_KM
    h0_planck = h0_s * T_PLANCK_S
    s_horizon = math.pi / (h0_planck * h0_planck)
    log_s_horizon = math.log(s_horizon)
    lift_log_per_sr = log_s_horizon / s_recursive
    lift_log_per_ne = log_s_horizon / n_e

    print("# FLRW Scale Map Gate")
    print()
    print("## Dimensionless CE core")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"S_R = -log(x) = {s_recursive:.8f}")
    print(f"contraction k = D_eff x = {contraction:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"alpha_spec = {alpha_spec:.8e}")
    print(f"r_tensor = {r_tensor:.8f}")
    print()

    print("## Curvature/flatness map candidates")
    print()
    print("| map | equation | value | status |")
    print("|---|---|---:|---|")
    print(f"| standard e-fold curvature dilution | `exp(-2 N_e)` | {curvature_efold:.8e} | closes dimensionlessly |")
    print(f"| recursive residual dilution | `k^N_e` | {curvature_recursive:.8e} | closes dimensionlessly |")
    print(f"| iterations for `k^n < 1e-5` | `log(1e-5)/log(k)` | {n_for_1e5_curvature:.2f} | flatness easy |")
    print(f"| iterations for `k^n < 1e-60` | `log(1e-60)/log(k)` | {n_for_1e60_residual:.2f} | deep residual erasure |")
    print()

    print("## Inflation scale lift")
    print()
    print("| quantity | equation | value | status |")
    print("|---|---|---:|---|")
    print(f"| A_s A3c | projected residual readout | {a_s_a3c:.8e} | candidate |")
    print(f"| H_inf/M_pl using observed A_s | `pi sqrt(A_s r/2)` | {h_inf_over_mpl:.8e} | needs amplitude scale |")
    print(f"| H_inf [GeV] using observed A_s | above * M_pl | {h_inf_gev:.8e} | scale-lifted |")
    print(f"| H_inf [GeV] using A3c | same with A_s^A3c | {h_inf_gev_a3c:.8e} | candidate |")
    print(f"| V^(1/4) [GeV] | `(3 pi^2 A_s r/2)^(1/4) M_pl` | {v_quarter_gev:.8e} | scale-lifted |")
    print()

    print("## Horizon entropy lift")
    print()
    print("| quantity | value | status |")
    print("|---|---:|---|")
    print(f"| H0 [s^-1] | {h0_s:.8e} | external scale |")
    print(f"| H0 t_Pl | {h0_planck:.8e} | external scale ratio |")
    print(f"| de Sitter horizon entropy pi/(H0 t_Pl)^2 | {s_horizon:.8e} | not predicted by S_R alone |")
    print(f"| log horizon entropy | {log_s_horizon:.8f} | scale-lift target |")
    print(f"| log(S_horizon)/S_R | {lift_log_per_sr:.8f} | required lift factor |")
    print(f"| log(S_horizon)/N_e | {lift_log_per_ne:.8f} | required lift per e-fold |")
    print()

    print("## Verdict")
    print()
    print("Closed without a new scale: curvature dilution ratios exp(-2N_e), k^N_e, and stability counts.")
    print("Scale-lifted but plausible: inflation H and V scales once A_s/r and M_pl are supplied.")
    print("Open: absolute horizon entropy and reheating entropy; S_R alone is far too small to be the physical entropy.")
    print("Therefore d=0 remains a dimensionless boundary, not yet a full FLRW initial-state theorem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
