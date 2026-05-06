"""Candidate lift from recursive e-fold data to de Sitter horizon entropy.

This gate probes whether the large lift factor between the dimensionless
recursive entropy S_R and the physical de Sitter horizon entropy can be
organized by CE dimensionless data.  It is a candidate test, not a derivation.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
H0_REF_KM_S_MPC = 67.4
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


def h0_from_log_entropy(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def rel_error(pred: float, obs: float) -> float:
    return 100.0 * (pred / obs - 1.0)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    s_recursive = -math.log(x)
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    h0_s = H0_REF_KM_S_MPC / MPC_KM
    log_s_obs = math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)
    lift_per_efold = log_s_obs / n_e

    models = [
        ("base phase-area", 0.5 * math.pi * math.pi * n_e),
        ("subtract pi delta", 0.5 * math.pi * math.pi * n_e - math.pi * delta),
        ("subtract pi delta sigma", 0.5 * math.pi * math.pi * n_e - math.pi * delta * sigma),
        ("subtract pi delta(1-x/2)", 0.5 * math.pi * math.pi * n_e - math.pi * delta * (1.0 - 0.5 * x)),
        ("subtract S_R/(2pi)", 0.5 * math.pi * math.pi * n_e - s_recursive / (2.0 * math.pi)),
    ]

    print("# Horizon Entropy Lift Gate")
    print()
    print("## Inputs")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"S_R = {s_recursive:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"H0 reference = {H0_REF_KM_S_MPC:.4f} km/s/Mpc")
    print(f"log S_dS(reference) = {log_s_obs:.8f}")
    print(f"log S_dS / N_e = {lift_per_efold:.8f}")
    print(f"pi^2/2 = {0.5 * math.pi * math.pi:.8f}")
    print()

    print("## Candidate log-entropy lifts")
    print()
    print("| model | log S_pred | Delta logS | H0_pred | H0 error | status |")
    print("|---|---:|---:|---:|---:|---|")
    for name, log_s_pred in models:
        h0_pred = h0_from_log_entropy(log_s_pred)
        delta_log = log_s_pred - log_s_obs
        status = "candidate" if abs(rel_error(h0_pred, H0_REF_KM_S_MPC)) < 2.0 else "weak"
        print(
            f"| {name} | {log_s_pred:.8f} | {delta_log:+.8f} | "
            f"{h0_pred:.6f} | {rel_error(h0_pred, H0_REF_KM_S_MPC):+.3f}% | {status} |"
        )
    print()

    best_log = 0.5 * math.pi * math.pi * n_e - math.pi * delta * sigma
    print("## Candidate formula")
    print()
    print("log S_dS ~= (pi^2/2) N_e - pi delta sigma")
    print(f"predicted log S_dS = {best_log:.8f}")
    print(f"reference log S_dS = {log_s_obs:.8f}")
    print(f"implied H0 = {h0_from_log_entropy(best_log):.6f} km/s/Mpc")
    print()

    print("## Verdict")
    print()
    print("A pure (pi^2/2)N_e lift is too large and predicts H0 about 23.5% low.")
    print("Adding a defect-dark correction pi*delta*sigma brings H0 within about 0.23% of the reference.")
    print("This is a strong candidate pattern but not yet a derivation; it may still be numerology.")
    print("To promote it, derive the pi^2/2 phase-area lift and the pi*delta*sigma boundary correction independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
