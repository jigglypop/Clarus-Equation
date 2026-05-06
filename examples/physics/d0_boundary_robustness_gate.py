"""Robustness and falsification audit for the d=0 boundary interpretation.

This gate asks a sharper question than d0_boundary_verification_gate.py:
does the d=0 boundary interpretation survive obvious null models and small
kernel deformations, or is it just a formal ideal with no pressure?
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0

OMEGA_B_OBS = 0.0493
OMEGA_B_SIGMA = 0.0004
N_S_OBS = 0.9649
N_S_SIGMA = 0.0042
A_S_OBS = 2.10e-9
A_S_SIGMA = 0.03e-9


def bootstrap_x(d_eff: float, c_kernel: float = 1.0, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(1000):
        nxt = math.exp(-c_kernel * (1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def dx_dD(x: float, d_eff: float, c_kernel: float = 1.0) -> float:
    return -c_kernel * x * (1.0 - x) / (1.0 - c_kernel * d_eff * x)


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def required_kernel_for_x(x_target: float, d_eff: float) -> float:
    return -math.log(x_target) / (d_eff * (1.0 - x_target))


def model_row(name: str, d_eff: float, c_kernel: float = 1.0) -> dict[str, float | str]:
    x = bootstrap_x(d_eff, c_kernel)
    sigma = 1.0 - x
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    q_total = abs(dx_dD(x, d_eff, c_kernel))
    q_source = x * sigma
    gamma_eff = d_eff / (d_eff + 1.0)
    q_a3c = (2.0 / math.pi) * sigma**gamma_eff * q_source
    a_s_raw = compute_a_s(x, sigma, n_e, q_total)
    a_s_a3c = compute_a_s(x, sigma, n_e, q_a3c)
    chi2 = pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA) ** 2 + pull(n_s, N_S_OBS, N_S_SIGMA) ** 2
    return {
        "name": name,
        "D": d_eff,
        "c": c_kernel,
        "x": x,
        "sigma": sigma,
        "n_s": n_s,
        "omega_b_pull": pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA),
        "n_s_pull": pull(n_s, N_S_OBS, N_S_SIGMA),
        "a_s_raw": a_s_raw,
        "a_s_a3c": a_s_a3c,
        "a_s_raw_pull": pull(a_s_raw, A_S_OBS, A_S_SIGMA),
        "a_s_a3c_pull": pull(a_s_a3c, A_S_OBS, A_S_SIGMA),
        "chi2_omega_ns": chi2,
    }


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta

    ce = model_row("CE minimal kernel", d_eff, 1.0)
    plain = model_row("plain d=3, no delta", D_SPATIAL, 1.0)
    tuned_c = required_kernel_for_x(OMEGA_B_OBS, d_eff)

    print("# d=0 Boundary Robustness Gate")
    print()
    print("## Null and deformation tests")
    print()
    print("| model | D | c in K=c(1-x) | x | Omega_b pull | n_s | n_s pull | chi2(Omega_b,n_s) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in [plain, ce, model_row("kernel tuned to Omega_b", d_eff, tuned_c)]:
        print(
            f"| {row['name']} | {row['D']:.8f} | {row['c']:.8f} | {row['x']:.8f} | "
            f"{row['omega_b_pull']:+.2f} | {row['n_s']:.8f} | {row['n_s_pull']:+.2f} | "
            f"{row['chi2_omega_ns']:.3f} |"
        )
    print()

    print("## Kernel deformation scan")
    print()
    print("| c | x | Omega_b pull | contraction cDx | A_s A3c pull |")
    print("|---:|---:|---:|---:|---:|")
    for c_kernel in [0.80, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20]:
        row = model_row("scan", d_eff, c_kernel)
        contraction = c_kernel * d_eff * float(row["x"])
        print(
            f"| {c_kernel:.2f} | {row['x']:.8f} | {row['omega_b_pull']:+.2f} | "
            f"{contraction:.8f} | {row['a_s_a3c_pull']:+.2f} |"
        )
    print()

    print("## Readout falsification")
    print()
    print("| readout | value | A_s pull | verdict |")
    print("|---|---:|---:|---|")
    print(f"| total sensitivity | {ce['a_s_raw']:.8e} | {ce['a_s_raw_pull']:+.2f} | rejected |")
    print(f"| projected residual | {ce['a_s_a3c']:.8e} | {ce['a_s_a3c_pull']:+.2f} | survives |")
    print()

    print("## What would make d=0 physical rather than ideal?")
    print()
    print("1. The minimal kernel c=1 must keep surviving independent observables.")
    print("2. Running and tensor must follow the same N_e family, not be separately tuned.")
    print("3. A scale map from recursive entropy to curvature/reheating/horizon entropy is needed.")
    print("4. A raw-sensitivity readout must remain rejected; otherwise the readout taxonomy is wrong.")
    print()

    print("## Verdict")
    print()
    print("The d=0 boundary is not directly proven.")
    print("The no-delta null is strongly worse for Omega_b, so the d=3 branch needs the defect trace.")
    print("Kernel deformations show c=1 is plausible but not uniquely proven by this small trace set.")
    print("The next hard tests are running/tensor and an entropy-to-FLRW scale map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
