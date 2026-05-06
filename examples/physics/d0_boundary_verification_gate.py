"""Audit whether the d=0 identity boundary is physical, ideal, or only formal.

This gate does not claim direct observation of d=0.  It tests the weaker and
testable question: if d=0 is used as the zero-residual boundary of the
self-recursive map, do the compulsory traces in the d=3 branch cohere?
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


def bootstrap_x(d_eff: float, start: float = 0.05, tol: float = 1e-15) -> float:
    x = start
    for _ in range(1000):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def residual(x: float, d_eff: float) -> float:
    return x - math.exp(-(1.0 - x) * d_eff)


def dx_dD(x: float, d_eff: float) -> float:
    return -x * (1.0 - x) / (1.0 - d_eff * x)


def compute_a_s(x: float, sigma: float, n_e: float, readout: float) -> float:
    return (readout * readout) / (sigma * sigma) * x / (2.0 * math.pi * n_e * n_e)


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta

    x_boundary = 1.0
    sigma_boundary = 0.0
    entropy_boundary = 0.0
    boundary_residual = residual(x_boundary, 0.0)

    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    entropy = -math.log(x)
    fixed_entropy = d_eff * sigma
    contraction = d_eff * x
    reverse_amplification = 1.0 / contraction
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE
    n_s = 1.0 - 2.0 / n_e

    q_total = abs(dx_dD(x, d_eff))
    q_source = x * sigma
    gamma_eff = d_eff / (d_eff + 1.0)
    q_a3c = (2.0 / math.pi) * sigma**gamma_eff * q_source
    a_s_raw = compute_a_s(x, sigma, n_e, q_total)
    a_s_a3c = compute_a_s(x, sigma, n_e, q_a3c)

    d_plain = D_SPATIAL
    x_plain = bootstrap_x(d_plain)
    sigma_plain = 1.0 - x_plain
    n_e_plain = (D_SPATIAL / 2.0) * d_plain * N_GAUGE
    n_s_plain = 1.0 - 2.0 / n_e_plain

    print("# d=0 Boundary Verification Gate")
    print()
    print("## What can be known")
    print()
    print("| question | status | reason |")
    print("|---|---|---|")
    print("| Does the mathematical boundary exist? | yes | D=0 gives x0=1, sigma0=0, S_R0=0 |")
    print("| Is d=0 directly observable as a place/state? | no | it is a boundary branch, not a d=3 event |")
    print("| Can d=0 be physically inferred? | candidate | only through compulsory d=3 traces |")
    print("| Can ordinary d=3 dynamics reach it? | no | physical branch is W0; d=0 is the W-1 boundary branch |")
    print()

    print("## Boundary identities")
    print()
    print(f"x0 = {x_boundary:.8f}")
    print(f"sigma0 = {sigma_boundary:.8f}")
    print(f"S_R0 = {entropy_boundary:.8f}")
    print(f"r_R(x0;0) = x0 - exp(-(1-x0)0) = {boundary_residual:+.3e}")
    print()

    print("## d=3 trace package")
    print()
    print(f"sin2(theta_W) = {sin2_theta_w:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"S_R = -log(x) = {entropy:.8f}")
    print(f"S_R - D_eff(1-x) = {entropy - fixed_entropy:+.3e}")
    print(f"contraction k = D_eff x = {contraction:.8f}")
    print(f"reverse step amplification 1/k = {reverse_amplification:.8f}")
    print(f"N_e = {n_e:.8f}")
    print()

    print("## Observational trace checks")
    print()
    print("| trace | prediction | reference | pull | passed broad gate? |")
    print("|---|---:|---:|---:|---|")
    print(
        f"| Omega_b as x | {x:.8f} | {OMEGA_B_OBS:.8f} +/- {OMEGA_B_SIGMA:.1e} | "
        f"{pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA):+.2f} sigma | "
        f"{yes_no(abs(pull(x, OMEGA_B_OBS, OMEGA_B_SIGMA)) < 3.0)} |"
    )
    print(
        f"| n_s from N_e | {n_s:.8f} | {N_S_OBS:.8f} +/- {N_S_SIGMA:.1e} | "
        f"{pull(n_s, N_S_OBS, N_S_SIGMA):+.2f} sigma | "
        f"{yes_no(abs(pull(n_s, N_S_OBS, N_S_SIGMA)) < 3.0)} |"
    )
    print(
        f"| A_s raw sensitivity | {a_s_raw:.8e} | {A_S_OBS:.8e} +/- {A_S_SIGMA:.1e} | "
        f"{pull(a_s_raw, A_S_OBS, A_S_SIGMA):+.2f} sigma | no |"
    )
    print(
        f"| A_s projected residual | {a_s_a3c:.8e} | {A_S_OBS:.8e} +/- {A_S_SIGMA:.1e} | "
        f"{pull(a_s_a3c, A_S_OBS, A_S_SIGMA):+.2f} sigma | "
        f"{yes_no(abs(pull(a_s_a3c, A_S_OBS, A_S_SIGMA)) < 3.0)} |"
    )
    print()

    print("## Null comparison: no fractional defect")
    print()
    print("| model | D | x | sigma | n_s |")
    print("|---|---:|---:|---:|---:|")
    print(f"| plain d=3 | {d_plain:.8f} | {x_plain:.8f} | {sigma_plain:.8f} | {n_s_plain:.8f} |")
    print(f"| CE d=3+delta | {d_eff:.8f} | {x:.8f} | {sigma:.8f} | {n_s:.8f} |")
    print()

    print("## Verdict")
    print()
    print("Direct existence of d=0 is not established.")
    print("What survives this gate is weaker: d=0 works as a zero-residual boundary condition.")
    print("The d=3 branch carries coherent traces: x, S_R, contraction, n_s, and the A3c residual readout.")
    print("The raw A_s failure means the boundary is not proven; it forces the readout-layer distinction.")
    print("If future tests reject n_s/running/tensor or the residual readout, d=0 falls back to a formal ideal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
