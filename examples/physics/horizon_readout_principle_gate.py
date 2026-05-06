"""Readout-principle audit for the horizon entropy lift.

This gate tests whether the successful horizon entropy lift is specific to the
half-cycle phase-area readout, or whether many neighboring readouts work just
as well.  It also separates this late-time boundary lift from standard
slow-roll horizon entropy evolution.
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
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    h0_s = H0_REF_KM_S_MPC / MPC_KM
    log_s_ref = math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)

    boundary = math.pi * delta * sigma
    readouts = [
        ("linear e-fold count", 1.0),
        ("curvature dilution count", 2.0),
        ("mean half-cycle phase", math.pi / 2.0),
        ("half-cycle survival projection", 2.0 / math.pi),
        ("half-cycle ordered area", math.pi * math.pi / 2.0),
        ("full-cycle ordered area", 2.0 * math.pi * math.pi),
        ("d=3 adjoint phase measure", (2.0 * math.pi) ** 2 / (D_SPATIAL * D_SPATIAL - 1.0)),
    ]

    print("# Horizon Readout Principle Gate")
    print()
    print("## Fixed inputs")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"boundary correction pi*delta*sigma = {boundary:.8f}")
    print(f"log S_dS reference = {log_s_ref:.8f}")
    print()

    print("## Competing readouts")
    print()
    print("| readout | per-e-fold coefficient | logS pred | H0 pred | H0 error | verdict |")
    print("|---|---:|---:|---:|---:|---|")
    for name, coeff in readouts:
        log_s_pred = coeff * n_e - boundary
        h0_pred = h0_from_log_entropy(log_s_pred)
        err = rel_error(h0_pred, H0_REF_KM_S_MPC)
        verdict = "survives" if abs(err) < 1.0 else "fails"
        print(
            f"| {name} | {coeff:.8f} | {log_s_pred:.8f} | "
            f"{h0_pred:.8e} | {err:+.3e}% | {verdict} |"
        )
    print()

    coeff_required = (log_s_ref + boundary) / n_e
    half_cycle_area = math.pi * math.pi / 2.0
    adjoint_d3 = (2.0 * math.pi) ** 2 / (D_SPATIAL * D_SPATIAL - 1.0)
    print("## Inverse coefficient")
    print()
    print(f"required coefficient = (logS_ref + pi delta sigma)/N_e = {coeff_required:.8f}")
    print(f"pi^2/2 = {half_cycle_area:.8f}")
    print(f"d=3 adjoint phase measure = {adjoint_d3:.8f}")
    print(f"coefficient error vs pi^2/2 = {rel_error(half_cycle_area, coeff_required):+.3f}%")
    print()

    print("## Slow-roll separation")
    print()
    print("For ordinary quasi-de Sitter evolution, S_hor ~ H^-2 gives d log S/dN = 2 epsilon_H.")
    print("The CE coefficient pi^2/2 is much larger than a slow-roll epsilon_H term.")
    print("So this readout cannot be interpreted as local inflationary horizon entropy growth.")
    print("It is a boundary lift from the primordial e-fold phase count to the late-time de Sitter entropy.")
    print()

    print("## Verdict")
    print()
    print("Most simple readouts fail by huge margins.")
    print("The surviving coefficient is uniquely the half-cycle ordered area, equal to the d=3 adjoint phase measure.")
    print("The remaining Bridge assumption is not the value of the coefficient, but why late-time horizon log-entropy reads this primordial phase-area count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
