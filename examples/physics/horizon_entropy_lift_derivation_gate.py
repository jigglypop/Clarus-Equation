"""Conditional derivation audit for the horizon entropy lift candidate.

Candidate:
    log S_dS ~= (pi^2/2) N_e - pi delta sigma

This script writes the candidate as two half-cycle integrals:
    A_phase = integral_0^pi theta dtheta = pi^2/2
    B_defect = integral_0^pi delta sigma dtheta = pi delta sigma

It then checks the implied H0.  The derivation is conditional on the readout
assumption that horizon log-entropy counts half-cycle phase area per e-fold.
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

    phase_area = 0.5 * math.pi * math.pi
    defect_boundary = math.pi * delta * sigma
    log_s_pred = phase_area * n_e - defect_boundary
    h0_pred = h0_from_log_entropy(log_s_pred)

    required_boundary = phase_area * n_e - log_s_ref
    inferred_density = required_boundary / math.pi
    candidate_density = delta * sigma

    print("# Horizon Entropy Lift Derivation Gate")
    print()
    print("## Conditional derivation")
    print()
    print("Assumption A: each e-fold contributes half-cycle phase area.")
    print("A_phase = integral_0^pi theta dtheta = pi^2/2")
    print("Assumption B: the boundary defect density over the same half-cycle is delta*sigma.")
    print("B_defect = integral_0^pi delta*sigma dtheta = pi*delta*sigma")
    print()
    print("Therefore:")
    print("log S_dS ~= N_e A_phase - B_defect")
    print("          = (pi^2/2) N_e - pi delta sigma")
    print()

    print("## Numerical check")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"A_phase = {phase_area:.8f}")
    print(f"B_defect = pi delta sigma = {defect_boundary:.8f}")
    print(f"log S_pred = {log_s_pred:.8f}")
    print(f"log S_ref = {log_s_ref:.8f}")
    print(f"Delta log S = {log_s_pred - log_s_ref:+.8f}")
    print(f"H0_pred = {h0_pred:.6f} km/s/Mpc")
    print(f"H0_error = {rel_error(h0_pred, H0_REF_KM_S_MPC):+.3f}%")
    print()

    print("## Inverse boundary-density check")
    print()
    print(f"required boundary = (pi^2/2)N_e - logS_ref = {required_boundary:.8f}")
    print(f"required density = boundary/pi = {inferred_density:.8f}")
    print(f"candidate density delta*sigma = {candidate_density:.8f}")
    print(f"density error = {rel_error(candidate_density, inferred_density):+.3f}%")
    print()

    print("## Verdict")
    print()
    print("The candidate has a compact conditional derivation from two half-cycle integrals.")
    print("It is promoted from raw numerology to Conditional/Bridge, not to Exact.")
    print("Remaining open point: justify why horizon log-entropy must count this phase area per e-fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
