"""Early-late boundary invariant audit.

This gate tests the strongest version of the horizon readout idea:

    log S_dS + pi delta sigma ~= (pi^2/2) N_e

If this is more than numerology, late-time H0 should invert back to the same
integer gauge count N_gauge=12 used by the primordial spectrum.
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


def log_s_from_h0(h0_km_s_mpc: float) -> float:
    h0_s = h0_km_s_mpc / MPC_KM
    return math.log(math.pi / (h0_s * T_PLANCK_S) ** 2)


def h0_from_log_s(log_s: float) -> float:
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
    phase_area = 0.5 * math.pi * math.pi
    boundary = math.pi * delta * sigma
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    log_s_ref = log_s_from_h0(H0_REF_KM_S_MPC)
    invariant_pred = phase_area * n_e
    invariant_obs = log_s_ref + boundary
    invariant_residual = invariant_obs - invariant_pred

    n_e_from_h0 = invariant_obs / phase_area
    n_gauge_from_h0 = 2.0 * n_e_from_h0 / (D_SPATIAL * d_eff)

    log_s_pred = invariant_pred - boundary
    h0_pred = h0_from_log_s(log_s_pred)

    print("# Early-Late Boundary Invariant Gate")
    print()
    print("## Proposed invariant")
    print()
    print("I_H = log S_dS + pi delta sigma")
    print("I_phase = (pi^2/2) N_e")
    print("Boundary readout claims I_H ~= I_phase.")
    print()

    print("## Forward prediction")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"N_gauge = {N_GAUGE:.8f}")
    print(f"N_e = (3/2) D_eff N_gauge = {n_e:.8f}")
    print(f"phase_area = pi^2/2 = {phase_area:.8f}")
    print(f"boundary = pi delta sigma = {boundary:.8f}")
    print(f"log S_pred = I_phase - boundary = {log_s_pred:.8f}")
    print(f"H0_pred = {h0_pred:.6f} km/s/Mpc")
    print(f"H0_ref = {H0_REF_KM_S_MPC:.6f} km/s/Mpc")
    print(f"H0_error = {rel_error(h0_pred, H0_REF_KM_S_MPC):+.3f}%")
    print()

    print("## Inverse check from late-time H0")
    print()
    print(f"log S_ref = {log_s_ref:.8f}")
    print(f"I_H(ref) = logS_ref + boundary = {invariant_obs:.8f}")
    print(f"I_phase = (pi^2/2)N_e = {invariant_pred:.8f}")
    print(f"invariant residual I_H - I_phase = {invariant_residual:+.8f}")
    print(f"N_e inferred from H0 = {n_e_from_h0:.8f}")
    print(f"N_e CE = {n_e:.8f}")
    print(f"Delta N_e = {n_e_from_h0 - n_e:+.8f}")
    print(f"N_gauge inferred from H0 = {n_gauge_from_h0:.8f}")
    print(f"N_gauge target = {N_GAUGE:.8f}")
    print(f"Delta N_gauge = {n_gauge_from_h0 - N_GAUGE:+.8f}")
    print()

    print("## Sensitivity")
    print()
    print("Because H0 ~ exp[-(phase_area/2) N_e], one e-fold changes H0 by exp(-pi^2/4).")
    print(f"exp(-pi^2/4) = {math.exp(-0.25 * math.pi * math.pi):.8f}")
    print("So matching H0 within percent-level is a sharp test of the boundary invariant.")
    print()

    print("## Verdict")
    print()
    print("Late-time H0 inverts back to N_gauge very close to 12.")
    print("This supports an early-late boundary invariant, but still assumes the reference H0 scale.")
    print("The next falsification is to compare this H0 prediction against independent H0 datasets and BAO/CMB fits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
