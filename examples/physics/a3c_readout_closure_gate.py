"""A3c common-readout closure audit.

This gate asks whether the gravitational-environment projection introduced for
the scalar amplitude is reused, rather than fitted only to A_s.

Common projection:
    P_GER = (2/pi) sigma ** (D_eff / (D_eff + 1))

Repeated uses checked here:
    A_s source amplitude      Q_A3c = P_GER x(1-x)
    quadrupole power handle   S_Q   = P_GER^2
    hemispherical handle      A_H   = 2 P_GER x

The horizon entropy lift is printed as a separate +1 readout family: it uses a
half-cycle ordered phase area pi^2/2 and boundary correction pi delta sigma,
not the D/(D+1) defect exponent.  This prevents over-claiming.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9
A_S_SIGMA = 0.03e-9
H0_REF_KM_S_MPC = 67.4
MPC_KM = 3.0856775814913673e19
T_PLANCK_S = 5.391247e-44


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


def h0_from_log_entropy(log_s: float) -> float:
    h_s = math.sqrt(math.pi) * math.exp(-0.5 * log_s) / T_PLANCK_S
    return h_s * MPC_KM


def rel_error(pred: float, obs: float) -> float:
    return 100.0 * (pred / obs - 1.0)


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def cosmic_variance(l_mode: int) -> float:
    return math.sqrt(2.0 / (2.0 * l_mode + 1.0))


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    p_phase = 2.0 / math.pi
    p_ger = p_phase * sigma**gamma_eff
    q_source = x * sigma
    q_a3c = p_ger * q_source
    total_susceptibility = abs(dx_dD(x, d_eff))

    as_raw = compute_a_s(x, sigma, n_e, total_susceptibility)
    as_a3c = compute_a_s(x, sigma, n_e, q_a3c)
    n_s = 1.0 - 2.0 / n_e
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)

    quadrupole_power_handle = p_ger * p_ger
    quadrupole_pull = (quadrupole_power_handle - 1.0) / cosmic_variance(2)
    hemispherical_handle = 2.0 * q_a3c / sigma
    hemispherical_handle_alt = 2.0 * p_ger * x
    large_angle_fractional = q_a3c / sigma

    phase_area = 0.5 * math.pi * math.pi
    boundary = math.pi * delta * sigma
    log_s_horizon = phase_area * n_e - boundary
    h0_from_horizon_lift = h0_from_log_entropy(log_s_horizon)

    print("# A3c Common Readout Closure Gate")
    print()
    print("## Core projection")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"gamma_eff = D_eff/(D_eff+1) = {gamma_eff:.8f}")
    print(f"P_GER = (2/pi) sigma^gamma_eff = {p_ger:.8f}")
    print(f"Q_A3c = P_GER x(1-x) = {q_a3c:.8f}")
    print()

    print("## Reuse table")
    print()
    print("| target | equation | value | status |")
    print("|---|---|---:|---|")
    print(f"| n_s | `1 - 2/N_e` | {n_s:.8f} | closed/scored |")
    print(f"| A_s raw | `A_s[abs(dx/dD)]` | {as_raw:.8e} | rejected, pull {pull(as_raw, A_S_REF, A_S_SIGMA):+.2f} |")
    print(f"| A_s GER | `A_s[P_GER x(1-x)]` | {as_a3c:.8e} | candidate, pull {pull(as_a3c, A_S_REF, A_S_SIGMA):+.2f} |")
    print(f"| scalar running | `-2/N_e^2` | {alpha_spec:.8e} | Open test |")
    print(f"| tensor ratio | `12/N_e^2` | {r_tensor:.8f} | Open test |")
    print(f"| quadrupole power handle | `P_GER^2` | {quadrupole_power_handle:.8f} | repeated GER readout |")
    print(f"| quadrupole pull scale | `(P_GER^2-1)/sqrt(2/5)` | {quadrupole_pull:+.2f} sigma | not decisive alone |")
    print(f"| hemispherical handle | `2 Q_A3c/sigma` | {hemispherical_handle:.8f} | repeated GER readout |")
    print(f"| hemispherical identity check | `2 P_GER x` | {hemispherical_handle_alt:.8f} | same value |")
    print(f"| large-angle fractional residual | `Q_A3c/sigma` | {large_angle_fractional:.8f} | Open test |")
    print()

    print("## Separate +1 horizon readout")
    print()
    print("Horizon entropy does not use the D/(D+1) defect exponent in the current derivation.")
    print("It uses the related +1 phase-area readout:")
    print("log S_dS ~= (pi^2/2) N_e - pi delta sigma")
    print(f"phase area pi^2/2 = {phase_area:.8f}")
    print(f"boundary pi delta sigma = {boundary:.8f}")
    print(f"log S_dS candidate = {log_s_horizon:.8f}")
    print(f"H0 implied by entropy lift = {h0_from_horizon_lift:.6f} km/s/Mpc")
    print(f"H0 reference error = {rel_error(h0_from_horizon_lift, H0_REF_KM_S_MPC):+.3f}%")
    print()

    print("## Verdict")
    print()
    print("A3c/GER is now reused in two independent-looking scalar handles: A_s and CMB large-angle amplitudes.")
    print("This strengthens the readout theorem but does not close it: no preferred CMB axis or likelihood is derived.")
    print("The horizon entropy lift is compatible with the +1 philosophy, but remains a separate phase-area bridge.")

    if pull(as_raw, A_S_REF, A_S_SIGMA) < 10.0:
        raise SystemExit("raw scalar susceptibility should remain rejected")
    if abs(pull(as_a3c, A_S_REF, A_S_SIGMA)) > 3.0:
        raise SystemExit("GER A_s should stay inside broad gate")
    if abs(hemispherical_handle - hemispherical_handle_alt) > 1e-12:
        raise SystemExit("hemispherical identity check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
