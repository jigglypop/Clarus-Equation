"""Primitive spectrum prediction card for the self-recursive cosmology gate.

This is the narrow Gate A audit from plan.md.  It keeps the scored spectral
tilt, the reopened scalar amplitude, and the next non-scored tests in one
place so the primitive spectrum can be checked as a coupled prediction family.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9


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


def rel_error(pred: float, obs: float) -> float:
    return 100.0 * (pred / obs - 1.0)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    residual = x - math.exp(-(1.0 - x) * d_eff)

    n_e = (D / 2.0) * d_eff * N_GAUGE
    n_s = 1.0 - 2.0 / n_e
    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)

    q_total = abs(dx_dD(x, d_eff))
    q_source = x * sigma
    q_phase = (2.0 / math.pi) * q_source
    gamma_eff = d_eff / (d_eff + 1.0)
    q_a3c = q_phase * sigma**gamma_eff

    a_s_raw = compute_a_s(x, sigma, n_e, q_total)
    a_s_source = compute_a_s(x, sigma, n_e, q_source)
    a_s_phase = compute_a_s(x, sigma, n_e, q_phase)
    a_s_a3c = compute_a_s(x, sigma, n_e, q_a3c)

    target_projection = math.sqrt(A_S_REF / a_s_source)
    a3c_projection = (2.0 / math.pi) * sigma**gamma_eff
    target_gamma = math.log(target_projection / (2.0 / math.pi)) / math.log(sigma)

    print("# Primitive Spectrum Prediction Card")
    print()
    print("## Fixed-point inputs")
    print()
    print(f"sin2(theta_W) = 4 alpha_s^(4/3) = {sin2_theta_w:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"r_R(x;D_eff) = x - exp(-(1-x)D_eff) = {residual:+.3e}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"sigma = 1 - x = {sigma:.8f}")
    print(f"N_e = (3/2) D_eff N_gauge = {n_e:.8f}")
    print()

    print("## Readout audit")
    print()
    print("| quantity | equation | value | reference/status | error |")
    print("|---|---|---:|---|---:|")
    print(f"| n_s | `1 - 2/N_e` | {n_s:.8f} | scored spectral tilt | {rel_error(n_s, 0.9649):+.3f}% |")
    print(f"| A_s raw | `Q=|dx/dD|` | {a_s_raw:.8e} | rejected total sensitivity | {rel_error(a_s_raw, A_S_REF):+.2f}% |")
    print(f"| A_s source | `Q=x(1-x)` | {a_s_source:.8e} | residual source only | {rel_error(a_s_source, A_S_REF):+.2f}% |")
    print(f"| A_s phase | `Q=(2/pi)x(1-x)` | {a_s_phase:.8e} | half-cycle projection | {rel_error(a_s_phase, A_S_REF):+.2f}% |")
    print(f"| A_s A3c | `Q=(2/pi)sigma^(D_eff/(D_eff+1))x(1-x)` | {a_s_a3c:.8e} | Open candidate | {rel_error(a_s_a3c, A_S_REF):+.2f}% |")
    print(f"| alpha_spec | `-2/N_e^2` | {alpha_spec:.8e} | Open test | non-scored |")
    print(f"| r_tensor | `12/N_e^2` | {r_tensor:.8f} | Open test | non-scored |")
    print()

    print("## Projection check")
    print()
    print(f"target projection from A_s = sqrt(A_s_ref/A_s_source) = {target_projection:.8f}")
    print(f"A3c projection = (2/pi)sigma^(D_eff/(D_eff+1)) = {a3c_projection:.8f}")
    print(f"projection error = {rel_error(a3c_projection, target_projection):+.3f}%")
    print(f"target gamma = log(P_target/(2/pi))/log(sigma) = {target_gamma:.8f}")
    print(f"gamma_eff = D_eff/(D_eff+1) = {gamma_eff:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
