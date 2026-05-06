"""A3c/GER closure package gate.

This gate is the final compact status card for the current A3c proof attempt.
It does not promote A3c to Exact.  It records the achieved closure level:

Closed:
    - raw susceptibility is rejected as scalar readout
    - GER projection is fixed by +1 gravitational environment
    - A_s and large-angle amplitude handles reuse the same projection

Conditional:
    - if an axis n_i is supplied, tensor normalization is fixed

Data-facing proxy:
    - representative HPA amplitudes prefer CE fixed A over null

Open:
    - derive n_i internally or run a full CMB map/covariance likelihood
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9
A_S_SIGMA = 0.03e-9


HPA_ROWS = [
    (0.070, 0.021),
    (0.090, 0.035),
]


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


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def weighted_hpa_fit() -> tuple[float, float]:
    w_sum = sum(1.0 / (sigma * sigma) for _, sigma in HPA_ROWS)
    mean = sum(a / (sigma * sigma) for a, sigma in HPA_ROWS) / w_sum
    err = math.sqrt(1.0 / w_sum)
    return mean, err


def chi2_hpa(amplitude: float) -> float:
    return sum(((amplitude - a) / sigma) ** 2 for a, sigma in HPA_ROWS)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    q_total = abs(dx_dD(x, d_eff))
    p_ger = (2.0 / math.pi) * sigma**gamma_eff
    q_a3c = p_ger * x * sigma
    as_raw = compute_a_s(x, sigma, n_e, q_total)
    as_a3c = compute_a_s(x, sigma, n_e, q_a3c)
    a_h = 2.0 * q_a3c / sigma
    s_q = p_ger * p_ger
    tensor_norm = s_q * math.sqrt(2.0 / 3.0)
    large_angle_fractional = q_a3c / sigma

    a_fit, a_fit_sigma = weighted_hpa_fit()
    chi2_null = chi2_hpa(0.0)
    chi2_ce = chi2_hpa(a_h)
    chi2_fit = chi2_hpa(a_fit)
    ce_fit_pull = (a_h - a_fit) / a_fit_sigma

    alpha_spec = -2.0 / (n_e * n_e)
    r_tensor = 12.0 / (n_e * n_e)

    print("# A3c/GER Closure Package Gate")
    print()
    print("## Core numbers")
    print()
    print(f"D_eff = {d_eff:.8f}")
    print(f"x = {x:.8f}")
    print(f"sigma = {sigma:.8f}")
    print(f"N_e = {n_e:.8f}")
    print(f"P_GER = (2/pi)sigma^[D/(D+1)] = {p_ger:.8f}")
    print(f"Q_A3c = P_GER x(1-x) = {q_a3c:.8f}")
    print()

    print("## Closure ledger")
    print()
    print("| layer | quantity | value | status |")
    print("|---|---|---:|---|")
    print(f"| raw scalar readout | A_s[abs(dx/dD)] | {as_raw:.8e} | rejected, pull {pull(as_raw, A_S_REF, A_S_SIGMA):+.2f} |")
    print(f"| GER scalar readout | A_s[Q_A3c] | {as_a3c:.8e} | candidate, pull {pull(as_a3c, A_S_REF, A_S_SIGMA):+.2f} |")
    print(f"| common projection | P_GER | {p_ger:.8f} | fixed by +1 gravity environment candidate |")
    print(f"| scalar running | -2/N_e^2 | {alpha_spec:.8e} | Open test |")
    print(f"| tensor ratio | 12/N_e^2 | {r_tensor:.8f} | Open test |")
    print(f"| quadrupole handle | S_Q=P_GER^2 | {s_q:.8f} | amplitude handle |")
    print(f"| hemispherical handle | A_H=2Q_A3c/sigma | {a_h:.8f} | amplitude handle |")
    print(f"| fractional residual | Q_A3c/sigma | {large_angle_fractional:.8f} | large-angle handle |")
    print(f"| conditional tensor norm | S_Q sqrt(2/3) | {tensor_norm:.8f} | conditional on axis n_i |")
    print()

    print("## HPA proxy likelihood")
    print()
    print(f"weighted HPA proxy fit = {a_fit:.8f} +/- {a_fit_sigma:.8f}")
    print(f"CE A_H vs proxy fit = {ce_fit_pull:+.2f} sigma")
    print(f"chi2 null A=0 = {chi2_null:.4f}")
    print(f"chi2 CE fixed A_H = {chi2_ce:.4f}")
    print(f"chi2 best-fit A = {chi2_fit:.4f}")
    print(f"Delta chi2 CE-null = {chi2_ce - chi2_null:+.4f}")
    print(f"Delta chi2 CE-fit = {chi2_ce - chi2_fit:+.4f}")
    print()

    print("## Final status")
    print()
    print("Promote to: Selection candidate / pre-likelihood CMB amplitude bridge.")
    print("Do not promote to: Exact theorem or full CMB anomaly closure.")
    print("Remaining blocker: derive n_i or run map/covariance likelihood with A fixed to A_H.")
    print()

    print("## Falsification handles")
    print()
    print("1. A future A_s/running/tensor family rejects the shared N_e/A3c structure.")
    print("2. Robust CMB HPA amplitude is far from A_H=0.05963341.")
    print("3. Full CMB likelihood with A fixed to A_H is worse than null after trials/masks.")
    print("4. A derived or ingested axis n_i fails quadrupole/octupole phase tests.")

    if pull(as_raw, A_S_REF, A_S_SIGMA) < 10.0:
        raise SystemExit("raw A_s should remain rejected")
    if abs(pull(as_a3c, A_S_REF, A_S_SIGMA)) > 3.0:
        raise SystemExit("A3c A_s should remain inside the broad gate")
    if abs(ce_fit_pull) > 2.0:
        raise SystemExit("CE HPA amplitude should remain near the proxy fit")
    if chi2_ce >= chi2_null:
        raise SystemExit("CE HPA amplitude should improve over null in proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
