"""A3c gravitational-environment readout gate.

This gate checks the proposed proof spine:

1. A scalar transition does not read the total fixed-point susceptibility.
2. It reads the unrelaxed residual drive x(1-x).
3. The d=0 -> d=3 transition contributes the half-cycle projection 2/pi.
4. Because scalar perturbations are observed through metric/gravity, the
   recursive defect sigma is read through the D+1 environment:
   sigma_env = sigma ** (D_eff / (D_eff + 1)).

The gate does not claim the theorem is fully proven.  It records the exact
places where the proposal is closed, candidate, or falsifiable.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D_SPATIAL = 3.0
N_GAUGE = 12.0
A_S_REF = 2.10e-9
A_S_SIGMA = 0.03e-9


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


def pull(pred: float, obs: float, sigma: float) -> float:
    return (pred - obs) / sigma


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D_SPATIAL + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    residual = x - math.exp(-(1.0 - x) * d_eff)
    n_e = (D_SPATIAL / 2.0) * d_eff * N_GAUGE

    total_susceptibility = abs(dx_dD(x, d_eff))
    residual_drive = x * sigma
    half_cycle = 2.0 / math.pi
    gamma_integer = D_SPATIAL / (D_SPATIAL + 1.0)
    gamma_eff = d_eff / (d_eff + 1.0)
    sigma_env_integer = sigma**gamma_integer
    sigma_env_eff = sigma**gamma_eff

    q_source = residual_drive
    q_phase = half_cycle * residual_drive
    q_integer_env = half_cycle * sigma_env_integer * residual_drive
    q_eff_env = half_cycle * sigma_env_eff * residual_drive

    as_total = compute_a_s(x, sigma, n_e, total_susceptibility)
    as_source = compute_a_s(x, sigma, n_e, q_source)
    as_phase = compute_a_s(x, sigma, n_e, q_phase)
    as_integer_env = compute_a_s(x, sigma, n_e, q_integer_env)
    as_eff_env = compute_a_s(x, sigma, n_e, q_eff_env)

    target_projection = math.sqrt(A_S_REF / as_source)
    ger_projection = half_cycle * sigma_env_eff
    target_gamma = math.log(target_projection / half_cycle) / math.log(sigma)

    print("# A3c Gravitational-Environment Readout Gate")
    print()
    print("## Fixed-point branch")
    print()
    print(f"sin2(theta_W) = {sin2_theta_w:.8f}")
    print(f"delta = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"sigma = 1 - x = {sigma:.8f}")
    print(f"r_R(x;D_eff) = {residual:+.3e}")
    print(f"N_e = {n_e:.8f}")
    print()

    print("## Principle decomposition")
    print()
    print("| component | value | status |")
    print("|---|---:|---|")
    print(f"| residual drive `x(1-x)` | {residual_drive:.8f} | definition from partial_D r_R |")
    print(f"| total susceptibility `abs(dx/dD)` | {total_susceptibility:.8f} | rejected for scalar readout |")
    print(f"| half-cycle projection `2/pi` | {half_cycle:.8f} | phase-readout candidate |")
    print(f"| integer gravity exponent `3/(3+1)` | {gamma_integer:.8f} | coarse +1 environment |")
    print(f"| effective gravity exponent `D_eff/(D_eff+1)` | {gamma_eff:.8f} | CE +1 environment candidate |")
    print(f"| sigma_env effective | {sigma_env_eff:.8f} | defect read through gravity environment |")
    print()

    print("## Scalar amplitude audit")
    print()
    print("| readout | Q | A_s | pull | verdict |")
    print("|---|---:|---:|---:|---|")
    print(
        f"| total fixed-point susceptibility | {total_susceptibility:.8f} | "
        f"{as_total:.8e} | {pull(as_total, A_S_REF, A_S_SIGMA):+.2f} | rejected raw |"
    )
    print(
        f"| residual source only | {q_source:.8f} | "
        f"{as_source:.8e} | {pull(as_source, A_S_REF, A_S_SIGMA):+.2f} | too large |"
    )
    print(
        f"| half-cycle residual | {q_phase:.8f} | "
        f"{as_phase:.8e} | {pull(as_phase, A_S_REF, A_S_SIGMA):+.2f} | close but not closed |"
    )
    print(
        f"| integer +1 environment | {q_integer_env:.8f} | "
        f"{as_integer_env:.8e} | {pull(as_integer_env, A_S_REF, A_S_SIGMA):+.2f} | strong candidate |"
    )
    print(
        f"| effective +1 environment | {q_eff_env:.8f} | "
        f"{as_eff_env:.8e} | {pull(as_eff_env, A_S_REF, A_S_SIGMA):+.2f} | A3c/GER candidate |"
    )
    print()

    print("## Non-fit check")
    print()
    print(f"target projection from A_s = {target_projection:.8f}")
    print(f"GER projection = (2/pi) sigma^[D_eff/(D_eff+1)] = {ger_projection:.8f}")
    print(f"projection relative error = {rel_error(ger_projection, target_projection):+.3f}%")
    print(f"target gamma from A_s = {target_gamma:.8f}")
    print(f"CE gamma_eff = {gamma_eff:.8f}")
    print(f"gamma difference = {gamma_eff - target_gamma:+.8f}")
    print()

    print("## Verdict")
    print()
    print("Closed: the residual drive x(1-x) and the total/raw susceptibility are distinct.")
    print("Rejected: scalar A_s cannot be read as the total fixed-point susceptibility.")
    print("Candidate: the +1 gravitational environment explains the D/(D+1) exponent.")
    print("Next proof burden: reuse the same +1 readout in horizon entropy, FLRW scale mapping, or CMB large-angle gates.")

    if abs(residual) > 1e-12:
        raise SystemExit("fixed-point residual is too large")
    if pull(as_total, A_S_REF, A_S_SIGMA) < 10.0:
        raise SystemExit("raw susceptibility should remain rejected")
    if abs(pull(as_eff_env, A_S_REF, A_S_SIGMA)) > 3.0:
        raise SystemExit("GER candidate should stay within the broad scalar-amplitude gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
