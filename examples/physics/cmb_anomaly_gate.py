"""Gate E: CMB large-angle anomaly audit.

This is a candidate-level calculation.  It asks whether the same projected
recursive residual used in the A_s audit gives natural O(1) low-ell suppression
and O(5%) hemispherical modulation handles.  No axis direction or full CMB
likelihood is fitted here.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D = 3.0


def bootstrap_x(d_eff: float, tol: float = 1e-15) -> float:
    x = 0.05
    for _ in range(500):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def cosmic_variance(l_mode: int) -> float:
    return math.sqrt(2.0 / (2.0 * l_mode + 1.0))


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D + delta
    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    gamma_eff = d_eff / (d_eff + 1.0)

    residual_source = x * sigma
    half_cycle = 2.0 / math.pi
    q_phase = half_cycle * residual_source
    geom_projection = half_cycle * sigma**gamma_eff
    q_a3c = geom_projection * residual_source

    power_suppression_phase = half_cycle**2
    power_suppression_geom = geom_projection**2
    quadrupole_cv = cosmic_variance(2)
    octupole_cv = cosmic_variance(3)
    quadrupole_pull_phase = (power_suppression_phase - 1.0) / quadrupole_cv
    quadrupole_pull_geom = (power_suppression_geom - 1.0) / quadrupole_cv

    hemispherical_contrast = 2.0 * geom_projection * residual_source / sigma
    hemispherical_contrast_plain = 2.0 * q_a3c
    large_angle_fractional_residual = q_a3c / sigma

    print("# CMB Large-Angle Anomaly Gate")
    print()
    print("## Inputs")
    print()
    print(f"sin2(theta_W) = {sin2_theta_w:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"sigma = 1 - x = {sigma:.8f}")
    print(f"gamma_eff = D_eff/(D_eff+1) = {gamma_eff:.8f}")
    print()

    print("## Projected residual")
    print()
    print(f"residual source = x(1-x) = {residual_source:.8f}")
    print(f"half-cycle projection = 2/pi = {half_cycle:.8f}")
    print(f"phase residual Q_phase = (2/pi)x(1-x) = {q_phase:.8f}")
    print(f"geometric projection = (2/pi)sigma^gamma_eff = {geom_projection:.8f}")
    print(f"A3c residual Q_A3c = (2/pi)sigma^gamma_eff x(1-x) = {q_a3c:.8f}")
    print(f"large-angle fractional residual Q_A3c/sigma = {large_angle_fractional_residual:.8f}")
    print()

    print("## Candidate anomaly amplitudes")
    print()
    print("| quantity | equation | CE value | status |")
    print("|---|---|---:|---|")
    print(f"| quadrupole power suppression, phase | `(2/pi)^2` | {power_suppression_phase:.8f} | Open test |")
    print(f"| quadrupole power suppression, geom | `[(2/pi)sigma^gamma]^2` | {power_suppression_geom:.8f} | Open test |")
    print(f"| quadrupole cosmic variance | `sqrt(2/5)` | {quadrupole_cv:.8f} | reference noise floor |")
    print(f"| octupole cosmic variance | `sqrt(2/7)` | {octupole_cv:.8f} | reference noise floor |")
    print(f"| quadrupole pull, phase suppression | `(S_Q-1)/sqrt(2/5)` | {quadrupole_pull_phase:+.2f} sigma | not decisive |")
    print(f"| quadrupole pull, geom suppression | `(S_Q-1)/sqrt(2/5)` | {quadrupole_pull_geom:+.2f} sigma | not decisive |")
    print(f"| hemispherical contrast, normalized | `2 Q_A3c/sigma` | {hemispherical_contrast:.8f} | Open test |")
    print(f"| hemispherical contrast, plain | `2 Q_A3c` | {hemispherical_contrast_plain:.8f} | Open test |")
    print()

    print("## Verdict")
    print()
    print("The residual projection naturally gives O(0.38-0.41) quadrupole power suppression")
    print("and O(5.7-6.0%) hemispherical contrast handles.")
    print("This is not a closure: no preferred axis, phase map, or likelihood comparison is derived yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
