"""Gate D: initial-condition and recursive-entropy audit.

This card tests how far the d=0 self-recursive identity boundary can be used as
an initial-cosmology condition.  The closed object is the dimensionless
recursive entropy S_R = -log(x) = D(1-x), not the absolute thermodynamic entropy
of the observed universe.
"""

from __future__ import annotations

import math


ALPHA_S = 0.11789
D = 3.0
N_GAUGE = 12.0


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


def iterations_for_decay(k: float, target: float) -> float:
    if not (0.0 < k < 1.0):
        return float("nan")
    return math.log(target) / math.log(k)


def main() -> int:
    sin2_theta_w = 4.0 * ALPHA_S ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = D + delta
    x0 = 1.0
    sigma0 = 0.0
    s0 = 0.0

    x = bootstrap_x(d_eff)
    sigma = 1.0 - x
    s_recursive = -math.log(x)
    s_fixed = d_eff * sigma
    residual = x - math.exp(-(1.0 - x) * d_eff)
    contraction = d_eff * x
    lyapunov = math.log(contraction)
    n_e = (D / 2.0) * d_eff * N_GAUGE
    entropy_per_efold = s_recursive / n_e

    derivative_x = dx_dD(x, d_eff)
    derivative_s = -derivative_x / x
    derivative_s_closed = sigma / (1.0 - d_eff * x)

    half_cycle_projection = 2.0 / math.pi
    phase_entropy = half_cycle_projection * s_recursive
    spatial_projection = d_eff / (d_eff + 1.0)
    spatial_entropy = spatial_projection * s_recursive

    print("# Initial Conditions and Recursive Entropy Gate")
    print()
    print("## Boundary and physical branch")
    print()
    print(f"d=0 boundary: x0 = {x0:.8f}, sigma0 = {sigma0:.8f}, S_R0 = {s0:.8f}")
    print(f"sin2(theta_W) = {sin2_theta_w:.8f}")
    print(f"delta = sin2(theta_W)(1-sin2(theta_W)) = {delta:.8f}")
    print(f"D_eff = 3 + delta = {d_eff:.8f}")
    print(f"x = epsilon^2 = {x:.8f}")
    print(f"sigma = 1 - x = {sigma:.8f}")
    print(f"r_R = x - exp(-(1-x)D_eff) = {residual:+.3e}")
    print()

    print("## Recursive entropy")
    print()
    print(f"S_R = -log(x) = {s_recursive:.8f}")
    print(f"S_R = D_eff(1-x) = {s_fixed:.8f}")
    print(f"Delta S_R = S_R - S_R0 = {s_recursive - s0:.8f}")
    print(f"dS_R/dD = (1-x)/(1-Dx) = {derivative_s_closed:.8f}")
    print(f"check: -x^(-1) dx/dD = {derivative_s:.8f}")
    print(f"half-cycle projected entropy = (2/pi) S_R = {phase_entropy:.8f}")
    print(f"spatial/spacetime projected entropy = [D/(D+1)] S_R = {spatial_entropy:.8f}")
    print()

    print("## Stability and e-fold link")
    print()
    print(f"contraction k = F_D'(x*) = D_eff x = {contraction:.8f}")
    print(f"Lyapunov log(k) = {lyapunov:.8f}")
    print(f"k^10 = {contraction**10:.8e}")
    print(f"k^20 = {contraction**20:.8e}")
    print(f"iterations to suppress residual by 1e-60 = {iterations_for_decay(contraction, 1.0e-60):.2f}")
    print(f"N_e = (3/2) D_eff N_gauge = {n_e:.8f}")
    print(f"S_R/N_e = {entropy_per_efold:.8f}")
    print()

    print("## Verdict")
    print()
    print("Closed: d=0 is a zero-recursive-entropy boundary and d=3 is a finite-entropy contracted branch.")
    print("Open: absolute thermodynamic entropy, curvature initial data, and singularity avoidance need a physical scale map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
