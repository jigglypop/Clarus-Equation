"""Independent, standard-library numerical certificate for DSO-1..DSO-7.

It certifies algebraic consequences and counterexamples only; PASS is not a
derivation of the physical history-to-stress map.
"""
from __future__ import annotations

import math


def q_root(d: float) -> float:
    q = 0.0
    for _ in range(10000):
        nxt = math.exp(-d * (1.0 - q))
        if abs(nxt - q) < 2e-16:
            return nxt
        q = nxt
    raise RuntimeError("fixed-point iteration did not converge")


def main() -> None:
    d = 3.1777584234099736
    q = q_root(d)
    residual = q - math.exp(-d * (1.0 - q))
    dq_dd = -q * (1.0 - q) / (1.0 - d * q)
    assert abs(residual) < 3e-16
    assert abs(q - 0.0486467196440282) < 3e-15
    assert dq_dd < 0.0

    # The two-node exact CTMC has vacuum as the unique stationary measure.
    # From state 11 an activation is blocked by exclusion, disproving linear
    # first-moment closure: exact d<n1>/dt=-gamma1, not kappa12-gamma1.
    gamma1, kappa12 = 1.0, 2.0
    exact_11 = -gamma1
    naive_11 = kappa12 - gamma1
    assert exact_11 != naive_11

    # Random exponential parent lifetime makes a mixed Poisson offspring law.
    kappa, gamma = 2.0, 5.0
    mean = kappa / gamma
    variance = mean + mean * mean
    assert variance > mean

    # Probability is not an energy fraction without an equal conditional mean.
    p, w_event, w_else = q, 9.0, 1.0
    energy_fraction = p * w_event / (p * w_event + (1.0 - p) * w_else)
    assert abs(energy_fraction - p) > 1e-3

    # Same fixed point, distinct scalar amplitudes/vacuum offsets -> abundances.
    m, a1, a2, v1, v2 = 7.0, 1.0, 2.0, 0.0, 5.0
    rho1 = 0.5 * m * m * a1 * a1 + v1
    rho2 = 0.5 * m * m * a2 * a2 + v2
    assert rho1 != rho2

    # Period average of quadratic oscillator and constant-offset equation of state.
    n = 100000
    kin = pot = 0.0
    for j in range(n):
        x = 2.0 * math.pi * (j + 0.5) / n
        kin += 0.5 * (m * a1 * math.sin(x)) ** 2
        pot += 0.5 * (m * a1 * math.cos(x)) ** 2
    w_osc = (kin - pot) / (kin + pot)
    assert abs(w_osc) < 1e-12

    print("PASS DSO mathematical certificate")
    print(f"D={d:.16g} q_ext={q:.17g} residual={residual:.3e} dq_dD={dq_dd:.17g}")
    print(f"CTMC exact_dn1_at_11={exact_11:g} naive_linear={naive_11:g}")
    print(f"mixed_poisson_mean={mean:g} variance={variance:g}")
    print(f"probability={p:.12g} weighted_energy_fraction={energy_fraction:.12g}")
    print(f"same_q_different_rho={rho1:g},{rho2:g}; oscillator_w={w_osc:.3e}; vacuum_w=-1")


if __name__ == "__main__":
    main()
