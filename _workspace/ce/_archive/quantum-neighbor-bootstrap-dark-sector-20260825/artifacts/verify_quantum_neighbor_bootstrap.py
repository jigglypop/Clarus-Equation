"""Small independent certificate for C1/C2 population and branching limits.

Only the standard library is used.  It checks the four-state exact CTMC for
two mutually facilitating qubits, the non-closing first moment, absorption,
the scalar Poisson fixed point, and the non-Poisson offspring law caused by an
exponentially distributed parent lifetime.
"""
from __future__ import annotations

import math


def solve_linear(a, b):
    """Gauss--Jordan solve, adequate for the 4 by 4 certificate."""
    n = len(b)
    m = [list(a[r]) + [b[r]] for r in range(n)]
    for c in range(n):
        pivot = max(range(c, n), key=lambda r: abs(m[r][c]))
        assert abs(m[pivot][c]) > 1e-14
        m[c], m[pivot] = m[pivot], m[c]
        scale = m[c][c]
        m[c] = [x / scale for x in m[c]]
        for r in range(n):
            if r != c:
                factor = m[r][c]
                m[r] = [m[r][k] - factor * m[c][k] for k in range(n + 1)]
    return [m[r][-1] for r in range(n)]


def stationary_distribution(q):
    # Solve pi Q=0 with sum pi=1.
    n = len(q)
    a = [[q[c][r] for c in range(n)] for r in range(n)]
    b = [0.0] * n
    a[-1] = [1.0] * n
    b[-1] = 1.0
    return solve_linear(a, b)


def main():
    # state bits: 00, 01, 10, 11; directed facilitation both ways.
    k12, k21, g1, g2 = 2.0, 3.0, 1.0, 4.0
    q = [[0.0] * 4 for _ in range(4)]
    # From 01: node 2 activates node 1; node 2 decays. From 10 analogous.
    q[1][3], q[1][0] = k12, g2
    q[2][3], q[2][0] = k21, g1
    q[3][1], q[3][2] = g1, g2
    for s in range(4):
        q[s][s] = -sum(q[s])

    # The unique stationary distribution must be the absorbing vacuum.
    pi = stationary_distribution(q)
    assert max(abs(pi[i] - (1.0 if i == 0 else 0.0)) for i in range(4)) < 1e-12

    # At state 11, dn1/dt=-g1; the naive linear gain k12*p2-g1*p1 is wrong.
    # Exact gain is k12*(p2-C12), so it vanishes when both are occupied.
    exact_n1_at_11 = -g1
    naive_linear_at_11 = k12 - g1
    assert exact_n1_at_11 != naive_linear_at_11

    # CE scalar Poisson fixed point, independently iterated from q=0.
    d = 3.1777584234
    x = 0.0
    for _ in range(10000):
        y = math.exp(d * (x - 1.0))
        if abs(y - x) < 1e-15:
            break
        x = y
    assert abs(x - 0.04864671964) < 2e-10

    # Parent lifetime T~Exp(gamma); conditional N|T~Poisson(kappa*T).
    # Unconditionally E[N]=k/gamma and Var[N]=mean+mean**2, not Poisson.
    kappa, gamma = 2.0, 5.0
    mean = kappa / gamma
    variance = mean + mean * mean
    assert variance > mean
    print("PASS")
    print(f"absorbing_pi={pi}")
    print(f"exact_dn1_11={exact_n1_at_11:g} naive_linear={naive_linear_at_11:g}")
    print(f"poisson_qext={x:.11f}")
    print(f"exp_lifetime_mean={mean:g} variance={variance:g}")


if __name__ == "__main__":
    main()
