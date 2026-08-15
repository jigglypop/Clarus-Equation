"""Independent L4 algebra. No production import.

Preregister: artifacts/l4_preregister.md (written first; no occupancy).
This script checks only the new L4 reductions:
  - u=0 kills the growth term and forces m_32 = 0 on B_c
  - W=I, A_1, A_J, A_L send e1, e2 to the registered drives
It does not re-prove N-E1, N-E3, or N-E2 / O-E1 occupancy on U0.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
R0_M = (Fraction(2, 5), Fraction(3, 5))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))
T = 32

E1 = (Fraction(1), Fraction(0))
E2 = (Fraction(0), Fraction(1))

W_I = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
A_1 = ((Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2)))
A_J = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
A_L = ((Fraction(1), Fraction(0)), (Fraction(1), Fraction(0)))


def matvec(a, e):
    return (
        a[0][0] * e[0] + a[0][1] * e[1],
        a[1][0] * e[0] + a[1][1] * e[1],
    )


def mtilde_u0(m, b):
    raw = m * (1 - LAM * (1 - b))
    return raw if raw > 0 else Fraction(0)


def step_u0(m, b):
    mt = mtilde_u0(m, b)
    next_m = mt / 2 if mt >= THETA else mt
    next_b = (1 - DELTA) * b + RHO * m * (1 - b)
    return next_m, next_b


def lines():
    out = []
    b_hi = BC_B[1]
    crit = 1 - 1 / LAM
    gap = crit - b_hi
    factor_at_bhi = 1 - LAM * (1 - b_hi)
    out.append(f"B_c b_hi = {b_hi}")
    out.append(f"u=0 sign change at b = 1 - 1/lambda = {crit}")
    out.append(f"crit - b_hi = {gap} > 0 : {gap > 0}")
    out.append(f"1 - lambda(1-b_hi) = {factor_at_bhi} < 0 : {factor_at_bhi < 0}")
    out.append(f"B_c m_lo = {BC_M[0]} > 0 : {BC_M[0] > 0}")

    mt = mtilde_u0(BC_M[0], b_hi)
    m1, _ = step_u0(BC_M[0], b_hi)
    out.append(f"worst-case mtilde(u=0) on closed B_c = {mt}")
    out.append(f"first-step m from B_c corner (m_lo,b_hi) = {m1}")
    assert mt == 0
    assert m1 == 0
    assert factor_at_bhi < 0
    assert gap > 0

    m, b = BC_M[0], b_hi
    for _ in range(T):
        m, b = step_u0(m, b)
    out.append(
        f"after T=32 from B_c corner: m={m} in R0_m? {R0_M[0] <= m <= R0_M[1]}"
    )
    assert m == 0
    assert not (R0_M[0] <= m <= R0_M[1])

    for name, a in (("W=I", W_I), ("A_1", A_1), ("A_J", A_J), ("A_L", A_L)):
        u1 = matvec(a, E1)
        u2 = matvec(a, E2)
        out.append(f"{name} e1 -> u={u1}")
        out.append(f"{name} e2 -> u={u2}")
        out.append(f"{name} same drive: {u1 == u2}")

    assert matvec(W_I, E1) == (Fraction(1), Fraction(0))
    assert matvec(W_I, E2) == (Fraction(0), Fraction(1))
    assert matvec(A_1, E1) == matvec(A_1, E2) == (Fraction(1, 2), Fraction(1, 2))
    assert matvec(A_J, E1) == (Fraction(0), Fraction(1))
    assert matvec(A_J, E2) == (Fraction(1), Fraction(0))
    assert matvec(A_L, E1) == (Fraction(1), Fraction(1))
    assert matvec(A_L, E2) == (Fraction(0), Fraction(0))

    out.append("q0 registered = 3/4 (fixed; not scored here)")
    out.append("u=1 occupancy on U0 is cited from N-E2 / O-E1, not recomputed")
    out.append("ALGEBRA_OK")
    return "\n".join(out) + "\n"


def main() -> None:
    text = lines()
    path = Path(__file__).with_suffix(".txt")
    path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
