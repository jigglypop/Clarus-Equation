"""Independent L5 algebra. No production import.

Preregister: artifacts/l5_preregister.md (written first; no occupancy).
This script checks only the new L5 reductions:
  - body indexing (S,A)=(L,R) and W=I drives
  - sigma gate u^A = sigma * u_I(e^beta)
  - u=0 one-step extinction on closed B_c
  - m=0 absorbs any later drive
  - wash no-store: both tasks share the same beta drive and start
It does not re-prove N-E1, N-E3, O-E1, or L4-E1--E3 occupancy on U0.
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


def matvec(a, e):
    return (
        a[0][0] * e[0] + a[0][1] * e[1],
        a[1][0] * e[0] + a[1][1] * e[1],
    )


def mtilde(m, b, u):
    raw = m * (1 + u * Fraction(0) - LAM * (1 - b))
    # growth term omitted on purpose for the u=0 identity;
    # the general absorbing-zero check uses the full bracket below.
    return raw if raw > 0 else Fraction(0)


def mtilde_full(m, b, u, r=Fraction(81, 16)):
    raw = m * (1 + u * r * (1 - m) - LAM * (1 - b))
    return raw if raw > 0 else Fraction(0)


def lines():
    out = []

    u_e1 = matvec(W_I, E1)
    u_e2 = matvec(W_I, E2)
    out.append(f"I e1 = {u_e1}")
    out.append(f"I e2 = {u_e2}")
    assert u_e1 == (Fraction(1), Fraction(0))
    assert u_e2 == (Fraction(0), Fraction(1))

    u_s_e1, u_a_e1 = u_e1
    u_s_e2, u_a_e2 = u_e2
    out.append(f"sensor u_I(e1) = {u_s_e1}")
    out.append(f"sensor u_I(e2) = {u_s_e2}")
    out.append(f"action u_I(e1) = {u_a_e1}")
    out.append(f"action u_I(e2) = {u_a_e2}")
    assert u_s_e1 == 1 and u_s_e2 == 0
    assert u_a_e1 == 0 and u_a_e2 == 1

    # Role-split drives in epoch beta = e2. Occupancy bits themselves
    # are not scored here; only the gated drives.
    sigma_if_e1 = 1  # placeholder for the cited O-E1 / L4-E1 bit
    sigma_if_e2 = 0  # placeholder for the cited u=0 extinction bit
    u_a_tau1 = sigma_if_e1 * u_a_e2
    u_a_tau2 = sigma_if_e2 * u_a_e2
    out.append(f"role-split u^A(tau1) = {u_a_tau1}")
    out.append(f"role-split u^A(tau2) = {u_a_tau2}")
    assert u_a_tau1 == 1
    assert u_a_tau2 == 0
    assert u_a_tau1 != u_a_tau2

    u_a_nostore_tau1 = u_a_e2
    u_a_nostore_tau2 = u_a_e2
    out.append(f"no-store u^A(tau1) = {u_a_nostore_tau1}")
    out.append(f"no-store u^A(tau2) = {u_a_nostore_tau2}")
    assert u_a_nostore_tau1 == u_a_nostore_tau2 == 1

    b_hi = BC_B[1]
    crit = 1 - 1 / LAM
    gap = crit - b_hi
    factor_at_bhi = 1 - LAM * (1 - b_hi)
    out.append(f"B_c b_hi = {b_hi}")
    out.append(f"u=0 sign change at b = 1 - 1/lambda = {crit}")
    out.append(f"crit - b_hi = {gap} > 0 : {gap > 0}")
    out.append(f"1 - lambda(1-b_hi) = {factor_at_bhi}")
    assert factor_at_bhi == Fraction(-53, 297)
    assert gap == Fraction(106, 1485)
    assert factor_at_bhi < 0
    assert gap > 0
    assert BC_M[0] > 0

    mt = mtilde(BC_M[0], b_hi, 0)
    out.append(f"worst-case mtilde(u=0) on closed B_c = {mt}")
    assert mt == 0

    # m=0 absorbs every later drive, including u=1 at q=3/4.
    for u in (Fraction(0), Fraction(1)):
        for b in (Fraction(0), b_hi, Fraction(1)):
            abs_mt = mtilde_full(Fraction(0), b, u)
            assert abs_mt == 0
    out.append("m=0 absorbs u in {0,1} at every tested b")

    # After one u=0 step from any closed B_c point, mass is 0, so the
    # no-wash right copy of tau1 stays at m=0 through epoch beta.
    m, b = BC_M[0], b_hi
    raw = m * (1 - LAM * (1 - b))
    m1 = Fraction(0) if raw <= 0 else raw
    assert m1 == 0
    for _ in range(T):
        m = mtilde_full(m, b, Fraction(1))
        b = (1 - DELTA) * b + RHO * m * (1 - b)
    out.append(f"no-wash tau1 action after beta from m=0: m={m}")
    assert m == 0
    assert not (R0_M[0] <= m <= R0_M[1])

    out.append("operators: role-split drives differ; no-store drives coincide")
    out.append("H1 readout on tau2 after beta is not scored (start not U0)")
    return out


def main() -> None:
    text = "\n".join(lines()) + "\n"
    out_path = Path(__file__).with_suffix(".txt")
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
