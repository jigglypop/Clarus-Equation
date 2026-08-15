"""L5-H1 operator comparison. Independent algebra. No production import.

Exact identities first. One float witness at the U0 center.
Not a theorem, not a promotion.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
R_HI = Fraction(81, 16)
R_LO = Fraction(63, 16)
R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))
BC_B_HI = Fraction(157, 297)
R0_B_HI = Fraction(6, 11)
CENTER_M = 0.5
CENTER_B = 49.0 / 99.0
T = 32


def mtilde(m, b, u, r):
    return m * (1.0 + u * r * (1.0 - m) - float(LAM) * (1.0 - b))


def step_float(m, b, u, r):
    raw = mtilde(m, b, u, r)
    mt = raw if raw > 0.0 else 0.0
    next_m = mt / 2.0 if mt >= float(THETA) else mt
    next_b = (1.0 - float(DELTA)) * b + float(RHO) * m * (1.0 - b)
    return next_m, next_b


def iterate_float(m, b, u, r, ticks=T):
    for _ in range(ticks):
        m, b = step_float(m, b, u, r)
    return m, b


def occ_float(m, b) -> int:
    return int(float(R0_M[0]) <= m <= float(R0_M[1]) and float(R0_B[0]) <= b <= float(R0_B[1]))


def lines():
    out = []
    r_hi = float(R_HI)
    r_lo = float(R_LO)

    factor_bc = 1 - LAM * (1 - BC_B_HI)
    factor_r0 = 1 - LAM * (1 - R0_B_HI)
    crit = 1 - 1 / LAM
    out.append(f"r(3/4) = {R_HI}")
    out.append(f"r(1/4) = {R_LO}")
    out.append(f"u=0 sign change at b = {crit}")
    out.append(f"1-lambda(1-b_hi B_c) = {factor_bc} < 0 : {factor_bc < 0}")
    out.append(f"1-lambda(1-b_hi R0) = {factor_r0} < 0 : {factor_r0 < 0}")
    out.append(f"m=0 absorbing under any u: tilde m = 0 identically")

    # wash+sigma / no-store / dual: exact from inherited cuts, no u=1 walk
    out.append("wash+sigma tau1: sigma=1 oA=1  (O-E1 after wash, uA=1)")
    out.append("wash+sigma tau2: sigma=0 oA=0  (uA=0 one-step extinction)")
    out.append("no-store both: oA=1  (wash then uA=1, O-E1)")
    out.append("dual-product both: bits (1,0) vs (0,1), product=0, oA=0")
    out.append("dual-own tau1: bitA=0 oA=0; tau2: bitA=1 oA=1")
    out.append("q-memory tau1: qA=1/4 uA=1 oA=0 (O-E1); tau2: qA=3/4 uA=1 oA=1")

    # float witness: leftover after alpha, then beta
    m, b = CENTER_M, CENTER_B
    m_s1, b_s1 = iterate_float(m, b, 1.0, r_hi)
    m_a1, b_a1 = iterate_float(m, b, 0.0, r_hi)
    m_s2, b_s2 = iterate_float(m, b, 0.0, r_hi)
    m_a2, b_a2 = iterate_float(m, b, 1.0, r_hi)
    out.append(
        f"center leftover tau1: oS={occ_float(m_s1, b_s1)} oA={occ_float(m_a1, b_a1)} "
        f"mA={m_a1:.6e} bA={b_a1:.6e}"
    )
    out.append(
        f"center leftover tau2: oS={occ_float(m_s2, b_s2)} oA={occ_float(m_a2, b_a2)} "
        f"mA={m_a2:.6f} bA={b_a2:.6f}"
    )

    m_a1b, b_a1b = iterate_float(m_a1, b_a1, 1.0, r_hi)
    m_a2b, b_a2b = iterate_float(m_a2, b_a2, 1.0, r_hi)
    out.append(
        f"no-wash tau1 after beta uA=1: oA={occ_float(m_a1b, b_a1b)} mA={m_a1b:.6e}"
    )
    out.append(
        f"no-wash tau2 after beta uA=1: oA={occ_float(m_a2b, b_a2b)} "
        f"mA={m_a2b:.6f} bA={b_a2b:.6f}"
    )

    m64, b64 = iterate_float(m_s1, b_s1, 1.0, r_hi)
    out.append(
        f"center u=1 window2 from window1 image: o={occ_float(m64, b64)} "
        f"m={m64:.6f} b={b64:.6f}"
    )

    # q-memory float: wash (m,b), keep written q
    m_q1, b_q1 = iterate_float(m, b, 1.0, r_lo)
    m_q2, b_q2 = iterate_float(m, b, 1.0, r_hi)
    out.append(
        f"q-memory float tau1 U0 x 1/4 u=1: oA={occ_float(m_q1, b_q1)} m={m_q1:.6f}"
    )
    out.append(
        f"q-memory float tau2 U0 x 3/4 u=1: oA={occ_float(m_q2, b_q2)} m={m_q2:.6f}"
    )
    return out


def main() -> None:
    text = "\n".join(lines()) + "\n"
    path = Path(__file__).with_suffix(".txt")
    path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
