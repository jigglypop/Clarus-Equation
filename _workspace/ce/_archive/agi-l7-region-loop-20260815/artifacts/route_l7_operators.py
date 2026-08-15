"""L7-H1 operator comparison. Independent algebra. No production import.

Exact identities from inherited cuts. One float witness at the U0 center.
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
    return int(
        float(R0_M[0]) <= m <= float(R0_M[1])
        and float(R0_B[0]) <= b <= float(R0_B[1])
    )


def lines():
    out = []
    r_hi = float(R_HI)

    factor_bc = 1 - LAM * (1 - BC_B_HI)
    factor_r0 = 1 - LAM * (1 - R0_B_HI)
    crit = 1 - 1 / LAM
    out.append(f"r(3/4) = {R_HI}")
    out.append(f"u=0 sign change at b = {crit}")
    out.append(f"1-lambda(1-b_hi B_c) = {factor_bc} < 0 : {factor_bc < 0}")
    out.append(f"1-lambda(1-b_hi R0) = {factor_r0} < 0 : {factor_r0 < 0}")
    out.append("m=0 absorbing under any u: tilde m = 0 identically")

    # Shared prefix: alpha = e1, W=I, wash. Exact from inherited cuts.
    out.append("alpha both: e1, uS=1 uA=0, oS=1 oA=0, sigma=1 (O-E1 / u=0 extinction)")

    # beta under L5 gate with sigma=1
    out.append("beta phi1 e2: uS=0 uA=1, oS=0 oA=1, I=1")
    out.append("beta phi2 e1: uS=1 uA=0, oS=1 oA=0, I=0")
    out.append("on this pair oS(beta) = 1 - oA(beta)")

    # gamma = e2, u_I^A = 1
    out.append("named-I loop: uA = I; phi1 oA=1; phi2 oA=0")
    out.append("frozen-sigma: uA = sigma = 1 both; oA=1 vs 1")
    out.append("sigma-overwrite: sigma <- oA(beta); uA = sigma; oA=1 vs 0")
    out.append("third-cube occupancy encode: oR = oA(beta); uA = oR; oA=1 vs 0 if bit preserved")

    # float witness: one washed epoch from center
    m, b = CENTER_M, CENTER_B
    m1, b1 = iterate_float(m, b, 1.0, r_hi)
    m0, b0 = iterate_float(m, b, 0.0, r_hi)
    out.append(
        f"center wash u=1: o={occ_float(m1, b1)} m={m1:.6f} b={b1:.6f}"
    )
    out.append(
        f"center wash u=0: o={occ_float(m0, b0)} m={m0:.6e} b={b0:.6f}"
    )
    return out


def main() -> None:
    text = "\n".join(lines()) + "\n"
    path = Path(__file__).with_suffix(".txt")
    path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
