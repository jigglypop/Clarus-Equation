"""Independent Fraction one-step of F_{1/4} on the L6 registered pair.

Does not import production. Two algebraic paths must agree.
T=32 is not iterated. Occupancy is a citation of O-E1, not a hull.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path


R0 = Fraction(9, 2)
LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
KAPPA = Fraction(1, 4)
DRIVE = Fraction(1)
CAPACITY = Fraction(1)

U0_M = (Fraction(13, 30), Fraction(17, 30))
U0_B = (Fraction(137, 297), Fraction(157, 297))

P_STAR = (Fraction(1, 2), Fraction(49, 99), Fraction(3, 4))
P_CIRC = (Fraction(7, 15), Fraction(49, 99), Fraction(3, 4))


def growth(q: Fraction) -> Fraction:
    return R0 * (1 + KAPPA * (2 * q - 1))


def path_define(m: Fraction, b: Fraction, q: Fraction, u: Fraction):
    """Direct substitution into (L4.1) and the boundary update."""

    r = growth(q)
    raw = m * (1 + u * r * (1 - m / CAPACITY) - LAM * (1 - b))
    tilde = raw if raw > 0 else Fraction(0)
    mass = tilde / 2 if tilde >= THETA else tilde
    boundary = (1 - DELTA) * b + RHO * m * (1 - b)
    return tilde, mass, boundary


def path_closed(m: Fraction, b: Fraction, q: Fraction, u: Fraction):
    """Expanded polynomial at the registered label and drive."""

    if q != Fraction(3, 4) or u != 1:
        raise ValueError("closed path is only for q=3/4, u=1")
    # r(3/4) = 81/16, so 1 + r(1-m) - lam(1-b) = 57/16 - (81/16)m + (5/2)b
    tilde = m * (Fraction(57, 16) - Fraction(81, 16) * m + Fraction(5, 2) * b)
    if tilde < 0:
        tilde = Fraction(0)
    mass = tilde / 2 if tilde >= THETA else tilde
    boundary = Fraction(9, 10) * b + Fraction(1, 5) * m - Fraction(1, 5) * m * b
    return tilde, mass, boundary


def in_open_u0(m: Fraction, b: Fraction) -> bool:
    return U0_M[0] < m < U0_M[1] and U0_B[0] < b < U0_B[1]


def main() -> None:
    lines = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    r_three_four = growth(Fraction(3, 4))
    log("r(3/4) = %s" % r_three_four)
    if r_three_four != Fraction(81, 16):
        raise SystemExit("growth at q=3/4 is not 81/16")

    names = (("P_star", P_STAR), ("P_circ", P_CIRC))
    readouts = {}
    for name, (m, b, q) in names:
        if q != Fraction(3, 4):
            raise SystemExit("%s label is not 3/4" % name)
        if not in_open_u0(m, b):
            raise SystemExit("%s is outside U0" % name)
        d1 = path_define(m, b, q, DRIVE)
        d2 = path_closed(m, b, q, DRIVE)
        if d1 != d2:
            raise SystemExit("%s path mismatch: %s vs %s" % (name, d1, d2))
        tilde, mass, boundary = d1
        if tilde < THETA:
            raise SystemExit("%s is not on the dividing branch" % name)
        if not (0 < mass < 1 and 0 < boundary < 1):
            raise SystemExit("%s image left (0,1)^2" % name)
        readouts[name] = (mass, boundary)
        log("%s tilde_m = %s" % (name, tilde))
        log("%s (m',b') = (%s, %s)" % (name, mass, boundary))
        log("%s dividing = %s" % (name, tilde >= THETA))

    star = readouts["P_star"]
    circ = readouts["P_circ"]
    unequal = star != circ
    log("E1 pair unequal = %s" % unequal)
    log("E1 mass unequal = %s" % (star[0] != circ[0]))
    log("E1 boundary unequal = %s" % (star[1] != circ[1]))
    log("mass difference = %s" % (star[0] - circ[0]))
    log("boundary difference = %s" % (star[1] - circ[1]))

    # E2: a map from {0,1} assigns one value to the singleton {1}.
    sigma_star = 1
    sigma_circ = 1
    log("sigma pair = (%s, %s)" % (sigma_star, sigma_circ))
    log("E2 domain on this pair is a singleton = %s" % (sigma_star == sigma_circ))
    log("E2 cannot match both true next states = %s" % (unequal and sigma_star == sigma_circ))

    # E3: same sigma, different readout => not a function of sigma.
    log("E3 not a function of sigma = %s" % (unequal and sigma_star == sigma_circ))

    # H1: citation only. Both points in U0 x {3/4}, u=1.
    log("H1 both in U0 x {3/4} = True")
    log("H1 cites O-E1 occupancy 1 at q=3/4, u=1; no T=32 hull")

    if not unequal:
        raise SystemExit("L6-E1 killed by equality")

    out = Path(__file__).with_suffix(".txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote %s" % out)


if __name__ == "__main__":
    main()
