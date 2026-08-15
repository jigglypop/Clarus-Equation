"""L6-H1 operator comparison. Independent algebra. No production import.

Exact one-step first. Float T=32 is a witness, not a hull.
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
U0_M = (Fraction(13, 30), Fraction(17, 30))
U0_B = (Fraction(137, 297), Fraction(157, 297))
P_STAR = (Fraction(1, 2), Fraction(49, 99))
P_CIRC = (Fraction(7, 15), Fraction(49, 99))
T = 32


def clip(value: Fraction) -> Fraction:
    if value < 0:
        return Fraction(0)
    if value > 1:
        return Fraction(1)
    return value


def step(mass: Fraction, boundary: Fraction, drive: Fraction):
    raw = mass * (1 + drive * R_HI * (1 - mass) - LAM * (1 - boundary))
    tilde = raw if raw > 0 else Fraction(0)
    divided = int(tilde >= THETA)
    next_mass = tilde / 2 if divided else tilde
    next_boundary = (1 - DELTA) * boundary + RHO * mass * (1 - boundary)
    return clip(next_mass), clip(next_boundary), tilde, divided


def in_u0(mass: Fraction, boundary: Fraction) -> bool:
    return U0_M[0] < mass < U0_M[1] and U0_B[0] < boundary < U0_B[1]


def occ_frac(mass: Fraction, boundary: Fraction) -> int:
    return int(R0_M[0] <= mass <= R0_M[1] and R0_B[0] <= boundary <= R0_B[1])


def step_float(mass: float, boundary: float, drive: float):
    raw = mass * (1.0 + drive * float(R_HI) * (1.0 - mass) - float(LAM) * (1.0 - boundary))
    tilde = raw if raw > 0.0 else 0.0
    divided = int(tilde >= float(THETA))
    next_mass = tilde / 2.0 if divided else tilde
    next_boundary = (1.0 - float(DELTA)) * boundary + float(RHO) * mass * (1.0 - boundary)
    next_mass = 0.0 if next_mass < 0.0 else (1.0 if next_mass > 1.0 else next_mass)
    next_boundary = (
        0.0 if next_boundary < 0.0 else (1.0 if next_boundary > 1.0 else next_boundary)
    )
    return next_mass, next_boundary, tilde, divided


def occ_float(mass: float, boundary: float) -> int:
    return int(
        float(R0_M[0]) <= mass <= float(R0_M[1])
        and float(R0_B[0]) <= boundary <= float(R0_B[1])
    )


def run_float(mass: float, boundary: float, drive_fn, ticks: int = T):
    n_div = 0
    n_out = 0
    first_out = None
    extinct_at = None
    for tick in range(ticks):
        bit = occ_float(mass, boundary)
        if bit == 0:
            n_out += 1
            if first_out is None:
                first_out = tick
        drive = drive_fn(mass, boundary)
        next_mass, next_boundary, tilde, divided = step_float(mass, boundary, drive)
        n_div += divided
        if tilde <= 0.0 and extinct_at is None:
            extinct_at = tick
        mass, boundary = next_mass, next_boundary
    return mass, boundary, occ_float(mass, boundary), n_div, n_out, first_out, extinct_at


def lines() -> list[str]:
    out: list[str] = []
    out.append("r(3/4) = %s" % R_HI)
    for name, point in (("P_star", P_STAR), ("P_circ", P_CIRC)):
        mass, boundary = point
        out.append(
            "%s in_U0=%s occ0=%s" % (name, in_u0(mass, boundary), occ_frac(mass, boundary))
        )

    out.append("=== one-step u=1 ===")
    readouts = {}
    for name, point in (("P_star", P_STAR), ("P_circ", P_CIRC)):
        mass, boundary, tilde, divided = step(point[0], point[1], Fraction(1))
        readouts[name] = (mass, boundary, tilde, divided)
        out.append("%s tilde=%s d=%s (m',b')=(%s, %s)" % (name, tilde, divided, mass, boundary))
    star = readouts["P_star"]
    circ = readouts["P_circ"]
    out.append("pair unequal = %s" % ((star[0], star[1]) != (circ[0], circ[1])))
    out.append("mass diff = %s" % (star[0] - circ[0]))
    out.append("boundary diff = %s" % (star[1] - circ[1]))
    out.append("division bits equal = %s" % (star[3] == circ[3]))

    out.append("=== one-step u=m ===")
    for name, point in (("P_star", P_STAR), ("P_circ", P_CIRC)):
        mass, boundary, tilde, divided = step(point[0], point[1], point[0])
        out.append("%s tilde=%s d=%s (m',b')=(%s, %s)" % (name, tilde, divided, mass, boundary))

    out.append("=== T=32 float witnesses (not a hull) ===")
    drives = {
        "const_u1": lambda mass, boundary: 1.0,
        "u=m": lambda mass, boundary: mass,
        "u=1[R0]": lambda mass, boundary: float(occ_float(mass, boundary)),
    }
    for label, drive_fn in drives.items():
        out.append("--- %s ---" % label)
        bits = []
        for name, point in (("P_star", P_STAR), ("P_circ", P_CIRC)):
            end = run_float(float(point[0]), float(point[1]), drive_fn)
            bits.append(end[2])
            out.append(
                "%s m32=%.12f b32=%.12f occ32=%s divs=%s out=%s first_out=%s extinct_at=%s"
                % (name, end[0], end[1], end[2], end[3], end[4], end[5], end[6])
            )
        out.append("occupancy equal = %s" % (bits[0] == bits[1]))

    out.append("H1 cites O-E1 occupancy 1 at q=3/4, u=1; no T=32 hull")
    out.append("sigma pair = (1, 1); bit predictor domain is a singleton")
    return out


def main() -> None:
    text = lines()
    for row in text:
        print(row)
    path = Path(__file__).with_suffix(".txt")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
