"""L8-H1 route comparison: four maps on S. Not a theorem."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
KAPPA = Fraction(1, 4)
R0 = Fraction(9, 2)
COPY_SEL = Fraction(1, 2)
MUT = Fraction(3, 32)
INH = Fraction(1)

P_STAR = (Fraction(1, 2), Fraction(49, 99), Fraction(3, 4))
P_CIRC = (Fraction(7, 15), Fraction(49, 99), Fraction(3, 4))
E2 = (Fraction(0), Fraction(1))
ACTION_STAR = (Fraction(7187, 12672), Fraction(491, 990))
ACTION_CIRC = (Fraction(16891, 29700), Fraction(133, 270))
R0_MASS = (Fraction(2, 5), Fraction(3, 5))
R0_BOUND = (Fraction(4, 9), Fraction(6, 11))


def r_of(q: Fraction) -> Fraction:
    return R0 * (1 + KAPPA * (2 * q - 1))


def next_mass(m: Fraction, b: Fraction, q: Fraction, u: Fraction) -> Fraction:
    raw = m * (1 + u * r_of(q) * (1 - m) - LAM * (1 - b))
    pred = max(Fraction(0), raw)
    return pred / 2 if pred >= THETA else pred


def next_boundary(m: Fraction, b: Fraction) -> Fraction:
    return (1 - DELTA) * b + RHO * m * (1 - b)


def next_label(q: Fraction) -> Fraction:
    copied = q + COPY_SEL * q * (1 - q) * (2 * q - 1) + MUT * (1 - 2 * q)
    return Fraction(1, 2) + INH * (copied - Fraction(1, 2))


def in_r0(m: Fraction, b: Fraction) -> int:
    return int(R0_MASS[0] <= m <= R0_MASS[1] and R0_BOUND[0] <= b <= R0_BOUND[1])


def cube_step(p: tuple[Fraction, Fraction, Fraction], u: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    m, b, q = p
    return next_mass(m, b, q, u), next_boundary(m, b), next_label(q)


def phi(p: tuple[Fraction, Fraction, Fraction]) -> tuple:
    z_s = cube_step(p, Fraction(0))
    z_a = cube_step(p, Fraction(1))
    return (1, E2, z_s, z_a, 1, 1)


def main() -> None:
    lines: list[str] = []
    lines.append("L8-H1 route numbers. Not a theorem. Not AGI GO.")
    lines.append(f"r(3/4) = {r_of(Fraction(3, 4))}")
    leak_factor = 1 - LAM * (1 - Fraction(49, 99))
    lines.append(f"1 - lambda(1-b) at b=49/99 = {leak_factor}")
    for name, p in (("star", P_STAR), ("circ", P_CIRC)):
        z_s = cube_step(p, Fraction(0))
        z_a = cube_step(p, Fraction(1))
        o_a = in_r0(p[0], p[1])
        lines.append(f"{name} current o^A = {o_a}")
        lines.append(f"{name} Z^S(u=0) = {z_s}")
        lines.append(f"{name} Z^A(u=1) = {z_a}")
        lines.append(f"{name} Phi slots = t=1 E=e2 sigma=1 I=1")
    assert cube_step(P_STAR, Fraction(1))[:2] == ACTION_STAR
    assert cube_step(P_CIRC, Fraction(1))[:2] == ACTION_CIRC
    assert cube_step(P_STAR, Fraction(0))[0] == 0
    assert cube_step(P_CIRC, Fraction(0))[0] == 0
    assert cube_step(P_STAR, Fraction(0))[1] == ACTION_STAR[1]
    assert cube_step(P_CIRC, Fraction(0))[1] == ACTION_CIRC[1]
    assert next_label(Fraction(3, 4)) == Fraction(3, 4)
    assert in_r0(*P_STAR[:2]) == 1 and in_r0(*P_CIRC[:2]) == 1
    phi_star = phi(P_STAR)
    phi_circ = phi(P_CIRC)
    lines.append(f"Phi(star) == Phi(circ): {phi_star == phi_circ}")
    lines.append(f"o^A constant on S: {in_r0(*P_STAR[:2]) == in_r0(*P_CIRC[:2]) == 1}")
    lines.append("K_bit codomain {0,1}; Phi codomain is H-space. Equality type-fails.")
    lines.append("K_act emits only action (m',b'). Missing t,E,Z^S,q,sigma,I.")
    lines.append("K_3 emits a third cube. Codomain is not H. Already rejected as required.")
    lines.append("K=Phi matches by construction on S.")
    text = "\n".join(lines) + "\n"
    out = Path(__file__).with_suffix(".txt")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
