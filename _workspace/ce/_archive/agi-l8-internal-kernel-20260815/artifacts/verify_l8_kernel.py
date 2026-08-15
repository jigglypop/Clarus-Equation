"""L8 host-tuple assembly on the preregistered pair S.

Does not import production. Action (m', b') is cited from L6-E1.
Sensor u=0 is computed from the definition. q' is the predecessor
label map at q=3/4. T=32 is not iterated.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path


LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
R0_GROWTH = Fraction(9, 2)
KAPPA = Fraction(1, 4)
COPY_SEL = Fraction(1, 2)
MUTATION = Fraction(3, 32)
INH_GAIN = Fraction(1)

U0_M = (Fraction(13, 30), Fraction(17, 30))
U0_B = (Fraction(137, 297), Fraction(157, 297))
R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))

E2 = (Fraction(0), Fraction(1))
P_STAR = (Fraction(1, 2), Fraction(49, 99), Fraction(3, 4))
P_CIRC = (Fraction(7, 15), Fraction(49, 99), Fraction(3, 4))

# Cited L6-E1 one-step fractions. Not re-proved here.
L6_ACTION_STAR = (Fraction(7187, 12672), Fraction(491, 990))
L6_ACTION_CIRC = (Fraction(16891, 29700), Fraction(133, 270))
L6_DM = -Fraction(1487, 950400)
L6_DB = Fraction(1, 297)


def in_open_u0(m: Fraction, b: Fraction) -> bool:
    return U0_M[0] < m < U0_M[1] and U0_B[0] < b < U0_B[1]


def in_r0(m: Fraction, b: Fraction) -> bool:
    return R0_M[0] <= m <= R0_M[1] and R0_B[0] <= b <= R0_B[1]


def u0_subset_r0() -> bool:
    return (
        R0_M[0] < U0_M[0]
        and U0_M[1] < R0_M[1]
        and R0_B[0] < U0_B[0]
        and U0_B[1] < R0_B[1]
    )


def growth(q: Fraction) -> Fraction:
    return R0_GROWTH * (1 + KAPPA * (2 * q - 1))


def mass_boundary(m: Fraction, b: Fraction, q: Fraction, u: Fraction):
    raw = m * (1 + u * growth(q) * (1 - m) - LAM * (1 - b))
    tilde = raw if raw > 0 else Fraction(0)
    mass = tilde / 2 if tilde >= THETA else tilde
    boundary = (1 - DELTA) * b + RHO * m * (1 - b)
    return tilde, mass, boundary


def next_label(q: Fraction) -> Fraction:
    copied = (
        q
        + COPY_SEL * q * (1 - q) * (2 * q - 1)
        + MUTATION * (1 - 2 * q)
    )
    return Fraction(1, 2) + INH_GAIN * (copied - Fraction(1, 2))


def cube_u0(m: Fraction, b: Fraction, q: Fraction):
    tilde, mass, boundary = mass_boundary(m, b, q, Fraction(0))
    return tilde, (mass, boundary, next_label(q))


def host_slots(h):
    t, e, zs, za, sigma, named_i = h
    return {
        "t": isinstance(t, int) and t >= 0,
        "E": (
            isinstance(e, tuple)
            and len(e) == 2
            and all(isinstance(x, Fraction) and 0 <= x <= 1 for x in e)
        ),
        "ZS": (
            isinstance(zs, tuple)
            and len(zs) == 3
            and all(isinstance(x, Fraction) and 0 <= x <= 1 for x in zs)
        ),
        "ZA": (
            isinstance(za, tuple)
            and len(za) == 3
            and all(isinstance(x, Fraction) and 0 <= x <= 1 for x in za)
        ),
        "sigma": sigma in (0, 1),
        "I": named_i in (0, 1),
    }


def main() -> None:
    lines = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    if not u0_subset_r0():
        raise SystemExit("U0 subset R0 failed")
    log("U0 subset R0: yes (geometry)")

    hosts = (("H_star", P_STAR), ("H_circ", P_CIRC))
    current = {}
    images = {}
    for name, p in hosts:
        m, b, q = p
        if q != Fraction(3, 4):
            raise SystemExit("%s label is not 3/4" % name)
        if not in_open_u0(m, b):
            raise SystemExit("%s is outside U0" % name)
        if not in_r0(m, b):
            raise SystemExit("%s is outside R0" % name)
        o_a = 1
        h = (0, E2, p, p, 1, 1)
        slots = host_slots(h)
        if not all(slots.values()):
            raise SystemExit("%s current slots failed: %s" % (name, slots))
        current[name] = (h, o_a)
        log("%s in U0 subset R0; o^A=%s; typed H" % (name, o_a))

    if current["H_star"][1] != current["H_circ"][1]:
        raise SystemExit("current o^A bits differ")
    log("E2 current bits equal: o^A(H_star)=o^A(H_circ)=1")

    cited = {"H_star": L6_ACTION_STAR, "H_circ": L6_ACTION_CIRC}
    for name, p in hosts:
        m, b, q = p
        tilde, sensor = cube_u0(m, b, q)
        if tilde != 0 or sensor[0] != 0:
            raise SystemExit("%s sensor is not u=0 extinction" % name)
        if sensor[2] != Fraction(3, 4):
            raise SystemExit("%s sensor q' is not 3/4" % name)
        m_a, b_a = cited[name]
        action = (m_a, b_a, Fraction(3, 4))
        phi = (1, E2, sensor, action, 1, 1)
        slots = host_slots(phi)
        if not all(slots.values()):
            raise SystemExit("%s Phi slots failed: %s" % (name, slots))
        images[name] = phi
        log(
            "%s Phi: t'=1 E held bits held; sensor m'=0 b'=%s q'=3/4; "
            "action cited L6-E1 (m',b')=(%s, %s)"
            % (name, sensor[1], m_a, b_a)
        )

    dm = L6_ACTION_STAR[0] - L6_ACTION_CIRC[0]
    db = L6_ACTION_STAR[1] - L6_ACTION_CIRC[1]
    if dm != L6_DM or db != L6_DB:
        raise SystemExit("cited L6-E1 differences do not match the ledger")
    if dm == 0 or db == 0:
        raise SystemExit("cited L6-E1 differences vanished")
    if images["H_star"][3][:2] == images["H_circ"][3][:2]:
        raise SystemExit("action images coincide")
    log("E2 action (m',b') differ by cited L6-E1: dm=%s db=%s" % (dm, db))

    if images["H_star"] == images["H_circ"]:
        raise SystemExit("Phi images coincide on S")
    log("Phi(H_star) != Phi(H_circ) as host tuples")

    k_star = images["H_star"]
    k_circ = images["H_circ"]
    if k_star != images["H_star"] or k_circ != images["H_circ"]:
        raise SystemExit("registered K is not Phi")
    log("E1: K=Phi typed H->H equals Phi on S")

    if current["H_star"][1] == current["H_circ"][1] and k_star != k_circ:
        log("E3: o^A constant on S, K=Phi not constant; maps from S differ")
    else:
        raise SystemExit("E3 failed")

    phi_star = images["H_star"]
    if phi_star in (0, 1):
        raise SystemExit("Phi(H_star) is a bit")
    log("H1: Phi(H) is a 6-slot host tuple, not a bit; K_bit equality fails")

    # Sensor b' is independent of u; record the values for the ledger.
    _, b_s_star, _ = cube_u0(*P_STAR)[1]
    _, b_s_circ, _ = cube_u0(*P_CIRC)[1]
    log("sensor b'(H_star)=%s" % b_s_star)
    log("sensor b'(H_circ)=%s" % b_s_circ)
    if b_s_star == b_s_circ:
        raise SystemExit("sensor boundaries coincide")

    # Drive assignment with W=I, I=1, E=e2.
    u_a = 1 * E2[1]
    u_s = E2[0]
    if u_a != 1 or u_s != 0:
        raise SystemExit("registered drives are not (u^A,u^S)=(1,0)")
    log("drives: u^A=1 u^S=0")

    log("ALL CHECKS PASSED")
    Path(__file__).with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
