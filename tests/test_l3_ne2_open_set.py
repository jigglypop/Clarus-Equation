"""Locks for N-E2 occupancy split on preregistered U0 at T=32.

Occupancy split on U0 = int(Bc), not an R0-wide split and not a
division-count theorem. Machine pass is not a theorem. This file does
not claim 닫힘, 유도됨, autonomy, or AGI.

Interval helper is copied here (outer 10^18 rounding). The production
kernel is not changed. No V15–V18b or runtime import.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from reality_stone.clarus.universe_life_kernel import NOMINAL_GROWTH, growth_at_label


KAPPA = Fraction(1, 4)
LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
SCALE = 10**18
TICKS = 32

R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))
CENTER = (Fraction(1, 2), Fraction(49, 99))
U1_M = (Fraction(43, 90), Fraction(47, 90))
U1_B = (Fraction(431, 891), Fraction(451, 891))

# 11-math T=32 outer hull, q=1/4 on closed U0 = Bc.
U0_Q14_MHI = Fraction(48924156634417547, 125000000000000000)


def outward_lo(x: Fraction) -> Fraction:
    k = (x.numerator * SCALE) // x.denominator
    return Fraction(k, SCALE)


def outward_hi(x: Fraction) -> Fraction:
    k = (x.numerator * SCALE + x.denominator - 1) // x.denominator
    return Fraction(k, SCALE)


@dataclass(frozen=True)
class Box:
    mlo: Fraction
    mhi: Fraction
    blo: Fraction
    bhi: Fraction

    def outward(self) -> Box:
        return Box(
            outward_lo(self.mlo),
            outward_hi(self.mhi),
            outward_lo(self.blo),
            outward_hi(self.bhi),
        )


def raw_range(box: Box, r: Fraction) -> tuple[Fraction, Fraction]:
    """Exact coordinate range of m * (1 + r(1-m) - lam(1-b)) on a closed box."""

    alpha = 1 + r - LAM
    vals: list[Fraction] = []
    for b in (box.blo, box.bhi):
        a = alpha + LAM * b
        for m in (box.mlo, box.mhi):
            vals.append(a * m - r * m * m)
        mcrit = a / (2 * r)
        if box.mlo <= mcrit <= box.mhi:
            vals.append(a * mcrit - r * mcrit * mcrit)
    return min(vals), max(vals)


def bnext_range(box: Box) -> tuple[Fraction, Fraction]:
    def bp(m: Fraction, b: Fraction) -> Fraction:
        return (1 - DELTA) * b + RHO * m * (1 - b)

    corners = (
        bp(box.mlo, box.blo),
        bp(box.mlo, box.bhi),
        bp(box.mhi, box.blo),
        bp(box.mhi, box.bhi),
    )
    return min(corners), max(corners)


def classify(box: Box, r: Fraction) -> str:
    mt_lo, mt_hi = raw_range(box, r)
    if mt_lo >= THETA:
        return "div"
    if mt_hi < THETA:
        return "nodiv"
    return "mixed"


def image_single_branch(box: Box, r: Fraction, branch: str) -> Box:
    mt_lo, mt_hi = raw_range(box, r)
    if branch == "div":
        nlo, nhi = mt_lo / 2, mt_hi / 2
    elif branch == "nodiv":
        nlo, nhi = max(Fraction(0), mt_lo), max(Fraction(0), mt_hi)
    else:
        raise ValueError("mixed branches are not wrapped into one hull")
    blo2, bhi2 = bnext_range(box)
    return Box(nlo, nhi, blo2, bhi2).outward()


def subset(inner: Box, outer_m: tuple[Fraction, Fraction], outer_b: tuple[Fraction, Fraction]) -> bool:
    return (
        outer_m[0] <= inner.mlo
        and inner.mhi <= outer_m[1]
        and outer_b[0] <= inner.blo
        and inner.bhi <= outer_b[1]
    )


def disjoint_r0(box: Box) -> bool:
    return (
        box.mhi < R0_M[0]
        or box.mlo > R0_M[1]
        or box.bhi < R0_B[0]
        or box.blo > R0_B[1]
    )


def inside_r0(box: Box) -> bool:
    return subset(box, R0_M, R0_B)


def closed_box(mm: tuple[Fraction, Fraction], bb: tuple[Fraction, Fraction]) -> Box:
    return Box(mm[0], mm[1], bb[0], bb[1])


def trace_full(box: Box, r: Fraction, ticks: int = TICKS) -> tuple[bool, int, Box]:
    """T-step outer hull. Mixed boxes are not wrapped."""

    h = box
    mixed = False
    divs = 0
    for _t in range(ticks):
        br = classify(h, r)
        if br == "mixed":
            mixed = True
            break
        if br == "div":
            divs += 1
        h = image_single_branch(h, r, br)
    return mixed, divs, h


def test_preregistered_u0_is_interior_of_bc() -> None:
    # Geometry only. U0 = int(Bc) ⊂ Bc ⊂ R0. Not an R0-wide claim.
    assert BC_M[0] == Fraction(13, 30)
    assert BC_M[1] == Fraction(17, 30)
    assert BC_B[0] == Fraction(137, 297)
    assert BC_B[1] == Fraction(157, 297)
    assert R0_M[0] <= BC_M[0] < BC_M[1] <= R0_M[1]
    assert R0_B[0] <= BC_B[0] < BC_B[1] <= R0_B[1]
    assert (BC_M[0] + BC_M[1]) / 2 == CENTER[0]
    assert (BC_B[0] + BC_B[1]) / 2 == CENTER[1]
    assert R0_M[0] <= CENTER[0] <= R0_M[1]
    assert R0_B[0] <= CENTER[1] <= R0_B[1]

    assert U1_M[0] == Fraction(43, 90)
    assert U1_M[1] == Fraction(47, 90)
    assert U1_B[0] == Fraction(431, 891)
    assert U1_B[1] == Fraction(451, 891)
    assert BC_M[0] < U1_M[0] < U1_M[1] < BC_M[1]
    assert BC_B[0] < U1_B[0] < U1_B[1] < BC_B[1]
    assert (U1_M[1] - U1_M[0]) / 2 == (BC_M[1] - BC_M[0]) / 6
    assert (U1_B[1] - U1_B[0]) / 2 == (BC_B[1] - BC_B[0]) / 6


def test_u0_first_step_is_division_on_both_labels() -> None:
    r_lo = growth_at_label(Fraction(1, 4), KAPPA)
    r_hi = growth_at_label(Fraction(3, 4), KAPPA)
    assert r_lo == Fraction(63, 16)
    assert r_hi == Fraction(81, 16)
    assert NOMINAL_GROWTH == Fraction(9, 2)

    u0 = closed_box(BC_M, BC_B)
    lo14, hi14 = raw_range(u0, r_lo)
    lo34, hi34 = raw_range(u0, r_hi)
    assert (lo14, hi14) == (
        Fraction(1098217, 1425600),
        Fraction(319086769, 355658688),
    )
    assert lo14 - THETA == Fraction(29017, 1425600)
    assert lo14 > THETA
    assert (lo34, hi34) == (
        Fraction(1492039, 1425600),
        Fraction(538657681, 457275456),
    )
    assert lo34 > THETA
    assert classify(u0, r_lo) == "div"
    assert classify(u0, r_hi) == "div"


def test_u0_t32_hull_occupancy_split_is_not_a_count_theorem() -> None:
    # Occupancy split on U0, not R0-wide, not a count theorem.
    r_lo = growth_at_label(Fraction(1, 4), KAPPA)
    r_hi = growth_at_label(Fraction(3, 4), KAPPA)
    u0 = closed_box(BC_M, BC_B)

    mixed_lo, divs_lo, hull_lo = trace_full(u0, r_lo)
    mixed_hi, divs_hi, hull_hi = trace_full(u0, r_hi)

    assert mixed_lo is False
    assert mixed_hi is False
    assert hull_lo.mhi == U0_Q14_MHI
    assert hull_lo.mhi < R0_M[0]
    assert disjoint_r0(hull_lo)
    assert not inside_r0(hull_lo)

    assert hull_hi.mlo >= R0_M[0]
    assert hull_hi.mhi <= R0_M[1]
    assert hull_hi.blo >= R0_B[0]
    assert hull_hi.bhi <= R0_B[1]
    assert inside_r0(hull_hi)
    assert not disjoint_r0(hull_hi)

    # Count split is false: both labels divide on every one of the 32 steps.
    assert divs_lo == TICKS
    assert divs_hi == TICKS
    count_split = divs_lo != divs_hi
    assert count_split is False
    occupancy_split = disjoint_r0(hull_lo) and inside_r0(hull_hi)
    assert occupancy_split is True
