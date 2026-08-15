"""L4-H1 router comparison algebra. No production import.

Gated growth (L4.1):
    mtilde = [m (1 + u r(q) (1-m) - lam (1-b))]_+
u=1 recovers F_{1/4}. u=0 drops the growth term.

T=32 images use the predecessor outward fixed-denominator hull, not
unbounded Fraction iteration. This file does not assign theorem status.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


R0 = Fraction(9, 2)
LAM = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
THETA = Fraction(3, 4)
KAPPA = Fraction(1, 4)
SCALE = 10**18

R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))
BC_M = (Fraction(13, 30), Fraction(17, 30))
BC_B = (Fraction(137, 297), Fraction(157, 297))
CENTER = (Fraction(1, 2), Fraction(49, 99))
Q34 = Fraction(3, 4)
Q14 = Fraction(1, 4)


def growth(q: Fraction) -> Fraction:
    return R0 * (1 + KAPPA * (2 * q - 1))


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

    def outward(self) -> "Box":
        return Box(
            outward_lo(self.mlo),
            outward_hi(self.mhi),
            outward_lo(self.blo),
            outward_hi(self.bhi),
        )


def raw_range_u(box: Box, u: Fraction, r: Fraction) -> tuple[Fraction, Fraction]:
    alpha = 1 + u * r - LAM
    vals: list[Fraction] = []
    for b in (box.blo, box.bhi):
        a = alpha + LAM * b
        for m in (box.mlo, box.mhi):
            vals.append(a * m - u * r * m * m)
        if u * r != 0:
            mcrit = a / (2 * u * r)
            if box.mlo <= mcrit <= box.mhi:
                vals.append(a * mcrit - u * r * mcrit * mcrit)
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


def classify(box: Box, u: Fraction, r: Fraction) -> str:
    if box.mhi <= 0:
        return "nodiv"
    mt_lo, mt_hi = raw_range_u(box, u, r)
    if mt_lo >= THETA:
        return "div"
    if mt_hi < THETA:
        return "nodiv"
    return "mixed"


def image_single_branch(box: Box, u: Fraction, r: Fraction, branch: str) -> Box:
    mt_lo, mt_hi = raw_range_u(box, u, r)
    if branch == "div":
        nlo, nhi = mt_lo / 2, mt_hi / 2
    elif branch == "nodiv":
        nlo, nhi = max(Fraction(0), mt_lo), max(Fraction(0), mt_hi)
    else:
        raise ValueError("mixed forbidden")
    blo2, bhi2 = bnext_range(box)
    return Box(nlo, nhi, blo2, bhi2).outward()


def iterate_hull(box: Box, u: Fraction, r: Fraction, ticks: int) -> tuple[Box, list[str]]:
    branches: list[str] = []
    current = box
    for _ in range(ticks):
        br = classify(current, u, r)
        branches.append(br)
        if br == "mixed":
            return current, branches
        current = image_single_branch(current, u, r, br)
    return current, branches


def disjoint_r0(box: Box) -> bool:
    return (
        box.mhi < R0_M[0]
        or box.mlo > R0_M[1]
        or box.bhi < R0_B[0]
        or box.blo > R0_B[1]
    )


def inside_r0(box: Box) -> bool:
    return (
        R0_M[0] <= box.mlo
        and box.mhi <= R0_M[1]
        and R0_B[0] <= box.blo
        and box.bhi <= R0_B[1]
    )


def occupancy_of(box: Box) -> str:
    if inside_r0(box):
        return "in"
    if disjoint_r0(box):
        return "out"
    return "straddle"


def row_normalize(row: tuple[int, int]) -> tuple[Fraction, Fraction]:
    total = row[0] + row[1]
    return (Fraction(row[0], total), Fraction(row[1], total))


def drive(A: tuple[tuple[int, int], tuple[int, int]], e: tuple[Fraction, Fraction]):
    rows = [row_normalize(r) for r in A]
    return (
        rows[0][0] * e[0] + rows[0][1] * e[1],
        rows[1][0] * e[0] + rows[1][1] * e[1],
    )


def symmetric_support(A: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    return A[0][0] == A[1][1] and A[0][1] == A[1][0]


BINARIES = {
    "I": ((1, 0), (0, 1)),
    "P": ((0, 1), (1, 0)),
    "A1": ((1, 1), (1, 1)),
    "A_lt": ((1, 0), (1, 1)),
    "A_ut": ((1, 1), (0, 1)),
    "A_ul": ((1, 1), (1, 0)),
    "A_lr": ((0, 1), (1, 1)),
    "B_L": ((1, 0), (1, 0)),
    "B_R": ((0, 1), (0, 1)),
}


def main() -> None:
    lines: list[str] = []
    r14 = growth(Q14)
    r34 = growth(Q34)
    lines.append(f"r(1/4)={r14}")
    lines.append(f"r(3/4)={r34}")

    leak_hi_R0 = 1 - LAM * (1 - R0_B[1])
    leak_hi_U0 = 1 - LAM * (1 - BC_B[1])
    lines.append("")
    lines.append("u=0 coefficient 1-lam(1-b):")
    lines.append(f"  at b=6/11 (R0 hi): {leak_hi_R0}")
    lines.append(f"  at b=157/297 (U0 hi): {leak_hi_U0}")
    lines.append(f"  1-1/lam = {1 - 1 / LAM}")
    lines.append(f"  6/11 < 3/5: {R0_B[1] < Fraction(3, 5)}")

    bc = Box(BC_M[0], BC_M[1], BC_B[0], BC_B[1])
    lines.append("")
    lines.append("first-step mtilde on closed B_c, q=3/4:")
    for u in (Fraction(0), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1)):
        lo, hi = raw_range_u(bc, u, r34)
        br = classify(bc, u, r34)
        lines.append(
            f"  u={u}: [{lo}, {hi}] floats=[{float(lo):.6f},{float(hi):.6f}] "
            f"lo-theta={lo - THETA} branch={br}"
        )

    lines.append("")
    lines.append("T=32 outward hull on closed B_c, q=3/4 frozen r:")
    for u in (Fraction(0), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1)):
        hull, branches = iterate_hull(bc, u, r34, 32)
        mixed_at = branches.index("mixed") + 1 if "mixed" in branches else None
        if mixed_at is None:
            lines.append(
                f"  u={u}: occ={occupancy_of(hull)} "
                f"m=[{float(hull.mlo):.6f},{float(hull.mhi):.6f}] "
                f"b=[{float(hull.blo):.6f},{float(hull.bhi):.6f}] "
                f"branches={branches[0]}..{branches[-1]} unique={set(branches)}"
            )
        else:
            lines.append(
                f"  u={u}: MIXED at t={mixed_at} "
                f"m=[{float(hull.mlo):.6f},{float(hull.mhi):.6f}] "
                f"last_pure={set(branches[: mixed_at - 1]) if mixed_at > 1 else 'none'}"
            )

    e1 = (Fraction(1), Fraction(0))
    e2 = (Fraction(0), Fraction(1))
    lines.append("")
    lines.append("binary census:")
    for name, A in BINARIES.items():
        d1 = drive(A, e1)
        d2 = drive(A, e2)
        lines.append(
            f"  {name}: supp={A} sym={symmetric_support(A)} "
            f"u(e1)={d1} u(e2)={d2} drive_sep={d1 != d2}"
        )

    text = "\n".join(lines) + "\n"
    out = Path(__file__).with_name("route_router_algebra.txt")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
