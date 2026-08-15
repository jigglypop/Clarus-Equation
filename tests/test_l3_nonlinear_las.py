"""Locks for the L3 boxed map at kappa=1/4.

Machine pass is not a theorem. N-E2 stays 미완성. This file does not
claim 닫힘, 유도됨, or AGI. The 5x5 R0 occupancy check is a construction
witness, not an open-set theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from reality_stone.clarus.universe_life_kernel import (
    EXTINCTION_AREA_FLOOR,
    KAPPA_OPEN_RIGHT,
    NOMINAL_GROWTH,
    SOURCE_EXTINCTION_AREA,
    HybridState,
    UniverseKernel,
    growth_at_label,
    hosted_states,
    source_hybrid_step,
    source_one_step_extinction_area,
)


KAPPA = Fraction(1, 4)
THETA = Fraction(3, 4)
LEAK = Fraction(5, 2)
RHO = Fraction(1, 5)
DELTA = Fraction(1, 10)
DISC = 18601
Z_MINUS = HybridState.from_values(Fraction(7, 18), Fraction(7, 16), Fraction(1, 4))
R0_M = (Fraction(2, 5), Fraction(3, 5))
R0_B = (Fraction(4, 9), Fraction(6, 11))


@dataclass(frozen=True)
class Qs:
    """Q(sqrt(18601)) element a + b*sqrt(18601). Copied from the math-lane helper."""

    a: Fraction
    b: Fraction

    def __add__(self, other: object) -> Qs:
        o = _qs(other)
        return Qs(self.a + o.a, self.b + o.b)

    def __radd__(self, other: object) -> Qs:
        return self + other

    def __sub__(self, other: object) -> Qs:
        o = _qs(other)
        return Qs(self.a - o.a, self.b - o.b)

    def __rsub__(self, other: object) -> Qs:
        return _qs(other) - self

    def __neg__(self) -> Qs:
        return Qs(-self.a, -self.b)

    def __mul__(self, other: object) -> Qs:
        o = _qs(other)
        return Qs(self.a * o.a + self.b * o.b * DISC, self.a * o.b + self.b * o.a)

    def __rmul__(self, other: object) -> Qs:
        return self * other

    def inv(self) -> Qs:
        n = self.a * self.a - DISC * self.b * self.b
        return Qs(self.a / n, -self.b / n)

    def __truediv__(self, other: object) -> Qs:
        return self * _qs(other).inv()

    def __eq__(self, other: object) -> bool:
        o = _qs(other)
        return self.a == o.a and self.b == o.b

    def sign(self) -> int:
        if self.b == 0:
            return (self.a > 0) - (self.a < 0)
        if self.b > 0:
            if self.a >= 0:
                return 1
            return 1 if Fraction(DISC) > (self.a / self.b) ** 2 else -1
        if self.a <= 0:
            return -1
        return 1 if self.a * self.a > Fraction(DISC) * self.b * self.b else -1

    def __gt__(self, other: object) -> bool:
        return (self - _qs(other)).sign() > 0

    def __ge__(self, other: object) -> bool:
        return (self - _qs(other)).sign() >= 0

    def as_float(self) -> float:
        return float(self.a) + float(self.b) * math.sqrt(DISC)


def _qs(value: object) -> Qs:
    if isinstance(value, Qs):
        return value
    return Qs(Fraction(value), Fraction(0))


@dataclass(frozen=True)
class Ival:
    lo: Fraction
    hi: Fraction

    def __add__(self, other: object) -> Ival:
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        return Ival(self.lo + o.lo, self.hi + o.hi)

    def __sub__(self, other: object) -> Ival:
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        return Ival(self.lo - o.hi, self.hi - o.lo)

    def __mul__(self, other: object) -> Ival:
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        xs = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Ival(min(xs), max(xs))

    def __truediv__(self, other: object) -> Ival:
        o = other if isinstance(other, Ival) else Ival(Fraction(other), Fraction(other))
        xs = (self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi)
        return Ival(min(xs), max(xs))

    def absbound(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))


def _f_prime(label: Fraction) -> Fraction:
    return -3 * label * label + 3 * label + Fraction(5, 16)


def _iv(value: object) -> Ival:
    if isinstance(value, Ival):
        return value
    v = Fraction(value)
    return Ival(v, v)


def _z_plus() -> tuple[Qs, Qs, Fraction]:
    return (
        Qs(Fraction(49, 324), Fraction(1, 324)),
        Qs(Fraction(-51, 160), Fraction(1, 160)),
        Fraction(3, 4),
    )


def _q_next(label: Fraction) -> Fraction:
    copied = (
        label
        + Fraction(1, 2) * label * (1 - label) * (2 * label - 1)
        + Fraction(3, 32) * (1 - 2 * label)
    )
    return Fraction(1, 2) + (copied - Fraction(1, 2))


def _step_qs(mass: Qs, boundary: Qs, label: Fraction) -> tuple[Qs, Qs, Fraction]:
    growth = growth_at_label(label, KAPPA)
    raw = mass * (1 + growth * (1 - mass) - LEAK * (1 - boundary))
    predivision = raw if raw > 0 else _qs(0)
    next_mass = predivision / 2 if predivision >= THETA else predivision
    next_boundary = (1 - DELTA) * boundary + RHO * mass * (1 - boundary)
    return next_mass, next_boundary, _q_next(label)


def _preregistered_r0_grid() -> list[tuple[Fraction, Fraction]]:
    nodes_m = [R0_M[0] + Fraction(i, 4) * (R0_M[1] - R0_M[0]) for i in range(5)]
    nodes_b = [R0_B[0] + Fraction(j, 4) * (R0_B[1] - R0_B[0]) for j in range(5)]
    return [(mass, boundary) for mass in nodes_m for boundary in nodes_b]


def _step_float(mass: float, boundary: float, label: float) -> tuple[float, float, float, float]:
    growth = float(NOMINAL_GROWTH) * (1.0 + float(KAPPA) * (2.0 * label - 1.0))
    raw = mass * (1.0 + growth * (1.0 - mass) - float(LEAK) * (1.0 - boundary))
    predivision = raw if raw > 0.0 else 0.0
    next_mass = predivision / 2.0 if predivision >= float(THETA) else predivision
    next_boundary = (1.0 - float(DELTA)) * boundary + float(RHO) * mass * (1.0 - boundary)
    next_label = (
        label
        + 0.5 * label * (1.0 - label) * (2.0 * label - 1.0)
        + float(Fraction(3, 32)) * (1.0 - 2.0 * label)
    )
    return next_mass, next_boundary, next_label, predivision


def _iterate_float(
    mass: float, boundary: float, label: float, ticks: int
) -> tuple[float, float, float, int]:
    divisions = 0
    for _ in range(ticks):
        mass, boundary, label, predivision = _step_float(mass, boundary, label)
        if predivision >= float(THETA):
            divisions += 1
    return mass, boundary, label, divisions


def _in_r0_float(mass: float, boundary: float) -> bool:
    return 0.4 <= mass <= 0.6 and float(R0_B[0]) <= boundary <= float(R0_B[1])


def test_z_minus_is_fixed_point_of_f_quarter() -> None:
    assert Fraction(0) < KAPPA < KAPPA_OPEN_RIGHT
    assert growth_at_label(Fraction(1, 4), KAPPA) == Fraction(63, 16)
    assert source_hybrid_step(Z_MINUS, kappa=KAPPA) == Z_MINUS

    growth = growth_at_label(Z_MINUS.label, KAPPA)
    predivision = Z_MINUS.mass * (
        1 + growth * (1 - Z_MINUS.mass) - LEAK * (1 - Z_MINUS.boundary)
    )
    assert predivision == Fraction(7, 9)
    assert predivision > THETA
    assert Z_MINUS.boundary == (2 * Z_MINUS.mass) / (1 + 2 * Z_MINUS.mass)

    kernel = UniverseKernel(kappa=KAPPA)
    hosted = kernel.host(Z_MINUS, flux=1)
    assert hosted_states(kernel.step(hosted))[0] == Z_MINUS
    assert source_hybrid_step(Z_MINUS) != Z_MINUS


def test_z_plus_exact_coordinates_are_fixed_in_quadratic_field() -> None:
    mass, boundary, label = _z_plus()
    growth = growth_at_label(label, KAPPA)
    assert growth == Fraction(81, 16)
    assert label == Fraction(3, 4)

    quad_a = 2 * growth
    quad_b = 2 - growth
    quad_c = Fraction(7, 2) - growth
    disc = quad_b * quad_b - 4 * quad_a * quad_c
    assert disc == Fraction(18601, 256)
    assert (2 * mass) / (1 + 2 * mass) == boundary

    next_mass, next_boundary, next_label = _step_qs(mass, boundary, label)
    assert next_mass == mass
    assert next_boundary == boundary
    assert next_label == label

    predivision = mass * (1 + growth * (1 - mass) - LEAK * (1 - boundary))
    assert predivision == 2 * mass
    assert predivision > THETA
    assert 0 < mass.as_float() < 1
    assert 0 < boundary.as_float() < 1
    assert abs(mass.as_float() - 0.572177) < 1e-5
    assert abs(boundary.as_float() - 0.533659) < 1e-5


def test_n_e3_half_label_extinction_area_is_one_tenth() -> None:
    assert growth_at_label(Fraction(1, 2), KAPPA) == NOMINAL_GROWTH
    assert growth_at_label(Fraction(1, 2), KAPPA) == Fraction(9, 2)
    assert source_one_step_extinction_area() == SOURCE_EXTINCTION_AREA
    assert SOURCE_EXTINCTION_AREA == Fraction(1, 10)
    assert SOURCE_EXTINCTION_AREA >= EXTINCTION_AREA_FLOOR

    ceiling = Fraction(3, 5)
    integral = ceiling / 3 - 5 * ceiling**2 / 18
    assert integral == Fraction(1, 10)
    assert integral == Fraction(1, 5) - Fraction(1, 10)

    collapsed = source_hybrid_step(
        HybridState.from_values(1, 0, Fraction(1, 2)),
        kappa=KAPPA,
    )
    assert collapsed.mass == 0


def test_z_minus_contracting_box_row_sums() -> None:
    """Finite certified box for Z_-. Not a universal LAS claim on I_r."""

    nu = Fraction(1, 200)
    weight = Fraction(1)
    label_weight = Fraction(2)
    mass0, boundary0, label0 = Fraction(7, 18), Fraction(7, 16), Fraction(1, 4)
    mass_iv = Ival(mass0 - nu, mass0 + nu)
    boundary_iv = Ival(boundary0 - nu, boundary0 + nu)
    label_iv = Ival(label0 - nu, label0 + nu)
    growth_iv = _iv(NOMINAL_GROWTH) * (
        _iv(1) + _iv(KAPPA) * (_iv(2) * label_iv - 1)
    )
    gain_iv = _iv(1) + growth_iv * (_iv(1) - mass_iv) - _iv(LEAK) * (_iv(1) - boundary_iv)
    # Interval Jacobian on the splitting branch, matching the math-lane box_rows.
    j11 = gain_iv / 2 - growth_iv * mass_iv / 2
    j12 = _iv(LEAK) * mass_iv / 2
    j13 = (mass_iv / 2) * (_iv(1) - mass_iv) * Fraction(9, 4)
    j21 = _iv(RHO) * (_iv(1) - boundary_iv)
    j22 = _iv(1 - DELTA) - _iv(RHO) * mass_iv
    fp_vals = [_f_prime(label_iv.lo), _f_prime(label_iv.hi)]
    if label_iv.lo <= Fraction(1, 2) <= label_iv.hi:
        fp_vals.append(_f_prime(Fraction(1, 2)))
    j33 = Ival(min(fp_vals), max(fp_vals))
    row1 = j11.absbound() + j12.absbound() / weight + j13.absbound() / label_weight
    row2 = weight * j21.absbound() + j22.absbound()
    row3 = j33.absbound()
    lip = max(row1, row2, row3)
    mt_lo = (mass_iv * gain_iv).lo
    assert mt_lo == Fraction(216807469, 288000000)
    assert mt_lo - THETA == Fraction(807469, 288000000)
    assert lip == Fraction(16861, 18000)
    assert lip < 1
    assert mt_lo > THETA


def test_n_e2_preregistered_grid_is_a_construction_check() -> None:
    # Construction lock of the preregistered 5x5 product. Occupancy numbers
    # below are a witness sample, not an open-set theorem. N-E2 stays 미완성.
    grid = _preregistered_r0_grid()
    assert len(grid) == 25
    assert (R0_M[0], R0_B[0]) in grid
    assert (R0_M[1], R0_B[1]) in grid
    assert (Fraction(1, 2), Fraction(49, 99)) in grid
    for mass, boundary in grid:
        assert R0_M[0] <= mass <= R0_M[1]
        assert R0_B[0] <= boundary <= R0_B[1]

    low_in = high_in = occupancy_split = 0
    for mass, boundary in grid:
        low_m, low_b, _low_q, _low_div = _iterate_float(float(mass), float(boundary), 0.25, 32)
        high_m, high_b, _high_q, _high_div = _iterate_float(
            float(mass), float(boundary), 0.75, 32
        )
        low_occ = _in_r0_float(low_m, low_b)
        high_occ = _in_r0_float(high_m, high_b)
        low_in += int(low_occ)
        high_in += int(high_occ)
        occupancy_split += int(low_occ != high_occ)
    assert low_in == 0
    assert high_in == 25
    assert occupancy_split == 25
