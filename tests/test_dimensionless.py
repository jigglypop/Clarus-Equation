from __future__ import annotations

import math
from fractions import Fraction

import pytest

from clarus.dimensionless import (
    CURVATURE,
    DIMENSIONLESS,
    LENGTH,
    Quantity,
    buckingham_pi_groups,
    dim,
    evaluate_group,
    exp_argument,
    group_dimension,
    nondimensionalize,
    require_dimensionless,
)


def test_curvature_must_be_scaled_before_exponential() -> None:
    ricci = Quantity("R", 2.5, CURVATURE)
    length_c = Quantity("L_c", 3.0, LENGTH)

    with pytest.raises(ValueError, match="exponential"):
        exp_argument(ricci)

    r_tilde = nondimensionalize(ricci, [length_c])

    assert r_tilde.dims == DIMENSIONLESS
    assert math.isclose(r_tilde.value, 22.5)
    assert math.isclose(exp_argument(r_tilde), 22.5)


def test_mass_scale_lift_closes_as_ratio() -> None:
    m_phi = Quantity("m_phi", 29.65, dim(1, 0, 0, 0))
    m_p = Quantity("m_p", 938.2720813, dim(1, 0, 0, 0))

    ratio = nondimensionalize(m_phi, [m_p])

    assert ratio.dims == DIMENSIONLESS
    assert math.isclose(ratio.value, 29.65 / 938.2720813)


def test_buckingham_pi_finds_reynolds_number_shape() -> None:
    rho = Quantity("rho", 1.2, dim(1, -3, 0, 0))
    velocity = Quantity("v", 3.0, dim(0, 1, -1, 0))
    length = Quantity("L", 2.0, dim(0, 1, 0, 0))
    viscosity = Quantity("mu", 1.8e-5, dim(1, -1, -1, 0))

    groups = buckingham_pi_groups([rho, velocity, length, viscosity])

    assert groups == [
        {"rho": Fraction(-1, 1), "v": Fraction(-1, 1), "L": Fraction(-1, 1), "mu": Fraction(1, 1)}
    ]
    assert group_dimension([rho, velocity, length, viscosity], groups[0]) == DIMENSIONLESS
    assert math.isclose(evaluate_group([rho, velocity, length, viscosity], groups[0]), 1 / (1.2 * 3.0 * 2.0 / 1.8e-5))


def test_dimensionless_guard_accepts_ce_core_ratio() -> None:
    epsilon2 = Quantity("epsilon^2", 0.04865)

    assert require_dimensionless(epsilon2).value == 0.04865
