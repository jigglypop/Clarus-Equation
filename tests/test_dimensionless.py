from __future__ import annotations

import math
from fractions import Fraction

import pytest

from reality_stone.clarus.dimensionless import (
    CURVATURE,
    DIMENSIONLESS,
    LENGTH,
    Quantity,
    audit_dimensionless,
    buckingham_pi_groups,
    check_dimensionless,
    dim,
    evaluate_group,
    exp_argument,
    exp_arguments,
    group_dimension,
    nondimensionalize,
    require_dimensionless,
)
from reality_stone.clarus.dimensionless_checker import (
    Dimension,
    DimensionVector,
    DimensionlessChecker,
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
    assert math.isclose(
        evaluate_group([rho, velocity, length, viscosity], groups[0]),
        1 / (1.2 * 3.0 * 2.0 / 1.8e-5),
    )


def test_dimensionless_guard_accepts_ce_core_ratio() -> None:
    epsilon2 = Quantity("epsilon^2", 0.04865)

    assert require_dimensionless(epsilon2).value == 0.04865


def test_dimensionless_gate_result_composes_value_transform() -> None:
    epsilon2 = Quantity("epsilon^2", 0.25)

    result = (
        check_dimensionless(epsilon2)
        .map(lambda q: q.value)
        .bind(lambda value: check_dimensionless(Quantity("sqrt_epsilon2", math.sqrt(value))))
    )

    assert result.passed
    assert math.isclose(result.unwrap().value, 0.5)


def test_dimensionless_audit_accumulates_all_failures() -> None:
    checks = [
        Quantity("epsilon^2", 0.04865),
        Quantity("R", 2.5, CURVATURE),
        Quantity("L", 3.0, LENGTH),
    ]

    result = audit_dimensionless(checks, context="CE selection gate")

    assert not result.passed
    assert len(result.errors) == 2
    assert "R must be dimensionless for CE selection gate" in result.errors[0]
    assert "L must be dimensionless for CE selection gate" in result.errors[1]
    with pytest.raises(ValueError, match="CE selection gate"):
        result.unwrap()


def test_exp_arguments_validates_batch_before_kernel_use() -> None:
    args = exp_arguments(
        [
            Quantity("D_eff", 0.31),
            Quantity("phi", 1.7),
        ]
    )

    assert args.passed
    assert args.unwrap() == (0.31, 1.7)


def test_checker_preserves_unnamed_inverse_time_dimension() -> None:
    inverse_time = Dimension.TIME**-1

    assert isinstance(inverse_time, DimensionVector)
    assert inverse_time.exponents == tuple(map(Fraction, (0, 0, -1, 0)))
    assert not inverse_time.is_dimensionless()


def test_checker_preserves_mass_squared_and_composes_back_to_mass() -> None:
    mass_squared = Dimension.MASS**2

    assert isinstance(mass_squared, DimensionVector)
    assert mass_squared.exponents == tuple(map(Fraction, (2, 0, 0, 0)))
    assert not mass_squared.is_dimensionless()
    assert mass_squared / Dimension.MASS == Dimension.MASS


def test_registered_rate_and_magnetic_field_have_nontrivial_dimensions() -> None:
    formulas = {formula.name: formula for formula in DimensionlessChecker().formulas}

    rate = formulas["STDP learning rate upper bound"].expected_dim
    magnetic_field = formulas["Critical magnetic field"].expected_dim

    assert rate == Dimension.TIME**-1
    assert not rate.is_dimensionless()
    assert magnetic_field == Dimension.MASS**2
    assert not magnetic_field.is_dimensionless()


def test_clarus_field_gate_and_phase_score_are_registered_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["g_CF"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["chi_CF"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["g_CF"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["chi_CF"])["status"].startswith("PASS")


def test_unified_metric_surprise_and_condition_ratio_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["chi_UM"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["kappa_UM"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["chi_UM"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["kappa_UM"])["status"].startswith("PASS")


def test_v16_metric_flow_residual_and_regret_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["r_V16"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["rho_V16"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["r_V16"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["rho_V16"])["status"].startswith("PASS")


def test_v17_conditional_information_and_lift_margin_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["I_V17"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["delta_V17"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["I_V17"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["delta_V17"])["status"].startswith("PASS")


def test_v18b_reward_decoder_and_classifier_increment_are_dimensionless() -> None:
    checker = DimensionlessChecker()
    formulas = {formula.symbol: formula for formula in checker.formulas}

    assert formulas["y_tilde_V18b"].expected_dim == Dimension.DIMENSIONLESS
    assert formulas["delta_w_V18b"].expected_dim == Dimension.DIMENSIONLESS
    assert checker.check_formula(formulas["y_tilde_V18b"])["status"].startswith("PASS")
    assert checker.check_formula(formulas["delta_w_V18b"])["status"].startswith("PASS")
