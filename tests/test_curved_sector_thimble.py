import math

import numpy as np
import pytest

from examples.physics.curved_sector_thimble import (
    dimensionless_curved_regge_phase,
    projected_curved_thimble_normal_form,
)


def _closed_normal_form(scale: float = 20.0):
    return projected_curved_thimble_normal_form(
        ((2.0, 0.0), (0.0, 8.0)),
        dimensionless_target_phase=0.7,
        large_dimensionless_parameter=scale,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )


def test_curved_regge_phase_combines_only_dimensionless_terms() -> None:
    audit = dimensionless_curved_regge_phase(
        (0.5, 1.25),
        (0.2, -0.4),
        cosmological_constant_times_reference_length_squared=3.0,
        four_volume_over_reference_length_fourth=0.1,
        inverse_dimensionless_gravitational_coupling=2.0,
    )

    assert audit.area_angle_sum == pytest.approx(-0.4)
    assert audit.cosmological_volume_term == pytest.approx(0.3)
    assert audit.dimensionless_curved_regge_phase == pytest.approx(-1.4)
    assert audit.all_input_normalizations_declared_dimensionless
    assert audit.status == "DIMENSIONLESS_CURVED_REGGE_PHASE_CLOSED"


@pytest.mark.parametrize(
    "areas,angles",
    (
        ((), ()),
        ((1.0,), (1.0, 2.0)),
        ((0.0,), (1.0,)),
        ((math.inf,), (1.0,)),
    ),
)
def test_curved_regge_phase_rejects_invalid_normalized_inputs(
    areas: tuple[float, ...],
    angles: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        dimensionless_curved_regge_phase(
            areas,
            angles,
            cosmological_constant_times_reference_length_squared=3.0,
            four_volume_over_reference_length_fourth=0.1,
            inverse_dimensionless_gravitational_coupling=2.0,
        )


def test_positive_transverse_hessian_has_exact_gaussian_thimble_amplitude() -> None:
    audit = _closed_normal_form(scale=20.0)
    expected_prefactor = (2.0 * math.pi / 20.0) / 4.0
    expected = expected_prefactor * np.exp(1j * 20.0 * 0.7)

    assert audit.variable_count == 2
    assert audit.transverse_hessian_eigenvalues == pytest.approx((2.0, 8.0))
    assert audit.transverse_hessian_determinant == pytest.approx(16.0)
    assert audit.exact_quadratic_prefactor_magnitude == pytest.approx(
        expected_prefactor
    )
    assert audit.exact_quadratic_normal_form_amplitude == pytest.approx(expected)
    assert audit.large_parameter_power == pytest.approx(-1.0)
    assert audit.numerically_well_conditioned_positive_definite
    assert audit.quadratic_gaussian_local_template_exact
    assert audit.conditional_local_single_branch_stationary_phase_template
    assert not audit.regulated_curved_block_single_branch_asymptotic_proved


def test_gaussian_prefactor_scales_with_declared_stationary_power() -> None:
    first = _closed_normal_form(scale=20.0)
    fourth = _closed_normal_form(scale=80.0)

    assert fourth.exact_quadratic_prefactor_magnitude == pytest.approx(
        first.exact_quadratic_prefactor_magnitude / 4.0
    )


@pytest.mark.parametrize(
    "missing",
    (
        "finite_dimensional_regulator_fixed",
        "gauge_zero_modes_removed",
        "admissible_middle_dimensional_cycle_fixed",
        "cutoff_equal_to_one_near_target",
        "other_critical_points_excluded_from_support",
        "uniform_nonstationary_remainder_bound_supplied",
    ),
)
def test_each_missing_global_hypothesis_blocks_regulated_branch_promotion(
    missing: str,
) -> None:
    hypotheses = {
        "finite_dimensional_regulator_fixed": True,
        "gauge_zero_modes_removed": True,
        "admissible_middle_dimensional_cycle_fixed": True,
        "cutoff_equal_to_one_near_target": True,
        "other_critical_points_excluded_from_support": True,
        "uniform_nonstationary_remainder_bound_supplied": True,
    }
    hypotheses[missing] = False
    audit = projected_curved_thimble_normal_form(
        ((1.0, 0.0), (0.0, 1.0)),
        dimensionless_target_phase=0.4,
        large_dimensionless_parameter=10.0,
        **hypotheses,
    )

    assert audit.quadratic_gaussian_local_template_exact
    assert not audit.conditional_local_single_branch_stationary_phase_template
    assert audit.status == "LOCAL_GAUSSIAN_TEMPLATE_HYPOTHESES_INCOMPLETE"


def test_nonpositive_hessian_is_not_a_steepest_descent_normal_form() -> None:
    audit = projected_curved_thimble_normal_form(
        ((1.0, 0.0), (0.0, -1.0)),
        dimensionless_target_phase=0.4,
        large_dimensionless_parameter=10.0,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )

    assert not audit.quadratic_gaussian_local_template_exact
    assert not audit.conditional_local_single_branch_stationary_phase_template
    assert math.isinf(audit.exact_quadratic_prefactor_magnitude)
    assert math.isnan(audit.exact_quadratic_normal_form_amplitude.real)


def test_local_normal_form_keeps_global_curved_spin_foam_claims_false() -> None:
    audit = _closed_normal_form()

    assert not audit.global_chern_simons_functional_integral_defined
    assert not audit.regulator_removal_proved
    assert not audit.equivalent_to_engle_proper_projector
    assert not audit.multi_block_gluing_proved
    assert audit.claim_ceiling.endswith("NOT_A_REGULATED_CURVED_BLOCK")


def test_off_diagonal_positive_hessian_uses_basis_invariant_determinant() -> None:
    audit = projected_curved_thimble_normal_form(
        ((5.0, 3.0), (3.0, 5.0)),
        dimensionless_target_phase=-0.2,
        large_dimensionless_parameter=8.0,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )

    assert audit.transverse_hessian_eigenvalues == pytest.approx((2.0, 8.0))
    assert audit.transverse_hessian_determinant == pytest.approx(16.0)
    assert abs(audit.exact_quadratic_normal_form_amplitude) == pytest.approx(
        (2.0 * math.pi / 8.0) / 4.0
    )


def test_one_dimensional_template_has_half_power() -> None:
    audit = projected_curved_thimble_normal_form(
        ((4.0,),),
        dimensionless_target_phase=0.0,
        large_dimensionless_parameter=9.0,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )

    assert audit.large_parameter_power == pytest.approx(-0.5)
    assert audit.exact_quadratic_prefactor_magnitude == pytest.approx(
        math.sqrt(2.0 * math.pi / 9.0) / 2.0
    )


def test_singular_hessian_fails_the_numerical_spd_gate() -> None:
    audit = projected_curved_thimble_normal_form(
        ((1.0, 0.0), (0.0, 0.0)),
        dimensionless_target_phase=0.0,
        large_dimensionless_parameter=2.0,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )

    assert not audit.numerically_well_conditioned_positive_definite
    assert not audit.quadratic_gaussian_local_template_exact


@pytest.mark.parametrize(
    "matrix",
    (
        ((1.0, 2.0), (0.0, 1.0)),
        ((1.0, math.inf), (math.inf, 1.0)),
    ),
)
def test_invalid_hessian_matrices_are_rejected(
    matrix: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ValueError, match="transverse_hessian"):
        projected_curved_thimble_normal_form(
            matrix,
            dimensionless_target_phase=0.0,
            large_dimensionless_parameter=1.0,
            finite_dimensional_regulator_fixed=True,
            gauge_zero_modes_removed=True,
            admissible_middle_dimensional_cycle_fixed=True,
            cutoff_equal_to_one_near_target=True,
            other_critical_points_excluded_from_support=True,
            uniform_nonstationary_remainder_bound_supplied=True,
        )


def test_tiny_strictly_positive_hessian_is_below_numerical_acceptance_gate() -> None:
    audit = projected_curved_thimble_normal_form(
        ((1.0e-16,),),
        dimensionless_target_phase=0.0,
        large_dimensionless_parameter=1.0,
        finite_dimensional_regulator_fixed=True,
        gauge_zero_modes_removed=True,
        admissible_middle_dimensional_cycle_fixed=True,
        cutoff_equal_to_one_near_target=True,
        other_critical_points_excluded_from_support=True,
        uniform_nonstationary_remainder_bound_supplied=True,
    )

    assert not audit.numerically_well_conditioned_positive_definite
    assert not audit.quadratic_gaussian_local_template_exact


@pytest.mark.parametrize("scale,tolerance", ((0.0, 1.0e-12), (1.0, 0.0)))
def test_scale_and_tolerance_must_be_positive(
    scale: float, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        projected_curved_thimble_normal_form(
            ((1.0,),),
            dimensionless_target_phase=0.0,
            large_dimensionless_parameter=scale,
            finite_dimensional_regulator_fixed=True,
            gauge_zero_modes_removed=True,
            admissible_middle_dimensional_cycle_fixed=True,
            cutoff_equal_to_one_near_target=True,
            other_critical_points_excluded_from_support=True,
            uniform_nonstationary_remainder_bound_supplied=True,
            tolerance=tolerance,
        )


def test_thimble_hypothesis_flags_must_be_actual_booleans() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        projected_curved_thimble_normal_form(
            ((1.0,),),
            dimensionless_target_phase=0.0,
            large_dimensionless_parameter=1.0,
            finite_dimensional_regulator_fixed=1,  # type: ignore[arg-type]
            gauge_zero_modes_removed=True,
            admissible_middle_dimensional_cycle_fixed=True,
            cutoff_equal_to_one_near_target=True,
            other_critical_points_excluded_from_support=True,
            uniform_nonstationary_remainder_bound_supplied=True,
        )
