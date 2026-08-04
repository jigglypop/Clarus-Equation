from __future__ import annotations

import math

from reality_stone.clarus.nonminimal_global_reconstruction import (
    nonminimal_throat_codesign_audit,
    nonminimal_throat_reconstruction_audit,
)


def test_target_fixes_planck_factor_slope_and_second_derivative() -> None:
    audit = nonminimal_throat_reconstruction_audit()

    assert math.isclose(audit.logarithmic_planck_factor_radial_slope, 1.0 / 8.0)
    assert math.isclose(audit.proper_planck_factor_second_derivative, 1.0 / 12.0)


def test_required_scalar_kinetic_has_the_wrong_sign() -> None:
    audit = nonminimal_throat_reconstruction_audit()

    assert math.isclose(audit.required_positive_metric_scalar_kinetic, -17.0 / 12.0)
    assert audit.positive_effective_planck_mass_assumed
    assert audit.positive_field_space_metric_assumed
    assert not audit.healthy_single_scalar_possible
    assert not audit.healthy_multiscalar_modes_possible
    assert not audit.potential_reconstruction_reached
    assert audit.target_refuted_for_healthy_nonminimal_scalars


def test_second_order_codesign_can_restore_healthy_local_kinetic_sign() -> None:
    audit = nonminimal_throat_codesign_audit(
        shape_second_derivative=-5.0,
        redshift_second_derivative=0.0,
    )

    assert math.isclose(audit.required_scalar_kinetic_over_planck_factor, 0.25)
    assert audit.positive_kinetic_gate
    assert audit.exact_casimir_throat_values_retained
    assert audit.local_codesign_survives
    assert not audit.global_solution_derived
    assert not audit.perturbative_stability_derived


def test_original_exponential_second_derivatives_reproduce_refutation() -> None:
    audit = nonminimal_throat_codesign_audit(
        shape_second_derivative=1.0 / 3.0,
        redshift_second_derivative=1.0 / 2.0,
    )

    assert math.isclose(audit.required_scalar_kinetic_over_planck_factor, -17.0 / 12.0)
    assert not audit.local_codesign_survives
