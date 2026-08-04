from __future__ import annotations

import math

import pytest

from reality_stone.clarus.global_throat_exact_certificate import (
    global_throat_exact_certificate,
)


def test_original_geometry_is_certified_without_a_radial_cutoff() -> None:
    audit = global_throat_exact_certificate()
    original = audit.original

    assert original.throat_matches_ideal_casimir
    assert original.shape_gap_at_throat == 0.0
    assert original.shape_gap_derivative_infimum == 1.0
    assert original.shape_gap_positive_for_every_x_above_one
    assert math.isclose(original.metric_factor_throat_slope, 4.0 / 3.0)
    assert original.two_sided_extension_available
    assert original.horizon_free_for_every_x
    assert original.asymptotically_flat_exact
    assert math.isclose(original.adm_mass_per_end_in_throat_radii, 1.0 / 3.0)
    assert original.adm_mass_per_end_positive


def test_bianchi_identity_does_not_claim_an_independent_matter_eom() -> None:
    audit = global_throat_exact_certificate()

    assert audit.original.bianchi_conservation_identity_exact
    assert not audit.original.independent_matter_eom_derived
    assert audit.localized_phi_match.bianchi_conservation_identity_exact
    assert not audit.localized_phi_match.independent_matter_eom_derived


def test_original_affine_anec_is_finite_but_volume_nec_diverges() -> None:
    original = global_throat_exact_certificate().original

    assert original.radial_nec_negative_for_every_x
    assert math.isclose(original.radial_pressure_cubic_tail_coefficient, -2.0 / 3.0)
    assert math.isclose(original.tangential_pressure_cubic_tail_coefficient, 1.0 / 3.0)
    assert original.radial_affine_killing_energy_normalization == 1.0
    assert math.isclose(
        original.radial_affine_anec_dimensionless,
        -2.497555417277317,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    )
    assert original.radial_affine_anec_finite
    assert original.radial_affine_anec_negative
    assert math.isclose(original.coordinate_volume_nec_log_coefficient, -2.0 / 3.0)
    assert original.volume_nec_diverges_logarithmically
    assert not original.coordinate_volume_nec_finite
    assert not original.proper_volume_nec_finite
    assert not original.stress_l1_localized


def test_exact_properties_are_independent_of_numerical_controls() -> None:
    coarse = global_throat_exact_certificate(
        quadrature_order=64,
        kinetic_sample_count=4_096,
        kinetic_radial_cutoff=2.0,
    )
    fine = global_throat_exact_certificate(
        quadrature_order=256,
        kinetic_sample_count=40_000,
        kinetic_radial_cutoff=100.0,
    )

    assert coarse.original.asymptotically_flat_exact
    assert fine.original.asymptotically_flat_exact
    assert (
        coarse.original.adm_mass_per_end_in_throat_radii
        == fine.original.adm_mass_per_end_in_throat_radii
    )
    assert math.isclose(
        coarse.original.radial_affine_anec_dimensionless,
        fine.original.radial_affine_anec_dimensionless,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    )


def test_localized_phi_match_preserves_throat_and_positive_adm_mass() -> None:
    localized = global_throat_exact_certificate().localized_phi_match

    assert localized.throat_matches_ideal_casimir
    assert math.isclose(localized.throat_shape_derivative, -1.0 / 3.0)
    assert math.isclose(localized.throat_redshift_derivative, -1.0 / 2.0)
    assert math.isclose(localized.throat_redshift_second_derivative, -5.0 / 2.0)
    assert localized.shape_gap_positive_for_every_x_above_one
    assert localized.lapse_squared_global_lower_bound == 1.0 / 3.0
    assert localized.horizon_free_for_every_x
    assert localized.asymptotically_schwarzschild
    assert math.isclose(localized.adm_mass_per_end_in_throat_radii, 1.0 / 3.0)
    assert localized.adm_mass_per_end_positive


def test_localized_phi_match_has_finite_negative_nec_integrals() -> None:
    localized = global_throat_exact_certificate().localized_phi_match

    assert localized.radial_nec_negative_for_every_x
    assert localized.radial_affine_killing_energy_normalization == 1.0
    assert math.isclose(
        localized.radial_affine_anec_dimensionless,
        -2.292728133816266,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    )
    assert math.isclose(
        localized.coordinate_volume_nec_dimensionless_per_end,
        -4.21893534547003,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    )
    assert math.isclose(
        localized.proper_volume_nec_dimensionless_per_end,
        -6.091787247550253,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    )
    assert localized.radial_affine_anec_finite
    assert localized.radial_affine_anec_negative
    assert localized.coordinate_volume_nec_finite
    assert localized.proper_volume_nec_finite
    assert not localized.volume_nec_diverges_logarithmically
    assert localized.stress_tail_exponentially_localized
    assert localized.stress_l1_localized


def test_local_nonminimal_sign_does_not_survive_the_global_counterexample() -> None:
    localized = global_throat_exact_certificate().localized_phi_match

    assert math.isclose(localized.local_nonminimal_kinetic_over_planck_factor, 7.0 / 12.0)
    assert localized.local_nonminimal_kinetic_positive
    assert math.isclose(localized.global_kinetic_counterexample_radius, 37.0 / 32.0)
    assert localized.global_kinetic_counterexample_value < -1.8
    assert localized.minimum_sampled_kinetic_over_planck_factor < -1.8
    assert 1.14 < localized.minimum_sampled_kinetic_radius < 1.17
    assert not localized.global_nonminimal_kinetic_positive
    assert not localized.potential_reconstructed
    assert not localized.perturbative_stability_derived


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("quadrature_order", True),
        ("quadrature_order", 31),
        ("quadrature_order", 1_025),
        ("kinetic_sample_count", 1_023),
        ("kinetic_sample_count", 20_000.0),
        ("kinetic_radial_cutoff", True),
        ("kinetic_radial_cutoff", "40"),
        ("kinetic_radial_cutoff", 1.999),
        ("kinetic_radial_cutoff", float("inf")),
    ],
)
def test_invalid_numerical_controls_are_rejected(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        global_throat_exact_certificate(**{keyword: value})  # type: ignore[arg-type]
