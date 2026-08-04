from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.multimode_global_throat import (
    _target_profiles,
    global_anisotropic_target_audit,
    multimode_target_fit_audit,
)


def test_global_target_exactly_matches_casimir_at_throat() -> None:
    audit = global_anisotropic_target_audit()

    assert math.isclose(audit.throat_density, -1.0 / 3.0)
    assert math.isclose(audit.throat_radial_pressure, -1.0)
    assert math.isclose(audit.throat_tangential_pressure, 1.0 / 3.0)
    assert audit.throat_matches_ideal_casimir
    assert audit.flare_out_satisfied


def test_variable_anisotropy_closes_global_geometry_control_gate() -> None:
    audit = global_anisotropic_target_audit()

    assert audit.horizon_free
    assert audit.analytic_lapse_squared_lower_bound == 1.0
    assert audit.shape_gap_at_throat == 0.0
    assert audit.minimum_shape_gap > 0.0
    assert audit.shape_gap_derivative_lower_bound == 1.0
    assert audit.shape_gap_positive_on_entire_exterior_proved
    assert audit.maximum_conservation_residual < 1.0e-10
    assert audit.conservation_identity_exact
    assert audit.throat_proper_distance_integrable_proved
    assert audit.shape_over_radius_at_cutoff < 1.0e-3
    assert abs(audit.redshift_at_cutoff) < 1.0e-10
    assert audit.finite_adm_mass
    assert math.isclose(audit.adm_mass_in_throat_radii, 1.0 / 3.0)
    assert audit.asymptotically_flat
    assert audit.asymptotic_flatness_proved_without_cutoff
    assert audit.two_sided_geometric_extension_available
    assert audit.global_geometry_control_pass


def test_asymptotic_flatness_is_exact_and_does_not_depend_on_cutoff() -> None:
    near = global_anisotropic_target_audit(radial_cutoff=2.0)
    far = global_anisotropic_target_audit(radial_cutoff=1.0e6)

    assert near.asymptotically_flat
    assert far.asymptotically_flat
    assert near.global_geometry_control_pass
    assert far.global_geometry_control_pass
    assert near.shape_over_radius_at_cutoff > far.shape_over_radius_at_cutoff


def test_original_exponential_redshift_has_log_divergent_volume_nec_tail() -> None:
    lower = global_anisotropic_target_audit(radial_cutoff=100.0)
    upper = global_anisotropic_target_audit(radial_cutoff=10_000.0)

    expected_change = -(2.0 / 3.0) * math.log(100.0)
    actual_change = (
        upper.sampled_coordinate_volume_nec_integral
        - lower.sampled_coordinate_volume_nec_integral
    )
    assert math.isclose(
        upper.asymptotic_radial_nec_x_cubed_coefficient,
        -2.0 / 3.0,
    )
    assert math.isclose(actual_change, expected_change, rel_tol=5.0e-5)
    assert upper.radial_nec_strictly_negative_everywhere_proved
    assert upper.complete_radial_affine_anec_finite_proved
    assert upper.complete_radial_affine_anec_negative_proved
    assert math.isclose(
        upper.sampled_dimensionless_two_sided_radial_anec,
        -2.4975554173,
        rel_tol=1.0e-7,
    )
    assert abs(upper.radial_anec_quadrature_refinement_delta) < 1.0e-7
    assert not upper.coordinate_volume_nec_burden_finite
    assert not upper.proper_volume_nec_burden_finite
    assert not upper.stress_l1_localized
    assert not upper.source_tail_control_pass


def test_schwarzschild_matched_tail_keeps_throat_and_closes_volume_burden() -> None:
    lower = global_anisotropic_target_audit(
        radial_cutoff=100.0,
        redshift_profile="schwarzschild_matched",
    )
    upper = global_anisotropic_target_audit(
        radial_cutoff=10_000.0,
        redshift_profile="schwarzschild_matched",
    )

    assert upper.throat_matches_ideal_casimir
    assert upper.analytic_lapse_squared_lower_bound == 1.0 / 3.0
    assert upper.horizon_free
    assert upper.asymptotic_radial_nec_x_cubed_coefficient == 0.0
    assert upper.radial_nec_strictly_negative_everywhere_proved
    assert upper.coordinate_volume_nec_burden_finite
    assert upper.proper_volume_nec_burden_finite
    assert upper.stress_l1_localized
    assert upper.source_tail_control_pass
    assert abs(
        upper.sampled_coordinate_volume_nec_integral
        - lower.sampled_coordinate_volume_nec_integral
    ) < 2.0e-5
    assert upper.sampled_numerical_identity_pass


def test_matched_profile_radial_nec_matches_closed_negative_formula() -> None:
    x = np.array([1.0, 1.01, 1.5, 3.0, 10.0, 100.0])
    _, _, density, radial, _ = _target_profiles(x, "schwarzschild_matched")
    exponential = np.exp(-(x - 1.0))
    closed = -(exponential / x**2) * (
        1.0 / 3.0
        + 1.0 / (3.0 * x - 2.0)
        + 3.0 * x
        - 2.0
        - exponential
    )

    assert np.all(closed < 0.0)
    assert np.allclose(density + radial, closed, rtol=1.0e-13, atol=1.0e-15)


def test_geometry_target_does_not_claim_the_missing_physics() -> None:
    audit = global_anisotropic_target_audit()

    assert not audit.fixed_casimir_eos_preserved_globally
    assert not audit.ce_multimode_stress_derived
    assert not audit.independent_matter_eom_derived
    assert not audit.perturbative_stability_derived


def test_shared_spectral_modes_converge_to_the_target_tensor() -> None:
    audit = multimode_target_fit_audit()
    errors = [level.maximum_normalized_error for level in audit.levels]

    assert audit.error_decreases
    assert errors[-1] < 1.0e-6
    assert audit.finite_mode_target_approximation_pass
    assert not audit.basis_is_physical_resonator_spectrum
    assert not audit.carrier_envelope_bridge_derived
    assert not audit.quantized_negative_stress_derived


@pytest.mark.parametrize("cutoff", [1.0, 0.0, float("inf")])
def test_invalid_radial_cutoff_is_rejected(cutoff: float) -> None:
    with pytest.raises(ValueError):
        global_anisotropic_target_audit(radial_cutoff=cutoff)


@pytest.mark.parametrize("sample_count", [32.0, True, False])
def test_noninteger_or_boolean_global_sample_count_is_rejected(
    sample_count: object,
) -> None:
    with pytest.raises(ValueError, match="integer"):
        global_anisotropic_target_audit(sample_count=sample_count)  # type: ignore[arg-type]


def test_unknown_redshift_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="redshift_profile"):
        global_anisotropic_target_audit(redshift_profile="unknown")  # type: ignore[arg-type]


def test_invalid_mode_sequence_is_rejected() -> None:
    with pytest.raises(ValueError):
        multimode_target_fit_audit(mode_counts=(8, 4))
