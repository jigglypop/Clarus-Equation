from __future__ import annotations

import math

import pytest

from examples.physics.finite_probability_deformation_readout import (
    certify_finite_probability_deformation_readout,
)


def test_default_finite_newtonian_rn_certificate() -> None:
    certificate = certify_finite_probability_deformation_readout()
    assert certificate.normalizer == pytest.approx(1.0014984243089056, abs=2.0e-14)
    assert certificate.log_normalizer == pytest.approx(0.001497302791400418, abs=2.0e-14)
    assert certificate.holdout_probability == pytest.approx(0.019046610299066694, abs=2.0e-14)
    assert certificate.normalization_residual < 2.0e-14
    assert certificate.constant_shift_invariance_residual < 2.0e-14
    assert certificate.inward_likelihood_ratio > 1.0
    assert certificate.chi_continuity_residual_at_surface == 0.0
    assert certificate.scaled_radial_laplacian_inside == pytest.approx(-0.03)
    assert certificate.scaled_radial_laplacian_outside == 0.0
    assert certificate.inside_chi_prime_over_x == pytest.approx(-0.01)
    assert certificate.outside_x_squared_chi_prime == pytest.approx(-0.01)
    assert certificate.scaled_acceleration_at_x_half == pytest.approx(-0.005)
    assert certificate.scaled_acceleration_at_holdout_x1 == pytest.approx(-0.0025)


def test_finite_scope_and_dimensions_are_explicit() -> None:
    certificate = certify_finite_probability_deformation_readout()
    assert certificate.chi_equals_minus_newtonian_potential_over_c_squared
    assert certificate.finite_sphere_regulates_normalization
    assert not certificate.point_source_global_normalization_available
    assert certificate.point_source_uniform_volume_integral_diverges
    assert certificate.dimensions_pass
    assert certificate.parameter_fit_count == 0
    assert certificate.internal_radial_holdout_only
    assert not certificate.observational_holdout_gate_closed
    assert certificate.newtonian_reparameterization_only
    assert certificate.no_probability_double_weighting


def test_large_finite_domain_uses_scaled_measure_without_overflow() -> None:
    certificate = certify_finite_probability_deformation_readout(domain_ratio=1.0e103)
    assert math.isfinite(certificate.normalizer) and certificate.normalizer > 0.0
    assert math.isfinite(certificate.log_normalizer)
    assert math.isfinite(certificate.holdout_probability) and certificate.holdout_probability > 0.0
    assert certificate.normalization_residual < 1.0e-11


def test_sharp_0d_readout_is_cptp_repeatable_with_single_no_signalling_witness() -> None:
    certificate = certify_finite_probability_deformation_readout()
    assert certificate.distinct_microstates_same_sharp_record
    assert certificate.record_probability_rho0 == certificate.record_probability_rho1 == (1.0, 0.0)
    assert certificate.record_probability_rho2 == (0.0, 1.0)
    assert certificate.kraus_completeness_residual < 1.0e-14
    assert certificate.choi_minimum_eigenvalue >= -1.0e-13
    assert certificate.channel_trace_preservation_residual < 1.0e-14
    assert certificate.channel_completely_positive
    assert certificate.channel_trace_preserving
    assert certificate.sharp_projector_repeatability_residual == 0.0
    assert certificate.immediate_sharp_repeatability
    assert certificate.classical_record_dephasing_idempotence_residual == 0.0
    assert certificate.single_witness_remote_marginal_residual < 1.0e-14


def test_false_claim_ceiling_is_not_promoted() -> None:
    certificate = certify_finite_probability_deformation_readout()
    assert not any((
        certificate.independent_chi_action_or_dynamics_derived,
        certificate.probability_current_or_attraction_mechanism_derived,
        certificate.causal_retarded_field_or_c_front_derived,
        certificate.scalar_to_gr_or_lensing_derived,
        certificate.gravity_energy_or_backreaction_derived,
        certificate.quantum_matter_dependent_chi_channel_derived,
        certificate.general_observation_repeatability_derived,
        certificate.physical_selection_derived,
        certificate.ideal_point_source_normalization_derived,
        certificate.homology_cohomology_self_duality_derived,
        certificate.actual_data_holdout_or_gates_5_to_8_closed,
        certificate.two_residuals_or_complexity_success,
    ))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"compactness": 0.0}, "compactness"),
        ({"compactness": 1.0}, "compactness"),
        ({"compactness": 1.0e300}, "compactness"),
        ({"domain_ratio": 1.0}, "domain_ratio"),
        ({"holdout_x1": 1.0}, "holdout"),
        ({"holdout_x2": 10.0}, "holdout"),
        ({"holdout_x1": 3.0, "holdout_x2": 2.0}, "holdout"),
    ],
)
def test_finite_domain_contract_fails_closed(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        certify_finite_probability_deformation_readout(**kwargs)
