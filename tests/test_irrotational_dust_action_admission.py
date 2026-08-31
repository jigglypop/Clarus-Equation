from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.irrotational_dust_action_admission import (
    admit_irrotational_dust_action,
    canonical_dust_data,
    certificate,
    clock_constraint_residual,
    clock_gradient_from_receipt,
    epsilon_reparameterization,
    lower_minkowski,
    multiplier_dust_stress_covariant,
    raise_rank_two_minkowski,
    velocity_vorticity_residual,
    vortical_monokinetic_counterexample,
)


def _canonical_arguments() -> tuple[object, tuple[float, ...], float]:
    dust = canonical_dust_data()
    scale = 2.0
    velocity_covector = lower_minkowski(dust.four_velocity)
    receipt_gradient = tuple(-scale * value for value in velocity_covector)
    return dust, receipt_gradient, scale


def test_multiplier_action_matches_existing_monokinetic_dust_exactly() -> None:
    receipt = certificate()

    assert receipt.clock_constraint_residual == pytest.approx(0.0)
    assert receipt.continuity_residual == pytest.approx(0.0)
    assert receipt.vorticity_residual == pytest.approx(0.0)
    assert receipt.geodesic_residual == pytest.approx(0.0)
    assert receipt.ward_residual == pytest.approx(0.0)
    assert receipt.stress_match_residual == pytest.approx(0.0)
    assert receipt.mass_current_match_residual == pytest.approx(0.0)
    assert receipt.action_stress_contravariant == receipt.kinetic_stress_contravariant
    assert receipt.action_energy_current == receipt.kinetic_mass_current


def test_metric_variation_has_pressureless_on_shell_form() -> None:
    receipt = certificate()
    covariant = multiplier_dust_stress_covariant(
        receipt.lambda_density, receipt.clock_gradient_covector
    )

    assert covariant == receipt.action_stress_covariant
    assert raise_rank_two_minkowski(covariant) == receipt.kinetic_stress_contravariant
    assert receipt.rest_energy_density == pytest.approx(6.0)
    assert receipt.isotropic_pressure == pytest.approx(0.0)
    assert receipt.equation_of_state == 0.0


def test_receipt_clock_requires_a_nonconstant_unit_timelike_gradient() -> None:
    gradient = clock_gradient_from_receipt(
        (2.5, -1.5, 0.0, 0.0), reference_mass_scale=2.0
    )

    assert gradient == pytest.approx((1.25, -0.75, 0.0, 0.0))
    assert clock_constraint_residual(gradient) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="constant receipt"):
        clock_gradient_from_receipt(
            (0.0, 0.0, 0.0, 0.0), reference_mass_scale=2.0
        )


def test_dimension_ledger_is_explicit_and_closes_the_action() -> None:
    receipt = certificate()

    assert receipt.receipt_mass_dimension == 0
    assert receipt.reference_scale_mass_dimension == 1
    assert receipt.clock_mass_dimension == -1
    assert receipt.multiplier_mass_dimension == 4
    assert receipt.stress_mass_dimension == 4
    assert receipt.action_density_mass_dimension == 4
    assert receipt.volume_element_mass_dimension == -4
    assert receipt.action_mass_dimension == 0
    assert receipt.dimensions_pass


def test_overall_epsilon_is_an_exact_multiplier_reparameterization() -> None:
    receipt = epsilon_reparameterization(epsilon=0.25, multiplier_density=20.0)

    assert receipt.physical_density == pytest.approx(5.0)
    assert receipt.absorbed_multiplier_density == pytest.approx(5.0)
    assert receipt.exact_stress_reparameterization
    assert not receipt.epsilon_is_independent_dust_coupling
    assert not receipt.finite_epsilon_gr_limit_derived


def test_vortical_geodesic_dust_is_a_complete_limit_counterexample() -> None:
    witness = vortical_monokinetic_counterexample(kappa=0.2, y=1.0)

    assert witness.norm_residual == pytest.approx(0.0)
    assert witness.continuity_residual == pytest.approx(0.0)
    assert witness.geodesic_residual == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert witness.vorticity_residual == pytest.approx(
        0.2 / (1.0 - 0.2**2) ** 1.5
    )
    assert witness.kinetic_dust_admissible_at_point
    assert not witness.single_clock_admissible


def test_admission_rejects_vorticity_caustics_and_multistream_data() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()
    vortical_jacobian = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )

    assert velocity_vorticity_residual(vortical_jacobian) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="irrotational"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            velocity_covector_jacobian=vortical_jacobian,
        )
    with pytest.raises(ValueError, match="caustic|shell crossing"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            lagrangian_flow_jacobian=0.0,
        )
    with pytest.raises(ValueError, match="multistream"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            single_stream=False,
        )


def test_action_and_kinetic_sources_cannot_be_summed() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()

    with pytest.raises(ValueError, match="matched, not summed"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            accounting_mode="kinetic_plus_action",
        )


def test_lambda_positivity_is_an_admission_not_an_action_theorem() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()
    negative = replace(dust, rest_energy_density=-1.0)

    with pytest.raises(ValueError, match="non-negative by admission"):
        admit_irrotational_dust_action(
            negative,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
        )
    assert certificate().lambda_nonnegative_admission


def test_timelike_dust_worldline_is_subluminal_but_not_no_signalling_proof() -> None:
    receipt = certificate()

    assert receipt.coordinate_speed == pytest.approx(0.6)
    assert receipt.coordinate_speed < 1.0
    assert receipt.worldline_speed_below_c
    assert receipt.proper_time_interval_squared == pytest.approx(-1.0)
    assert not receipt.qft_microcausality_derived
    assert not receipt.operational_no_signalling_derived


def test_certificate_keeps_the_physical_claim_ceiling_false() -> None:
    receipt = certificate()

    assert receipt.smooth_single_stream_irrotational_precaustic
    assert receipt.matched_not_summed
    assert receipt.action_is_variational_reexpression_not_new_energy
    assert not receipt.local_receipt_field_map_derived
    assert not receipt.reference_mass_scale_derived
    assert not receipt.multiplier_initial_law_derived
    assert not receipt.supplied_metric_derived
    assert not receipt.gravitational_boson_derived
    assert not receipt.cptp_quantum_dynamics_derived
    assert not receipt.finite_coefficient_gr_phenomenology_derived
    assert not receipt.independent_holdout_prediction_derived
    assert not receipt.two_residual_classes_reduced
    assert not receipt.complexity_penalty_success


def test_invalid_kinematics_and_transport_fail_closed() -> None:
    dust, receipt_gradient, scale = _canonical_arguments()

    with pytest.raises(ValueError, match="u_mu=-partial_mu tau"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=(2.0, 0.0, 0.0, 0.0),
            reference_mass_scale=scale,
        )
    with pytest.raises(ValueError, match="continuity"):
        admit_irrotational_dust_action(
            dust,
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
            density_gradient_covector=(1.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="future unit timelike"):
        admit_irrotational_dust_action(
            replace(dust, four_velocity=(1.0, 1.0, 0.0, 0.0)),
            receipt_gradient_covector=receipt_gradient,
            reference_mass_scale=scale,
        )
    with pytest.raises(ValueError, match="positive"):
        epsilon_reparameterization(epsilon=0.0, multiplier_density=1.0)
    assert math.isfinite(certificate().lambda_density)
