from __future__ import annotations

import numpy as np
import pytest

from examples.physics.scalar_receipt_source_admission import (
    COMPONENT_ORDER,
    additive_action_countermodel,
    canonical_scalar_eom,
    canonical_scalar_potential,
    canonical_scalar_potential_derivative,
    canonical_scalar_principal_coefficient,
    canonical_scalar_stress_at_flat_point,
    canonical_scalar_ward_divergence,
    certificate,
    infinitesimal_invariance_constraint,
    lorentz_generators,
    lorentz_natural_tensor_certificate,
    minkowski_metric,
    source_accounting_receipt,
    tensor_from_components,
    vacuum_form_receipt,
)


def test_exact_rotation_and_lorentz_invariant_ranks() -> None:
    receipt = lorentz_natural_tensor_certificate()

    assert receipt.symmetric_tensor_dimension == 10
    assert receipt.rotation_constraint_shape == (30, 10)
    assert receipt.rotation_constraint_rank == 8
    assert receipt.rotation_invariant_nullity == 2
    assert receipt.full_lorentz_constraint_shape == (60, 10)
    assert receipt.full_lorentz_constraint_rank == 9
    assert receipt.full_lorentz_invariant_nullity == 1
    assert receipt.full_metric_span_unique


def test_metric_and_isotropic_rotation_basis_satisfy_exact_constraints() -> None:
    rotations = infinitesimal_invariance_constraint(("J12", "J13", "J23"))
    full = infinitesimal_invariance_constraint(
        ("J12", "J13", "J23", "K01", "K02", "K03")
    )
    metric_components = np.array((-1, 0, 0, 0, 1, 0, 0, 1, 0, 1))
    time_components = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    spatial_components = np.array((0, 0, 0, 0, 1, 0, 0, 1, 0, 1))

    assert np.array_equal(full @ metric_components, np.zeros(60, dtype=int))
    assert np.array_equal(rotations @ time_components, np.zeros(30, dtype=int))
    assert np.array_equal(
        rotations @ spatial_components,
        np.zeros(30, dtype=int),
    )
    assert not np.array_equal(full @ time_components, np.zeros(60, dtype=int))


def test_all_six_generators_preserve_the_supplied_minkowski_metric() -> None:
    metric = minkowski_metric()

    for generator in lorentz_generators().values():
        assert np.array_equal(
            generator.T @ metric + metric @ generator,
            np.zeros((4, 4), dtype=int),
        )


def test_vacuum_form_has_positive_density_and_w_minus_one_for_negative_c() -> None:
    receipt = vacuum_form_receipt(-81.0)

    assert receipt.energy_density == 81.0
    assert receipt.isotropic_pressure == -81.0
    assert receipt.equation_of_state == -1.0
    assert vacuum_form_receipt(0.0).equation_of_state is None


def test_additive_action_countermodel_keeps_field_dynamics_but_changes_stress() -> None:
    receipt = additive_action_countermodel(
        receipt_value=0.4,
        reference_mass_scale=3.0,
        scalar_mass=2.0,
        hidden_action_coefficient=0.2,
    )
    zero_stress = np.asarray(receipt.zero_source_stress_covariant)
    nonzero_stress = np.asarray(receipt.nonzero_source_stress_covariant)

    assert receipt.constant_field_value == pytest.approx(1.2)
    assert receipt.same_operational_receipt_without_action_normalization
    assert receipt.same_constant_on_shell_field
    assert receipt.same_scalar_eom_for_positive_coefficient
    assert receipt.same_principal_symbol_for_positive_coefficient
    assert receipt.both_stresses_conserved_on_shell
    assert receipt.finite_coefficient_metric_sources_distinct
    assert not receipt.additive_source_selected_by_receipt
    assert np.array_equal(zero_stress, np.zeros((4, 4)))
    assert nonzero_stress[0, 0] > 0.0
    assert np.all(np.diag(nonzero_stress)[1:] < 0.0)
    assert receipt.normalized_stress_difference == pytest.approx(1.0)


def test_epsilon_zero_is_only_a_source_decoupling_statement() -> None:
    receipt = additive_action_countermodel()

    assert receipt.zero_coefficient_hidden_stress_residual == 0.0
    assert receipt.zero_coefficient_hidden_eom_coefficient == 0.0
    assert receipt.zero_coefficient_hidden_metric_source_vanishes
    assert not receipt.metric_solution_convergence_derived
    with pytest.raises(ValueError, match="hidden_action_coefficient"):
        additive_action_countermodel(hidden_action_coefficient=0.0)


def test_dimension_ledger_requires_a_mass_four_source_scale() -> None:
    receipt = certificate()

    assert receipt.receipt_mass_dimension == 0
    assert receipt.metric_mass_dimension == 0
    assert receipt.reference_scale_mass_dimension == 1
    assert receipt.scalar_field_mass_dimension == 1
    assert receipt.scalar_mass_dimension == 1
    assert receipt.derivative_mass_dimension == 1
    assert receipt.potential_mass_dimension == 4
    assert receipt.stress_mass_dimension == 4
    assert receipt.action_density_mass_dimension == 4
    assert receipt.volume_element_mass_dimension == -4
    assert receipt.action_mass_dimension == 0
    assert receipt.hidden_action_coefficient_mass_dimension == 0
    assert receipt.dimensions_pass


def test_rank_complete_e31_receipt_still_does_not_select_a_source() -> None:
    receipt = certificate()

    assert receipt.e31_full_receipt_combined_rank == 16
    assert receipt.e31_receipt_kernel_rank == 7
    assert receipt.e31_rank_complete_receipt
    assert not receipt.rank_complete_receipt_selects_physical_source


def test_accounting_modes_are_exclusive_and_do_not_rebook_probability() -> None:
    retained = source_accounting_receipt("retained_hidden_field")
    influence = source_accounting_receipt("integrated_out_influence")
    no_source = source_accounting_receipt("receipt_only_no_source")

    assert retained.retained_hidden_stress_added
    assert not retained.integrated_out_influence_response_added
    assert influence.integrated_out_influence_response_added
    assert not influence.retained_hidden_stress_added
    assert not no_source.retained_hidden_stress_added
    assert not no_source.integrated_out_influence_response_added
    for receipt in (retained, influence, no_source):
        assert receipt.mutually_exclusive_source_accounting
        assert receipt.declared_no_probability_energy_rebooking
        assert not receipt.rn_probability_reweighting_added_as_energy
        assert not receipt.rank_or_volume_added_as_energy
    with pytest.raises(ValueError, match="unknown source accounting mode"):
        source_accounting_receipt("retained_plus_integrated")


def test_certificate_keeps_physical_claim_ceiling_false() -> None:
    receipt = certificate()

    assert receipt.scalar_only_order_zero_source_is_vacuum_form
    assert not receipt.dust_source_derived
    assert receipt.current_gradient_or_kinetic_data_required_for_dust
    assert not receipt.local_receipt_to_field_map_derived
    assert not receipt.supplied_metric_derived_from_receipt
    assert not receipt.metric_variation_machine_verified
    assert not receipt.conditional_ward_theorem_replaced_by_numerics
    assert not receipt.cptp_quantum_dynamics_derived
    assert not receipt.qft_microcausality_derived
    assert not receipt.operational_no_signalling_derived
    assert not receipt.finite_coefficient_gr_phenomenology_derived
    assert not receipt.independent_holdout_prediction_derived
    assert not receipt.two_residual_classes_reduced
    assert not receipt.complexity_penalty_success


def test_component_order_and_invalid_inputs_fail_closed() -> None:
    assert COMPONENT_ORDER == (
        "00",
        "01",
        "02",
        "03",
        "11",
        "12",
        "13",
        "22",
        "23",
        "33",
    )
    assert np.array_equal(
        tensor_from_components((-1, 0, 0, 0, 1, 0, 0, 1, 0, 1)),
        minkowski_metric(),
    )
    with pytest.raises(ValueError, match="ten finite"):
        tensor_from_components((1.0, 2.0))
    with pytest.raises(ValueError, match="unknown Lorentz generator"):
        infinitesimal_invariance_constraint(("not-a-generator",))
    with pytest.raises(ValueError, match="at least one"):
        infinitesimal_invariance_constraint(())
    with pytest.raises(ValueError, match="reference_mass_scale"):
        additive_action_countermodel(reference_mass_scale=0.0)


def test_action_coefficients_are_computed_from_the_declared_formulas() -> None:
    phi0 = 1.25
    phi = 1.75
    mass = 2.0
    epsilon = 0.3
    gradient = np.array((0.2, -0.1, 0.05, 0.0))

    assert canonical_scalar_potential(
        phi,
        field_minimum=phi0,
        scalar_mass=mass,
        additive_density=7.0,
    ) == pytest.approx(7.5)
    assert canonical_scalar_potential_derivative(
        phi,
        field_minimum=phi0,
        scalar_mass=mass,
    ) == pytest.approx(2.0)
    assert canonical_scalar_eom(
        phi,
        box_field=3.0,
        field_minimum=phi0,
        scalar_mass=mass,
        hidden_action_coefficient=epsilon,
    ) == pytest.approx(0.3)
    assert canonical_scalar_principal_coefficient(epsilon) == epsilon
    stress = canonical_scalar_stress_at_flat_point(
        phi,
        gradient_covector=gradient,
        field_minimum=phi0,
        scalar_mass=mass,
        additive_density=7.0,
        hidden_action_coefficient=epsilon,
    )
    assert stress.shape == (4, 4)
    assert np.allclose(stress, stress.T)
    ward = canonical_scalar_ward_divergence(
        0.3,
        gradient_covector=gradient,
    )
    assert np.allclose(ward, 0.3 * gradient)
    assert np.array_equal(
        canonical_scalar_stress_at_flat_point(
            phi,
            gradient_covector=gradient,
            field_minimum=phi0,
            scalar_mass=mass,
            additive_density=7.0,
            hidden_action_coefficient=0.0,
        ),
        np.zeros((4, 4)),
    )
