from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.causal.contextual_obstruction import (
    QUANTUM_ETA,
    isotropic_chsh_box,
    marginal_incidence_matrix,
    marginalize_global_weights,
    quantum_kernel_perturbed_extension,
    swap_opposite_score_weights,
    symmetric_signed_global_extension,
    walsh_kernel_vectors,
)
from examples.physics.causal.receipt_source import (
    COMPONENT_ORDER,
    UNIFORM_CONTEXT_WEIGHTS,
    additive_action_countermodel,
    atom_permutation_matrix,
    canonical_scalar_eom,
    canonical_scalar_potential,
    canonical_scalar_potential_derivative,
    canonical_scalar_principal_coefficient,
    canonical_scalar_stress_at_flat_point,
    canonical_scalar_ward_divergence,
    certificate,
    combined_readout_rank,
    conditional_conformal_metric,
    conditional_fisher_quadratic,
    context_block_permutation,
    factor_linear_source,
    fisher_pullback_metric,
    hellinger_coordinates,
    hellinger_tangent,
    high_frequency_volume_witness,
    infinitesimal_invariance_constraint,
    isotropic_fisher_component,
    isotropic_fisher_distance,
    linear_source_factorization_residual,
    linear_source_kernel_residual,
    lorentz_generators,
    lorentz_natural_tensor_certificate,
    lorentzian_signature,
    matrix_inertia,
    metric_volume_ratio,
    minkowski_metric,
    normalized_atom_tangent_basis,
    product_fisher_rao_distance,
    receipt_kernel_rank,
    reconstruct_from_visible_and_walsh,
    representation_certificate,
    representation_run,
    run,
    scalar_receipt_certificate,
    source_accounting_receipt,
    tensor_from_components,
    vacuum_form_receipt,
    visible_and_walsh_receipt,
    walsh_receipt_matrix,
)


def _single_context_tangent() -> np.ndarray:
    tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    tangent[0, 0] = np.asarray(((0.25, -0.25), (-0.25, 0.25)))
    return tangent


def test_context_weighted_fisher_and_hellinger_factor_count_each_context_once() -> None:
    box = isotropic_chsh_box(0.0)
    tangent = _single_context_tangent()
    coordinates = hellinger_coordinates(box)
    differential = hellinger_tangent(box, tangent)
    quadratic = conditional_fisher_quadratic(box, tangent)

    assert float(np.sum(coordinates * coordinates)) == pytest.approx(4.0)
    assert quadratic == pytest.approx(0.25)
    assert float(np.sum(differential * differential)) == pytest.approx(quadratic)


def test_pullback_has_exact_incidence_rank_and_same_seven_kernel_directions() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    incidence = marginal_incidence_matrix().astype(np.float64)
    metric = fisher_pullback_metric(target, incidence=incidence)

    assert matrix_inertia(metric) == (9, 0, 7)
    assert np.allclose(metric, metric.T, atol=1.0e-12, rtol=0.0)
    for vector in walsh_kernel_vectors().values():
        direction = np.asarray(vector, dtype=np.float64)
        assert np.array_equal(incidence @ direction, np.zeros(16))
        assert np.max(np.abs(metric @ direction)) < 1.0e-12


def test_normalized_global_tangent_quotient_has_rank_eight() -> None:
    metric = fisher_pullback_metric(isotropic_chsh_box(QUANTUM_ETA))
    basis = normalized_atom_tangent_basis()
    restricted = basis.T @ metric @ basis

    assert basis.shape == (16, 15)
    assert np.allclose(np.sum(basis, axis=0), 0.0)
    assert matrix_inertia(restricted) == (8, 0, 7)


def test_signed_kernel_lifts_have_identical_visible_box_and_fisher_pullback() -> None:
    base_q = symmetric_signed_global_extension(QUANTUM_ETA)
    shifted_q = quantum_kernel_perturbed_extension(0.1)
    base_box = marginalize_global_weights(base_q)
    shifted_box = marginalize_global_weights(shifted_q)

    assert base_q != shifted_q
    assert np.allclose(base_box, shifted_box, atol=1.0e-12, rtol=0.0)
    assert np.allclose(
        fisher_pullback_metric(base_box),
        fisher_pullback_metric(shifted_box),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_simultaneous_coordinate_relabel_is_a_metric_congruence() -> None:
    q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    incidence = marginal_incidence_matrix().astype(np.float64)
    box = marginalize_global_weights(q)
    metric = fisher_pullback_metric(box, incidence=incidence)
    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    atom_permutation = atom_permutation_matrix(tuple(reversed(range(16))))

    relabelled_q = atom_permutation @ q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_box = (row_permutation @ box.reshape(-1)).reshape(2, 2, 2, 2)
    relabelled_weights = tuple(
        UNIFORM_CONTEXT_WEIGHTS[index] for index in context_order
    )
    relabelled_metric = fisher_pullback_metric(
        relabelled_box,
        incidence=relabelled_incidence,
        context_weights=relabelled_weights,
    )

    assert np.allclose(
        relabelled_incidence @ relabelled_q,
        relabelled_box.reshape(-1),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.allclose(
        relabelled_metric,
        atom_permutation @ metric @ atom_permutation.T,
        atol=1.0e-12,
        rtol=0.0,
    )
    result = representation_certificate()
    assert result.general_relabel_fixed_incidence_residual > 0.5
    assert result.status["chosen_general_relabel_is_not_fixed_incidence_automorphism"]


def test_context_design_must_cotransform_unless_uniform_symmetry_is_assumed() -> None:
    result = representation_certificate()
    target = isotropic_chsh_box(QUANTUM_ETA)
    atom_only = marginalize_global_weights(
        swap_opposite_score_weights(symmetric_signed_global_extension(QUANTUM_ETA))
    )

    assert result.fixed_nonuniform_context_swap_residual > 0.1
    assert result.co_transformed_context_swap_residual < 1.0e-12
    assert result.uniform_context_swap_residual < 1.0e-12
    assert not np.allclose(atom_only, target, atol=1.0e-12, rtol=0.0)
    assert result.atom_only_probability_residual > 0.1
    assert result.atom_only_fixed_incidence_residual > 0.5
    assert result.status["atom_only_fixed_incidence_automorphism_excluded"]


def test_isotropic_line_has_arcsine_coordinate_and_completion_boundary() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    origin = isotropic_chsh_box(0.0)

    assert isotropic_fisher_component(QUANTUM_ETA) == pytest.approx(2.0)
    assert isotropic_fisher_distance(0.0, QUANTUM_ETA) == pytest.approx(math.pi / 4.0)
    assert product_fisher_rao_distance(origin, target) == pytest.approx(math.pi / 4.0)
    assert isotropic_fisher_component(0.999) > 500.0
    with pytest.raises(ValueError, match="strict chart"):
        isotropic_fisher_component(1.0)
    with pytest.raises(ValueError, match="positive"):
        product_fisher_rao_distance(origin, isotropic_chsh_box(1.0))


def test_fisher_psd_does_not_supply_a_lorentzian_signature() -> None:
    result = representation_certificate()

    assert result.pullback_inertia == (9, 0, 7)
    assert result.normalized_tangent_inertia == (8, 0, 7)
    assert result.status["fisher_form_positive_semidefinite"]
    assert not result.status["fisher_metric_is_spacetime_lorentz_metric_derived"]
    assert not result.status["lorentzian_signature_or_lightcone_derived_from_fisher"]
    assert result.boundaries["fisher_psd_no_go_is_not_a_general_lorentz_geometry_no_go"]


def test_supplied_lorentz_metric_and_volume_ratio_only_fix_conformal_control() -> None:
    reference = np.diag((-1.0, 1.0, 1.0, 1.0))
    metric = conditional_conformal_metric(reference, 16.0)
    null = np.asarray((1.0, 1.0, 0.0, 0.0))

    assert np.allclose(metric, 4.0 * reference)
    assert lorentzian_signature(metric) == lorentzian_signature(reference) == (3, 1, 0)
    assert metric_volume_ratio(metric, reference) == pytest.approx(16.0)
    assert float(null @ reference @ null) == pytest.approx(0.0)
    assert float(null @ metric @ null) == pytest.approx(0.0)
    assert np.allclose(conditional_conformal_metric(reference, 1.0), reference)


def test_uniform_volume_convergence_does_not_control_second_derivatives() -> None:
    small = high_frequency_volume_witness(10)
    large = high_frequency_volume_witness(100)

    assert small.minimum_volume_ratio > 0.0
    assert large.uniform_value_residual_bound < small.uniform_value_residual_bound
    assert large.probe_value_residual == pytest.approx(1.0e-4)
    assert abs(large.probe_first_derivative) < 1.0e-12
    assert large.probe_second_derivative == pytest.approx(-10000.0)


def test_dimension_accounting_alternatives_and_claim_ceiling_are_explicit() -> None:
    result = representation_certificate()

    assert all(result.dimensions.values())
    assert result.accounting["context_weights_sum_to_one"]
    assert result.accounting["each_context_counted_once_not_once_per_outcome_cell"]
    assert not result.accounting["probability_energy_or_volume_double_counted"]
    assert all(result.alternatives.values())
    assert not result.status["physical_volume_law_derived"]
    assert not result.status["curvature_einstein_dynamics_or_gravity_derived"]
    assert not result.status["gr_c2_limit_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_representation_public_contract_fails_closed_and_serializes() -> None:
    box = isotropic_chsh_box(0.0)
    tangent = _single_context_tangent()

    with pytest.raises(ValueError, match="four finite positive"):
        hellinger_coordinates(box, context_weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        hellinger_coordinates(box, context_weights=(0.4, 0.3, 0.2, 0.2))
    with pytest.raises(ValueError, match="positive"):
        hellinger_coordinates(isotropic_chsh_box(1.0))
    with pytest.raises(ValueError, match="zero sum"):
        conditional_fisher_quadratic(box, np.ones_like(tangent))
    with pytest.raises(ValueError, match="sixteen"):
        fisher_pullback_metric(box, incidence=np.zeros((15, 16)))
    with pytest.raises(ValueError, match="symmetric"):
        matrix_inertia(np.asarray(((1.0, 1.0), (0.0, 1.0))))
    with pytest.raises(ValueError, match="permutation"):
        context_block_permutation((0, 0, 2, 3))
    with pytest.raises(ValueError, match="permutation"):
        atom_permutation_matrix(tuple(range(15)))
    with pytest.raises(ValueError, match="Lorentzian"):
        conditional_conformal_metric(np.eye(4), 1.0)
    with pytest.raises(ValueError, match="positive"):
        conditional_conformal_metric(np.diag((-1.0, 1.0)), 0.0)
    with pytest.raises(ValueError, match="determinant"):
        metric_volume_ratio(np.eye(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        metric_volume_ratio(np.asarray(((1.0, 1.0), (0.0, 1.0))), np.eye(2))
    with pytest.raises(ValueError, match="determinant"):
        metric_volume_ratio(np.diag((1.0, 0.0)), np.eye(2))
    with pytest.raises(ValueError, match="integer"):
        high_frequency_volume_witness(True)
    with pytest.raises(ValueError, match="tolerance"):
        representation_certificate(tolerance=0.0)

    result = representation_certificate()
    payload = representation_run()
    json.dumps(payload)
    json.loads(result.to_json())
    assert payload["incidence_rank"] == 9
    assert payload["normalized_tangent_rank"] == 8


def test_walsh_rows_are_an_exact_orthogonal_basis_of_the_hidden_kernel() -> None:
    incidence = marginal_incidence_matrix()
    walsh = walsh_receipt_matrix()

    assert incidence.shape == (16, 16)
    assert walsh.shape == (7, 16)
    assert np.array_equal(incidence @ walsh.T, np.zeros((16, 7), dtype=int))
    assert np.array_equal(walsh @ walsh.T, 16 * np.eye(7, dtype=int))
    result = certificate()
    assert result.incidence_rank == 9
    assert result.incidence_nullity == 7
    assert result.walsh_rank == 7


def test_one_six_and_seven_receipts_give_the_exact_rank_ladder() -> None:
    walsh = walsh_receipt_matrix()
    result = certificate()

    assert combined_readout_rank(walsh[:1]) == 10
    assert combined_readout_rank(walsh[:6]) == 15
    assert combined_readout_rank(walsh) == 16
    assert receipt_kernel_rank(walsh[:1]) == 1
    assert receipt_kernel_rank(walsh[:6]) == 6
    assert receipt_kernel_rank(walsh) == 7
    assert result.minimum_receipt_rows_for_full_recovery == 7
    assert result.status["receipt_rank_lower_bound_witness_certified"]


def test_normalized_tangent_has_eight_visible_plus_seven_hidden_directions() -> None:
    incidence = marginal_incidence_matrix()
    walsh = walsh_receipt_matrix()
    tangent = normalized_atom_tangent_basis()

    assert tangent.shape == (16, 15)
    assert np.allclose(np.sum(tangent, axis=0), 0.0)
    assert np.linalg.matrix_rank(incidence @ tangent) == 8
    assert combined_readout_rank(walsh[:1], normalized_tangent=True) == 9
    assert combined_readout_rank(walsh[:6], normalized_tangent=True) == 14
    assert combined_readout_rank(walsh, normalized_tangent=True) == 15


def test_linear_source_factors_exactly_when_it_annihilates_the_hidden_kernel() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    visible_source = incidence[[0, 5]]
    factor = factor_linear_source(visible_source)

    assert linear_source_kernel_residual(visible_source) == pytest.approx(0.0)
    assert linear_source_factorization_residual(visible_source) < 1.0e-12
    assert np.allclose(factor @ incidence, visible_source, atol=1.0e-12, rtol=0.0)

    hidden_source = walsh_receipt_matrix()[[0]]
    assert linear_source_kernel_residual(hidden_source) == pytest.approx(16.0)
    assert linear_source_factorization_residual(hidden_source) > 0.5
    with pytest.raises(ValueError, match="not constant"):
        factor_linear_source(hidden_source)


def test_ambient_factor_extension_is_not_unique() -> None:
    result = certificate()

    assert result.visible_source_factorization_residual < 1.0e-12
    assert result.alternative_ambient_factor_residual < 1.0e-12
    assert result.ambient_factor_extension_difference > 0.1
    assert result.status["ambient_factor_extension_nonuniqueness_certified"]


def test_permutation_invariant_norm_is_not_constant_on_visible_fibres() -> None:
    base = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    shifted = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_visible, _ = visible_and_walsh_receipt(base)
    shifted_visible, _ = visible_and_walsh_receipt(shifted)
    result = certificate()

    assert np.allclose(base_visible, shifted_visible, atol=1.0e-12, rtol=0.0)
    assert float(np.dot(shifted, shifted)) - float(np.dot(base, base)) == pytest.approx(
        0.000625
    )
    assert result.permutation_norm_residual < 1.0e-12
    assert result.same_fibre_norm_square_difference == pytest.approx(0.000625)
    assert result.status["permutation_covariance_not_fibre_invariance_certified"]


def test_walsh_receipts_detect_q_delta_and_reconstruct_the_full_coordinate() -> None:
    coordinates = np.asarray(quantum_kernel_perturbed_extension(0.1))
    visible, receipt = visible_and_walsh_receipt(coordinates)
    reconstructed = reconstruct_from_visible_and_walsh(visible, receipt)
    result = certificate()

    assert np.allclose(reconstructed, coordinates, atol=1.0e-12, rtol=0.0)
    assert result.q_delta_visible_residual < 1.0e-12
    assert result.q_delta_first_walsh_change == pytest.approx(0.1)
    assert result.q_delta_other_walsh_residual < 1.0e-12
    assert result.status["full_walsh_coordinate_reconstruction_certified"]


def test_visible_duplicate_receipts_add_no_hidden_rank_or_new_source() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    duplicate = incidence[:7]
    result = certificate()

    assert receipt_kernel_rank(duplicate) == 0
    assert combined_readout_rank(duplicate) == 9
    assert result.accounting["duplicate_receipt_factors_through_visible_map"]
    assert result.accounting["duplicate_receipt_not_added_as_new_source"]
    assert result.status["duplicate_visible_receipt_adds_no_rank_certified"]


def test_general_relabel_covariance_is_not_fixed_map_automorphism() -> None:
    result = certificate()

    assert result.relabel_visible_residual < 1.0e-12
    assert result.relabel_receipt_residual < 1.0e-12
    assert result.relabel_combined_rank == 16
    assert result.relabel_fixed_incidence_residual == pytest.approx(1.0)
    assert result.relabel_fixed_receipt_residual == pytest.approx(2.0)
    assert result.status["general_relabel_covariance_certified"]
    assert not result.status["chosen_general_relabel_is_fixed_map_automorphism"]


def test_dimension_accounting_alternatives_and_physical_claim_ceiling() -> None:
    result = certificate()

    assert all(result.dimensions.values())
    assert not result.accounting["receipt_probability_energy_or_volume_double_counted"]
    assert all(result.alternatives.values())
    assert result.boundaries["seven_rows_are_necessary_only_for_full_linear_q_recovery"]
    assert result.boundaries["seven_is_not_a_gravity_component_field_or_boson_count"]
    assert not result.status["physical_walsh_receipt_derived"]
    assert not result.status["hidden_signed_coordinate_is_physical_state_derived"]
    assert not result.status["local_covariant_action_or_stress_derived"]
    assert not result.status["spacetime_metric_curvature_or_gravity_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed_and_serializes() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    visible, receipt = visible_and_walsh_receipt(np.full(16, 1.0 / 16.0))

    with pytest.raises(ValueError, match="sixteen"):
        linear_source_kernel_residual(np.ones(15))
    with pytest.raises(ValueError, match="finite"):
        receipt_kernel_rank(np.full((1, 16), np.nan))
    with pytest.raises(ValueError, match="seven"):
        reconstruct_from_visible_and_walsh(visible, receipt[:6])
    inconsistent = np.array(visible, copy=True)
    inconsistent[0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        reconstruct_from_visible_and_walsh(inconsistent, receipt)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)

    payload = run()
    json.dumps(payload)
    json.loads(certificate().to_json())
    assert payload["combined_rank_seven_receipts"] == 16
    assert np.array_equal(incidence @ walsh_receipt_matrix().T, np.zeros((16, 7)))


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
    receipt = scalar_receipt_certificate()

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
    receipt = scalar_receipt_certificate()

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
    receipt = scalar_receipt_certificate()

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
