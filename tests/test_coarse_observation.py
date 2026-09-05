"""거친 관측 섹터 병합 모듈(coarse_observation)의 테스트다."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.record.coarse_observation import (
    CANONICAL_BOUNDARY_PARAMETERS,
    CANONICAL_STABLE_PARAMETERS,
    CANONICAL_TACHYON_PARAMETERS,
    I2,
    P0,
    P1,
    apparatus_zero_embedding,
    apply_cp_map,
    basis_obstruction_audit,
    bilinear_spectrum_audit,
    canonical_on_shell_ward_audit,
    certificate,
    certify_finite_probability_deformation_readout,
    choi_matrix,
    coarse_visibility_labels,
    controlled_record_inverse,
    controlled_record_unitary,
    dimension_audit,
    duplicate_operation,
    fine_visibility_labels,
    isometric_refinement,
    kraus_certificate,
    partial_trace_apparatus,
    projective_dephasing,
    raw_count_conformal_factor,
    record_fold_certificate,
    record_fold_run,
    record_isometry,
    record_kraus_operators,
    require_stable_spectrum,
    run,
    schur_complement_audit,
    selective_update,
    source_accounting_audit,
    ward_exchange_audit,
)


@pytest.mark.parametrize("dimension", [3, 4])
def test_controlled_shift_is_a_bijective_unitary_with_explicit_inverse(
    dimension: int,
) -> None:
    unitary = controlled_record_unitary(dimension)
    inverse = controlled_record_inverse(dimension)
    identity = np.eye(dimension * dimension)
    assert np.allclose(unitary.conj().T @ unitary, identity)
    assert np.allclose(unitary @ unitary.conj().T, identity)
    assert np.allclose(inverse, unitary.conj().T)
    assert np.allclose(inverse @ unitary, identity)


def test_record_restriction_is_isometric_and_bijective_only_onto_its_image() -> None:
    dimension = 3
    isometry = record_isometry(dimension)
    assert isometry.shape == (dimension * dimension, dimension)
    assert np.linalg.matrix_rank(isometry) == dimension
    assert np.allclose(isometry.conj().T @ isometry, np.eye(dimension))
    assert np.allclose(
        isometry,
        controlled_record_unitary(dimension) @ apparatus_zero_embedding(dimension),
    )
    result = certificate(dimension=dimension)
    assert result.record_output_residual < 1.0e-12
    assert result.inverse_recovery_residual < 1.0e-12
    assert result.status["record_isometry_bijective_onto_its_image"]
    assert not result.status["record_isometry_surjective_onto_full_joint_space"]


def test_partial_trace_is_the_cptp_dephasing_channel() -> None:
    result = certificate()
    assert result.reduced_system_residual < 1.0e-12
    assert result.reduced_apparatus_residual < 1.0e-12
    assert result.kraus_completeness_residual < 1.0e-12
    assert result.kraus_channel_residual < 1.0e-12
    assert result.choi_minimum_eigenvalue >= -1.0e-12
    assert result.status["explicit_projective_record_channel_cptp"]

    operators = record_kraus_operators(3)
    assert len(operators) == 3
    assert all(np.allclose(operator, np.diag(np.eye(3)[index])) for index, operator in enumerate(operators))


def test_fine_sort_is_bijective_while_coarse_readout_forgets_hidden_identity() -> None:
    fine = fine_visibility_labels(3, 1)
    coarse = coarse_visibility_labels(3, 1)
    assert fine == (("hidden", 0), ("visible", 1), ("hidden", 2))
    assert len(set(fine)) == 3
    assert len(set(coarse)) == 2

    # 경계 사례: 숨은 라벨이 하나뿐이면 이진 읽기는 우연히 단사로 남는다.
    # 그렇다고 선택적 상태 갱신이 단사가 되지는 않는다.
    assert len(set(coarse_visibility_labels(2, 0))) == 2
    assert certificate(dimension=2, selected=0).status[
        "coarse_visibility_readout_injective"
    ]
    assert not certificate().status["coarse_visibility_readout_injective"]


def test_selective_and_nonselective_state_updates_are_many_to_one() -> None:
    result = certificate()
    assert result.distinct_input_residual > 1.0e-3
    assert result.nonselective_collision_residual < 1.0e-12
    assert result.selective_operation_collision_residual < 1.0e-12
    assert result.selective_posterior_collision_residual < 1.0e-12
    assert result.status["nonselective_dephasing_many_to_one_witness"]
    assert result.status["selective_update_many_to_one_witness"]


def test_probability_accounting_and_claim_ceiling_are_explicit() -> None:
    result = certificate()
    assert result.status["declared_finite_controlled_unitary_bijective"]
    assert result.status["fine_discrete_label_sort_bijective_onto_declared_image"]
    assert result.status[
        "fine_discrete_label_bijection_onto_image_is_homeomorphism"
    ]
    assert result.status["premeasurement_components_preserved_by_fine_unitary"]
    assert all(result.dimensions.values())
    assert result.accounting["branch_probabilities_sum_to_one"]
    assert result.accounting["visible_plus_hidden_probability_sum_to_one"]
    assert result.accounting[
        "hidden_labels_retained_individually_in_fine_label_map"
    ]
    assert not result.accounting["coarse_and_fine_probabilities_added_as_separate_energy"]
    assert result.boundaries["selected_label_is_not_an_input_to_controlled_unitary"]
    assert result.boundaries[
        "finite_w_is_declared_model_not_actual_universe_dynamics"
    ]
    assert result.boundaries["fine_sort_is_label_only_not_physical_branch_dynamics"]
    assert result.boundaries[
        "fine_sort_codomain_is_declared_image_not_full_cartesian_product"
    ]
    assert result.boundaries["finite_label_topology_declared_discrete"]
    assert result.boundaries[
        "finite_dimension_is_hilbert_label_dimension_not_spacetime_dimension"
    ]
    assert result.boundaries["cptp_claim_is_for_the_explicit_projective_instrument"]
    assert not result.status["unitary_selects_one_unique_actual_outcome"]
    assert not result.status["durable_physical_pointer_derived"]
    assert not result.status["energy_hamiltonian_or_transfer_derived"]
    assert not result.status["spacetime_homeomorphism_derived"]
    assert not result.status["spacetime_metric_or_curvature_derived"]
    assert not result.status["fold_stress_or_gravity_derived"]
    assert not result.status["success_gates_5_to_8_complete"]


def test_public_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        controlled_record_unitary(1)
    with pytest.raises(ValueError, match="at least two"):
        controlled_record_inverse(True)
    with pytest.raises(ValueError, match="record range"):
        fine_visibility_labels(3, 3)
    with pytest.raises(ValueError, match="positive probability"):
        selective_update(np.diag([1.0, 0.0]), 1)
    with pytest.raises(ValueError, match="unit trace"):
        projective_dephasing(np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        partial_trace_apparatus(np.eye(3) / 3.0, 2)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)


def test_run_payload_is_json_serializable_and_keeps_status_ceiling() -> None:
    payload = run()
    json.dumps(payload)
    assert payload["status"]["declared_finite_controlled_unitary_bijective"]
    assert not payload["status"]["unitary_selects_one_unique_actual_outcome"]


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


def test_nonzero_duplication_preserves_operation_probability_and_posterior() -> None:
    result = kraus_certificate()
    assert result.hidden_multiplicities == (1, 2, 16, 37)
    assert result.maximum_operation_residual < 1.0e-12
    assert result.maximum_coarse_probability_residual < 1.0e-12
    assert result.maximum_posterior_residual < 1.0e-12
    assert all(math.isclose(value, result.outcome_probability) for value in result.sublabel_probability_sums)
    assert result.status["outcome_operation_isometry_invariant"]
    assert result.status["coarse_probability_invariant"]
    assert result.status["posterior_invariant"]


def test_full_instrument_remains_cptp_and_choi_matrix_is_invariant() -> None:
    result = kraus_certificate()
    assert result.maximum_full_completeness_residual < 1.0e-12
    assert result.maximum_total_probability_residual < 1.0e-12
    assert result.maximum_choi_residual < 1.0e-12
    assert result.numerical_choi_ranks == (1, 1, 1, 1)
    assert result.status["cptp_completeness_preserved"]
    assert result.status["choi_matrix_invariant"]
    assert result.status["choi_rank_numerically_invariant"]
    assert not result.status["minimal_kraus_rank_theorem_proved_by_finite_regression"]


def test_general_isometric_mixing_preserves_a_nontrivial_dephasing_channel() -> None:
    isometry = 0.5 * np.array(
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0j], [1.0, -1.0j]],
        dtype=np.complex128,
    )
    refined = isometric_refinement((P0, P1), isometry)
    state = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=np.complex128)
    assert len(refined) == 4
    assert np.allclose(apply_cp_map(refined, state), apply_cp_map((P0, P1), state))
    assert np.allclose(choi_matrix(refined), choi_matrix((P0, P1)))
    result = kraus_certificate()
    assert result.general_isometry_shape == (4, 2)
    assert result.general_isometry_residual < 1.0e-12
    assert result.general_channel_residual < 1.0e-12


def test_raw_count_metric_candidate_fails_representation_invariance() -> None:
    result = kraus_certificate(spacetime_dimension=4)
    assert result.raw_conformal_factors[0] == 1.0
    assert result.raw_conformal_factors[2] == 2.0
    assert result.raw_metric_coefficient_ratios[0] == 1.0
    assert result.raw_metric_coefficient_ratios[2] == 4.0
    assert not result.status["raw_hidden_count_invariant"]
    assert result.status["raw_count_metric_changes_for_same_instrument"]
    assert not result.status["raw_count_defines_physical_volume_or_metric"]


def test_dimension_accounting_boundary_and_status_ceiling() -> None:
    result = kraus_certificate()
    assert all(result.dimensions.values())
    assert result.accounting["refined_sublabel_probabilities_sum_to_coarse_probability"]
    assert result.accounting["coarse_plus_refined_probability_double_counting_forbidden"]
    assert not result.accounting["representation_only_sublabel_adds_energy_or_stress"]
    assert not result.accounting["energy_receipt_or_stress_used"]
    assert result.boundaries["sublabel_is_unobserved"]
    assert not result.boundaries["physical_pointer_record_derived"]
    assert all(result.alternatives.values())
    assert not result.status["local_volume_measure_derived"]
    assert not result.status["metric_or_curvature_derived"]
    assert not result.status["gr_lensing_backreaction_derived"]
    assert not result.status["success_gates_5_to_8_complete"]


def test_public_helpers_fail_closed__kraus() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        duplicate_operation(I2, 0)
    with pytest.raises(ValueError, match="positive integer"):
        duplicate_operation(I2, True)
    with pytest.raises(ValueError, match="isometry columns"):
        isometric_refinement((P0, P1), np.eye(3))
    with pytest.raises(ValueError, match=r"u\^dagger u"):
        isometric_refinement((P0, P1), np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least two"):
        raw_count_conformal_factor(1, spacetime_dimension=1)
    with pytest.raises(ValueError, match="outcome_probability"):
        kraus_certificate(outcome_probability=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        kraus_certificate(hidden_multiplicities=())
    with pytest.raises(ValueError, match="positive integer"):
        kraus_certificate(hidden_multiplicities=(1, False))


def test_relative_choi_rank_remains_one_for_a_tiny_nonzero_outcome() -> None:
    result = kraus_certificate(outcome_probability=1.0e-20)
    assert result.numerical_choi_ranks == (1, 1, 1, 1)
    assert result.maximum_total_probability_residual < 1.0e-12


def test_dimension_ledger_does_not_convert_probability_into_energy() -> None:
    receipt = dimension_audit()

    assert receipt.record_field_mass_dimension == 1
    assert receipt.fold_field_mass_dimension == 1
    assert receipt.mixing_kappa_mass_dimension == 2
    assert receipt.source_coefficient_mass_dimension == 3
    assert receipt.lagrangian_density_mass_dimension == 4
    assert receipt.stress_mass_dimension == 4
    assert receipt.ward_current_mass_dimension == 5
    assert receipt.action_mass_dimension == 0
    assert receipt.dimensions_pass
    assert not receipt.probability_used_as_source_coefficient


def test_stable_witness_has_the_analytic_eigenmass_squared_values() -> None:
    receipt = bilinear_spectrum_audit(*CANONICAL_STABLE_PARAMETERS)
    expected_high = 0.5 * (13.0 + math.sqrt(41.0))
    expected_low = 0.5 * (13.0 - math.sqrt(41.0))
    rotated = np.asarray(receipt.rotated_mass_squared_matrix)

    assert receipt.determinant_mass_four == pytest.approx(32.0)
    assert receipt.eigenmass_squared_high == pytest.approx(expected_high)
    assert receipt.eigenmass_squared_low == pytest.approx(expected_low)
    assert receipt.positive_by_principal_minors
    assert receipt.strictly_stable
    assert not receipt.tachyonic_mode_present
    assert not receipt.boundary_zero_mode_present
    assert receipt.canonical_kinetic_ghost_free
    assert receipt.rotated_off_diagonal_residual < 1.0e-12
    assert receipt.kinetic_rotation_residual < 1.0e-12
    assert sorted(np.diag(rotated)) == pytest.approx(
        sorted((expected_low, expected_high))
    )


def test_tachyon_and_zero_mode_counterexamples_fail_closed() -> None:
    tachyon = bilinear_spectrum_audit(*CANONICAL_TACHYON_PARAMETERS)
    boundary = bilinear_spectrum_audit(*CANONICAL_BOUNDARY_PARAMETERS)

    assert tachyon.determinant_mass_four == pytest.approx(-3.0)
    assert tachyon.eigenmass_squared_high == pytest.approx(3.0)
    assert tachyon.eigenmass_squared_low == pytest.approx(-1.0)
    assert tachyon.tachyonic_mode_present
    assert not tachyon.strictly_stable
    assert boundary.determinant_mass_four == pytest.approx(0.0)
    assert boundary.eigenmass_squared_low == pytest.approx(0.0)
    assert boundary.boundary_zero_mode_present
    assert not boundary.strictly_stable
    with pytest.raises(ValueError, match="not strictly stable"):
        require_stable_spectrum(*CANONICAL_TACHYON_PARAMETERS)
    with pytest.raises(ValueError, match="not strictly stable"):
        require_stable_spectrum(*CANONICAL_BOUNDARY_PARAMETERS)


def test_on_shell_ward_witness_has_equal_and_opposite_exchange() -> None:
    receipt = canonical_on_shell_ward_audit()
    free_fold = np.asarray(receipt.free_fold_stress_divergence)
    record_and_interaction = np.asarray(
        receipt.record_plus_interaction_divergence
    )

    assert receipt.source_coefficient == pytest.approx(1.0)
    assert receipt.record_eom_residual == pytest.approx(0.0)
    assert receipt.fold_eom_residual == pytest.approx(0.0)
    assert np.allclose(free_fold, -record_and_interaction)
    assert np.allclose(receipt.total_stress_divergence, np.zeros(4))
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_exchange_balance_residual < 1.0e-12
    assert receipt.both_field_equations_on_shell
    assert receipt.total_stress_conserved_on_shell
    assert receipt.interaction_counted_once


def test_off_shell_ward_identity_is_not_misreported_as_conservation() -> None:
    receipt = ward_exchange_audit(
        record_value=0.5,
        fold_value=-0.25,
        box_record=0.0,
        box_fold=0.0,
        record_gradient_covector=(0.3, -0.2, 0.1, 0.0),
        fold_gradient_covector=(-0.4, 0.05, 0.0, 0.2),
    )

    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence > 0.0
    assert not receipt.both_field_equations_on_shell
    assert not receipt.total_stress_conserved_on_shell


def test_action_sign_convention_puts_minus_j_on_the_fold_rhs() -> None:
    receipt = ward_exchange_audit(
        record_value=0.5,
        fold_value=0.0,
        box_record=4.5,
        box_fold=-1.0,
        record_gradient_covector=(1.0, 0.0, 0.0, 0.0),
        fold_gradient_covector=(0.0, 1.0, 0.0, 0.0),
    )

    assert receipt.source_coefficient == pytest.approx(1.0)
    assert receipt.fold_eom_residual == pytest.approx(0.0)
    assert receipt.free_fold_stress_divergence == pytest.approx(
        (0.0, -1.0, 0.0, 0.0)
    )


def test_static_schur_complement_is_positive_only_inside_the_stable_witness() -> None:
    stable = schur_complement_audit(*CANONICAL_STABLE_PARAMETERS)
    tachyon = schur_complement_audit(*CANONICAL_TACHYON_PARAMETERS)

    assert stable.static_effective_fold_mass_squared == pytest.approx(32.0 / 9.0)
    assert stable.determinant_over_record_mass_squared == pytest.approx(32.0 / 9.0)
    assert stable.positive_static_effective_mass
    assert stable.operator_kernel == "D_phi - kappa^2 D_R^{-1}"
    assert stable.zero_momentum_local_formula_only
    assert stable.inverse_boundary_or_state_prescription_required
    assert not stable.retarded_inverse_automatically_selected
    assert not stable.closed_time_path_noise_derived
    assert not stable.local_effective_stress_automatically_derived
    assert tachyon.static_effective_fold_mass_squared == pytest.approx(-3.0)
    assert not tachyon.positive_static_effective_mass


def test_retained_and_integrated_out_source_ledgers_are_exclusive() -> None:
    retained = source_accounting_audit("retained_fields")
    influence = source_accounting_audit("integrated_out_influence")

    assert retained.retained_record_and_fold_fields
    assert retained.original_bilinear_interaction_retained
    assert not retained.integrated_out_influence_kernel
    assert influence.integrated_out_influence_kernel
    assert not influence.retained_record_and_fold_fields
    assert not influence.original_bilinear_interaction_retained
    for receipt in (retained, influence):
        assert receipt.mutually_exclusive_representations
        assert not receipt.probability_rebooked_as_energy
        assert not receipt.source_stress_counted_twice
    with pytest.raises(ValueError, match="unknown source accounting mode"):
        source_accounting_audit("retained_plus_influence")


def test_basis_rotation_removes_mixing_but_does_not_select_a_record() -> None:
    receipt = basis_obstruction_audit()

    assert receipt.eigenmass_squared_set == pytest.approx((4.0, 6.0))
    assert receipt.absolute_rotation_angle_degrees == pytest.approx(45.0)
    assert receipt.rotated_off_diagonal_residual < 1.0e-12
    assert receipt.kinetic_rotation_residual < 1.0e-12
    assert receipt.hypothetical_pointer_vector_eigenbasis == pytest.approx(
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    )
    assert receipt.hypothetical_pointer_is_extra_input
    assert receipt.eigenmass_squared_set_basis_invariant
    assert not receipt.record_and_fold_labels_basis_invariant
    assert not receipt.bilinear_mixing_selects_pointer_basis
    assert not receipt.bilinear_mixing_derives_observed_outcome
    assert not receipt.bilinear_mixing_derives_dark_source


def test_zero_mixing_decouples_classical_equations_but_does_not_prove_gr_limit() -> None:
    spectrum = bilinear_spectrum_audit(9.0, 4.0, 0.0)
    top = record_fold_certificate()

    assert spectrum.mass_squared_matrix == ((9.0, -0.0), (-0.0, 4.0))
    assert spectrum.strictly_stable
    assert not top.zero_stress_qm_gr_limit_derived
    assert not top.gravitational_solution_derived


def test_certificate_preserves_all_physical_claim_ceilings() -> None:
    receipt = record_fold_certificate()

    assert receipt.status == "CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION"
    assert receipt.one_total_action_accounting_admitted
    assert receipt.classical_principal_symbol_uses_metric_cone
    assert not receipt.nonselected_quantum_to_record_map_derived
    assert not receipt.pointer_selection_and_durable_record_derived
    assert not receipt.probability_deformation_defined
    assert not receipt.cptp_and_normalization_derived
    assert not receipt.qft_microcausality_derived
    assert not receipt.operational_no_signalling_derived
    assert not receipt.fixed_parameter_manifest_established
    assert not receipt.independent_holdout_prediction_derived
    assert not receipt.two_residual_classes_reduced
    assert not receipt.complexity_penalized_improvement_established
    assert "=-J_ns" in receipt.source_sign_convention


def test_invalid_numeric_inputs_fail_closed_and_run_is_serializable() -> None:
    with pytest.raises(ValueError, match="record_mass_squared"):
        bilinear_spectrum_audit(float("nan"), 4.0, 2.0)
    with pytest.raises(ValueError, match="record_mass_squared"):
        schur_complement_audit(0.0, 4.0, 2.0)
    with pytest.raises(ValueError, match="four finite"):
        ward_exchange_audit(
            record_value=0.0,
            fold_value=0.0,
            box_record=0.0,
            box_fold=0.0,
            record_gradient_covector=(1.0, 2.0),
            fold_gradient_covector=(0.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="reference_mass_scale"):
        ward_exchange_audit(
            record_value=0.0,
            fold_value=0.0,
            box_record=0.0,
            box_fold=0.0,
            record_gradient_covector=(0.0, 0.0, 0.0, 0.0),
            fold_gradient_covector=(0.0, 0.0, 0.0, 0.0),
            reference_mass_scale=0.0,
        )
    payload = record_fold_run()
    assert payload["status"] == "CONDITIONAL_CLASSICAL_TWO_FIELD_ADMISSION"
    assert payload["stable_witness"]["strictly_stable"]
