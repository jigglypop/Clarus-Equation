from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_fp_berezin_weight import (
    SOURCE_TRANSCRIPTION_SHA256,
    berezin_gaussian_integral,
    berezin_matrix_fixtures,
    commutator_residual_l1,
    covariant_second_jet,
    determinant_leibniz,
    determinant_reference_ratio,
    derive_fp_length_dimensions,
    evaluate_fp_berezin_gate,
    expanded_gauge_variation,
    fp_berezin_contract,
    fp_potential_action,
    permutation_basis,
    similarity_transform,
    singular_berezin_fixture,
    source_payload_sha256,
    target_gauge_variation,
    validate_contract,
    xi_fixture,
)
from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    kulkarni_nomizu_curvature,
    symmetric_fixture,
    vector_fixture,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_fp_berezin_gate()


def test_exact_fp_variation_and_operator_relation_pass(receipt) -> None:
    assert receipt.fixture_dimensions == (3, 4, 5)
    assert receipt.fixture_count == 9
    assert receipt.generic_vector_fixture_count == 3
    assert receipt.zero_vector_fixture_count == 3
    assert receipt.pure_weyl_fixture_count == 2
    assert receipt.flat_fixture_count == 1
    assert receipt.curvature_audit_count == 6
    assert receipt.curvature_audits_all_passed
    assert receipt.weyl_fixtures_nonzero_and_ricci_flat
    assert receipt.commutator_residuals_l1 == ('0',) * 9
    assert receipt.gauge_variation_residuals == ('0',) * 36
    assert receipt.fp_operator_residuals == ('0',) * 36
    assert receipt.exact_component_count == 72
    assert receipt.exact_fp_variation_all_passed
    assert receipt.declared_finite_fp_berezin_gate_passed


def test_local_jet_derivation_is_direct_and_exact() -> None:
    dimension = 4
    curvature = kulkarni_nomizu_curvature(
        symmetric_fixture(dimension)
    )
    xi = xi_fixture(dimension)
    gradient = vector_fixture(dimension)
    second_jet = covariant_second_jet(curvature, xi)
    expanded = expanded_gauge_variation(
        second_jet,
        xi,
        gradient,
    )
    target = target_gauge_variation(
        curvature,
        second_jet,
        xi,
        gradient,
    )
    laplacian = tuple(
        sum(
            (second_jet[mu][mu][nu] for mu in range(dimension)),
            Fraction(0),
        )
        for nu in range(dimension)
    )
    potential = fp_potential_action(curvature, xi, gradient)

    assert commutator_residual_l1(curvature, xi, second_jet) == 0
    assert expanded == target
    assert tuple(-value for value in expanded) == tuple(
        -laplacian[index] + potential[index]
        for index in range(dimension)
    )


def test_finite_berezin_integral_is_computed_independently(receipt) -> None:
    matrices = berezin_matrix_fixtures()
    expected = (Fraction(2), Fraction(7), Fraction(16))
    assert tuple(determinant_leibniz(matrix) for matrix in matrices) == expected
    assert tuple(
        berezin_gaussian_integral(matrix) for matrix in matrices
    ) == expected
    assert receipt.determinant_values == ('2', '7', '16')
    assert receipt.berezin_integral_values == ('2', '7', '16')
    assert receipt.berezin_residuals == ('0', '0', '0')
    assert receipt.finite_berezin_identity_all_passed

    singular = singular_berezin_fixture()
    assert determinant_leibniz(singular) == 0
    assert berezin_gaussian_integral(singular) == 0
    with pytest.raises(ValueError):
        determinant_reference_ratio(singular, singular)
    assert receipt.zero_mode_rejected


def test_basis_covariance_reference_scaling_and_sign_limit(receipt) -> None:
    for matrix in berezin_matrix_fixtures():
        transformed = similarity_transform(
            matrix,
            permutation_basis(len(matrix)),
        )
        assert determinant_leibniz(transformed) == determinant_leibniz(
            matrix
        )
        assert berezin_gaussian_integral(transformed) == (
            berezin_gaussian_integral(matrix)
        )
    assert receipt.transpose_determinant_covariant
    assert receipt.diagonal_similarity_determinant_covariant
    assert receipt.permutation_basis_determinant_covariant
    assert receipt.dimensionless_reference_ratios == ('2', '4', '8')
    assert receipt.expected_reference_ratios == ('2', '4', '8')
    assert receipt.reference_scale_law_passed
    assert receipt.operator_sign_abs_reference_ratios_preserved
    assert receipt.odd_dimension_operator_sign_changed
    assert not receipt.overall_operator_sign_phase_resolved
    assert not receipt.log_branch_resolved


def test_all_fp_berezin_kill_controls_are_live(receipt) -> None:
    controls = (
        receipt.wrong_commutator_sign_mismatch_l1,
        receipt.wrong_gauge_trace_coefficient_mismatch_l1,
        receipt.omitted_scalar_gauge_term_mismatch_l1,
        receipt.flipped_scalar_gauge_term_mismatch_l1,
        receipt.wrong_ricci_sign_mismatch_l1,
        receipt.wrong_fp_operator_sign_mismatch_l1,
        receipt.positive_exponent_sign_mismatch_l1,
        receipt.wrong_orientation_mismatch_l1,
        receipt.inverse_determinant_confusion_mismatch_l1,
        receipt.wrong_inverse_ghost_weight_mismatch,
        receipt.half_ghost_weight_mismatch,
        receipt.doubled_ghost_multiplicity_mismatch,
    )
    assert all(Fraction(value) > 0 for value in controls)
    assert receipt.gauge_parameter_rescaling_residual_l1 == '0'
    assert receipt.gauge_parameter_rescaling_covariant
    assert receipt.generic_laplacian_ricci_scalar_terms_live
    assert receipt.zero_vector_limits_all_passed
    assert receipt.flat_curvature_limit_passed


def test_relative_weight_is_a_declared_finite_convention(receipt) -> None:
    assert receipt.ghost_effective_action_exponent == '-1'
    assert receipt.real_boson_effective_action_exponent == '1/2'
    assert receipt.relative_ghost_weight == '-2'
    assert receipt.relative_ghost_weight_computed
    assert receipt.euclidean_real_boson_gaussian_assumed
    assert not receipt.ghost_minus_two_derivation_source_explicit
    assert not receipt.action_prefactor_derived


def test_source_and_claim_contract_fail_closed() -> None:
    contract = fp_berezin_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'fixture_dimensions': (4,)},
        {'frame_convention': 'Lorentzian'},
        {'fp_formula': 'Delta_FP=delta_chi'},
        {'berezin_variable_order': 'c_then_barc'},
        {'reference_scale': Fraction(1)},
        {'linearized_background_split_assumed': False},
        {'fp_derivation_source_explicit': True},
        {'grassmann_measure_source_explicit': True},
        {'ghost_minus_two_derivation_source_explicit': True},
        {'action_prefactor_derived': True},
        {'overall_operator_sign_phase_resolved': True},
        {'zero_mode_sector_resolved': True},
        {'functional_determinant_computed': True},
        {'brst_bv_measure_proved': True},
        {'quantum_hda_m2_proved': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_dimensions_and_scope_remain_bounded(receipt) -> None:
    assert receipt.primitive_dimension_basis == (
        'GaugeParameter',
        'Derivative',
        'Curvature',
        'ScalarGradient',
    )
    assert receipt.primitive_length_dimensions == (1, -1, -2, -1)
    assert derive_fp_length_dimensions() == (
        1,
        -1,
        -1,
        -1,
        -2,
        0,
        0,
        0,
    )
    assert receipt.quantity_length_dimensions == (
        1,
        -1,
        -1,
        -1,
        -2,
        0,
        0,
        0,
    )
    assert receipt.corrupted_gradient_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.corrupted_derivative_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.dimension_gate_passed
    assert receipt.derivation_status == (
        'finite_linear_fp_variation_and_berezin_weight_only'
    )
    bounded_false = (
        receipt.fp_derivation_source_explicit,
        receipt.grassmann_measure_source_explicit,
        receipt.ghost_minus_two_derivation_source_explicit,
        receipt.action_prefactor_derived,
        receipt.overall_operator_sign_phase_resolved,
        receipt.global_fp_operator_completed,
        receipt.boundary_conditions_completed,
        receipt.zero_mode_sector_resolved,
        receipt.functional_measure_derived,
        receipt.functional_determinant_computed,
        receipt.log_branch_resolved,
        receipt.brst_bv_measure_proved,
        receipt.heat_kernel_derived,
        receipt.loop_integral_evaluated,
        receipt.renormalization_proof,
        receipt.continuum_st_qme_proved,
        receipt.local_covariance_proved,
        receipt.in_in_ctp_completed,
        receipt.positive_physical_hilbert_proved,
        receipt.quantum_hda_m2_proved,
    )
    assert not any(bounded_false)
