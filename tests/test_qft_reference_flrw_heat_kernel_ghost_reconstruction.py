from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_heat_kernel_ghost_reconstruction import (
    EXPECTED_EQ23_AT_FOUR,
    EXPECTED_EQ27_AT_FOUR,
    EXPECTED_RAW_AT_FOUR,
    SOURCE_TRANSCRIPTION_SHA256,
    combine_ghost_weight,
    derive_monomial_length_dimensions,
    equation_23_coefficients,
    equation_27_ghost_coefficients,
    evaluate_heat_kernel_ghost_reconstruction_gate,
    four_dimensional_bulk_gb_quotient,
    heat_kernel_ghost_contract,
    source_payload_sha256,
    validate_contract,
)


def test_eq23_and_eq27_reconstruct_eq28_exactly() -> None:
    receipt = evaluate_heat_kernel_ghost_reconstruction_gate()

    assert receipt.equation_23_exact_vector == (
        '191/180', '-551/180', '119/72', '-2', '-1/6', '2', '1'
    )
    assert receipt.equation_27_exact_vector == (
        '-11/180', '43/90', '2/9', '-1', '-1/6', '1/2', '0'
    )
    assert receipt.raw_exact_vector == (
        '71/60', '-241/60', '29/24', '0', '1/6', '1', '1'
    )
    assert receipt.reduced_exact_vector_with_p_slot == (
        '43/60', '1/40', '0', '1/6', '1', '1'
    )
    assert receipt.source_equation_28_vector_with_p_slot == (
        '43/60', '1/40', '0', '1/6', '1', '1'
    )
    assert receipt.p_term_cancels_only_after_ghost_subtraction
    assert receipt.declared_source_coefficient_assembly_gate_passed


def test_source_contract_and_local_transcription_fail_closed() -> None:
    contract = heat_kernel_ghost_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256

    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'html_internal_heading': contract.source_metadata_title},
        {'source_transcription_sha256': '0' * 64},
        {'ghost_weight': -1},
        {'spacetime_dimension': 5},
        {'gauss_bonnet_bulk_quotient_used': False},
        {'gauss_bonnet_pointwise_identity_claimed': True},
        {'heat_kernel_trace_derived': True},
        {'independent_source_artifact_authenticated': True},
        {'finite_boundary_completed': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_n_dependent_formulae_and_dimension_controls_are_nonvacuous() -> None:
    assert equation_23_coefficients(4) == EXPECTED_EQ23_AT_FOUR
    assert equation_27_ghost_coefficients(4) == EXPECTED_EQ27_AT_FOUR
    assert combine_ghost_weight(
        EXPECTED_EQ23_AT_FOUR, EXPECTED_EQ27_AT_FOUR, -2
    ) == EXPECTED_RAW_AT_FOUR
    with pytest.raises(ValueError):
        equation_23_coefficients(2)
    with pytest.raises(ValueError):
        four_dimensional_bulk_gb_quotient(EXPECTED_RAW_AT_FOUR, spacetime_dimension=5)

    assert derive_monomial_length_dimensions() == (-4,) * 7
    assert derive_monomial_length_dimensions(
        {'ScalarGradientSquared': -1}
    ) == (-4, -4, -4, -3, -3, -2, -4)


def test_wrong_ghost_and_gauss_bonnet_paths_are_detected() -> None:
    receipt = evaluate_heat_kernel_ghost_reconstruction_gate()

    assert Fraction(receipt.wrong_plus_two_ghost_mismatch_l1) > 0
    assert Fraction(receipt.wrong_minus_one_ghost_mismatch_l1) > 0
    assert Fraction(receipt.omitted_ghost_mismatch_l1) > 0
    assert Fraction(receipt.premature_p_deletion_input_mismatch_l1) == 3
    assert Fraction(receipt.wrong_gauss_bonnet_sign_mismatch_l1) > 0
    assert Fraction(receipt.raw_equation_28_confusion_mismatch_l1) > 0
    assert Fraction(receipt.permuted_eq23_basis_mismatch_l1) > 0
    assert Fraction(receipt.omitted_r_squared_mismatch_l1) == Fraction(1, 40)
    assert Fraction(receipt.omitted_r_x_mismatch_l1) == Fraction(1, 6)
    assert Fraction(receipt.omitted_x_squared_mismatch_l1) == 1
    assert Fraction(receipt.dimension_five_raw_mismatch_l1) > 0
    assert receipt.bulk_representative_sample_count == 3
    assert receipt.bulk_representative_samples_all_exact
    assert Fraction(receipt.broken_gb_representative_residual) != 0


def test_claim_ceiling_remains_fail_closed() -> None:
    receipt = evaluate_heat_kernel_ghost_reconstruction_gate()

    assert receipt.local_transcription_lock_passed
    assert not receipt.independent_source_artifact_authenticated
    assert receipt.gauss_bonnet_bulk_quotient_used
    assert not receipt.gauss_bonnet_pointwise_identity_claimed
    assert receipt.derivation_status == 'source_coefficient_assembly_only'
    assert not receipt.heat_kernel_trace_derived
    assert not receipt.ghost_determinant_derived
    assert not receipt.loop_integral_evaluated
    assert not receipt.regularization_scheme_implemented
    assert not receipt.finite_boundary_completed
    assert not receipt.evanescent_terms_controlled
    assert not receipt.renormalization_proof
    assert not receipt.continuum_st_qme_proved
    assert not receipt.local_covariance_proved
    assert not receipt.in_in_ctp_completed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved
