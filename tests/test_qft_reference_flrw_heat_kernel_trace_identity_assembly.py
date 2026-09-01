from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_heat_kernel_trace_identity_assembly import (
    RationalPolynomial,
    SOURCE_TRANSCRIPTION_SHA256,
    eq22_trace_inputs,
    evaluate_trace_identity_assembly_gate,
    ghost_trace_inputs,
    source_payload_sha256,
    trace_identity_assembly_contract,
    universal_eq19_assembly,
    validate_contract,
)


def test_eq19_and_supplied_traces_symbolically_reproduce_eq23_and_eq27() -> None:
    receipt = evaluate_trace_identity_assembly_gate()

    assert receipt.eq23_symbolic_cross_residuals == ('0',) * 7
    assert receipt.eq27_symbolic_cross_residuals == ('0',) * 7
    assert receipt.eq23_symbolic_identity_passed
    assert receipt.eq27_symbolic_identity_passed
    assert receipt.exact_spot_component_count == 42
    assert receipt.exact_spot_checks_all_passed
    assert receipt.verification_dimensions == (3, 4, 5)
    assert receipt.n_two_pole_rejected
    assert receipt.declared_source_trace_identity_assembly_gate_passed


def test_rational_polynomial_equivalence_and_pole_are_exact() -> None:
    n = RationalPolynomial.variable()
    left = (n**2 - 4) / (n - 2)
    right = n + 2

    assert left.equivalent(right)
    assert left.cross_residual(right) == (Fraction(0),)
    assert left.evaluate(3) == 5
    with pytest.raises(ZeroDivisionError):
        left.evaluate(2)
    with pytest.raises(ZeroDivisionError):
        _ = left / 0


def test_source_trace_contract_fails_closed() -> None:
    contract = trace_identity_assembly_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256

    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'ordered_basis': tuple(reversed(contract.ordered_basis))},
        {'verification_dimensions': (4,)},
        {'downstream_ghost_weight': -1},
        {'source_bulk_total_derivatives_omitted': False},
        {'gauss_bonnet_applied_in_this_gate': True},
        {'eq22_trace_tensors_derived': True},
        {'ghost_determinant_derived': True},
        {'finite_boundary_completed': True},
        {'independent_source_artifact_authenticated': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_symbolic_negative_controls_are_nonzero() -> None:
    receipt = evaluate_trace_identity_assembly_gate()

    assert Fraction(receipt.n_four_only_eq23_impostor_mismatch_l1) > 0
    assert Fraction(receipt.n_four_only_eq27_impostor_mismatch_l1) > 0
    assert receipt.missing_r_potential_nonzero_component_count == 2
    assert receipt.wrong_eq22_field_strength_sign_nonzero_component_count == 1
    assert receipt.missing_scalar_trace_identity_nonzero_component_count == 3
    assert receipt.omitted_eq22_p_nonzero_component_count == 1
    assert receipt.omitted_eq22_r_x_nonzero_component_count == 1
    assert receipt.wrong_ghost_p_sign_nonzero_component_count == 1
    assert receipt.wrong_ghost_field_strength_sign_nonzero_component_count == 1
    assert receipt.permuted_curvature_basis_nonzero_component_count == 2


def test_dimensions_and_claim_ceiling_are_fail_closed() -> None:
    receipt = evaluate_trace_identity_assembly_gate()

    assert receipt.monomial_length_dimensions == (-4,) * 7
    assert receipt.corrupted_x_dimension_vector == (
        -4, -4, -4, -3, -3, -2, -4
    )
    assert receipt.universal_contribution_length_dimensions == (
        -4, -4, -4, -4
    )
    assert receipt.corrupted_potential_contribution_dimensions == (
        -4, -2, -3, -2
    )
    assert receipt.dimension_gate_passed
    assert receipt.downstream_ghost_weight == -2
    assert receipt.source_bulk_total_derivatives_omitted
    assert not receipt.gauss_bonnet_applied_in_this_gate
    assert receipt.derivation_status == 'source_trace_identity_assembly_only'
    assert not receipt.universal_heat_kernel_formula_derived
    assert not receipt.eq22_trace_tensors_derived
    assert not receipt.ghost_determinant_derived
    assert not receipt.ghost_weight_derived
    assert not receipt.loop_integral_evaluated
    assert not receipt.regularization_scheme_implemented
    assert not receipt.finite_boundary_completed
    assert not receipt.evanescent_terms_controlled
    assert not receipt.independent_source_artifact_authenticated
    assert not receipt.renormalization_proof
    assert not receipt.continuum_st_qme_proved
    assert not receipt.local_covariance_proved
    assert not receipt.in_in_ctp_completed
    assert not receipt.positive_physical_hilbert_proved
    assert not receipt.quantum_hda_m2_proved


def test_trace_input_vectors_keep_the_declared_basis() -> None:
    assert len(eq22_trace_inputs().potential_squared) == 7
    assert len(eq22_trace_inputs().field_strength_squared) == 7
    assert len(ghost_trace_inputs().potential_squared) == 7
    assert len(universal_eq19_assembly(eq22_trace_inputs())) == 7
