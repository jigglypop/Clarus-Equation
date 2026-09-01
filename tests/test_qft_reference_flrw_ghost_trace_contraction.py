from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    SOURCE_TRANSCRIPTION_SHA256,
    audit_curvature,
    evaluate_ghost_trace_contraction_gate,
    field_strength_frobenius,
    field_strength_matrix_trace,
    ghost_invariants,
    ghost_potential,
    ghost_trace_contract,
    identity_matrix,
    kulkarni_nomizu_curvature,
    ricci_from_curvature,
    source_payload_sha256,
    symmetric_fixture,
    validate_contract,
    vector_fixture,
)


def test_finite_ghost_trace_contractions_are_exact() -> None:
    receipt = evaluate_ghost_trace_contraction_gate()

    assert receipt.fixture_dimensions == (3, 4, 5)
    assert receipt.generic_fixture_count == 3
    assert receipt.zero_vector_fixture_count == 3
    assert receipt.exact_trace_component_count == 24
    assert receipt.exact_trace_residuals == ('0',) * 24
    assert receipt.exact_trace_contractions_all_passed
    assert receipt.zero_vector_limits_all_passed
    assert receipt.generic_invariants_all_nonzero
    assert receipt.declared_finite_ghost_trace_contraction_gate_passed


def test_curvature_symmetries_and_matrix_trace_sign_are_explicit() -> None:
    symmetric = symmetric_fixture(4)
    curvature = kulkarni_nomizu_curvature(symmetric)
    ricci = ricci_from_curvature(curvature)
    invariants = ghost_invariants(curvature, ricci, vector_fixture(4))

    assert audit_curvature(curvature, ricci, symmetric).passed
    assert field_strength_matrix_trace(curvature) == -invariants.riemann_squared
    assert field_strength_frobenius(curvature) == invariants.riemann_squared
    assert field_strength_matrix_trace(
        curvature, linear_sign=-1
    ) == field_strength_matrix_trace(curvature, linear_sign=1)


def test_source_and_claim_contract_fails_closed() -> None:
    contract = ghost_trace_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256

    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'fixture_dimensions': (4,)},
        {'frame_convention': 'Lorentzian'},
        {'source_eq22_reused_for_ghost': True},
        {'source_lorentzian_sign_extended': True},
        {'background_eom_used': True},
        {'ghost_weight_applied': True},
        {'w_linear_sign_determined': True},
        {'fp_operator_derived': True},
        {'fp_determinant_derived': True},
        {'finite_boundary_completed': True},
        {'independent_source_artifact_authenticated': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_all_negative_controls_are_nonzero_and_corruption_is_rejected() -> None:
    receipt = evaluate_ghost_trace_contraction_gate()

    assert receipt.curvature_audit_count == 3
    assert receipt.curvature_audits_all_passed
    assert receipt.corrupted_curvature_rejected
    assert Fraction(receipt.frobenius_vs_matrix_trace_mismatch_l1) > 0
    assert Fraction(receipt.wrong_ricci_contraction_mismatch_l1) > 0
    assert Fraction(receipt.wrong_outer_sign_mismatch_l1) > 0
    assert Fraction(receipt.omitted_outer_product_mismatch_l1) > 0
    assert Fraction(receipt.omitted_cross_term_mismatch_l1) > 0
    assert Fraction(receipt.wrong_w_index_placement_mismatch_l1) > 0
    assert Fraction(receipt.omitted_generic_fixture_magnitude) > 0
    assert Fraction(receipt.rank_deficient_identity_mismatch_l1) > 0
    assert Fraction(receipt.w_linear_sign_flip_squared_trace_residual) == 0
    assert not receipt.w_linear_sign_determined


def test_dimensions_and_scope_remain_bounded() -> None:
    receipt = evaluate_ghost_trace_contraction_gate()

    assert receipt.primitive_operator_length_dimensions == (-2, -2, -2)
    assert receipt.invariant_dimension_basis == (
        'RiemannSq',
        'RicciSq',
        'RicciScalar',
        'ScalarGradientSquared',
        'RicciGradientContraction',
        'ScalarGradientFourth',
    )
    assert receipt.invariant_length_dimensions == (-4, -4, -2, -2, -4, -4)
    assert receipt.corrupted_invariant_length_dimensions == (
        -4, -4, -2, -1, -3, -2
    )
    assert receipt.dimension_gate_passed
    assert receipt.finite_trace_contractions_computed
    assert receipt.derivation_status == 'finite_ghost_trace_contraction_only'
    assert not receipt.source_eq22_reused_for_ghost
    assert not receipt.source_lorentzian_sign_extended
    assert not receipt.background_eom_used
    assert not receipt.ghost_weight_applied
    assert not receipt.fp_operator_derived
    assert not receipt.fp_determinant_derived
    assert not receipt.ghost_weight_derived
    assert not receipt.eq19_heat_kernel_derived
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


def test_potential_matrix_identity_at_one_generic_fixture() -> None:
    curvature = kulkarni_nomizu_curvature(symmetric_fixture(3))
    ricci = ricci_from_curvature(curvature)
    vector = vector_fixture(3)
    potential = ghost_potential(ricci, vector)
    invariants = ghost_invariants(curvature, ricci, vector)

    direct = sum(
        (
            potential[row][column] * potential[column][row]
            for row in range(3)
            for column in range(3)
        ),
        Fraction(0),
    )
    assert direct == (
        invariants.ricci_squared
        - 2 * invariants.ricci_gradient_contraction
        + invariants.scalar_gradient_squared**2
    )


def test_identity_trace_is_constructed_from_matrix_entries() -> None:
    identity = identity_matrix(4)

    assert len(identity) == 4
    assert all(len(row) == 4 for row in identity)
    assert sum(identity[index][index] for index in range(4)) == 4
    assert sum(sum(row, Fraction(0)) for row in identity) == 4
