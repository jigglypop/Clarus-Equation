from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    audit_curvature,
    kulkarni_nomizu_curvature,
    ricci_from_curvature,
    symmetric_fixture,
)
from examples.physics.qft_reference_flrw_sym2_curvature_trace import (
    SOURCE_TRANSCRIPTION_SHA256,
    add_curvatures,
    bundle_curvature_matrix,
    bundle_curvature_squared_trace,
    bundle_identity_matrix,
    bundle_rank,
    curvature_squared,
    evaluate_sym2_curvature_trace_gate,
    raw_basis_roundtrip_passed,
    raw_symmetric_basis,
    source_payload_sha256,
    sym2_curvature_trace_contract,
    symmetric_pairs,
    validate_contract,
    weyl_fixture,
    zero_matrix,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_sym2_curvature_trace_gate()


def test_declared_sym2_bundle_traces_are_exact(receipt) -> None:
    assert receipt.fixture_dimensions == (3, 4, 5)
    assert receipt.weyl_fixture_dimensions == (4, 5)
    assert receipt.bundle_ranks == (7, 11, 16)
    assert receipt.generic_fixture_count == 3
    assert receipt.weyl_added_fixture_count == 2
    assert receipt.exact_trace_component_count == 8
    assert receipt.exact_trace_residuals == ('0',) * 8
    assert receipt.exact_trace_contractions_all_passed
    assert receipt.scalar_curvature_blocks_all_zero
    assert receipt.declared_finite_sym2_curvature_trace_gate_passed


def test_raw_basis_and_scalar_identity_are_constructed() -> None:
    for dimension, expected_rank in ((3, 7), (4, 11), (5, 16)):
        assert raw_basis_roundtrip_passed(dimension)
        assert len(symmetric_pairs(dimension)) + 1 == expected_rank
        identity = bundle_identity_matrix(dimension)
        assert len(identity) == expected_rank
        assert sum(
            (identity[index][index] for index in range(expected_rank)),
            Fraction(0),
        ) == expected_rank

    off_diagonal = raw_symmetric_basis(3, (0, 2))
    assert off_diagonal[0][2] == 1
    assert off_diagonal[2][0] == 1
    assert sum(sum(row, Fraction(0)) for row in off_diagonal) == 2


def test_weyl_fixture_is_nonzero_ricci_flat_and_trace_sensitive() -> None:
    for dimension in (4, 5):
        weyl = weyl_fixture(dimension)
        zero = zero_matrix(dimension)
        ricci = ricci_from_curvature(weyl)
        assert curvature_squared(weyl) > 0
        assert ricci == zero
        assert audit_curvature(weyl, ricci, zero).passed

        generic = kulkarni_nomizu_curvature(symmetric_fixture(dimension))
        combined = add_curvatures(generic, weyl)
        assert bundle_curvature_squared_trace(combined) == (
            -(dimension + 2) * curvature_squared(combined)
        )
        assert bundle_curvature_squared_trace(generic) != (
            -(dimension + 2) * curvature_squared(combined)
        )


def test_scalar_curvature_block_is_explicitly_zero() -> None:
    dimension = 4
    curvature = kulkarni_nomizu_curvature(symmetric_fixture(dimension))
    matrix = bundle_curvature_matrix(curvature, 0, 1)
    scalar_index = bundle_rank(dimension) - 1
    assert all(value == 0 for value in matrix[scalar_index])
    assert all(row[scalar_index] == 0 for row in matrix)


def test_all_negative_controls_are_nonzero(receipt) -> None:
    assert receipt.curvature_audit_count == 7
    assert receipt.curvature_audits_all_passed
    assert receipt.weyl_fixtures_nonzero_and_ricci_flat
    assert receipt.corrupted_curvature_rejected
    assert Fraction(receipt.missing_scalar_identity_mismatch_l1) > 0
    assert Fraction(receipt.off_diagonal_normalization_mismatch_l1) > 0
    assert Fraction(receipt.half_action_mismatch_l1) > 0
    assert Fraction(receipt.omitted_second_slot_mismatch_l1) > 0
    assert Fraction(receipt.wrong_relative_slot_sign_mismatch_l1) > 0
    assert Fraction(receipt.wrong_curvature_index_mismatch_l1) > 0
    assert Fraction(receipt.frobenius_vs_matrix_trace_mismatch_l1) > 0
    assert Fraction(receipt.dropped_weyl_mismatch_l1) > 0
    assert Fraction(receipt.omitted_generic_fixture_magnitude) > 0
    assert Fraction(receipt.w_linear_sign_flip_squared_trace_residual) == 0
    assert not receipt.w_linear_sign_determined


def test_source_and_claim_contract_fails_closed() -> None:
    contract = sym2_curvature_trace_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256

    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'fixture_dimensions': (4,)},
        {'weyl_fixture_dimensions': (4,)},
        {'frame_convention': 'Lorentzian'},
        {'basis_formula': 'orthonormalized'},
        {'curvature_action_formula': 'half-action'},
        {'w_linear_sign_determined': True},
        {'source_lorentzian_sign_extended': True},
        {'eq22_trY_derived': True},
        {'eq18_operator_derived': True},
        {'functional_determinant_derived': True},
        {'independent_source_artifact_authenticated': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_dimensions_and_scope_remain_bounded(receipt) -> None:
    assert receipt.primitive_length_dimensions == (0, -2)
    assert receipt.trace_dimension_basis == (
        'IdentityTrace',
        'RiemannSq',
        'BundleCurvatureSqTrace',
    )
    assert receipt.trace_length_dimensions == (0, -4, -4)
    assert receipt.corrupted_trace_length_dimensions == (0, -2, -2)
    assert receipt.dimension_gate_passed
    assert receipt.finite_sym2_bundle_curvature_traces_computed
    assert receipt.derivation_status == 'finite_sym2_bundle_curvature_trace_only'

    bounded_false = (
        receipt.source_lorentzian_sign_extended,
        receipt.background_eom_used,
        receipt.eq22_trY_derived,
        receipt.eq22_trY2_derived,
        receipt.eq18_operator_derived,
        receipt.gauge_fixing_derived,
        receipt.functional_determinant_derived,
        receipt.heat_kernel_trace_derived,
        receipt.fp_determinant_derived,
        receipt.ghost_weight_derived,
        receipt.loop_integral_evaluated,
        receipt.regularization_scheme_implemented,
        receipt.finite_boundary_completed,
        receipt.evanescent_terms_controlled,
        receipt.independent_source_artifact_authenticated,
        receipt.renormalization_proof,
        receipt.continuum_st_qme_proved,
        receipt.local_covariance_proved,
        receipt.in_in_ctp_completed,
        receipt.positive_physical_hilbert_proved,
        receipt.quantum_hda_m2_proved,
    )
    assert not any(bounded_false)
