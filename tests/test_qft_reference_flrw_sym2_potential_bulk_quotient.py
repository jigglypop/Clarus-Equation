from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_heat_kernel_trace_identity_assembly import (
    eq22_trace_inputs,
)
from examples.physics.qft_reference_flrw_sym2_potential_bulk_quotient import (
    SOURCE_TRANSCRIPTION_SHA256,
    PotentialInvariants,
    bulk_divergence,
    evaluate_sym2_potential_bulk_gate,
    flat_hessian_fixture,
    invert_matrix,
    matrix_identity_residual_l1,
    potential_invariants,
    potential_trace_values,
    raw_dewitt_metric,
    raw_potential_matrix,
    source_eq22_bulk_potential_squared,
    source_payload_sha256,
    sym2_potential_bulk_contract,
    trace_potential_target,
    validate_contract,
    zero_curvature,
)
from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    zero_vector,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_sym2_potential_bulk_gate()


def test_declared_raw_and_bulk_trace_relations_are_exact(receipt) -> None:
    assert receipt.bundle_ranks == (7, 11, 16)
    assert receipt.fixture_count == 9
    assert receipt.generic_vector_fixture_count == 3
    assert receipt.zero_vector_fixture_count == 3
    assert receipt.weyl_added_fixture_count == 2
    assert receipt.flat_hessian_fixture_count == 1
    assert receipt.exact_trace_component_count == 18
    assert receipt.exact_trace_residuals == ('0',) * 18
    assert receipt.exact_trace_relations_all_passed
    assert receipt.declared_finite_sym2_potential_bulk_gate_passed


def test_dewitt_metric_inverse_and_n2_pole_are_explicit(receipt) -> None:
    assert receipt.metric_inverse_residuals_l1 == ('0', '0', '0')
    assert receipt.metric_inverse_all_passed
    assert receipt.n2_pole_rejected
    for dimension in (3, 4, 5):
        metric = raw_dewitt_metric(dimension)
        inverse = invert_matrix(metric)
        assert matrix_identity_residual_l1(inverse, metric) == 0
    with pytest.raises(ValueError):
        raw_dewitt_metric(2)


def test_flat_hessian_separates_pointwise_trace_from_bulk() -> None:
    curvature = zero_curvature(3)
    vector = zero_vector(3)
    hessian = flat_hessian_fixture()
    invariants = potential_invariants(curvature, vector, hessian)
    values = potential_trace_values(curvature, vector, hessian)
    bulk = source_eq22_bulk_potential_squared(3, invariants)

    assert invariants.hessian_squared == 21
    assert invariants.box_phi == 7
    assert bulk_divergence(invariants) == -28
    assert values.trace_potential == 0
    assert values.trace_potential_squared_raw == -14
    assert bulk == 98
    assert values.trace_potential_squared_raw - bulk == -112
    assert values.trace_potential_squared_raw - bulk == (
        4 * bulk_divergence(invariants)
    )


def test_source_eq22_coefficients_match_locked_symbolic_inputs() -> None:
    source = eq22_trace_inputs()
    for dimension in (3, 4, 5):
        invariants = PotentialInvariants(
            riemann_squared=Fraction(2),
            ricci_squared=Fraction(3),
            ricci_scalar=Fraction(5),
            scalar_gradient_squared=Fraction(7),
            ricci_gradient_contraction=Fraction(11),
            hessian_squared=Fraction(13),
            box_phi=Fraction(17),
        )
        expected_trace = (
            source.potential_r.evaluate(dimension)
            * invariants.ricci_scalar
            + source.potential_x.evaluate(dimension)
            * invariants.scalar_gradient_squared
        )
        basis_values = (
            invariants.riemann_squared,
            invariants.ricci_squared,
            invariants.ricci_scalar**2,
            invariants.ricci_gradient_contraction,
            invariants.ricci_scalar
            * invariants.scalar_gradient_squared,
            invariants.scalar_gradient_squared**2,
            invariants.box_phi**2,
        )
        expected_squared = sum(
            (
                coefficient.evaluate(dimension) * value
                for coefficient, value in zip(
                    source.potential_squared,
                    basis_values,
                    strict=True,
                )
            ),
            Fraction(0),
        )
        assert trace_potential_target(dimension, invariants) == expected_trace
        assert (
            source_eq22_bulk_potential_squared(dimension, invariants)
            == expected_squared
        )


def test_raw_potential_is_covariantly_symmetric(receipt) -> None:
    curvature = zero_curvature(3)
    vector = zero_vector(3)
    potential = raw_potential_matrix(
        curvature,
        vector,
        flat_hessian_fixture(),
    )
    assert all(
        potential[row][column] == potential[column][row]
        for row in range(len(potential))
        for column in range(len(potential))
    )
    assert receipt.potential_matrices_symmetric
    assert receipt.curvature_audit_count == 7
    assert receipt.curvature_audits_all_passed
    assert receipt.weyl_fixtures_nonzero_and_ricci_flat
    assert receipt.generic_invariants_live
    assert receipt.zero_vector_limits_all_passed


def test_all_kill_controls_are_nonzero_and_sign_blindness_is_recorded(
    receipt,
) -> None:
    controls = (
        receipt.wrong_dewitt_metric_mismatch_l1,
        receipt.euclidean_raw_metric_mismatch_l1,
        receipt.off_diagonal_basis_mismatch_l1,
        receipt.corrupted_yhh_component_mismatch_l1,
        receipt.omitted_mixed_blocks_mismatch_l1,
        receipt.wrong_relative_mixed_sign_mismatch_l1,
        receipt.wrong_hessian_trace_sign_mismatch_l1,
        receipt.omitted_scalar_block_mismatch_l1,
        receipt.trace_square_confusion_mismatch_l1,
        receipt.wrong_ricci_divergence_sign_mismatch_l1,
        receipt.wrong_quotient_coefficient_mismatch_l1,
        receipt.forced_pointwise_identity_mismatch_l1,
        receipt.dropped_weyl_mismatch_l1,
        receipt.n4_coefficient_copy_mismatch_l1,
    )
    assert all(Fraction(value) > 0 for value in controls)
    assert receipt.flat_divergence == '-28'
    assert receipt.flat_raw_minus_bulk == '-112'
    assert (
        Fraction(
            receipt.simultaneous_mixed_sign_flip_squared_trace_residual
        )
        == 0
    )
    assert not receipt.mixed_linear_sign_determined


def test_source_and_claim_contract_fails_closed() -> None:
    contract = sym2_potential_bulk_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'fixture_dimensions': (4,)},
        {'frame_convention': 'Lorentzian'},
        {'dewitt_formula': 'identity'},
        {'potential_formula': 'supplied traces only'},
        {'bulk_quotient_formula': 'pointwise equality'},
        {'mixed_linear_sign_determined': True},
        {'source_eq22_pointwise_identity_proved': True},
        {'integration_by_parts_source_explicit': True},
        {'finite_boundary_completed': True},
        {'eq18_operator_derived': True},
        {'heat_kernel_trace_derived': True},
        {'independent_source_artifact_authenticated': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_dimensions_and_scope_remain_bounded(receipt) -> None:
    assert receipt.primitive_length_dimensions == (0, -2, -1, -2)
    assert receipt.quantity_length_dimensions == (0, -2, -2) + (-4,) * 11
    assert receipt.corrupted_gradient_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.corrupted_hessian_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.dimension_gate_passed
    assert receipt.raw_potential_traces_computed
    assert receipt.bulk_quotient_applied
    assert receipt.derivation_status == (
        'finite_sym2_potential_bulk_quotient_only'
    )
    bounded_false = (
        receipt.source_eq22_pointwise_identity_proved,
        receipt.integration_by_parts_source_explicit,
        receipt.source_lorentzian_sign_extended,
        receipt.background_eom_used,
        receipt.finite_boundary_completed,
        receipt.endpoint_terms_computed,
        receipt.eq18_operator_derived,
        receipt.gauge_fixing_derived,
        receipt.functional_determinant_derived,
        receipt.heat_kernel_trace_derived,
        receipt.fp_determinant_derived,
        receipt.ghost_weight_derived,
        receipt.loop_integral_evaluated,
        receipt.regularization_scheme_implemented,
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
