from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_operator_trace_synthesis import (
    SOURCE_TRANSCRIPTION_SHA256,
    derive_synthesis_length_dimensions,
    evaluate_operator_trace_synthesis_gate,
    evaluate_polynomial,
    exact_fit,
    interpolate_polynomial,
    operator_trace_synthesis_contract,
    source_payload_sha256,
    synthesis_fixtures,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_operator_trace_synthesis_gate()


def test_all_raw_designs_are_full_rank_and_audited(receipt) -> None:
    assert receipt.identification_dimensions == (4, 5, 6, 7)
    assert receipt.holdout_dimensions == (8,)
    assert receipt.fixtures_per_dimension == 12
    assert receipt.fixture_count == 60
    assert receipt.curvature_audit_count == 60
    assert receipt.curvature_audits_all_passed
    assert receipt.direct_field_strength_audit_count == 20
    assert receipt.direct_field_strength_audit_residuals == ('0',) * 20
    assert receipt.direct_field_strength_audits_all_passed
    assert receipt.pure_weyl_nonzero_and_ricci_flat
    assert receipt.generic_invariants_live
    assert receipt.design_ranks == (7, 7, 7, 7, 7)
    assert receipt.no_weyl_design_ranks == (6, 6, 6, 6, 6)
    assert receipt.full_rank_identification_passed
    assert receipt.no_weyl_rank_loss_detected
    assert receipt.declared_operator_trace_synthesis_gate_passed


def test_independent_fits_match_source_only_after_fit(receipt) -> None:
    assert receipt.exact_fit_residual_count == 120
    assert receipt.bosonic_fit_residuals == ('0',) * 60
    assert receipt.ghost_fit_residuals == ('0',) * 60
    assert receipt.exact_fits_all_passed
    assert receipt.source_coefficient_component_count == 70
    assert receipt.source_coefficient_residuals == ('0',) * 70
    assert receipt.source_coefficients_all_matched
    assert receipt.source_targets_used_only_after_fit
    assert not receipt.source_eq22_coefficients_used_as_fit_input
    assert not receipt.source_eq23_eq27_used_as_fit_input


def test_n4_raw_ghost_combination_and_gb_match_source(receipt) -> None:
    assert receipt.bosonic_coefficients[0] == (
        '191/180',
        '-551/180',
        '119/72',
        '-2',
        '-1/6',
        '2',
        '1',
    )
    assert receipt.ghost_coefficients[0] == (
        '-11/180',
        '43/90',
        '2/9',
        '-1',
        '-1/6',
        '1/2',
        '0',
    )
    assert receipt.n4_combined_raw_coefficients == (
        '71/60',
        '-241/60',
        '29/24',
        '0',
        '1/6',
        '1',
        '1',
    )
    assert receipt.n4_combined_p_coefficient == '0'
    assert receipt.n4_gb_reduced_coefficients == (
        '43/60',
        '1/40',
        '0',
        '1/6',
        '1',
        '1',
    )
    assert receipt.n4_gb_reduced_coefficients == (
        receipt.source_eq28_with_p_slot
    )
    assert receipt.n4_combination_and_gb_passed
    assert Fraction(receipt.generic_euler_density_mismatch_l1) > 0
    assert receipt.pointwise_gauss_bonnet_rejected


def test_degree_bound_interpolation_has_independent_n8_holdout(
    receipt,
) -> None:
    assert receipt.admitted_lift_degree == 3
    assert receipt.lift_holdout_component_count == 14
    assert receipt.lift_holdout_residuals == ('0',) * 14
    assert receipt.polynomial_lift_holdout_passed
    assert receipt.polynomial_lift_degree_bound_admitted
    assert not receipt.all_n_symbolic_identity_proved

    polynomial = interpolate_polynomial(
        (4, 5, 6, 7),
        tuple(Fraction(value) for value in (1, 4, 9, 16)),
        3,
    )
    assert evaluate_polynomial(polynomial, 8) == 25


def test_exact_fit_is_independent_of_source_oracles() -> None:
    rows = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(2), Fraction(3)),
    )
    values = (Fraction(5), Fraction(7), Fraction(31))
    fit = exact_fit(rows, values)
    assert fit.rank == 2
    assert fit.coefficients == (Fraction(5), Fraction(7))
    assert fit.residuals == (Fraction(0),) * 3


def test_all_synthesis_kill_controls_are_live(receipt) -> None:
    controls = (
        receipt.omitted_bulk_quotient_residual_l1,
        receipt.wrong_plus_bulk_quotient_residual_l1,
        receipt.wrong_field_strength_sign_mismatch_l1,
        receipt.omitted_scalar_identity_mismatch_l1,
        receipt.omitted_r_potential_term_mismatch_l1,
        receipt.wrong_ghost_outer_sign_mismatch_l1,
        receipt.coefficient_permutation_mismatch_l1,
        receipt.corrupted_raw_density_residual_l1,
        receipt.wrong_ghost_weight_plus_two_mismatch_l1,
        receipt.wrong_ghost_weight_minus_one_mismatch_l1,
        receipt.wrong_ghost_weight_zero_mismatch_l1,
        receipt.premature_bosonic_p_deletion_mismatch_l1,
        receipt.n4_copy_to_n8_mismatch_l1,
    )
    assert all(Fraction(value) > 0 for value in controls)
    assert receipt.no_weyl_rank_loss_detected
    assert receipt.pointwise_gauss_bonnet_rejected
    assert receipt.n3_full_rank_identification_rejected
    assert receipt.n2_pole_rejected
    with pytest.raises(ValueError):
        synthesis_fixtures(3)


def test_source_upstream_and_claim_contract_fail_closed() -> None:
    contract = operator_trace_synthesis_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'raw_basis': tuple(reversed(contract.raw_basis))},
        {'identification_dimensions': (4,)},
        {'holdout_dimensions': (7,)},
        {'fixture_types': contract.fixture_types[:-1]},
        {'ghost_weight': -1},
        {'admitted_lift_degree': 4},
        {'source_targets_used_only_after_fit': False},
        {'source_eq22_coefficients_used_as_fit_input': True},
        {'source_eq23_eq27_used_as_fit_input': True},
        {'eq19_theorem_independently_derived': True},
        {'all_n_symbolic_identity_proved': True},
        {'gauss_bonnet_pointwise_zero_claimed': True},
        {'finite_boundary_completed': True},
        {'functional_determinant_computed': True},
        {'renormalization_proof': True},
        {'quantum_hda_m2_proved': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_dimensions_and_scope_remain_bounded(receipt) -> None:
    assert receipt.primitive_dimension_basis == (
        'Curvature',
        'ScalarGradient',
        'ScalarHessian',
        'Coefficient',
    )
    assert receipt.primitive_length_dimensions == (-2, -1, -2, 0)
    assert derive_synthesis_length_dimensions() == (-4,) * 9 + (0,)
    assert receipt.quantity_length_dimensions == (-4,) * 9 + (0,)
    assert receipt.corrupted_gradient_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.corrupted_curvature_length_dimensions != (
        receipt.quantity_length_dimensions
    )
    assert receipt.dimension_gate_passed
    assert receipt.eq19_source_supplied_theorem
    assert receipt.derivation_status == (
        'finite_raw_trace_to_source_coefficient_synthesis_only'
    )
    bounded_false = (
        receipt.eq19_theorem_independently_derived,
        receipt.all_n_symbolic_identity_proved,
        receipt.bulk_divergence_pointwise_zero_claimed,
        receipt.gauss_bonnet_pointwise_zero_claimed,
        receipt.finite_boundary_completed,
        receipt.global_minimal_operator_derived,
        receipt.functional_measure_derived,
        receipt.functional_determinant_computed,
        receipt.heat_kernel_proper_time_integral_derived,
        receipt.loop_integral_evaluated,
        receipt.regularization_scheme_implemented,
        receipt.evanescent_terms_controlled,
        receipt.renormalization_proof,
        receipt.continuum_st_qme_proved,
        receipt.local_covariance_proved,
        receipt.in_in_ctp_completed,
        receipt.positive_physical_hilbert_proved,
        receipt.quantum_hda_m2_proved,
    )
    assert not any(bounded_false)
