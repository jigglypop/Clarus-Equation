from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    SOURCE_TRANSCRIPTION_SHA256,
    RationalBackgroundPoint,
    counterterm_density,
    derive_monomial_length_dimensions,
    eom_ideal_identity,
    evaluate_one_loop_source_reproduction_gate,
    one_loop_source_contract,
    source_coefficient_vector,
    source_payload_sha256,
    validate_contract,
)


def test_v7_equation_28_reproduces_equation_30_exactly() -> None:
    receipt = evaluate_one_loop_source_reproduction_gate()

    assert receipt.source_lock_passed
    assert receipt.source_transcription_sha256 == SOURCE_TRANSCRIPTION_SHA256
    assert receipt.exact_coefficient_vector == ('43/60', '1/40', '1/6', '1', '1')
    assert receipt.monomial_length_dimensions == (-4, -4, -4, -4, -4)
    assert receipt.dimension_gate_passed
    assert receipt.background_eom.scalar_gradient_squared_over_r == '2'
    assert receipt.background_eom.ricci_tensor_squared_over_r_squared == '1'
    assert receipt.background_eom.box_scalar_over_r == '0'
    assert receipt.on_shell_term_contributions == (
        '43/60',
        '1/40',
        '1/3',
        '4',
        '0',
    )
    assert receipt.direct_equation_30_coefficient == '203/40'
    assert receipt.source_equation_30_coefficient == '203/40'
    assert receipt.on_shell_rational_sample_count == 4
    assert receipt.on_shell_samples_all_exact
    assert receipt.off_shell_rational_sample_count == 3
    assert receipt.eom_ideal_identity_all_exact
    assert receipt.declared_one_loop_source_reproduction_gate_passed


def test_source_and_field_content_changes_fail_closed() -> None:
    contract = one_loop_source_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256

    changes = (
        {'source_id': 'arXiv:1706.02622v6'},
        {'source_transcription_sha256': '0' * 64},
        {'quantum_scalar_multiplicity': 0},
        {'scalar_loop_retained': False},
        {'scalar_background_zero_removes_scalar_loop': True},
        {'boundary_counterterm_computed': True},
        {'pure_einstein_coefficients_claimed': True},
        {'coefficients': ('7/10', '1/60', '0', '0', '0')},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_counterterm_dimensions_are_derived_from_primitive_assignments() -> None:
    assert derive_monomial_length_dimensions() == (-4, -4, -4, -4, -4)
    assert derive_monomial_length_dimensions(
        {'ScalarGradientSquared': -1}
    ) == (-4, -4, -3, -2, -4)


def test_eom_ideal_identity_holds_off_shell_as_exact_rationals() -> None:
    contract = one_loop_source_contract()
    coefficients = source_coefficient_vector(contract)
    point = RationalBackgroundPoint(
        ricci_scalar=Fraction(-7, 11),
        ricci_tensor_squared=Fraction(13, 17),
        scalar_gradient_squared=Fraction(19, 23),
        box_scalar=Fraction(-29, 31),
    )
    left, right = eom_ideal_identity(coefficients, point, Fraction(203, 40))

    assert left == right
    assert left != 0
    assert counterterm_density(coefficients, point) == left + Fraction(203, 40) * point.ricci_scalar**2


def test_stale_and_scalar_omission_controls_are_detected() -> None:
    receipt = evaluate_one_loop_source_reproduction_gate()

    assert Fraction(receipt.stale_pure_gravity_vector_mismatch_l1) > 0
    assert Fraction(receipt.equation_28_30_confusion_mismatch_l1) > 0
    assert Fraction(receipt.scalar_background_zero_shortcut_residual) == Fraction(13, 3)
    assert Fraction(receipt.omitted_x_squared_control_residual) == 4
    assert Fraction(receipt.omitted_r_x_control_residual) == Fraction(1, 3)
    assert Fraction(receipt.wrong_eom_substitution_residual) > 0
    assert Fraction(receipt.half_ricci_coefficient_control_residual) == Fraction(43, 120)
    assert Fraction(receipt.linear_box_scalar_control_residual) == 2
    assert receipt.quantum_scalar_multiplicity == 1
    assert receipt.scalar_loop_retained
    assert not receipt.scalar_background_zero_removes_scalar_loop
    assert receipt.gauss_bonnet_identity_used_by_source
    assert not receipt.boundary_counterterm_computed
    assert receipt.derivation_status == 'source_reproduction_only'
    assert not receipt.loop_integral_evaluated
    assert not receipt.heat_kernel_trace_derived
    assert not receipt.ghost_determinant_derived
    assert not receipt.regularization_scheme_implemented
    assert not receipt.independent_feynman_diagram_check
    assert not receipt.renormalization_proof
    assert not receipt.pure_einstein_coefficients_claimed
    assert not receipt.continuum_st_qme_proved
    assert not receipt.in_in_ctp_computed
    assert not receipt.positive_physical_hilbert_computed
    assert not receipt.nonperturbative_m2_passed
