import pytest

from examples.physics.qft_reference_flrw_cubic_exchange import (
    ResonanceCertificate,
)
from examples.physics.qft_reference_flrw_quartic_contact_gate import (
    reference_state,
)
from examples.physics.qft_reference_flrw_quartic_exchange import (
    finite_time_ordered_double_kernel,
    evaluate_scalar_quartic_exchange_gate,
    select_certified_resonant_matrix_element,
    simpson_ordered_double_kernel,
    simpson_time_ordered_square_kernel_with_stability,
)


def _certificate(
    disposition: str,
    *,
    matrix_element: complex,
) -> ResonanceCertificate:
    resolved = disposition == 'resolved'
    return ResonanceCertificate(
        first_branch=0,
        second_branch=0,
        third_branch=0,
        energy_mismatch_bar=0.0,
        production_matrix_element_real=float(matrix_element.real),
        production_matrix_element_imag=float(matrix_element.imag),
        production_error_envelope=1.0e-6 if resolved else 1.0e-7,
        signal_to_error_ratio=1000.0 if resolved else 0.1,
        linear_second_order_ratio_residual=1.0 if resolved else 0.01,
        richardson_matrix_element_real=(
            float(matrix_element.real) if resolved else 1.0e-13
        ),
        richardson_matrix_element_imag=(
            float(matrix_element.imag) if resolved else 0.0
        ),
        richardson_stability_residual=1.0e-12,
        linear_grid_residual=1.0e-12,
        linear_gauge_residual=1.0e-12,
        null_error_envelope=1.0e-12,
        null_relative_envelope=1.0 if resolved else 1.0e-7,
        disposition=disposition,
        local_exchange_elimination_rejected=resolved,
    )


def test_branch_keyed_resonance_certificate_is_fail_closed() -> None:
    key = (0, 0, 0)
    null_raw = 1.0e-8 + 2.0e-9j
    null_certificate = _certificate('null', matrix_element=null_raw)
    assert (
        select_certified_resonant_matrix_element(
            key,
            null_raw,
            {key: null_certificate},
        )
        == 0.0j
    )

    resolved_raw = 1.0e-3 + 2.0e-4j
    resolved_certificate = _certificate(
        'resolved',
        matrix_element=resolved_raw,
    )
    assert select_certified_resonant_matrix_element(
        key,
        resolved_raw,
        {key: resolved_certificate},
    ) == resolved_raw

    unclassified = _certificate(
        'unclassified',
        matrix_element=null_raw,
    )
    with pytest.raises(ValueError, match='unclassified'):
        select_certified_resonant_matrix_element(
            key,
            null_raw,
            {key: unclassified},
        )
    with pytest.raises(ValueError, match='missing'):
        select_certified_resonant_matrix_element(key, null_raw, {})
    with pytest.raises(ValueError, match='does not match'):
        select_certified_resonant_matrix_element(
            key,
            2.0 * null_raw,
            {key: null_certificate},
        )


def test_ordered_double_kernel_has_finite_resonant_limit() -> None:
    interval = 0.5
    assert abs(
        finite_time_ordered_double_kernel(0.0, 0.0, interval)
        - interval**2 / 2.0
    ) < 1.0e-14
    for later, earlier in ((0.0, 0.0), (0.3, -0.3), (0.2, 0.1)):
        analytic = finite_time_ordered_double_kernel(
            later,
            earlier,
            interval,
        )
        quadrature = simpson_ordered_double_kernel(
            later,
            earlier,
            interval,
            subintervals=512,
        )
        assert abs(analytic - quadrature) < 1.0e-10
        square, square_stability = (
            simpson_time_ordered_square_kernel_with_stability(
            later,
            earlier,
            interval,
            subintervals=128,
            )
        )
        assert abs(analytic - square) < 1.0e-10
        assert square_stability < 1.0e-10


def test_preregistered_quartic_contact_rotating_exchange_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_scalar_quartic_exchange_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        simpson_subintervals=(512, 1024),
    )

    assert len(receipt.branches) == 2
    assert receipt.quartic_contact_gate_passed
    assert receipt.cubic_resonance_classification_gate_passed
    assert receipt.maximum_cubic_step_residual < 2.0e-4
    assert receipt.maximum_cubic_grid_residual < 1.0e-8
    assert receipt.maximum_cubic_gauge_residual < 1.0e-6
    assert receipt.maximum_kernel_quadrature_residual < 1.0e-10
    assert receipt.maximum_kernel_grid_refinement < 1.0e-10
    assert receipt.maximum_time_ordered_square_residual < 1.0e-10
    assert receipt.maximum_time_ordered_square_grid_refinement < 1.0e-10
    assert receipt.maximum_hermiticity_residual < 1.0e-12
    assert receipt.maximum_fock_matrix_element_residual < 1.0e-12
    assert receipt.maximum_finest_exact_normalized_error < 1.0e-4
    assert receipt.maximum_lambda_quarter_scaling_residual < 0.1
    assert receipt.minimum_negative_control_to_numerical_error_ratio > 10.0
    assert receipt.branches[0].resonant_null_channel_count == 1
    assert receipt.branches[0].discarded_raw_resonant_magnitude > 0.0
    assert receipt.branches[0].intermediate_resonance_dispositions == (
        'null',
        'nonresonant',
    )
    assert receipt.declared_rotating_exchange_gate_passed
