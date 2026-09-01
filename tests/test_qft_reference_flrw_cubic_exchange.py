from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_cubic_exchange import (
    classify_resonant_channel,
    evaluate_scalar_cubic_exchange_gate,
)
from examples.physics.qft_reference_flrw_cubic_dyson import (
    finite_time_exponential_kernel,
    simpson_exponential_kernel,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    state = ReferenceFlrwState(
        n=0.0,
        h=expanding_h_from_constraint(u=u, b=b, parameters=parameters),
        clock=0.0,
        u=u,
        b=b,
    )
    return state, parameters


def test_resonant_finite_time_kernel_is_finite_while_unit_denominator_doubles() -> None:
    interval = 0.5
    assert abs(finite_time_exponential_kernel(0.0, interval) - interval) < 1.0e-14
    assert abs(
        simpson_exponential_kernel(0.0, interval, subintervals=4096)
        - interval
    ) < 1.0e-14
    magnitudes = [abs(1.0 / (1j * value)) for value in (1.0e-2, 5.0e-3, 2.5e-3)]
    assert abs(magnitudes[1] / magnitudes[0] - 2.0) < 1.0e-12
    assert abs(magnitudes[2] / magnitudes[1] - 2.0) < 1.0e-12


def test_resonance_classifier_separates_null_resolved_and_unclassified() -> None:
    assert classify_resonant_channel(
        production_matrix_element=1.0e-8 + 0.0j,
        production_error_envelope=1.0e-7,
        linear_second_order_ratio_residual=0.01,
        richardson_matrix_element=1.0e-13 + 0.0j,
        null_error_envelope=1.0e-12,
        null_relative_envelope=1.0e-7,
    ) == 'null'
    assert classify_resonant_channel(
        production_matrix_element=1.0e-3 + 0.0j,
        production_error_envelope=1.0e-6,
        linear_second_order_ratio_residual=1.0,
        richardson_matrix_element=1.0e-3 + 0.0j,
        null_error_envelope=1.0e-12,
        null_relative_envelope=1.0,
    ) == 'resolved'
    assert classify_resonant_channel(
        production_matrix_element=1.0e-8 + 0.0j,
        production_error_envelope=1.0e-8,
        linear_second_order_ratio_residual=0.01,
        richardson_matrix_element=1.0e-13 + 0.0j,
        null_error_envelope=1.0e-12,
        null_relative_envelope=1.0e-7,
    ) == 'unclassified'


def test_preregistered_cubic_exchange_resonance_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_scalar_cubic_exchange_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
    )

    assert receipt.channel_count == 8
    assert receipt.resonant_channel_count > 0
    assert receipt.nonresonant_channel_count > 0
    assert len(receipt.resonance_certificates) == receipt.resonant_channel_count
    assert tuple(item.key for item in receipt.resonance_certificates) == tuple(
        sorted(item.key for item in receipt.resonance_certificates)
    )
    assert all(
        item.disposition == 'null'
        for item in receipt.resonance_certificates
    )
    assert receipt.resonant_signal_to_error_ratio < 1.0
    assert receipt.resonant_linear_second_order_ratio_residual < 0.15
    assert (
        receipt.resonant_richardson_matrix_element_magnitude
        <= receipt.resonant_null_error_envelope
    )
    assert receipt.resonant_null_relative_envelope < 1.0e-5
    assert receipt.resonant_null_consistent
    assert receipt.resonant_kernel_limit_residual < 1.0e-10
    assert receipt.vertex_step_refinement < 2.0e-4
    assert receipt.vertex_grid_refinement < 1.0e-8
    assert receipt.vertex_gauge_residual < 1.0e-6
    assert receipt.hermiticity_residual < 1.0e-10
    assert receipt.same_k_exchange_residual < 1.0e-8
    assert receipt.kernel_quadrature_residual < 1.0e-10
    assert receipt.unit_denominator_growth_ratio_residual < 1.0e-6
    assert receipt.unit_denominator_relative_nonconvergence_witness > 0.49
    assert receipt.wrong_frequency_assignment_negative_control > 1.0e-6
    assert receipt.wrong_frequency_assignment_relative_control > 1.0e-3
    assert receipt.wrong_repeated_leg_negative_control > 1.0e-6
    assert receipt.wrong_repeated_leg_relative_control > 1.0e-3
    assert not receipt.local_exchange_elimination_rejected
    assert receipt.declared_exchange_gate_passed
