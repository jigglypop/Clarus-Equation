import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_cubic_dyson import (
    evaluate_scalar_cubic_dyson_gate,
    finite_time_exponential_kernel,
    scalar_cubic_dyson_channels,
    simpson_exponential_kernel,
    two_state_dyson_result,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    frozen_symplectic_scalar_modes,
)
from examples.physics.qft_reference_flrw_cubic_eom_boundary import (
    deterministic_boundary_tensor,
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


def test_two_state_exact_evolution_matches_closed_form_and_dyson_scaling() -> None:
    matrix_element = 0.037 + 0.021j
    omega = 0.7
    interval = 0.5
    kernel = finite_time_exponential_kernel(omega, interval)
    quadrature = simpson_exponential_kernel(
        omega,
        interval,
        subintervals=4096,
    )
    first = -1j * matrix_element * kernel
    transition_errors = []
    unitary_errors = []
    for coupling in (1.0, 0.5, 0.25):
        result = two_state_dyson_result(
            matrix_element,
            omega,
            interval,
            coupling=coupling,
        )
        assert abs(
            result.exact_interaction_amplitude
            - result.closed_form_interaction_amplitude
        ) < 1.0e-12
        transition_errors.append(
            abs(result.exact_interaction_amplitude / coupling - first)
        )
        unitary_errors.append(
            np.linalg.norm(
                result.exact_interaction_unitary - result.first_dyson_unitary
            )
        )

    assert abs(kernel - quadrature) < 1.0e-10
    assert abs(transition_errors[1] / transition_errors[0] - 0.25) < 0.05
    assert abs(transition_errors[2] / transition_errors[1] - 0.25) < 0.05
    assert abs(unitary_errors[1] / unitary_errors[0] - 0.25) < 0.05
    assert abs(unitary_errors[2] / unitary_errors[1] - 0.25) < 0.05


def test_dyson_boundary_column_retains_the_e68_q_half_factor() -> None:
    state, parameters = reference_state()
    channels = scalar_cubic_dyson_channels(
        state,
        parameters,
        base_wavenumber_bar=0.2,
        hamiltonian_tensor=np.zeros((4, 4, 4)),
        interval_bar=0.5,
    )
    first_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
    )
    third_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.4,
    )
    boundary = deterministic_boundary_tensor()
    first = channels[0]
    raw = np.einsum(
        'abc,a,b,c->',
        boundary,
        first_modes[0].configuration.conj(),
        first_modes[0].configuration.conj(),
        third_modes[0].configuration.conj(),
    )
    expected = (
        0.5
        / np.sqrt(2.0)
        * raw
        * (np.exp(1j * first.frequency_gap_bar * 0.5) - 1.0)
    )

    assert abs(first.boundary_endpoint - expected) < 1.0e-12


def test_preregistered_finite_time_cubic_dyson_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_scalar_cubic_dyson_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
    )

    assert receipt.channel_count == 8
    assert receipt.vertex_step_refinement < 2.0e-4
    assert receipt.vertex_grid_refinement < 1.0e-8
    assert receipt.vertex_gauge_residual < 1.0e-6
    assert receipt.hermiticity_residual < 1.0e-10
    assert receipt.kernel_quadrature_residual < 1.0e-10
    assert receipt.kernel_grid_refinement < 1.0e-8
    assert receipt.exact_closed_form_residual < 1.0e-10
    assert receipt.maximum_active_transition_relative_residual < 2.0e-4
    assert receipt.maximum_inactive_transition_absolute_residual < 1.0e-8
    assert receipt.maximum_inactive_unitary_absolute_residual < 1.0e-8
    assert receipt.wrong_frequency_sign_negative_control > 1.0e-6
    assert receipt.wrong_repeated_leg_negative_control > 1.0e-6
    assert receipt.overlarge_coupling_negative_control > 1.0e-4
    assert receipt.minimum_boundary_endpoint > 0.0
    assert receipt.boundary_was_kept_separate
    assert receipt.declared_dyson_gate_passed
