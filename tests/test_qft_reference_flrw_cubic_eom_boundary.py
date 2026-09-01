import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    frozen_symplectic_scalar_modes,
)
from examples.physics.qft_reference_flrw_cubic_eom_boundary import (
    evaluate_scalar_eom_boundary_gate,
    frozen_scalar_pencil,
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


def test_signed_pencil_annihilates_matched_kg_mode_and_wrong_sign_does_not() -> None:
    state, parameters = reference_state()
    modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=0.2,
    )
    correct_residuals = []
    wrong_residuals = []
    for mode in modes:
        for sign in (-1, 1):
            configuration = mode.configuration if sign == 1 else mode.configuration.conj()
            frequency = sign * mode.frequency_bar
            correct = frozen_scalar_pencil(
                state,
                parameters,
                comoving_wavenumber_bar=0.2,
                signed_frequency_bar=frequency,
            )
            wrong = frozen_scalar_pencil(
                state,
                parameters,
                comoving_wavenumber_bar=0.2,
                signed_frequency_bar=frequency,
                reverse_gyroscopic_sign=True,
            )
            correct_residuals.append(np.linalg.norm(correct @ configuration))
            wrong_residuals.append(np.linalg.norm(wrong @ configuration))

    assert max(correct_residuals) < 1.0e-8
    assert max(wrong_residuals) > 1.0e-6


def test_preregistered_eom_quotient_and_boundary_endpoint_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_scalar_eom_boundary_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
    )

    assert receipt.assignment_count == 64
    assert receipt.vertex_step_refinement < 2.0e-4
    assert receipt.vertex_grid_refinement < 1.0e-8
    assert receipt.vertex_gauge_residual < 1.0e-6
    assert receipt.configuration_map_residual < 1.0e-10
    assert receipt.correct_eom_quotient_residual < 1.0e-8
    assert receipt.transposed_pencil_negative_control > 1.0e-6
    assert receipt.reversed_gyroscopic_negative_control > 1.0e-6
    assert receipt.non_eom_negative_control > 1.0e-6
    assert receipt.maximum_pencil_residual < 1.0e-8
    assert receipt.same_k_exchange_residual < 1.0e-8
    assert receipt.resonant_assignment_count > 0
    assert receipt.nonresonant_assignment_count > 0
    assert receipt.maximum_resonant_endpoint < 1.0e-10
    assert receipt.maximum_normalized_boundary_endpoint > 1.0e-6
    assert receipt.boundary_quadrature_residual < 1.0e-8
    assert receipt.declared_eom_boundary_gate_passed
