import numpy as np
import pytest

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
    state_from_conserved_charges,
)
from examples.physics.qft_reference_flrw_principal_stability import (
    PHYSICAL_MODE_NAMES,
    audit_finite_wavenumber_tachyon,
    audit_frozen_principal_stability,
    audit_subprincipal_power_counting,
    faddeev_popov_principal_symbol,
    linearized_derivative_order_audit,
    physical_principal_matrices,
    physical_principal_pencil,
    spatial_tt_basis,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    h = expanding_h_from_constraint(u=u, b=b, parameters=parameters)
    return ReferenceFlrwState(n=0.0, h=h, clock=0.0, u=u, b=b), parameters


def test_minimal_physical_principal_matrices_have_seven_positive_modes() -> None:
    _, parameters = reference_state()
    matrices = physical_principal_matrices(parameters)

    assert matrices.mode_names == PHYSICAL_MODE_NAMES
    assert np.allclose(np.diag(matrices.kinetic), (25.0, 25.0, 1, 1, 1, 1, 1))
    assert np.allclose(matrices.gradient, matrices.kinetic)
    assert np.all(np.linalg.eigvalsh(matrices.kinetic) > 0.0)


def test_admitted_background_passes_only_the_frozen_high_frequency_gate() -> None:
    state, parameters = reference_state()
    audit = audit_frozen_principal_stability(state, parameters)

    assert audit.declared_physical_principal_mode_count == 7
    assert audit.physical_kinetic_positive
    assert audit.physical_gradient_positive
    assert audit.real_characteristics
    assert audit.uniformly_diagonalizable_physical_symbol
    assert np.allclose(audit.speed_squared_eigenvalues, np.ones(7))
    assert audit.principal_background_mixing_norm == 0.0
    assert audit.background_gradient_subprincipal_norm > 0.0
    assert audit.background_gradients_are_subprincipal
    assert audit.high_frequency_principal_gate_passed
    assert not audit.auxiliary_metric_components_counted_as_physical
    assert not audit.harmonic_constraint_propagation_computed
    assert not audit.finite_k_hessian_computed
    assert not audit.strong_coupling_scale_computed
    assert not audit.brst_physical_inner_product_constructed
    assert not audit.one_loop_st_identity_computed
    assert not audit.nonperturbative_m2_passed


def test_principal_matrices_are_independent_of_admitted_background_gradients() -> None:
    state, parameters = reference_state()
    moved = state_from_conserved_charges(
        n=0.4,
        clock=0.2,
        clock_charge=np.exp(3.0 * state.n) * state.u,
        rod_charge=np.exp(state.n) * state.b,
        parameters=parameters,
    )
    first = audit_frozen_principal_stability(state, parameters)
    second = audit_frozen_principal_stability(moved, parameters)

    assert np.allclose(first.kinetic_eigenvalues, second.kinetic_eigenvalues)
    assert np.allclose(first.gradient_eigenvalues, second.gradient_eigenvalues)
    assert np.allclose(first.speed_squared_eigenvalues, second.speed_squared_eigenvalues)


def test_linearized_equation_mixing_is_one_derivative_below_the_wave_blocks() -> None:
    state, _ = reference_state()
    audit = linearized_derivative_order_audit(state)

    assert audit.derivative_orders == ((2, 1), (1, 2))
    assert audit.background_gradient_norm > 0.0
    assert audit.subprincipal_gradient_coefficient_norm > 0.0
    assert audit.principal_off_diagonal_norm == 0.0
    assert audit.metric_scalar_mixing_is_strictly_subprincipal


@pytest.mark.parametrize(
    'wavevector',
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    ],
)
def test_tt_basis_is_transverse_traceless_for_arbitrary_direction(
    wavevector: np.ndarray,
) -> None:
    basis = spatial_tt_basis(wavevector)
    direction = wavevector / np.linalg.norm(wavevector)

    assert basis.shape == (2, 3, 3)
    assert np.allclose(np.trace(basis, axis1=1, axis2=2), 0.0)
    assert np.allclose(np.einsum('i,aij->aj', direction, basis), 0.0)
    assert np.allclose(np.einsum('aij,bij->ab', basis, basis), np.eye(2))


def test_wrong_sign_internal_metric_is_a_high_frequency_ghost() -> None:
    state, parameters = reference_state()
    wrong = np.diag([-1.0, 1.0, 1.0, 1.0])
    audit = audit_frozen_principal_stability(
        state,
        parameters,
        internal_kinetic_metric=wrong,
    )

    assert min(audit.kinetic_eigenvalues) < 0.0
    assert not audit.physical_kinetic_positive
    assert not audit.high_frequency_principal_gate_passed


def test_gradient_sign_flip_has_positive_kinetic_but_negative_speed_squared() -> None:
    state, parameters = reference_state()
    wrong_gradient = np.diag([-1.0, 1.0, 1.0, 1.0])
    audit = audit_frozen_principal_stability(
        state,
        parameters,
        internal_gradient_metric=wrong_gradient,
    )

    assert audit.physical_kinetic_positive
    assert not audit.physical_gradient_positive
    assert min(audit.speed_squared_eigenvalues) < 0.0
    assert not audit.high_frequency_principal_gate_passed


def test_lorentz_invariant_derivative_mixing_has_a_sharp_kinetic_threshold() -> None:
    state, parameters = reference_state()
    below = audit_frozen_principal_stability(
        state,
        parameters,
        chi_x0_kinetic_mixing=0.75,
    )
    at_threshold = audit_frozen_principal_stability(
        state,
        parameters,
        chi_x0_kinetic_mixing=1.0,
    )

    assert np.isclose(min(below.kinetic_eigenvalues), 0.25)
    assert below.high_frequency_principal_gate_passed
    assert np.allclose(below.speed_squared_eigenvalues, np.ones(7))
    assert np.isclose(min(at_threshold.kinetic_eigenvalues), 0.0)
    assert not at_threshold.high_frequency_principal_gate_passed


def test_independent_gradient_mixing_can_destabilize_an_unchanged_kinetic_form() -> None:
    state, parameters = reference_state()
    audit = audit_frozen_principal_stability(
        state,
        parameters,
        chi_x0_kinetic_mixing=0.0,
        chi_x0_gradient_mixing=1.1,
    )

    assert audit.physical_kinetic_positive
    assert not audit.physical_gradient_positive
    assert min(audit.speed_squared_eigenvalues) < 0.0


def test_physical_and_fp_symbols_share_the_null_characteristic() -> None:
    _, parameters = reference_state()
    matrices = physical_principal_matrices(parameters)

    assert np.linalg.norm(physical_principal_pencil(2.0, 2.0, matrices)) < 1.0e-14
    assert np.linalg.norm(faddeev_popov_principal_symbol(2.0, np.array([2.0, 0.0, 0.0]))) < 1.0e-14
    assert np.linalg.norm(physical_principal_pencil(1.0, 2.0, matrices)) > 1.0


def test_subprincipal_background_mixing_dies_relative_to_k_squared() -> None:
    state, parameters = reference_state()
    low = audit_subprincipal_power_counting(state, parameters, wavenumber_bar=100.0)
    high = audit_subprincipal_power_counting(state, parameters, wavenumber_bar=1000.0)

    assert np.isclose(high.one_derivative_to_principal_ratio, low.one_derivative_to_principal_ratio / 10.0)
    assert np.isclose(high.curvature_to_principal_ratio, low.curvature_to_principal_ratio / 100.0)
    assert high.strict_high_frequency_limit_zero


def test_positive_principal_matrices_do_not_exclude_a_low_k_tachyon() -> None:
    unstable = audit_finite_wavenumber_tachyon(
        mass_squared_bar=-1.0,
        wavenumber_bar=0.5,
    )
    stable = audit_finite_wavenumber_tachyon(
        mass_squared_bar=-1.0,
        wavenumber_bar=2.0,
    )

    assert unstable.principal_kinetic_positive
    assert unstable.principal_gradient_positive
    assert unstable.omega_squared_bar < 0.0
    assert not unstable.finite_wavenumber_stable
    assert stable.finite_wavenumber_stable


def test_principal_gate_rejects_a_degenerate_e62_background() -> None:
    state, parameters = reference_state()
    degenerate = ReferenceFlrwState(
        n=state.n,
        h=expanding_h_from_constraint(u=0.0, b=state.b, parameters=parameters),
        clock=state.clock,
        u=0.0,
        b=state.b,
    )
    with pytest.raises(ValueError, match='nondegenerate'):
        audit_frozen_principal_stability(degenerate, parameters)


def test_principal_matrix_inputs_fail_closed() -> None:
    _, parameters = reference_state()
    with pytest.raises(ValueError, match='4 by 4'):
        physical_principal_matrices(parameters, internal_kinetic_metric=np.eye(3))
    nonsymmetric = np.eye(4)
    nonsymmetric[0, 1] = 1.0
    with pytest.raises(ValueError, match='symmetric'):
        physical_principal_matrices(parameters, internal_kinetic_metric=nonsymmetric)
    with pytest.raises(ValueError, match='positive'):
        audit_subprincipal_power_counting(
            reference_state()[0],
            parameters,
            wavenumber_bar=0.0,
        )
