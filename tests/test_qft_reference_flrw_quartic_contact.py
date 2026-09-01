import numpy as np

from examples.physics.qft_reference_flrw_quartic_contact import (
    _fock_factor_residual,
    _fourth_mode_projection,
    evaluate_scalar_quartic_contact_gate,
)
from examples.physics.qft_reference_flrw_quartic_contact_gate import (
    reference_state,
)


def test_quartic_stencil_and_normal_ordered_factor() -> None:
    first = np.array([1.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0, 0.0])

    def quartic(vector: np.ndarray) -> np.ndarray:
        value = (vector[0] ** 2 + vector[1] ** 2) ** 2 / 24.0
        return np.array([value], dtype=np.longdouble)

    def quadratic(vector: np.ndarray) -> np.ndarray:
        return np.array([vector @ vector], dtype=np.longdouble)

    projected = _fourth_mode_projection(
        quartic,
        first,
        second,
        epsilon=0.1,
    )
    negative = _fourth_mode_projection(
        quadratic,
        first,
        second,
        epsilon=0.1,
    )
    assert abs(float(projected[0]) - 8.0 / 3.0) < 1.0e-12
    assert abs(float(negative[0])) < 1.0e-12
    assert _fock_factor_residual(4.0 / 3.0, 8.0 / 3.0) < 1.0e-12


def test_preregistered_scalar_quartic_contact_gate() -> None:
    state, parameters = reference_state()
    receipt = evaluate_scalar_quartic_contact_gate(
        state,
        parameters,
        base_wavenumber_bar=0.2,
    )

    assert len(receipt.branches) == 2
    assert receipt.maximum_analytic_direct_relative_residual < 2.0e-4
    assert receipt.maximum_step_residual < 2.0e-4
    assert receipt.maximum_grid_residual < 1.0e-8
    assert receipt.maximum_gauge_residual < 1.0e-6
    assert receipt.minimum_signal_to_error_ratio > 10.0
    assert receipt.maximum_fock_factor_residual < 1.0e-12
    assert receipt.minimum_induced_legendre_omission_ratio > 10.0
    assert receipt.minimum_wrong_induced_sign_ratio > 10.0
    assert receipt.maximum_momentum_residual < 2.0e-13
    assert receipt.maximum_constraint_residual < 1.0e-10
    assert receipt.declared_quartic_contact_gate_passed
