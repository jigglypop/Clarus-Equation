from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.massless_spin2_physical_pole_gate import (
    audit_massless_spin2_physical_pole_gate,
    higher_derivative_physical_roots,
    normalized_transverse_traceless_basis,
    physical_tt_action_hessian,
    physical_tt_propagator,
)


def test_plus_cross_basis_is_frobenius_orthonormal() -> None:
    basis = normalized_transverse_traceless_basis()
    weights = np.asarray((1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0))

    assert basis.T @ np.diag(weights) @ basis == pytest.approx(np.eye(2))


@pytest.mark.parametrize(
    "frequency,wavenumber,coefficient",
    (
        (0.7, 1.3, 1.0),
        (1.4, 0.2, 2.5),
        (-0.9, 1.7, -3.0),
    ),
)
def test_tt_hessian_is_exactly_a_q_squared_identity(
    frequency: float, wavenumber: float, coefficient: float
) -> None:
    momentum_squared = -frequency**2 + wavenumber**2
    hessian = physical_tt_action_hessian(
        (frequency, wavenumber), overall_coefficient=coefficient
    )

    assert hessian == pytest.approx(coefficient * momentum_squared * np.eye(2))


def test_each_tt_propagator_entry_has_the_single_massless_denominator() -> None:
    frequency = 0.4
    wavenumber = 1.1
    coefficient = 2.0
    momentum_squared = -frequency**2 + wavenumber**2
    propagator = physical_tt_propagator(
        (frequency, wavenumber), overall_coefficient=coefficient
    )

    assert propagator == pytest.approx(
        np.eye(2) / (coefficient * momentum_squared)
    )


def test_determinant_zero_order_two_is_the_product_of_two_simple_channels() -> None:
    frequency = 0.6
    wavenumber = 1.4
    coefficient = -1.7
    momentum_squared = -frequency**2 + wavenumber**2
    hessian = physical_tt_action_hessian(
        (frequency, wavenumber), overall_coefficient=coefficient
    )
    channel_eigenvalues = np.linalg.eigvalsh(hessian)

    assert channel_eigenvalues == pytest.approx(
        (coefficient * momentum_squared, coefficient * momentum_squared)
    )
    assert np.linalg.det(hessian) == pytest.approx(
        (coefficient * momentum_squared) ** 2
    )


def test_exact_two_derivative_polynomial_has_only_q_squared_zero_root() -> None:
    assert higher_derivative_physical_roots(
        overall_coefficient=2.0,
        dimensionless_higher_derivative_coefficient=0.0,
    ) == (0.0,)


@pytest.mark.parametrize("beta", (-2.0, 0.25, 3.0))
def test_four_derivative_deformation_has_an_additional_root(beta: float) -> None:
    roots = higher_derivative_physical_roots(
        overall_coefficient=1.0,
        dimensionless_higher_derivative_coefficient=beta,
    )

    assert roots == pytest.approx((0.0, -1.0 / beta))
    assert roots[1] != 0.0


def test_propagator_rejects_the_massless_shell() -> None:
    with pytest.raises(ValueError, match="singular"):
        physical_tt_propagator((1.0, 1.0), overall_coefficient=1.0)


def test_audit_closes_only_the_declared_two_derivative_tt_gate() -> None:
    audit = audit_massless_spin2_physical_pole_gate()

    assert np.allclose(np.asarray(audit.normalized_tt_gram), np.eye(2))
    assert audit.physical_helicity_count == 2
    assert audit.each_helicity_pole_order == 1
    assert audit.physical_pole_root_in_q_squared == 0.0
    assert audit.determinant_zero_multiplicity_from_helicity_count == 2
    assert audit.exact_two_derivative_physical_tt_pole_gate_closed
    assert audit.no_additional_physical_tt_poles_within_declared_ansatz
    assert audit.status == "TWO_HELICITY_SIMPLE_MASSLESS_TT_POLE_GATE_CLOSED"
    assert not audit.overall_kinetic_sign_fixed
    assert not audit.positive_residue_derived
    assert not audit.higher_derivative_and_nonlocal_corrections_excluded
    assert not audit.full_gauge_fixed_microscopic_propagator_constructed
    assert not audit.microscopic_refinement_kernel_proved_to_use_this_pole_polynomial
    assert audit.claim_ceiling.endswith("NOT_MICROSCOPIC_SPECTRUM")


@pytest.mark.parametrize(
    "call,arguments,message",
    (
        (physical_tt_action_hessian, ((1.0,),), "must be finite"),
        (physical_tt_action_hessian, ((1.0, math.nan),), "must be finite"),
        (physical_tt_action_hessian, ((1.0, 0.0),), "overall_coefficient"),
        (higher_derivative_physical_roots, (), "overall_coefficient"),
    ),
)
def test_invalid_inputs_are_rejected(
    call: object, arguments: tuple[object, ...], message: str
) -> None:
    if call is physical_tt_action_hessian:
        coefficient = 0.0 if arguments == ((1.0, 0.0),) else 1.0
        with pytest.raises(ValueError, match=message):
            physical_tt_action_hessian(
                arguments[0], overall_coefficient=coefficient  # type: ignore[arg-type]
            )
    else:
        with pytest.raises(ValueError, match=message):
            higher_derivative_physical_roots(
                overall_coefficient=0.0,
                dimensionless_higher_derivative_coefficient=1.0,
            )


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.inf, math.nan))
def test_audit_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        audit_massless_spin2_physical_pole_gate(tolerance=tolerance)
