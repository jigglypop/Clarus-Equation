"""Physical transverse-traceless pole gate for the unique spin-2 kernel.

Under the declared real, local, Lorentz-covariant, parity-even two-derivative
single-field ansatz, the unique gauge-invariant action Hessian is an overall
nonzero multiple ``A`` of the Fierz--Pauli Hessian.  For momentum
``q=(omega,0,0,k)`` and normalized plus/cross representatives, its restriction
to the physical transverse-traceless subspace is exactly

``H_TT(q) = A q^2 I_2``,  with ``q^2=-omega^2+k^2``.

Consequently each helicity propagator has a simple ``1/(A q^2)`` pole and no
other finite pole inside this two-derivative ansatz.  The determinant has a
double zero only because there are two independent helicities; neither entry
has a double pole.  The sign of ``A`` is not fixed by the uniqueness theorem.

A dimensionless four-derivative deformation
``A q^2 (1 + beta q^2)`` supplies the negative control: for ``beta != 0`` it
has another root ``q^2=-1/beta``.  Thus the result does not exclude extra poles
from a microscopic kernel until higher-derivative/nonlocal terms are bounded
or absent by a separate derivation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from examples.physics.linearized_spin2_acceptance import (
    transverse_traceless_basis,
)
from examples.physics.two_derivative_spin2_uniqueness import (
    general_two_derivative_spin2_symbol,
)


_ETA_DIAGONAL = np.asarray((-1.0, 1.0, 1.0, 1.0))
_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _finite_axis_momentum(values: Sequence[float]) -> np.ndarray:
    momentum = np.asarray(values, dtype=float)
    if momentum.shape != (2,) or not np.all(np.isfinite(momentum)):
        raise ValueError("dimensionless_frequency_and_wavenumber must be finite (omega,k)")
    return momentum


def normalized_transverse_traceless_basis() -> np.ndarray:
    """Return Frobenius-normalized plus/cross columns in component order."""

    return np.asarray(transverse_traceless_basis(), dtype=float) / math.sqrt(2.0)


def physical_tt_action_hessian(
    dimensionless_frequency_and_wavenumber: Sequence[float],
    *,
    overall_coefficient: float,
) -> np.ndarray:
    """Restrict the unique two-derivative action Hessian to plus/cross."""

    frequency, wavenumber = _finite_axis_momentum(
        dimensionless_frequency_and_wavenumber
    )
    coefficient = float(overall_coefficient)
    if not math.isfinite(coefficient) or coefficient == 0.0:
        raise ValueError("overall_coefficient must be finite and nonzero")
    momentum = (frequency, 0.0, 0.0, wavenumber)
    equation_symbol = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -1.0)
    )
    weights = np.asarray(
        [
            (1.0 if mu == nu else 2.0)
            * _ETA_DIAGONAL[mu]
            * _ETA_DIAGONAL[nu]
            for mu, nu in _COMPONENTS
        ]
    )
    action_hessian = coefficient * np.diag(weights) @ equation_symbol
    tt_basis = normalized_transverse_traceless_basis()
    return np.asarray(tt_basis.T @ action_hessian @ tt_basis, dtype=float)


def physical_tt_propagator(
    dimensionless_frequency_and_wavenumber: Sequence[float],
    *,
    overall_coefficient: float,
) -> np.ndarray:
    """Invert the physical TT Hessian away from the massless shell."""

    frequency, wavenumber = _finite_axis_momentum(
        dimensionless_frequency_and_wavenumber
    )
    momentum_squared = -frequency**2 + wavenumber**2
    if momentum_squared == 0.0:
        raise ValueError("physical TT propagator is singular on q^2=0")
    return np.linalg.inv(
        physical_tt_action_hessian(
            (frequency, wavenumber), overall_coefficient=overall_coefficient
        )
    )


def higher_derivative_physical_roots(
    *, overall_coefficient: float, dimensionless_higher_derivative_coefficient: float
) -> tuple[float, ...]:
    """Return roots in ``z=q^2`` of ``A z (1+beta z)``."""

    coefficient = float(overall_coefficient)
    beta = float(dimensionless_higher_derivative_coefficient)
    if not math.isfinite(coefficient) or coefficient == 0.0:
        raise ValueError("overall_coefficient must be finite and nonzero")
    if not math.isfinite(beta):
        raise ValueError("dimensionless_higher_derivative_coefficient must be finite")
    return (0.0,) if beta == 0.0 else (0.0, -1.0 / beta)


@dataclass(frozen=True)
class MasslessSpin2PhysicalPoleAudit:
    dimensionless_frequency: float
    dimensionless_wavenumber: float
    dimensionless_momentum_squared: float
    overall_coefficient: float
    normalized_tt_gram: tuple[tuple[float, float], tuple[float, float]]
    physical_tt_hessian: tuple[tuple[float, float], tuple[float, float]]
    expected_physical_tt_eigenvalue: float
    physical_tt_hessian_residual: float
    physical_helicity_count: int
    each_helicity_pole_order: int
    physical_pole_root_in_q_squared: float
    determinant_zero_multiplicity_from_helicity_count: int
    exact_two_derivative_physical_tt_pole_gate_closed: bool
    no_additional_physical_tt_poles_within_declared_ansatz: bool
    overall_kinetic_sign_fixed: bool
    positive_residue_derived: bool
    higher_derivative_and_nonlocal_corrections_excluded: bool
    full_gauge_fixed_microscopic_propagator_constructed: bool
    microscopic_refinement_kernel_proved_to_use_this_pole_polynomial: bool
    status: str
    claim_ceiling: str = (
        "EXACT_TT_POLE_GATE_WITHIN_TWO_DERIVATIVE_ANSATZ_NOT_MICROSCOPIC_SPECTRUM"
    )


def audit_massless_spin2_physical_pole_gate(
    dimensionless_frequency_and_wavenumber: Sequence[float] = (0.7, 1.3),
    *,
    overall_coefficient: float = 1.0,
    tolerance: float = 1.0e-12,
) -> MasslessSpin2PhysicalPoleAudit:
    """Audit the exact two-helicity pole polynomial at one off-shell point."""

    frequency, wavenumber = _finite_axis_momentum(
        dimensionless_frequency_and_wavenumber
    )
    coefficient = float(overall_coefficient)
    tolerance = float(tolerance)
    if not math.isfinite(coefficient) or coefficient == 0.0:
        raise ValueError("overall_coefficient must be finite and nonzero")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    momentum_squared = float(-frequency**2 + wavenumber**2)
    if abs(momentum_squared) <= tolerance:
        raise ValueError("audit point must be off the q^2=0 pole")

    tt_basis = normalized_transverse_traceless_basis()
    component_frobenius_weights = np.asarray(
        [1.0 if mu == nu else 2.0 for mu, nu in _COMPONENTS]
    )
    tt_gram = tt_basis.T @ np.diag(component_frobenius_weights) @ tt_basis
    physical_hessian = physical_tt_action_hessian(
        (frequency, wavenumber), overall_coefficient=coefficient
    )
    expected_eigenvalue = coefficient * momentum_squared
    expected_hessian = expected_eigenvalue * np.eye(2)
    residual = float(np.linalg.norm(physical_hessian - expected_hessian))
    roots = higher_derivative_physical_roots(
        overall_coefficient=coefficient,
        dimensionless_higher_derivative_coefficient=0.0,
    )
    closed = (
        np.linalg.norm(tt_gram - np.eye(2)) <= tolerance
        and residual <= tolerance
        and roots == (0.0,)
    )

    return MasslessSpin2PhysicalPoleAudit(
        dimensionless_frequency=float(frequency),
        dimensionless_wavenumber=float(wavenumber),
        dimensionless_momentum_squared=momentum_squared,
        overall_coefficient=coefficient,
        normalized_tt_gram=(
            (float(tt_gram[0, 0]), float(tt_gram[0, 1])),
            (float(tt_gram[1, 0]), float(tt_gram[1, 1])),
        ),
        physical_tt_hessian=(
            (float(physical_hessian[0, 0]), float(physical_hessian[0, 1])),
            (float(physical_hessian[1, 0]), float(physical_hessian[1, 1])),
        ),
        expected_physical_tt_eigenvalue=expected_eigenvalue,
        physical_tt_hessian_residual=residual,
        physical_helicity_count=2,
        each_helicity_pole_order=1,
        physical_pole_root_in_q_squared=0.0,
        determinant_zero_multiplicity_from_helicity_count=2,
        exact_two_derivative_physical_tt_pole_gate_closed=closed,
        no_additional_physical_tt_poles_within_declared_ansatz=closed,
        overall_kinetic_sign_fixed=False,
        positive_residue_derived=False,
        higher_derivative_and_nonlocal_corrections_excluded=False,
        full_gauge_fixed_microscopic_propagator_constructed=False,
        microscopic_refinement_kernel_proved_to_use_this_pole_polynomial=False,
        status=(
            "TWO_HELICITY_SIMPLE_MASSLESS_TT_POLE_GATE_CLOSED"
            if closed
            else "MASSLESS_SPIN2_PHYSICAL_POLE_AUDIT_FAILED"
        ),
    )
