"""Uniqueness of the two-derivative massless spin-2 quadratic kernel.

Assume locality and translation invariance, no background tensors except
``eta`` and ``q``, parity evenness, no additional fields, and no higher-
derivative or nonlocal terms.  The most general Lorentz-covariant linear
operator on a symmetric tensor that is homogeneous quadratic in momentum is

``E_mn = a q^2 h_mn
       + b (q_m q^r h_rn + q_n q^r h_rm)
       + c q_m q_n h
       + d eta_mn q^r q^s h_rs
       + e eta_mn q^2 h``.

Gauge-nullness under ``delta h_mn=q_m xi_n+q_n xi_m`` gives three independent
relations and leaves a two-dimensional family.  Formal self-adjointness under
the symmetric-tensor action pairing adds ``c=d``.  Equivalently, one may add
the off-shell Bianchi identity ``q^m E_mn=0``.  Either route leaves the single
coefficient ray

``(a,b,c,d,e) = A (1,-1,1,1,-1)``,

which is twice the normalization used by ``linearized_einstein_symbol`` when
``A=1``.  Thus a real local quadratic action in the declared ansatz is the
Fierz--Pauli quadratic action up to integration by parts and an overall scale.
The theorem does not prove that a microscopic/refined CE kernel lies in that
ansatz, choose the positive scale, or construct/demonstrate dominance of the
nonlinear Einstein action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from examples.physics.lattice_fierz_pauli_refinement import (
    linearized_bianchi_divergence_matrix,
    linearized_einstein_symbol,
    linearized_gauge_direction_matrix,
)


RationalMatrix = tuple[tuple[Fraction, ...], ...]

_ETA = np.diag((-1.0, 1.0, 1.0, 1.0))
_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _rank(matrix: RationalMatrix) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    pivot_row = 0
    for column in range(len(rows[0])):
        pivot = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def gauge_coefficient_constraints() -> RationalMatrix:
    """Return ``a+b=0``, ``b+c=0`` and ``d+e=0``."""

    return (
        (Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )


def gauge_and_bianchi_coefficient_constraints() -> RationalMatrix:
    """Add ``b+d=0`` and ``c+e=0`` to the gauge relations."""

    return gauge_coefficient_constraints() + (
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
    )


def gauge_and_self_adjoint_coefficient_constraints() -> RationalMatrix:
    """Add formal action-Hessian self-adjointness ``c=d``."""

    return gauge_coefficient_constraints() + (
        (Fraction(0), Fraction(0), Fraction(1), Fraction(-1), Fraction(0)),
    )


def _finite_momentum(values: Sequence[float]) -> np.ndarray:
    momentum = np.asarray(values, dtype=float)
    if momentum.shape != (4,) or not np.all(np.isfinite(momentum)):
        raise ValueError("dimensionless_momentum_up must contain four finite values")
    return momentum


def _finite_coefficients(values: Sequence[float]) -> np.ndarray:
    coefficients = np.asarray(values, dtype=float)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("operator coefficients must contain five finite values")
    return coefficients


def _symmetric_basis_tensor(component: tuple[int, int]) -> np.ndarray:
    first, second = component
    tensor = np.zeros((4, 4), dtype=float)
    tensor[first, second] = 1.0
    tensor[second, first] = 1.0
    return tensor


def general_two_derivative_spin2_symbol(
    dimensionless_momentum_up: Sequence[float],
    coefficients: Sequence[float],
) -> np.ndarray:
    """Materialize the displayed five-coefficient 10-by-10 ansatz."""

    momentum_up = _finite_momentum(dimensionless_momentum_up)
    coefficient_values = _finite_coefficients(coefficients)
    a, b, c, d, e = coefficient_values
    momentum_down = _ETA @ momentum_up
    momentum_squared = float(momentum_up @ momentum_down)
    columns = []
    for component in _COMPONENTS:
        field = _symmetric_basis_tensor(component)
        trace = float(np.sum(np.diag(_ETA) * np.diag(field)))
        double_divergence = float(momentum_up @ field @ momentum_up)
        output = np.zeros((4, 4), dtype=float)
        for mu in range(4):
            for nu in range(4):
                output[mu, nu] = (
                    a * momentum_squared * field[mu, nu]
                    + b
                    * (
                        momentum_down[mu] * float(momentum_up @ field[:, nu])
                        + momentum_down[nu] * float(momentum_up @ field[:, mu])
                    )
                    + c * momentum_down[mu] * momentum_down[nu] * trace
                    + d * _ETA[mu, nu] * double_divergence
                    + e * _ETA[mu, nu] * momentum_squared * trace
                )
        columns.append(np.asarray([output[index] for index in _COMPONENTS]))
    return np.column_stack(columns)


@dataclass(frozen=True)
class TwoDerivativeSpin2UniquenessAudit:
    ansatz_coefficient_count: int
    gauge_constraint_rank: int
    gauge_invariant_family_dimension: int
    gauge_bianchi_constraint_rank: int
    gauge_bianchi_family_dimension: int
    gauge_self_adjoint_constraint_rank: int
    gauge_self_adjoint_family_dimension: int
    unique_coefficient_ray: tuple[int, int, int, int, int]
    linearized_einstein_match_residual: float
    unique_ray_gauge_null_residual: float
    unique_ray_bianchi_residual: float
    unique_ray_weighted_self_adjoint_residual: float
    gauge_only_counterexample_gauge_residual: float
    gauge_only_counterexample_bianchi_residual: float
    gauge_only_counterexample_weighted_self_adjoint_residual: float
    exact_unique_fierz_pauli_ray_within_ansatz: bool
    gauge_invariance_alone_is_insufficient: bool
    conditional_unique_fierz_pauli_quadratic_kernel_ray_within_declared_operator_ansatz: bool
    conditional_quadratic_action_is_fp_up_to_boundary_and_overall_scale: bool
    positive_overall_normalization_fixed: bool
    microscopic_effective_kernel_proved_to_lie_in_ansatz: bool
    higher_derivative_and_nonlocal_terms_excluded_microscopically: bool
    nonlinear_einstein_completion_derived_from_ce: bool
    status: str
    claim_ceiling: str = (
        "UNIQUE_TWO_DERIVATIVE_SPIN2_RAY_NOT_MICROSCOPIC_EH_DOMINANCE"
    )


def audit_two_derivative_spin2_uniqueness(
    dimensionless_momentum_up: Sequence[float] = (1.2, 0.3, -0.4, 0.8),
    *,
    tolerance: float = 1.0e-12,
) -> TwoDerivativeSpin2UniquenessAudit:
    """Audit the exact coefficient ranks and representative symbols."""

    momentum = _finite_momentum(dimensionless_momentum_up)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    gauge_constraints = gauge_coefficient_constraints()
    full_constraints = gauge_and_bianchi_coefficient_constraints()
    action_constraints = gauge_and_self_adjoint_coefficient_constraints()
    gauge_rank = _rank(gauge_constraints)
    full_rank = _rank(full_constraints)
    action_rank = _rank(action_constraints)
    unique_ray = (1, -1, 1, 1, -1)
    unique_symbol = general_two_derivative_spin2_symbol(momentum, unique_ray)
    target_symbol = 2.0 * linearized_einstein_symbol(momentum)
    gauge = linearized_gauge_direction_matrix(momentum)
    bianchi = linearized_bianchi_divergence_matrix(momentum)
    match_residual = float(np.linalg.norm(unique_symbol - target_symbol))
    unique_gauge_residual = float(np.linalg.norm(unique_symbol @ gauge))
    unique_bianchi_residual = float(np.linalg.norm(bianchi @ unique_symbol))
    component_weights = np.asarray(
        [
            (1.0 if mu == nu else 2.0) * _ETA[mu, mu] * _ETA[nu, nu]
            for mu, nu in _COMPONENTS
        ]
    )
    unique_weighted = np.diag(component_weights) @ unique_symbol
    unique_self_adjoint_residual = float(
        np.linalg.norm(unique_weighted - unique_weighted.T)
    )

    gauge_only_ray = (1, -1, 1, 0, 0)
    gauge_only_symbol = general_two_derivative_spin2_symbol(momentum, gauge_only_ray)
    gauge_only_gauge_residual = float(np.linalg.norm(gauge_only_symbol @ gauge))
    gauge_only_bianchi_residual = float(np.linalg.norm(bianchi @ gauge_only_symbol))
    gauge_only_weighted = np.diag(component_weights) @ gauge_only_symbol
    gauge_only_self_adjoint_residual = float(
        np.linalg.norm(gauge_only_weighted - gauge_only_weighted.T)
    )
    unique = (
        full_rank == 4
        and 5 - full_rank == 1
        and action_rank == 4
        and 5 - action_rank == 1
        and match_residual <= tolerance
        and unique_gauge_residual <= tolerance
        and unique_bianchi_residual <= tolerance
        and unique_self_adjoint_residual <= tolerance
    )
    gauge_insufficient = (
        gauge_rank == 3
        and 5 - gauge_rank == 2
        and gauge_only_gauge_residual <= tolerance
        and gauge_only_bianchi_residual > tolerance
        and gauge_only_self_adjoint_residual > tolerance
    )
    return TwoDerivativeSpin2UniquenessAudit(
        ansatz_coefficient_count=5,
        gauge_constraint_rank=gauge_rank,
        gauge_invariant_family_dimension=5 - gauge_rank,
        gauge_bianchi_constraint_rank=full_rank,
        gauge_bianchi_family_dimension=5 - full_rank,
        gauge_self_adjoint_constraint_rank=action_rank,
        gauge_self_adjoint_family_dimension=5 - action_rank,
        unique_coefficient_ray=unique_ray,
        linearized_einstein_match_residual=match_residual,
        unique_ray_gauge_null_residual=unique_gauge_residual,
        unique_ray_bianchi_residual=unique_bianchi_residual,
        unique_ray_weighted_self_adjoint_residual=unique_self_adjoint_residual,
        gauge_only_counterexample_gauge_residual=gauge_only_gauge_residual,
        gauge_only_counterexample_bianchi_residual=gauge_only_bianchi_residual,
        gauge_only_counterexample_weighted_self_adjoint_residual=(
            gauge_only_self_adjoint_residual
        ),
        exact_unique_fierz_pauli_ray_within_ansatz=unique,
        gauge_invariance_alone_is_insufficient=gauge_insufficient,
        conditional_unique_fierz_pauli_quadratic_kernel_ray_within_declared_operator_ansatz=(
            unique
        ),
        conditional_quadratic_action_is_fp_up_to_boundary_and_overall_scale=unique,
        positive_overall_normalization_fixed=False,
        microscopic_effective_kernel_proved_to_lie_in_ansatz=False,
        higher_derivative_and_nonlocal_terms_excluded_microscopically=False,
        nonlinear_einstein_completion_derived_from_ce=False,
        status=(
            "UNIQUE_TWO_DERIVATIVE_MASSLESS_SPIN2_KERNEL_RAY_CLOSED"
            if unique and gauge_insufficient
            else "TWO_DERIVATIVE_SPIN2_UNIQUENESS_AUDIT_FAILED"
        ),
    )
