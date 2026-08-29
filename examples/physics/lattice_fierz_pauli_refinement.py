"""Gauge-preserving lattice-symbol refinement of the Fierz--Pauli model.

Let ``q^mu = L_ref p^mu`` and ``abar = a/L_ref`` be dimensionless.  Replacing
each momentum component by the central-difference symbol

``qhat^mu = sin(abar q^mu)/abar``

inside the linearized Einstein symbol preserves its algebraic gauge-null and
Bianchi identities at every fixed lattice spacing.  On every compact momentum
box, ``qhat -> q`` uniformly because

``|sin(abar q)/abar - q| <= abar^2 |q|^3 / 6``.

Since the Fierz--Pauli symbol is polynomial in momentum, its lattice symbol
therefore converges uniformly to the continuum symbol.  More explicitly, if
``|q_mu| <= B`` and ``delta=max_mu|qhat_mu-q_mu|``, direct term counting in the
implemented 10-by-10 symbol gives

``||K(qhat)-K(q)||_F <= 130 B delta <= (130/6) abar^2 B^4``.

The bound tends to zero for every fixed finite ``B``.  Along the null ray
``q=(omega,0,0,omega)`` the central-difference symbol is a nonzero scalar
multiple of the same null direction, so the exact two-polarization quotient is
unchanged in the declared low-momentum window.

This is a constructive refinement family for an already supplied free action.
It is not obtained from EPRL/spin-foam amplitudes and does not prove nonlinear
Einstein dynamics, an interacting renormalized limit, or global absence of
lattice doublers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


_ETA = np.diag((-1.0, 1.0, 1.0, 1.0))
_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _finite_four_vector(name: str, values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain four finite values")
    return vector


def _symmetric_basis_tensor(component: tuple[int, int]) -> np.ndarray:
    first, second = component
    tensor = np.zeros((4, 4), dtype=float)
    tensor[first, second] = 1.0
    tensor[second, first] = 1.0
    return tensor


def central_difference_dimensionless_momentum(
    dimensionless_momentum_up: Sequence[float],
    *,
    lattice_spacing_over_reference_length: float,
) -> np.ndarray:
    """Return ``sin(abar q^mu)/abar`` component by component."""

    momentum = _finite_four_vector(
        "dimensionless_momentum_up", dimensionless_momentum_up
    )
    spacing = float(lattice_spacing_over_reference_length)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice spacing ratio must be finite and positive")
    return np.sin(spacing * momentum) / spacing


def linearized_einstein_symbol(
    dimensionless_momentum_up: Sequence[float],
) -> np.ndarray:
    """Return the 10-by-10 Fourier symbol ``h -> G^(1)[h]``.

    The output and input use the independent lower-index symmetric components
    ``(00,01,02,03,11,12,13,22,23,33)``.
    """

    momentum_up = _finite_four_vector(
        "dimensionless_momentum_up", dimensionless_momentum_up
    )
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
                output[mu, nu] = 0.5 * (
                    -momentum_down[nu] * float(momentum_up @ field[:, mu])
                    -momentum_down[mu] * float(momentum_up @ field[:, nu])
                    + momentum_squared * field[mu, nu]
                    + momentum_down[mu] * momentum_down[nu] * trace
                    + _ETA[mu, nu] * double_divergence
                    - _ETA[mu, nu] * momentum_squared * trace
                )
        columns.append(np.asarray([output[index] for index in _COMPONENTS]))
    return np.column_stack(columns)


def linearized_gauge_direction_matrix(
    dimensionless_momentum_up: Sequence[float],
) -> np.ndarray:
    """Return columns ``delta h_{mu nu}=q_mu xi_nu+q_nu xi_mu``."""

    momentum_up = _finite_four_vector(
        "dimensionless_momentum_up", dimensionless_momentum_up
    )
    momentum_down = _ETA @ momentum_up
    matrix = np.zeros((10, 4), dtype=float)
    for row, (mu, nu) in enumerate(_COMPONENTS):
        for gauge_index in range(4):
            matrix[row, gauge_index] = (
                momentum_down[mu] * float(nu == gauge_index)
                + momentum_down[nu] * float(mu == gauge_index)
            )
    return matrix


def linearized_bianchi_divergence_matrix(
    dimensionless_momentum_up: Sequence[float],
) -> np.ndarray:
    """Return the map from lower symmetric tensors to ``q^mu X_{mu nu}``."""

    momentum_up = _finite_four_vector(
        "dimensionless_momentum_up", dimensionless_momentum_up
    )
    matrix = np.zeros((4, 10), dtype=float)
    for column, component in enumerate(_COMPONENTS):
        tensor = _symmetric_basis_tensor(component)
        matrix[:, column] = momentum_up @ tensor
    return matrix


@dataclass(frozen=True)
class LatticeFierzPauliRefinementAudit:
    lattice_spacing_over_reference_length: float
    compact_dimensionless_momentum_bound: float
    generic_dimensionless_momentum_up: tuple[float, float, float, float]
    lattice_symbol_momentum_up: tuple[float, float, float, float]
    component_symbol_error_bound: float
    maximum_component_symbol_error: float
    fierz_pauli_symbol_frobenius_error: float
    compact_uniform_fierz_pauli_symbol_error_bound: float
    lattice_gauge_null_residual: float
    lattice_bianchi_residual: float
    weighted_action_self_adjoint_residual: float
    null_ray_lattice_norm_squared: float
    null_ray_harmonic_constraint_rank: int
    null_ray_residual_gauge_rank: int
    null_ray_physical_quotient_dimension: int
    all_momentum_arguments_dimensionless: bool
    central_difference_error_bound_satisfied: bool
    sample_fierz_pauli_error_within_uniform_analytic_bound: bool
    algebraic_gauge_and_bianchi_identities_preserved: bool
    quadratic_action_symbol_self_adjoint: bool
    compact_uniform_free_symbol_limit_closed: bool
    low_momentum_null_ray_two_polarization_gate_preserved: bool
    low_momentum_component_doubler_window: bool
    global_lattice_doublers_excluded: bool
    geometric_or_spin_foam_refinement_derived: bool
    interacting_renormalized_limit_proved: bool
    nonlinear_constraint_algebra_proved: bool
    einstein_hilbert_dominance_from_ce_proved: bool
    status: str
    claim_ceiling: str = (
        "DECLARED_FREE_LATTICE_FIERZ_PAULI_REFINEMENT_NOT_SPINFOAM_EH_LIMIT"
    )


def _harmonic_constraint_rank(momentum_up: np.ndarray, tolerance: float) -> int:
    momentum_down = _ETA @ momentum_up
    columns = []
    for component in _COMPONENTS:
        field = _symmetric_basis_tensor(component)
        trace = float(np.sum(np.diag(_ETA) * np.diag(field)))
        trace_reversed = field - 0.5 * _ETA * trace
        columns.append(momentum_up @ trace_reversed)
    matrix = np.column_stack(columns)
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def audit_lattice_fierz_pauli_refinement(
    *,
    lattice_spacing_over_reference_length: float,
    compact_dimensionless_momentum_bound: float,
    generic_dimensionless_momentum_up: Sequence[float] = (1.1, 0.2, -0.4, 0.7),
    null_ray_frequency: float = 0.8,
    tolerance: float = 1.0e-10,
) -> LatticeFierzPauliRefinementAudit:
    """Audit one spacing and the analytic compact-limit acceptance conditions."""

    spacing = float(lattice_spacing_over_reference_length)
    momentum_bound = float(compact_dimensionless_momentum_bound)
    frequency = float(null_ray_frequency)
    tolerance = float(tolerance)
    if (
        not math.isfinite(spacing)
        or not math.isfinite(momentum_bound)
        or not math.isfinite(frequency)
        or not math.isfinite(tolerance)
        or spacing <= 0.0
        or momentum_bound <= 0.0
        or frequency <= 0.0
        or tolerance <= 0.0
    ):
        raise ValueError("spacing, bound, frequency, and tolerance must be positive finite")
    momentum = _finite_four_vector(
        "generic_dimensionless_momentum_up", generic_dimensionless_momentum_up
    )
    if np.max(np.abs(momentum)) > momentum_bound:
        raise ValueError("generic momentum must lie inside the declared compact box")
    if frequency > momentum_bound:
        raise ValueError("null-ray frequency must lie inside the declared compact box")
    low_momentum_window = spacing * momentum_bound < math.pi / 2.0
    if not low_momentum_window:
        raise ValueError("declared compact box must satisfy abar*qmax < pi/2")

    lattice_momentum = central_difference_dimensionless_momentum(
        momentum, lattice_spacing_over_reference_length=spacing
    )
    component_errors = np.abs(lattice_momentum - momentum)
    component_bound = spacing**2 * momentum_bound**3 / 6.0
    component_bound_satisfied = bool(np.max(component_errors) <= component_bound + tolerance)

    continuum_symbol = linearized_einstein_symbol(momentum)
    lattice_symbol = linearized_einstein_symbol(lattice_momentum)
    symbol_error = float(np.linalg.norm(lattice_symbol - continuum_symbol))
    uniform_symbol_bound = 130.0 * momentum_bound * component_bound
    sample_within_uniform_bound = symbol_error <= uniform_symbol_bound + tolerance
    gauge = linearized_gauge_direction_matrix(lattice_momentum)
    bianchi = linearized_bianchi_divergence_matrix(lattice_momentum)
    gauge_residual = float(np.linalg.norm(lattice_symbol @ gauge))
    bianchi_residual = float(np.linalg.norm(bianchi @ lattice_symbol))
    component_weights = np.asarray(
        [
            (1.0 if mu == nu else 2.0) * _ETA[mu, mu] * _ETA[nu, nu]
            for mu, nu in _COMPONENTS
        ]
    )
    weighted_symbol = np.diag(component_weights) @ lattice_symbol
    self_adjoint_residual = float(
        np.linalg.norm(weighted_symbol - weighted_symbol.T)
    )

    null_ray = np.asarray((frequency, 0.0, 0.0, frequency))
    lattice_null_ray = central_difference_dimensionless_momentum(
        null_ray, lattice_spacing_over_reference_length=spacing
    )
    lattice_null_down = _ETA @ lattice_null_ray
    null_norm = float(lattice_null_ray @ lattice_null_down)
    harmonic_rank = _harmonic_constraint_rank(lattice_null_ray, tolerance)
    gauge_rank = int(
        np.linalg.matrix_rank(
            linearized_gauge_direction_matrix(lattice_null_ray), tol=tolerance
        )
    )
    physical_dimension = (10 - harmonic_rank) - gauge_rank
    identities = gauge_residual <= tolerance and bianchi_residual <= tolerance
    self_adjoint = self_adjoint_residual <= tolerance
    two_polarization = (
        abs(null_norm) <= tolerance
        and harmonic_rank == 4
        and gauge_rank == 4
        and physical_dimension == 2
    )
    free_limit_closed = (
        component_bound_satisfied
        and sample_within_uniform_bound
        and identities
        and self_adjoint
    )

    return LatticeFierzPauliRefinementAudit(
        lattice_spacing_over_reference_length=spacing,
        compact_dimensionless_momentum_bound=momentum_bound,
        generic_dimensionless_momentum_up=tuple(float(value) for value in momentum),
        lattice_symbol_momentum_up=tuple(float(value) for value in lattice_momentum),
        component_symbol_error_bound=component_bound,
        maximum_component_symbol_error=float(np.max(component_errors)),
        fierz_pauli_symbol_frobenius_error=symbol_error,
        compact_uniform_fierz_pauli_symbol_error_bound=uniform_symbol_bound,
        lattice_gauge_null_residual=gauge_residual,
        lattice_bianchi_residual=bianchi_residual,
        weighted_action_self_adjoint_residual=self_adjoint_residual,
        null_ray_lattice_norm_squared=null_norm,
        null_ray_harmonic_constraint_rank=harmonic_rank,
        null_ray_residual_gauge_rank=gauge_rank,
        null_ray_physical_quotient_dimension=physical_dimension,
        all_momentum_arguments_dimensionless=True,
        central_difference_error_bound_satisfied=component_bound_satisfied,
        sample_fierz_pauli_error_within_uniform_analytic_bound=(
            sample_within_uniform_bound
        ),
        algebraic_gauge_and_bianchi_identities_preserved=identities,
        quadratic_action_symbol_self_adjoint=self_adjoint,
        compact_uniform_free_symbol_limit_closed=free_limit_closed,
        low_momentum_null_ray_two_polarization_gate_preserved=two_polarization,
        low_momentum_component_doubler_window=low_momentum_window,
        global_lattice_doublers_excluded=False,
        geometric_or_spin_foam_refinement_derived=False,
        interacting_renormalized_limit_proved=False,
        nonlinear_constraint_algebra_proved=False,
        einstein_hilbert_dominance_from_ce_proved=False,
        status=(
            "GAUGE_PRESERVING_FREE_FIERZ_PAULI_COMPACT_REFINEMENT_CLOSED"
            if free_limit_closed and two_polarization
            else "LATTICE_FIERZ_PAULI_REFINEMENT_AUDIT_FAILED"
        ),
    )
