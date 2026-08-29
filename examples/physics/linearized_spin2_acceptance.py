"""Exact two-polarization acceptance witness for linearized Einstein gravity.

The calculation fixes a dimensionless null momentum direction
``k^mu/omega = (1, 0, 0, 1)`` in four-dimensional Minkowski space and uses the
massless Fierz--Pauli/linearized-Einstein equations as a model input.  A
symmetric polarization tensor has ten components.  The harmonic-gauge
conditions leave six, and the four residual null-momentum gauge directions
are contained in that solution space.  Their quotient is therefore two
dimensional, with the usual plus and cross transverse-traceless representatives.

This closes the linearized IR acceptance theorem for the supplied action.  It
does not show that a spin-foam refinement limit produces that action, restores
the nonlinear constraint algebra, or contains no extra poles away from the
declared Fierz--Pauli model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


RationalMatrix = tuple[tuple[Fraction, ...], ...]

_ETA_DIAGONAL = (-1, 1, 1, 1)
_NULL_MOMENTUM_UP = (1, 0, 0, 1)
_NULL_MOMENTUM_DOWN = (-1, 0, 0, 1)
_SYMMETRIC_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _fraction_matrix(rows: list[list[int | Fraction]]) -> RationalMatrix:
    return tuple(tuple(Fraction(value) for value in row) for row in rows)


def _rank(matrix: RationalMatrix) -> int:
    if not matrix:
        return 0
    rows = [list(row) for row in matrix]
    row_count = len(rows)
    column_count = len(rows[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (row for row in range(pivot_row, row_count) if rows[row][column]),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot_value for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix dimensions do not compose")
    return tuple(
        tuple(
            sum(
                (left[row][index] * right[index][column] for index in range(len(right))),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _tensor_basis(component: tuple[int, int]) -> list[list[Fraction]]:
    first, second = component
    tensor = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    tensor[first][second] = Fraction(1)
    tensor[second][first] = Fraction(1)
    return tensor


def harmonic_constraint_matrix() -> RationalMatrix:
    """Return ``k^mu (epsilon_{mu nu} - eta_{mu nu} epsilon/2)=0``."""

    columns: list[list[Fraction]] = []
    for component in _SYMMETRIC_COMPONENTS:
        tensor = _tensor_basis(component)
        trace = sum(
            (Fraction(_ETA_DIAGONAL[index]) * tensor[index][index] for index in range(4)),
            Fraction(0),
        )
        column: list[Fraction] = []
        for nu in range(4):
            divergence = sum(
                (
                    Fraction(_NULL_MOMENTUM_UP[mu])
                    * (
                        tensor[mu][nu]
                        - Fraction(1, 2)
                        * (Fraction(_ETA_DIAGONAL[mu]) if mu == nu else Fraction(0))
                        * trace
                    )
                    for mu in range(4)
                ),
                Fraction(0),
            )
            column.append(divergence)
        columns.append(column)
    return tuple(
        tuple(columns[column][row] for column in range(len(columns)))
        for row in range(4)
    )


def residual_gauge_matrix() -> RationalMatrix:
    """Return columns of ``delta epsilon_{mu nu}=k_mu xi_nu+k_nu xi_mu``."""

    rows: list[list[Fraction]] = []
    for mu, nu in _SYMMETRIC_COMPONENTS:
        row = []
        for gauge_index in range(4):
            value = (
                _NULL_MOMENTUM_DOWN[mu] * int(nu == gauge_index)
                + _NULL_MOMENTUM_DOWN[nu] * int(mu == gauge_index)
            )
            row.append(Fraction(value))
        rows.append(row)
    return tuple(tuple(row) for row in rows)


def transverse_traceless_basis() -> RationalMatrix:
    """Return plus/cross representatives as two columns in component order."""

    plus = []
    cross = []
    for component in _SYMMETRIC_COMPONENTS:
        plus.append(
            Fraction(1)
            if component == (1, 1)
            else Fraction(-1)
            if component == (2, 2)
            else Fraction(0)
        )
        cross.append(Fraction(1) if component == (1, 2) else Fraction(0))
    return tuple((plus[index], cross[index]) for index in range(len(plus)))


def massive_fierz_pauli_constraint_matrix() -> RationalMatrix:
    """Return rest-frame transversality plus trace constraints for massive FP."""

    rows: list[list[Fraction]] = []
    for nu in range(4):
        row = []
        for component in _SYMMETRIC_COMPONENTS:
            tensor = _tensor_basis(component)
            row.append(tensor[0][nu])
        rows.append(row)
    trace_row = []
    for component in _SYMMETRIC_COMPONENTS:
        tensor = _tensor_basis(component)
        trace_row.append(
            sum(
                (
                    Fraction(_ETA_DIAGONAL[index]) * tensor[index][index]
                    for index in range(4)
                ),
                Fraction(0),
            )
        )
    rows.append(trace_row)
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class LinearizedSpin2AcceptanceAudit:
    spacetime_dimension: int
    symmetric_tensor_component_count: int
    dimensionless_null_momentum_direction: tuple[int, int, int, int]
    null_momentum_norm_squared: int
    harmonic_constraint_rank: int
    harmonic_solution_dimension: int
    residual_gauge_rank: int
    gauge_image_within_harmonic_kernel: bool
    harmonic_kernel_spanned_by_gauge_plus_tt: bool
    physical_quotient_dimension: int
    transverse_traceless_representative_count: int
    massive_fierz_pauli_constraint_rank: int
    massive_fierz_pauli_polarization_count: int
    arithmetic_spin2_plus_one_scalar_count: int
    supplied_linearized_einstein_action: bool
    supplied_massless_null_dispersion: bool
    exact_two_polarization_quotient_closed: bool
    nonlinear_diffeomorphism_symmetry_derived: bool
    refinement_limit_kernel_derived: bool
    refinement_uniform_ward_identities_proved: bool
    extra_poles_excluded_for_a_microscopic_model: bool
    einstein_hilbert_dominance_from_ce_proved: bool
    status: str
    claim_ceiling: str = (
        "EXACT_LINEARIZED_FIERZ_PAULI_TWO_DOF_NOT_MICROSCOPIC_EH_DERIVATION"
    )


def audit_linearized_spin2_acceptance() -> LinearizedSpin2AcceptanceAudit:
    """Perform the exact rational polarization and quotient count."""

    constraints = harmonic_constraint_matrix()
    gauge = residual_gauge_matrix()
    tt_basis = transverse_traceless_basis()
    constraint_rank = _rank(constraints)
    gauge_rank = _rank(gauge)
    constraint_on_gauge = _multiply(constraints, gauge)
    constraint_on_tt = _multiply(constraints, tt_basis)
    zero_gauge = all(value == 0 for row in constraint_on_gauge for value in row)
    zero_tt = all(value == 0 for row in constraint_on_tt for value in row)
    gauge_plus_tt = tuple(
        tuple(gauge[row][column] for column in range(4)) + tt_basis[row]
        for row in range(len(gauge))
    )
    combined_rank = _rank(gauge_plus_tt)
    solution_dimension = len(_SYMMETRIC_COMPONENTS) - constraint_rank
    quotient_dimension = solution_dimension - gauge_rank

    massive_constraints = massive_fierz_pauli_constraint_matrix()
    massive_rank = _rank(massive_constraints)
    massive_polarizations = len(_SYMMETRIC_COMPONENTS) - massive_rank
    exact_two = (
        constraint_rank == 4
        and solution_dimension == 6
        and gauge_rank == 4
        and zero_gauge
        and zero_tt
        and combined_rank == solution_dimension
        and quotient_dimension == 2
    )
    return LinearizedSpin2AcceptanceAudit(
        spacetime_dimension=4,
        symmetric_tensor_component_count=len(_SYMMETRIC_COMPONENTS),
        dimensionless_null_momentum_direction=_NULL_MOMENTUM_UP,
        null_momentum_norm_squared=0,
        harmonic_constraint_rank=constraint_rank,
        harmonic_solution_dimension=solution_dimension,
        residual_gauge_rank=gauge_rank,
        gauge_image_within_harmonic_kernel=zero_gauge,
        harmonic_kernel_spanned_by_gauge_plus_tt=zero_tt
        and combined_rank == solution_dimension,
        physical_quotient_dimension=quotient_dimension,
        transverse_traceless_representative_count=2,
        massive_fierz_pauli_constraint_rank=massive_rank,
        massive_fierz_pauli_polarization_count=massive_polarizations,
        arithmetic_spin2_plus_one_scalar_count=quotient_dimension + 1,
        supplied_linearized_einstein_action=True,
        supplied_massless_null_dispersion=True,
        exact_two_polarization_quotient_closed=exact_two,
        nonlinear_diffeomorphism_symmetry_derived=False,
        refinement_limit_kernel_derived=False,
        refinement_uniform_ward_identities_proved=False,
        extra_poles_excluded_for_a_microscopic_model=False,
        einstein_hilbert_dominance_from_ce_proved=False,
        status=(
            "EXACT_LINEARIZED_FIERZ_PAULI_TWO_POLARIZATION_GATE_CLOSED"
            if exact_two
            else "LINEARIZED_SPIN2_ACCEPTANCE_AUDIT_FAILED"
        ),
    )
