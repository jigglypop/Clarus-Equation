"""Conditional structural gates for a quantum-jump-to-branching bridge.

This module does **not** derive a GKSL generator, jump operators, or offspring
rates from the CE+SM action.  It only audits algebraic conditions after those
objects have been supplied:

* Hermiticity and positive semidefiniteness of a Kossakowski matrix;
* row-source classical off-diagonal rates extracted in a declared type basis;
* population/coherence closure of a Lindblad generator;
* invariance and constant hazard of a no-jump sector;
* the next-generation conversion ``A_ij = lifetime_i * birth_rate_ij``.

Passing every gate is therefore necessary structural evidence for the selected
basis and unraveling, not a derivation of CE+SM physics or of independent
Poisson genealogy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexMatrix: TypeAlias = NDArray[np.complex128]
FloatMatrix: TypeAlias = NDArray[np.float64]
FloatVector: TypeAlias = NDArray[np.float64]

STRUCTURAL_SCOPE = "conditional_quantum_jump_structure_only"
CONDITIONAL_PASS = "STRUCTURAL_CONDITIONAL_PASS"
CONDITIONAL_FAIL = "STRUCTURAL_CONDITIONAL_FAIL"

_PASS_CONCLUSION = (
    "Conditional algebraic gates passed for the supplied basis and unraveling. "
    "This does not derive them from the CE+SM action and does not establish "
    "independent Poisson offspring."
)
_FAIL_CONCLUSION = (
    "At least one conditional algebraic gate failed, so the supplied objects do "
    "not support this classical jump reduction. This does not assess the CE+SM "
    "action itself."
)


@dataclass(frozen=True)
class KossakowskiAudit:
    """Hermiticity and positive-semidefiniteness diagnostics."""

    dimension: int
    hermiticity_residual: float
    minimum_eigenvalue: float
    tolerance: float
    hermitian: bool
    positive_semidefinite: bool
    structural_pass: bool


@dataclass(frozen=True)
class PopulationCoherenceLeakageAudit:
    """Hilbert-Schmidt block norms of population/coherence coupling."""

    dimension: int
    population_to_coherence_norm: float
    coherence_to_population_norm: float
    tolerance: float
    populations_invariant: bool
    populations_autonomous: bool
    classical_closed: bool


@dataclass(frozen=True)
class NoJumpSectorAudit:
    """No-jump sector invariance and constant-hazard diagnostics."""

    dimension: int
    sector_rank: int
    hazard: float
    invariance_residual: float
    constant_hazard_residual: float
    tolerance: float
    invariant: bool
    constant_hazard: bool
    structural_pass: bool


@dataclass(frozen=True)
class QuantumJumpBridgeReport:
    """Serializable report whose scope explicitly excludes a CE+SM derivation."""

    schema_version: str
    scope: str
    structural_status: str
    ce_sm_derivation_complete: bool
    poisson_branching_derived: bool
    kossakowski: KossakowskiAudit
    leakage: PopulationCoherenceLeakageAudit
    no_jump: NoJumpSectorAudit
    classical_offdiagonal_rates: tuple[tuple[float, ...], ...]
    next_generation_matrix: tuple[tuple[float, ...], ...]
    assumptions_not_audited: tuple[str, ...]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _as_square_complex(matrix: ArrayLike, *, name: str) -> ComplexMatrix:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} entries must be finite")
    return value


def _as_hermitian(
    matrix: ArrayLike,
    *,
    name: str,
    tolerance: float,
) -> ComplexMatrix:
    value = _as_square_complex(matrix, name=name)
    residual = _spectral_norm(value - value.conj().T)
    if residual > tolerance:
        raise ValueError(f"{name} must be Hermitian within tolerance")
    return 0.5 * (value + value.conj().T)


def _as_jump_operators(
    jump_operators: ArrayLike,
    *,
    dimension: int | None = None,
) -> tuple[ComplexMatrix, ...]:
    values = np.asarray(jump_operators, dtype=np.complex128)
    if values.ndim == 2:
        values = values[np.newaxis, :, :]
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[1] != values.shape[2]
        or values.shape[1] == 0
    ):
        raise ValueError(
            "jump_operators must contain at least one non-empty square matrix"
        )
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError("jump operator dimension does not match the Hamiltonian")
    if not np.all(np.isfinite(values)):
        raise ValueError("jump operator entries must be finite")
    return tuple(values[index] for index in range(values.shape[0]))


def _as_nonnegative_square(matrix: ArrayLike, *, name: str) -> FloatMatrix:
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError(f"{name} entries must be real")
    value = np.asarray(np.real(raw), dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} entries must be finite")
    if np.any(value < 0.0):
        raise ValueError(f"{name} entries must be non-negative")
    return value


def _as_nonnegative_vector(
    vector: ArrayLike,
    *,
    size: int,
    name: str,
) -> FloatVector:
    raw = np.asarray(vector)
    if np.iscomplexobj(raw) and np.any(np.abs(np.imag(raw)) > 0.0):
        raise ValueError(f"{name} entries must be real")
    value = np.asarray(np.real(raw), dtype=np.float64)
    if value.ndim != 1 or value.shape[0] != size:
        raise ValueError(f"{name} must be a vector of length {size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} entries must be finite")
    if np.any(value < 0.0):
        raise ValueError(f"{name} entries must be non-negative")
    return value


def _spectral_norm(matrix: ComplexMatrix) -> float:
    if matrix.size == 0 or 0 in matrix.shape:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2))


def _lindblad_action(
    state: ComplexMatrix,
    hamiltonian: ComplexMatrix,
    jump_operators: tuple[ComplexMatrix, ...],
) -> ComplexMatrix:
    derivative = -1.0j * (hamiltonian @ state - state @ hamiltonian)
    for jump in jump_operators:
        gamma = jump.conj().T @ jump
        derivative += (
            jump @ state @ jump.conj().T
            - 0.5 * gamma @ state
            - 0.5 * state @ gamma
        )
    return derivative


def _matrix_as_tuple(matrix: FloatMatrix) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(matrix, dtype=np.float64)
    )


def audit_kossakowski(
    matrix: ArrayLike,
    *,
    tolerance: float = 1.0e-12,
) -> KossakowskiAudit:
    """Audit a candidate Kossakowski matrix without repairing it.

    The minimum eigenvalue is evaluated on the Hermitian part for a useful
    diagnostic, but ``structural_pass`` also requires the original matrix to be
    Hermitian within ``tolerance``.
    """
    threshold = _validate_tolerance(tolerance)
    value = _as_square_complex(matrix, name="Kossakowski matrix")
    hermitian_part = 0.5 * (value + value.conj().T)
    hermiticity_residual = _spectral_norm(value - value.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_part)))
    hermitian = hermiticity_residual <= threshold
    positive_semidefinite = minimum_eigenvalue >= -threshold
    return KossakowskiAudit(
        dimension=value.shape[0],
        hermiticity_residual=hermiticity_residual,
        minimum_eigenvalue=minimum_eigenvalue,
        tolerance=threshold,
        hermitian=hermitian,
        positive_semidefinite=positive_semidefinite,
        structural_pass=hermitian and positive_semidefinite,
    )


def classical_offdiagonal_rates(jump_operators: ArrayLike) -> FloatMatrix:
    """Extract row-source rates in the declared computational type basis.

    For source type ``i`` and target type ``j``,

    ``rates[i, j] = sum_r |<j|L_r|i>|^2``.

    The diagonal is set to zero because this function returns transition rates,
    not a continuous-time generator and not an offspring matrix.
    """
    operators = _as_jump_operators(jump_operators)
    dimension = operators[0].shape[0]
    rates = np.zeros((dimension, dimension), dtype=np.float64)
    for jump in operators:
        rates += np.abs(jump.T) ** 2
    np.fill_diagonal(rates, 0.0)
    return rates


def audit_population_coherence_leakage(
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    *,
    tolerance: float = 1.0e-12,
) -> PopulationCoherenceLeakageAudit:
    """Audit whether populations form an invariant autonomous classical block.

    The norms are induced Hilbert-Schmidt 2-norms of
    ``(I-P) L P`` and ``P L (I-P)``, where ``P`` removes off-diagonal
    matrix entries in the declared type basis.
    """
    threshold = _validate_tolerance(tolerance)
    hamiltonian_matrix = _as_hermitian(
        hamiltonian,
        name="hamiltonian",
        tolerance=threshold,
    )
    dimension = hamiltonian_matrix.shape[0]
    operators = _as_jump_operators(jump_operators, dimension=dimension)
    coherence_indices = [
        (row, column)
        for row in range(dimension)
        for column in range(dimension)
        if row != column
    ]

    population_to_coherence = np.zeros(
        (len(coherence_indices), dimension),
        dtype=np.complex128,
    )
    for source in range(dimension):
        basis = np.zeros((dimension, dimension), dtype=np.complex128)
        basis[source, source] = 1.0
        derivative = _lindblad_action(basis, hamiltonian_matrix, operators)
        for index, (row, column) in enumerate(coherence_indices):
            population_to_coherence[index, source] = derivative[row, column]

    coherence_to_population = np.zeros(
        (dimension, len(coherence_indices)),
        dtype=np.complex128,
    )
    for index, (row, column) in enumerate(coherence_indices):
        basis = np.zeros((dimension, dimension), dtype=np.complex128)
        basis[row, column] = 1.0
        derivative = _lindblad_action(basis, hamiltonian_matrix, operators)
        coherence_to_population[:, index] = np.diag(derivative)

    outgoing_norm = _spectral_norm(population_to_coherence)
    incoming_norm = _spectral_norm(coherence_to_population)
    populations_invariant = outgoing_norm <= threshold
    populations_autonomous = incoming_norm <= threshold
    return PopulationCoherenceLeakageAudit(
        dimension=dimension,
        population_to_coherence_norm=outgoing_norm,
        coherence_to_population_norm=incoming_norm,
        tolerance=threshold,
        populations_invariant=populations_invariant,
        populations_autonomous=populations_autonomous,
        classical_closed=populations_invariant and populations_autonomous,
    )


def audit_no_jump_sector(
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    sector_projector: ArrayLike,
    *,
    tolerance: float = 1.0e-12,
) -> NoJumpSectorAudit:
    """Audit an invariant sector with state-independent no-jump hazard.

    The reported hazard is the best scalar ``kappa`` on the sector:
    ``Tr(P Gamma) / rank(P)``.  Exact exponential no-jump survival on every
    state in the sector requires both ``(I-P) H_eff P = 0`` and
    ``P Gamma P = kappa P``.
    """
    threshold = _validate_tolerance(tolerance)
    hamiltonian_matrix = _as_hermitian(
        hamiltonian,
        name="hamiltonian",
        tolerance=threshold,
    )
    dimension = hamiltonian_matrix.shape[0]
    operators = _as_jump_operators(jump_operators, dimension=dimension)
    projector = _as_hermitian(
        sector_projector,
        name="sector_projector",
        tolerance=threshold,
    )
    if projector.shape != hamiltonian_matrix.shape:
        raise ValueError("sector_projector dimension does not match the Hamiltonian")
    idempotence_residual = _spectral_norm(projector @ projector - projector)
    if idempotence_residual > threshold:
        raise ValueError("sector_projector must be idempotent within tolerance")
    sector_rank = int(round(float(np.real(np.trace(projector)))))
    if sector_rank < 1:
        raise ValueError("sector_projector must have positive rank")

    gamma = np.zeros_like(hamiltonian_matrix)
    for jump in operators:
        gamma += jump.conj().T @ jump
    effective_hamiltonian = hamiltonian_matrix - 0.5j * gamma
    complement = np.eye(dimension, dtype=np.complex128) - projector
    invariance_residual = _spectral_norm(
        complement @ effective_hamiltonian @ projector
    )
    hazard = float(np.real(np.trace(projector @ gamma)) / sector_rank)
    constant_hazard_residual = _spectral_norm(
        projector @ gamma @ projector - hazard * projector
    )
    invariant = invariance_residual <= threshold
    constant_hazard = constant_hazard_residual <= threshold
    return NoJumpSectorAudit(
        dimension=dimension,
        sector_rank=sector_rank,
        hazard=hazard,
        invariance_residual=invariance_residual,
        constant_hazard_residual=constant_hazard_residual,
        tolerance=threshold,
        invariant=invariant,
        constant_hazard=constant_hazard,
        structural_pass=invariant and constant_hazard,
    )


def next_generation_from_constant_rates(
    birth_rates: ArrayLike,
    mean_lifetimes: ArrayLike,
) -> FloatMatrix:
    """Convert constant row-source birth rates to a next-generation matrix.

    ``A_ij = mean_lifetimes[i] * birth_rates[i, j]``.

    The caller must establish that the inputs count offspring births.  A Markov
    transition rate extracted from jump operators is not automatically a birth
    rate, and this function does not test Poisson or genealogical independence.
    """
    rates = _as_nonnegative_square(birth_rates, name="birth_rates")
    lifetimes = _as_nonnegative_vector(
        mean_lifetimes,
        size=rates.shape[0],
        name="mean_lifetimes",
    )
    return lifetimes[:, np.newaxis] * rates


def structural_bridge_report(
    *,
    kossakowski_matrix: ArrayLike,
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    sector_projector: ArrayLike,
    birth_rates: ArrayLike,
    mean_lifetimes: ArrayLike,
    tolerance: float = 1.0e-12,
) -> QuantumJumpBridgeReport:
    """Run the five conditional structural gates and preserve their scope."""
    kossakowski = audit_kossakowski(
        kossakowski_matrix,
        tolerance=tolerance,
    )
    leakage = audit_population_coherence_leakage(
        hamiltonian,
        jump_operators,
        tolerance=tolerance,
    )
    no_jump = audit_no_jump_sector(
        hamiltonian,
        jump_operators,
        sector_projector,
        tolerance=tolerance,
    )
    classical_rates = classical_offdiagonal_rates(jump_operators)
    next_generation = next_generation_from_constant_rates(
        birth_rates,
        mean_lifetimes,
    )
    if next_generation.shape != classical_rates.shape:
        raise ValueError(
            "birth-rate dimension must match the supplied jump-operator dimension"
        )

    structural_pass = (
        kossakowski.structural_pass
        and leakage.classical_closed
        and no_jump.structural_pass
    )
    return QuantumJumpBridgeReport(
        schema_version="1.0",
        scope=STRUCTURAL_SCOPE,
        structural_status=CONDITIONAL_PASS if structural_pass else CONDITIONAL_FAIL,
        ce_sm_derivation_complete=False,
        poisson_branching_derived=False,
        kossakowski=kossakowski,
        leakage=leakage,
        no_jump=no_jump,
        classical_offdiagonal_rates=_matrix_as_tuple(classical_rates),
        next_generation_matrix=_matrix_as_tuple(next_generation),
        assumptions_not_audited=(
            "CE+SM action, background, gauge fixing, and system-bath split",
            "Born-Markov-secular or Davies weak-coupling limit",
            "consistency between the Kossakowski matrix and jump operators",
            "physical selection of the type basis and jump unraveling",
            "interpretation of supplied rates as offspring births",
            "independent Poisson streams, reset, and genealogical independence",
        ),
        conclusion=_PASS_CONCLUSION if structural_pass else _FAIL_CONCLUSION,
    )
