'''Construct an energy-resolved quantum instrument record kernel.

This module supplies one explicit upstream realization for the marked record
contract in causal_record_dust_bridge.  The instrument choice is part of the
input: a nonselective channel does not determine a unique outcome tree.

For a normalized state rho_a and Kraus branch K_b|a,

    P_b|a = Tr(K_b|a rho_a K_b|a^dagger)

is a dimensionless Born transition probability.  A complete family has a
harmonic energy ledger when

    sum_b K_b|a^dagger (H + Delta E_b I) K_b|a = H.

For QND records Delta E_b is zero and every Kraus operator commutes with H.
For a measured energy-conserving collision, Delta E_b is the fresh ancilla's
energy gain; the system energy may then change while system-plus-record energy
remains harmonic.

An internal node energy is still only a conditional expectation.  A terminal
record may feed the exact mass-shell bridge only when Var(H) vanishes there.
The spectrum of H supplies the physical energy unit; no dimensionless Born
probability can determine that absolute scale.

For every accepted finite tree, the module also constructs the standard
append-only history register.  Its orthonormal basis is indexed by complete
declared outcome histories, including zero- or subthreshold-probability
histories omitted from the supported flow.  The diagonal register algebra is
therefore exactly commutative, while the chained Kraus operators give a
numerical Stinespring-isometry and Born-cylinder-probability certificate.
This is an abstract record algebra construction; it does not derive a robust
physical pointer, a spacetime embedding, or a mass-shell mark.

When only H is supplied, ``construct_luders_energy_instrument`` gives the
coarsest distinct-energy spectral PVM as one canonical realization.  This
does not make the physical instrument unique: refinements and unitary motion
inside a degenerate energy sector remain additional choices.  A numerically
ambiguous near-degeneracy is rejected instead of being silently merged.
'''

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from examples.physics.causal_record_dust_bridge import (
    CausalRecordFlow,
    CausalRecordNode,
    CausalTransition,
    construct_conserved_record_flow,
)


DEFAULT_TOLERANCE = 1.0e-10  # Dimensionless relative numerical tolerance.


@dataclass(frozen=True)
class KrausBranch:
    '''One explicitly named outcome of a chosen quantum instrument.'''

    target: str
    label: str
    operator: np.ndarray
    energy_transfer: float = 0.0


@dataclass(frozen=True)
class RecordInstrument:
    '''The chosen Kraus decomposition at one record-tree node.'''

    node: str
    instrument_id: str
    branches: tuple[KrausBranch, ...]


@dataclass(frozen=True)
class ConditionalEnergyState:
    node: str
    density: np.ndarray
    probability_from_root: float
    system_energy_expectation: float
    cumulative_energy_transfer: float
    energy_expectation: float
    energy_variance: float


@dataclass(frozen=True)
class PersistentRecordHistory:
    '''One orthogonal basis label of the append-only history register.'''

    basis_index: int
    terminal_node: str
    path: tuple[str, ...]
    probability: float
    supported_by_root: bool


@dataclass(frozen=True)
class ClassicalRecordAlgebraCertificate:
    '''Finite Stinespring/history certificate for a declared instrument tree.

    The history basis and its diagonal algebra are mathematical constructions.
    The physical_pointer_dynamics_derived flag remains false because
    orthogonality and append-only labelling do not prove that a real apparatus
    stores the record robustly.
    '''

    histories: tuple[PersistentRecordHistory, ...]
    node_probabilities: tuple[tuple[str, float], ...]
    global_isometry_residual: float
    probability_normalization_residual: float
    max_history_probability_residual: float
    max_prefix_probability_residual: float
    orthogonal_history_basis: bool
    commutative_diagonal_algebra: bool
    append_only_history_labels: bool
    physical_pointer_dynamics_derived: bool

    def probability(self, node: str) -> float:
        for candidate, probability in self.node_probabilities:
            if candidate == node:
                return probability
        raise KeyError(node)


@dataclass(frozen=True)
class QuantumKernelCertificate:
    '''Certified Born kernel and its downstream Kirchhoff flow.'''

    root: str
    nodes: tuple[CausalRecordNode, ...]
    transitions: tuple[CausalTransition, ...]
    conditional_states: tuple[ConditionalEnergyState, ...]
    instrument_ids: tuple[tuple[str, str], ...]
    terminal_nodes: tuple[str, ...]
    terminal_energy_resolved: bool
    qnd_required: bool
    hamiltonian_energy_scale: float
    hamiltonian_energy_min: float
    hamiltonian_energy_max: float
    max_completeness_residual: float
    max_energy_channel_residual: float
    max_relative_energy_channel_residual: float
    max_qnd_commutator_residual: float
    max_relative_qnd_commutator_residual: float
    support_probability_tolerance: float
    record_algebra: ClassicalRecordAlgebraCertificate
    flow: CausalRecordFlow

    def state(self, node: str) -> ConditionalEnergyState:
        for candidate in self.conditional_states:
            if candidate.node == node:
                return candidate
        raise KeyError(node)


@dataclass(frozen=True)
class LudersInstrumentConstruction:
    '''Numerical certificate for the coarsest spectral Lüders instrument.'''

    instrument: RecordInstrument
    spectral_energies: tuple[float, ...]
    spectral_multiplicities: tuple[int, ...]
    hamiltonian_energy_scale: float
    max_projector_idempotence_residual: float
    max_projector_orthogonality_residual: float
    resolution_of_identity_residual: float
    max_hamiltonian_commutator_residual: float
    max_relative_hamiltonian_commutator_residual: float
    max_spectral_cluster_width: float
    max_relative_spectral_cluster_width: float
    max_eigenprojector_residual: float
    max_relative_eigenprojector_residual: float


@dataclass(frozen=True)
class EnergyConservingCollisionConstruction:
    '''Kraus instrument induced by one measured fresh-ancilla collision.'''

    instrument: RecordInstrument
    initial_pointer_index: int
    initial_pointer_energy: float
    pointer_energies: tuple[float, ...]
    branch_energy_transfers: tuple[float, ...]
    pointer_basis_unitarity_residual: float
    collision_unitarity_residual: float
    total_energy_commutator_residual: float
    relative_total_energy_commutator_residual: float
    max_relative_pointer_energy_variance: float
    kraus_completeness_residual: float
    ledger_identity_residual: float
    relative_ledger_identity_residual: float
    physical_pointer_persistence_derived: bool


def _square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f'{name} must be a non-empty square matrix')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{name} must have finite entries')
    return matrix


def _validate_density(
    value: np.ndarray,
    *,
    dimension: int,
    tolerance: float,
    name: str,
) -> np.ndarray:
    density = _square_matrix(value, name)
    if density.shape != (dimension, dimension):
        raise ValueError(f'{name} has the wrong Hilbert-space dimension')
    if np.linalg.norm(density - density.conj().T, ord='fro') > tolerance:
        raise ValueError(f'{name} must be Hermitian')
    eigenvalues = np.linalg.eigvalsh(density)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f'{name} must be positive semidefinite')
    trace = np.trace(density)
    if abs(float(trace.imag)) > tolerance or not math.isclose(
        float(trace.real), 1.0, rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f'{name} must have unit trace')
    return density


def _real_trace(
    value: np.ndarray,
    *,
    tolerance: float,
    name: str,
    scale: float = 1.0,
) -> float:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'{name} trace scale must be finite and positive')
    trace = np.trace(value)
    if abs(float(trace.imag)) > tolerance * scale:
        raise ArithmeticError(f'{name} has a non-real trace')
    result = float(trace.real)
    if not math.isfinite(result):
        raise ArithmeticError(f'{name} trace is not finite')
    return result


def _energy_state(
    node: str,
    density: np.ndarray,
    probability_from_root: float,
    hamiltonian: np.ndarray,
    *,
    cumulative_energy_transfer: float,
    energy_scale: float,
    tolerance: float,
) -> ConditionalEnergyState:
    system_energy = _real_trace(
        hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy at {node}',
        scale=energy_scale,
    )
    second_moment = _real_trace(
        hamiltonian @ hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy second moment at {node}',
        scale=energy_scale**2,
    )
    variance = second_moment - system_energy * system_energy
    if variance < -tolerance * energy_scale**2:
        raise ArithmeticError(f'energy variance at {node} is negative')
    return ConditionalEnergyState(
        node=node,
        density=density,
        probability_from_root=probability_from_root,
        system_energy_expectation=system_energy,
        cumulative_energy_transfer=cumulative_energy_transfer,
        energy_expectation=system_energy + cumulative_energy_transfer,
        energy_variance=max(0.0, variance),
    )


def construct_luders_energy_instrument(
    hamiltonian: np.ndarray,
    *,
    node: str = 'root',
    instrument_id: str = 'spectral-luders-energy',
    target_prefix: str = 'energy-sector',
    tolerance: float = DEFAULT_TOLERANCE,
) -> LudersInstrumentConstruction:
    '''Construct the coarsest distinct-energy spectral PVM of finite H.

    The spectral projectors are canonical functions of the supplied finite,
    positive Hermitian Hamiltonian.  The Lüders operations use one projector
    as the sole Kraus operator for each distinct energy.  Canonicality here is
    only spectral-PVM canonicality; it excludes neither a finer pointer inside
    a degenerate sector nor another QND operation within that sector.  Exact
    equality in the returned numerical spectrum defines a degenerate cluster.
    Distinct eigenvalues closer than the declared relative numerical tolerance are
    rejected as ambiguous instead of being merged into a non-sharp sector.
    '''

    if not node or not instrument_id or not target_prefix:
        raise ValueError('node, instrument identifier, and target prefix are required')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    hamiltonian_matrix_scale = float(np.linalg.norm(hamiltonian_matrix, ord='fro'))
    if hamiltonian_matrix_scale <= 0.0:
        raise ValueError('Hamiltonian must have a non-zero physical energy scale')
    if (
        np.linalg.norm(
            hamiltonian_matrix - hamiltonian_matrix.conj().T,
            ord='fro',
        )
        / hamiltonian_matrix_scale
        > tolerance
    ):
        raise ValueError('Hamiltonian must be Hermitian')
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    energy_scale = float(np.max(np.abs(eigenvalues)))
    if float(np.min(eigenvalues)) <= tolerance * energy_scale:
        raise ValueError('Hamiltonian must be strictly positive on this contract')

    clusters: list[list[int]] = []
    for index, energy in enumerate(eigenvalues):
        if clusters:
            reference = float(eigenvalues[clusters[-1][0]])
            if float(energy) == reference:
                clusters[-1].append(index)
                continue
            if abs(float(energy) - reference) <= tolerance * energy_scale:
                raise ValueError(
                    'Hamiltonian has a numerically ambiguous near-degeneracy; '
                    'lower the tolerance or supply an exactly resolved representation'
                )
        clusters.append([index])

    dimension = hamiltonian_matrix.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    projectors: list[np.ndarray] = []
    energies: list[float] = []
    multiplicities: list[int] = []
    for cluster in clusters:
        basis = eigenvectors[:, cluster]
        projector = basis @ basis.conj().T
        projectors.append(projector)
        energies.append(float(np.mean(eigenvalues[cluster])))
        multiplicities.append(len(cluster))

    max_idempotence = max(
        float(np.linalg.norm(projector @ projector - projector, ord='fro'))
        for projector in projectors
    )
    max_orthogonality = max(
        (
            float(np.linalg.norm(left @ right, ord='fro'))
            for index, left in enumerate(projectors)
            for right in projectors[index + 1 :]
        ),
        default=0.0,
    )
    resolution_residual = float(
        np.linalg.norm(sum(projectors, start=np.zeros_like(identity)) - identity, ord='fro')
    )
    max_commutator = max(
        float(
            np.linalg.norm(
                projector @ hamiltonian_matrix - hamiltonian_matrix @ projector,
                ord='fro',
            )
        )
        for projector in projectors
    )
    max_cluster_width = max(
        float(np.max(eigenvalues[cluster]) - np.min(eigenvalues[cluster]))
        for cluster in clusters
    )
    max_eigenprojector_residual = max(
        float(
            np.linalg.norm(
                hamiltonian_matrix @ projector - energy * projector,
                ord='fro',
            )
        )
        for energy, projector in zip(energies, projectors)
    )
    max_relative_commutator = max_commutator / energy_scale
    max_relative_cluster_width = max_cluster_width / energy_scale
    max_relative_eigenprojector_residual = (
        max_eigenprojector_residual / energy_scale
    )
    numerical_limit = tolerance * max(1.0, float(dimension))
    if max(
        max_idempotence,
        max_orthogonality,
        resolution_residual,
        max_relative_commutator,
        max_relative_cluster_width,
        max_relative_eigenprojector_residual,
    ) > numerical_limit:
        raise ArithmeticError('spectral projector construction exceeded tolerance')

    branches = tuple(
        KrausBranch(
            target=f'{target_prefix}-{index}',
            label=f'Luders energy {energy:.16g}',
            operator=projector,
        )
        for index, (energy, projector) in enumerate(zip(energies, projectors))
    )
    return LudersInstrumentConstruction(
        instrument=RecordInstrument(
            node=node,
            instrument_id=instrument_id,
            branches=branches,
        ),
        spectral_energies=tuple(energies),
        spectral_multiplicities=tuple(multiplicities),
        hamiltonian_energy_scale=energy_scale,
        max_projector_idempotence_residual=max_idempotence,
        max_projector_orthogonality_residual=max_orthogonality,
        resolution_of_identity_residual=resolution_residual,
        max_hamiltonian_commutator_residual=max_commutator,
        max_relative_hamiltonian_commutator_residual=max_relative_commutator,
        max_spectral_cluster_width=max_cluster_width,
        max_relative_spectral_cluster_width=max_relative_cluster_width,
        max_eigenprojector_residual=max_eigenprojector_residual,
        max_relative_eigenprojector_residual=max_relative_eigenprojector_residual,
    )


def construct_energy_conserving_collision_instrument(
    system_hamiltonian: np.ndarray,
    ancilla_hamiltonian: np.ndarray,
    collision_unitary: np.ndarray,
    *,
    pointer_basis: np.ndarray | None = None,
    initial_pointer_index: int = 0,
    outcome_targets: Sequence[str] | None = None,
    outcome_labels: Sequence[str] | None = None,
    node: str = 'root',
    instrument_id: str = 'energy-conserving-collision',
    target_prefix: str = 'ancilla-energy',
    require_ground_input: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> EnergyConservingCollisionConstruction:
    '''Derive a transfer-labelled instrument from one local collision.

    The tensor ordering is system then ancilla.  The supplied unitary must
    commute with H_system + H_ancilla.  The columns of pointer_basis must be an
    orthonormal energy eigenbasis of the ancilla, and the fresh ancilla starts
    in one declared basis vector.  Measuring the outgoing ancilla then gives

        K_r = <r| U |a0>,
        Delta E_r = e_r - e_a0,

    and the exact operator ledger

        sum_r K_r^dagger (H_system + Delta E_r I) K_r = H_system.

    With a ground-state input every transfer is non-negative up to numerical
    precision.  This constructs the abstract outcome instrument and its energy
    receipt; it does not derive a durable apparatus pointer or event location.
    '''

    if not node or not instrument_id or not target_prefix:
        raise ValueError('node, instrument identifier, and target prefix are required')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    system_h = _square_matrix(system_hamiltonian, 'system Hamiltonian')
    ancilla_h = _square_matrix(ancilla_hamiltonian, 'ancilla Hamiltonian')
    system_dimension = system_h.shape[0]
    ancilla_dimension = ancilla_h.shape[0]
    if outcome_targets is None:
        targets = tuple(
            f'{target_prefix}-{index}' for index in range(ancilla_dimension)
        )
    else:
        targets = tuple(outcome_targets)
    if outcome_labels is None:
        labels: tuple[str, ...] | None = None
    else:
        labels = tuple(outcome_labels)
    if len(targets) != ancilla_dimension or (
        labels is not None and len(labels) != ancilla_dimension
    ):
        raise ValueError('collision outcomes must cover the ancilla pointer basis')
    if any(not target for target in targets) or len(set(targets)) != len(targets):
        raise ValueError('collision outcome targets must be unique and non-empty')
    if labels is not None and any(not label for label in labels):
        raise ValueError('collision outcome labels must be non-empty')
    system_scale = float(np.linalg.norm(system_h, ord='fro'))
    ancilla_scale = float(np.linalg.norm(ancilla_h, ord='fro'))
    if system_scale <= 0.0 or ancilla_scale <= 0.0:
        raise ValueError('system and ancilla need non-zero energy scales')
    if (
        np.linalg.norm(system_h - system_h.conj().T, ord='fro') / system_scale
        > tolerance
    ):
        raise ValueError('system Hamiltonian must be Hermitian')
    if (
        np.linalg.norm(ancilla_h - ancilla_h.conj().T, ord='fro') / ancilla_scale
        > tolerance
    ):
        raise ValueError('ancilla Hamiltonian must be Hermitian')
    system_spectrum = np.linalg.eigvalsh(system_h)
    system_spectral_scale = float(np.max(np.abs(system_spectrum)))
    if float(np.min(system_spectrum)) <= tolerance * system_spectral_scale:
        raise ValueError('system Hamiltonian must be strictly positive')
    ancilla_spectrum = np.linalg.eigvalsh(ancilla_h)
    ancilla_spectral_scale = float(np.max(np.abs(ancilla_spectrum)))
    if float(np.min(ancilla_spectrum)) < -tolerance * ancilla_spectral_scale:
        raise ValueError('ancilla Hamiltonian must be positive semidefinite')

    if pointer_basis is None:
        pointer = np.eye(ancilla_dimension, dtype=np.complex128)
    else:
        pointer = _square_matrix(pointer_basis, 'pointer basis')
        if pointer.shape != (ancilla_dimension, ancilla_dimension):
            raise ValueError('pointer basis has the wrong ancilla dimension')
    ancilla_identity = np.eye(ancilla_dimension, dtype=np.complex128)
    pointer_basis_unitarity_residual = float(
        np.linalg.norm(pointer.conj().T @ pointer - ancilla_identity, ord='fro')
    )
    numerical_limit = tolerance * max(
        1.0,
        float(system_dimension),
        float(ancilla_dimension),
    )
    if pointer_basis_unitarity_residual > numerical_limit:
        raise ValueError('pointer basis must be unitary')

    ancilla_h_pointer = pointer.conj().T @ ancilla_h @ pointer
    pointer_energies = tuple(
        float(ancilla_h_pointer[index, index].real)
        for index in range(ancilla_dimension)
    )
    ancilla_h_squared_pointer = (
        pointer.conj().T @ ancilla_h @ ancilla_h @ pointer
    )
    pointer_variances = tuple(
        max(
            0.0,
            float(ancilla_h_squared_pointer[index, index].real)
            - pointer_energies[index] ** 2,
        )
        for index in range(ancilla_dimension)
    )
    max_relative_pointer_energy_variance = (
        max(pointer_variances, default=0.0) / ancilla_spectral_scale**2
    )
    off_diagonal_pointer_h = ancilla_h_pointer - np.diag(
        np.diag(ancilla_h_pointer)
    )
    if (
        np.linalg.norm(off_diagonal_pointer_h, ord='fro') / ancilla_scale
        > numerical_limit
        or max_relative_pointer_energy_variance > numerical_limit
    ):
        raise ValueError('pointer basis must resolve sharp ancilla energies')
    if not 0 <= initial_pointer_index < ancilla_dimension:
        raise ValueError('initial pointer index is outside the ancilla basis')
    initial_pointer_energy = pointer_energies[initial_pointer_index]
    if (
        require_ground_input
        and initial_pointer_energy
        > min(pointer_energies) + tolerance * ancilla_spectral_scale
    ):
        raise ValueError('fresh ancilla must start in a pointer-basis ground state')

    collision = _square_matrix(collision_unitary, 'collision unitary')
    joint_dimension = system_dimension * ancilla_dimension
    if collision.shape != (joint_dimension, joint_dimension):
        raise ValueError('collision unitary has the wrong joint dimension')
    joint_identity = np.eye(joint_dimension, dtype=np.complex128)
    collision_unitarity_residual = float(
        np.linalg.norm(collision.conj().T @ collision - joint_identity, ord='fro')
    )
    if collision_unitarity_residual > numerical_limit:
        raise ValueError('collision operator must be unitary')

    system_identity = np.eye(system_dimension, dtype=np.complex128)
    total_hamiltonian = (
        np.kron(system_h, ancilla_identity)
        + np.kron(system_identity, ancilla_h)
    )
    total_energy_scale = float(np.linalg.norm(total_hamiltonian, ord='fro'))
    total_energy_commutator_residual = float(
        np.linalg.norm(
            collision @ total_hamiltonian - total_hamiltonian @ collision,
            ord='fro',
        )
    )
    relative_total_energy_commutator_residual = (
        total_energy_commutator_residual / total_energy_scale
    )
    if relative_total_energy_commutator_residual > numerical_limit:
        raise ValueError('collision unitary must conserve total energy')

    basis_change = np.kron(system_identity, pointer)
    collision_pointer = basis_change.conj().T @ collision @ basis_change
    collision_tensor = collision_pointer.reshape(
        system_dimension,
        ancilla_dimension,
        system_dimension,
        ancilla_dimension,
    )
    kraus_operators = tuple(
        collision_tensor[:, outcome, :, initial_pointer_index]
        for outcome in range(ancilla_dimension)
    )
    kraus_completeness = sum(
        (operator.conj().T @ operator for operator in kraus_operators),
        start=np.zeros_like(system_identity),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(kraus_completeness - system_identity, ord='fro')
    )
    if kraus_completeness_residual > numerical_limit:
        raise ArithmeticError('collision Kraus family is not complete')

    energy_transfers = []
    for pointer_energy in pointer_energies:
        transfer = pointer_energy - initial_pointer_energy
        if abs(transfer) <= tolerance * ancilla_spectral_scale:
            transfer = 0.0
        if require_ground_input and transfer < 0.0:
            raise ArithmeticError('ground-input collision produced negative transfer')
        energy_transfers.append(transfer)
    ledger_operator = sum(
        (
            operator.conj().T
            @ (system_h + transfer * system_identity)
            @ operator
            for operator, transfer in zip(kraus_operators, energy_transfers)
        ),
        start=np.zeros_like(system_h),
    )
    ledger_identity_residual = float(
        np.linalg.norm(ledger_operator - system_h, ord='fro')
    )
    relative_ledger_identity_residual = ledger_identity_residual / system_scale
    if relative_ledger_identity_residual > numerical_limit:
        raise ArithmeticError('collision energy-transfer ledger identity failed')

    branches = tuple(
        KrausBranch(
            target=targets[index],
            label=(
                labels[index]
                if labels is not None
                else f'ancilla energy {energy:.16g}'
            ),
            operator=operator,
            energy_transfer=transfer,
        )
        for index, (energy, transfer, operator) in enumerate(
            zip(pointer_energies, energy_transfers, kraus_operators)
        )
    )
    return EnergyConservingCollisionConstruction(
        instrument=RecordInstrument(
            node=node,
            instrument_id=instrument_id,
            branches=branches,
        ),
        initial_pointer_index=initial_pointer_index,
        initial_pointer_energy=initial_pointer_energy,
        pointer_energies=pointer_energies,
        branch_energy_transfers=tuple(energy_transfers),
        pointer_basis_unitarity_residual=pointer_basis_unitarity_residual,
        collision_unitarity_residual=collision_unitarity_residual,
        total_energy_commutator_residual=total_energy_commutator_residual,
        relative_total_energy_commutator_residual=(
            relative_total_energy_commutator_residual
        ),
        max_relative_pointer_energy_variance=(
            max_relative_pointer_energy_variance
        ),
        kraus_completeness_residual=kraus_completeness_residual,
        ledger_identity_residual=ledger_identity_residual,
        relative_ledger_identity_residual=relative_ledger_identity_residual,
        physical_pointer_persistence_derived=False,
    )


def apply_nonselective_channel(
    density: np.ndarray,
    branches: Sequence[KrausBranch],
) -> np.ndarray:
    '''Apply a declared Kraus family while discarding its outcome label.'''

    state = np.asarray(density, dtype=np.complex128)
    return sum(
        (
            np.asarray(branch.operator, dtype=np.complex128)
            @ state
            @ np.asarray(branch.operator, dtype=np.complex128).conj().T
            for branch in branches
        ),
        start=np.zeros_like(state),
    )


def _certify_classical_record_algebra(
    *,
    root: str,
    root_density: np.ndarray,
    dimension: int,
    branches_by_node: dict[str, tuple[tuple[KrausBranch, np.ndarray], ...]],
    state_by_node: dict[str, ConditionalEnergyState],
    supported_terminal_nodes: tuple[str, ...],
    tolerance: float,
) -> ClassicalRecordAlgebraCertificate:
    '''Construct the full orthogonal history register of a validated tree.'''

    identity = np.eye(dimension, dtype=np.complex128)
    stack: list[tuple[str, tuple[str, ...], np.ndarray]] = [
        (root, (root,), identity)
    ]
    raw_histories: list[tuple[str, tuple[str, ...], np.ndarray, float]] = []
    while stack:
        node, path, chain = stack.pop()
        branches = branches_by_node.get(node)
        if branches is None:
            unnormalized = chain @ root_density @ chain.conj().T
            probability = _real_trace(
                unnormalized,
                tolerance=tolerance,
                name=f'history probability at {node}',
            )
            if probability < -tolerance:
                raise ArithmeticError('history probability is negative')
            raw_histories.append((node, path, chain, max(0.0, probability)))
            continue
        for branch, operator in reversed(branches):
            stack.append(
                (
                    branch.target,
                    path + (branch.target,),
                    operator @ chain,
                )
            )

    if not raw_histories:
        raise ArithmeticError('a finite instrument tree needs a terminal history')

    global_effect = sum(
        (chain.conj().T @ chain for _, _, chain, _ in raw_histories),
        start=np.zeros_like(identity),
    )
    global_isometry_residual = float(
        np.linalg.norm(global_effect - identity, ord='fro')
    )
    probability_normalization_residual = abs(
        sum(probability for _, _, _, probability in raw_histories) - 1.0
    )

    supported = set(supported_terminal_nodes)
    history_probability_residuals: list[float] = []
    prefix_probabilities: dict[str, float] = {}
    histories: list[PersistentRecordHistory] = []
    for index, (terminal, path, _, probability) in enumerate(raw_histories):
        expected = (
            state_by_node[terminal].probability_from_root
            if terminal in supported
            else 0.0
        )
        history_probability_residuals.append(abs(probability - expected))
        for node in path:
            prefix_probabilities[node] = prefix_probabilities.get(node, 0.0) + probability
        histories.append(
            PersistentRecordHistory(
                basis_index=index,
                terminal_node=terminal,
                path=path,
                probability=probability,
                supported_by_root=terminal in supported,
            )
        )

    prefix_probability_residuals = [
        abs(
            probability
            - (
                state_by_node[node].probability_from_root
                if node in state_by_node
                else 0.0
            )
        )
        for node, probability in prefix_probabilities.items()
    ]
    max_history_probability_residual = max(history_probability_residuals, default=0.0)
    max_prefix_probability_residual = max(prefix_probability_residuals, default=0.0)
    numerical_limit = tolerance * max(1.0, float(len(raw_histories)), float(dimension))
    if max(
        global_isometry_residual,
        probability_normalization_residual,
        max_history_probability_residual,
        max_prefix_probability_residual,
    ) > numerical_limit:
        raise ArithmeticError('declared history register exceeded numerical tolerance')

    return ClassicalRecordAlgebraCertificate(
        histories=tuple(histories),
        node_probabilities=tuple(prefix_probabilities.items()),
        global_isometry_residual=global_isometry_residual,
        probability_normalization_residual=probability_normalization_residual,
        max_history_probability_residual=max_history_probability_residual,
        max_prefix_probability_residual=max_prefix_probability_residual,
        orthogonal_history_basis=True,
        commutative_diagonal_algebra=True,
        append_only_history_labels=True,
        physical_pointer_dynamics_derived=False,
    )


def build_energy_resolved_instrument_tree(
    hamiltonian: np.ndarray,
    root_density: np.ndarray,
    instruments: Sequence[RecordInstrument],
    *,
    root: str = 'root',
    root_label: str = 'prepared ensemble',
    initial_weight: float = 1.0,
    require_terminal_energy_sharp: bool = True,
    require_qnd: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> QuantumKernelCertificate:
    '''Derive a finite Born record tree and certify its energy ledger.

    The input instruments must form a tree: every non-root target has exactly
    one parent.  Every Kraus family is complete and preserves the declared
    Hamiltonian-plus-transfer ledger as an operator,

        sum_b K_b^dagger (H + Delta E_b I) K_b = H.

    The default additionally requires every branch to be QND and therefore
    covers Delta E_b=0 energy records.  An explicitly energy-conserving
    system--ancilla collision may set require_qnd=False and record the measured
    ancilla energy gain in energy_transfer.  Only outcomes whose probability
    exceeds the numerical support threshold enter the supported tree;
    subthreshold outcomes remain in the completeness, ledger, and full
    abstract history-register checks.
    '''

    if not root or not root_label:
        raise ValueError('root name and label must be non-empty')
    if not math.isfinite(initial_weight) or initial_weight <= 0.0:
        raise ValueError('initial weight must be finite and positive')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    hamiltonian_matrix_scale = float(np.linalg.norm(hamiltonian_matrix, ord='fro'))
    if hamiltonian_matrix_scale <= 0.0:
        raise ValueError('Hamiltonian must have a non-zero physical energy scale')
    if (
        np.linalg.norm(
            hamiltonian_matrix - hamiltonian_matrix.conj().T,
            ord='fro',
        )
        / hamiltonian_matrix_scale
        > tolerance
    ):
        raise ValueError('Hamiltonian must be Hermitian')
    energy_spectrum = np.linalg.eigvalsh(hamiltonian_matrix)
    energy_scale = float(np.max(np.abs(energy_spectrum)))
    if float(np.min(energy_spectrum)) <= tolerance * energy_scale:
        raise ValueError('Hamiltonian must be strictly positive on this contract')

    dimension = hamiltonian_matrix.shape[0]
    root_state = _validate_density(
        root_density,
        dimension=dimension,
        tolerance=tolerance,
        name='root density',
    )

    instrument_by_node: dict[str, RecordInstrument] = {}
    for instrument in instruments:
        if not instrument.node or not instrument.instrument_id:
            raise ValueError('instrument node and identifier must be non-empty')
        if instrument.node in instrument_by_node:
            raise ValueError('instrument nodes must be unique')
        if not instrument.branches:
            raise ValueError('a nonterminal instrument needs at least one branch')
        instrument_by_node[instrument.node] = instrument

    identity = np.eye(dimension, dtype=np.complex128)
    state_by_node = {
        root: _energy_state(
            root,
            root_state,
            1.0,
            hamiltonian_matrix,
            cumulative_energy_transfer=0.0,
            energy_scale=energy_scale,
            tolerance=tolerance,
        )
    }
    label_by_node = {root: root_label}
    order = [root]
    queue = [root]
    declared_targets = {root}
    transitions: list[CausalTransition] = []
    used_instruments: list[tuple[str, str]] = []
    validated_branches_by_node: dict[
        str,
        tuple[tuple[KrausBranch, np.ndarray], ...],
    ] = {}
    max_completeness_residual = 0.0
    max_energy_channel_residual = 0.0
    max_relative_energy_channel_residual = 0.0
    max_qnd_commutator_residual = 0.0
    max_relative_qnd_commutator_residual = 0.0

    while queue:
        node = queue.pop(0)
        instrument = instrument_by_node.get(node)
        if instrument is None:
            continue
        used_instruments.append((node, instrument.instrument_id))
        parent = state_by_node[node]
        targets: set[str] = set()
        completeness = np.zeros_like(identity)
        energy_channel = np.zeros_like(hamiltonian_matrix)
        prepared: list[tuple[KrausBranch, np.ndarray]] = []
        for branch in instrument.branches:
            if not branch.target or not branch.label:
                raise ValueError('branch target and label must be non-empty')
            if branch.target in targets:
                raise ValueError('branch targets must be unique at each instrument')
            if branch.target in declared_targets:
                raise ValueError('instrument records must form a tree without merging')
            if not math.isfinite(branch.energy_transfer):
                raise ValueError('branch energy transfer must be finite')
            targets.add(branch.target)
            declared_targets.add(branch.target)
            operator = _square_matrix(
                branch.operator,
                f'Kraus operator {node}->{branch.target}',
            )
            if operator.shape != (dimension, dimension):
                raise ValueError('Kraus operator has the wrong Hilbert-space dimension')
            completeness += operator.conj().T @ operator
            energy_channel += operator.conj().T @ (
                hamiltonian_matrix
                + branch.energy_transfer * identity
            ) @ operator
            qnd_residual = float(
                np.linalg.norm(
                    operator @ hamiltonian_matrix
                    - hamiltonian_matrix @ operator,
                    ord='fro',
                )
            )
            max_qnd_commutator_residual = max(
                max_qnd_commutator_residual,
                qnd_residual,
            )
            max_relative_qnd_commutator_residual = max(
                max_relative_qnd_commutator_residual,
                qnd_residual / hamiltonian_matrix_scale,
            )
            prepared.append((branch, operator))

        completeness_residual = float(
            np.linalg.norm(completeness - identity, ord='fro')
        )
        energy_channel_residual = float(
            np.linalg.norm(
                energy_channel - hamiltonian_matrix,
                ord='fro',
            )
        )
        max_completeness_residual = max(
            max_completeness_residual,
            completeness_residual,
        )
        max_energy_channel_residual = max(
            max_energy_channel_residual,
            energy_channel_residual,
        )
        relative_energy_channel_residual = (
            energy_channel_residual / hamiltonian_matrix_scale
        )
        max_relative_energy_channel_residual = max(
            max_relative_energy_channel_residual,
            relative_energy_channel_residual,
        )
        if completeness_residual > tolerance:
            raise ValueError('Kraus family must be trace-preserving')
        if relative_energy_channel_residual > tolerance:
            raise ValueError(
                'instrument must preserve the declared Hamiltonian-plus-transfer ledger'
            )
        if require_qnd and max(
            float(
                np.linalg.norm(
                    operator @ hamiltonian_matrix
                    - hamiltonian_matrix @ operator,
                    ord='fro',
                )
            )
            / hamiltonian_matrix_scale
            for _, operator in prepared
        ) > tolerance:
            raise ValueError('every Kraus branch must be QND with the Hamiltonian')
        validated_branches_by_node[node] = tuple(prepared)

        child_probabilities = 0.0
        for branch, operator in prepared:
            unnormalized = operator @ parent.density @ operator.conj().T
            probability = _real_trace(
                unnormalized,
                tolerance=tolerance,
                name=f'Born weight {node}->{branch.target}',
            )
            if probability < -tolerance:
                raise ArithmeticError('Born probability is negative')
            if probability <= tolerance:
                continue
            if probability > 1.0 + tolerance:
                raise ArithmeticError('Born probability exceeds one')
            probability = min(1.0, probability)
            child_probabilities += probability
            posterior = _validate_density(
                unnormalized / probability,
                dimension=dimension,
                tolerance=tolerance,
                name=f'posterior density at {branch.target}',
            )
            child = _energy_state(
                branch.target,
                posterior,
                parent.probability_from_root * probability,
                hamiltonian_matrix,
                cumulative_energy_transfer=(
                    parent.cumulative_energy_transfer
                    + branch.energy_transfer
                ),
                energy_scale=energy_scale,
                tolerance=tolerance,
            )
            state_by_node[branch.target] = child
            label_by_node[branch.target] = branch.label
            transitions.append(
                CausalTransition(
                    source=node,
                    target=branch.target,
                    probability=probability,
                )
            )
            order.append(branch.target)
            queue.append(branch.target)
        if not math.isclose(
            child_probabilities,
            1.0,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            raise ArithmeticError('Born branch probabilities do not sum to one')

    unused = set(instrument_by_node) - {node for node, _ in used_instruments}
    if unused:
        raise ValueError('every declared instrument must be reachable from the root')

    terminal_nodes = tuple(
        node for node in order if node not in instrument_by_node
    )
    terminal_energy_resolved = all(
        state_by_node[node].energy_variance
        <= tolerance * energy_scale**2
        for node in terminal_nodes
    )
    if require_terminal_energy_sharp and not terminal_energy_resolved:
        raise ValueError('every terminal record must have sharp Hamiltonian energy')

    nodes = tuple(
        CausalRecordNode(
            name=node,
            label=label_by_node[node],
            energy=state_by_node[node].energy_expectation,
        )
        for node in order
    )
    flow = construct_conserved_record_flow(
        nodes,
        transitions,
        {root: initial_weight},
        tolerance=tolerance,
    )
    record_algebra = _certify_classical_record_algebra(
        root=root,
        root_density=root_state,
        dimension=dimension,
        branches_by_node=validated_branches_by_node,
        state_by_node=state_by_node,
        supported_terminal_nodes=terminal_nodes,
        tolerance=tolerance,
    )
    return QuantumKernelCertificate(
        root=root,
        nodes=nodes,
        transitions=tuple(transitions),
        conditional_states=tuple(state_by_node[node] for node in order),
        instrument_ids=tuple(used_instruments),
        terminal_nodes=terminal_nodes,
        terminal_energy_resolved=terminal_energy_resolved,
        qnd_required=require_qnd,
        hamiltonian_energy_scale=energy_scale,
        hamiltonian_energy_min=float(np.min(energy_spectrum)),
        hamiltonian_energy_max=float(np.max(energy_spectrum)),
        max_completeness_residual=max_completeness_residual,
        max_energy_channel_residual=max_energy_channel_residual,
        max_relative_energy_channel_residual=max_relative_energy_channel_residual,
        max_qnd_commutator_residual=max_qnd_commutator_residual,
        max_relative_qnd_commutator_residual=max_relative_qnd_commutator_residual,
        support_probability_tolerance=tolerance,
        record_algebra=record_algebra,
        flow=flow,
    )
