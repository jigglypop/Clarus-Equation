'''Construct an energy-resolved quantum instrument record kernel.

This module supplies one explicit upstream realization for the marked record
contract in causal_record_dust_bridge.  The instrument choice is part of the
input: a nonselective channel does not determine a unique outcome tree.

For a normalized state rho_a and Kraus branch K_b|a,

    P_b|a = Tr(K_b|a rho_a K_b|a^dagger)

is a dimensionless Born transition probability.  If the Kraus family is
complete and preserves a declared positive total Hamiltonian H as an operator,
the conditional energy expectations are harmonic.  Requiring every Kraus
operator to commute with H gives the stronger QND certificate that exact
energy spectral sectors are not mixed along a history.

An internal node energy is still only a conditional expectation.  A terminal
record may feed the exact mass-shell bridge only when Var(H) vanishes there.
The spectrum of H supplies the physical energy unit; no dimensionless Born
probability can determine that absolute scale.

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


DEFAULT_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class KrausBranch:
    '''One explicitly named outcome of a chosen quantum instrument.'''

    target: str
    label: str
    operator: np.ndarray


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
    energy_expectation: float
    energy_variance: float


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
    hamiltonian_energy_min: float
    hamiltonian_energy_max: float
    max_completeness_residual: float
    max_energy_channel_residual: float
    max_qnd_commutator_residual: float
    support_probability_tolerance: float
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
    max_projector_idempotence_residual: float
    max_projector_orthogonality_residual: float
    resolution_of_identity_residual: float
    max_hamiltonian_commutator_residual: float
    max_spectral_cluster_width: float
    max_eigenprojector_residual: float


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


def _real_trace(value: np.ndarray, *, tolerance: float, name: str) -> float:
    trace = np.trace(value)
    if abs(float(trace.imag)) > tolerance:
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
    tolerance: float,
) -> ConditionalEnergyState:
    energy = _real_trace(
        hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy at {node}',
    )
    second_moment = _real_trace(
        hamiltonian @ hamiltonian @ density,
        tolerance=tolerance,
        name=f'energy second moment at {node}',
    )
    variance = second_moment - energy * energy
    variance_scale = max(1.0, second_moment, energy * energy)
    if variance < -tolerance * variance_scale:
        raise ArithmeticError(f'energy variance at {node} is negative')
    return ConditionalEnergyState(
        node=node,
        density=density,
        probability_from_root=probability_from_root,
        energy_expectation=energy,
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
    Distinct eigenvalues closer than the declared numerical tolerance are
    rejected as ambiguous instead of being merged into a non-sharp sector.
    '''

    if not node or not instrument_id or not target_prefix:
        raise ValueError('node, instrument identifier, and target prefix are required')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    if np.linalg.norm(
        hamiltonian_matrix - hamiltonian_matrix.conj().T, ord='fro'
    ) > tolerance:
        raise ValueError('Hamiltonian must be Hermitian')
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    if float(np.min(eigenvalues)) <= tolerance:
        raise ValueError('Hamiltonian must be strictly positive on this contract')

    clusters: list[list[int]] = []
    for index, energy in enumerate(eigenvalues):
        if clusters:
            reference = float(eigenvalues[clusters[-1][0]])
            if float(energy) == reference:
                clusters[-1].append(index)
                continue
            if math.isclose(
                float(energy),
                reference,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
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
    numerical_limit = tolerance * max(1.0, float(dimension))
    if max(
        max_idempotence,
        max_orthogonality,
        resolution_residual,
        max_commutator,
        max_cluster_width,
        max_eigenprojector_residual,
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
        max_projector_idempotence_residual=max_idempotence,
        max_projector_orthogonality_residual=max_orthogonality,
        resolution_of_identity_residual=resolution_residual,
        max_hamiltonian_commutator_residual=max_commutator,
        max_spectral_cluster_width=max_cluster_width,
        max_eigenprojector_residual=max_eigenprojector_residual,
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


def build_energy_resolved_instrument_tree(
    hamiltonian: np.ndarray,
    root_density: np.ndarray,
    instruments: Sequence[RecordInstrument],
    *,
    root: str = 'root',
    root_label: str = 'prepared ensemble',
    initial_weight: float = 1.0,
    require_terminal_energy_sharp: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> QuantumKernelCertificate:
    '''Derive a finite Born record tree and certify its energy ledger.

    The input instruments must form a tree: every non-root target has exactly
    one parent.  Every Kraus family is complete, preserves the Hamiltonian as
    an operator, and is QND with respect to that Hamiltonian.  Only outcomes
    whose probability exceeds the numerical support threshold enter this
    supported tree; subthreshold outcomes remain in the Kraus completeness
    and energy-channel checks.
    '''

    if not root or not root_label:
        raise ValueError('root name and label must be non-empty')
    if not math.isfinite(initial_weight) or initial_weight <= 0.0:
        raise ValueError('initial weight must be finite and positive')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    hamiltonian_matrix = _square_matrix(hamiltonian, 'Hamiltonian')
    if np.linalg.norm(
        hamiltonian_matrix - hamiltonian_matrix.conj().T, ord='fro'
    ) > tolerance:
        raise ValueError('Hamiltonian must be Hermitian')
    energy_spectrum = np.linalg.eigvalsh(hamiltonian_matrix)
    if float(np.min(energy_spectrum)) <= tolerance:
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
            tolerance=tolerance,
        )
    }
    label_by_node = {root: root_label}
    order = [root]
    queue = [root]
    declared_targets = {root}
    transitions: list[CausalTransition] = []
    used_instruments: list[tuple[str, str]] = []
    max_completeness_residual = 0.0
    max_energy_channel_residual = 0.0
    max_qnd_commutator_residual = 0.0

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
            targets.add(branch.target)
            declared_targets.add(branch.target)
            operator = _square_matrix(
                branch.operator,
                f'Kraus operator {node}->{branch.target}',
            )
            if operator.shape != (dimension, dimension):
                raise ValueError('Kraus operator has the wrong Hilbert-space dimension')
            completeness += operator.conj().T @ operator
            energy_channel += (
                operator.conj().T @ hamiltonian_matrix @ operator
            )
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
        if completeness_residual > tolerance:
            raise ValueError('Kraus family must be trace-preserving')
        if energy_channel_residual > tolerance:
            raise ValueError('instrument must preserve the declared total Hamiltonian')
        if max(
            float(
                np.linalg.norm(
                    operator @ hamiltonian_matrix
                    - hamiltonian_matrix @ operator,
                    ord='fro',
                )
            )
            for _, operator in prepared
        ) > tolerance:
            raise ValueError('every Kraus branch must be QND with the Hamiltonian')

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
        <= tolerance
        * max(1.0, state_by_node[node].energy_expectation ** 2)
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
    return QuantumKernelCertificate(
        root=root,
        nodes=nodes,
        transitions=tuple(transitions),
        conditional_states=tuple(state_by_node[node] for node in order),
        instrument_ids=tuple(used_instruments),
        terminal_nodes=terminal_nodes,
        terminal_energy_resolved=terminal_energy_resolved,
        hamiltonian_energy_min=float(np.min(energy_spectrum)),
        hamiltonian_energy_max=float(np.max(energy_spectrum)),
        max_completeness_residual=max_completeness_residual,
        max_energy_channel_residual=max_energy_channel_residual,
        max_qnd_commutator_residual=max_qnd_commutator_residual,
        support_probability_tolerance=tolerance,
        flow=flow,
    )
