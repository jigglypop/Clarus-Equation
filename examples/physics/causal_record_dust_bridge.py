"""Constructive bridge from marked causal records to a dust stress tensor.

The bridge is deliberately forward directed.  A finite causal record DAG is
first equipped with a normalized transition kernel and a positive energy mark.
If the energy mark is harmonic under that kernel, the induced number and
expected-energy flows obey exact Kirchhoff balance at every internal record.

An exit antichain can then be marked by points and future mass-shell momenta on
an already specified Cauchy cell.  Its pushforward is a positive kinetic
measure.  The moment formulas below use a surface-number convention:

    J^mu = integral (p^mu / E_N) dN,
    T^munu = integral (p^mu p^nu / E_N) dN,

where ``E_N = -N.p``.  Thus ``-N.J`` is the number density measured on the
surface and ``T_NN`` is number density times particle energy.

The construction does not derive the DAG, its physical energy scale, the
Cauchy geometry, or the mass-shell marks from a bare singleton.  It proves
that once those typed data are supplied, current, stress, no-double-counting,
and the monokinetic dust limit are consequences rather than extra axioms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


DEFAULT_TOLERANCE = 1.0e-12


def _finite_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _finite_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


@dataclass(frozen=True)
class CausalRecordNode:
    """One zero-dimensional record state with a physical energy mark."""

    name: str
    label: str
    energy: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("record name must be non-empty")
        if not self.label:
            raise ValueError("record label must be non-empty")
        _finite_positive(self.energy, "record energy")


@dataclass(frozen=True)
class CausalTransition:
    """A dimensionless transition probability on a directed causal edge."""

    source: str
    target: str
    probability: float

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("transition endpoints must be non-empty")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("transition probability must lie in [0, 1]")


@dataclass(frozen=True)
class EdgeFlux:
    source: str
    target: str
    number_flux: float
    energy_flux: float


@dataclass(frozen=True)
class NodeBalance:
    node: str
    number_in: float
    number_out: float
    number_injected: float
    number_extracted: float
    number_residual: float
    energy_in: float
    energy_out: float
    energy_injected: float
    energy_extracted: float
    energy_residual: float


@dataclass(frozen=True)
class CausalRecordFlow:
    # Positive is used in the measure-theory sense: every flux is non-negative.
    """Unique positive flow induced by a normalized causal kernel."""

    topological_order: tuple[str, ...]
    node_weights: tuple[tuple[str, float], ...]
    edge_fluxes: tuple[EdgeFlux, ...]
    balances: tuple[NodeBalance, ...]
    terminal_nodes: tuple[str, ...]
    terminal_composition: tuple[tuple[str, float], ...]
    initial_number: float
    terminal_number: float
    initial_energy: float
    terminal_energy: float

    def weight(self, node: str) -> float:
        for candidate, value in self.node_weights:
            if candidate == node:
                return value
        raise KeyError(node)


def _topological_order(
    node_names: set[str], transitions: Sequence[CausalTransition]
) -> tuple[str, ...]:
    outgoing: dict[str, list[str]] = {name: [] for name in node_names}
    indegree = {name: 0 for name in node_names}
    seen_edges: set[tuple[str, str]] = set()
    for transition in transitions:
        edge = (transition.source, transition.target)
        if edge in seen_edges:
            raise ValueError("duplicate causal transition")
        seen_edges.add(edge)
        outgoing[transition.source].append(transition.target)
        indegree[transition.target] += 1

    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(node_names):
        raise ValueError("causal record graph must be acyclic")
    return tuple(order)


def construct_conserved_record_flow(
    nodes: Sequence[CausalRecordNode],
    transitions: Sequence[CausalTransition],
    initial_weights: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CausalRecordFlow:
    """Construct the unique number and expected-energy flow on a finite DAG.

    Every nonterminal row of the transition kernel must sum to one.  Its
    energy mark must also be harmonic,

        E(source) = sum_target P(source, target) E(target).

    Initial weight is allowed only on graph roots.  Under these assumptions a
    topological recursion fixes every node and edge weight uniquely.
    """

    _finite_positive(tolerance, "tolerance")
    node_by_name = {node.name: node for node in nodes}
    if len(node_by_name) != len(nodes) or not node_by_name:
        raise ValueError("record node names must be unique and non-empty")
    node_names = set(node_by_name)
    for transition in transitions:
        if transition.source not in node_names or transition.target not in node_names:
            raise ValueError("every transition endpoint must name a record node")

    order = _topological_order(node_names, transitions)
    outgoing: dict[str, list[CausalTransition]] = {name: [] for name in node_names}
    incoming: dict[str, list[CausalTransition]] = {name: [] for name in node_names}
    for transition in transitions:
        outgoing[transition.source].append(transition)
        incoming[transition.target].append(transition)

    roots = {name for name in node_names if not incoming[name]}
    for name, weight in initial_weights.items():
        if name not in node_names:
            raise ValueError("initial weight names an unknown record node")
        _finite_nonnegative(weight, "initial weight")
        if name not in roots and weight > tolerance:
            raise ValueError("initial weight is allowed only on causal roots")

    for name, row in outgoing.items():
        if not row:
            continue
        probability_sum = sum(edge.probability for edge in row)
        if not _close(probability_sum, 1.0, tolerance):
            raise ValueError("each nonterminal transition row must sum to one")
        expected_energy = sum(
            edge.probability * node_by_name[edge.target].energy for edge in row
        )
        if not _close(expected_energy, node_by_name[name].energy, tolerance):
            raise ValueError("record energy must be harmonic under the transition kernel")

    weights = {name: float(initial_weights.get(name, 0.0)) for name in node_names}
    edge_flux_by_pair: dict[tuple[str, str], EdgeFlux] = {}
    for name in order:
        for edge in outgoing[name]:
            number_flux = weights[name] * edge.probability
            energy_flux = number_flux * node_by_name[edge.target].energy
            edge_flux_by_pair[(edge.source, edge.target)] = EdgeFlux(
                source=edge.source,
                target=edge.target,
                number_flux=number_flux,
                energy_flux=energy_flux,
            )
            weights[edge.target] += number_flux

    terminal_nodes = tuple(sorted(name for name in node_names if not outgoing[name]))
    balances: list[NodeBalance] = []
    for name in order:
        number_in = sum(
            edge_flux_by_pair[(edge.source, edge.target)].number_flux
            for edge in incoming[name]
        )
        number_out = sum(
            edge_flux_by_pair[(edge.source, edge.target)].number_flux
            for edge in outgoing[name]
        )
        number_injected = float(initial_weights.get(name, 0.0))
        number_extracted = weights[name] if name in terminal_nodes else 0.0
        energy_in = sum(
            edge_flux_by_pair[(edge.source, edge.target)].energy_flux
            for edge in incoming[name]
        )
        energy_out = sum(
            edge_flux_by_pair[(edge.source, edge.target)].energy_flux
            for edge in outgoing[name]
        )
        energy_injected = number_injected * node_by_name[name].energy
        energy_extracted = (
            weights[name] * node_by_name[name].energy if name in terminal_nodes else 0.0
        )
        balances.append(
            NodeBalance(
                node=name,
                number_in=number_in,
                number_out=number_out,
                number_injected=number_injected,
                number_extracted=number_extracted,
                number_residual=(
                    number_out - number_in - number_injected + number_extracted
                ),
                energy_in=energy_in,
                energy_out=energy_out,
                energy_injected=energy_injected,
                energy_extracted=energy_extracted,
                energy_residual=(
                    energy_out - energy_in - energy_injected + energy_extracted
                ),
            )
        )

    initial_number = sum(float(initial_weights.get(name, 0.0)) for name in roots)
    terminal_number = sum(weights[name] for name in terminal_nodes)
    initial_energy = sum(
        float(initial_weights.get(name, 0.0)) * node_by_name[name].energy
        for name in roots
    )
    terminal_energy = sum(
        weights[name] * node_by_name[name].energy for name in terminal_nodes
    )
    if not _close(initial_number, terminal_number, tolerance):
        raise ArithmeticError("constructed record flow violates number conservation")
    if not _close(initial_energy, terminal_energy, tolerance):
        raise ArithmeticError("constructed record flow violates energy conservation")
    if any(
        abs(balance.number_residual) > tolerance
        or abs(balance.energy_residual) > tolerance
        for balance in balances
    ):
        raise ArithmeticError("constructed record flow violates a local balance law")

    label_weights: dict[str, float] = {}
    for name in terminal_nodes:
        label = node_by_name[name].label
        label_weights[label] = label_weights.get(label, 0.0) + weights[name]
    terminal_composition = tuple(
        sorted(
            (
                label,
                0.0 if terminal_number == 0.0 else weight / terminal_number,
            )
            for label, weight in label_weights.items()
        )
    )
    return CausalRecordFlow(
        topological_order=order,
        node_weights=tuple((name, weights[name]) for name in order),
        edge_fluxes=tuple(
            edge_flux_by_pair[pair] for pair in sorted(edge_flux_by_pair)
        ),
        balances=tuple(balances),
        terminal_nodes=terminal_nodes,
        terminal_composition=terminal_composition,
        initial_number=initial_number,
        terminal_number=terminal_number,
        initial_energy=initial_energy,
        terminal_energy=terminal_energy,
    )


Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Tensor4 = tuple[Vector4, Vector4, Vector4, Vector4]


@dataclass(frozen=True)
class ExitPhaseMark:
    """A terminal record embedded in one local orthonormal Cauchy frame."""

    node: str
    position: Vector3
    mass: float
    spatial_momentum: Vector3
    residual_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if not self.node:
            raise ValueError("phase mark must name a terminal record")
        _finite_positive(self.mass, "mass")
        if len(self.position) != 3 or len(self.spatial_momentum) != 3:
            raise ValueError("position and spatial momentum must have three components")
        if not all(math.isfinite(value) for value in self.position):
            raise ValueError("position components must be finite")
        if not all(math.isfinite(value) for value in self.spatial_momentum):
            raise ValueError("momentum components must be finite")
        if (
            not math.isfinite(self.residual_efficiency)
            or not 0.0 <= self.residual_efficiency <= 1.0
        ):
            raise ValueError("residual efficiency must lie in [0, 1]")

    @property
    def energy(self) -> float:
        return math.sqrt(
            self.mass * self.mass
            + sum(component * component for component in self.spatial_momentum)
        )

    @property
    def four_momentum(self) -> Vector4:
        return (self.energy, *self.spatial_momentum)


@dataclass(frozen=True)
class MatchedExitRecord:
    node: str
    label: str
    position: Vector3
    mass: float
    four_momentum: Vector4
    full_number: float
    residual_number: float
    complement_number: float


@dataclass(frozen=True)
class KineticMatching:
    """Positive exit measure and its number/stress moments on one Cauchy cell."""

    records: tuple[MatchedExitRecord, ...]
    cell_volume: float
    current: Vector4
    stress: Tensor4
    surface_number_density: float
    complement_number_density: float
    total_number_density: float
    residual_energy_density: float
    complement_energy_density: float
    total_energy_density: float
    no_double_counting_residual: float


def match_exit_antichain(
    flow: CausalRecordFlow,
    nodes: Sequence[CausalRecordNode],
    marks: Sequence[ExitPhaseMark],
    *,
    cell_volume: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> KineticMatching:
    """Push the terminal record measure to a future mass-shell measure.

    ``residual_efficiency`` performs an explicit partition: the residual and
    complementary weights add to the full terminal weight record by record.
    The record energy mark must equal the local mass-shell energy, so the
    discrete energy ledger and the Cauchy data are the same typed quantity.
    """

    _finite_positive(cell_volume, "cell volume")
    _finite_positive(tolerance, "tolerance")
    node_by_name = {node.name: node for node in nodes}
    if len(node_by_name) != len(nodes):
        raise ValueError("record node names must be unique")
    mark_by_node = {mark.node: mark for mark in marks}
    if len(mark_by_node) != len(marks):
        raise ValueError("terminal phase marks must be unique by node")
    if set(mark_by_node) != set(flow.terminal_nodes):
        raise ValueError("phase marks must cover exactly the terminal antichain")

    records: list[MatchedExitRecord] = []
    current = [0.0, 0.0, 0.0, 0.0]
    stress = [[0.0 for _ in range(4)] for _ in range(4)]
    residual_number = 0.0
    complement_number = 0.0
    total_number = 0.0
    residual_energy = 0.0
    complement_energy = 0.0
    total_energy = 0.0
    for name in flow.terminal_nodes:
        if name not in node_by_name:
            raise ValueError("terminal flow node is absent from node metadata")
        node = node_by_name[name]
        mark = mark_by_node[name]
        if not _close(node.energy, mark.energy, tolerance):
            raise ValueError("record energy and future mass-shell energy must agree")
        full_weight = flow.weight(name)
        residual_weight = full_weight * mark.residual_efficiency
        complement_weight = full_weight - residual_weight
        momentum = mark.four_momentum
        energy = momentum[0]
        records.append(
            MatchedExitRecord(
                node=name,
                label=node.label,
                position=mark.position,
                mass=mark.mass,
                four_momentum=momentum,
                full_number=full_weight,
                residual_number=residual_weight,
                complement_number=complement_weight,
            )
        )
        total_number += full_weight
        residual_number += residual_weight
        complement_number += complement_weight
        total_energy += full_weight * energy
        residual_energy += residual_weight * energy
        complement_energy += complement_weight * energy
        for mu in range(4):
            current[mu] += residual_weight * momentum[mu] / energy
            for nu in range(4):
                stress[mu][nu] += (
                    residual_weight * momentum[mu] * momentum[nu] / energy
                )

    inverse_volume = 1.0 / cell_volume
    current_tuple: Vector4 = tuple(  # type: ignore[assignment]
        value * inverse_volume for value in current
    )
    stress_tuple: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(value * inverse_volume for value in row) for row in stress
    )
    no_double_counting_residual = total_energy - residual_energy - complement_energy
    if abs(no_double_counting_residual) > tolerance:
        raise ArithmeticError("residual and complement energy double-count or lose weight")
    if not _close(total_number, flow.terminal_number, tolerance):
        raise ArithmeticError("Cauchy matching changed the terminal record number")
    if not _close(total_energy, flow.terminal_energy, tolerance):
        raise ArithmeticError("Cauchy matching changed the terminal record energy")
    return KineticMatching(
        records=tuple(records),
        cell_volume=cell_volume,
        current=current_tuple,
        stress=stress_tuple,
        surface_number_density=residual_number * inverse_volume,
        complement_number_density=complement_number * inverse_volume,
        total_number_density=total_number * inverse_volume,
        residual_energy_density=residual_energy * inverse_volume,
        complement_energy_density=complement_energy * inverse_volume,
        total_energy_density=total_energy * inverse_volume,
        no_double_counting_residual=no_double_counting_residual * inverse_volume,
    )


@dataclass(frozen=True)
class DustInitialData:
    mass: float
    energy: float
    gamma: float
    four_velocity: Vector4
    surface_number_density: float
    rest_number_density: float
    rest_energy_density: float
    current: Vector4
    stress: Tensor4


def monokinetic_dust_data(
    matching: KineticMatching,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> DustInitialData:
    """Reduce a positive single-momentum kinetic measure to exact dust data."""

    _finite_positive(tolerance, "tolerance")
    active = [record for record in matching.records if record.residual_number > tolerance]
    if not active:
        raise ValueError("monokinetic dust requires positive residual weight")
    reference = active[0]
    for record in active[1:]:
        if not _close(record.mass, reference.mass, tolerance) or any(
            not _close(left, right, tolerance)
            for left, right in zip(record.four_momentum, reference.four_momentum)
        ):
            raise ValueError("multi-stream or multi-mass data require kinetic/multifluid EFT")

    mass = reference.mass
    energy = reference.four_momentum[0]
    gamma = energy / mass
    four_velocity: Vector4 = tuple(  # type: ignore[assignment]
        component / mass for component in reference.four_momentum
    )
    rest_number_density = matching.surface_number_density / gamma
    rest_energy_density = mass * rest_number_density
    expected_current: Vector4 = tuple(  # type: ignore[assignment]
        rest_number_density * component for component in four_velocity
    )
    expected_stress: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(
            rest_energy_density * four_velocity[mu] * four_velocity[nu]
            for nu in range(4)
        )
        for mu in range(4)
    )
    if any(
        not _close(left, right, tolerance)
        for left, right in zip(expected_current, matching.current)
    ):
        raise ArithmeticError("monokinetic current does not equal n u")
    if any(
        not _close(expected_stress[mu][nu], matching.stress[mu][nu], tolerance)
        for mu in range(4)
        for nu in range(4)
    ):
        raise ArithmeticError("monokinetic stress does not equal rho u tensor u")
    return DustInitialData(
        mass=mass,
        energy=energy,
        gamma=gamma,
        four_velocity=four_velocity,
        surface_number_density=matching.surface_number_density,
        rest_number_density=rest_number_density,
        rest_energy_density=rest_energy_density,
        current=matching.current,
        stress=matching.stress,
    )


def free_stream_exit_marks(
    marks: Sequence[ExitPhaseMark],
    *,
    coordinate_time: float,
    box_length: float,
) -> tuple[ExitPhaseMark, ...]:
    """Exact Minkowski geodesic pushforward on a periodic spatial box."""

    if not math.isfinite(coordinate_time):
        raise ValueError("coordinate time must be finite")
    _finite_positive(box_length, "box length")
    streamed = []
    for mark in marks:
        velocity = tuple(component / mark.energy for component in mark.spatial_momentum)
        position: Vector3 = tuple(  # type: ignore[assignment]
            (coordinate + coordinate_time * speed) % box_length
            for coordinate, speed in zip(mark.position, velocity)
        )
        streamed.append(
            ExitPhaseMark(
                node=mark.node,
                position=position,
                mass=mark.mass,
                spatial_momentum=mark.spatial_momentum,
                residual_efficiency=mark.residual_efficiency,
            )
        )
    return tuple(streamed)


@dataclass(frozen=True)
class FlatFLRWCauchyWitness:
    """Constraint-satisfying homogeneous flat-torus dust initial data."""

    energy_density: float
    newton_constant: float
    hubble_rate: float
    spatial_metric_diagonal: Vector3
    extrinsic_curvature_diagonal: Vector3
    hamiltonian_left: float
    hamiltonian_right: float
    hamiltonian_residual: float
    momentum_residual: Vector3


def flat_flrw_cauchy_witness(
    dust: DustInitialData,
    *,
    newton_constant: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatFLRWCauchyWitness:
    """Embed homogeneous comoving dust in data satisfying the GR constraints.

    The witness uses a unit flat three-torus, ``h_ij = delta_ij`` and
    ``K_ij = -H h_ij``.  For comoving dust the momentum constraint vanishes,
    while the Hamiltonian constraint fixes ``H^2 = 8 pi G rho / 3``.
    """

    _finite_positive(newton_constant, "Newton constant")
    _finite_positive(tolerance, "tolerance")
    if any(abs(component) > tolerance for component in dust.four_velocity[1:]):
        raise ValueError("flat FLRW witness requires comoving monokinetic dust")
    density = dust.rest_energy_density
    hubble = math.sqrt(8.0 * math.pi * newton_constant * density / 3.0)
    hamiltonian_left = 6.0 * hubble * hubble
    hamiltonian_right = 16.0 * math.pi * newton_constant * density
    return FlatFLRWCauchyWitness(
        energy_density=density,
        newton_constant=newton_constant,
        hubble_rate=hubble,
        spatial_metric_diagonal=(1.0, 1.0, 1.0),
        extrinsic_curvature_diagonal=(-hubble, -hubble, -hubble),
        hamiltonian_left=hamiltonian_left,
        hamiltonian_right=hamiltonian_right,
        hamiltonian_residual=hamiltonian_left - hamiltonian_right,
        momentum_residual=(0.0, 0.0, 0.0),
    )
