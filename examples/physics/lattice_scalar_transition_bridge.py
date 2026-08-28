'''Constructive lattice embedding and scalar total-stress transition.

The lattice spacing, clock step, causal speed, and flat background foliation
are explicit microscopic model inputs.  The routines certify that a declared
record graph respects that embedding; they do not derive spacetime from a
dimensionless probability kernel.

The scalar transition uses two identical free real scalar species.  An
orthogonal rotation in species space is canonical and leaves the sum of their
quadratic stress tensors invariant.  If the residual species initially
vanishes, a rotation with sin(theta)^2 = eta gives an exact eta/(1-eta)
partition without copying energy.  This is a concrete transition model, not a
claim that a quantum instrument uniquely selects eta.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


DEFAULT_TOLERANCE = 1.0e-11


@dataclass(frozen=True)
class LayeredRecordEvent:
    name: str
    time_index: int
    site: tuple[int, int, int]


@dataclass(frozen=True)
class LayeredCausalEdge:
    source: str
    target: str


@dataclass(frozen=True)
class CauchyEmbeddingCertificate:
    coordinates: tuple[tuple[str, tuple[float, float, float, float]], ...]
    exit_nodes: tuple[str, ...]
    exit_time: float
    lattice_spacing: float
    clock_step: float
    causal_speed: float
    spatial_shape: tuple[int, int, int]
    spatial_volume: float
    minimum_causal_margin: float

    def coordinate(self, node: str) -> tuple[float, float, float, float]:
        for candidate, coordinate in self.coordinates:
            if candidate == node:
                return coordinate
        raise KeyError(node)


def certify_layered_cauchy_embedding(
    events: Sequence[LayeredRecordEvent],
    edges: Sequence[LayeredCausalEdge],
    *,
    exit_nodes: Sequence[str],
    lattice_spacing: float,
    clock_step: float,
    causal_speed: float,
    spatial_shape: tuple[int, int, int],
    tolerance: float = DEFAULT_TOLERANCE,
) -> CauchyEmbeddingCertificate:
    '''Check an order-preserving embedding into a flat periodic time foliation.'''

    for value, name in (
        (lattice_spacing, 'lattice spacing'),
        (clock_step, 'clock step'),
        (causal_speed, 'causal speed'),
        (tolerance, 'tolerance'),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
    if len(spatial_shape) != 3 or any(size <= 0 for size in spatial_shape):
        raise ValueError('spatial shape must contain three positive sizes')

    event_by_name: dict[str, LayeredRecordEvent] = {}
    for event in events:
        if not event.name:
            raise ValueError('event names must be non-empty')
        if event.name in event_by_name:
            raise ValueError('event names must be unique')
        if event.time_index < 0:
            raise ValueError('time indices must be non-negative')
        if len(event.site) != 3 or any(
            index < 0 or index >= size
            for index, size in zip(event.site, spatial_shape)
        ):
            raise ValueError('event site lies outside the periodic lattice')
        event_by_name[event.name] = event
    if not event_by_name:
        raise ValueError('at least one record event is required')

    outgoing = {name: 0 for name in event_by_name}
    minimum_causal_margin = math.inf
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge.source, edge.target)
        if pair in seen_edges:
            raise ValueError('causal edges must be unique')
        seen_edges.add(pair)
        if edge.source not in event_by_name or edge.target not in event_by_name:
            raise ValueError('causal edge names an unknown event')
        source = event_by_name[edge.source]
        target = event_by_name[edge.target]
        delta_steps = target.time_index - source.time_index
        if delta_steps <= 0:
            raise ValueError('causal edges must point to a later time layer')
        squared_distance = 0.0
        for left, right, size in zip(source.site, target.site, spatial_shape):
            raw = abs(right - left)
            periodic_steps = min(raw, size - raw)
            squared_distance += (lattice_spacing * periodic_steps) ** 2
        elapsed = clock_step * delta_steps
        causal_margin = (causal_speed * elapsed) ** 2 - squared_distance
        if causal_margin < -tolerance:
            raise ValueError('declared edge is spacelike in the lattice embedding')
        minimum_causal_margin = min(minimum_causal_margin, causal_margin)
        outgoing[edge.source] += 1

    declared_exits = tuple(exit_nodes)
    if not declared_exits or len(set(declared_exits)) != len(declared_exits):
        raise ValueError('exit nodes must be a non-empty unique list')
    if any(node not in event_by_name for node in declared_exits):
        raise ValueError('exit set names an unknown event')
    graph_exits = {name for name, count in outgoing.items() if count == 0}
    if set(declared_exits) != graph_exits:
        raise ValueError('exit set must equal the terminal graph antichain')
    exit_time_indices = {event_by_name[node].time_index for node in declared_exits}
    if len(exit_time_indices) != 1:
        raise ValueError('all exit records must lie on one time layer')
    exit_index = next(iter(exit_time_indices))

    coordinates = tuple(
        (
            event.name,
            (
                event.time_index * clock_step,
                event.site[0] * lattice_spacing,
                event.site[1] * lattice_spacing,
                event.site[2] * lattice_spacing,
            ),
        )
        for event in events
    )
    spatial_volume = (
        spatial_shape[0]
        * spatial_shape[1]
        * spatial_shape[2]
        * lattice_spacing**3
    )
    if math.isinf(minimum_causal_margin):
        minimum_causal_margin = 0.0
    return CauchyEmbeddingCertificate(
        coordinates=coordinates,
        exit_nodes=declared_exits,
        exit_time=exit_index * clock_step,
        lattice_spacing=lattice_spacing,
        clock_step=clock_step,
        causal_speed=causal_speed,
        spatial_shape=spatial_shape,
        spatial_volume=spatial_volume,
        minimum_causal_margin=minimum_causal_margin,
    )


@dataclass(frozen=True)
class ScalarLatticeState:
    '''Classical Cauchy data for one free scalar on a periodic cubic lattice.'''

    field: np.ndarray
    momentum: np.ndarray
    spacing: float
    mass: float

    def __post_init__(self) -> None:
        field = np.asarray(self.field, dtype=np.float64)
        momentum = np.asarray(self.momentum, dtype=np.float64)
        if field.ndim != 3 or any(size == 0 for size in field.shape):
            raise ValueError('scalar field must be a non-empty three-dimensional array')
        if momentum.shape != field.shape:
            raise ValueError('field and momentum must have the same lattice shape')
        if not np.all(np.isfinite(field)) or not np.all(np.isfinite(momentum)):
            raise ValueError('scalar Cauchy data must be finite')
        if not math.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError('lattice spacing must be finite and positive')
        if not math.isfinite(self.mass) or self.mass < 0.0:
            raise ValueError('scalar mass must be finite and non-negative')
        object.__setattr__(self, 'field', field)
        object.__setattr__(self, 'momentum', momentum)

    @property
    def spatial_volume(self) -> float:
        return float(np.prod(self.field.shape)) * self.spacing**3


def zero_scalar_lattice_like(state: ScalarLatticeState) -> ScalarLatticeState:
    return ScalarLatticeState(
        field=np.zeros_like(state.field),
        momentum=np.zeros_like(state.momentum),
        spacing=state.spacing,
        mass=state.mass,
    )


def _forward_gradients(state: ScalarLatticeState) -> tuple[np.ndarray, ...]:
    return tuple(
        (np.roll(state.field, -1, axis=axis) - state.field) / state.spacing
        for axis in range(3)
    )


def cell_averaged_scalar_stress(state: ScalarLatticeState) -> np.ndarray:
    '''Return a continuum-form lattice average of contravariant T^{mu nu}.'''

    gradients = _forward_gradients(state)
    momentum = state.momentum
    gradient_square = sum(gradient * gradient for gradient in gradients)
    mass_term = state.mass * state.mass * state.field * state.field
    stress = np.zeros((4, 4), dtype=np.float64)
    stress[0, 0] = 0.5 * float(
        np.mean(momentum * momentum + gradient_square + mass_term)
    )
    for axis, gradient in enumerate(gradients):
        flux = -float(np.mean(momentum * gradient))
        stress[0, axis + 1] = flux
        stress[axis + 1, 0] = flux
    isotropic = 0.5 * (momentum * momentum - gradient_square - mass_term)
    for left in range(3):
        for right in range(3):
            value = float(np.mean(gradients[left] * gradients[right]))
            if left == right:
                value += float(np.mean(isotropic))
            stress[left + 1, right + 1] = value
    return stress


def scalar_lattice_energy(state: ScalarLatticeState) -> float:
    return state.spatial_volume * float(cell_averaged_scalar_stress(state)[0, 0])


def rescale_scalar_to_total_energy(
    template: ScalarLatticeState,
    *,
    target_energy: float,
) -> ScalarLatticeState:
    '''Fix a declared scalar profile amplitude by an exact energy ledger.'''

    if not math.isfinite(target_energy) or target_energy <= 0.0:
        raise ValueError('target scalar energy must be finite and positive')
    template_energy = scalar_lattice_energy(template)
    if template_energy <= 0.0:
        raise ValueError('scalar template must carry positive quadratic energy')
    amplitude = math.sqrt(target_energy / template_energy)
    return ScalarLatticeState(
        field=amplitude * template.field,
        momentum=amplitude * template.momentum,
        spacing=template.spacing,
        mass=template.mass,
    )


@dataclass(frozen=True)
class ScalarTransitionMatching:
    complement: ScalarLatticeState
    residual: ScalarLatticeState
    residual_efficiency: float
    source_stress_before: np.ndarray
    complement_stress_after: np.ndarray
    residual_stress_after: np.ndarray
    total_stress_residual: np.ndarray
    source_energy_before: float
    complement_energy_after: float
    residual_energy_after: float
    total_energy_residual: float
    partition_stress_residual: float


def split_identical_scalar_species(
    source: ScalarLatticeState,
    *,
    residual_efficiency: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ScalarTransitionMatching:
    '''Rotate a source scalar into identical complement and residual species.'''

    if (
        not math.isfinite(residual_efficiency)
        or not 0.0 <= residual_efficiency <= 1.0
    ):
        raise ValueError('residual efficiency must lie in [0, 1]')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    residual_amplitude = math.sqrt(residual_efficiency)
    complement_amplitude = math.sqrt(1.0 - residual_efficiency)
    complement = ScalarLatticeState(
        field=complement_amplitude * source.field,
        momentum=complement_amplitude * source.momentum,
        spacing=source.spacing,
        mass=source.mass,
    )
    residual = ScalarLatticeState(
        field=residual_amplitude * source.field,
        momentum=residual_amplitude * source.momentum,
        spacing=source.spacing,
        mass=source.mass,
    )
    source_stress = cell_averaged_scalar_stress(source)
    complement_stress = cell_averaged_scalar_stress(complement)
    residual_stress = cell_averaged_scalar_stress(residual)
    total_stress_residual = (
        source_stress - complement_stress - residual_stress
    )
    expected_residual = residual_efficiency * source_stress
    expected_complement = (1.0 - residual_efficiency) * source_stress
    partition_stress_residual = max(
        float(np.max(np.abs(residual_stress - expected_residual))),
        float(np.max(np.abs(complement_stress - expected_complement))),
    )
    source_energy = scalar_lattice_energy(source)
    complement_energy = scalar_lattice_energy(complement)
    residual_energy = scalar_lattice_energy(residual)
    total_energy_residual = source_energy - complement_energy - residual_energy
    scale = max(1.0, float(np.max(np.abs(source_stress))), abs(source_energy))
    if float(np.max(np.abs(total_stress_residual))) > tolerance * scale:
        raise ArithmeticError('species rotation failed total-stress matching')
    if partition_stress_residual > tolerance * scale:
        raise ArithmeticError('species rotation failed the eta stress partition')
    if abs(total_energy_residual) > tolerance * scale:
        raise ArithmeticError('species rotation failed total-energy matching')
    return ScalarTransitionMatching(
        complement=complement,
        residual=residual,
        residual_efficiency=residual_efficiency,
        source_stress_before=source_stress,
        complement_stress_after=complement_stress,
        residual_stress_after=residual_stress,
        total_stress_residual=total_stress_residual,
        source_energy_before=source_energy,
        complement_energy_after=complement_energy,
        residual_energy_after=residual_energy,
        total_energy_residual=total_energy_residual,
        partition_stress_residual=partition_stress_residual,
    )


def lattice_mode_frequency(
    wave_vector: tuple[float, float, float],
    *,
    spacing: float,
    mass: float,
) -> float:
    '''Exact normal-mode frequency of the nearest-neighbor scalar lattice.'''

    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError('lattice spacing must be finite and positive')
    if not math.isfinite(mass) or mass < 0.0:
        raise ValueError('mass must be finite and non-negative')
    if len(wave_vector) != 3 or not all(
        math.isfinite(component) for component in wave_vector
    ):
        raise ValueError('wave vector must have three finite components')
    frequency_squared = mass * mass + 4.0 / (spacing * spacing) * sum(
        math.sin(0.5 * spacing * component) ** 2
        for component in wave_vector
    )
    return math.sqrt(frequency_squared)


def continuum_mode_frequency(
    wave_vector: tuple[float, float, float],
    *,
    mass: float,
) -> float:
    if len(wave_vector) != 3 or not all(
        math.isfinite(component) for component in wave_vector
    ):
        raise ValueError('wave vector must have three finite components')
    if not math.isfinite(mass) or mass < 0.0:
        raise ValueError('mass must be finite and non-negative')
    return math.sqrt(mass * mass + sum(value * value for value in wave_vector))


@dataclass(frozen=True)
class VacuumCauchyTransition:
    '''One-slice energy match to a declared covariant constant vacuum action.'''

    battery_energy_before: float
    residual_efficiency: float
    transferred_energy: float
    complement_energy_after: float
    spatial_volume: float
    vacuum_density: float
    vacuum_stress: np.ndarray
    no_double_counting_residual: float
    energy_preserving_rotation_commutator_residual: float
    battery_energy_after: float
    vacuum_register_energy_after: float
    unitary_total_energy_residual: float


def match_constant_vacuum_on_cauchy_slice(
    *,
    battery_energy: float,
    spatial_volume: float,
    residual_efficiency: float,
) -> VacuumCauchyTransition:
    '''Match a battery gap to S_vac = -rho_vac integral sqrt(-g) on one slice.

    The equal-gap two-state swap certifies energy matching at the declared
    Cauchy geometry.  The covariant action supplies T^{mu nu} = -rho g^{mu nu}
    and hence constant-vacuum conservation.  This does not predict the
    efficiency, volume, or vacuum density from a Born kernel.
    '''

    if not math.isfinite(battery_energy) or battery_energy <= 0.0:
        raise ValueError('battery energy must be finite and positive')
    if not math.isfinite(spatial_volume) or spatial_volume <= 0.0:
        raise ValueError('spatial volume must be finite and positive')
    if (
        not math.isfinite(residual_efficiency)
        or not 0.0 <= residual_efficiency <= 1.0
    ):
        raise ValueError('residual efficiency must lie in [0, 1]')
    transferred_energy = residual_efficiency * battery_energy
    complement_energy = battery_energy - transferred_energy
    vacuum_density = transferred_energy / spatial_volume
    vacuum_stress = np.diag(
        [
            vacuum_density,
            -vacuum_density,
            -vacuum_density,
            -vacuum_density,
        ]
    )
    no_double_counting_residual = (
        battery_energy - complement_energy - vacuum_density * spatial_volume
    )
    battery_number = np.diag([0.0, 0.0, 1.0, 1.0])
    vacuum_number = np.diag([0.0, 1.0, 0.0, 1.0])
    battery_hamiltonian = battery_energy * battery_number
    vacuum_hamiltonian = battery_energy * vacuum_number
    total_hamiltonian = battery_hamiltonian + vacuum_hamiltonian
    residual_amplitude = math.sqrt(residual_efficiency)
    complement_amplitude = math.sqrt(1.0 - residual_efficiency)
    swap_unitary = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, complement_amplitude, residual_amplitude, 0.0],
            [0.0, -residual_amplitude, complement_amplitude, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    initial_density = np.zeros((4, 4))
    initial_density[2, 2] = 1.0
    final_density = swap_unitary @ initial_density @ swap_unitary.T
    battery_energy_after = float(np.trace(battery_hamiltonian @ final_density))
    vacuum_register_energy_after = float(
        np.trace(vacuum_hamiltonian @ final_density)
    )
    swap_commutator_residual = float(
        np.linalg.norm(
            swap_unitary @ total_hamiltonian
            - total_hamiltonian @ swap_unitary,
            ord='fro',
        )
    )
    unitary_total_energy_residual = (
        battery_energy
        - battery_energy_after
        - vacuum_register_energy_after
    )
    return VacuumCauchyTransition(
        battery_energy_before=battery_energy,
        residual_efficiency=residual_efficiency,
        transferred_energy=transferred_energy,
        complement_energy_after=complement_energy,
        spatial_volume=spatial_volume,
        vacuum_density=vacuum_density,
        vacuum_stress=vacuum_stress,
        no_double_counting_residual=no_double_counting_residual,
        energy_preserving_rotation_commutator_residual=swap_commutator_residual,
        battery_energy_after=battery_energy_after,
        vacuum_register_energy_after=vacuum_register_energy_after,
        unitary_total_energy_residual=unitary_total_energy_residual,
    )


class DarkReadoutChoice(str, Enum):
    DUST = 'dust'
    SCALAR = 'scalar'
    VACUUM = 'vacuum'


@dataclass(frozen=True)
class SelectedDarkReadout:
    '''Exactly one stress readout selected from mutually exclusive alternatives.'''

    choice: DarkReadoutChoice
    stress: np.ndarray

    @property
    def energy_density(self) -> float:
        return float(self.stress[0, 0])


def select_single_dark_readout(
    choice: DarkReadoutChoice,
    *,
    dust_stress: np.ndarray,
    scalar_stress: np.ndarray,
    vacuum_stress: np.ndarray,
) -> SelectedDarkReadout:
    '''Return one alternative stress tensor and never their sum.'''

    if not isinstance(choice, DarkReadoutChoice):
        raise ValueError('choice must be one DarkReadoutChoice value')
    candidates = {
        DarkReadoutChoice.DUST: np.asarray(dust_stress, dtype=np.float64),
        DarkReadoutChoice.SCALAR: np.asarray(scalar_stress, dtype=np.float64),
        DarkReadoutChoice.VACUUM: np.asarray(vacuum_stress, dtype=np.float64),
    }
    for stress in candidates.values():
        if stress.shape != (4, 4) or not np.all(np.isfinite(stress)):
            raise ValueError('every candidate stress must be a finite 4 by 4 tensor')
    return SelectedDarkReadout(
        choice=choice,
        stress=candidates[choice].copy(),
    )


@dataclass(frozen=True)
class FlatFLRWScalarCauchyWitness:
    energy_density: float
    pressure_diagonal: tuple[float, float, float]
    newton_constant: float
    hubble_rate: float
    hamiltonian_residual: float
    momentum_residual: tuple[float, float, float]


def flat_flrw_scalar_cauchy_witness(
    state: ScalarLatticeState,
    *,
    newton_constant: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatFLRWScalarCauchyWitness:
    '''Embed homogeneous scalar data in flat-torus GR initial constraints.'''

    if not math.isfinite(newton_constant) or newton_constant <= 0.0:
        raise ValueError('Newton constant must be finite and positive')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    if (
        float(np.max(np.abs(state.field - np.mean(state.field)))) > tolerance
        or float(np.max(np.abs(state.momentum - np.mean(state.momentum))))
        > tolerance
    ):
        raise ValueError('flat FLRW scalar witness requires homogeneous Cauchy data')
    stress = cell_averaged_scalar_stress(state)
    density = float(stress[0, 0])
    if density <= 0.0:
        raise ValueError('flat FLRW scalar witness requires positive energy density')
    hubble = math.sqrt(8.0 * math.pi * newton_constant * density / 3.0)
    hamiltonian_residual = 6.0 * hubble * hubble - (
        16.0 * math.pi * newton_constant * density
    )
    momentum_residual = tuple(float(stress[0, index]) for index in range(1, 4))
    if any(abs(value) > tolerance for value in momentum_residual):
        raise ArithmeticError('homogeneous scalar data have nonzero momentum density')
    return FlatFLRWScalarCauchyWitness(
        energy_density=density,
        pressure_diagonal=tuple(float(stress[index, index]) for index in range(1, 4)),
        newton_constant=newton_constant,
        hubble_rate=hubble,
        hamiltonian_residual=hamiltonian_residual,
        momentum_residual=momentum_residual,
    )
