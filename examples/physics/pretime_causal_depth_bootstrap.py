"""Conditional pre-time causal-depth and Planck-rendering audit.

The finite witness keeps three statements separate.

1. A dimensionless Feynman--Kitaev history constraint stores an ordered
   sequence without using a physical time parameter.  It is a static ground
   state constraint, not a Hamiltonian that dynamically fires one gate per
   Planck tick.
2. A separately supplied nearest-neighbour circuit has an exact graph causal
   cone.  Planck matching uses only the dimensionless ratios
   ``alpha_t=Delta t/t_P`` and ``alpha_l=ell/ell_P`` with
   ``v_front/c=alpha_l/alpha_t``.
3. A physical gate completed in ``Delta t`` is subject to a quantum speed
   limit.  The corresponding generator-energy spread is distinct from the
   equal bare excitation gap in the battery receipt.

The audit does not derive a first seed, a clock arrow, Planck units, a durable
pointer, a covariant stress tensor, dark matter, dark energy, or an
observational prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from collections.abc import Sequence

import numpy as np

from examples.physics.causal_quantum_domino import (
    CausalQuantumDominoCertificate,
    certify_causal_quantum_domino,
    homogeneous_continuous_time_early_arrival_probability,
)


DEFAULT_TOLERANCE = 1.0e-10


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: int, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def pointer_rotation(theta: float) -> np.ndarray:
    """Return a real two-state pointer rotation by a dimensionless angle."""

    theta = float(theta)
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5 * math.pi:
        raise ValueError("theta must be finite and lie in [0, pi/2]")
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.complex128)


def _validated_unitaries(
    unitaries: Sequence[np.ndarray], *, tolerance: float
) -> tuple[np.ndarray, ...]:
    normalized: list[np.ndarray] = []
    dimension: int | None = None
    for index, raw in enumerate(unitaries):
        unitary = np.asarray(raw, dtype=np.complex128)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError(f"unitary[{index}] must be square")
        if dimension is None:
            dimension = unitary.shape[0]
        if unitary.shape != (dimension, dimension):
            raise ValueError("all unitaries must have the same dimension")
        identity = np.eye(dimension, dtype=np.complex128)
        residual = max(
            float(np.linalg.norm(unitary.conj().T @ unitary - identity, ord="fro")),
            float(np.linalg.norm(unitary @ unitary.conj().T - identity, ord="fro")),
        )
        if residual > tolerance * math.sqrt(dimension):
            raise ValueError(f"unitary[{index}] is not unitary within tolerance")
        normalized.append(unitary)
    return tuple(normalized)


def feynman_kitaev_propagation_hamiltonian(
    unitaries: Sequence[np.ndarray], *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    """Return the dimensionless legal-clock propagation constraint.

    For work dimension ``d`` and clock labels ``0,...,N``, the blocks are

    ``H_nn += I/2``, ``H_(n+1,n+1) += I/2``,
    ``H_(n,n+1) += -U_n^dagger/2``, and
    ``H_(n+1,n) += -U_n/2``.
    """

    tolerance = _positive_finite(tolerance, "tolerance")
    normalized = _validated_unitaries(unitaries, tolerance=tolerance)
    if not normalized:
        raise ValueError("at least one unitary is required")
    depth = len(normalized)
    dimension = normalized[0].shape[0]
    hamiltonian = np.zeros(
        ((depth + 1) * dimension, (depth + 1) * dimension),
        dtype=np.complex128,
    )
    identity = np.eye(dimension, dtype=np.complex128)
    for step, unitary in enumerate(normalized):
        left = slice(step * dimension, (step + 1) * dimension)
        right = slice((step + 1) * dimension, (step + 2) * dimension)
        hamiltonian[left, left] += 0.5 * identity
        hamiltonian[right, right] += 0.5 * identity
        hamiltonian[left, right] += -0.5 * unitary.conj().T
        hamiltonian[right, left] += -0.5 * unitary
    return hamiltonian


def feynman_kitaev_history_state(
    unitaries: Sequence[np.ndarray],
    initial_state: Sequence[complex],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return the normalized legal-clock history state for one supplied seed."""

    tolerance = _positive_finite(tolerance, "tolerance")
    normalized = _validated_unitaries(unitaries, tolerance=tolerance)
    if not normalized:
        raise ValueError("at least one unitary is required")
    dimension = normalized[0].shape[0]
    state = np.asarray(initial_state, dtype=np.complex128)
    if state.shape != (dimension,):
        raise ValueError("initial_state has the wrong dimension")
    norm = float(np.linalg.norm(state))
    if not math.isfinite(norm) or norm <= tolerance:
        raise ValueError("initial_state must have non-zero finite norm")
    current = state / norm
    snapshots = [current]
    for unitary in normalized:
        current = unitary @ current
        snapshots.append(current)
    return np.concatenate(snapshots) / math.sqrt(len(snapshots))


@dataclass(frozen=True)
class HistoryConstraintAudit:
    depth: int
    work_dimension: int
    legal_clock_dimension: int
    matrix_dimension: int
    hermiticity_residual: float
    minimum_eigenvalue: float
    analytic_spectrum_residual: float
    numerical_kernel_dimension: int
    expected_kernel_dimension: int
    spectral_gap: float
    analytic_spectral_gap: float
    history_norm_residual: float
    history_constraint_residual: float
    maximum_conditioned_history_residual: float
    clock_uniformity_residual: float
    alternate_seed_constraint_residual: float
    seed_history_overlap: float
    reverse_arrow_spectrum_residual: float
    reverse_arrow_clock_reflection_residual: float
    positive_semidefinite: bool
    history_constraint_closed: bool
    seed_is_unique: bool = False
    arrow_is_unique: bool = False
    physical_tick_derived: bool = False
    static_constraint_is_real_time_generator: bool = False
    clock_register_and_boundary_are_supplied: bool = True
    status: str = "DIMENSIONLESS_HISTORY_CONSTRAINT_CLOSED_SEED_AND_ARROW_OPEN"


def audit_history_constraint(
    *,
    depth: int = 3,
    theta: float = math.pi / 5.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> HistoryConstraintAudit:
    """Audit a finite two-state Feynman--Kitaev history constraint."""

    depth = _positive_integer(depth, "depth")
    tolerance = _positive_finite(tolerance, "tolerance")
    unitary = pointer_rotation(theta)
    unitaries = tuple(unitary.copy() for _ in range(depth))
    hamiltonian = feynman_kitaev_propagation_hamiltonian(
        unitaries, tolerance=tolerance
    )
    initial = np.asarray((1.0, 0.0), dtype=np.complex128)
    alternate = np.asarray((0.0, 1.0), dtype=np.complex128)
    history = feynman_kitaev_history_state(
        unitaries, initial, tolerance=tolerance
    )
    alternate_history = feynman_kitaev_history_state(
        unitaries, alternate, tolerance=tolerance
    )

    hermiticity_residual = float(
        np.linalg.norm(hamiltonian - hamiltonian.conj().T, ord="fro")
    )
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    analytic_base = np.asarray(
        [1.0 - math.cos(index * math.pi / (depth + 1)) for index in range(depth + 1)],
        dtype=float,
    )
    analytic_spectrum = np.sort(np.repeat(analytic_base, 2))
    analytic_spectrum_residual = float(
        np.max(np.abs(np.sort(eigenvalues) - analytic_spectrum))
    )
    numerical_kernel_dimension = int(np.count_nonzero(np.abs(eigenvalues) <= tolerance))

    snapshots = history.reshape(depth + 1, 2)
    current = initial.copy()
    conditioned_residuals: list[float] = []
    clock_probabilities: list[float] = []
    for step, snapshot in enumerate(snapshots):
        clock_probability = float(np.vdot(snapshot, snapshot).real)
        clock_probabilities.append(clock_probability)
        conditioned = snapshot / math.sqrt(clock_probability)
        conditioned_residuals.append(float(np.linalg.norm(conditioned - current)))
        if step < depth:
            current = unitaries[step] @ current

    reversed_unitaries = tuple(
        candidate.conj().T for candidate in reversed(unitaries)
    )
    reverse_hamiltonian = feynman_kitaev_propagation_hamiltonian(
        reversed_unitaries, tolerance=tolerance
    )
    reverse_arrow_spectrum_residual = float(
        np.max(
            np.abs(
                np.sort(np.linalg.eigvalsh(reverse_hamiltonian))
                - np.sort(eigenvalues)
            )
        )
    )
    clock_reflection = np.kron(
        np.fliplr(np.eye(depth + 1, dtype=np.complex128)),
        np.eye(2, dtype=np.complex128),
    )
    reverse_arrow_clock_reflection_residual = float(
        np.linalg.norm(
            reverse_hamiltonian
            - clock_reflection @ hamiltonian @ clock_reflection.conj().T,
            ord=fro,
        )
    )

    history_norm_residual = abs(float(np.vdot(history, history).real) - 1.0)
    history_constraint_residual = float(np.linalg.norm(hamiltonian @ history))
    alternate_seed_constraint_residual = float(
        np.linalg.norm(hamiltonian @ alternate_history)
    )
    seed_history_overlap = abs(complex(np.vdot(history, alternate_history)))
    expected_clock_probability = 1.0 / (depth + 1)
    clock_uniformity_residual = max(
        abs(probability - expected_clock_probability)
        for probability in clock_probabilities
    )
    positive_semidefinite = float(np.min(eigenvalues)) >= -tolerance
    closed = (
        hermiticity_residual <= tolerance
        and positive_semidefinite
        and analytic_spectrum_residual <= tolerance
        and numerical_kernel_dimension == 2
        and history_norm_residual <= tolerance
        and history_constraint_residual <= tolerance
        and alternate_seed_constraint_residual <= tolerance
        and seed_history_overlap <= tolerance
        and max(conditioned_residuals) <= tolerance
        and clock_uniformity_residual <= tolerance
        and reverse_arrow_spectrum_residual <= tolerance
        and reverse_arrow_clock_reflection_residual <= tolerance
    )
    return HistoryConstraintAudit(
        depth=depth,
        work_dimension=2,
        legal_clock_dimension=depth + 1,
        matrix_dimension=2 * (depth + 1),
        hermiticity_residual=hermiticity_residual,
        minimum_eigenvalue=float(np.min(eigenvalues)),
        analytic_spectrum_residual=analytic_spectrum_residual,
        numerical_kernel_dimension=numerical_kernel_dimension,
        expected_kernel_dimension=2,
        spectral_gap=float(eigenvalues[2]),
        analytic_spectral_gap=1.0 - math.cos(math.pi / (depth + 1)),
        history_norm_residual=history_norm_residual,
        history_constraint_residual=history_constraint_residual,
        maximum_conditioned_history_residual=max(conditioned_residuals),
        clock_uniformity_residual=clock_uniformity_residual,
        alternate_seed_constraint_residual=alternate_seed_constraint_residual,
        seed_history_overlap=seed_history_overlap,
        reverse_arrow_spectrum_residual=reverse_arrow_spectrum_residual,
        reverse_arrow_clock_reflection_residual=(
            reverse_arrow_clock_reflection_residual
        ),
        positive_semidefinite=positive_semidefinite,
        history_constraint_closed=closed,
    )


def _pure_density(state: np.ndarray) -> np.ndarray:
    return np.outer(state, state.conj())


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left - right, dtype=np.complex128)
    difference = 0.5 * (difference + difference.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


@dataclass(frozen=True)
class SelfNonidentityStitchAudit:
    microsteps_per_planck_render: int
    zero_d_event_register_dimension: int
    declared_render_algebra_dimension: int
    step_angle: float
    dimensionless_depth_increment: float
    adjacent_state_changes: tuple[float, ...]
    adjacent_bures_angles: tuple[float, ...]
    normalized_stitch_coordinates: tuple[float, ...]
    total_bures_arclength: float
    endpoint_state_change: float
    stitch_coordinate_uniformity_residual: float
    joint_event_render_history_norm_residual: float
    faithful_renderer_distance_residual: float
    identity_channel_state_change: float
    erasing_probe_event_change: float
    erasing_probe_render_change: float
    recurrence_first_step_change: float
    recurrence_endpoint_change: float
    every_step_operationally_nonfixed: bool
    nonidentity_stitch_defined: bool
    stitch_spans_unit_render_interval: bool
    event_register_and_rendered_slices_coencoded: bool
    faithful_renderer_preserves_change: bool
    identity_counterexample_closed: bool
    erasing_renderer_counterexample_closed: bool
    recurrence_counterexample_closed: bool
    literal_self_nonidentity_asserted: bool = False
    zero_d_means_event_type: bool = True
    render_algebra_dimension_is_spacetime_dimension: bool = False
    four_dimensional_geometry_derived: bool = False
    physical_time_from_change_derived: bool = False
    arrow_of_time_derived: bool = False
    selective_measurement_implemented: bool = False
    renderer_and_slice_identification_supplied: bool = True
    planck_endpoint_calibration_supplied: bool = True
    unique_stitch_law_derived: bool = False
    status: str = (
        "OPERATIONAL_NONIDENTITY_STITCH_CLOSED_RENDERER_TIME_AND_GEOMETRY_SUPPLIED"
    )


def _self_nonidentity_path_data(
    microsteps: int, theta: float, tolerance: float
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
]:
    unitary = pointer_rotation(theta)
    state = np.asarray((1.0, 0.0), dtype=np.complex128)
    states = [state]
    for _ in range(microsteps):
        state = unitary @ state
        states.append(state)
    state_tuple = tuple(states)
    densities = tuple(_pure_density(candidate) for candidate in state_tuple)
    changes = tuple(
        _trace_distance(densities[index], densities[index + 1])
        for index in range(microsteps)
    )
    angles = tuple(
        math.acos(
            min(
                1.0,
                abs(
                    complex(
                        np.vdot(state_tuple[index], state_tuple[index + 1])
                    )
                ),
            )
        )
        for index in range(microsteps)
    )
    total_angle = float(sum(angles))
    if total_angle <= tolerance:
        coordinates = tuple(0.0 for _ in range(microsteps + 1))
        uniformity_residual = 1.0
    else:
        cumulative = 0.0
        coordinate_values = [0.0]
        for angle in angles:
            cumulative += angle
            coordinate_values.append(cumulative / total_angle)
        coordinates = tuple(coordinate_values)
        uniformity_residual = max(
            abs(value - index / microsteps)
            for index, value in enumerate(coordinates)
        )
    return (
        state_tuple,
        densities,
        changes,
        angles,
        coordinates,
        total_angle,
        uniformity_residual,
    )


def _self_nonidentity_render_witness(
    states: tuple[np.ndarray, ...],
    densities: tuple[np.ndarray, ...],
) -> tuple[float, float, float, float, float, float, float]:
    embedding = np.zeros((4, 2), dtype=np.complex128)
    embedding[0, 0] = 1.0
    embedding[1, 1] = 1.0
    rendered_states = tuple(embedding @ candidate for candidate in states)
    rendered_densities = tuple(
        _pure_density(candidate) for candidate in rendered_states
    )
    joint_history = np.concatenate(rendered_states) / math.sqrt(len(states))
    joint_norm_residual = abs(
        float(np.vdot(joint_history, joint_history).real) - 1.0
    )
    event_changes = tuple(
        _trace_distance(densities[index], densities[index + 1])
        for index in range(len(densities) - 1)
    )
    render_changes = tuple(
        _trace_distance(rendered_densities[index], rendered_densities[index + 1])
        for index in range(len(rendered_densities) - 1)
    )
    faithful_residual = max(
        (
            abs(rendered - event)
            for rendered, event in zip(
                render_changes, event_changes, strict=True
            )
        ),
        default=0.0,
    )
    initial_density = densities[0]
    identity_change = _trace_distance(initial_density, initial_density)
    orthogonal_density = _pure_density(
        np.asarray((0.0, 1.0), dtype=np.complex128)
    )
    erasing_event_change = _trace_distance(initial_density, orthogonal_density)
    fixed_render = np.zeros((4, 4), dtype=np.complex128)
    fixed_render[0, 0] = 1.0
    erasing_render_change = _trace_distance(fixed_render, fixed_render)

    recurrence_unitary = pointer_rotation(0.5 * math.pi)
    recurrence_state_1 = recurrence_unitary @ states[0]
    recurrence_state_2 = recurrence_unitary @ recurrence_state_1
    recurrence_first_change = _trace_distance(
        initial_density, _pure_density(recurrence_state_1)
    )
    recurrence_endpoint_change = _trace_distance(
        initial_density, _pure_density(recurrence_state_2)
    )
    return (
        joint_norm_residual,
        faithful_residual,
        identity_change,
        erasing_event_change,
        erasing_render_change,
        recurrence_first_change,
        recurrence_endpoint_change,
    )


def audit_self_nonidentity_stitch(
    *,
    microsteps_per_planck_render: int = 4,
    theta: float = math.pi / 5.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SelfNonidentityStitchAudit:
    """Normalize dimensionless boundary changes to one rendered interval."""

    microsteps = _positive_integer(
        microsteps_per_planck_render, "microsteps_per_planck_render"
    )
    tolerance = _positive_finite(tolerance, "tolerance")
    pointer_rotation(theta)
    path_data = _self_nonidentity_path_data(microsteps, theta, tolerance)
    states, densities, changes, angles, coordinates = path_data[:5]
    total_angle, uniformity_residual = path_data[5:]
    render_data = _self_nonidentity_render_witness(states, densities)
    (
        joint_norm_residual,
        faithful_residual,
        identity_change,
        erasing_event_change,
        erasing_render_change,
        recurrence_first_change,
        recurrence_endpoint_change,
    ) = render_data
    every_step_nonfixed = all(change > tolerance for change in changes)
    stitch_defined = total_angle > tolerance
    spans_interval = bool(
        stitch_defined
        and abs(coordinates[0]) <= tolerance
        and abs(coordinates[-1] - 1.0) <= tolerance
    )
    coencoded = joint_norm_residual <= tolerance
    faithful = faithful_residual <= tolerance
    identity_counterexample = identity_change <= tolerance
    erasing_counterexample = bool(
        erasing_event_change > tolerance
        and erasing_render_change <= tolerance
    )
    recurrence_counterexample = bool(
        recurrence_first_change > tolerance
        and recurrence_endpoint_change <= tolerance
    )
    closed = bool(
        every_step_nonfixed
        and spans_interval
        and coencoded
        and faithful
        and identity_counterexample
        and erasing_counterexample
        and recurrence_counterexample
    )
    status = (
        "OPERATIONAL_NONIDENTITY_STITCH_CLOSED_RENDERER_TIME_AND_GEOMETRY_SUPPLIED"
        if closed
        else "OPERATIONAL_NONIDENTITY_STITCH_DEGENERATE_OR_AUDIT_FAILED"
    )
    return SelfNonidentityStitchAudit(
        microsteps_per_planck_render=microsteps,
        zero_d_event_register_dimension=microsteps + 1,
        declared_render_algebra_dimension=4,
        step_angle=float(theta),
        dimensionless_depth_increment=1.0 / microsteps,
        adjacent_state_changes=changes,
        adjacent_bures_angles=angles,
        normalized_stitch_coordinates=coordinates,
        total_bures_arclength=total_angle,
        endpoint_state_change=_trace_distance(densities[0], densities[-1]),
        stitch_coordinate_uniformity_residual=uniformity_residual,
        joint_event_render_history_norm_residual=joint_norm_residual,
        faithful_renderer_distance_residual=faithful_residual,
        identity_channel_state_change=identity_change,
        erasing_probe_event_change=erasing_event_change,
        erasing_probe_render_change=erasing_render_change,
        recurrence_first_step_change=recurrence_first_change,
        recurrence_endpoint_change=recurrence_endpoint_change,
        every_step_operationally_nonfixed=every_step_nonfixed,
        nonidentity_stitch_defined=stitch_defined,
        stitch_spans_unit_render_interval=spans_interval,
        event_register_and_rendered_slices_coencoded=coencoded,
        faithful_renderer_preserves_change=faithful,
        identity_counterexample_closed=identity_counterexample,
        erasing_renderer_counterexample_closed=erasing_counterexample,
        recurrence_counterexample_closed=recurrence_counterexample,
        status=status,
    )


@dataclass(frozen=True)
class PlanckRenderingMatchingAudit:
    microsteps_per_planck_render: int
    micro_time_over_planck_time: float
    micro_length_over_planck_length: float
    front_speed_over_c: float
    block_time_over_planck_time: float
    block_length_over_planck_length: float
    block_front_speed_over_c: float
    full_planck_cell_one_microstep_speed_over_c: float
    subplanck_time: bool
    subplanck_length: bool
    local_causality_satisfied: bool
    full_planck_cell_one_microstep_causal: bool
    planck_units_enter_probability_core: bool = False
    planck_time_is_minimum_time_theorem: bool = False
    planck_scale_derived_from_zero_d: bool = False
    physical_time_is_post_rendering_map: bool = True
    matching_is_adopted: bool = True
    status: str = "DIMENSIONLESS_PLANCK_RENDERING_MATCHING_CONDITIONAL"


def audit_planck_rendering_matching(
    *,
    microsteps_per_planck_render: int = 4,
    micro_length_over_planck_length: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PlanckRenderingMatchingAudit:
    """Audit ``v/c=(ell/ell_P)/(Delta t/t_P)`` without using SI values."""

    microsteps = _positive_integer(
        microsteps_per_planck_render, "microsteps_per_planck_render"
    )
    tolerance = _positive_finite(tolerance, "tolerance")
    alpha_t = 1.0 / microsteps
    alpha_l = (
        alpha_t
        if micro_length_over_planck_length is None
        else _positive_finite(
            micro_length_over_planck_length,
            "micro_length_over_planck_length",
        )
    )
    front_speed = alpha_l / alpha_t
    block_time = microsteps * alpha_t
    block_length = microsteps * alpha_l
    block_speed = block_length / block_time
    full_cell_speed = 1.0 / alpha_t
    return PlanckRenderingMatchingAudit(
        microsteps_per_planck_render=microsteps,
        micro_time_over_planck_time=alpha_t,
        micro_length_over_planck_length=alpha_l,
        front_speed_over_c=front_speed,
        block_time_over_planck_time=block_time,
        block_length_over_planck_length=block_length,
        block_front_speed_over_c=block_speed,
        full_planck_cell_one_microstep_speed_over_c=full_cell_speed,
        subplanck_time=alpha_t < 1.0,
        subplanck_length=alpha_l < 1.0,
        local_causality_satisfied=front_speed <= 1.0 + tolerance,
        full_planck_cell_one_microstep_causal=full_cell_speed <= 1.0 + tolerance,
    )


@dataclass(frozen=True)
class QuantumSpeedLimitAudit:
    theta: float
    trigger_probability: float
    micro_time_over_planck_time: float
    interaction_energy_cap_over_planck_energy: float
    minimum_generator_spread_over_planck_energy: float
    full_transfer_minimum_generator_spread_over_planck_energy: float
    maximum_angle_per_microstep_under_cap: float
    minimum_coherent_microsteps_under_cap: int
    minimum_discrete_total_time_over_planck_time: float
    single_microstep_not_excluded_by_cap_bound: bool
    deterministic_pointer_label_transfer: bool
    generator_spread_is_battery_gap: bool = False
    energy_cap_is_a_physical_axiom: bool = True
    bound_proves_realizability: bool = False
    status: str = "QUANTUM_SPEED_LIMIT_SEPARATED_FROM_BATTERY_RECEIPT"


def audit_quantum_speed_limit(
    *,
    theta: float,
    micro_time_over_planck_time: float,
    interaction_energy_cap_over_planck_energy: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> QuantumSpeedLimitAudit:
    """Return dimensionless Mandelstam--Tamm angle bounds.

    ``minimum_generator_spread_over_planck_energy`` is a lower bound on the
    relevant (possibly time-averaged) generator-energy spread.  It is not the
    target/battery bare excitation gap and it does not prove that the gate is
    dynamically realizable.
    """

    pointer_rotation(theta)
    alpha_t = _positive_finite(
        micro_time_over_planck_time, "micro_time_over_planck_time"
    )
    energy_cap = _positive_finite(
        interaction_energy_cap_over_planck_energy,
        "interaction_energy_cap_over_planck_energy",
    )
    tolerance = _positive_finite(tolerance, "tolerance")
    minimum_spread = theta / alpha_t
    full_transfer_spread = 0.5 * math.pi / alpha_t
    maximum_angle = energy_cap * alpha_t
    if theta == 0.0:
        minimum_steps = 0
    else:
        minimum_steps = max(
            1,
            math.ceil(theta / maximum_angle - tolerance),
        )
    probability = math.sin(theta) ** 2
    return QuantumSpeedLimitAudit(
        theta=float(theta),
        trigger_probability=probability,
        micro_time_over_planck_time=alpha_t,
        interaction_energy_cap_over_planck_energy=energy_cap,
        minimum_generator_spread_over_planck_energy=minimum_spread,
        full_transfer_minimum_generator_spread_over_planck_energy=(
            full_transfer_spread
        ),
        maximum_angle_per_microstep_under_cap=maximum_angle,
        minimum_coherent_microsteps_under_cap=minimum_steps,
        minimum_discrete_total_time_over_planck_time=minimum_steps * alpha_t,
        single_microstep_not_excluded_by_cap_bound=(
            minimum_spread <= energy_cap + tolerance
        ),
        deterministic_pointer_label_transfer=math.isclose(
            probability, 1.0, rel_tol=0.0, abs_tol=tolerance
        ),
    )


@dataclass(frozen=True)
class PretimeCausalDepthCertificate:
    history_constraint: HistoryConstraintAudit
    planck_matching: PlanckRenderingMatchingAudit
    quantum_speed_limit: QuantumSpeedLimitAudit
    self_nonidentity_stitch: SelfNonidentityStitchAudit
    local_domino: CausalQuantumDominoCertificate | None
    continuous_time_two_hop_early_arrival_probability: float
    exact_graph_cone_and_history_constraint_closed: bool
    formal_history_stitch_and_exact_graph_cone_closed: bool
    single_microstep_qsl_cap_necessary_condition_satisfied: bool
    parent_subplanck_tick_solves_bootstrap: bool
    first_seed_derived: bool
    clock_order_derived: bool
    arrow_of_time_derived: bool
    durable_physical_pointer_derived: bool
    zero_to_three_spatial_dimension_derived: bool
    record_to_covariant_stress_derived: bool
    dark_matter_dark_energy_derived: bool
    absolute_dark_scale_derived: bool
    unique_observational_prediction: bool
    required_adopted_inputs: tuple[str, ...]
    dimensionless_arguments: tuple[tuple[str, str], ...]
    status: str


def certify_pretime_causal_depth_bootstrap(
    *,
    microsteps_per_planck_render: int = 4,
    micro_length_over_planck_length: float | None = None,
    theta: float = 0.5 * math.pi,
    interaction_energy_cap_over_planck_energy: float = 1.0,
    history_depth: int = 3,
    domino_depth: int = 2,
    energy_gap: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PretimeCausalDepthCertificate:
    """Build the strongest finite pre-time witness without status promotion."""

    tolerance = _positive_finite(tolerance, "tolerance")
    matching = audit_planck_rendering_matching(
        microsteps_per_planck_render=microsteps_per_planck_render,
        micro_length_over_planck_length=micro_length_over_planck_length,
        tolerance=tolerance,
    )
    history = audit_history_constraint(
        depth=history_depth,
        theta=theta,
        tolerance=tolerance,
    )
    qsl = audit_quantum_speed_limit(
        theta=theta,
        micro_time_over_planck_time=matching.micro_time_over_planck_time,
        interaction_energy_cap_over_planck_energy=(
            interaction_energy_cap_over_planck_energy
        ),
        tolerance=tolerance,
    )
    stitch = audit_self_nonidentity_stitch(
        microsteps_per_planck_render=microsteps_per_planck_render,
        theta=theta,
        tolerance=tolerance,
    )
    domino: CausalQuantumDominoCertificate | None = None
    if matching.local_causality_satisfied:
        domino = certify_causal_quantum_domino(
            site_count=domino_depth + 2,
            depth=domino_depth,
            theta=theta,
            lattice_spacing=matching.micro_length_over_planck_length,
            clock_step=matching.micro_time_over_planck_time,
            causal_speed=1.0,
            energy_gap=_positive_finite(energy_gap, "energy_gap"),
            tolerance=tolerance,
        )
    early_tail = homogeneous_continuous_time_early_arrival_probability(
        rate_per_time=1.0,
        hops=2,
        elapsed_time=matching.micro_time_over_planck_time,
    )
    finite_closed = bool(
        history.history_constraint_closed
        and stitch.nonidentity_stitch_defined
        and stitch.stitch_spans_unit_render_interval
        and stitch.event_register_and_rendered_slices_coencoded
        and stitch.faithful_renderer_preserves_change
        and stitch.identity_counterexample_closed
        and stitch.erasing_renderer_counterexample_closed
        and stitch.recurrence_counterexample_closed
        and matching.local_causality_satisfied
        and domino is not None
        and domino.structural_causal_support_exact
        and domino.cptp_within_tolerance
        and domino.energy_conserved_within_tolerance
        and domino.energy_resolved_instrument_within_tolerance
        and early_tail > 0.0
    )
    return PretimeCausalDepthCertificate(
        history_constraint=history,
        planck_matching=matching,
        quantum_speed_limit=qsl,
        self_nonidentity_stitch=stitch,
        local_domino=domino,
        continuous_time_two_hop_early_arrival_probability=early_tail,
        exact_graph_cone_and_history_constraint_closed=finite_closed,
        formal_history_stitch_and_exact_graph_cone_closed=finite_closed,
        single_microstep_qsl_cap_necessary_condition_satisfied=(
            qsl.single_microstep_not_excluded_by_cap_bound
        ),
        parent_subplanck_tick_solves_bootstrap=False,
        first_seed_derived=False,
        clock_order_derived=False,
        arrow_of_time_derived=False,
        durable_physical_pointer_derived=False,
        zero_to_three_spatial_dimension_derived=False,
        record_to_covariant_stress_derived=False,
        dark_matter_dark_energy_derived=False,
        absolute_dark_scale_derived=False,
        unique_observational_prediction=False,
        required_adopted_inputs=(
            "finite legal clock labels and their order",
            "initial seed and low-entropy boundary state",
            "nearest-neighbour graph and scheduled local gates",
            "fresh equal-gap batteries and their readout basis",
            "Planck matching ratio between causal depth and rendered units",
            "ordered operational boundary changes in the event register",
            "faithful renderer, slice identification, and endpoint calibration",
            "interaction generator capable of the declared pointer angle",
            "durable local record and covariant source map",
        ),
        dimensionless_arguments=(
            ("n", "integer causal-depth label, not physical time"),
            ("theta", "pointer rotation angle"),
            ("sin(theta)^2", "one-edge pointer-label probability"),
            ("J_n=||rho_(n+1)-rho_n||_1/2", "boundary change readout"),
            ("A_n=acos|<psi_n|psi_(n+1)>", "pure-state Bures angle"),
            ("zeta_n=sum_(r<n)A_r/sum_r A_r", "normalized stitch coordinate"),
            ("alpha_t=Delta t/t_P", "rendered time ratio"),
            ("alpha_l=ell/ell_P", "rendered length ratio"),
            ("alpha_l/alpha_t", "front speed divided by c"),
            ("Delta E/E_P >= theta/alpha_t", "quantum-speed lower bound"),
        ),
        status=(
            "PRETIME_FORMAL_HISTORY_NONIDENTITY_STITCH_AND_EXACT_LOCAL_CONE_CLOSED_QSL_EXECUTION_AND_DARK_SOURCE_OPEN"
            if finite_closed
            else "PRETIME_HISTORY_OR_CAUSAL_MATCHING_AUDIT_FAILED"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microsteps", type=int, default=4)
    parser.add_argument("--micro-length-ratio", type=float)
    parser.add_argument("--theta", type=float, default=0.5 * math.pi)
    parser.add_argument("--energy-cap-ratio", type=float, default=1.0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    certificate = certify_pretime_causal_depth_bootstrap(
        microsteps_per_planck_render=args.microsteps,
        micro_length_over_planck_length=args.micro_length_ratio,
        theta=args.theta,
        interaction_energy_cap_over_planck_energy=args.energy_cap_ratio,
    )
    print(
        json.dumps(
            asdict(certificate),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0 if certificate.exact_graph_cone_and_history_constraint_closed else 2


if __name__ == "__main__":
    raise SystemExit(main())
