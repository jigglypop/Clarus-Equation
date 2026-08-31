"""Finite Born-selection kernel with energy and no-signalling boundaries.

For a finite grouped instrument ``{I_a}``, this module partitions a supplied
dimensionless seed ``u in [0, 1)`` into half-open intervals of Born length,

    p_a = Tr I_a(rho),       I_a = [C_{a-1}, C_a).

A uniform seed therefore samples the Born distribution and the seed average
of normalized posteriors equals the nonselective CPTP channel.  In floating
point arithmetic the raw Born traces and the explicitly normalized interval
weights are exposed separately.  The seed law is an explicit stochastic
axiom: neither a unitary premeasurement nor this inverse-CDF representation
derives physical randomness or an ontic single world.

The finite certificate also reuses a supplied energy-conserving collision.
Energy residuals and variances are divided by an independent positive energy
scale before comparison with a dimensionless tolerance.  It checks a local
Bell-pair nonselective marginal and locks two counterexamples: a controllable
seed can force remote conditional states, and outcome-scalar energy receipts
cannot close an arbitrary measurement energy ledger.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.quantum_instrument_record_kernel import (
    construct_energy_conserving_collision_instrument,
)


DEFAULT_TOLERANCE = 1.0e-12


def _positive_tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_density(state: np.ndarray, *, tolerance: float) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1] or density.shape[0] < 2:
        raise ValueError("state must be a square density matrix of dimension at least two")
    if not np.isfinite(density).all():
        raise ValueError("density matrix entries must be finite")
    if np.linalg.norm(density - density.conj().T, ord="fro") > tolerance:
        raise ValueError("density matrix must be Hermitian")
    trace = np.trace(density)
    if abs(float(trace.real) - 1.0) > tolerance or abs(float(trace.imag)) > tolerance:
        raise ValueError("density matrix must have unit trace")
    if float(np.linalg.eigvalsh(density).min()) < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")
    return density


@dataclass(frozen=True)
class CoarseOutcomeOperation:
    """One declared coarse outcome, possibly with several internal Kraus terms."""

    label: str
    operators: tuple[np.ndarray, ...]
    energy_transfer: float | None = None


def _validated_outcomes(
    outcomes: Sequence[CoarseOutcomeOperation],
    dimension: int,
) -> tuple[CoarseOutcomeOperation, ...]:
    declared = tuple(outcomes)
    if not declared:
        raise ValueError("instrument outcomes must be non-empty")
    if any(not isinstance(outcome, CoarseOutcomeOperation) for outcome in declared):
        raise TypeError("every outcome must be a CoarseOutcomeOperation")
    if any(not outcome.label for outcome in declared):
        raise ValueError("outcome labels must be non-empty")
    if len({outcome.label for outcome in declared}) != len(declared):
        raise ValueError("coarse outcome labels must be unique")
    for outcome in declared:
        if not outcome.operators:
            raise ValueError("every coarse outcome needs at least one Kraus operator")
        for operator in outcome.operators:
            matrix = np.asarray(operator, dtype=np.complex128)
            if matrix.shape != (dimension, dimension):
                raise ValueError("Kraus operators must match the state dimension")
            if not np.isfinite(matrix).all():
                raise ValueError("Kraus operator entries must be finite")
        if outcome.energy_transfer is not None and not math.isfinite(outcome.energy_transfer):
            raise ValueError("energy transfer must be finite when declared")
    return declared


def _outcome_output(outcome: CoarseOutcomeOperation, state: np.ndarray) -> np.ndarray:
    return sum(
        (
            np.asarray(operator, dtype=np.complex128)
            @ state
            @ np.asarray(operator, dtype=np.complex128).conj().T
            for operator in outcome.operators
        ),
        np.zeros_like(state),
    )


def instrument_completeness_residual(
    outcomes: Sequence[CoarseOutcomeOperation],
    dimension: int,
) -> float:
    """Return ``||sum K^dagger K - I||_2`` for all internal Kraus terms."""

    declared = _validated_outcomes(outcomes, dimension)
    completeness = sum(
        (
            np.asarray(operator, dtype=np.complex128).conj().T
            @ np.asarray(operator, dtype=np.complex128)
            for outcome in declared
            for operator in outcome.operators
        ),
        np.zeros((dimension, dimension), dtype=np.complex128),
    )
    return float(np.linalg.norm(completeness - np.eye(dimension), ord=2))


def born_probabilities(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, ...]:
    """Return coarse Born probabilities after a fail-closed instrument check."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    if instrument_completeness_residual(declared, density.shape[0]) > 10.0 * tol:
        raise ValueError("Kraus family must be complete")
    probabilities: list[float] = []
    for outcome in declared:
        probability = float(np.trace(_outcome_output(outcome, density)).real)
        if probability < -tol or not math.isfinite(probability):
            raise ArithmeticError("Born probability must be finite and nonnegative")
        probabilities.append(max(0.0, probability))
    if abs(math.fsum(probabilities) - 1.0) > 10.0 * tol:
        raise ArithmeticError("Born probabilities must sum to one")
    return tuple(probabilities)


@dataclass(frozen=True)
class SeedPartition:
    """Numerically normalized half-open partition of the unit seed interval."""

    input_probabilities: tuple[float, ...]
    cell_probabilities: tuple[float, ...]
    intervals: tuple[tuple[float, float], ...]
    input_normalization_residual: float


def build_seed_partition(
    probabilities: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SeedPartition:
    """Build ``[C_{a-1}, C_a)`` while keeping zero-probability cells empty."""

    tol = _positive_tolerance(tolerance)
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("probabilities must be non-empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and nonnegative")
    total = math.fsum(values)
    residual = abs(total - 1.0)
    if total <= 0.0 or residual > 10.0 * tol:
        raise ValueError("probabilities must sum to one within tolerance")

    # This explicit numerical normalization prevents a floating-point gap at
    # u -> 1^- while preserving exactly zero cells.  The input residual is
    # returned rather than hidden.
    normalized = tuple(value / total for value in values)
    intervals: list[tuple[float, float]] = []
    cumulative = 0.0
    last_positive_index = max(
        index for index, probability in enumerate(normalized) if probability > 0.0
    )
    for index, probability in enumerate(normalized):
        start = cumulative
        if probability == 0.0:
            end = start
        elif index == last_positive_index:
            end = 1.0
        else:
            end = math.fsum(normalized[: index + 1])
        cumulative = end
        intervals.append((start, end))
    return SeedPartition(
        input_probabilities=values,
        cell_probabilities=normalized,
        intervals=tuple(intervals),
        input_normalization_residual=residual,
    )


def select_partition_cell(partition: SeedPartition, seed: float) -> int:
    """Return the unique positive-measure cell containing ``seed``."""

    value = float(seed)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("seed must be finite and lie in [0, 1)")
    matches = tuple(
        index
        for index, ((start, end), probability) in enumerate(
            zip(partition.intervals, partition.cell_probabilities)
        )
        if probability > 0.0 and start <= value < end
    )
    if len(matches) != 1:
        raise ArithmeticError("valid seed must belong to exactly one positive interval")
    return matches[0]


@dataclass(frozen=True)
class OutcomeSelection:
    outcome_index: int
    label: str
    seed: float
    raw_born_probability: float
    partition_probability: float
    interval: tuple[float, float]
    subnormalized_state: np.ndarray
    posterior: np.ndarray


def select_outcome(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    seed: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> OutcomeSelection:
    """Select one coarse outcome for a supplied seed; no randomness is derived."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    probabilities = born_probabilities(declared, density, tolerance=tol)
    partition = build_seed_partition(probabilities, tolerance=tol)
    index = select_partition_cell(partition, seed)
    operation = _outcome_output(declared[index], density)
    raw_probability = probabilities[index]
    partition_probability = partition.cell_probabilities[index]
    if raw_probability <= 0.0 or partition_probability <= 0.0:
        raise ArithmeticError("zero-probability outcome has no posterior")
    posterior = operation / raw_probability
    if abs(float(np.trace(posterior).real) - 1.0) > 20.0 * tol:
        raise ArithmeticError("selected posterior failed normalization")
    return OutcomeSelection(
        outcome_index=index,
        label=declared[index].label,
        seed=float(seed),
        raw_born_probability=raw_probability,
        partition_probability=partition_probability,
        interval=partition.intervals[index],
        subnormalized_state=operation,
        posterior=posterior,
    )


def apply_nonselective_instrument(
    outcomes: Sequence[CoarseOutcomeOperation],
    state: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Apply all coarse operations exactly once while discarding the label."""

    tol = _positive_tolerance(tolerance)
    density = _validated_density(state, tolerance=tol)
    declared = _validated_outcomes(outcomes, density.shape[0])
    if instrument_completeness_residual(declared, density.shape[0]) > 10.0 * tol:
        raise ValueError("Kraus family must be complete")
    return sum(
        (_outcome_output(outcome, density) for outcome in declared),
        np.zeros_like(density),
    )


def equal_copy_internal_refinement(
    outcome: CoarseOutcomeOperation,
    multiplicity: int,
) -> CoarseOutcomeOperation:
    """Replace every internal ``K`` by ``k`` copies ``K/sqrt(k)``."""

    count = _positive_integer(multiplicity, "multiplicity")
    if not outcome.operators:
        raise ValueError("outcome must contain at least one Kraus operator")
    refined = tuple(
        np.asarray(operator, dtype=np.complex128) / math.sqrt(count)
        for operator in outcome.operators
        for _ in range(count)
    )
    return CoarseOutcomeOperation(
        label=outcome.label,
        operators=refined,
        energy_transfer=outcome.energy_transfer,
    )


def _two_channel_emission_collision(left_probability: float) -> np.ndarray:
    left_amplitude = math.sqrt(left_probability)
    right_amplitude = math.sqrt(1.0 - left_probability)
    collision = np.eye(6, dtype=np.complex128)
    energy_two_sector = (1, 2, 3)
    collision[np.ix_(energy_two_sector, energy_two_sector)] = np.array(
        [
            [right_amplitude, 0.0, left_amplitude],
            [-left_amplitude, 0.0, right_amplitude],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return collision


def _partial_trace_a(joint_state: np.ndarray, dimension_a: int, dimension_b: int) -> np.ndarray:
    tensor = joint_state.reshape(dimension_a, dimension_b, dimension_a, dimension_b)
    return np.trace(tensor, axis1=0, axis2=2)


def _joint_outcome_output(
    outcome: CoarseOutcomeOperation,
    joint_state: np.ndarray,
    remote_dimension: int,
) -> np.ndarray:
    local_identity = np.eye(remote_dimension, dtype=np.complex128)
    return sum(
        (
            np.kron(np.asarray(operator, dtype=np.complex128), local_identity)
            @ joint_state
            @ np.kron(np.asarray(operator, dtype=np.complex128), local_identity).conj().T
            for operator in outcome.operators
        ),
        np.zeros_like(joint_state),
    )


@dataclass(frozen=True)
class LocalBornSelectionCertificate:
    outcome_labels: tuple[str, ...]
    raw_born_probabilities: tuple[float, ...]
    partition_probabilities: tuple[float, ...]
    seed_intervals: tuple[tuple[float, float], ...]
    probe_seeds: tuple[float, ...]
    probe_labels: tuple[str, ...]
    probability_normalization_residual: float
    maximum_interval_probability_residual: float
    maximum_posterior_trace_residual: float
    seed_average_channel_residual: float
    completeness_residual: float
    refinement_operation_residual: float
    refinement_probability_residual: float
    refinement_posterior_residual: float
    refinement_interval_residual: float
    refinement_same_seed_label_mismatches: int
    energy_scale: float
    collision_operator_energy_ledger_residual: float
    maximum_supported_branch_relative_energy_residual: float
    maximum_supported_branch_dimensionless_energy_variance: float
    remote_nonselective_marginal_residual: float
    forced_seed_remote_trace_distance: float
    fixed_seed_born_frequency_error: float
    x_measurement_best_scalar_receipts: tuple[float, float]
    x_measurement_relative_frobenius_receipt_residual: float
    x_measurement_relative_operator_receipt_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *,
    left_probability: float = 0.4,
    energy_scale: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> LocalBornSelectionCertificate:
    """Build the E27 finite selection, energy, and signalling certificate."""

    tol = _positive_tolerance(tolerance)
    if not math.isfinite(left_probability) or not 0.0 < left_probability < 1.0:
        raise ValueError("left_probability must be finite and lie in (0, 1)")
    if not math.isfinite(energy_scale) or energy_scale <= 0.0:
        raise ValueError("energy_scale must be finite and positive")

    system_hamiltonian = energy_scale * np.diag([1.0, 2.0]).astype(np.complex128)
    ancilla_hamiltonian = energy_scale * np.diag([0.0, 1.0, 1.0]).astype(
        np.complex128
    )
    collision = construct_energy_conserving_collision_instrument(
        system_hamiltonian,
        ancilla_hamiltonian,
        _two_channel_emission_collision(left_probability),
        outcome_targets=("silent", "left", "right"),
        outcome_labels=("silent", "left", "right"),
        tolerance=tol,
    )
    outcomes = tuple(
        CoarseOutcomeOperation(
            label=branch.target,
            operators=(np.asarray(branch.operator, dtype=np.complex128),),
            energy_transfer=float(branch.energy_transfer),
        )
        for branch in collision.instrument.branches
    )
    initial_state = np.diag([0.0, 1.0]).astype(np.complex128)
    probabilities = born_probabilities(outcomes, initial_state, tolerance=tol)
    partition = build_seed_partition(probabilities, tolerance=tol)
    left_boundary = partition.intervals[1][1]
    probe_seeds = (
        0.0,
        math.nextafter(left_boundary, 0.0),
        left_boundary,
        math.nextafter(1.0, 0.0),
    )
    selections = tuple(
        select_outcome(outcomes, initial_state, seed, tolerance=tol)
        for seed in probe_seeds
    )

    interval_residual = max(
        abs((end - start) - probability)
        for (start, end), probability in zip(
            partition.intervals, partition.cell_probabilities
        )
    )
    posterior_trace_residuals: list[float] = []
    seed_average = np.zeros_like(initial_state)
    for outcome, raw_probability, partition_probability in zip(
        outcomes, probabilities, partition.cell_probabilities
    ):
        if raw_probability <= 0.0:
            continue
        operation = _outcome_output(outcome, initial_state)
        posterior = operation / raw_probability
        posterior_trace_residuals.append(abs(float(np.trace(posterior).real) - 1.0))
        seed_average += partition_probability * posterior
    nonselective = apply_nonselective_instrument(outcomes, initial_state, tolerance=tol)
    seed_average_channel_residual = float(np.linalg.norm(seed_average - nonselective, ord=2))
    completeness_residual = instrument_completeness_residual(outcomes, 2)

    refined_outcomes = list(outcomes)
    refined_outcomes[1] = equal_copy_internal_refinement(refined_outcomes[1], 7)
    refined_tuple = tuple(refined_outcomes)
    refined_probabilities = born_probabilities(refined_tuple, initial_state, tolerance=tol)
    refined_partition = build_seed_partition(refined_probabilities, tolerance=tol)
    base_left_output = _outcome_output(outcomes[1], initial_state)
    refined_left_output = _outcome_output(refined_tuple[1], initial_state)
    refinement_operation_residual = float(
        np.linalg.norm(refined_left_output - base_left_output, ord=2)
    )
    refinement_probability_residual = max(
        abs(left - right)
        for left, right in zip(
            partition.cell_probabilities, refined_partition.cell_probabilities
        )
    )
    refinement_posterior_residual = float(
        np.linalg.norm(
            refined_left_output / refined_probabilities[1]
            - base_left_output / probabilities[1],
            ord=2,
        )
    )
    refinement_interval_residual = max(
        abs(left_endpoint - right_endpoint)
        for left_interval, right_interval in zip(
            partition.intervals, refined_partition.intervals
        )
        for left_endpoint, right_endpoint in zip(left_interval, right_interval)
    )
    refinement_probe_seeds = tuple(
        0.5 * (start + end)
        for (start, end), probability in zip(
            partition.intervals, partition.cell_probabilities
        )
        if probability > 0.0
    )
    refinement_same_seed_label_mismatches = sum(
        select_outcome(outcomes, initial_state, seed, tolerance=tol).label
        != select_outcome(refined_tuple, initial_state, seed, tolerance=tol).label
        for seed in refinement_probe_seeds
    )

    initial_energy = float(np.trace(system_hamiltonian @ initial_state).real)
    supported_branch_relative_energy_residuals: list[float] = []
    supported_branch_dimensionless_variances: list[float] = []
    for outcome, raw_probability in zip(outcomes, probabilities):
        if raw_probability <= 0.0:
            continue
        posterior = _outcome_output(outcome, initial_state) / raw_probability
        system_energy = float(np.trace(system_hamiltonian @ posterior).real)
        system_energy_squared = float(
            np.trace(system_hamiltonian @ system_hamiltonian @ posterior).real
        )
        supported_branch_dimensionless_variances.append(
            max(0.0, system_energy_squared - system_energy * system_energy)
            / (energy_scale * energy_scale)
        )
        if outcome.energy_transfer is None:
            raise ArithmeticError("collision outcome must carry an energy receipt")
        supported_branch_relative_energy_residuals.append(
            abs(system_energy + outcome.energy_transfer - initial_energy) / energy_scale
        )

    projector_zero = np.diag([1.0, 0.0]).astype(np.complex128)
    projector_one = np.diag([0.0, 1.0]).astype(np.complex128)
    local_outcomes = (
        CoarseOutcomeOperation("zero", (projector_zero,)),
        CoarseOutcomeOperation("one", (projector_one,)),
    )
    bell_vector = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    bell_state = np.outer(bell_vector, bell_vector.conj())
    remote_before = _partial_trace_a(bell_state, 2, 2)
    joint_outputs = tuple(
        _joint_outcome_output(outcome, bell_state, 2) for outcome in local_outcomes
    )
    remote_after = _partial_trace_a(sum(joint_outputs, np.zeros_like(bell_state)), 2, 2)
    remote_nonselective_marginal_residual = float(
        np.linalg.norm(remote_after - remote_before, ord=2)
    )
    joint_probabilities = tuple(float(np.trace(output).real) for output in joint_outputs)
    remote_conditionals = tuple(
        _partial_trace_a(output / probability, 2, 2)
        for output, probability in zip(joint_outputs, joint_probabilities)
    )
    forced_seed_remote_trace_distance = 0.5 * float(
        np.linalg.norm(remote_conditionals[0] - remote_conditionals[1], ord="nuc")
    )
    bell_partition = build_seed_partition(
        joint_probabilities,
        tolerance=tol,
    )
    fixed_seed = 0.25
    repeated_labels = tuple(
        select_partition_cell(bell_partition, fixed_seed) for _ in range(32)
    )
    empirical = tuple(repeated_labels.count(index) / len(repeated_labels) for index in range(2))
    fixed_seed_born_frequency_error = max(
        abs(observed - expected)
        for observed, expected in zip(empirical, bell_partition.cell_probabilities)
    )

    energy_unit = energy_scale
    incompatible_hamiltonian = np.diag([0.0, energy_unit]).astype(np.complex128)
    plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    minus = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)
    measured_energy = plus @ incompatible_hamiltonian @ plus + minus @ incompatible_hamiltonian @ minus
    target_receipt = incompatible_hamiltonian - measured_energy
    design = np.column_stack((plus.reshape(-1), minus.reshape(-1)))
    best_receipts, _, _, _ = np.linalg.lstsq(design, target_receipt.reshape(-1), rcond=None)
    best_operator = (
        measured_energy + best_receipts[0] * plus + best_receipts[1] * minus
    )
    receipt_residual = best_operator - incompatible_hamiltonian
    relative_frobenius_receipt_residual = float(
        np.linalg.norm(receipt_residual, ord="fro")
        / np.linalg.norm(incompatible_hamiltonian, ord="fro")
    )
    relative_operator_receipt_residual = float(
        np.linalg.norm(receipt_residual, ord=2)
        / np.linalg.norm(incompatible_hamiltonian, ord=2)
    )

    numerical_limit = 20.0 * tol
    inverse_cdf_certified = max(
        partition.input_normalization_residual,
        interval_residual,
        max(posterior_trace_residuals, default=0.0),
        seed_average_channel_residual,
        completeness_residual,
    ) <= numerical_limit
    refinement_certified = max(
        refinement_operation_residual,
        refinement_probability_residual,
        refinement_posterior_residual,
        refinement_interval_residual,
    ) <= numerical_limit and refinement_same_seed_label_mismatches == 0
    collision_energy_certified = max(
        collision.relative_ledger_identity_residual,
        max(supported_branch_relative_energy_residuals, default=0.0),
        max(supported_branch_dimensionless_variances, default=0.0),
    ) <= numerical_limit

    return LocalBornSelectionCertificate(
        outcome_labels=tuple(outcome.label for outcome in outcomes),
        raw_born_probabilities=probabilities,
        partition_probabilities=partition.cell_probabilities,
        seed_intervals=partition.intervals,
        probe_seeds=probe_seeds,
        probe_labels=tuple(selection.label for selection in selections),
        probability_normalization_residual=partition.input_normalization_residual,
        maximum_interval_probability_residual=interval_residual,
        maximum_posterior_trace_residual=max(posterior_trace_residuals, default=0.0),
        seed_average_channel_residual=seed_average_channel_residual,
        completeness_residual=completeness_residual,
        refinement_operation_residual=refinement_operation_residual,
        refinement_probability_residual=refinement_probability_residual,
        refinement_posterior_residual=refinement_posterior_residual,
        refinement_interval_residual=refinement_interval_residual,
        refinement_same_seed_label_mismatches=refinement_same_seed_label_mismatches,
        energy_scale=energy_scale,
        collision_operator_energy_ledger_residual=collision.relative_ledger_identity_residual,
        maximum_supported_branch_relative_energy_residual=max(
            supported_branch_relative_energy_residuals, default=0.0
        ),
        maximum_supported_branch_dimensionless_energy_variance=max(
            supported_branch_dimensionless_variances, default=0.0
        ),
        remote_nonselective_marginal_residual=remote_nonselective_marginal_residual,
        forced_seed_remote_trace_distance=forced_seed_remote_trace_distance,
        fixed_seed_born_frequency_error=fixed_seed_born_frequency_error,
        x_measurement_best_scalar_receipts=tuple(
            float(value.real) for value in best_receipts
        ),
        x_measurement_relative_frobenius_receipt_residual=(
            relative_frobenius_receipt_residual
        ),
        x_measurement_relative_operator_receipt_residual=(
            relative_operator_receipt_residual
        ),
        dimensions={
            "seed_dimensionless": True,
            "born_probabilities_dimensionless": True,
            "cumulative_intervals_dimensionless": True,
            "kraus_and_density_entries_dimensionless": True,
            "hamiltonian_and_receipt_share_energy_dimension": True,
            "branch_energy_residual_divided_by_energy_scale": True,
            "branch_energy_variance_divided_by_energy_scale_squared": True,
            "seed_or_label_does_not_supply_energy_scale": True,
        },
        accounting={
            "probabilities_partition_seed_measure_once": True,
            "weighted_posteriors_equal_nonselective_channel_once": (
                seed_average_channel_residual <= numerical_limit
            ),
            "all_zero_probability_outcomes_not_conditioned": all(
                start == end
                for (start, end), probability in zip(
                    partition.intervals, partition.cell_probabilities
                )
                if probability == 0.0
            ),
            "unselected_probabilities_not_added_as_energy": True,
            "selected_record_energy_receipt_counted_once": collision_energy_certified,
            "seed_carries_energy": False,
        },
        boundaries={
            "uniform_independent_uncontrollable_seed_is_explicit_axiom": True,
            "unitary_or_stinespring_does_not_derive_seed": True,
            "forced_seed_is_prohibited_external_intervention": True,
            "half_open_intervals_use_declared_coarse_outcomes_only": True,
            "internal_kraus_labels_do_not_enter_seed_partition": True,
            "finite_refinement_probe_set_excludes_boundary_neighborhoods": True,
            "same_seed_refinement_claim_limited_to_declared_probe_set": True,
            "outcome_order_is_declared_input": True,
            "physical_seed_independence_from_settings_derived": False,
            "finite_bipartite_witness_is_not_relativistic_qft": True,
            "collision_hamiltonians_and_pointer_basis_are_supplied": True,
            "supported_collision_branches_are_sharp_energy_only": True,
        },
        alternatives={
            "operational_uniform_seed_sampler_route_open": True,
            "microscopic_local_uncontrollable_seed_route_open": True,
            "durable_local_pointer_route_open": True,
            "covariant_selection_and_geometry_route_open": True,
            "deterministic_hidden_variable_route_requires_bell_audit": True,
        },
        status={
            "inverse_cdf_born_partition_certified": inverse_cdf_certified,
            "valid_probe_seed_returns_one_coarse_label": len(selections) == len(probe_seeds),
            "uniform_seed_average_recovers_nonselective_channel": (
                seed_average_channel_residual <= numerical_limit
            ),
            "explicit_collision_instrument_cptp": completeness_residual <= numerical_limit,
            "coarse_selection_internal_refinement_invariant": refinement_certified,
            "supplied_collision_operator_energy_ledger_certified": (
                collision.relative_ledger_identity_residual <= numerical_limit
            ),
            "sharp_supported_branch_energy_receipts_certified": collision_energy_certified,
            "single_local_nonselective_marginal_witness": (
                remote_nonselective_marginal_residual <= numerical_limit
            ),
            "fixed_seed_born_frequency_counterexample": (
                fixed_seed_born_frequency_error > 0.49
            ),
            "controllable_seed_signalling_counterexample": (
                forced_seed_remote_trace_distance > 0.99
            ),
            "general_scalar_energy_receipt_counterexample": (
                relative_frobenius_receipt_residual > 0.7
                and relative_operator_receipt_residual > 0.49
            ),
            "physical_uniform_seed_law_derived": False,
            "objective_single_outcome_selection_derived": False,
            "durable_physical_pointer_derived": False,
            "relativistic_no_signalling_derived": False,
            "general_measurement_energy_conservation_derived": False,
            "spacetime_metric_curvature_or_gravity_derived": False,
            "independent_holdout_complete": False,
            "success_gates_1_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-probability", type=float, default=0.4)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(
                certificate(
                    left_probability=args.left_probability,
                    energy_scale=args.energy_scale,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
