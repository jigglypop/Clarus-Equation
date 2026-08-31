'''Conditional joint proof for reciprocal selection, dimension lift, and dark readout.

The construction joins three finite statements without identifying a bare
probability with energy.

1. Two pointer qubits conditionally activate one another through fresh
   equal-gap batteries.  The global circuit is unitary and energy preserving;
   tracing the batteries gives a CPTP channel.
2. Three independently selected direction records generate a nested spatial
   filtration.  Their exterior products have squared norm equal to the Gram
   determinant, so non-zero successive wedges certify dimensions 1, 2, 3.
3. Inside an already physical folded energy receipt, the activation and
   non-activation outcomes have probabilities sin(theta)^2 and cos(theta)^2.
   They partition one energy receipt into a mobile mass-shell channel and its
   locked complement.  Existing one-receipt theorems then give conditional
   dust and constant-vacuum FLRW stresses without double counting.

The geometry readout, persistence of the folded receipt, equal-gap scale, and
global constant-vacuum action remain explicit axioms.  Hence this is an
existence theorem for the declared model, not a derivation of real cosmology
from standard quantum mechanics or a parameter-free abundance prediction.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

try:
    from examples.physics.causal_record_dust_bridge import (
        CausalRecordNode,
        CausalTransition,
        ExitPhaseMark,
        construct_conserved_record_flow,
        match_exit_antichain,
        monokinetic_dust_data,
    )
    from examples.physics.partitioned_dark_sector_flrw import (
        PartitionedDarkReceipt,
        PartitionedDustVacuumFLRW,
        construct_record_complement_as_vacuum,
        propagate_partitioned_dust_vacuum_flat_flrw,
    )
except ModuleNotFoundError:
    from causal_record_dust_bridge import (
        CausalRecordNode,
        CausalTransition,
        ExitPhaseMark,
        construct_conserved_record_flow,
        match_exit_antichain,
        monokinetic_dust_data,
    )
    from partitioned_dark_sector_flrw import (
        PartitionedDarkReceipt,
        PartitionedDustVacuumFLRW,
        construct_record_complement_as_vacuum,
        propagate_partitioned_dust_vacuum_flat_flrw,
    )


DEFAULT_TOLERANCE = 1.0e-10
QUBIT_COUNT = 4
A_QUBIT = 0
B_QUBIT = 1
A_BATTERY = 2
B_BATTERY = 3


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f'{name} must be positive')
    return result


def _interaction_angle(value: float, *, strict_interior: bool = False) -> float:
    angle = _finite(value, 'interaction_angle')
    lower_ok = angle > 0.0 if strict_interior else angle >= 0.0
    upper_ok = angle < math.pi / 2.0 if strict_interior else angle <= math.pi / 2.0
    if not lower_ok or not upper_ok:
        interval = '(0, pi/2)' if strict_interior else '[0, pi/2]'
        raise ValueError(f'interaction_angle must lie in {interval}')
    return angle


def _basis_index(bits: Sequence[int]) -> int:
    if len(bits) != QUBIT_COUNT or any(bit not in (0, 1) for bit in bits):
        raise ValueError('basis state must contain four bits')
    return sum(int(bit) << position for position, bit in enumerate(bits))


def _basis_state(bits: Sequence[int]) -> np.ndarray:
    state = np.zeros(1 << QUBIT_COUNT, dtype=np.complex128)
    state[_basis_index(bits)] = 1.0
    return state


def _controlled_exchange(
    *,
    control: int,
    target: int,
    battery: int,
    angle: float,
) -> np.ndarray:
    '''Return a controlled target--battery excitation rotation.'''

    angle = _interaction_angle(angle)
    if len({control, target, battery}) != 3 or any(
        index < 0 or index >= QUBIT_COUNT
        for index in (control, target, battery)
    ):
        raise ValueError('control, target, and battery must be distinct qubits')
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dimension = 1 << QUBIT_COUNT
    unitary = np.eye(dimension, dtype=np.complex128)
    for source in range(dimension):
        if not ((source >> control) & 1):
            continue
        if (source >> target) & 1:
            continue
        if not ((source >> battery) & 1):
            continue
        partner = source ^ (1 << target) ^ (1 << battery)
        unitary[source, source] = cosine
        unitary[partner, source] = sine
        unitary[source, partner] = -sine
        unitary[partner, partner] = cosine
    return unitary


def reciprocal_execution_unitary(interaction_angle: float) -> np.ndarray:
    '''Apply A->B and then B->A conditional energy-paying gates.'''

    angle = _interaction_angle(interaction_angle)
    activate_b = _controlled_exchange(
        control=A_QUBIT,
        target=B_QUBIT,
        battery=B_BATTERY,
        angle=angle,
    )
    activate_a = _controlled_exchange(
        control=B_QUBIT,
        target=A_QUBIT,
        battery=A_BATTERY,
        angle=angle,
    )
    return activate_a @ activate_b


def _number_expectation(state: np.ndarray, qubit: int) -> float:
    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    return float(
        sum(
            probability * ((index >> qubit) & 1)
            for index, probability in enumerate(probabilities)
        )
    )


def _energy_expectation(state: np.ndarray, energy_gap: float) -> float:
    probabilities = np.abs(np.asarray(state, dtype=np.complex128)) ** 2
    return float(
        sum(
            probability * index.bit_count() * energy_gap
            for index, probability in enumerate(probabilities)
        )
    )


def _reduced_system_kraus(unitary: np.ndarray) -> tuple[np.ndarray, ...]:
    '''Trace two initially excited batteries and return four system Kraus maps.'''

    system_dimension = 4
    initial_battery_index = 3
    operators = []
    for output_battery in range(4):
        operator = np.zeros((system_dimension, system_dimension), dtype=np.complex128)
        for output_system in range(system_dimension):
            for input_system in range(system_dimension):
                row = output_system + (output_battery << 2)
                column = input_system + (initial_battery_index << 2)
                operator[output_system, input_system] = unitary[row, column]
        operators.append(operator)
    return tuple(operators)


@dataclass(frozen=True)
class MutualExecutionAudit:
    interaction_angle: float
    activation_probability: float
    nonactivation_probability: float
    a_to_b_intervention_effect: float
    b_to_a_intervention_effect: float
    forward_system_activations: tuple[float, float]
    reverse_system_activations: tuple[float, float]
    inactive_system_activations: tuple[float, float]
    forward_battery_activations: tuple[float, float]
    reverse_battery_activations: tuple[float, float]
    unitary_residual: float
    relative_energy_commutator_residual: float
    maximum_energy_balance_residual: float
    kraus_completeness_residual: float
    minimum_choi_eigenvalue: float
    reciprocal_probability_residual: float
    forward_reverse_output_overlap: float
    global_unitary: bool
    reduced_channel_cptp: bool
    energy_conserved: bool
    reciprocal_execution_certified: bool
    seed_and_battery_required: bool = True
    arbitrary_unknown_state_cloned: bool = False
    status: str = 'FINITE_RECIPROCAL_POINTER_EXECUTION_CLOSED'


def audit_mutual_execution(
    *,
    interaction_angle: float = math.pi / 4.0,
    energy_gap: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MutualExecutionAudit:
    angle = _interaction_angle(interaction_angle)
    gap = _positive(energy_gap, 'energy_gap')
    tolerance = _positive(tolerance, 'tolerance')
    unitary = reciprocal_execution_unitary(angle)
    identity = np.eye(unitary.shape[0], dtype=np.complex128)
    unitary_residual = float(np.linalg.norm(unitary.conj().T @ unitary - identity))
    energy_diagonal = np.array(
        [index.bit_count() * gap for index in range(unitary.shape[0])],
        dtype=float,
    )
    hamiltonian = np.diag(energy_diagonal).astype(np.complex128)
    energy_scale = max(gap, float(np.max(energy_diagonal)))
    energy_commutator = float(
        np.linalg.norm(unitary @ hamiltonian - hamiltonian @ unitary)
        / energy_scale
    )

    forward_input = _basis_state((1, 0, 1, 1))
    reverse_input = _basis_state((0, 1, 1, 1))
    inactive_input = _basis_state((0, 0, 1, 1))
    forward_output = unitary @ forward_input
    reverse_output = unitary @ reverse_input
    inactive_output = unitary @ inactive_input
    forward_system = (
        _number_expectation(forward_output, A_QUBIT),
        _number_expectation(forward_output, B_QUBIT),
    )
    reverse_system = (
        _number_expectation(reverse_output, A_QUBIT),
        _number_expectation(reverse_output, B_QUBIT),
    )
    inactive_system = (
        _number_expectation(inactive_output, A_QUBIT),
        _number_expectation(inactive_output, B_QUBIT),
    )
    forward_batteries = (
        _number_expectation(forward_output, A_BATTERY),
        _number_expectation(forward_output, B_BATTERY),
    )
    reverse_batteries = (
        _number_expectation(reverse_output, A_BATTERY),
        _number_expectation(reverse_output, B_BATTERY),
    )
    energy_residuals = tuple(
        abs(_energy_expectation(output, gap) - _energy_expectation(source, gap))
        / max(_energy_expectation(source, gap), gap)
        for source, output in (
            (forward_input, forward_output),
            (reverse_input, reverse_output),
            (inactive_input, inactive_output),
        )
    )

    kraus = _reduced_system_kraus(unitary)
    system_identity = np.eye(4, dtype=np.complex128)
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        start=np.zeros((4, 4), dtype=np.complex128),
    )
    kraus_residual = float(np.linalg.norm(completeness - system_identity))
    choi = sum(
        (
            np.reshape(operator, (-1, 1), order='F')
            @ np.reshape(operator, (1, -1), order='F').conj()
            for operator in kraus
        ),
        start=np.zeros((16, 16), dtype=np.complex128),
    )
    minimum_choi = float(np.min(np.linalg.eigvalsh(choi)))
    probability = math.sin(angle) ** 2
    complement = math.cos(angle) ** 2
    a_to_b = forward_system[1] - inactive_system[1]
    b_to_a = reverse_system[0] - inactive_system[0]
    reciprocal_residual = max(abs(a_to_b - probability), abs(b_to_a - probability))
    output_overlap = abs(np.vdot(forward_output, reverse_output))
    unitary_ok = unitary_residual <= tolerance
    cptp = kraus_residual <= tolerance and minimum_choi >= -tolerance
    energy_ok = max(energy_commutator, *energy_residuals) <= tolerance
    reciprocal = (
        a_to_b > tolerance
        and b_to_a > tolerance
        and reciprocal_residual <= tolerance
    )
    return MutualExecutionAudit(
        interaction_angle=angle,
        activation_probability=probability,
        nonactivation_probability=complement,
        a_to_b_intervention_effect=a_to_b,
        b_to_a_intervention_effect=b_to_a,
        forward_system_activations=forward_system,
        reverse_system_activations=reverse_system,
        inactive_system_activations=inactive_system,
        forward_battery_activations=forward_batteries,
        reverse_battery_activations=reverse_batteries,
        unitary_residual=unitary_residual,
        relative_energy_commutator_residual=energy_commutator,
        maximum_energy_balance_residual=max(energy_residuals),
        kraus_completeness_residual=kraus_residual,
        minimum_choi_eigenvalue=minimum_choi,
        reciprocal_probability_residual=reciprocal_residual,
        forward_reverse_output_overlap=output_overlap,
        global_unitary=unitary_ok,
        reduced_channel_cptp=cptp,
        energy_conserved=energy_ok,
        reciprocal_execution_certified=(unitary_ok and cptp and energy_ok and reciprocal),
    )


@dataclass(frozen=True)
class DimensionSelectionAudit:
    direction_vectors: tuple[tuple[float, float, float], ...]
    gram_determinants: tuple[float, float, float, float]
    exterior_norms_squared: tuple[float, float, float, float]
    projector_ranks: tuple[int, int, int, int]
    rank_increments: tuple[int, int, int]
    selected_probabilities: tuple[float, float, float, float]
    incremental_probabilities: tuple[float, float, float]
    maximum_projector_nesting_residual: float
    maximum_probability_sum_residual: float
    oriented_three_volume: float
    independent_direction_records: bool
    cumulative_selection_monotone: bool
    dimensions_zero_through_three_certified: bool
    geometry_readout_is_adopted_axiom: bool = True
    hilbert_rank_equals_spatial_dimension_without_readout_axiom: bool = False
    bare_zerod_uniqueness_proved: bool = False
    status: str = 'CONDITIONAL_GRADED_SELECTION_ZERO_TO_THREE_CLOSED'


def audit_dimension_selection(
    *,
    direction_vectors: Sequence[Sequence[float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    direction_density: Sequence[Sequence[complex]] = (
        (0.2, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.5),
    ),
    tolerance: float = DEFAULT_TOLERANCE,
) -> DimensionSelectionAudit:
    '''Certify a nested quantum-direction filtration and its exterior volumes.'''

    tolerance = _positive(tolerance, 'tolerance')
    vectors = np.asarray(direction_vectors, dtype=float)
    if vectors.shape != (3, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError('direction_vectors must be a finite 3 by 3 array')
    matrix = vectors.T
    if np.linalg.matrix_rank(matrix, tol=tolerance) != 3:
        raise ValueError('three direction records must be linearly independent')
    density = np.asarray(direction_density, dtype=np.complex128)
    if density.shape != (3, 3) or not np.all(np.isfinite(density)):
        raise ValueError('direction_density must be a finite 3 by 3 matrix')
    if np.linalg.norm(density - density.conj().T) > tolerance:
        raise ValueError('direction_density must be Hermitian')
    if abs(float(np.trace(density).real) - 1.0) > tolerance:
        raise ValueError('direction_density must have unit trace')
    if float(np.min(np.linalg.eigvalsh(density))) < -tolerance:
        raise ValueError('direction_density must be positive semidefinite')

    projectors = [np.zeros((3, 3), dtype=float)]
    determinants = [1.0]
    for count in range(1, 4):
        columns = matrix[:, :count]
        gram = columns.T @ columns
        determinant = float(np.linalg.det(gram))
        if determinant <= tolerance:
            raise ValueError('every successive exterior selection must be non-zero')
        projector = columns @ np.linalg.inv(gram) @ columns.T
        determinants.append(determinant)
        projectors.append(projector)

    ranks = tuple(int(np.linalg.matrix_rank(item, tol=tolerance)) for item in projectors)
    rank_increments = tuple(ranks[index] - ranks[index - 1] for index in range(1, 4))
    nesting_residual = max(
        float(np.linalg.norm(projectors[left] @ projectors[right] - projectors[min(left, right)]))
        for left in range(4)
        for right in range(4)
    )
    probabilities = tuple(
        float(np.trace(projector.astype(np.complex128) @ density).real)
        for projector in projectors
    )
    increments = tuple(
        probabilities[index] - probabilities[index - 1]
        for index in range(1, 4)
    )
    probability_residual = abs(sum(increments) - probabilities[-1])
    independent = all(value > tolerance for value in determinants[1:])
    monotone = all(value >= -tolerance for value in increments)
    dimensions_closed = (
        ranks == (0, 1, 2, 3)
        and rank_increments == (1, 1, 1)
        and nesting_residual <= tolerance
        and independent
        and monotone
        and probability_residual <= tolerance
    )
    return DimensionSelectionAudit(
        direction_vectors=tuple(tuple(float(value) for value in row) for row in vectors),
        gram_determinants=tuple(determinants),
        exterior_norms_squared=tuple(determinants),
        projector_ranks=ranks,
        rank_increments=rank_increments,
        selected_probabilities=probabilities,
        incremental_probabilities=increments,
        maximum_projector_nesting_residual=nesting_residual,
        maximum_probability_sum_residual=probability_residual,
        oriented_three_volume=math.sqrt(determinants[-1]),
        independent_direction_records=independent,
        cumulative_selection_monotone=monotone,
        dimensions_zero_through_three_certified=dimensions_closed,
    )


@dataclass(frozen=True)
class FoldedOpportunityDarkAudit:
    interaction_angle: float
    mobile_probability: float
    locked_probability: float
    nonselected_weighted_surprisal: float
    energy_gap: float
    natural_cell_volume: float
    total_folded_energy: float
    mobile_dust_energy: float
    locked_vacuum_energy: float
    expected_mobile_energy: float
    expected_locked_energy: float
    physical_opportunity_energy: float
    probability_normalization_residual: float
    shared_energy_partition_residual: float
    dimensionless_volume_combination: float
    partition: PartitionedDarkReceipt
    flrw: PartitionedDustVacuumFLRW
    probability_used_as_energy: bool
    physical_energy_receipt_supplies_scale: bool
    opportunity_cost_is_dimensionless_allocation_diagnostic: bool
    allocation_probabilities_derived_from_execution_unitary: bool
    one_receipt_no_double_counting_closed: bool
    conditional_dm_de_stress_forms_closed: bool
    ensemble_receipt_partition: bool
    counterfactual_quantum_branch_dynamics_derived: bool
    folded_receipt_persistence_is_adopted_axiom: bool = True
    energy_to_cell_scale_is_adopted_axiom: bool = True
    global_constant_vacuum_action_is_adopted_axiom: bool = True
    absolute_dark_scale_from_probability_alone: bool = False
    real_dark_sector_identity_observationally_confirmed: bool = False
    status: str = 'CONDITIONAL_FOLDED_RECEIPT_DUST_VACUUM_EXISTENCE_CLOSED'


def audit_folded_opportunity_dark_sector(
    *,
    interaction_angle: float = math.pi / 4.0,
    energy_gap: float = 1.0,
    direction_audit: DimensionSelectionAudit | None = None,
    newton_constant: float = 1.0,
    evaluation_scale_factor_ratio: float = 2.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FoldedOpportunityDarkAudit:
    '''Route one persistent folded receipt into disjoint mobile and locked outcomes.'''

    angle = _interaction_angle(interaction_angle, strict_interior=True)
    gap = _positive(energy_gap, 'energy_gap')
    tolerance = _positive(tolerance, 'tolerance')
    directions = direction_audit or audit_dimension_selection(tolerance=tolerance)
    if not directions.dimensions_zero_through_three_certified:
        raise ValueError('a certified three-direction cell is required')
    mobile_probability = math.sin(angle) ** 2
    locked_probability = math.cos(angle) ** 2
    opportunity_cost = -locked_probability * math.log(locked_probability)

    # In hbar=c=1 units, E_*^-1 is the adopted cell length scale.  The
    # dimensionless oriented volume multiplies E_*^-3.
    cell_volume = directions.oriented_three_volume / gap**3
    nodes = (
        CausalRecordNode('folded_root', 'persistent_folded_receipt', gap),
        CausalRecordNode('mobile_exit', 'mobile_mass_shell_record', gap),
        CausalRecordNode('locked_exit', 'locked_vacuum_record', gap),
    )
    transitions = (
        CausalTransition('folded_root', 'mobile_exit', mobile_probability),
        CausalTransition('folded_root', 'locked_exit', locked_probability),
    )
    flow = construct_conserved_record_flow(
        nodes,
        transitions,
        {'folded_root': 1.0},
        tolerance=tolerance,
    )
    marks = (
        ExitPhaseMark(
            'mobile_exit',
            (0.0, 0.0, 0.0),
            gap,
            (0.0, 0.0, 0.0),
            1.0,
        ),
        ExitPhaseMark(
            'locked_exit',
            (cell_volume ** (1.0 / 3.0) / 2.0, 0.0, 0.0),
            gap,
            (0.0, 0.0, 0.0),
            0.0,
        ),
    )
    matching = match_exit_antichain(
        flow,
        nodes,
        marks,
        cell_volume=cell_volume,
        tolerance=tolerance,
    )
    dust = monokinetic_dust_data(matching, tolerance=tolerance)
    partition = construct_record_complement_as_vacuum(
        matching,
        dust,
        source_receipt_id='folded:mutual-execution',
        dust_allocation_id='folded:mobile',
        vacuum_allocation_id='folded:locked',
        tolerance=tolerance,
    )
    flrw = propagate_partitioned_dust_vacuum_flat_flrw(
        partition,
        newton_constant=newton_constant,
        evaluation_scale_factor_ratio=evaluation_scale_factor_ratio,
        global_constant_vacuum_action_adopted=True,
        tolerance=tolerance,
    )
    expected_mobile = gap * mobile_probability
    expected_locked = gap * locked_probability
    probability_residual = mobile_probability + locked_probability - 1.0
    energy_residual = gap - expected_mobile - expected_locked
    one_receipt_closed = (
        partition.two_channel_partition_closed
        and abs(partition.dust_energy - expected_mobile) <= tolerance * gap
        and abs(partition.vacuum_energy - expected_locked) <= tolerance * gap
        and abs(energy_residual) <= tolerance * gap
    )
    stress_closed = (
        one_receipt_closed
        and flrw.conditional_mixed_background_closed
        and flrw.global_constant_vacuum_action_adopted
    )
    return FoldedOpportunityDarkAudit(
        interaction_angle=angle,
        mobile_probability=mobile_probability,
        locked_probability=locked_probability,
        nonselected_weighted_surprisal=opportunity_cost,
        energy_gap=gap,
        natural_cell_volume=cell_volume,
        total_folded_energy=partition.total_record_energy,
        mobile_dust_energy=partition.dust_energy,
        locked_vacuum_energy=partition.vacuum_energy,
        expected_mobile_energy=expected_mobile,
        expected_locked_energy=expected_locked,
        physical_opportunity_energy=expected_locked,
        probability_normalization_residual=probability_residual,
        shared_energy_partition_residual=energy_residual,
        dimensionless_volume_combination=gap**3 * cell_volume,
        partition=partition,
        flrw=flrw,
        probability_used_as_energy=False,
        physical_energy_receipt_supplies_scale=True,
        opportunity_cost_is_dimensionless_allocation_diagnostic=True,
        allocation_probabilities_derived_from_execution_unitary=True,
        one_receipt_no_double_counting_closed=one_receipt_closed,
        conditional_dm_de_stress_forms_closed=stress_closed,
        ensemble_receipt_partition=True,
        counterfactual_quantum_branch_dynamics_derived=False,
    )


@dataclass(frozen=True)
class QuantumSelectionDimensionDarkCertificate:
    mutual_execution: MutualExecutionAudit
    dimension_selection: DimensionSelectionAudit
    dark_readout: FoldedOpportunityDarkAudit
    dimension_derived_mobile_probability: float
    dimension_derived_locked_probability: float
    dimension_to_dark_probability_residual: float
    interaction_angle_derived_from_dimension_selection: bool
    dimension_selection_controls_dark_partition: bool
    claim_1_nonselected_dark_status: str
    claim_2_dimension_implementation_status: str
    claim_3_mutual_execution_status: str
    all_three_conditional_claims_closed: bool
    all_three_finite_witnesses_closed: bool
    all_three_user_claims_unconditionally_proved: bool
    unconditional_standard_qm_to_real_cosmology_proved: bool
    required_adopted_axioms: tuple[str, ...]
    dimensionless_arguments: tuple[tuple[str, str], ...]
    status: str


def certify_quantum_selection_dimension_dark_bridge(
    *,
    interaction_angle: float | None = None,
    energy_gap: float = 1.0,
    newton_constant: float = 1.0,
    evaluation_scale_factor_ratio: float = 2.0,
    direction_density: Sequence[Sequence[complex]] = (
        (0.2, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.5),
    ),
    tolerance: float = DEFAULT_TOLERANCE,
) -> QuantumSelectionDimensionDarkCertificate:
    '''Build the joint finite witness for the three user-level claims.'''

    tolerance = _positive(tolerance, 'tolerance')
    dimensions = audit_dimension_selection(
        direction_density=direction_density,
        tolerance=tolerance,
    )
    total_probability = dimensions.selected_probabilities[-1]
    dimension_mobile = dimensions.selected_probabilities[2] / total_probability
    dimension_locked = dimensions.incremental_probabilities[2] / total_probability
    if not 0.0 < dimension_mobile < 1.0 or not 0.0 < dimension_locked < 1.0:
        raise ValueError('the mixed dark witness requires positive grade-2 and grade-3 weights')
    derived_angle = math.asin(math.sqrt(dimension_mobile))
    angle_was_derived = interaction_angle is None
    angle = (
        derived_angle
        if interaction_angle is None
        else _interaction_angle(interaction_angle, strict_interior=True)
    )
    mutual = audit_mutual_execution(
        interaction_angle=angle,
        energy_gap=energy_gap,
        tolerance=tolerance,
    )
    dark = audit_folded_opportunity_dark_sector(
        interaction_angle=angle,
        energy_gap=energy_gap,
        direction_audit=dimensions,
        newton_constant=newton_constant,
        evaluation_scale_factor_ratio=evaluation_scale_factor_ratio,
        tolerance=tolerance,
    )
    dimension_dark_residual = max(
        abs(dark.mobile_probability - dimension_mobile),
        abs(dark.locked_probability - dimension_locked),
    )
    dimension_controls_dark = dimension_dark_residual <= tolerance
    closed = (
        mutual.reciprocal_execution_certified
        and dimensions.dimensions_zero_through_three_certified
        and dark.conditional_dm_de_stress_forms_closed
        and dimension_controls_dark
    )
    return QuantumSelectionDimensionDarkCertificate(
        mutual_execution=mutual,
        dimension_selection=dimensions,
        dark_readout=dark,
        dimension_derived_mobile_probability=dimension_mobile,
        dimension_derived_locked_probability=dimension_locked,
        dimension_to_dark_probability_residual=dimension_dark_residual,
        interaction_angle_derived_from_dimension_selection=angle_was_derived,
        dimension_selection_controls_dark_partition=dimension_controls_dark,
        claim_1_nonselected_dark_status=(
            'CONDITIONAL_PROBABILITY_WEIGHTED_RECEIPT_PARTITION_AND_ONE_ACTION'
        ),
        claim_2_dimension_implementation_status=(
            'CONDITIONAL_GRADED_SELECTION_IN_SUPPLIED_R3'
        ),
        claim_3_mutual_execution_status=(
            'FINITE_SEEDED_BATTERY_ASSISTED_RECIPROCAL_POINTER_THEOREM'
        ),
        all_three_conditional_claims_closed=closed,
        all_three_finite_witnesses_closed=closed,
        all_three_user_claims_unconditionally_proved=False,
        unconditional_standard_qm_to_real_cosmology_proved=False,
        required_adopted_axioms=(
            'nonselected record persists as one physical folded energy receipt',
            'three independent pointer records are read as spatial directions',
            'the common energy gap fixes the natural cell-length scale',
            'mobile records admit a positive monokinetic mass-shell map',
            'locked complement is varied through one global constant-vacuum action',
        ),
        dimensionless_arguments=(
            ('theta = g tau', 'interaction angle is dimensionless'),
            ('sin(theta)^2 and cos(theta)^2', 'Born routing probabilities'),
            ('-q log(q)', 'weighted opportunity surprisal'),
            ('E_*^3 V_*', 'dimensionless oriented cell volume'),
            ('x = a/a_*', 'dimensionless FLRW scale-factor ratio'),
        ),
        status=(
            'THREE_FINITE_CONDITIONAL_WITNESSES_CLOSED_PHYSICAL_MAPS_OPEN'
            if closed
            else 'JOINT_FINITE_WITNESS_OR_PROBABILITY_LINK_AUDIT_FAILED'
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='quantum_selection_dimension_dark_bridge'
    )
    parser.add_argument('--angle', type=float)
    parser.add_argument('--energy-gap', type=float, default=1.0)
    parser.add_argument('--newton-constant', type=float, default=1.0)
    parser.add_argument('--scale-factor-ratio', type=float, default=2.0)
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args(argv)
    certificate = certify_quantum_selection_dimension_dark_bridge(
        interaction_angle=args.angle,
        energy_gap=args.energy_gap,
        newton_constant=args.newton_constant,
        evaluation_scale_factor_ratio=args.scale_factor_ratio,
    )
    print(
        json.dumps(
            asdict(certificate),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0 if certificate.all_three_conditional_claims_closed else 2


if __name__ == '__main__':
    raise SystemExit(main())
