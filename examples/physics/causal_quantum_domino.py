"""Finite-depth causal quantum domino with an explicit energy payer.

The continuous-time nearest-neighbour jump picture does not have a strict
front: a chain of exponential waiting times has a non-zero early-arrival tail
at every positive time.  This module therefore uses a discrete local circuit.

At tick ``j`` the already reached system qubit ``j`` controls a partial swap
between the neighbouring target qubit ``j + 1`` and one fresh battery qubit.
The battery starts excited.  For the orthogonal pointer label ``1`` the gate is

    |1,0,1> -> cos(theta)|1,0,1> + sin(theta)|1,1,0>.

The reverse state is rotated by the same two-dimensional unitary.  The parent
is unchanged, the target excitation is paid for by the battery, and the gate
commutes with the supplied equal-gap excitation Hamiltonian.  ``theta`` and
``sin(theta)**2`` are dimensionless.  Length and time enter the trigonometric
core only through the separately audited ratio ``a / (c * delta_t)``.

Tracing the fresh batteries gives a CPTP channel.  A depth-d nearest-neighbour
circuit has an exact structural influence cone of graph radius d; requiring
``delta_t >= a / c`` bounds its front by c.  This is a finite conditional
witness.  It does not derive a durable apparatus pointer, a record-to-gravity
source map, spacetime, or a cosmological abundance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8
MAX_SYSTEM_QUBITS = 5
MAX_TOTAL_QUBITS = 9


@dataclass(frozen=True)
class BatteryOutcomeReceipt:
    """One energy-resolved outcome of the supplied battery instrument."""

    basis_label: str
    probability: float
    final_battery_energy: float
    energy_paid_to_system: float
    conditional_system_energy: float | None
    relative_branch_energy_residual: float | None


@dataclass(frozen=True)
class CausalQuantumDominoCertificate:
    """Numerical receipt for one declared finite-depth domino circuit."""

    site_count: int
    depth: int
    theta: float
    trigger_probability: float
    lattice_spacing: float
    clock_step: float
    causal_speed: float
    causal_ratio: float
    front_speed_bound: float
    structural_influence_cone: tuple[int, ...]
    spacelike_sites: tuple[int, ...]
    activation_probabilities: tuple[float, ...]
    channel_dimension: int
    kraus_count: int
    unitary_residual: float
    relative_energy_commutator_residual: float
    kraus_completeness_residual: float
    minimum_choi_eigenvalue: float
    output_trace_residual: float
    minimum_output_eigenvalue: float
    born_probability_sum_residual: float
    minimum_born_probability: float
    maximum_sampled_spacelike_trace_distance: float
    kraus_vs_direct_partial_trace_residual: float
    standard_limit_superoperator_residual: float
    energy_gap: float
    initial_system_energy: float
    final_system_energy: float
    initial_battery_energy: float
    final_battery_energy: float
    relative_total_energy_balance_residual: float
    battery_outcomes: tuple[BatteryOutcomeReceipt, ...]
    expected_battery_energy_paid: float
    relative_reverse_transfer_identity_residual: float
    maximum_relative_branch_energy_residual: float
    structural_causal_support_exact: bool
    sampled_spacelike_marginals_pass: bool
    cptp_within_tolerance: bool
    energy_conserved_within_tolerance: bool
    energy_resolved_instrument_within_tolerance: bool
    durable_physical_pointer_derived: bool
    covariant_matching_current_derived: bool
    record_to_gravity_source_derived: bool


def homogeneous_continuous_time_early_arrival_probability(
    rate_per_time: float,
    hops: int,
    elapsed_time: float,
) -> float:
    """Return the Erlang early-arrival probability for equal exponential hops.

    ``rate_per_time * elapsed_time`` is the dimensionless argument.  In exact
    real arithmetic, every positive rate, positive time, and finite positive
    ``hops`` gives a strictly positive probability.  The implementation uses a
    lower-tail series when direct subtraction would cancel; values smaller than
    the floating-point range can still underflow to zero.  Hence a
    continuous-time Markov domino cannot by itself implement an exact
    light-cone front.
    """

    if not math.isfinite(rate_per_time) or rate_per_time <= 0.0:
        raise ValueError("rate_per_time must be finite and positive")
    if not isinstance(hops, int) or isinstance(hops, bool) or hops <= 0:
        raise ValueError("hops must be a positive integer")
    if not math.isfinite(elapsed_time) or elapsed_time < 0.0:
        raise ValueError("elapsed_time must be finite and non-negative")
    if elapsed_time == 0.0:
        return 0.0

    argument = rate_per_time * elapsed_time
    if argument == 0.0:
        return 0.0
    if argument < hops + 1.0:
        log_term = -argument + hops * math.log(argument) - math.lgamma(hops + 1.0)
        term = math.exp(log_term)
        probability = term
        order = hops
        for _ in range(100000):
            order += 1
            term *= argument / order
            probability += term
            if order > argument and term <= math.ulp(1.0) * probability:
                break
    else:
        log_terms = tuple(
            -argument + order * math.log(argument) - math.lgamma(order + 1.0)
            for order in range(hops)
        )
        maximum_log = max(log_terms)
        log_survival = maximum_log + math.log(
            math.fsum(math.exp(value - maximum_log) for value in log_terms)
        )
        probability = -math.expm1(log_survival)
    return min(1.0, max(0.0, probability))


def _bit(index: int, qubit: int, total_qubits: int) -> int:
    return (index >> (total_qubits - qubit - 1)) & 1


def _controlled_partial_swap(
    total_qubits: int,
    parent: int,
    target: int,
    battery: int,
    theta: float,
) -> np.ndarray:
    """Construct one number-conserving controlled target--battery rotation."""

    dimension = 1 << total_qubits
    gate = np.eye(dimension, dtype=np.complex128)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    target_mask = 1 << (total_qubits - target - 1)
    battery_mask = 1 << (total_qubits - battery - 1)

    for basis_index in range(dimension):
        if (
            _bit(basis_index, parent, total_qubits) == 1
            and _bit(basis_index, target, total_qubits) == 0
            and _bit(basis_index, battery, total_qubits) == 1
        ):
            partner = basis_index ^ target_mask ^ battery_mask
            gate[basis_index, basis_index] = cosine
            gate[basis_index, partner] = -sine
            gate[partner, basis_index] = sine
            gate[partner, partner] = cosine
    return gate


def _domino_unitary(site_count: int, depth: int, theta: float) -> np.ndarray:
    total_qubits = site_count + depth
    dimension = 1 << total_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    for tick in range(depth):
        gate = _controlled_partial_swap(
            total_qubits,
            parent=tick,
            target=tick + 1,
            battery=site_count + tick,
            theta=theta,
        )
        unitary = gate @ unitary
    return unitary


def _kraus_operators(
    unitary: np.ndarray,
    site_count: int,
    depth: int,
) -> tuple[np.ndarray, ...]:
    system_dimension = 1 << site_count
    battery_dimension = 1 << depth
    initial_battery_index = battery_dimension - 1
    tensor = unitary.reshape(
        system_dimension,
        battery_dimension,
        system_dimension,
        battery_dimension,
    )
    return tuple(
        tensor[:, output_battery, :, initial_battery_index]
        for output_battery in range(battery_dimension)
    )


def _apply_channel(
    kraus_operators: tuple[np.ndarray, ...],
    density: np.ndarray,
) -> np.ndarray:
    return sum(
        (operator @ density @ operator.conj().T for operator in kraus_operators),
        start=np.zeros_like(density, dtype=np.complex128),
    )


def _single_site_reduced(
    density: np.ndarray,
    site_count: int,
    site: int,
) -> np.ndarray:
    dimensions = (2,) * site_count
    tensor = density.reshape(dimensions + dimensions)
    traced_sites = tuple(index for index in range(site_count) if index != site)
    permutation = (
        (site,)
        + traced_sites
        + (site_count + site,)
        + tuple(site_count + index for index in traced_sites)
    )
    trace_dimension = 1 << (site_count - 1)
    ordered = np.transpose(tensor, permutation).reshape(
        2,
        trace_dimension,
        2,
        trace_dimension,
    )
    return np.einsum("aibi->ab", ordered)


def _seed_product_state(site_count: int, seed: np.ndarray) -> np.ndarray:
    vector = np.asarray(seed, dtype=np.complex128)
    if vector.shape != (2,):
        raise ValueError("seed must be a two-component state vector")
    state = vector
    ground = np.array([1.0, 0.0], dtype=np.complex128)
    for _ in range(site_count - 1):
        state = np.kron(state, ground)
    return state


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = 0.5 * (left - right + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _number_expectations(
    system_density: np.ndarray,
    site_count: int,
) -> tuple[float, ...]:
    diagonal = np.real(np.diag(system_density))
    return tuple(
        float(
            sum(
                probability * _bit(index, site, site_count)
                for index, probability in enumerate(diagonal)
            )
        )
        for site in range(site_count)
    )


def _channel_superoperator(kraus_operators: tuple[np.ndarray, ...]) -> np.ndarray:
    dimension = kraus_operators[0].shape[0]
    return sum(
        (np.kron(operator, operator.conj()) for operator in kraus_operators),
        start=np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128),
    )


def certify_causal_quantum_domino(
    *,
    site_count: int,
    depth: int,
    theta: float,
    lattice_spacing: float,
    clock_step: float,
    causal_speed: float,
    energy_gap: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CausalQuantumDominoCertificate:
    """Build and audit one finite-depth energy-conserving causal channel.

    ``lattice_spacing``, ``clock_step``, and ``causal_speed`` must use one
    compatible length/time convention.  ``energy_gap`` uses one declared energy
    unit shared by the system and batteries; it never enters a trigonometric or
    probability argument.
    """

    if not isinstance(site_count, int) or isinstance(site_count, bool) or site_count < 2:
        raise ValueError("site_count must be an integer at least two")
    if site_count > MAX_SYSTEM_QUBITS:
        raise ValueError(
            f"site_count exceeds the finite certificate limit {MAX_SYSTEM_QUBITS}"
        )
    if (
        not isinstance(depth, int)
        or isinstance(depth, bool)
        or depth <= 0
        or depth >= site_count
    ):
        raise ValueError("depth must be a positive integer smaller than site_count")
    if site_count + depth > MAX_TOTAL_QUBITS:
        raise ValueError(
            f"site_count + depth exceeds the finite certificate limit {MAX_TOTAL_QUBITS}"
        )
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5 * math.pi:
        raise ValueError("theta must be a finite angle in [0, pi/2]")
    for value, name in (
        (lattice_spacing, "lattice_spacing"),
        (clock_step, "clock_step"),
        (causal_speed, "causal_speed"),
        (energy_gap, "energy_gap"),
        (tolerance, "tolerance"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")

    causal_ratio = lattice_spacing / (causal_speed * clock_step)
    if causal_ratio > 1.0:
        raise ValueError("causal timing requires clock_step >= lattice_spacing / causal_speed")

    unitary = _domino_unitary(site_count, depth, theta)
    total_qubits = site_count + depth
    global_dimension = 1 << total_qubits
    identity_global = np.eye(global_dimension, dtype=np.complex128)
    unitary_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity_global, ord="fro")
    )

    number_diagonal = np.array(
        [
            sum(_bit(index, qubit, total_qubits) for qubit in range(total_qubits))
            * energy_gap
            for index in range(global_dimension)
        ],
        dtype=np.float64,
    )
    hamiltonian = np.diag(number_diagonal).astype(np.complex128)
    hamiltonian_scale = float(np.linalg.norm(hamiltonian, ord="fro"))
    relative_energy_commutator_residual = float(
        np.linalg.norm(unitary @ hamiltonian - hamiltonian @ unitary, ord="fro")
        / hamiltonian_scale
    )

    kraus = _kraus_operators(unitary, site_count, depth)
    system_dimension = 1 << site_count
    identity_system = np.eye(system_dimension, dtype=np.complex128)
    system_number_diagonal = np.array(
        [
            sum(_bit(index, site, site_count) for site in range(site_count))
            * energy_gap
            for index in range(system_dimension)
        ],
        dtype=np.float64,
    )
    system_hamiltonian = np.diag(system_number_diagonal).astype(np.complex128)
    system_hamiltonian_scale = max(
        float(np.linalg.norm(system_hamiltonian, ord="fro")),
        energy_gap,
    )
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        start=np.zeros_like(identity_system),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(completeness - identity_system, ord="fro")
    )
    choi = sum(
        (
            np.outer(
                operator.reshape(-1, order="F"),
                operator.reshape(-1, order="F").conj(),
            )
            for operator in kraus
        ),
        start=np.zeros(
            (system_dimension * system_dimension, system_dimension * system_dimension),
            dtype=np.complex128,
        ),
    )
    minimum_choi_eigenvalue = float(np.min(np.linalg.eigvalsh(choi)))

    excited = np.array([0.0, 1.0], dtype=np.complex128)
    seed_state = _seed_product_state(site_count, excited)
    seed_density = np.outer(seed_state, seed_state.conj())
    initial_system_energy = float(
        np.vdot(seed_state, system_hamiltonian @ seed_state).real
    )
    initial_battery_energy = depth * energy_gap
    output_density = _apply_channel(kraus, seed_density)
    output_trace_residual = abs(float(np.trace(output_density).real) - 1.0)
    minimum_output_eigenvalue = float(np.min(np.linalg.eigvalsh(output_density)))
    born_probabilities = np.real(np.diag(output_density))
    born_probability_sum_residual = abs(float(np.sum(born_probabilities)) - 1.0)
    minimum_born_probability = float(np.min(born_probabilities))
    activation_probabilities = _number_expectations(output_density, site_count)

    structural_influence_cone = tuple(range(depth + 1))
    spacelike_sites = tuple(range(depth + 1, site_count))
    seed_family = (
        np.array([1.0, 0.0], dtype=np.complex128),
        excited,
        np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),
        np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0),
    )
    output_family = tuple(
        _apply_channel(
            kraus,
            np.outer(state := _seed_product_state(site_count, seed), state.conj()),
        )
        for seed in seed_family
    )
    maximum_sampled_spacelike_trace_distance = 0.0
    for site in spacelike_sites:
        reference = _single_site_reduced(output_family[0], site_count, site)
        for candidate in output_family[1:]:
            maximum_sampled_spacelike_trace_distance = max(
                maximum_sampled_spacelike_trace_distance,
                _trace_distance(
                    _single_site_reduced(candidate, site_count, site),
                    reference,
                ),
            )

    zero_unitary = _domino_unitary(site_count, depth, 0.0)
    zero_kraus = _kraus_operators(zero_unitary, site_count, depth)
    standard_limit_superoperator_residual = float(
        np.linalg.norm(
            _channel_superoperator(zero_kraus)
            - np.eye(system_dimension * system_dimension, dtype=np.complex128),
            ord="fro",
        )
    )

    battery_dimension = 1 << depth
    battery_input = np.zeros(battery_dimension, dtype=np.complex128)
    battery_input[-1] = 1.0
    global_input = np.kron(seed_state, battery_input)
    global_output = unitary @ global_input
    direct_output_matrix = global_output.reshape(system_dimension, battery_dimension)
    direct_output_density = direct_output_matrix @ direct_output_matrix.conj().T
    kraus_vs_direct_partial_trace_residual = float(
        np.linalg.norm(output_density - direct_output_density, ord="fro")
    )

    battery_outcomes: list[BatteryOutcomeReceipt] = []
    reverse_transfer_operator = np.zeros_like(system_hamiltonian)
    expected_battery_energy_paid = 0.0
    branch_residuals: list[float] = []
    for output_battery, operator in enumerate(kraus):
        final_outcome_energy = output_battery.bit_count() * energy_gap
        energy_paid = initial_battery_energy - final_outcome_energy
        branch_state = operator @ seed_state
        branch_probability = float(np.vdot(branch_state, branch_state).real)
        expected_battery_energy_paid += branch_probability * energy_paid
        reverse_transfer_operator += (
            operator.conj().T
            @ (system_hamiltonian - energy_paid * identity_system)
            @ operator
        )
        conditional_system_energy: float | None = None
        relative_branch_energy_residual: float | None = None
        if branch_probability > tolerance:
            conditional_system_energy = float(
                np.vdot(branch_state, system_hamiltonian @ branch_state).real
                / branch_probability
            )
            branch_energy_scale = max(
                initial_system_energy + energy_paid,
                energy_gap,
            )
            relative_branch_energy_residual = abs(
                conditional_system_energy - initial_system_energy - energy_paid
            ) / branch_energy_scale
            branch_residuals.append(relative_branch_energy_residual)
        battery_outcomes.append(
            BatteryOutcomeReceipt(
                basis_label=format(output_battery, f"0{depth}b"),
                probability=branch_probability,
                final_battery_energy=final_outcome_energy,
                energy_paid_to_system=energy_paid,
                conditional_system_energy=conditional_system_energy,
                relative_branch_energy_residual=relative_branch_energy_residual,
            )
        )
    relative_reverse_transfer_identity_residual = float(
        np.linalg.norm(
            reverse_transfer_operator - system_hamiltonian,
            ord="fro",
        )
        / system_hamiltonian_scale
    )
    maximum_relative_branch_energy_residual = max(branch_residuals, default=0.0)

    global_probabilities = np.abs(global_output) ** 2
    final_system_excitation = 0.0
    final_battery_excitation = 0.0
    for index, probability in enumerate(global_probabilities):
        final_system_excitation += probability * sum(
            _bit(index, site, total_qubits) for site in range(site_count)
        )
        final_battery_excitation += probability * sum(
            _bit(index, site_count + battery, total_qubits)
            for battery in range(depth)
        )
    final_system_energy = float(final_system_excitation * energy_gap)
    final_battery_energy = float(final_battery_excitation * energy_gap)
    initial_total_energy = initial_system_energy + initial_battery_energy
    relative_total_energy_balance_residual = abs(
        final_system_energy + final_battery_energy - initial_total_energy
    ) / initial_total_energy

    cptp_within_tolerance = (
        unitary_residual <= tolerance * math.sqrt(global_dimension)
        and kraus_completeness_residual <= tolerance * math.sqrt(system_dimension)
        and minimum_choi_eigenvalue >= -tolerance
        and output_trace_residual <= tolerance
        and minimum_output_eigenvalue >= -tolerance
        and born_probability_sum_residual <= tolerance
        and minimum_born_probability >= -tolerance
        and kraus_vs_direct_partial_trace_residual <= tolerance
    )
    energy_conserved_within_tolerance = (
        relative_energy_commutator_residual <= tolerance
        and relative_total_energy_balance_residual <= tolerance
    )
    energy_resolved_instrument_within_tolerance = (
        relative_reverse_transfer_identity_residual <= tolerance
        and maximum_relative_branch_energy_residual <= tolerance
        and abs(
            final_system_energy
            - initial_system_energy
            - expected_battery_energy_paid
        )
        <= tolerance * max(initial_system_energy + initial_battery_energy, energy_gap)
    )
    structural_causal_support_exact = (
        causal_ratio <= 1.0
        and structural_influence_cone == tuple(range(depth + 1))
        and all(site > depth for site in spacelike_sites)
    )
    sampled_spacelike_marginals_pass = (
        maximum_sampled_spacelike_trace_distance <= tolerance
    )

    return CausalQuantumDominoCertificate(
        site_count=site_count,
        depth=depth,
        theta=theta,
        trigger_probability=math.sin(theta) ** 2,
        lattice_spacing=lattice_spacing,
        clock_step=clock_step,
        causal_speed=causal_speed,
        causal_ratio=causal_ratio,
        front_speed_bound=lattice_spacing / clock_step,
        structural_influence_cone=structural_influence_cone,
        spacelike_sites=spacelike_sites,
        activation_probabilities=activation_probabilities,
        channel_dimension=system_dimension,
        kraus_count=len(kraus),
        unitary_residual=unitary_residual,
        relative_energy_commutator_residual=relative_energy_commutator_residual,
        kraus_completeness_residual=kraus_completeness_residual,
        minimum_choi_eigenvalue=minimum_choi_eigenvalue,
        output_trace_residual=output_trace_residual,
        minimum_output_eigenvalue=minimum_output_eigenvalue,
        born_probability_sum_residual=born_probability_sum_residual,
        minimum_born_probability=minimum_born_probability,
        maximum_sampled_spacelike_trace_distance=(
            maximum_sampled_spacelike_trace_distance
        ),
        kraus_vs_direct_partial_trace_residual=(
            kraus_vs_direct_partial_trace_residual
        ),
        standard_limit_superoperator_residual=standard_limit_superoperator_residual,
        energy_gap=energy_gap,
        initial_system_energy=initial_system_energy,
        final_system_energy=final_system_energy,
        initial_battery_energy=initial_battery_energy,
        final_battery_energy=final_battery_energy,
        relative_total_energy_balance_residual=relative_total_energy_balance_residual,
        battery_outcomes=tuple(battery_outcomes),
        expected_battery_energy_paid=expected_battery_energy_paid,
        relative_reverse_transfer_identity_residual=(
            relative_reverse_transfer_identity_residual
        ),
        maximum_relative_branch_energy_residual=(
            maximum_relative_branch_energy_residual
        ),
        structural_causal_support_exact=structural_causal_support_exact,
        sampled_spacelike_marginals_pass=sampled_spacelike_marginals_pass,
        cptp_within_tolerance=cptp_within_tolerance,
        energy_conserved_within_tolerance=energy_conserved_within_tolerance,
        energy_resolved_instrument_within_tolerance=(
            energy_resolved_instrument_within_tolerance
        ),
        durable_physical_pointer_derived=False,
        covariant_matching_current_derived=False,
        record_to_gravity_source_derived=False,
    )
