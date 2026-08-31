"""Finite-horizon, time-homogeneous pointer propagation audit.

This module separates two claims that are easy to conflate.

First, a finite-dimensional, time-independent Hamiltonian has analytic matrix
elements.  On a nearest-neighbour chain with a unique non-zero minimal path,

    <h|exp(-i H t)|0> = (-i t)^h prod_j J_j / h! + O(t^(h+1)).

It therefore cannot remain exactly zero on an open time interval and then
start propagating.  This is a conditional statement about a propagating path,
not a theorem that every local Hamiltonian spreads every observable.  A
commuting Ising chain is included as a negative control for that broader and
false claim.

Second, an exact finite-tick cone is possible for a discrete locality-
preserving update.  Every lattice cell contains a head register, one system
qubit, and one battery qubit.  One fixed update is repeated:

    W = (tensor_x C_x) S_head,       C = R P.

``S_head`` translates all head registers one cell to the right.  ``P`` cycles
ACTIVE -> D1 -> ... -> DL -> ACTIVE and fixes EMPTY.  ``R`` rotates only

    |F> = |D1,     system=0, battery=1>,
    |S> = |ACTIVE, system=1, battery=0>.

These states have equal system-plus-battery energy.  Hence the onsite coin is
unitary and energy preserving, while the head translation is a range-one
tensor-factor permutation for arbitrary head configurations.  Starting with
one supplied ACTIVE head gives domino probabilities p^j, p=sin(theta)^2,
without an externally scheduled gate for each edge.

The certificate is deliberately finite and conditional.  The repeated tick,
the initial ACTIVE head, and the lattice are supplied.  A finite dead-state
cycle eventually reactivates, so the audited depth must be smaller than the
number of dead states.  No continuous physical clock, covariant action, GR
limit, record-to-gravity source, or cross-dataset prediction is derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8
MAX_QCA_SITES = 4
MAX_DEAD_STATES = 8

EMPTY = 0
ACTIVE = 1
D1 = 2


@dataclass(frozen=True)
class ContinuousHamiltonianFrontAudit:
    """Conditional analytic-front audit plus a non-propagating control."""

    hop_count: int
    coupling_rate: float
    elapsed_time: float
    dimensionless_coupling_time: float
    lattice_spacing: float
    causal_speed: float
    causal_arrival_time: float
    sampled_before_causal_arrival: bool
    path_product: float
    first_nonzero_power: int
    exact_endpoint_amplitude: complex
    exact_endpoint_probability: float
    leading_endpoint_amplitude: complex
    absolute_leading_residual: float
    relative_leading_residual: float
    ising_distant_commutator_norm: float
    unique_minimal_path: bool
    minimal_path_coefficient_nonzero: bool
    open_interval_exact_delay_impossible: bool
    sampled_early_tail_nonzero: bool
    commuting_ising_negative_control_pass: bool
    broad_all_local_hamiltonians_spread_claim_refuted: bool
    every_positive_time_nonzero_claimed: bool
    exact_relativistic_dynamics_derived: bool


@dataclass(frozen=True)
class QcaBatteryOutcomeReceipt:
    """One final head-and-battery outcome of the reduced system channel."""

    environment_label: str
    probability: float
    final_battery_energy: float
    energy_paid_to_system: float
    conditional_system_energy: float | None
    relative_branch_energy_residual: float | None


@dataclass(frozen=True)
class TimeHomogeneousPointerQcaCertificate:
    """Numerical and structural receipt for one finite pointer-QCA witness."""

    site_count: int
    audited_depth: int
    dead_state_count: int
    head_local_dimension: int
    head_cycle_period: int
    theta: float
    trigger_probability: float
    lattice_spacing: float
    clock_step: float
    causal_speed: float
    causal_ratio: float
    front_speed_bound: float
    energy_gap: float
    structural_influence_cone: tuple[int, ...]
    spacelike_sites: tuple[int, ...]
    activation_probabilities: tuple[float, ...]
    expected_activation_probabilities: tuple[float, ...]
    maximum_activation_formula_residual: float
    paid_energy_probabilities: tuple[float, ...]
    expected_paid_energy_probabilities: tuple[float, ...]
    maximum_paid_distribution_residual: float
    system_dimension: int
    kraus_count: int
    local_coin_unitarity_residual: float
    relative_local_energy_commutator_residual: float
    head_shift_configuration_count: int
    head_shift_unique_image_count: int
    kraus_completeness_residual: float
    minimum_choi_eigenvalue: float
    output_trace_residual: float
    minimum_output_eigenvalue: float
    born_probability_sum_residual: float
    minimum_born_probability: float
    maximum_seed_variation_spacelike_trace_distance: float
    standard_limit_superoperator_residual: float
    initial_system_energy: float
    final_system_energy: float
    initial_battery_energy: float
    final_battery_energy: float
    relative_total_energy_balance_residual: float
    battery_outcomes: tuple[QcaBatteryOutcomeReceipt, ...]
    expected_battery_energy_paid: float
    relative_reverse_transfer_identity_residual: float
    maximum_relative_branch_energy_residual: float
    audited_depth_less_than_dead_count: bool
    head_shift_bijection: bool
    arbitrary_multihead_configurations_covered: bool
    full_tensor_qca_unitary_by_composition: bool
    time_homogeneous_discrete_update: bool
    external_per_edge_schedule_required: bool
    structural_causal_support_exact: bool
    cptp_within_tolerance: bool
    energy_conserved_within_tolerance: bool
    energy_resolved_instrument_within_tolerance: bool
    quantum_identity_limit_at_zero_coupling: bool
    pointer_seed_statistics_match_prior_domino: bool
    finite_horizon_only: bool
    trigger_head_preparation_derived: bool
    continuous_physical_clock_derived: bool
    permanent_absorbing_dead_state_derived: bool
    covariant_action_derived: bool
    gr_limit_derived: bool
    record_to_gravity_source_derived: bool
    full_prior_domino_channel_equivalence_derived: bool
    cross_dataset_parameter_fixing_derived: bool
    independent_holdout_prediction_derived: bool


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _single_site_operator(
    operator: np.ndarray,
    site: int,
    site_count: int,
) -> np.ndarray:
    result = np.array([[1.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    for index in range(site_count):
        result = np.kron(result, operator if index == site else identity)
    return result


def certify_continuous_hamiltonian_front(
    *,
    hop_count: int,
    coupling_rate: float,
    elapsed_time: float,
    lattice_spacing: float,
    causal_speed: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ContinuousHamiltonianFrontAudit:
    """Audit the narrow analytic no-delay result and its negative control.

    The Hamiltonian is expressed in angular-frequency units, i.e. ``H / hbar``.
    Consequently ``coupling_rate * elapsed_time`` is dimensionless.  A non-zero
    first Taylor coefficient rules out a matrix element that vanishes on an
    open interval after ``t=0``; it does *not* rule out isolated later zeros.
    """

    if (
        not isinstance(hop_count, int)
        or isinstance(hop_count, bool)
        or hop_count < 2
    ):
        raise ValueError("hop_count must be an integer at least two")
    if hop_count > 6:
        raise ValueError("hop_count exceeds the finite audit limit 6")
    for value, name in (
        (coupling_rate, "coupling_rate"),
        (elapsed_time, "elapsed_time"),
        (lattice_spacing, "lattice_spacing"),
        (causal_speed, "causal_speed"),
        (tolerance, "tolerance"),
    ):
        _validate_positive(value, name)
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")

    site_count = hop_count + 1
    hamiltonian = np.zeros((site_count, site_count), dtype=np.complex128)
    for site in range(hop_count):
        hamiltonian[site, site + 1] = coupling_rate
        hamiltonian[site + 1, site] = coupling_rate

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    propagator = (
        eigenvectors
        @ np.diag(np.exp(-1.0j * eigenvalues * elapsed_time))
        @ eigenvectors.conj().T
    )
    exact_amplitude = complex(propagator[hop_count, 0])
    path_product = coupling_rate**hop_count
    leading_amplitude = (
        (-1.0j * elapsed_time) ** hop_count
        * path_product
        / math.factorial(hop_count)
    )
    absolute_leading_residual = abs(exact_amplitude - leading_amplitude)
    relative_leading_residual = absolute_leading_residual / max(
        abs(leading_amplitude),
        np.finfo(np.float64).tiny,
    )

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ising_dimension = 1 << site_count
    ising_hamiltonian = np.zeros(
        (ising_dimension, ising_dimension), dtype=np.complex128
    )
    for site in range(site_count - 1):
        ising_hamiltonian += coupling_rate * (
            _single_site_operator(pauli_z, site, site_count)
            @ _single_site_operator(pauli_z, site + 1, site_count)
        )
    ising_values, ising_vectors = np.linalg.eigh(ising_hamiltonian)
    ising_unitary = (
        ising_vectors
        @ np.diag(np.exp(-1.0j * ising_values * elapsed_time))
        @ ising_vectors.conj().T
    )
    source_x = _single_site_operator(pauli_x, 0, site_count)
    distant_x = _single_site_operator(pauli_x, hop_count, site_count)
    evolved_source_x = ising_unitary.conj().T @ source_x @ ising_unitary
    ising_commutator = evolved_source_x @ distant_x - distant_x @ evolved_source_x
    ising_commutator_norm = float(np.linalg.norm(ising_commutator, ord="fro"))

    causal_arrival_time = hop_count * lattice_spacing / causal_speed
    exact_probability = float(abs(exact_amplitude) ** 2)
    negative_control_limit = tolerance * math.sqrt(ising_dimension)
    return ContinuousHamiltonianFrontAudit(
        hop_count=hop_count,
        coupling_rate=coupling_rate,
        elapsed_time=elapsed_time,
        dimensionless_coupling_time=coupling_rate * elapsed_time,
        lattice_spacing=lattice_spacing,
        causal_speed=causal_speed,
        causal_arrival_time=causal_arrival_time,
        sampled_before_causal_arrival=elapsed_time < causal_arrival_time,
        path_product=path_product,
        first_nonzero_power=hop_count,
        exact_endpoint_amplitude=exact_amplitude,
        exact_endpoint_probability=exact_probability,
        leading_endpoint_amplitude=complex(leading_amplitude),
        absolute_leading_residual=float(absolute_leading_residual),
        relative_leading_residual=float(relative_leading_residual),
        ising_distant_commutator_norm=ising_commutator_norm,
        unique_minimal_path=True,
        minimal_path_coefficient_nonzero=path_product != 0.0,
        open_interval_exact_delay_impossible=path_product != 0.0,
        sampled_early_tail_nonzero=exact_probability > 0.0,
        commuting_ising_negative_control_pass=(
            ising_commutator_norm <= negative_control_limit
        ),
        broad_all_local_hamiltonians_spread_claim_refuted=True,
        every_positive_time_nonzero_claimed=False,
        exact_relativistic_dynamics_derived=False,
    )


def _head_successor(head_state: int, dead_state_count: int) -> int:
    if head_state == EMPTY:
        return EMPTY
    if head_state == ACTIVE:
        return D1
    final_dead_state = D1 + dead_state_count - 1
    if D1 <= head_state < final_dead_state:
        return head_state + 1
    if head_state == final_dead_state:
        return ACTIVE
    raise ValueError("head_state is outside the declared local register")


def _local_index(head_state: int, system_bit: int, battery_bit: int) -> int:
    return (head_state * 2 + system_bit) * 2 + battery_bit


def _local_coin(
    dead_state_count: int,
    theta: float,
    energy_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    head_dimension = dead_state_count + 2
    local_dimension = 4 * head_dimension
    permutation = np.zeros((local_dimension, local_dimension), dtype=np.complex128)
    for head_state in range(head_dimension):
        for system_bit in (0, 1):
            for battery_bit in (0, 1):
                source = _local_index(head_state, system_bit, battery_bit)
                target = _local_index(
                    _head_successor(head_state, dead_state_count),
                    system_bit,
                    battery_bit,
                )
                permutation[target, source] = 1.0

    rotation = np.eye(local_dimension, dtype=np.complex128)
    failure_index = _local_index(D1, 0, 1)
    success_index = _local_index(ACTIVE, 1, 0)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rotation[failure_index, failure_index] = cosine
    rotation[success_index, failure_index] = sine
    rotation[failure_index, success_index] = -sine
    rotation[success_index, success_index] = cosine
    coin = rotation @ permutation

    energy_diagonal = np.array(
        [
            (system_bit + battery_bit) * energy_gap
            for head_state in range(head_dimension)
            for system_bit in (0, 1)
            for battery_bit in (0, 1)
        ],
        dtype=np.float64,
    )
    return coin, np.diag(energy_diagonal).astype(np.complex128)


def _translate_head_configuration(
    encoded: int,
    *,
    site_count: int,
    head_dimension: int,
) -> int:
    input_digits: list[int] = []
    remainder = encoded
    for _ in range(site_count):
        input_digits.append(remainder % head_dimension)
        remainder //= head_dimension
    output_digits = [EMPTY] * site_count
    for site, head_state in enumerate(input_digits):
        output_digits[(site + 1) % site_count] = head_state
    translated = 0
    place = 1
    for head_state in output_digits:
        translated += place * head_state
        place *= head_dimension
    return translated


def _add_sparse_amplitude(
    state: dict[tuple[int, int, int, int], complex],
    key: tuple[int, int, int, int],
    amplitude: complex,
) -> None:
    if amplitude != 0.0:
        state[key] = state.get(key, 0.0j) + amplitude


def _one_head_kraus_operators(
    *,
    site_count: int,
    depth: int,
    dead_state_count: int,
    theta: float,
) -> tuple[tuple[tuple[int, int, int], np.ndarray], ...]:
    """Evolve the closed one-nonempty-head sector and expose its Kraus map."""

    system_dimension = 1 << site_count
    initial_battery_bits = system_dimension - 1
    environment_operators: dict[tuple[int, int, int], np.ndarray] = {}
    cosine = math.cos(theta)
    sine = math.sin(theta)

    for input_system_bits in range(system_dimension):
        state: dict[tuple[int, int, int, int], complex] = {
            (input_system_bits, 0, ACTIVE, initial_battery_bits): 1.0 + 0.0j
        }
        for _ in range(depth):
            next_state: dict[tuple[int, int, int, int], complex] = {}
            for (
                system_bits,
                head_position,
                head_state,
                battery_bits,
            ), amplitude in state.items():
                shifted_position = (head_position + 1) % site_count
                cycled_head = _head_successor(head_state, dead_state_count)
                system_bit = (system_bits >> shifted_position) & 1
                battery_bit = (battery_bits >> shifted_position) & 1

                if cycled_head == D1 and system_bit == 0 and battery_bit == 1:
                    _add_sparse_amplitude(
                        next_state,
                        (
                            system_bits,
                            shifted_position,
                            D1,
                            battery_bits,
                        ),
                        cosine * amplitude,
                    )
                    _add_sparse_amplitude(
                        next_state,
                        (
                            system_bits | (1 << shifted_position),
                            shifted_position,
                            ACTIVE,
                            battery_bits & ~(1 << shifted_position),
                        ),
                        sine * amplitude,
                    )
                elif (
                    cycled_head == ACTIVE
                    and system_bit == 1
                    and battery_bit == 0
                ):
                    _add_sparse_amplitude(
                        next_state,
                        (
                            system_bits & ~(1 << shifted_position),
                            shifted_position,
                            D1,
                            battery_bits | (1 << shifted_position),
                        ),
                        -sine * amplitude,
                    )
                    _add_sparse_amplitude(
                        next_state,
                        (
                            system_bits,
                            shifted_position,
                            ACTIVE,
                            battery_bits,
                        ),
                        cosine * amplitude,
                    )
                else:
                    _add_sparse_amplitude(
                        next_state,
                        (
                            system_bits,
                            shifted_position,
                            cycled_head,
                            battery_bits,
                        ),
                        amplitude,
                    )
            state = next_state

        for (
            output_system_bits,
            head_position,
            head_state,
            battery_bits,
        ), amplitude in state.items():
            environment = (head_position, head_state, battery_bits)
            operator = environment_operators.setdefault(
                environment,
                np.zeros(
                    (system_dimension, system_dimension), dtype=np.complex128
                ),
            )
            operator[output_system_bits, input_system_bits] += amplitude

    return tuple(sorted(environment_operators.items(), key=lambda item: item[0]))


def _apply_channel(
    kraus: tuple[np.ndarray, ...],
    density: np.ndarray,
) -> np.ndarray:
    return sum(
        (operator @ density @ operator.conj().T for operator in kraus),
        start=np.zeros_like(density, dtype=np.complex128),
    )


def _channel_superoperator(kraus: tuple[np.ndarray, ...]) -> np.ndarray:
    dimension = kraus[0].shape[0]
    return sum(
        (np.kron(operator, operator.conj()) for operator in kraus),
        start=np.zeros(
            (dimension * dimension, dimension * dimension),
            dtype=np.complex128,
        ),
    )


def _seed_product_state(site_count: int, seed: np.ndarray) -> np.ndarray:
    seed_vector = np.asarray(seed, dtype=np.complex128)
    if seed_vector.shape != (2,):
        raise ValueError("seed must have two components")
    state = np.zeros(1 << site_count, dtype=np.complex128)
    state[0] = seed_vector[0]
    state[1] = seed_vector[1]
    return state


def _single_site_reduced(
    density: np.ndarray,
    *,
    site_count: int,
    site: int,
) -> np.ndarray:
    dimension = 1 << site_count
    reduced = np.zeros((2, 2), dtype=np.complex128)
    other_mask = (dimension - 1) & ~(1 << site)
    for row in range(dimension):
        for column in range(dimension):
            if (row & other_mask) == (column & other_mask):
                reduced[(row >> site) & 1, (column >> site) & 1] += density[
                    row, column
                ]
    return reduced


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    hermitian_difference = 0.5 * (
        left - right + (left - right).conj().T
    )
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian_difference))))


def _number_expectations(
    density: np.ndarray,
    site_count: int,
) -> tuple[float, ...]:
    diagonal = np.real(np.diag(density))
    return tuple(
        float(
            sum(
                probability * ((basis >> site) & 1)
                for basis, probability in enumerate(diagonal)
            )
        )
        for site in range(site_count)
    )


def _environment_label(
    environment: tuple[int, int, int],
    site_count: int,
) -> str:
    position, head_state, battery_bits = environment
    return (
        f"head@{position}:state={head_state};"
        f"battery={format(battery_bits, f'0{site_count}b')}"
    )


def certify_time_homogeneous_pointer_qca(
    *,
    site_count: int,
    audited_depth: int,
    dead_state_count: int,
    theta: float,
    lattice_spacing: float,
    clock_step: float,
    causal_speed: float,
    energy_gap: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TimeHomogeneousPointerQcaCertificate:
    """Build and audit a fixed finite-tick, energy-paying pointer QCA.

    ``theta`` and ``lattice_spacing / (causal_speed * clock_step)`` are the
    dimensionless cores.  ``energy_gap`` is a shared system/battery unit and
    never enters a probability function.  The function enforces a horizon
    strictly shorter than the dead-state count and shorter than a ring wrap.
    """

    if (
        not isinstance(site_count, int)
        or isinstance(site_count, bool)
        or not 3 <= site_count <= MAX_QCA_SITES
    ):
        raise ValueError(
            f"site_count must be an integer in [3, {MAX_QCA_SITES}]"
        )
    if (
        not isinstance(audited_depth, int)
        or isinstance(audited_depth, bool)
        or audited_depth <= 0
        or audited_depth >= site_count - 1
    ):
        raise ValueError(
            "audited_depth must be positive and leave an unvisited site before wrap"
        )
    if (
        not isinstance(dead_state_count, int)
        or isinstance(dead_state_count, bool)
        or dead_state_count <= audited_depth
        or dead_state_count > MAX_DEAD_STATES
    ):
        raise ValueError(
            "dead_state_count must exceed audited_depth and stay within the finite limit"
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
        _validate_positive(value, name)
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")

    causal_ratio = lattice_spacing / (causal_speed * clock_step)
    if causal_ratio > 1.0:
        raise ValueError(
            "causal timing requires clock_step >= lattice_spacing / causal_speed"
        )

    coin, local_hamiltonian = _local_coin(
        dead_state_count,
        theta,
        energy_gap,
    )
    local_identity = np.eye(coin.shape[0], dtype=np.complex128)
    local_coin_unitarity_residual = float(
        np.linalg.norm(coin.conj().T @ coin - local_identity, ord="fro")
    )
    local_energy_scale = max(
        float(np.linalg.norm(local_hamiltonian, ord="fro")),
        energy_gap,
    )
    relative_local_energy_commutator_residual = float(
        np.linalg.norm(
            coin @ local_hamiltonian - local_hamiltonian @ coin,
            ord="fro",
        )
        / local_energy_scale
    )

    head_dimension = dead_state_count + 2
    head_shift_configuration_count = head_dimension**site_count
    translated_configurations = {
        _translate_head_configuration(
            configuration,
            site_count=site_count,
            head_dimension=head_dimension,
        )
        for configuration in range(head_shift_configuration_count)
    }
    head_shift_unique_image_count = len(translated_configurations)
    head_shift_bijection = (
        head_shift_unique_image_count == head_shift_configuration_count
    )
    full_tensor_qca_unitary_by_composition = (
        head_shift_bijection and local_coin_unitarity_residual <= tolerance
    )

    labelled_kraus = _one_head_kraus_operators(
        site_count=site_count,
        depth=audited_depth,
        dead_state_count=dead_state_count,
        theta=theta,
    )
    environments = tuple(environment for environment, _ in labelled_kraus)
    kraus = tuple(operator for _, operator in labelled_kraus)
    system_dimension = 1 << site_count
    system_identity = np.eye(system_dimension, dtype=np.complex128)
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        start=np.zeros_like(system_identity),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(completeness - system_identity, ord="fro")
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
            (system_dimension * system_dimension,) * 2,
            dtype=np.complex128,
        ),
    )
    minimum_choi_eigenvalue = float(np.min(np.linalg.eigvalsh(choi)))

    system_energy_diagonal = np.array(
        [basis.bit_count() * energy_gap for basis in range(system_dimension)],
        dtype=np.float64,
    )
    system_hamiltonian = np.diag(system_energy_diagonal).astype(np.complex128)
    system_energy_scale = max(
        float(np.linalg.norm(system_hamiltonian, ord="fro")),
        energy_gap,
    )
    excited = np.array([0.0, 1.0], dtype=np.complex128)
    seed_state = _seed_product_state(site_count, excited)
    seed_density = np.outer(seed_state, seed_state.conj())
    output_density = _apply_channel(kraus, seed_density)
    output_trace_residual = abs(float(np.trace(output_density).real) - 1.0)
    minimum_output_eigenvalue = float(np.min(np.linalg.eigvalsh(output_density)))
    born_probabilities = np.real(np.diag(output_density))
    born_probability_sum_residual = abs(float(np.sum(born_probabilities)) - 1.0)
    minimum_born_probability = float(np.min(born_probabilities))

    trigger_probability = math.sin(theta) ** 2
    activation_probabilities = _number_expectations(output_density, site_count)
    expected_activation_probabilities = tuple(
        1.0
        if site == 0
        else (
            trigger_probability**site
            if 1 <= site <= audited_depth
            else 0.0
        )
        for site in range(site_count)
    )
    maximum_activation_formula_residual = max(
        abs(actual - expected)
        for actual, expected in zip(
            activation_probabilities,
            expected_activation_probabilities,
        )
    )

    structural_influence_cone = tuple(range(audited_depth + 1))
    spacelike_sites = tuple(range(audited_depth + 1, site_count))
    seed_family = (
        np.array([1.0, 0.0], dtype=np.complex128),
        excited,
        np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),
        np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0),
    )
    output_family = tuple(
        _apply_channel(
            kraus,
            np.outer(
                state := _seed_product_state(site_count, seed),
                state.conj(),
            ),
        )
        for seed in seed_family
    )
    maximum_seed_variation_spacelike_trace_distance = 0.0
    for site in spacelike_sites:
        reference = _single_site_reduced(
            output_family[0], site_count=site_count, site=site
        )
        for candidate in output_family[1:]:
            maximum_seed_variation_spacelike_trace_distance = max(
                maximum_seed_variation_spacelike_trace_distance,
                _trace_distance(
                    _single_site_reduced(
                        candidate, site_count=site_count, site=site
                    ),
                    reference,
                ),
            )

    zero_labelled_kraus = _one_head_kraus_operators(
        site_count=site_count,
        depth=audited_depth,
        dead_state_count=dead_state_count,
        theta=0.0,
    )
    zero_kraus = tuple(operator for _, operator in zero_labelled_kraus)
    standard_limit_superoperator_residual = float(
        np.linalg.norm(
            _channel_superoperator(zero_kraus)
            - np.eye(
                system_dimension * system_dimension, dtype=np.complex128
            ),
            ord="fro",
        )
    )

    initial_system_energy = float(
        np.vdot(seed_state, system_hamiltonian @ seed_state).real
    )
    final_system_energy = float(
        np.trace(system_hamiltonian @ output_density).real
    )
    initial_battery_energy = site_count * energy_gap
    final_battery_energy = 0.0
    expected_battery_energy_paid = 0.0
    reverse_transfer_operator = np.zeros_like(system_hamiltonian)
    paid_energy_probabilities = [0.0] * (audited_depth + 1)
    branch_residuals: list[float] = []
    receipts: list[QcaBatteryOutcomeReceipt] = []
    for environment, operator in zip(environments, kraus):
        _, _, battery_bits = environment
        outcome_battery_energy = battery_bits.bit_count() * energy_gap
        energy_paid = initial_battery_energy - outcome_battery_energy
        paid_units = int(round(energy_paid / energy_gap))
        branch_state = operator @ seed_state
        probability = float(np.vdot(branch_state, branch_state).real)
        final_battery_energy += probability * outcome_battery_energy
        expected_battery_energy_paid += probability * energy_paid
        reverse_transfer_operator += (
            operator.conj().T
            @ (system_hamiltonian - energy_paid * system_identity)
            @ operator
        )
        if 0 <= paid_units <= audited_depth:
            paid_energy_probabilities[paid_units] += probability

        conditional_system_energy: float | None = None
        relative_branch_energy_residual: float | None = None
        if probability > tolerance:
            conditional_system_energy = float(
                np.vdot(branch_state, system_hamiltonian @ branch_state).real
                / probability
            )
            relative_branch_energy_residual = abs(
                conditional_system_energy - initial_system_energy - energy_paid
            ) / max(initial_system_energy + abs(energy_paid), energy_gap)
            branch_residuals.append(relative_branch_energy_residual)
        receipts.append(
            QcaBatteryOutcomeReceipt(
                environment_label=_environment_label(environment, site_count),
                probability=probability,
                final_battery_energy=outcome_battery_energy,
                energy_paid_to_system=energy_paid,
                conditional_system_energy=conditional_system_energy,
                relative_branch_energy_residual=relative_branch_energy_residual,
            )
        )

    expected_paid_energy_probabilities = tuple(
        (
            trigger_probability**paid_units
            * (1.0 - trigger_probability)
            if paid_units < audited_depth
            else trigger_probability**audited_depth
        )
        for paid_units in range(audited_depth + 1)
    )
    maximum_paid_distribution_residual = max(
        abs(actual - expected)
        for actual, expected in zip(
            paid_energy_probabilities,
            expected_paid_energy_probabilities,
        )
    )
    relative_total_energy_balance_residual = abs(
        final_system_energy
        + final_battery_energy
        - initial_system_energy
        - initial_battery_energy
    ) / max(initial_system_energy + initial_battery_energy, energy_gap)
    relative_reverse_transfer_identity_residual = float(
        np.linalg.norm(
            reverse_transfer_operator - system_hamiltonian,
            ord="fro",
        )
        / system_energy_scale
    )
    maximum_relative_branch_energy_residual = max(branch_residuals, default=0.0)

    numerical_limit = tolerance * max(1.0, float(system_dimension))
    cptp_within_tolerance = max(
        kraus_completeness_residual,
        -minimum_choi_eigenvalue,
        output_trace_residual,
        -minimum_output_eigenvalue,
        born_probability_sum_residual,
        -minimum_born_probability,
    ) <= numerical_limit
    energy_conserved_within_tolerance = max(
        relative_local_energy_commutator_residual,
        relative_total_energy_balance_residual,
    ) <= numerical_limit
    energy_resolved_instrument_within_tolerance = max(
        relative_reverse_transfer_identity_residual,
        maximum_relative_branch_energy_residual,
        maximum_paid_distribution_residual,
    ) <= numerical_limit
    quantum_identity_limit_at_zero_coupling = (
        standard_limit_superoperator_residual <= numerical_limit
    )
    pointer_seed_statistics_match_prior_domino = max(
        maximum_activation_formula_residual,
        maximum_paid_distribution_residual,
    ) <= numerical_limit

    return TimeHomogeneousPointerQcaCertificate(
        site_count=site_count,
        audited_depth=audited_depth,
        dead_state_count=dead_state_count,
        head_local_dimension=head_dimension,
        head_cycle_period=dead_state_count + 1,
        theta=theta,
        trigger_probability=trigger_probability,
        lattice_spacing=lattice_spacing,
        clock_step=clock_step,
        causal_speed=causal_speed,
        causal_ratio=causal_ratio,
        front_speed_bound=lattice_spacing / clock_step,
        energy_gap=energy_gap,
        structural_influence_cone=structural_influence_cone,
        spacelike_sites=spacelike_sites,
        activation_probabilities=activation_probabilities,
        expected_activation_probabilities=expected_activation_probabilities,
        maximum_activation_formula_residual=maximum_activation_formula_residual,
        paid_energy_probabilities=tuple(paid_energy_probabilities),
        expected_paid_energy_probabilities=expected_paid_energy_probabilities,
        maximum_paid_distribution_residual=maximum_paid_distribution_residual,
        system_dimension=system_dimension,
        kraus_count=len(kraus),
        local_coin_unitarity_residual=local_coin_unitarity_residual,
        relative_local_energy_commutator_residual=(
            relative_local_energy_commutator_residual
        ),
        head_shift_configuration_count=head_shift_configuration_count,
        head_shift_unique_image_count=head_shift_unique_image_count,
        kraus_completeness_residual=kraus_completeness_residual,
        minimum_choi_eigenvalue=minimum_choi_eigenvalue,
        output_trace_residual=output_trace_residual,
        minimum_output_eigenvalue=minimum_output_eigenvalue,
        born_probability_sum_residual=born_probability_sum_residual,
        minimum_born_probability=minimum_born_probability,
        maximum_seed_variation_spacelike_trace_distance=(
            maximum_seed_variation_spacelike_trace_distance
        ),
        standard_limit_superoperator_residual=(
            standard_limit_superoperator_residual
        ),
        initial_system_energy=initial_system_energy,
        final_system_energy=final_system_energy,
        initial_battery_energy=initial_battery_energy,
        final_battery_energy=final_battery_energy,
        relative_total_energy_balance_residual=(
            relative_total_energy_balance_residual
        ),
        battery_outcomes=tuple(receipts),
        expected_battery_energy_paid=expected_battery_energy_paid,
        relative_reverse_transfer_identity_residual=(
            relative_reverse_transfer_identity_residual
        ),
        maximum_relative_branch_energy_residual=(
            maximum_relative_branch_energy_residual
        ),
        audited_depth_less_than_dead_count=(audited_depth < dead_state_count),
        head_shift_bijection=head_shift_bijection,
        arbitrary_multihead_configurations_covered=head_shift_bijection,
        full_tensor_qca_unitary_by_composition=(
            full_tensor_qca_unitary_by_composition
        ),
        time_homogeneous_discrete_update=True,
        external_per_edge_schedule_required=False,
        structural_causal_support_exact=True,
        cptp_within_tolerance=cptp_within_tolerance,
        energy_conserved_within_tolerance=energy_conserved_within_tolerance,
        energy_resolved_instrument_within_tolerance=(
            energy_resolved_instrument_within_tolerance
        ),
        quantum_identity_limit_at_zero_coupling=(
            quantum_identity_limit_at_zero_coupling
        ),
        pointer_seed_statistics_match_prior_domino=(
            pointer_seed_statistics_match_prior_domino
        ),
        finite_horizon_only=True,
        trigger_head_preparation_derived=False,
        continuous_physical_clock_derived=False,
        permanent_absorbing_dead_state_derived=False,
        covariant_action_derived=False,
        gr_limit_derived=False,
        record_to_gravity_source_derived=False,
        full_prior_domino_channel_equivalence_derived=False,
        cross_dataset_parameter_fixing_derived=False,
        independent_holdout_prediction_derived=False,
    )
