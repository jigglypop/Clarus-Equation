'''All-signed normal-ordered scalar cubic exchange on the E68 frozen ansatz.

The four oscillators are the two scalar branches of the real k and 2k
harmonics.  This module adopts normal ordering as an explicit finite-ansatz
quantization convention and closes every state reachable from |2_k,a> by one
of the 64 signed projected cubic monomials.  It is not a continuum momentum
Fock space or a Schwinger--Keldysh correlator.
'''

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    FrozenCubicVertex,
    dynamic_reduced_scalar_cubic_tensor_pair,
    frozen_symplectic_scalar_modes,
    project_frozen_scalar_hamiltonian_vertices,
    scalar_interaction_hamiltonian_cubic_tensor_pair,
)
from examples.physics.qft_reference_flrw_cubic_exchange import (
    RESONANCE_TOL,
    evaluate_scalar_cubic_exchange_gate,
    scalar_cubic_exchange_channels,
)
from examples.physics.qft_reference_flrw_quartic_contact import (
    evaluate_scalar_quartic_contact_gate,
)
from examples.physics.qft_reference_flrw_quartic_exchange import (
    finite_time_ordered_double_kernel,
    select_certified_resonant_matrix_element,
    simpson_ordered_double_kernel,
)


FockState = tuple[int, int, int, int]


@dataclass(frozen=True)
class NormalOrderedFockTransition:
    target: FockState
    matrix_element: complex
    contributing_monomial_count: int


def _validate_vertices(
    vertices: tuple[FrozenCubicVertex, ...],
) -> None:
    if len(vertices) != 64:
        raise ValueError('the all-signed cubic gate requires 64 assignments')
    keys = {
        (
            item.first_mode,
            item.second_mode,
            item.third_mode,
            item.first_frequency_sign,
            item.second_frequency_sign,
            item.third_frequency_sign,
        )
        for item in vertices
    }
    if len(keys) != 64:
        raise ValueError('the signed cubic assignment keys must be unique')
    if not all(np.isfinite(item.value) for item in vertices):
        raise ValueError('the signed cubic vertices must be finite')


def _normal_ordered_counts(
    vertex: FrozenCubicVertex,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    creation = [0, 0, 0, 0]
    annihilation = [0, 0, 0, 0]
    legs = (
        (vertex.first_mode, vertex.first_frequency_sign),
        (vertex.second_mode, vertex.second_frequency_sign),
        (2 + vertex.third_mode, vertex.third_frequency_sign),
    )
    for mode, sign in legs:
        if sign == -1:
            creation[mode] += 1
        elif sign == 1:
            annihilation[mode] += 1
        else:
            raise ValueError('a signed cubic vertex must use signs +/-1')
    return tuple(creation), tuple(annihilation)


def _apply_normal_ordered_counts(
    source: FockState,
    creation: tuple[int, ...],
    annihilation: tuple[int, ...],
) -> tuple[FockState, float] | None:
    target = []
    factor = 1.0
    for occupation, create_count, annihilate_count in zip(
        source,
        creation,
        annihilation,
        strict=True,
    ):
        if occupation < annihilate_count:
            return None
        remaining = occupation - annihilate_count
        final = remaining + create_count
        factor *= np.sqrt(
            factorial(occupation)
            * factorial(final)
        ) / factorial(remaining)
        target.append(final)
    return tuple(target), float(factor)


def normal_ordered_cubic_transitions(
    vertices: tuple[FrozenCubicVertex, ...],
    source: FockState,
    *,
    taylor_factor: float = 0.5,
) -> tuple[NormalOrderedFockTransition, ...]:
    '''Sum every signed monomial after explicit normal ordering.'''

    _validate_vertices(vertices)
    if len(source) != 4 or any(
        not isinstance(value, int) or value < 0 for value in source
    ):
        raise ValueError('a Fock source must contain four nonnegative integers')
    if not np.isfinite(taylor_factor):
        raise ValueError('the cubic Taylor factor must be finite')
    amplitudes: dict[FockState, complex] = {}
    counts: dict[FockState, int] = {}
    for vertex in vertices:
        creation, annihilation = _normal_ordered_counts(vertex)
        applied = _apply_normal_ordered_counts(
            source,
            creation,
            annihilation,
        )
        if applied is None:
            continue
        target, fock_factor = applied
        amplitudes[target] = amplitudes.get(target, 0.0j) + (
            float(taylor_factor) * vertex.value * fock_factor
        )
        counts[target] = counts.get(target, 0) + 1
    return tuple(
        NormalOrderedFockTransition(
            target=target,
            matrix_element=complex(amplitudes[target]),
            contributing_monomial_count=counts[target],
        )
        for target in sorted(amplitudes)
    )


def expected_one_insertion_states(initial_branch: int) -> tuple[FockState, ...]:
    '''Return the algebraic 12-state closure of one signed H3 insertion.'''

    if initial_branch == 0:
        first_harmonic = (
            (0, 0),
            (1, 1),
            (2, 0),
            (2, 2),
            (3, 1),
            (4, 0),
        )
    elif initial_branch == 1:
        first_harmonic = (
            (0, 0),
            (0, 2),
            (0, 4),
            (1, 1),
            (1, 3),
            (2, 2),
        )
    else:
        raise ValueError('the initial scalar branch must be zero or one')
    states = []
    for first, second in first_harmonic:
        states.append((first, second, 1, 0))
        states.append((first, second, 0, 1))
    return tuple(sorted(states))


def _single_mode_annihilation(maximum_occupation: int) -> np.ndarray:
    dimension = maximum_occupation + 1
    operator = np.zeros((dimension, dimension), dtype=complex)
    for occupation in range(1, dimension):
        operator[occupation - 1, occupation] = np.sqrt(occupation)
    return operator


def explicit_normal_ordered_cubic_matrix(
    vertices: tuple[FrozenCubicVertex, ...],
    maximum_occupations: FockState = (4, 4, 1, 1),
    *,
    taylor_factor: float = 0.5,
) -> np.ndarray:
    '''Build the independent tensor-product matrix of the normal-ordered H3.'''

    _validate_vertices(vertices)
    if len(maximum_occupations) != 4 or any(
        not isinstance(value, int) or value < 1
        for value in maximum_occupations
    ):
        raise ValueError('the explicit Fock caps must be four positive integers')
    dimensions = tuple(value + 1 for value in maximum_occupations)
    total_dimension = int(np.prod(dimensions))
    hamiltonian = np.zeros(
        (total_dimension, total_dimension),
        dtype=complex,
    )
    annihilations = tuple(
        _single_mode_annihilation(value)
        for value in maximum_occupations
    )
    identities = tuple(np.eye(value, dtype=complex) for value in dimensions)
    for vertex in vertices:
        creation, annihilation = _normal_ordered_counts(vertex)
        factors = []
        for mode in range(4):
            local = (
                np.linalg.matrix_power(
                    annihilations[mode].conj().T,
                    creation[mode],
                )
                @ np.linalg.matrix_power(
                    annihilations[mode],
                    annihilation[mode],
                )
            )
            factors.append(local if creation[mode] + annihilation[mode] else identities[mode])
        monomial = factors[0]
        for factor in factors[1:]:
            monomial = np.kron(monomial, factor)
        hamiltonian += float(taylor_factor) * vertex.value * monomial
    return hamiltonian


def _fock_index(
    state: FockState,
    maximum_occupations: FockState = (4, 4, 1, 1),
) -> int:
    return int(
        np.ravel_multi_index(
            state,
            tuple(value + 1 for value in maximum_occupations),
        )
    )


def _raw_ordered_cubic_transitions(
    vertices: tuple[FrozenCubicVertex, ...],
    source: FockState,
) -> dict[FockState, complex]:
    '''Negative control: retain the written leg order instead of normal ordering.'''

    amplitudes: dict[FockState, complex] = {}
    for vertex in vertices:
        legs = (
            (vertex.first_mode, vertex.first_frequency_sign),
            (vertex.second_mode, vertex.second_frequency_sign),
            (2 + vertex.third_mode, vertex.third_frequency_sign),
        )
        state = list(source)
        factor = 1.0
        allowed = True
        for mode, sign in reversed(legs):
            if sign == 1:
                if state[mode] == 0:
                    allowed = False
                    break
                factor *= np.sqrt(state[mode])
                state[mode] -= 1
            else:
                factor *= np.sqrt(state[mode] + 1)
                state[mode] += 1
        if allowed:
            target = tuple(state)
            amplitudes[target] = amplitudes.get(target, 0.0j) + (
                0.5 * vertex.value * factor
            )
    return amplitudes


def _exact_star_survival_amplitude(
    initial_energy: float,
    intermediate_energies: tuple[float, ...],
    matrix_elements: tuple[complex, ...],
    contact: float,
    interval_bar: float,
    coupling: float,
) -> tuple[complex, float]:
    dimension = 1 + len(intermediate_energies)
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    hamiltonian[0, 0] = initial_energy + coupling**2 * contact
    for index, (energy, matrix_element) in enumerate(
        zip(intermediate_energies, matrix_elements, strict=True),
        start=1,
    ):
        hamiltonian[index, index] = energy
        hamiltonian[index, 0] = coupling * matrix_element
        hamiltonian[0, index] = coupling * matrix_element.conjugate()
    hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    evolution = (
        eigenvectors
        @ np.diag(np.exp(-1j * eigenvalues * interval_bar))
        @ eigenvectors.conj().T
    )
    interaction = np.exp(1j * initial_energy * interval_bar) * evolution[0, 0]
    return complex(interaction), hermiticity


@dataclass(frozen=True)
class AllSignedFockExchangeBranch:
    branch: int
    source_occupations: FockState
    candidate_intermediate_count: int
    active_intermediate_count: int
    intermediate_occupations: tuple[FockState, ...]
    intermediate_matrix_element_magnitudes: tuple[float, ...]
    rotating_exchange_coefficient_real: float
    rotating_exchange_coefficient_imag: float
    all_signed_exchange_coefficient_real: float
    all_signed_exchange_coefficient_imag: float
    all_signed_minus_rotating_magnitude: float
    finest_exact_normalized_error: float
    lambda_quarter_scaling_residual: float
    all_signed_minus_rotating_to_error_ratio: float
    omitted_target_to_error_ratio: float
    wrong_taylor_factor_to_error_ratio: float
    unordered_contraction_to_error_ratio: float
    reduced_cap_to_error_ratio: float


@dataclass(frozen=True)
class AllSignedFockExchangeReceipt:
    base_wavenumber_bar: float
    phase_points: int
    grid_phase_points: int
    branches: tuple[AllSignedFockExchangeBranch, ...]
    signed_assignment_count: int
    maximum_signed_conjugation_residual: float
    maximum_first_leg_exchange_residual: float
    maximum_vertex_step_residual: float
    maximum_vertex_grid_residual: float
    maximum_vertex_gauge_residual: float
    maximum_explicit_fock_matrix_residual: float
    maximum_projected_h3_hermiticity_residual: float
    maximum_intermediate_parity_residual: float
    maximum_rotating_subset_reproduction_residual: float
    maximum_kernel_quadrature_residual: float
    maximum_kernel_grid_refinement: float
    minimum_nonrotating_energy_gap_magnitude: float
    maximum_finest_exact_normalized_error: float
    maximum_lambda_quarter_scaling_residual: float
    maximum_zero_coupling_residual: float
    minimum_negative_control_to_numerical_error_ratio: float
    quartic_contact_gate_passed: bool
    cubic_resonance_classification_gate_passed: bool
    declared_all_signed_diagonal_exchange_gate_passed: bool


def evaluate_all_signed_fock_exchange_gate(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    interval_bar: float = 0.5,
    cubic_steps: tuple[float, float] = (5.0e-3, 2.5e-3),
    phase_points: int = 1024,
    grid_phase_points: int = 2048,
    simpson_subintervals: tuple[int, int] = (512, 1024),
    couplings: tuple[float, float, float] = (1.0, 0.5, 0.25),
) -> AllSignedFockExchangeReceipt:
    '''Run the preregistered all-signed diagonal finite-Fock exchange gate.'''

    if not np.allclose(
        np.asarray(cubic_steps[1:]),
        0.5 * np.asarray(cubic_steps[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the cubic steps must be successive halvings')
    if simpson_subintervals[1] != 2 * simpson_subintervals[0]:
        raise ValueError('the ordered Simpson grids must differ by two')
    if not np.allclose(
        np.asarray(couplings[1:]),
        0.5 * np.asarray(couplings[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the star couplings must be successive halvings')

    contact_receipt = evaluate_scalar_quartic_contact_gate(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        phase_points=phase_points,
        grid_phase_points=grid_phase_points,
    )
    resonance_receipt = evaluate_scalar_cubic_exchange_gate(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        phase_points=phase_points,
        grid_phase_points=grid_phase_points,
        interval_bar=interval_bar,
    )
    certificate_lookup = {
        item.key: item for item in resonance_receipt.resonance_certificates
    }
    lagrangian_pairs = tuple(
        dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
        )
        for step in cubic_steps
    )
    hamiltonian_pairs = tuple(
        scalar_interaction_hamiltonian_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            flat_lagrangian_tensor=pair[0],
            unitary_lagrangian_tensor=pair[1],
        )
        for pair in lagrangian_pairs
    )
    grid_lagrangian = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=cubic_steps[-1],
        phase_points=grid_phase_points,
    )
    grid_hamiltonian = scalar_interaction_hamiltonian_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_lagrangian_tensor=grid_lagrangian[0],
        unitary_lagrangian_tensor=grid_lagrangian[1],
    )
    first_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=base_wavenumber_bar,
    )
    second_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base_wavenumber_bar,
    )

    def project(tensor: np.ndarray) -> tuple[FrozenCubicVertex, ...]:
        return project_frozen_scalar_hamiltonian_vertices(
            tensor,
            first_modes,
            second_modes,
        )

    coarse_vertices = project(hamiltonian_pairs[0][0])
    fine_vertices = project(hamiltonian_pairs[1][0])
    unitary_vertices = project(hamiltonian_pairs[1][1])
    grid_vertices = project(grid_hamiltonian[0])
    _validate_vertices(fine_vertices)

    def vertex_lookup(
        vertices: tuple[FrozenCubicVertex, ...],
    ) -> dict[tuple[int, int, int, int, int, int], complex]:
        return {
            (
                item.first_mode,
                item.second_mode,
                item.third_mode,
                item.first_frequency_sign,
                item.second_frequency_sign,
                item.third_frequency_sign,
            ): item.value
            for item in vertices
        }

    fine_vertex_lookup = vertex_lookup(fine_vertices)
    conjugation = 0.0
    first_leg_exchange = 0.0
    for key, value in fine_vertex_lookup.items():
        conjugation = max(
            conjugation,
            abs(fine_vertex_lookup[(*key[:3], -key[3], -key[4], -key[5])] - value.conjugate()),
        )
        swapped = (
            key[1],
            key[0],
            key[2],
            key[4],
            key[3],
            key[5],
        )
        first_leg_exchange = max(
            first_leg_exchange,
            abs(fine_vertex_lookup[swapped] - value),
        )

    explicit_matrix = explicit_normal_ordered_cubic_matrix(fine_vertices)
    explicit_hermiticity = float(
        np.max(np.abs(explicit_matrix - explicit_matrix.conj().T))
    )
    rotating_channels = scalar_cubic_exchange_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-1][0],
        interval_bar=interval_bar,
    )
    rotating_lookup = {
        (item.first_branch, item.second_branch, item.third_branch): item
        for item in rotating_channels
    }
    frequencies = (
        first_modes[0].frequency_bar,
        first_modes[1].frequency_bar,
        second_modes[0].frequency_bar,
        second_modes[1].frequency_bar,
    )

    maximum_step = 0.0
    maximum_grid = 0.0
    maximum_gauge = 0.0
    maximum_fock = 0.0
    maximum_parity = 0.0
    maximum_rwa = 0.0
    maximum_kernel = 0.0
    maximum_kernel_grid = 0.0
    minimum_nonrotating_gap = np.inf
    maximum_exact = 0.0
    maximum_scaling = 0.0
    maximum_zero_coupling = 0.0
    minimum_control_ratio = np.inf
    branch_receipts = []
    for branch, contact_branch in enumerate(contact_receipt.branches):
        source: FockState = (
            (2, 0, 0, 0) if branch == 0 else (0, 2, 0, 0)
        )
        fine_items = normal_ordered_cubic_transitions(
            fine_vertices,
            source,
        )
        coarse_items = normal_ordered_cubic_transitions(
            coarse_vertices,
            source,
        )
        unitary_items = normal_ordered_cubic_transitions(
            unitary_vertices,
            source,
        )
        grid_items = normal_ordered_cubic_transitions(
            grid_vertices,
            source,
        )

        def transition_lookup(
            items: tuple[NormalOrderedFockTransition, ...],
        ) -> dict[FockState, complex]:
            return {item.target: item.matrix_element for item in items}

        fine = transition_lookup(fine_items)
        coarse = transition_lookup(coarse_items)
        unitary = transition_lookup(unitary_items)
        grid = transition_lookup(grid_items)
        expected = set(expected_one_insertion_states(branch))
        if any(set(items) != expected for items in (fine, coarse, unitary, grid)):
            raise ValueError('the signed cubic reachable-state closure changed')

        raw_fine = dict(fine)
        rotating_targets = ((0, 0, 1, 0), (0, 0, 0, 1))
        for third_branch, target in enumerate(rotating_targets):
            key = (branch, branch, third_branch)
            fine[target] = select_certified_resonant_matrix_element(
                key,
                fine[target],
                certificate_lookup,
            ) if rotating_lookup[key].resonant else fine[target]
            expected_rotating = (
                select_certified_resonant_matrix_element(
                    key,
                    rotating_lookup[key].matrix_element,
                    certificate_lookup,
                )
                if rotating_lookup[key].resonant
                else rotating_lookup[key].matrix_element
            )
            maximum_rwa = max(
                maximum_rwa,
                abs(fine[target] - expected_rotating),
            )

        local_step = max(abs(raw_fine[key] - coarse[key]) for key in expected)
        local_grid = max(abs(raw_fine[key] - grid[key]) for key in expected)
        local_gauge = max(abs(raw_fine[key] - unitary[key]) for key in expected)
        maximum_step = max(maximum_step, local_step)
        maximum_grid = max(maximum_grid, local_grid)
        maximum_gauge = max(maximum_gauge, local_gauge)

        source_index = _fock_index(source)
        local_fock = 0.0
        for target in expected:
            local_fock = max(
                local_fock,
                abs(
                    raw_fine[target]
                    - explicit_matrix[_fock_index(target), source_index]
                ),
            )
        maximum_fock = max(maximum_fock, local_fock)
        intermediate_indices = [_fock_index(target) for target in expected]
        intermediate_block = explicit_matrix[
            np.ix_(intermediate_indices, intermediate_indices)
        ]
        local_parity = float(np.max(np.abs(intermediate_block)))
        maximum_parity = max(maximum_parity, local_parity)

        initial_energy = sum(
            occupation * frequency
            for occupation, frequency in zip(source, frequencies, strict=True)
        )
        intermediate_states = tuple(sorted(expected))
        intermediate_energies = []
        matrix_elements = []
        exchange_terms: dict[FockState, complex] = {}
        local_kernel = 0.0
        local_kernel_grid = 0.0
        for target in intermediate_states:
            energy = sum(
                occupation * frequency
                for occupation, frequency in zip(target, frequencies, strict=True)
            )
            gap = initial_energy - energy
            if target not in rotating_targets:
                minimum_nonrotating_gap = min(
                    minimum_nonrotating_gap,
                    abs(gap),
                )
                if abs(gap) <= RESONANCE_TOL:
                    raise ValueError(
                        'a nonrotating resonant target lacks a '
                        'branch-keyed certificate'
                    )
            kernel = finite_time_ordered_double_kernel(
                gap,
                -gap,
                interval_bar,
            )
            coarse_kernel = simpson_ordered_double_kernel(
                gap,
                -gap,
                interval_bar,
                subintervals=simpson_subintervals[0],
            )
            fine_kernel = simpson_ordered_double_kernel(
                gap,
                -gap,
                interval_bar,
                subintervals=simpson_subintervals[1],
            )
            local_kernel = max(local_kernel, abs(kernel - fine_kernel))
            local_kernel_grid = max(
                local_kernel_grid,
                abs(fine_kernel - coarse_kernel),
            )
            intermediate_energies.append(float(energy))
            matrix_elements.append(complex(fine[target]))
            exchange_terms[target] = -abs(fine[target]) ** 2 * kernel
        maximum_kernel = max(maximum_kernel, local_kernel)
        maximum_kernel_grid = max(
            maximum_kernel_grid,
            local_kernel_grid,
        )

        rotating_exchange = sum(
            (exchange_terms[target] for target in rotating_targets),
            0.0j,
        )
        all_signed_exchange = sum(exchange_terms.values(), 0.0j)
        full_minus_rotating = abs(all_signed_exchange - rotating_exchange)
        contact = float(contact_branch.flat_analytic_contact)
        total_coefficient = (
            -1j * contact * interval_bar + all_signed_exchange
        )
        exact_errors = []
        local_star_hermiticity = 0.0
        for coupling in couplings:
            exact, hermiticity = _exact_star_survival_amplitude(
                float(initial_energy),
                tuple(intermediate_energies),
                tuple(matrix_elements),
                contact,
                interval_bar,
                coupling,
            )
            local_star_hermiticity = max(
                local_star_hermiticity,
                hermiticity,
            )
            exact_errors.append(
                abs((exact - 1.0) / coupling**2 - total_coefficient)
            )
        scaling = max(
            abs(exact_errors[1] / max(exact_errors[0], 1.0e-300) - 0.25),
            abs(exact_errors[2] / max(exact_errors[1], 1.0e-300) - 0.25),
        )
        maximum_exact = max(maximum_exact, exact_errors[-1])
        maximum_scaling = max(maximum_scaling, scaling)
        zero_coupling, _ = _exact_star_survival_amplitude(
            float(initial_energy),
            tuple(intermediate_energies),
            tuple(matrix_elements),
            contact,
            interval_bar,
            0.0,
        )
        maximum_zero_coupling = max(
            maximum_zero_coupling,
            abs(zero_coupling - 1.0),
        )

        contact_error = interval_bar * max(
            contact_receipt.maximum_step_residual,
            contact_receipt.maximum_grid_residual,
            contact_receipt.maximum_gauge_residual,
            contact_receipt.maximum_analytic_direct_relative_residual
            * abs(contact),
        )
        cubic_scale = max(abs(value) for value in matrix_elements)
        numerical_error = max(
            contact_error,
            interval_bar**2
            * cubic_scale
            * max(local_step, local_grid, local_gauge),
            local_kernel
            * sum(abs(value) ** 2 for value in matrix_elements),
            local_fock,
            explicit_hermiticity,
            local_star_hermiticity,
            1.0e-30,
        )
        omitted_target = max(
            abs(value) for value in exchange_terms.values()
        )

        wrong_factor = transition_lookup(
            normal_ordered_cubic_transitions(
                fine_vertices,
                source,
                taylor_factor=1.0,
            )
        )
        wrong_factor_exchange = 0.0j
        unordered = _raw_ordered_cubic_transitions(fine_vertices, source)
        unordered_exchange = 0.0j
        reduced_cap_exchange = 0.0j
        for target, correct_term in exchange_terms.items():
            energy = sum(
                occupation * frequency
                for occupation, frequency in zip(target, frequencies, strict=True)
            )
            gap = initial_energy - energy
            kernel = finite_time_ordered_double_kernel(
                gap,
                -gap,
                interval_bar,
            )
            wrong_factor_exchange += -abs(wrong_factor[target]) ** 2 * kernel
            unordered_exchange += -abs(unordered[target]) ** 2 * kernel
            if target[0] <= 2 and target[1] <= 2:
                reduced_cap_exchange += correct_term
        wrong_factor_control = abs(
            wrong_factor_exchange - all_signed_exchange
        )
        unordered_control = abs(unordered_exchange - all_signed_exchange)
        reduced_cap_control = abs(
            reduced_cap_exchange - all_signed_exchange
        )
        ratios = (
            full_minus_rotating / numerical_error,
            omitted_target / numerical_error,
            wrong_factor_control / numerical_error,
            unordered_control / numerical_error,
            reduced_cap_control / numerical_error,
        )
        minimum_control_ratio = min(minimum_control_ratio, *ratios)
        active_count = sum(
            abs(fine[target]) > 10.0 * max(
                abs(raw_fine[target] - coarse[target]),
                abs(raw_fine[target] - grid[target]),
                abs(raw_fine[target] - unitary[target]),
                1.0e-30,
            )
            for target in intermediate_states
        )
        branch_receipts.append(
            AllSignedFockExchangeBranch(
                branch=branch,
                source_occupations=source,
                candidate_intermediate_count=len(intermediate_states),
                active_intermediate_count=int(active_count),
                intermediate_occupations=intermediate_states,
                intermediate_matrix_element_magnitudes=tuple(
                    float(abs(value)) for value in matrix_elements
                ),
                rotating_exchange_coefficient_real=float(
                    rotating_exchange.real
                ),
                rotating_exchange_coefficient_imag=float(
                    rotating_exchange.imag
                ),
                all_signed_exchange_coefficient_real=float(
                    all_signed_exchange.real
                ),
                all_signed_exchange_coefficient_imag=float(
                    all_signed_exchange.imag
                ),
                all_signed_minus_rotating_magnitude=float(
                    full_minus_rotating
                ),
                finest_exact_normalized_error=float(exact_errors[-1]),
                lambda_quarter_scaling_residual=float(scaling),
                all_signed_minus_rotating_to_error_ratio=float(ratios[0]),
                omitted_target_to_error_ratio=float(ratios[1]),
                wrong_taylor_factor_to_error_ratio=float(ratios[2]),
                unordered_contraction_to_error_ratio=float(ratios[3]),
                reduced_cap_to_error_ratio=float(ratios[4]),
            )
        )

    passed = bool(
        contact_receipt.declared_quartic_contact_gate_passed
        and resonance_receipt.declared_exchange_gate_passed
        and len(fine_vertices) == 64
        and all(
            item.candidate_intermediate_count == 12
            and item.active_intermediate_count == 4
            for item in branch_receipts
        )
        and conjugation < 1.0e-10
        and first_leg_exchange < 1.0e-10
        and maximum_step < 2.0e-4
        and maximum_grid < 1.0e-8
        and maximum_gauge < 1.0e-6
        and maximum_fock < 1.0e-12
        and explicit_hermiticity < 1.0e-10
        and maximum_parity < 1.0e-12
        and maximum_rwa < 1.0e-12
        and maximum_kernel < 1.0e-10
        and maximum_kernel_grid < 1.0e-10
        and minimum_nonrotating_gap > 100.0 * RESONANCE_TOL
        and maximum_exact < 1.0e-4
        and maximum_scaling < 0.1
        and maximum_zero_coupling < 1.0e-12
        and minimum_control_ratio > 10.0
    )
    return AllSignedFockExchangeReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        phase_points=phase_points,
        grid_phase_points=grid_phase_points,
        branches=tuple(branch_receipts),
        signed_assignment_count=len(fine_vertices),
        maximum_signed_conjugation_residual=float(conjugation),
        maximum_first_leg_exchange_residual=float(first_leg_exchange),
        maximum_vertex_step_residual=float(maximum_step),
        maximum_vertex_grid_residual=float(maximum_grid),
        maximum_vertex_gauge_residual=float(maximum_gauge),
        maximum_explicit_fock_matrix_residual=float(maximum_fock),
        maximum_projected_h3_hermiticity_residual=float(
            explicit_hermiticity
        ),
        maximum_intermediate_parity_residual=float(maximum_parity),
        maximum_rotating_subset_reproduction_residual=float(maximum_rwa),
        maximum_kernel_quadrature_residual=float(maximum_kernel),
        maximum_kernel_grid_refinement=float(maximum_kernel_grid),
        minimum_nonrotating_energy_gap_magnitude=float(
            minimum_nonrotating_gap
        ),
        maximum_finest_exact_normalized_error=float(maximum_exact),
        maximum_lambda_quarter_scaling_residual=float(maximum_scaling),
        maximum_zero_coupling_residual=float(maximum_zero_coupling),
        minimum_negative_control_to_numerical_error_ratio=float(
            minimum_control_ratio
        ),
        quartic_contact_gate_passed=(
            contact_receipt.declared_quartic_contact_gate_passed
        ),
        cubic_resonance_classification_gate_passed=(
            resonance_receipt.declared_exchange_gate_passed
        ),
        declared_all_signed_diagonal_exchange_gate_passed=passed,
    )
