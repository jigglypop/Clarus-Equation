'''Frozen scalar quartic contact for the next E68 admission subgate.

The module extracts the normal-ordered diagonal contact on ``|2_k,a>`` from
the exact projected scalar action.  It compares a direct finite-amplitude
Legendre transform with the quartic Legendre formula

    H4 = a^3[-ell4(v0) + 1/2 d_v ell3 K^{-1} d_v ell3].

Only the declared one-harmonic contact and the induced k+k -> 2k velocity
piece are included.  This is not a full quartic Hamiltonian or a cutoff.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    constraint_solved_scalar_phase_point,
    dynamic_reduced_scalar_cubic_tensor_pair,
    frozen_symplectic_scalar_modes,
    harmonic_scalar_phase_space_map,
    solve_direct_scalar_legendre_point,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


ArrayValue = Callable[[np.ndarray], np.ndarray]


def _fourth_mode_projection(
    value: ArrayValue,
    real_direction: np.ndarray,
    imaginary_direction: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    '''Return Q(u,u,u*,u*) for u=A+iB from real finite differences.'''

    step = np.longdouble(epsilon)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError('the quartic finite-difference step must be positive')
    first = np.asarray(real_direction, dtype=float)
    second = np.asarray(imaginary_direction, dtype=float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError('quartic mode directions must be four-vectors')

    def pure(direction: np.ndarray) -> np.ndarray:
        return (
            value(2.0 * float(step) * direction)
            - 4.0 * value(float(step) * direction)
            + 6.0 * value(np.zeros(4))
            - 4.0 * value(-float(step) * direction)
            + value(-2.0 * float(step) * direction)
        ) / step**4

    weights = ((-1.0, 1.0), (0.0, -2.0), (1.0, 1.0))
    mixed = None
    for first_offset, first_weight in weights:
        for second_offset, second_weight in weights:
            sample = value(
                float(step)
                * (first_offset * first + second_offset * second)
            )
            contribution = first_weight * second_weight * sample
            mixed = contribution if mixed is None else mixed + contribution
    if mixed is None:
        raise RuntimeError('the quartic mixed stencil did not evaluate')
    mixed = mixed / step**4
    return pure(first) + 2.0 * mixed + pure(second)


def _normal_ordered_two_particle_contact(
    fourth_derivative: np.ndarray,
) -> np.ndarray:
    '''Map Q(u,u,u*,u*) to <2|:H4:|2> = Q/2.'''

    return np.asarray(fourth_derivative, dtype=np.longdouble) / 2.0


def _fock_factor_residual(contact: float, fourth_derivative: float) -> float:
    '''Check the declared normal-ordered 2!/(2!2!) factorial convention.'''

    dimension = 5
    annihilation = np.zeros((dimension, dimension))
    for occupation in range(1, dimension):
        annihilation[occupation - 1, occupation] = np.sqrt(occupation)
    creation = annihilation.T
    operator = (
        float(fourth_derivative)
        / 4.0
        * creation
        @ creation
        @ annihilation
        @ annihilation
    )
    direct = float(operator[2, 2])
    return abs(float(contact) - direct)


@dataclass(frozen=True)
class ScalarQuarticContactBranch:
    branch: int
    frequency_bar: float
    flat_bare_contact: float
    flat_induced_legendre_contact: float
    flat_analytic_contact: float
    flat_direct_contact: float
    unitary_bare_contact: float
    unitary_induced_legendre_contact: float
    unitary_analytic_contact: float
    unitary_direct_contact: float
    maximum_fock_factor_residual: float


@dataclass(frozen=True)
class ScalarQuarticContactEvaluation:
    base_wavenumber_bar: float
    epsilon: float
    phase_points: int
    branches: tuple[ScalarQuarticContactBranch, ...]
    maximum_momentum_residual: float
    maximum_constraint_residual: float


@dataclass(frozen=True)
class ScalarQuarticContactGateReceipt:
    base_wavenumber_bar: float
    branches: tuple[ScalarQuarticContactBranch, ...]
    maximum_analytic_direct_relative_residual: float
    maximum_step_residual: float
    maximum_grid_residual: float
    maximum_gauge_residual: float
    minimum_signal_to_error_ratio: float
    maximum_fock_factor_residual: float
    minimum_induced_legendre_omission_ratio: float
    minimum_wrong_induced_sign_ratio: float
    maximum_momentum_residual: float
    maximum_constraint_residual: float
    declared_quartic_contact_gate_passed: bool


def scalar_quartic_contact_evaluation(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    epsilon: float,
    phase_points: int = 256,
    cubic_epsilon: float = 2.5e-3,
    lagrangian_cubic_tensor_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> ScalarQuarticContactEvaluation:
    '''Extract the two diagonal normal-ordered scalar quartic contacts.'''

    base = float(base_wavenumber_bar)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the quartic contact requires positive base momentum')
    if not isinstance(phase_points, int) or phase_points < 32:
        raise ValueError('phase_points must be an integer of at least thirty-two')
    if lagrangian_cubic_tensor_pair is None:
        cubic_pair = dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base,
            epsilon=cubic_epsilon,
            phase_points=phase_points,
        )
    else:
        cubic_pair = tuple(
            np.asarray(tensor, dtype=float)
            for tensor in lagrangian_cubic_tensor_pair
        )
    if len(cubic_pair) != 2 or any(
        tensor.shape != (4, 4, 4) for tensor in cubic_pair
    ):
        raise ValueError('the quartic contact requires two 4x4x4 cubic tensors')

    first_map = harmonic_scalar_phase_space_map(
        state,
        parameters,
        comoving_wavenumber_bar=base,
    ).matrix
    second_blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base,
    )
    second_kinetic, _, _ = reduced_scalar_matrices(second_blocks)
    inverse_second_kinetic = np.linalg.inv(second_kinetic)
    modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=base,
    )
    a3 = float(np.exp(3.0 * state.n))

    lagrangian_cache: dict[tuple[float, ...], np.ndarray] = {}
    hamiltonian_cache: dict[tuple[float, ...], np.ndarray] = {}
    maximum_momentum_residual = 0.0
    maximum_constraint_residual = 0.0

    def lagrangian_value(canonical_first: np.ndarray) -> np.ndarray:
        nonlocal maximum_constraint_residual
        key = tuple(float(item) for item in canonical_first)
        if key not in lagrangian_cache:
            physical = np.zeros((2, 4))
            physical[0] = first_map @ canonical_first
            point = constraint_solved_scalar_phase_point(
                state,
                parameters,
                base_wavenumber_bar=base,
                physical_modes=physical,
                phase_points=phase_points,
            )
            maximum_constraint_residual = max(
                maximum_constraint_residual,
                point.maximum_constraint_residual,
            )
            lagrangian_cache[key] = np.array(
                [
                    point.flat_lagrangian_bar_per_a3,
                    point.unitary_lagrangian_bar_per_a3,
                ],
                dtype=np.longdouble,
            )
        return lagrangian_cache[key]

    def hamiltonian_value(canonical_first: np.ndarray) -> np.ndarray:
        nonlocal maximum_momentum_residual
        nonlocal maximum_constraint_residual
        key = tuple(float(item) for item in canonical_first)
        if key not in hamiltonian_cache:
            canonical = np.zeros((2, 4))
            canonical[0] = canonical_first
            point = solve_direct_scalar_legendre_point(
                state,
                parameters,
                base_wavenumber_bar=base,
                canonical_modes=canonical,
                phase_points=phase_points,
            )
            maximum_momentum_residual = max(
                maximum_momentum_residual,
                point.maximum_momentum_residual,
            )
            maximum_constraint_residual = max(
                maximum_constraint_residual,
                point.maximum_constraint_residual,
            )
            hamiltonian_cache[key] = np.array(
                [
                    point.flat_interaction_hamiltonian_bar,
                    point.unitary_interaction_hamiltonian_bar,
                ],
                dtype=np.longdouble,
            )
        return hamiltonian_cache[key]

    def induced_value(canonical_first: np.ndarray) -> np.ndarray:
        jet = first_map @ canonical_first
        values = []
        for tensor in cubic_pair:
            velocity_gradient = 0.5 * np.einsum(
                'ijl,i,j->l',
                tensor[:, :, :2],
                jet,
                jet,
            )
            values.append(
                0.5
                * a3
                * velocity_gradient
                @ inverse_second_kinetic
                @ velocity_gradient
            )
        return np.asarray(values, dtype=np.longdouble)

    branches = []
    for branch, mode in enumerate(modes):
        phase_direction = np.concatenate((mode.momentum, mode.configuration))
        real_direction = np.asarray(phase_direction.real, dtype=float)
        imaginary_direction = np.asarray(phase_direction.imag, dtype=float)
        lagrangian_fourth = _fourth_mode_projection(
            lagrangian_value,
            real_direction,
            imaginary_direction,
            epsilon=epsilon,
        )
        induced_fourth = _fourth_mode_projection(
            induced_value,
            real_direction,
            imaginary_direction,
            epsilon=epsilon,
        )
        direct_fourth = _fourth_mode_projection(
            hamiltonian_value,
            real_direction,
            imaginary_direction,
            epsilon=epsilon,
        )
        bare_contacts = _normal_ordered_two_particle_contact(
            -a3 * lagrangian_fourth
        )
        induced_contacts = _normal_ordered_two_particle_contact(
            induced_fourth
        )
        analytic_contacts = bare_contacts + induced_contacts
        direct_contacts = _normal_ordered_two_particle_contact(direct_fourth)
        fock_residual = max(
            _fock_factor_residual(
                float(analytic_contacts[index]),
                float(2.0 * analytic_contacts[index]),
            )
            for index in range(2)
        )
        branches.append(
            ScalarQuarticContactBranch(
                branch=branch,
                frequency_bar=mode.frequency_bar,
                flat_bare_contact=float(bare_contacts[0]),
                flat_induced_legendre_contact=float(induced_contacts[0]),
                flat_analytic_contact=float(analytic_contacts[0]),
                flat_direct_contact=float(direct_contacts[0]),
                unitary_bare_contact=float(bare_contacts[1]),
                unitary_induced_legendre_contact=float(induced_contacts[1]),
                unitary_analytic_contact=float(analytic_contacts[1]),
                unitary_direct_contact=float(direct_contacts[1]),
                maximum_fock_factor_residual=float(fock_residual),
            )
        )
    return ScalarQuarticContactEvaluation(
        base_wavenumber_bar=base,
        epsilon=float(epsilon),
        phase_points=phase_points,
        branches=tuple(branches),
        maximum_momentum_residual=maximum_momentum_residual,
        maximum_constraint_residual=maximum_constraint_residual,
    )


def evaluate_scalar_quartic_contact_gate(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    quartic_steps: tuple[float, ...] = (8.0e-2, 4.0e-2, 2.0e-2),
    cubic_epsilon: float = 2.5e-3,
    phase_points: int = 256,
    grid_phase_points: int = 512,
) -> ScalarQuarticContactGateReceipt:
    '''Run the preregistered direct/analytic scalar quartic contact gate.'''

    if len(quartic_steps) != 3 or not np.allclose(
        np.asarray(quartic_steps[1:]),
        0.5 * np.asarray(quartic_steps[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the quartic gate requires three successive halvings')
    if grid_phase_points <= phase_points:
        raise ValueError('the quartic grid refinement must increase resolution')
    cubic_pair = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=cubic_epsilon,
        phase_points=phase_points,
    )
    grid_cubic_pair = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=cubic_epsilon,
        phase_points=grid_phase_points,
    )
    evaluations = tuple(
        scalar_quartic_contact_evaluation(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
            cubic_epsilon=cubic_epsilon,
            lagrangian_cubic_tensor_pair=cubic_pair,
        )
        for step in quartic_steps
    )
    grid_evaluations = tuple(
        scalar_quartic_contact_evaluation(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=grid_phase_points,
            cubic_epsilon=cubic_epsilon,
            lagrangian_cubic_tensor_pair=grid_cubic_pair,
        )
        for step in quartic_steps
    )

    def richardson(values: tuple[float, float, float]) -> tuple[float, float]:
        coarse = (4.0 * values[1] - values[0]) / 3.0
        fine = (4.0 * values[2] - values[1]) / 3.0
        extrapolated = (16.0 * fine - coarse) / 15.0
        return extrapolated, abs(extrapolated - fine)

    def branch_values(
        source: tuple[ScalarQuarticContactEvaluation, ...],
        branch_index: int,
        attribute: str,
    ) -> tuple[float, float, float]:
        return tuple(
            float(getattr(item.branches[branch_index], attribute))
            for item in source
        )

    analytic_direct_relative = 0.0
    step_residual = 0.0
    grid_residual = 0.0
    gauge_residual = 0.0
    minimum_signal_to_error = np.inf
    maximum_fock_residual = 0.0
    minimum_omission_ratio = np.inf
    minimum_wrong_sign_ratio = np.inf
    extrapolated_branches = []
    attributes = (
        'flat_bare_contact',
        'flat_induced_legendre_contact',
        'flat_analytic_contact',
        'flat_direct_contact',
        'unitary_bare_contact',
        'unitary_induced_legendre_contact',
        'unitary_analytic_contact',
        'unitary_direct_contact',
    )
    for branch_index in range(len(evaluations[-1].branches)):
        estimates: dict[str, float] = {}
        stabilities: dict[str, float] = {}
        grid_estimates: dict[str, float] = {}
        for attribute in attributes:
            estimates[attribute], stabilities[attribute] = richardson(
                branch_values(evaluations, branch_index, attribute)
            )
            grid_estimates[attribute], _ = richardson(
                branch_values(grid_evaluations, branch_index, attribute)
            )
        source_branch = evaluations[-1].branches[branch_index]
        branch = ScalarQuarticContactBranch(
            branch=source_branch.branch,
            frequency_bar=source_branch.frequency_bar,
            flat_bare_contact=estimates['flat_bare_contact'],
            flat_induced_legendre_contact=estimates[
                'flat_induced_legendre_contact'
            ],
            flat_analytic_contact=estimates['flat_analytic_contact'],
            flat_direct_contact=estimates['flat_direct_contact'],
            unitary_bare_contact=estimates['unitary_bare_contact'],
            unitary_induced_legendre_contact=estimates[
                'unitary_induced_legendre_contact'
            ],
            unitary_analytic_contact=estimates[
                'unitary_analytic_contact'
            ],
            unitary_direct_contact=estimates['unitary_direct_contact'],
            maximum_fock_factor_residual=source_branch.maximum_fock_factor_residual,
        )
        extrapolated_branches.append(branch)
        signal = max(
            abs(branch.flat_analytic_contact),
            abs(branch.flat_direct_contact),
            1.0e-30,
        )
        analytic_direct = abs(
            branch.flat_analytic_contact - branch.flat_direct_contact
        )
        local_step = max(
            stabilities['flat_analytic_contact'],
            stabilities['flat_direct_contact'],
        )
        local_grid = max(
            abs(
                branch.flat_analytic_contact
                - grid_estimates['flat_analytic_contact']
            ),
            abs(
                branch.flat_direct_contact
                - grid_estimates['flat_direct_contact']
            ),
        )
        local_gauge = max(
            abs(
                branch.flat_analytic_contact
                - branch.unitary_analytic_contact
            ),
            abs(branch.flat_direct_contact - branch.unitary_direct_contact),
        )
        error_envelope = max(
            analytic_direct,
            local_step,
            local_grid,
            local_gauge,
            1.0e-30,
        )
        omission = abs(branch.flat_induced_legendre_contact)
        wrong_sign = 2.0 * omission
        analytic_direct_relative = max(
            analytic_direct_relative,
            analytic_direct / signal,
        )
        step_residual = max(step_residual, local_step)
        grid_residual = max(grid_residual, local_grid)
        gauge_residual = max(gauge_residual, local_gauge)
        minimum_signal_to_error = min(
            minimum_signal_to_error,
            signal / error_envelope,
        )
        maximum_fock_residual = max(
            maximum_fock_residual,
            branch.maximum_fock_factor_residual,
        )
        minimum_omission_ratio = min(
            minimum_omission_ratio,
            omission / error_envelope,
        )
        minimum_wrong_sign_ratio = min(
            minimum_wrong_sign_ratio,
            wrong_sign / error_envelope,
        )
    maximum_momentum = max(
        evaluation.maximum_momentum_residual
        for evaluation in evaluations + grid_evaluations
    )
    maximum_constraint = max(
        evaluation.maximum_constraint_residual
        for evaluation in evaluations + grid_evaluations
    )
    passed = bool(
        analytic_direct_relative < 2.0e-4
        and step_residual < 2.0e-4
        and grid_residual < 1.0e-8
        and gauge_residual < 1.0e-6
        and minimum_signal_to_error > 10.0
        and maximum_fock_residual < 1.0e-12
        and minimum_omission_ratio > 10.0
        and minimum_wrong_sign_ratio > 10.0
        and maximum_momentum < 2.0e-13
        and maximum_constraint < 1.0e-10
    )
    return ScalarQuarticContactGateReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        branches=tuple(extrapolated_branches),
        maximum_analytic_direct_relative_residual=float(
            analytic_direct_relative
        ),
        maximum_step_residual=float(step_residual),
        maximum_grid_residual=float(grid_residual),
        maximum_gauge_residual=float(gauge_residual),
        minimum_signal_to_error_ratio=float(minimum_signal_to_error),
        maximum_fock_factor_residual=float(maximum_fock_residual),
        minimum_induced_legendre_omission_ratio=float(minimum_omission_ratio),
        minimum_wrong_induced_sign_ratio=float(minimum_wrong_sign_ratio),
        maximum_momentum_residual=float(maximum_momentum),
        maximum_constraint_residual=float(maximum_constraint),
        declared_quartic_contact_gate_passed=passed,
    )
