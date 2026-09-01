'''Frozen scalar cubic EOM-ideal and finite-time boundary audit for E68.

The gate acts on the already admitted k,k,-2k Lagrangian derivative tensor.
It checks that external-leg terms proportional to the E66 quadratic equations
vanish on the KG-normalized right-null modes, while a total time derivative
leaves an explicit endpoint on a finite interval.  It is not an in-in
correlator, an asymptotic equivalence theorem, or a strong-coupling bound.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    FrozenScalarMode,
    dynamic_reduced_scalar_cubic_tensor_pair,
    frozen_symplectic_scalar_modes,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


TOL = 1.0e-10


def frozen_scalar_pencil(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    signed_frequency_bar: float,
    reverse_gyroscopic_sign: bool = False,
) -> np.ndarray:
    '''Return P(omega)=-omega^2 K-i omega(R^T-R)+V.'''

    omega = float(signed_frequency_bar)
    if not np.isfinite(omega):
        raise ValueError('the frozen scalar frequency must be finite')
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks)
    antisymmetric = gyroscopic.T - gyroscopic
    sign = 1.0 if reverse_gyroscopic_sign else -1.0
    return -omega**2 * kinetic + sign * 1j * omega * antisymmetric + potential


def frequency_jet_map(signed_frequency_bar: float) -> np.ndarray:
    '''Map a configuration amplitude to Z=(velocity,configuration).'''

    omega = float(signed_frequency_bar)
    if not np.isfinite(omega):
        raise ValueError('the frequency jet map requires a finite frequency')
    mapping = np.zeros((4, 2), dtype=complex)
    mapping[:2] = -1j * omega * np.eye(2)
    mapping[2:] = np.eye(2)
    return mapping


def frequency_configuration_cubic_tensor(
    tensor: np.ndarray,
    *,
    signed_frequencies_bar: tuple[float, float, float],
    scale_factor: float,
) -> np.ndarray:
    '''Lower a jet derivative tensor to a frequency-dependent configuration tensor.'''

    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (4, 4, 4) or not np.all(np.isfinite(tensor)):
        raise ValueError('the frozen cubic jet tensor must be finite and 4x4x4')
    a = float(scale_factor)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError('the frozen cubic tensor requires a positive scale factor')
    maps = tuple(frequency_jet_map(omega) for omega in signed_frequencies_bar)
    return a**3 * np.einsum(
        'ijk,ia,jb,kc->abc',
        tensor,
        maps[0],
        maps[1],
        maps[2],
    )


def deterministic_eom_deformation_coefficients() -> tuple[np.ndarray, np.ndarray]:
    '''Return fixed nonzero same-k-compatible EOM coefficient tensors.'''

    first = np.array(
        [
            [[0.31, -0.17], [0.23, 0.41]],
            [[-0.29, 0.37], [0.19, -0.11]],
        ],
        dtype=float,
    )
    third = np.array(
        [
            [[0.27, -0.14], [-0.14, 0.35]],
            [[-0.21, 0.32], [0.32, 0.16]],
        ],
        dtype=float,
    )
    first /= np.linalg.norm(first)
    third /= np.linalg.norm(third)
    return first, third


def deterministic_boundary_tensor() -> np.ndarray:
    '''Return a unit-norm B_ab;c symmetric in the repeated k legs.'''

    boundary = np.array(
        [
            [[0.43, -0.22], [0.17, 0.31]],
            [[0.17, 0.31], [-0.28, 0.39]],
        ],
        dtype=float,
    )
    boundary /= np.linalg.norm(boundary)
    return boundary


def eom_exact_configuration_deformation(
    first_pencil: np.ndarray,
    second_pencil: np.ndarray,
    third_pencil: np.ndarray,
    *,
    transpose_pencils: bool = False,
) -> np.ndarray:
    '''Construct Delta C with P_ia A_i... on each external right-null leg.'''

    pencils = tuple(
        np.asarray(item, dtype=complex)
        for item in (first_pencil, second_pencil, third_pencil)
    )
    if any(item.shape != (2, 2) for item in pencils):
        raise ValueError('each scalar EOM pencil must be two-by-two')
    if not all(np.all(np.isfinite(item)) for item in pencils):
        raise ValueError('each scalar EOM pencil must be finite')
    first_coefficient, third_coefficient = (
        deterministic_eom_deformation_coefficients()
    )
    first, second, third = pencils
    if transpose_pencils:
        first = first.T
        second = second.T
        third = third.T
    return (
        np.einsum('ia,ibc->abc', first, first_coefficient)
        + np.einsum('ib,iac->abc', second, first_coefficient)
        + np.einsum('ic,iab->abc', third, third_coefficient)
    )


def _signed_mode(mode: FrozenScalarMode, sign: int) -> tuple[float, np.ndarray]:
    if sign not in (-1, 1):
        raise ValueError('a frozen frequency sign must be -1 or +1')
    if sign == 1:
        return mode.frequency_bar, mode.configuration
    return -mode.frequency_bar, mode.configuration.conj()


def _contract_configuration_tensor(
    tensor: np.ndarray,
    configurations: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> complex:
    return complex(
        np.einsum(
            'abc,a,b,c->',
            tensor,
            configurations[0],
            configurations[1],
            configurations[2],
        )
    )


def _trapezoid_total_derivative(
    amplitude: complex,
    signed_frequency_sum: float,
    interval_bar: float,
    *,
    subintervals: int = 4096,
) -> complex:
    times = np.linspace(0.0, interval_bar, subintervals + 1)
    values = (
        -1j
        * signed_frequency_sum
        * amplitude
        * np.exp(-1j * signed_frequency_sum * times)
    )
    spacing = interval_bar / subintervals
    return complex(
        spacing
        * (0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1])
    )


@dataclass(frozen=True)
class EomBoundaryAssignment:
    key: tuple[int, int, int, int, int, int]
    original_flat_vertex: complex
    original_unitary_vertex: complex
    configuration_map_residual: float
    correct_eom_deformation: complex
    transposed_pencil_deformation: complex
    reversed_gyroscopic_deformation: complex
    non_eom_deformation: complex
    signed_frequency_sum: float
    boundary_endpoint: complex
    boundary_quadrature_residual: float
    maximum_pencil_residual: float


def scalar_eom_boundary_assignments(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    flat_tensor: np.ndarray,
    unitary_tensor: np.ndarray,
    interval_bar: float = 0.5,
) -> tuple[EomBoundaryAssignment, ...]:
    '''Evaluate all 64 branch/frequency assignments for one tensor pair.'''

    base = float(base_wavenumber_bar)
    interval = float(interval_bar)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the EOM/boundary audit requires positive base momentum')
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError('the EOM/boundary audit requires a positive interval')
    first_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=base,
    )
    third_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base,
    )
    boundary = deterministic_boundary_tensor()
    non_eom = 0.1 * boundary
    scale_factor = float(np.exp(state.n))
    assignments = []
    for first_branch in range(2):
        for second_branch in range(2):
            for third_branch in range(2):
                for first_sign in (-1, 1):
                    for second_sign in (-1, 1):
                        for third_sign in (-1, 1):
                            first_frequency, first_configuration = _signed_mode(
                                first_modes[first_branch], first_sign
                            )
                            second_frequency, second_configuration = _signed_mode(
                                first_modes[second_branch], second_sign
                            )
                            third_frequency, third_configuration = _signed_mode(
                                third_modes[third_branch], third_sign
                            )
                            frequencies = (
                                first_frequency,
                                second_frequency,
                                third_frequency,
                            )
                            configurations = (
                                first_configuration,
                                second_configuration,
                                third_configuration,
                            )
                            flat_configuration_tensor = (
                                frequency_configuration_cubic_tensor(
                                    flat_tensor,
                                    signed_frequencies_bar=frequencies,
                                    scale_factor=scale_factor,
                                )
                            )
                            unitary_configuration_tensor = (
                                frequency_configuration_cubic_tensor(
                                    unitary_tensor,
                                    signed_frequencies_bar=frequencies,
                                    scale_factor=scale_factor,
                                )
                            )
                            original_flat = _contract_configuration_tensor(
                                flat_configuration_tensor,
                                configurations,
                            )
                            original_unitary = _contract_configuration_tensor(
                                unitary_configuration_tensor,
                                configurations,
                            )
                            jets = tuple(
                                frequency_jet_map(frequency) @ configuration
                                for frequency, configuration in zip(
                                    frequencies, configurations, strict=True
                                )
                            )
                            direct_jet_vertex = scale_factor**3 * np.einsum(
                                'ijk,i,j,k->',
                                np.asarray(flat_tensor, dtype=float),
                                jets[0],
                                jets[1],
                                jets[2],
                            )
                            pencils = (
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=base,
                                    signed_frequency_bar=first_frequency,
                                ),
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=base,
                                    signed_frequency_bar=second_frequency,
                                ),
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=2.0 * base,
                                    signed_frequency_bar=third_frequency,
                                ),
                            )
                            reversed_pencils = (
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=base,
                                    signed_frequency_bar=first_frequency,
                                    reverse_gyroscopic_sign=True,
                                ),
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=base,
                                    signed_frequency_bar=second_frequency,
                                    reverse_gyroscopic_sign=True,
                                ),
                                frozen_scalar_pencil(
                                    state,
                                    parameters,
                                    comoving_wavenumber_bar=2.0 * base,
                                    signed_frequency_bar=third_frequency,
                                    reverse_gyroscopic_sign=True,
                                ),
                            )
                            correct_deformation = _contract_configuration_tensor(
                                eom_exact_configuration_deformation(*pencils),
                                configurations,
                            )
                            transpose_deformation = _contract_configuration_tensor(
                                eom_exact_configuration_deformation(
                                    *pencils,
                                    transpose_pencils=True,
                                ),
                                configurations,
                            )
                            reversed_deformation = _contract_configuration_tensor(
                                eom_exact_configuration_deformation(
                                    *reversed_pencils
                                ),
                                configurations,
                            )
                            non_eom_deformation = _contract_configuration_tensor(
                                non_eom,
                                configurations,
                            )
                            omega_sum = float(sum(frequencies))
                            boundary_amplitude = 0.5 * _contract_configuration_tensor(
                                boundary,
                                configurations,
                            )
                            endpoint = boundary_amplitude * (
                                np.exp(-1j * omega_sum * interval) - 1.0
                            )
                            quadrature = _trapezoid_total_derivative(
                                boundary_amplitude,
                                omega_sum,
                                interval,
                            )
                            pencil_residual = max(
                                float(
                                    np.linalg.norm(pencil @ configuration)
                                    / max(1.0, np.linalg.norm(configuration))
                                )
                                for pencil, configuration in zip(
                                    pencils, configurations, strict=True
                                )
                            )
                            assignments.append(
                                EomBoundaryAssignment(
                                    key=(
                                        first_branch,
                                        second_branch,
                                        third_branch,
                                        first_sign,
                                        second_sign,
                                        third_sign,
                                    ),
                                    original_flat_vertex=original_flat,
                                    original_unitary_vertex=original_unitary,
                                    configuration_map_residual=float(
                                        abs(original_flat - direct_jet_vertex)
                                    ),
                                    correct_eom_deformation=correct_deformation,
                                    transposed_pencil_deformation=transpose_deformation,
                                    reversed_gyroscopic_deformation=reversed_deformation,
                                    non_eom_deformation=non_eom_deformation,
                                    signed_frequency_sum=omega_sum,
                                    boundary_endpoint=complex(endpoint),
                                    boundary_quadrature_residual=float(
                                        abs(endpoint - quadrature)
                                    ),
                                    maximum_pencil_residual=pencil_residual,
                                )
                            )
    return tuple(assignments)


@dataclass(frozen=True)
class ScalarEomBoundaryReceipt:
    base_wavenumber_bar: float
    assignment_count: int
    vertex_step_refinement: float
    vertex_grid_refinement: float
    vertex_gauge_residual: float
    configuration_map_residual: float
    correct_eom_quotient_residual: float
    transposed_pencil_negative_control: float
    reversed_gyroscopic_negative_control: float
    non_eom_negative_control: float
    maximum_pencil_residual: float
    same_k_exchange_residual: float
    resonant_assignment_count: int
    nonresonant_assignment_count: int
    minimum_nonzero_frequency_sum: float
    maximum_resonant_endpoint: float
    maximum_normalized_boundary_endpoint: float
    boundary_quadrature_residual: float
    declared_eom_boundary_gate_passed: bool


def evaluate_scalar_eom_boundary_gate(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    cubic_steps: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    phase_points: int = 256,
    grid_phase_points: int = 512,
    interval_bar: float = 0.5,
) -> ScalarEomBoundaryReceipt:
    '''Run the preregistered frozen EOM-ideal and endpoint-boundary gate.'''

    if len(cubic_steps) < 2:
        raise ValueError('the EOM/boundary gate requires at least two cubic steps')
    if grid_phase_points <= phase_points:
        raise ValueError('the EOM/boundary grid refinement must increase resolution')
    tensor_pairs = tuple(
        dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
        )
        for step in cubic_steps
    )
    grid_pair = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=float(cubic_steps[-1]),
        phase_points=grid_phase_points,
    )
    previous = scalar_eom_boundary_assignments(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_tensor=tensor_pairs[-2][0],
        unitary_tensor=tensor_pairs[-2][1],
        interval_bar=interval_bar,
    )
    fine = scalar_eom_boundary_assignments(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_tensor=tensor_pairs[-1][0],
        unitary_tensor=tensor_pairs[-1][1],
        interval_bar=interval_bar,
    )
    grid = scalar_eom_boundary_assignments(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_tensor=grid_pair[0],
        unitary_tensor=grid_pair[1],
        interval_bar=interval_bar,
    )
    vertex_scale = max(
        1.0,
        float(
            np.linalg.norm(
                np.array([item.original_flat_vertex for item in fine])
            )
        ),
    )
    step_refinement = max(
        abs(first.original_flat_vertex - second.original_flat_vertex)
        for first, second in zip(fine, previous, strict=True)
    ) / vertex_scale
    grid_refinement = max(
        abs(first.original_flat_vertex - second.original_flat_vertex)
        for first, second in zip(fine, grid, strict=True)
    ) / vertex_scale
    gauge_residual = max(
        abs(item.original_flat_vertex - item.original_unitary_vertex)
        for item in fine
    ) / vertex_scale
    lookup = {item.key: item for item in fine}
    exchange = 0.0
    for key, item in lookup.items():
        exchange_key = (
            key[1],
            key[0],
            key[2],
            key[4],
            key[3],
            key[5],
        )
        exchanged = lookup[exchange_key]
        exchange = max(
            exchange,
            abs(item.original_flat_vertex - exchanged.original_flat_vertex),
            abs(item.correct_eom_deformation - exchanged.correct_eom_deformation),
            abs(item.boundary_endpoint - exchanged.boundary_endpoint),
        )
    resonant = [item for item in fine if abs(item.signed_frequency_sum) < TOL]
    nonresonant = [item for item in fine if abs(item.signed_frequency_sum) >= TOL]
    minimum_nonzero = min(
        abs(item.signed_frequency_sum) for item in nonresonant
    )
    maximum_resonant_endpoint = max(
        (abs(item.boundary_endpoint) for item in resonant),
        default=0.0,
    )
    maximum_endpoint = max(
        abs(item.boundary_endpoint) for item in nonresonant
    ) / vertex_scale
    correct_eom = max(
        abs(item.correct_eom_deformation) for item in fine
    ) / vertex_scale
    transpose_control = max(
        abs(item.transposed_pencil_deformation) for item in fine
    ) / vertex_scale
    reversed_control = max(
        abs(item.reversed_gyroscopic_deformation) for item in fine
    ) / vertex_scale
    non_eom_control = max(
        abs(item.non_eom_deformation) for item in fine
    ) / vertex_scale
    map_residual = max(item.configuration_map_residual for item in fine)
    pencil_residual = max(item.maximum_pencil_residual for item in fine)
    quadrature_residual = max(
        item.boundary_quadrature_residual for item in fine
    ) / vertex_scale
    passed = (
        len(fine) == 64
        and step_refinement < 2.0e-4
        and grid_refinement < 1.0e-8
        and gauge_residual < 1.0e-6
        and map_residual < 1.0e-10
        and correct_eom < 1.0e-8
        and transpose_control > 1.0e-6
        and reversed_control > 1.0e-6
        and non_eom_control > 1.0e-6
        and pencil_residual < 1.0e-8
        and exchange < 1.0e-8
        and len(resonant) > 0
        and len(nonresonant) > 0
        and maximum_resonant_endpoint < 1.0e-10
        and maximum_endpoint > 1.0e-6
        and quadrature_residual < 1.0e-8
    )
    return ScalarEomBoundaryReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        assignment_count=len(fine),
        vertex_step_refinement=float(step_refinement),
        vertex_grid_refinement=float(grid_refinement),
        vertex_gauge_residual=float(gauge_residual),
        configuration_map_residual=float(map_residual),
        correct_eom_quotient_residual=float(correct_eom),
        transposed_pencil_negative_control=float(transpose_control),
        reversed_gyroscopic_negative_control=float(reversed_control),
        non_eom_negative_control=float(non_eom_control),
        maximum_pencil_residual=float(pencil_residual),
        same_k_exchange_residual=float(exchange),
        resonant_assignment_count=len(resonant),
        nonresonant_assignment_count=len(nonresonant),
        minimum_nonzero_frequency_sum=float(minimum_nonzero),
        maximum_resonant_endpoint=float(maximum_resonant_endpoint),
        maximum_normalized_boundary_endpoint=float(maximum_endpoint),
        boundary_quadrature_residual=float(quadrature_residual),
        declared_eom_boundary_gate_passed=passed,
    )
