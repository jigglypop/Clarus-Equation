'''Finite-time two-state Dyson witness for the E68 frozen scalar cubic vertex.

The module tests the time phase, repeated-leg Fock normalization, and first
Dyson coefficient of the admitted local cubic Hamiltonian tensor.  The exact
two-state evolution is an independent finite-dimensional witness, not the
full constrained Fock space or a cosmological in-in correlator.
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
    project_frozen_scalar_hamiltonian_vertices,
    scalar_interaction_hamiltonian_cubic_tensor_pair,
)
from examples.physics.qft_reference_flrw_cubic_eom_boundary import (
    deterministic_boundary_tensor,
)


TOL = 1.0e-12


def finite_time_exponential_kernel(
    frequency_bar: float,
    interval_bar: float,
) -> complex:
    '''Return integral_0^T exp(i omega t) dt with a stable zero limit.'''

    omega = float(frequency_bar)
    interval = float(interval_bar)
    if not np.isfinite(omega) or not np.isfinite(interval) or interval <= 0.0:
        raise ValueError('the finite-time kernel requires finite omega and positive T')
    if abs(omega * interval) <= 1.0e-8:
        return complex(
            interval
            * np.exp(0.5j * omega * interval)
            * np.sinc(omega * interval / (2.0 * np.pi))
        )
    return complex(np.expm1(1j * omega * interval) / (1j * omega))


def simpson_exponential_kernel(
    frequency_bar: float,
    interval_bar: float,
    *,
    subintervals: int,
) -> complex:
    '''Independently integrate the frozen Dyson phase by composite Simpson.'''

    if (
        not isinstance(subintervals, int)
        or subintervals < 2
        or subintervals % 2 != 0
    ):
        raise ValueError('the Dyson Simpson rule requires a positive even grid')
    times = np.linspace(0.0, float(interval_bar), subintervals + 1)
    values = np.exp(1j * float(frequency_bar) * times)
    spacing = float(interval_bar) / subintervals
    return complex(
        spacing
        / 3.0
        * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )
    )


@dataclass(frozen=True)
class TwoStateDysonResult:
    exact_interaction_amplitude: complex
    closed_form_interaction_amplitude: complex
    first_dyson_amplitude: complex
    exact_interaction_unitary: np.ndarray
    first_dyson_unitary: np.ndarray


def two_state_dyson_result(
    matrix_element: complex,
    frequency_gap_bar: float,
    interval_bar: float,
    *,
    coupling: float,
) -> TwoStateDysonResult:
    '''Compare matrix-exponential evolution with the first Dyson term.'''

    matrix_element = complex(matrix_element)
    omega = float(frequency_gap_bar)
    interval = float(interval_bar)
    coupling = float(coupling)
    if not np.isfinite(matrix_element):
        raise ValueError('the two-state matrix element must be finite')
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError('the two-state witness requires a positive gap')
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError('the two-state witness requires a positive interval')
    if not np.isfinite(coupling) or coupling < 0.0:
        raise ValueError('the bookkeeping coupling must be finite and nonnegative')
    hamiltonian = np.array(
        [
            [0.0, coupling * matrix_element.conjugate()],
            [coupling * matrix_element, omega],
        ],
        dtype=complex,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    schrodinger = (
        eigenvectors
        @ np.diag(np.exp(-1j * eigenvalues * interval))
        @ eigenvectors.conj().T
    )
    interaction_rotation = np.diag([1.0, np.exp(1j * omega * interval)])
    interaction = interaction_rotation @ schrodinger
    kernel = finite_time_exponential_kernel(omega, interval)
    first_amplitude = -1j * coupling * matrix_element * kernel
    first_reverse = -1j * coupling * matrix_element.conjugate() * kernel.conjugate()
    first_unitary = np.array(
        [[1.0, first_reverse], [first_amplitude, 1.0]],
        dtype=complex,
    )
    generalized_frequency = np.sqrt(
        (0.5 * omega) ** 2 + coupling**2 * abs(matrix_element) ** 2
    )
    closed_form = (
        -1j
        * coupling
        * matrix_element
        * np.exp(0.5j * omega * interval)
        * np.sin(generalized_frequency * interval)
        / generalized_frequency
    )
    return TwoStateDysonResult(
        exact_interaction_amplitude=complex(interaction[1, 0]),
        closed_form_interaction_amplitude=complex(closed_form),
        first_dyson_amplitude=complex(first_amplitude),
        exact_interaction_unitary=interaction,
        first_dyson_unitary=first_unitary,
    )


@dataclass(frozen=True)
class CubicDysonChannel:
    first_branch: int
    second_branch: int
    third_branch: int
    frequency_gap_bar: float
    creation_vertex: complex
    annihilation_vertex: complex
    matrix_element: complex
    boundary_endpoint: complex


def _creation_configuration(mode: FrozenScalarMode) -> np.ndarray:
    return mode.configuration.conj()


def scalar_cubic_dyson_channels(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    hamiltonian_tensor: np.ndarray,
    interval_bar: float = 0.5,
) -> tuple[CubicDysonChannel, ...]:
    '''Return the eight all-creation branch triples and separate B endpoints.'''

    base = float(base_wavenumber_bar)
    interval = float(interval_bar)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the cubic Dyson channels require positive base momentum')
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError('the cubic Dyson channels require a positive interval')
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
    vertices = project_frozen_scalar_hamiltonian_vertices(
        hamiltonian_tensor,
        first_modes,
        third_modes,
    )
    lookup = {
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
    boundary = deterministic_boundary_tensor()
    channels = []
    for first_branch in range(2):
        for second_branch in range(2):
            for third_branch in range(2):
                creation = lookup[
                    (first_branch, second_branch, third_branch, -1, -1, -1)
                ]
                annihilation = lookup[
                    (first_branch, second_branch, third_branch, 1, 1, 1)
                ]
                repeated_factor = (
                    1.0 / np.sqrt(2.0)
                    if first_branch == second_branch
                    else 1.0
                )
                matrix_element = repeated_factor * creation
                omega = (
                    first_modes[first_branch].frequency_bar
                    + first_modes[second_branch].frequency_bar
                    + third_modes[third_branch].frequency_bar
                )
                boundary_vertex = 0.5 * np.einsum(
                    'abc,a,b,c->',
                    boundary,
                    _creation_configuration(first_modes[first_branch]),
                    _creation_configuration(first_modes[second_branch]),
                    _creation_configuration(third_modes[third_branch]),
                )
                boundary_matrix_element = repeated_factor * boundary_vertex
                boundary_endpoint = boundary_matrix_element * (
                    np.exp(1j * omega * interval) - 1.0
                )
                channels.append(
                    CubicDysonChannel(
                        first_branch=first_branch,
                        second_branch=second_branch,
                        third_branch=third_branch,
                        frequency_gap_bar=float(omega),
                        creation_vertex=complex(creation),
                        annihilation_vertex=complex(annihilation),
                        matrix_element=complex(matrix_element),
                        boundary_endpoint=complex(boundary_endpoint),
                    )
                )
    return tuple(channels)


@dataclass(frozen=True)
class ScalarCubicDysonReceipt:
    base_wavenumber_bar: float
    channel_count: int
    vertex_step_refinement: float
    vertex_grid_refinement: float
    vertex_gauge_residual: float
    hermiticity_residual: float
    kernel_quadrature_residual: float
    kernel_grid_refinement: float
    exact_closed_form_residual: float
    maximum_active_transition_relative_residual: float
    maximum_inactive_transition_absolute_residual: float
    maximum_inactive_unitary_absolute_residual: float
    transition_lambda_ratio_deviation: float
    unitary_lambda_ratio_deviation: float
    fixed_witness_transition_ratio_deviation: float
    fixed_witness_unitary_ratio_deviation: float
    zero_coupling_residual: float
    wrong_frequency_sign_negative_control: float
    wrong_repeated_leg_negative_control: float
    overlarge_coupling_negative_control: float
    minimum_boundary_endpoint: float
    maximum_boundary_endpoint: float
    boundary_was_kept_separate: bool
    declared_dyson_gate_passed: bool


def _ratio_deviation(values: tuple[float, float, float], target: float) -> float:
    deviations = []
    if values[0] > 1.0e-15:
        deviations.append(abs(values[1] / values[0] - target))
    if values[1] > 1.0e-15:
        deviations.append(abs(values[2] / values[1] - target))
    return max(deviations, default=0.0)


def evaluate_scalar_cubic_dyson_gate(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    cubic_steps: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    phase_points: int = 256,
    grid_phase_points: int = 512,
    interval_bar: float = 0.5,
    couplings: tuple[float, float, float] = (1.0, 0.5, 0.25),
    time_subintervals: tuple[int, int] = (2048, 4096),
) -> ScalarCubicDysonReceipt:
    '''Run the preregistered finite-time cubic Dyson bookkeeping gate.'''

    if len(cubic_steps) < 2:
        raise ValueError('the cubic Dyson gate requires at least two cubic steps')
    if grid_phase_points <= phase_points:
        raise ValueError('the cubic Dyson grid refinement must increase resolution')
    if len(couplings) != 3 or not np.allclose(
        np.asarray(couplings[1:]),
        0.5 * np.asarray(couplings[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the cubic Dyson couplings must be three successive halvings')
    if (
        len(time_subintervals) != 2
        or time_subintervals[1] != 2 * time_subintervals[0]
    ):
        raise ValueError('the Dyson time grids must use one factor-two refinement')

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
        epsilon=float(cubic_steps[-1]),
        phase_points=grid_phase_points,
    )
    grid_hamiltonian = scalar_interaction_hamiltonian_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_lagrangian_tensor=grid_lagrangian[0],
        unitary_lagrangian_tensor=grid_lagrangian[1],
    )
    previous_channels = scalar_cubic_dyson_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-2][0],
        interval_bar=interval_bar,
    )
    flat_channels = scalar_cubic_dyson_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-1][0],
        interval_bar=interval_bar,
    )
    unitary_channels = scalar_cubic_dyson_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-1][1],
        interval_bar=interval_bar,
    )
    grid_channels = scalar_cubic_dyson_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=grid_hamiltonian[0],
        interval_bar=interval_bar,
    )
    vertex_scale = max(
        1.0,
        float(np.linalg.norm([item.matrix_element for item in flat_channels])),
    )
    step_refinement = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat_channels, previous_channels, strict=True)
    ) / vertex_scale
    grid_refinement = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat_channels, grid_channels, strict=True)
    ) / vertex_scale
    gauge_residual = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat_channels, unitary_channels, strict=True)
    ) / vertex_scale
    hermiticity = max(
        abs(item.creation_vertex - item.annihilation_vertex.conjugate())
        for item in flat_channels
    )

    kernel_residual = 0.0
    kernel_refinement = 0.0
    closed_form_residual = 0.0
    active_transition_residual = 0.0
    inactive_transition_residual = 0.0
    inactive_unitary_residual = 0.0
    transition_ratio_deviation = 0.0
    unitary_ratio_deviation = 0.0
    wrong_frequency = 0.0
    wrong_repeated = 0.0
    for channel in flat_channels:
        kernel = finite_time_exponential_kernel(
            channel.frequency_gap_bar,
            interval_bar,
        )
        coarse_kernel = simpson_exponential_kernel(
            channel.frequency_gap_bar,
            interval_bar,
            subintervals=time_subintervals[0],
        )
        fine_kernel = simpson_exponential_kernel(
            channel.frequency_gap_bar,
            interval_bar,
            subintervals=time_subintervals[1],
        )
        kernel_residual = max(kernel_residual, abs(fine_kernel - kernel))
        kernel_refinement = max(kernel_refinement, abs(fine_kernel - coarse_kernel))
        first_amplitude = -1j * channel.matrix_element * kernel
        wrong_kernel = finite_time_exponential_kernel(
            -channel.frequency_gap_bar,
            interval_bar,
        )
        wrong_amplitude = -1j * channel.matrix_element * wrong_kernel
        wrong_frequency = max(
            wrong_frequency,
            abs(first_amplitude - wrong_amplitude),
        )
        if channel.first_branch == channel.second_branch:
            wrong_matrix_element = channel.creation_vertex
            wrong_repeated = max(
                wrong_repeated,
                abs(
                    -1j * wrong_matrix_element * kernel
                    - first_amplitude
                ),
            )

        transition_errors = []
        unitary_errors = []
        for coupling in couplings:
            result = two_state_dyson_result(
                channel.matrix_element,
                channel.frequency_gap_bar,
                interval_bar,
                coupling=coupling,
            )
            closed_form_residual = max(
                closed_form_residual,
                abs(
                    result.exact_interaction_amplitude
                    - result.closed_form_interaction_amplitude
                ),
            )
            normalized_transition_error = abs(
                result.exact_interaction_amplitude / coupling
                - first_amplitude
            )
            transition_errors.append(normalized_transition_error)
            unitary_errors.append(
                float(
                    np.linalg.norm(
                        result.exact_interaction_unitary
                        - result.first_dyson_unitary
                    )
                )
            )
            if abs(first_amplitude) >= 1.0e-6:
                active_transition_residual = max(
                    active_transition_residual,
                    normalized_transition_error / abs(first_amplitude),
                )
            else:
                inactive_transition_residual = max(
                    inactive_transition_residual,
                    normalized_transition_error,
                )
        if abs(first_amplitude) >= 1.0e-6:
            transition_ratio_deviation = max(
                transition_ratio_deviation,
                _ratio_deviation(tuple(transition_errors), 0.25),
            )
            unitary_ratio_deviation = max(
                unitary_ratio_deviation,
                _ratio_deviation(tuple(unitary_errors), 0.25),
            )
        else:
            inactive_unitary_residual = max(
                inactive_unitary_residual,
                max(unitary_errors),
            )

    reference_matrix_element = 0.037 + 0.021j
    reference_gap = 0.7
    reference_transition_errors = []
    reference_unitary_errors = []
    for coupling in couplings:
        result = two_state_dyson_result(
            reference_matrix_element,
            reference_gap,
            interval_bar,
            coupling=coupling,
        )
        reference_first = -1j * reference_matrix_element * finite_time_exponential_kernel(
            reference_gap,
            interval_bar,
        )
        reference_transition_errors.append(
            abs(result.exact_interaction_amplitude / coupling - reference_first)
        )
        reference_unitary_errors.append(
            float(
                np.linalg.norm(
                    result.exact_interaction_unitary
                    - result.first_dyson_unitary
                )
            )
        )
    fixed_transition_ratio = _ratio_deviation(
        tuple(reference_transition_errors), 0.25
    )
    fixed_unitary_ratio = _ratio_deviation(tuple(reference_unitary_errors), 0.25)
    zero_result = two_state_dyson_result(
        reference_matrix_element,
        reference_gap,
        interval_bar,
        coupling=0.0,
    )
    zero_residual = max(
        abs(zero_result.exact_interaction_amplitude),
        abs(zero_result.first_dyson_amplitude),
    )
    overlarge = two_state_dyson_result(
        reference_matrix_element,
        reference_gap,
        interval_bar,
        coupling=20.0,
    )
    overlarge_control = abs(
        overlarge.exact_interaction_amplitude
        - overlarge.first_dyson_amplitude
    )
    boundary_magnitudes = [abs(item.boundary_endpoint) for item in flat_channels]
    passed = (
        len(flat_channels) == 8
        and step_refinement < 2.0e-4
        and grid_refinement < 1.0e-8
        and gauge_residual < 1.0e-6
        and hermiticity < 1.0e-10
        and kernel_residual < 1.0e-10
        and kernel_refinement < 1.0e-8
        and closed_form_residual < 1.0e-10
        and active_transition_residual < 2.0e-4
        and inactive_transition_residual < 1.0e-8
        and inactive_unitary_residual < 1.0e-8
        and transition_ratio_deviation < 5.0e-2
        and unitary_ratio_deviation < 5.0e-2
        and fixed_transition_ratio < 5.0e-2
        and fixed_unitary_ratio < 5.0e-2
        and zero_residual < 1.0e-12
        and wrong_frequency > 1.0e-6
        and wrong_repeated > 1.0e-6
        and overlarge_control > 1.0e-4
        and min(boundary_magnitudes) > 0.0
    )
    return ScalarCubicDysonReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        channel_count=len(flat_channels),
        vertex_step_refinement=float(step_refinement),
        vertex_grid_refinement=float(grid_refinement),
        vertex_gauge_residual=float(gauge_residual),
        hermiticity_residual=float(hermiticity),
        kernel_quadrature_residual=float(kernel_residual),
        kernel_grid_refinement=float(kernel_refinement),
        exact_closed_form_residual=float(closed_form_residual),
        maximum_active_transition_relative_residual=float(
            active_transition_residual
        ),
        maximum_inactive_transition_absolute_residual=float(
            inactive_transition_residual
        ),
        maximum_inactive_unitary_absolute_residual=float(
            inactive_unitary_residual
        ),
        transition_lambda_ratio_deviation=float(transition_ratio_deviation),
        unitary_lambda_ratio_deviation=float(unitary_ratio_deviation),
        fixed_witness_transition_ratio_deviation=float(fixed_transition_ratio),
        fixed_witness_unitary_ratio_deviation=float(fixed_unitary_ratio),
        zero_coupling_residual=float(zero_residual),
        wrong_frequency_sign_negative_control=float(wrong_frequency),
        wrong_repeated_leg_negative_control=float(wrong_repeated),
        overlarge_coupling_negative_control=float(overlarge_control),
        minimum_boundary_endpoint=float(min(boundary_magnitudes)),
        maximum_boundary_endpoint=float(max(boundary_magnitudes)),
        boundary_was_kept_separate=True,
        declared_dyson_gate_passed=passed,
    )
