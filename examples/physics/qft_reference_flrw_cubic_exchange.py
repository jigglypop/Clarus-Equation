'''Resonance admission for the E68 frozen scalar cubic exchange channel.

The gate inspects k+k <-> 2k branch triples.  It distinguishes channels that
may carry a nonresonant second-order exchange weight from exact resonances
that must remain explicit degrees of freedom.  The reported weights are not
scattering amplitudes, Wilson coefficients, or strong-coupling scales.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
)
from examples.physics.qft_reference_flrw_cubic_dynamics import (
    dynamic_reduced_scalar_cubic_tensor_pair,
    frozen_symplectic_scalar_modes,
    project_frozen_scalar_hamiltonian_vertices,
    scalar_interaction_hamiltonian_cubic_tensor_pair,
)
from examples.physics.qft_reference_flrw_cubic_dyson import (
    finite_time_exponential_kernel,
    simpson_exponential_kernel,
)


RESONANCE_TOL = 1.0e-10


@dataclass(frozen=True)
class CubicExchangeChannel:
    first_branch: int
    second_branch: int
    third_branch: int
    energy_mismatch_bar: float
    raw_vertex: complex
    conjugate_vertex: complex
    matrix_element: complex
    wrong_sign_vertex: complex
    finite_time_amplitude: complex
    exchange_weight: float | None
    resonant: bool


ResonanceDisposition = Literal['null', 'resolved', 'unclassified']


@dataclass(frozen=True)
class ResonanceCertificate:
    first_branch: int
    second_branch: int
    third_branch: int
    energy_mismatch_bar: float
    production_matrix_element_real: float
    production_matrix_element_imag: float
    production_error_envelope: float
    signal_to_error_ratio: float
    linear_second_order_ratio_residual: float
    richardson_matrix_element_real: float
    richardson_matrix_element_imag: float
    richardson_stability_residual: float
    linear_grid_residual: float
    linear_gauge_residual: float
    null_error_envelope: float
    null_relative_envelope: float
    disposition: ResonanceDisposition
    local_exchange_elimination_rejected: bool

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.first_branch, self.second_branch, self.third_branch)

    @property
    def production_matrix_element(self) -> complex:
        return complex(
            self.production_matrix_element_real,
            self.production_matrix_element_imag,
        )

    @property
    def richardson_matrix_element(self) -> complex:
        return complex(
            self.richardson_matrix_element_real,
            self.richardson_matrix_element_imag,
        )


def classify_resonant_channel(
    *,
    production_matrix_element: complex,
    production_error_envelope: float,
    linear_second_order_ratio_residual: float,
    richardson_matrix_element: complex,
    null_error_envelope: float,
    null_relative_envelope: float,
) -> ResonanceDisposition:
    '''Classify one resonant branch from its own numerical evidence.'''

    finite_values = (
        production_matrix_element.real,
        production_matrix_element.imag,
        production_error_envelope,
        linear_second_order_ratio_residual,
        richardson_matrix_element.real,
        richardson_matrix_element.imag,
        null_error_envelope,
        null_relative_envelope,
    )
    if not np.all(np.isfinite(finite_values)):
        raise ValueError('resonance evidence must be finite')
    if (
        production_error_envelope < 0.0
        or linear_second_order_ratio_residual < 0.0
        or null_error_envelope < 0.0
        or null_relative_envelope < 0.0
    ):
        raise ValueError('resonance error measures must be nonnegative')
    signal_to_error = abs(production_matrix_element) / max(
        production_error_envelope,
        1.0e-30,
    )
    is_null = bool(
        signal_to_error < 1.0
        and linear_second_order_ratio_residual < 0.15
        and abs(richardson_matrix_element) <= null_error_envelope
        and null_relative_envelope < 1.0e-5
    )
    is_resolved = bool(
        abs(production_matrix_element) > 1.0e-10
        and signal_to_error > 10.0
    )
    if is_null and is_resolved:
        raise RuntimeError('resonance classifications must be exclusive')
    if is_null:
        return 'null'
    if is_resolved:
        return 'resolved'
    return 'unclassified'


def scalar_cubic_exchange_channels(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    hamiltonian_tensor: np.ndarray,
    interval_bar: float = 0.5,
) -> tuple[CubicExchangeChannel, ...]:
    '''Return the eight k+k <-> 2k branch channels at one base momentum.'''

    base = float(base_wavenumber_bar)
    interval = float(interval_bar)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the cubic exchange gate requires positive base momentum')
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError('the cubic exchange gate requires a positive interval')
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
    channels = []
    for first_branch in range(2):
        for second_branch in range(2):
            for third_branch in range(2):
                raw = lookup[
                    (first_branch, second_branch, third_branch, 1, 1, -1)
                ]
                conjugate = lookup[
                    (first_branch, second_branch, third_branch, -1, -1, 1)
                ]
                wrong_sign = lookup[
                    (first_branch, second_branch, third_branch, 1, 1, 1)
                ]
                repeated_factor = (
                    1.0 / np.sqrt(2.0)
                    if first_branch == second_branch
                    else 1.0
                )
                matrix_element = repeated_factor * raw
                mismatch = (
                    first_modes[first_branch].frequency_bar
                    + first_modes[second_branch].frequency_bar
                    - third_modes[third_branch].frequency_bar
                )
                kernel = finite_time_exponential_kernel(-mismatch, interval)
                amplitude = -1j * matrix_element * kernel
                resonant = abs(mismatch) < RESONANCE_TOL
                weight = (
                    None
                    if resonant
                    else float(abs(matrix_element) ** 2 / mismatch)
                )
                channels.append(
                    CubicExchangeChannel(
                        first_branch=first_branch,
                        second_branch=second_branch,
                        third_branch=third_branch,
                        energy_mismatch_bar=float(mismatch),
                        raw_vertex=complex(raw),
                        conjugate_vertex=complex(conjugate),
                        matrix_element=complex(matrix_element),
                        wrong_sign_vertex=complex(wrong_sign),
                        finite_time_amplitude=complex(amplitude),
                        exchange_weight=weight,
                        resonant=resonant,
                    )
                )
    return tuple(channels)


@dataclass(frozen=True)
class ScalarCubicExchangeReceipt:
    base_wavenumber_bar: float
    channel_count: int
    resonant_channel_count: int
    nonresonant_channel_count: int
    resonance_certificates: tuple[ResonanceCertificate, ...]
    resonant_matrix_element_magnitude: float
    resonant_step_residual: float
    resonant_grid_residual: float
    resonant_gauge_residual: float
    resonant_signal_to_error_ratio: float
    resonant_linear_second_order_ratio_residual: float
    resonant_richardson_matrix_element_magnitude: float
    resonant_richardson_stability_residual: float
    resonant_linear_grid_residual: float
    resonant_linear_gauge_residual: float
    resonant_null_error_envelope: float
    resonant_null_relative_envelope: float
    resonant_null_consistent: bool
    resonant_finite_time_amplitude_magnitude: float
    resonant_kernel_limit_residual: float
    minimum_nonresonant_mismatch_magnitude: float
    minimum_exchange_weight: float
    maximum_exchange_weight: float
    vertex_step_refinement: float
    vertex_grid_refinement: float
    vertex_gauge_residual: float
    hermiticity_residual: float
    same_k_exchange_residual: float
    kernel_quadrature_residual: float
    unit_denominator_growth_ratio_residual: float
    unit_denominator_relative_nonconvergence_witness: float
    wrong_frequency_assignment_negative_control: float
    wrong_frequency_assignment_relative_control: float
    wrong_repeated_leg_negative_control: float
    wrong_repeated_leg_relative_control: float
    local_exchange_elimination_rejected: bool
    declared_exchange_gate_passed: bool


def evaluate_scalar_cubic_exchange_gate(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    cubic_steps: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    phase_points: int = 256,
    grid_phase_points: int = 512,
    interval_bar: float = 0.5,
    regulator_steps: tuple[float, float, float] = (1.0e-2, 5.0e-3, 2.5e-3),
    null_probe_steps: tuple[float, float, float, float] = (
        8.0e-2,
        4.0e-2,
        2.0e-2,
        1.0e-2,
    ),
    time_subintervals: int = 4096,
) -> ScalarCubicExchangeReceipt:
    '''Run the preregistered cubic-exchange resonance admission gate.'''

    if len(cubic_steps) < 2:
        raise ValueError('the exchange gate requires at least two cubic steps')
    if grid_phase_points <= phase_points:
        raise ValueError('the exchange grid refinement must increase resolution')
    if not np.allclose(
        np.asarray(regulator_steps[1:]),
        0.5 * np.asarray(regulator_steps[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the exchange regulators must be successive halvings')
    if len(null_probe_steps) != 4 or not np.allclose(
        np.asarray(null_probe_steps[1:]),
        0.5 * np.asarray(null_probe_steps[:-1]),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError('the resonant null probe requires four successive halvings')

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
    # A resonant coefficient that is zero at h=0 is easily mistaken for the
    # O(h^2) truncation error of a centered third derivative.  At cubic order
    # the linear constraint solution is sufficient by stationarity of the
    # eliminated lapse/shift.  Use that faster independent path at four
    # deliberately coarser halvings and Richardson-extrapolate the putative
    # null; the production tensor above remains the nonlinear-constraint path.
    null_lagrangian_pairs = tuple(
        dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
            constraint_scheme='linear',
        )
        for step in null_probe_steps
    )
    null_hamiltonian_pairs = tuple(
        scalar_interaction_hamiltonian_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            flat_lagrangian_tensor=pair[0],
            unitary_lagrangian_tensor=pair[1],
        )
        for pair in null_lagrangian_pairs
    )
    null_grid_lagrangian_pairs = tuple(
        dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=grid_phase_points,
            constraint_scheme='linear',
        )
        for step in null_probe_steps
    )
    null_grid_hamiltonian_pairs = tuple(
        scalar_interaction_hamiltonian_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            flat_lagrangian_tensor=pair[0],
            unitary_lagrangian_tensor=pair[1],
        )
        for pair in null_grid_lagrangian_pairs
    )
    previous = scalar_cubic_exchange_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-2][0],
        interval_bar=interval_bar,
    )
    flat = scalar_cubic_exchange_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-1][0],
        interval_bar=interval_bar,
    )
    unitary = scalar_cubic_exchange_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=hamiltonian_pairs[-1][1],
        interval_bar=interval_bar,
    )
    grid = scalar_cubic_exchange_channels(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        hamiltonian_tensor=grid_hamiltonian[0],
        interval_bar=interval_bar,
    )
    null_flat_sets = tuple(
        scalar_cubic_exchange_channels(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            hamiltonian_tensor=pair[0],
            interval_bar=interval_bar,
        )
        for pair in null_hamiltonian_pairs
    )
    null_unitary_sets = tuple(
        scalar_cubic_exchange_channels(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            hamiltonian_tensor=pair[1],
            interval_bar=interval_bar,
        )
        for pair in null_hamiltonian_pairs
    )
    null_grid_sets = tuple(
        scalar_cubic_exchange_channels(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            hamiltonian_tensor=pair[0],
            interval_bar=interval_bar,
        )
        for pair in null_grid_hamiltonian_pairs
    )
    scale = max(1.0, float(np.linalg.norm([item.matrix_element for item in flat])))
    step_refinement = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat, previous, strict=True)
    ) / scale
    grid_refinement = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat, grid, strict=True)
    ) / scale
    gauge_residual = max(
        abs(first.matrix_element - second.matrix_element)
        for first, second in zip(flat, unitary, strict=True)
    ) / scale
    hermiticity = max(
        abs(item.raw_vertex - item.conjugate_vertex.conjugate()) for item in flat
    )
    lookup = {
        (item.first_branch, item.second_branch, item.third_branch): item
        for item in flat
    }
    exchange_residual = 0.0
    for key, item in lookup.items():
        swapped = lookup[(key[1], key[0], key[2])]
        exchange_residual = max(
            exchange_residual,
            abs(item.matrix_element - swapped.matrix_element),
        )

    resonant = [item for item in flat if item.resonant]
    nonresonant = [item for item in flat if not item.resonant]
    if not resonant or not nonresonant:
        raise ValueError('the declared exchange gate requires both channel types')
    resonant_matrix = max(abs(item.matrix_element) for item in resonant)
    resonant_indices = [index for index, item in enumerate(flat) if item.resonant]
    resonant_step_residual = max(
        abs(flat[index].matrix_element - previous[index].matrix_element)
        for index in resonant_indices
    )
    resonant_grid_residual = max(
        abs(flat[index].matrix_element - grid[index].matrix_element)
        for index in resonant_indices
    )
    resonant_gauge_residual = max(
        abs(flat[index].matrix_element - unitary[index].matrix_element)
        for index in resonant_indices
    )
    resonant_error = max(
        resonant_step_residual,
        resonant_grid_residual,
        resonant_gauge_residual,
        1.0e-30,
    )
    resonant_signal_to_error = resonant_matrix / resonant_error
    resonant_keys = tuple(sorted({
        (item.first_branch, item.second_branch, item.third_branch)
        for item in resonant
    }))

    def channel_lookup(
        channels: tuple[CubicExchangeChannel, ...],
    ) -> dict[tuple[int, int, int], CubicExchangeChannel]:
        return {
            (item.first_branch, item.second_branch, item.third_branch): item
            for item in channels
        }

    null_lookups = tuple(channel_lookup(items) for items in null_flat_sets)
    null_unitary_lookups = tuple(
        channel_lookup(items) for items in null_unitary_sets
    )
    null_grid_lookups = tuple(
        channel_lookup(items) for items in null_grid_sets
    )
    previous_lookup = channel_lookup(previous)
    unitary_lookup = channel_lookup(unitary)
    grid_lookup = channel_lookup(grid)

    def richardson_limit(values: list[complex]) -> tuple[complex, float]:
        first = [
            (4.0 * values[index + 1] - values[index]) / 3.0
            for index in range(3)
        ]
        second = [
            (16.0 * first[index + 1] - first[index]) / 15.0
            for index in range(2)
        ]
        extrapolated = (64.0 * second[1] - second[0]) / 63.0
        stability = max(
            abs(extrapolated - second[-1]),
            abs(second[-1] - first[-1]),
        )
        return extrapolated, float(stability)

    ratio_residual = 0.0
    richardson_magnitude = 0.0
    richardson_stability = 0.0
    null_grid_residual = 0.0
    null_gauge_residual = 0.0
    vertex_scale = max(abs(item.matrix_element) for item in flat)
    certificate_inputs: list[dict[str, object]] = []
    for key in resonant_keys:
        production_matrix_element = lookup[key].matrix_element
        production_step_residual = abs(
            production_matrix_element - previous_lookup[key].matrix_element
        )
        production_grid_residual = abs(
            production_matrix_element - grid_lookup[key].matrix_element
        )
        production_gauge_residual = abs(
            production_matrix_element - unitary_lookup[key].matrix_element
        )
        production_error_envelope = max(
            production_step_residual,
            production_grid_residual,
            production_gauge_residual,
            1.0e-30,
        )
        signal_to_error = (
            abs(production_matrix_element) / production_error_envelope
        )
        values = [lookup_[key].matrix_element for lookup_ in null_lookups]
        unitary_values = [
            lookup_[key].matrix_element for lookup_ in null_unitary_lookups
        ]
        grid_values = [
            lookup_[key].matrix_element for lookup_ in null_grid_lookups
        ]
        local_ratio_residual = max(
            (
                abs(abs(values[index]) / max(abs(values[index + 1]), 1.0e-300) - 4.0)
                for index in range(len(values) - 1)
            )
        )
        extrapolated, stability = richardson_limit(values)
        unitary_extrapolated, unitary_stability = richardson_limit(
            unitary_values
        )
        grid_extrapolated, grid_stability = richardson_limit(grid_values)
        # Flat gauge is the preregistered primary estimator.  The independently
        # extrapolated unitary pullback and denser phase grid enter its error
        # envelope below; treating their maximum as a second signal would mix
        # an estimator with its systematic-error probes.
        local_stability = max(
            stability,
            unitary_stability,
            grid_stability,
        )
        local_grid_residual = abs(extrapolated - grid_extrapolated)
        local_gauge_residual = abs(extrapolated - unitary_extrapolated)
        local_null_error_envelope = max(
            local_stability,
            local_grid_residual,
            local_gauge_residual,
            1.0e-30,
        )
        local_null_relative_envelope = (
            abs(extrapolated) + local_null_error_envelope
        ) / max(vertex_scale, 1.0e-30)
        disposition = classify_resonant_channel(
            production_matrix_element=production_matrix_element,
            production_error_envelope=production_error_envelope,
            linear_second_order_ratio_residual=local_ratio_residual,
            richardson_matrix_element=extrapolated,
            null_error_envelope=local_null_error_envelope,
            null_relative_envelope=local_null_relative_envelope,
        )
        certificate_inputs.append(
            {
                'first_branch': key[0],
                'second_branch': key[1],
                'third_branch': key[2],
                'energy_mismatch_bar': lookup[key].energy_mismatch_bar,
                'production_matrix_element_real': production_matrix_element.real,
                'production_matrix_element_imag': production_matrix_element.imag,
                'production_error_envelope': production_error_envelope,
                'signal_to_error_ratio': signal_to_error,
                'linear_second_order_ratio_residual': local_ratio_residual,
                'richardson_matrix_element_real': extrapolated.real,
                'richardson_matrix_element_imag': extrapolated.imag,
                'richardson_stability_residual': local_stability,
                'linear_grid_residual': local_grid_residual,
                'linear_gauge_residual': local_gauge_residual,
                'null_error_envelope': local_null_error_envelope,
                'null_relative_envelope': local_null_relative_envelope,
                'disposition': disposition,
            }
        )
        ratio_residual = max(ratio_residual, local_ratio_residual)
        richardson_magnitude = max(richardson_magnitude, abs(extrapolated))
        richardson_stability = max(richardson_stability, local_stability)
        null_grid_residual = max(null_grid_residual, local_grid_residual)
        null_gauge_residual = max(null_gauge_residual, local_gauge_residual)
    null_error_envelope = max(
        richardson_stability,
        null_grid_residual,
        null_gauge_residual,
        1.0e-30,
    )
    null_relative_envelope = max(
        float(item['null_relative_envelope'])
        for item in certificate_inputs
    )
    resonant_signal_to_error = max(
        float(item['signal_to_error_ratio'])
        for item in certificate_inputs
    )
    null_consistent = all(
        item['disposition'] == 'null' for item in certificate_inputs
    )
    resonant_amplitude = max(abs(item.finite_time_amplitude) for item in resonant)
    resonant_kernel_residual = max(
        abs(
            finite_time_exponential_kernel(
                -item.energy_mismatch_bar,
                interval_bar,
            )
            - interval_bar
        )
        for item in resonant
    )
    minimum_mismatch = min(
        abs(item.energy_mismatch_bar) for item in nonresonant
    )
    exchange_weights = [
        float(item.exchange_weight)
        for item in nonresonant
        if item.exchange_weight is not None
    ]
    kernel_residual = max(
        abs(
            finite_time_exponential_kernel(
                -item.energy_mismatch_bar,
                interval_bar,
            )
            - simpson_exponential_kernel(
                -item.energy_mismatch_bar,
                interval_bar,
                subintervals=time_subintervals,
            )
        )
        for item in flat
    )
    # This unit-numerator diagnostic checks only that a formal resonant
    # denominator is singular.  It is deliberately not multiplied by the
    # unresolved physical matrix element and is not evidence that this null
    # channel must be retained as an interacting coupled channel.
    regulator_magnitudes = [1.0 / abs(regulator) for regulator in regulator_steps]
    growth_residual = max(
        abs(regulator_magnitudes[1] / regulator_magnitudes[0] - 2.0),
        abs(regulator_magnitudes[2] / regulator_magnitudes[1] - 2.0),
    )
    nonconvergence = abs(
        regulator_magnitudes[2] - regulator_magnitudes[1]
    ) / max(regulator_magnitudes[2], 1.0e-300)
    wrong_frequency = 0.0
    wrong_repeated = 0.0
    correct_amplitude_scale = max(
        abs(item.finite_time_amplitude) for item in flat
    )
    repeated_amplitude_scale = max(
        abs(item.finite_time_amplitude)
        for item in flat
        if item.first_branch == item.second_branch
    )
    first_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=base_wavenumber_bar,
    )
    third_modes = frozen_symplectic_scalar_modes(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base_wavenumber_bar,
    )
    for item in flat:
        wrong_gap = (
            first_modes[item.first_branch].frequency_bar
            + first_modes[item.second_branch].frequency_bar
            + third_modes[item.third_branch].frequency_bar
        )
        repeated_factor = (
            1.0 / np.sqrt(2.0)
            if item.first_branch == item.second_branch
            else 1.0
        )
        wrong_amplitude = (
            -1j
            * repeated_factor
            * item.wrong_sign_vertex
            * finite_time_exponential_kernel(-wrong_gap, interval_bar)
        )
        wrong_frequency = max(
            wrong_frequency,
            abs(item.finite_time_amplitude - wrong_amplitude),
        )
        if item.first_branch == item.second_branch:
            wrong_matrix = item.raw_vertex
            wrong_amplitude = (
                -1j
                * wrong_matrix
                * finite_time_exponential_kernel(
                    -item.energy_mismatch_bar,
                    interval_bar,
                )
            )
            wrong_repeated = max(
                wrong_repeated,
                abs(item.finite_time_amplitude - wrong_amplitude),
            )
    wrong_frequency_relative = wrong_frequency / max(
        correct_amplitude_scale,
        1.0e-30,
    )
    wrong_repeated_relative = wrong_repeated / max(
        repeated_amplitude_scale,
        1.0e-30,
    )
    regulator_supports_rejection = bool(
        growth_residual < 1.0e-6
        and nonconvergence > 0.49
    )
    resonance_certificates = tuple(
        ResonanceCertificate(
            **certificate_input,
            local_exchange_elimination_rejected=bool(
                certificate_input['disposition'] == 'resolved'
                and regulator_supports_rejection
            ),
        )
        for certificate_input in certificate_inputs
    )
    resolved_nonzero = any(
        item.disposition == 'resolved' for item in resonance_certificates
    )
    rejected = any(
        item.local_exchange_elimination_rejected
        for item in resonance_certificates
    )
    resonance_classified = all(
        item.disposition != 'unclassified'
        for item in resonance_certificates
    )
    passed = (
        len(flat) == 8
        and len(resonant) > 0
        and len(nonresonant) > 0
        and resonance_classified
        and step_refinement < 2.0e-4
        and grid_refinement < 1.0e-8
        and gauge_residual < 1.0e-6
        and hermiticity < 1.0e-10
        and exchange_residual < 1.0e-8
        and kernel_residual < 1.0e-10
        and resonant_kernel_residual < 1.0e-10
        and growth_residual < 1.0e-6
        and nonconvergence > 0.49
        and wrong_frequency > 1.0e-6
        and wrong_frequency_relative > 1.0e-3
        and wrong_repeated > 1.0e-6
        and wrong_repeated_relative > 1.0e-3
        and all(
            item.disposition != 'resolved'
            or item.local_exchange_elimination_rejected
            for item in resonance_certificates
        )
    )
    return ScalarCubicExchangeReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        channel_count=len(flat),
        resonant_channel_count=len(resonant),
        nonresonant_channel_count=len(nonresonant),
        resonance_certificates=resonance_certificates,
        resonant_matrix_element_magnitude=float(resonant_matrix),
        resonant_step_residual=float(resonant_step_residual),
        resonant_grid_residual=float(resonant_grid_residual),
        resonant_gauge_residual=float(resonant_gauge_residual),
        resonant_signal_to_error_ratio=float(resonant_signal_to_error),
        resonant_linear_second_order_ratio_residual=float(ratio_residual),
        resonant_richardson_matrix_element_magnitude=float(
            richardson_magnitude
        ),
        resonant_richardson_stability_residual=float(richardson_stability),
        resonant_linear_grid_residual=float(null_grid_residual),
        resonant_linear_gauge_residual=float(null_gauge_residual),
        resonant_null_error_envelope=float(null_error_envelope),
        resonant_null_relative_envelope=float(null_relative_envelope),
        resonant_null_consistent=null_consistent,
        resonant_finite_time_amplitude_magnitude=float(resonant_amplitude),
        resonant_kernel_limit_residual=float(resonant_kernel_residual),
        minimum_nonresonant_mismatch_magnitude=float(minimum_mismatch),
        minimum_exchange_weight=float(min(exchange_weights)),
        maximum_exchange_weight=float(max(exchange_weights)),
        vertex_step_refinement=float(step_refinement),
        vertex_grid_refinement=float(grid_refinement),
        vertex_gauge_residual=float(gauge_residual),
        hermiticity_residual=float(hermiticity),
        same_k_exchange_residual=float(exchange_residual),
        kernel_quadrature_residual=float(kernel_residual),
        unit_denominator_growth_ratio_residual=float(growth_residual),
        unit_denominator_relative_nonconvergence_witness=float(nonconvergence),
        wrong_frequency_assignment_negative_control=float(wrong_frequency),
        wrong_frequency_assignment_relative_control=float(
            wrong_frequency_relative
        ),
        wrong_repeated_leg_negative_control=float(wrong_repeated),
        wrong_repeated_leg_relative_control=float(wrong_repeated_relative),
        local_exchange_elimination_rejected=rejected,
        declared_exchange_gate_passed=passed,
    )
