'''Cubic-triad admission checks for the E62 reference background.

The implemented vertex is deliberately narrow: a frozen, static q/r scalar
triad in the E66 flat gauge and its exact time-independent spatial
diffeomorphism to rod-unitary gauge.  It demonstrates why a one-mode cubic
derivative is a false negative and checks a two-gauge off-shell tensor.
It does not derive the full on-shell cubic action or a strong-coupling scale.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    audit_reference_flrw_background,
    rod_charge_bar,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


TOL = 1.0e-10
E68_BASE_WAVENUMBERS_BAR = (0.05, 0.1, 0.2, 0.4)
E68_CUBIC_STEPS = (1.0e-2, 5.0e-3, 2.5e-3)
E68_GAUGE_TOL = 1.0e-6
E68_REFINEMENT_TOL = 2.0e-4


def normalized_cubic_profile_overlaps(phase_points: int = 512) -> tuple[float, float]:
    '''Return <f_1^3> and <f_1^2 f_2> for f_n=sqrt(2) cos(n theta).'''

    if not isinstance(phase_points, int) or phase_points < 16:
        raise ValueError('phase_points must be an integer of at least sixteen')
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    first = np.sqrt(2.0) * np.cos(phase)
    second = np.sqrt(2.0) * np.cos(2.0 * phase)
    return float(np.mean(first**3)), float(np.mean(first**2 * second))


def _validate_static_triad(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    rod_amplitudes: np.ndarray,
    phase_points: int,
) -> tuple[float, float, float, float]:
    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('E68 requires an admitted nondegenerate E62 background')
    base = float(base_wavenumber_bar)
    amplitudes = np.asarray(rod_amplitudes, dtype=float)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the cubic triad requires positive base momentum')
    if amplitudes.shape != (2,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('the k and 2k rod amplitudes must be a finite pair')
    if not isinstance(phase_points, int) or phase_points < 32:
        raise ValueError('phase_points must be an integer of at least thirty-two')
    a = float(np.exp(state.n))
    beta_bar = rod_charge_bar(state)
    m = float(parameters.m_planck_over_mu_x)
    if abs(beta_bar) <= TOL or not np.isfinite(m) or m <= 0.0:
        raise ValueError('the cubic rod chart requires nonzero beta and positive M_P/mu_X')
    return a, beta_bar, m, base / a


def _linear_static_constraints(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    rod_amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lapse_amplitudes = []
    shift_amplitudes = []
    for harmonic, rod_amplitude in enumerate(rod_amplitudes, start=1):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=harmonic * base_wavenumber_bar,
        )
        source = blocks.field_coupling_bar @ np.array([0.0, rod_amplitude])
        constraint = -np.linalg.solve(blocks.constraint_matrix_bar, source)
        lapse_amplitudes.append(constraint[0])
        shift_amplitudes.append(constraint[1])
    return np.array(lapse_amplitudes), np.array(shift_amplitudes)


@dataclass(frozen=True)
class StaticTriadFields:
    lapse: np.ndarray
    flat_shift_contravariant: np.ndarray
    flat_shift_covariant_derivative: np.ndarray
    rod_gradient: np.ndarray
    rod_gradient_derivative: np.ndarray
    coordinate_jacobian: np.ndarray
    coordinate_jacobian_derivative: np.ndarray


@dataclass(frozen=True)
class StaticScalarTriadFields:
    lapse: np.ndarray
    flat_shift_contravariant: np.ndarray
    flat_shift_covariant_derivative: np.ndarray
    clock_gradient: np.ndarray
    rod_gradient: np.ndarray
    coordinate_jacobian: np.ndarray


def _static_scalar_triad_fields(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    mode_amplitudes: np.ndarray,
    phase_points: int,
) -> StaticScalarTriadFields:
    amplitudes = np.asarray(mode_amplitudes, dtype=float)
    if amplitudes.shape != (2, 2) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('static scalar amplitudes must have shape (k,2k) by (q,r)')
    a, beta_bar, _, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=amplitudes[:, 1],
        phase_points=phase_points,
    )
    lapse_modes = []
    shift_modes = []
    for harmonic, mode in enumerate(amplitudes, start=1):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=harmonic * base_wavenumber_bar,
        )
        source = blocks.field_coupling_bar @ mode
        constraint = -np.linalg.solve(blocks.constraint_matrix_bar, source)
        lapse_modes.append(constraint[0])
        shift_modes.append(constraint[1])
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    lapse = np.ones(phase_points)
    flat_shift = np.zeros(phase_points)
    flat_shift_covariant_derivative = np.zeros(phase_points)
    clock_gradient = np.zeros(phase_points)
    rod_displacement_derivative = np.zeros(phase_points)
    for harmonic, (mode, alpha, theta) in enumerate(
        zip(amplitudes, lapse_modes, shift_modes, strict=True),
        start=1,
    ):
        q_amplitude, rod_amplitude = mode
        wavenumber = harmonic * base_wavenumber_bar
        cosine = np.sqrt(2.0) * np.cos(harmonic * phase)
        sine = np.sqrt(2.0) * np.sin(harmonic * phase)
        lapse += alpha * cosine
        flat_shift += -theta * sine / beta_bar
        flat_shift_covariant_derivative += (
            -a**2 * theta * wavenumber * cosine / beta_bar
        )
        clock_gradient += -q_amplitude * wavenumber * sine
        # E66 uses r=k s.  The rod displacement delta X=d s has amplitude r,
        # so its spatial derivative entering d X is -k r cos(k x).
        rod_displacement_derivative += -rod_amplitude * wavenumber * cosine
    if np.min(lapse) <= 0.0:
        raise ValueError('the static scalar sampled lapse must stay positive')
    rod_gradient = beta_bar + rod_displacement_derivative
    jacobian = rod_gradient / beta_bar
    if np.min(jacobian) <= 0.0:
        raise ValueError('the static scalar rod-unitary map must stay invertible')
    return StaticScalarTriadFields(
        lapse=lapse,
        flat_shift_contravariant=flat_shift,
        flat_shift_covariant_derivative=flat_shift_covariant_derivative,
        clock_gradient=clock_gradient,
        rod_gradient=rod_gradient,
        coordinate_jacobian=jacobian,
    )


def _static_triad_fields(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    rod_amplitudes: np.ndarray,
    phase_points: int,
) -> StaticTriadFields:
    a, beta_bar, _, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=rod_amplitudes,
        phase_points=phase_points,
    )
    lapse_modes, shift_modes = _linear_static_constraints(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=rod_amplitudes,
    )
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    lapse = np.ones(phase_points)
    flat_shift = np.zeros(phase_points)
    flat_shift_covariant_derivative = np.zeros(phase_points)
    rod_displacement_derivative = np.zeros(phase_points)
    rod_displacement_second_derivative = np.zeros(phase_points)
    for harmonic, (rod_amplitude, alpha, theta) in enumerate(
        zip(rod_amplitudes, lapse_modes, shift_modes, strict=True),
        start=1,
    ):
        wavenumber = harmonic * base_wavenumber_bar
        cosine = np.sqrt(2.0) * np.cos(harmonic * phase)
        sine = np.sqrt(2.0) * np.sin(harmonic * phase)
        lapse += alpha * cosine
        flat_shift += -theta * sine / beta_bar
        flat_shift_covariant_derivative += (
            -a**2 * theta * wavenumber * cosine / beta_bar
        )
        # E66 uses r=k s; this is d(delta X), not the amplitude of s.
        rod_displacement_derivative += -rod_amplitude * wavenumber * cosine
        rod_displacement_second_derivative += (
            rod_amplitude * wavenumber**2 * sine
        )
    if np.min(lapse) <= 0.0:
        raise ValueError('the cubic sampled lapse must stay positive')
    rod_gradient = beta_bar + rod_displacement_derivative
    jacobian = rod_gradient / beta_bar
    if np.min(jacobian) <= 0.0:
        raise ValueError('the rod-unitary coordinate map must stay invertible')
    return StaticTriadFields(
        lapse=lapse,
        flat_shift_contravariant=flat_shift,
        flat_shift_covariant_derivative=flat_shift_covariant_derivative,
        rod_gradient=rod_gradient,
        rod_gradient_derivative=rod_displacement_second_derivative,
        coordinate_jacobian=jacobian,
        coordinate_jacobian_derivative=(
            rod_displacement_second_derivative / beta_bar
        ),
    )


def flat_gauge_static_triad_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    rod_amplitudes: np.ndarray,
    phase_points: int = 512,
) -> float:
    '''Evaluate the exact static longitudinal triad in the E66 flat gauge.'''

    a, beta_bar, m, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=rod_amplitudes,
        phase_points=phase_points,
    )
    fields = _static_triad_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=np.asarray(rod_amplitudes, dtype=float),
        phase_points=phase_points,
    )
    lapse = fields.lapse
    shift = fields.flat_shift_contravariant
    rod_gradient = fields.rod_gradient
    clock = 0.5 * np.mean(state.u**2 / lapse)
    rods = 0.5 * np.mean(
        (shift * rod_gradient) ** 2 / lapse
        - lapse * (2.0 * beta_bar**2 + rod_gradient**2) / a**2
    )
    transverse_extrinsic = a**2 * state.h / lapse
    longitudinal_extrinsic = (
        a**2 * state.h - fields.flat_shift_covariant_derivative
    ) / lapse
    mixed_transverse = transverse_extrinsic / a**2
    mixed_longitudinal = longitudinal_extrinsic / a**2
    extrinsic_combination = (
        2.0 * mixed_transverse**2
        + mixed_longitudinal**2
        - (2.0 * mixed_transverse + mixed_longitudinal) ** 2
    )
    lam = float(parameters.lambda_over_mu_x_squared)
    gravity = 0.5 * m**2 * np.mean(
        lapse * (extrinsic_combination - 2.0 * lam)
    )
    background = (
        0.5 * state.u**2
        - 1.5 * beta_bar**2 / a**2
        + 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    )
    return float(clock + rods + gravity - background)


def flat_gauge_static_scalar_triad_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    mode_amplitudes: np.ndarray,
    phase_points: int = 512,
) -> float:
    '''Evaluate all static q/r field directions for k,k,-2k in flat gauge.'''

    amplitudes = np.asarray(mode_amplitudes, dtype=float)
    a, beta_bar, m, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=amplitudes[:, 1],
        phase_points=phase_points,
    )
    fields = _static_scalar_triad_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        mode_amplitudes=amplitudes,
        phase_points=phase_points,
    )
    lapse = fields.lapse
    shift = fields.flat_shift_contravariant
    clock_convective = state.u - shift * fields.clock_gradient
    clock = 0.5 * np.mean(
        clock_convective**2 / lapse
        - lapse * fields.clock_gradient**2 / a**2
    )
    rods = 0.5 * np.mean(
        (shift * fields.rod_gradient) ** 2 / lapse
        - lapse
        * (2.0 * beta_bar**2 + fields.rod_gradient**2)
        / a**2
    )
    transverse_extrinsic = a**2 * state.h / lapse
    longitudinal_extrinsic = (
        a**2 * state.h - fields.flat_shift_covariant_derivative
    ) / lapse
    mixed_transverse = transverse_extrinsic / a**2
    mixed_longitudinal = longitudinal_extrinsic / a**2
    extrinsic_combination = (
        2.0 * mixed_transverse**2
        + mixed_longitudinal**2
        - (2.0 * mixed_transverse + mixed_longitudinal) ** 2
    )
    lam = float(parameters.lambda_over_mu_x_squared)
    gravity = 0.5 * m**2 * np.mean(
        lapse * (extrinsic_combination - 2.0 * lam)
    )
    background = (
        0.5 * state.u**2
        - 1.5 * beta_bar**2 / a**2
        + 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    )
    return float(clock + rods + gravity - background)


def rod_unitary_static_triad_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    rod_amplitudes: np.ndarray,
    phase_points: int = 512,
    include_coordinate_measure: bool = True,
) -> float:
    '''Evaluate the exactly spatially transformed static triad in rod-unitary gauge.'''

    a, beta_bar, m, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=rod_amplitudes,
        phase_points=phase_points,
    )
    fields = _static_triad_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=np.asarray(rod_amplitudes, dtype=float),
        phase_points=phase_points,
    )
    lapse = fields.lapse
    jacobian = fields.coordinate_jacobian
    flat_shift = fields.flat_shift_contravariant
    flat_shift_covariant = a**2 * flat_shift
    unitary_shift_covariant = flat_shift_covariant / jacobian
    # D_y N_y, not partial_y N_y.  For h_yy=a^2/J^2 the nonzero
    # Christoffel term cancels the derivative of the covector Jacobian.
    unitary_shift_covariant_derivative = (
        fields.flat_shift_covariant_derivative / jacobian**2
    )
    unitary_shift_contravariant = (
        jacobian**2 * unitary_shift_covariant / a**2
    )
    clock_integrand = 0.5 * state.u**2 / lapse
    rod_spatial_norm = (
        2.0 * beta_bar**2 / a**2
        + beta_bar**2 * jacobian**2 / a**2
    )
    rod_integrand = 0.5 * (
        (unitary_shift_contravariant * beta_bar) ** 2 / lapse
        - lapse * rod_spatial_norm
    )
    transverse_mixed_extrinsic = state.h / lapse
    longitudinal_metric = a**2 / jacobian**2
    longitudinal_extrinsic = (
        state.h * longitudinal_metric
        - unitary_shift_covariant_derivative
    ) / lapse
    longitudinal_mixed_extrinsic = (
        jacobian**2 * longitudinal_extrinsic / a**2
    )
    extrinsic_combination = (
        2.0 * transverse_mixed_extrinsic**2
        + longitudinal_mixed_extrinsic**2
        - (
            2.0 * transverse_mixed_extrinsic
            + longitudinal_mixed_extrinsic
        )
        ** 2
    )
    lam = float(parameters.lambda_over_mu_x_squared)
    gravity_integrand = 0.5 * m**2 * lapse * (
        extrinsic_combination - 2.0 * lam
    )
    measure_weight = np.ones_like(jacobian)
    if not include_coordinate_measure:
        measure_weight = 1.0 / jacobian
    total = np.mean(
        measure_weight
        * (clock_integrand + rod_integrand + gravity_integrand)
    )
    background = (
        0.5 * state.u**2
        - 1.5 * beta_bar**2 / a**2
        + 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    )
    return float(total - background)


def rod_unitary_static_scalar_triad_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    mode_amplitudes: np.ndarray,
    phase_points: int = 512,
) -> float:
    '''Evaluate the exact static q/r triad after the rod-unitary spatial map.'''

    amplitudes = np.asarray(mode_amplitudes, dtype=float)
    a, beta_bar, m, _ = _validate_static_triad(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        rod_amplitudes=amplitudes[:, 1],
        phase_points=phase_points,
    )
    fields = _static_scalar_triad_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        mode_amplitudes=amplitudes,
        phase_points=phase_points,
    )
    lapse = fields.lapse
    jacobian = fields.coordinate_jacobian
    shift_flat = fields.flat_shift_contravariant
    shift_unitary = jacobian * shift_flat
    clock_gradient_unitary = fields.clock_gradient / jacobian
    clock_convective = state.u - shift_unitary * clock_gradient_unitary
    clock = 0.5 * np.mean(
        clock_convective**2 / lapse
        - lapse
        * (jacobian**2 / a**2)
        * clock_gradient_unitary**2
    )
    rods = 0.5 * np.mean(
        (shift_unitary * beta_bar) ** 2 / lapse
        - lapse
        * (
            2.0 * beta_bar**2 / a**2
            + beta_bar**2 * jacobian**2 / a**2
        )
    )
    transverse_mixed_extrinsic = state.h / lapse
    longitudinal_mixed_extrinsic = (
        state.h - fields.flat_shift_covariant_derivative / a**2
    ) / lapse
    extrinsic_combination = (
        2.0 * transverse_mixed_extrinsic**2
        + longitudinal_mixed_extrinsic**2
        - (
            2.0 * transverse_mixed_extrinsic
            + longitudinal_mixed_extrinsic
        )
        ** 2
    )
    lam = float(parameters.lambda_over_mu_x_squared)
    gravity = 0.5 * m**2 * np.mean(
        lapse * (extrinsic_combination - 2.0 * lam)
    )
    background = (
        0.5 * state.u**2
        - 1.5 * beta_bar**2 / a**2
        + 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    )
    return float(clock + rods + gravity - background)


def mixed_static_cubic_coefficient(
    action,
    *,
    epsilon: float,
) -> float:
    '''Return d^3 L/(d r_k^2 d r_2k) at the origin by central differences.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('epsilon must be finite and positive')

    def value(first: float, second: float) -> float:
        return float(action(np.array([first, second], dtype=float)))

    positive_second = (
        value(epsilon, epsilon)
        - 2.0 * value(0.0, epsilon)
        + value(-epsilon, epsilon)
    )
    negative_second = (
        value(epsilon, -epsilon)
        - 2.0 * value(0.0, -epsilon)
        + value(-epsilon, -epsilon)
    )
    return float(
        (positive_second - negative_second) / (2.0 * epsilon**3)
    )


def static_scalar_cubic_tensor(
    action,
    *,
    epsilon: float,
) -> np.ndarray:
    '''Return T_abc for two k legs and one 2k leg, a,b,c in (q,r).'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('epsilon must be finite and positive')

    def value(first: np.ndarray, second: np.ndarray) -> float:
        return float(action(np.stack((first, second))))

    tensor = np.zeros((2, 2, 2))
    for first_index in range(2):
        for second_index in range(2):
            for third_index in range(2):
                first_direction = np.eye(2)[first_index]
                second_direction = np.eye(2)[second_index]
                third_direction = np.eye(2)[third_index]
                if first_index == second_index:
                    positive = (
                        value(epsilon * first_direction, epsilon * third_direction)
                        - 2.0 * value(np.zeros(2), epsilon * third_direction)
                        + value(-epsilon * first_direction, epsilon * third_direction)
                    )
                    negative = (
                        value(epsilon * first_direction, -epsilon * third_direction)
                        - 2.0 * value(np.zeros(2), -epsilon * third_direction)
                        + value(-epsilon * first_direction, -epsilon * third_direction)
                    )
                    coefficient = (positive - negative) / (2.0 * epsilon**3)
                else:
                    coefficient = 0.0
                    for first_sign in (-1.0, 1.0):
                        for second_sign in (-1.0, 1.0):
                            for third_sign in (-1.0, 1.0):
                                first_mode = epsilon * (
                                    first_sign * first_direction
                                    + second_sign * second_direction
                                )
                                second_mode = (
                                    epsilon * third_sign * third_direction
                                )
                                coefficient += (
                                    first_sign
                                    * second_sign
                                    * third_sign
                                    * value(first_mode, second_mode)
                                )
                    coefficient /= 8.0 * epsilon**3
                tensor[first_index, second_index, third_index] = coefficient
    return tensor


def _positive_inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    if np.min(eigenvalues) <= TOL:
        raise ValueError('canonical cubic normalization requires positive kinetic matrix')
    return eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T


def canonically_normalized_static_scalar_tensor(
    tensor: np.ndarray,
    first_kinetic: np.ndarray,
    second_kinetic: np.ndarray,
    *,
    scale_factor: float,
) -> np.ndarray:
    first_inverse_root = _positive_inverse_square_root(first_kinetic)
    second_inverse_root = _positive_inverse_square_root(second_kinetic)
    return np.einsum(
        'ijk,ia,jb,kc->abc',
        np.asarray(tensor, dtype=float),
        first_inverse_root,
        first_inverse_root,
        second_inverse_root,
    ) / float(scale_factor) ** 1.5


@dataclass(frozen=True)
class CubicPowerCountingAudit:
    background_gradient_expansion_parameter_squared: float
    curvature_expansion_parameter_squared: float
    supplied_gravity_cutoff_over_mu_x: float
    covariant_power_counting_small: bool
    reduced_low_k_cutoff_derived: bool


def audit_covariant_cubic_power_counting(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
) -> CubicPowerCountingAudit:
    m = float(parameters.m_planck_over_mu_x)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    gradient = (state.u**2 + state.b**2) / m**2
    curvature = state.h**2 / m**2
    return CubicPowerCountingAudit(
        background_gradient_expansion_parameter_squared=float(gradient),
        curvature_expansion_parameter_squared=float(curvature),
        supplied_gravity_cutoff_over_mu_x=m,
        covariant_power_counting_small=(gradient < 1.0 and curvature < 1.0),
        reduced_low_k_cutoff_derived=False,
    )


@dataclass(frozen=True)
class StaticCubicTriadResult:
    base_wavenumber_bar: float
    flat_cubic_coefficients: tuple[float, ...]
    unitary_cubic_coefficients: tuple[float, ...]
    flat_refinement_spread: float
    gauge_relative_residual: float
    q_fixed_coordinate_direction_proxy: float
    coordinate_measure_negative_control_residual: float
    static_two_gauge_gate_passed: bool


@dataclass(frozen=True)
class StaticScalarTensorResult:
    base_wavenumber_bar: float
    flat_canonical_tensor_norm: float
    unitary_canonical_tensor_norm: float
    tensor_refinement_spread: float
    tensor_gauge_relative_residual: float
    first_leg_permutation_residual: float
    static_scalar_tensor_gate_passed: bool


@dataclass(frozen=True)
class CubicAdmissionAudit:
    single_mode_cubic_overlap: float
    momentum_conserving_triad_overlap: float
    power_counting: CubicPowerCountingAudit
    triad_results: tuple[StaticCubicTriadResult, ...]
    scalar_tensor_results: tuple[StaticScalarTensorResult, ...]
    static_off_shell_two_gauge_gate_passed: bool
    complete_static_qr_triad_tensor_computed: bool
    on_shell_cubic_residue_computed: bool
    all_scalar_vector_tensor_vertices_computed: bool
    second_order_constraint_and_gauge_completion_computed: bool
    physical_strong_coupling_scale_derived: bool
    one_loop_st_identity_computed: bool
    brst_physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_static_cubic_triad_precursor(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
) -> CubicAdmissionAudit:
    '''Run the restricted E68 static two-gauge triad precursor.'''

    single_overlap, triad_overlap = normalized_cubic_profile_overlaps()
    results = []
    scalar_tensor_results = []
    a = float(np.exp(state.n))
    for base in E68_BASE_WAVENUMBERS_BAR:
        flat_action = lambda amplitudes, base=base: (
            flat_gauge_static_triad_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base,
                rod_amplitudes=amplitudes,
            )
        )
        unitary_action = lambda amplitudes, base=base: (
            rod_unitary_static_triad_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base,
                rod_amplitudes=amplitudes,
            )
        )
        wrong_measure_action = lambda amplitudes, base=base: (
            rod_unitary_static_triad_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base,
                rod_amplitudes=amplitudes,
                include_coordinate_measure=False,
            )
        )
        flat_coefficients = tuple(
            mixed_static_cubic_coefficient(flat_action, epsilon=epsilon)
            for epsilon in E68_CUBIC_STEPS
        )
        unitary_coefficients = tuple(
            mixed_static_cubic_coefficient(unitary_action, epsilon=epsilon)
            for epsilon in E68_CUBIC_STEPS
        )
        wrong_coefficient = mixed_static_cubic_coefficient(
            wrong_measure_action,
            epsilon=E68_CUBIC_STEPS[-1],
        )
        scale = max(1.0, abs(flat_coefficients[-1]))
        refinement = abs(flat_coefficients[-1] - flat_coefficients[-2]) / scale
        gauge_residual = abs(
            flat_coefficients[-1] - unitary_coefficients[-1]
        ) / scale
        wrong_residual = abs(wrong_coefficient - flat_coefficients[-1]) / scale
        first_blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=base,
        )
        second_blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=2.0 * base,
        )
        first_kinetic, _, _ = reduced_scalar_matrices(first_blocks)
        second_kinetic, _, _ = reduced_scalar_matrices(second_blocks)
        first_norm = first_kinetic[1, 1]
        second_norm = second_kinetic[1, 1]
        q_fixed_proxy = flat_coefficients[-1] / (
            a**1.5 * first_norm * np.sqrt(second_norm)
        )
        passed = (
            refinement < E68_REFINEMENT_TOL
            and gauge_residual < E68_GAUGE_TOL
            and wrong_residual > E68_GAUGE_TOL
        )
        results.append(
            StaticCubicTriadResult(
                base_wavenumber_bar=base,
                flat_cubic_coefficients=flat_coefficients,
                unitary_cubic_coefficients=unitary_coefficients,
                flat_refinement_spread=float(refinement),
                gauge_relative_residual=float(gauge_residual),
                q_fixed_coordinate_direction_proxy=float(q_fixed_proxy),
                coordinate_measure_negative_control_residual=float(wrong_residual),
                static_two_gauge_gate_passed=passed,
            )
        )
        flat_scalar_action = lambda mode_amplitudes, base=base: (
            flat_gauge_static_scalar_triad_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base,
                mode_amplitudes=mode_amplitudes,
            )
        )
        unitary_scalar_action = lambda mode_amplitudes, base=base: (
            rod_unitary_static_scalar_triad_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base,
                mode_amplitudes=mode_amplitudes,
            )
        )
        flat_tensors = tuple(
            static_scalar_cubic_tensor(flat_scalar_action, epsilon=epsilon)
            for epsilon in E68_CUBIC_STEPS
        )
        unitary_tensors = tuple(
            static_scalar_cubic_tensor(unitary_scalar_action, epsilon=epsilon)
            for epsilon in E68_CUBIC_STEPS
        )
        first_blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=base,
        )
        second_blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=2.0 * base,
        )
        first_kinetic, _, _ = reduced_scalar_matrices(first_blocks)
        second_kinetic, _, _ = reduced_scalar_matrices(second_blocks)
        flat_canonical = canonically_normalized_static_scalar_tensor(
            flat_tensors[-1],
            first_kinetic,
            second_kinetic,
            scale_factor=a,
        )
        unitary_canonical = canonically_normalized_static_scalar_tensor(
            unitary_tensors[-1],
            first_kinetic,
            second_kinetic,
            scale_factor=a,
        )
        tensor_scale = max(1.0, float(np.linalg.norm(flat_canonical)))
        tensor_refinement = float(
            np.linalg.norm(flat_tensors[-1] - flat_tensors[-2])
            / max(1.0, np.linalg.norm(flat_tensors[-1]))
        )
        tensor_gauge = float(
            np.linalg.norm(flat_canonical - unitary_canonical) / tensor_scale
        )
        permutation = float(
            np.linalg.norm(flat_tensors[-1] - np.swapaxes(flat_tensors[-1], 0, 1))
            / max(1.0, np.linalg.norm(flat_tensors[-1]))
        )
        tensor_passed = (
            tensor_refinement < E68_REFINEMENT_TOL
            and tensor_gauge < E68_GAUGE_TOL
            and permutation < E68_GAUGE_TOL
        )
        scalar_tensor_results.append(
            StaticScalarTensorResult(
                base_wavenumber_bar=base,
                flat_canonical_tensor_norm=float(np.linalg.norm(flat_canonical)),
                unitary_canonical_tensor_norm=float(
                    np.linalg.norm(unitary_canonical)
                ),
                tensor_refinement_spread=tensor_refinement,
                tensor_gauge_relative_residual=tensor_gauge,
                first_leg_permutation_residual=permutation,
                static_scalar_tensor_gate_passed=tensor_passed,
            )
        )
    static_passed = (
        all(result.static_two_gauge_gate_passed for result in results)
        and all(
            result.static_scalar_tensor_gate_passed
            for result in scalar_tensor_results
        )
    )
    return CubicAdmissionAudit(
        single_mode_cubic_overlap=single_overlap,
        momentum_conserving_triad_overlap=triad_overlap,
        power_counting=audit_covariant_cubic_power_counting(state, parameters),
        triad_results=tuple(results),
        scalar_tensor_results=tuple(scalar_tensor_results),
        static_off_shell_two_gauge_gate_passed=static_passed,
        complete_static_qr_triad_tensor_computed=True,
        on_shell_cubic_residue_computed=False,
        all_scalar_vector_tensor_vertices_computed=False,
        second_order_constraint_and_gauge_completion_computed=False,
        physical_strong_coupling_scale_derived=False,
        one_loop_st_identity_computed=False,
        brst_physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=(
            'STATIC_OFF_SHELL_TWO_GAUGE_PRECURSOR_PASSED'
            if static_passed
            else 'STATIC_OFF_SHELL_TWO_GAUGE_PRECURSOR_FAILED'
        ),
    )
