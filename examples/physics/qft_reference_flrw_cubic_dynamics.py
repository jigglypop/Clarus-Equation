'''Frozen-event dynamic scalar cubic completion on the E62 FLRW background.

The physical jet for each of the k and 2k harmonics is ordered as
(qdot, rdot, q, r), with the E66 convention r=k s.  Lapse and longitudinal
shift are nondynamical.  Their projected nonlinear equations retain lapse
cosines 0..4 and zero-mean shift sines 1..4, exactly the harmonics sourced at
second order by a (k,k,-2k) scalar triad.

This module only constructs a local frozen-background cubic diagnostic.  It
does not construct an asymptotic S-matrix or derive a strong-coupling scale.
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
    frozen_scalar_frequency_squared,
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


TOL = 1.0e-10
DYNAMIC_LAPSE_HARMONICS = (0, 1, 2, 3, 4)
DYNAMIC_SHIFT_HARMONICS = (1, 2, 3, 4)
DYNAMIC_CONSTRAINT_SIZE = 9


def _validate_dynamic_inputs(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    constraint_coefficients: np.ndarray,
    phase_points: int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('dynamic E68 requires an admitted E62 reference patch')
    base = float(base_wavenumber_bar)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError('the dynamic cubic triad requires positive base momentum')
    modes = np.asarray(physical_modes, dtype=float)
    if modes.shape != (2, 4) or not np.all(np.isfinite(modes)):
        raise ValueError('physical_modes must have shape (k,2k) by (qdot,rdot,q,r)')
    constraints = np.asarray(constraint_coefficients, dtype=float)
    if constraints.shape != (DYNAMIC_CONSTRAINT_SIZE,) or not np.all(
        np.isfinite(constraints)
    ):
        raise ValueError('dynamic constraint coefficients must be a finite nine-vector')
    if not isinstance(phase_points, int) or phase_points < 32:
        raise ValueError('phase_points must be an integer of at least thirty-two')
    a = float(np.exp(state.n))
    beta_bar = float(rod_charge_bar(state))
    m = float(parameters.m_planck_over_mu_x)
    if a <= 0.0 or abs(beta_bar) <= TOL or not np.isfinite(m) or m <= 0.0:
        raise ValueError('dynamic E68 requires positive a,m and nonzero beta')
    return a, beta_bar, m, base, modes, constraints


def linear_dynamic_constraint_coefficients(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
) -> np.ndarray:
    '''Embed the two E66 linear lapse/shift solutions in the 0..4 basis.'''

    modes = np.asarray(physical_modes, dtype=float)
    if modes.shape != (2, 4) or not np.all(np.isfinite(modes)):
        raise ValueError('physical_modes must have shape (2,4)')
    coefficients = np.zeros(DYNAMIC_CONSTRAINT_SIZE)
    for harmonic, mode in enumerate(modes, start=1):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=harmonic * float(base_wavenumber_bar),
        )
        source = (
            blocks.velocity_coupling_bar @ mode[:2]
            + blocks.field_coupling_bar @ mode[2:]
        )
        alpha, theta = -np.linalg.solve(blocks.constraint_matrix_bar, source)
        coefficients[harmonic] = alpha
        coefficients[5 + harmonic - 1] = theta
    return coefficients


@dataclass(frozen=True)
class DynamicScalarFields:
    lapse: np.ndarray
    lapse_derivative: np.ndarray
    shift_contravariant: np.ndarray
    shift_derivative: np.ndarray
    clock_velocity: np.ndarray
    clock_gradient: np.ndarray
    rod_velocity: np.ndarray
    rod_velocity_derivative: np.ndarray
    rod_gradient: np.ndarray
    rod_gradient_derivative: np.ndarray
    scale_factor: float
    rod_background_gradient: float
    m_planck_over_mu_x: float


def dynamic_scalar_fields(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    constraint_coefficients: np.ndarray,
    phase_points: int = 256,
) -> DynamicScalarFields:
    '''Build the exact frozen scalar jet in the E66 flat scalar gauge.'''

    a, beta_bar, m, base, modes, constraints = _validate_dynamic_inputs(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
        constraint_coefficients=constraint_coefficients,
        phase_points=phase_points,
    )
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    lapse = np.ones(phase_points) + constraints[0]
    lapse_derivative = np.zeros(phase_points)
    shift = np.zeros(phase_points)
    shift_derivative = np.zeros(phase_points)
    for harmonic in DYNAMIC_SHIFT_HARMONICS:
        wavenumber = harmonic * base
        cosine = np.sqrt(2.0) * np.cos(harmonic * phase)
        sine = np.sqrt(2.0) * np.sin(harmonic * phase)
        alpha = constraints[harmonic]
        theta = constraints[5 + harmonic - 1]
        lapse += alpha * cosine
        lapse_derivative += -alpha * wavenumber * sine
        shift += -theta * sine / beta_bar
        shift_derivative += -theta * wavenumber * cosine / beta_bar

    clock_velocity = np.zeros(phase_points)
    clock_gradient = np.zeros(phase_points)
    rod_velocity = np.zeros(phase_points)
    rod_velocity_derivative = np.zeros(phase_points)
    rod_gradient = np.full(phase_points, beta_bar)
    rod_gradient_derivative = np.zeros(phase_points)
    for harmonic, mode in enumerate(modes, start=1):
        q_velocity, r_velocity, q_amplitude, r_amplitude = mode
        wavenumber = harmonic * base
        cosine = np.sqrt(2.0) * np.cos(harmonic * phase)
        sine = np.sqrt(2.0) * np.sin(harmonic * phase)
        clock_velocity += q_velocity * cosine
        clock_gradient += -q_amplitude * wavenumber * sine
        # r=k s: delta X=d s has amplitude r and sine phase.
        rod_velocity += -r_velocity * sine
        rod_velocity_derivative += -r_velocity * wavenumber * cosine
        rod_gradient += -r_amplitude * wavenumber * cosine
        rod_gradient_derivative += r_amplitude * wavenumber**2 * sine
    if np.min(lapse) <= 0.0:
        raise ValueError('the sampled dynamic lapse must stay positive')
    if np.min(rod_gradient / beta_bar) <= 0.0:
        raise ValueError('the dynamic rod-unitary map must stay invertible')
    return DynamicScalarFields(
        lapse=lapse,
        lapse_derivative=lapse_derivative,
        shift_contravariant=shift,
        shift_derivative=shift_derivative,
        clock_velocity=clock_velocity,
        clock_gradient=clock_gradient,
        rod_velocity=rod_velocity,
        rod_velocity_derivative=rod_velocity_derivative,
        rod_gradient=rod_gradient,
        rod_gradient_derivative=rod_gradient_derivative,
        scale_factor=a,
        rod_background_gradient=beta_bar,
        m_planck_over_mu_x=m,
    )


def _background_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    scale_factor: float,
    beta_bar: float,
) -> float:
    m = float(parameters.m_planck_over_mu_x)
    lam = float(parameters.lambda_over_mu_x_squared)
    return float(
        0.5 * state.u**2
        - 1.5 * beta_bar**2 / scale_factor**2
        + 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    )


def flat_gauge_dynamic_scalar_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    constraint_coefficients: np.ndarray,
    phase_points: int = 256,
) -> np.longdouble:
    '''Evaluate the exact frozen dynamic scalar ADM functional in flat gauge.'''

    fields = dynamic_scalar_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
        constraint_coefficients=constraint_coefficients,
        phase_points=phase_points,
    )
    # Subtract the background pointwise before the periodic sum.  Cubic finite
    # differences otherwise recover an O(epsilon^3) number by cancelling
    # several O(1) averaged terms and lose avoidable digits.
    n = np.asarray(fields.lapse, dtype=np.longdouble)
    shift = np.asarray(fields.shift_contravariant, dtype=np.longdouble)
    clock_velocity = np.asarray(fields.clock_velocity, dtype=np.longdouble)
    clock_gradient = np.asarray(fields.clock_gradient, dtype=np.longdouble)
    rod_velocity = np.asarray(fields.rod_velocity, dtype=np.longdouble)
    rod_gradient = np.asarray(fields.rod_gradient, dtype=np.longdouble)
    shift_derivative = np.asarray(fields.shift_derivative, dtype=np.longdouble)
    a = np.longdouble(fields.scale_factor)
    beta_bar = np.longdouble(fields.rod_background_gradient)
    m = np.longdouble(fields.m_planck_over_mu_x)
    u = np.longdouble(state.u)
    h = np.longdouble(state.h)
    lam = np.longdouble(parameters.lambda_over_mu_x_squared)
    clock_convective = u + clock_velocity - shift * clock_gradient
    rod_convective = rod_velocity - shift * rod_gradient
    density = (
        np.longdouble(0.5)
        * ((clock_convective**2 + rod_convective**2) / n - u**2)
        - np.longdouble(0.5)
        * (
            n * (clock_gradient**2 + 2.0 * beta_bar**2 + rod_gradient**2)
            - 3.0 * beta_bar**2
        )
        / a**2
        + np.longdouble(0.5)
        * m**2
        * (
            (-6.0 * h**2 + 4.0 * h * shift_derivative) / n
            - 2.0 * lam * n
            + 6.0 * h**2
            + 2.0 * lam
        )
    )
    return np.mean(density, dtype=np.longdouble)


def rod_unitary_dynamic_scalar_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    constraint_coefficients: np.ndarray,
    phase_points: int = 256,
    include_coordinate_time_shift: bool = True,
) -> np.longdouble:
    '''Evaluate the exact time-dependent spatial pullback in rod-unitary gauge.'''

    fields = dynamic_scalar_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
        constraint_coefficients=constraint_coefficients,
        phase_points=phase_points,
    )
    n = np.asarray(fields.lapse, dtype=np.longdouble)
    a = np.longdouble(fields.scale_factor)
    beta_bar = np.longdouble(fields.rod_background_gradient)
    m = np.longdouble(fields.m_planck_over_mu_x)
    shift_flat = np.asarray(fields.shift_contravariant, dtype=np.longdouble)
    shift_flat_derivative = np.asarray(fields.shift_derivative, dtype=np.longdouble)
    clock_velocity = np.asarray(fields.clock_velocity, dtype=np.longdouble)
    clock_gradient = np.asarray(fields.clock_gradient, dtype=np.longdouble)
    rod_gradient = np.asarray(fields.rod_gradient, dtype=np.longdouble)
    rod_gradient_derivative = np.asarray(
        fields.rod_gradient_derivative, dtype=np.longdouble
    )
    rod_velocity = np.asarray(fields.rod_velocity, dtype=np.longdouble)
    rod_velocity_derivative = np.asarray(
        fields.rod_velocity_derivative, dtype=np.longdouble
    )
    jacobian = rod_gradient / beta_bar
    jacobian_derivative = rod_gradient_derivative / beta_bar
    coordinate_velocity = rod_velocity / beta_bar
    coordinate_velocity_derivative = rod_velocity_derivative / beta_bar
    if include_coordinate_time_shift:
        mapped_velocity = coordinate_velocity
        mapped_velocity_derivative = coordinate_velocity_derivative
    else:
        mapped_velocity = np.zeros_like(coordinate_velocity)
        mapped_velocity_derivative = np.zeros_like(coordinate_velocity_derivative)
    unitary_shift = jacobian * shift_flat - mapped_velocity
    clock_time_at_fixed_y = (
        np.longdouble(state.u)
        + clock_velocity
        - mapped_velocity * clock_gradient / jacobian
    )
    clock_gradient_y = clock_gradient / jacobian
    clock_convective = clock_time_at_fixed_y - unitary_shift * clock_gradient_y
    inverse_longitudinal_metric = jacobian**2 / a**2
    clock_density = np.longdouble(0.5) * (
        clock_convective**2 / n
        - n * inverse_longitudinal_metric * clock_gradient_y**2
        - np.longdouble(state.u) ** 2
    )
    rod_density = np.longdouble(0.5) * (
        (beta_bar * unitary_shift) ** 2 / n
        - n
        * (
            2.0 * beta_bar**2 / a**2
            + beta_bar**2 * inverse_longitudinal_metric
        )
        + 3.0 * beta_bar**2 / a**2
    )

    unitary_shift_derivative_x = (
        jacobian_derivative * shift_flat
        + jacobian * shift_flat_derivative
        - mapped_velocity_derivative
    )
    metric_time_at_fixed_y = (
        2.0 * a**2 * np.longdouble(state.h) / jacobian**2
        - 2.0 * a**2 * mapped_velocity_derivative / jacobian**3
        + 2.0
        * a**2
        * mapped_velocity
        * jacobian_derivative
        / jacobian**4
    )
    covariant_shift_derivative_y = a**2 * (
        unitary_shift_derivative_x / jacobian**3
        - unitary_shift * jacobian_derivative / jacobian**4
    )
    longitudinal_extrinsic_covariant = (
        metric_time_at_fixed_y - 2.0 * covariant_shift_derivative_y
    ) / (2.0 * n)
    expanded_longitudinal_extrinsic_mixed = (
        inverse_longitudinal_metric * longitudinal_extrinsic_covariant
    )
    if include_coordinate_time_shift:
        # The expanded time-dependent pullback above independently checks all
        # Ydot, Jdot and Christoffel terms.  Once checked, use the exact tensor
        # transformation identity for the action so a cubic finite difference
        # does not amplify pointwise cancellation noise.
        transformed_longitudinal_extrinsic_mixed = (
            np.longdouble(state.h) - shift_flat_derivative
        ) / n
        pullback_residual = np.max(
            np.abs(
                expanded_longitudinal_extrinsic_mixed
                - transformed_longitudinal_extrinsic_mixed
            )
        )
        if pullback_residual > np.longdouble(1.0e-10):
            raise ValueError('the dynamic extrinsic-curvature pullback identity failed')
        longitudinal_extrinsic_mixed = transformed_longitudinal_extrinsic_mixed
    else:
        longitudinal_extrinsic_mixed = expanded_longitudinal_extrinsic_mixed
    transverse_extrinsic_mixed = np.longdouble(state.h) / n
    extrinsic_combination = (
        2.0 * transverse_extrinsic_mixed**2
        + longitudinal_extrinsic_mixed**2
        - (2.0 * transverse_extrinsic_mixed + longitudinal_extrinsic_mixed) ** 2
    )
    lam = np.longdouble(parameters.lambda_over_mu_x_squared)
    gravity_density = np.longdouble(0.5) * m**2 * (
        n * (extrinsic_combination - 2.0 * lam)
        + 6.0 * np.longdouble(state.h) ** 2
        + 2.0 * lam
    )
    # dy sqrt(h_yy) = dx J (a/J) = dx a, so the x-sampled measure has no
    # additional Jacobian weight.
    return np.mean(
        clock_density + rod_density + gravity_density,
        dtype=np.longdouble,
    )


def projected_dynamic_constraint_residual(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    constraint_coefficients: np.ndarray,
    phase_points: int = 256,
) -> np.ndarray:
    '''Project the exact nonlinear Hamiltonian/momentum equations on 0..4.'''

    fields = dynamic_scalar_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
        constraint_coefficients=constraint_coefficients,
        phase_points=phase_points,
    )
    n = fields.lapse
    shift = fields.shift_contravariant
    a = fields.scale_factor
    beta_bar = fields.rod_background_gradient
    m = fields.m_planck_over_mu_x
    clock_convective = state.u + fields.clock_velocity - shift * fields.clock_gradient
    rod_convective = fields.rod_velocity - shift * fields.rod_gradient
    lam = float(parameters.lambda_over_mu_x_squared)
    hamiltonian = (
        -0.5 * (clock_convective**2 + rod_convective**2) / n**2
        - 0.5
        * (
            fields.clock_gradient**2
            + 2.0 * beta_bar**2
            + fields.rod_gradient**2
        )
        / a**2
        + m**2
        * (
            (3.0 * state.h**2 - 2.0 * state.h * fields.shift_derivative)
            / n**2
            - lam
        )
    )
    momentum = (
        -(fields.clock_gradient * clock_convective + fields.rod_gradient * rod_convective)
        / n
        + 2.0 * m**2 * state.h * fields.lapse_derivative / n**2
    )
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    residual = np.zeros(DYNAMIC_CONSTRAINT_SIZE)
    residual[0] = np.mean(hamiltonian)
    for harmonic in DYNAMIC_SHIFT_HARMONICS:
        cosine = np.sqrt(2.0) * np.cos(harmonic * phase)
        sine = np.sqrt(2.0) * np.sin(harmonic * phase)
        residual[harmonic] = np.mean(hamiltonian * cosine)
        residual[5 + harmonic - 1] = np.mean(
            momentum * (-sine / beta_bar)
        )
    return residual


@dataclass(frozen=True)
class DynamicConstraintSolution:
    coefficients: np.ndarray
    projected_residual: np.ndarray
    iterations: int
    converged: bool


def solve_projected_dynamic_constraints(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    phase_points: int = 256,
    residual_tolerance: float = 1.0e-12,
    max_iterations: int = 16,
) -> DynamicConstraintSolution:
    '''Newton-solve the exact nine projected nondynamical equations.'''

    coefficients = linear_dynamic_constraint_coefficients(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
    )
    for iteration in range(max_iterations + 1):
        residual = projected_dynamic_constraint_residual(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            physical_modes=physical_modes,
            constraint_coefficients=coefficients,
            phase_points=phase_points,
        )
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= residual_tolerance:
            return DynamicConstraintSolution(
                coefficients=coefficients.copy(),
                projected_residual=residual,
                iterations=iteration,
                converged=True,
            )
        jacobian = np.zeros((DYNAMIC_CONSTRAINT_SIZE, DYNAMIC_CONSTRAINT_SIZE))
        for column in range(DYNAMIC_CONSTRAINT_SIZE):
            step = 1.0e-6 * max(1.0, abs(coefficients[column]))
            direction = np.zeros(DYNAMIC_CONSTRAINT_SIZE)
            direction[column] = step
            positive = projected_dynamic_constraint_residual(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                physical_modes=physical_modes,
                constraint_coefficients=coefficients + direction,
                phase_points=phase_points,
            )
            negative = projected_dynamic_constraint_residual(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                physical_modes=physical_modes,
                constraint_coefficients=coefficients - direction,
                phase_points=phase_points,
            )
            jacobian[:, column] = (positive - negative) / (2.0 * step)
        update = np.linalg.solve(jacobian, -residual)
        accepted = False
        scale = 1.0
        for _ in range(12):
            candidate = coefficients + scale * update
            try:
                candidate_residual = projected_dynamic_constraint_residual(
                    state,
                    parameters,
                    base_wavenumber_bar=base_wavenumber_bar,
                    physical_modes=physical_modes,
                    constraint_coefficients=candidate,
                    phase_points=phase_points,
                )
            except ValueError:
                scale *= 0.5
                continue
            if np.max(np.abs(candidate_residual)) < residual_norm:
                coefficients = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    residual = projected_dynamic_constraint_residual(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical_modes,
        constraint_coefficients=coefficients,
        phase_points=phase_points,
    )
    return DynamicConstraintSolution(
        coefficients=coefficients.copy(),
        projected_residual=residual,
        iterations=max_iterations,
        converged=False,
    )


def _physical_vector_to_modes(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (8,) or not np.all(np.isfinite(vector)):
        raise ValueError('the flattened dynamic physical jet must be a finite eight-vector')
    return np.stack((vector[:4], vector[4:]))


def second_order_dynamic_constraint_tensor(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    epsilon: float,
    phase_points: int = 256,
) -> np.ndarray:
    '''Return d^2 c_I/d z_A d z_B for the exact projected constraint solution.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('the constraint-tensor step must be finite and positive')
    cache: dict[tuple[float, ...], np.ndarray] = {}

    def solved(vector: np.ndarray) -> np.ndarray:
        key = tuple(float(value) for value in vector)
        if key not in cache:
            solution = solve_projected_dynamic_constraints(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                physical_modes=_physical_vector_to_modes(vector),
                phase_points=phase_points,
                residual_tolerance=1.0e-13,
            )
            if not solution.converged:
                raise ValueError('the projected dynamic constraint solve did not converge')
            cache[key] = solution.coefficients
        return cache[key]

    origin = np.zeros(8)
    origin_solution = solved(origin)
    tensor = np.zeros((DYNAMIC_CONSTRAINT_SIZE, 8, 8))
    for first in range(8):
        first_direction = np.eye(8)[first]
        tensor[:, first, first] = (
            solved(epsilon * first_direction)
            - 2.0 * origin_solution
            + solved(-epsilon * first_direction)
        ) / epsilon**2
        for second in range(first + 1, 8):
            second_direction = np.eye(8)[second]
            coefficient = (
                solved(epsilon * (first_direction + second_direction))
                - solved(epsilon * (first_direction - second_direction))
                - solved(epsilon * (-first_direction + second_direction))
                + solved(-epsilon * (first_direction + second_direction))
            ) / (4.0 * epsilon**2)
            tensor[:, first, second] = coefficient
            tensor[:, second, first] = coefficient
    return tensor


def dynamic_reduced_scalar_cubic_tensor_pair(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    epsilon: float,
    phase_points: int = 256,
    constraint_scheme: str = 'nonlinear',
) -> tuple[np.ndarray, np.ndarray]:
    '''Return flat/unitary T_ab;c on z=(qdot,rdot,q,r) for k,k,2k.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('the dynamic cubic step must be finite and positive')
    if constraint_scheme not in {'nonlinear', 'linear', 'zero'}:
        raise ValueError('constraint_scheme must be nonlinear, linear, or zero')
    cache: dict[tuple[float, ...], np.ndarray] = {}

    def values(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        key = tuple(float(value) for value in np.concatenate((first, second)))
        if key not in cache:
            physical = np.stack((first, second))
            if constraint_scheme == 'nonlinear':
                solution = solve_projected_dynamic_constraints(
                    state,
                    parameters,
                    base_wavenumber_bar=base_wavenumber_bar,
                    physical_modes=physical,
                    phase_points=phase_points,
                    residual_tolerance=1.0e-13,
                )
                if not solution.converged:
                    raise ValueError('the nonlinear cubic constraint solve did not converge')
                constraints = solution.coefficients
            elif constraint_scheme == 'linear':
                constraints = linear_dynamic_constraint_coefficients(
                    state,
                    parameters,
                    base_wavenumber_bar=base_wavenumber_bar,
                    physical_modes=physical,
                )
            else:
                constraints = np.zeros(DYNAMIC_CONSTRAINT_SIZE)
            flat = flat_gauge_dynamic_scalar_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                physical_modes=physical,
                constraint_coefficients=constraints,
                phase_points=phase_points,
            )
            unitary = rod_unitary_dynamic_scalar_lagrangian_bar_per_a3(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                physical_modes=physical,
                constraint_coefficients=constraints,
                phase_points=phase_points,
            )
            cache[key] = np.array([flat, unitary], dtype=np.longdouble)
        return cache[key]

    tensors = np.zeros((2, 4, 4, 4), dtype=np.longdouble)
    basis = np.eye(4)
    zero = np.zeros(4)
    for first_index in range(4):
        for second_index in range(4):
            for third_index in range(4):
                first_direction = basis[first_index]
                second_direction = basis[second_index]
                third_direction = basis[third_index]
                if first_index == second_index:
                    positive = (
                        values(epsilon * first_direction, epsilon * third_direction)
                        - 2.0 * values(zero, epsilon * third_direction)
                        + values(-epsilon * first_direction, epsilon * third_direction)
                    )
                    negative = (
                        values(epsilon * first_direction, -epsilon * third_direction)
                        - 2.0 * values(zero, -epsilon * third_direction)
                        + values(-epsilon * first_direction, -epsilon * third_direction)
                    )
                    coefficient = (positive - negative) / (2.0 * epsilon**3)
                else:
                    coefficient = np.zeros(2, dtype=np.longdouble)
                    for first_sign in (-1.0, 1.0):
                        for second_sign in (-1.0, 1.0):
                            for third_sign in (-1.0, 1.0):
                                first_mode = epsilon * (
                                    first_sign * first_direction
                                    + second_sign * second_direction
                                )
                                second_mode = epsilon * third_sign * third_direction
                                coefficient += (
                                    first_sign
                                    * second_sign
                                    * third_sign
                                    * values(first_mode, second_mode)
                                )
                    coefficient /= 8.0 * epsilon**3
                tensors[:, first_index, second_index, third_index] = coefficient
    return np.asarray(tensors[0], dtype=float), np.asarray(tensors[1], dtype=float)


@dataclass(frozen=True)
class ScalarCanonicalPhaseSpaceMap:
    '''Quadratic map from W=(p_q,p_r,q,r) to Z=(qdot,rdot,q,r).'''

    matrix: np.ndarray
    kinetic_eigenvalues: np.ndarray
    determinant: float


def canonical_scalar_phase_space_map(
    kinetic: np.ndarray,
    gyroscopic: np.ndarray,
    *,
    scale_factor: float,
    tol: float = TOL,
) -> ScalarCanonicalPhaseSpaceMap:
    '''Construct the invertible quadratic Legendre map for one harmonic.'''

    kinetic = np.asarray(kinetic, dtype=float)
    gyroscopic = np.asarray(gyroscopic, dtype=float)
    a = float(scale_factor)
    if kinetic.shape != (2, 2) or gyroscopic.shape != (2, 2):
        raise ValueError('the scalar phase-space map requires two-by-two K and R')
    if not np.all(np.isfinite(kinetic)) or not np.all(np.isfinite(gyroscopic)):
        raise ValueError('the scalar phase-space matrices must be finite')
    if not np.isfinite(a) or a <= 0.0 or not np.isfinite(tol) or tol <= 0.0:
        raise ValueError('the scalar phase-space map requires positive a and tolerance')
    if not np.allclose(kinetic, kinetic.T, rtol=0.0, atol=tol):
        raise ValueError('the scalar kinetic matrix must be symmetric')
    eigenvalues = np.linalg.eigvalsh(kinetic)
    if np.min(eigenvalues) <= tol:
        raise ValueError('the scalar kinetic matrix must be positive definite')
    inverse = np.linalg.inv(kinetic)
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = inverse / a**3
    matrix[:2, 2:] = inverse @ gyroscopic
    matrix[2:, 2:] = np.eye(2)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= tol:
        raise ValueError('the scalar quadratic Legendre map is singular')
    return ScalarCanonicalPhaseSpaceMap(
        matrix=matrix,
        kinetic_eigenvalues=eigenvalues,
        determinant=determinant,
    )


def harmonic_scalar_phase_space_map(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
) -> ScalarCanonicalPhaseSpaceMap:
    '''Return the E66 quadratic Legendre map at one comoving momentum.'''

    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic, gyroscopic, _ = reduced_scalar_matrices(blocks)
    return canonical_scalar_phase_space_map(
        kinetic,
        gyroscopic,
        scale_factor=np.exp(state.n),
    )


def scalar_interaction_hamiltonian_cubic_tensor_pair(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    flat_lagrangian_tensor: np.ndarray,
    unitary_lagrangian_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Legendre-transform a k,k,2k Lagrangian derivative tensor at cubic order.'''

    flat = np.asarray(flat_lagrangian_tensor, dtype=float)
    unitary = np.asarray(unitary_lagrangian_tensor, dtype=float)
    if any(tensor.shape != (4, 4, 4) for tensor in (flat, unitary)):
        raise ValueError('the scalar cubic tensors must be four-by-four-by-four')
    if not np.all(np.isfinite(flat)) or not np.all(np.isfinite(unitary)):
        raise ValueError('the scalar cubic tensors must be finite')
    first_map = harmonic_scalar_phase_space_map(
        state,
        parameters,
        comoving_wavenumber_bar=base_wavenumber_bar,
    ).matrix
    second_map = harmonic_scalar_phase_space_map(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base_wavenumber_bar,
    ).matrix
    factor = -float(np.exp(3.0 * state.n))

    def transformed(tensor: np.ndarray) -> np.ndarray:
        return factor * np.einsum(
            'ijk,ia,jb,kc->abc',
            tensor,
            first_map,
            first_map,
            second_map,
        )

    return transformed(flat), transformed(unitary)


@dataclass(frozen=True)
class ConstraintSolvedScalarPhasePoint:
    '''Exact reduced Lagrangian and momenta at a real two-harmonic jet.'''

    canonical_momenta: np.ndarray
    flat_lagrangian_bar_per_a3: np.longdouble
    unitary_lagrangian_bar_per_a3: np.longdouble
    constraint_coefficients: np.ndarray
    maximum_constraint_residual: float


def constraint_solved_scalar_phase_point(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    physical_modes: np.ndarray,
    phase_points: int = 256,
) -> ConstraintSolvedScalarPhasePoint:
    '''Solve the nonlinear constraints and differentiate the exact flat action.'''

    physical = np.asarray(physical_modes, dtype=float)
    if physical.shape != (2, 4) or not np.all(np.isfinite(physical)):
        raise ValueError('physical_modes must have shape (2,4)')
    solution = solve_projected_dynamic_constraints(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical,
        phase_points=phase_points,
        residual_tolerance=1.0e-13,
    )
    if not solution.converged:
        raise ValueError('the direct Legendre constraint solve did not converge')
    fields = dynamic_scalar_fields(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical,
        constraint_coefficients=solution.coefficients,
        phase_points=phase_points,
    )
    phase = (
        np.longdouble(2.0)
        * np.longdouble(np.pi)
        * np.arange(phase_points, dtype=np.longdouble)
        / np.longdouble(phase_points)
    )
    lapse = np.asarray(fields.lapse, dtype=np.longdouble)
    shift = np.asarray(fields.shift_contravariant, dtype=np.longdouble)
    clock_convective = (
        np.longdouble(state.u)
        + np.asarray(fields.clock_velocity, dtype=np.longdouble)
        - shift * np.asarray(fields.clock_gradient, dtype=np.longdouble)
    )
    rod_convective = (
        np.asarray(fields.rod_velocity, dtype=np.longdouble)
        - shift * np.asarray(fields.rod_gradient, dtype=np.longdouble)
    )
    momenta = np.zeros((2, 2), dtype=np.longdouble)
    a3 = np.longdouble(fields.scale_factor) ** 3
    for index, harmonic in enumerate((1, 2)):
        cosine = np.sqrt(np.longdouble(2.0)) * np.cos(harmonic * phase)
        sine = np.sqrt(np.longdouble(2.0)) * np.sin(harmonic * phase)
        momenta[index, 0] = a3 * np.mean(
            clock_convective * cosine / lapse,
            dtype=np.longdouble,
        )
        momenta[index, 1] = a3 * np.mean(
            rod_convective * (-sine) / lapse,
            dtype=np.longdouble,
        )
    flat = flat_gauge_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical,
        constraint_coefficients=solution.coefficients,
        phase_points=phase_points,
    )
    unitary = rod_unitary_dynamic_scalar_lagrangian_bar_per_a3(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        physical_modes=physical,
        constraint_coefficients=solution.coefficients,
        phase_points=phase_points,
    )
    return ConstraintSolvedScalarPhasePoint(
        canonical_momenta=np.asarray(momenta, dtype=float),
        flat_lagrangian_bar_per_a3=flat,
        unitary_lagrangian_bar_per_a3=unitary,
        constraint_coefficients=solution.coefficients,
        maximum_constraint_residual=float(
            np.max(np.abs(solution.projected_residual))
        ),
    )


def _quadratic_scalar_hamiltonian_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    canonical_modes: np.ndarray,
) -> np.longdouble:
    '''Evaluate H2 in the same dimensionless normalization as a^3 Lbar.'''

    a3 = np.longdouble(np.exp(3.0 * state.n))
    value = np.longdouble(0.0)
    for index, harmonic in enumerate((1, 2)):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=harmonic * base_wavenumber_bar,
        )
        kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks)
        phase_mode = canonical_modes[index]
        momentum = phase_mode[:2]
        configuration = phase_mode[2:]
        velocity = np.linalg.solve(
            kinetic,
            momentum / float(a3) + gyroscopic @ configuration,
        )
        lagrangian_two = (
            0.5 * velocity @ kinetic @ velocity
            - velocity @ gyroscopic @ configuration
            - 0.5 * configuration @ potential @ configuration
        )
        value += np.longdouble(momentum @ velocity) - a3 * np.longdouble(
            lagrangian_two
        )
    return value


@dataclass(frozen=True)
class DirectScalarLegendrePoint:
    '''Finite-amplitude direct inverse and interaction Hamiltonian remainder.'''

    physical_modes: np.ndarray
    recovered_momenta: np.ndarray
    flat_interaction_hamiltonian_bar: np.longdouble
    unitary_interaction_hamiltonian_bar: np.longdouble
    maximum_momentum_residual: float
    maximum_constraint_residual: float
    iterations: int


def solve_direct_scalar_legendre_point(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    canonical_modes: np.ndarray,
    phase_points: int = 256,
    momentum_tolerance: float = 2.0e-13,
    max_iterations: int = 12,
) -> DirectScalarLegendrePoint:
    '''Numerically invert p=dL/dv without using the analytic R-dependent map.'''

    canonical = np.asarray(canonical_modes, dtype=float)
    if canonical.shape != (2, 4) or not np.all(np.isfinite(canonical)):
        raise ValueError('canonical_modes must have shape (2,4)')
    if (
        not np.isfinite(momentum_tolerance)
        or momentum_tolerance <= 0.0
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError('the direct Legendre solver requires positive controls')
    target_momenta = canonical[:, :2]
    configurations = canonical[:, 2:]
    velocities = np.zeros((2, 2))
    a3 = float(np.exp(3.0 * state.n))
    kinetic_matrices = []
    for harmonic in (1, 2):
        blocks = scalar_constraint_blocks(
            state,
            parameters,
            comoving_wavenumber_bar=harmonic * base_wavenumber_bar,
        )
        kinetic, _, _ = reduced_scalar_matrices(blocks)
        if np.min(np.linalg.eigvalsh(kinetic)) <= TOL:
            raise ValueError('the direct scalar Legendre kinetic block is not positive')
        kinetic_matrices.append(kinetic)

    point = None
    for iteration in range(max_iterations + 1):
        physical = np.concatenate((velocities, configurations), axis=1)
        point = constraint_solved_scalar_phase_point(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            physical_modes=physical,
            phase_points=phase_points,
        )
        residual = point.canonical_momenta - target_momenta
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= momentum_tolerance:
            break
        if iteration == max_iterations:
            raise ValueError('the direct scalar momentum inversion did not converge')
        for harmonic_index, kinetic in enumerate(kinetic_matrices):
            velocities[harmonic_index] -= np.linalg.solve(
                a3 * kinetic,
                residual[harmonic_index],
            )
    if point is None:
        raise RuntimeError('the direct scalar Legendre solver did not evaluate a point')
    physical = np.concatenate((velocities, configurations), axis=1)
    p_dot_v = np.sum(
        np.asarray(target_momenta, dtype=np.longdouble)
        * np.asarray(velocities, dtype=np.longdouble),
        dtype=np.longdouble,
    )
    quadratic = _quadratic_scalar_hamiltonian_bar(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        canonical_modes=canonical,
    )
    scale = np.longdouble(a3)
    flat_hamiltonian = p_dot_v - scale * point.flat_lagrangian_bar_per_a3
    unitary_hamiltonian = p_dot_v - scale * point.unitary_lagrangian_bar_per_a3
    return DirectScalarLegendrePoint(
        physical_modes=physical,
        recovered_momenta=point.canonical_momenta,
        flat_interaction_hamiltonian_bar=flat_hamiltonian - quadratic,
        unitary_interaction_hamiltonian_bar=unitary_hamiltonian - quadratic,
        maximum_momentum_residual=float(
            np.max(np.abs(point.canonical_momenta - target_momenta))
        ),
        maximum_constraint_residual=point.maximum_constraint_residual,
        iterations=iteration,
    )


@dataclass(frozen=True)
class DirectScalarHamiltonianTensorPair:
    flat_tensor: np.ndarray
    unitary_tensor: np.ndarray
    maximum_momentum_residual: float
    maximum_constraint_residual: float
    maximum_iterations: int


def direct_scalar_interaction_hamiltonian_tensor_pair(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    epsilon: float,
    phase_points: int = 256,
) -> DirectScalarHamiltonianTensorPair:
    '''Third-differentiate a finite-amplitude direct Legendre transform.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('the direct Hamiltonian tensor step must be positive')
    cache: dict[tuple[float, ...], np.ndarray] = {}
    maximum_momentum_residual = 0.0
    maximum_constraint_residual = 0.0
    maximum_iterations = 0

    def values(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        nonlocal maximum_momentum_residual
        nonlocal maximum_constraint_residual
        nonlocal maximum_iterations
        key = tuple(float(value) for value in np.concatenate((first, second)))
        if key not in cache:
            point = solve_direct_scalar_legendre_point(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                canonical_modes=np.stack((first, second)),
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
            maximum_iterations = max(maximum_iterations, point.iterations)
            cache[key] = np.array(
                [
                    point.flat_interaction_hamiltonian_bar,
                    point.unitary_interaction_hamiltonian_bar,
                ],
                dtype=np.longdouble,
            )
        return cache[key]

    tensors = np.zeros((2, 4, 4, 4), dtype=np.longdouble)
    basis = np.eye(4)
    zero = np.zeros(4)
    for first_index in range(4):
        for second_index in range(4):
            for third_index in range(4):
                first_direction = basis[first_index]
                second_direction = basis[second_index]
                third_direction = basis[third_index]
                if first_index == second_index:
                    positive = (
                        values(epsilon * first_direction, epsilon * third_direction)
                        - 2.0 * values(zero, epsilon * third_direction)
                        + values(-epsilon * first_direction, epsilon * third_direction)
                    )
                    negative = (
                        values(epsilon * first_direction, -epsilon * third_direction)
                        - 2.0 * values(zero, -epsilon * third_direction)
                        + values(-epsilon * first_direction, -epsilon * third_direction)
                    )
                    coefficient = (positive - negative) / (2.0 * epsilon**3)
                else:
                    coefficient = np.zeros(2, dtype=np.longdouble)
                    for first_sign in (-1.0, 1.0):
                        for second_sign in (-1.0, 1.0):
                            for third_sign in (-1.0, 1.0):
                                first_mode = epsilon * (
                                    first_sign * first_direction
                                    + second_sign * second_direction
                                )
                                second_mode = epsilon * third_sign * third_direction
                                coefficient += (
                                    first_sign
                                    * second_sign
                                    * third_sign
                                    * values(first_mode, second_mode)
                                )
                    coefficient /= 8.0 * epsilon**3
                tensors[:, first_index, second_index, third_index] = coefficient
    return DirectScalarHamiltonianTensorPair(
        flat_tensor=np.asarray(tensors[0], dtype=float),
        unitary_tensor=np.asarray(tensors[1], dtype=float),
        maximum_momentum_residual=maximum_momentum_residual,
        maximum_constraint_residual=maximum_constraint_residual,
        maximum_iterations=maximum_iterations,
    )


def quadratic_scalar_hamiltonian_cubic_negative_control(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    epsilon: float,
) -> np.ndarray:
    '''Apply the triad third-derivative stencil to the pure quadratic H2.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('the quadratic negative-control step must be positive')

    def value(first: np.ndarray, second: np.ndarray) -> np.longdouble:
        return _quadratic_scalar_hamiltonian_bar(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            canonical_modes=np.stack((first, second)),
        )

    tensor = np.zeros((4, 4, 4), dtype=np.longdouble)
    basis = np.eye(4)
    zero = np.zeros(4)
    for first_index in range(4):
        for second_index in range(4):
            for third_index in range(4):
                first_direction = basis[first_index]
                second_direction = basis[second_index]
                third_direction = basis[third_index]
                if first_index == second_index:
                    positive = (
                        value(epsilon * first_direction, epsilon * third_direction)
                        - 2.0 * value(zero, epsilon * third_direction)
                        + value(-epsilon * first_direction, epsilon * third_direction)
                    )
                    negative = (
                        value(epsilon * first_direction, -epsilon * third_direction)
                        - 2.0 * value(zero, -epsilon * third_direction)
                        + value(-epsilon * first_direction, -epsilon * third_direction)
                    )
                    coefficient = (positive - negative) / (2.0 * epsilon**3)
                else:
                    coefficient = np.longdouble(0.0)
                    for first_sign in (-1.0, 1.0):
                        for second_sign in (-1.0, 1.0):
                            for third_sign in (-1.0, 1.0):
                                first_mode = epsilon * (
                                    first_sign * first_direction
                                    + second_sign * second_direction
                                )
                                second_mode = epsilon * third_sign * third_direction
                                coefficient += (
                                    first_sign
                                    * second_sign
                                    * third_sign
                                    * value(first_mode, second_mode)
                                )
                    coefficient /= 8.0 * epsilon**3
                tensor[first_index, second_index, third_index] = coefficient
    return np.asarray(tensor, dtype=float)


@dataclass(frozen=True)
class FrozenScalarMode:
    frequency_bar: float
    configuration: np.ndarray
    momentum: np.ndarray
    symplectic_norm: float
    pencil_residual: float


def frozen_symplectic_scalar_modes(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
) -> tuple[FrozenScalarMode, FrozenScalarMode]:
    '''Solve and unit-normalize the two positive-frequency E66 scalar modes.'''

    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks)
    roots = frozen_scalar_frequency_squared(kinetic, gyroscopic, potential)
    if np.max(np.abs(roots.imag)) > 1.0e-8 or np.min(roots.real) <= 0.0:
        raise ValueError('frozen scalar modes require positive real frequency squared')
    antisymmetric = gyroscopic.T - gyroscopic
    a = float(np.exp(state.n))
    modes = []
    for root in roots.real:
        frequency = float(np.sqrt(root))
        pencil = -frequency**2 * kinetic - 1j * frequency * antisymmetric + potential
        _, _, right_vectors = np.linalg.svd(pencil)
        configuration = right_vectors.conj().T[:, -1]
        pivot = int(np.argmax(np.abs(configuration)))
        configuration *= np.exp(-1j * np.angle(configuration[pivot]))
        momentum = a**3 * (
            -1j * frequency * kinetic - gyroscopic
        ) @ configuration
        norm = float(
            np.real(
                1j
                * (
                    np.vdot(configuration, momentum)
                    - np.vdot(momentum, configuration)
                )
            )
        )
        if not np.isfinite(norm) or norm <= TOL:
            raise ValueError('the positive-frequency scalar symplectic norm is not positive')
        configuration = configuration / np.sqrt(norm)
        momentum = momentum / np.sqrt(norm)
        normalized_norm = float(
            np.real(
                1j
                * (
                    np.vdot(configuration, momentum)
                    - np.vdot(momentum, configuration)
                )
            )
        )
        residual = float(
            np.linalg.norm(pencil @ configuration)
            / max(1.0, np.linalg.norm(configuration))
        )
        modes.append(
            FrozenScalarMode(
                frequency_bar=frequency,
                configuration=configuration,
                momentum=momentum,
                symplectic_norm=normalized_norm,
                pencil_residual=residual,
            )
        )
    return modes[0], modes[1]


def scalar_mode_symplectic_overlap(
    first: FrozenScalarMode,
    second: FrozenScalarMode,
) -> complex:
    return 1j * (
        np.vdot(first.configuration, second.momentum)
        - np.vdot(first.momentum, second.configuration)
    )


@dataclass(frozen=True)
class FrozenCubicVertex:
    first_mode: int
    second_mode: int
    third_mode: int
    first_frequency_sign: int
    second_frequency_sign: int
    third_frequency_sign: int
    value: complex


def _signed_mode_jet(mode: FrozenScalarMode, sign: int) -> np.ndarray:
    if sign not in (-1, 1):
        raise ValueError('a frozen mode frequency sign must be -1 or +1')
    configuration = mode.configuration if sign == 1 else mode.configuration.conj()
    velocity = -1j * sign * mode.frequency_bar * configuration
    return np.concatenate((velocity, configuration))


def project_frozen_scalar_cubic_vertices(
    tensor: np.ndarray,
    first_modes: tuple[FrozenScalarMode, FrozenScalarMode],
    second_harmonic_modes: tuple[FrozenScalarMode, FrozenScalarMode],
    *,
    scale_factor: float,
) -> tuple[FrozenCubicVertex, ...]:
    '''Project T_ab;c onto all 2^3 mode and 2^3 frequency-sign assignments.'''

    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (4, 4, 4) or not np.all(np.isfinite(tensor)):
        raise ValueError('the dynamic cubic tensor must be a finite 4x4x4 array')
    vertices = []
    for first_mode in range(2):
        for second_mode in range(2):
            for third_mode in range(2):
                for first_sign in (-1, 1):
                    for second_sign in (-1, 1):
                        for third_sign in (-1, 1):
                            first_jet = _signed_mode_jet(
                                first_modes[first_mode], first_sign
                            )
                            second_jet = _signed_mode_jet(
                                first_modes[second_mode], second_sign
                            )
                            third_jet = _signed_mode_jet(
                                second_harmonic_modes[third_mode], third_sign
                            )
                            value = float(scale_factor) ** 3 * np.einsum(
                                'ijk,i,j,k->',
                                tensor,
                                first_jet,
                                second_jet,
                                third_jet,
                            )
                            vertices.append(
                                FrozenCubicVertex(
                                    first_mode=first_mode,
                                    second_mode=second_mode,
                                    third_mode=third_mode,
                                    first_frequency_sign=first_sign,
                                    second_frequency_sign=second_sign,
                                    third_frequency_sign=third_sign,
                                    value=complex(value),
                                )
                            )
    return tuple(vertices)


@dataclass(frozen=True)
class DynamicCubicMomentumReceipt:
    base_wavenumber_bar: float
    flat_tensor_norm: float
    unitary_tensor_norm: float
    tensor_step_refinement: float
    tensor_gauge_residual: float
    same_k_leg_exchange_residual: float
    maximum_pencil_residual: float
    maximum_cross_mode_symplectic_overlap: float
    minimum_vertex_magnitude: float
    maximum_vertex_magnitude: float
    maximum_vertex_gauge_residual: float
    frequency_conjugation_residual: float
    same_k_vertex_exchange_residual: float
    assignment_count: int
    declared_momentum_gate_passed: bool


def evaluate_dynamic_cubic_momentum(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    cubic_steps: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    phase_points: int = 256,
) -> DynamicCubicMomentumReceipt:
    '''Run the preregistered dynamic scalar cubic receipt at one base momentum.'''

    if len(cubic_steps) < 2:
        raise ValueError('the dynamic cubic receipt requires at least two steps')
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
    flat = tensor_pairs[-1][0]
    unitary = tensor_pairs[-1][1]
    scale = max(1.0, float(np.linalg.norm(flat)))
    refinement = float(
        np.linalg.norm(tensor_pairs[-1][0] - tensor_pairs[-2][0]) / scale
    )
    gauge = float(np.linalg.norm(flat - unitary) / scale)
    tensor_exchange = float(
        np.linalg.norm(flat - np.swapaxes(flat, 0, 1)) / scale
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
    flat_vertices = project_frozen_scalar_cubic_vertices(
        flat,
        first_modes,
        second_modes,
        scale_factor=np.exp(state.n),
    )
    unitary_vertices = project_frozen_scalar_cubic_vertices(
        unitary,
        first_modes,
        second_modes,
        scale_factor=np.exp(state.n),
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
        for item in flat_vertices
    }
    conjugation = 0.0
    exchange = 0.0
    for key, value in lookup.items():
        conjugate_key = (*key[:3], -key[3], -key[4], -key[5])
        exchange_key = (
            key[1],
            key[0],
            key[2],
            key[4],
            key[3],
            key[5],
        )
        conjugation = max(
            conjugation,
            abs(lookup[conjugate_key] - value.conjugate()),
        )
        exchange = max(exchange, abs(lookup[exchange_key] - value))
    maximum_pencil = max(
        mode.pencil_residual for mode in first_modes + second_modes
    )
    maximum_overlap = max(
        abs(scalar_mode_symplectic_overlap(first_modes[0], first_modes[1])),
        abs(scalar_mode_symplectic_overlap(second_modes[0], second_modes[1])),
    )
    maximum_vertex_gauge = max(
        abs(first.value - second.value)
        for first, second in zip(flat_vertices, unitary_vertices, strict=True)
    )
    passed = (
        refinement < 2.0e-4
        and gauge < 1.0e-6
        and tensor_exchange < 1.0e-6
        and maximum_pencil < 1.0e-8
        and maximum_overlap < 1.0e-8
        and maximum_vertex_gauge < 1.0e-6
        and conjugation < 1.0e-8
        and exchange < 1.0e-8
        and all(np.isfinite(item.value) for item in flat_vertices)
    )
    return DynamicCubicMomentumReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        flat_tensor_norm=float(np.linalg.norm(flat)),
        unitary_tensor_norm=float(np.linalg.norm(unitary)),
        tensor_step_refinement=refinement,
        tensor_gauge_residual=gauge,
        same_k_leg_exchange_residual=tensor_exchange,
        maximum_pencil_residual=float(maximum_pencil),
        maximum_cross_mode_symplectic_overlap=float(maximum_overlap),
        minimum_vertex_magnitude=float(
            min(abs(item.value) for item in flat_vertices)
        ),
        maximum_vertex_magnitude=float(
            max(abs(item.value) for item in flat_vertices)
        ),
        maximum_vertex_gauge_residual=float(maximum_vertex_gauge),
        frequency_conjugation_residual=float(conjugation),
        same_k_vertex_exchange_residual=float(exchange),
        assignment_count=len(flat_vertices),
        declared_momentum_gate_passed=passed,
    )


def _signed_mode_phase_space(mode: FrozenScalarMode, sign: int) -> np.ndarray:
    if sign not in (-1, 1):
        raise ValueError('a frozen mode frequency sign must be -1 or +1')
    if sign == 1:
        return np.concatenate((mode.momentum, mode.configuration))
    return np.concatenate((mode.momentum.conj(), mode.configuration.conj()))


def project_frozen_scalar_hamiltonian_vertices(
    tensor: np.ndarray,
    first_modes: tuple[FrozenScalarMode, FrozenScalarMode],
    second_harmonic_modes: tuple[FrozenScalarMode, FrozenScalarMode],
) -> tuple[FrozenCubicVertex, ...]:
    '''Project a canonical H tensor onto all branch and frequency assignments.'''

    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (4, 4, 4) or not np.all(np.isfinite(tensor)):
        raise ValueError('the scalar Hamiltonian tensor must be finite and 4x4x4')
    vertices = []
    for first_mode in range(2):
        for second_mode in range(2):
            for third_mode in range(2):
                for first_sign in (-1, 1):
                    for second_sign in (-1, 1):
                        for third_sign in (-1, 1):
                            first_phase = _signed_mode_phase_space(
                                first_modes[first_mode], first_sign
                            )
                            second_phase = _signed_mode_phase_space(
                                first_modes[second_mode], second_sign
                            )
                            third_phase = _signed_mode_phase_space(
                                second_harmonic_modes[third_mode], third_sign
                            )
                            value = np.einsum(
                                'ijk,i,j,k->',
                                tensor,
                                first_phase,
                                second_phase,
                                third_phase,
                            )
                            vertices.append(
                                FrozenCubicVertex(
                                    first_mode=first_mode,
                                    second_mode=second_mode,
                                    third_mode=third_mode,
                                    first_frequency_sign=first_sign,
                                    second_frequency_sign=second_sign,
                                    third_frequency_sign=third_sign,
                                    value=complex(value),
                                )
                            )
    return tuple(vertices)


@dataclass(frozen=True)
class ScalarInteractionHamiltonianReceipt:
    base_wavenumber_bar: float
    analytic_tensor_norm: float
    direct_tensor_norm: float
    analytic_direct_tensor_residual: float
    maximum_component_residual: float
    maximum_active_component_relative_residual: float
    maximum_inactive_component_absolute_residual: float
    tensor_step_refinement: float
    tensor_grid_refinement: float
    tensor_gauge_residual: float
    minimum_kinetic_eigenvalue: float
    minimum_map_determinant_magnitude: float
    maximum_momentum_inverse_residual: float
    maximum_constraint_residual: float
    maximum_velocity_iterations: int
    naive_coordinate_slot_negative_control: float
    wrong_r_sign_negative_control: float
    quadratic_only_cubic_negative_control: float
    maximum_mode_map_residual: float
    maximum_vertex_analytic_direct_residual: float
    maximum_vertex_lagrangian_sign_residual: float
    maximum_vertex_gauge_residual: float
    frequency_conjugation_residual: float
    same_k_vertex_exchange_residual: float
    assignment_count: int
    declared_legendre_gate_passed: bool


def evaluate_scalar_interaction_hamiltonian(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    base_wavenumber_bar: float,
    cubic_steps: tuple[float, ...] = (1.0e-2, 5.0e-3, 2.5e-3),
    phase_points: int = 256,
    grid_phase_points: int = 512,
) -> ScalarInteractionHamiltonianReceipt:
    '''Run the preregistered finite-triad cubic Legendre-transform gate.'''

    if len(cubic_steps) < 2:
        raise ValueError('the scalar Legendre receipt requires at least two steps')
    if grid_phase_points <= phase_points:
        raise ValueError('the scalar Legendre grid refinement must increase resolution')

    lagrangian_pairs = []
    analytic_pairs = []
    direct_pairs = []
    for step in cubic_steps:
        lagrangian = dynamic_reduced_scalar_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
        )
        analytic = scalar_interaction_hamiltonian_cubic_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            flat_lagrangian_tensor=lagrangian[0],
            unitary_lagrangian_tensor=lagrangian[1],
        )
        direct = direct_scalar_interaction_hamiltonian_tensor_pair(
            state,
            parameters,
            base_wavenumber_bar=base_wavenumber_bar,
            epsilon=step,
            phase_points=phase_points,
        )
        lagrangian_pairs.append(lagrangian)
        analytic_pairs.append(analytic)
        direct_pairs.append(direct)

    fine_step = float(cubic_steps[-1])
    grid_lagrangian = dynamic_reduced_scalar_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=fine_step,
        phase_points=grid_phase_points,
    )
    grid_analytic = scalar_interaction_hamiltonian_cubic_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        flat_lagrangian_tensor=grid_lagrangian[0],
        unitary_lagrangian_tensor=grid_lagrangian[1],
    )
    grid_direct = direct_scalar_interaction_hamiltonian_tensor_pair(
        state,
        parameters,
        base_wavenumber_bar=base_wavenumber_bar,
        epsilon=fine_step,
        phase_points=grid_phase_points,
    )

    lagrangian_flat = lagrangian_pairs[-1][0]
    analytic_flat, analytic_unitary = analytic_pairs[-1]
    direct = direct_pairs[-1]
    direct_flat = direct.flat_tensor
    direct_unitary = direct.unitary_tensor
    scale = max(1.0, float(np.linalg.norm(analytic_flat)))
    analytic_direct = float(np.linalg.norm(analytic_flat - direct_flat) / scale)
    component_residual = float(np.max(np.abs(analytic_flat - direct_flat)))
    component_difference = np.abs(analytic_flat - direct_flat)
    active_components = np.abs(analytic_flat) >= 1.0e-5
    if np.any(active_components):
        active_component_residual = float(
            np.max(
                component_difference[active_components]
                / np.abs(analytic_flat[active_components])
            )
        )
    else:
        active_component_residual = 0.0
    if np.any(~active_components):
        inactive_component_residual = float(
            np.max(component_difference[~active_components])
        )
    else:
        inactive_component_residual = 0.0
    step_refinement = max(
        float(
            np.linalg.norm(analytic_pairs[-1][0] - analytic_pairs[-2][0])
            / scale
        ),
        float(
            np.linalg.norm(
                direct_pairs[-1].flat_tensor - direct_pairs[-2].flat_tensor
            )
            / scale
        ),
    )
    grid_refinement = max(
        float(np.linalg.norm(grid_analytic[0] - analytic_flat) / scale),
        float(np.linalg.norm(grid_direct.flat_tensor - direct_flat) / scale),
    )
    gauge_residual = max(
        float(np.linalg.norm(analytic_flat - analytic_unitary) / scale),
        float(np.linalg.norm(direct_flat - direct_unitary) / scale),
    )

    first_blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=base_wavenumber_bar,
    )
    second_blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=2.0 * base_wavenumber_bar,
    )
    first_kinetic, first_gyroscopic, _ = reduced_scalar_matrices(first_blocks)
    second_kinetic, second_gyroscopic, _ = reduced_scalar_matrices(second_blocks)
    a = float(np.exp(state.n))
    first_map = canonical_scalar_phase_space_map(
        first_kinetic,
        first_gyroscopic,
        scale_factor=a,
    )
    second_map = canonical_scalar_phase_space_map(
        second_kinetic,
        second_gyroscopic,
        scale_factor=a,
    )
    wrong_first = canonical_scalar_phase_space_map(
        first_kinetic,
        -first_gyroscopic,
        scale_factor=a,
    ).matrix
    wrong_second = canonical_scalar_phase_space_map(
        second_kinetic,
        -second_gyroscopic,
        scale_factor=a,
    ).matrix
    wrong_r_tensor = -a**3 * np.einsum(
        'ijk,ia,jb,kc->abc',
        lagrangian_flat,
        wrong_first,
        wrong_first,
        wrong_second,
    )
    naive_tensor = -a**3 * lagrangian_flat
    naive_control = float(np.linalg.norm(analytic_flat - naive_tensor) / scale)
    wrong_r_control = float(np.linalg.norm(analytic_flat - wrong_r_tensor) / scale)
    quadratic_control = float(
        np.linalg.norm(
            quadratic_scalar_hamiltonian_cubic_negative_control(
                state,
                parameters,
                base_wavenumber_bar=base_wavenumber_bar,
                epsilon=fine_step,
            )
        )
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
    mode_map_residual = 0.0
    for mapping, modes in (
        (first_map.matrix, first_modes),
        (second_map.matrix, second_modes),
    ):
        for mode in modes:
            phase_mode = np.concatenate((mode.momentum, mode.configuration))
            mapped = mapping @ phase_mode
            expected = np.concatenate(
                (
                    -1j * mode.frequency_bar * mode.configuration,
                    mode.configuration,
                )
            )
            mode_map_residual = max(
                mode_map_residual,
                float(np.linalg.norm(mapped - expected)),
            )

    lagrangian_vertices = project_frozen_scalar_cubic_vertices(
        lagrangian_flat,
        first_modes,
        second_modes,
        scale_factor=a,
    )
    analytic_vertices = project_frozen_scalar_hamiltonian_vertices(
        analytic_flat,
        first_modes,
        second_modes,
    )
    direct_vertices = project_frozen_scalar_hamiltonian_vertices(
        direct_flat,
        first_modes,
        second_modes,
    )
    unitary_vertices = project_frozen_scalar_hamiltonian_vertices(
        direct_unitary,
        first_modes,
        second_modes,
    )
    vertex_analytic_direct = max(
        abs(first.value - second.value)
        for first, second in zip(analytic_vertices, direct_vertices, strict=True)
    )
    vertex_lagrangian_sign = max(
        abs(first.value + second.value)
        for first, second in zip(analytic_vertices, lagrangian_vertices, strict=True)
    )
    vertex_gauge = max(
        abs(first.value - second.value)
        for first, second in zip(direct_vertices, unitary_vertices, strict=True)
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
        for item in analytic_vertices
    }
    conjugation = 0.0
    exchange = 0.0
    for key, value in lookup.items():
        conjugate_key = (*key[:3], -key[3], -key[4], -key[5])
        exchange_key = (
            key[1],
            key[0],
            key[2],
            key[4],
            key[3],
            key[5],
        )
        conjugation = max(
            conjugation,
            abs(lookup[conjugate_key] - value.conjugate()),
        )
        exchange = max(exchange, abs(lookup[exchange_key] - value))

    maximum_momentum_residual = max(
        item.maximum_momentum_residual for item in direct_pairs + [grid_direct]
    )
    maximum_constraint_residual = max(
        item.maximum_constraint_residual for item in direct_pairs + [grid_direct]
    )
    maximum_iterations = max(
        item.maximum_iterations for item in direct_pairs + [grid_direct]
    )
    minimum_kinetic = min(
        float(np.min(first_map.kinetic_eigenvalues)),
        float(np.min(second_map.kinetic_eigenvalues)),
    )
    minimum_determinant = min(abs(first_map.determinant), abs(second_map.determinant))
    passed = (
        analytic_direct < 2.0e-4
        and component_residual < 2.0e-4
        and active_component_residual < 2.0e-4
        and inactive_component_residual < 2.0e-4
        and step_refinement < 2.0e-4
        and grid_refinement < 1.0e-8
        and gauge_residual < 1.0e-6
        and minimum_kinetic > TOL
        and minimum_determinant > TOL
        and maximum_momentum_residual < 1.0e-11
        and maximum_constraint_residual < 1.0e-11
        and naive_control > 1.0e-6
        and wrong_r_control > 1.0e-6
        and quadratic_control < 1.0e-8
        and mode_map_residual < 1.0e-8
        and vertex_analytic_direct < 2.0e-4
        and vertex_lagrangian_sign < 1.0e-8
        and vertex_gauge < 1.0e-6
        and conjugation < 1.0e-8
        and exchange < 1.0e-8
        and len(analytic_vertices) == 64
    )
    return ScalarInteractionHamiltonianReceipt(
        base_wavenumber_bar=float(base_wavenumber_bar),
        analytic_tensor_norm=float(np.linalg.norm(analytic_flat)),
        direct_tensor_norm=float(np.linalg.norm(direct_flat)),
        analytic_direct_tensor_residual=analytic_direct,
        maximum_component_residual=component_residual,
        maximum_active_component_relative_residual=active_component_residual,
        maximum_inactive_component_absolute_residual=inactive_component_residual,
        tensor_step_refinement=step_refinement,
        tensor_grid_refinement=grid_refinement,
        tensor_gauge_residual=gauge_residual,
        minimum_kinetic_eigenvalue=minimum_kinetic,
        minimum_map_determinant_magnitude=minimum_determinant,
        maximum_momentum_inverse_residual=float(maximum_momentum_residual),
        maximum_constraint_residual=float(maximum_constraint_residual),
        maximum_velocity_iterations=maximum_iterations,
        naive_coordinate_slot_negative_control=naive_control,
        wrong_r_sign_negative_control=wrong_r_control,
        quadratic_only_cubic_negative_control=quadratic_control,
        maximum_mode_map_residual=mode_map_residual,
        maximum_vertex_analytic_direct_residual=float(vertex_analytic_direct),
        maximum_vertex_lagrangian_sign_residual=float(vertex_lagrangian_sign),
        maximum_vertex_gauge_residual=float(vertex_gauge),
        frequency_conjugation_residual=float(conjugation),
        same_k_vertex_exchange_residual=float(exchange),
        assignment_count=len(analytic_vertices),
        declared_legendre_gate_passed=passed,
    )
