'''Finite-k scalar constraint reduction on the admitted E62 FLRW background.

The flat scalar gauge keeps the clock fluctuation q, the longitudinal rod
phonon r=k s, the lapse alpha, and the longitudinal shift
theta=beta k B/a^2.  Lapse and shift are eliminated by an exact 2x2 Schur
complement.  A separate periodic ADM contraction extracts the unreduced 6x6
Hessian without using the closed-form reduced matrices.
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


TOL = 1.0e-10
ADM_TOL = 5.0e-6


def _validate_scalar_inputs(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
) -> tuple[float, float, float, float]:
    values = state.as_array()
    m = float(parameters.m_planck_over_mu_x)
    k_comoving = float(comoving_wavenumber_bar)
    if not np.all(np.isfinite(values)):
        raise ValueError('the scalar background state must be finite')
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    if not np.isfinite(k_comoving) or k_comoving <= 0.0:
        raise ValueError('the finite-k scalar chart requires positive momentum')
    a = float(np.exp(state.n))
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError('the scale factor must be finite and positive')
    if abs(state.b) <= TOL:
        raise ValueError('the longitudinal rod chart requires nonzero beta')
    return a, k_comoving / a, m, float(state.b)


@dataclass(frozen=True)
class ScalarConstraintBlocks:
    physical_wavenumber_bar: float
    lapse_coefficient_bar: float
    lapse_shift_coefficient_bar: float
    constraint_matrix_bar: np.ndarray
    velocity_coupling_bar: np.ndarray
    field_coupling_bar: np.ndarray
    constraint_determinant_bar: float
    positive_q_polynomial_bar: float
    friedmann_identity_residual: float
    lapse_has_time_derivative: bool
    longitudinal_shift_has_time_derivative: bool


def scalar_constraint_blocks(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
) -> ScalarConstraintBlocks:
    '''Return C,D,E in L/a^3=(ydot^2-kappa^2 y^2)/2+c^TCc/2+c^T(Dydot+Ey).'''

    _, kappa, m, b = _validate_scalar_inputs(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    u = float(state.u)
    h = float(state.h)
    lam = float(parameters.lambda_over_mu_x_squared)
    lapse_coefficient = u**2 - 6.0 * m**2 * h**2
    lapse_shift = 2.0 * m**2 * h * kappa / b
    constraint = np.array(
        [[lapse_coefficient, lapse_shift], [lapse_shift, 1.0]], dtype=float
    )
    velocity = np.diag([-u, -1.0])
    field = np.array(
        [[0.0, b * kappa], [-u * kappa / b, 0.0]], dtype=float
    )
    determinant = float(np.linalg.det(constraint))
    q_polynomial = float(
        3.0 * b**4
        + 2.0 * b**2 * m**2 * lam
        + 4.0 * m**4 * h**2 * kappa**2
    )
    friedmann_identity = abs(lapse_coefficient + 3.0 * b**2 + 2.0 * m**2 * lam)
    return ScalarConstraintBlocks(
        physical_wavenumber_bar=kappa,
        lapse_coefficient_bar=lapse_coefficient,
        lapse_shift_coefficient_bar=lapse_shift,
        constraint_matrix_bar=constraint,
        velocity_coupling_bar=velocity,
        field_coupling_bar=field,
        constraint_determinant_bar=determinant,
        positive_q_polynomial_bar=q_polynomial,
        friedmann_identity_residual=float(friedmann_identity),
        lapse_has_time_derivative=False,
        longitudinal_shift_has_time_derivative=False,
    )


def reduced_scalar_matrices(
    blocks: ScalarConstraintBlocks,
    *,
    tol: float = TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Return K,R,V after c=-C^{-1}(D ydot+E y).'''

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError('tol must be finite and positive')
    if abs(blocks.constraint_determinant_bar) <= tol:
        raise ValueError('the lapse-shift scalar constraint block is singular')
    inverse = np.linalg.inv(blocks.constraint_matrix_bar)
    derivative = blocks.velocity_coupling_bar
    field = blocks.field_coupling_bar
    identity = np.eye(2)
    kinetic = identity - derivative.T @ inverse @ derivative
    gyroscopic = derivative.T @ inverse @ field
    potential = (
        blocks.physical_wavenumber_bar**2 * identity
        + field.T @ inverse @ field
    )
    return kinetic, gyroscopic, potential


def frozen_scalar_frequency_squared(
    kinetic: np.ndarray,
    gyroscopic: np.ndarray,
    potential: np.ndarray,
    *,
    tol: float = TOL,
) -> np.ndarray:
    '''Solve det[-omega^2 K-i omega(R^T-R)+V]=0 for omega^2.'''

    kinetic = np.asarray(kinetic, dtype=float)
    gyroscopic = np.asarray(gyroscopic, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if any(matrix.shape != (2, 2) for matrix in (kinetic, gyroscopic, potential)):
        raise ValueError('the frozen scalar pencil requires three 2x2 matrices')
    antisymmetric = gyroscopic.T - gyroscopic
    g = float(antisymmetric[0, 1])
    determinant_kinetic = float(np.linalg.det(kinetic))
    if abs(determinant_kinetic) <= tol:
        raise ValueError('the reduced scalar kinetic matrix is singular')
    linear = -(
        potential[0, 0] * kinetic[1, 1]
        + potential[1, 1] * kinetic[0, 0]
        - 2.0 * potential[0, 1] * kinetic[0, 1]
    ) - g**2
    coefficients = np.array(
        [determinant_kinetic, linear, float(np.linalg.det(potential))]
    )
    roots = np.roots(coefficients)
    return roots[np.argsort(roots.real)]


def analytic_scalar_mass_squared_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
) -> float:
    '''Return the second frozen scalar mass squared in units of mu_X^2.'''

    m = float(parameters.m_planck_over_mu_x)
    lam = float(parameters.lambda_over_mu_x_squared)
    h = float(state.h)
    if not np.all(np.isfinite([m, lam, h, state.b, state.u])) or m <= 0.0:
        raise ValueError('the scalar mass inputs must be finite with positive M_P/mu_X')
    if abs(h) <= TOL:
        raise ValueError('the frozen scalar mass formula requires nonzero H')
    return float(
        (state.b**2 + m**2 * lam)
        * (state.b**2 + state.u**2)
        / (2.0 * m**4 * h**2)
    )


def expected_unreduced_scalar_hessian(
    blocks: ScalarConstraintBlocks,
) -> np.ndarray:
    '''Assemble the analytic Hessian in (qdot,rdot,alpha,theta,q,r).'''

    hessian = np.zeros((6, 6), dtype=float)
    hessian[0:2, 0:2] = np.eye(2)
    hessian[2:4, 2:4] = blocks.constraint_matrix_bar
    hessian[2:4, 0:2] = blocks.velocity_coupling_bar
    hessian[0:2, 2:4] = blocks.velocity_coupling_bar.T
    hessian[2:4, 4:6] = blocks.field_coupling_bar
    hessian[4:6, 2:4] = blocks.field_coupling_bar.T
    hessian[4:6, 4:6] = -blocks.physical_wavenumber_bar**2 * np.eye(2)
    return hessian


def exact_adm_scalar_mode_lagrangian_bar_per_a3(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray,
    amplitudes: np.ndarray,
    phase_points: int = 128,
) -> float:
    '''Evaluate the exact background-subtracted ADM ansatz on one scalar mode.

    amplitudes=(qdot,rdot,alpha,theta,q,r).  Cosine scalar potentials and
    their sine longitudinal gradients are kept with their exact relative
    phase before the periodic average.
    '''

    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the scalar wavevector must have three finite components')
    k_comoving = float(np.linalg.norm(wavevector))
    a, _, m, _ = _validate_scalar_inputs(
        state,
        parameters,
        comoving_wavenumber_bar=k_comoving,
    )
    if amplitudes.shape != (6,) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('scalar ADM amplitudes must be a finite six-vector')
    if not isinstance(phase_points, int) or phase_points < 8:
        raise ValueError('phase_points must be an integer of at least eight')
    q_velocity, r_velocity, alpha, theta, q_amplitude, r_amplitude = amplitudes
    beta_bar = rod_charge_bar(state)
    if abs(beta_bar) <= TOL:
        raise ValueError('the scalar ADM shift variable requires nonzero beta')

    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    cosine = np.sqrt(2.0) * np.cos(phase)
    gradient_cosine = (
        -np.sqrt(2.0) * np.sin(phase)[:, None] * wavevector[None, :]
    )
    hessian_cosine = (
        -np.sqrt(2.0)
        * np.cos(phase)[:, None, None]
        * np.outer(wavevector, wavevector)[None, :, :]
    )
    lapse = 1.0 + alpha * cosine
    if np.min(lapse) <= 0.0:
        raise ValueError('the sampled ADM lapse must stay positive')
    inverse_spatial_metric = np.eye(3) / a**2

    b_potential_bar = theta * a**2 / (beta_bar * k_comoving)
    shift_contravariant = (
        b_potential_bar
        * np.einsum('ij,nj->ni', inverse_spatial_metric, gradient_cosine)
    )
    derivative_covariant_shift = b_potential_bar * hessian_cosine

    q_gradient = q_amplitude * gradient_cosine
    q_dot = q_velocity * cosine
    clock_convective = (
        state.u
        + q_dot
        - np.einsum('ni,ni->n', shift_contravariant, q_gradient)
    )
    q_spatial_norm = np.einsum(
        'ij,ni,nj->n', inverse_spatial_metric, q_gradient, q_gradient
    )
    clock_lagrangian = 0.5 * np.mean(
        clock_convective**2 / lapse - lapse * q_spatial_norm
    )
    clock_background = 0.5 * state.u**2

    direction_derivative = gradient_cosine / k_comoving
    rod_velocity = r_velocity * direction_derivative
    rod_delta_gradient = r_amplitude * hessian_cosine / k_comoving
    rod_gradient = beta_bar * np.eye(3)[None, :, :] + rod_delta_gradient
    rod_convective = rod_velocity - np.einsum(
        'nj,nij->ni', shift_contravariant, rod_gradient
    )
    rod_spatial_norm = np.einsum(
        'jk,nij,nik->n',
        inverse_spatial_metric,
        rod_gradient,
        rod_gradient,
    )
    rod_lagrangian = 0.5 * np.mean(
        np.einsum('ni,ni->n', rod_convective, rod_convective) / lapse
        - lapse * rod_spatial_norm
    )
    rod_background = -1.5 * beta_bar**2 / a**2

    background_extrinsic_curvature = a**2 * state.h * np.eye(3)
    extrinsic_curvature = (
        background_extrinsic_curvature[None, :, :]
        - derivative_covariant_shift
    ) / lapse[:, None, None]
    extrinsic_norm = np.einsum(
        'ik,jl,nij,nkl->n',
        inverse_spatial_metric,
        inverse_spatial_metric,
        extrinsic_curvature,
        extrinsic_curvature,
    )
    extrinsic_trace = np.einsum(
        'ij,nij->n', inverse_spatial_metric, extrinsic_curvature
    )
    lam = float(parameters.lambda_over_mu_x_squared)
    gravity_lagrangian = 0.5 * m**2 * np.mean(
        lapse * (extrinsic_norm - extrinsic_trace**2 - 2.0 * lam)
    )
    gravity_background = 0.5 * m**2 * (-6.0 * state.h**2 - 2.0 * lam)
    return float(
        clock_lagrangian
        - clock_background
        + rod_lagrangian
        - rod_background
        + gravity_lagrangian
        - gravity_background
    )


def finite_difference_adm_scalar_hessian(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray,
    epsilon: float = 2.0e-4,
) -> np.ndarray:
    '''Extract the unreduced six-variable Hessian from the exact ADM ansatz.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('epsilon must be finite and positive')
    zero = np.zeros(6)

    def value(vector: np.ndarray) -> float:
        return exact_adm_scalar_mode_lagrangian_bar_per_a3(
            state,
            parameters,
            spatial_wavevector_bar=spatial_wavevector_bar,
            amplitudes=vector,
        )

    baseline = value(zero)
    hessian = np.zeros((6, 6), dtype=float)
    for index in range(6):
        direction = np.zeros(6)
        direction[index] = epsilon
        hessian[index, index] = (
            value(direction) - 2.0 * baseline + value(-direction)
        ) / epsilon**2
    for row in range(6):
        for column in range(row + 1, 6):
            first = np.zeros(6)
            second = np.zeros(6)
            first[row] = epsilon
            second[column] = epsilon
            mixed = (
                value(first + second)
                - value(first - second)
                - value(-first + second)
                + value(-first - second)
            ) / (4.0 * epsilon**2)
            hessian[row, column] = hessian[column, row] = mixed
    return hessian


def wrong_sign_reference_principal_kinetic() -> np.ndarray:
    '''Algebraic negative control for a reversed X^A kinetic sign.'''

    return -np.eye(2)


@dataclass(frozen=True)
class ScalarStabilityAudit:
    comoving_wavenumber_bar: float
    physical_wavenumber_bar: float
    constraint_determinant_bar: float
    positive_q_polynomial_bar: float
    q_determinant_identity_residual: float
    friedmann_identity_residual: float
    kinetic_matrix_bar: np.ndarray
    kinetic_eigenvalues: tuple[float, float]
    potential_matrix_bar: np.ndarray
    gyroscopic_antisymmetric_bar: np.ndarray
    coupled_frequency_squared_bar: tuple[complex, complex]
    analytic_frequency_squared_bar: tuple[float, float]
    scalar_mass_squared_bar: float
    spectator_chi_frequency_squared_bar: float
    root_factorization_residual: float
    adm_hessian_relative_residual: float
    adm_hessian_convergence_spread: float
    lapse_has_time_derivative: bool
    longitudinal_shift_has_time_derivative: bool
    constraint_block_regular: bool
    reduced_kinetic_positive: bool
    frozen_roots_real: bool
    frozen_roots_nonnegative: bool
    adm_action_extraction_passed: bool
    reduced_coupled_scalar_count: int
    spectator_scalar_count: int
    total_scalar_physical_count: int
    finite_k_scalar_gate_passed: bool
    homogeneous_k_zero_sector_resolved: bool
    beta_zero_branch_resolved: bool
    time_dependent_mode_equations_solved: bool
    strong_coupling_scale_derived: bool
    one_loop_st_identity_computed: bool
    brst_physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_flrw_scalar_sector(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    chi_mass_over_mu_x: float = 0.0,
    tol: float = TOL,
) -> ScalarStabilityAudit:
    '''Audit the k>0 flat-gauge scalar Schur complement and frozen poles.'''

    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('E66 requires an admitted nondegenerate E62 background')
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError('tol must be finite and positive')
    chi_mass = float(chi_mass_over_mu_x)
    if not np.isfinite(chi_mass) or chi_mass < 0.0:
        raise ValueError('m_chi/mu_X must be finite and nonnegative')
    if abs(state.h) <= tol:
        raise ValueError('the E66 flat scalar reduction requires nonzero H')
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks, tol=tol)
    roots = frozen_scalar_frequency_squared(
        kinetic, gyroscopic, potential, tol=tol
    )
    mass_squared = analytic_scalar_mass_squared_bar(state, parameters)
    analytic_roots = np.sort(
        np.array(
            [
                blocks.physical_wavenumber_bar**2,
                blocks.physical_wavenumber_bar**2 + mass_squared,
            ]
        )
    )
    sorted_roots = roots[np.argsort(roots.real)]
    root_residual = float(
        np.max(np.abs(sorted_roots - analytic_roots))
        / max(1.0, float(np.max(np.abs(analytic_roots))))
    )

    k_comoving = float(comoving_wavenumber_bar)
    wavevector = k_comoving * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    coarse_hessian = finite_difference_adm_scalar_hessian(
        state,
        parameters,
        spatial_wavevector_bar=wavevector,
        epsilon=4.0e-4,
    )
    fine_hessian = finite_difference_adm_scalar_hessian(
        state,
        parameters,
        spatial_wavevector_bar=wavevector,
        epsilon=2.0e-4,
    )
    expected_hessian = expected_unreduced_scalar_hessian(blocks)
    hessian_scale = max(1.0, float(np.linalg.norm(expected_hessian)))
    hessian_residual = float(
        np.linalg.norm(fine_hessian - expected_hessian) / hessian_scale
    )
    convergence_spread = float(
        np.linalg.norm(fine_hessian - coarse_hessian) / hessian_scale
    )
    adm_passed = hessian_residual <= ADM_TOL and convergence_spread <= ADM_TOL

    kinetic_eigenvalues_array = np.linalg.eigvalsh(kinetic)
    kinetic_positive = bool(np.min(kinetic_eigenvalues_array) > tol)
    roots_real = bool(np.max(np.abs(sorted_roots.imag)) <= 1.0e-8)
    roots_nonnegative = bool(
        roots_real and np.min(sorted_roots.real) >= -1.0e-8
    )
    determinant_identity = abs(
        blocks.positive_q_polynomial_bar
        + state.b**2 * blocks.constraint_determinant_bar
    )
    regular = abs(blocks.constraint_determinant_bar) > tol
    passed = (
        regular
        and kinetic_positive
        and roots_real
        and roots_nonnegative
        and root_residual <= 1.0e-8
        and adm_passed
        and blocks.friedmann_identity_residual <= 1.0e-10
        and determinant_identity <= 1.0e-10
    )
    return ScalarStabilityAudit(
        comoving_wavenumber_bar=k_comoving,
        physical_wavenumber_bar=blocks.physical_wavenumber_bar,
        constraint_determinant_bar=blocks.constraint_determinant_bar,
        positive_q_polynomial_bar=blocks.positive_q_polynomial_bar,
        q_determinant_identity_residual=float(determinant_identity),
        friedmann_identity_residual=blocks.friedmann_identity_residual,
        kinetic_matrix_bar=kinetic,
        kinetic_eigenvalues=tuple(float(value) for value in kinetic_eigenvalues_array),
        potential_matrix_bar=potential,
        gyroscopic_antisymmetric_bar=gyroscopic.T - gyroscopic,
        coupled_frequency_squared_bar=tuple(complex(value) for value in sorted_roots),
        analytic_frequency_squared_bar=tuple(float(value) for value in analytic_roots),
        scalar_mass_squared_bar=mass_squared,
        spectator_chi_frequency_squared_bar=(
            blocks.physical_wavenumber_bar**2 + chi_mass**2
        ),
        root_factorization_residual=root_residual,
        adm_hessian_relative_residual=hessian_residual,
        adm_hessian_convergence_spread=convergence_spread,
        lapse_has_time_derivative=blocks.lapse_has_time_derivative,
        longitudinal_shift_has_time_derivative=(
            blocks.longitudinal_shift_has_time_derivative
        ),
        constraint_block_regular=regular,
        reduced_kinetic_positive=kinetic_positive,
        frozen_roots_real=roots_real,
        frozen_roots_nonnegative=roots_nonnegative,
        adm_action_extraction_passed=adm_passed,
        reduced_coupled_scalar_count=2,
        spectator_scalar_count=1,
        total_scalar_physical_count=3,
        finite_k_scalar_gate_passed=passed,
        homogeneous_k_zero_sector_resolved=False,
        beta_zero_branch_resolved=False,
        time_dependent_mode_equations_solved=False,
        strong_coupling_scale_derived=False,
        one_loop_st_identity_computed=False,
        brst_physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=(
            'FINITE_K_SCALAR_GATE_PASSED'
            if passed
            else 'FINITE_K_SCALAR_GATE_FAILED'
        ),
    )
