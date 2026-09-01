'''Transverse rod-phonon and ADM-shift reduction on the E62 background.

Use vector spatial gauge, N_i=a S_i, div S=0 and

    X^i = beta x^i + pi_T^i,   div pi_T=0.

For each nonzero comoving Fourier wavenumber the shift has no time derivative.
Its exact quadratic Schur complement gives a positive reduced kinetic residue
for the admitted M1 signs.  The k=0 sector is treated separately because the
transverse decomposition and spatial vector gauge degenerate there.
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


def transverse_vector_basis(spatial_wavevector_bar: np.ndarray) -> np.ndarray:
    '''Return two unit vectors transverse to an arbitrary nonzero wavevector.'''

    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the dimensionless spatial wavevector must have three components')
    norm = float(np.linalg.norm(wavevector))
    if norm <= TOL:
        raise ValueError('the transverse decomposition requires nonzero momentum')
    direction = wavevector / norm
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.stack((first, second))


def exact_adm_vector_mode_lagrangian_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray,
    polarization: np.ndarray,
    phonon_amplitude: float = 0.0,
    phonon_velocity: float = 0.0,
    shift_amplitude: float = 0.0,
    rod_kinetic_sign: float = 1.0,
    phase_points: int = 128,
) -> float:
    '''Evaluate the background-subtracted exact ADM action on one periodic mode.

    The calculation contracts the full rod gradients and the ADM
    K_ij K^ij-K^2 term before averaging.  The real mode profile is normalized
    so that its squared spatial average is one.
    '''

    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    polarization = np.asarray(polarization, dtype=float)
    amplitudes = np.array(
        [phonon_amplitude, phonon_velocity, shift_amplitude, rod_kinetic_sign],
        dtype=float,
    )
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the dimensionless spatial wavevector must have three components')
    k = float(np.linalg.norm(wavevector))
    if k <= TOL:
        raise ValueError('the periodic vector mode requires nonzero momentum')
    if polarization.shape != (3,) or not np.all(np.isfinite(polarization)):
        raise ValueError('the vector polarization must have three finite components')
    if not np.isclose(np.linalg.norm(polarization), 1.0, atol=TOL):
        raise ValueError('the vector polarization must have unit norm')
    if abs(float(np.dot(wavevector, polarization))) > TOL * max(1.0, k):
        raise ValueError('the vector polarization must be transverse')
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError('the vector mode amplitudes and sign must be finite')
    sign = float(rod_kinetic_sign)
    if sign == 0.0:
        raise ValueError('rod kinetic sign must be nonzero')
    if not isinstance(phase_points, int) or phase_points < 8:
        raise ValueError('phase_points must be an integer of at least eight')
    state_values = state.as_array()
    m = float(parameters.m_planck_over_mu_x)
    if not np.all(np.isfinite(state_values)) or not np.isfinite(m) or m <= 0.0:
        raise ValueError('the ADM background data must be finite with positive M_P/mu_X')

    a = float(np.exp(state.n))
    beta_bar = rod_charge_bar(state)
    phase = 2.0 * np.pi * np.arange(phase_points) / phase_points
    profile = np.sqrt(2.0) * np.cos(phase)
    gradient_profile = (
        -np.sqrt(2.0) * np.sin(phase)[:, None] * wavevector[None, :]
    )
    background_gradient = beta_bar * np.eye(3)
    delta_gradient = float(phonon_amplitude) * np.einsum(
        'i,nj->nij', polarization, gradient_profile
    )
    full_gradient = background_gradient[None, :, :] + delta_gradient
    field_velocity = (
        float(phonon_velocity) * profile[:, None] * polarization[None, :]
    )
    shift_contravariant = (
        float(shift_amplitude)
        * profile[:, None]
        * polarization[None, :]
        / a
    )
    convective_velocity = field_velocity - np.einsum(
        'nj,nij->ni', shift_contravariant, full_gradient
    )
    inverse_spatial_metric = np.eye(3) / a**2
    spatial_cross = 2.0 * np.einsum(
        'jk,ij,nik->n',
        inverse_spatial_metric,
        background_gradient,
        delta_gradient,
    )
    spatial_quadratic = np.einsum(
        'jk,nij,nik->n',
        inverse_spatial_metric,
        delta_gradient,
        delta_gradient,
    )
    rod_contribution = 0.5 * sign * a**3 * np.mean(
        np.einsum('ni,ni->n', convective_velocity, convective_velocity)
        - spatial_cross
        - spatial_quadratic
    )

    derivative_covariant_shift = (
        a
        * float(shift_amplitude)
        * np.einsum('ni,j->nij', gradient_profile, polarization)
    )
    delta_extrinsic_curvature = -0.5 * (
        derivative_covariant_shift
        + np.swapaxes(derivative_covariant_shift, 1, 2)
    )
    background_extrinsic_curvature = a**2 * state.h * np.eye(3)
    background_trace = float(
        np.einsum(
            'ij,ij->',
            inverse_spatial_metric,
            background_extrinsic_curvature,
        )
    )
    delta_trace = np.einsum(
        'ij,nij->n', inverse_spatial_metric, delta_extrinsic_curvature
    )
    extrinsic_cross = 2.0 * np.einsum(
        'ik,jl,ij,nkl->n',
        inverse_spatial_metric,
        inverse_spatial_metric,
        background_extrinsic_curvature,
        delta_extrinsic_curvature,
    )
    extrinsic_quadratic = np.einsum(
        'ik,jl,nij,nkl->n',
        inverse_spatial_metric,
        inverse_spatial_metric,
        delta_extrinsic_curvature,
        delta_extrinsic_curvature,
    )
    gravitational_contribution = 0.5 * m**2 * a**3 * np.mean(
        extrinsic_cross
        + extrinsic_quadratic
        - 2.0 * background_trace * delta_trace
        - delta_trace**2
    )
    return float(rod_contribution + gravitational_contribution)


def finite_difference_adm_vector_block(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray,
    polarization: np.ndarray,
    rod_kinetic_sign: float = 1.0,
    epsilon: float = 1.0e-3,
) -> np.ndarray:
    '''Extract the (pi_dot,S) Hessian from the exact periodic ADM action.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('epsilon must be finite and positive')

    def value(velocity: float, shift: float) -> float:
        return exact_adm_vector_mode_lagrangian_bar(
            state,
            parameters,
            spatial_wavevector_bar=spatial_wavevector_bar,
            polarization=polarization,
            phonon_velocity=velocity,
            shift_amplitude=shift,
            rod_kinetic_sign=rod_kinetic_sign,
        )

    zero = value(0.0, 0.0)
    velocity_diagonal = (
        value(epsilon, 0.0) - 2.0 * zero + value(-epsilon, 0.0)
    ) / epsilon**2
    shift_diagonal = (
        value(0.0, epsilon) - 2.0 * zero + value(0.0, -epsilon)
    ) / epsilon**2
    mixed = (
        value(epsilon, epsilon)
        - value(epsilon, -epsilon)
        - value(-epsilon, epsilon)
        + value(-epsilon, -epsilon)
    ) / (4.0 * epsilon**2)
    return np.array(
        [[velocity_diagonal, mixed], [mixed, shift_diagonal]], dtype=float
    )


def finite_difference_adm_vector_gradient_potential_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray,
    polarization: np.ndarray,
    rod_kinetic_sign: float = 1.0,
    epsilon: float = 1.0e-3,
) -> float:
    '''Extract positive G from L=-G pi^2/2 in the exact periodic ADM action.'''

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('epsilon must be finite and positive')

    def value(amplitude: float) -> float:
        return exact_adm_vector_mode_lagrangian_bar(
            state,
            parameters,
            spatial_wavevector_bar=spatial_wavevector_bar,
            polarization=polarization,
            phonon_amplitude=amplitude,
            rod_kinetic_sign=rod_kinetic_sign,
        )

    zero = value(0.0)
    return float(
        -(value(epsilon) - 2.0 * zero + value(-epsilon)) / epsilon**2
    )


def vector_shift_denominator_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Return D/mu_X^4=s(beta/mu_X)^2+(m^2/2)k_com^2.'''

    k = float(comoving_wavenumber_bar)
    sign = float(rod_kinetic_sign)
    m = float(parameters.m_planck_over_mu_x)
    if not np.all(np.isfinite([k, sign, m])):
        raise ValueError('vector parameters must be finite')
    if k < 0.0:
        raise ValueError('comoving wavenumber must be nonnegative')
    if sign == 0.0:
        raise ValueError('rod kinetic sign must be nonzero')
    if m <= 0.0:
        raise ValueError('M_P/mu_X must be positive')
    beta_bar = rod_charge_bar(state)
    return float(sign * beta_bar**2 + 0.5 * m**2 * k**2)


def uneliminated_vector_quadratic_block(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    rod_kinetic_sign: float = 1.0,
) -> np.ndarray:
    '''Return the symmetric block in 1/2 (pi_dot,S) H (pi_dot,S)^T.'''

    k = float(comoving_wavenumber_bar)
    sign = float(rod_kinetic_sign)
    a = float(np.exp(state.n))
    beta_bar = rod_charge_bar(state)
    denominator = vector_shift_denominator_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    return np.array(
        [
            [sign * a**3, -sign * a**2 * beta_bar],
            [-sign * a**2 * beta_bar, a * denominator],
        ]
    )


def reduced_vector_kinetic_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Return K_V/mu_X^2 from the exact shift Schur complement.'''

    k = float(comoving_wavenumber_bar)
    if k <= 0.0:
        raise ValueError('the reduced vector formula requires nonzero k')
    block = uneliminated_vector_quadratic_block(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=rod_kinetic_sign,
    )
    if abs(block[1, 1]) <= TOL:
        raise ValueError('the shift Hessian is singular')
    return float(block[0, 0] - block[0, 1] ** 2 / block[1, 1])


def analytic_reduced_vector_kinetic_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    rod_kinetic_sign: float = 1.0,
) -> float:
    k = float(comoving_wavenumber_bar)
    if k <= 0.0:
        raise ValueError('the reduced vector formula requires nonzero k')
    sign = float(rod_kinetic_sign)
    a = float(np.exp(state.n))
    m = parameters.m_planck_over_mu_x
    denominator = vector_shift_denominator_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    if abs(denominator) <= TOL:
        raise ValueError('the shift constraint denominator is singular')
    return float(sign * a**3 * (0.5 * m**2 * k**2) / denominator)


@dataclass(frozen=True)
class VectorHomogeneousSector:
    comoving_wavenumber_bar: float
    transverse_decomposition_defined: bool
    reduced_formula_applied: bool
    propagating_vector_count_claimed: int
    status: str


def audit_homogeneous_vector_sector() -> VectorHomogeneousSector:
    '''Fail closed at k=0 without interpreting K_V -> 0 as a ghost pole.'''

    return VectorHomogeneousSector(
        comoving_wavenumber_bar=0.0,
        transverse_decomposition_defined=False,
        reduced_formula_applied=False,
        propagating_vector_count_claimed=0,
        status='K_ZERO_REQUIRES_SEPARATE_HOMOGENEOUS_CONSTRAINT_ANALYSIS',
    )


@dataclass(frozen=True)
class VectorStabilityAudit:
    comoving_wavenumber_bar: float
    physical_wavenumber_bar: float
    shift_denominator_bar: float
    shift_solution_per_phonon_velocity: float
    schur_kinetic_bar: float
    analytic_kinetic_bar: float
    schur_relative_residual: float
    gradient_potential_coefficient_bar: float
    frozen_omega_squared_bar: float
    vector_mass_squared_bar: float
    vector_speed_squared: float
    high_k_kinetic_limit_bar: float
    adm_block_relative_residual: float
    adm_gradient_relative_residual: float
    adm_polarization_spread: float
    adm_action_extraction_passed: bool
    transverse_polarization_count: int
    polarizations_degenerate: bool
    shift_has_time_derivative: bool
    shift_constraint_regular: bool
    reduced_kinetic_positive: bool
    gradient_positive: bool
    frozen_pole_nonnegative: bool
    finite_k_vector_gate_passed: bool
    homogeneous_k_zero_sector_resolved: bool
    time_dependent_mode_equation_solved: bool
    scalar_sector_computed: bool
    strong_coupling_scale_derived: bool
    one_loop_st_identity_computed: bool
    brst_physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_flrw_vector_sector(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    rod_kinetic_sign: float = 1.0,
    tol: float = TOL,
) -> VectorStabilityAudit:
    '''Eliminate one transverse shift polarization and audit the reduced pole.'''

    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('E65 requires an admitted nondegenerate E62 background')
    k = float(comoving_wavenumber_bar)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError('E65 finite-k vector audit requires positive k')
    sign = float(rod_kinetic_sign)
    a = float(np.exp(state.n))
    beta_bar = rod_charge_bar(state)
    m = parameters.m_planck_over_mu_x
    denominator = vector_shift_denominator_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    if abs(denominator) <= tol:
        raise ValueError('the shift constraint denominator is singular')
    schur = reduced_vector_kinetic_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    analytic = analytic_reduced_vector_kinetic_bar(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    schur_residual = abs(schur - analytic) / max(1.0, abs(analytic))
    gradient_potential = float(sign * a * k**2)
    wavevector = k * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    polarizations = transverse_vector_basis(wavevector)
    adm_blocks = np.stack(
        [
            finite_difference_adm_vector_block(
                state,
                parameters,
                spatial_wavevector_bar=wavevector,
                polarization=polarization,
                rod_kinetic_sign=sign,
            )
            for polarization in polarizations
        ]
    )
    adm_gradients = np.array(
        [
            finite_difference_adm_vector_gradient_potential_bar(
                state,
                parameters,
                spatial_wavevector_bar=wavevector,
                polarization=polarization,
                rod_kinetic_sign=sign,
            )
            for polarization in polarizations
        ]
    )
    expected_block = uneliminated_vector_quadratic_block(
        state,
        parameters,
        comoving_wavenumber_bar=k,
        rod_kinetic_sign=sign,
    )
    block_residual = float(
        np.max(np.linalg.norm(adm_blocks - expected_block, axis=(1, 2)))
        / max(1.0, float(np.linalg.norm(expected_block)))
    )
    gradient_residual = float(
        np.max(np.abs(adm_gradients - gradient_potential))
        / max(1.0, abs(gradient_potential))
    )
    polarization_spread = float(
        max(
            np.linalg.norm(adm_blocks[0] - adm_blocks[1]),
            abs(adm_gradients[0] - adm_gradients[1]),
        )
    )
    adm_extraction_passed = (
        block_residual <= tol
        and gradient_residual <= tol
        and polarization_spread <= tol
    )
    omega_squared = float(gradient_potential / schur) if abs(schur) > tol else np.nan
    mass_squared = float(2.0 * sign * beta_bar**2 / (m**2 * a**2))
    physical_k = float(k / a)
    speed_squared = (
        (omega_squared - mass_squared) / physical_k**2
        if physical_k > tol and np.isfinite(omega_squared)
        else np.nan
    )
    shift_solution = float(sign * a * beta_bar / denominator)
    high_k_limit = float(sign * a**3)
    regular = abs(denominator) > tol
    kinetic_positive = schur > tol
    gradient_positive = gradient_potential > tol
    pole_nonnegative = np.isfinite(omega_squared) and omega_squared >= -tol
    passed = (
        regular
        and kinetic_positive
        and gradient_positive
        and pole_nonnegative
        and schur_residual <= tol
        and abs(speed_squared - 1.0) <= tol
        and adm_extraction_passed
    )
    return VectorStabilityAudit(
        comoving_wavenumber_bar=k,
        physical_wavenumber_bar=physical_k,
        shift_denominator_bar=denominator,
        shift_solution_per_phonon_velocity=shift_solution,
        schur_kinetic_bar=schur,
        analytic_kinetic_bar=analytic,
        schur_relative_residual=float(schur_residual),
        gradient_potential_coefficient_bar=gradient_potential,
        frozen_omega_squared_bar=omega_squared,
        vector_mass_squared_bar=mass_squared,
        vector_speed_squared=float(speed_squared),
        high_k_kinetic_limit_bar=high_k_limit,
        adm_block_relative_residual=block_residual,
        adm_gradient_relative_residual=gradient_residual,
        adm_polarization_spread=polarization_spread,
        adm_action_extraction_passed=adm_extraction_passed,
        transverse_polarization_count=len(polarizations),
        polarizations_degenerate=polarization_spread <= tol,
        shift_has_time_derivative=False,
        shift_constraint_regular=regular,
        reduced_kinetic_positive=kinetic_positive,
        gradient_positive=gradient_positive,
        frozen_pole_nonnegative=pole_nonnegative,
        finite_k_vector_gate_passed=passed,
        homogeneous_k_zero_sector_resolved=False,
        time_dependent_mode_equation_solved=False,
        scalar_sector_computed=False,
        strong_coupling_scale_derived=False,
        one_loop_st_identity_computed=False,
        brst_physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=(
            'FINITE_K_TRANSVERSE_VECTOR_GATE_PASSED'
            if passed
            else 'FINITE_K_TRANSVERSE_VECTOR_GATE_FAILED'
        ),
    )


def naive_shift_zero_omega_squared_bar(
    state: ReferenceFlrwState, *, comoving_wavenumber_bar: float
) -> float:
    '''Negative control that incorrectly sets the shift to zero before varying.'''

    k = float(comoving_wavenumber_bar)
    if not np.isfinite(k) or k < 0.0:
        raise ValueError('comoving wavenumber must be finite and nonnegative')
    a = float(np.exp(state.n))
    return float(k**2 / a**2)
