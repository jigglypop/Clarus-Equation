'''Finite-wavenumber TT tensor gate on the E62 reference background.

For q_ij=a^2 exp(gamma)_ij with trace(gamma)=0 and X^i=beta x^i,
the volume element is gamma independent and

    tr exp(-gamma) = 3 + 1/2 tr(gamma^2) + O(gamma^3).

The canonical rods therefore generate the positive tensor mass

    m_T^2 = 2 mu_X^2 beta^2 / (M_P^2 a^2).

Only the helicity-two quadratic sector is audited here.  Vector/scalar
constraints, a derived strong-coupling scale, loops and BRST positivity remain
outside this module.
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
from examples.physics.qft_reference_flrw_principal_stability import spatial_tt_basis


TOL = 1.0e-10


def _validate_symmetric_generator(generator: np.ndarray) -> np.ndarray:
    generator = np.asarray(generator, dtype=float)
    if generator.shape != (3, 3):
        raise ValueError('a tensor generator must be 3 by 3')
    if not np.all(np.isfinite(generator)):
        raise ValueError('the tensor generator must be finite')
    if np.linalg.norm(generator - generator.T) > TOL:
        raise ValueError('the tensor generator must be symmetric')
    return generator


def validate_tt_generator(
    generator: np.ndarray,
    spatial_wavevector_bar: np.ndarray,
    *,
    tol: float = TOL,
) -> None:
    generator = _validate_symmetric_generator(generator)
    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the spatial wavevector must have three finite components')
    norm = float(np.linalg.norm(wavevector))
    if norm <= tol:
        raise ValueError('a TT Fourier generator requires nonzero wavevector')
    direction = wavevector / norm
    if abs(float(np.trace(generator))) > tol:
        raise ValueError('the tensor generator must be traceless')
    if np.linalg.norm(direction @ generator) > tol:
        raise ValueError('the tensor generator must be transverse')
    if float(np.einsum('ij,ij->', generator, generator)) <= tol:
        raise ValueError('the tensor generator must be nonzero')


def rotate_tt_generator_to_z(
    generator: np.ndarray,
    spatial_wavevector_bar: np.ndarray,
) -> np.ndarray:
    '''Rotate a TT Fourier pair so its wavevector points along the grid z axis.'''

    validate_tt_generator(generator, spatial_wavevector_bar)
    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    direction = wavevector / np.linalg.norm(wavevector)
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    transverse_one = np.cross(direction, reference)
    transverse_one /= np.linalg.norm(transverse_one)
    transverse_two = np.cross(direction, transverse_one)
    rotation = np.vstack((transverse_one, transverse_two, direction))
    rotated = rotation @ np.asarray(generator, dtype=float) @ rotation.T
    validate_tt_generator(rotated, np.array([0.0, 0.0, 1.0]))
    return rotated


def symmetric_matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    matrix = _validate_symmetric_generator(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


def symmetric_exponential_trace_increment(matrix: np.ndarray) -> float:
    '''Return tr exp(matrix)-3 using expm1 to avoid subtractive cancellation.'''

    matrix = _validate_symmetric_generator(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(np.sum(np.expm1(eigenvalues)))


def rod_lagrangian_density_bar(
    epsilon: float,
    generator: np.ndarray,
    state: ReferenceFlrwState,
    *,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Return L_rod/mu_X^4 per comoving volume along gamma=epsilon Q.'''

    generator = _validate_symmetric_generator(generator)
    epsilon = float(epsilon)
    sign = float(rod_kinetic_sign)
    if not np.all(np.isfinite([epsilon, sign])) or sign == 0.0:
        raise ValueError('epsilon and a nonzero rod sign must be finite')
    inverse_metric_shape = symmetric_matrix_exponential(-epsilon * generator)
    a_cubed = float(np.exp(3.0 * state.n))
    return float(-0.5 * sign * a_cubed * state.b**2 * np.trace(inverse_metric_shape))


def rod_lagrangian_increment_bar(
    epsilon: float,
    generator: np.ndarray,
    state: ReferenceFlrwState,
    *,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Return L_rod(epsilon Q)-L_rod(0) with a stable exponential increment.'''

    generator = _validate_symmetric_generator(generator)
    epsilon = float(epsilon)
    sign = float(rod_kinetic_sign)
    if not np.all(np.isfinite([epsilon, sign])) or sign == 0.0:
        raise ValueError('epsilon and a nonzero rod sign must be finite')
    trace_increment = symmetric_exponential_trace_increment(-epsilon * generator)
    a_cubed = float(np.exp(3.0 * state.n))
    return float(-0.5 * sign * a_cubed * state.b**2 * trace_increment)


def tensor_mass_squared_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Return m_T^2/mu_X^2=2 sign b^2/m^2.'''

    m = float(parameters.m_planck_over_mu_x)
    sign = float(rod_kinetic_sign)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    if not np.isfinite(sign) or sign == 0.0:
        raise ValueError('rod kinetic sign must be finite and nonzero')
    return float(2.0 * sign * state.b**2 / m**2)


def finite_difference_tensor_mass_squared_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    generator: np.ndarray,
    *,
    epsilon: float = 1.0e-4,
    rod_kinetic_sign: float = 1.0,
) -> float:
    '''Infer m_T^2/mu_X^2 from the central Hessian of the exact rod trace.'''

    generator = _validate_symmetric_generator(generator)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('finite-difference epsilon must be positive')
    l_plus_increment = rod_lagrangian_increment_bar(
        epsilon, generator, state, rod_kinetic_sign=rod_kinetic_sign
    )
    l_minus_increment = rod_lagrangian_increment_bar(
        -epsilon, generator, state, rod_kinetic_sign=rod_kinetic_sign
    )
    second_derivative = (l_plus_increment + l_minus_increment) / epsilon**2
    norm_squared = float(np.einsum('ij,ij->', generator, generator))
    a_cubed = float(np.exp(3.0 * state.n))
    m = parameters.m_planck_over_mu_x
    return float(-4.0 * second_derivative / (m**2 * a_cubed * norm_squared))


def eh_tensor_kinetic_coefficient_from_adm(
    parameters: ReferenceFlrwParameters,
    generator: np.ndarray,
) -> float:
    '''Extract K_T from K^i_j=H delta^i_j+(qdot/2)Q^i_j.

    The ADM increment is (m^2/8) a^3 qdot^2 tr(Q^2).  Its velocity Hessian,
    divided by a^3 tr(Q^2), is m^2/4.
    '''

    generator = _validate_symmetric_generator(generator)
    norm_squared = float(np.einsum('ij,ij->', generator, generator))
    if norm_squared <= TOL:
        raise ValueError('the tensor generator must be nonzero')
    m = float(parameters.m_planck_over_mu_x)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    velocity_hessian_per_a_cubed = m**2 * norm_squared / 4.0
    return float(velocity_hessian_per_a_cubed / norm_squared)


def _periodic_derivative(values: np.ndarray) -> np.ndarray:
    points = values.shape[0]
    spacing = 2.0 * np.pi / points
    frequencies = 2.0 * np.pi * np.fft.fftfreq(points, d=spacing)
    reshape = (points,) + (1,) * (values.ndim - 1)
    transformed = np.fft.fft(values, axis=0)
    derivative = np.fft.ifft(1.0j * frequencies.reshape(reshape) * transformed, axis=0)
    return np.real_if_close(derivative, tol=1000).real


def integrated_spatial_ricci_density_bar(
    epsilon: float,
    generator: np.ndarray,
    state: ReferenceFlrwState,
    *,
    wave_number: int = 1,
    grid_points: int = 96,
) -> float:
    '''Integrate sqrt(q) R^(3)/mu_X^0 for a periodic exponential TT wave.

    The dimensionless coordinate is z in [0,2 pi), and
    gamma(z)=epsilon cos(wave_number z) Q.  Christoffels and the Ricci tensor
    are built from the metric, while their z derivatives use a spectral grid.
    '''

    generator = _validate_symmetric_generator(generator)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite')
    if wave_number <= 0:
        raise ValueError('wave number must be a positive integer')
    if grid_points < 24 or grid_points % 2:
        raise ValueError('grid_points must be an even integer at least 24')
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    if abs(epsilon) * float(np.max(np.abs(eigenvalues))) > 40.0:
        raise ValueError('the exponential TT probe is outside the finite audit range')
    z = 2.0 * np.pi * np.arange(grid_points) / grid_points
    amplitudes = epsilon * np.cos(wave_number * z)
    exponentials = np.exp(amplitudes[:, None] * eigenvalues[None, :])
    inverse_exponentials = np.exp(-amplitudes[:, None] * eigenvalues[None, :])
    shape_metric = np.einsum('ia,za,ja->zij', eigenvectors, exponentials, eigenvectors)
    shape_inverse = np.einsum(
        'ia,za,ja->zij', eigenvectors, inverse_exponentials, eigenvectors
    )
    a = float(np.exp(state.n))
    metric = a**2 * shape_metric
    inverse_metric = a**-2 * shape_inverse
    dz_metric = _periodic_derivative(metric)
    derivatives = np.zeros((grid_points, 3, 3, 3))
    derivatives[:, 2, :, :] = dz_metric

    christoffel = np.zeros((grid_points, 3, 3, 3))
    for upper in range(3):
        for lower_one in range(3):
            for lower_two in range(3):
                for contracted in range(3):
                    christoffel[:, upper, lower_one, lower_two] += 0.5 * inverse_metric[
                        :, upper, contracted
                    ] * (
                        derivatives[:, lower_one, contracted, lower_two]
                        + derivatives[:, lower_two, contracted, lower_one]
                        - derivatives[:, contracted, lower_one, lower_two]
                    )
    dz_christoffel = _periodic_derivative(christoffel)
    ricci = np.zeros((grid_points, 3, 3))
    for first in range(3):
        for second in range(3):
            ricci[:, first, second] += dz_christoffel[:, 2, first, second]
            if second == 2:
                for upper in range(3):
                    ricci[:, first, second] -= dz_christoffel[:, upper, first, upper]
            for upper in range(3):
                for contracted in range(3):
                    ricci[:, first, second] += (
                        christoffel[:, upper, upper, contracted]
                        * christoffel[:, contracted, first, second]
                        - christoffel[:, upper, second, contracted]
                        * christoffel[:, contracted, first, upper]
                    )
    scalar_curvature = np.einsum('zij,zij->z', inverse_metric, ricci)
    volume = np.sqrt(np.linalg.det(metric))
    return float(np.mean(volume * scalar_curvature))


def finite_difference_eh_gradient_coefficient_bar(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    generator: np.ndarray,
    *,
    epsilon: float = 2.0e-3,
    wave_number: int = 1,
    grid_points: int = 96,
) -> float:
    '''Extract G_T from the Hessian of (m^2/2) integral sqrt(q) R^(3).'''

    generator = _validate_symmetric_generator(generator)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('finite-difference epsilon must be positive')
    values = [
        integrated_spatial_ricci_density_bar(
            sign * epsilon,
            generator,
            state,
            wave_number=wave_number,
            grid_points=grid_points,
        )
        for sign in (1.0, 0.0, -1.0)
    ]
    m = parameters.m_planck_over_mu_x
    lagrangian = [0.5 * m**2 * value for value in values]
    second_derivative = (lagrangian[0] - 2.0 * lagrangian[1] + lagrangian[2]) / epsilon**2
    norm_squared = float(np.einsum('ij,ij->', generator, generator))
    a = float(np.exp(state.n))
    return float(-2.0 * second_derivative / (a * wave_number**2 * norm_squared))


@dataclass(frozen=True)
class TensorStabilityAudit:
    kinetic_coefficient_bar: float
    gradient_coefficient_bar: float
    expected_eh_coefficient_bar: float
    kinetic_coefficient_residual: float
    gradient_coefficient_residual: float
    gradient_epsilon_grid_wavenumber_spread: float
    tensor_speed_squared: float
    tensor_mass_squared_bar: float
    finite_difference_mass_squared_bar: float
    mass_relative_residual: float
    exponential_determinant_residual: float
    comoving_mass_invariant_bar: float
    kinetic_positive: bool
    gradient_positive: bool
    mass_nonnegative: bool
    finite_k_tensor_gate_passed: bool
    vector_sector_computed: bool
    scalar_sector_computed: bool
    strong_coupling_scale_derived: bool
    one_loop_st_identity_computed: bool
    brst_physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_flrw_tensor_sector(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    spatial_wavevector_bar: np.ndarray | None = None,
    polarization_index: int = 0,
    epsilon: float = 1.0e-4,
    rod_kinetic_sign: float = 1.0,
    tol: float = 2.0e-7,
) -> TensorStabilityAudit:
    '''Audit one TT polarization; isotropy makes the other coefficient identical.'''

    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('E64 requires an admitted nondegenerate E62 background')
    if spatial_wavevector_bar is None:
        spatial_wavevector_bar = np.array([0.0, 0.0, 1.0])
    basis = spatial_tt_basis(spatial_wavevector_bar)
    if polarization_index not in (0, 1):
        raise ValueError('polarization index must be zero or one')
    generator = basis[polarization_index]
    validate_tt_generator(generator, spatial_wavevector_bar)
    grid_generator = rotate_tt_generator_to_z(generator, spatial_wavevector_bar)

    m = parameters.m_planck_over_mu_x
    expected_eh = float(m**2 / 4.0)
    kinetic = eh_tensor_kinetic_coefficient_from_adm(parameters, generator)
    gradient = finite_difference_eh_gradient_coefficient_bar(
        state,
        parameters,
        grid_generator,
        epsilon=2.0e-3,
        wave_number=1,
        grid_points=96,
    )
    gradient_coarse = finite_difference_eh_gradient_coefficient_bar(
        state,
        parameters,
        grid_generator,
        epsilon=4.0e-3,
        wave_number=1,
        grid_points=48,
    )
    gradient_second_harmonic = finite_difference_eh_gradient_coefficient_bar(
        state,
        parameters,
        grid_generator,
        epsilon=2.0e-3,
        wave_number=2,
        grid_points=96,
    )
    eh_scale = max(1.0, abs(expected_eh))
    kinetic_residual = abs(kinetic - expected_eh) / eh_scale
    gradient_residual = abs(gradient - expected_eh) / eh_scale
    gradient_spread = (
        max(gradient, gradient_coarse, gradient_second_harmonic)
        - min(gradient, gradient_coarse, gradient_second_harmonic)
    ) / eh_scale
    mass = tensor_mass_squared_bar(
        state, parameters, rod_kinetic_sign=rod_kinetic_sign
    )
    finite_difference_mass = finite_difference_tensor_mass_squared_bar(
        state,
        parameters,
        generator,
        epsilon=epsilon,
        rod_kinetic_sign=rod_kinetic_sign,
    )
    mass_scale = max(1.0, abs(mass))
    mass_residual = abs(finite_difference_mass - mass) / mass_scale
    exponential = symmetric_matrix_exponential(epsilon * generator)
    determinant_residual = abs(float(np.linalg.det(exponential)) - 1.0)
    comoving_invariant = float(np.exp(2.0 * state.n) * mass)
    passed = (
        kinetic > tol
        and gradient > tol
        and kinetic_residual <= tol
        and gradient_residual <= tol
        and gradient_spread <= tol
        and mass >= -tol
        and mass_residual <= tol
        and determinant_residual <= tol
    )
    return TensorStabilityAudit(
        kinetic_coefficient_bar=kinetic,
        gradient_coefficient_bar=gradient,
        expected_eh_coefficient_bar=expected_eh,
        kinetic_coefficient_residual=float(kinetic_residual),
        gradient_coefficient_residual=float(gradient_residual),
        gradient_epsilon_grid_wavenumber_spread=float(gradient_spread),
        tensor_speed_squared=gradient / kinetic,
        tensor_mass_squared_bar=mass,
        finite_difference_mass_squared_bar=finite_difference_mass,
        mass_relative_residual=float(mass_residual),
        exponential_determinant_residual=float(determinant_residual),
        comoving_mass_invariant_bar=comoving_invariant,
        kinetic_positive=kinetic > tol,
        gradient_positive=gradient > tol,
        mass_nonnegative=mass >= -tol,
        finite_k_tensor_gate_passed=passed,
        vector_sector_computed=False,
        scalar_sector_computed=False,
        strong_coupling_scale_derived=False,
        one_loop_st_identity_computed=False,
        brst_physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=(
            'FINITE_K_TENSOR_GATE_PASSED'
            if passed
            else 'FINITE_K_TENSOR_GATE_FAILED'
        ),
    )


@dataclass(frozen=True)
class SuppliedTensorCutoffAudit:
    supplied_cutoff_over_mu_x: float
    max_background_or_tensor_scale_bar: float
    hierarchy_ratio: float
    below_supplied_cutoff: bool
    cutoff_derived_from_m1: bool


def audit_supplied_tensor_cutoff(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    cutoff_over_mu_x: float,
) -> SuppliedTensorCutoffAudit:
    '''Compare classical scales to a caller-supplied cutoff; do not derive it.'''

    cutoff = float(cutoff_over_mu_x)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError('the supplied cutoff ratio must be finite and positive')
    mass = tensor_mass_squared_bar(state, parameters)
    scale = max(abs(state.h), abs(state.u), abs(state.b), np.sqrt(max(0.0, mass)))
    ratio = float(scale / cutoff)
    return SuppliedTensorCutoffAudit(
        supplied_cutoff_over_mu_x=cutoff,
        max_background_or_tensor_scale_bar=float(scale),
        hierarchy_ratio=ratio,
        below_supplied_cutoff=ratio < 1.0,
        cutoff_derived_from_m1=False,
    )
