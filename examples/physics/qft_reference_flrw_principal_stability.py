'''Frozen high-frequency stability gate for the E62 reference background.

In harmonic/de Donder gauge, minimally coupled Einstein gravity plus the M1
canonical scalars has a block-diagonal second-derivative principal part.  The
background gradients Tdot and beta/a enter metric--scalar mixing with at most
one perturbation derivative, so they are subprincipal in the strict k ->
infinity limit.

This module audits only the reduced physical principal quadratic form: two TT
metric polarizations, chi, and four X^A perturbations.  It does not eliminate
the full finite-k lapse/shift system, compute mass/mixing eigenvalues, establish
a strong-coupling scale, or construct a BRST physical inner product.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    audit_reference_flrw_background,
)


TOL = 1.0e-10
PHYSICAL_MODE_NAMES = ('TT_plus', 'TT_cross', 'chi', 'X0', 'X1', 'X2', 'X3')


@dataclass(frozen=True)
class PrincipalMatrices:
    kinetic: np.ndarray
    gradient: np.ndarray
    mode_names: tuple[str, ...]


def _validate_symmetric_matrix(matrix: np.ndarray, size: int, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f'{label} must be {size} by {size}')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{label} must be finite')
    if np.linalg.norm(matrix - matrix.T) > TOL:
        raise ValueError(f'{label} must be symmetric')
    return matrix


def physical_principal_matrices(
    parameters: ReferenceFlrwParameters,
    *,
    internal_kinetic_metric: np.ndarray | None = None,
    internal_gradient_metric: np.ndarray | None = None,
    chi_x0_kinetic_mixing: float = 0.0,
    chi_x0_gradient_mixing: float | None = None,
) -> PrincipalMatrices:
    '''Return dimensionless K and G in 1/2 qdot K qdot-1/2 grad q G grad q.

    TT polarization tensors have unit Frobenius norm, giving K_TT=G_TT=m^2/4.
    In dimensionless coordinates the scalar variables are normalized as
    (delta chi/mu_X, delta X^A), so the admitted internal metric is I_4.
    Optional deformations exist only for negative-control tests.
    '''

    m = float(parameters.m_planck_over_mu_x)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    if internal_kinetic_metric is None:
        internal_kinetic_metric = np.eye(4)
    internal_kinetic_metric = _validate_symmetric_matrix(
        internal_kinetic_metric, 4, 'internal kinetic metric'
    )
    if internal_gradient_metric is None:
        internal_gradient_metric = internal_kinetic_metric.copy()
    internal_gradient_metric = _validate_symmetric_matrix(
        internal_gradient_metric, 4, 'internal gradient metric'
    )
    kinetic_mixing = float(chi_x0_kinetic_mixing)
    if not np.isfinite(kinetic_mixing):
        raise ValueError('kinetic mixing must be finite')
    if chi_x0_gradient_mixing is None:
        gradient_mixing = kinetic_mixing
    else:
        gradient_mixing = float(chi_x0_gradient_mixing)
    if not np.isfinite(gradient_mixing):
        raise ValueError('gradient mixing must be finite')

    kinetic = np.zeros((7, 7))
    gradient = np.zeros((7, 7))
    kinetic[:2, :2] = np.eye(2) * (m**2 / 4.0)
    gradient[:2, :2] = np.eye(2) * (m**2 / 4.0)
    kinetic[2, 2] = 1.0
    gradient[2, 2] = 1.0
    kinetic[3:, 3:] = internal_kinetic_metric
    gradient[3:, 3:] = internal_gradient_metric
    kinetic[2, 3] = kinetic[3, 2] = kinetic_mixing
    gradient[2, 3] = gradient[3, 2] = gradient_mixing
    return PrincipalMatrices(
        kinetic=kinetic,
        gradient=gradient,
        mode_names=PHYSICAL_MODE_NAMES,
    )


@dataclass(frozen=True)
class LinearizedDerivativeOrderAudit:
    equation_field_blocks: tuple[str, str]
    derivative_orders: tuple[tuple[int, int], tuple[int, int]]
    background_gradient_norm: float
    subprincipal_gradient_coefficient_norm: float
    principal_off_diagonal_norm: float
    metric_scalar_mixing_is_strictly_subprincipal: bool


def linearized_derivative_order_audit(
    state: ReferenceFlrwState,
) -> LinearizedDerivativeOrderAudit:
    '''Encode derivative orders of the linearized metric/scalar equation blocks.

    Rows are (Einstein, scalar) equations and columns are (metric, scalar)
    perturbations.  Harmonic reduction gives two derivatives in the diagonal
    wave blocks.  Variation of T_mn gives background-gradient times one
    derivative of delta X, while variation of box X gives background-gradient
    times the one-derivative de Donder residual.  Curvature/Hessian terms have
    order zero.
    '''

    gradient_norm = float(np.sqrt(state.u**2 + 3.0 * state.b**2))
    derivative_orders = ((2, 1), (1, 2))
    principal_off_diagonal = np.array(
        [
            0.0 if derivative_orders[0][1] < 2 else gradient_norm,
            0.0 if derivative_orders[1][0] < 2 else gradient_norm,
        ]
    )
    return LinearizedDerivativeOrderAudit(
        equation_field_blocks=('metric', 'five_scalars'),
        derivative_orders=derivative_orders,
        background_gradient_norm=gradient_norm,
        subprincipal_gradient_coefficient_norm=float(np.sqrt(2.0) * gradient_norm),
        principal_off_diagonal_norm=float(np.linalg.norm(principal_off_diagonal)),
        metric_scalar_mixing_is_strictly_subprincipal=(
            derivative_orders[0][1] < 2 and derivative_orders[1][0] < 2
        ),
    )


def spatial_tt_basis(spatial_wavevector_bar: np.ndarray) -> np.ndarray:
    '''Return two Frobenius-normalized 3x3 TT tensors for any nonzero direction.'''

    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the dimensionless spatial wavevector must have three components')
    norm = float(np.linalg.norm(wavevector))
    if norm <= TOL:
        raise ValueError('the TT decomposition requires nonzero spatial momentum')
    direction = wavevector / norm
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    transverse_one = np.cross(direction, reference)
    transverse_one /= np.linalg.norm(transverse_one)
    transverse_two = np.cross(direction, transverse_one)
    plus = (
        np.outer(transverse_one, transverse_one)
        - np.outer(transverse_two, transverse_two)
    ) / np.sqrt(2.0)
    cross = (
        np.outer(transverse_one, transverse_two)
        + np.outer(transverse_two, transverse_one)
    ) / np.sqrt(2.0)
    return np.stack((plus, cross))


@dataclass(frozen=True)
class PrincipalStabilityAudit:
    kinetic_eigenvalues: tuple[float, ...]
    gradient_eigenvalues: tuple[float, ...]
    speed_squared_eigenvalues: tuple[float, ...]
    declared_physical_principal_mode_count: int
    principal_background_mixing_norm: float
    background_gradient_subprincipal_norm: float
    background_gradients_are_subprincipal: bool
    real_characteristics: bool
    uniformly_diagonalizable_physical_symbol: bool
    physical_kinetic_positive: bool
    physical_gradient_positive: bool
    high_frequency_principal_gate_passed: bool
    auxiliary_metric_components_counted_as_physical: bool
    harmonic_constraint_propagation_computed: bool
    finite_k_hessian_computed: bool
    strong_coupling_scale_computed: bool
    brst_physical_inner_product_constructed: bool
    one_loop_st_identity_computed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_frozen_principal_stability(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    internal_kinetic_metric: np.ndarray | None = None,
    internal_gradient_metric: np.ndarray | None = None,
    chi_x0_kinetic_mixing: float = 0.0,
    chi_x0_gradient_mixing: float | None = None,
    tol: float = TOL,
) -> PrincipalStabilityAudit:
    '''Audit the physical seven-mode principal pencil G-c^2 K.'''

    background = audit_reference_flrw_background(state, parameters)
    if not background.local_reference_patch_admitted:
        raise ValueError('E63 requires an admitted nondegenerate E62 background')
    matrices = physical_principal_matrices(
        parameters,
        internal_kinetic_metric=internal_kinetic_metric,
        internal_gradient_metric=internal_gradient_metric,
        chi_x0_kinetic_mixing=chi_x0_kinetic_mixing,
        chi_x0_gradient_mixing=chi_x0_gradient_mixing,
    )
    kinetic_eigenvalues = np.linalg.eigvalsh(matrices.kinetic)
    gradient_eigenvalues = np.linalg.eigvalsh(matrices.gradient)
    kinetic_positive = bool(np.all(kinetic_eigenvalues > tol))
    gradient_positive = bool(np.all(gradient_eigenvalues > tol))

    if kinetic_positive:
        kinetic_values, kinetic_vectors = np.linalg.eigh(matrices.kinetic)
        inverse_sqrt = kinetic_vectors @ np.diag(kinetic_values ** -0.5) @ kinetic_vectors.T
        symmetric_pencil = inverse_sqrt @ matrices.gradient @ inverse_sqrt
        speeds = np.linalg.eigvalsh(symmetric_pencil)
        real_characteristics = bool(np.all(np.isfinite(speeds)))
        diagonalizable = real_characteristics
    else:
        speeds = np.full(7, np.nan)
        real_characteristics = False
        diagonalizable = False
    principal_passed = (
        kinetic_positive
        and gradient_positive
        and real_characteristics
        and diagonalizable
        and bool(np.all(speeds > tol))
    )
    derivative_order = linearized_derivative_order_audit(state)
    return PrincipalStabilityAudit(
        kinetic_eigenvalues=tuple(float(value) for value in kinetic_eigenvalues),
        gradient_eigenvalues=tuple(float(value) for value in gradient_eigenvalues),
        speed_squared_eigenvalues=tuple(float(value) for value in speeds),
        declared_physical_principal_mode_count=7,
        principal_background_mixing_norm=derivative_order.principal_off_diagonal_norm,
        background_gradient_subprincipal_norm=(
            derivative_order.subprincipal_gradient_coefficient_norm
        ),
        background_gradients_are_subprincipal=True,
        real_characteristics=real_characteristics,
        uniformly_diagonalizable_physical_symbol=diagonalizable,
        physical_kinetic_positive=kinetic_positive,
        physical_gradient_positive=gradient_positive,
        high_frequency_principal_gate_passed=principal_passed,
        auxiliary_metric_components_counted_as_physical=False,
        harmonic_constraint_propagation_computed=False,
        finite_k_hessian_computed=False,
        strong_coupling_scale_computed=False,
        brst_physical_inner_product_constructed=False,
        one_loop_st_identity_computed=False,
        nonperturbative_m2_passed=False,
        status=(
            'FROZEN_HIGH_FREQUENCY_PRINCIPAL_GATE_PASSED'
            if principal_passed
            else 'FROZEN_HIGH_FREQUENCY_PRINCIPAL_GATE_FAILED'
        ),
    )


def physical_principal_pencil(
    frequency_bar: float,
    spatial_wavenumber_bar: float,
    matrices: PrincipalMatrices,
) -> np.ndarray:
    '''Return -omega^2 K+k_phys^2 G for the seven physical modes.'''

    frequency = float(frequency_bar)
    wavenumber = float(spatial_wavenumber_bar)
    if not np.all(np.isfinite([frequency, wavenumber])):
        raise ValueError('dimensionless frequency and wavenumber must be finite')
    return -(frequency**2) * matrices.kinetic + (wavenumber**2) * matrices.gradient


def faddeev_popov_principal_symbol(
    frequency_bar: float, spatial_wavevector_bar: np.ndarray
) -> np.ndarray:
    '''Return the de Donder FP wave symbol (-omega^2+|k|^2) I_4.'''

    wavevector = np.asarray(spatial_wavevector_bar, dtype=float)
    if wavevector.shape != (3,) or not np.all(np.isfinite(wavevector)):
        raise ValueError('the dimensionless spatial wavevector must have three components')
    frequency = float(frequency_bar)
    if not np.isfinite(frequency):
        raise ValueError('the dimensionless frequency must be finite')
    wave_symbol = -frequency**2 + float(wavevector @ wavevector)
    return np.eye(4) * wave_symbol


@dataclass(frozen=True)
class SubprincipalPowerCounting:
    background_scale_bar: float
    wavenumber_bar: float
    one_derivative_to_principal_ratio: float
    curvature_to_principal_ratio: float
    strict_high_frequency_limit_zero: bool


def audit_subprincipal_power_counting(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    wavenumber_bar: float,
) -> SubprincipalPowerCounting:
    '''Power-count frozen O(k) gradient mixing and O(k^0) curvature terms.'''

    wavenumber = float(wavenumber_bar)
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError('the dimensionless wavenumber must be finite and positive')
    scale = max(
        abs(state.h),
        abs(state.u),
        abs(state.b),
        np.sqrt(abs(parameters.lambda_over_mu_x_squared)),
    )
    return SubprincipalPowerCounting(
        background_scale_bar=float(scale),
        wavenumber_bar=wavenumber,
        one_derivative_to_principal_ratio=float(scale / wavenumber),
        curvature_to_principal_ratio=float((scale / wavenumber) ** 2),
        strict_high_frequency_limit_zero=True,
    )


@dataclass(frozen=True)
class FiniteWavenumberTachyonAudit:
    mass_squared_bar: float
    wavenumber_bar: float
    omega_squared_bar: float
    principal_kinetic_positive: bool
    principal_gradient_positive: bool
    finite_wavenumber_stable: bool


def audit_finite_wavenumber_tachyon(
    *, mass_squared_bar: float, wavenumber_bar: float
) -> FiniteWavenumberTachyonAudit:
    '''Negative control: K=G=1 can coexist with omega^2=k^2+m_eff^2<0.'''

    mass_squared = float(mass_squared_bar)
    wavenumber = float(wavenumber_bar)
    if not np.all(np.isfinite([mass_squared, wavenumber])) or wavenumber < 0.0:
        raise ValueError('finite dimensionless mass and nonnegative wavenumber are required')
    omega_squared = wavenumber**2 + mass_squared
    return FiniteWavenumberTachyonAudit(
        mass_squared_bar=mass_squared,
        wavenumber_bar=wavenumber,
        omega_squared_bar=float(omega_squared),
        principal_kinetic_positive=True,
        principal_gradient_positive=True,
        finite_wavenumber_stable=omega_squared >= 0.0,
    )
