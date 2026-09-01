'''Finite-time canonical mode evolution for the E62--E66 quadratic sectors.

This module integrates the E62 background and tensor, vector, coupled scalar,
and free-chi fundamental matrices in dimensionless time tau=mu_X t.  It is a
finite-interval linear screening receipt, not an asymptotic or nonlinear
stability theorem.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    energy_density_bar,
    expanding_h_from_constraint,
    flrw_rhs,
    friedmann_residual,
    reference_jacobian_bar,
)
from examples.physics.qft_reference_flrw_scalar_stability import (
    reduced_scalar_matrices,
    scalar_constraint_blocks,
)


TOL = 1.0e-10
E67_TAU_FINAL = 0.5
E67_WAVENUMBERS_BAR = (0.05, 0.2, 1.0, 3.0)
E67_CHI_MASS_OVER_MU_X = 0.4
E67_COARSE_STEPS = 1000
E67_FINE_STEPS = 2000
E67_BACKGROUND_TOL = 1.0e-10
E67_SYMPLECTIC_TOL = 1.0e-8
E67_REFINEMENT_TOL = 1.0e-6
E67_SECOND_ORDER_TOL = 1.0e-7
E67_GROWTH_LIMIT = 100.0


def canonical_symplectic_matrix(configuration_count: int) -> np.ndarray:
    if not isinstance(configuration_count, int) or configuration_count <= 0:
        raise ValueError('configuration_count must be a positive integer')
    identity = np.eye(configuration_count)
    zero = np.zeros_like(identity)
    return np.block([[zero, identity], [-identity, zero]])


def hamiltonian_generator_residual(generator: np.ndarray) -> float:
    generator = np.asarray(generator, dtype=float)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError('a Hamiltonian generator must be square')
    if generator.shape[0] % 2:
        raise ValueError('a canonical generator must have even dimension')
    symplectic = canonical_symplectic_matrix(generator.shape[0] // 2)
    return float(
        np.linalg.norm(generator.T @ symplectic + symplectic @ generator)
    )


@dataclass(frozen=True)
class ModeGenerators:
    tensor: np.ndarray
    vector: np.ndarray
    scalar: np.ndarray
    chi: np.ndarray
    physical_wavenumber_bar: float
    tensor_mass_squared_bar: float
    vector_kinetic_ratio: float
    scalar_constraint_determinant_bar: float
    minimum_scalar_kinetic_eigenvalue: float
    maximum_hamiltonian_generator_residual: float


def canonical_mode_generators(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    chi_mass_over_mu_x: float,
    tol: float = TOL,
) -> ModeGenerators:
    '''Return first-order canonical generators for all quadratic sectors.'''

    values = state.as_array()
    m = float(parameters.m_planck_over_mu_x)
    k_comoving = float(comoving_wavenumber_bar)
    chi_mass = float(chi_mass_over_mu_x)
    if not np.all(np.isfinite(values)):
        raise ValueError('mode evolution requires a finite FLRW state')
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    if not np.isfinite(k_comoving) or k_comoving <= 0.0:
        raise ValueError('finite-time mode evolution requires positive k')
    if not np.isfinite(chi_mass) or chi_mass < 0.0:
        raise ValueError('m_chi/mu_X must be finite and nonnegative')
    a = float(np.exp(state.n))
    kappa = k_comoving / a
    a_cubed = a**3
    tensor_mass_squared = 2.0 * state.b**2 / m**2

    tensor = np.array(
        [
            [0.0, 1.0 / a_cubed],
            [-a_cubed * (kappa**2 + tensor_mass_squared), 0.0],
        ]
    )
    vector_kinetic_ratio = (
        m**2 * kappa**2 / (m**2 * kappa**2 + 2.0 * state.b**2)
    )
    if vector_kinetic_ratio <= tol:
        raise ValueError('the finite-k vector canonical kinetic ratio is singular')
    vector = np.array(
        [
            [0.0, 1.0 / (a_cubed * vector_kinetic_ratio)],
            [-a_cubed * kappa**2, 0.0],
        ]
    )

    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=k_comoving,
    )
    kinetic, gyroscopic, potential = reduced_scalar_matrices(blocks, tol=tol)
    kinetic_eigenvalues = np.linalg.eigvalsh(kinetic)
    if np.min(kinetic_eigenvalues) <= tol:
        raise ValueError('the reduced scalar kinetic matrix is not positive')
    inverse_kinetic = np.linalg.inv(kinetic)
    scalar = np.block(
        [
            [
                inverse_kinetic @ gyroscopic,
                inverse_kinetic / a_cubed,
            ],
            [
                -a_cubed
                * (
                    gyroscopic.T @ inverse_kinetic @ gyroscopic
                    + potential
                ),
                -gyroscopic.T @ inverse_kinetic,
            ],
        ]
    )
    chi = np.array(
        [
            [0.0, 1.0 / a_cubed],
            [-a_cubed * (kappa**2 + chi_mass**2), 0.0],
        ]
    )
    residual = max(
        hamiltonian_generator_residual(generator)
        for generator in (tensor, vector, scalar, chi)
    )
    return ModeGenerators(
        tensor=tensor,
        vector=vector,
        scalar=scalar,
        chi=chi,
        physical_wavenumber_bar=kappa,
        tensor_mass_squared_bar=tensor_mass_squared,
        vector_kinetic_ratio=float(vector_kinetic_ratio),
        scalar_constraint_determinant_bar=blocks.constraint_determinant_bar,
        minimum_scalar_kinetic_eigenvalue=float(np.min(kinetic_eigenvalues)),
        maximum_hamiltonian_generator_residual=float(residual),
    )


def _scalar_reduced_coefficients(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = scalar_constraint_blocks(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    return reduced_scalar_matrices(blocks)


def scalar_second_order_equivalence_residual(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    comoving_wavenumber_bar: float,
    chi_mass_over_mu_x: float,
    scalar_fundamental: np.ndarray,
    derivative_step: float = 1.0e-5,
    scalar_generator: np.ndarray | None = None,
) -> float:
    '''Check the canonical flow against the equivalent scalar second-order EOM.'''

    fundamental = np.asarray(scalar_fundamental, dtype=float)
    if fundamental.shape != (4, 4) or not np.all(np.isfinite(fundamental)):
        raise ValueError('the scalar fundamental matrix must be finite and 4x4')
    derivative_step = float(derivative_step)
    if not np.isfinite(derivative_step) or derivative_step <= 0.0:
        raise ValueError('derivative_step must be finite and positive')
    tangent = flrw_rhs(state, parameters).as_array()
    plus_state = ReferenceFlrwState.from_array(
        state.as_array() + derivative_step * tangent
    )
    minus_state = ReferenceFlrwState.from_array(
        state.as_array() - derivative_step * tangent
    )
    kinetic, gyroscopic, potential = _scalar_reduced_coefficients(
        state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic_plus, gyroscopic_plus, _ = _scalar_reduced_coefficients(
        plus_state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic_minus, gyroscopic_minus, _ = _scalar_reduced_coefficients(
        minus_state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
    )
    kinetic_derivative = (
        kinetic_plus - kinetic_minus
    ) / (2.0 * derivative_step)
    gyroscopic_derivative = (
        gyroscopic_plus - gyroscopic_minus
    ) / (2.0 * derivative_step)
    generator = (
        canonical_mode_generators(
            state,
            parameters,
            comoving_wavenumber_bar=comoving_wavenumber_bar,
            chi_mass_over_mu_x=chi_mass_over_mu_x,
        ).scalar
        if scalar_generator is None
        else np.asarray(scalar_generator, dtype=float)
    )
    if generator.shape != (4, 4):
        raise ValueError('the scalar canonical generator must be 4x4')
    generator_plus = canonical_mode_generators(
        plus_state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
        chi_mass_over_mu_x=chi_mass_over_mu_x,
    ).scalar
    generator_minus = canonical_mode_generators(
        minus_state,
        parameters,
        comoving_wavenumber_bar=comoving_wavenumber_bar,
        chi_mass_over_mu_x=chi_mass_over_mu_x,
    ).scalar
    generator_derivative = (
        generator_plus - generator_minus
    ) / (2.0 * derivative_step)
    phase_velocity = generator @ fundamental
    phase_acceleration = (
        generator_derivative + generator @ generator
    ) @ fundamental
    configuration = fundamental[:2, :]
    configuration_velocity = phase_velocity[:2, :]
    configuration_acceleration = phase_acceleration[:2, :]
    kinetic_term = kinetic @ configuration_acceleration
    velocity_term = (
        kinetic_derivative
        + 3.0 * state.h * kinetic
        + gyroscopic.T
        - gyroscopic
    ) @ configuration_velocity
    field_term = (
        potential
        - gyroscopic_derivative
        - 3.0 * state.h * gyroscopic
    ) @ configuration
    residual = kinetic_term + velocity_term + field_term
    scale = max(
        1.0,
        float(
            np.linalg.norm(kinetic_term)
            + np.linalg.norm(velocity_term)
            + np.linalg.norm(field_term)
        ),
    )
    return float(np.linalg.norm(residual) / scale)


@dataclass(frozen=True)
class FundamentalMatrixDiagnostics:
    symplectic_residual: float
    determinant_residual: float
    largest_singular_value: float


def fundamental_matrix_diagnostics(
    fundamental_matrix: np.ndarray,
) -> FundamentalMatrixDiagnostics:
    fundamental = np.asarray(fundamental_matrix, dtype=float)
    if (
        fundamental.ndim != 2
        or fundamental.shape[0] != fundamental.shape[1]
        or fundamental.shape[0] % 2
    ):
        raise ValueError('a canonical fundamental matrix must be even and square')
    symplectic = canonical_symplectic_matrix(fundamental.shape[0] // 2)
    return FundamentalMatrixDiagnostics(
        symplectic_residual=float(
            np.linalg.norm(fundamental.T @ symplectic @ fundamental - symplectic)
        ),
        determinant_residual=float(abs(np.linalg.det(fundamental) - 1.0)),
        largest_singular_value=float(
            np.linalg.svd(fundamental, compute_uv=False)[0]
        ),
    )


SECTOR_DIMENSIONS = {
    'tensor': 2,
    'vector': 2,
    'scalar': 4,
    'chi': 2,
}


def _mode_layout(
    wavenumbers_bar: tuple[float, ...],
) -> dict[tuple[int, str], slice]:
    layout: dict[tuple[int, str], slice] = {}
    cursor = 5
    for wavenumber_index in range(len(wavenumbers_bar)):
        for sector, dimension in SECTOR_DIMENSIONS.items():
            layout[(wavenumber_index, sector)] = slice(
                cursor, cursor + dimension**2
            )
            cursor += dimension**2
    return layout


def _initial_joint_state(
    state: ReferenceFlrwState,
    wavenumbers_bar: tuple[float, ...],
    layout: dict[tuple[int, str], slice],
) -> np.ndarray:
    total_size = max(section.stop for section in layout.values())
    joint = np.zeros(total_size)
    joint[:5] = state.as_array()
    for wavenumber_index in range(len(wavenumbers_bar)):
        for sector, dimension in SECTOR_DIMENSIONS.items():
            joint[layout[(wavenumber_index, sector)]] = np.eye(dimension).ravel()
    return joint


def _joint_rhs(
    joint: np.ndarray,
    parameters: ReferenceFlrwParameters,
    wavenumbers_bar: tuple[float, ...],
    chi_mass_over_mu_x: float,
    layout: dict[tuple[int, str], slice],
) -> np.ndarray:
    state = ReferenceFlrwState.from_array(joint[:5])
    derivative = np.zeros_like(joint)
    derivative[:5] = flrw_rhs(state, parameters).as_array()
    for wavenumber_index, wavenumber in enumerate(wavenumbers_bar):
        generators = canonical_mode_generators(
            state,
            parameters,
            comoving_wavenumber_bar=wavenumber,
            chi_mass_over_mu_x=chi_mass_over_mu_x,
        )
        for sector, dimension in SECTOR_DIMENSIONS.items():
            fundamental = joint[
                layout[(wavenumber_index, sector)]
            ].reshape(dimension, dimension)
            generator = getattr(generators, sector)
            derivative[layout[(wavenumber_index, sector)]] = (
                generator @ fundamental
            ).ravel()
    return derivative


def _friedmann_relative_residual(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
) -> float:
    m = float(parameters.m_planck_over_mu_x)
    lam = float(parameters.lambda_over_mu_x_squared)
    rho = energy_density_bar(state)
    scale = 1.0 + abs(3.0 * m**2 * state.h**2) + abs(rho) + abs(m**2 * lam)
    return float(abs(friedmann_residual(state, parameters)) / scale)


@dataclass(frozen=True)
class WavenumberModeEvolution:
    comoving_wavenumber_bar: float
    final_tensor_fundamental: np.ndarray
    final_vector_fundamental: np.ndarray
    final_scalar_fundamental: np.ndarray
    final_chi_fundamental: np.ndarray
    maximum_symplectic_residual: float
    maximum_determinant_residual: float
    maximum_largest_singular_value: float
    minimum_vector_kinetic_ratio: float
    minimum_scalar_kinetic_eigenvalue: float
    minimum_abs_scalar_constraint_determinant: float
    maximum_hamiltonian_generator_residual: float
    maximum_scalar_second_order_residual: float


@dataclass(frozen=True)
class FiniteTimeModeRun:
    tau_final: float
    steps: int
    final_background_state: ReferenceFlrwState
    maximum_friedmann_relative_residual: float
    minimum_abs_reference_jacobian_bar: float
    wavenumber_evolutions: tuple[WavenumberModeEvolution, ...]
    refinement_checkpoint_joint_states: tuple[np.ndarray, ...]


def integrate_finite_time_modes(
    initial_state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    tau_final: float,
    steps: int,
    wavenumbers_bar: tuple[float, ...],
    chi_mass_over_mu_x: float,
) -> FiniteTimeModeRun:
    '''Joint RK4 integration of the E62 background and canonical matrices.'''

    tau_final = float(tau_final)
    chi_mass = float(chi_mass_over_mu_x)
    if not np.isfinite(tau_final) or tau_final <= 0.0:
        raise ValueError('tau_final must be finite and positive')
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if not wavenumbers_bar:
        raise ValueError('at least one comoving wavenumber is required')
    wavenumbers = tuple(float(value) for value in wavenumbers_bar)
    if any(not np.isfinite(value) or value <= 0.0 for value in wavenumbers):
        raise ValueError('all comoving wavenumbers must be finite and positive')
    if len(set(wavenumbers)) != len(wavenumbers):
        raise ValueError('comoving wavenumbers must be unique')
    if not np.isfinite(chi_mass) or chi_mass < 0.0:
        raise ValueError('m_chi/mu_X must be finite and nonnegative')

    layout = _mode_layout(wavenumbers)
    joint = _initial_joint_state(initial_state, wavenumbers, layout)
    step_size = tau_final / steps
    maximum_friedmann = 0.0
    minimum_jacobian = np.inf
    statistics = [
        {
            'symplectic': 0.0,
            'determinant': 0.0,
            'growth': 1.0,
            'vector_kinetic': np.inf,
            'scalar_kinetic': np.inf,
            'constraint': np.inf,
            'generator': 0.0,
            'second_order': 0.0,
        }
        for _ in wavenumbers
    ]
    checkpoint_indices = set(
        int(value) for value in np.linspace(0, steps, min(steps, 20) + 1)
    )
    checkpoints = [joint.copy()]

    def monitor(vector: np.ndarray) -> None:
        nonlocal maximum_friedmann, minimum_jacobian
        state = ReferenceFlrwState.from_array(vector[:5])
        maximum_friedmann = max(
            maximum_friedmann,
            _friedmann_relative_residual(state, parameters),
        )
        minimum_jacobian = min(
            minimum_jacobian, abs(reference_jacobian_bar(state))
        )
        for wavenumber_index, wavenumber in enumerate(wavenumbers):
            generators = canonical_mode_generators(
                state,
                parameters,
                comoving_wavenumber_bar=wavenumber,
                chi_mass_over_mu_x=chi_mass,
            )
            statistic = statistics[wavenumber_index]
            statistic['vector_kinetic'] = min(
                statistic['vector_kinetic'], generators.vector_kinetic_ratio
            )
            statistic['scalar_kinetic'] = min(
                statistic['scalar_kinetic'],
                generators.minimum_scalar_kinetic_eigenvalue,
            )
            statistic['constraint'] = min(
                statistic['constraint'],
                abs(generators.scalar_constraint_determinant_bar),
            )
            statistic['generator'] = max(
                statistic['generator'],
                generators.maximum_hamiltonian_generator_residual,
            )
            scalar_fundamental = vector[
                layout[(wavenumber_index, 'scalar')]
            ].reshape(4, 4)
            statistic['second_order'] = max(
                statistic['second_order'],
                scalar_second_order_equivalence_residual(
                    state,
                    parameters,
                    comoving_wavenumber_bar=wavenumber,
                    chi_mass_over_mu_x=chi_mass,
                    scalar_fundamental=scalar_fundamental,
                    scalar_generator=generators.scalar,
                ),
            )
            for sector, dimension in SECTOR_DIMENSIONS.items():
                fundamental = vector[
                    layout[(wavenumber_index, sector)]
                ].reshape(dimension, dimension)
                diagnostics = fundamental_matrix_diagnostics(fundamental)
                statistic['symplectic'] = max(
                    statistic['symplectic'], diagnostics.symplectic_residual
                )
                statistic['determinant'] = max(
                    statistic['determinant'], diagnostics.determinant_residual
                )
                statistic['growth'] = max(
                    statistic['growth'], diagnostics.largest_singular_value
                )

    monitor(joint)
    for step_index in range(1, steps + 1):
        first = _joint_rhs(
            joint, parameters, wavenumbers, chi_mass, layout
        )
        second = _joint_rhs(
            joint + 0.5 * step_size * first,
            parameters,
            wavenumbers,
            chi_mass,
            layout,
        )
        third = _joint_rhs(
            joint + 0.5 * step_size * second,
            parameters,
            wavenumbers,
            chi_mass,
            layout,
        )
        fourth = _joint_rhs(
            joint + step_size * third,
            parameters,
            wavenumbers,
            chi_mass,
            layout,
        )
        joint = joint + (step_size / 6.0) * (
            first + 2.0 * second + 2.0 * third + fourth
        )
        if not np.all(np.isfinite(joint)):
            raise ValueError('the finite-time joint evolution became nonfinite')
        monitor(joint)
        if step_index in checkpoint_indices:
            checkpoints.append(joint.copy())

    evolutions = []
    for wavenumber_index, wavenumber in enumerate(wavenumbers):
        statistic = statistics[wavenumber_index]
        matrices = {
            sector: joint[layout[(wavenumber_index, sector)]].reshape(
                dimension, dimension
            )
            for sector, dimension in SECTOR_DIMENSIONS.items()
        }
        evolutions.append(
            WavenumberModeEvolution(
                comoving_wavenumber_bar=wavenumber,
                final_tensor_fundamental=matrices['tensor'],
                final_vector_fundamental=matrices['vector'],
                final_scalar_fundamental=matrices['scalar'],
                final_chi_fundamental=matrices['chi'],
                maximum_symplectic_residual=float(statistic['symplectic']),
                maximum_determinant_residual=float(statistic['determinant']),
                maximum_largest_singular_value=float(statistic['growth']),
                minimum_vector_kinetic_ratio=float(statistic['vector_kinetic']),
                minimum_scalar_kinetic_eigenvalue=float(
                    statistic['scalar_kinetic']
                ),
                minimum_abs_scalar_constraint_determinant=float(
                    statistic['constraint']
                ),
                maximum_hamiltonian_generator_residual=float(
                    statistic['generator']
                ),
                maximum_scalar_second_order_residual=float(
                    statistic['second_order']
                ),
            )
        )
    return FiniteTimeModeRun(
        tau_final=tau_final,
        steps=steps,
        final_background_state=ReferenceFlrwState.from_array(joint[:5]),
        maximum_friedmann_relative_residual=float(maximum_friedmann),
        minimum_abs_reference_jacobian_bar=float(minimum_jacobian),
        wavenumber_evolutions=tuple(evolutions),
        refinement_checkpoint_joint_states=tuple(checkpoints),
    )


def _run_refinement_residual(
    coarse: FiniteTimeModeRun,
    fine: FiniteTimeModeRun,
) -> float:
    residuals = [
        np.linalg.norm(
            coarse.final_background_state.as_array()
            - fine.final_background_state.as_array()
        )
        / max(1.0, np.linalg.norm(fine.final_background_state.as_array()))
    ]
    if len(coarse.refinement_checkpoint_joint_states) != len(
        fine.refinement_checkpoint_joint_states
    ):
        raise ValueError('coarse and fine checkpoint manifests differ')
    residuals.extend(
        np.linalg.norm(coarse_checkpoint - fine_checkpoint)
        / max(1.0, np.linalg.norm(fine_checkpoint))
        for coarse_checkpoint, fine_checkpoint in zip(
            coarse.refinement_checkpoint_joint_states,
            fine.refinement_checkpoint_joint_states,
            strict=True,
        )
    )
    for coarse_mode, fine_mode in zip(
        coarse.wavenumber_evolutions,
        fine.wavenumber_evolutions,
        strict=True,
    ):
        if coarse_mode.comoving_wavenumber_bar != fine_mode.comoving_wavenumber_bar:
            raise ValueError('coarse and fine wavenumber manifests differ')
        for sector in ('tensor', 'vector', 'scalar', 'chi'):
            coarse_matrix = getattr(
                coarse_mode, f'final_{sector}_fundamental'
            )
            fine_matrix = getattr(fine_mode, f'final_{sector}_fundamental')
            residuals.append(
                np.linalg.norm(coarse_matrix - fine_matrix)
                / max(1.0, np.linalg.norm(fine_matrix))
            )
    return float(max(residuals))


@dataclass(frozen=True)
class FiniteTimeSymplecticAudit:
    coarse_run: FiniteTimeModeRun
    fine_run: FiniteTimeModeRun
    maximum_refinement_residual: float
    maximum_symplectic_residual: float
    maximum_determinant_residual: float
    maximum_largest_singular_value: float
    minimum_vector_kinetic_ratio: float
    minimum_scalar_kinetic_eigenvalue: float
    minimum_abs_scalar_constraint_determinant: float
    maximum_scalar_second_order_residual: float
    finite_time_symplectic_gate_passed: bool
    predeclared_finite_interval_only: bool
    uniform_wavenumber_band_proved: bool
    asymptotic_stability_proved: bool
    nonlinear_stability_proved: bool
    strong_coupling_scale_derived: bool
    one_loop_st_identity_computed: bool
    brst_physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_preregistered_e67_witness() -> FiniteTimeSymplecticAudit:
    '''Run exactly the E67 witness and thresholds frozen in plan 36.'''

    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    initial_state = ReferenceFlrwState(
        n=0.0,
        h=expanding_h_from_constraint(u=u, b=b, parameters=parameters),
        clock=0.0,
        u=u,
        b=b,
    )
    coarse = integrate_finite_time_modes(
        initial_state,
        parameters,
        tau_final=E67_TAU_FINAL,
        steps=E67_COARSE_STEPS,
        wavenumbers_bar=E67_WAVENUMBERS_BAR,
        chi_mass_over_mu_x=E67_CHI_MASS_OVER_MU_X,
    )
    fine = integrate_finite_time_modes(
        initial_state,
        parameters,
        tau_final=E67_TAU_FINAL,
        steps=E67_FINE_STEPS,
        wavenumbers_bar=E67_WAVENUMBERS_BAR,
        chi_mass_over_mu_x=E67_CHI_MASS_OVER_MU_X,
    )
    refinement = _run_refinement_residual(coarse, fine)
    symplectic = max(
        evolution.maximum_symplectic_residual
        for evolution in fine.wavenumber_evolutions
    )
    determinant = max(
        evolution.maximum_determinant_residual
        for evolution in fine.wavenumber_evolutions
    )
    growth = max(
        evolution.maximum_largest_singular_value
        for evolution in fine.wavenumber_evolutions
    )
    minimum_vector_kinetic = min(
        evolution.minimum_vector_kinetic_ratio
        for evolution in fine.wavenumber_evolutions
    )
    minimum_scalar_kinetic = min(
        evolution.minimum_scalar_kinetic_eigenvalue
        for evolution in fine.wavenumber_evolutions
    )
    minimum_constraint = min(
        evolution.minimum_abs_scalar_constraint_determinant
        for evolution in fine.wavenumber_evolutions
    )
    second_order = max(
        evolution.maximum_scalar_second_order_residual
        for evolution in fine.wavenumber_evolutions
    )
    passed = (
        fine.maximum_friedmann_relative_residual < E67_BACKGROUND_TOL
        and fine.minimum_abs_reference_jacobian_bar > TOL
        and refinement < E67_REFINEMENT_TOL
        and symplectic < E67_SYMPLECTIC_TOL
        and determinant < E67_SYMPLECTIC_TOL
        and growth < E67_GROWTH_LIMIT
        and minimum_vector_kinetic > TOL
        and minimum_scalar_kinetic > TOL
        and minimum_constraint > TOL
        and second_order < E67_SECOND_ORDER_TOL
    )
    return FiniteTimeSymplecticAudit(
        coarse_run=coarse,
        fine_run=fine,
        maximum_refinement_residual=refinement,
        maximum_symplectic_residual=symplectic,
        maximum_determinant_residual=determinant,
        maximum_largest_singular_value=growth,
        minimum_vector_kinetic_ratio=minimum_vector_kinetic,
        minimum_scalar_kinetic_eigenvalue=minimum_scalar_kinetic,
        minimum_abs_scalar_constraint_determinant=minimum_constraint,
        maximum_scalar_second_order_residual=second_order,
        finite_time_symplectic_gate_passed=passed,
        predeclared_finite_interval_only=True,
        uniform_wavenumber_band_proved=False,
        asymptotic_stability_proved=False,
        nonlinear_stability_proved=False,
        strong_coupling_scale_derived=False,
        one_loop_st_identity_computed=False,
        brst_physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=(
            'PREREGISTERED_FINITE_TIME_SYMPLECTIC_GATE_PASSED'
            if passed
            else 'PREREGISTERED_FINITE_TIME_SYMPLECTIC_GATE_FAILED'
        ),
    )
