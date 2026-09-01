'''Classical E62 admission for a nondegenerate FLRW reference background.

The M1 action contains four dimensionless canonical scalars X^A with positive
internal metric and coefficient mu_X^2.  In a local flat-FLRW chart this module
tests

    X^0 = T(t),                 X^i = beta x^i.

All numerical core variables are dimensionless:

    tau = mu_X t,  n = log(a/a_ref),  h = H/mu_X,
    u = dT/dt / mu_X,  b = beta/(mu_X a),
    m = M_P/mu_X,  lambda = Lambda/mu_X^2.

The construction is classical and local.  It does not calculate perturbation
stability, a loop Slavnov--Taylor identity, a BRST physical Hilbert space, or a
global real-valued rod chart on compact spatial slices.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOL = 1.0e-10
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


@dataclass(frozen=True)
class DimensionlessBackgroundAudit:
    base_mass_dimensions: tuple[tuple[str, int], ...]
    combination_mass_dimensions: tuple[tuple[str, int], ...]
    all_core_variables_dimensionless: bool
    physical_correctness_implied: bool


def audit_dimensionless_background_variables() -> DimensionlessBackgroundAudit:
    '''Audit natural-unit mass dimensions of every E62 numerical core variable.'''

    base = {
        't': -1,
        'a': 0,
        'T': 0,
        'X': 0,
        'mu_X': 1,
        'beta': 1,
        'H': 1,
        'dot_T': 1,
        'M_P': 1,
        'Lambda': 2,
        'R': 2,
        'J': 4,
    }
    combinations = {
        'tau=mu_X*t': base['mu_X'] + base['t'],
        'n=log(a/a_ref)': base['a'] - base['a'],
        'h=H/mu_X': base['H'] - base['mu_X'],
        'u=dot_T/mu_X': base['dot_T'] - base['mu_X'],
        'b=beta/(mu_X*a)': base['beta'] - base['mu_X'] - base['a'],
        'm=M_P/mu_X': base['M_P'] - base['mu_X'],
        'lambda=Lambda/mu_X^2': base['Lambda'] - 2 * base['mu_X'],
        'R_bar=R/mu_X^2': base['R'] - 2 * base['mu_X'],
        'J_bar=J/mu_X^4': base['J'] - 4 * base['mu_X'],
    }
    return DimensionlessBackgroundAudit(
        base_mass_dimensions=tuple(sorted(base.items())),
        combination_mass_dimensions=tuple(sorted(combinations.items())),
        all_core_variables_dimensionless=all(value == 0 for value in combinations.values()),
        physical_correctness_implied=False,
    )


@dataclass(frozen=True)
class ReferenceFlrwParameters:
    m_planck_over_mu_x: float
    lambda_over_mu_x_squared: float = 0.0


@dataclass(frozen=True)
class ReferenceFlrwState:
    n: float
    h: float
    clock: float
    u: float
    b: float

    def as_array(self) -> np.ndarray:
        return np.array([self.n, self.h, self.clock, self.u, self.b], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray) -> 'ReferenceFlrwState':
        values = np.asarray(values, dtype=float)
        if values.shape != (5,):
            raise ValueError('an FLRW reference state must have five components')
        return cls(*(float(value) for value in values))


def validate_parameters(parameters: ReferenceFlrwParameters) -> None:
    m = float(parameters.m_planck_over_mu_x)
    lam = float(parameters.lambda_over_mu_x_squared)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError('M_P/mu_X must be finite and positive')
    if not np.isfinite(lam):
        raise ValueError('Lambda/mu_X^2 must be finite')


def energy_density_bar(state: ReferenceFlrwState) -> float:
    '''Return rho_X/mu_X^4.'''

    return 0.5 * state.u**2 + 1.5 * state.b**2


def pressure_bar(state: ReferenceFlrwState) -> float:
    '''Return p_X/mu_X^4.'''

    return 0.5 * state.u**2 - 0.5 * state.b**2


def orthonormal_stress_bar(state: ReferenceFlrwState) -> np.ndarray:
    '''Construct T_(a)(b)/mu_X^4 directly from clock and rod gradients.'''

    gradients = np.zeros((4, 4))
    gradients[0, 0] = state.u
    for index in range(1, 4):
        gradients[index, index] = state.b
    stress = np.zeros((4, 4))
    for gradient in gradients:
        contraction = float(gradient @ ETA @ gradient)
        stress += np.outer(gradient, gradient) - 0.5 * ETA * contraction
    return stress


def bare_internal_kinetic_gram() -> np.ndarray:
    '''Return the declared M1 internal kinetic metric delta_AB.'''

    return np.eye(4)


def friedmann_residual(
    state: ReferenceFlrwState, parameters: ReferenceFlrwParameters
) -> float:
    m = parameters.m_planck_over_mu_x
    lam = parameters.lambda_over_mu_x_squared
    return 3.0 * m**2 * state.h**2 - energy_density_bar(state) - m**2 * lam


def expanding_h_from_constraint(
    *,
    u: float,
    b: float,
    parameters: ReferenceFlrwParameters,
) -> float:
    '''Return the expanding Friedmann branch, failing on a negative radicand.'''

    validate_parameters(parameters)
    m = parameters.m_planck_over_mu_x
    lam = parameters.lambda_over_mu_x_squared
    radicand = (0.5 * float(u) ** 2 + 1.5 * float(b) ** 2 + m**2 * lam) / (
        3.0 * m**2
    )
    if not np.isfinite(radicand) or radicand < 0.0:
        raise ValueError('the Friedmann initializer has a negative radicand')
    return float(np.sqrt(radicand))


def flrw_rhs(
    state: ReferenceFlrwState, parameters: ReferenceFlrwParameters
) -> ReferenceFlrwState:
    '''Return d(n,h,T,u,b)/d tau for the exact homogeneous equations.'''

    validate_parameters(parameters)
    m = parameters.m_planck_over_mu_x
    return ReferenceFlrwState(
        n=state.h,
        h=-(state.u**2 + state.b**2) / (2.0 * m**2),
        clock=state.u,
        u=-3.0 * state.h * state.u,
        b=-state.h * state.b,
    )


def clock_charge_bar(state: ReferenceFlrwState) -> float:
    '''Return the conserved a^3 u charge.'''

    return float(np.exp(3.0 * state.n) * state.u)


def rod_charge_bar(state: ReferenceFlrwState) -> float:
    '''Return the conserved beta/mu_X=a b.'''

    return float(np.exp(state.n) * state.b)


def reference_jacobian_bar(state: ReferenceFlrwState) -> float:
    '''Return det(d_mu X^A)/mu_X^4=(beta/mu_X)^3 u.'''

    return float(rod_charge_bar(state) ** 3 * state.u)


def curvature_scalar_bar(
    state: ReferenceFlrwState, parameters: ReferenceFlrwParameters
) -> float:
    '''Return R/mu_X^2 from the traced Einstein equation.'''

    m = parameters.m_planck_over_mu_x
    lam = parameters.lambda_over_mu_x_squared
    return float(4.0 * lam - state.u**2 / m**2 + 3.0 * state.b**2 / m**2)


@dataclass(frozen=True)
class ReferenceFlrwAudit:
    friedmann_relative_residual: float
    continuity_residual: float
    constraint_propagation_residual: float
    acceleration_residual: float
    trace_residual: float
    isotropy_residual: float
    reference_jacobian_bar: float
    clock_gradient_norm_bar: float
    curvature_scalar_bar: float
    local_reference_patch_admitted: bool
    compact_global_real_rod_claim_admitted: bool
    bare_internal_kinetic_eigenvalues: tuple[float, ...]
    bare_homogeneous_kinetic_positive: bool
    perturbation_stability_computed: bool
    one_loop_st_identity_computed: bool
    physical_inner_product_constructed: bool
    nonperturbative_m2_passed: bool
    status: str


def audit_reference_flrw_background(
    state: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    ricci_scalar_cutoff_bar: float | None = None,
    tol: float = TOL,
) -> ReferenceFlrwAudit:
    '''Audit all classical equations and the local reference Jacobian at one event.'''

    validate_parameters(parameters)
    values = state.as_array()
    if not np.all(np.isfinite(values)):
        raise ValueError('the background state must be finite')
    if ricci_scalar_cutoff_bar is not None:
        if not np.isfinite(ricci_scalar_cutoff_bar) or ricci_scalar_cutoff_bar <= 0.0:
            raise ValueError('the dimensionless Ricci-scalar cutoff must be positive')

    rhs = flrw_rhs(state, parameters)
    rho = energy_density_bar(state)
    pressure = pressure_bar(state)
    m = parameters.m_planck_over_mu_x
    lam = parameters.lambda_over_mu_x_squared
    constraint = friedmann_residual(state, parameters)
    constraint_scale = 1.0 + abs(3.0 * m**2 * state.h**2) + abs(rho) + abs(m**2 * lam)
    constraint_relative = abs(constraint) / constraint_scale

    rho_prime = state.u * rhs.u + 3.0 * state.b * rhs.b
    continuity = rho_prime + 3.0 * state.h * (rho + pressure)
    propagation = 6.0 * m**2 * state.h * rhs.h - rho_prime
    acceleration = rhs.h + state.h**2
    acceleration_target = lam / 3.0 - state.u**2 / (3.0 * m**2)
    acceleration_residual = acceleration - acceleration_target
    curvature_geometric = 6.0 * (rhs.h + 2.0 * state.h**2)
    curvature_trace = curvature_scalar_bar(state, parameters)
    trace_residual = curvature_geometric - curvature_trace

    stress = orthonormal_stress_bar(state)
    spatial_diagonal = np.diag(stress)[1:]
    off_diagonal = stress - np.diag(np.diag(stress))
    isotropy_residual = float(
        np.linalg.norm(off_diagonal)
        + np.linalg.norm(spatial_diagonal - pressure)
        + abs(stress[0, 0] - rho)
    )

    jacobian = reference_jacobian_bar(state)
    clock_norm = -state.u**2
    reference_admitted = (
        abs(jacobian) > tol
        and clock_norm < -(tol**2)
        and constraint_relative <= tol
        and abs(continuity) <= tol
        and abs(propagation) <= tol
        and abs(acceleration_residual) <= tol
        and abs(trace_residual) <= tol
        and isotropy_residual <= tol
    )
    if ricci_scalar_cutoff_bar is not None:
        reference_admitted = reference_admitted and (
            abs(curvature_trace) <= ricci_scalar_cutoff_bar
        )
    kinetic_eigenvalues = np.linalg.eigvalsh(bare_internal_kinetic_gram())
    bare_kinetic_positive = bool(np.all(kinetic_eigenvalues > tol))
    status = (
        'CLASSICAL_LOCAL_REFERENCE_BACKGROUND_ADMITTED'
        if reference_admitted
        else 'CLASSICAL_BACKGROUND_REJECTED'
    )
    return ReferenceFlrwAudit(
        friedmann_relative_residual=float(constraint_relative),
        continuity_residual=float(continuity),
        constraint_propagation_residual=float(propagation),
        acceleration_residual=float(acceleration_residual),
        trace_residual=float(trace_residual),
        isotropy_residual=isotropy_residual,
        reference_jacobian_bar=jacobian,
        clock_gradient_norm_bar=float(clock_norm),
        curvature_scalar_bar=curvature_trace,
        local_reference_patch_admitted=reference_admitted,
        compact_global_real_rod_claim_admitted=False,
        bare_internal_kinetic_eigenvalues=tuple(
            float(value) for value in kinetic_eigenvalues
        ),
        bare_homogeneous_kinetic_positive=bare_kinetic_positive,
        perturbation_stability_computed=False,
        one_loop_st_identity_computed=False,
        physical_inner_product_constructed=False,
        nonperturbative_m2_passed=False,
        status=status,
    )


def state_from_conserved_charges(
    *,
    n: float,
    clock: float,
    clock_charge: float,
    rod_charge: float,
    parameters: ReferenceFlrwParameters,
    expanding: bool = True,
) -> ReferenceFlrwState:
    '''Construct a constraint-satisfying state from a^3 u and a b.'''

    if not np.all(np.isfinite([n, clock, clock_charge, rod_charge])):
        raise ValueError('charges and state labels must be finite')
    u = float(clock_charge * np.exp(-3.0 * n))
    b = float(rod_charge * np.exp(-n))
    h = expanding_h_from_constraint(u=u, b=b, parameters=parameters)
    if not expanding:
        h = -h
    return ReferenceFlrwState(n=float(n), h=h, clock=float(clock), u=u, b=b)


def clock_dominated_past_time_bound(
    state: ReferenceFlrwState, parameters: ReferenceFlrwParameters
) -> float:
    '''Upper bound on dimensionless expanding proper time to a=0 for Lambda>=0.

    Friedmann gives h >= |u|/(sqrt(6)m), while u=u_0 exp[-3(n-n_0)].
    Hence d tau/dn <= sqrt(6)m exp[3(n-n_0)]/|u_0|.  Integrating from
    n=-infinity to n_0 yields sqrt(6)m/(3|u_0|).
    '''

    validate_parameters(parameters)
    if parameters.lambda_over_mu_x_squared < 0.0:
        raise ValueError('the simple past-time bound requires nonnegative Lambda')
    if state.h <= 0.0:
        raise ValueError('the past-time bound is for the expanding branch')
    if abs(state.u) <= TOL:
        raise ValueError('the bound requires a nonzero clock charge')
    return float(np.sqrt(6.0) * parameters.m_planck_over_mu_x / (3.0 * abs(state.u)))


@dataclass(frozen=True)
class ReferenceFlrwTrajectory:
    tau: tuple[float, ...]
    states: tuple[ReferenceFlrwState, ...]
    max_friedmann_relative_residual: float
    max_clock_charge_drift: float
    max_rod_charge_drift: float
    minimum_abs_reference_jacobian_bar: float
    all_local_reference_patches_admitted: bool
    perturbation_stability_computed: bool
    one_loop_st_identity_computed: bool
    nonperturbative_m2_passed: bool


def integrate_reference_flrw_background(
    initial: ReferenceFlrwState,
    parameters: ReferenceFlrwParameters,
    *,
    tau_final: float,
    steps: int,
    ricci_scalar_cutoff_bar: float | None = None,
) -> ReferenceFlrwTrajectory:
    '''Integrate the full first-order system with fixed-step RK4.'''

    if not np.isfinite(tau_final):
        raise ValueError('tau_final must be finite')
    if steps <= 0:
        raise ValueError('steps must be positive')
    initial_audit = audit_reference_flrw_background(
        initial,
        parameters,
        ricci_scalar_cutoff_bar=ricci_scalar_cutoff_bar,
    )
    if not initial_audit.local_reference_patch_admitted:
        raise ValueError('the initial state does not pass the local reference gate')

    delta = float(tau_final) / steps
    times = [0.0]
    states = [initial]

    def rhs_array(values: np.ndarray) -> np.ndarray:
        state = ReferenceFlrwState.from_array(values)
        return flrw_rhs(state, parameters).as_array()

    values = initial.as_array()
    for index in range(steps):
        k1 = rhs_array(values)
        k2 = rhs_array(values + 0.5 * delta * k1)
        k3 = rhs_array(values + 0.5 * delta * k2)
        k4 = rhs_array(values + delta * k3)
        values = values + delta * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        state = ReferenceFlrwState.from_array(values)
        audit = audit_reference_flrw_background(
            state,
            parameters,
            ricci_scalar_cutoff_bar=ricci_scalar_cutoff_bar,
            tol=2.0e-7,
        )
        if not audit.local_reference_patch_admitted:
            raise ValueError('the trajectory left the admitted local reference region')
        times.append((index + 1) * delta)
        states.append(state)

    constraints = [
        audit_reference_flrw_background(state, parameters).friedmann_relative_residual
        for state in states
    ]
    clock_charges = np.array([clock_charge_bar(state) for state in states])
    rod_charges = np.array([rod_charge_bar(state) for state in states])
    jacobians = np.array([abs(reference_jacobian_bar(state)) for state in states])
    clock_scale = max(1.0, abs(clock_charges[0]))
    rod_scale = max(1.0, abs(rod_charges[0]))
    return ReferenceFlrwTrajectory(
        tau=tuple(float(value) for value in times),
        states=tuple(states),
        max_friedmann_relative_residual=float(max(constraints)),
        max_clock_charge_drift=float(np.max(np.abs(clock_charges - clock_charges[0])) / clock_scale),
        max_rod_charge_drift=float(np.max(np.abs(rod_charges - rod_charges[0])) / rod_scale),
        minimum_abs_reference_jacobian_bar=float(np.min(jacobians)),
        all_local_reference_patches_admitted=True,
        perturbation_stability_computed=False,
        one_loop_st_identity_computed=False,
        nonperturbative_m2_passed=False,
    )
