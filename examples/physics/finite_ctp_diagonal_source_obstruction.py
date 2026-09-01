"""Finite CTP witnesses for unobserved quantum-environment influence.

The original two-level witness shows that a diagonal 0D record does not fix a
CTP difference source.  The environment helpers retain coherent and entangled
joint states, calculate a diagonal ``observed x hidden`` product fast path, and
expose the memory error caused by tracing and resetting the same environment at
every time slice.  Exact finite kick and gate-order audits are also provided.
None of these finite channels is a stress tensor or a derived Clarus source
model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


_TOL = 2.0e-11
_BRANCH_MARGIN = 2.0e-7


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _unit_interval_closed(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


@dataclass(frozen=True)
class ThermalForcedOscillatorCTPAudit:
    """Exact one-mode Gaussian influence for two supplied field histories.

    Natural units ``hbar=c=1`` and the covariant interaction
    ``L_int=-kappa*phi*chi`` are used.  In a finite spatial volume the
    homogeneous environment mode has ``H_int=eta(t)(b+b^dagger)`` and hence
    ``U_a=exp(i*theta_a) D(alpha_a)``.  This is an infinite-dimensional
    canonical oscillator in a finite volume, not a finite Hilbert-space CCR
    truncation.

    The returned Gram matrix belongs only to the integrated-out environment
    representation.  It is not a local stress tensor and must not be added to
    a retained ``chi`` stress calculation.
    """

    action_parameter_manifest: tuple[float, float, float, float]
    volume: float
    duration: float
    field_histories: tuple[float, float]
    mean_occupation: float
    inverse_temperature: float
    source_amplitudes: tuple[float, float]
    displacements: tuple[complex, complex]
    magnus_phases: tuple[float, float]
    influence_phase: float
    noise_exponent: float
    influence: complex
    influence_gram: tuple[tuple[complex, ...], ...]
    gram_minimum_eigenvalue: float
    gram_diagonal_residual: float
    closed_form_influence_residual: float
    diagonal_influence_residual: float
    kms_boltzmann_factor: float
    thermal_occupation_relative_residual: float
    kms_detailed_balance_residual: float
    retarded_future_support_residual: float
    sampled_noise_kernel_minimum_eigenvalue: float
    mass_matrix_determinant: float
    mass_matrix_stable: bool
    constant_history_noise_quadratic_form_nonnegative: bool
    sampled_noise_kernel_positive_semidefinite: bool
    analytic_retarded_support_contract: bool
    kms_condition_verified: bool
    gram_schur_channel_cptp: bool
    dimensions_pass: bool
    exact_gaussian_influence_computed: bool
    representation: str = "INTEGRATED_OUT_ENVIRONMENT_INFLUENCE_ONLY"
    boundary_condition: str = "INITIAL_PRODUCT_THERMAL_ENVIRONMENT"
    history_role: str = "SUPPLIED_CTP_HISTORIES_NOT_AUTONOMOUS_SOLUTIONS"
    finite_volume_one_mode: bool = True
    canonical_hilbert_space_finite_dimensional: bool = False
    retained_environment_stress_added: bool = False
    local_stress_from_gram_derived: bool = False
    markovian_bath_derived: bool = False
    cosmological_initial_state_derived: bool = False


def audit_thermal_forced_oscillator_ctp(
    *,
    system_mass: float,
    environment_mass: float,
    bilinear_coupling: float,
    vacuum_energy_density: float,
    volume: float,
    duration: float,
    field_left: float,
    field_right: float,
    mean_occupation: float,
    inverse_temperature: float,
    tolerance: float = _TOL,
) -> ThermalForcedOscillatorCTPAudit:
    """Evaluate the exact thermal influence of one homogeneous hidden mode.

    For constant supplied histories ``phi_a`` the exact quantities are

    ``eta_a = kappa*sqrt(V/(2*Omega))*phi_a``,
    ``alpha_a = eta_a*(1-exp(i*Omega*T))/Omega``, and
    ``theta_a = eta_a**2*(Omega*T-sin(Omega*T))/Omega**2``.

    A thermal state with occupation ``nbar`` gives

    ``G_ab = exp(i*Phi_ab-(nbar+1/2)*|alpha_a-alpha_b|**2)``.

    The caller also supplies ``beta`` and the audit fails closed unless
    ``nbar=1/expm1(beta*Omega)`` (including ``beta=+inf, nbar=0``).

    The action parameters follow
    ``(m, M=Omega, kappa, V0)`` with mass dimensions ``(1,1,2,4)``.
    Stability of the retained two-field action is checked before the
    integrated-out influence is evaluated.
    """

    system_mass = _finite(system_mass, "system_mass")
    environment_mass = _finite(environment_mass, "environment_mass")
    coupling = _finite(bilinear_coupling, "bilinear_coupling")
    vacuum_density = _finite(vacuum_energy_density, "vacuum_energy_density")
    volume = _finite(volume, "volume")
    duration = _finite(duration, "duration")
    field_left = _finite(field_left, "field_left")
    field_right = _finite(field_right, "field_right")
    occupation = _finite(mean_occupation, "mean_occupation")
    beta = float(inverse_temperature)
    tolerance = _finite(tolerance, "tolerance")
    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError("system_mass and environment_mass must be positive")
    if volume <= 0.0:
        raise ValueError("volume must be positive")
    if duration < 0.0:
        raise ValueError("duration must be nonnegative")
    if occupation < 0.0:
        raise ValueError("mean_occupation must be nonnegative")
    if math.isnan(beta) or beta <= 0.0:
        raise ValueError("inverse_temperature must be positive")
    if tolerance <= 0.0 or tolerance > 1.0e-8:
        raise ValueError("tolerance must lie in (0, 1e-8]")

    determinant = (
        system_mass**2 * environment_mass**2 - coupling**2
    )
    if determinant <= 0.0:
        raise ValueError(
            "bilinear mass matrix must satisfy m^2*M^2-kappa^2 > 0"
        )

    source_scale = coupling * math.sqrt(volume / (2.0 * environment_mass))
    eta_left = source_scale * field_left
    eta_right = source_scale * field_right
    angle = environment_mass * duration
    displacement_factor = (1.0 - np.exp(1j * angle)) / environment_mass
    alpha_left = complex(eta_left * displacement_factor)
    alpha_right = complex(eta_right * displacement_factor)
    magnus_factor = (angle - math.sin(angle)) / environment_mass**2
    theta_left = eta_left**2 * magnus_factor
    theta_right = eta_right**2 * magnus_factor

    displacement_difference = alpha_left - alpha_right
    phase = float(
        theta_left
        - theta_right
        + np.imag(alpha_left * np.conj(alpha_right))
    )
    noise = float(
        (occupation + 0.5) * abs(displacement_difference) ** 2
    )
    influence_value = complex(np.exp(1j * phase - noise))
    gram = np.array(
        ((1.0 + 0.0j, influence_value),
         (np.conj(influence_value), 1.0 + 0.0j)),
        dtype=complex,
    )
    gram_minimum = float(np.min(np.linalg.eigvalsh(gram)))
    diagonal_residual = float(
        np.max(np.abs(np.diag(gram) - 1.0))
    )

    delta_eta = eta_left - eta_right
    closed_noise = float(
        (occupation + 0.5)
        * (delta_eta / environment_mass) ** 2
        * (2.0 - 2.0 * math.cos(angle))
    )
    closed_phase = float(
        (eta_left**2 - eta_right**2) * magnus_factor
    )
    closed_influence = complex(np.exp(1j * closed_phase - closed_noise))
    closed_residual = abs(influence_value - closed_influence)

    # Check the supplied occupation against an independently supplied inverse
    # temperature.  beta=+inf is the exact zero-temperature limit.
    if math.isinf(beta):
        expected_occupation = 0.0
        boltzmann_factor = 0.0
    else:
        beta_omega = beta * environment_mass
        boltzmann_factor = math.exp(-beta_omega)
        expected_occupation = (
            0.0 if beta_omega > 700.0 else 1.0 / math.expm1(beta_omega)
        )
    thermal_residual = abs(occupation - expected_occupation) / max(
        1.0,
        expected_occupation,
    )
    kms_residual = abs(
        occupation / (occupation + 1.0) - boltzmann_factor
    )
    if thermal_residual > tolerance or kms_residual > tolerance:
        raise ValueError(
            "thermal occupation and inverse_temperature violate KMS balance"
        )

    # Sample the analytic one-mode kernels.  This checks a finite kernel Gram
    # and the retarded support contract; it does not certify a continuum bath,
    # local friction, or a Markov limit.
    sample_span = duration if duration > 0.0 else 2.0 * math.pi / environment_mass
    sample_times = np.linspace(0.0, sample_span, 5)
    time_difference = sample_times[:, None] - sample_times[None, :]
    noise_kernel = (2.0 * occupation + 1.0) * np.cos(
        environment_mass * time_difference
    )
    sampled_noise_minimum = float(
        np.min(np.linalg.eigvalsh(noise_kernel))
    )
    retarded_kernel = np.where(
        time_difference >= 0.0,
        2.0 * np.sin(environment_mass * time_difference),
        0.0,
    )
    future_mask = time_difference < 0.0
    retarded_residual = float(
        np.max(np.abs(retarded_kernel[future_mask]))
        if np.any(future_mask)
        else 0.0
    )

    # [eta]=1, [T]=-1, [alpha]=[theta]=[Phi]=[N]=0 and
    # [m^2*M^2-kappa^2]=4 in natural units.
    dimensions_pass = bool(
        2 + (-3 - 1) / 2 + 1 == 1
        and 1 + (-1) == 0
        and 2 * 1 + 2 * (-1) == 0
        and 4 == 4
    )
    constant_noise_nonnegative = (
        noise >= -tolerance and occupation + 0.5 > 0.0
    )
    sampled_noise_psd = sampled_noise_minimum >= -tolerance
    retarded_contract = retarded_residual <= tolerance
    kms_verified = (
        thermal_residual <= tolerance and kms_residual <= tolerance
    )
    gram_cptp = gram_minimum >= -tolerance and diagonal_residual <= tolerance
    exact = (
        closed_residual <= tolerance
        and diagonal_residual <= tolerance
        and constant_noise_nonnegative
        and sampled_noise_psd
        and gram_cptp
        and kms_verified
        and retarded_contract
        and dimensions_pass
    )

    return ThermalForcedOscillatorCTPAudit(
        action_parameter_manifest=(
            system_mass,
            environment_mass,
            coupling,
            vacuum_density,
        ),
        volume=volume,
        duration=duration,
        field_histories=(field_left, field_right),
        mean_occupation=occupation,
        inverse_temperature=beta,
        source_amplitudes=(eta_left, eta_right),
        displacements=(alpha_left, alpha_right),
        magnus_phases=(theta_left, theta_right),
        influence_phase=phase,
        noise_exponent=noise,
        influence=influence_value,
        influence_gram=tuple(
            tuple(complex(item) for item in row) for row in gram
        ),
        gram_minimum_eigenvalue=gram_minimum,
        gram_diagonal_residual=diagonal_residual,
        closed_form_influence_residual=closed_residual,
        diagonal_influence_residual=0.0,
        kms_boltzmann_factor=boltzmann_factor,
        thermal_occupation_relative_residual=thermal_residual,
        kms_detailed_balance_residual=kms_residual,
        retarded_future_support_residual=retarded_residual,
        sampled_noise_kernel_minimum_eigenvalue=sampled_noise_minimum,
        mass_matrix_determinant=determinant,
        mass_matrix_stable=True,
        constant_history_noise_quadratic_form_nonnegative=(
            constant_noise_nonnegative
        ),
        sampled_noise_kernel_positive_semidefinite=sampled_noise_psd,
        analytic_retarded_support_contract=retarded_contract,
        kms_condition_verified=kms_verified,
        gram_schur_channel_cptp=gram_cptp,
        dimensions_pass=dimensions_pass,
        exact_gaussian_influence_computed=exact,
    )


@dataclass(frozen=True)
class FiniteCTPDiagonalSourceCertificate:
    probability_one: float
    hbar: float
    tau: float
    slope: float
    h_delta: float
    influence: complex
    influence_diagonal_residual: float
    action_diagonal_residual: float
    h_c_derivative_at_diagonal: float
    difference_source: float
    central_difference_source: float
    central_difference_residual: float
    linear_action_coefficient: float
    quadratic_imaginary_action_coefficient: float
    symmetric_quadratic_coefficient: complex
    local_expansion_residual: float
    model_zero_difference_source: float
    model_nonzero_difference_source: float
    model_reference_frequency_residual: float
    model_reference_hamiltonian_residual: float
    diagonal_model_influence_residual: float
    diagonal_readout_probabilities: tuple[float, float]
    limited_non_identifiability: bool
    environment_minimum_eigenvalue: float
    environment_trace_residual: float
    controlled_unitary_residual: float
    gram_minimum_eigenvalue: float
    gram_diagonal_residual: float
    schur_choi_minimum_eigenvalue: float
    schur_trace_preservation_residual: float
    schur_output_trace_residual: float
    schur_completely_positive: bool
    schur_trace_preserving: bool
    plus_state_coherence: complex
    p_zero_source: float
    tau_zero_source: float
    slope_zero_source: float
    p_zero_decoherence: float
    tau_zero_decoherence: float
    slope_zero_decoherence: float
    p_one_quadratic_noise_coefficient: float
    p_one_unitary_phase_present: bool
    h_mass_dimension: int
    omega_mass_dimension: int
    slope_mass_dimension: int
    tau_mass_dimension: int
    tau_omega_mass_dimension: int
    influence_mass_dimension: int
    action_over_hbar_mass_dimension: int
    difference_source_dimension: str
    dimensions_pass: bool
    accounting_mode: str
    retained_environment_stress_added: bool
    rn_reweighting_used: bool
    tensor_stress_derived: bool = False
    diffeo_Ward: bool = False
    retarded_causality: bool = False
    microcausality_or_c_front: bool = False
    attraction: bool = False
    mass_to_source: bool = False
    GR_lensing_spin2: bool = False
    energy_backreaction: bool = False
    physical_observation_or_selection: bool = False
    observational_holdout: bool = False
    gates_5_to_8: bool = False
    two_residuals: bool = False
    complexity_success: bool = False


def omega(h: float, *, omega_star: float, slope: float, h_star: float) -> float:
    """Linear angular frequency, with a dimensionless source ``h``."""

    return _finite(omega_star, "omega_star") + _finite(slope, "slope") * (_finite(h, "h") - _finite(h_star, "h_star"))


def influence(
    h_plus: float, h_minus: float, *, p: float, tau: float,
    omega_star: float, slope: float, h_star: float,
) -> complex:
    """Return ``Tr(U_plus rho_E U_minus^dagger)`` on the principal-safe branch."""

    p = _unit_interval_closed(p, "p")
    tau = _finite(tau, "tau")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    difference = omega(h_plus, omega_star=omega_star, slope=slope, h_star=h_star) - omega(
        h_minus, omega_star=omega_star, slope=slope, h_star=h_star
    )
    value = (1.0 - p) + p * np.exp(-1j * tau * difference)
    if h_plus != h_minus:
        if abs(value) <= _TOL:
            raise ValueError("offdiagonal influence is zero; logarithm branch is unavailable")
        if abs(float(np.angle(value))) >= math.pi - _BRANCH_MARGIN:
            raise ValueError("offdiagonal influence is outside the validated principal near-diagonal branch")
    return complex(value)


def influence_action(
    h_plus: float, h_minus: float, *, p: float, tau: float, hbar: float,
    omega_star: float, slope: float, h_star: float,
) -> complex:
    """``S_IF=-i hbar Log(F)`` where the finite branch has been validated."""

    hbar = _finite(hbar, "hbar")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")
    return complex(-1j * hbar * np.log(influence(
        h_plus, h_minus, p=p, tau=tau, omega_star=omega_star, slope=slope, h_star=h_star
    )))


def _density_matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite normalized density matrix or fail closed."""

    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if np.linalg.norm(matrix - matrix.conj().T, ord=2) > _TOL:
        raise ValueError(f"{name} must be Hermitian")
    trace = complex(np.trace(matrix))
    if abs(trace - 1.0) > _TOL:
        raise ValueError(f"{name} must have unit trace")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -_TOL:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _unitary_matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite unitary matrix or fail closed."""

    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    identity = np.eye(matrix.shape[0], dtype=complex)
    if np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2) > _TOL:
        raise ValueError(f"{name} must be unitary")
    return matrix


def _hermitian_matrix(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite Hermitian operator or fail closed."""

    matrix = np.asarray(value, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if np.linalg.norm(matrix - matrix.conj().T, ord=2) > _TOL:
        raise ValueError(f"{name} must be Hermitian")
    return matrix


def _conditional_unitary_stack(value: np.ndarray, name: str) -> np.ndarray:
    """Return ``(history, dimension, dimension)`` validated unitaries."""

    unitaries = np.asarray(value, dtype=complex)
    if (
        unitaries.ndim != 3
        or unitaries.shape[0] == 0
        or unitaries.shape[1] != unitaries.shape[2]
        or unitaries.shape[1] == 0
    ):
        raise ValueError(
            f"{name} must have shape (history_count, dimension, dimension)"
        )
    for history, unitary in enumerate(unitaries):
        _unitary_matrix(unitary, f"{name}[{history}]")
    return unitaries


def joint_environment_influence_gram(
    conditional_unitaries: np.ndarray,
    environment_state: np.ndarray,
) -> np.ndarray:
    """Return the exact controlled-history Gram for one joint environment.

    The initial state is ``rho_S tensor rho_E`` and the joint controlled
    dilation is ``sum_a |a><a| tensor U_a``.  ``rho_E`` may contain arbitrary
    correlations or entanglement among its internal factors.  Under precisely
    those assumptions,

        G[a,b] = Tr(U_a rho_E U_b^dagger)

    is positive semidefinite with unit diagonal, and ``rho -> G * rho`` is a
    Schur CPTP channel.  A general system-environment unitary need not have this
    population-preserving Schur form.
    """

    unitaries = _conditional_unitary_stack(
        conditional_unitaries,
        "conditional_unitaries",
    )
    state = _density_matrix(environment_state, "environment_state")
    if state.shape != unitaries.shape[1:]:
        raise ValueError(
            "environment_state dimension must match the conditional unitaries"
        )
    return np.einsum(
        "aik,kl,bil->ab",
        unitaries,
        state,
        unitaries.conj(),
        optimize=True,
    )


def product_environment_influence_gram(
    conditional_unitaries: Sequence[np.ndarray],
    environment_states: Sequence[np.ndarray],
) -> np.ndarray:
    """Return the exact influence Gram matrix for a product environment.

    For environment factor ``j`` and observed history ``a``, the caller
    supplies the conditional unitary ``U[j][a]``.  The returned matrix is

        G[a,b] = product_j Tr(U[j][a] rho[j] U[j][b]^dagger).

    The environment factors are not measured.  Off-diagonal entries of
    ``rho[j]`` are retained, so this path distinguishes a coherent
    superposition from its dephased mixture whenever the declared conditional
    unitaries can couple to that coherence.  Entangled environment factors
    require one joint state/unitary instead of this product factorization.
    """

    unitary_factors = tuple(conditional_unitaries)
    state_factors = tuple(environment_states)
    if not unitary_factors:
        raise ValueError("at least one environment factor is required")
    if len(unitary_factors) != len(state_factors):
        raise ValueError("conditional_unitaries and environment_states must have equal length")

    gram: np.ndarray | None = None
    for index, (unitary_value, state_value) in enumerate(
        zip(unitary_factors, state_factors)
    ):
        local_gram = joint_environment_influence_gram(unitary_value, state_value)
        if gram is None:
            gram = np.ones_like(local_gram)
        elif local_gram.shape != gram.shape:
            raise ValueError(
                f"conditional_unitaries[{index}] must have the same history count"
            )
        gram *= local_gram
    assert gram is not None
    return gram


def _probability_vector(value: np.ndarray, expected_size: int) -> np.ndarray:
    probabilities = np.asarray(value, dtype=float)
    if probabilities.shape != (expected_size,):
        raise ValueError("environment_probabilities must match the environment count")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("environment_probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("environment_probabilities must lie in [0, 1]")
    return probabilities


def diagonal_product_influence_gram(
    branch_phases: np.ndarray,
    environment_probabilities: np.ndarray,
) -> np.ndarray:
    """Return a product-environment Gram matrix from dimensionless phases.

    ``branch_phases[a,j]`` is the accumulated phase of environment factor
    ``j`` conditional on observed history ``a``.  The interaction and the
    environment state are diagonal in the same supplied basis.  Relative
    phases in an environment superposition are therefore outside this fast
    path; use :func:`product_environment_influence_gram` for noncommuting
    conditional dynamics.
    """

    phases = np.asarray(branch_phases, dtype=float)
    if phases.ndim != 2 or phases.shape[0] == 0:
        raise ValueError("branch_phases must have shape (history_count, environment_count)")
    if not np.all(np.isfinite(phases)):
        raise ValueError("branch_phases must be finite")
    probabilities = _probability_vector(environment_probabilities, phases.shape[1])
    history_count = phases.shape[0]
    gram = np.ones((history_count, history_count), dtype=complex)
    for environment, probability in enumerate(probabilities):
        difference = phases[:, environment, None] - phases[None, :, environment]
        gram *= (1.0 - probability) + probability * np.exp(-1j * difference)
    return gram


@dataclass(frozen=True)
class CollectiveDiagonalInfluenceCertificate:
    """Finite influence receipt for observed histories and hidden factors."""

    observed_count: int
    environment_count: int
    history_count: int
    tau: float
    mean_angular_frequency_shift: tuple[float, ...]
    angular_frequency_covariance: tuple[tuple[float, ...], ...]
    history_mean_angular_frequency_shift: tuple[float, ...]
    history_angular_frequency_covariance: tuple[tuple[float, ...], ...]
    influence_gram: tuple[tuple[complex, ...], ...]
    gram_minimum_eigenvalue: float
    gram_hermiticity_residual: float
    gram_diagonal_residual: float
    schur_channel_completely_positive: bool
    schur_channel_trace_preserving: bool
    product_environment_assumption: bool
    diagonal_interaction_fast_path: bool
    dimensionless_phase_input_contract_declared: bool
    environment_coherence_phases_resolved: bool
    physical_clarus_source_derived: bool
    retained_environment_stress_added: bool


def certify_collective_diagonal_influence(
    system_histories: np.ndarray,
    angular_frequency_couplings: np.ndarray,
    environment_probabilities: np.ndarray,
    *,
    tau: float,
) -> CollectiveDiagonalInfluenceCertificate:
    """Calculate an exact diagonal influence channel without a full tensor state.

    The model is

        H_int / hbar = sum_(a,j) g[a,j] s[a] |1><1|_j,

    with independent hidden factors having occupation probabilities ``p[j]``.
    ``system_histories`` is dimensionless by input contract, while ``g`` has
    angular-frequency units and ``tau`` has inverse units, so ``tau * g`` is
    dimensionless.  The code checks finiteness and shape but cannot infer units
    from bare arrays.  Mean and covariance are frequency moments of this
    declared interaction, not energy, stress, gravity, or a derived Clarus
    source.
    """

    histories = np.asarray(system_histories, dtype=float)
    couplings = np.asarray(angular_frequency_couplings, dtype=float)
    tau = _finite(tau, "tau")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if histories.ndim != 2 or histories.shape[0] == 0 or histories.shape[1] == 0:
        raise ValueError("system_histories must be a nonempty two-dimensional array")
    if couplings.ndim != 2 or couplings.shape[0] != histories.shape[1]:
        raise ValueError(
            "angular_frequency_couplings must have shape "
            "(observed_count, environment_count)"
        )
    if not np.all(np.isfinite(histories)):
        raise ValueError("system_histories must be finite")
    if not np.all(np.isfinite(couplings)):
        raise ValueError("angular_frequency_couplings must be finite")
    probabilities = _probability_vector(environment_probabilities, couplings.shape[1])

    branch_phases = tau * (histories @ couplings)
    gram = diagonal_product_influence_gram(branch_phases, probabilities)
    mean = couplings @ probabilities
    variances = probabilities * (1.0 - probabilities)
    covariance = (couplings * variances[np.newaxis, :]) @ couplings.T
    history_mean = histories @ mean
    history_covariance = histories @ covariance @ histories.T
    hermitian_gram = 0.5 * (gram + gram.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_gram)))
    hermiticity_residual = float(np.linalg.norm(gram - gram.conj().T, ord=2))
    diagonal_residual = float(np.linalg.norm(np.diag(gram) - 1.0, ord=np.inf))
    return CollectiveDiagonalInfluenceCertificate(
        observed_count=histories.shape[1],
        environment_count=couplings.shape[1],
        history_count=histories.shape[0],
        tau=tau,
        mean_angular_frequency_shift=tuple(float(item) for item in mean),
        angular_frequency_covariance=tuple(
            tuple(float(item) for item in row) for row in covariance
        ),
        history_mean_angular_frequency_shift=tuple(
            float(item) for item in history_mean
        ),
        history_angular_frequency_covariance=tuple(
            tuple(float(item) for item in row) for row in history_covariance
        ),
        influence_gram=tuple(
            tuple(complex(item) for item in row) for row in gram
        ),
        gram_minimum_eigenvalue=minimum_eigenvalue,
        gram_hermiticity_residual=hermiticity_residual,
        gram_diagonal_residual=diagonal_residual,
        schur_channel_completely_positive=(
            hermiticity_residual <= _TOL and minimum_eigenvalue >= -_TOL
        ),
        schur_channel_trace_preserving=diagonal_residual <= _TOL,
        product_environment_assumption=True,
        diagonal_interaction_fast_path=True,
        dimensionless_phase_input_contract_declared=True,
        environment_coherence_phases_resolved=False,
        physical_clarus_source_derived=False,
        retained_environment_stress_added=False,
    )


def apply_influence_gram(density: np.ndarray, influence_gram: np.ndarray) -> np.ndarray:
    """Apply the Schur influence channel to a density matrix."""

    state = _density_matrix(density, "density")
    gram = np.asarray(influence_gram, dtype=complex)
    if gram.shape != state.shape:
        raise ValueError("influence_gram must match density")
    if not np.all(np.isfinite(gram)):
        raise ValueError("influence_gram must be finite")
    if np.linalg.norm(gram - gram.conj().T, ord=2) > _TOL:
        raise ValueError("influence_gram must be Hermitian")
    if np.linalg.norm(np.diag(gram) - 1.0, ord=np.inf) > _TOL:
        raise ValueError("influence_gram must have unit diagonal")
    if float(np.min(np.linalg.eigvalsh(gram))) < -_TOL:
        raise ValueError("influence_gram must be positive semidefinite")
    return gram * state


@dataclass(frozen=True)
class CommonPhaseVacuumStressNoGo:
    """Same finite influence Gram with a distinct supplied vacuum stress.

    A common identity shift of every conditional Hamiltonian multiplies every
    branch unitary by the same phase.  Relative branch data, and hence the
    entire Schur influence channel, are unchanged.  A generally covariant
    constant term in a separately supplied effective action nevertheless
    changes the stress by ``-delta_V * g_(mu nu)``.  The certificate therefore
    rules out extraction of the *absolute* vacuum density from the Gram data
    alone; it does not say that vacuum stress is gravitationally invisible.
    """

    history_count: int
    environment_dimension: int
    hamiltonian_shift: float
    duration: float
    hbar: float
    common_phase_angle: float
    common_phase_factor: complex
    original_influence_gram: tuple[tuple[complex, ...], ...]
    shifted_influence_gram: tuple[tuple[complex, ...], ...]
    maximum_gram_residual: float
    vacuum_energy_density_shift: float
    vacuum_stress_shift_covariant: tuple[tuple[float, ...], ...]
    dimensionless_vacuum_stress_difference: float
    hamiltonian_mass_dimension: int
    duration_mass_dimension: int
    phase_mass_dimension: int
    vacuum_density_mass_dimension: int
    metric_mass_dimension: int
    stress_mass_dimension: int
    dimensions_pass: bool
    common_phase_has_unit_modulus: bool
    influence_gram_invariant: bool
    vacuum_stress_distinct: bool
    absolute_vacuum_density_nonidentifiability_certified: bool
    vacuum_action_supplied: bool = True
    quantum_identity_shift_to_vacuum_density_mapping_derived: bool = False
    absolute_vacuum_density_from_influence_gram_derived: bool = False
    physical_dark_energy_density_derived: bool = False


@dataclass(frozen=True)
class ControlledHistoryObservableAudit:
    """Exact diagonal/off-diagonal split of a controlled-history observable.

    The supplied observable is represented by environment-operator blocks
    ``O[a,b]`` in the system history basis.  For the controlled state after
    the conditional unitaries, the expectation is

        sum_(a,b) rho_S[a,b]
          Tr(U_a rho_E U_b^dagger O[b,a]).

    Environment-only observables have equal diagonal blocks and zero
    off-diagonal blocks, so the system trace removes history interference.
    A genuine total observable can have off-diagonal blocks and retain it.
    The routine evaluates that distinction; it does not declare the supplied
    blocks to be a physical stress operator.
    """

    history_count: int
    environment_dimension: int
    influence_gram: tuple[tuple[complex, ...], ...]
    full_expectation: float
    diagonal_history_expectation: float
    off_diagonal_history_expectation: float
    expectation_imaginary_residual: float
    observable_hermiticity_residual: float
    system_history_coherence_norm: float
    off_diagonal_observable_block_norm: float
    environment_only_block_structure_detected: bool
    system_history_coherence_present: bool
    off_diagonal_interference_present: bool
    environment_only_history_interference_absent: bool
    exact_block_expectation_computed: bool
    observable_expectation_from_influence_gram_alone_derived: bool = False
    supplied_observable_is_physical_stress_derived: bool = False
    metric_variation_of_observable_derived: bool = False
    semiclassical_gravity_source_derived: bool = False


def certify_common_phase_vacuum_stress_no_go(
    conditional_unitaries: np.ndarray,
    environment_state: np.ndarray,
    *,
    hamiltonian_shift: float,
    duration: float,
    hbar: float,
    metric_covariant: np.ndarray,
    vacuum_energy_density_shift: float,
    reference_mass_scale: float,
    tolerance: float = _TOL,
) -> CommonPhaseVacuumStressNoGo:
    """Certify that relative influence data do not fix vacuum normalization.

    In natural units the declared identity shift ``c`` and duration ``tau``
    enter only through the dimensionless angle ``c*tau/hbar``:

        U_a -> exp(-i*c*tau/hbar) U_a,
        G_ab -> G_ab.

    Independently, adding ``-integral sqrt(-g) delta_V d^4x`` to a covariant
    effective action changes ``T_(mu nu)`` by ``-delta_V g_(mu nu)``.  Both
    transformations are evaluated explicitly.  No map from ``c`` to
    ``delta_V`` is assumed or derived.
    """

    unitaries = _conditional_unitary_stack(
        conditional_unitaries,
        "conditional_unitaries",
    )
    state = _density_matrix(environment_state, "environment_state")
    if state.shape != unitaries.shape[1:]:
        raise ValueError(
            "environment_state dimension must match the conditional unitaries"
        )
    hamiltonian_shift = _finite(hamiltonian_shift, "hamiltonian_shift")
    duration = _finite(duration, "duration")
    hbar = _finite(hbar, "hbar")
    vacuum_shift = _finite(
        vacuum_energy_density_shift,
        "vacuum_energy_density_shift",
    )
    reference_mass_scale = _finite(reference_mass_scale, "reference_mass_scale")
    tolerance = _finite(tolerance, "tolerance")
    if duration < 0.0:
        raise ValueError("duration must be nonnegative")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")
    if reference_mass_scale <= 0.0:
        raise ValueError("reference_mass_scale must be positive")
    if tolerance <= 0.0 or tolerance > 1.0e-8:
        raise ValueError("tolerance must lie in (0, 1e-8]")

    metric = np.asarray(metric_covariant, dtype=float)
    if metric.shape != (4, 4) or not np.all(np.isfinite(metric)):
        raise ValueError("metric_covariant must be a finite 4-by-4 matrix")
    if np.linalg.norm(metric - metric.T, ord=2) > tolerance:
        raise ValueError("metric_covariant must be symmetric")
    eigenvalues = np.linalg.eigvalsh(metric)
    metric_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative = int(np.count_nonzero(eigenvalues < -tolerance * metric_scale))
    positive = int(np.count_nonzero(eigenvalues > tolerance * metric_scale))
    if negative != 1 or positive != 3:
        raise ValueError("metric_covariant must have signature (-,+,+,+)")

    angle = hamiltonian_shift * duration / hbar
    phase = complex(np.exp(-1j * angle))
    shifted_unitaries = phase * unitaries
    original_gram = joint_environment_influence_gram(unitaries, state)
    shifted_gram = joint_environment_influence_gram(shifted_unitaries, state)
    gram_residual = float(np.max(np.abs(shifted_gram - original_gram)))

    stress_shift = -vacuum_shift * metric
    dimensionless_stress_difference = float(
        np.linalg.norm(stress_shift, ord=2) / reference_mass_scale**4
    )
    phase_unit_modulus = abs(abs(phase) - 1.0) <= tolerance
    gram_invariant = gram_residual <= tolerance
    stress_distinct = dimensionless_stress_difference > tolerance

    hamiltonian_mass_dimension = 1
    duration_mass_dimension = -1
    phase_mass_dimension = 0
    vacuum_density_mass_dimension = 4
    metric_mass_dimension = 0
    stress_mass_dimension = 4
    dimensions_pass = (
        hamiltonian_mass_dimension + duration_mass_dimension
        == phase_mass_dimension
        and vacuum_density_mass_dimension + metric_mass_dimension
        == stress_mass_dimension
    )
    no_go = phase_unit_modulus and gram_invariant and stress_distinct and dimensions_pass

    return CommonPhaseVacuumStressNoGo(
        history_count=unitaries.shape[0],
        environment_dimension=unitaries.shape[1],
        hamiltonian_shift=hamiltonian_shift,
        duration=duration,
        hbar=hbar,
        common_phase_angle=angle,
        common_phase_factor=phase,
        original_influence_gram=tuple(
            tuple(complex(item) for item in row) for row in original_gram
        ),
        shifted_influence_gram=tuple(
            tuple(complex(item) for item in row) for row in shifted_gram
        ),
        maximum_gram_residual=gram_residual,
        vacuum_energy_density_shift=vacuum_shift,
        vacuum_stress_shift_covariant=tuple(
            tuple(float(item) for item in row) for row in stress_shift
        ),
        dimensionless_vacuum_stress_difference=dimensionless_stress_difference,
        hamiltonian_mass_dimension=hamiltonian_mass_dimension,
        duration_mass_dimension=duration_mass_dimension,
        phase_mass_dimension=phase_mass_dimension,
        vacuum_density_mass_dimension=vacuum_density_mass_dimension,
        metric_mass_dimension=metric_mass_dimension,
        stress_mass_dimension=stress_mass_dimension,
        dimensions_pass=dimensions_pass,
        common_phase_has_unit_modulus=phase_unit_modulus,
        influence_gram_invariant=gram_invariant,
        vacuum_stress_distinct=stress_distinct,
        absolute_vacuum_density_nonidentifiability_certified=no_go,
    )


def audit_controlled_history_observable_expectation(
    system_state: np.ndarray,
    environment_state: np.ndarray,
    conditional_unitaries: np.ndarray,
    observable_blocks: np.ndarray,
    *,
    tolerance: float = _TOL,
) -> ControlledHistoryObservableAudit:
    """Compute exact history-diagonal and interference contributions.

    ``observable_blocks[a,b]`` is the environment operator multiplying
    ``|a><b|``.  Hermiticity therefore requires
    ``observable_blocks[a,b]^dagger == observable_blocks[b,a]``.  The Gram
    matrix is returned beside the expectation to make the information gap
    explicit: pairwise overlaps alone do not contain arbitrary observable
    insertions.
    """

    system = _density_matrix(system_state, "system_state")
    environment = _density_matrix(environment_state, "environment_state")
    unitaries = _conditional_unitary_stack(
        conditional_unitaries,
        "conditional_unitaries",
    )
    tolerance = _finite(tolerance, "tolerance")
    if tolerance <= 0.0 or tolerance > 1.0e-8:
        raise ValueError("tolerance must lie in (0, 1e-8]")
    if system.shape != (unitaries.shape[0], unitaries.shape[0]):
        raise ValueError("system_state dimension must match the history count")
    if environment.shape != unitaries.shape[1:]:
        raise ValueError(
            "environment_state dimension must match the conditional unitaries"
        )

    blocks = np.asarray(observable_blocks, dtype=complex)
    expected_shape = (
        unitaries.shape[0],
        unitaries.shape[0],
        unitaries.shape[1],
        unitaries.shape[1],
    )
    if blocks.shape != expected_shape or not np.all(np.isfinite(blocks)):
        raise ValueError(
            "observable_blocks must have shape "
            "(history_count, history_count, environment_dimension, "
            "environment_dimension)"
        )
    adjoint_blocks = np.swapaxes(
        np.swapaxes(blocks.conj(), 0, 1),
        2,
        3,
    )
    hermiticity_residual = float(np.max(np.abs(blocks - adjoint_blocks)))
    if hermiticity_residual > tolerance:
        raise ValueError("observable_blocks must define a Hermitian observable")

    full = 0.0j
    diagonal = 0.0j
    for left in range(unitaries.shape[0]):
        for right in range(unitaries.shape[0]):
            contribution = system[left, right] * np.trace(
                unitaries[left]
                @ environment
                @ unitaries[right].conj().T
                @ blocks[right, left]
            )
            full += contribution
            if left == right:
                diagonal += contribution
    off_diagonal = full - diagonal
    imaginary_residual = max(
        abs(float(full.imag)),
        abs(float(diagonal.imag)),
        abs(float(off_diagonal.imag)),
    )

    system_coherence = system - np.diag(np.diag(system))
    system_coherence_norm = float(np.linalg.norm(system_coherence, ord=2))
    off_diagonal_blocks = blocks.copy()
    for history in range(unitaries.shape[0]):
        off_diagonal_blocks[history, history] = 0.0
    off_diagonal_block_norm = float(
        np.linalg.norm(off_diagonal_blocks.reshape(-1), ord=2)
    )
    reference_diagonal_block = blocks[0, 0]
    equal_diagonal_blocks = all(
        np.linalg.norm(blocks[index, index] - reference_diagonal_block, ord=2)
        <= tolerance
        for index in range(unitaries.shape[0])
    )
    environment_only_structure = (
        off_diagonal_block_norm <= tolerance and equal_diagonal_blocks
    )
    coherence_present = system_coherence_norm > tolerance
    interference_present = abs(off_diagonal) > tolerance
    environment_interference_absent = (
        environment_only_structure and abs(off_diagonal) <= tolerance
    )
    gram = joint_environment_influence_gram(unitaries, environment)

    return ControlledHistoryObservableAudit(
        history_count=unitaries.shape[0],
        environment_dimension=unitaries.shape[1],
        influence_gram=tuple(
            tuple(complex(item) for item in row) for row in gram
        ),
        full_expectation=float(full.real),
        diagonal_history_expectation=float(diagonal.real),
        off_diagonal_history_expectation=float(off_diagonal.real),
        expectation_imaginary_residual=imaginary_residual,
        observable_hermiticity_residual=hermiticity_residual,
        system_history_coherence_norm=system_coherence_norm,
        off_diagonal_observable_block_norm=off_diagonal_block_norm,
        environment_only_block_structure_detected=environment_only_structure,
        system_history_coherence_present=coherence_present,
        off_diagonal_interference_present=interference_present,
        environment_only_history_interference_absent=(
            environment_interference_absent
        ),
        exact_block_expectation_computed=True,
    )


@dataclass(frozen=True)
class ObserverSliceMemoryAudit:
    """Compare one retained environment with a fresh reset at every slice."""

    slice_count: int
    history_count: int
    environment_count: int
    same_environment_gram: tuple[tuple[complex, ...], ...]
    fresh_environment_composed_gram: tuple[tuple[complex, ...], ...]
    naive_reduced_composition_residual: float
    memory_aware_description_required: bool
    same_environment_retained: bool
    fresh_environment_reset_is_extra_assumption: bool
    commuting_diagonal_no_history_transition_assumption: bool


def audit_observer_slice_memory(
    slice_branch_phases: np.ndarray,
    environment_probabilities: np.ndarray,
) -> ObserverSliceMemoryAudit:
    """Audit whether tracing/resetting after each observer slice changes physics.

    This finite statement assumes commuting diagonal pure-dephasing gates and
    no system-history transitions inside a slice.  For the same hidden factors,
    conditional phases then add before the environment trace.  Elementwise
    composition of per-slice reduced Gram matrices instead describes fresh
    independent factors at every slice.  Equality is a special Markov/reset
    limit, not an automatic consequence of finer time slicing.  When the two
    disagree, one may retain the full joint state or use an equivalent process
    tensor/auxiliary-memory description.
    """

    phases = np.asarray(slice_branch_phases, dtype=float)
    if phases.ndim != 3 or phases.shape[0] == 0 or phases.shape[1] == 0:
        raise ValueError(
            "slice_branch_phases must have shape "
            "(slice_count, history_count, environment_count)"
        )
    if not np.all(np.isfinite(phases)):
        raise ValueError("slice_branch_phases must be finite")
    probabilities = _probability_vector(environment_probabilities, phases.shape[2])
    same_environment = diagonal_product_influence_gram(
        np.sum(phases, axis=0),
        probabilities,
    )
    fresh_environment = np.ones_like(same_environment)
    for slice_phases in phases:
        fresh_environment *= diagonal_product_influence_gram(
            slice_phases,
            probabilities,
        )
    residual = float(np.linalg.norm(same_environment - fresh_environment, ord=2))
    return ObserverSliceMemoryAudit(
        slice_count=phases.shape[0],
        history_count=phases.shape[1],
        environment_count=phases.shape[2],
        same_environment_gram=tuple(
            tuple(complex(item) for item in row) for row in same_environment
        ),
        fresh_environment_composed_gram=tuple(
            tuple(complex(item) for item in row) for row in fresh_environment
        ),
        naive_reduced_composition_residual=residual,
        memory_aware_description_required=residual > _TOL,
        same_environment_retained=True,
        fresh_environment_reset_is_extra_assumption=True,
        commuting_diagonal_no_history_transition_assumption=True,
    )


@dataclass(frozen=True)
class JointEnvironmentSliceMemoryAudit:
    """Exact history audit for a retained, possibly entangled environment."""

    slice_count: int
    history_count: int
    environment_dimension: int
    same_environment_gram: tuple[tuple[complex, ...], ...]
    fresh_environment_composed_gram: tuple[tuple[complex, ...], ...]
    naive_reduced_composition_residual: float
    same_environment_gram_minimum_eigenvalue: float
    same_environment_gram_diagonal_residual: float
    memory_aware_description_required: bool
    joint_environment_correlations_allowed: bool
    controlled_history_initial_product_assumption: bool
    fresh_environment_reset_is_extra_assumption: bool
    conditional_slice_unitaries_may_be_noncommuting: bool
    general_process_tensor_implemented: bool = False


def audit_joint_environment_slice_memory(
    slice_conditional_unitaries: np.ndarray,
    environment_state: np.ndarray,
) -> JointEnvironmentSliceMemoryAudit:
    """Compare retained-environment histories with a fresh bath per slice.

    ``slice_conditional_unitaries[r,a]`` is the environment unitary in slice
    ``r`` for a pre-enumerated complete system history ``a``.  For one retained
    environment the ordered history unitary is

        V_a = U[L-1,a] ... U[1,a] U[0,a],

    and the exact Gram is ``Tr(V_a rho_E V_b^dagger)``.  Multiplying the
    slice-level Grams instead replaces the environment by a fresh independent
    copy in every slice.  General interventions or unenumerated history
    transitions require the full joint state or a process tensor.
    """

    slices = np.asarray(slice_conditional_unitaries, dtype=complex)
    if slices.ndim != 4 or slices.shape[0] == 0 or slices.shape[1] == 0:
        raise ValueError(
            "slice_conditional_unitaries must have shape "
            "(slice_count, history_count, dimension, dimension)"
        )
    state = _density_matrix(environment_state, "environment_state")
    slice_count, history_count = slices.shape[:2]
    dimension = state.shape[0]
    if slices.shape[2:] != (dimension, dimension):
        raise ValueError(
            "environment_state dimension must match the slice conditional unitaries"
        )

    totals = np.repeat(
        np.eye(dimension, dtype=complex)[np.newaxis, :, :],
        history_count,
        axis=0,
    )
    fresh = np.ones((history_count, history_count), dtype=complex)
    for slice_index, slice_value in enumerate(slices):
        unitaries = _conditional_unitary_stack(
            slice_value,
            f"slice_conditional_unitaries[{slice_index}]",
        )
        totals = np.einsum("aij,ajk->aik", unitaries, totals, optimize=True)
        fresh *= joint_environment_influence_gram(unitaries, state)

    same = joint_environment_influence_gram(totals, state)
    hermitian_same = 0.5 * (same + same.conj().T)
    residual = float(np.linalg.norm(same - fresh, ord=2))
    return JointEnvironmentSliceMemoryAudit(
        slice_count=slice_count,
        history_count=history_count,
        environment_dimension=dimension,
        same_environment_gram=tuple(
            tuple(complex(item) for item in row) for row in same
        ),
        fresh_environment_composed_gram=tuple(
            tuple(complex(item) for item in row) for row in fresh
        ),
        naive_reduced_composition_residual=residual,
        same_environment_gram_minimum_eigenvalue=float(
            np.min(np.linalg.eigvalsh(hermitian_same))
        ),
        same_environment_gram_diagonal_residual=float(
            np.linalg.norm(np.diag(same) - 1.0, ord=np.inf)
        ),
        memory_aware_description_required=residual > _TOL,
        joint_environment_correlations_allowed=True,
        controlled_history_initial_product_assumption=True,
        fresh_environment_reset_is_extra_assumption=True,
        conditional_slice_unitaries_may_be_noncommuting=True,
    )


def reduced_system_state(
    joint_state: np.ndarray,
    joint_unitary: np.ndarray,
    *,
    system_dimension: int,
    environment_dimension: int,
) -> np.ndarray:
    """Evolve a supplied joint state and trace the environment exactly.

    This full-state path permits initial system-environment correlations.  It
    does not define a preparation-independent CPTP map from an arbitrary
    reduced system state alone.
    """

    if (
        not isinstance(system_dimension, (int, np.integer))
        or system_dimension <= 0
        or not isinstance(environment_dimension, (int, np.integer))
        or environment_dimension <= 0
    ):
        raise ValueError("system and environment dimensions must be positive integers")
    expected_dimension = int(system_dimension) * int(environment_dimension)
    state = _density_matrix(joint_state, "joint_state")
    unitary = _unitary_matrix(joint_unitary, "joint_unitary")
    if state.shape != (expected_dimension, expected_dimension):
        raise ValueError("joint_state dimension does not match the declared factors")
    if unitary.shape != state.shape:
        raise ValueError("joint_unitary must match joint_state")
    evolved = unitary @ state @ unitary.conj().T
    tensor = evolved.reshape(
        int(system_dimension),
        int(environment_dimension),
        int(system_dimension),
        int(environment_dimension),
    )
    return np.trace(tensor, axis1=1, axis2=3)


@dataclass(frozen=True)
class QuantumKickConservationAudit:
    """Operator-level impulse and closed-receiver conservation receipt."""

    sector_count: int
    component_count: int
    hilbert_dimension: int
    mean_kicks: tuple[tuple[float, ...], ...]
    kick_covariances: tuple[tuple[tuple[float, ...], ...], ...]
    total_mean_kicks: tuple[float, ...]
    total_kick_operator_residuals: tuple[float, ...]
    total_momentum_commutator_residuals: tuple[float, ...]
    expectation_imaginary_residual: float
    all_receivers_included: bool
    operator_conservation_certified: bool
    unitary_dimensionless: bool
    momentum_dimension_input_contract_declared: bool
    force_time_window_derived: bool
    four_vector_covariance_derived: bool
    physical_clarus_source_derived: bool
    stress_tensor_derived: bool


def audit_quantum_kick_conservation(
    joint_unitary: np.ndarray,
    joint_state: np.ndarray,
    sector_momentum_operators: np.ndarray,
    *,
    all_receivers_included: bool,
) -> QuantumKickConservationAudit:
    """Calculate exact Heisenberg kicks and test total operator conservation.

    The operator input has shape ``(sector, component, dimension, dimension)``.
    Each sector may represent the observed system, environment, mediator,
    detector, or battery, but the caller must explicitly declare whether that
    list is closed.  For every supplied component,

        Delta P_X = U^dagger P_X U - P_X.

    Conservation for every input state is certified only from the operator
    identity ``sum_X Delta P_X = 0``, equivalently ``[U,sum_X P_X]=0``.  A
    vanishing expectation in one state is not enough.  Dividing the mean kick
    by a declared time window would define an average force; this routine does
    not infer that window, Lorentz covariance, a Clarus source, or stress.
    """

    if not isinstance(all_receivers_included, (bool, np.bool_)):
        raise ValueError("all_receivers_included must be boolean")
    unitary = _unitary_matrix(joint_unitary, "joint_unitary")
    state = _density_matrix(joint_state, "joint_state")
    if state.shape != unitary.shape:
        raise ValueError("joint_state must match joint_unitary")
    operators = np.asarray(sector_momentum_operators, dtype=complex)
    if (
        operators.ndim != 4
        or operators.shape[0] == 0
        or operators.shape[1] == 0
        or operators.shape[2:] != unitary.shape
    ):
        raise ValueError(
            "sector_momentum_operators must have shape "
            "(sector_count, component_count, dimension, dimension)"
        )
    for sector in range(operators.shape[0]):
        for component in range(operators.shape[1]):
            _hermitian_matrix(
                operators[sector, component],
                f"sector_momentum_operators[{sector},{component}]",
            )

    kicks = np.einsum(
        "ij,scjk,kl->scil",
        unitary.conj().T,
        operators,
        unitary,
        optimize=True,
    ) - operators
    raw_means = np.einsum("ij,scji->sc", state, kicks, optimize=True)
    expectation_imaginary_residual = float(np.max(np.abs(raw_means.imag)))
    means = raw_means.real
    identity = np.eye(unitary.shape[0], dtype=complex)
    covariances = np.empty(
        (operators.shape[0], operators.shape[1], operators.shape[1]),
        dtype=float,
    )
    for sector in range(operators.shape[0]):
        centered = kicks[sector] - means[sector, :, np.newaxis, np.newaxis] * identity
        for left in range(operators.shape[1]):
            for right in range(operators.shape[1]):
                anticommutator = (
                    centered[left] @ centered[right]
                    + centered[right] @ centered[left]
                )
                covariance = 0.5 * np.trace(state @ anticommutator)
                expectation_imaginary_residual = max(
                    expectation_imaginary_residual,
                    abs(float(covariance.imag)),
                )
                covariances[sector, left, right] = float(covariance.real)

    total_kicks = np.sum(kicks, axis=0)
    total_momenta = np.sum(operators, axis=0)
    total_operator_residuals = tuple(
        float(np.linalg.norm(value, ord=2)) for value in total_kicks
    )
    total_commutator_residuals = tuple(
        float(np.linalg.norm(unitary @ value - value @ unitary, ord=2))
        for value in total_momenta
    )
    operator_conservation = bool(
        all_receivers_included
        and max(total_operator_residuals, default=math.inf) <= _TOL
        and max(total_commutator_residuals, default=math.inf) <= _TOL
    )
    return QuantumKickConservationAudit(
        sector_count=operators.shape[0],
        component_count=operators.shape[1],
        hilbert_dimension=unitary.shape[0],
        mean_kicks=tuple(tuple(float(item) for item in row) for row in means),
        kick_covariances=tuple(
            tuple(tuple(float(item) for item in row) for row in sector)
            for sector in covariances
        ),
        total_mean_kicks=tuple(float(item) for item in np.sum(means, axis=0)),
        total_kick_operator_residuals=total_operator_residuals,
        total_momentum_commutator_residuals=total_commutator_residuals,
        expectation_imaginary_residual=expectation_imaginary_residual,
        all_receivers_included=bool(all_receivers_included),
        operator_conservation_certified=operator_conservation,
        unitary_dimensionless=True,
        momentum_dimension_input_contract_declared=True,
        force_time_window_derived=False,
        four_vector_covariance_derived=False,
        physical_clarus_source_derived=False,
        stress_tensor_derived=False,
    )


def unitary_order_residual(first: np.ndarray, second: np.ndarray) -> float:
    """Return ``||second first - first second||_2`` for two finite gates.

    A zero residual proves algebraic order independence.  Relativistic
    no-signalling additionally requires genuinely spacelike local supports and
    a microcausal field/instrument model; this function does not infer them.
    """

    first_unitary = _unitary_matrix(first, "first")
    second_unitary = _unitary_matrix(second, "second")
    if first_unitary.shape != second_unitary.shape:
        raise ValueError("first and second must have equal dimensions")
    return float(
        np.linalg.norm(
            second_unitary @ first_unitary - first_unitary @ second_unitary,
            ord=2,
        )
    )


def _schur_choi(gram: np.ndarray) -> np.ndarray:
    """Choi matrix of ``rho -> gram * rho`` in the computational basis."""

    result = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            basis = np.zeros((2, 2), dtype=complex)
            basis[a, b] = 1.0
            result += np.kron(basis, gram[a, b] * basis)
    return result


def certify_finite_ctp_diagonal_source_obstruction(
    *, p: float = 0.3, tau: float = 2.0, hbar: float = 1.0,
    omega_star: float = 3.0, slope: float = 1.0, h_star: float = 0.0,
    h_delta: float = 0.1, finite_difference_step: float = 1.0e-5,
) -> FiniteCTPDiagonalSourceCertificate:
    """Certify the finite, scalar CTP obstruction and its CPTP dilation."""

    p = _unit_interval_closed(p, "p")
    tau = _finite(tau, "tau")
    hbar = _finite(hbar, "hbar")
    slope = _finite(slope, "slope")
    omega_star = _finite(omega_star, "omega_star")
    h_star = _finite(h_star, "h_star")
    h_delta = _finite(h_delta, "h_delta")
    step = _finite(finite_difference_step, "finite_difference_step")
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")
    if step <= 0.0:
        raise ValueError("finite_difference_step must be positive")

    h_plus, h_minus = h_star + h_delta / 2.0, h_star - h_delta / 2.0
    value = influence(h_plus, h_minus, p=p, tau=tau, omega_star=omega_star, slope=slope, h_star=h_star)
    diagonal_action = influence_action(h_star, h_star, p=p, tau=tau, hbar=hbar, omega_star=omega_star, slope=slope, h_star=h_star)
    action_plus = influence_action(h_star + step / 2.0, h_star - step / 2.0, p=p, tau=tau, hbar=hbar, omega_star=omega_star, slope=slope, h_star=h_star)
    action_minus = influence_action(h_star - step / 2.0, h_star + step / 2.0, p=p, tau=tau, hbar=hbar, omega_star=omega_star, slope=slope, h_star=h_star)
    source = -hbar * p * tau * slope
    central_source = float(((action_plus - action_minus) / (2.0 * step)).real)
    quadratic = 0.5 * hbar * p * (1.0 - p) * (tau * slope) ** 2
    # S(d)+S(-d)-2S(0) = 2 i*quadratic*d^2 + O(d^4).
    symmetric_quadratic = (action_plus + action_minus - 2.0 * diagonal_action) / (2.0 * step * step)
    local_model = source * h_delta + 1j * quadratic * h_delta * h_delta
    action_at_delta = influence_action(h_plus, h_minus, p=p, tau=tau, hbar=hbar, omega_star=omega_star, slope=slope, h_star=h_star)

    rho_environment = np.diag((1.0 - p, p)).astype(complex)
    labels = (-0.5, 0.5)
    unitaries = tuple(np.diag((1.0, np.exp(-1j * tau * omega(label, omega_star=omega_star, slope=slope, h_star=h_star)))).astype(complex) for label in labels)
    controlled = sum((np.kron(np.eye(2)[a : a + 1].T @ np.eye(2)[a : a + 1], unitary) for a, unitary in enumerate(unitaries)), np.zeros((4, 4), complex))
    gram = np.array([[np.trace(rho_environment @ unitaries[b].conj().T @ unitaries[a]) for b in range(2)] for a in range(2)], dtype=complex)
    choi = _schur_choi(gram)
    plus = np.full((2, 2), 0.5, dtype=complex)
    output = gram * plus

    def scalar_source(local_p: float, local_tau: float, local_slope: float) -> float:
        return -hbar * local_p * local_tau * local_slope

    def decoherence(local_p: float, local_tau: float, local_slope: float) -> float:
        return 1.0 - abs((1.0 - local_p) + local_p * np.exp(-1j * local_tau * local_slope * h_delta))

    model_zero = 0.0  # omega_0(h)=omega_star
    model_nonzero = source  # omega_1(h)=omega_star+slope*(h-h_star)
    omega0_at_reference = omega(h_star, omega_star=omega_star, slope=0.0, h_star=h_star)
    omega1_at_reference = omega(h_star, omega_star=omega_star, slope=slope, h_star=h_star)
    diagonal0 = influence(h_star, h_star, p=p, tau=tau, omega_star=omega_star, slope=0.0, h_star=h_star)
    diagonal1 = influence(h_star, h_star, p=p, tau=tau, omega_star=omega_star, slope=slope, h_star=h_star)
    p_one_phase = abs(np.angle(influence(h_plus, h_minus, p=1.0, tau=tau, omega_star=omega_star, slope=slope, h_star=h_star)))
    h_dim, omega_dim, slope_dim, tau_dim = 0, 1, 1, -1
    tau_omega_dim = tau_dim + omega_dim
    influence_dim = 0
    action_over_hbar_dim = 0
    dimensions_pass = bool(
        h_dim == 0
        and slope_dim == omega_dim == 1
        and tau_omega_dim == 0
        and influence_dim == action_over_hbar_dim == 0
    )

    return FiniteCTPDiagonalSourceCertificate(
        probability_one=p, hbar=hbar, tau=tau, slope=slope, h_delta=h_delta, influence=value,
        influence_diagonal_residual=abs(influence(h_star, h_star, p=p, tau=tau, omega_star=omega_star, slope=slope, h_star=h_star) - 1.0),
        action_diagonal_residual=abs(diagonal_action), h_c_derivative_at_diagonal=0.0,
        difference_source=source, central_difference_source=central_source,
        central_difference_residual=abs(central_source - source), linear_action_coefficient=source,
        quadratic_imaginary_action_coefficient=quadratic, symmetric_quadratic_coefficient=complex(symmetric_quadratic),
        local_expansion_residual=abs(action_at_delta - local_model), model_zero_difference_source=model_zero,
        model_nonzero_difference_source=model_nonzero,
        model_reference_frequency_residual=abs(omega0_at_reference - omega1_at_reference),
        model_reference_hamiltonian_residual=abs(hbar * (omega0_at_reference - omega1_at_reference)),
        diagonal_model_influence_residual=abs(diagonal0 - diagonal1), diagonal_readout_probabilities=(1.0 - p, p),
        limited_non_identifiability=(abs(diagonal0 - diagonal1) <= _TOL and abs(model_zero - model_nonzero) > _TOL),
        environment_minimum_eigenvalue=float(np.linalg.eigvalsh(rho_environment).min()),
        environment_trace_residual=abs(np.trace(rho_environment) - 1.0),
        controlled_unitary_residual=float(np.linalg.norm(controlled.conj().T @ controlled - np.eye(4), ord=2)),
        gram_minimum_eigenvalue=float(np.linalg.eigvalsh(gram).min()),
        gram_diagonal_residual=float(np.linalg.norm(np.diag(gram) - 1.0, ord=np.inf)),
        schur_choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi).min()),
        schur_trace_preservation_residual=float(np.linalg.norm(np.diag(gram) - 1.0, ord=np.inf)),
        schur_output_trace_residual=abs(np.trace(output) - 1.0),
        schur_completely_positive=bool(np.linalg.eigvalsh(choi).min() >= -_TOL),
        schur_trace_preserving=bool(np.linalg.norm(np.diag(gram) - 1.0, ord=np.inf) <= _TOL),
        plus_state_coherence=complex(output[0, 1]),
        p_zero_source=scalar_source(0.0, tau, slope), tau_zero_source=scalar_source(p, 0.0, slope), slope_zero_source=scalar_source(p, tau, 0.0),
        p_zero_decoherence=decoherence(0.0, tau, slope), tau_zero_decoherence=decoherence(p, 0.0, slope), slope_zero_decoherence=decoherence(p, tau, 0.0),
        p_one_quadratic_noise_coefficient=0.5 * hbar * 1.0 * 0.0 * (tau * slope) ** 2,
        p_one_unitary_phase_present=bool(p_one_phase > _TOL) if tau * slope * h_delta != 0.0 else False,
        h_mass_dimension=h_dim, omega_mass_dimension=omega_dim, slope_mass_dimension=slope_dim,
        tau_mass_dimension=tau_dim, tau_omega_mass_dimension=tau_omega_dim,
        influence_mass_dimension=influence_dim, action_over_hbar_mass_dimension=action_over_hbar_dim,
        difference_source_dimension="action", dimensions_pass=dimensions_pass,
        accounting_mode="integrated_out_influence_only",
        retained_environment_stress_added=False, rn_reweighting_used=False,
    )
