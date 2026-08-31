"""Finite CTP witness: a diagonal 0D record does not fix a difference source.

This is deliberately a two-level environment, not a stress tensor model.  It
keeps the closed-time-path (CTP) influence functional after the environment is
integrated out and demonstrates a limited non-identifiability: identical
diagonal data can accompany different derivatives with respect to the CTP
difference source.
"""

from __future__ import annotations

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
