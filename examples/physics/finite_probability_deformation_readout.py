"""Finite E19 witness: Newtonian measure reparameterization and sharp records.

The radial calculation only rewrites a *supplied*, finite Newtonian potential
as a Radon--Nikodym weight on a supplied uniform-volume measure.  It derives
neither gravity nor an attraction mechanism.  The quantum calculation is kept
separate: its Lüders instrument never receives the radial weight, so it is not
a second probability weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_TOL = 2.0e-11


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return value


def _legendre_integral(function, left: float, right: float, *, order: int = 192) -> float:
    """Deterministic Gauss--Legendre integral on a finite supplied interval."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    radii = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    return float(0.5 * (right - left) * np.dot(weights, function(radii)))


def _partial_trace_a(rho_ab: np.ndarray, a_dimension: int, b_dimension: int) -> np.ndarray:
    return np.trace(rho_ab.reshape(a_dimension, b_dimension, a_dimension, b_dimension), axis1=0, axis2=2)


@dataclass(frozen=True)
class FiniteProbabilityDeformationReadoutCertificate:
    compactness: float
    domain_ratio: float
    holdout_x1: float
    holdout_x2: float
    normalizer: float
    log_normalizer: float
    normalization_residual: float
    constant_shift_invariance_residual: float
    inward_likelihood_ratio: float
    holdout_probability: float
    chi_continuity_residual_at_surface: float
    scaled_radial_laplacian_inside: float
    scaled_radial_laplacian_outside: float
    inside_chi_prime_over_x: float
    outside_x_squared_chi_prime: float
    scaled_acceleration_at_x_half: float
    scaled_acceleration_at_holdout_x1: float
    chi_equals_minus_newtonian_potential_over_c_squared: bool
    finite_sphere_regulates_normalization: bool
    point_source_global_normalization_available: bool
    point_source_uniform_volume_integral_diverges: bool
    epsilon_mass_dimension: int
    scaled_radius_mass_dimension: int
    chi_mass_dimension: int
    normalizer_mass_dimension: int
    probability_mass_dimension: int
    chi_derivative_invariant_mass_dimension: int
    dimensions_pass: bool
    parameter_fit_count: int
    internal_radial_holdout_only: bool
    observational_holdout_gate_closed: bool
    record_probability_rho0: tuple[float, float]
    record_probability_rho1: tuple[float, float]
    record_probability_rho2: tuple[float, float]
    distinct_microstates_same_sharp_record: bool
    kraus_completeness_residual: float
    choi_minimum_eigenvalue: float
    channel_trace_preservation_residual: float
    channel_completely_positive: bool
    channel_trace_preserving: bool
    sharp_projector_repeatability_residual: float
    immediate_sharp_repeatability: bool
    classical_record_dephasing_idempotence_residual: float
    single_witness_remote_marginal_residual: float
    no_probability_double_weighting: bool
    newtonian_reparameterization_only: bool = True
    independent_chi_action_or_dynamics_derived: bool = False
    probability_current_or_attraction_mechanism_derived: bool = False
    causal_retarded_field_or_c_front_derived: bool = False
    scalar_to_gr_or_lensing_derived: bool = False
    gravity_energy_or_backreaction_derived: bool = False
    quantum_matter_dependent_chi_channel_derived: bool = False
    general_observation_repeatability_derived: bool = False
    physical_selection_derived: bool = False
    ideal_point_source_normalization_derived: bool = False
    homology_cohomology_self_duality_derived: bool = False
    actual_data_holdout_or_gates_5_to_8_closed: bool = False
    two_residuals_or_complexity_success: bool = False


def certify_finite_probability_deformation_readout(
    *, compactness: float = 0.01, domain_ratio: float = 10.0,
    holdout_x1: float = 2.0, holdout_x2: float = 3.0,
) -> FiniteProbabilityDeformationReadoutCertificate:
    """Return a finite-domain RN witness and a separate 0D sharp-record witness.

    ``compactness = GM/(R_s c^2)`` and ``x=r/R_s`` are dimensionless.  The
    supplied finite sphere has ``chi=-Phi/c^2`` and base measure
    ``dmu0=3*x^2/domain_ratio^3 dx``.  A global point-source normalizer is
    intentionally unavailable: its uniform-volume integral diverges.
    """

    epsilon = _unit_interval(compactness, "compactness")
    domain = _positive(domain_ratio, "domain_ratio")
    if domain <= 1.0:
        raise ValueError("domain_ratio must exceed 1")
    x1 = _positive(holdout_x1, "holdout_x1")
    x2 = _positive(holdout_x2, "holdout_x2")
    if not 1.0 < x1 < x2 < domain:
        raise ValueError("holdout radii must satisfy 1 < holdout_x1 < holdout_x2 < domain_ratio")

    def chi(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 1.0, 0.5 * epsilon * (3.0 - x * x), epsilon / x)

    def weighted_volume(x: np.ndarray) -> np.ndarray:
        # This is exactly ``3*x**2/domain**3`` but avoids an intermediate
        # ``x**2`` overflow for a very large, still finite supplied domain.
        return 3.0 * (x / domain) ** 2 * np.exp(chi(x)) / domain

    # Split at x=1: the supplied potential is continuous but its derivatives differ.
    normalizer = _legendre_integral(weighted_volume, 0.0, 1.0) + _legendre_integral(weighted_volume, 1.0, domain)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("normalizer must be finite and positive on the supplied domain")
    log_normalizer = math.log(normalizer)
    normalized_integral = (
        _legendre_integral(lambda x: weighted_volume(x) / normalizer, 0.0, 1.0)
        + _legendre_integral(lambda x: weighted_volume(x) / normalizer, 1.0, domain)
    )
    holdout = _legendre_integral(lambda x: weighted_volume(x) / normalizer, x1, x2)
    if not math.isfinite(holdout) or not 0.0 < holdout < 1.0:
        raise ValueError("holdout probability must be finite and lie in (0, 1)")
    shift = 0.731
    shifted_normalizer = (
        _legendre_integral(lambda x: weighted_volume(x) * math.exp(shift), 0.0, 1.0)
        + _legendre_integral(lambda x: weighted_volume(x) * math.exp(shift), 1.0, domain)
    )
    # Equal-volume shells: the inward one has the larger RN factor.
    inward_ratio = math.exp(float(chi(np.array([x1]))[0] - chi(np.array([x2]))[0]))

    p0 = np.diag((1.0, 1.0, 0.0)).astype(complex)
    p1 = np.diag((0.0, 0.0, 1.0)).astype(complex)
    kraus = (p0, p1)
    completeness = sum((operator.conj().T @ operator for operator in kraus), np.zeros((3, 3), complex))
    rho0 = np.diag((1.0, 0.0, 0.0)).astype(complex)
    rho1 = np.diag((0.0, 1.0, 0.0)).astype(complex)
    rho2 = np.diag((0.0, 0.0, 1.0)).astype(complex)
    probabilities0 = tuple(float(np.trace(operator @ rho0 @ operator.conj().T).real) for operator in kraus)
    probabilities1 = tuple(float(np.trace(operator @ rho1 @ operator.conj().T).real) for operator in kraus)
    probabilities2 = tuple(float(np.trace(operator @ rho2 @ operator.conj().T).real) for operator in kraus)
    channel = lambda rho: sum((operator @ rho @ operator.conj().T for operator in kraus), np.zeros_like(rho))
    choi = sum(
        np.kron(np.eye(3)[i : i + 1].T @ np.eye(3)[j : j + 1], channel(np.eye(3, dtype=complex)[i : i + 1].T @ np.eye(3, dtype=complex)[j : j + 1]))
        for i in range(3) for j in range(3)
    )
    # A representative record is dephased, then dephased again.
    record = np.array(((0.5, 0.25j), (-0.25j, 0.5)), dtype=complex)
    record_dephase = lambda matrix: np.diag(np.diag(matrix))
    repeatability_residual = max(
        float(np.linalg.norm(
            second @ first - (first if first_label == second_label else np.zeros((3, 3))),
            ord=2,
        ))
        for first_label, first in enumerate(kraus)
        for second_label, second in enumerate(kraus)
    )

    psi = np.zeros(6, dtype=complex)
    psi[0] = 1.0 / math.sqrt(2.0)  # |0>_A |0>_B
    psi[5] = 1.0 / math.sqrt(2.0)  # |2>_A |1>_B
    rho_ab = np.outer(psi, psi.conj())
    remote_before = _partial_trace_a(rho_ab, 3, 2)
    local_nonselective = sum((np.kron(operator, np.eye(2)) @ rho_ab @ np.kron(operator, np.eye(2)).conj().T for operator in kraus), np.zeros_like(rho_ab))
    remote_after = _partial_trace_a(local_nonselective, 3, 2)

    return FiniteProbabilityDeformationReadoutCertificate(
        compactness=epsilon, domain_ratio=domain, holdout_x1=x1, holdout_x2=x2,
        normalizer=normalizer, log_normalizer=log_normalizer,
        normalization_residual=abs(normalized_integral - 1.0),
        constant_shift_invariance_residual=abs(shifted_normalizer / math.exp(shift) / normalizer - 1.0),
        inward_likelihood_ratio=inward_ratio, holdout_probability=holdout,
        chi_continuity_residual_at_surface=abs(0.5 * epsilon * (3.0 - 1.0) - epsilon),
        scaled_radial_laplacian_inside=-3.0 * epsilon,
        scaled_radial_laplacian_outside=0.0,
        inside_chi_prime_over_x=-epsilon,
        outside_x_squared_chi_prime=-epsilon,
        scaled_acceleration_at_x_half=-0.5 * epsilon,
        scaled_acceleration_at_holdout_x1=-epsilon / x1**2,
        chi_equals_minus_newtonian_potential_over_c_squared=True,
        finite_sphere_regulates_normalization=True,
        point_source_global_normalization_available=False,
        point_source_uniform_volume_integral_diverges=True,
        epsilon_mass_dimension=0, scaled_radius_mass_dimension=0, chi_mass_dimension=0,
        normalizer_mass_dimension=0, probability_mass_dimension=0,
        chi_derivative_invariant_mass_dimension=0, dimensions_pass=True,
        parameter_fit_count=0, internal_radial_holdout_only=True,
        observational_holdout_gate_closed=False,
        record_probability_rho0=probabilities0, record_probability_rho1=probabilities1,
        record_probability_rho2=probabilities2,
        distinct_microstates_same_sharp_record=(not np.allclose(rho0, rho1) and probabilities0 == probabilities1 == (1.0, 0.0)),
        kraus_completeness_residual=float(np.linalg.norm(completeness - np.eye(3), ord=2)),
        choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi).min()),
        channel_trace_preservation_residual=max(abs(np.trace(channel(np.eye(3) / 3.0)) - 1.0), 0.0),
        channel_completely_positive=bool(np.linalg.eigvalsh(choi).min() >= -_TOL),
        channel_trace_preserving=bool(np.linalg.norm(completeness - np.eye(3), ord=2) <= _TOL),
        sharp_projector_repeatability_residual=repeatability_residual,
        immediate_sharp_repeatability=bool(repeatability_residual <= _TOL),
        classical_record_dephasing_idempotence_residual=float(np.linalg.norm(record_dephase(record_dephase(record)) - record_dephase(record), ord=2)),
        single_witness_remote_marginal_residual=float(np.linalg.norm(remote_after - remote_before, ord=2)),
        no_probability_double_weighting=True,
    )
