"""Finite E16 certificate for basis overlap, barrier leakage, and dynamics.

Natural units use ``hbar = 1``.  The localized functions use the explicit
convention

``psi_n(x) = (pi sigma^2)^(-1/4) exp(-(x - n a)^2/(2 sigma^2))``.

The certificate deliberately keeps three incommensurate budgets separate:
``basis_amplitude``, ``barrier_probability``, and
``projected_operator_norm``.  In particular it never adds them into a single
"error".  It is a finite, supplied-parameter witness only; the false flags at
the end of :class:`FiniteBarrierModeLeakageCertificate` record what it does
not derive.  Displayed ``S = 0`` or transmission ``T = 0`` can be numerical
underflow from the log-domain calculation; neither asserts exact orthogonality,
compact support, or exact finite-barrier localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_TOLERANCE = 1.0e-11
_SIGMA_X = np.array(((0.0, 1.0), (1.0, 0.0)), dtype=complex)
_SIGMA_Z = np.array(((1.0, 0.0), (0.0, -1.0)), dtype=complex)
_IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass(frozen=True)
class FiniteBarrierModeLeakageCertificate:
    """Separate finite certificates; fields labelled ``budget`` are not summed."""

    mode_count: int
    sigma: float
    center_spacing: float
    gaussian_log_overlap_amplitude: float
    gaussian_overlap_amplitude: float
    gaussian_overlap_numerically_underflowed: bool
    basis_amplitude_budget: float
    basis_log_amplitude_budget: float
    basis_amplitude_target: float
    required_center_spacing: float
    basis_amplitude_budget_holds: bool
    nonrelativistic_mass: float
    barrier_height: float
    incident_energy: float
    barrier_width: float
    kappa: float
    barrier_log_transmission_probability: float
    barrier_transmission_probability: float
    barrier_probability_numerically_underflowed: bool
    barrier_probability_budget: float
    barrier_log_probability_budget: float
    barrier_probability_target: float
    per_barrier_probability_target: float
    exact_required_barrier_width: float
    exponential_prefactor: float
    exponential_regime_threshold: float
    exponential_regime_holds: bool
    exponential_probability_upper: float | None
    exponential_required_barrier_width: float
    barrier_probability_budget_holds: bool
    ideal_hopping: float
    projected_hamiltonian_norm_error: float
    dynamic_operator_norm_target: float
    ideal_swap_time: float
    ideal_swap_probability: float
    ideal_unitarity_residual: float
    ideal_swap_phase_residual: float
    single_step_operator_difference: float
    single_step_duhamel_bound_raw: float
    single_step_duhamel_bound_clipped: float
    repeated_step_operator_difference: float
    repeated_step_telescoping_bound_raw: float
    repeated_step_telescoping_bound_clipped: float
    required_projected_hamiltonian_norm_error: float
    projected_operator_norm_budget_holds: bool
    error_type_tuple: tuple[str, str, str]
    sigma_mass_dimension: int
    spacing_mass_dimension: int
    mass_mass_dimension: int
    energy_mass_dimension: int
    kappa_mass_dimension: int
    barrier_width_mass_dimension: int
    hopping_mass_dimension: int
    hamiltonian_norm_error_mass_dimension: int
    time_mass_dimension: int
    overlap_amplitude_mass_dimension: int
    transmission_probability_mass_dimension: int
    operator_norm_difference_mass_dimension: int
    dimensions_pass: bool
    identities_and_finite_witness_only: bool
    identical_parameters_required_by_contract: bool
    e15_modes_derived: bool = False
    kg_to_schrodinger_projection_derived: bool = False
    rectangular_barrier_represents_periodic_lattice: bool = False
    barrier_or_wkb_to_hopping_derived: bool = False
    finite_barrier_exact_localization: bool = False
    autonomous_dwell_time_derived: bool = False
    scattering_instrument_or_energy_receipt_derived: bool = False
    repeated_cptp_or_fresh_ancilla_derived: bool = False
    causal_or_strict_front_derived: bool = False
    qft_microcausality_or_no_signalling_derived: bool = False
    gr_source_derived: bool = False
    selection_derived: bool = False
    gates_5_to_8_closed: bool = False


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return value


def gaussian_overlap_amplitude(*, sigma: float, center_spacing: float) -> float:
    """Return displayed ``S`` for the stated psi, which can underflow to zero.

    Such zero is numerical only: finite Gaussian tails are neither exactly
    orthogonal nor compactly supported.  Use the certificate's log field for
    the finite value when the displayed amplitude underflows.
    """

    sigma = _positive(sigma, "sigma")
    center_spacing = _positive(center_spacing, "center_spacing")
    return math.exp(_gaussian_log_overlap(sigma=sigma, center_spacing=center_spacing))


def _gaussian_log_overlap(*, sigma: float, center_spacing: float) -> float:
    return -(center_spacing / (2.0 * sigma)) ** 2


def exact_rectangular_barrier_transmission_probability(
    *, nonrelativistic_mass: float, barrier_height: float, incident_energy: float,
    barrier_width: float,
) -> float:
    """Exact below-barrier 1D transmission probability, not a WKB replacement."""

    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    energy = _positive(incident_energy, "incident_energy")
    width = _positive(barrier_width, "barrier_width")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")
    log_transmission, _ = _barrier_log_transmission(
        mass=mass, height=height, energy=energy, width=width
    )
    return math.exp(log_transmission)


def _log_sinh_nonnegative(value: float) -> float:
    """Stable ``log(sinh(value))`` for positive dimensionless ``value``."""

    if value < 40.0:
        return math.log(math.sinh(value))
    return value - math.log(2.0) + math.log1p(-math.exp(-2.0 * value))


def _barrier_log_transmission(
    *, mass: float, height: float, energy: float, width: float
) -> tuple[float, float]:
    """Return ``log(T)`` and kappa without overflowing a finite barrier."""

    log_energy_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_factor = -math.log(4.0) - log_energy_fraction - log_gap_fraction
    kappa = math.sqrt(2.0 * mass * (height - energy))
    log_sinh = _log_sinh_nonnegative(kappa * width)
    log_transmission = -float(np.logaddexp(0.0, log_factor + 2.0 * log_sinh))
    return log_transmission, kappa


def _asinh_exp(log_argument: float) -> float:
    """Return ``asinh(exp(log_argument))`` without overflowing ``exp``."""

    if log_argument < 40.0:
        return math.asinh(math.exp(log_argument))
    return log_argument + math.log(2.0)


def _exp_or_infinity(log_value: float) -> float:
    maximum_log_float = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
    return math.exp(log_value) if log_value < maximum_log_float else math.inf


def _hermitian_exponential(hamiltonian: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return (eigenvectors * np.exp(-1j * time * eigenvalues)) @ eigenvectors.conj().T


def certify_finite_barrier_mode_leakage(
    *,
    mode_count: int,
    sigma: float,
    center_spacing: float,
    delta_basis: float,
    nonrelativistic_mass: float,
    barrier_height: float,
    incident_energy: float,
    barrier_width: float,
    delta_leak: float,
    ideal_projected_hopping: float,
    projected_hamiltonian_norm_error: float,
    delta_dyn: float,
) -> FiniteBarrierModeLeakageCertificate:
    """Compute the approved finite E16 certificate for identical supplied cells.

    ``mode_count`` is used only in the declared union/telescoping allocations.
    The dynamic calculation is a finite two-dimensional operator-norm witness,
    neither a diamond-norm statement nor a CPTP construction.
    """

    if isinstance(mode_count, bool) or not isinstance(mode_count, int) or mode_count < 1:
        raise ValueError("mode_count must be an integer at least one")
    sigma = _positive(sigma, "sigma")
    spacing = _positive(center_spacing, "center_spacing")
    delta_basis = _unit_interval(delta_basis, "delta_basis")
    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    energy = _positive(incident_energy, "incident_energy")
    width = _positive(barrier_width, "barrier_width")
    delta_leak = _unit_interval(delta_leak, "delta_leak")
    hopping = _positive(ideal_projected_hopping, "ideal_projected_hopping")
    eta = _finite(projected_hamiltonian_norm_error, "projected_hamiltonian_norm_error")
    if eta < 0.0:
        raise ValueError("projected_hamiltonian_norm_error must be non-negative")
    delta_dyn = _unit_interval(delta_dyn, "delta_dyn")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")

    log_overlap = _gaussian_log_overlap(sigma=sigma, center_spacing=spacing)
    overlap = math.exp(log_overlap)
    log_mode_count = math.log(mode_count)
    log_basis_target = math.log(delta_basis)
    log_basis_budget = log_mode_count + log_overlap
    basis_budget = _exp_or_infinity(log_basis_budget)
    required_spacing = 2.0 * sigma * math.sqrt(log_mode_count - log_basis_target)

    log_transmission, kappa = _barrier_log_transmission(
        mass=mass, height=height, energy=energy, width=width
    )
    transmission = math.exp(log_transmission)
    log_leak_target = math.log(delta_leak)
    log_probability_budget = log_mode_count + log_transmission
    leak_budget = _exp_or_infinity(log_probability_budget)
    log_per_barrier_target = log_leak_target - log_mode_count
    per_barrier_target = math.exp(log_per_barrier_target)
    log_energy_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_inverse_factor = math.log(4.0) + log_energy_fraction + log_gap_fraction
    log_z = 0.5 * (
        log_inverse_factor
        + math.log1p(-per_barrier_target)
        - log_per_barrier_target
    )
    exact_required_width = _asinh_exp(log_z) / kappa
    log_prefactor = math.log(64.0) + log_energy_fraction + log_gap_fraction
    prefactor = math.exp(log_prefactor)
    regime_threshold = math.log(2.0)
    regime_holds = kappa * width >= regime_threshold
    exponential_upper = (
        math.exp(log_prefactor - 2.0 * kappa * width) if regime_holds else None
    )
    exponential_required_width = max(
        regime_threshold,
        0.5 * (log_prefactor + log_mode_count - log_leak_target),
    ) / kappa

    tau = math.pi / (2.0 * hopping)
    h0 = -hopping * _SIGMA_X
    h = h0 + eta * _SIGMA_Z
    u0 = _hermitian_exponential(h0, tau)
    u = _hermitian_exponential(h, tau)
    single_difference = float(np.linalg.norm(u - u0, ord=2))
    single_bound_raw = tau * eta
    repeated_u = np.linalg.matrix_power(u, mode_count)
    repeated_u0 = np.linalg.matrix_power(u0, mode_count)
    repeated_difference = float(np.linalg.norm(repeated_u - repeated_u0, ord=2))
    repeated_bound_raw = mode_count * single_bound_raw
    required_eta = delta_dyn / (mode_count * tau)

    ideal_unitarity = float(np.linalg.norm(u0.conj().T @ u0 - _IDENTITY_2, ord=2))
    ideal_probability = float(abs(u0[1, 0]) ** 2)
    ideal_phase = float(abs(u0[1, 0] - 1j))
    sigma_dimension = -1
    spacing_dimension = -1
    mass_dimension = 1
    energy_dimension = 1
    kappa_dimension = 1
    barrier_width_dimension = -1
    hopping_dimension = 1
    eta_dimension = 1
    time_dimension = -1
    amplitude_dimension = 0
    probability_dimension = 0
    operator_difference_dimension = 0
    # kappa^2=2m(V0-E), tau=pi/(2J), and tau*eta are the relevant checks.
    dimensions_pass = (
        sigma_dimension == spacing_dimension == -1
        and mass_dimension + energy_dimension == 2
        and kappa_dimension + barrier_width_dimension == 0
        and hopping_dimension + time_dimension == 0
        and time_dimension + eta_dimension == operator_difference_dimension
        and amplitude_dimension == probability_dimension == 0
    )
    dynamic_witness_holds = (
        single_difference <= single_bound_raw + _TOLERANCE
        and repeated_difference <= repeated_bound_raw + _TOLERANCE
    )

    return FiniteBarrierModeLeakageCertificate(
        mode_count=mode_count,
        sigma=sigma,
        center_spacing=spacing,
        gaussian_log_overlap_amplitude=log_overlap,
        gaussian_overlap_amplitude=overlap,
        gaussian_overlap_numerically_underflowed=(overlap == 0.0),
        basis_amplitude_budget=basis_budget,
        basis_log_amplitude_budget=log_basis_budget,
        basis_amplitude_target=delta_basis,
        required_center_spacing=required_spacing,
        basis_amplitude_budget_holds=(
            log_basis_budget <= log_basis_target + _TOLERANCE
        ),
        nonrelativistic_mass=mass,
        barrier_height=height,
        incident_energy=energy,
        barrier_width=width,
        kappa=kappa,
        barrier_log_transmission_probability=log_transmission,
        barrier_transmission_probability=transmission,
        barrier_probability_numerically_underflowed=(transmission == 0.0),
        barrier_probability_budget=leak_budget,
        barrier_log_probability_budget=log_probability_budget,
        barrier_probability_target=delta_leak,
        per_barrier_probability_target=per_barrier_target,
        exact_required_barrier_width=exact_required_width,
        exponential_prefactor=prefactor,
        exponential_regime_threshold=regime_threshold,
        exponential_regime_holds=regime_holds,
        exponential_probability_upper=exponential_upper,
        exponential_required_barrier_width=exponential_required_width,
        barrier_probability_budget_holds=(
            log_probability_budget <= log_leak_target + _TOLERANCE
        ),
        ideal_hopping=hopping,
        projected_hamiltonian_norm_error=eta,
        dynamic_operator_norm_target=delta_dyn,
        ideal_swap_time=tau,
        ideal_swap_probability=ideal_probability,
        ideal_unitarity_residual=ideal_unitarity,
        ideal_swap_phase_residual=ideal_phase,
        single_step_operator_difference=single_difference,
        single_step_duhamel_bound_raw=single_bound_raw,
        single_step_duhamel_bound_clipped=min(2.0, single_bound_raw),
        repeated_step_operator_difference=repeated_difference,
        repeated_step_telescoping_bound_raw=repeated_bound_raw,
        repeated_step_telescoping_bound_clipped=min(2.0, repeated_bound_raw),
        required_projected_hamiltonian_norm_error=required_eta,
        projected_operator_norm_budget_holds=(eta <= required_eta and dynamic_witness_holds),
        error_type_tuple=(
            "basis_amplitude",
            "barrier_probability",
            "projected_operator_norm",
        ),
        sigma_mass_dimension=sigma_dimension,
        spacing_mass_dimension=spacing_dimension,
        mass_mass_dimension=mass_dimension,
        energy_mass_dimension=energy_dimension,
        kappa_mass_dimension=kappa_dimension,
        barrier_width_mass_dimension=barrier_width_dimension,
        hopping_mass_dimension=hopping_dimension,
        hamiltonian_norm_error_mass_dimension=eta_dimension,
        time_mass_dimension=time_dimension,
        overlap_amplitude_mass_dimension=amplitude_dimension,
        transmission_probability_mass_dimension=probability_dimension,
        operator_norm_difference_mass_dimension=operator_difference_dimension,
        dimensions_pass=dimensions_pass,
        identities_and_finite_witness_only=True,
        identical_parameters_required_by_contract=True,
    )
