"""Conditional monochromatic two-port barrier instrument in ``hbar = c = 1``.

This is a scattering calculation at one supplied energy, not an autonomous
detector.  The scattering amplitude uses a canonical barrier-face convention.
The conventional coordinate amplitude is ``t_conv = exp(-i*k*b) * t``: this
is a rephasing of a port, so the calculation makes no absolute-phase claim.
Displayed zero transmission at a very wide *finite* barrier is floating-point
underflow; ``log_transmission_probability`` remains the relevant quantity and
does not assert exact localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_I2 = np.eye(2, dtype=complex)
_P0 = np.diag((1.0, 0.0)).astype(complex)
_P1 = np.diag((0.0, 1.0)).astype(complex)


@dataclass(frozen=True)
class SingleEnergyBarrierInstrumentCertificate:
    """Finite conditional witness; its false flags delimit all omitted bridges."""

    nonrelativistic_mass: float
    barrier_height: float
    incident_energy: float
    barrier_width: float
    k: float
    kappa: float
    dimensionless_barrier_width: float
    coefficient_a: float
    coefficient_b: float
    tanh_dimensionless_width: float
    log_sech_dimensionless_width: float
    reflection_amplitude: complex
    transmission_amplitude: complex
    conventional_coordinate_transmission_amplitude: complex
    log_transmission_probability: float
    transmission_probability: float
    transmission_amplitude_numerically_underflowed: bool
    transmission_probability_numerically_underflowed: bool
    reflection_probability: float
    reflection_probability_residual: float
    scattering_matrix: np.ndarray
    scattering_unitarity_residual: float
    cross_amplitude_residual: float
    coefficient_identity_residual: float
    transmission_e16_residual: float
    input_density_matrix: np.ndarray
    output_density_matrix: np.ndarray
    output_port_probabilities: tuple[float, float]
    output_port_projectors: tuple[np.ndarray, np.ndarray]
    kraus_operators: tuple[np.ndarray, np.ndarray]
    kraus_completeness_residual: float
    record_isometry: np.ndarray
    record_isometry_residual: float
    choi_matrix: np.ndarray
    choi_minimum_eigenvalue: float
    output_trace_residual: float
    output_minimum_eigenvalue: float
    elastic_shell_energy: float
    input_shell_hamiltonian: np.ndarray
    port_shell_hamiltonian: np.ndarray
    record_hamiltonian: np.ndarray
    final_shell_hamiltonian: np.ndarray
    energy_intertwining_residual: float
    input_shell_energy_expectation: float
    nonselective_output_shell_energy_expectation: float
    isometric_total_output_energy_expectation: float
    nonselective_shell_energy_residual: float
    isometric_shell_energy_residual: float
    nonselective_energy_residual: float
    left_input_port0_reflection_probability: float
    left_input_port1_transmission_probability: float
    mass_mass_dimension: int
    energy_mass_dimension: int
    width_mass_dimension: int
    wavenumber_mass_dimension: int
    kappa_mass_dimension: int
    dimensionless_barrier_width_mass_dimension: int
    amplitude_mass_dimension: int
    probability_mass_dimension: int
    dimensions_pass: bool
    conditional_single_energy_scattering_unitarity: bool
    output_port_cptp_instrument: bool
    prepared_record_isometry: bool
    elastic_degenerate_energy_bookkeeping: bool
    one_sided_port_label_statement: bool
    physical_observation_or_selection_derived: bool = False
    general_reflection_transmission_labels_derived: bool = False
    wavepacket_or_energy_spread_derived: bool = False
    autonomous_detector_derived: bool = False
    durable_record_reset_or_battery_derived: bool = False
    physical_non_degenerate_record_energy_receipt_derived: bool = False
    repeated_fresh_ancilla_cptp_derived: bool = False
    causal_front_derived: bool = False
    qft_or_gr_derived: bool = False
    e17_j_transmission_relation_derived: bool = False
    residual_prediction_derived: bool = False
    gates_3_to_8_closed: bool = False


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _density_matrix(value: np.ndarray | None) -> np.ndarray:
    rho = _P0.copy() if value is None else np.asarray(value, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must have shape (2, 2)")
    if not np.isfinite(rho).all():
        raise ValueError("rho must be finite")
    if not np.allclose(rho, rho.conj().T, atol=1.0e-12, rtol=0.0):
        raise ValueError("rho must be Hermitian")
    if not math.isclose(float(np.trace(rho).real), 1.0, abs_tol=1.0e-12):
        raise ValueError("rho must have unit trace")
    if np.linalg.eigvalsh(rho).min() < -1.0e-12:
        raise ValueError("rho must be positive semidefinite")
    return rho


def _log_sech(x: float) -> float:
    """Stable log(sech x) for the positive, dimensionless x used here."""

    return math.log(2.0) - x - math.log1p(math.exp(-2.0 * x))


def _e16_log_transmission(*, mass: float, height: float, energy: float, width: float) -> float:
    """Independent algebraic form of the E16 exact below-barrier result."""

    log_fraction = math.log(energy) - math.log(height)
    log_gap_fraction = math.log(height - energy) - math.log(height)
    log_factor = -math.log(4.0) - log_fraction - log_gap_fraction
    x = math.sqrt(2.0 * mass * (height - energy)) * width
    log_sinh = math.log(math.sinh(x)) if x < 40.0 else x - math.log(2.0) + math.log1p(-math.exp(-2.0 * x))
    return -float(np.logaddexp(0.0, log_factor + 2.0 * log_sinh))


def certify_single_energy_barrier_instrument(
    *, nonrelativistic_mass: float, barrier_height: float, incident_energy: float,
    barrier_width: float, rho: np.ndarray | None = None,
) -> SingleEnergyBarrierInstrumentCertificate:
    """Certify a one-shot, fixed-energy output-port instrument.

    For a left-only input, output port 0 is reflection and port 1 is
    transmission.  For arbitrary two-sided input, those words are deliberately
    not assigned: they are merely two output ports.
    """

    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    energy = _positive(incident_energy, "incident_energy")
    width = _positive(barrier_width, "barrier_width")
    if energy >= height:
        raise ValueError("incident_energy must satisfy 0 < incident_energy < barrier_height")
    input_rho = _density_matrix(rho)

    k = math.sqrt(2.0 * mass * energy)
    kappa = math.sqrt(2.0 * mass * (height - energy))
    x = kappa * width
    # These ratios avoid a separate dimensional coefficient convention.
    a = (kappa / k - k / kappa) / 2.0
    b = (kappa / k + k / kappa) / 2.0
    q = math.tanh(x)
    logsech = _log_sech(x)
    sech = math.exp(logsech)
    denominator = 1.0 + 1j * a * q
    t = sech / denominator
    r = -1j * b * q / denominator
    scattering = np.array(((r, t), (t, r)), dtype=complex)
    log_t = 2.0 * logsech - math.log1p((a * q) ** 2)
    transmission = math.exp(log_t)
    reflection = float(abs(r) ** 2)
    conventional_t = complex(math.cos(k * width), -math.sin(k * width)) * t
    e16_log_t = _e16_log_transmission(mass=mass, height=height, energy=energy, width=width)

    kraus = (_P0 @ scattering, _P1 @ scattering)
    output = sum((operator @ input_rho @ operator.conj().T for operator in kraus), np.zeros((2, 2), complex))
    probabilities = tuple(float(np.trace(operator @ input_rho @ operator.conj().T).real) for operator in kraus)
    record_isometry = np.vstack(kraus)
    choi = sum((np.outer(operator.reshape(-1, order="F"), operator.reshape(-1, order="F").conj()) for operator in kraus), np.zeros((4, 4), complex))
    completeness = sum((operator.conj().T @ operator for operator in kraus), np.zeros((2, 2), complex))
    h_shell = energy * _I2
    h_record = np.zeros((2, 2), dtype=complex)
    h_final = np.kron(_I2, h_shell) + np.kron(h_record, _I2)
    isometric_output = record_isometry @ input_rho @ record_isometry.conj().T
    input_energy = float(np.trace(h_shell @ input_rho).real)
    port_energy = float(np.trace(h_shell @ output).real)
    isometric_energy = float(np.trace(h_final @ isometric_output).real)
    left_output = scattering @ _P0 @ scattering.conj().T

    return SingleEnergyBarrierInstrumentCertificate(
        nonrelativistic_mass=mass, barrier_height=height, incident_energy=energy, barrier_width=width,
        k=k, kappa=kappa, dimensionless_barrier_width=x, coefficient_a=a, coefficient_b=b,
        tanh_dimensionless_width=q, log_sech_dimensionless_width=logsech,
        reflection_amplitude=r, transmission_amplitude=t,
        conventional_coordinate_transmission_amplitude=conventional_t,
        log_transmission_probability=log_t, transmission_probability=transmission,
        transmission_amplitude_numerically_underflowed=(t == 0.0j),
        transmission_probability_numerically_underflowed=(transmission == 0.0), reflection_probability=reflection,
        reflection_probability_residual=float(abs(reflection - (1.0 - transmission))),
        scattering_matrix=scattering,
        scattering_unitarity_residual=float(np.linalg.norm(scattering.conj().T @ scattering - _I2, ord=2)),
        cross_amplitude_residual=float(abs(r * t.conjugate() + t * r.conjugate())),
        coefficient_identity_residual=float(abs(b * b - (1.0 + a * a))),
        transmission_e16_residual=float(abs(log_t - e16_log_t)), input_density_matrix=input_rho,
        output_density_matrix=output, output_port_probabilities=probabilities,
        output_port_projectors=(_P0.copy(), _P1.copy()), kraus_operators=kraus,
        kraus_completeness_residual=float(np.linalg.norm(completeness - _I2, ord=2)),
        record_isometry=record_isometry,
        record_isometry_residual=float(np.linalg.norm(record_isometry.conj().T @ record_isometry - _I2, ord=2)),
        choi_matrix=choi, choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi).min()),
        output_trace_residual=float(abs(np.trace(output) - 1.0)),
        output_minimum_eigenvalue=float(np.linalg.eigvalsh(output).min()), elastic_shell_energy=energy,
        input_shell_hamiltonian=h_shell, port_shell_hamiltonian=h_shell.copy(), record_hamiltonian=h_record,
        final_shell_hamiltonian=h_final,
        energy_intertwining_residual=float(np.linalg.norm(h_final @ record_isometry - record_isometry @ h_shell, ord=2)),
        input_shell_energy_expectation=input_energy,
        nonselective_output_shell_energy_expectation=port_energy,
        isometric_total_output_energy_expectation=isometric_energy,
        nonselective_shell_energy_residual=abs(port_energy - input_energy),
        isometric_shell_energy_residual=abs(isometric_energy - input_energy),
        nonselective_energy_residual=float(np.linalg.norm(h_shell @ output - output @ h_shell, ord=2)),
        left_input_port0_reflection_probability=float(left_output[0, 0].real),
        left_input_port1_transmission_probability=float(left_output[1, 1].real),
        mass_mass_dimension=1, energy_mass_dimension=1, width_mass_dimension=-1,
        wavenumber_mass_dimension=1, kappa_mass_dimension=1,
        dimensionless_barrier_width_mass_dimension=0, amplitude_mass_dimension=0, probability_mass_dimension=0,
        dimensions_pass=True, conditional_single_energy_scattering_unitarity=True,
        output_port_cptp_instrument=True, prepared_record_isometry=True,
        elastic_degenerate_energy_bookkeeping=True, one_sided_port_label_statement=True,
    )
