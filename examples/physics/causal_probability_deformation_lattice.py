"""E20 finite causal witness: a supplied scalar front and a local Born response.

This is deliberately not E19's Radon--Nikodym reweighting.  ``chi`` is a
dimensionless, supplied classical lattice field.  Its compact impulse travels
one lattice cell per time step (``dt = a/c``); only after that front reaches a
detector does a local Hamiltonian alter that detector's Born probability.
Nothing here identifies the effect with attraction or with gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_TOL = 2.0e-12


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


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class CausalProbabilityDeformationLatticeCertificate:
    """Numerical certificate for one finite, padded causal-lattice experiment."""

    lattice_spacing: float
    light_speed: float
    time_step: float
    detector_distance_cells: int
    time_steps: int
    grid_radius_cells: int
    source_amplitude: float
    coupling_energy: float
    hbar: float
    omega: float
    source_detector_chi: tuple[float, ...]
    control_detector_chi: tuple[float, ...]
    source_probabilities: tuple[float, ...]
    control_probabilities: tuple[float, ...]
    support_violation: float
    first_nonzero_detector_sample: int | None
    expected_first_detector_sample: int
    prearrival_probability_difference: float
    postarrival_probability_difference: float
    front_speed: float
    local_unitary_residual: float
    local_trace_residual: float
    local_choi_minimum_eigenvalue: float
    source_off_probability_difference: float
    coupling_off_probability_difference: float
    boundary_clearance_cells: int
    chi_dimensionless: bool
    source_q_dimensionless: bool
    continuum_source_s_length_power: int
    coupling_g_is_energy: bool
    dt_hamiltonian_over_hbar_dimensionless: bool
    dimensions_pass: bool
    rn_reweighting_used: bool
    finite_lattice_causal_front_witness: bool
    mass_to_q_derived: bool = False
    energy_current_or_backreaction_derived: bool = False
    probability_deformation_equals_attraction_derived: bool = False
    continuous_qft_microcausality_derived: bool = False
    gr_or_lensing_derived: bool = False
    repeated_measurement_or_physical_selection_derived: bool = False
    observational_holdout_derived: bool = False
    gates_5_to_8_closed: bool = False
    two_residuals_reduced: bool = False
    complexity_success: bool = False


def _unitary(chi: float, *, dt: float, hbar: float, omega: float, coupling: float) -> np.ndarray:
    """Exact exponential for H=hbar*omega*X/2 + coupling*chi*Z."""

    hamiltonian = np.array(
        ((coupling * chi, 0.5 * hbar * omega), (0.5 * hbar * omega, -coupling * chi)),
        dtype=complex,
    )
    norm = float(math.hypot(coupling * chi, 0.5 * hbar * omega))
    if norm == 0.0:
        return np.eye(2, dtype=complex)
    angle = dt * norm / hbar
    return math.cos(angle) * np.eye(2, dtype=complex) - 1j * math.sin(angle) * hamiltonian / norm


def _unitary_choi_minimum_eigenvalue(unitary: np.ndarray) -> float:
    """Choi positivity witness of rho -> U rho U dagger."""

    choi = np.zeros((4, 4), dtype=complex)
    for row in range(2):
        for column in range(2):
            basis = np.zeros((2, 2), dtype=complex)
            basis[row, column] = 1.0
            output = unitary @ basis @ unitary.conj().T
            choi += np.kron(basis, output)
    return float(np.linalg.eigvalsh(choi).min())


def certify_causal_probability_deformation_lattice(
    *, lattice_spacing: float = 0.1, light_speed: float = 1.0,
    detector_distance_cells: int = 3, time_steps: int = 6,
    grid_radius_cells: int = 6, source_amplitude: float = 0.8,
    coupling_energy: float = 0.7, omega: float = 1.0, hbar: float = 1.0,
) -> CausalProbabilityDeformationLatticeCertificate:
    """Evolve a compact source and compare local source/control Born responses.

    The recurrence is exactly the CFL-one update
    ``chi[j,n+1]=chi[j+1,n]+chi[j-1,n]-chi[j,n-1]+q delta[j,0] delta[n,0]``.
    Initial slices ``chi[-1]`` and ``chi[0]`` vanish.  A grid radius at least
    ``time_steps`` keeps this finite realization inside the infinite-lattice
    horizon for every returned slice.
    """

    a = _positive(lattice_spacing, "lattice_spacing")
    c = _positive(light_speed, "light_speed")
    distance = _nonnegative_integer(detector_distance_cells, "detector_distance_cells")
    steps = _nonnegative_integer(time_steps, "time_steps")
    radius = _nonnegative_integer(grid_radius_cells, "grid_radius_cells")
    q = _finite(source_amplitude, "source_amplitude")
    g = _finite(coupling_energy, "coupling_energy")
    angular_frequency = _positive(omega, "omega")
    planck = _positive(hbar, "hbar")
    if steps < distance + 1:
        raise ValueError("time_steps must include the first detector-arrival sample")
    if radius < steps:
        raise ValueError("grid_radius_cells must be at least time_steps to avoid boundary/reflection")
    if distance > radius:
        raise ValueError("detector_distance_cells must lie inside the padded grid")

    dt = a / c
    width = 2 * radius + 1
    origin = radius
    detector = origin + distance

    def lattice_history(amplitude: float) -> tuple[np.ndarray, ...]:
        previous = np.zeros(width, dtype=float)  # chi^-1
        current = np.zeros(width, dtype=float)   # chi^0
        history = [current.copy()]
        for n in range(steps):
            following = np.zeros(width, dtype=float)
            following[1:-1] = current[2:] + current[:-2] - previous[1:-1]
            if n == 0:
                following[origin] += amplitude
            previous, current = current, following
            history.append(current.copy())
        return tuple(history)

    source_history = lattice_history(q)
    control_history = lattice_history(0.0)
    source_chi = tuple(float(slice_[detector]) for slice_ in source_history)
    control_chi = tuple(float(slice_[detector]) for slice_ in control_history)
    expected_first = distance + 1
    nonzero = next((n for n, value in enumerate(source_chi) if abs(value) > _TOL), None)
    support_violation = max(
        (abs(value)
         for n, slice_ in enumerate(source_history)
         for j, value in enumerate(slice_)
         if abs(j - origin) > n - 1),
        default=0.0,
    )

    initial_rho = np.array(((1.0, 0.0), (0.0, 0.0)), dtype=complex)
    projector_zero = np.diag((1.0, 0.0)).astype(complex)

    def probabilities(field: tuple[float, ...], coupling: float) -> tuple[tuple[float, ...], np.ndarray]:
        rho = initial_rho.copy()
        values = [float(np.trace(projector_zero @ rho).real)]
        last = np.eye(2, dtype=complex)
        for n in range(1, steps + 1):
            last = _unitary(field[n], dt=dt, hbar=planck, omega=angular_frequency, coupling=coupling)
            rho = last @ rho @ last.conj().T
            values.append(float(np.trace(projector_zero @ rho).real))
        return tuple(values), last

    source_probabilities, arrival_unitary = probabilities(source_chi, g)
    control_probabilities, _ = probabilities(control_chi, g)
    source_off_probabilities, _ = probabilities(control_chi, g)
    coupling_off_probabilities, _ = probabilities(source_chi, 0.0)
    prearrival = max(abs(source_probabilities[n] - control_probabilities[n]) for n in range(expected_first))
    postarrival = max(abs(source_probabilities[n] - control_probabilities[n]) for n in range(expected_first, steps + 1))
    local_unitarity = float(np.linalg.norm(arrival_unitary.conj().T @ arrival_unitary - np.eye(2), ord=2))
    rho_after = arrival_unitary @ initial_rho @ arrival_unitary.conj().T
    trace_residual = abs(float(np.trace(rho_after).real) - 1.0)

    return CausalProbabilityDeformationLatticeCertificate(
        lattice_spacing=a, light_speed=c, time_step=dt,
        detector_distance_cells=distance, time_steps=steps, grid_radius_cells=radius,
        source_amplitude=q, coupling_energy=g, hbar=planck, omega=angular_frequency,
        source_detector_chi=source_chi, control_detector_chi=control_chi,
        source_probabilities=source_probabilities, control_probabilities=control_probabilities,
        support_violation=float(support_violation), first_nonzero_detector_sample=nonzero,
        expected_first_detector_sample=expected_first,
        prearrival_probability_difference=prearrival, postarrival_probability_difference=postarrival,
        front_speed=a / dt, local_unitary_residual=local_unitarity,
        local_trace_residual=trace_residual,
        local_choi_minimum_eigenvalue=_unitary_choi_minimum_eigenvalue(arrival_unitary),
        source_off_probability_difference=max(abs(x - y) for x, y in zip(source_off_probabilities, control_probabilities)),
        coupling_off_probability_difference=max(abs(x - y) for x, y in zip(coupling_off_probabilities, control_probabilities)),
        boundary_clearance_cells=radius - steps + 1,
        chi_dimensionless=True, source_q_dimensionless=True, continuum_source_s_length_power=-2,
        coupling_g_is_energy=True, dt_hamiltonian_over_hbar_dimensionless=True,
        dimensions_pass=True, rn_reweighting_used=False,
        finite_lattice_causal_front_witness=bool(
            support_violation <= _TOL and nonzero == expected_first and abs(a / dt - c) <= _TOL
        ),
    )
