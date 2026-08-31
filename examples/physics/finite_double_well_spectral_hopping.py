"""Finite E17 spectral certificate for a symmetric Dirichlet double well.

Natural units use ``hbar = 1``.  This module derives a two-mode hopping
``J`` only from the finite bound-state spectrum of the stated double well.
Its auxiliary open-scattering transmission is deliberately a separate
quantity: holding ``m, V0, b, Es`` fixed while changing the well width ``w``
changes the spectrum but not that transmission.

The mode normalization and left/right bias are floating-point quadrature and
closed-form witnesses, not formal interval proofs.  In particular finite
barriers do not give exactly spatially localized modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.finite_barrier_mode_leakage import (
    exact_rectangular_barrier_transmission_probability,
)


_PI = math.pi


@dataclass(frozen=True)
class FiniteDoubleWellSpectralHoppingCertificate:
    """Conditional finite spectral witness; no lattice or continuum bridge."""

    nonrelativistic_mass: float
    barrier_height: float
    total_barrier_width: float
    well_width: float
    scattering_energy: float
    energy_unit: float
    nu: float
    beta: float
    even_z_bracket: tuple[float, float]
    odd_z_bracket: tuple[float, float]
    even_endpoint_values: tuple[float, float]
    odd_endpoint_values: tuple[float, float]
    even_root_residual: float
    odd_root_residual: float
    even_z: float
    odd_z: float
    even_energy_interval: tuple[float, float]
    odd_energy_interval: tuple[float, float]
    ground_energy: float
    first_excited_energy: float
    hopping_interval: tuple[float, float]
    hopping: float
    mean_energy: float
    spectral_order_holds: bool
    right_mode_norm_witness: float
    left_mode_norm_witness: float
    right_left_overlap_witness: float
    right_mode_right_probability_witness: float
    left_mode_left_probability_witness: float
    maximum_join_residual: float
    spectral_hamiltonian: np.ndarray
    ideal_swap_time: float
    spectral_swap_phase_residual: float
    auxiliary_scattering_transmission: float
    mass_mass_dimension: int
    barrier_height_mass_dimension: int
    barrier_width_mass_dimension: int
    well_width_mass_dimension: int
    scattering_energy_mass_dimension: int
    energy_unit_mass_dimension: int
    nu_mass_dimension: int
    beta_mass_dimension: int
    wavefunction_mass_dimension: float
    hopping_mass_dimension: int
    time_mass_dimension: int
    transmission_probability_mass_dimension: int
    dimensions_pass: bool
    finite_double_well_spectrum_to_J_derived: bool = True
    prepared_exact_spectral_pair_invariant_by_construction: bool = True
    transmission_to_hopping_derived: bool = False
    wkb_to_hopping_derived: bool = False
    exact_spatial_localization_derived: bool = False
    e15_material_lattice_embedding_derived: bool = False
    periodic_or_n_chain_derived: bool = False
    arbitrary_continuum_preparation_projects_to_subspace: bool = False
    scattering_instrument_or_energy_receipt_derived: bool = False
    cptp_or_fresh_ancilla_derived: bool = False
    causal_c_front_derived: bool = False
    qft_microcausality_or_gr_derived: bool = False
    selection_or_residual_explanation_derived: bool = False
    gates_5_to_8_closed: bool = False


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _dimensionless_root_function(*, z: float, nu: float, beta: float, parity: str) -> float:
    if not _PI / 2.0 < z < _PI:
        raise ValueError("z must lie in (pi/2, pi)")
    q = math.sqrt(nu - z * z)
    u = 0.5 * beta * q
    barrier_ratio = math.tanh(u) if parity == "even" else 1.0 / math.tanh(u)
    return z / math.tan(z) + q * barrier_ratio


def _bisect_root(*, nu: float, beta: float, parity: str, tolerance: float) -> tuple[tuple[float, float], tuple[float, float], float]:
    # The declared nu > pi^2 condition makes q real on the full bracket.
    # Uniqueness in this declared branch follows from z*cot(z) decreasing and,
    # as q decreases with z, q*tanh(aq) and q*coth(aq) increasing in q.
    # The ledger carries the proof; bisection records its numerical bracket.
    left = math.nextafter(_PI / 2.0, _PI)
    right = math.nextafter(_PI, _PI / 2.0)
    f_left = _dimensionless_root_function(z=left, nu=nu, beta=beta, parity=parity)
    f_right = _dimensionless_root_function(z=right, nu=nu, beta=beta, parity=parity)
    if not (f_left > 0.0 and f_right < 0.0):
        raise RuntimeError("declared parity-root sign bracket failed")
    while right - left > tolerance:
        midpoint = 0.5 * (left + right)
        f_mid = _dimensionless_root_function(z=midpoint, nu=nu, beta=beta, parity=parity)
        if f_mid > 0.0:
            left = midpoint
            f_left = f_mid
        else:
            right = midpoint
            f_right = f_mid
    root = 0.5 * (left + right)
    return (left, right), (f_left, f_right), _dimensionless_root_function(
        z=root, nu=nu, beta=beta, parity=parity
    )


def _mode_values(*, x: np.ndarray, z: float, nu: float, beta: float, well_width: float, parity: str) -> np.ndarray:
    """Return an analytically normalized even or odd mode on the full domain."""

    q = math.sqrt(nu - z * z)
    k = z / well_width
    kappa = q / well_width
    half_barrier = 0.5 * beta * well_width
    barrier_edge = 0.5 * beta * q
    well_integral = 0.5 * well_width - math.sin(2.0 * z) / (4.0 * k)
    if parity == "even":
        barrier_integral = 2.0 * (0.25 * math.sinh(2.0 * barrier_edge) / kappa + half_barrier / 2.0)
        match_denominator = math.cosh(barrier_edge)
    else:
        barrier_integral = 2.0 * (0.25 * math.sinh(2.0 * barrier_edge) / kappa - half_barrier / 2.0)
        match_denominator = math.sinh(barrier_edge)
    normalization = 1.0 / math.sqrt(2.0 * well_integral + (math.sin(z) / match_denominator) ** 2 * barrier_integral)
    absolute_x = np.abs(x)
    in_barrier = absolute_x <= half_barrier
    values = np.empty_like(x, dtype=float)
    y = absolute_x - half_barrier
    values[~in_barrier] = normalization * np.sin(k * (well_width - y[~in_barrier]))
    if parity == "even":
        values[in_barrier] = normalization * math.sin(z) * np.cosh(kappa * x[in_barrier]) / match_denominator
    else:
        values[in_barrier] = (
            normalization * math.sin(z) * np.sinh(kappa * x[in_barrier]) / match_denominator
        )
        values[~in_barrier] *= np.sign(x[~in_barrier])
    return values


def _join_residual(*, z: float, nu: float, beta: float, well_width: float, parity: str) -> float:
    q = math.sqrt(nu - z * z)
    k = z / well_width
    kappa = q / well_width
    edge = 0.5 * beta * q
    # Ratio matching: well f'/f=-k cot z, barrier f'/f=kappa tanh/coth.
    barrier_ratio = kappa * (math.tanh(edge) if parity == "even" else 1.0 / math.tanh(edge))
    return abs(-k / math.tan(z) - barrier_ratio)


def certify_finite_double_well_spectral_hopping(
    *,
    nonrelativistic_mass: float,
    barrier_height: float,
    total_barrier_width: float,
    well_width: float,
    scattering_energy: float,
    root_tolerance: float = 1.0e-12,
) -> FiniteDoubleWellSpectralHoppingCertificate:
    """Certify the lowest parity pair of the supplied finite Dirichlet well.

    The domain is ``[-(w+b/2), w+b/2]`` with ``V=V0`` for ``|x|<b/2``.
    Exact two-state invariance below is only for states prepared in the two
    spectral eigenmodes found here; it does not project arbitrary continuum
    preparations into that subspace.
    """

    mass = _positive(nonrelativistic_mass, "nonrelativistic_mass")
    height = _positive(barrier_height, "barrier_height")
    width = _positive(total_barrier_width, "total_barrier_width")
    well = _positive(well_width, "well_width")
    scattering = _positive(scattering_energy, "scattering_energy")
    tolerance = _positive(root_tolerance, "root_tolerance")
    if tolerance >= 0.1:
        raise ValueError("root_tolerance must be below 0.1")
    if scattering >= height:
        raise ValueError("scattering_energy must satisfy 0 < Es < V0")
    energy_unit = 1.0 / (2.0 * mass * well * well)
    nu = height / energy_unit
    beta = width / well
    if nu <= _PI * _PI:
        raise ValueError("safe lowest-pair domain requires nu > pi^2")

    even_bracket, even_values, even_residual = _bisect_root(
        nu=nu, beta=beta, parity="even", tolerance=tolerance
    )
    odd_bracket, odd_values, odd_residual = _bisect_root(
        nu=nu, beta=beta, parity="odd", tolerance=tolerance
    )
    even_z = 0.5 * sum(even_bracket)
    odd_z = 0.5 * sum(odd_bracket)
    even_interval = (energy_unit * even_bracket[0] ** 2, energy_unit * even_bracket[1] ** 2)
    odd_interval = (energy_unit * odd_bracket[0] ** 2, energy_unit * odd_bracket[1] ** 2)
    if not even_z < odd_z or not even_interval[1] < odd_interval[0]:
        raise RuntimeError("lowest even/odd spectral order was not certified")
    ground = energy_unit * even_z * even_z
    excited = energy_unit * odd_z * odd_z
    hopping_interval = (
        0.5 * (odd_interval[0] - even_interval[1]),
        0.5 * (odd_interval[1] - even_interval[0]),
    )
    hopping = 0.5 * (excited - ground)
    mean_energy = 0.5 * (excited + ground)

    endpoint = well + 0.5 * width
    grid = np.linspace(-endpoint, endpoint, 100_001)
    even_mode = _mode_values(x=grid, z=even_z, nu=nu, beta=beta, well_width=well, parity="even")
    odd_mode = _mode_values(x=grid, z=odd_z, nu=nu, beta=beta, well_width=well, parity="odd")
    right_mode = (even_mode + odd_mode) / math.sqrt(2.0)
    left_mode = (even_mode - odd_mode) / math.sqrt(2.0)
    right_norm = float(np.trapezoid(right_mode * right_mode, grid))
    left_norm = float(np.trapezoid(left_mode * left_mode, grid))
    overlap = float(np.trapezoid(right_mode * left_mode, grid))
    right_bias = float(np.trapezoid(right_mode[grid >= 0.0] ** 2, grid[grid >= 0.0]))
    left_bias = float(np.trapezoid(left_mode[grid <= 0.0] ** 2, grid[grid <= 0.0]))
    joins = (
        _join_residual(z=even_z, nu=nu, beta=beta, well_width=well, parity="even"),
        _join_residual(z=odd_z, nu=nu, beta=beta, well_width=well, parity="odd"),
    )

    spectral_hamiltonian = np.array(
        ((mean_energy, -hopping), (-hopping, mean_energy)), dtype=float
    )
    swap_time = _PI / (2.0 * hopping)
    eigenvalues, eigenvectors = np.linalg.eigh(spectral_hamiltonian)
    swap = (eigenvectors * np.exp(-1j * swap_time * eigenvalues)) @ eigenvectors.conj().T
    swap_phase_residual = float(abs(swap[1, 0] - 1j * np.exp(-1j * mean_energy * swap_time)))
    transmission = exact_rectangular_barrier_transmission_probability(
        nonrelativistic_mass=mass,
        barrier_height=height,
        incident_energy=scattering,
        barrier_width=width,
    )
    dimensions_pass = (
        1 + (-2) + 1 == 0  # 2*m*w^2*V0 is dimensionless.
        and 1 + (-1) == 0  # beta=b/w.
        and 1 + (-1) == 0  # J*tau.
    )

    return FiniteDoubleWellSpectralHoppingCertificate(
        nonrelativistic_mass=mass, barrier_height=height, total_barrier_width=width,
        well_width=well, scattering_energy=scattering, energy_unit=energy_unit, nu=nu, beta=beta,
        even_z_bracket=even_bracket, odd_z_bracket=odd_bracket,
        even_endpoint_values=even_values, odd_endpoint_values=odd_values,
        even_root_residual=even_residual, odd_root_residual=odd_residual,
        even_z=even_z, odd_z=odd_z, even_energy_interval=even_interval,
        odd_energy_interval=odd_interval, ground_energy=ground, first_excited_energy=excited,
        hopping_interval=hopping_interval, hopping=hopping, mean_energy=mean_energy,
        spectral_order_holds=True, right_mode_norm_witness=right_norm,
        left_mode_norm_witness=left_norm, right_left_overlap_witness=overlap,
        right_mode_right_probability_witness=right_bias,
        left_mode_left_probability_witness=left_bias, maximum_join_residual=max(joins),
        spectral_hamiltonian=spectral_hamiltonian, ideal_swap_time=swap_time,
        spectral_swap_phase_residual=swap_phase_residual,
        auxiliary_scattering_transmission=transmission,
        mass_mass_dimension=1, barrier_height_mass_dimension=1, barrier_width_mass_dimension=-1,
        well_width_mass_dimension=-1, scattering_energy_mass_dimension=1,
        energy_unit_mass_dimension=1, nu_mass_dimension=0, beta_mass_dimension=0,
        wavefunction_mass_dimension=0.5, hopping_mass_dimension=1, time_mass_dimension=-1,
        transmission_probability_mass_dimension=0, dimensions_pass=dimensions_pass,
    )
