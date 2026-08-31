"""Finite repeated-domino obstruction and resource receipt.

This is a deliberately finite certificate, not a continuum domino theory.  At
each coordinate-labelled candidate cell let

``S_j = |h_j|^2 + |d_j|^2 + |b_j|^2``

and couple adjacent cells by

``g h[j + 1]^* h[j] d[j + 1]^* b[j + 1] + c.c.``.

The labels are *not* proved to be physical distances.  They are only a local
in-coordinate species-field candidate.  For an open chain (each cell has
degree at most two), the normalization used here gives

``|V_link| <= |g|/4 (S_j^2 + S_{j+1}^2)``.

Consequently ``lambda >= 2 |g|`` is a sufficient bound for
``lambda/4 sum_j S_j^2 + sum_links V_link``.  The one-excitation Hermitian
matrix uses projected nearest-neighbour rates ``J_j`` (angular-frequency,
hence mass, dimension), rather than treating the four-dimensional quartic
``g`` as a bare rate.  It makes the analytic-onset obstruction explicit: a
nonzero nearest-neighbour path has its first endpoint Taylor term at order
``N``.  It cannot provide an exactly delayed front on an open time interval.

The preparation stability check uses a unit-peak clock bump, equivalently
``||r||_infty <= 1``: the quadratic carrier minimum is then
``m_H^2 - mu_P^2``.  A differently normalized bump would require its own
correspondingly rescaled bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AutonomousRepeatedDominoObstructionCertificate:
    """Finite-chain audit receipt; false ceilings are intentional."""

    n_links: int
    couplings: tuple[float, ...]
    field_mass: float
    clock_scale: float
    prep_mass_squared: float
    carrier_quadratic_minimum_mass_squared: float
    battery_energy_per_cell: float
    exchange_coupling: float
    quartic_coupling: float
    quartic_lower_bound_coefficient: float
    hamiltonian_hermiticity_residual: float
    lower_order_endpoint_power_residual: float
    endpoint_order_n_value: complex
    expected_endpoint_order_n_value: float
    endpoint_taylor_coefficient: complex
    expected_endpoint_taylor_coefficient: complex
    small_time: float
    small_time_endpoint_amplitude: complex
    small_time_leading_term: complex
    small_time_remainder_magnitude: float
    all_success_initially_clean_battery_count: int
    all_success_initially_clean_record_count: int
    all_success_battery_energy: float
    dimensionless_core_arguments: tuple[tuple[str, str], ...]
    local_coordinate_species_field_candidate_by_construction: bool
    explicit_coordinate_time_switching_present: bool
    dimensions_closed: bool
    stability_bound_pass: bool
    carrier_prep_stability_pass: bool
    finite_hamiltonian_hermitian: bool
    analytic_coefficient_conditions_pass: bool
    finite_all_success_resource_receipt: bool
    species_index_is_physical_spatial_distance: bool
    physical_lattice_or_worldtube_completion: bool
    coupled_clock_global_monotonicity_one_pass: bool
    exact_delayed_front_derived: bool
    projected_link_rates_from_action_derived: bool
    iterated_fresh_ancilla_cptp_instrument_derived: bool
    continuum_qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    gr_source_stress_matching_derived: bool
    unbounded_front_from_finite_resources_derived: bool
    durable_records_derived: bool
    cross_dataset_parameter_fixing_derived: bool
    independent_holdout_prediction_derived: bool


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite_nonzero(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(f"{name} must be finite and nonzero")
    return value


def certify_autonomous_repeated_domino_obstruction(
    *,
    n_links: int,
    couplings: Sequence[float],
    field_mass: float,
    clock_scale: float,
    prep_mass_squared: float,
    battery_energy_per_cell: float,
    exchange_coupling: float,
    quartic_coupling: float,
    small_time: float = 1.0e-3,
) -> AutonomousRepeatedDominoObstructionCertificate:
    """Return the finite analytic-onset and all-success resource certificate.

    ``all_success_battery_energy`` is capacity for the branch where every
    initially clean cell records success.  It is not an expected branch
    energy; durable records and reset are not proved here.
    """

    if not isinstance(n_links, int) or isinstance(n_links, bool) or n_links < 1:
        raise ValueError("n_links must be an integer at least one")
    # Projected one-excitation rates J_j [mass], not the dimensionless 4D g.
    supplied_couplings = tuple(
        _finite_nonzero(value, f"couplings[{index}]")
        for index, value in enumerate(couplings)
    )
    if len(supplied_couplings) != n_links:
        raise ValueError("couplings must contain exactly n_links entries")
    field_mass = _positive(field_mass, "field_mass")
    clock_scale = _positive(clock_scale, "clock_scale")
    prep_mass_squared = _positive(prep_mass_squared, "prep_mass_squared")
    carrier_quadratic_minimum_mass_squared = field_mass**2 - prep_mass_squared
    if carrier_quadratic_minimum_mass_squared <= 0.0:
        raise ValueError(
            "carrier preparation stability requires field_mass**2 > prep_mass_squared"
        )
    battery_energy_per_cell = _positive(
        battery_energy_per_cell, "battery_energy_per_cell"
    )
    exchange_coupling = _finite_nonzero(exchange_coupling, "exchange_coupling")
    quartic_coupling = _positive(quartic_coupling, "quartic_coupling")
    small_time = _positive(small_time, "small_time")
    if quartic_coupling < 2.0 * abs(exchange_coupling):
        raise ValueError("quartic stability requires lambda >= 2 |g|")
    quartic_lower_bound_coefficient = (
        quartic_coupling / 4.0 - abs(exchange_coupling) / 2.0
    )

    hamiltonian = np.zeros((n_links + 1, n_links + 1), dtype=np.complex128)
    for index, coupling in enumerate(supplied_couplings):
        hamiltonian[index, index + 1] = coupling
        hamiltonian[index + 1, index] = coupling
    powers = [np.linalg.matrix_power(hamiltonian, power) for power in range(n_links + 1)]
    endpoint_values = [power[n_links, 0] for power in powers]
    lower_residual = max(abs(value) for value in endpoint_values[:-1])
    endpoint_value = endpoint_values[-1]
    product = math.prod(supplied_couplings)
    coefficient = ((-1j) ** n_links) * product / math.factorial(n_links)
    propagator = _antihermitian_propagator(-1j * hamiltonian * small_time)
    amplitude = propagator[n_links, 0]
    leading = coefficient * small_time**n_links

    return AutonomousRepeatedDominoObstructionCertificate(
        n_links=n_links,
        couplings=supplied_couplings,
        field_mass=field_mass,
        clock_scale=clock_scale,
        prep_mass_squared=prep_mass_squared,
        carrier_quadratic_minimum_mass_squared=carrier_quadratic_minimum_mass_squared,
        battery_energy_per_cell=battery_energy_per_cell,
        exchange_coupling=exchange_coupling,
        quartic_coupling=quartic_coupling,
        quartic_lower_bound_coefficient=quartic_lower_bound_coefficient,
        hamiltonian_hermiticity_residual=float(
            np.max(np.abs(hamiltonian - hamiltonian.conj().T))
        ),
        lower_order_endpoint_power_residual=float(lower_residual),
        endpoint_order_n_value=complex(endpoint_value),
        expected_endpoint_order_n_value=product,
        endpoint_taylor_coefficient=complex(coefficient),
        expected_endpoint_taylor_coefficient=complex(coefficient),
        small_time=small_time,
        small_time_endpoint_amplitude=complex(amplitude),
        small_time_leading_term=complex(leading),
        small_time_remainder_magnitude=float(abs(amplitude - leading)),
        all_success_initially_clean_battery_count=n_links,
        all_success_initially_clean_record_count=n_links,
        all_success_battery_energy=n_links * battery_energy_per_cell,
        dimensionless_core_arguments=(
            ("T / M_T", "dimensionless dynamical-clock argument"),
            ("mu_P^2 Delta_tau / omega", "dimensionless preparation area"),
            ("J_j Delta_tau", "dimensionless projected rate-time area; J_j has mass dimension"),
            ("(-i)^N prod(J_j) t^N / N!", "dimensionless endpoint amplitude; prod(J_j) has mass^N"),
        ),
        local_coordinate_species_field_candidate_by_construction=True,
        explicit_coordinate_time_switching_present=False,
        dimensions_closed=True,
        stability_bound_pass=quartic_lower_bound_coefficient >= 0.0,
        carrier_prep_stability_pass=carrier_quadratic_minimum_mass_squared > 0.0,
        finite_hamiltonian_hermitian=True,
        analytic_coefficient_conditions_pass=(
            lower_residual < 1.0e-12 and abs(endpoint_value - product) < 1.0e-12
        ),
        finite_all_success_resource_receipt=True,
        species_index_is_physical_spatial_distance=False,
        physical_lattice_or_worldtube_completion=False,
        coupled_clock_global_monotonicity_one_pass=False,
        exact_delayed_front_derived=False,
        projected_link_rates_from_action_derived=False,
        iterated_fresh_ancilla_cptp_instrument_derived=False,
        continuum_qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        gr_source_stress_matching_derived=False,
        unbounded_front_from_finite_resources_derived=False,
        durable_records_derived=False,
        cross_dataset_parameter_fixing_derived=False,
        independent_holdout_prediction_derived=False,
    )


def _antihermitian_propagator(antihermitian_generator: np.ndarray) -> np.ndarray:
    """Exponentiate an anti-Hermitian ``-i H t`` built from Hermitian ``H``."""

    eigenvalues, eigenvectors = np.linalg.eigh(1j * antihermitian_generator)
    return (eigenvectors * np.exp(-1j * eigenvalues)) @ eigenvectors.conj().T
