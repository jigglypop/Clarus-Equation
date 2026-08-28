"""Exact no-go: a zero-dimensional endpoint record does not fix opening yield.

Work in units of a declared reference energy ``E_*``.  For the exact tanh
mass-squared quench, choose

    m_in / E_* = 1,  m_out / E_* = 3,  k / E_* = 0.

The two positive durations ``tau E_*=log(2)/pi`` and
``tau E_*=2 log(2)/pi`` have the same endpoint record, but give

    |beta_0|^2 = 4/21  and  |beta_0|^2 = 16/273.

The result follows from ``sinh(3x)=sinh(x)[3+4 sinh(x)^2]``.  Hence endpoint
masses do not select a unique production spectrum or quench history.

Initial-state freedom independently defeats a unique abundance.  For a
diagonal bosonic state,

    n_created(k) = [1 + 2 n_in(k)] |beta_k|^2.

Changing ``n_in`` from zero to three on any finite momentum band multiplies
the created occupation there by seven.  A smooth compactly supported band is
UV admissible and changes the integrated number and energy densities by a
strictly positive amount.  This counterexample deletes the implication

    zero-dimensional endpoint labels -> unique profile/state/abundance.

It does not forbid a future constructive theorem after a duration/profile
law and an initial-state selection axiom are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
    bosonic_out_occupation,
    smooth_tanh_mode,
)


@dataclass(frozen=True)
class TheaterOpeningInputNoGoAudit:
    mass_in_over_reference_energy: float
    mass_out_over_reference_energy: float
    momentum_over_reference_energy: float
    short_duration_times_reference_energy: float
    long_duration_times_reference_energy: float
    short_beta_squared: float
    long_beta_squared: float
    short_exact_beta_squared: float
    long_exact_beta_squared: float
    vacuum_created_occupation: float
    occupied_band_created_occupation: float
    occupied_band_stimulation_factor: float
    same_endpoint_record: bool
    unique_quench_spectrum_follows: bool
    compact_support_state_changes_integrated_abundance: bool
    unique_abundance_follows: bool
    status: str = "ZEROD_ENDPOINT_TO_UNIQUE_OPENING_ABUNDANCE_DISPROVED"
    claim_ceiling: str = "COMPLETE_EXACT_TANH_AND_INITIAL_STATE_COUNTEREXAMPLE"


def theater_opening_input_no_go() -> TheaterOpeningInputNoGoAudit:
    """Return exact duration and initial-state witnesses in ``E_*=1`` units."""

    mass_in = 1.0
    mass_out = 3.0
    momentum = 0.0
    short_duration = math.log(2.0) / math.pi
    long_duration = 2.0 * math.log(2.0) / math.pi

    short = QuantumSeatSpecies(
        label="same-endpoint-record",
        degeneracy=1,
        mass_in=mass_in,
        mass_out=mass_out,
        duration=short_duration,
    )
    long = QuantumSeatSpecies(
        label="same-endpoint-record",
        degeneracy=1,
        mass_in=mass_in,
        mass_out=mass_out,
        duration=long_duration,
    )
    short_mode = smooth_tanh_mode(short, momentum)
    long_mode = smooth_tanh_mode(long, momentum)

    initial_band_occupation = 3.0
    occupied_created = (
        bosonic_out_occupation(
            beta_squared=short_mode.beta_squared,
            initial_occupation=initial_band_occupation,
        )
        - initial_band_occupation
    )
    stimulation_factor = 1.0 + 2.0 * initial_band_occupation

    return TheaterOpeningInputNoGoAudit(
        mass_in_over_reference_energy=mass_in,
        mass_out_over_reference_energy=mass_out,
        momentum_over_reference_energy=momentum,
        short_duration_times_reference_energy=short_duration,
        long_duration_times_reference_energy=long_duration,
        short_beta_squared=short_mode.beta_squared,
        long_beta_squared=long_mode.beta_squared,
        short_exact_beta_squared=4.0 / 21.0,
        long_exact_beta_squared=16.0 / 273.0,
        vacuum_created_occupation=short_mode.created_occupation,
        occupied_band_created_occupation=occupied_created,
        occupied_band_stimulation_factor=stimulation_factor,
        same_endpoint_record=(
            short.degeneracy == long.degeneracy
            and short.mass_in == long.mass_in
            and short.mass_out == long.mass_out
        ),
        unique_quench_spectrum_follows=False,
        compact_support_state_changes_integrated_abundance=True,
        unique_abundance_follows=False,
    )
