"""Focused checks for the analytic smooth-quench ultraviolet tail bound."""

from __future__ import annotations

import math

import pytest

from examples.physics.kinetic_dark_sector_quench_tail_bound import (
    smooth_quench_created_occupation_tail_upper,
    smooth_quench_present_tail_certificate,
)
from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
    smooth_tanh_mode,
)


def _species(
    *,
    mass_in: float = 1.0,
    mass_out: float = 3.0,
    initial_occupation: float = 0.0,
    degeneracy: int = 2,
) -> QuantumSeatSpecies:
    return QuantumSeatSpecies(
        label="tail-test",
        degeneracy=degeneracy,
        mass_in=mass_in,
        mass_out=mass_out,
        duration=0.2,
        initial_mode_occupation=initial_occupation,
    )


def test_pointwise_bound_dominates_exact_created_occupation() -> None:
    species = _species()
    start = 10.0
    for momentum in (10.0, 12.0, 20.0, 50.0):
        exact = smooth_tanh_mode(species, momentum).created_occupation
        upper = smooth_quench_created_occupation_tail_upper(
            species,
            momentum=momentum,
            momentum_start=start,
        )
        assert exact <= upper


def test_integrated_tail_bound_dominates_direct_finite_tail_segment() -> None:
    species = _species()
    start, stop, intervals = 10.0, 80.0, 4000
    scale_factor = 0.1
    step = (stop - start) / intervals
    number_terms = []
    energy_terms = []
    for index in range(intervals + 1):
        momentum = start + index * step
        weight = (
            1.0
            if index in (0, intervals)
            else (4.0 if index % 2 else 2.0)
        )
        occupation = smooth_tanh_mode(species, momentum).created_occupation
        radial = momentum * momentum * occupation
        number_terms.append(weight * radial)
        energy_terms.append(
            weight
            * radial
            * math.hypot(species.mass_out, scale_factor * momentum)
        )
    prefactor = (
        species.degeneracy
        / (2.0 * math.pi * math.pi)
        * scale_factor**3
        * step
        / 3.0
    )
    number_segment = prefactor * math.fsum(number_terms)
    energy_segment = prefactor * math.fsum(energy_terms)
    certificate = smooth_quench_present_tail_certificate(
        species,
        momentum_start=start,
        scale_factor_at_production=scale_factor,
        critical_density_today=108.0,
    )
    assert number_segment <= certificate.present_number_density_upper
    assert energy_segment <= certificate.present_energy_density_upper
    assert certificate.present_pressure_upper >= 0.0
    assert certificate.omega_produced_upper >= (
        certificate.present_energy_density_upper / 108.0
    )
    assert certificate.numerical_status.endswith("NOT_INTERVAL_CERTIFIED")


def test_tail_bound_tracks_degeneracy_stimulation_and_redshift_volume() -> None:
    common = dict(
        momentum_start=10.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    base = smooth_quench_present_tail_certificate(_species(degeneracy=1), **common)
    doubled = smooth_quench_present_tail_certificate(
        _species(degeneracy=2),
        **common,
    )
    stimulated = smooth_quench_present_tail_certificate(
        _species(degeneracy=1, initial_occupation=2.0),
        **common,
    )
    assert doubled.present_number_density_upper / base.present_number_density_upper == (
        pytest.approx(2.0, rel=3.0e-15)
    )
    assert (
        stimulated.present_number_density_upper
        / base.present_number_density_upper
        == pytest.approx(5.0, rel=3.0e-15)
    )
    half_scale = smooth_quench_present_tail_certificate(
        _species(degeneracy=1),
        momentum_start=10.0,
        scale_factor_at_production=0.05,
        critical_density_today=108.0,
    )
    assert half_scale.present_number_density_upper / base.present_number_density_upper == (
        pytest.approx(1.0 / 8.0, rel=3.0e-15)
    )


def test_later_tail_start_gives_a_stronger_bound() -> None:
    lower = smooth_quench_present_tail_certificate(
        _species(),
        momentum_start=8.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    upper = smooth_quench_present_tail_certificate(
        _species(),
        momentum_start=12.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    assert upper.present_number_density_upper < lower.present_number_density_upper
    assert upper.present_energy_density_upper < lower.present_energy_density_upper


def test_equal_masses_have_an_exact_zero_created_tail() -> None:
    species = _species(
        mass_in=2.0,
        mass_out=2.0,
        initial_occupation=4.0,
    )
    certificate = smooth_quench_present_tail_certificate(
        species,
        momentum_start=10.0,
        scale_factor_at_production=0.1,
        critical_density_today=108.0,
    )
    assert certificate.log_occupation_coefficient == -math.inf
    assert certificate.present_number_density_upper == 0.0
    assert certificate.present_energy_density_upper == 0.0
    assert certificate.omega_produced_upper == 0.0
    assert smooth_quench_created_occupation_tail_upper(
        species,
        momentum=10.0,
        momentum_start=10.0,
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(momentum_start=0.0, scale_factor_at_production=0.1, critical_density_today=1.0),
        dict(momentum_start=1.0, scale_factor_at_production=1.1, critical_density_today=1.0),
        dict(momentum_start=1.0, scale_factor_at_production=0.1, critical_density_today=math.nan),
    ],
)
def test_tail_certificate_rejects_invalid_domain(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        smooth_quench_present_tail_certificate(_species(), **kwargs)


def test_pointwise_bound_rejects_momentum_below_tail_start() -> None:
    with pytest.raises(ValueError):
        smooth_quench_created_occupation_tail_upper(
            _species(),
            momentum=9.0,
            momentum_start=10.0,
        )
