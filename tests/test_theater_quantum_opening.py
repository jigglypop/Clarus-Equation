from __future__ import annotations

import math

import pytest

from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
    SuddenQuenchUVVerdict,
    bosonic_out_occupation,
    instantaneous_mode,
    integrate_quench_densities,
    multi_species_opening,
    scalar_energy_transfer_rate,
    smooth_tanh_mode,
    total_ward_residual,
)


def test_seat_weight_becomes_final_rest_mass_not_generic_mean_energy() -> None:
    species = QuantumSeatSpecies.from_seat_weight(
        label="heavy",
        degeneracy=3,
        mass_in=1.0,
        reference_energy=2.0,
        relative_rest_mass=4.0,
        duration=0.5,
    )

    assert species.mass_out == 8.0
    assert species.degeneracy == 3


def test_instantaneous_mode_matches_exact_oscillator_example() -> None:
    species = QuantumSeatSpecies(
        label="example",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=1.0,
    )
    mode = instantaneous_mode(species, momentum=0.0)

    assert mode.alpha_squared == pytest.approx(1.125)
    assert mode.beta_squared == pytest.approx(0.125)
    assert mode.normalization_residual == pytest.approx(0.0, abs=1.0e-15)


def test_smooth_tanh_mode_has_exact_bogoliubov_normalization() -> None:
    species = QuantumSeatSpecies(
        label="smooth",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=1.0,
    )
    mode = smooth_tanh_mode(species, momentum=0.0)

    assert mode.beta_squared == pytest.approx(0.001712735436006, rel=1.0e-12)
    assert mode.alpha_squared == pytest.approx(
        1.001712735436006,
        rel=1.0e-12,
    )
    assert mode.normalization_residual == pytest.approx(0.0, abs=2.0e-15)


def test_smooth_quench_tends_to_sudden_mode_for_short_duration() -> None:
    species = QuantumSeatSpecies(
        label="fast",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=1.0e-6,
    )
    smooth = smooth_tanh_mode(species, momentum=0.7)
    sudden = instantaneous_mode(species, momentum=0.7)

    assert smooth.beta_squared == pytest.approx(
        sudden.beta_squared,
        rel=1.0e-10,
    )


def test_bosonic_initial_population_stimulates_pair_creation() -> None:
    assert bosonic_out_occupation(
        beta_squared=0.2,
        initial_occupation=0.0,
    ) == pytest.approx(0.2)
    assert bosonic_out_occupation(
        beta_squared=0.2,
        initial_occupation=3.0,
    ) == pytest.approx(4.4)


def test_sudden_quench_is_a_finite_number_but_divergent_energy_control() -> None:
    verdict = SuddenQuenchUVVerdict()

    assert verdict.beta_squared_power == -4
    assert verdict.number_density_uv_convergent
    assert not verdict.energy_density_uv_convergent


def test_instantaneous_density_requires_cutoff_and_exposes_uv_growth() -> None:
    species = QuantumSeatSpecies(
        label="sudden",
        degeneracy=1,
        mass_in=1.0,
        mass_out=3.0,
        duration=0.2,
    )

    with pytest.raises(ValueError, match="explicit UV cutoff"):
        integrate_quench_densities(species, protocol="instantaneous")

    lower = integrate_quench_densities(
        species,
        protocol="instantaneous",
        momentum_max=20.0,
        intervals=1200,
    )
    upper = integrate_quench_densities(
        species,
        protocol="instantaneous",
        momentum_max=80.0,
        intervals=2400,
    )

    assert upper.number_density > lower.number_density
    assert upper.excess_energy_density > lower.excess_energy_density
    assert upper.ultraviolet_status.startswith("ENERGY_LOG_DIVERGENT")


def test_smooth_density_is_finite_positive_and_non_vacuum_like() -> None:
    species = QuantumSeatSpecies(
        label="finite",
        degeneracy=2,
        mass_in=1.0,
        mass_out=3.0,
        duration=0.4,
    )
    audit = integrate_quench_densities(
        species,
        momentum_max=30.0,
        intervals=1600,
    )

    assert audit.number_density > 0.0
    assert audit.excess_energy_density > 0.0
    assert 0.0 <= audit.equation_of_state <= 1.0 / 3.0
    assert audit.mean_energy_per_created_quantum >= species.mass_out
    assert audit.ultraviolet_status == "FINITE_FOR_POSITIVE_DURATION"
    assert audit.maximum_bogoliubov_residual < 1.0e-12


def test_smooth_density_converges_when_momentum_window_is_extended() -> None:
    species = QuantumSeatSpecies(
        label="convergence",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=0.5,
    )
    lower = integrate_quench_densities(
        species,
        momentum_max=20.0,
        intervals=1600,
    )
    upper = integrate_quench_densities(
        species,
        momentum_max=30.0,
        intervals=2400,
    )

    assert upper.number_density == pytest.approx(
        lower.number_density,
        rel=2.0e-8,
    )
    assert upper.excess_energy_density == pytest.approx(
        lower.excess_energy_density,
        rel=2.0e-8,
    )


def test_multiple_material_weights_produce_normalized_distinct_fractions() -> None:
    light = QuantumSeatSpecies.from_seat_weight(
        label="light",
        degeneracy=2,
        mass_in=4.0,
        reference_energy=1.0,
        relative_rest_mass=1.0,
        duration=0.3,
    )
    heavy = QuantumSeatSpecies.from_seat_weight(
        label="heavy",
        degeneracy=5,
        mass_in=4.0,
        reference_energy=1.0,
        relative_rest_mass=2.0,
        duration=0.3,
    )
    fractions = multi_species_opening((light, heavy), intervals=1000)

    assert math.fsum(item.energy_fraction for item in fractions) == pytest.approx(1.0)
    assert fractions[0].energy_fraction != fractions[1].energy_fraction
    assert fractions[0].mean_energy_over_rest_mass > 1.0
    assert fractions[1].mean_energy_over_rest_mass > 1.0


def test_dynamic_clock_closes_energy_transfer_but_external_quench_does_not() -> None:
    first = scalar_energy_transfer_rate(
        degeneracy=2,
        mass_squared_rate=3.0,
        renormalized_field_squared=0.5,
    )
    second = scalar_energy_transfer_rate(
        degeneracy=1,
        mass_squared_rate=-1.0,
        renormalized_field_squared=0.25,
    )

    assert total_ward_residual(
        scalar_transfer_rates=(first, second),
        clock_transfer_rate=-(first + second),
    ) == pytest.approx(0.0)
    assert total_ward_residual(
        scalar_transfer_rates=(first, second),
        clock_transfer_rate=0.0,
    ) != 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mass_in", 0.0, "masses must be positive"),
        ("mass_out", -1.0, "masses must be positive"),
        ("duration", 0.0, "duration must be positive"),
        ("initial_mode_occupation", -0.1, "occupation must be non-negative"),
    ),
)
def test_quantum_species_rejects_invalid_opening_domain(
    field: str,
    value: float,
    message: str,
) -> None:
    values: dict[str, object] = {
        "label": "bad",
        "degeneracy": 1,
        "mass_in": 1.0,
        "mass_out": 2.0,
        "duration": 0.5,
        "initial_mode_occupation": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        QuantumSeatSpecies(**values)
