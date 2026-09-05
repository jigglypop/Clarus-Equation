"""examples/physics/record/theater_opening.py 의 검사다.

앞부분은 양자 개장 모형(보골리우보프 모드·급랭 밀도·늦은 압착 응력 상계)을,
뒷부분은 틱 접힘 규칙 스캔을 검사한다.

틱 접힘 쪽의 해석적 기대: ln a 당 3*Omega_Lambda 의 절대 생성률을 먼지 싱크에 넣으면
LCDM 배경을 정확히 재현하고(rho_D = Om_c a^-3 + Om_L), 무시할 만한 허블 틱 접힘은
DM/바리온 비를 평평하게 유지한다.
"""
from __future__ import annotations

import math

import pytest

from examples.physics.record import theater_opening as m
from examples.physics.record.theater_opening import (
    QuantumSeatSpecies,
    SuddenQuenchUVVerdict,
    bosonic_out_occupation,
    instantaneous_mode,
    integrate_late_squeezed_stress_envelope,
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


def test_late_squeezed_stress_separates_energy_from_hidden_phase() -> None:
    species = QuantumSeatSpecies(
        label="squeezed",
        degeneracy=1,
        mass_in=math.sqrt(3.0),
        mass_out=math.sqrt(7.0),
        duration=0.2,
    )
    audit = integrate_late_squeezed_stress_envelope(
        species,
        averaging_duration=100.0,
        momentum_max=30.0,
        intervals=400,
    )

    assert audit.created_energy_density == pytest.approx(
        0.008675684216652192,
        rel=2.0e-12,
    )
    assert audit.dephased_created_pressure == pytest.approx(
        0.001206593673204608,
        rel=2.0e-12,
    )
    assert audit.dephased_created_equation_of_state == pytest.approx(
        0.1390776384977983,
        rel=2.0e-12,
    )
    assert audit.static_out_created_anomalous_energy_density_coefficient == 0.0
    assert audit.anomalous_energy_cancels_exactly
    assert (
        audit.boxcar_anomalous_pressure_integrated_triangle_upper
        <= audit.one_over_time_anomalous_pressure_upper
        <= audit.instantaneous_anomalous_pressure_independent_phase_upper
    )
    assert (
        audit.boxcar_anomalous_field_variance_integrated_triangle_upper
        <= audit.one_over_time_anomalous_field_variance_upper
        <= audit.instantaneous_anomalous_field_variance_independent_phase_upper
    )
    assert audit.boxcar_pressure_lower > 0.0
    assert audit.dimensions_pass
    assert dict(audit.dimensionless_core_argument_mass_dimensions) == {
        "omega_times_averaging_duration": 0.0,
        "boxcar_sinc_argument": 0.0,
        "dephased_created_equation_of_state": 0.0,
    }
    assert not audit.phase_resolved_value_available
    assert not audit.global_time_supremum_computed
    assert not audit.full_renormalized_flrw_stress
    assert audit.initial_state_assumed_isotropic_number_diagonal
    assert audit.static_out_created_excess_scope
    assert audit.finite_momentum_window_only
    assert not audit.analytic_uv_tail_certificate
    assert audit.conditional_long_time_no_sustained_dark_energy_scope_declared
    assert not audit.long_time_no_sustained_dark_energy_numerically_certified
    assert not audit.physical_dark_matter_dark_energy_identification


def test_late_squeezed_one_over_time_bound_and_certificates_scale() -> None:
    species = QuantumSeatSpecies(
        label="dephasing",
        degeneracy=1,
        mass_in=math.sqrt(3.0),
        mass_out=math.sqrt(7.0),
        duration=0.2,
    )
    short = integrate_late_squeezed_stress_envelope(
        species,
        averaging_duration=10.0,
        momentum_max=30.0,
        intervals=400,
    )
    long = integrate_late_squeezed_stress_envelope(
        species,
        averaging_duration=100.0,
        momentum_max=30.0,
        intervals=400,
    )

    assert long.one_over_time_anomalous_pressure_coefficient == pytest.approx(
        short.one_over_time_anomalous_pressure_coefficient,
    )
    assert long.one_over_time_anomalous_pressure_upper == pytest.approx(
        short.one_over_time_anomalous_pressure_upper / 10.0,
    )
    assert (
        long.one_over_time_anomalous_field_variance_upper
        == pytest.approx(
            short.one_over_time_anomalous_field_variance_upper / 10.0
        )
    )
    assert short.sufficient_averaging_duration_for_no_acceleration == (
        pytest.approx(14.384712836797233)
    )
    assert not short.no_acceleration_certified_by_one_over_time_bound
    assert not short.nonnegative_pressure_certified_by_one_over_time_bound
    assert long.no_acceleration_certified_by_one_over_time_bound
    assert long.nonnegative_pressure_certified_by_one_over_time_bound


def test_late_squeezed_created_excess_bose_stimulation_is_not_full_state() -> None:
    vacuum = QuantumSeatSpecies(
        label="vacuum",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=0.3,
    )
    occupied = QuantumSeatSpecies(
        label="occupied",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=0.3,
        initial_mode_occupation=2.0,
    )
    vacuum_audit = integrate_late_squeezed_stress_envelope(
        vacuum,
        averaging_duration=2.0,
        momentum_max=30.0,
        intervals=400,
    )
    occupied_audit = integrate_late_squeezed_stress_envelope(
        occupied,
        averaging_duration=2.0,
        momentum_max=30.0,
        intervals=400,
    )

    for occupied_value, vacuum_value in (
        (
            occupied_audit.created_energy_density,
            vacuum_audit.created_energy_density,
        ),
        (
            occupied_audit.dephased_created_pressure,
            vacuum_audit.dephased_created_pressure,
        ),
        (
            occupied_audit.one_over_time_anomalous_pressure_coefficient,
            vacuum_audit.one_over_time_anomalous_pressure_coefficient,
        ),
        (
            occupied_audit.dephased_created_field_variance,
            vacuum_audit.dephased_created_field_variance,
        ),
    ):
        assert occupied_value == pytest.approx(5.0 * vacuum_value)
    assert (
        occupied_audit
        .constant_initial_occupation_used_only_as_created_excess_stimulation
    )
    assert not occupied_audit.full_initial_state_stress_computed


def test_late_squeezed_envelope_converges_and_zero_quench_is_exact() -> None:
    species = QuantumSeatSpecies(
        label="envelope-convergence",
        degeneracy=1,
        mass_in=math.sqrt(3.0),
        mass_out=math.sqrt(7.0),
        duration=0.2,
    )
    lower = integrate_late_squeezed_stress_envelope(
        species,
        averaging_duration=1.0,
        momentum_max=20.0,
        intervals=400,
    )
    upper = integrate_late_squeezed_stress_envelope(
        species,
        averaging_duration=1.0,
        momentum_max=30.0,
        intervals=600,
    )
    assert upper.one_over_time_anomalous_pressure_coefficient == pytest.approx(
        lower.one_over_time_anomalous_pressure_coefficient,
        rel=1.0e-4,
    )
    assert (
        upper.instantaneous_anomalous_field_variance_independent_phase_upper
        == pytest.approx(
            lower.instantaneous_anomalous_field_variance_independent_phase_upper,
            rel=3.0e-5,
        )
    )

    no_change = QuantumSeatSpecies(
        label="no-change",
        degeneracy=1,
        mass_in=2.0,
        mass_out=2.0,
        duration=0.2,
    )
    zero = integrate_late_squeezed_stress_envelope(
        no_change,
        averaging_duration=1.0,
        momentum_max=20.0,
        intervals=400,
    )
    assert zero.created_energy_density == 0.0
    assert zero.instantaneous_anomalous_pressure_independent_phase_upper == 0.0
    assert zero.exact_no_mass_quench
    assert zero.status == "PASS_ZERO_QUENCH_NO_SQUEEZED_EXCESS"

    underflowed = QuantumSeatSpecies(
        label="underflowed-nonzero-change",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=1.0e6,
    )
    unresolved = integrate_late_squeezed_stress_envelope(
        underflowed,
        averaging_duration=1.0,
        momentum_max=20.0,
        intervals=400,
    )
    assert unresolved.created_energy_density == 0.0
    assert not unresolved.exact_no_mass_quench
    assert not unresolved.numerical_created_excess_resolved
    assert not unresolved.no_acceleration_certified_by_one_over_time_bound
    assert unresolved.status == "FAIL_NUMERICAL_CREATED_EXCESS_UNRESOLVED"


@pytest.mark.parametrize("averaging_duration", (0.0, -1.0, math.inf))
def test_late_squeezed_envelope_requires_positive_finite_time(
    averaging_duration: float,
) -> None:
    species = QuantumSeatSpecies(
        label="bad-time",
        degeneracy=1,
        mass_in=1.0,
        mass_out=2.0,
        duration=0.2,
    )

    with pytest.raises(ValueError, match="averaging_duration"):
        integrate_late_squeezed_stress_envelope(
            species,
            averaging_duration=averaging_duration,
            momentum_max=20.0,
            intervals=400,
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


@pytest.mark.parametrize("degeneracy", (True, 0, 1.5))
def test_degeneracy_must_be_a_positive_integer(degeneracy: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        QuantumSeatSpecies(
            label="invalid-degeneracy",
            degeneracy=degeneracy,  # type: ignore[arg-type]
            mass_in=1.0,
            mass_out=2.0,
            duration=0.5,
        )

    with pytest.raises(ValueError, match="positive integer"):
        scalar_energy_transfer_rate(
            degeneracy=degeneracy,  # type: ignore[arg-type]
            mass_squared_rate=1.0,
            renormalized_field_squared=1.0,
        )


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


# ---------------------------------------------------------------- 틱 접힘 규칙 스캔
def test_r5_constant_creation_reproduces_lcdm_background():
    rule = dict(rate="R5", source="const", w=0.0, conserve="copy", gamma=3 * m.OM_L, x_star=0.0, f=0.0)
    r = m.test_single(rule)
    assert r["ok"], r
    assert r["dev_rho"] < 1e-3 and r["dev_E"] < 1e-3


def test_r5_wrong_rate_fails():
    rule = dict(rate="R5", source="const", w=0.0, conserve="copy", gamma=1.5 * m.OM_L, x_star=0.0, f=0.0)
    r = m.test_single(rule)
    assert not r["ok"], r


def test_negligible_fold_keeps_dm_ratio_flat():
    rule = dict(rate="R1", source="S1", w=0.0, conserve="copy", gamma=1e-8, x_star=0.0, f=0.0)
    r = m.test_dm(rule)
    assert r["ok"], r


def test_growth_lcdm_matter_era_normalisation():
    g = m.growth(None)
    # 물질 시대에 delta 는 ~a 로 자란다. Lambda 에 의한 억제로 D(a=1)/a_ini 는 약 0.78 * 1000 이다
    assert 600 < g["D0"] < 900, g
