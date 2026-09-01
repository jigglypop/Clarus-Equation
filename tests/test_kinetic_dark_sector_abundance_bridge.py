"""Focused tests for the conditional quench-to-abundance forward map."""

from __future__ import annotations

import math

import pytest

from examples.physics.kinetic_dark_sector_abundance_bridge import (
    EntropyRedshiftContract,
    NaturalUnitCosmology,
    audit_fixed_basis_bilinear_quench_cosmology,
    smooth_quench_collisionless_abundance,
)
from examples.physics.theater_quantum_opening import (
    QuantumSeatSpecies,
    integrate_quench_densities,
)


def _species(
    *,
    mass_in: float = 1.0,
    mass_out: float = 3.0,
    degeneracy: int = 2,
    initial_occupation: float = 0.0,
) -> QuantumSeatSpecies:
    return QuantumSeatSpecies(
        label="created-dark-scalar",
        degeneracy=degeneracy,
        mass_in=mass_in,
        mass_out=mass_out,
        duration=0.2,
        initial_mode_occupation=initial_occupation,
    )


def _entropy(*, growth: float = 1.0) -> EntropyRedshiftContract:
    return EntropyRedshiftContract(
        temperature_at_production=10.0,
        temperature_today=1.0,
        entropy_dof_at_production=4.0,
        entropy_dof_today=4.0,
        comoving_entropy_growth=growth,
    )


def _cosmology() -> NaturalUnitCosmology:
    return NaturalUnitCosmology(hubble_today=2.0, reduced_planck_mass=3.0)


def _fixed_basis_audit(**overrides: object):
    parameters: dict[str, object] = dict(
        initial_system_mass_ev=2.0,
        initial_environment_mass_ev=2.0,
        initial_bilinear_coupling_ev2=1.0,
        initial_vacuum_energy_density_ev4=0.01,
        final_system_mass_ev=3.0,
        final_environment_mass_ev=3.0,
        final_bilinear_coupling_ev2=2.0,
        final_vacuum_energy_density_ev4=0.02,
        quench_duration_ev_minus1=0.2,
        initial_mode_occupations=(0.0, 0.0),
        entropy=EntropyRedshiftContract(
            temperature_at_production=100.0,
            temperature_today=1.0,
            entropy_dof_at_production=4.0,
            entropy_dof_today=4.0,
        ),
        cosmology=NaturalUnitCosmology(
            hubble_today=1.0e-3,
            reduced_planck_mass=100.0,
        ),
        hubble_at_production_ev=1.0e-2,
        comoving_wavenumber_ev=0.1,
        momentum_max_ev=30.0,
        intervals=400,
        validity_limit=0.1,
    )
    parameters.update(overrides)
    return audit_fixed_basis_bilinear_quench_cosmology(**parameters)


def test_entropy_contract_fixes_dimensionless_production_scale_factor() -> None:
    assert _entropy().scale_factor_at_production == pytest.approx(0.1)
    assert _entropy(growth=8.0).scale_factor_at_production == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature_at_production", 0.0),
        ("temperature_today", math.nan),
        ("entropy_dof_at_production", -1.0),
        ("entropy_dof_today", True),
        ("comoving_entropy_growth", 0.5),
    ],
)
def test_entropy_contract_rejects_invalid_inputs(field: str, value: object) -> None:
    values = dict(
        temperature_at_production=10.0,
        temperature_today=1.0,
        entropy_dof_at_production=4.0,
        entropy_dof_today=4.0,
        comoving_entropy_growth=1.0,
    )
    values[field] = value
    with pytest.raises(ValueError):
        EntropyRedshiftContract(**values)


def test_entropy_contract_rejects_a_future_production_surface() -> None:
    with pytest.raises(ValueError):
        EntropyRedshiftContract(
            temperature_at_production=1.0,
            temperature_today=2.0,
            entropy_dof_at_production=1.0,
            entropy_dof_today=1.0,
        )


def test_natural_unit_critical_density_is_exact() -> None:
    assert _cosmology().critical_density_today == 108.0
    with pytest.raises(ValueError):
        NaturalUnitCosmology(hubble_today=0.0, reduced_planck_mass=3.0)


def test_collisionless_number_dilution_and_absolute_omega() -> None:
    certificate = smooth_quench_collisionless_abundance(
        _species(),
        entropy=_entropy(),
        cosmology=_cosmology(),
        momentum_max=30.0,
        intervals=600,
    )
    expected_number = (
        certificate.scale_factor_at_production**3
        * certificate.production_number_density
    )
    assert certificate.present_number_density == pytest.approx(
        expected_number,
        rel=2.0e-15,
    )
    assert certificate.present_number_density_from_dilution == pytest.approx(
        expected_number,
        rel=2.0e-15,
    )
    assert abs(certificate.number_dilution_residual) <= 2.0e-15 * expected_number
    assert certificate.omega_produced_today == pytest.approx(
        certificate.present_energy_density / 108.0,
        rel=2.0e-15,
    )
    assert certificate.omitted_number_density_upper > 0.0
    assert certificate.omitted_energy_density_upper > 0.0
    assert certificate.omitted_omega_upper >= (
        certificate.omitted_energy_density_upper / 108.0
    )


def test_redshifted_energy_obeys_rest_and_production_bounds() -> None:
    species = _species()
    production = integrate_quench_densities(
        species,
        momentum_max=30.0,
        intervals=600,
    )
    certificate = smooth_quench_collisionless_abundance(
        species,
        entropy=_entropy(),
        cosmology=_cosmology(),
        momentum_max=30.0,
        intervals=600,
    )
    a_star = certificate.scale_factor_at_production
    assert certificate.present_rest_density <= certificate.present_energy_density
    production_redshift_upper = (
        a_star**3 * production.excess_energy_density
    )
    assert certificate.present_energy_density <= (
        production_redshift_upper * (1.0 + 2.0e-15)
    )
    assert 0.0 <= certificate.present_equation_of_state <= 1.0 / 3.0
    assert certificate.cold_bound_residual <= (
        2.0e-14 * certificate.relative_kinetic_energy_upper_bound
    )


def test_a_star_one_recovers_post_quench_energy_and_pressure() -> None:
    species = _species()
    entropy = EntropyRedshiftContract(
        temperature_at_production=1.0,
        temperature_today=1.0,
        entropy_dof_at_production=1.0,
        entropy_dof_today=1.0,
    )
    production = integrate_quench_densities(
        species,
        momentum_max=30.0,
        intervals=600,
    )
    certificate = smooth_quench_collisionless_abundance(
        species,
        entropy=entropy,
        cosmology=_cosmology(),
        momentum_max=30.0,
        intervals=600,
    )
    assert certificate.present_number_density == pytest.approx(
        production.number_density,
        rel=2.0e-15,
    )
    assert certificate.present_energy_density == pytest.approx(
        production.excess_energy_density,
        rel=2.0e-15,
    )
    assert certificate.present_pressure == pytest.approx(
        production.dephased_pressure,
        rel=2.0e-15,
    )


def test_entropy_growth_dilutes_created_comoving_number() -> None:
    common = dict(
        species=_species(),
        cosmology=_cosmology(),
        momentum_max=30.0,
        intervals=600,
    )
    adiabatic = smooth_quench_collisionless_abundance(
        entropy=_entropy(growth=1.0),
        **common,
    )
    diluted = smooth_quench_collisionless_abundance(
        entropy=_entropy(growth=8.0),
        **common,
    )
    assert diluted.present_number_density / adiabatic.present_number_density == (
        pytest.approx(1.0 / 8.0, rel=2.0e-15)
    )


def test_no_mass_change_creates_no_excess_abundance() -> None:
    certificate = smooth_quench_collisionless_abundance(
        _species(mass_in=2.0, mass_out=2.0, initial_occupation=3.0),
        entropy=_entropy(),
        cosmology=_cosmology(),
        momentum_max=20.0,
        intervals=400,
    )
    assert certificate.production_number_density == 0.0
    assert certificate.present_number_density == 0.0
    assert certificate.present_energy_density == 0.0
    assert certificate.present_pressure == 0.0
    assert certificate.omega_produced_today == 0.0
    assert certificate.omitted_number_density_upper == 0.0
    assert certificate.omitted_energy_density_upper == 0.0
    assert certificate.omitted_omega_upper == 0.0
    assert certificate.relative_kinetic_energy_upper_bound == 0.0


def test_created_excess_is_linear_in_degeneracy_and_bose_stimulation() -> None:
    common = dict(
        entropy=_entropy(),
        cosmology=_cosmology(),
        momentum_max=20.0,
        intervals=400,
    )
    base = smooth_quench_collisionless_abundance(
        _species(degeneracy=1),
        **common,
    )
    doubled = smooth_quench_collisionless_abundance(
        _species(degeneracy=2),
        **common,
    )
    stimulated = smooth_quench_collisionless_abundance(
        _species(degeneracy=1, initial_occupation=2.0),
        **common,
    )
    assert doubled.present_energy_density / base.present_energy_density == (
        pytest.approx(2.0, rel=2.0e-15)
    )
    assert stimulated.present_energy_density / base.present_energy_density == (
        pytest.approx(5.0, rel=2.0e-15)
    )


def test_larger_momentum_window_is_a_convergence_check_not_a_tail_proof() -> None:
    common = dict(
        species=_species(),
        entropy=_entropy(),
        cosmology=_cosmology(),
        intervals=800,
    )
    lower = smooth_quench_collisionless_abundance(
        momentum_max=20.0,
        **common,
    )
    upper = smooth_quench_collisionless_abundance(
        momentum_max=30.0,
        **common,
    )
    assert upper.omega_produced_today == pytest.approx(
        lower.omega_produced_today,
        rel=2.0e-8,
    )
    assert upper.omitted_omega_upper < lower.omitted_omega_upper
    assert upper.quadrature_status.endswith("NOT_UV_TAIL_CERTIFICATE")
    assert upper.ultraviolet_status == (
        "ANALYTIC_EXPONENTIAL_OMITTED_TAIL_BOUND_ATTACHED"
    )
    assert upper.tail_numerical_status.endswith("NOT_INTERVAL_CERTIFIED")


def test_abundance_bridge_is_typed_and_keeps_tail_status_explicit() -> None:
    certificate = smooth_quench_collisionless_abundance(
        _species(),
        entropy=_entropy(),
        cosmology=_cosmology(),
        momentum_max=20.0,
        intervals=401,
    )
    assert certificate.intervals == 402
    assert certificate.quadrature_status == (
        "FINITE_WINDOW_SIMPSON_NOT_UV_TAIL_CERTIFICATE"
    )
    assert certificate.role.endswith("NOT_ABUNDANCE_PREDICTION")
    assert certificate.production_approximation.startswith("ASYMPTOTIC_MINKOWSKI")
    with pytest.raises(ValueError):
        smooth_quench_collisionless_abundance(
            object(),
            entropy=_entropy(),
            cosmology=_cosmology(),
        )
    for invalid_intervals in (True, 400.5):
        with pytest.raises(ValueError):
            smooth_quench_collisionless_abundance(
                _species(),
                entropy=_entropy(),
                cosmology=_cosmology(),
                intervals=invalid_intervals,
            )
    for invalid_momentum_max in (math.nan, "20"):
        with pytest.raises(ValueError):
            smooth_quench_collisionless_abundance(
                _species(),
                entropy=_entropy(),
                cosmology=_cosmology(),
                momentum_max=invalid_momentum_max,
            )


def test_fixed_basis_bilinear_quench_closes_created_density_to_wkb_input() -> None:
    audit = _fixed_basis_audit()

    assert audit.matrix_commutator_frobenius_ev4 == 0.0
    assert audit.matrix_commutator_relative_residual == 0.0
    assert audit.fixed_basis_off_diagonal_relative_residual < 2.0e-16
    assert audit.initial_normal_mode_gap_fraction > 0.0
    assert audit.final_normal_mode_gap_fraction > 0.0
    assert audit.fixed_basis_mode_mass_squared_at_initial_ev2 == pytest.approx(
        (3.0, 5.0)
    )
    assert audit.fixed_basis_mode_mass_squared_at_final_ev2 == pytest.approx(
        (7.0, 11.0)
    )
    assert tuple(species.mass_in for species in audit.quench_species) == (
        pytest.approx(math.sqrt(3.0)),
        pytest.approx(math.sqrt(5.0)),
    )
    assert tuple(species.mass_out for species in audit.quench_species) == (
        pytest.approx(math.sqrt(7.0)),
        pytest.approx(math.sqrt(11.0)),
    )
    assert all(
        density > 0.0 for density in audit.present_created_energy_densities_ev4
    )
    assert audit.perturbation_audit.comoving_mode_density_constants_ev4 == (
        audit.present_created_rest_density_constants_ev4
    )
    assert audit.omega_created_today == pytest.approx(
        audit.present_created_energy_density_total_ev4
        / audit.perturbation_audit.reduced_planck_mass_ev**2
        / (3.0 * audit.perturbation_audit.hubble_ev**2)
    )
    assert audit.omega_created_energy_plus_vacuum_today == pytest.approx(
        audit.omega_created_today + audit.omega_vacuum_today
    )
    assert audit.omega_dust_input_plus_vacuum_today == pytest.approx(
        audit.omega_dust_input_today + audit.omega_vacuum_today
    )
    assert audit.omega_created_energy_plus_vacuum_today >= (
        audit.omega_dust_input_plus_vacuum_today
    )
    assert audit.local_quench_pass
    assert audit.background_cold_pass
    assert audit.perturbation_audit.perturbation_discriminant_derived
    assert audit.dimensions_pass
    assert not audit.failed_gates
    assert audit.status == "PASS_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB"
    assert audit.conditional_created_abundance_forward_map_derived
    assert audit.conditional_wkb_composition_derived
    assert audit.late_dephased_created_excess_only
    assert not audit.bogoliubov_phase_and_anomalous_correlator_propagated
    assert not audit.created_state_is_e45_product_gaussian
    assert not audit.absolute_abundance_predicted
    assert not audit.vacuum_energy_derived
    assert not audit.covariant_qft_stress_renormalized
    assert not audit.physical_dark_matter_dark_energy_identification


def test_vacuum_shift_changes_only_the_supplied_de_ledger() -> None:
    baseline = _fixed_basis_audit(final_vacuum_energy_density_ev4=0.02)
    shifted = _fixed_basis_audit(final_vacuum_energy_density_ev4=0.05)

    assert shifted.present_created_number_densities_ev3 == pytest.approx(
        baseline.present_created_number_densities_ev3
    )
    assert shifted.present_created_energy_densities_ev4 == pytest.approx(
        baseline.present_created_energy_densities_ev4
    )
    assert shifted.omega_created_today == pytest.approx(
        baseline.omega_created_today
    )
    assert shifted.omega_vacuum_today - baseline.omega_vacuum_today == (
        pytest.approx(0.03 / 0.03)
    )
    assert not shifted.vacuum_energy_derived


def test_noncommuting_or_degenerate_endpoint_paths_fail_closed() -> None:
    with pytest.raises(ValueError, match="fixed normal basis"):
        _fixed_basis_audit(
            initial_environment_mass_ev=3.0,
            final_system_mass_ev=2.0,
            final_environment_mass_ev=3.0,
        )
    with pytest.raises(ValueError, match="degenerate"):
        _fixed_basis_audit(
            final_bilinear_coupling_ev2=0.0,
        )
    with pytest.raises(ValueError, match="initial normal masses are degenerate"):
        _fixed_basis_audit(
            initial_bilinear_coupling_ev2=0.0,
        )


def test_local_flrw_quench_approximation_is_an_explicit_gate() -> None:
    audit = _fixed_basis_audit(hubble_at_production_ev=1.0)

    assert audit.local_quench_ratio == pytest.approx(0.2)
    assert not audit.local_quench_pass
    assert "local_quench_Hstar_tau" in audit.failed_gates
    assert audit.status == "FAIL_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB_GATES"
