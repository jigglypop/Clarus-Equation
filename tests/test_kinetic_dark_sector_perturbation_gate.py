from __future__ import annotations

import math

import pytest

from examples.physics.finite_ctp_diagonal_source_obstruction import (
    audit_thermal_forced_oscillator_ctp,
)
from examples.physics.kinetic_dark_sector_gate import KineticClockConfig, solve_background
from examples.physics.kinetic_dark_sector_perturbation_gate import (
    audit_finite_product_gaussian_state_densities,
    audit_gaussian_normal_mode_perturbations,
    audit_product_gaussian_state_wkb_perturbations,
    evaluate_single_clock_gate,
    quasi_static_growth_diagnostic,
    scan_kappa_sensitivity,
)


def test_product_gaussian_state_has_an_exact_finite_mode_energy_ledger() -> None:
    audit = audit_finite_product_gaussian_state_densities(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        comoving_volume_ev_minus3=1.0,
        system_field_mean_ev=1.0,
        system_field_velocity_ev2=0.0,
        environment_mean_occupation=0.0,
        environment_inverse_temperature_ev_minus1=math.inf,
    )

    assert audit.covariance_ordering == (
        "q_phi",
        "q_chi",
        "p_phi",
        "p_chi",
    )
    expected_covariance = (
        (0.25, 0.0, 0.0, 0.0),
        (0.0, 1.0 / 6.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.5),
    )
    for actual_row, expected_row in zip(
        audit.centered_symmetrized_covariance,
        expected_covariance,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert audit.symplectic_eigenvalues == pytest.approx((0.5, 0.5))
    assert audit.normal_mode_mass_squared == pytest.approx(
        (3.298437881284, 9.701562118716)
    )
    assert audit.normal_mode_masses == pytest.approx(
        (1.816160202538, 3.114733073430)
    )
    assert audit.normal_mode_position_cross_covariance == pytest.approx(
        0.026028960315
    )
    assert audit.normal_mode_momentum_cross_covariance == pytest.approx(
        -0.156173761889
    )
    assert audit.finite_mode_vacuum_subtracted_energies_ev == pytest.approx(
        (1.485079180462, 0.549474181554)
    )
    assert audit.finite_mode_density_constants_ev4 == pytest.approx(
        audit.finite_mode_vacuum_subtracted_energies_ev
    )
    assert audit.finite_mode_vacuum_subtracted_total_energy_ev == pytest.approx(
        2.034553362016
    )
    assert audit.uncoupled_coherent_preparation_energy_ev == pytest.approx(2.0)
    assert audit.uncoupled_thermal_preparation_energy_ev == 0.0
    assert audit.vacuum_mismatch_quench_energy_ev == pytest.approx(
        0.034553362016
    )
    assert audit.excitation_energy_ledger_relative_residual < 1.0e-14
    assert audit.raw_energy_rotation_relative_residual < 1.0e-14
    assert audit.mode_sign_flip_energy_relative_residual < 1.0e-14
    assert audit.uncertainty_principle_pass
    assert audit.covariance_physicality_pass
    assert audit.canonical_transform_pass
    assert audit.mass_matrix_stable
    assert audit.finite_mode_excitation_nonnegative
    mass_dimensions = dict(audit.mass_dimension_manifest)
    assert mass_dimensions["canonical_q"] == -0.5
    assert mass_dimensions["canonical_p"] == 0.5
    assert mass_dimensions["mode_density_constant"] == 4.0
    assert all(
        dimension == 0.0
        for _, dimension in audit.dimensionless_core_argument_mass_dimensions
    )
    assert audit.dimensions_pass
    assert audit.same_state_finite_mode_energy_map_derived
    assert not audit.ctp_to_cosmological_state_map_derived
    assert audit.finite_mode_vacuum_subtraction_only
    assert not audit.covariant_qft_stress_renormalized
    assert not audit.preparation_battery_dynamics_derived
    assert not audit.physical_dark_matter_dark_energy_identification


def test_product_gaussian_energy_ledger_covers_thermal_motion_and_volume() -> None:
    volume = 8.0
    audit = audit_finite_product_gaussian_state_densities(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        comoving_volume_ev_minus3=volume,
        system_field_mean_ev=0.25,
        system_field_velocity_ev2=0.5,
        environment_mean_occupation=1.0,
        environment_inverse_temperature_ev_minus1=math.log(2.0) / 3.0,
    )

    assert audit.canonical_mean == pytest.approx(
        (math.sqrt(volume) * 0.25, 0.0, math.sqrt(volume) * 0.5, 0.0)
    )
    assert audit.uncoupled_thermal_preparation_energy_ev == pytest.approx(3.0)
    assert audit.thermal_occupation_relative_residual < 1.0e-14
    assert audit.finite_mode_vacuum_subtracted_total_density_ev4 == pytest.approx(
        audit.finite_mode_vacuum_subtracted_total_energy_ev / volume
    )
    assert audit.excitation_energy_ledger_relative_residual < 1.0e-14
    assert audit.covariance_physicality_pass
    assert audit.dimensions_pass


def test_gaussian_normal_modes_supply_a_scale_dependent_sound_discriminant() -> None:
    audit = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        scale_factor=1.0,
        hubble_ev=1.0e-3,
        comoving_wavenumber_ev=0.1,
        comoving_mode_density_constants_ev4=(1.0, 2.0),
        reduced_planck_mass_ev=100.0,
    )

    assert audit.normal_mode_mass_squared == pytest.approx(
        (3.298437881284, 9.701562118716)
    )
    assert audit.normal_mode_masses == pytest.approx(
        (1.816160202538, 3.114733073430)
    )
    assert audit.mass_matrix_determinant_ev4 == pytest.approx(32.0)
    assert audit.mode_densities_ev4 == pytest.approx((1.0, 2.0))
    assert audit.background_density_ev4 == pytest.approx(8.0)
    assert audit.background_pressure_ev4 == pytest.approx(-5.0)
    assert audit.vacuum_equation_of_state == -1.0
    assert audit.vacuum_density_perturbation_ev4 == 0.0
    assert audit.linear_anisotropic_stress == 0.0
    assert audit.microscopic_characteristic_speed_squared == (1.0, 1.0)
    assert audit.effective_sound_speed_squared[0] > audit.effective_sound_speed_squared[1]
    assert audit.effective_sound_speed_squared[0] == pytest.approx(
        0.1**2 / (4.0 * audit.normal_mode_mass_squared[0])
    )
    assert audit.four_pi_g_density_sources_ev2 == pytest.approx(
        (5.0e-5, 1.0e-4)
    )
    for mass_squared, jeans in zip(
        audit.normal_mode_mass_squared,
        audit.jeans_comoving_wavenumbers_ev,
    ):
        assert jeans**4 == pytest.approx(
            4.0 * mass_squared * sum(audit.four_pi_g_density_sources_ev2)
        )
    assert audit.wkb_domain_pass
    assert audit.nonrelativistic_domain_pass
    assert audit.subhorizon_domain_pass
    assert audit.positive_vacuum_pass
    assert audit.background_dm_de_limit
    assert audit.perturbation_discriminant_derived
    perturbation_dimensions = dict(audit.mass_dimension_manifest)
    assert perturbation_dimensions["effective_sound_speed_squared"] == 0.0
    assert perturbation_dimensions["pressure_frequency_squared"] == 2.0
    assert perturbation_dimensions["four_pi_g_density_source"] == 2.0
    assert perturbation_dimensions["jeans_wavenumber_fourth_power"] == 4.0
    assert perturbation_dimensions["jeans_wavenumber"] == 1.0
    assert all(
        dimension == 0.0
        for _, dimension in audit.dimensionless_core_argument_mass_dimensions
    )
    assert audit.dimensions_pass
    assert audit.status == "PASS_CONDITIONAL_GAUSSIAN_WKB_PERTURBATIONS"
    assert audit.failed_gates == ()
    assert audit.representation.startswith("RETAINED_TWO_FIELD")
    assert not audit.integrated_out_environment_stress_added
    assert not audit.influence_gram_used_as_gravity_source
    assert not audit.growth_history_derived
    assert not audit.physical_dark_matter_dark_energy_identification


def test_gaussian_ctp_and_external_cosmology_share_only_the_action_manifest() -> None:
    ctp = audit_thermal_forced_oscillator_ctp(
        system_mass=2.0,
        environment_mass=3.0,
        bilinear_coupling=2.0,
        vacuum_energy_density=5.0,
        volume=1.0,
        duration=math.pi / 3.0,
        field_left=1.0,
        field_right=0.0,
        mean_occupation=0.0,
        inverse_temperature=math.inf,
    )
    perturbation = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        scale_factor=1.0,
        hubble_ev=1.0e-3,
        comoving_wavenumber_ev=0.1,
        comoving_mode_density_constants_ev4=(1.0, 2.0),
        reduced_planck_mass_ev=100.0,
    )

    assert perturbation.action_parameter_manifest == ctp.action_parameter_manifest
    assert ctp.representation.startswith("INTEGRATED_OUT")
    assert perturbation.representation.startswith("RETAINED_TWO_FIELD")
    assert not ctp.retained_environment_stress_added
    assert not perturbation.integrated_out_environment_stress_added
    assert not perturbation.influence_gram_used_as_gravity_source
    assert not perturbation.ctp_to_cosmological_state_map_derived


def test_same_supplied_state_feeds_its_derived_densities_into_the_wkb_gate() -> None:
    ctp = audit_thermal_forced_oscillator_ctp(
        system_mass=2.0,
        environment_mass=3.0,
        bilinear_coupling=2.0,
        vacuum_energy_density=5.0,
        volume=1.0,
        duration=math.pi / 3.0,
        field_left=1.0,
        field_right=0.0,
        mean_occupation=0.0,
        inverse_temperature=math.inf,
    )
    bridge = audit_product_gaussian_state_wkb_perturbations(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        comoving_volume_ev_minus3=1.0,
        system_field_mean_ev=1.0,
        system_field_velocity_ev2=0.0,
        environment_mean_occupation=0.0,
        environment_inverse_temperature_ev_minus1=math.inf,
        scale_factor=1.0,
        hubble_ev=1.0e-3,
        comoving_wavenumber_ev=0.1,
        reduced_planck_mass_ev=100.0,
    )

    state = bridge.state_density_audit
    perturbation = bridge.perturbation_audit
    assert state.action_parameter_manifest == ctp.action_parameter_manifest
    assert state.environment_thermal_marginal_manifest == (
        ctp.action_parameter_manifest[1],
        ctp.volume,
        ctp.mean_occupation,
        ctp.inverse_temperature,
    )
    assert perturbation.comoving_mode_density_constants_ev4 == pytest.approx(
        state.finite_mode_density_constants_ev4
    )
    assert bridge.action_parameter_manifest_match
    assert bridge.derived_density_constants_match
    assert bridge.same_state_finite_mode_energy_map_derived
    assert bridge.perturbation_discriminant_derived
    assert bridge.status == "PASS_CONDITIONAL_SAME_STATE_GAUSSIAN_WKB_BRIDGE"
    assert ctp.representation.startswith("INTEGRATED_OUT")
    assert bridge.representation.startswith("RETAINED_TWO_FIELD")
    assert not bridge.integrated_out_environment_stress_added
    assert not bridge.influence_gram_used_as_gravity_source
    assert not bridge.ctp_to_cosmological_state_map_derived
    assert not bridge.cosmological_initial_state_derived
    assert not bridge.absolute_abundance_predicted


def test_state_density_map_exposes_independence_and_degeneracy_counterexamples() -> None:
    common = dict(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        comoving_volume_ev_minus3=1.0,
        system_field_velocity_ev2=0.0,
        environment_mean_occupation=0.0,
        environment_inverse_temperature_ev_minus1=math.inf,
    )
    displaced = audit_finite_product_gaussian_state_densities(
        **common,
        vacuum_energy_density_ev4=5.0,
        system_field_mean_ev=1.0,
    )
    shifted_vacuum = audit_finite_product_gaussian_state_densities(
        **common,
        vacuum_energy_density_ev4=11.0,
        system_field_mean_ev=1.0,
    )
    undisplaced = audit_finite_product_gaussian_state_densities(
        **common,
        vacuum_energy_density_ev4=5.0,
        system_field_mean_ev=0.0,
    )

    assert shifted_vacuum.finite_mode_density_constants_ev4 == pytest.approx(
        displaced.finite_mode_density_constants_ev4
    )
    assert shifted_vacuum.vacuum_cell_energy_ev != displaced.vacuum_cell_energy_ev
    assert displaced.finite_mode_vacuum_subtracted_total_energy_ev > (
        undisplaced.finite_mode_vacuum_subtracted_total_energy_ev
    )

    with pytest.raises(ValueError, match="degenerate"):
        audit_finite_product_gaussian_state_densities(
            system_mass_ev=2.0,
            environment_mass_ev=2.0,
            bilinear_coupling_ev2=0.0,
            vacuum_energy_density_ev4=5.0,
            comoving_volume_ev_minus3=1.0,
            system_field_mean_ev=1.0,
            system_field_velocity_ev2=0.0,
            environment_mean_occupation=0.0,
            environment_inverse_temperature_ev_minus1=math.inf,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        audit_finite_product_gaussian_state_densities(
            **{**common, "environment_mean_occupation": 1.0},
            vacuum_energy_density_ev4=5.0,
            system_field_mean_ev=1.0,
        )


def test_gaussian_perturbation_approximation_fails_outside_nr_domain() -> None:
    audit = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=5.0,
        scale_factor=1.0,
        hubble_ev=1.0e-3,
        comoving_wavenumber_ev=1.0,
        comoving_mode_density_constants_ev4=(1.0, 2.0),
        reduced_planck_mass_ev=100.0,
    )

    assert not audit.nonrelativistic_domain_pass
    assert "nonrelativistic" in audit.failed_gates
    assert audit.status == "FAIL_CONDITIONAL_APPROXIMATION_GATE"
    assert not audit.perturbation_discriminant_derived

    zero_vacuum = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=2.0,
        environment_mass_ev=3.0,
        bilinear_coupling_ev2=2.0,
        vacuum_energy_density_ev4=0.0,
        scale_factor=1.0,
        hubble_ev=1.0e-3,
        comoving_wavenumber_ev=0.1,
        comoving_mode_density_constants_ev4=(1.0, 2.0),
        reduced_planck_mass_ev=100.0,
    )
    assert not zero_vacuum.positive_vacuum_pass
    assert not zero_vacuum.background_dm_de_limit
    assert "positive_vacuum" in zero_vacuum.failed_gates


def test_single_clock_gate_passes_without_claiming_matter_growth() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=600))
    gate = evaluate_single_clock_gate(solution)

    assert gate.status == "PASS_SINGLE_CLOCK_ONLY"
    assert gate.failed_gates == ()
    assert gate.matter_growth_likelihood.startswith("NOT_IMPLEMENTED")
    assert gate.min_friction > 0.0
    assert gate.max_tachyon_ratio < 1.0
    assert gate.max_log_growth_bound < 1.0
    assert gate.min_pump_slope > 0.0
    assert gate.min_zeta_decay_slope > 0.0
    assert gate.min_energy_cutoff_over_h > 1.0
    assert gate.min_wavenumber_cutoff_over_k_1mpc > 1.0
    assert math.isfinite(gate.fixed_coordinate_growth_minus_one)


def test_quasi_static_growth_is_explicitly_approximate_and_finite() -> None:
    solution = solve_background(KineticClockConfig(gamma=10.0, steps=600))
    diagnostic = quasi_static_growth_diagnostic(solution)

    assert 0.0 < diagnostic.predicted_fsigma8 < 1.0
    assert math.isfinite(diagnostic.pull)
    assert diagnostic.closure == "KINETIC_CLUSTERS_VACUUM_SMOOTH_GR_SUBHORIZON"
    assert diagnostic.role.startswith("APPROXIMATE_DIAGNOSTIC")


def test_kappa_scan_exposes_the_conditional_stability_threshold() -> None:
    rows = scan_kappa_sensitivity((3.0e11, 1.0e12), steps=600)

    assert rows[0].status == "FAIL_SINGLE_CLOCK_GATE"
    assert "positive_friction" in rows[0].failed_gates
    assert rows[1].status == "PASS_SINGLE_CLOCK_ONLY"
    assert rows[1].failed_gates == ()
