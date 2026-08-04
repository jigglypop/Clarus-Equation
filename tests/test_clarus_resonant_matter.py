from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.clarus_resonant_matter import (
    ClarusPumpMode,
    boundary_response_audit,
    boundary_stress_match_audit,
    canonical_daughter_stress_audit,
    current_clarus_resonant_matter_report,
    finite_pulse_bogoliubov_audit,
    pair_kinematics_scan,
    spectral_target_audit,
    squared_field_spectrum_audit,
    standing_wave_target_audit,
)


def test_single_mode_squared_spectrum_is_exact() -> None:
    spectrum = squared_field_spectrum_audit(
        [ClarusPumpMode(energy_ev=2.0, axial_momentum_ev=1.0, amplitude_ev=4.0)]
    )

    assert spectrum.input_mode_count == 1
    assert spectrum.coherent_mode_count == 1
    assert spectrum.dc_field_squared_ev2 == 8.0
    assert len(spectrum.spectral_lines) == 1
    line = spectrum.spectral_lines[0]
    assert line.energy_transfer_ev == 4.0
    assert line.axial_momentum_transfer_ev == 2.0
    assert line.invariant_mass_squared_ev2 == 12.0
    assert line.cosine_amplitude_ev2 == 8.0
    assert line.combined_linewidth_ev is None
    assert not line.combined_linewidth_model_derived
    assert line.origins == ("self:0",)
    assert spectrum.exact_quadratic_identity_used
    assert spectrum.exact_fourier_key_grouping_used
    assert spectrum.phase_coherent_aggregation_used
    assert not spectrum.full_spacetime_normalization_derived


def test_degenerate_quadratic_lines_cancel_as_phasors() -> None:
    # Phi=cos(t)-0.5*cos(3t): the self 2t line and the difference 3t-t
    # line have equal magnitude and opposite phase.
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(1.0, 0.0, 1.0, 0.0),
            ClarusPumpMode(3.0, 0.0, 0.5, math.pi),
        ]
    )

    energies = [line.energy_transfer_ev for line in spectrum.spectral_lines]
    assert not any(math.isclose(energy, 2.0) for energy in energies)
    assert energies == [4.0, 6.0]


def test_identical_modes_with_opposite_phase_cancel_before_squaring() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(1.0, 0.0, 2.0, 0.0),
            ClarusPumpMode(1.0, 0.0, 2.0, math.pi),
        ]
    )

    assert spectrum.zero_after_coherent_cancellation
    assert spectrum.coherent_mode_count == 0
    assert spectrum.dc_field_squared_ev2 == 0.0
    assert spectrum.spectral_lines == ()


def test_near_but_distinct_modes_are_not_erased_as_exact_cancellation() -> None:
    energy = 1.0e11
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(energy, 0.0, 1.0, 0.0),
            ClarusPumpMode(energy + 0.05, 0.0, 1.0, math.pi),
        ]
    )

    assert spectrum.coherent_mode_count == 2
    assert not spectrum.zero_after_coherent_cancellation
    assert spectrum.spectral_lines
    assert spectrum.exact_fourier_key_grouping_used


def test_near_opposite_phase_is_not_deleted_by_the_exact_gate() -> None:
    modes = [
        ClarusPumpMode(1.0, 0.0, 1.0, 0.0),
        ClarusPumpMode(1.0, 0.0, 1.0, math.pi + 1.0e-13),
    ]

    exact = squared_field_spectrum_audit(modes)
    approximate = squared_field_spectrum_audit(
        modes,
        cancellation_relative_tolerance=1.0e-12,
    )

    assert not exact.zero_after_coherent_cancellation
    assert exact.spectral_lines
    assert exact.exact_fourier_key_grouping_used
    assert approximate.zero_after_coherent_cancellation
    assert not approximate.exact_fourier_key_grouping_used


def test_quadratic_linewidths_are_not_invented_without_a_line_shape_model() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(2.0, 0.0, 1.0, linewidth_ev=0.2),
            ClarusPumpMode(3.0, 0.0, 1.0, linewidth_ev=0.3),
        ]
    )

    self_lines = [line for line in spectrum.spectral_lines if line.origins == ("self:0",)]
    assert self_lines[0].combined_linewidth_ev is None
    assert not self_lines[0].combined_linewidth_model_derived
    cross_lines = [
        line
        for line in spectrum.spectral_lines
        if line.origins[0].startswith(("sum:", "difference:"))
    ]
    assert all(line.combined_linewidth_ev is None for line in cross_lines)
    assert all(not line.combined_linewidth_model_derived for line in cross_lines)


def test_counterpropagating_sum_opens_pair_threshold_but_self_lines_do_not() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(10.0, 10.0, 1.0),
            ClarusPumpMode(10.0, -10.0, 1.0),
        ]
    )
    channels = pair_kinematics_scan(spectrum, daughter_mass_ev=4.0)

    open_channels = [channel for channel in channels if channel.invariant_pair_threshold_open]
    assert len(open_channels) == 1
    open_line = open_channels[0].line
    assert open_line.energy_transfer_ev == 20.0
    assert open_line.axial_momentum_transfer_ev == 0.0
    assert open_line.invariant_mass_squared_ev2 == 400.0
    assert open_channels[0].centre_of_mass_energy_per_daughter_ev == 10.0
    assert not open_channels[0].particle_production_dynamics_derived

    null_self_channels = [
        channel
        for channel in channels
        if channel.line.energy_transfer_ev == 20.0
        and abs(channel.line.axial_momentum_transfer_ev) == 20.0
    ]
    assert len(null_self_channels) == 2
    assert all(not channel.invariant_pair_threshold_open for channel in null_self_channels)
    assert all(
        channel.temporal_frequency_only_would_be_misleading for channel in null_self_channels
    )


def test_large_copropagating_frequency_is_not_massive_pair_energy() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(100.0, 100.0, 1.0),
            ClarusPumpMode(80.0, 80.0, 1.0),
        ]
    )
    channels = pair_kinematics_scan(spectrum, daughter_mass_ev=1.0)

    assert channels
    assert not any(channel.invariant_pair_threshold_open for channel in channels)
    assert any(channel.temporal_frequency_only_would_be_misleading for channel in channels)


def test_pair_threshold_tolerance_never_promotes_a_spacelike_line() -> None:
    spectrum = squared_field_spectrum_audit([ClarusPumpMode(1.0, 2.0, 1.0)])
    channel = pair_kinematics_scan(
        spectrum,
        daughter_mass_ev=0.0,
        threshold_tolerance_ev2=12.0,
    )[0]

    assert channel.line.invariant_mass_squared_ev2 == -12.0
    assert not channel.timelike_or_null_transfer
    assert not channel.invariant_pair_threshold_open
    assert channel.threshold_within_numerical_tolerance_only


def test_massless_null_transfer_has_no_centre_of_mass_frame() -> None:
    spectrum = squared_field_spectrum_audit([ClarusPumpMode(1.0, 1.0, 1.0)])
    channel = pair_kinematics_scan(spectrum, daughter_mass_ev=0.0)[0]

    assert channel.invariant_pair_threshold_open
    assert not channel.centre_of_mass_frame_exists
    assert channel.centre_of_mass_energy_per_daughter_ev is None
    assert channel.centre_of_mass_momentum_per_daughter_ev is None


def test_matching_a_coherent_line_does_not_imply_particles_or_negative_stress() -> None:
    spectrum = squared_field_spectrum_audit([ClarusPumpMode(5.0, 0.0, 1.0)])
    audit = spectral_target_audit(
        spectrum,
        target_energy_ev=10.0,
        target_axial_momentum_ev=0.0,
        energy_linewidth_ev=0.1,
        momentum_tolerance_ev=0.1,
    )

    assert audit.within_supplied_resolution
    assert audit.coherent_excitation_conditionally_matched
    assert not audit.particle_production_implied
    assert not audit.negative_stress_implied


def test_target_search_prefers_any_line_inside_the_rectangular_gate() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(45.0, 0.45, 1.0),
            ClarusPumpMode(0.5, 1.0, 1.0),
        ]
    )
    audit = spectral_target_audit(
        spectrum,
        target_energy_ev=0.0,
        target_axial_momentum_ev=0.0,
        energy_linewidth_ev=200.0,
        momentum_tolerance_ev=1.0,
    )

    assert audit.within_supplied_resolution
    assert audit.nearest_line is not None
    assert abs(audit.nearest_line.axial_momentum_transfer_ev) <= 1.0


def test_standing_wave_separates_boundary_carrier_from_pair_line() -> None:
    audit = standing_wave_target_audit()

    assert audit.grating_matches_target_separation
    assert audit.pump_wavelength_m == pytest.approx(audit.target.wavelength_m)
    assert audit.static_grating_period_m == pytest.approx(audit.target.separation_m)
    assert not audit.pair_line_is_twice_boundary_carrier
    assert audit.pair_line_detuning_from_twice_boundary_carrier_ev == pytest.approx(
        5747.5,
        rel=1.0e-4,
    )
    assert audit.pair_line_within_supplied_linewidth is None
    assert audit.pair_line_total_energy_ev == pytest.approx(
        2.0 * audit.target.carrier_energy_ev,
        rel=1.0e-7,
    )
    assert audit.target_carrier_is_not_daughter_mass


def test_smooth_finite_pulse_resolves_conditional_bosonic_excitation() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=8,
        ramp_cycles=2,
        integration_steps_per_cycle=128,
    )

    assert audit.central_drive_matches_selected_pair_energy
    assert audit.global_pair_phase_space_open
    assert audit.selected_mode_detuning_ev == pytest.approx(0.0)
    assert audit.leading_order_first_resonance_band_estimate
    assert audit.numerical_periodic_floquet_instability_resolved
    assert abs(audit.floquet_monodromy_trace_4n) > 2.0
    assert audit.smooth_in_out_switching
    assert audit.tachyon_free_certified_by_lower_bound
    assert not audit.tachyonic_during_pulse_derived
    assert audit.occupation_4n > 1.0
    assert audit.occupation_4n > audit.no_drive_occupation_4n
    assert audit.refinement_delta_2n_4n < audit.refinement_delta_n_2n
    assert audit.wronskian_residual_4n < 1.0e-8
    assert audit.occupation_numerically_resolved
    assert audit.conditional_asymptotic_daughter_excitation
    assert not audit.physical_particle_production_derived
    assert audit.maximum_supported_stage == ("CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION")


def test_zero_drive_coupling_has_no_particle_signal() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.0,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )

    assert audit.occupation_4n <= 1.0e-12
    assert not audit.occupation_above_no_drive_control
    assert not audit.conditional_asymptotic_daughter_excitation
    assert not audit.physical_particle_production_derived


def test_unresolved_tachyon_lower_bound_is_not_promoted_to_pair_resonance() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=2.0,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )

    assert not audit.tachyon_free_certified_by_lower_bound
    assert not audit.tachyonic_during_pulse_derived
    assert not audit.conditional_asymptotic_daughter_excitation
    assert audit.maximum_supported_stage == ("TACHYON_STATUS_UNRESOLVED_BY_CONSERVATIVE_BOUND")


def test_sudden_switching_is_not_a_physical_in_out_pass() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=0,
        integration_steps_per_cycle=64,
    )

    assert not audit.smooth_in_out_switching
    assert not audit.conditional_asymptotic_daughter_excitation


def test_off_resonant_selected_mode_is_not_promoted_from_finite_pulse_leakage() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=10.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=128,
    )

    assert audit.global_pair_phase_space_open
    assert not audit.central_drive_matches_selected_pair_energy
    assert not audit.leading_order_first_resonance_band_estimate
    assert not audit.numerical_periodic_floquet_instability_resolved
    assert not audit.conditional_asymptotic_daughter_excitation


def test_leading_mathieu_band_estimate_does_not_replace_monodromy() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=0.0,
        daughter_momentum_ev=math.sqrt(0.3),
        drive_energy_ev=1.0,
        modulation_mass_squared_ev2=0.1,
        pulse_cycles=8,
        ramp_cycles=2,
        integration_steps_per_cycle=128,
    )

    assert audit.leading_order_first_resonance_band_estimate
    assert abs(audit.floquet_monodromy_trace_4n) < 2.0
    assert not audit.numerical_periodic_floquet_instability_resolved
    assert not audit.conditional_asymptotic_daughter_excitation


def test_self_reported_particle_prerequisites_never_self_certify_physics() -> None:
    audit = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
        physical_clarus_pole_derived=True,
        action_vertex_derived=True,
        pump_backreaction_solved=True,
        pump_work_energy_accounted=True,
    )

    assert audit.all_physical_prerequisites_self_reported
    assert audit.conditional_asymptotic_daughter_excitation
    assert not audit.physical_particle_production_derived
    assert audit.maximum_supported_stage == ("CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daughter_mass_ev": -1.0},
        {"drive_energy_ev": 0.0},
        {"modulation_mass_squared_ev2": math.nan},
        {"pulse_cycles": True},
        {"ramp_cycles": 1.5},
        {"integration_steps_per_cycle": 63},
        {"daughter_statistics": "fermion"},
        {"physical_clarus_pole_derived": "False"},
    ],
)
def test_bogoliubov_gate_rejects_adversarial_inputs(kwargs: dict[str, object]) -> None:
    inputs: dict[str, object] = {
        "daughter_mass_ev": 1.0,
        "daughter_momentum_ev": 0.0,
        "drive_energy_ev": 2.0,
        "modulation_mass_squared_ev2": 0.2,
        "pulse_cycles": 4,
        "ramp_cycles": 1,
        "integration_steps_per_cycle": 64,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        finite_pulse_bogoliubov_audit(**inputs)  # type: ignore[arg-type]


def test_canonical_daughters_cannot_directly_supply_negative_null_stress() -> None:
    audit = canonical_daughter_stress_audit([3.0, -4.0])

    assert audit.classical_null_projection == 25.0
    assert audit.classical_null_projection_nonnegative
    assert audit.dephased_particle_null_projection_nonnegative
    assert not audit.directly_supplies_negative_throat_source
    assert not audit.occupation_determines_quantum_stress_sign
    assert audit.anomalous_correlator_and_phase_required_to_infer_from_occupation
    assert audit.boundary_subtraction_required_for_casimir_route


def test_single_frequency_reflectivity_never_completes_boundary_gate() -> None:
    audit = boundary_response_audit(
        state_kind="passive_equilibrium",
        target_energy_ev=1.5293e11,
        reflectivity_at_target=1.0,
    )

    assert not audit.single_real_frequency_reflectivity_sufficient
    assert not audit.causal_broadband_response_pass
    assert not audit.equilibrium_lifshitz_applicable
    assert not audit.physical_boundary_response_pass


def test_active_boundary_cannot_use_equilibrium_lifshitz_shortcut() -> None:
    audit = boundary_response_audit(
        state_kind="active_driven",
        target_energy_ev=1.5293e11,
        reflectivity_at_target=1.2,
        imaginary_frequency_grid_complete=True,
        transverse_momentum_grid_complete=True,
        polarization_response_complete=True,
        retarded_susceptibility_derived=True,
        kramers_kronig_residual=0.0,
        gain_balance_residual=0.0,
        nonequilibrium_keldysh_stress_derived=True,
    )

    assert audit.conditional_response_metadata_gate_pass
    assert not audit.causal_broadband_response_pass
    assert not audit.equilibrium_lifshitz_applicable
    assert not audit.physical_boundary_response_pass


def test_transparent_or_self_reported_boundary_metadata_never_self_certifies() -> None:
    transparent = boundary_response_audit(
        state_kind="passive_equilibrium",
        target_energy_ev=1.5293e11,
        reflectivity_at_target=0.0,
        imaginary_frequency_grid_complete=True,
        transverse_momentum_grid_complete=True,
        polarization_response_complete=True,
        retarded_susceptibility_derived=True,
        kramers_kronig_residual=0.0,
    )

    assert not transparent.nonzero_target_reflectivity
    assert not transparent.conditional_response_metadata_gate_pass
    assert not transparent.physical_boundary_response_pass

    with pytest.raises(ValueError):
        boundary_response_audit(
            state_kind="passive_equilibrium",
            target_energy_ev=1.0,
            reflectivity_at_target=1.0,
            imaginary_frequency_grid_complete="False",  # type: ignore[arg-type]
        )


def test_exact_target_components_without_provenance_do_not_realize_throat() -> None:
    audit = boundary_stress_match_audit(
        rho_over_curvature_scale=-1.0 / 3.0,
        radial_pressure_over_curvature_scale=-1.0,
        tangential_pressure_over_curvature_scale=1.0 / 3.0,
    )

    assert audit.component_match
    assert audit.radial_null_projection_over_curvature_scale == pytest.approx(-4.0 / 3.0)
    assert not audit.renormalized_stress_derived
    assert not audit.throat_realization_pass


def test_self_reported_stress_prerequisites_do_not_self_certify_a_throat() -> None:
    audit = boundary_stress_match_audit(
        rho_over_curvature_scale=-1.0 / 3.0,
        radial_pressure_over_curvature_scale=-1.0,
        tangential_pressure_over_curvature_scale=1.0 / 3.0,
        full_net_stress_includes_boundary_matter_pump_and_vacuum=True,
        renormalized_stress_derived=True,
        conservation_derived=True,
        finite_tail_certified=True,
        physical_affine_anec_negative=True,
        quantum_inequality_pass=True,
        backreaction_solved=True,
        perturbative_stability_pass=True,
    )

    assert audit.all_realization_prerequisites_self_reported
    assert not audit.throat_realization_pass


def test_current_report_stops_at_conditional_excitation() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(1.0, 1.0, 1.0),
            ClarusPumpMode(1.0, -1.0, 1.0),
        ]
    )
    channels = pair_kinematics_scan(spectrum, daughter_mass_ev=1.0)
    bogoliubov = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )
    report = current_clarus_resonant_matter_report(
        spectrum=spectrum,
        pair_channels=channels,
        bogoliubov=bogoliubov,
    )

    assert report.conditional_toy_excitation
    assert not report.physical_particle_production_derived
    assert not report.physical_boundary_derived
    assert not report.renormalized_negative_stress_derived
    assert not report.stable_backreacted_throat_derived
    assert not report.wormhole_realization_derived
    assert report.maximum_supported_stage == ("CONDITIONAL_ASYMPTOTIC_DAUGHTER_EXCITATION")
    assert report.maximum_conditional_toy_stage == report.maximum_supported_stage
    assert report.maximum_ce_physical_stage == "TARGET_SCALE_CALIBRATION_ONLY"


def test_report_rejects_an_unlinked_pair_scan_and_oscillator() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(10.0, 10.0, 1.0),
            ClarusPumpMode(10.0, -10.0, 1.0),
        ]
    )
    channels = pair_kinematics_scan(spectrum, daughter_mass_ev=4.0)
    unrelated_oscillator = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )

    report = current_clarus_resonant_matter_report(
        spectrum=spectrum,
        pair_channels=channels,
        bogoliubov=unrelated_oscillator,
    )

    assert not report.conditional_toy_excitation
    assert report.maximum_supported_stage == "CONDITIONAL_NONLINEAR_SPECTRUM"


def test_report_rejects_a_hostile_daughter_momentum_rewrite() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(1.0, 1.0, 1.0),
            ClarusPumpMode(1.0, -1.0, 1.0),
        ]
    )
    channels = pair_kinematics_scan(spectrum, daughter_mass_ev=1.0)
    oscillator = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )
    hostile = replace(oscillator, daughter_momentum_ev=0.5)

    report = current_clarus_resonant_matter_report(
        spectrum=spectrum,
        pair_channels=channels,
        bogoliubov=hostile,
    )

    assert not report.conditional_toy_excitation


def test_report_keeps_exact_field_cancellation_as_a_null_control() -> None:
    spectrum = squared_field_spectrum_audit(
        [
            ClarusPumpMode(1.0, 0.0, 1.0, 0.0),
            ClarusPumpMode(1.0, 0.0, 1.0, math.pi),
        ]
    )

    report = current_clarus_resonant_matter_report(spectrum=spectrum)

    assert report.maximum_supported_stage == ("COHERENT_FIELD_CANCELLATION_NULL_CONTROL")
    assert report.stages[3].status == "NULL_CONTROL"


def test_current_report_ignores_hostile_downstream_self_certification() -> None:
    bogoliubov = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=1.0,
        daughter_momentum_ev=0.0,
        drive_energy_ev=2.0,
        modulation_mass_squared_ev2=0.2,
        pulse_cycles=4,
        ramp_cycles=1,
        integration_steps_per_cycle=64,
    )
    hostile = replace(
        bogoliubov,
        physical_particle_production_derived=True,
        maximum_supported_stage="PHYSICAL_PARTICLE_PRODUCTION_CONTROL",
    )
    report = current_clarus_resonant_matter_report(bogoliubov=hostile)

    assert not report.physical_particle_production_derived
    assert not report.physical_boundary_derived
    assert not report.renormalized_negative_stress_derived
    assert not report.stable_backreacted_throat_derived
    assert report.maximum_supported_stage == "KINEMATIC_CORRELATION_ANSATZ"


@pytest.mark.parametrize(
    "modes",
    [
        [],
        [ClarusPumpMode(0.0, 0.0, 1.0)],
        [ClarusPumpMode(1.0, 0.0, -1.0)],
        [ClarusPumpMode(1.0, math.inf, 1.0)],
        [ClarusPumpMode(1.0, 0.0, 1.0, linewidth_ev=-1.0)],
    ],
)
def test_spectrum_gate_rejects_invalid_modes(modes: list[ClarusPumpMode]) -> None:
    with pytest.raises(ValueError):
        squared_field_spectrum_audit(modes)
