from __future__ import annotations

from reality_stone.clarus.casimir_carrier_target import (
    exact_casimir_carrier_target,
    legacy_bprime_minus_one_null_control,
)
from reality_stone.clarus.clarus_resonant_matter import (
    ClarusPumpMode,
    boundary_response_audit,
    boundary_stress_match_audit,
    canonical_daughter_stress_audit,
    current_clarus_resonant_matter_report,
    finite_pulse_bogoliubov_audit,
    pair_kinematics_scan,
    squared_field_spectrum_audit,
    standing_wave_target_audit,
)
from reality_stone.clarus.global_throat_exact_certificate import (
    global_throat_exact_certificate,
)


def main() -> None:
    target = exact_casimir_carrier_target()
    legacy = legacy_bprime_minus_one_null_control()
    standing = standing_wave_target_audit()
    global_throat = global_throat_exact_certificate()

    pump_energy = standing.pump_energy_ev
    pump_momentum = standing.pump_axial_momentum_ev
    toy_amplitude = 0.1 * pump_energy
    spectrum = squared_field_spectrum_audit(
        (
            ClarusPumpMode(pump_energy, pump_momentum, toy_amplitude),
            ClarusPumpMode(pump_energy, -pump_momentum, toy_amplitude),
        )
    )
    pair_channels = pair_kinematics_scan(
        spectrum,
        daughter_mass_ev=pump_energy,
    )

    # A deliberately supplied dimensionless toy modulation.  Its amplitude is
    # not derived from CE, so a numerical excitation cannot become a physical
    # Clarus-particle claim.
    drive_energy = standing.pair_line_total_energy_ev
    bogoliubov = finite_pulse_bogoliubov_audit(
        daughter_mass_ev=pump_energy,
        daughter_momentum_ev=0.0,
        drive_energy_ev=drive_energy,
        modulation_mass_squared_ev2=0.05 * drive_energy**2,
        pulse_cycles=8,
        ramp_cycles=2,
        integration_steps_per_cycle=128,
    )
    canonical_stress = canonical_daughter_stress_audit((1.0, -0.5))
    boundary = boundary_response_audit(
        state_kind="active_driven",
        target_energy_ev=target.carrier_energy_ev,
        reflectivity_at_target=1.0,
    )
    stress = boundary_stress_match_audit(
        rho_over_curvature_scale=-1.0 / 3.0,
        radial_pressure_over_curvature_scale=-1.0,
        tangential_pressure_over_curvature_scale=1.0 / 3.0,
    )
    report = current_clarus_resonant_matter_report(
        spectrum=spectrum,
        pair_channels=pair_channels,
        bogoliubov=bogoliubov,
        boundary_response=boundary,
        stress_match=stress,
    )

    print("CLARUS RESONANT MATTER LOOP")
    print(" current ideal-planar scale (formal lambda=2a choice)")
    print(f"  separation m                 {target.separation_m:.12e}")
    print(f"  wavelength m                 {target.wavelength_m:.12e}")
    print(f"  formal cavity energy eV      {target.carrier_energy_ev:.12e}")
    print(f"  CE-pole ratio                {target.carrier_to_ce_pole_ratio:.9f}")
    print(f"  nearest harmonic             {target.nearest_integer_harmonic}")
    print(f"  nearest detuning eV          {target.nearest_harmonic_detuning_ev:.9e}")
    print(" legacy control is separate")
    print(f"  legacy carrier eV            {legacy.carrier_energy_ev:.12e}")
    print(" counter-propagating standing-wave control")
    print(f"  grating period m             {standing.static_grating_period_m:.12e}")
    print(f"  pair-line total eV           {standing.pair_line_total_energy_ev:.12e}")
    print(
        "  pair-line 2E* detuning eV    "
        f"{standing.pair_line_detuning_from_twice_boundary_carrier_ev:.9e}"
    )
    print(f"  spectral lines               {len(spectrum.spectral_lines)}")
    print(
        "  invariant pair channels      ",
        sum(channel.invariant_pair_threshold_open for channel in pair_channels),
    )
    print(" smooth finite-pulse toy")
    print(f"  occupation N,2N,4N           {bogoliubov.occupation_n:.9e}")
    print(f"                                {bogoliubov.occupation_2n:.9e}")
    print(f"                                {bogoliubov.occupation_4n:.9e}")
    print(f"  Wronskian residual           {bogoliubov.wronskian_residual_4n:.9e}")
    print(
        "  conditional excitation       ",
        bogoliubov.conditional_asymptotic_daughter_excitation,
    )
    print("  physical particles           ", bogoliubov.physical_particle_production_derived)
    print(" canonical daughter source")
    print(f"  T_kk control                 {canonical_stress.classical_null_projection:.9e}")
    print(
        "  directly negative            ",
        canonical_stress.directly_supplies_negative_throat_source,
    )
    print(" boundary/stress bridge")
    print("  single-line boundary pass    ", boundary.physical_boundary_response_pass)
    print("  target components match      ", stress.component_match)
    print("  throat realization           ", stress.throat_realization_pass)
    print(" exact global source target")
    print(
        "  original volume NEC finite   ",
        global_throat.original.coordinate_volume_nec_finite,
    )
    print(
        "  localized volume NEC/end     "
        f"{global_throat.localized_phi_match.coordinate_volume_nec_dimensionless_per_end:.9e}"
    )
    print(
        "  global scalar K/F control     "
        f"{global_throat.localized_phi_match.global_kinetic_counterexample_value:.9e}"
    )
    print(
        "  healthy global scalar        ",
        global_throat.localized_phi_match.global_nonminimal_kinetic_positive,
    )
    print(" conditional toy stage", report.maximum_conditional_toy_stage)
    print(" CE-physical stage", report.maximum_ce_physical_stage)
    print(" wormhole realization", report.wormhole_realization_derived)


if __name__ == "__main__":
    main()
