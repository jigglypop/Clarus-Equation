from __future__ import annotations

from reality_stone.clarus.fusion_full_loop import current_full_fusion_loop_report


def main() -> None:
    report = current_full_fusion_loop_report()
    pair = report.z2_pair
    broken = report.broken_z2
    background = report.coherent_background
    thermal = report.thermal_reactivity
    icf = report.icf

    print("CE FULL FUSION LOOP")
    print(" canonical Z2 pair branch")
    print(f"  pair vertex                     {pair.tree_pair_vertex_present}")
    print(f"  two-scalar range fm             {pair.two_scalar_asymptotic_range_fm:.9e}")
    print(f"  zero-bare portal pole GeV       {pair.zero_bare_mass_portal_pole_gev:.9e}")
    print(f"  invisible branching fraction   {pair.invisible_branching_fraction:.9e}")
    print(f"  supplied benchmark allowed     {pair.supplied_portal_benchmark_allowed}")
    print(" broken Z2 branch")
    print(f"  mixing / supplied limit         {broken.mixing_ratio_to_limit:.9e}")
    print(f"  branching-like ratio            {broken.branching_like_ratio_to_limit:.9e}")
    print(f"  static force / Coulomb at rn    {broken.legacy_static_force_ratio_at_nuclear_radius:.9e}")
    print(f"  allowed-limit force / Coulomb   {broken.maximum_static_force_ratio_under_supplied_limit:.9e}")
    print(" coherent background control")
    print(f"  fractional mass modulation      {background.fractional_nucleon_mass_modulation:.9e}")
    print(f"  field amplitude MeV             {background.required_field_amplitude_mev:.9e}")
    print(f"  energy density J/m^3            {background.energy_density_j_m3:.9e}")
    print(f"  replenishment power W/m^3       {background.replenishment_power_density_w_m3:.9e}")
    print(f"  drive / transit frequency       {background.drive_to_transit_frequency_ratio:.9e}")
    print(" standard D-T baseline")
    print(f"  T keV                           {thermal.temperature_kev:.6f}")
    print(f"  Bosch-Hale <sigma v> cm^3/s     {thermal.baseline_reactivity_cm3_s:.9e}")
    print(f"  baseline n tau cm^-3 s          {thermal.baseline_ignition_n_tau_cm3_s:.9e}")
    print(" NIF boundary")
    print(f"  published target gain           {icf.published_target_gain:.9f}")
    print(f"  rejected linear-rescale kJ      {icf.rejected_linear_rescale_energy_kj:.9f}")
    print(" final stage ledger")
    for stage in report.stages:
        print(f"  {stage.name:42s} {stage.status}")
    print(" maximum supported stage", report.maximum_supported_stage)
    print(" physical D-T amplitude", report.physical_dt_amplitude_modified)
    print(" modified Lawson", report.modified_lawson_derived)
    print(" NIF prediction", report.nif_ignition_prediction_derived)


if __name__ == "__main__":
    main()
