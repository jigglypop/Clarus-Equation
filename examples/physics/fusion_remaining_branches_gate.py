from __future__ import annotations

from reality_stone.clarus.fusion_remaining_branches_loop import (
    current_fusion_remaining_branches_report,
)


def main() -> None:
    report = current_fusion_remaining_branches_report()
    direct = report.direct_operator
    drive = report.time_dependent_drive
    reactor = report.reactor_propagation
    print("CE FUSION REMAINING BRANCHES")
    print(" direct operator")
    print(f"  massless required g_N            {direct.massless_required_nucleon_coupling:.9e}")
    print(f"  29.65 MeV required g_N           {direct.registered_mass_required_nucleon_coupling:.9e}")
    print(f"  completion scale GeV             {direct.mass_proportional_completion_scale_registered_gev:.9e}")
    print(f"  nuclear mean field MeV/N         {direct.registered_nuclear_matter_mean_field_mev_per_nucleon:.9e}")
    print(f"  experimental gate                {direct.experimental_constraint_gate_pass}")
    print(" time-dependent drive")
    print(f"  published max EM density J/m3    {drive.published_field_max_energy_density_j_m3:.9e}")
    print(f"  CE quantum / published ceiling   {drive.ce_energy_to_published_photon_ceiling_ratio:.9e}")
    print(f"  1 keV quiver at 1e16 V/m fm      {drive.quiver_amplitude_at_one_kev_and_max_field_fm:.9e}")
    print(f"  CE-frequency quiver fm           {drive.quiver_amplitude_at_ce_frequency_and_max_field_fm:.9e}")
    print(f"  field for CE-frequency rn V/m    {drive.field_for_one_nuclear_radius_quiver_at_ce_frequency_v_m:.9e}")
    print(f"  Floquet D-T solved               {drive.floquet_dt_scattering_solved}")
    print(" reactor / ICF boundary")
    print(f"  allowed reactivity gain          {reactor.allowed_static_reactivity_fractional_gain:.9e}")
    print(f"  model-class upper gain           {reactor.higgs_model_class_reactivity_fractional_upper_bound:.9e}")
    print(f"  rejected NIF saving upper J      {reactor.rejected_nif_linear_energy_saving_upper_bound_j:.9e}")
    print(f" physical one-percent gain         {report.physical_one_percent_reactivity_gain_derived}")
    print(f" maximum supported stage           {report.maximum_supported_stage}")


if __name__ == "__main__":
    main()
