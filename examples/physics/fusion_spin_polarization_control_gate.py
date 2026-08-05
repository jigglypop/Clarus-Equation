from __future__ import annotations

from reality_stone.clarus.fusion_spin_polarization_control_loop import (
    current_fusion_spin_polarization_control_report,
)


def main() -> None:
    report = current_fusion_spin_polarization_control_report()
    reaction = report.target_reaction
    design = report.source_design_reaction
    retention = report.retention
    equilibrium = report.thermal_equilibrium
    pump = report.pump_ledger
    print("FUSION SPIN-POLARIZATION STANDARD-MODEL CONTROL")
    print(" reaction layer")
    print(f"  required P_D P_T                {reaction.required_polarization_product:.9e}")
    print(f"  Maxwellian reactivity ratio     {reaction.maxwellian_reactivity_ratio:.9e}")
    print(
        f"  ideal-projector target reached  {reaction.conditional_ideal_projector_target_reached}"
    )
    print(
        "  energy-dependent rate integral  "
        f"{reaction.energy_dependent_polarized_cross_section_integrated}"
    )
    print(
        "  reaction-operator provenance    "
        f"{reaction.energy_dependent_polarized_reaction_operator_provenance_pass}"
    )
    print(f"  Czz directly measured           {reaction.spin_correlation_czz_directly_measured}")
    print(f"  polarized D-T rate validated    {reaction.polarized_dt_rate_directly_validated}")
    print(" source margin")
    print(f"  design P_D P_T                  {design.polarization_product:.9e}")
    print(
        f"  common retention threshold      {retention.minimum_common_species_retention_fraction:.9e}"
    )
    print(
        "  reactor-rate D source           "
        f"{retention.deuteron_pellet_source_demonstrated_at_reactor_throughput}"
    )
    print(f"  D-T retention measured          {retention.dt_in_plasma_retention_measured}")
    print(f"  burn-weighted product measured  {retention.burn_weighted_product_measured}")
    print(f"  burn-weighted product value     {retention.burn_weighted_polarization_product}")
    print(f"  burn-weighted product required  {retention.burn_weighted_product_required:.9e}")
    print(
        "  burn product meets threshold    "
        f"{retention.burn_weighted_product_meets_required_threshold}"
    )
    print(
        "  burn product <= source product  "
        f"{retention.burn_weighted_product_not_above_source_product}"
    )
    print(
        "  reactor-rate T source           "
        f"{retention.tritium_polarization_source_demonstrated_at_reactor_throughput}"
    )
    print(" equilibrium route")
    print(f"  required field T                {equilibrium.required_uniform_magnetic_field_t:.9e}")
    print(f"  field energy density J/m3       {equilibrium.magnetic_field_energy_density_j_m3:.9e}")
    print(" pump ledger")
    print(f"  D-T pair throughput /s          {pump.declared_dt_pair_injection_rate_s:.9e}")
    print(
        "  linearized budget keV/pair      "
        f"{pump.linearized_incremental_fusion_energy_budget_per_injected_pair_kev:.9e}"
    )
    print(
        "  fixed-exposure budget keV/pair  "
        f"{pump.incremental_fusion_energy_budget_per_injected_pair_kev:.9e}"
    )
    print(
        f"  wall-plug break-even eV/pair    {pump.electrical_break_even_energy_per_injected_pair_ev:.9e}"
    )
    print(f"  10x-margin pump power W         {pump.engineering_margin_wall_plug_power_w:.9e}")
    print(
        f"  measured wall-plug ledger       {pump.measured_wall_plug_energy_per_polarized_dt_pair_available}"
    )
    print(
        f"  measured pair energy eV/pair    {pump.measured_wall_plug_energy_per_polarized_dt_pair_ev}"
    )
    print(f"  measured energy below ceiling   {pump.measured_pair_energy_below_break_even}")
    print(f"  cryo/microwave accounted        {pump.cryogenic_and_microwave_power_accounted}")
    print(f"  tritium handling accounted      {pump.tritium_handling_power_accounted}")
    print(
        "  recycle/repolarize accounted    "
        f"{pump.recycle_depolarization_and_repolarization_accounted}"
    )
    print(f"  net incremental positive        {pump.net_incremental_energy_positive_demonstrated}")
    print(f" physical spin branch             {report.physical_spin_polarized_branch_pass}")
    print(f" physical CE branch               {report.physical_ce_one_percent_branch_pass}")
    print(f" maximum supported stage          {report.maximum_supported_stage}")


if __name__ == "__main__":
    main()
