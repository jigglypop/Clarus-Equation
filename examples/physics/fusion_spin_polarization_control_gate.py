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
    evidence = report.published_evidence
    print("FUSION SPIN-POLARIZATION STANDARD-MODEL CONTROL")
    print(" reaction layer")
    print(f"  required P_D P_T                {reaction.required_polarization_product:.9e}")
    print(f"  deuteron p_zz                   {reaction.deuteron_tensor_polarization:.9e}")
    print(
        "  deuteron m+/m0/m- populations  "
        f"{reaction.deuteron_mplus_population:.6f}/"
        f"{reaction.deuteron_mzero_population:.6f}/"
        f"{reaction.deuteron_mminus_population:.6f}"
    )
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
    print(" published energy-dependent evidence")
    print(
        "  digitized full ratio            "
        f"{evidence.reaction.full_alignment_maxwellian_reactivity_ratio:.9e}"
    )
    print(
        "  digitized lower ratio           "
        f"{evidence.reaction.digitization_lower_maxwellian_reactivity_ratio:.9e}"
    )
    print(f"  physical evidence gate          {evidence.physical_spin_fusion_evidence_gate_pass}")
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
    print(
        f"  burn-weighted p_zz value        {retention.burn_weighted_deuteron_tensor_polarization}"
    )
    print(
        "  burn-weighted p_zz measured     "
        f"{retention.burn_weighted_deuteron_tensor_polarization_measured}"
    )
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
    print(f"  ledger denominator              {pump.energy_ledger_denominator}")
    print(
        "  per baseline reacted pair eV    "
        f"{pump.electrical_break_even_energy_per_baseline_reacted_pair_ev:.9e}"
    )
    print(
        "  per incremental reaction eV     "
        f"{pump.electrical_break_even_energy_per_incremental_fusion_reaction_ev:.9e}"
    )
    print(f"  10x-margin pump power W         {pump.engineering_margin_wall_plug_power_w:.9e}")
    print(
        f"  measured wall-plug ledger       {pump.measured_wall_plug_energy_per_injected_dt_pair_available}"
    )
    print(
        f"  measured pair energy eV/pair    {pump.measured_wall_plug_energy_per_injected_dt_pair_ev}"
    )
    print(
        "  measured uncertainty eV/pair   "
        f"{pump.measured_wall_plug_energy_std_per_injected_dt_pair_ev}"
    )
    print(f"  3sigma upper below 10x margin  {pump.uncertainty_upper_below_engineering_margin}")
    print(
        "  measured flow meets throughput "
        f"{pump.wall_plug_measurement_pair_flow_meets_declared_throughput}"
    )
    print(f"  measurement provenance          {pump.wall_plug_measurement_provenance}")
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

    if not evidence.energy_dependent_figure_control_reproduced:
        raise RuntimeError("published figure reproduction control regressed")
    if report.physical_spin_polarized_branch_pass:
        raise RuntimeError("unexpected physical spin-branch promotion requires review")
    if report.physical_ce_one_percent_branch_pass:
        raise RuntimeError("unexpected CE-branch promotion requires review")


if __name__ == "__main__":
    main()
