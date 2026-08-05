from __future__ import annotations

from reality_stone.clarus.fusion_resonance_loop import (
    current_fusion_resonance_loop_report,
)


def main() -> None:
    report = current_fusion_resonance_loop_report()
    line = report.scalar_line
    exchange = report.static_exchange
    wkb = report.wkb_q_1e9

    print("CE FUSION RESONANCE LOOP")
    print(" legacy scalar line arithmetic")
    print(f"  vacuum Q                       {line.vacuum_quality_factor:.9e}")
    print(f"  angular frequency rad/s        {line.angular_frequency_rad_s:.9e}")
    print(f"  cyclic frequency Hz            {line.cyclic_frequency_hz:.9e}")
    print(f"  cyclic linewidth Hz            {line.cyclic_linewidth_hz:.9e}")
    print(f"  collision sigma ansatz m^2     {line.collision_cross_section_ansatz_m2:.9e}")
    print(f"  collision width ansatz MeV     {line.collision_width_ansatz_mev:.9e}")
    print(f"  plasma Q under ansatz           {line.plasma_quality_factor_under_ansatz:.9e}")
    print(f"  scalar one-loop delta a_e       {line.electron_g_minus_two_one_loop:.9e}")
    print(" static exchange gate")
    print(f"  invariant transfer MeV^2        {exchange.invariant_transfer_mev2:.9e}")
    print(f"  timelike pole reached           {exchange.timelike_pole_reached}")
    print(" counterfactual Q times Yukawa WKB")
    print(f"  Q supplied                      {wkb.supplied_quality_factor:.9e}")
    print(f"  gamma_0                         {wkb.baseline_exponent:.9f}")
    print(f"  gamma_Q                         {wkb.modified_exponent:.9f}")
    print(f"  counterfactual enhancement      {wkb.counterfactual_tunnelling_enhancement:.9f}")
    print(f"  inner-radius cancellation Q     {wkb.inner_radius_cancellation_quality_factor:.9e}")
    print(f"  whole-barrier removal Q         {wkb.whole_barrier_removal_quality_factor:.9e}")
    print(f"  required fractional bandwidth  {wkb.whole_barrier_fractional_bandwidth:.9e}")
    print(" stage ledger")
    for stage in report.stages:
        print(f"  {stage.name:42s} {stage.status}")
    print(" maximum supported stage", report.maximum_supported_stage)
    print(" physical barrier reduction", report.physical_resonant_barrier_reduction_derived)
    print(" ignition energy derived", report.ignition_energy_derived)


if __name__ == "__main__":
    main()
