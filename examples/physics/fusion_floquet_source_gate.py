from __future__ import annotations

from reality_stone.clarus.fusion_floquet_source_loop import (
    current_fusion_floquet_source_report,
)


def main() -> None:
    report = current_fusion_floquet_source_report()
    regression = report.regression_point
    threshold = report.qed_threshold
    pump = report.pump_ledger
    beat = report.ce_scalar_beat
    print("CE FUSION FLOQUET / SOURCE LOOP")
    print(" QED Floquet--Volkov control")
    print(f"  gain at 0.3 keV, 1e16 V/m       {regression.reactivity_fractional_gain:.9e}")
    print(f"  field for 1 percent V/m         {threshold.required_electric_field_v_m:.9e}")
    print(f"  ponderomotive energy eV         {threshold.ponderomotive_energy_ev:.9e}")
    print(f"  Gamow Keldysh parameter         {threshold.keldysh_gamow_parameter:.9e}")
    print(f"  grid gain spread                {threshold.maximum_grid_fractional_gain_spread:.9e}")
    print(f"  prescribed QED branch pass      {threshold.prescribed_qed_reactivity_branch_pass}")
    print(" pump ledger")
    print(f"  10 fs, 10 nm pulse energy J     {pump.incident_pulse_energy_j:.9e}")
    print(f"  extra fusion / incident pulse   {pump.incremental_fusion_to_incident_pulse_energy_ratio:.9e}")
    print(f"  net energy positive             {pump.net_energy_positive}")
    print(" exact-Z2 CE scalar beat")
    print(f"  beat reduced wavelength fm      {beat.beat_reduced_wavelength_fm:.9e}")
    print(f"  required mass modulation        {beat.required_fractional_mass_modulation:.9e}")
    print(f"  required scalar density J/m3    {beat.required_scalar_energy_density_j_m3:.9e}")
    print(f"  physical CE scalar branch       {beat.physical_ce_scalar_reactivity_branch_pass}")
    print(f" maximum supported stage          {report.maximum_supported_stage}")


if __name__ == "__main__":
    main()
