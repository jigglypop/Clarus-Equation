"""Run the CE spatial-folding wormhole control audit."""

from __future__ import annotations

from reality_stone.clarus.spatial_folding import (
    casimir_cell_conversion_audit,
    spatial_shortcut_audit,
    wormhole_throat_audit,
)


def main() -> None:
    light_year_m = 9.4607304725808e15
    shortcut = spatial_shortcut_audit(light_year_m, 10.0, 0.1)
    conversion = casimir_cell_conversion_audit()
    throat = wormhole_throat_audit(
        throat_radius_m=1.0,
        candidate_negative_density_j_m3=conversion.energy_density_j_m3,
    )

    print("CE SPATIAL-FOLDING WORMHOLE LOOP 1")
    print(f"  local speed / c             {shortcut.local_speed_fraction_c:.12g}")
    print(f"  throat traversal time s    {shortcut.throat_traversal_time_s:.12g}")
    print(f"  traveler proper time s     {shortcut.traveler_proper_time_s:.12g}")
    print(f"  apparent speed / c         {shortcut.apparent_speed_over_c:.12g}")
    print(f"  locally causal             {shortcut.locally_subluminal}")
    print(f"  exactly instantaneous      {shortcut.exactly_instantaneous}")
    print(f"  CE cell density J/m^3      {conversion.energy_density_j_m3:.12g}")
    print(f"  CE cell mass kg/m^3        {conversion.mass_equivalent_density_kg_m3:.12g}")
    print(f"  old/report conversion gap  {conversion.reported_to_correct_ratio:.12g}")
    print(f"  throat NEC J/m^3           {throat.nec_energy_density_j_m3:.12g}")
    print(f"  local density gap          {throat.local_density_gap:.12g}")
    print(f"  coherence Q required       {throat.coherence_q_required:.12g}")
    print(f"  CE stress tensor derived   {throat.ce_stress_tensor_bridge_derived}")
    print(f"  physical wormhole          {throat.physical_wormhole_established}")


if __name__ == "__main__":
    main()
