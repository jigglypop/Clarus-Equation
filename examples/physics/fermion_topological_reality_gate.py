from __future__ import annotations

from reality_stone.clarus.fermion_topological_reality import (
    charged_fermion_topology_audit,
    flux_multiplicity_control,
)


def main() -> None:
    audit = charged_fermion_topology_audit()
    flux = flux_multiplicity_control(
        wormhole_length_m=1000.0,
        magnetic_radius_m=1.0,
        flux_zero_mode_count=1000,
    )
    print("CE CHARGED-FERMION TOPOLOGY REALITY GATE")
    print(" massless bound eV", audit.effectively_massless_energy_bound_ev)
    print(" electron/bound ratio", audit.electron_to_massless_bound_ratio)
    print(" CE charged massless fermion", audit.ce_charged_massless_fermion_specified)
    print(" ambient shortcut", audit.ambient_space_shortcut)
    print(" flux lower bound", flux.dsnec_scale_lower_bound)
    print(" human-scale CE mapping", audit.human_scale_ce_mapping_pass)


if __name__ == "__main__":
    main()
