from __future__ import annotations

from reality_stone.clarus.physical_multimode_realization import (
    physical_multimode_realization_audit,
)


def main() -> None:
    audit = physical_multimode_realization_audit()

    print("CE PHYSICAL MULTI-MODE REALIZATION GATE")
    print(" required density J/m^3", audit.required_density_magnitude_j_m3)
    print(" Casimir separation m", audit.ideal_casimir_separation_m)
    print(" carrier energy eV", audit.fundamental_energy_ev)
    print(" separation/boundary resolution", audit.separation_to_boundary_resolution_ratio)
    print(" two-sided negative energy J", audit.two_sided_negative_energy_magnitude_j)
    print(" mass equivalent Earths", audit.two_sided_mass_equivalent_earths)
    print(" radius for nuclear resolution m", audit.radius_for_boundary_resolution_m)
    print(" resolved-radius mass equivalent Suns", audit.resolved_radius_mass_equivalent_solar)
    print(" crossing/QI duration", audit.crossing_to_qi_duration_ratio)
    print(" physical realization pass", audit.current_physical_realization_pass)


if __name__ == "__main__":
    main()
