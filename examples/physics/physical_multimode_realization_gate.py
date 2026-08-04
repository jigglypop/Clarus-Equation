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
    print(
        " coordinate-density integral J",
        audit.two_sided_coordinate_density_integral_magnitude_j,
    )
    print(
        " coordinate integral mass equivalent Earths",
        audit.two_sided_coordinate_mass_equivalent_earths,
    )
    print(
        " proper-volume integral I",
        audit.proper_volume_integral_dimensionless,
    )
    print(
        " proper matter energy J",
        audit.two_sided_proper_matter_energy_magnitude_j,
    )
    print(
        " proper mass equivalent Earths",
        audit.two_sided_proper_mass_equivalent_earths,
    )
    print(" proper/coordinate ratio", audit.proper_to_coordinate_energy_ratio)
    print(" radius for nuclear resolution m", audit.radius_for_boundary_resolution_m)
    print(
        " resolved-radius coordinate mass equivalent Suns",
        audit.resolved_radius_coordinate_mass_equivalent_solar,
    )
    print(
        " resolved-radius proper mass equivalent Suns",
        audit.resolved_radius_proper_mass_equivalent_solar,
    )
    print(" crossing/QI duration", audit.crossing_to_qi_duration_ratio)
    print(" physical realization pass", audit.current_physical_realization_pass)


if __name__ == "__main__":
    main()
