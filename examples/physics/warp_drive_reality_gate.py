from reality_stone.clarus.warp_drive_reality import (
    audit_alcubierre_tanh_wall,
    warp_pathway_portfolio,
)


def main() -> None:
    for speed in (0.5, 1.0, 2.0, 10.0):
        audit = audit_alcubierre_tanh_wall(speed_over_c=speed)
        print(f"speed/c={speed:g}")
        print(f"  Eulerian wall energy [J]        {audit.total_eulerian_energy_j:.6e}")
        print(f"  negative mass [Earth]           {audit.negative_mass_earth:.6e}")
        print(
            "  minimum Eulerian density [J/m3] "
            f"{audit.minimum_eulerian_energy_density_j_m3:.6e}"
        )
        print(
            "  Eulerian WEC violated            "
            f"{audit.eulerian_weak_energy_condition_violated}"
        )
        print(
            "  generic NEC no-go applicable     "
            f"{audit.generic_warp_nec_no_go_applicable}"
        )
        print(
            "  explicit null projection         "
            f"{audit.explicit_null_projection_computed}"
        )
        print(f"  superluminal shortcut           {audit.superluminal_shortcut}")
        print(f"  axial horizon pair              {audit.axis_horizon_pair_exists}")
        print(f"  axial horizon radius [m]        {audit.axis_horizon_radius_m}")

    print("warp pathway portfolio")
    for candidate in warp_pathway_portfolio():
        print(f"  {candidate.name}: {candidate.verdict}")


if __name__ == "__main__":
    main()
