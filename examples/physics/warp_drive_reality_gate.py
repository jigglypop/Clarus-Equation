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
        print(f"  superluminal shortcut           {audit.superluminal_shortcut}")
        print(f"  horizon pair expected           {audit.front_back_horizons_expected}")

    print("warp pathway portfolio")
    for candidate in warp_pathway_portfolio():
        print(f"  {candidate.name}: {candidate.verdict}")


if __name__ == "__main__":
    main()
