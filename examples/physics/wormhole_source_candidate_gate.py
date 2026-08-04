from __future__ import annotations

from reality_stone.clarus.spatial_folding import casimir_cell_conversion_audit
from reality_stone.clarus.wormhole_source_candidates import (
    resonance_source_audit,
    scalar_null_energy_audit,
    source_candidate_catalog,
)


def main() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    linear = resonance_source_audit(
        throat_radius_m=1.0,
        base_negative_density_j_m3=density,
        base_correlation_length_m=6.65e-15,
        density_gain_exponent=1.0,
    )
    quadratic = resonance_source_audit(
        throat_radius_m=1.0,
        base_negative_density_j_m3=density,
        base_correlation_length_m=6.65e-15,
        density_gain_exponent=2.0,
    )
    canonical = scalar_null_energy_audit(null_directional_derivative=1.0)
    phantom = scalar_null_energy_audit(
        null_directional_derivative=1.0,
        kinetic_sign=-1.0,
    )

    print("CE WORMHOLE SOURCE CANDIDATE LOOP")
    print(" linear-Q required", linear.combined_q_required)
    print(" quadratic-Q required", quadratic.combined_q_required)
    print(" Q scaling derived", linear.density_scaling_law_derived_from_ce)
    print(" canonical NEC violation", canonical.violates_nec)
    print(" phantom ghost free", phantom.ghost_free_kinetic_term)
    for candidate in source_candidate_catalog():
        print(" candidate", candidate.name, "->", candidate.first_failed_or_open_gate)


if __name__ == "__main__":
    main()
