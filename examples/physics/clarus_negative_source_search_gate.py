from __future__ import annotations

from reality_stone.clarus.clarus_negative_source_search import (
    averaged_nonminimal_null_audit,
    casimir_plate_scale_audit,
    clarus_negative_source_funnel,
    effective_planck_amplification_audit,
    nonminimal_scalar_null_audit,
)
from reality_stone.clarus.spatial_folding import (
    casimir_cell_conversion_audit,
    wormhole_throat_audit,
)


def main() -> None:
    local = nonminimal_scalar_null_audit(
        nonminimal_coupling=0.49,
        field_value_planck=0.5,
        affine_first_derivative=0.1,
        affine_second_derivative=1.0,
    )
    averaged = averaged_nonminimal_null_audit(
        nonminimal_coupling=0.49,
        gradient_squared_integral=2.0,
        endpoint_field_squared_derivative_jump=0.0,
    )
    density = casimir_cell_conversion_audit().energy_density_j_m3
    throat = wormhole_throat_audit(
        throat_radius_m=1.0,
        candidate_negative_density_j_m3=density,
    )
    plate = casimir_plate_scale_audit(
        required_null_magnitude_j_m3=abs(throat.nec_energy_density_j_m3),
        ce_correlation_length_m=6.65e-15,
    )
    amplification = effective_planck_amplification_audit(
        required_amplification=throat.local_density_gap,
        nonminimal_coupling=0.49,
    )

    print("CE CLARUS NEGATIVE-SOURCE SEARCH LOOP")
    print(" local null numerator", local.null_numerator)
    print(" effective Planck factor", local.effective_planck_factor)
    print(" local candidate", local.local_candidate_survives)
    print(" localized averaged NEC violation", averaged.averaged_nec_violated)
    print(" ideal plate separation m", plate.plate_separation_m)
    print(" required Planck factor", amplification.required_effective_planck_factor)
    print(" relative distance to gravity pole", amplification.relative_distance_below_critical)
    for candidate in clarus_negative_source_funnel():
        print(candidate.frontier, candidate.name, "->", candidate.decisive_next_calculation)


if __name__ == "__main__":
    main()
