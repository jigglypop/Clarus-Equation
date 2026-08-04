from __future__ import annotations

from reality_stone.clarus.clarus_backreaction_candidates import (
    ideal_casimir_zero_redshift_match_audit,
    vacuum_polarization_scale_audit,
)


def main() -> None:
    casimir = ideal_casimir_zero_redshift_match_audit()
    vacuum = vacuum_polarization_scale_audit(
        throat_radius_m=1.0,
        field_correlation_length_m=6.65e-15,
    )

    print("CE CLARUS BACKREACTION CANDIDATE LOOP")
    print(" Casimir b-prime", casimir.shape_derivative_from_radial_match)
    print(" Casimir tangential residual", casimir.residual_tangential_pressure_over_scale)
    print(" exact tensor match", casimir.exact_zero_redshift_tensor_match)
    print(" vacuum large-mass parameter", vacuum.large_mass_expansion_parameter)
    print(" vacuum backreaction ratio", vacuum.backreaction_ratio)
    print(" required field multiplicity", vacuum.multiplicity_required)
    print(" order-one backreaction", vacuum.order_one_backreaction_reached)


if __name__ == "__main__":
    main()
