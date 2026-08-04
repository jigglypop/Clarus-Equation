from __future__ import annotations

from reality_stone.clarus.nonminimal_global_reconstruction import (
    nonminimal_throat_codesign_audit,
    nonminimal_throat_reconstruction_audit,
)


def main() -> None:
    audit = nonminimal_throat_reconstruction_audit()
    codesign = nonminimal_throat_codesign_audit(
        shape_second_derivative=-5.0,
        redshift_second_derivative=0.0,
    )

    print("CE NONMINIMAL GLOBAL RECONSTRUCTION GATE")
    print(" d ln F / dx at throat", audit.logarithmic_planck_factor_radial_slope)
    print(" F_ss / F at throat", audit.proper_planck_factor_second_derivative)
    print(" required scalar kinetic / F", audit.required_positive_metric_scalar_kinetic)
    print(" healthy single scalar", audit.healthy_single_scalar_possible)
    print(" healthy multimode scalars", audit.healthy_multiscalar_modes_possible)
    print(" target refuted", audit.target_refuted_for_healthy_nonminimal_scalars)
    print(" co-design kinetic / F", codesign.required_scalar_kinetic_over_planck_factor)
    print(" local co-design survives", codesign.local_codesign_survives)


if __name__ == "__main__":
    main()
