from __future__ import annotations

from reality_stone.clarus.multimode_global_throat import (
    global_anisotropic_target_audit,
    multimode_target_fit_audit,
)


def main() -> None:
    audit = global_anisotropic_target_audit()
    repaired = global_anisotropic_target_audit(
        redshift_profile="schwarzschild_matched"
    )
    fit = multimode_target_fit_audit()

    print("CE MULTI-MODE GLOBAL THROAT TARGET")
    print(" throat Casimir match", audit.throat_matches_ideal_casimir)
    print(" conservation residual", audit.maximum_conservation_residual)
    print(" finite ADM mass", audit.finite_adm_mass)
    print(" asymptotically flat", audit.asymptotically_flat)
    print(" global geometry/control pass", audit.global_geometry_control_pass)
    print(" radial NEC negative everywhere", audit.radial_nec_strictly_negative_everywhere_proved)
    print(" radial affine ANEC finite", audit.complete_radial_affine_anec_finite_proved)
    print(" dimensionless radial ANEC", audit.sampled_dimensionless_two_sided_radial_anec)
    print(" x^3 NEC tail coefficient", audit.asymptotic_radial_nec_x_cubed_coefficient)
    print(" coordinate-volume NEC finite", audit.coordinate_volume_nec_burden_finite)
    print("SCHWARZSCHILD-MATCHED TAIL REPAIR")
    print(" throat Casimir match", repaired.throat_matches_ideal_casimir)
    print(" analytic lapse^2 lower bound", repaired.analytic_lapse_squared_lower_bound)
    print(" conservation residual", repaired.sampled_maximum_conservation_residual)
    print(" x^3 NEC tail coefficient", repaired.asymptotic_radial_nec_x_cubed_coefficient)
    print(" coordinate-volume NEC finite", repaired.coordinate_volume_nec_burden_finite)
    print(" source-tail control pass", repaired.source_tail_control_pass)
    print(" independent matter EOM derived", repaired.independent_matter_eom_derived)
    print(" CE multi-mode stress derived", audit.ce_multimode_stress_derived)
    print(" perturbative stability derived", audit.perturbative_stability_derived)
    for level in fit.levels:
        print(" modes/error", level.mode_count, level.maximum_normalized_error)
    print(" finite-mode approximation pass", fit.finite_mode_target_approximation_pass)
    print(" carrier/envelope bridge derived", fit.carrier_envelope_bridge_derived)


if __name__ == "__main__":
    main()
