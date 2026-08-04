from __future__ import annotations

from reality_stone.clarus.multimode_global_throat import (
    global_anisotropic_target_audit,
    multimode_target_fit_audit,
)


def main() -> None:
    audit = global_anisotropic_target_audit()
    fit = multimode_target_fit_audit()

    print("CE MULTI-MODE GLOBAL THROAT TARGET")
    print(" throat Casimir match", audit.throat_matches_ideal_casimir)
    print(" conservation residual", audit.maximum_conservation_residual)
    print(" finite ADM mass", audit.finite_adm_mass)
    print(" asymptotically flat", audit.asymptotically_flat)
    print(" global geometry/control pass", audit.global_geometry_control_pass)
    print(" CE multi-mode stress derived", audit.ce_multimode_stress_derived)
    print(" perturbative stability derived", audit.perturbative_stability_derived)
    for level in fit.levels:
        print(" modes/error", level.mode_count, level.maximum_normalized_error)
    print(" finite-mode approximation pass", fit.finite_mode_target_approximation_pass)
    print(" carrier/envelope bridge derived", fit.carrier_envelope_bridge_derived)


if __name__ == "__main__":
    main()
