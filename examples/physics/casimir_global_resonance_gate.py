from __future__ import annotations

from reality_stone.clarus.casimir_global_resonance import (
    engineered_eight_thirds_tail_audit,
    fixed_casimir_eos_asymptotic_audit,
    wavelength_resonance_audit,
)


def main() -> None:
    finite_redshift = fixed_casimir_eos_asymptotic_audit(
        density_tail_power=8.0 / 3.0
    )
    finite_mass = fixed_casimir_eos_asymptotic_audit(density_tail_power=4.0)
    engineered = engineered_eight_thirds_tail_audit()
    wavelength = wavelength_resonance_audit(
        cavity_separation_m=3.662808556063564e-18
    )

    print("CE CASIMIR GLOBAL/RESONANCE LOOP")
    print(" finite-redshift tail finite mass", finite_redshift.finite_adm_mass_falloff)
    print(" finite-mass tail finite redshift", finite_mass.finite_redshift_at_infinity)
    print(" engineered b/r -> 0", engineered.shape_over_radius_tends_to_zero)
    print(" engineered finite energy", engineered.total_source_energy_finite)
    print(" required wavelength m", wavelength.fundamental_wavelength_m)
    print(" required quantum eV", wavelength.fundamental_quantum_energy_ev)
    print(" CE pole harmonic ratio", wavelength.required_harmonic_ratio)
    print(" Q changes carrier", wavelength.quality_factor_changes_carrier_frequency)


if __name__ == "__main__":
    main()
