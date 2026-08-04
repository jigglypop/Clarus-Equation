from reality_stone.clarus.thin_shell_defect_reality import (
    audit_floquet_radial_control,
    audit_quantum_negative_layer,
    audit_static_schwarzschild_thin_shell,
    barotropic_radial_stability,
)


def main() -> None:
    for lapse in (1.0, 1.0e-6, 1.0e-12):
        audit = audit_static_schwarzschild_thin_shell(lapse=lapse)
        print(f"lapse f(a)                         {audit.lapse:.3e}")
        print(f"  surface energy [J/m^2]          {audit.surface_energy_j_m2:.6e}")
        print(f"  tangential pressure [N/m]       {audit.tangential_pressure_n_m:.6e}")
        print(f"  p/|sigma|                       {audit.pressure_to_abs_energy_ratio:.6e}")
        print(f"  shell mass [Earth masses]       {audit.shell_mass_earth:.6e}")
        print(f"  scale-free QFT EoS match        {audit.conformal_eos_match}")
        print(f"  required effective degrees      {audit.required_effective_degrees:.6e}")
        print(f"  species cutoff / radius         {audit.species_cutoff_to_radius:.6e}")
        print(f"  full reality pass               {audit.reality_pass}")

    print("barotropic radial-stability gate")
    for lapse in (0.1, 1.0 / 3.0, 0.5, 1.0):
        audit = barotropic_radial_stability(lapse, 1.0)
        print(
            f"  f={lapse:.6g}: a^2 V''={audit.potential_curvature_times_radius_squared:.6e}, "
            f"stable={audit.radially_stable}, requirement={audit.required_inequality}"
        )

    quantum = audit_quantum_negative_layer()
    print("one-species smooth quantum-layer control")
    print(
        "  maximum thickness [m]           "
        f"{quantum.maximum_negative_layer_thickness_m:.6e}"
    )
    print(f"  UV energy [eV]                  {quantum.ultraviolet_energy_ev:.6e}")
    print(f"  sampling time [s]               {quantum.sampling_time_s:.6e}")
    print(f"  boundary completion required    {quantum.boundary_completion_required}")

    floquet = audit_floquet_radial_control(0.05, 0.1)
    print("active Floquet radial control")
    print(f"  averaged curvature              {floquet.averaged_curvature:.6e}")
    print(f"  monodromy trace                 {floquet.monodromy_trace:.6e}")
    print(f"  monodromy determinant           {floquet.monodromy_determinant:.6e}")
    print(f"  exact stable                    {floquet.exact_floquet_stable}")
    print(f"  static source supplied          {floquet.supplies_static_negative_stress}")


if __name__ == "__main__":
    main()
