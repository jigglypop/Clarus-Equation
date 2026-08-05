from __future__ import annotations

from reality_stone.clarus.fusion_equation_iteration_loop import (
    current_fusion_equation_iteration_report,
)


def main() -> None:
    report = current_fusion_equation_iteration_report()
    print("CE FUSION EQUATION ITERATION LOOP")
    print(f" Bosch-Hale numeric / closed       {report.bosch_hale_numeric_to_closed_ratio:.9e}")
    for audit in (
        report.allowed_broken_z2,
        report.massless_unit_mixing_upper_bound,
        report.allowed_z2_pair,
        report.massless_z2_pair_upper_bound,
    ):
        print(f" {audit.branch}")
        print(f"  V/Coulomb at nuclear radius      {audit.potential_to_coulomb_ratio_at_nuclear_radius:.9e}")
        print(f"  20 keV enhancement - 1          {audit.wkb_enhancement_minus_one_at_20_kev:.9e}")
        print(f"  thermal reactivity ratio - 1    {audit.thermal_reactivity_ratio_minus_one:.9e}")
        print(f"  target reached                  {audit.engineering_gain_reached}")
    direct = report.direct_coupling_requirement
    print(" DIRECT COUPLING DIAGNOSTIC")
    print(f"  required g_N                    {direct.required_direct_nucleon_coupling:.9e}")
    print(f"  equivalent Higgs mixing         {direct.equivalent_higgs_mixing_sine:.9e}")
    print(
        "  registered-mass required g_N    "
        f"{report.direct_coupling_registered_mass_requirement.required_direct_nucleon_coupling:.9e}"
    )
    print(f"  physical gate pass              {direct.physical_gate_pass}")
    print(f" selected action meets target      {report.current_selected_action_meets_target}")
    print(f" model class meets target          {report.higgs_proportional_model_class_meets_target}")
    print(f" maximum supported stage           {report.maximum_supported_stage}")


if __name__ == "__main__":
    main()
