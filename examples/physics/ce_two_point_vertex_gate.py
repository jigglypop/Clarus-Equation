"""Run the conditional CE two-point and portal-vertex certificate."""

from __future__ import annotations

import json

from reality_stone.clarus.ce_two_point_vertex_certificate import (
    ce_light_pole_q04_q05_certificate,
)


def main() -> None:
    report = ce_light_pole_q04_q05_certificate()
    compatibility = report.mass_compatibility
    width = report.invisible_width

    print("=" * 88)
    print("CE Q0.4--Q0.5 CONDITIONAL TWO-POINT / PORTAL-VERTEX CERTIFICATE")
    print("=" * 88)
    print(f"status={report.status}")
    print(f"target={report.registered_target_mass_mev:.8f} MeV")
    print(f"portal-dominated pole={compatibility.zero_bare_mass_portal_pole_gev:.9f} GeV")
    print(f"required bare mass^2={compatibility.required_bare_mass_squared_gev2:.9f} GeV^2")
    print(f"target^2 / portal shift={compatibility.target_squared_to_portal_shift_ratio:.12e}")
    print(f"portal / target mass ratio={compatibility.portal_to_target_mass_ratio:.6f}")
    print(
        "nonnegative-bare lambda ceiling="
        f"{compatibility.maximum_lambda_for_nonnegative_bare_mass:.12e}"
    )
    print(
        "minimum lambda_phi for EW vacuum vs singlet-only control="
        f"{report.vacuum.minimum_singlet_self_coupling_against_singlet_only_vacuum:.9f}"
    )
    print(
        "selected EW/Z2 vacuum global in tree control="
        f"{report.vacuum.selected_ew_vacuum_global_among_tree_stationary_points}"
    )
    print(f"tree invariant residue={report.two_point.invariant_pole_residue:.1f}")
    print(f"h-phi-phi derivative={report.vertices.h_phi_phi_derivative_gev:.9f} GeV")
    print(f"BR(h->phi phi)={width.branching_fraction:.9f}")
    print(f"supplied BR limit={width.supplied_branching_fraction_upper_limit:.6f}")
    print(f"maximum |lambda_HP| from supplied limit={width.maximum_allowed_abs_lambda:.9f}")
    print("-" * 88)
    print(f"Q0.4 singlet tree control={report.singlet_block_q0_4_tree_control_pass}")
    print(f"Q0.5 local vertex control={report.singlet_block_q0_5_tree_control_pass}")
    print(f"physical Clarus pole derived={report.physical_clarus_pole_derived}")
    print(f"full LSZ passed={report.full_lsz_passed}")
    print(f"physical SM production rate derived={report.physical_sm_production_rate_derived}")
    print(f"maximum stage={report.maximum_supported_stage}")
    print("-" * 88)
    print(report.conclusion)
    print("-" * 88)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
