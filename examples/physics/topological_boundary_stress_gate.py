from __future__ import annotations

from reality_stone.clarus.topological_boundary_stress import (
    topological_boundary_stress_audit,
)


def main() -> None:
    audit = topological_boundary_stress_audit()
    print("CE TOPOLOGICAL BOUNDARY STRESS GATE")
    for term in audit.terms:
        print(term.name, "bulk stress", term.local_bulk_stress_nonzero)
    print(" topology selects sectors", audit.topology_can_select_sectors_and_boundary_conditions)
    print(" edge theory required", audit.dynamical_edge_theory_required)
    print(" CE edge action specified", audit.ce_covariant_edge_action_specified)
    print(" current reality pass", audit.current_reality_pass)


if __name__ == "__main__":
    main()
