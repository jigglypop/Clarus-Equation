from __future__ import annotations

from reality_stone.clarus.topological_boundary_stress import (
    topological_boundary_stress_audit,
)


def test_constant_topological_terms_have_no_local_bulk_stress() -> None:
    audit = topological_boundary_stress_audit()

    assert len(audit.terms) == 4
    assert all(term.constant_coefficient for term in audit.terms)
    assert all(term.metric_independent_or_topological_in_bulk for term in audit.terms)
    assert not any(term.local_bulk_stress_nonzero for term in audit.terms)
    assert not any(term.negative_bulk_null_stress_derived for term in audit.terms)
    assert not audit.pure_topology_has_local_bulk_stress
    assert not audit.pure_topology_can_support_throat


def test_topology_can_select_edge_modes_but_does_not_supply_their_stress() -> None:
    audit = topological_boundary_stress_audit()

    assert audit.topology_can_select_sectors_and_boundary_conditions
    assert audit.dynamical_edge_theory_required
    assert all(term.requires_dynamical_edge_or_defect_fields for term in audit.terms)
    assert not audit.edge_stress_sign_derived
    assert not audit.ce_covariant_edge_action_specified
    assert not audit.current_reality_pass


def test_nec_preserving_pure_topology_does_not_evade_topological_censorship() -> None:
    audit = topological_boundary_stress_audit()

    assert audit.topological_censorship_blocks_nec_shortcut
    assert not audit.pure_topology_can_support_throat
