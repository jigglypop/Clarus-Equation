"""Metric-variation gate for proposed topological wormhole sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologicalTermAudit:
    name: str
    constant_coefficient: bool
    metric_independent_or_topological_in_bulk: bool
    local_bulk_stress_nonzero: bool
    negative_bulk_null_stress_derived: bool
    requires_dynamical_edge_or_defect_fields: bool


@dataclass(frozen=True)
class TopologicalBoundaryPortfolioAudit:
    terms: tuple[TopologicalTermAudit, ...]
    pure_topology_has_local_bulk_stress: bool
    pure_topology_can_support_throat: bool
    topology_can_select_sectors_and_boundary_conditions: bool
    dynamical_edge_theory_required: bool
    edge_stress_sign_derived: bool
    topological_censorship_blocks_nec_shortcut: bool
    ce_covariant_edge_action_specified: bool
    current_reality_pass: bool


def topological_boundary_stress_audit() -> TopologicalBoundaryPortfolioAudit:
    """Audit constant-coefficient topological terms by metric variation.

    A metric-independent action has identically zero Hilbert stress tensor.
    Four-dimensional Euler/Gauss--Bonnet and Pontryagin densities with constant
    coefficients vary to boundary terms and likewise supply no local bulk
    source.  Edge modes may be dynamical and metric-sensitive, but their stress
    belongs to the edge QFT and must be computed separately.
    """

    terms = (
        TopologicalTermAudit(
            name="4D Euler/Gauss-Bonnet invariant",
            constant_coefficient=True,
            metric_independent_or_topological_in_bulk=True,
            local_bulk_stress_nonzero=False,
            negative_bulk_null_stress_derived=False,
            requires_dynamical_edge_or_defect_fields=True,
        ),
        TopologicalTermAudit(
            name="4D gravitational Pontryagin invariant",
            constant_coefficient=True,
            metric_independent_or_topological_in_bulk=True,
            local_bulk_stress_nonzero=False,
            negative_bulk_null_stress_derived=False,
            requires_dynamical_edge_or_defect_fields=True,
        ),
        TopologicalTermAudit(
            name="gauge theta F wedge F",
            constant_coefficient=True,
            metric_independent_or_topological_in_bulk=True,
            local_bulk_stress_nonzero=False,
            negative_bulk_null_stress_derived=False,
            requires_dynamical_edge_or_defect_fields=True,
        ),
        TopologicalTermAudit(
            name="3D boundary Chern-Simons term",
            constant_coefficient=True,
            metric_independent_or_topological_in_bulk=True,
            local_bulk_stress_nonzero=False,
            negative_bulk_null_stress_derived=False,
            requires_dynamical_edge_or_defect_fields=True,
        ),
    )
    any_bulk_stress = any(term.local_bulk_stress_nonzero for term in terms)
    return TopologicalBoundaryPortfolioAudit(
        terms=terms,
        pure_topology_has_local_bulk_stress=any_bulk_stress,
        pure_topology_can_support_throat=any_bulk_stress,
        topology_can_select_sectors_and_boundary_conditions=True,
        dynamical_edge_theory_required=True,
        edge_stress_sign_derived=False,
        topological_censorship_blocks_nec_shortcut=True,
        ce_covariant_edge_action_specified=False,
        current_reality_pass=False,
    )
