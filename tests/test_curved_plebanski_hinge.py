import math

import numpy as np
import pytest

from examples.physics.curved_plebanski_hinge import (
    constructive_curved_plebanski_hinge_witness,
    curved_hinge_trace_audit,
    de_sitter_plebanski_point_audit,
    exact_dual_hinge_holonomy,
    exact_primal_triangle_holonomy,
)


def test_one_added_simplex_closes_the_original_open_hinge_link() -> None:
    audit = curved_hinge_trace_audit()

    assert audit.original_simplices == (
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
    )
    assert audit.added_simplex == (0, 1, 2, 3, 5)
    assert audit.original_link_edges == ((0, 4), (4, 5))
    assert audit.extended_link_edges == ((0, 4), (0, 5), (4, 5))
    assert audit.extended_link_degrees == ((0, 2), (4, 2), (5, 2))
    assert audit.simplex_oriented_coordinate_volumes == pytest.approx(
        (-1.0, 1.0, -0.4)
    )
    assert audit.original_two_cell_link_open
    assert audit.closed_three_cycle
    assert audit.minimal_one_simplex_closure
    assert audit.original_cells_preserved
    assert audit.causal_label_matches_hinge


def test_removing_closing_simplex_restores_open_link_negative_control() -> None:
    audit = curved_hinge_trace_audit(include_closing_simplex=False)

    assert audit.original_two_cell_link_open
    assert not audit.closed_three_cycle
    assert not audit.minimal_one_simplex_closure
    assert audit.status == "NO_CLOSED_DUAL_FACE"
    with pytest.raises(ValueError, match="NO_CLOSED_DUAL_FACE"):
        exact_dual_hinge_holonomy(
            audit,
            curvature_times_reference_length_squared=1.0,
        )


@pytest.mark.parametrize(
    "point",
    (
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.2, 0.2, 0.2),
        (-1.0, 0.2, 0.2, 0.2),
        (0.2, 0.24, 0.24, 0.24),
    ),
)
def test_exact_de_sitter_fields_satisfy_plebanski_identities(point: tuple[float, ...]) -> None:
    audit = de_sitter_plebanski_point_audit(
        point,
        curvature_times_reference_length_squared=1.0,
    )

    assert audit.patch_denominator > 0.0
    assert audit.torsion_residual < 1.0e-12
    assert audit.riemann_constant_curvature_residual < 1.0e-12
    assert audit.self_duality_residual < 1.0e-12
    assert audit.simplicity_tracefree_residual < 1.0e-12
    assert audit.covariant_constancy_residual < 1.0e-12
    assert audit.plebanski_curvature_residual < 1.0e-12


def test_wrong_cosmological_constant_breaks_plebanski_equation() -> None:
    audit = de_sitter_plebanski_point_audit(
        (0.0, 0.0, 0.0, 0.0),
        curvature_times_reference_length_squared=1.0,
        cosmological_constant_times_reference_length_squared=2.0,
    )
    certificate = constructive_curved_plebanski_hinge_witness(
        curvature_times_reference_length_squared=1.0,
        cosmological_constant_times_reference_length_squared=2.0,
    )

    assert audit.plebanski_curvature_residual > 0.1
    assert not certificate.same_history_nonflat_plebanski_witness_closed
    assert certificate.status == "NONFLAT_PLEBANSKI_HINGE_WITNESS_FAILED"


def test_labelled_primal_face_has_exact_nonidentity_lorentz_holonomy() -> None:
    audit = exact_primal_triangle_holonomy(
        curvature_times_reference_length_squared=1.0
    )

    assert audit.face_id == (1, 2, 3)
    assert audit.coordinate_triangle_area_over_reference_length_squared == pytest.approx(
        0.5
    )
    assert audit.transport_convention == "dV + omega V = 0"
    assert audit.oriented_boundary == (1, 2, 3, 1)
    assert audit.rotation_angle == pytest.approx(0.4290007391955229)
    assert audit.lorentz_residual < 1.0e-12
    assert audit.flatness_residual > 0.1
    assert audit.nontrivial_curvature_holonomy
    assert np.linalg.det(audit.holonomy) == pytest.approx(1.0)


def test_zero_curvature_collapses_primal_holonomy_to_identity() -> None:
    audit = exact_primal_triangle_holonomy(
        curvature_times_reference_length_squared=0.0
    )

    assert audit.rotation_angle == pytest.approx(0.0)
    assert audit.flatness_residual == pytest.approx(0.0)
    assert np.allclose(audit.holonomy, np.eye(4))
    assert not audit.nontrivial_curvature_holonomy
    assert audit.status == "FLAT_PRIMAL_HOLONOMY"


def test_wrong_primal_face_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="labelled internal hinge"):
        exact_primal_triangle_holonomy(
            curvature_times_reference_length_squared=1.0,
            face_id=(0, 1, 2),
        )


def test_exact_closed_dual_loop_detects_curvature() -> None:
    trace = curved_hinge_trace_audit()
    audit = exact_dual_hinge_holonomy(
        trace,
        curvature_times_reference_length_squared=1.0,
    )

    assert len(audit.path_labels) == 6
    assert audit.minimum_geodesic_denominator > 0.0
    assert audit.maximum_hyperboloid_residual < 1.0e-12
    assert audit.maximum_segment_tangency_residual < 1.0e-12
    assert audit.lorentz_residual < 1.0e-12
    assert audit.flatness_residual > 0.01
    assert audit.projective_barycentric_realization
    assert audit.positive_cone_segment_inclusion
    assert not audit.regge_deficit_angle_derived
    assert audit.closed_dual_curvature_holonomy


def test_aggregate_certificate_stays_at_its_declared_claim_ceiling() -> None:
    certificate = constructive_curved_plebanski_hinge_witness()

    assert certificate.same_history_nonflat_plebanski_witness_closed
    assert certificate.cosmological_constant_times_reference_length_squared == pytest.approx(
        3.0 * certificate.curvature_times_reference_length_squared
    )
    assert certificate.convex_hull_patch_denominator_lower_bound > 0.0
    assert certificate.maximum_sampled_field_residual < 1.0e-12
    assert certificate.field_evidence_scope.startswith("ANALYTIC_ALL_POINTS")
    assert certificate.primal_holonomy.nontrivial_curvature_holonomy
    assert certificate.dual_holonomy.closed_dual_curvature_holonomy
    assert certificate.einstein_endpoint.lorentzian_einstein_geometry
    assert not certificate.proper_vertex_amplitude_derived
    assert not certificate.continuum_refinement_derived
    assert not certificate.two_dof_ir_spectrum_derived
    assert certificate.claim_ceiling.endswith("NOT_CONTINUUM_QG")


@pytest.mark.parametrize("curvature", (0.0, 4.0, math.inf))
def test_aggregate_certificate_rejects_uncertified_curvature_patch(curvature: float) -> None:
    with pytest.raises(ValueError):
        constructive_curved_plebanski_hinge_witness(
            curvature_times_reference_length_squared=curvature
        )
