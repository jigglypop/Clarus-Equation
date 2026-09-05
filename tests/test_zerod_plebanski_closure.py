from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.gravity.causal_face_simplicity import (
    hard_shared_spacelike_face_match,
)
from examples.physics.gravity.zerod_plebanski_closure import (
    MINKOWSKI_METRIC,
    bivector_face_reconstruction_audit,
    bivector_from_normal_edge,
    block_rg_verdict,
    blocked_rendered_mean,
    common_linear_simplicity_nullity,
    constant_curvature_einstein_audit,
    constructive_zerod_to_plebanski_witness,
    continuum_gr_dof_no_go,
    critical_block_moments,
    critical_borel_asymptotic_ratio,
    critical_side_tree_total_progeny_probability,
    critical_split_merge,
    decoherent_fold_audit,
    face_holonomy_audit,
    finite_constraint_concentration,
    flat_chiral_plebanski_audit,
    form_degree_closure,
    heat_time_from_area,
    hodge_dual,
    ir_einstein_dominance_audit,
    marked_joint_probability,
    massive_spin_two_polarization_count,
    massive_traceless_transverse_basis_4d,
    massless_spin_two_polarization_count,
    massless_tt_basis_4d,
    planck_resolution_quotient,
    q_spine_distinct_probability,
    quadratic_stationary_phase_audit,
    simplex_interaction_audit,
    spine_fixed_point,
    typed_rank_four_event_trace,
)


D_CE = 3.1777584234


# ---------------------------------------------------------------------------
# 분할/병합 블록 RG (구 test_planck_rendering_block_rg)
# ---------------------------------------------------------------------------


def test_critical_marking_is_unique_and_gives_exact_ce_numbers() -> None:
    params = critical_split_merge(D_CE)

    assert params.distinct_probability == pytest.approx(1.0 / D_CE)
    assert params.merge_probability == pytest.approx(1.0 - 1.0 / D_CE)
    assert params.distinct_intensity == pytest.approx(1.0)
    assert params.face_intensity == pytest.approx(D_CE - 1.0)


def test_marked_poisson_joint_law_is_normalized_on_a_large_window() -> None:
    params = critical_split_merge(D_CE)
    probability = math.fsum(
        marked_joint_probability(
            branch_mean=D_CE,
            distinct_probability=params.distinct_probability,
            distinct_count=rendered,
            face_count=faces,
        )
        for rendered in range(16)
        for faces in range(24)
    )

    assert probability == pytest.approx(1.0, abs=1.0e-12)


def test_blocking_fixes_the_mean_but_not_the_poisson_law() -> None:
    assert blocked_rendered_mean(1.0, depth=8) == pytest.approx(1.0)
    audit = critical_block_moments(D_CE, depth=8)

    assert audit.output_mean == pytest.approx(1.0)
    assert audit.output_variance == pytest.approx(8.0)
    assert not audit.poisson_family_closed
    assert audit.expected_face_events == pytest.approx(8.0 * (D_CE - 1.0))


def test_noncritical_rendered_mean_flows_away_under_blocking() -> None:
    assert blocked_rendered_mean(0.9, depth=4) == pytest.approx(0.9**4)
    assert blocked_rendered_mean(1.1, depth=4) == pytest.approx(1.1**4)
    assert blocked_rendered_mean(0.9, depth=4) < 0.9
    assert blocked_rendered_mean(1.1, depth=4) > 1.1


def test_spine_conditioning_has_one_persistent_line_and_stationary_side_law() -> None:
    audit = spine_fixed_point(D_CE)

    assert audit.persistent_spine_count == 1
    assert audit.rendered_continuation_mean == pytest.approx(2.0)
    assert audit.folded_side_branch_mean == pytest.approx(1.0)
    assert audit.face_event_mean == pytest.approx(D_CE - 1.0)
    assert audit.shift_invariant_local_law
    assert math.fsum(q_spine_distinct_probability(k) for k in range(1, 20)) == pytest.approx(
        1.0,
        abs=1.0e-15,
    )


def test_critical_side_tree_has_borel_three_halves_tail() -> None:
    first_mass = critical_side_tree_total_progeny_probability(1)
    assert first_mass == pytest.approx(math.exp(-1.0))
    assert critical_borel_asymptotic_ratio(10_000) == pytest.approx(1.0, rel=5.0e-5)


def test_heat_time_is_additive_in_area() -> None:
    first = heat_time_from_area(area=2.0, planck_area=0.5)
    second = heat_time_from_area(area=3.0, planck_area=0.5)
    combined = heat_time_from_area(area=5.0, planck_area=0.5)

    assert combined == pytest.approx(first + second)


def test_verdict_keeps_the_remaining_geometry_obligation_explicit() -> None:
    verdict = block_rg_verdict(D_CE)

    assert verdict.critical_rendered_mean == pytest.approx(1.0)
    assert verdict.critical_face_intensity == pytest.approx(D_CE - 1.0)
    assert not verdict.full_poisson_measure_fixed
    assert verdict.spine_local_measure_fixed
    assert verdict.side_sector_scale_free
    assert "simplicity" in verdict.remaining_obligation


@pytest.mark.parametrize("branch_mean", (1.0, 0.5, -1.0, math.inf))
def test_critical_split_merge_rejects_invalid_domain(branch_mean: float) -> None:
    with pytest.raises(ValueError):
        critical_split_merge(branch_mean)


# ---------------------------------------------------------------------------
# 연속 GR / 두 자유도 무유도 정리 (구 test_continuum_gr_dof_no_go)
# ---------------------------------------------------------------------------


def _minkowski_trace(tensor: np.ndarray) -> float:
    return float(np.sum(MINKOWSKI_METRIC * tensor))


def test_covariant_r_squared_counterexample_deletes_unique_gr_implication() -> None:
    audit = continuum_gr_dof_no_go(alpha_over_reference_length_squared=2.0)

    assert audit.scalaron_mass_squared_times_reference_length_squared == pytest.approx(
        1.0 / 12.0
    )
    assert audit.einstein_hilbert_polarizations == 2
    assert audit.r_plus_r_squared_polarizations == 3
    assert audit.shared_flat_stationary_background
    assert audit.shared_finite_flat_face_data
    assert audit.both_actions_diffeomorphism_invariant
    assert not audit.unique_continuum_action_follows
    assert not audit.exactly_two_dof_follow
    assert audit.status == "FINITE_FACE_TO_UNIQUE_CONTINUUM_GR_IMPLICATION_DISPROVED"


def test_massless_tt_basis_has_exactly_plus_and_cross() -> None:
    basis = massless_tt_basis_4d()
    null_covector = np.array((-1.0, 0.0, 0.0, 1.0))

    assert massless_spin_two_polarization_count(4) == 2
    assert len(basis) == 2
    for tensor in basis:
        np.testing.assert_allclose(tensor, tensor.T)
        assert _minkowski_trace(tensor) == pytest.approx(0.0)
        np.testing.assert_allclose(null_covector @ tensor, np.zeros(4))
    assert np.linalg.matrix_rank(np.stack([item.ravel() for item in basis])) == 2


def test_massive_rest_frame_basis_has_five_independent_polarizations() -> None:
    basis = massive_traceless_transverse_basis_4d()
    timelike_covector = np.array((-1.0, 0.0, 0.0, 0.0))

    assert massive_spin_two_polarization_count(4) == 5
    assert len(basis) == 5
    for tensor in basis:
        np.testing.assert_allclose(tensor, tensor.T)
        assert _minkowski_trace(tensor) == pytest.approx(0.0)
        np.testing.assert_allclose(timelike_covector @ tensor, np.zeros(4))
    assert np.linalg.matrix_rank(np.stack([item.ravel() for item in basis])) == 5


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf")])
def test_counterexample_requires_nontachyonic_finite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        continuum_gr_dof_no_go(alpha)


# ---------------------------------------------------------------------------
# 로렌츠 이중벡터 재구성 (구 test_lorentzian_bivector_reconstruction)
# ---------------------------------------------------------------------------


def _boost() -> np.ndarray:
    rapidity = 0.7
    cosine = math.cosh(rapidity)
    sine = math.sinh(rapidity)
    return np.array(
        [
            [cosine, sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _geometric_face() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = _boost()
    normal = transform @ np.array([1.0, 0.0, 0.0, 0.0])
    rest_edges = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    edges = rest_edges @ transform.T
    bivectors = np.asarray(
        [bivector_from_normal_edge(normal, edge) for edge in edges]
    )
    return normal, edges, bivectors


def test_lorentzian_hodge_star_squares_to_minus_one() -> None:
    normal = np.array([1.0, 0.0, 0.0, 0.0])
    edge = np.array([0.0, 1.0, 0.0, 0.0])
    bivector = bivector_from_normal_edge(normal, edge)
    dual = hodge_dual(bivector)

    assert dual[2, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(hodge_dual(dual), -bivector, atol=1.0e-15)


def test_linear_simple_bivectors_reconstruct_edges_and_gram() -> None:
    normal, edges, bivectors = _geometric_face()
    audit = bivector_face_reconstruction_audit(normal, bivectors)

    np.testing.assert_allclose(audit.reconstructed_edges, edges, atol=1.0e-14)
    np.testing.assert_allclose(audit.edge_gram, np.diag((1.0, 4.0, 9.0)), atol=1.0e-14)
    np.testing.assert_allclose(audit.bivector_gram, audit.edge_gram, atol=1.0e-14)
    assert audit.common_normal_nullity == 1
    assert audit.hard_reconstruction
    assert audit.status == "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"
    assert audit.plebanski_branch == "NOT_TESTED_BY_LINEAR_FACE_DATA"
    assert audit.claim_ceiling == "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTION_ONLY"


def test_rank_deficient_linear_simple_data_are_rejected() -> None:
    normal, _, bivectors = _geometric_face()
    bivectors[2] = bivectors[1]

    audit = bivector_face_reconstruction_audit(normal, bivectors)

    assert not audit.hard_reconstruction
    assert audit.status == "NONSPACELIKE_OR_RANK_DEFICIENT_FACE"


def test_bf_pair_has_no_common_linear_simplicity_normal() -> None:
    basis = np.eye(4)
    time_space = np.outer(basis[0], basis[1]) - np.outer(basis[1], basis[0])
    space_space = np.outer(basis[2], basis[3]) - np.outer(basis[3], basis[2])

    assert common_linear_simplicity_nullity(
        np.asarray((time_space, space_space))
    ) == 0


def test_declared_wrong_normal_fails_linear_simplicity() -> None:
    basis = np.eye(4)
    normal = basis[0]
    spatial = np.outer(basis[2], basis[3]) - np.outer(basis[3], basis[2])
    bivectors = np.asarray((spatial, 2.0 * spatial, 3.0 * spatial))

    audit = bivector_face_reconstruction_audit(normal, bivectors)

    assert not audit.hard_reconstruction
    assert audit.status == "LINEAR_SIMPLICITY_FAILED"


def test_reconstructed_bivector_faces_feed_proper_shared_face_gluing() -> None:
    left_normal = np.array([1.0, 0.0, 0.0, 0.0])
    left_edges = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    left_bivectors = np.asarray(
        [bivector_from_normal_edge(left_normal, edge) for edge in left_edges]
    )
    right_to_left = _boost()
    left_to_right = np.linalg.inv(right_to_left)
    right_normal = left_to_right @ left_normal
    right_edges = left_edges @ left_to_right.T
    right_bivectors = np.asarray(
        [bivector_from_normal_edge(right_normal, edge) for edge in right_edges]
    )
    left_audit = bivector_face_reconstruction_audit(left_normal, left_bivectors)
    right_audit = bivector_face_reconstruction_audit(right_normal, right_bivectors)

    gluing = hard_shared_spacelike_face_match(
        left_audit.reconstructed_edges,
        left_normal,
        np.array([2.0, 0.2, 0.1, 0.0]),
        right_audit.reconstructed_edges,
        right_normal,
        left_to_right @ np.array([-1.5, 0.3, 0.0, 0.2]),
        right_to_left,
    )

    assert left_audit.hard_reconstruction
    assert right_audit.hard_reconstruction
    assert gluing.hard_match
    assert gluing.status == "FINITE_SHARED_SPACELIKE_FACE_MATCH"


@pytest.mark.parametrize(
    "length_scale", [1.0e-200, 1.0e-100, 1.0e-6, 1.0e100, 1.0e200]
)
def test_bivector_reconstruction_is_invariant_under_length_units(
    length_scale: float,
) -> None:
    normal, edges, _ = _geometric_face()
    scaled_edges = edges * length_scale
    bivectors = np.asarray(
        [bivector_from_normal_edge(normal, edge) for edge in scaled_edges]
    )

    audit = bivector_face_reconstruction_audit(normal, bivectors)

    assert audit.hard_reconstruction
    np.testing.assert_allclose(
        audit.reconstructed_edges / length_scale,
        edges,
        rtol=2.0e-15,
        atol=1.0e-14,
    )


# ---------------------------------------------------------------------------
# 0D→플레바인스키 구성적 유한 닫힘
# ---------------------------------------------------------------------------


def test_same_type_two_form_closure_selects_three_plus_one() -> None:
    audit = form_degree_closure()

    assert audit.spacetime_dimension == 4
    assert audit.spatial_dimension == 3
    assert audit.same_type_pair
    assert audit.hodge_degree_closes
    assert audit.nondegenerate_lorentzian_signature
    assert audit.lorentzian_three_plus_one


def test_form_degree_condition_is_not_bare_dimension_uniqueness() -> None:
    audit = form_degree_closure(1, 2)

    assert audit.spacetime_dimension == 3
    assert not audit.same_type_pair
    assert not audit.hodge_degree_closes
    assert not audit.lorentzian_three_plus_one


def test_one_named_time_direction_without_lorentz_signature_is_rejected() -> None:
    audit = form_degree_closure(metric_signature=(1, 1, 1, 1))

    assert audit.one_time_direction
    assert not audit.nondegenerate_lorentzian_signature
    assert not audit.lorentzian_three_plus_one


def test_rank_four_simplex_strands_pair_ten_faces() -> None:
    audit = simplex_interaction_audit(4)

    assert audit.interaction_valence == 5
    assert audit.strand_ends == 20
    assert audit.paired_codimension_two_faces == 10
    assert audit.every_strand_paired_twice
    assert audit.boundary_euler_characteristic == 0
    assert audit.coordinate_free
    assert audit.target_four_simplex


def test_non_target_rank_remains_a_valid_but_different_simplex() -> None:
    audit = simplex_interaction_audit(3)

    assert audit.every_strand_paired_twice
    assert audit.boundary_euler_characteristic == 2
    assert not audit.target_four_simplex


def test_nontrivial_lorentz_face_holonomy_carries_curvature() -> None:
    angle = 0.3
    rotation = np.eye(4)
    rotation[1:3, 1:3] = (
        (math.cos(angle), -math.sin(angle)),
        (math.sin(angle), math.cos(angle)),
    )

    audit = face_holonomy_audit(
        (rotation, np.eye(4), np.eye(4)),
        face_id=(1, 2, 3),
        attached_contractible_face=True,
    )

    assert audit.maximum_lorentz_residual < 1.0e-12
    assert audit.flatness_residual > 0.1
    assert audit.nontrivial_curvature_carrier


def test_identity_face_holonomy_is_flat_negative_control() -> None:
    audit = face_holonomy_audit(
        (np.eye(4), np.eye(4), np.eye(4)),
        face_id=(1, 2, 3),
        attached_contractible_face=True,
    )

    assert audit.flatness_residual == pytest.approx(0.0)
    assert not audit.nontrivial_curvature_carrier


def test_unattached_nontrivial_loop_is_not_a_curvature_certificate() -> None:
    rotation = np.eye(4)
    rotation[1:3, 1:3] = ((0.0, -1.0), (1.0, 0.0))
    audit = face_holonomy_audit(
        (rotation,),
        face_id="loop",
        attached_contractible_face=False,
    )

    assert audit.flatness_residual > 0.0
    assert not audit.nontrivial_curvature_carrier
    assert audit.status == "UNATTACHED_LOOP_IS_NOT_A_CURVATURE_CERTIFICATE"


def test_planck_readout_is_an_equivalence_without_deleting_microstates() -> None:
    audit = planck_resolution_quotient(
        ((0.10, 0.20), (0.15, 0.25), (1.20, 0.10)),
        observable_dimensions=("L", "L^2"),
        reference_dimensions=("L", "L^2"),
    )

    assert audit.microscopic_history_count == 3
    assert audit.coarse_class_count == 2
    assert audit.folded_pair_count == 1
    assert audit.equivalence_reflexive
    assert audit.equivalence_symmetric
    assert audit.equivalence_transitive
    assert audit.all_microscopic_histories_retained
    assert audit.dimension_match


def test_planck_readout_rejects_mixed_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        planck_resolution_quotient(
            ((1.0, 2.0), (1.1, 2.1)),
            observable_dimensions=("L", "L^2"),
            reference_dimensions=("L^2", "L^2"),
        )


def test_orthogonal_environment_decoheres_and_preserves_folded_norm() -> None:
    probabilities = np.asarray((0.6, 0.25, 0.15))
    audit = decoherent_fold_audit(
        np.sqrt(probabilities),
        np.eye(3),
        ("rendered", "folded", "folded"),
        rendered_label="rendered",
    )

    assert audit.global_norm == pytest.approx(1.0)
    assert audit.reduced_trace == pytest.approx(1.0)
    assert audit.maximum_interclass_record_overlap == pytest.approx(0.0)
    assert audit.maximum_interclass_coherence == pytest.approx(0.0)
    assert audit.rendered_probability == pytest.approx(0.6)
    assert audit.folded_probability == pytest.approx(0.4)
    assert audit.decoherent
    assert audit.folded_sector_preserved


def test_identical_environment_records_do_not_decohere_classes() -> None:
    environment = np.ones((2, 1), dtype=complex)
    audit = decoherent_fold_audit(
        np.sqrt((0.5, 0.5)),
        environment,
        ("rendered", "folded"),
        rendered_label="rendered",
    )

    assert audit.maximum_interclass_coherence == pytest.approx(0.5)
    assert not audit.decoherent
    assert audit.folded_sector_preserved


def test_finite_constraint_selection_preserves_support_and_obeys_bound() -> None:
    audit = finite_constraint_concentration(
        (1.0, 2.0, 1.0),
        (0.0, 0.5, 1.0),
        inverse_temperature=10.0,
    )

    assert audit.zero_defect_count == 1
    assert audit.positive_defect_gap == pytest.approx(0.5)
    assert audit.good_probability > 0.98
    assert audit.bad_probability <= audit.exponential_bad_probability_bound
    assert audit.bound_holds
    assert audit.finite_beta_preserves_full_support
    assert all(probability > 0.0 for probability in audit.probabilities)


def test_constraint_selection_rejects_deleted_base_history() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        finite_constraint_concentration(
            (1.0, 0.0),
            (0.0, 1.0),
            inverse_temperature=2.0,
        )


def test_nondegenerate_stationary_phase_localizes_with_large_parameter() -> None:
    first = quadratic_stationary_phase_audit(
        np.diag((1.0, 2.0, -3.0)),
        gradient_at_candidate=(0.0, 0.0, 0.0),
        large_dimensionless_parameter=100.0,
        continuous_variable_domain="x in R^3",
        gauge_fixing="test gauge fixed",
        contour="real R^3",
    )
    second = quadratic_stationary_phase_audit(
        np.diag((1.0, 2.0, -3.0)),
        gradient_at_candidate=(0.0, 0.0, 0.0),
        large_dimensionless_parameter=400.0,
        continuous_variable_domain="x in R^3",
        gauge_fixing="test gauge fixed",
        contour="real R^3",
    )

    assert first.hessian_rank == 3
    assert first.hessian_signature == (2, 1)
    assert first.nondegenerate_stationary_sector
    assert second.localization_scale == pytest.approx(first.localization_scale / 2.0)
    assert second.leading_prefactor_magnitude == pytest.approx(
        first.leading_prefactor_magnitude / 8.0
    )


def test_singular_hessian_is_not_a_nondegenerate_saddle() -> None:
    audit = quadratic_stationary_phase_audit(
        np.diag((1.0, 0.0)),
        gradient_at_candidate=(0.0, 0.0),
        large_dimensionless_parameter=100.0,
        continuous_variable_domain="x in R^2",
        gauge_fixing="test gauge fixed",
        contour="real R^2",
    )

    assert audit.hessian_rank == 1
    assert not audit.nondegenerate_stationary_sector
    assert math.isinf(audit.localization_scale)


def test_nondegenerate_hessian_with_nonzero_gradient_is_not_stationary() -> None:
    audit = quadratic_stationary_phase_audit(
        np.eye(2),
        gradient_at_candidate=(0.1, 0.0),
        large_dimensionless_parameter=100.0,
        continuous_variable_domain="x in R^2",
        gauge_fixing="test gauge fixed",
        contour="real R^2",
    )

    assert audit.hessian_rank == 2
    assert audit.gradient_residual > 0.0
    assert not audit.nondegenerate_stationary_sector


def test_ir_power_counting_passes_and_fails_at_declared_scales() -> None:
    passed = ir_einstein_dominance_audit(
        1.0e-4,
        (1.0, 2.0),
        tolerance=1.0e-6,
    )
    failed = ir_einstein_dominance_audit(
        0.5,
        (1.0,),
        tolerance=1.0e-2,
    )

    assert passed.correction_ratios == pytest.approx((1.0e-8, 2.0e-16))
    assert passed.einstein_hilbert_dominates
    assert not failed.einstein_hilbert_dominates


def test_constant_curvature_endpoint_is_exact_four_dimensional_einstein() -> None:
    audit = constant_curvature_einstein_audit(0.125)

    assert audit.cosmological_constant_times_reference_length_squared == pytest.approx(
        0.375
    )
    assert audit.ricci_residual < 1.0e-14
    assert audit.scalar_curvature_residual < 1.0e-14
    assert audit.einstein_equation_residual < 1.0e-14
    assert audit.massless_spin_two_polarizations == 2
    assert not audit.two_dof_spectrum_derived_from_action
    assert audit.lorentzian_einstein_geometry


def test_one_typed_trace_generates_rank_faces_and_causal_face_map() -> None:
    trace = typed_rank_four_event_trace()

    assert trace.connected_two_cell_block
    assert trace.boundary_atom_occurrences == 10
    assert trace.strand_end_count == 40
    assert len(trace.unique_triangle_ids) == 16
    assert len(trace.causal_composition_faces) == 4
    assert trace.rank_four_pairing_consistent
    assert trace.causal_face_map_bijective
    assert trace.exact_typed_trace_probability > 0.0


def test_flat_chiral_plebanski_solution_is_exact_on_real_tetrad() -> None:
    coordinates = {
        0: (1.0, 0.2, 0.2, 0.2),
        1: (0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0),
        5: (-1.0, 0.2, 0.2, 0.2),
    }
    audit = flat_chiral_plebanski_audit(
        "test-history",
        vertex_coordinates=coordinates,
        simplex_cells=((0, 1, 2, 3, 4), (1, 2, 3, 4, 5)),
        shared_tetrahedron=(1, 2, 3, 4),
        selected_face_vectors=np.asarray(
            ((0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 1.0))
        ),
        selected_face_id=(2, 3, 4),
        selected_face_holonomy=np.eye(4),
    )

    assert audit.induced_by_selected_simplex_geometry
    assert audit.shared_face_embedding_residual == pytest.approx(0.0)
    assert audit.selected_holonomy_flatness_residual == pytest.approx(0.0)
    assert audit.complex_self_duality_residual < 1.0e-14
    assert audit.simplicity_tracefree_residual < 1.0e-14
    assert abs(audit.simplicity_volume) > 0.0
    assert audit.covariant_constancy_residual == 0.0
    assert audit.curvature_equation_residual == 0.0
    assert audit.flat_lorentzian_plebanski_solution


def test_flat_plebanski_copy_cannot_be_attached_to_mismatched_face_data() -> None:
    coordinates = {
        0: (1.0, 0.2, 0.2, 0.2),
        1: (0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0),
        4: (0.0, 0.0, 0.0, 1.0),
        5: (-1.0, 0.2, 0.2, 0.2),
    }
    mismatched = np.asarray(
        ((0.0, 1.1, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0),
         (0.0, 0.0, 0.0, 1.0))
    )
    audit = flat_chiral_plebanski_audit(
        "test-history",
        vertex_coordinates=coordinates,
        simplex_cells=((0, 1, 2, 3, 4), (1, 2, 3, 4, 5)),
        shared_tetrahedron=(1, 2, 3, 4),
        selected_face_vectors=mismatched,
        selected_face_id=(2, 3, 4),
        selected_face_holonomy=np.eye(4),
    )

    assert not audit.induced_by_selected_simplex_geometry
    assert not audit.flat_lorentzian_plebanski_solution


def test_constructive_chain_uses_one_trace_and_closes_finite_flat_case() -> None:
    audit = constructive_zerod_to_plebanski_witness()

    assert audit.form_degree.lorentzian_three_plus_one
    assert audit.simplex_interaction.target_four_simplex
    assert len(audit.typed_trace.causal_composition_faces) == 4
    assert audit.typed_trace.causal_face_map_bijective
    assert audit.face_holonomies[0].flatness_residual == pytest.approx(0.0)
    assert any(
        holonomy.nontrivial_curvature_carrier
        for holonomy in audit.face_holonomies[1:]
    )
    assert audit.causal_relation_realized_by_metric
    assert audit.planck_quotient.folded_pair_count >= 1
    assert audit.decoherence.decoherent
    assert audit.decoherence.class_record_map_consistent
    assert audit.constraint_concentration.bound_holds
    assert audit.stationary_phase.nondegenerate_stationary_sector
    assert audit.bivector_reconstruction_status == (
        "FINITE_LINEAR_SIMPLE_FACE_RECONSTRUCTED"
    )
    assert audit.selected_shared_face_status == "FINITE_SHARED_SPACELIKE_FACE_MATCH"
    assert audit.flat_plebanski.flat_lorentzian_plebanski_solution
    assert audit.all_finite_projections_share_one_trace
    assert audit.folded_possibilities_preserved
    assert audit.single_history_finite_flat_witness_closed
    assert audit.conditional_local_plebanski_einstein_existence_closed
    assert not audit.continuum_refinement_derived
    assert not audit.two_dof_ir_spectrum_derived
    assert not audit.bare_zerod_uniqueness_proved


def test_every_member_uses_same_trace_and_nonselected_members_keep_support() -> None:
    audit = constructive_zerod_to_plebanski_witness(history_id="linked-history")

    assert all(
        member.member_id.startswith("linked-history:")
        for member in audit.history_members
    )
    assert all(
        member.shared_tetrahedron == audit.typed_trace.shared_tetrahedron
        for member in audit.history_members
    )
    assert all(probability > 0.0 for probability in audit.constraint_concentration.probabilities)
    assert audit.status == (
        "SINGLE_TYPED_HISTORY_FINITE_FLAT_CONDITIONAL_WITNESS_CLOSED"
    )


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    (
        (simplex_interaction_audit, (True,), "rank"),
        (constant_curvature_einstein_audit, (math.inf,), "finite"),
    ),
)
def test_invalid_inputs_are_rejected(function, arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        function(*arguments)


def test_empty_planck_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        planck_resolution_quotient(
            (),
            observable_dimensions=("L",),
            reference_dimensions=("L",),
        )
