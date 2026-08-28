from __future__ import annotations

import pytest

from examples.physics.causal_face_simplicity import composition_faces
from examples.physics.causal_gr_action_no_go import (
    action_underdetermination_audit,
    bf_simplicity_counterexample,
    flat_action,
    flat_gradient,
    measure_underdetermination_audit,
    quadratic_action,
    quadratic_gradient,
)


def test_same_composition_faces_admit_inequivalent_invariant_actions() -> None:
    faces = composition_faces(
        {(0, 1), (1, 3), (0, 2), (2, 3)},
        {(0, 3)},
    )
    audit = action_underdetermination_audit(len(faces), coupling=0.7)

    assert audit.same_incidence
    assert audit.same_normalized_measure
    assert audit.both_relabel_invariant
    assert audit.flat_stationary_dimension == len(faces)
    assert audit.quadratic_stationary_dimension == 0
    assert audit.flat_hessian_rank == 0
    assert audit.quadratic_hessian_rank == len(faces)
    assert audit.flat_partition_amplitude == 1.0 + 0.0j
    assert audit.quadratic_partition_amplitude != audit.flat_partition_amplitude
    assert not audit.unique_action_selected
    assert audit.status == "BARE_INCIDENCE_CANNOT_SELECT_ACTION"
    assert audit.claim_ceiling == "FINITE_REGULATED_ACTION_UNDERDETERMINATION"


def test_stationary_sets_are_constructively_different() -> None:
    nonzero = (0.3, -0.8, 1.1)
    origin = (0.0, 0.0, 0.0)

    assert flat_action(nonzero) == 0.0
    assert flat_gradient(nonzero) == origin
    assert quadratic_action(nonzero, coupling=2.0) > 0.0
    assert quadratic_gradient(nonzero, coupling=2.0) != origin
    assert quadratic_gradient(origin, coupling=2.0) == origin


def test_quadratic_action_is_invariant_under_a_nontrivial_permutation() -> None:
    labels = (0.2, -0.7, 1.3, 0.4)
    permuted = (labels[2], labels[0], labels[3], labels[1])

    assert quadratic_action(labels, coupling=0.9) == pytest.approx(
        quadratic_action(permuted, coupling=0.9)
    )


@pytest.mark.parametrize("face_count", [0, -1, True])
def test_no_go_requires_a_nonempty_finite_face_set(face_count: object) -> None:
    with pytest.raises(ValueError, match="face_count"):
        action_underdetermination_audit(face_count)  # type: ignore[arg-type]


def test_zero_coupling_is_rejected_because_it_erases_the_countermodel() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        action_underdetermination_audit(3, coupling=0.0)


def test_four_face_bf_saddle_is_rejected_by_linear_simplicity() -> None:
    faces = composition_faces(
        {
            ("u", "m0"),
            ("m0", "v"),
            ("u", "m1"),
            ("m1", "v"),
            ("u", "m2"),
            ("m2", "v"),
            ("u", "m3"),
            ("m3", "v"),
        },
        {("u", "v")},
    )
    audit = bf_simplicity_counterexample()

    assert len(faces) == audit.face_count == 4
    assert audit.identity_holonomy_flatness_residual == 0.0
    assert audit.internal_edge_closure_residual == 0.0
    assert audit.common_simplicity_normal_nullity == 0
    assert audit.finite_bf_saddle_conditions
    assert not audit.linear_simplicity_sector
    assert not audit.same_incidence_selects_constrained_gravity
    assert audit.status == "BF_SADDLE_REJECTED_BY_LINEAR_SIMPLICITY"
    assert audit.claim_ceiling == "FINITE_BF_VS_SIMPLICITY_COUNTEREXAMPLE"


def test_relabel_invariance_does_not_select_a_unique_normalized_measure() -> None:
    audit = measure_underdetermination_audit(
        4,
        first_variance=1.0,
        second_variance=2.5,
    )

    assert audit.both_normalized
    assert audit.both_relabel_invariant
    assert audit.first_total_second_moment == pytest.approx(4.0)
    assert audit.second_total_second_moment == pytest.approx(10.0)
    assert not audit.unique_measure_selected
    assert audit.status == "BARE_INCIDENCE_CANNOT_SELECT_MEASURE"
    assert audit.claim_ceiling == "FINITE_NORMALIZED_MEASURE_UNDERDETERMINATION"


def test_measure_countermodel_requires_distinct_positive_variances() -> None:
    with pytest.raises(ValueError, match="distinct"):
        measure_underdetermination_audit(2, first_variance=1.0, second_variance=1.0)
    with pytest.raises(ValueError, match="positive"):
        measure_underdetermination_audit(2, first_variance=0.0, second_variance=1.0)
