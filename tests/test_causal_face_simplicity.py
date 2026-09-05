from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.gravity.causal_face_simplicity import (
    CompositionFace,
    composition_faces,
    cross_simplicity_residual,
    face_incidence_audit,
    face_simplicity_verdict,
    fan_euler_characteristic,
    geometric_self_dual_triple,
    hard_shared_spacelike_face_match,
    maximum_poisson_exact_valence_probability,
    minimum_block_depth,
    proper_orthochronous_residual,
    random_tetrad_block_audit,
    simplicity_block_audit,
    simplicity_residual,
    soft_block_simplicity_weight,
)


D = 3.1777584234


def _lorentzian_shared_face_fixture() -> tuple[np.ndarray, ...]:
    left_face = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    left_normal = np.array([1.0, 0.0, 0.0, 0.0])
    left_apex = np.array([2.0, 0.2, 0.1, 0.0])
    rapidity = 0.7
    cosine = math.cosh(rapidity)
    sine = math.sinh(rapidity)
    right_to_left = np.array(
        [
            [cosine, sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    left_to_right = np.linalg.inv(right_to_left)
    right_face = left_face @ left_to_right.T
    right_normal = left_to_right @ left_normal
    desired_right_apex = np.array([-1.5, 0.3, 0.0, 0.2])
    right_apex = left_to_right @ desired_right_apex
    return (
        left_face,
        left_normal,
        left_apex,
        right_face,
        right_normal,
        right_apex,
        right_to_left,
    )


def test_composition_face_is_canonical_factorization_triangle() -> None:
    fine = {("u", "a"), ("a", "v"), ("u", "b"), ("b", "v")}
    coarse = {("u", "v")}

    faces = composition_faces(fine, coarse)

    assert faces == (
        CompositionFace("u", "a", "v"),
        CompositionFace("u", "b", "v"),
    )
    assert all(face.oriented_boundary[-1] == ("v", "u") for face in faces)
    assert fan_euler_characteristic(len(faces)) == 1


def test_composition_faces_are_equivariant_under_relabeling() -> None:
    fine = {(0, 1), (1, 3), (0, 2), (2, 3)}
    coarse = {(0, 3)}
    relabel = {0: "z", 1: "x", 2: "y", 3: "w"}

    original = composition_faces(fine, coarse)
    mapped = {
        CompositionFace(
            relabel[face.source],
            relabel[face.middle],
            relabel[face.target],
        )
        for face in original
    }
    relabeled_faces = set(
        composition_faces(
            {(relabel[u], relabel[v]) for u, v in fine},
            {(relabel[u], relabel[v]) for u, v in coarse},
        )
    )

    assert mapped == relabeled_faces


def test_causal_cycles_are_rejected() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        composition_faces({(0, 1), (1, 0)}, {(0, 1)})


def test_ce_face_incidence_requires_a_coarse_block() -> None:
    one = face_incidence_audit(D, 1)
    four = face_incidence_audit(D, 4)

    assert one.expected_faces == pytest.approx(D - 1.0)
    assert one.probability_at_least_minimum == pytest.approx(
        0.17629188622245506,
        rel=1.0e-12,
    )
    assert four.probability_at_least_minimum == pytest.approx(
        0.9739978016710592,
        rel=1.0e-12,
    )
    assert minimum_block_depth(D, confidence=0.95) == 4
    assert minimum_block_depth(D, confidence=0.99) == 5


def test_raw_poisson_cannot_concentrate_on_exact_tetrahedral_valence() -> None:
    maximum = maximum_poisson_exact_valence_probability(4)

    assert maximum == pytest.approx(
        math.exp(-4.0) * 4.0**4 / math.factorial(4),
        rel=1.0e-15,
    )
    assert maximum < 0.20


def test_geometric_self_dual_triple_satisfies_plebanski_simplicity() -> None:
    triple = geometric_self_dual_triple(np.eye(4))

    assert simplicity_residual(triple) < 1.0e-14


def test_conformal_aligned_cells_remain_simple_after_blocking() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    second = geometric_self_dual_triple(1.7 * np.eye(4))
    audit = simplicity_block_audit(first, second)

    assert audit.first_local_residual < 1.0e-14
    assert audit.second_local_residual < 1.0e-14
    assert audit.cross_residual < 1.0e-14
    assert audit.blocked_residual < 1.0e-14
    assert audit.local_simplicity_sufficient


def test_individually_simple_mismatched_cells_fail_block_simplicity() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    mismatched_tetrad = np.array(
        [
            [0.74341184, 0.80016662, 0.30490388, -0.47971556],
            [0.02980649, 1.23067583, -0.07551285, 0.27316411],
            [-0.02660693, 0.26689902, 1.57540904, -0.27026490],
            [0.08125544, -0.18532303, 0.05090736, 0.52512219],
        ]
    )
    second = geometric_self_dual_triple(mismatched_tetrad)
    audit = simplicity_block_audit(first, second)

    assert audit.first_local_residual < 1.0e-12
    assert audit.second_local_residual < 1.0e-12
    assert audit.cross_residual > 0.03
    assert audit.blocked_residual > 0.10
    assert not audit.local_simplicity_sufficient


def test_soft_simplicity_weight_penalizes_shape_mismatch() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    aligned = geometric_self_dual_triple(1.2 * np.eye(4))
    mismatch = geometric_self_dual_triple(
        np.array(
            [
                [1.0, 0.4, 0.2, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, 0.7],
            ]
        )
    )

    assert soft_block_simplicity_weight(first, aligned, width=0.05) == pytest.approx(1.0)
    assert soft_block_simplicity_weight(first, mismatch, width=0.05) < 0.95


def test_cross_residual_is_exact_obstruction_for_two_simple_cells() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    second = geometric_self_dual_triple(
        np.array(
            [
                [1.0, 0.3, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, 0.8],
            ]
        )
    )

    assert simplicity_residual(first) < 1.0e-14
    assert simplicity_residual(second) < 1.0e-14
    assert cross_simplicity_residual(first, second) > 0.0
    assert simplicity_residual(first + second) > 0.0


def test_random_geometric_cells_generically_leave_the_simple_block_sector() -> None:
    audit = random_tetrad_block_audit()

    assert audit.sample_count == 1_000
    assert audit.fraction_above_tolerance == pytest.approx(1.0)
    assert audit.median_residual == pytest.approx(0.08707009, rel=1.0e-6)
    assert audit.ninety_percent_residual > audit.median_residual


def test_verdict_records_both_topology_and_simplicity_obstructions() -> None:
    verdict = face_simplicity_verdict(D)

    assert verdict.canonical_face_attachment
    assert verdict.block_depth_95_percent == 4
    assert verdict.block_depth_99_percent == 5
    assert not verdict.raw_poisson_simplicial_concentration_possible
    assert not verdict.local_simplicity_closed_under_blocking
    assert "shape-matching" in verdict.remaining_obligation


def test_hard_shared_spacelike_face_match_is_boost_invariant() -> None:
    fixture = _lorentzian_shared_face_fixture()
    audit = hard_shared_spacelike_face_match(*fixture)

    assert proper_orthochronous_residual(fixture[-1]) < 3.0e-16
    np.testing.assert_allclose(audit.left_gram, np.diag((1.0, 4.0, 9.0)))
    np.testing.assert_allclose(audit.right_gram, audit.left_gram, atol=1.0e-14)
    assert audit.left_wedge_determinant == pytest.approx(-12.0)
    assert audit.right_wedge_determinant == pytest.approx(9.0)
    assert audit.left_lapse == pytest.approx(-2.0)
    assert audit.right_lapse == pytest.approx(1.5)
    assert audit.hard_match
    assert audit.status == "FINITE_SHARED_SPACELIKE_FACE_MATCH"
    assert audit.plebanski_branch == "NOT_TESTED_BY_FACE_GRAM"
    assert audit.claim_ceiling == "FINITE_CONDITIONAL_SHARED_SPACELIKE_FACE_ONLY"


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("degenerate_wedge", "DEGENERATE_CELL_WEDGE"),
        ("nonspacelike", "NONSPACELIKE_OR_DEGENERATE_FACE"),
        ("invalid_normal", "INVALID_FUTURE_FACE_NORMAL"),
        ("invalid_transport", "NON_PROPER_OR_NON_ORTHOCHRONOUS_TRANSPORT"),
        ("incompatible_normal", "INCOMPATIBLE_FACE_NORMALS"),
        ("shape_mismatch", "SHAPE_MISMATCH"),
        ("orientation_reversal", "ORIENTATION_REVERSING_FACE_MAP"),
        ("tangent_transport", "INCOMPATIBLE_FACE_TANGENT_TRANSPORT"),
        ("same_side", "SAME_SIDE_APEX_CONFIGURATION"),
    ],
)
def test_hard_shared_face_match_rejects_each_missing_hypothesis(
    case: str,
    expected_status: str,
) -> None:
    data = list(_lorentzian_shared_face_fixture())
    left_face, left_normal, _, _, _, _, right_to_left = data
    left_to_right = np.linalg.inv(right_to_left)

    if case == "degenerate_wedge":
        data[2] = left_face[0].copy()
    elif case == "nonspacelike":
        bad_left_gauge_face = left_face.copy()
        bad_left_gauge_face[0] = np.array([2.0, 1.0, 0.0, 0.0])
        data[3] = bad_left_gauge_face @ left_to_right.T
    elif case == "invalid_normal":
        data[4] = -data[4]
    elif case == "invalid_transport":
        data[6] = np.diag((-1.0, 1.0, 1.0, 1.0))
    elif case == "incompatible_normal":
        data[3] = left_face.copy()
        data[4] = left_normal.copy()
        data[5] = np.array([-1.5, 0.3, 0.0, 0.2])
    elif case == "shape_mismatch":
        data[3] = data[3].copy()
        data[3][2] *= 1.1
    elif case == "orientation_reversal":
        data[3] = data[3].copy()
        data[3][0] *= -1.0
    elif case == "tangent_transport":
        spatial_rotation = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rotated_left_gauge = left_face @ spatial_rotation.T
        data[3] = rotated_left_gauge @ left_to_right.T
    elif case == "same_side":
        data[5] = left_to_right @ np.array([1.5, 0.3, 0.0, 0.2])
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(case)

    audit = hard_shared_spacelike_face_match(*data)

    assert not audit.hard_match
    assert audit.status == expected_status
    assert audit.plebanski_branch == "NOT_TESTED_BY_FACE_GRAM"
    assert "GR" not in audit.status
    assert "CONTINUUM" not in audit.status


def test_hard_shared_face_match_fails_closed_on_malformed_data() -> None:
    data = list(_lorentzian_shared_face_fixture())
    data[0] = np.zeros((2, 4))

    with pytest.raises(ValueError, match="left_face must have shape"):
        hard_shared_spacelike_face_match(*data)


@pytest.mark.parametrize(
    "length_scale", [1.0e-200, 1.0e-100, 1.0e-12, 1.0e-3, 1.0e3, 1.0e200]
)
def test_hard_shared_face_match_is_invariant_under_length_units(
    length_scale: float,
) -> None:
    data = list(_lorentzian_shared_face_fixture())
    for index in (0, 2, 3, 5):
        data[index] = data[index] * length_scale

    audit = hard_shared_spacelike_face_match(*data)

    assert audit.hard_match
    assert audit.status == "FINITE_SHARED_SPACELIKE_FACE_MATCH"


@pytest.mark.parametrize("length_scale", [1.0e-6, 1.0, 1.0e6])
def test_shape_mismatch_status_is_invariant_under_length_units(
    length_scale: float,
) -> None:
    data = list(_lorentzian_shared_face_fixture())
    data[3] = data[3].copy()
    data[3][2] *= 1.1
    for index in (0, 2, 3, 5):
        data[index] = data[index] * length_scale

    audit = hard_shared_spacelike_face_match(*data)

    assert not audit.hard_match
    assert audit.status == "SHAPE_MISMATCH"
