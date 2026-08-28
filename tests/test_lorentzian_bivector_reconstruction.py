from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_face_simplicity import (
    hard_shared_spacelike_face_match,
)
from examples.physics.lorentzian_bivector_reconstruction import (
    bivector_face_reconstruction_audit,
    bivector_from_normal_edge,
    common_linear_simplicity_nullity,
    hodge_dual,
)


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
