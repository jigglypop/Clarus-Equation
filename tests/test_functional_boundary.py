import numpy as np

from reality_stone.clarus.functional_boundary import (
    _heat_landmark_coordinates,
    _mesh_edges,
    _roc_auc,
    _weighted_ridge_scores,
)


def test_mesh_edges_are_unique_and_undirected() -> None:
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    edges = _mesh_edges(faces)
    assert edges.shape == (5, 2)
    assert np.all(edges[:, 0] < edges[:, 1])


def test_auc_and_balanced_ridge_separate_simple_classes() -> None:
    features = np.arange(20, dtype=float)[:, None]
    target = np.where(features[:, 0] >= 10.0, 1.0, -1.0)
    scores = _weighted_ridge_scores(features, target, features, penalty=1e-6)
    assert _roc_auc(target > 0, scores) == 1.0


def test_heat_landmark_coordinates_are_rigid_invariant() -> None:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original = _heat_landmark_coordinates(coordinates, faces, 2, 0.45, [1, 4])
    transformed = _heat_landmark_coordinates(
        coordinates @ rotation.T + np.array([5.0, -2.0, 3.0]),
        faces,
        2,
        0.45,
        [1, 4],
    )
    assert original.shape == (4, 4)
    assert np.allclose(original, transformed, atol=1e-12)
