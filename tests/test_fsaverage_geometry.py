import numpy as np
from reality_stone.clarus.fsaverage_geometry import (
    _anisotropic_laplace_beltrami_features,
    _apply_laplace_beltrami,
    _cotangent_operator,
    _curvature_guided_laplace_beltrami_features,
    _laplace_beltrami_features,
    _ridge_predict,
)


def test_ridge_recovers_linear_signal() -> None:
    x = np.arange(20, dtype=float)[:, None]
    y = 2.0 * x[:, 0] + 1.0
    prediction = _ridge_predict(x, y, x, 1e-8)
    assert np.max(np.abs(prediction - y)) < 1e-6


def _square_mesh() -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return coordinates, faces


def test_cotangent_laplace_beltrami_preserves_constants_and_integral() -> None:
    coordinates, faces = _square_mesh()
    mass, left, right, weight = _cotangent_operator(coordinates, faces)
    constant = _apply_laplace_beltrami(
        np.ones(4), mass, left, right, weight
    )
    field = np.array([0.0, 1.0, 3.0, 2.0])
    laplacian = _apply_laplace_beltrami(field, mass, left, right, weight)
    assert np.all(mass > 0.0)
    assert np.all(weight >= 0.0)
    assert np.max(np.abs(constant)) < 1e-12
    assert abs(float(mass @ laplacian)) < 1e-12


def test_laplace_beltrami_heat_features_are_finite() -> None:
    coordinates, faces = _square_mesh()
    curvature = np.array([-1.0, 1.0, 1.0, -1.0])
    features = _laplace_beltrami_features(
        coordinates, curvature, faces, heat_cfl=0.45, heat_steps=[1, 4, 16]
    )
    assert features.shape == (4, 6)
    assert np.all(np.isfinite(features))
    assert np.all(features[:, 1] >= 0.0)


def test_laplace_beltrami_features_ignore_rigid_embedding_transform() -> None:
    coordinates, faces = _square_mesh()
    curvature = np.array([-1.0, 1.0, 1.0, -1.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = coordinates @ rotation.T + np.array([4.0, -3.0, 2.0])
    original_features = _laplace_beltrami_features(
        coordinates, curvature, faces, heat_cfl=0.45, heat_steps=[1, 4, 16]
    )
    transformed_features = _laplace_beltrami_features(
        transformed, curvature, faces, heat_cfl=0.45, heat_steps=[1, 4, 16]
    )
    assert np.allclose(original_features, transformed_features, atol=1e-12)


def test_anisotropic_features_are_rigid_and_direction_sign_invariant() -> None:
    coordinates, faces = _square_mesh()
    curvature = np.array([-1.0, 1.0, 1.0, -1.0])
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    transformed = coordinates @ rotation.T + np.array([-2.0, 5.0, 7.0])
    original_features = _anisotropic_laplace_beltrami_features(
        coordinates,
        curvature,
        faces,
        anisotropy=0.75,
        heat_cfl=0.45,
        heat_steps=[1, 4, 16],
    )
    transformed_features = _anisotropic_laplace_beltrami_features(
        transformed,
        curvature,
        faces,
        anisotropy=0.75,
        heat_cfl=0.45,
        heat_steps=[1, 4, 16],
    )
    assert np.allclose(original_features, transformed_features, atol=1e-10)


def test_curvature_guided_features_are_rigid_invariant_and_finite() -> None:
    coordinates, faces = _square_mesh()
    curvature = np.array([-1.0, 1.0, 1.0, -1.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = coordinates @ rotation.T + np.array([3.0, 2.0, -4.0])
    arguments = {
        "curvature": curvature,
        "faces": faces,
        "conductivity_floor": 0.1,
        "contrast_scales": [0.5, 2.0],
        "heat_cfl": 0.45,
        "heat_steps": [1, 4, 16],
    }
    original = _curvature_guided_laplace_beltrami_features(
        coordinates=coordinates, **arguments
    )
    moved = _curvature_guided_laplace_beltrami_features(
        coordinates=transformed, **arguments
    )
    assert original.shape == (4, 12)
    assert np.all(np.isfinite(original))
    assert np.allclose(original, moved, atol=1e-12)
