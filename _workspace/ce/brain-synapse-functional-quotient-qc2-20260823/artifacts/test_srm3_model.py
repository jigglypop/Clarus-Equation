from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("srm3_model.py")
SPEC = importlib.util.spec_from_file_location("srm3_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model
SPEC.loader.exec_module(model)


def test_preprocessor_is_train_whitened_and_unknown_safe():
    numeric = np.arange(60, dtype=float).reshape(20, 3)
    numeric[0, 1] = np.nan
    categorical = np.asarray([["a" if i % 2 else "b"] for i in range(20)], dtype=object)
    params, coordinates = model.fit_preprocessor(numeric, categorical, max_dimension=4)
    dimension = coordinates.shape[1]
    transformed = model.transform_preprocessor(
        params, numeric, categorical, dimension=dimension
    )
    assert np.allclose(coordinates, transformed)
    assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1e-10)
    assert np.allclose(
        np.cov(transformed, rowvar=False), np.eye(dimension), atol=1e-10
    )
    unseen = model.transform_preprocessor(
        params,
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([["new"]], dtype=object),
        dimension,
    )
    assert unseen.shape == (1, dimension)
    assert np.all(np.isfinite(unseen))


def test_krr_matches_dense_system_and_jacobian_finite_difference():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(18, 3))
    y = rng.normal(size=(18, 4))
    scaler = model.fit_output_scaler(y)
    ys = model.standardize_target(y, scaler)
    fitted = model.fit_krr(x, ys, ell=1.3, ridge=1e-2, output_scaler=scaler)
    expected_alpha = np.linalg.solve(
        model.rbf_kernel(x, x, 1.3) + 1e-2 * np.eye(len(x)), ys
    )
    assert np.allclose(fitted.alpha, expected_alpha)
    q = rng.normal(size=(2, 3))
    analytic = model.jacobian_krr(fitted, q)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for j in range(3):
        plus = q.copy()
        minus = q.copy()
        plus[:, j] += step
        minus[:, j] -= step
        numeric[:, :, j] = (
            model.predict_krr(fitted, plus) - model.predict_krr(fitted, minus)
        ) / (2.0 * step)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_covariance_score_and_pullback_are_well_formed():
    rng = np.random.default_rng(8)
    residual = rng.normal(size=(40, 5))
    sample = model.residual_covariance(residual)
    covariance, floor = model.floored_covariance(model.shrink_covariance(sample, 0.5))
    assert floor > 0.0
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    score = model.gaussian_log_score(residual, np.zeros_like(residual), covariance)
    assert score.shape == (40,)
    jacobian = rng.normal(size=(7, 5, 3))
    metrics = model.pullback_metrics(jacobian, covariance)
    assert metrics.shape == (7, 3, 3)
    assert np.min(np.linalg.eigvalsh(metrics)) > -1e-10


def test_rank_bound_and_secant_symmetry():
    rng = np.random.default_rng(10)
    jacobian = rng.normal(size=(6, 4, 7))
    ranks, fifth = model.numerical_ranks(jacobian)
    assert np.all(ranks <= 4)
    assert np.all(fifth == 0.0)
    covariance = np.eye(4)
    metrics = model.pullback_metrics(jacobian, covariance)
    points = rng.normal(size=(6, 7))
    distance = model.secant_squared(points, points, metrics, metrics)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance >= 0.0)


def test_affine_metric_jacobian_and_generalized_spectrum_transport():
    rng = np.random.default_rng(12)
    j = rng.normal(size=(5, 3))
    g = j.T @ j
    reference = np.eye(3)
    transform = np.asarray([[2.0, 0.2, 0.0], [0.0, 0.75, 0.1], [0.0, 0.0, 1.2]])
    jt = model.transport_jacobian(j, transform)
    gt = model.transport_metric(g, transform)
    rt = model.transport_metric(reference, transform)
    assert np.allclose(jt.T @ jt, gt)
    assert np.allclose(
        model.generalized_spectrum(g, reference),
        model.generalized_spectrum(gt, rt),
    )
    vector = rng.normal(size=3)
    transformed_vector = transform @ vector
    assert np.allclose(vector @ g @ vector, transformed_vector @ gt @ transformed_vector)


def test_neighbor_abstention_and_slice_equal_score():
    distances = np.asarray([[0.0, 1.0], [100.0, 101.0]])
    target = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    prediction, effective, supported = model.neighbor_predict(
        distances, target, rho=1.0, minimum_effective_neighbors=1.5
    )
    assert supported[0]
    assert np.all(np.isfinite(prediction[0]))
    delta, se, groups = model.slice_equal_delta(
        np.asarray([2.0, 4.0, 8.0]),
        np.asarray([1.0, 1.0, 2.0]),
        ["a", "a", "b"],
    )
    assert groups == 2
    assert np.isclose(delta, ((1.0 + 3.0) / 2.0 + 6.0) / 2.0)
    assert se >= 0.0


def test_covariance_and_rho_guards_reject_invalid_inputs():
    with pytest.raises(model.ModelFailure):
        model.shrink_covariance(np.ones((2, 3)), 0.5)
    with pytest.raises(model.ModelFailure):
        model.floored_covariance(np.asarray([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(model.ModelFailure):
        model.neighbor_predict(np.zeros((1, 2)), np.zeros((2, 1)), rho=0.0)
