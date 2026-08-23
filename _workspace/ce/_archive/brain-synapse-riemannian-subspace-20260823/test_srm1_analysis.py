from __future__ import annotations

import math

import numpy as np

from srm1_analysis import (
    ResponseModel,
    build_graph,
    fixed_gauge_transforms,
    gauge_audit,
    query_distances,
    stable_split,
    symmetric_knn_adjacency,
    worst_support_eigenvalue_ratio,
)
from run_analysis import raw_z


def test_stable_split_is_exact_sha256_first_byte_rule() -> None:
    import hashlib

    for group in ("slice-a", "slice-b", "2019-01-01_001"):
        bucket = hashlib.sha256(group.encode("utf-8")).digest()[0] % 10
        expected = "train" if bucket <= 5 else "development" if bucket <= 7 else "confirmation"
        assert stable_split(group) == expected


def test_quadratic_response_jacobian_matches_finite_difference() -> None:
    rng = np.random.default_rng(80421)
    coefficient = rng.normal(size=(15, 4))
    model = ResponseModel(coefficient, np.array([0.7, 1.1, 1.3, 0.9]))
    x = rng.normal(size=(5, 4))
    analytic = model.jacobians(x)
    epsilon = 1e-6
    numeric = np.empty_like(analytic)
    for coordinate in range(4):
        step = np.zeros(4)
        step[coordinate] = epsilon
        numeric[:, :, coordinate] = (
            model.predict(x + step) - model.predict(x - step)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-9, atol=2e-9)


def test_pullback_metric_is_spd_for_identity_full_rank_map() -> None:
    coefficient = np.zeros((15, 4))
    coefficient[1:5] = np.eye(4)
    residual_variance = np.array([0.5, 1.0, 2.0, 4.0])
    model = ResponseModel(coefficient, residual_variance)
    metric = model.metrics(np.zeros((3, 4)))
    expected = np.diag(1.0 / residual_variance)
    np.testing.assert_allclose(metric, np.repeat(expected[None, :, :], 3, axis=0))
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)


def test_rank_bootstrap_statistic_uses_worst_support_point() -> None:
    eigenvalues = np.array(
        [
            [1e-8, 1.0, 2.0, 4.0],
            [0.2, 0.4, 0.8, 1.0],
            [0.5, 0.6, 0.7, 1.0],
        ]
    )
    assert worst_support_eigenvalue_ratio(eigenvalues) == 2.5e-9


def test_query_attachment_does_not_create_query_query_paths() -> None:
    rng = np.random.default_rng(80422)
    x_train = rng.normal(size=(40, 4))
    metrics = np.repeat(np.eye(4)[None, :, :], len(x_train), axis=0)
    adjacency = symmetric_knn_adjacency(x_train, 8)
    graph = build_graph(x_train, metrics, 8, adjacency=adjacency)
    x_query = rng.normal(size=(2, 4))
    query_metrics = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    together, neighbors = query_distances(
        x_train, metrics, graph, x_query, query_metrics, 8
    )
    for row in range(2):
        alone, _ = query_distances(
            x_train,
            metrics,
            graph,
            x_query[row : row + 1],
            query_metrics[row : row + 1],
            8,
            neighbor_indices=neighbors[row : row + 1],
        )
        np.testing.assert_allclose(together[row], alone[0], rtol=0.0, atol=0.0)


def test_fixed_gauge_suite_and_transport_invariance() -> None:
    rng = np.random.default_rng(80423)
    x_train = rng.normal(size=(48, 4))
    x_query = rng.normal(size=(6, 4))
    y_train = rng.normal(size=(48, 4))
    coefficient = np.zeros((15, 4))
    coefficient[1:5] = np.eye(4)
    model = ResponseModel(coefficient, np.ones(4))
    assert len(fixed_gauge_transforms()) == 64
    audit = gauge_audit(
        x_train,
        y_train,
        x_query,
        model,
        k=8,
        bandwidth_multiplier=1.0,
    )
    assert audit["status"] == "PASS", audit
    assert audit["transforms_tested"] == 64


def test_strict_chart_uses_only_dimensionless_si_ratios_inside_log() -> None:
    row = {
        "psp_amplitude": -2e-3,
        "distance": 200e-6,
        "input_resistance": 100e6,
        "tau": 20e-3,
    }
    observed = raw_z([row], r_reference=1e-3)[0]
    expected = np.log([2.0, 200e-6 / 1.0, 100e6 / 1.0, 20e-3 / 1.0])
    np.testing.assert_allclose(observed, expected)
    assert math.isfinite(float(np.sum(observed)))
