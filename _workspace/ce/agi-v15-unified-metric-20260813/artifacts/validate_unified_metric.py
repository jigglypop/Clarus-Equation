from __future__ import annotations

import json

import numpy as np

from reality_stone.clarus.unified_metric import (
    UnifiedMetricConfig,
    UnifiedMetricCore,
    affine_chart_change,
)


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def diamond() -> tuple[np.ndarray, np.ndarray]:
    points = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    return points, adjacency


def random_spd(rng: np.random.Generator, count: int) -> np.ndarray:
    result = np.empty((count, 2, 2))
    for index in range(count):
        basis, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        values = rng.uniform(0.4, 3.5, size=2)
        result[index] = basis @ np.diag(values) @ basis.T
    return result


def random_jacobian(rng: np.random.Generator) -> np.ndarray:
    left, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    right, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    scales = rng.uniform(0.4, 2.5, size=2)
    return left @ np.diag(scales) @ right


def main() -> None:
    rng = np.random.default_rng(150013)
    points, adjacency = diamond()
    core = UnifiedMetricCore(points, adjacency)
    maximum_local_error = 0.0
    maximum_edge_error = 0.0
    maximum_path_error = 0.0
    trials = 128
    for _ in range(trials):
        metric = random_spd(rng, len(points))
        state = core.make_state(metric)
        jacobian = random_jacobian(rng)
        offset = rng.normal(size=2)
        transformed_points, transformed_metric = affine_chart_change(
            points,
            metric,
            jacobian,
            offset,
        )
        transformed = UnifiedMetricCore(transformed_points, adjacency)
        transformed_state = transformed.make_state(transformed_metric)
        displacement = rng.normal(size=2)
        local_before = core.local_length_squared(state, 0, displacement)
        local_after = transformed.local_length_squared(
            transformed_state,
            0,
            jacobian @ displacement,
        )
        maximum_local_error = max(
            maximum_local_error,
            relative_error(local_before, local_after),
        )
        edges_before = core.edge_lengths(state)
        edges_after = transformed.edge_lengths(transformed_state)
        mask = np.isfinite(edges_before)
        maximum_edge_error = max(
            maximum_edge_error,
            float(
                np.max(
                    np.abs(edges_before[mask] - edges_after[mask])
                    / np.maximum(1.0, np.abs(edges_before[mask]))
                )
            ),
        )
        path_before = core.shortest_path(state, 0, 3).cost
        path_after = transformed.shortest_path(transformed_state, 0, 3).cost
        maximum_path_error = max(
            maximum_path_error,
            relative_error(path_before, path_after),
        )

    identity = core.identity_state()
    identity_goal = core.minimum_cost_targets(identity, 0, [2, 1])
    identity_path = core.shortest_path(identity, 0, 3)
    barrier_metric = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    barrier_metric[1] = 4.0 * np.eye(2)
    barrier = core.make_state(barrier_metric)
    barrier_goal = core.minimum_cost_targets(barrier, 0, [1, 2])
    barrier_path = core.shortest_path(barrier, 0, 3)

    single = UnifiedMetricCore(np.array([[0.0, 0.0]]), np.zeros((1, 1)))
    single_identity = single.identity_state()
    transformed_points, transformed_metric = affine_chart_change(
        single.points,
        np.asarray(single_identity.metric),
        np.diag([10.0, 1.0]),
    )
    transformed_single = UnifiedMetricCore(transformed_points, np.zeros((1, 1)))
    unprojected = transformed_single.make_state(transformed_metric)
    projected = transformed_single.project_metric(transformed_metric)
    vector = np.array([10.0, 0.0])
    projection_before = transformed_single.local_length_squared(unprojected, 0, vector)
    projection_after = transformed_single.local_length_squared(projected, 0, vector)

    bounded_core = UnifiedMetricCore(
        points,
        adjacency,
        UnifiedMetricConfig(
            min_eigenvalue=0.5,
            max_eigenvalue=2.0,
            source_rate=0.25,
        ),
    )
    source = np.repeat(np.diag([4.0, -1.0])[None, :, :], 4, axis=0)
    source_updated = bounded_core.apply_source_metric(bounded_core.identity_state(), source)
    source_certificate = bounded_core.certificate(source_updated)
    certificate = core.certificate(barrier)

    result = {
        "affine_trials": trials,
        "maximum_relative_local_error": maximum_local_error,
        "maximum_relative_edge_error": maximum_edge_error,
        "maximum_relative_path_error": maximum_path_error,
        "affine_tolerance": 1.0e-10,
        "identity_goal_minimizers": identity_goal.minimizers,
        "identity_path_unique": identity_path.unique,
        "barrier_goal_minimizers": barrier_goal.minimizers,
        "barrier_path": barrier_path.nodes,
        "barrier_path_cost": barrier_path.cost,
        "projection_length_before": projection_before,
        "projection_length_after": projection_after,
        "projection_covariance_defect": projection_after - projection_before,
        "source_observed_min_eigenvalue": source_certificate.observed_min_eigenvalue,
        "source_observed_max_eigenvalue": source_certificate.observed_max_eigenvalue,
        "source_condition_number": source_certificate.condition_number,
        "persistent_state": certificate.persistent_state,
        "role_parameter_count": certificate.role_parameter_count,
        "world_scope": certificate.world_scope,
        "full_geodesic_verified": certificate.full_geodesic_verified,
        "continuum_limit_verified": certificate.continuum_limit_verified,
        "irreversible_world_dynamics_verified": (
            certificate.irreversible_world_dynamics_verified
        ),
        "agi_evidence": certificate.agi_evidence,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
