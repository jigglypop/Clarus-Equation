from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest

from reality_stone.clarus.unified_metric import (
    MetricGoalReadout,
    UnifiedMetricConfig,
    UnifiedMetricCore,
    UnifiedMetricState,
    affine_chart_change,
)


def _diamond() -> tuple[np.ndarray, np.ndarray]:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [2.0, 0.0],
        ]
    )
    adjacency = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    return points, adjacency


def _core(config: UnifiedMetricConfig | None = None) -> UnifiedMetricCore:
    points, adjacency = _diamond()
    return UnifiedMetricCore(points, adjacency, config or UnifiedMetricConfig())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"min_eigenvalue": 0.0}, "positive"),
        ({"min_eigenvalue": 2.0, "max_eigenvalue": 1.0}, "at least"),
        ({"source_rate": -0.1}, r"\[0, 1\]"),
        ({"source_rate": 1.1}, r"\[0, 1\]"),
        ({"source_rate": True}, "finite real"),
    ],
)
def test_config_rejects_invalid_stabilisation(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        UnifiedMetricConfig(**options)


def test_core_rejects_invalid_points_and_topology() -> None:
    points, adjacency = _diamond()
    with pytest.raises(ValueError, match="dimension at least 2"):
        UnifiedMetricCore(points[:, :1], adjacency)
    duplicated = points.copy()
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="distinct"):
        UnifiedMetricCore(duplicated, adjacency)
    with pytest.raises(ValueError, match="size must match"):
        UnifiedMetricCore(points, np.ones((3, 3)) - np.eye(3))
    disconnected = adjacency.copy()
    disconnected[0, 1] = disconnected[1, 0] = 0.0
    disconnected[0, 2] = disconnected[2, 0] = 0.0
    with pytest.raises(ValueError, match="connected"):
        UnifiedMetricCore(points, disconnected)


def test_state_is_metric_only_and_rejects_non_spd_without_projection() -> None:
    core = _core()
    identity = core.identity_state()

    assert tuple(field.name for field in fields(UnifiedMetricState)) == ("metric",)
    assert np.asarray(identity.metric).shape == (4, 2, 2)

    nonsymmetric = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    nonsymmetric[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        core.make_state(nonsymmetric)
    indefinite = np.repeat(np.diag([-1.0, 1.0])[None, :, :], 4, axis=0)
    with pytest.raises(ValueError, match="positive definite"):
        core.make_state(indefinite)


def test_spectral_projection_and_certificate_are_fixed_chart_bounds() -> None:
    config = UnifiedMetricConfig(min_eigenvalue=0.2, max_eigenvalue=3.0)
    core = _core(config)
    raw = np.repeat(np.diag([-4.0, 9.0])[None, :, :], 4, axis=0)

    state = core.project_metric(raw)
    certificate = core.certificate(state)

    assert certificate.observed_min_eigenvalue == pytest.approx(0.2)
    assert certificate.observed_max_eigenvalue == pytest.approx(3.0)
    assert certificate.condition_number == pytest.approx(15.0)
    assert certificate.configured_condition_bound == pytest.approx(15.0)
    assert certificate.within_configured_bounds
    assert certificate.affine_readout_covariant
    assert not certificate.projection_affine_covariant


def test_local_edge_and_path_costs_are_affine_covariant_without_reprojection() -> None:
    core = _core()
    metric = np.array(
        [
            [[1.2, 0.1], [0.1, 0.8]],
            [[2.0, -0.2], [-0.2, 1.1]],
            [[0.7, 0.15], [0.15, 1.6]],
            [[1.4, 0.05], [0.05, 0.9]],
        ]
    )
    state = core.make_state(metric)
    jacobian = np.array([[2.0, 0.35], [-0.4, 1.3]])
    offset = np.array([3.0, -2.0])
    transformed_points, transformed_metric = affine_chart_change(
        core.points,
        metric,
        jacobian,
        offset,
    )
    transformed = UnifiedMetricCore(
        transformed_points,
        core.adjacency_mask.astype(float),
    )
    transformed_state = transformed.make_state(transformed_metric)
    displacement = np.array([0.3, -0.8])

    local_before = core.local_length_squared(state, 1, displacement)
    local_after = transformed.local_length_squared(
        transformed_state,
        1,
        jacobian @ displacement,
    )
    edges_before = core.edge_lengths(state)
    edges_after = transformed.edge_lengths(transformed_state)
    path_before = core.shortest_path(state, 0, 3)
    path_after = transformed.shortest_path(transformed_state, 0, 3)

    assert local_after == pytest.approx(local_before, rel=1e-12, abs=1e-12)
    mask = np.isfinite(edges_before)
    np.testing.assert_allclose(edges_after[mask], edges_before[mask], rtol=1e-12, atol=1e-12)
    assert path_after.cost == pytest.approx(path_before.cost, rel=1e-12, abs=1e-12)


def test_identity_metric_reduces_to_euclidean_edge_cost() -> None:
    core = _core()
    state = core.identity_state()
    lengths = core.edge_lengths(state)

    assert lengths[0, 1] == pytest.approx(math.sqrt(2.0))
    assert lengths[0, 2] == pytest.approx(math.sqrt(2.0))
    assert math.isinf(lengths[0, 3])


def test_source_update_is_bounded_deterministic_and_detached() -> None:
    config = UnifiedMetricConfig(
        min_eigenvalue=0.5,
        max_eigenvalue=2.0,
        source_rate=0.25,
    )
    core = _core(config)
    initial = core.identity_state()
    source = np.repeat(np.diag([4.0, -1.0])[None, :, :], 4, axis=0)

    first = core.apply_source_metric(initial, source)
    second = core.apply_source_metric(initial, source.copy())
    source[:] = 100.0
    certificate = core.certificate(first)

    assert first == second
    np.testing.assert_allclose(np.asarray(first.metric)[0], np.diag([1.25, 0.875]))
    assert certificate.within_configured_bounds
    assert certificate.condition_number <= certificate.configured_condition_bound + 1e-12


def test_one_metric_changes_memory_plan_critic_and_goal_readouts_together() -> None:
    core = _core()
    identity = core.identity_state()
    metric = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    metric[1] = 4.0 * np.eye(2)
    deformed = core.make_state(metric)

    before_edges = core.edge_lengths(identity)
    after_edges = core.edge_lengths(deformed)
    deformation = np.asarray(core.metric_deformation(deformed, identity))
    plan = core.shortest_path(deformed, 0, 3)
    critic_before = core.surprise_gate(
        identity,
        1,
        [1.0, 0.0],
        [0.0, 0.0],
        reference_scale=1.0,
        threshold=2.0,
    )
    critic_after = core.surprise_gate(
        deformed,
        1,
        [1.0, 0.0],
        [0.0, 0.0],
        reference_scale=1.0,
        threshold=2.0,
    )
    goal = core.minimum_cost_targets(deformed, 0, [1, 2])

    assert after_edges[0, 1] > before_edges[0, 1]
    assert np.linalg.norm(deformation[1]) > 0.0
    assert plan.nodes == (0, 2, 3)
    assert critic_before.hard_gate == 0
    assert critic_after.hard_gate == 1
    assert goal.minimizers == (2,)
    assert goal.unique


def test_symmetric_goal_preserves_all_ties_and_candidate_permutation() -> None:
    core = _core()
    state = core.identity_state()

    forward = core.minimum_cost_targets(state, 0, [1, 2])
    reversed_order = core.minimum_cost_targets(state, 0, [2, 1])
    path = core.shortest_path(state, 0, 3)

    assert isinstance(forward, MetricGoalReadout)
    assert forward == reversed_order
    assert forward.minimizers == (1, 2)
    assert not forward.unique
    assert not path.unique
    assert "representative" in path.tie_policy


def test_surprise_is_dimensionless_hard_and_affine_covariant() -> None:
    core = _core()
    state = core.identity_state()
    at_boundary = core.surprise_gate(
        state,
        0,
        [1.0, 0.0],
        [0.0, 0.0],
        reference_scale=1.0,
        threshold=1.0,
    )
    above = core.surprise_gate(
        state,
        0,
        [1.0, 0.0],
        [0.0, 0.0],
        reference_scale=1.0,
        threshold=0.99,
    )
    jacobian = np.array([[3.0, 0.5], [0.2, 0.8]])
    points, metric = affine_chart_change(
        core.points,
        np.asarray(state.metric),
        jacobian,
    )
    transformed = UnifiedMetricCore(points, core.adjacency_mask.astype(float))
    transformed_state = transformed.make_state(metric)
    transformed_surprise = transformed.surprise_gate(
        transformed_state,
        0,
        jacobian @ np.array([1.0, 0.0]),
        [0.0, 0.0],
        reference_scale=1.0,
        threshold=1.0,
    )

    assert at_boundary.normalized_squared_length == pytest.approx(1.0)
    assert at_boundary.hard_gate == 0
    assert above.hard_gate == 1
    assert transformed_surprise.normalized_squared_length == pytest.approx(
        at_boundary.normalized_squared_length,
        rel=1e-12,
        abs=1e-12,
    )
    assert transformed_surprise.hard_gate == at_boundary.hard_gate


def test_snapshot_roundtrip_is_exact_and_input_detached() -> None:
    core = _core()
    raw = np.repeat(np.eye(2)[None, :, :], 4, axis=0)
    state = core.make_state(raw)
    raw[:] = 3.0

    snapshot = core.snapshot(state)
    restored = core.from_snapshot(snapshot)

    assert restored == state
    assert np.asarray(restored.metric)[0, 0, 0] == 1.0


def test_certificate_refuses_unimplemented_geometry_and_agi_claims() -> None:
    certificate = _core().certificate(_core().identity_state())

    assert certificate.persistent_state == "metric_only"
    assert certificate.persistent_state_field_count == 1
    assert certificate.role_parameter_count == 0
    assert certificate.geometry_scope == "finite-point-local-quadratic+metric-graph"
    assert certificate.world_scope == "metric_cost_substrate"
    assert not certificate.projection_affine_covariant
    assert not certificate.full_geodesic_verified
    assert not certificate.connection_verified
    assert not certificate.curvature_verified
    assert not certificate.heat_kernel_verified
    assert not certificate.continuum_limit_verified
    assert not certificate.irreversible_world_dynamics_verified
    assert not certificate.agi_evidence
    assert not certificate.biological_evidence
    assert not certificate.cosmological_evidence


def test_unified_metric_is_exported_from_public_clarus_package() -> None:
    from reality_stone import clarus

    assert clarus.UnifiedMetricCore is UnifiedMetricCore
    assert clarus.UnifiedMetricConfig is UnifiedMetricConfig
    assert clarus.affine_chart_change is affine_chart_change
