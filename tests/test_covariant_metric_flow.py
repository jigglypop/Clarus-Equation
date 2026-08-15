from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest

import reality_stone.clarus as clarus
from reality_stone.clarus.covariant_metric_flow import (
    CovariantMetricConfig,
    CovariantMetricFlow,
    CovariantMetricState,
    MetricFlowCertificate,
    RouteChoice,
)


def _moderate_metric() -> np.ndarray:
    factor = np.array(
        [
            [1.2, 0.0, 0.0],
            [-0.25, 0.8, 0.0],
            [0.3, 0.15, 1.5],
        ]
    )
    return factor @ factor.T


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"eta": 0.0}, r"\(0, 1\]"),
        ({"eta": 1.1}, r"\(0, 1\]"),
        ({"eta": True}, "finite real"),
        ({"tie_tolerance_multiplier": -1.0}, "nonnegative"),
    ],
)
def test_config_rejects_invalid_values(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CovariantMetricConfig(**options)


def test_state_has_exactly_one_factor_field_and_no_metric_copy() -> None:
    flow = CovariantMetricFlow(3)
    state = flow.identity_state()
    certificate = flow.certificate(state)

    assert tuple(field.name for field in fields(CovariantMetricState)) == ("factor",)
    assert not hasattr(state, "metric")
    assert certificate.persistent_state == "factor_only_metric_encoding"
    assert certificate.persistent_state_field_count == 1
    assert certificate.semantic_state_degrees_of_freedom == 6
    assert certificate.optimizer_state_field_count == 0


def test_state_validation_requires_canonical_positive_diagonal_factor() -> None:
    flow = CovariantMetricFlow(2)
    with pytest.raises(ValueError, match="lower triangular"):
        flow.predict(CovariantMetricState(((1.0, 0.1), (0.0, 1.0))), [1.0, 0.0])
    with pytest.raises(ValueError, match="positive diagonal"):
        flow.predict(CovariantMetricState(((1.0, 0.0), (0.0, 0.0))), [1.0, 0.0])
    with pytest.raises(ValueError, match="positive definite"):
        flow.make_state_from_metric([[1.0, 2.0], [2.0, 1.0]])


def test_m1_update_preserves_spd_and_canonical_factor_without_projection() -> None:
    flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=0.7))
    state = flow.make_state_from_metric(_moderate_metric())
    updated = flow.update(state, [0.4, -1.1, 0.7], 0.13)
    factor = np.asarray(updated.factor)
    metric = flow.metric(updated)
    certificate = flow.certificate(updated)

    assert np.all(np.triu(factor, 1) == 0.0)
    assert np.all(np.diag(factor) > 0.0)
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)
    assert certificate.factor_congruence_update
    assert certificate.spd_preserved_in_exact_arithmetic
    assert not certificate.spectral_projection_used


def test_m2_update_is_affine_covariant_at_moderate_conditioning() -> None:
    flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=0.4))
    metric = _moderate_metric()
    state = flow.make_state_from_metric(metric)
    displacement = np.array([0.4, -1.1, 0.7])
    observed_cost = 2.3
    jacobian = np.array(
        [
            [1.7, 0.2, -0.1],
            [-0.3, 0.9, 0.15],
            [0.25, -0.2, 1.3],
        ]
    )
    inverse = np.linalg.inv(jacobian)
    transformed_metric = inverse.T @ metric @ inverse
    transformed_flow = CovariantMetricFlow(3, flow.config)
    transformed_state = transformed_flow.make_state_from_metric(transformed_metric)

    updated = flow.metric(flow.update(state, displacement, observed_cost))
    transformed_updated = transformed_flow.metric(
        transformed_flow.update(
            transformed_state,
            jacobian @ displacement,
            observed_cost,
        )
    )

    np.testing.assert_allclose(
        transformed_updated,
        inverse.T @ updated @ inverse,
        rtol=2e-12,
        atol=2e-12,
    )


def test_m3_same_observation_log_residual_contracts_by_one_minus_eta() -> None:
    eta = 0.37
    flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=eta))
    state = flow.make_state_from_metric(_moderate_metric())
    displacement = np.array([0.4, -1.1, 0.7])
    observed_cost = 0.27
    before = flow.residual(state, displacement, observed_cost)

    updated = flow.update(state, displacement, observed_cost)
    after = flow.residual(updated, displacement, observed_cost)

    assert after == pytest.approx((1.0 - eta) * before, rel=2e-13, abs=2e-13)


def test_m4_factor_update_matches_airm_rank_one_exponential_formula() -> None:
    eta = 0.4
    flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=eta))
    metric = _moderate_metric()
    state = flow.make_state_from_metric(metric)
    displacement = np.array([0.4, -1.1, 0.7])
    observed_cost = 0.73
    prediction = float(displacement @ metric @ displacement)
    residual = math.log(prediction / observed_cost)
    coefficient = math.expm1(-eta * residual)
    metric_x = metric @ displacement
    expected = metric + coefficient * np.outer(metric_x, metric_x) / prediction

    actual = flow.metric(flow.update(state, displacement, observed_cost))

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("initial_metric", "displacement", "observed_cost", "eta", "expected_metric"),
    [
        (1.0, 1.0, 1.0e-300, 1.0, 1.0e-300),
        (1.0, 1.0e-150, 1.0, 1.0, 1.0e300),
        (1.0e308, 1.0, 1.0e-308, 0.5, 1.0),
    ],
)
def test_registered_scalar_extremes_remain_positive_and_representable(
    initial_metric: float,
    displacement: float,
    observed_cost: float,
    eta: float,
    expected_metric: float,
) -> None:
    flow = CovariantMetricFlow(1, CovariantMetricConfig(eta=eta))
    state = flow.make_state_from_metric([[initial_metric]])

    updated = flow.update(state, [displacement], observed_cost)
    factor = float(updated.factor[0][0])
    result = flow.metric(updated)[0, 0]

    assert math.isfinite(factor) and factor > 0.0
    assert result == pytest.approx(expected_metric, rel=8e-14)
    assert flow.predict(updated, [1.0]) == pytest.approx(expected_metric, rel=8e-14)


def test_representable_near_equal_residual_is_not_rounded_to_zero() -> None:
    flow = CovariantMetricFlow(1, CovariantMetricConfig(eta=1.0))
    # The perturbed factor and the target factor 1 are both representable.
    # Using cost=nextafter(1, +inf) would instead demand a square root lying
    # strictly between adjacent binary64 factors.
    state = CovariantMetricState(((math.nextafter(1.0, math.inf),),))
    observed_cost = 1.0

    residual = flow.residual(state, [1.0], observed_cost)
    updated = flow.update(state, [1.0], observed_cost)

    assert residual != 0.0
    assert residual == pytest.approx(2.0 * math.ulp(1.0), rel=0.0, abs=1e-30)
    assert flow.predict(updated, [1.0]) == observed_cost


def test_nonrepresentable_updated_factor_is_rejected_explicitly() -> None:
    flow = CovariantMetricFlow(1, CovariantMetricConfig(eta=1.0))
    state = flow.make_state_from_metric([[1.0e308]])

    with pytest.raises(OverflowError, match="not representable"):
        flow.update(state, [1.0e-308], 1.0e308)


def test_route_choice_uses_declared_tolerance_and_deterministic_lowest_index() -> None:
    flow = CovariantMetricFlow(
        2,
        CovariantMetricConfig(eta=0.4, tie_tolerance_multiplier=64.0),
    )
    state = flow.identity_state()
    routes = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[2.0, 0.0]],
    ]

    choice = flow.choose_route(state, routes)

    assert choice.costs == pytest.approx((2.0, 2.0, 4.0))
    assert choice.minimizers == (0, 1)
    assert choice.selected_index == 0
    assert choice.selected_cost == pytest.approx(2.0)
    assert not choice.unique
    assert choice.tie_tolerance == pytest.approx(64.0 * np.finfo(float).eps * 4.0)
    assert "lowest index" in choice.tie_policy


def test_snapshot_roundtrip_is_exact_and_detached() -> None:
    flow = CovariantMetricFlow(3)
    metric = _moderate_metric()
    state = flow.make_state_from_metric(metric)
    metric[:] = 99.0

    snapshot = flow.snapshot(state)
    restored = flow.from_snapshot(snapshot)

    assert snapshot == state
    assert restored == state
    assert flow.metric(restored)[0, 0] != 99.0


def test_certificate_keeps_unproved_agent_claims_false() -> None:
    flow = CovariantMetricFlow(2)
    certificate = flow.certificate(flow.identity_state())

    assert certificate.affine_update_covariant_in_exact_arithmetic
    assert certificate.same_observation_contraction_in_exact_arithmetic
    assert certificate.airm_natural_gradient_identity
    assert not certificate.full_metric_identifiable_without_spanning_measurements
    assert not certificate.fixed_rate_noisy_point_convergence
    assert not certificate.raw_perception_verified
    assert not certificate.delayed_credit_verified
    assert not certificate.continuum_geometry_verified
    assert not certificate.agi_evidence


def test_public_exports_are_the_production_types() -> None:
    assert clarus.CovariantMetricConfig is CovariantMetricConfig
    assert clarus.CovariantMetricState is CovariantMetricState
    assert clarus.RouteChoice is RouteChoice
    assert clarus.MetricFlowCertificate is MetricFlowCertificate
    assert clarus.CovariantMetricFlow is CovariantMetricFlow
