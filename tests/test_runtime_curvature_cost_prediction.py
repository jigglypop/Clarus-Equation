from pathlib import Path

import torch

from reality_stone.clarus.runtime_curvature_cost_prediction import (
    analyze_development_artifact,
    flat_nonlinear_counterexample,
    geometry,
    ray_costs,
)


DEVELOPMENT = Path(
    "_workspace/ce/brainruntime-local-stochastic-binding-20260822/"
    "artifacts/development-results.json"
)


def test_exact_flat_nonlinear_counterexample_rejects_curvature_sufficiency() -> None:
    result = flat_nonlinear_counterexample()
    assert result["same_origin_metric"]
    assert result["both_curvature_cost_zero"]
    assert result["distortion_separated"]
    assert result["metric_strain_selects_lower_distortion"]
    assert result["route_1"]["heldout_distortion"] > result["route_2"]["heldout_distortion"]


def test_signed_hidden_permutation_is_an_exact_equality_null() -> None:
    actuator = torch.tensor(
        ((1.0, .2), (.3, .9), (.6, -.4), (-.2, .7)), dtype=torch.float64,
    )
    permutation = torch.tensor(
        ((0., 0., 0., -1.), (1., 0., 0., 0.), (0., -1., 0., 0.), (0., 0., 1., 0.)),
        dtype=torch.float64,
    )
    first = ray_costs(actuator, (1.0, 1.0))
    second = ray_costs(permutation @ actuator, (1.0, 1.0))
    for key in ("curvature_cost", "metric_strain_cost", "heldout_distortion"):
        assert abs(first[key] - second[key]) <= 1e-12


def test_geometry_has_zero_curvature_for_square_invertible_tanh_chart() -> None:
    actuator = torch.tensor(((1.0, .2), (.1, .9)), dtype=torch.float64)
    for point in ((0.0, 0.0), (.3, -.4), (.8, .2)):
        assert abs(geometry(actuator, torch.tensor(point))["curvature"]) <= 1e-12


def test_frozen_learned_family_shows_curvature_association_not_sufficiency() -> None:
    result = analyze_development_artifact(DEVELOPMENT)
    assert result["status"] == "CURVATURE_ASSOCIATION_NOT_SUFFICIENT"
    assert result["seed_count"] == 16
    assert result["maximum_origin_metric_error"] <= 1e-12
    assert result["maximum_equality_residual"] <= 1e-12
    assert result["mean_curvature_regret"] < result["mean_metric_strain_regret"]
    assert not result["endpoint_opened"]
