
from reality_stone.clarus.experiments.runtime_curvature_selector_confirmation import (
    CONFIRMATION_AMPLITUDE,
    CONFIRMATION_DIRECTIONS,
    confirm_development_artifact,
    confirmation_rotations,
)


from _run_paths import run_dir


DEVELOPMENT = run_dir("brainruntime-local-stochastic-binding-20260822") / "artifacts" / "development-results.json"


def test_confirmation_catalogue_is_fresh_and_fixed() -> None:
    assert CONFIRMATION_AMPLITUDE == 1.25
    assert len(CONFIRMATION_DIRECTIONS) == 8
    assert len(confirmation_rotations()) == 6
    assert all(abs(direction[0] ** 2 + direction[1] ** 2 - 1.0) <= 1e-12 for direction in CONFIRMATION_DIRECTIONS)


def test_fresh_geometry_curvature_selector_stops_at_frozen_gate() -> None:
    result = confirm_development_artifact(DEVELOPMENT)
    assert result["status"] == "CURVATURE_SELECTOR_STOP"
    assert result["seed_count"] == 16
    assert result["gates"]["equal_origin_metric"]
    assert result["gates"]["signed_permutation_equality"]
    assert not result["gates"]["curvature_hit_at_least_point70"]
    assert not result["gates"]["curvature_beats_static_hit"]
    assert result["maximum_origin_metric_error"] <= 1e-12
    assert result["maximum_equality_residual"] <= 1e-12
    assert not result["endpoint_opened"]
