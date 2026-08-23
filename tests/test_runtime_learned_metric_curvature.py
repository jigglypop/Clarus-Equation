"""Python float64 reference tests for the learned-metric curvature diagnostic.

These tests do not certify a Rust or CUDA implementation.  Native backend
certification must satisfy ``.codex/harnesses/curvature_backend_parity.md``;
an unavailable backend, a Python/Torch fallback, or a CUDA f32 result is not a
native scientific-validation pass.
"""

from pathlib import Path

import torch

from reality_stone.clarus.runtime_learned_metric_curvature import (
    analyze_development_artifact,
    analyze_weight_code,
    finite_difference_jacobian,
    linear_flatness_certificate,
    nonlinear_geometry,
    top_right_singular_plane,
)


DEVELOPMENT = Path(
    "_workspace/ce/brainruntime-local-stochastic-binding-20260822/"
    "artifacts/development-results.json"
)


def test_linear_code_is_flat_and_uniform_code_is_degenerate() -> None:
    full_rank = torch.diag(torch.tensor((1.0, 1.2, 1.4, 1.6), dtype=torch.float64))
    flat = linear_flatness_certificate(full_rank)
    uniform = linear_flatness_certificate(torch.ones(4, 4, dtype=torch.float64))
    assert flat["rank"] == 4
    assert flat["intrinsic_curvature"] == 0.0
    assert uniform["rank"] == 1
    assert uniform["intrinsic_curvature"] is None


def test_nonlinear_geometry_analytic_jacobian_matches_finite_difference() -> None:
    B = torch.tensor(
        ((1.1, .9, .8, .7), (.8, 1.2, .7, .6), (.7, .8, 1.3, .9), (.6, .7, .8, 1.4)),
        dtype=torch.float64,
    )
    plane = top_right_singular_plane(B)
    point = torch.tensor((.25, -.35), dtype=torch.float64)
    geometry = nonlinear_geometry(B, plane, point)
    analytic = torch.tensor(geometry["jacobian"], dtype=torch.float64)
    finite = finite_difference_jacobian(B, plane, point)
    torch.testing.assert_close(analytic, finite, rtol=5e-9, atol=5e-9)


def test_hidden_relabel_changes_binding_but_not_metric_or_curvature() -> None:
    payload = analyze_development_artifact(DEVELOPMENT)
    assert payload["status"] == "CURVATURE_MEMORY_IDENTITY_REJECTED"
    assert payload["seed_count"] == payload["pass_count"] == 16
    for row in payload["rows"]:
        assert row["row_permutation"]["original_winners"] != row["row_permutation"]["permuted_winners"]
        assert row["row_permutation"]["metric_error"] <= 1e-12
        assert row["row_permutation"]["curvature_error"] <= 1e-12
        assert row["output_read_count"] == 0
        assert row["decoder_read_count"] == 0


def test_one_learned_code_has_nonzero_derived_curvature_but_not_curvature_identity() -> None:
    import json

    raw = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    result = analyze_weight_code(torch.tensor(raw["rows"][0]["learned"]["candidate_weights"]))
    assert result["status"] == "CURVATURE_IS_DERIVED_NOT_MEMORY"
    assert result["max_abs_nonlinear_curvature"] > 1e-10
    assert result["hidden_rotation"]["origin_metric_error"] <= 1e-12
    assert result["hidden_rotation"]["max_curvature_difference"] > 1e-10
