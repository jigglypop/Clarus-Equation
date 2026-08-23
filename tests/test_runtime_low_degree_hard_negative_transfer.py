import json

import torch

from reality_stone.clarus.experiments.runtime_low_degree_hard_negative_transfer import (
    _hard_panel,
    analyze_low_degree_hard_negative_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-low-degree-hard-negative-transfer-20260822") / "artifacts" / "calibration-r1-input.json"


def test_midpoint_hard_negative_has_unique_opposite_model_choices() -> None:
    truth = torch.tensor((1.0, 0.9, 0.8, 0.7, 0.6, 0.5), dtype=torch.float64)
    affine = truth + torch.tensor((0.1, -0.05, 0.08, -0.04, 0.06, -0.03))
    content = torch.stack([truth + 0.01 * index for index in range(25)])
    panel, _, truth_role, affine_role, skew_role = _hard_panel(
        116001, 0, truth, affine, content
    )
    distances_truth = torch.linalg.vector_norm(panel[:, :3].T - truth, dim=1)
    distances_affine = torch.linalg.vector_norm(panel[:, :3].T - affine, dim=1)
    assert int(torch.argmin(distances_truth).item()) == truth_role
    assert int(torch.argmin(distances_affine).item()) == affine_role
    assert truth_role != affine_role != skew_role
    assert torch.allclose(
        panel[:, skew_role],
        0.5 * (truth + affine),
        atol=1e-12,
        rtol=0.0,
    )


def test_fresh_calibration_resolves_nearest_and_affine_panels() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "LOW_DEGREE_ROTATING_INPUTS_READY"
    result = analyze_low_degree_hard_negative_artifact(
        CALIBRATION_INPUT,
        stage="calibration",
    )
    assert result["status"] == "LOW_DEGREE_HARD_NEGATIVE_CALIBRATION_PASS"
    assert result["rotating_fold_count"] == 25
    assert result["hard_affine_decoy_selection_total"] == 25
    assert result["hard_affine_truth_selection_total"] == 0
    assert result["nearest_affine_truth_selection_total"] <= 6
    assert result["endpoint_opened"] is True
    assert result["confirmation_opened"] is False
