import json
from pathlib import Path

import torch

from reality_stone.clarus.runtime_independent_stream_degree_id import (
    _poly_features,
    _studentized_press,
    analyze_degree_id_artifact,
)


CALIBRATION_INPUT = Path(
    "_workspace/ce/brainruntime-independent-stream-degree-id-20260822/"
    "artifacts/calibration-input.json"
)


def test_studentized_press_hat_identity_matches_explicit_refit() -> None:
    generator = torch.Generator(device="cpu").manual_seed(424_242)
    cues = torch.randn(12, 2, generator=generator, dtype=torch.float64)
    coefficients = (
        2.0 * torch.rand(6, 6, generator=generator, dtype=torch.float64) - 1.0
    )
    design = _poly_features(cues, 2)
    observations = design @ coefficients + 1e-3 * torch.randn(
        12, 6, generator=generator, dtype=torch.float64
    )
    value, leverages, _ = _studentized_press(design, observations)
    total = 0.0
    for held_out in range(12):
        keep = [index for index in range(12) if index != held_out]
        refit = torch.linalg.pinv(design[keep]) @ observations[keep]
        residual = observations[held_out] - design[held_out] @ refit
        total += float(
            torch.linalg.vector_norm(residual).item()
        ) * float(torch.sqrt(1.0 - leverages[held_out]).item())
    explicit = total / 12
    assert abs(value - explicit) <= 1e-8 * max(1.0, abs(explicit))


def test_fresh_calibration_identifies_degrees_and_abstains_on_witness() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "DEGREE_ID_INPUTS_READY"
    result = analyze_degree_id_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "DEGREE_ID_CALIBRATION_PASS"
    assert result["main_fold_count"] == 9
    assert result["witness_fold_count"] == 1
    assert result["degree_identification_count"] == 9
    assert result["bank_truth_selection_count"] == 9
    assert result["witness_abstain_count"] == 1
    assert result["shuffle_rejection_count"] == 10
    assert result["wrong_cue_truth_selection_count"] == 0
    assert result["forced_affine_gate_failure_count"] == 6
    assert result["confirmation_opened"] is False
    for row in result["rows"]:
        for fold in row["folds"]:
            receipt = fold["bank_receipt"]
            assert receipt["bank_counter"] < receipt["decision_counter"]
            assert len(receipt["values"]) == 8
