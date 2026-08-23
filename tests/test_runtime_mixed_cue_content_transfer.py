import inspect
import json
from pathlib import Path

import torch

from reality_stone.clarus.runtime_mixed_cue_content_transfer import (
    analyze_mixed_cue_content_artifact,
    compile_arrived_packet_indices,
    predict_mixed_cue_content,
    train_mixed_cue_content_gate,
)


CALIBRATION_INPUT = Path(
    "_workspace/ce/brainruntime-mixed-cue-content-transfer-20260822/"
    "artifacts/calibration-input.json"
)


def test_rank_two_formula_is_exact_and_coordinate_equivariant() -> None:
    r0 = torch.tensor((2.0, -1.0, 0.0), dtype=torch.float64)
    u = torch.tensor((1.0, 2.0, 1.0), dtype=torch.float64)
    v = torch.tensor((2.0, -1.0, 2.0), dtype=torch.float64)
    cues = torch.stack((r0, r0 + u, r0 + v, r0 + u + v))
    content = torch.eye(4, dtype=torch.float64)
    sums = torch.stack(
        (
            content[0] + content[2],
            content[0] + content[3],
            content[1] + content[2],
            content[1] + content[3],
        )
    )
    gate = train_mixed_cue_content_gate(cues[:3], sums[:3])
    assert torch.allclose(
        predict_mixed_cue_content(gate, cues[3]), sums[3], atol=1e-12, rtol=0.0
    )

    weight = torch.zeros(8, 8, dtype=torch.float64)
    response = (0, 1, 2, 3)
    remapped = (6, 4, 7, 5)
    for role, coordinate in enumerate(remapped):
        weight[torch.tensor(response), coordinate] = content[role]
    arrived = (remapped[1], remapped[3], remapped[0])
    selected = compile_arrived_packet_indices(
        gate, cues[3], arrived, weight, response
    )
    assert set(selected) == {remapped[1], remapped[3]}


def test_generic_learner_and_compiler_have_no_supplied_factor_api() -> None:
    forbidden = (
        "factor_contexts",
        "composition_pairs",
        "target_mapping",
        "heldout",
        "decoder",
        "reward",
        "endpoint",
        "source[",
    )
    source = "\n".join(
        inspect.getsource(function).lower()
        for function in (
            train_mixed_cue_content_gate,
            predict_mixed_cue_content,
            compile_arrived_packet_indices,
        )
    )
    assert all(token not in source for token in forbidden)


def test_fresh_calibration_transfers_to_unseen_coordinates() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_mixed_cue_content_artifact(
        CALIBRATION_INPUT, stage="calibration"
    )
    assert result["status"] == "MIXED_CUE_CONTENT_CALIBRATION_PASS"
    assert result["learned_success_total"] == 1
    assert result["oracle_success_total"] == 1
    assert result["joint_lookup_success_total"] == 0
    assert result["coordinate_memorizer_success_total"] == 0
    assert result["wrong_cue_success_total"] == 0
    assert result["binding_shuffle_success_total"] == 0
    assert result["rank_one_success_total"] == 0
    assert result["no_context_success_total"] == 0
