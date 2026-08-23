import inspect
import json
from pathlib import Path

import pytest
import torch

from reality_stone.clarus.runtime_3x3_unlabeled_content_transfer import (
    analyze_3x3_unlabeled_content_artifact,
    compile_current_packet_indices,
    discover_unlabeled_parallelograms,
    predict_affine_content,
    train_affine_content_gate,
)


CALIBRATION_INPUT = Path(
    "_workspace/ce/brainruntime-3x3-unlabeled-content-transfer-20260822/"
    "artifacts/calibration-input.json"
)


def _exact_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(5, dtype=torch.float64)
    base = 1.25 * basis[0]
    first = (torch.zeros(5, dtype=torch.float64), basis[1], basis[2])
    second = (torch.zeros(5, dtype=torch.float64), basis[3], basis[4])
    cues = torch.stack(
        [base + first[a] + second[b] for a in range(3) for b in range(3)]
    )
    dictionary = torch.eye(6, dtype=torch.float64)
    content = torch.stack(
        [dictionary[a] + dictionary[3 + b] for a in range(3) for b in range(3)]
    )
    return cues, content


def test_rank_four_formula_finds_five_rectangles_and_predicts_22() -> None:
    cues, content = _exact_fixture()
    rectangles, residuals = discover_unlabeled_parallelograms(
        cues[:8], content[:8]
    )
    assert len(rectangles) == 5
    assert max(residuals) == 0.0
    gate = train_affine_content_gate(cues[:8], content[:8])
    assert gate.cue_rank == 4
    assert gate.content_rank == 4
    assert torch.allclose(
        predict_affine_content(gate, cues[8]),
        content[8],
        atol=1e-12,
        rtol=0.0,
    )

    order = torch.tensor((7, 0, 5, 2, 6, 1, 4, 3))
    reordered = train_affine_content_gate(
        cues[:8].index_select(0, order),
        content[:8].index_select(0, order),
    )
    assert torch.allclose(
        predict_affine_content(reordered, cues[8]),
        content[8],
        atol=1e-12,
        rtol=0.0,
    )


def test_global_law_is_conditional_and_off_span_query_fails_closed() -> None:
    cues, content = _exact_fixture()
    gate = train_affine_content_gate(cues[:8], content[:8])
    alternative = content[8] + torch.linspace(0.1, 0.6, 6)
    assert not torch.allclose(predict_affine_content(gate, cues[8]), alternative)
    off_span = torch.cat((cues[8], torch.tensor((1.0,), dtype=torch.float64)))
    with pytest.raises(ValueError):
        predict_affine_content(gate, off_span)


def test_generic_learner_and_compiler_have_no_factor_or_endpoint_api() -> None:
    forbidden = (
        "factor",
        "grid",
        "heldout",
        "target",
        "decoder",
        "reward",
        "endpoint",
        "role",
        "source[",
    )
    source = "\n".join(
        inspect.getsource(function).lower()
        for function in (
            train_affine_content_gate,
            predict_affine_content,
            compile_current_packet_indices,
        )
    )
    assert all(token not in source for token in forbidden)


def test_fresh_calibration_transfers_under_unseen_coordinate_map() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_SIX_CONTENT_INPUTS_READY"
    result = analyze_3x3_unlabeled_content_artifact(
        CALIBRATION_INPUT,
        stage="calibration",
    )
    assert result["status"] == "UNLABELED_3X3_AFFINE_CONTENT_CALIBRATION_PASS"
    assert result["learned_success_total"] == 1
    assert result["oracle_success_total"] == 1
    assert result["joint_lookup_success_total"] == 0
    assert result["coordinate_memorizer_success_total"] == 0
    assert result["wrong_cue_success_total"] == 0
    assert result["binding_shuffle_success_total"] == 0
    assert result["rank_three_success_total"] == 0
    assert result["no_context_success_total"] == 0
    assert result["endpoint_opened"] is True
    assert result["confirmation_opened"] is False
