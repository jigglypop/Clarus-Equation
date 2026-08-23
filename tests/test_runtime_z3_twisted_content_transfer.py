import inspect
import json

import pytest
import torch

from reality_stone.clarus.experiments.runtime_z3_twisted_content_transfer import (
    analyze_z3_twisted_content_artifact,
    compile_twisted_packet_indices,
    enumerate_unlabeled_cartesian_charts,
    predict_twisted_content,
    train_twisted_content_gate,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-z3-twisted-content-transfer-20260822") / "artifacts" / "calibration-input.json"


def _fixture(twist_class: int) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(5, dtype=torch.float64)
    base = 1.25 * basis[0]
    first = (torch.zeros(5, dtype=torch.float64), basis[1], basis[2])
    second = (torch.zeros(5, dtype=torch.float64), basis[3], basis[4])
    cues = torch.stack(
        [base + first[a] + second[b] for a in range(3) for b in range(3)]
    )
    dictionary = torch.eye(6, dtype=torch.float64)
    content = torch.stack(
        [
            dictionary[a] + dictionary[3 + ((b + twist_class * a) % 3)]
            for a in range(3)
            for b in range(3)
        ]
    )
    return cues, content


def test_unlabeled_chart_and_nonzero_twist_predict_query() -> None:
    cues, content = _fixture(1)
    charts = enumerate_unlabeled_cartesian_charts(cues)
    assert len(charts) == 72
    order = torch.tensor((5, 0, 7, 2, 6, 1, 4, 3))
    gate = train_twisted_content_gate(
        cues[:8].index_select(0, order),
        content[:8].index_select(0, order),
        cues[8],
    )
    assert gate.selected_residual <= 1e-12
    assert gate.best_additive_residual >= 1e-3
    assert torch.allclose(
        predict_twisted_content(gate, cues[8]),
        content[8],
        atol=1e-12,
        rtol=0.0,
    )


def test_additive_family_and_arbitrary_query_completion_are_not_admitted() -> None:
    cues, additive = _fixture(0)
    with pytest.raises(RuntimeError, match="additive content arm"):
        train_twisted_content_gate(cues[:8], additive[:8], cues[8])

    _, twisted = _fixture(2)
    gate = train_twisted_content_gate(cues[:8], twisted[:8], cues[8])
    alternative = twisted[8] + torch.linspace(0.1, 0.6, 6)
    assert not torch.allclose(predict_twisted_content(gate, cues[8]), alternative)


def test_generic_twist_learner_and_compiler_have_no_answer_api() -> None:
    forbidden = (
        "factor",
        "heldout",
        "target",
        "decoder",
        "reward",
        "endpoint",
        "role",
        "source[",
        "seed",
    )
    source = "\n".join(
        inspect.getsource(function).lower()
        for function in (
            train_twisted_content_gate,
            predict_twisted_content,
            compile_twisted_packet_indices,
        )
    )
    assert all(token not in source for token in forbidden)


def test_fresh_calibration_selects_nonzero_twist_and_routes_query() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_SIX_CONTENT_INPUTS_READY"
    result = analyze_z3_twisted_content_artifact(
        CALIBRATION_INPUT,
        stage="calibration",
    )
    assert result["status"] == "Z3_TWISTED_CONTENT_CALIBRATION_PASS"
    assert result["learned_success_total"] == 1
    assert result["oracle_success_total"] == 1
    assert result["joint_lookup_success_total"] == 0
    assert result["coordinate_memorizer_success_total"] == 0
    assert result["wrong_cue_success_total"] == 0
    assert result["binding_shuffle_success_total"] == 0
    assert result["no_context_success_total"] == 0
    assert result["endpoint_opened"] is True
    assert result["confirmation_opened"] is False
