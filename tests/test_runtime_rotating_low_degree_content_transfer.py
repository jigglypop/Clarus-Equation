import inspect
import json

import pytest
import torch

from reality_stone.clarus.experiments.runtime_rotating_low_degree_content_transfer import (
    analyze_rotating_low_degree_artifact,
    compile_low_degree_packet_index,
    generate_low_degree_inputs,
    predict_low_degree_content,
    train_low_degree_content_gate,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-rotating-low-degree-content-transfer-20260822") / "artifacts" / "calibration-input.json"


def _quadratic_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    points = torch.tensor(
        [(first, second) for first in levels for second in levels],
        dtype=torch.float64,
    )
    cues = torch.cat(
        (
            0.75 * torch.ones(25, 1, dtype=torch.float64),
            points,
            torch.zeros(25, 5, dtype=torch.float64),
        ),
        dim=1,
    )
    features = torch.stack(
        (
            torch.ones(25, dtype=torch.float64),
            points[:, 0],
            points[:, 1],
            points[:, 0].square(),
            points[:, 0] * points[:, 1],
            points[:, 1].square(),
        ),
        dim=1,
    )
    coefficients = torch.eye(6, dtype=torch.float64)
    content = features @ coefficients
    return cues, content


def test_every_rotating_holdout_is_identified_by_one_generic_quadratic() -> None:
    cues, content = _quadratic_fixture()
    for query in range(25):
        observed = [index for index in range(25) if index != query]
        gate = train_low_degree_content_gate(cues[observed], content[observed])
        assert torch.allclose(
            predict_low_degree_content(gate, cues[query]),
            content[query],
            atol=1e-11,
            rtol=0.0,
        )


def test_query_only_delta_is_indistinguishable_and_off_plane_query_abstains() -> None:
    cues, content = _quadratic_fixture()
    gate = train_low_degree_content_gate(cues[:-1], content[:-1])
    prediction = predict_low_degree_content(gate, cues[-1])
    alternative = content[-1] + torch.linspace(0.1, 0.6, 6)
    assert torch.allclose(prediction, content[-1], atol=1e-11, rtol=0.0)
    assert not torch.allclose(prediction, alternative)
    off_plane = cues[-1].clone()
    off_plane[-1] = 0.25
    with pytest.raises(RuntimeError, match="outside the observed cue plane"):
        predict_low_degree_content(gate, off_plane)


def test_generic_learner_and_compiler_have_no_answer_api() -> None:
    forbidden = (
        "factor",
        "grid",
        "heldout",
        "target",
        "decoder",
        "reward",
        "endpoint",
        "role",
        "source",
        "seed",
    )
    source = "\n".join(
        inspect.getsource(function).lower()
        for function in (
            train_low_degree_content_gate,
            predict_low_degree_content,
            compile_low_degree_packet_index,
        )
    )
    assert all(token not in source for token in forbidden)


def test_fresh_calibration_rotates_all_cells_and_routes_current_packet() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "LOW_DEGREE_ROTATING_INPUTS_READY"
    result = analyze_rotating_low_degree_artifact(
        CALIBRATION_INPUT,
        stage="calibration",
    )
    assert result["status"] == "ROTATING_LOW_DEGREE_CALIBRATION_PASS"
    assert result["rotating_fold_count"] == 25
    assert result["association_shuffle_rejection_total"] == 25
    assert result["endpoint_opened"] is True
    assert result["confirmation_opened"] is False


def test_input_generator_varies_coefficients_and_stays_dimensionless() -> None:
    payload = generate_low_degree_inputs((115001, 115002))
    assert payload["status"] == "LOW_DEGREE_ROTATING_INPUTS_READY"
    assert payload["rows"][0]["coefficient_sha256"] != payload["rows"][1]["coefficient_sha256"]
    assert all(all(row["gates"].values()) for row in payload["rows"])
