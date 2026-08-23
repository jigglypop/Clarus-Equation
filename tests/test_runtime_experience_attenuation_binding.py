import inspect
import json

import torch

from reality_stone.clarus.experiments.runtime_experience_attenuation_binding import (
    MAX_COMPENSATION,
    _compensation_vector,
    _experience_block_compensated,
    analyze_fresh_input_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-experience-attenuation-binding-20260822") / "artifacts" / "calibration-input.json"


def test_compensation_is_local_bounded_and_zero_without_packet() -> None:
    packet = torch.tensor((1e-5, 0.0, 1e-3, 0.0))
    trace = torch.tensor((1.0, 1.0, 1.0, 0.0))
    factor = _compensation_vector(packet, trace)
    assert torch.isclose(factor[0], torch.tensor(MAX_COMPENSATION / 1.6), atol=1e-5)
    assert factor[1] == 0.0
    assert factor[2] == 1.0
    assert factor[3] == 0.0
    source = inspect.getsource(_compensation_vector).lower()
    assert not any(token in source for token in ("decoder", "reward", "endpoint", "target", "label"))


def test_block_is_one_bounded_write_without_projection() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    weight = torch.tensor(payload["rows"][0]["candidate_weights"])
    result = _experience_block_compensated(weight, condition="compensated")
    assert result["mid_block_weight_unchanged"]
    assert result["block_boundary_count"] == 1
    assert result["mutation_count"] == 1
    assert result["outside_support_delta_norm"] == 0.0
    assert result["raw_install_max_error"] <= 1e-7
    assert result["edge_cap_hit_count"] == 0
    module_source = inspect.getsource(inspect.getmodule(_experience_block_compensated)).lower()
    assert "structural_projection" not in module_source


def test_frozen_calibration_passes_zero_store_recall() -> None:
    result = analyze_fresh_input_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "ATTENUATION_CALIBRATION_PASS"
    assert result["pass_count"] == 1
    assert not result["endpoint_opened"]
