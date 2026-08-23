import json
from pathlib import Path

import torch

from reality_stone.clarus.runtime_binding_composition_no_go import (
    _global_wta,
    analyze_composition_artifact,
)


CALIBRATION_INPUT = Path(
    "_workspace/ce/brainruntime-binding-composition-no-go-20260822/"
    "artifacts/calibration-input.json"
)


def test_global_wta_has_at_most_one_positive_component() -> None:
    for values in (
        torch.tensor((1.0, 0.5, 0.2, 0.1)),
        torch.tensor((1.0, 1.0, 0.2, 0.1)),
        torch.tensor((-1.0, -2.0, -3.0, -4.0)),
    ):
        assert torch.count_nonzero(_global_wta(values) > 0.0) <= 1


def test_fresh_calibration_confirms_composition_no_go() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_composition_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "GLOBAL_WTA_COMPOSITION_NO_GO_CONFIRMED"
    assert result["atomic_success_total"] == 4
    assert result["simultaneous_success_total"] == 0
    assert result["independent_union_success_total"] == 4

