import json
from pathlib import Path

from reality_stone.clarus.runtime_all_input_packet_factorization import (
    analyze_all_input_artifact,
)


CALIBRATION_INPUT = Path(
    "_workspace/ce/brainruntime-all-input-packet-factorization-20260822/"
    "artifacts/calibration-input.json"
)


def test_fresh_all_input_calibration_passes_with_identity_controls() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_all_input_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "ALL_INPUT_PACKET_CALIBRATION_PASS"
    assert result["atomic_success_total"] == 4
    assert result["all_input_pair_success_total"] == 4
    assert result["source_projected_pair_success_total"] == 4
    assert result["legacy_pair_success_total"] == 0
    assert result["shifted_column_pair_success_total"] == 0
    assert result["suppressed_pair_success_total"] == 0
    assert result["independent_union_success_total"] == 4

