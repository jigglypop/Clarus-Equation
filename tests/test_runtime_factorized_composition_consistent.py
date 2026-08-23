import json

from reality_stone.clarus.experiments.runtime_experience_delayed_binding import MIN_DECODE_ACTIVATION
from reality_stone.clarus.experiments.runtime_factorized_composition_consistent import (
    analyze_consistent_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-factorized-composition-consistent-20260822") / "artifacts" / "calibration-input.json"


def test_component_threshold_is_the_existing_atomic_decoder_threshold() -> None:
    assert MIN_DECODE_ACTIVATION == 1e-5


def test_fresh_calibration_passes_with_packet_stream_receipt() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_consistent_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "FACTORIZED_COMPOSITION_CALIBRATION_PASS"
    assert result["atomic_success_total"] == 4
    assert result["factorized_pair_success_total"] == 4
    assert result["legacy_pair_success_total"] == 0
    assert result["misaligned_pair_success_total"] == 0
    assert result["independent_union_success_total"] == 4

