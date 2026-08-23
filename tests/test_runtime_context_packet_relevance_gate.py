import json

from reality_stone.clarus.experiments.runtime_context_packet_relevance_gate import (
    analyze_context_gate_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-context-packet-relevance-gate-20260822") / "artifacts" / "calibration-input.json"


def test_context_event_cooccurrence_gate_rejects_matched_distractor() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_context_gate_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "CONTEXT_PACKET_GATE_CALIBRATION_PASS"
    assert result["learned_success_total"] == 4
    assert result["oracle_success_total"] == 4
    assert result["context_shuffle_success_total"] == 0
    assert result["static_success_total"] == 1
    assert result["no_context_success_total"] == 0

