import json

from reality_stone.clarus.experiments.runtime_three_event_relevance_no_go import (
    analyze_relevance_no_go_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-three-event-relevance-no-go-20260822") / "artifacts" / "calibration-input.json"


def test_three_locally_valid_events_do_not_identify_desired_pair() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_relevance_no_go_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "THREE_EVENT_RELEVANCE_CALIBRATION_PASS"
    assert result["pair_only_success_total"] == 4
    assert result["desired_pair_success_total"] == 0
    assert result["three_route_identity_total"] == 4

