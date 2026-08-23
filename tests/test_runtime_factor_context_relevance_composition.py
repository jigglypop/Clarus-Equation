import json

from reality_stone.clarus.experiments.runtime_factor_context_relevance_composition import (
    analyze_factor_context_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-factor-context-relevance-composition-20260822") / "artifacts" / "calibration-input.json"


def test_unseen_11_context_composes_two_seen_factor_values() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    assert payload["status"] == "FRESH_INPUTS_READY"
    result = analyze_factor_context_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "FACTOR_CONTEXT_CALIBRATION_PASS"
    assert result["heldout_success_total"] == 1
    assert result["oracle_success_total"] == 1
    assert result["joint_lookup_success_total"] == 0
    assert result["factor_a_shuffle_success_total"] == 0
    assert result["factor_b_shuffle_success_total"] == 0
    assert result["no_context_success_total"] == 0

