import json

import torch

from reality_stone.clarus.experiments.runtime_experience_attenuation_binding import (
    _experience_block_compensated,
)
from reality_stone.clarus.experiments.runtime_experience_delayed_binding import _seal
from reality_stone.clarus.experiments.runtime_factorized_competition_composition import (
    _factorized_snapshot,
)
from reality_stone.clarus.experiments.runtime_factorized_one_shot_event_composition import (
    _event_probe,
    analyze_one_shot_artifact,
)


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-factorized-one-shot-event-composition-20260822") / "artifacts" / "calibration-r2-input.json"


def test_one_shot_ring_gate_preserves_first_packet_and_removes_repeats() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    B = torch.tensor(payload["rows"][0]["candidate_weights"])
    block = _experience_block_compensated(B, condition="target_shuffle")
    snapshot, _cutoff = _seal(block["runtime"])
    factorized = _factorized_snapshot(snapshot, aligned=True)
    one_shot = _event_probe(factorized, (0, 2), emission="one_shot")
    stream = _event_probe(factorized, (0, 2), emission="stream")
    suppressed = _event_probe(factorized, (0, 2), emission="suppressed")
    assert one_shot["source_packet_count_by_tick"] == [0, 0, 0, 2, 0, 0, 0]
    assert one_shot["source_written_packet_count_by_tick"] == [0, 2, 0, 0, 0, 0, 0]
    assert stream["source_packet_count_by_tick"] == [0, 0, 0, 2, 2, 2, 2]
    assert suppressed["source_packet_count_by_tick"] == [0] * 7
    assert suppressed["source_written_packet_count_by_tick"] == [0] * 7


def test_fresh_calibration_requires_one_shot_composition_and_controls() -> None:
    result = analyze_one_shot_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "FACTORIZED_ONE_SHOT_CALIBRATION_PASS"
    assert result["atomic_success_total"] == 4
    assert result["factorized_pair_success_total"] == 4
    assert result["legacy_pair_success_total"] == 0
    assert result["misaligned_pair_success_total"] == 0
    assert result["independent_union_success_total"] == 4
    assert result["suppressed_pair_success_total"] == 0
    assert result["stream_pair_success_total"] == 4
