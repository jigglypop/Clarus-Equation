import json

import torch

from reality_stone.clarus.runtime import BrainRuntimeConfig
from reality_stone.clarus.experiments.runtime_adaptive_competition_composition import (
    analyze_adaptive_artifact,
)
from reality_stone.clarus.experiments.runtime_experience_delayed_binding import _blocks, _runtime


from _run_paths import run_dir


CALIBRATION_INPUT = run_dir("brainruntime-adaptive-competition-composition-20260822") / "artifacts" / "calibration-input.json"


def test_adaptive_config_fails_closed_outside_strict_domain() -> None:
    source, hidden, _target = _blocks()
    common = dict(
        dim=20,
        axon_delay=True,
        competition_indices=hidden,
        competition_input_indices=source,
        competition_k_from_delayed_input=True,
    )
    try:
        BrainRuntimeConfig(**common, competition_lateral_gain=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("adaptive k-WTA must reject lateral gain != 1")


def test_two_three_boundary_tie_fails_closed_and_singleton_matches() -> None:
    payload = json.loads(CALIBRATION_INPUT.read_text(encoding="utf-8"))
    B = torch.tensor(payload["rows"][0]["candidate_weights"])
    legacy = _runtime(B)
    source, hidden, _target = _blocks()
    adaptive = _runtime(B)
    adaptive.config.competition_input_indices = source
    adaptive.config.competition_k_from_delayed_input = True
    adaptive._competition_signature = (
        tuple(hidden), 1, 0.0, 0, tuple(source), True
    )
    adaptive._competition_input_indices = torch.tensor(source)
    recurrent = torch.zeros(20)
    recurrent[torch.tensor(hidden)] = torch.tensor((0.9, 0.7, 0.2, 0.1))
    legacy_out, _ = legacy._apply_local_competition(recurrent.clone(), input_packet_count=1)
    adaptive_one, _ = adaptive._apply_local_competition(recurrent.clone(), input_packet_count=1)
    assert torch.equal(legacy_out, adaptive_one)
    adaptive_two, _ = adaptive._apply_local_competition(recurrent.clone(), input_packet_count=2)
    assert torch.count_nonzero(adaptive_two[torch.tensor(hidden)] > 0.0) == 2
    tied = recurrent.clone()
    tied[torch.tensor(hidden)] = torch.tensor((0.9, 0.5, 0.5, 0.1))
    tied_out, _ = adaptive._apply_local_competition(tied, input_packet_count=2)
    assert torch.count_nonzero(tied_out[torch.tensor(hidden)]) == 0


def test_frozen_calibration_composes_two_unseen_atoms() -> None:
    result = analyze_adaptive_artifact(CALIBRATION_INPUT, stage="calibration")
    assert result["status"] == "ADAPTIVE_COMPETITION_CALIBRATION_PASS"
    assert result["atomic_success_total"] == 4
    assert result["adaptive_pair_success_total"] == 4
    assert result["legacy_pair_success_total"] == 0
    assert result["misaligned_pair_success_total"] == 0

