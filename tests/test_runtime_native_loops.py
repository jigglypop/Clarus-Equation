import torch

from reality_stone.clarus.runtime_native_loops import (
    NativeLoopsConfig, _codebook, _detach, _loop8_replay_source_audit, _probe_rollout, _runtime, loop6_temporal_selection, loop7_agent_route, loop8_native_replay, loop8_route_b, loop9_route_b, run_native_loops,
)
from reality_stone.clarus.temporal_memory import TemporalAuditedMemory, TemporalMemoryEvent


def test_loop6_and_native_runner_are_deterministic_and_report_all_loops():
    config = NativeLoopsConfig(dim=16, replay_epochs=2, rollout_horizon=2)
    first = run_native_loops(97101, config=config)
    second = run_native_loops(97101, config=config)
    assert loop6_temporal_selection()["status"] == "GO"
    assert first == second
    assert set(first) >= {"loop6", "loop7", "loop8", "loop9", "loop10"}
    assert first["loop8"]["weight_drift"] > 0.0


def test_cutoff_physically_replaces_both_external_stores():
    config = NativeLoopsConfig(dim=16)
    runtime = _runtime(7, config)
    temporal = TemporalAuditedMemory()
    temporal.ingest(TemporalMemoryEvent("s", "r", "v", 1, 1, "e"))
    runtime.step(external_input=torch.ones(16))
    audit = _detach(runtime, temporal)
    assert audit["temporal_rows_removed"] == 1
    assert audit["hippocampal_rows_removed"] > 0
    assert audit["temporal_rows_after"] == audit["hippocampal_rows_after"] == 0


def test_sealed_snapshot_probes_are_order_independent_and_cannot_reencode_memory():
    config = NativeLoopsConfig(dim=16, rollout_horizon=2)
    runtime, temporal = _runtime(11, config), TemporalAuditedMemory()
    runtime.step(external_input=torch.ones(16))
    temporal.ingest(TemporalMemoryEvent("s", "r", "v", 1, 1, "e"))
    _detach(runtime, temporal)
    sealed = runtime.snapshot()
    cue_a, cue_b = torch.eye(16)[0], torch.eye(16)[1]
    _, first_a, rows_a = _probe_rollout(sealed, cue_a, config)
    _probe_rollout(sealed, cue_b, config)
    _, second_a, rows_second_a = _probe_rollout(sealed, cue_a, config)
    assert torch.equal(first_a, second_a)
    assert rows_a == rows_second_a == 0


def test_route_b_is_separately_labeled_and_uses_only_bounded_runtime_write():
    config = NativeLoopsConfig(dim=16, replay_epochs=1, rollout_horizon=1)
    loop8 = loop8_route_b(19, config)
    loop9 = loop9_route_b(19, config)
    assert loop8["route"] == "B_bounded_supervised_recurrent_projection"
    assert loop8["installed_write_norm"] > 0.0
    assert loop8["controls"]["no_write"]["installed_write_norm"] == 0.0
    assert loop8["hippocampal_rows_after_rollout"] == 0
    assert loop9["route"] == "B_bounded_supervised_recurrent_projection"


def test_independent_orthogonal_codebook_and_native_projection_meet_development_gates():
    config = NativeLoopsConfig(seed=97101)
    cues, targets = _codebook(config.seed, config.dim)
    identity = torch.eye(len(cues))
    assert torch.allclose(cues @ cues.T, identity, atol=1e-6)
    assert torch.allclose(targets @ targets.T, identity, atol=1e-6)
    assert torch.equal(cues @ targets.T, torch.zeros_like(identity))

    loop8 = loop8_route_b(config.seed, config)
    loop9 = loop9_route_b(config.seed, config)
    assert loop8["status"] == "GO"
    assert loop8["clean_accuracy"] == 1.0
    assert loop8["controls"]["target_shuffled"]["clean_accuracy"] == 0.0
    assert loop8["cutoff_audit"]["temporal_rows_after"] == 0
    assert loop8["hippocampal_rows_after_rollout"] == 0
    assert loop9["status"] == "GO"
    assert loop9["native_accuracy"] == 1.0
    assert loop9["target_shuffled_accuracy"] == 0.0
    assert loop9["cutoff_audit"]["hippocampal_rows_after_rollout"] == 0


def test_loop7_context_precedence_is_a_zero_read_benchmark_condition():
    result = loop7_agent_route(23, NativeLoopsConfig(dim=16))
    assert result["context_precedence_accuracy"] == 1.0
    assert result["context_temporal_reads"] == 0
    assert result["disabled_temporal_reads"] == 0
    assert result["disabled_matches_base"]
    assert result["status"] == "GO"


def test_loop8_replay_source_uses_latest_valid_manifest_not_arrival_last():
    _, forward, arrival_last = _loop8_replay_source_audit()
    _, reversed_arrival, _ = _loop8_replay_source_audit(reverse_arrival=True)
    assert forward == reversed_arrival
    assert forward == [
        {"key": ["episode-0", "target"], "value": "0", "evidence": "e0-current", "session": 3},
        {"key": ["episode-2", "target"], "value": "2", "evidence": "e2-current", "session": 2},
        {"key": ["episode-3", "target"], "value": "3", "evidence": "e3-current", "session": 2},
    ]
    assert all(entry["key"][0] != "episode-1" for entry in forward)
    assert forward != arrival_last

    result = loop8_native_replay(31, NativeLoopsConfig(dim=16, replay_epochs=1, rollout_horizon=1))
    assert result["replay_source_audit"] == [dict(entry, replayed=True) for entry in forward]
    assert result["replay_source_reverse_equal"]
    assert result["arrival_last_mismatch"]
    assert result["controls"]["no_replay"]["replay_source_audit"] == [dict(entry, replayed=False) for entry in forward]
