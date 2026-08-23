import torch

from reality_stone.clarus.experiments.runtime_alternative_memory import (
    AlternativeMemoryConfig,
    DelayedSignedEligibility,
    _m1_runtime,
    m0_capacity_rank_sweep,
    m1_delayed_three_factor,
)


def test_delayed_signed_eligibility_uses_post_pre_direction() -> None:
    config = AlternativeMemoryConfig(
        dim=4,
        replay_epochs=1,
        replay_ticks=1,
        m1_trace_decay=1.0,
        m1_eligibility_decay=1.0,
        m1_ltp=1.0,
        m1_ltd=0.2,
    )
    tracker = DelayedSignedEligibility(config)
    cue, target = torch.zeros(4), torch.zeros(4)
    cue[0], target[1] = 1.0, 1.0
    tracker.observe(cue)
    tracker.observe(target)
    assert tracker.eligibility[1, 0] > 0.0
    assert tracker.eligibility[1, 0] > tracker.eligibility[0, 1]


def test_m0_rank_sweep_is_deterministic_and_controls_are_native() -> None:
    config = AlternativeMemoryConfig(dim=16, replay_epochs=1, replay_ticks=1, rollout_horizon=2, seed=97201)
    first = m0_capacity_rank_sweep(config.seed, config)
    second = m0_capacity_rank_sweep(config.seed, config)
    assert first == second
    assert first["route"] == "M0_supervised_low_rank_capacity_ceiling"
    assert set(first["results"]) == {"1", "2", "4", "full"}
    for result in first["results"].values():
        assert result["dale_law"] is False
        assert result["structural_projection_used"] is False
        assert result["hippocampal_rows_after_rollout"] == 0
        assert result["snapshot_restore_parity"]
        assert result["dense_sparse_parity"]
        random_control = result["controls"]["random_low_rank"]
        assert random_control["random_spectrum_parity"]
        assert random_control["installed_write_norm"] <= config.max_write_norm + 1e-6


def test_m1_uses_fixed_clock_equal_schedules_and_zero_store_probes() -> None:
    config = AlternativeMemoryConfig(
        dim=16,
        replay_epochs=1,
        replay_ticks=2,
        rollout_horizon=2,
        seed=97201,
    )
    result = m1_delayed_three_factor(config.seed, config)
    assert result["route"] == "M1_fixed_clock_delayed_three_factor"
    assert result["gate_audit"] == {
        "source": "fixed_block_end_clock",
        "base_value": 1.0,
        "reads_runtime_state": False,
        "reads_reward": False,
        "reads_target_identity": False,
        "reads_decoder": False,
        "reads_memory_value": False,
        "reads_condition_flag": False,
        "lesion_override": "none",
    }
    assert result["schedule_parity"]
    assert result["mid_block_weight_unchanged"]
    assert result["runtime_tick_count"] == result["expected_event_count"]
    assert result["interphase_reset_count"] == result["block_count"]
    assert result["hippocampal_rows_after_rollout"] == 0
    assert result["cutoff_audit"]["temporal_rows_after"] == 0
    assert result["cutoff_audit"]["hippocampal_rows_after"] == 0
    assert result["snapshot_restore_parity"]
    assert result["dense_sparse_parity"]
    assert result["abstain_threshold"] == config.m1_abstain_threshold == 0.20
    assert result["controls"]["zero_gate"]["weight_drift"] == 0.0
    for control in result["controls"].values():
        assert control["block_count"] == result["block_count"]
        assert control["pulse_count"] == result["pulse_count"]
        assert control["event_count"] == result["event_count"]
        assert control["interphase_reset_count"] == result["interphase_reset_count"]


def test_m1_memory_binding_survives_neuronwise_thresholds() -> None:
    dim = 16
    active = tuple(0.18 + 0.08 * i / (dim - 1) for i in range(dim))
    bit_lower = tuple(0.06 + 0.08 * i / (dim - 1) for i in range(dim))
    bit_upper = tuple(0.24 + 0.12 * i / (dim - 1) for i in range(dim))
    config = AlternativeMemoryConfig(
        dim=dim,
        replay_epochs=1,
        replay_ticks=2,
        rollout_horizon=2,
        neuronwise_active_threshold=active,
        neuronwise_bit_lower_threshold=bit_lower,
        neuronwise_bit_upper_threshold=bit_upper,
        seed=97201,
    )

    runtime = _m1_runtime(config)
    assert runtime.config.effective_active_thresholds() == active
    assert runtime.config.effective_bit_thresholds() == (bit_lower, bit_upper)

    result = m1_delayed_three_factor(config.seed, config)
    assert result["status"] == "GO"
    assert result["clean_accuracy"] == 1.0
    assert result["corrupt_accuracy"] == 1.0
    assert result["control_advantage"] == 1.0
    assert result["hippocampal_rows_after_rollout"] == 0
