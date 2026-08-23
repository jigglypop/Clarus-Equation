import torch

from reality_stone.clarus.experiments.runtime_alternative_memory import AlternativeMemoryConfig
from reality_stone.clarus.experiments.runtime_contrastive_predictive_memory import (
    _factor_codebooks,
    _m2_collect_phase,
    _m2_runtime,
    _m3_actual_gated_pre,
    _m3_feature,
    _m3_runtime,
    m2_lagged_contrastive_binding,
    m2_lagged_contrastive_factor_transfer,
    m3_predictor_audit,
    m3_replay_residual_binding,
    m3_replay_residual_factor_transfer,
    t1_m1_factor_transfer,
)
from reality_stone.clarus.runtime import BrainRuntime, RuntimeMode


def test_factor_codebook_excludes_heldout_pair_and_is_deterministic() -> None:
    first = _factor_codebooks(97301, 16)
    second = _factor_codebooks(97301, 16)
    assert first["sha256"] == second["sha256"]
    assert first["held_out"] == (1, 1)
    assert first["held_out"] not in first["train"]


def test_t1_reuses_frozen_m1_and_audits_factor_transfer() -> None:
    config = AlternativeMemoryConfig(
        dim=16, replay_epochs=1, replay_ticks=2, rollout_horizon=2, seed=97301,
    )
    result = t1_m1_factor_transfer(config.seed, config)
    assert result["route"] == "T1_frozen_M1_factor_transfer"
    assert result["heldout_absence_audit"]
    assert result["schedule_parity"]
    assert result["schedule_contract"]
    assert result["codebook_parity"]
    assert result["mid_block_weight_unchanged"]
    assert result["runtime_tick_count"] == result["expected_event_count"]
    assert result["interphase_reset_count"] == result["block_count"]
    assert result["hippocampal_rows_after_rollout"] == 0
    assert result["snapshot_restore_parity"]
    assert result["cutoff_audit"]["temporal_rows_after"] == 0
    assert result["cutoff_audit"]["hippocampal_rows_after"] == 0
    assert result["abstain_threshold"] == config.m1_abstain_threshold == 0.20
    assert result["chance_baseline"] == 0.25
    assert result["decoder_only_baseline_accuracy"] == 0.0
    assert not result["frozen_protocol"]
    assert result["controls"]["zero_gate"]["weight_drift"] == 0.0


def test_m2_fixed_point_and_virtual_collector_orientation() -> None:
    config = AlternativeMemoryConfig(
        dim=16, replay_epochs=1, replay_ticks=2, rollout_horizon=2, seed=97301,
    )
    runtime, residual, projection_passes = _m2_runtime(config.seed, config)
    books = _factor_codebooks(config.seed, config.dim)
    positive = _m2_collect_phase(
        runtime.snapshot(), books["cues"][0], books["targets"][0], config,
    )
    negative = _m2_collect_phase(
        runtime.snapshot(), books["cues"][0], books["targets"][0] * 0.0, config,
    )
    assert residual <= 1e-7
    assert projection_passes == 1
    assert positive["correlation"].shape == (config.dim, config.dim)
    assert positive["first_term_residual"] <= 1e-7
    assert positive["weight_unchanged"]
    assert positive["stdp_updates"] == 0
    assert negative["correlation"].norm().item() <= 1e-7


def test_m2_binding_and_factor_routes_expose_null_negative_phase() -> None:
    config = AlternativeMemoryConfig(
        dim=16, replay_epochs=1, replay_ticks=2, rollout_horizon=2, seed=97301,
    )
    binding = m2_lagged_contrastive_binding(config.seed, config)
    factor = m2_lagged_contrastive_factor_transfer(config.seed, config)
    for result in (binding, factor):
        assert result["schedule_parity"]
        assert result["identical_phase_zero_update"]
        assert result["positive_only_applied_delta_same"]
        assert not result["contrastive_negative_nonzero"]
        assert result["controls"]["no_write"]["applied_delta_norm"] == 0.0
        assert result["automatic_stdp_updates"] == 0
        assert result["snapshot_restore_parity"]
        assert result["hippocampal_rows_after_rollout"] == 0
        assert result["status"] == "STOP"
    assert factor["heldout_absence_audit"]


def test_m3_feature_and_gated_pre_match_native_torch_step() -> None:
    config = AlternativeMemoryConfig(dim=16, seed=97301)
    runtime = _m3_runtime(config.seed, config)
    runtime.step(
        external_input=torch.ones(config.dim),
        force_mode=RuntimeMode.WAKE,
        learning_signal=0.0,
    )
    feature = _m3_feature(
        runtime,
        torch.zeros(config.dim),
        torch.zeros(config.dim),
        RuntimeMode.WAKE,
        replay_present=False,
    )
    pre = _m3_actual_gated_pre(runtime)
    fork = BrainRuntime.from_snapshot(runtime.snapshot(), backend="torch", device="cpu")
    _, recurrent, _ = fork._step_torch(
        torch.zeros(config.dim), torch.zeros(config.dim), RuntimeMode.WAKE,
    )
    torch.testing.assert_close(recurrent, fork._matvec(pre), atol=1e-7, rtol=0.0)
    assert feature.shape == (12 * config.dim + 5,)


def test_m3_predictor_is_frozen_and_uses_disjoint_heldout_rows() -> None:
    config = AlternativeMemoryConfig(dim=16, seed=97301)
    result = m3_predictor_audit(config.seed, config)
    assert result["fit_row_count"] == 64
    assert result["heldout_row_count"] == 16
    assert result["feature_dim"] == 12 * config.dim + 5
    assert result["theta_shape"] == [12 * config.dim + 5, config.dim]
    assert result["theta_frozen_during_score"]
    assert result["fit_score_row_disjoint"]
    assert result["effective_replay_vector_residual_max"] <= 1e-7
    assert result["automatic_stdp_updates"] == 0
    assert result["weight_unchanged"]


def test_m3_replay_residual_controls_preserve_schedule_and_reconstruct_writes() -> None:
    config = AlternativeMemoryConfig(
        dim=16, replay_epochs=1, replay_ticks=2, rollout_horizon=2, seed=97301,
    )
    binding = m3_replay_residual_binding(config.seed, config)
    factor = m3_replay_residual_factor_transfer(config.seed, config)
    for result in (binding, factor):
        assert result["predictor_frozen"]
        assert result["schedule_parity"]
        assert result["mid_block_weight_unchanged"]
        assert result["automatic_stdp_updates"] == 0
        assert result["update_formula_residual_max"] <= 1e-7
        assert result["applied_reconstruction_residual_max"] <= 1e-7
        assert result["installed_norm_residual_max"] <= 1e-7
        assert result["controls"]["predictor_only"]["weight_drift"] == 0.0
        delayed = result["controls"]["one_block_delayed_error"]["block_apply_audits"]
        assert delayed[0]["delayed_source_block"] == -1
        assert delayed[0]["applied_delta_norm"] == 0.0
        if len(delayed) > 1:
            assert delayed[1]["delayed_source_block"] == 0
        shuffled = result["controls"]["transition_order_shuffled"]["block_apply_audits"]
        assert shuffled[0]["residual_credit_permutation"] == [1, 2, 0]
        assert all(
            pair["replay_present"] and pair["replay_value"] == "zero"
            for pair in result["controls"]["no_replay"]["learning_pairs"]
        )
        assert result["snapshot_restore_parity"]
        assert result["hippocampal_rows_after_rollout"] == 0
    assert factor["heldout_absence_audit"]
