import pytest

from reality_stone.clarus.dpc_benchmark import (
    DPCConfig,
    DPCEpisode,
    belief_mpc_decision,
    evaluate_learned_validation,
    evaluate_validation,
    posterior_positive,
)


def test_action_conditioning_reverses_posterior_direction():
    cfg = DPCConfig(dropout_probability=0.0)
    positive_probe = DPCEpisode(1, 1, 1, 0.5, 2)
    negative_probe = DPCEpisode(1, 1, -1, 0.5, 2)
    assert posterior_positive(positive_probe, cfg, action_conditioned=True) > 0.5
    assert posterior_positive(negative_probe, cfg, action_conditioned=True) < 0.5


def test_horizon_cannot_see_reward_outside_contract():
    cfg = DPCConfig(dropout_probability=0.0)
    episode = DPCEpisode(1, 1, 1, 3.0, 2)
    assert belief_mpc_decision(episode, cfg, horizon=1).action == 0
    assert belief_mpc_decision(episode, cfg, horizon=2).action == 1


def test_validation_directionality_gates_on_small_panel():
    result = evaluate_validation(start_seed=920000, episodes=256)
    assert result["hard_gate"] is True
    assert result["delay2"]["lcb_full_minus_reactive"] > 0.15
    assert result["delay3"]["lcb_full_minus_h1"] > 0.08
    assert result["delay2"]["lcb_full_minus_recurrent"] > -0.03
    assert result["action_sensitivity"] is True


def test_learned_belief_validation_passes_small_panel():
    result = evaluate_learned_validation(
        train_start=910000,
        train_episodes=500,
        validation_start=920000,
        validation_episodes=128,
    )
    assert result["hard_gate"] is True
    assert result["full_model"]["action_conditioned"] is True
    assert result["action_agnostic_model"]["action_conditioned"] is False
