import pytest

from reality_stone.clarus.reward_transfer_benchmark import (
    RewardTransferConfig,
    UtilityContext,
    action_return,
    choose_action,
    evaluate_reward_transfer,
)


def test_utility_context_changes_commit_threshold() -> None:
    permissive = UtilityContext(0.05, 0.8, 0.0)
    cautious = UtilityContext(0.45, 1.8, 0.1)
    assert choose_action(0.75, permissive) == 1
    assert choose_action(0.75, cautious) == 0
    assert action_return(1, 1, cautious) == pytest.approx(0.9)
    assert action_return(1, -1, cautious) == pytest.approx(-1.9)


def test_small_reward_transfer_benchmark_has_integrity_guards() -> None:
    result = evaluate_reward_transfer(
        RewardTransferConfig(train_episodes=96, validation_episodes=48, policy_epochs=40)
    )
    assert result["schema"] == "clarus.reward-transfer.validation.v1"
    assert result["belief_updates_during_transfer"] == 0
    assert result["policy_updates_during_transfer"] == 0
    assert result["future_reads"] == 0
