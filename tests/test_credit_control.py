import torch

from reality_stone.clarus.credit_control import (
    EligibilityQLearner,
    HomeostaticConfig,
    SignedHomeostaticController,
    TemporalCreditConfig,
)
from reality_stone.clarus.delayed_credit_benchmark import evaluate_delayed_credit


def test_td_error_preserves_reward_sign():
    learner = EligibilityQLearner(TemporalCreditConfig(state_count=2, action_count=2))
    learner.start_episode()
    positive = learner.update(0, 0, 1.0, None, done=True)
    learner.start_episode()
    negative = learner.update(1, 0, -1.0, None, done=True)
    assert positive > 0
    assert negative < 0


def test_homeostasis_preserves_over_under_direction():
    controller = SignedHomeostaticController(
        HomeostaticConfig(unit_count=2, target_rate=0.5, averaging_rate=1.0)
    )
    error = controller.update(torch.tensor([1.0, 0.0]))
    assert error[0] > 0
    assert error[1] < 0
    assert controller.threshold[0] > 0
    assert controller.threshold[1] < 0


def test_delayed_credit_requires_signed_trace():
    result = evaluate_delayed_credit(train_episodes=600, validation_episodes=256)
    assert result["hard_gate"] is True
    rates = result["success_rates"]
    assert rates["signed_td_lambda"] > rates["trace_off"]
    assert rates["signed_td_lambda"] > rates["absolute_td"]

