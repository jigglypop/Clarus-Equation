"""Agent loop tests: Critic, Action, Bootstrap (F.4, F.7, F.9)."""

import torch
import pytest
from clarus.agent import (
    compute_critic, select_action_discrete, select_action_continuous,
    bootstrap_operator, agent_step, CriticResult,
)
from clarus.constants import BOOTSTRAP_CONTRACTION


class TestCritic:
    def test_critic_result_fields(self):
        obs = torch.randn(16)
        pred = torch.randn(16)
        z = torch.randn(16)
        recalled = torch.randn(16)
        c = compute_critic(obs, pred, z, recalled)
        assert isinstance(c, CriticResult)
        assert c.c_pred >= 0
        assert c.c_cons >= 0
        assert c.c_nov >= 0
        assert c.score >= 0

    def test_critic_perfect_prediction(self):
        obs = torch.randn(16)
        c = compute_critic(obs, obs, obs, obs)
        assert c.c_pred == pytest.approx(0.0, abs=1e-5)
        assert c.c_cons == pytest.approx(0.0, abs=1e-5)

    def test_critic_weights_sum_one(self):
        from clarus.constants import CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV
        assert CRITIC_W_PRED + CRITIC_W_CONS + CRITIC_W_NOV == pytest.approx(1.0)

    def test_critic_with_novelty(self):
        obs = torch.randn(16)
        pred = torch.randn(16)
        prior = torch.randn(16)
        c = compute_critic(obs, pred, obs, obs, obs_prior=prior)
        assert c.c_nov >= 0


class TestAction:
    def test_discrete_selection(self):
        z_out = torch.randn(8)
        actions = torch.randn(5, 8)
        idx = select_action_discrete(z_out, actions)
        assert 0 <= idx < 5

    def test_discrete_selects_most_similar(self):
        z_out = torch.tensor([1.0, 0.0, 0.0])
        actions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        idx = select_action_discrete(z_out, actions)
        assert idx == 0

    def test_continuous_action(self):
        z = torch.randn(4)
        w = torch.randn(3, 4)
        b = torch.randn(3)
        a = select_action_continuous(z, w, b)
        assert a.shape == (3,)


class TestBootstrap:
    def test_contraction(self):
        x = torch.randn(8)
        target = torch.zeros(8)
        y = bootstrap_operator(x, target)
        assert (y - target).norm() < (x - target).norm()

    def test_contraction_rate(self):
        x = torch.ones(4)
        target = torch.zeros(4)
        y = bootstrap_operator(x, target)
        ratio = y.norm() / x.norm()
        assert ratio == pytest.approx(BOOTSTRAP_CONTRACTION, abs=0.01)

    def test_fixed_point(self):
        target = torch.ones(4) * 0.5
        y = bootstrap_operator(target, target)
        assert torch.allclose(y, target, atol=1e-6)

    def test_agent_step_contracts(self):
        x = torch.randn(8)
        target = torch.zeros(8)
        for _ in range(20):
            x = agent_step(
                x, torch.zeros(8), torch.zeros(8),
                torch.zeros(8), torch.zeros(8), target=target,
            )
        assert x.norm().item() < 0.1
