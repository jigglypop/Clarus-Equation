"""Agent loop tests: Critic, Action, Bootstrap (F.4, F.7, F.9)."""

import torch
import pytest
from reality_stone.clarus.agent import (
    compute_critic, select_action_discrete, select_action_continuous,
    bootstrap_operator, agent_step, CriticResult,
    RuntimeAgent, RuntimeAgentConfig, RuntimeAgentStep,
    RuntimeTextAgent, RuntimeTextAgentTurn, TextEnvironment, TextEnvironmentStep,
)
from reality_stone.clarus.constants import BOOTSTRAP_CONTRACTION, CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


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


def make_runtime(dim: int = 16) -> BrainRuntime:
    torch.manual_seed(11)
    weight = torch.randn(dim, dim) * 0.05
    weight = 0.5 * (weight + weight.T)
    weight.fill_diagonal_(0.0)
    return BrainRuntime(
        weight,
        config=BrainRuntimeConfig(
            dim=dim,
            active_ratio=0.25,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            memory_capacity=8,
            stdp_enabled=True,
            stdp_apply_interval=2,
            stdp_lr=0.01,
            stdp_density=1.0,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.0,
        ),
        backend="torch",
        device="cpu",
    )


class TestRuntimeAgent:
    def test_closed_loop_tick_returns_action_and_feedback(self):
        runtime = make_runtime()
        agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=3))

        out = agent.step(
            external_input=torch.linspace(0.0, 0.5, 16),
            force_mode=RuntimeMode.WAKE,
        )

        assert isinstance(out, RuntimeAgentStep)
        assert 0 <= out.action_index < 3
        assert out.runtime_step.step == 1
        assert out.working_memory_size == 1
        assert out.goal_norm > 0.0
        assert out.consciousness_depth > 0.0

    def test_closed_loop_uses_environment_observation_for_critic(self):
        runtime = make_runtime()
        agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=2))
        observation = torch.ones(16) * 0.25

        out = agent.step(
            external_input=torch.zeros(16),
            observation=observation,
            force_mode=RuntimeMode.WAKE,
        )

        assert out.critic.c_pred > 0.0
        assert len(agent.working_memory) == 1
        assert torch.allclose(agent.working_memory.contents()[0][1], observation)

    def test_closed_loop_preserves_runtime_stdp_updates(self):
        runtime = make_runtime()
        agent = RuntimeAgent(runtime, config=RuntimeAgentConfig(action_count=2))

        for _ in range(4):
            out = agent.step(
                external_input=torch.linspace(0.1, 0.6, 16),
                force_mode=RuntimeMode.WAKE,
            )

        assert out.runtime_step.stdp_updates > 0
        assert len(agent.working_memory) == 4


class TestTextEnvironment:
    def test_text_encoding_is_deterministic(self):
        env = TextEnvironment(dim=12)

        a = env.encode("hello clarus")
        b = env.encode("hello clarus")

        assert torch.allclose(a, b)
        assert a.shape == (12,)

    def test_environment_step_returns_text_and_observation(self):
        env = TextEnvironment(dim=8, actions=["answer"])
        env.reset("What is CE?")

        out = env.step(0)

        assert isinstance(out, TextEnvironmentStep)
        assert out.action_label == "answer"
        assert "What is CE?" in out.response
        assert out.observation.shape == (8,)

    def test_runtime_text_agent_runs_prompt_episode(self):
        runtime = make_runtime()
        env = TextEnvironment(dim=16, actions=["answer", "reflect"])
        agent = RuntimeTextAgent(runtime, environment=env)

        turn = agent.ask("Explain the bootstrap loop.", ticks=2)

        assert isinstance(turn, RuntimeTextAgentTurn)
        assert turn.env_step.action_label in env.actions
        assert turn.agent_step.working_memory_size == 2
        assert turn.agent_step.goal_norm > 0.0

    def test_runtime_text_agent_exports_from_package(self):
        import reality_stone.clarus as clarus

        assert clarus.RuntimeTextAgent is RuntimeTextAgent
        assert clarus.TextEnvironment is TextEnvironment
