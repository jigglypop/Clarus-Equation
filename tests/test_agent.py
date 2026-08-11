"""Agent loop tests: Critic, Action, Bootstrap (F.4, F.7, F.9)."""

import torch
import pytest
from reality_stone.clarus.agent import (
    compute_critic,
    select_action_discrete,
    select_action_continuous,
    bootstrap_operator,
    agent_step,
    CriticResult,
    RuntimeAgent,
    RuntimeAgentConfig,
    RuntimeAgentStep,
    RuntimeTextAgent,
    RuntimeTextAgentTurn,
    TextEnvironment,
    TextEnvironmentStep,
    cosine_action_evidence,
)
from reality_stone.clarus.constants import (
    BOOTSTRAP_CONTRACTION,
    CRITIC_W_PRED,
    CRITIC_W_CONS,
    CRITIC_W_NOV,
)
from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from reality_stone.clarus.belief_control import BeliefControlConfig, BeliefController
from reality_stone.clarus.adaptive_scc_tower_controller import AdaptiveTowerController
from reality_stone.clarus.nested_scc_tower import NestedTowerGenerator, TowerSpec


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

    def test_cosine_action_evidence_is_bounded_and_dimensionless(self):
        observation = torch.tensor([2.0, 0.0, 0.0])
        actions = torch.tensor([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-1.0, 0.0, 0.0]])
        assert cosine_action_evidence(observation, actions) == pytest.approx((1.0, 0.0, -1.0))

    def test_cosine_action_evidence_zero_norm_is_zero(self):
        assert cosine_action_evidence(torch.zeros(2), torch.eye(2)) == (0.0, 0.0)


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
                x,
                torch.zeros(8),
                torch.zeros(8),
                torch.zeros(8),
                torch.zeros(8),
                target=target,
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

    def test_belief_control_requires_external_task_goal(self):
        runtime = make_runtime()
        agent = RuntimeAgent(
            runtime,
            config=RuntimeAgentConfig(action_count=2, belief_control_enabled=True),
        )
        with pytest.raises(ValueError, match="task_goal"):
            agent.step(observation=torch.zeros(16), force_mode=RuntimeMode.WAKE)
        assert runtime.step_index == 0

    def test_disabled_belief_control_keeps_legacy_action_path(self):
        runtime = make_runtime()
        agent = RuntimeAgent(
            runtime,
            config=RuntimeAgentConfig(action_count=2, belief_control_enabled=False),
            belief_controller=object(),
        )
        out = agent.step(
            external_input=torch.linspace(0.0, 0.5, 16),
            force_mode=RuntimeMode.WAKE,
        )
        expected = select_action_discrete(runtime.activation, agent.action_embeddings)
        assert agent.belief_controller is None
        assert out.action_index == expected
        assert out.belief_plan is None
        assert out.belief_update is None

    def test_belief_control_uses_goal_conditioned_plan(self):
        runtime = make_runtime()
        controller = BeliefController(
            BeliefControlConfig(observation_dim=16, action_count=2, horizon=2),
            loading=torch.nn.functional.one_hot(torch.tensor(0), 16).float(),
        )
        controller.action_effect[:, 0] = -0.25
        controller.action_effect[:, 1] = 0.25
        agent = RuntimeAgent(
            runtime,
            config=RuntimeAgentConfig(action_count=2, belief_control_enabled=True),
            belief_controller=controller,
        )
        observation = torch.zeros(16)
        out = agent.step(
            observation=observation,
            task_goal=torch.ones(16),
            force_mode=RuntimeMode.WAKE,
        )
        assert out.action_index == 1
        assert out.belief_plan is not None
        assert out.belief_update is not None

    def test_nested_scc_action_is_read_from_issued_state_token(self):
        runtime = make_runtime(dim=4)
        controller = AdaptiveTowerController(
            NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=2))
        )
        embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        agent = RuntimeAgent(
            runtime,
            action_embeddings=embeddings,
            config=RuntimeAgentConfig(action_count=2, nested_scc_enabled=True),
            nested_scc_controller=controller,
        )

        out = agent.step(
            observation=torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            action_mask=(True, True),
            force_mode=RuntimeMode.WAKE,
        )

        assert out.nested_scc_token is controller.latest_token
        assert out.nested_scc_policy == controller.read_policy(out.nested_scc_token, (True, True))
        assert out.action_index == out.nested_scc_policy.selected_action
        assert out.nested_scc_evidence == pytest.approx((-1.0, 1.0))
        assert sum(out.nested_scc_policy.probabilities) == pytest.approx(1.0, abs=1e-12)
        assert out.nested_scc_audit.bounded
        assert out.nested_scc_audit.state_mediated
        assert out.nested_scc_audit.evidence_sup_norm <= 1.0
        assert out.nested_scc_audit.state_sup_norm <= 1.0

    def test_nested_scc_history_changes_policy_under_same_current_observation(self):
        embeddings = torch.eye(2)
        first_runtime = make_runtime(dim=2)
        second_runtime = make_runtime(dim=2)
        first = RuntimeAgent(
            first_runtime,
            action_embeddings=embeddings,
            config=RuntimeAgentConfig(action_count=2, nested_scc_enabled=True),
        )
        second = RuntimeAgent(
            second_runtime,
            action_embeddings=embeddings,
            config=RuntimeAgentConfig(action_count=2, nested_scc_enabled=True),
        )
        first.step(observation=torch.tensor([1.0, 0.0]), force_mode=RuntimeMode.WAKE)
        second.step(observation=torch.tensor([0.0, 1.0]), force_mode=RuntimeMode.WAKE)

        first_out = first.step(observation=torch.tensor([0.5, 0.5]), force_mode=RuntimeMode.WAKE)
        second_out = second.step(observation=torch.tensor([0.5, 0.5]), force_mode=RuntimeMode.WAKE)

        assert first_out.nested_scc_evidence == pytest.approx(second_out.nested_scc_evidence)
        assert first_out.nested_scc_policy.probabilities != pytest.approx(
            second_out.nested_scc_policy.probabilities
        )

    def test_nested_scc_mask_and_invalid_observation_fail_before_runtime_step(self):
        runtime = make_runtime(dim=2)
        agent = RuntimeAgent(
            runtime,
            action_embeddings=torch.eye(2),
            config=RuntimeAgentConfig(action_count=2, nested_scc_enabled=True),
        )
        with pytest.raises(ValueError, match="at least one"):
            agent.step(observation=torch.zeros(2), action_mask=(False, False))
        assert runtime.step_index == 0
        with pytest.raises(ValueError, match="finite"):
            agent.step(observation=torch.tensor([float("nan"), 0.0]))
        assert runtime.step_index == 0

    def test_nested_scc_and_belief_control_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="cannot be enabled together"):
            RuntimeAgentConfig(belief_control_enabled=True, nested_scc_enabled=True)

    def test_disabled_nested_scc_ignores_supplied_controller_and_preserves_legacy_path(self):
        runtime = make_runtime(dim=2)
        controller = AdaptiveTowerController(
            NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=1))
        )
        agent = RuntimeAgent(
            runtime,
            action_embeddings=torch.eye(2),
            config=RuntimeAgentConfig(action_count=2, nested_scc_enabled=False),
            nested_scc_controller=controller,
        )
        out = agent.step(observation=torch.tensor([1.0, 0.0]), force_mode=RuntimeMode.WAKE)
        assert agent.nested_scc_controller is None
        assert out.nested_scc_token is None
        assert out.action_index == select_action_discrete(
            runtime.activation, agent.action_embeddings
        )


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
        assert clarus.AdaptiveTowerController is AdaptiveTowerController
        assert clarus.NestedTowerGenerator is NestedTowerGenerator
        assert clarus.cosine_action_evidence is cosine_action_evidence
