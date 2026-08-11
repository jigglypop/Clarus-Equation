import torch
import pytest

from reality_stone.clarus.belief_control import (
    BeliefControlConfig,
    BeliefController,
)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def make_controller(**kwargs) -> BeliefController:
    config = BeliefControlConfig(observation_dim=2, action_count=2, **kwargs)
    return BeliefController(config, loading=torch.tensor([1.0, 0.0]))


def assert_same_plan(left, right) -> None:
    assert left.action_index == right.action_index
    assert left.sequence == right.sequence
    assert left.sequence_cost == pytest.approx(right.sequence_cost)
    assert left.trust == pytest.approx(right.trust)
    assert torch.equal(left.action_costs, right.action_costs)
    assert torch.equal(left.predicted_observation, right.predicted_observation)
    assert torch.equal(left.predicted_variance, right.predicted_variance)


def learn_symmetric_effects(controller: BeliefController, repeats: int = 40) -> None:
    effects = (torch.tensor([-0.5, 0.0]), torch.tensor([0.5, 0.0]))
    state = torch.zeros(2)
    controller.observe(state)
    for step in range(repeats):
        action = step % 2
        goal = effects[action]
        plan = controller.plan(state, goal, action_free_base_transition=identity)
        # Commit the desired exploration action without allowing the planner to
        # choose a different training action.
        plan = type(plan)(
            action_index=action,
            sequence=(action,) * controller.config.horizon,
            sequence_cost=plan.sequence_cost,
            action_costs=plan.action_costs,
            predicted_observation=plan.predicted_observation,
            predicted_variance=plan.predicted_variance,
            trust=plan.trust,
        )
        controller.commit(plan, base_prediction=state)
        state = state + effects[action]
        controller.observe(state)


def test_action_effect_is_learned_only_for_committed_action():
    controller = make_controller(horizon=2)
    controller.observe(torch.zeros(2))
    plan = controller.plan(torch.zeros(2), torch.tensor([1.0, 0.0]), action_free_base_transition=identity)
    before = controller.action_effect.clone()
    controller.commit(plan, base_prediction=torch.zeros(2))
    assert torch.equal(before, controller.action_effect)
    controller.observe(torch.tensor([0.8, 0.0]))
    other = 1 - plan.action_index
    assert not torch.equal(before[:, plan.action_index], controller.action_effect[:, plan.action_index])
    assert torch.equal(before[:, other], controller.action_effect[:, other])


def test_goal_changes_planned_action_after_effect_learning():
    controller = make_controller(horizon=2, action_effect_lr=0.25)
    learn_symmetric_effects(controller)
    state = controller.last_observation.clone()
    left = controller.plan(state, state + torch.tensor([-0.8, 0.0]), action_free_base_transition=identity)
    right = controller.plan(state, state + torch.tensor([0.8, 0.0]), action_free_base_transition=identity)
    assert left.action_index == 0
    assert right.action_index == 1


def test_plan_is_pure_and_deterministic():
    controller = make_controller(horizon=2)
    controller.observe(torch.zeros(2))
    before = controller.state_dict()
    first = controller.plan(torch.zeros(2), torch.ones(2), action_free_base_transition=identity)
    second = controller.plan(torch.zeros(2), torch.ones(2), action_free_base_transition=identity)
    after = controller.state_dict()
    assert_same_plan(first, second)
    for key in ("posterior_mean", "posterior_variance", "action_effect", "action_counts"):
        assert torch.equal(before[key], after[key])


def test_uncertainty_reduces_trust():
    controller = make_controller()
    controller.posterior_mean = torch.tensor(1.0)
    controller.posterior_variance = torch.tensor(0.01)
    low_uncertainty = controller._trust(controller.posterior_mean, controller.posterior_variance)
    controller.posterior_variance = torch.tensor(100.0)
    high_uncertainty = controller._trust(controller.posterior_mean, controller.posterior_variance)
    assert high_uncertainty < low_uncertainty


def test_outlier_is_robustly_downweighted():
    controller = make_controller(robust_threshold=1.0)
    controller.observe(torch.zeros(2))
    plan = controller.plan(torch.zeros(2), torch.zeros(2), action_free_base_transition=identity)
    controller.commit(plan, base_prediction=torch.zeros(2))
    update = controller.observe(torch.tensor([100.0, -100.0]))
    assert update.robust_weight < 0.1
    assert torch.isfinite(controller.action_effect).all()


def test_state_dict_round_trip_exact_continuation():
    controller = make_controller(horizon=2)
    learn_symmetric_effects(controller, repeats=6)
    restored = make_controller(horizon=2)
    restored.load_state_dict(controller.state_dict())
    goal = controller.last_observation + torch.tensor([1.0, 0.0])
    assert_same_plan(
        restored.plan(restored.last_observation, goal, action_free_base_transition=identity),
        controller.plan(controller.last_observation, goal, action_free_base_transition=identity),
    )


def test_state_dict_fails_closed_on_shape_mismatch():
    controller = make_controller()
    state = controller.state_dict()
    state["action_effect"] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="action_effect"):
        controller.load_state_dict(state)
