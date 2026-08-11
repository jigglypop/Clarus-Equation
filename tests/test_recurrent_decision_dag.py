import math

import pytest

from reality_stone.clarus.recurrent_decision_dag import (
    DagEdge,
    DagNode,
    RecurrentDagConfig,
    RecurrentDecisionDag,
    validate_topology,
)


def test_recurrent_dag_is_finite_and_commits_only_after_forward() -> None:
    model = RecurrentDecisionDag()
    before = model.state
    output = model.forward_step((1.0, -1.0, 1.0), (1.0, 0.0, 0.0, 0.0))
    assert model.state == before
    assert math.isclose(sum(output.probabilities), 1.0)
    assert output.evaluated_nodes == len(model.nodes)
    assert output.evaluated_edges == len(model.edges)
    model.commit_feedback(1.0)
    assert model.state != before
    assert math.sqrt(sum(value**2 for value in model.state)) <= model.config.state_norm_cap


def test_recurrent_dag_rejects_same_depth_or_reverse_edge() -> None:
    nodes = (DagNode(0, 0, "input"), DagNode(1, 1, "action"))
    with pytest.raises(ValueError, match="topological depth"):
        validate_topology(nodes, (DagEdge(1, 0),))


def test_recurrent_dag_requires_causal_feedback_order() -> None:
    model = RecurrentDecisionDag(RecurrentDagConfig())
    with pytest.raises(RuntimeError, match="preceding forward"):
        model.commit_feedback(1.0)
    model.forward_step((1.0, 1.0, -1.0), (0.0, 1.0, 0.0, 0.0))
    model.commit_feedback(-1.0, flip_sign=True)
    with pytest.raises(RuntimeError, match="preceding forward"):
        model.commit_feedback(1.0)


def test_soft_content_preserves_all_action_support_and_pending_order() -> None:
    model = RecurrentDecisionDag(
        RecurrentDagConfig(soft_content=True, strict_causal_order=True)
    )
    output = model.forward_step((0.2, -0.3, 0.1), (0.5, -0.2, 0.0, 0.1))
    assert all(probability > 0.0 for probability in output.probabilities)
    assert math.isclose(sum(output.probabilities), 1.0)
    with pytest.raises(RuntimeError, match="pending decision"):
        model.forward_step((0.2, -0.3, 0.1), (0.5, -0.2, 0.0, 0.1))
    model.commit_feedback(-1.0)


def test_context_boundary_is_directional_and_positive_safe() -> None:
    model = RecurrentDecisionDag(
        RecurrentDagConfig(soft_content=True, strict_causal_order=True)
    )
    model.forward_step((1.0, -1.0, 1.0), (2.0, 0.0, 0.0, 0.0))
    positive = model.commit_feedback_with_context_boundary(1.0)
    assert positive.reset_strength == 0.0
    model.forward_step((1.0, -1.0, 1.0), (2.0, 0.0, 0.0, 0.0))
    negative = model.commit_feedback_with_context_boundary(-1.0)
    assert math.isclose(negative.reset_strength, negative.confidence)
    assert negative.state_norm_after_labilization <= negative.state_norm_before + 1e-12
    assert negative.orthogonal_error <= 1e-12
