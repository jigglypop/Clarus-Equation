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
