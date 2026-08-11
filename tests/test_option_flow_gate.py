import math

import numpy as np
import pytest

from reality_stone.clarus.option_flow_gate import (
    HoldGateConfig,
    OptionEdge,
    OptionNode,
    edge_responsibilities,
    gpi_hold_gate,
    route_option_flow,
    validate_option_dag,
)


def test_stn_changes_hold_without_changing_conditional_action_identity() -> None:
    low_conflict = gpi_hold_gate(
        (8.0, -8.0),
        (3.0, 1.0),
        (0.5, 1.5),
    )
    high_conflict = gpi_hold_gate(
        (0.0, 0.0),
        (3.0, 1.0),
        (0.5, 1.5),
    )
    assert high_conflict.stn_drive > low_conflict.stn_drive
    assert high_conflict.hold_probability > low_conflict.hold_probability
    assert np.allclose(
        high_conflict.conditional_action_probabilities,
        low_conflict.conditional_action_probabilities,
        atol=1e-14,
    )
    assert high_conflict.normalization_error <= 1e-14


def test_action_only_softmax_common_stn_offset_is_exact_no_effect() -> None:
    logits = np.asarray((2.0, 0.3, -1.2), dtype=np.float64)
    baseline = np.exp(logits - np.max(logits))
    baseline /= np.sum(baseline)
    shifted = np.exp((logits - 7.0) - np.max(logits - 7.0))
    shifted /= np.sum(shifted)
    assert np.allclose(baseline, shifted, rtol=0.0, atol=1e-15)


def _shared_dag() -> tuple[tuple[OptionNode, ...], tuple[OptionEdge, ...]]:
    nodes = (
        OptionNode(0, 0, "root"),
        OptionNode(1, 1, "goal"),
        OptionNode(2, 1, "goal"),
        OptionNode(3, 2, "shared"),
        OptionNode(4, 2, "leaf", action_label=1),
        OptionNode(5, 2, "leaf", action_label=2),
        OptionNode(6, 3, "leaf", action_label=0),
    )
    edges = (
        OptionEdge(0, 1),
        OptionEdge(0, 2),
        OptionEdge(1, 3),
        OptionEdge(1, 4),
        OptionEdge(2, 3),
        OptionEdge(2, 5),
        OptionEdge(3, 6),
    )
    return nodes, edges


def test_reconvergent_dag_conserves_mass_and_shared_flow() -> None:
    nodes, edges = _shared_dag()
    probabilities = {
        (0, 1): 0.5,
        (0, 2): 0.5,
        (1, 3): 0.5,
        (1, 4): 0.5,
        (2, 3): 0.5,
        (2, 5): 0.5,
        (3, 6): 1.0,
    }
    output = route_option_flow(
        nodes,
        edges,
        root_id=0,
        edge_probabilities=probabilities,
        hold_probabilities={},
    )
    assert output.normalization_error <= 1e-14
    assert math.isclose(output.probability_of(0), 0.5)
    assert math.isclose(output.probability_of(1), 0.25)
    assert math.isclose(output.probability_of(2), 0.25)
    assert math.isclose(dict(output.node_flows)[3], 0.5)


def test_multi_path_credit_is_split_by_posterior_responsibility() -> None:
    nodes, edges = _shared_dag()
    probabilities = {
        (0, 1): 0.5,
        (0, 2): 0.5,
        (1, 3): 0.5,
        (1, 4): 0.5,
        (2, 3): 0.5,
        (2, 5): 0.5,
        (3, 6): 1.0,
    }
    output = route_option_flow(
        nodes,
        edges,
        root_id=0,
        edge_probabilities=probabilities,
        hold_probabilities={},
    )
    responsibility = edge_responsibilities(
        nodes,
        edges,
        chosen_action=0,
        flow=output,
        edge_probabilities=probabilities,
    )
    assert math.isclose(responsibility[(0, 1)], 0.5)
    assert math.isclose(responsibility[(0, 2)], 0.5)
    assert math.isclose(responsibility[(1, 3)], 0.5)
    assert math.isclose(responsibility[(2, 3)], 0.5)
    assert math.isclose(responsibility[(3, 6)], 1.0)


def test_option_dag_rejects_reverse_edges_and_nonconserved_local_gate() -> None:
    nodes, edges = _shared_dag()
    with pytest.raises(ValueError, match="increase depth"):
        validate_option_dag(nodes, (*edges, OptionEdge(3, 1)))
    with pytest.raises(ValueError, match="exactly one unit"):
        route_option_flow(
            nodes,
            edges,
            root_id=0,
            edge_probabilities={(0, 1): 0.6, (0, 2): 0.6},
            hold_probabilities={},
        )


def test_hold_mass_is_terminal_and_normalized() -> None:
    nodes = (
        OptionNode(0, 0, "root"),
        OptionNode(1, 1, "leaf", action_label=0),
        OptionNode(2, 1, "leaf", action_label=1),
    )
    edges = (OptionEdge(0, 1), OptionEdge(0, 2))
    output = route_option_flow(
        nodes,
        edges,
        root_id=0,
        edge_probabilities={(0, 1): 0.3, (0, 2): 0.2},
        hold_probabilities={0: 0.5},
    )
    assert math.isclose(output.hold_probability, 0.5)
    assert output.normalization_error <= 1e-14


def test_gate_rejects_dimensional_or_sign_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        gpi_hold_gate((0.0, 1.0), (-1.0, 1.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="positive"):
        HoldGateConfig(gpi_temperature=0.0)
