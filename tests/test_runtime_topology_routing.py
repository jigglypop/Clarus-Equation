import ast
import inspect
import math

import pytest
import torch

from reality_stone.clarus.runtime_topology_routing import (
    ApparatusInvalid, TopologyRoutingConfig, _shared_sparse_budget, construct_route_mask,
    run_binding_route, run_topology_circuit, run_topology_route,
)


def _blocks(): return ((0, 1), (2, 3), (4, 5), (6, 7))


def test_target_free_constructor_and_exact_budget() -> None:
    weight = torch.ones(8, 8) - torch.eye(8)
    cue = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    assert tuple(inspect.signature(construct_route_mask).parameters) == ("weight", "cue", "blocks", "seed", "route", "budget")
    tree = ast.parse(inspect.getsource(construct_route_mask))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not identifiers.intersection({"target", "targets", "decoder", "_decode", "endpoint", "rollout"})
    for route in ("WEIGHT", "CLUSTER", "PATH_ONLY", "TOPOLOGY", "RETURN_SHUFFLED", "RANDOM_MATCHED"):
        mask = construct_route_mask(weight, cue, _blocks(), 7, route, 7)
        assert int(mask.sum()) == 7 and not bool(mask.diagonal().any())


def test_topology_formula_return_changes_path_score() -> None:
    # Weight-only path ranking prefers 0->3, but return support makes 0->2 win.
    weight = torch.zeros(8, 8); weight[2, 0] = 1; weight[0, 2] = 1; weight[3, 0] = 1.1
    cue = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
    path = construct_route_mask(weight, cue, _blocks(), 1, "PATH_ONLY", 1)
    topology = construct_route_mask(weight, cue, _blocks(), 1, "TOPOLOGY", 1)
    assert int(path.sum()) == int(topology.sum()) == 1
    assert bool(path[3, 0])
    assert bool(topology[2, 0])
    assert not torch.equal(path, topology)


def test_shared_budget_is_exact_and_feasible_for_every_cue() -> None:
    weight = torch.ones(8, 8) - torch.eye(8)
    cues = torch.tensor([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
    ])
    budget, minimum = _shared_sparse_budget(weight, cues, _blocks())
    assert budget == math.ceil(0.25 * minimum)
    assert 0 < budget <= minimum
    for cue in cues:
        for route in ("CLUSTER", "PATH_ONLY", "TOPOLOGY", "RETURN_SHUFFLED"):
            assert int(construct_route_mask(weight, cue, _blocks(), 7, route, budget).sum()) == budget


def test_cluster_destination_excludes_every_cue_source_block() -> None:
    weight = torch.zeros(8, 8)
    weight[2, 0] = 10.0  # Strong but points into another cue-active block.
    weight[4, 0] = 1.0   # Best admissible non-source destination.
    weight[6, 2] = 0.9
    cue = torch.tensor([1., 0., 1., 0., 0., 0., 0., 0.])
    mask = construct_route_mask(weight, cue, _blocks(), 1, "CLUSTER", 1)
    assert bool(mask[4, 0])
    assert not bool(mask[2, 0])


def test_degenerate_inputs_fail_closed() -> None:
    with pytest.raises(ApparatusInvalid): construct_route_mask(torch.zeros(8, 8), torch.ones(8), _blocks(), 1, "WEIGHT", 1)
    with pytest.raises(ApparatusInvalid): construct_route_mask(torch.eye(8), torch.ones(8), _blocks(), 1, "WEIGHT", 1)
    with pytest.raises(ApparatusInvalid): construct_route_mask(torch.ones(8, 8) - torch.eye(8), torch.zeros(8), _blocks(), 1, "CLUSTER", 1)
    with pytest.raises(ApparatusInvalid): construct_route_mask(torch.ones(8, 8) - torch.eye(8), torch.ones(8), _blocks(), 1, "WEIGHT", 0)
    sparse = torch.zeros(8, 8); sparse[2, 0] = 1.0; sparse[4, 0] = 0.5
    with pytest.raises(ApparatusInvalid): construct_route_mask(sparse, torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]), _blocks(), 1, "CLUSTER", 2)


def test_actual_delayed_torch_smoke_snapshot_and_receipts() -> None:
    row = run_topology_route(97301, config=TopologyRoutingConfig(dim=16, replay_epochs=1, replay_ticks=1, rollout_horizon=1), route="TOPOLOGY")
    assert row["delay_ring_length"] == 2 and row["snapshot_immutable"] and row["finite"]
    assert row["retained_edges"] == row["edge_budget"]
    assert row["temporal_rows_after"] == 0
    assert row["cutoff_audit"]["hippocampal_rows_after"] == 0
    assert all(math.isfinite(float(row[key])) for key in ("runtime_energy", "active_fraction", "exposed_edge_fraction", "separation", "switch_cost", "topology_path_hamming"))

    binding = run_binding_route(97201, config=TopologyRoutingConfig(dim=16, replay_epochs=1, replay_ticks=1, rollout_horizon=2), route="FULL")
    assert binding["snapshot_immutable"] and binding["finite"]
    assert binding["cutoff_audit"]["temporal_rows_after"] == 0
    assert binding["cutoff_audit"]["hippocampal_rows_after"] == 0

    circuit = run_topology_circuit(97301, config=TopologyRoutingConfig(dim=16, replay_epochs=1, replay_ticks=1, rollout_horizon=1))
    assert set(circuit["routes"]) == {"FULL", "WEIGHT", "CLUSTER", "PATH_ONLY", "TOPOLOGY", "RETURN_SHUFFLED", "RANDOM_MATCHED", "WRONG_CONTEXT"}
    assert {row["source_snapshot_sha256"] for row in circuit["routes"].values()} == {circuit["source_snapshot_sha256"]}
