import ast
import inspect
import textwrap

import pytest
import torch

from reality_stone.clarus.runtime_context_branch_routing import architectural_blocks
from reality_stone.clarus.runtime_context_learned_gate import (
    FORBIDDEN_GATE_NAMES,
    GateSnapshot,
    LearnedContextGateConfig,
    LocalContextGate,
    compile_learned_mask,
    run_learned_context_gate_seed,
)


def _fixture_weight() -> torch.Tensor:
    weight = torch.zeros(20, 20)
    s0, s1, h0, h1, output = architectural_blocks(20)
    for index in range(4):
        weight[h0[index], s0[index]] = 1.0
        weight[h1[index], s1[index]] = 1.0
        weight[output[index], h0[index]] = 0.7
        weight[output[index], h1[index]] = 0.7
    return weight


def _identifiers(function) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_gate_update_and_compiler_depend_only_on_frozen_theta_and_cue() -> None:
    config = LearnedContextGateConfig(seed=97601)
    gate = LocalContextGate(config)
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q1 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    gate.observe(q0, torch.tensor([0.0, 0.5]))
    gate.observe(q1, torch.tensor([0.5, 0.0]))
    frozen = gate.snapshot()
    weight = _fixture_weight()
    blocks = architectural_blocks(20)

    mask0, info0 = compile_learned_mask(frozen, q0, weight, blocks)
    mask1, info1 = compile_learned_mask(frozen, q1, weight, blocks)
    assert (info0["selected_branch"], info1["selected_branch"]) == (1, 0)
    assert int(mask0.sum()) == int(mask1.sum()) == 12
    assert int((mask0 != mask1).sum()) == 8
    assert tuple(inspect.signature(compile_learned_mask).parameters) == (
        "gate_snapshot", "context_cue", "weight", "blocks",
    )
    assert compile_learned_mask.__closure__ is None
    identifiers = _identifiers(compile_learned_mask) | _identifiers(LocalContextGate.observe)
    assert not identifiers.intersection(FORBIDDEN_GATE_NAMES)


def test_gate_ties_and_invalid_states_fail_closed() -> None:
    blocks = architectural_blocks(20)
    weight = _fixture_weight()
    tied = GateSnapshot(torch.zeros(2, 4), 4, 1, 1.0, 4.0, 1e-6)
    with pytest.raises(ValueError, match="tie"):
        compile_learned_mask(tied, torch.tensor([1.0, 0.0, 0.0, 0.0]), weight, blocks)
    invalid = GateSnapshot(torch.full((2, 4), float("nan")), 4, 1, 1.0, 4.0, 1e-6)
    with pytest.raises(ValueError, match="invalid frozen gate"):
        compile_learned_mask(invalid, torch.tensor([1.0, 0.0, 0.0, 0.0]), weight, blocks)


def test_actual_runtime_learns_context_gate_before_endpoint() -> None:
    row = run_learned_context_gate_seed(97601)
    assert row["preflight"]["all_pass"]
    for gate in (
        "independent_theta_q_reference",
        "seed_sigma_schedule_metadata_invariance",
        "cue_swap_equivariance",
        "theta_counterfactual_dependence",
        "learned_selects_experienced_mapping",
        "shuffled_training_reverses_mapping",
        "gate_input_signature",
    ):
        assert row["preflight"]["gates"][gate]
    assert row["status"] == "LEARNED_CONTEXT_GATE_PASS"
    assert row["routes"]["LEARNED"]["accuracy"] >= 0.95
    assert row["routes"]["ORACLE"]["accuracy"] >= 0.95
    assert row["routes"]["CONTEXT_SHUFFLE_TRAIN"]["opposite_delivery"] >= 0.95
    assert row["routes"]["WRONG_CUE"]["opposite_delivery"] >= 0.95
    assert row["routes"]["GATE_LESION_STATIC_0"]["accuracy"] <= 0.55
    assert row["all_frozen_after_evaluation"]
