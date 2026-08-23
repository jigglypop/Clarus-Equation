import ast
import inspect
import textwrap

import pytest
import torch

from reality_stone.clarus.runtime_context_branch_routing import architectural_blocks
from reality_stone.clarus.runtime_factor_compositional_routing import (
    FORBIDDEN_FACTOR_GATE_NAMES,
    CountNormalizedFactorGate,
    CountNormalizedGateSnapshot,
    compile_factor_mask,
    run_factor_composition_seed,
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


def test_count_normalization_removes_unequal_factor_frequency() -> None:
    gate = CountNormalizedFactorGate()
    q0 = torch.tensor([1.0, 0.0])
    q1 = torch.tensor([0.0, 1.0])
    for _ in range(8):
        gate.observe(q0, torch.tensor([0.2, 0.8], dtype=torch.float64))
    for _ in range(4):
        gate.observe(q1, torch.tensor([0.7, 0.3], dtype=torch.float64))
    frozen = gate.snapshot()
    assert torch.equal(frozen.counts, torch.tensor([8.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(
        frozen.theta,
        torch.tensor([[0.2, 0.7], [0.8, 0.3]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.equal(frozen.theta, frozen.accumulator / frozen.counts.view(1, 2))


def test_factor_compiler_is_local_and_fails_closed_on_unseen_or_tied_state() -> None:
    weight = _fixture_weight()
    blocks = architectural_blocks(20)
    gate = CountNormalizedFactorGate()
    gate.observe(torch.tensor([1.0, 0.0]), torch.tensor([0.1, 0.9]))
    gate.observe(torch.tensor([0.0, 1.0]), torch.tensor([0.8, 0.2]))
    frozen = gate.snapshot()
    mask0, info0 = compile_factor_mask(frozen, torch.tensor([1.0, 0.0]), weight, blocks)
    mask1, info1 = compile_factor_mask(frozen, torch.tensor([0.0, 1.0]), weight, blocks)
    assert (info0["selected_branch"], info1["selected_branch"]) == (1, 0)
    assert int(mask0.sum()) == int(mask1.sum()) == 12
    assert int((mask0 != mask1).sum()) == 8
    assert tuple(inspect.signature(compile_factor_mask).parameters) == (
        "gate_snapshot", "factor_cue", "weight", "blocks",
    )
    assert compile_factor_mask.__closure__ is None
    identifiers = _identifiers(compile_factor_mask) | _identifiers(CountNormalizedFactorGate.observe)
    assert not identifiers.intersection(FORBIDDEN_FACTOR_GATE_NAMES)

    tied = CountNormalizedGateSnapshot(
        theta=torch.zeros(2, 2),
        accumulator=torch.zeros(2, 2),
        counts=torch.ones(2),
        update_count=2,
        min_logit_margin=1e-6,
    )
    with pytest.raises(ValueError, match="tie"):
        compile_factor_mask(tied, torch.tensor([1.0, 0.0]), weight, blocks)
    unseen = CountNormalizedGateSnapshot(
        theta=torch.zeros(2, 2),
        accumulator=torch.zeros(2, 2),
        counts=torch.tensor([1.0, 0.0]),
        update_count=1,
        min_logit_margin=1e-6,
    )
    with pytest.raises(ValueError, match="unobserved"):
        compile_factor_mask(unseen, torch.tensor([0.0, 1.0]), weight, blocks)


def test_actual_runtime_composes_the_heldout_11_pair() -> None:
    row = run_factor_composition_seed(97701)
    assert row["preflight"]["all_pass"]
    for gate in (
        "exact_training_multiset",
        "heldout_absent",
        "local_branch_use_separation",
        "normal_counts",
        "shuffled_counts",
        "factor_gate_input_signature",
        "pair_mask_budget",
        "common_output_trunk",
        "pair_mask_hamming",
        "direct_sum_cross_support_zero",
        "joint_lookup_holdout_abstains",
    ):
        assert row["preflight"]["gates"][gate]
    assert row["status"] == "FACTOR_COMPOSITION_PASS"
    assert row["heldout_context"] == (1, 1)
    assert row["routes"]["FACTORWISE_LEARNED"]["joint_accuracy"] >= 0.95
    assert row["routes"]["ORACLE"]["joint_accuracy"] >= 0.95
    assert row["routes"]["A_FACTOR_SHUFFLE_TRAIN"]["A_opposite_delivery"] >= 0.95
    assert row["routes"]["A_FACTOR_SHUFFLE_TRAIN"]["B_accuracy"] >= 0.95
    assert row["routes"]["B_FACTOR_SHUFFLE_TRAIN"]["B_opposite_delivery"] >= 0.95
    assert row["routes"]["B_FACTOR_SHUFFLE_TRAIN"]["A_accuracy"] >= 0.95
    assert row["routes"]["FACTORWISE_LEARNED"]["cartesian_trial_count"] == 144
    assert row["all_frozen_after_evaluation"]
