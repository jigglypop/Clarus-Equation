import ast
import inspect

import torch

from reality_stone.clarus.experiments.runtime_context_branch_routing import (
    ContextBranchConfig,
    ExactDelayEligibility,
    architectural_blocks,
    construct_context_branch_mask,
    run_context_branch_seed,
)


def _fixture_weight() -> torch.Tensor:
    weight = torch.zeros(20, 20)
    blocks = architectural_blocks(20)
    s0, s1, h0, h1, output = blocks
    for index in range(4):
        weight[h0[index], s0[index]] = 1.0
        weight[h1[index], s1[index]] = 1.0
        weight[output[index], h0[index]] = 0.7
        weight[output[index], h1[index]] = 0.7
    return weight


def test_mask_compiler_is_answer_blind_and_changes_only_entry_branch() -> None:
    function = construct_context_branch_mask
    assert tuple(inspect.signature(function).parameters) == ("weight", "context", "blocks", "seed", "route")
    identifiers = {node.id for node in ast.walk(ast.parse(inspect.getsource(function))) if isinstance(node, ast.Name)}
    assert not identifiers.intersection({"payload", "answer", "target", "decoder", "endpoint", "rollout"})

    weight = _fixture_weight()
    blocks = architectural_blocks(20)
    mask0 = function(weight, 0, blocks, 7, "CORRECT").bool()
    mask1 = function(weight, 1, blocks, 7, "CORRECT").bool()
    assert int(mask0.sum()) == int(mask1.sum()) == 12
    assert int((mask0 != mask1).sum()) == 8
    assert torch.equal(mask0[16:, :], mask1[16:, :])
    assert int(function(weight, 0, blocks, 7, "RANDOM_MATCHED").sum()) == 12
    assert int(function(weight, 0, blocks, 7, "STATIC_UNION").sum()) == 16


def test_exact_delay_eligibility_uses_delayed_pre_state() -> None:
    tracker = ExactDelayEligibility(dim=4, delay=2, decay=0.99, ltd=0.2)
    source = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden = torch.tensor([0.0, 1.0, 0.0, 0.0])
    tracker.observe(source)
    tracker.observe(torch.zeros(4))
    tracker.observe(hidden)
    assert tracker.eligibility[1, 0] > 0.0
    assert tracker.eligibility[0, 1] < 0.0
    assert tracker.paired_observations == 1


def test_actual_delayed_runtime_context_branch_smoke() -> None:
    row = run_context_branch_seed(97501, config=ContextBranchConfig(seed=97501))
    assert row["preflight"]["all_pass"]
    for gate in (
        "no_context_state_path",
        "no_context_decoder_path",
        "correct_wrong_mask_parity",
        "delay_histogram_parity",
        "threshold_profile_parity",
        "stp_profile_parity",
        "decoder_hash_parity",
    ):
        assert row["preflight"]["gates"][gate]
    assert row["learning"]["outside_allowed_actual_delta_norm"] == 0.0
    assert row["learning"]["cutoff"]["hippocampal_rows_after"] == 0
    assert row["learning"]["cutoff"]["delay_ring_zero"]
    assert row["source_snapshot_immutable"]
    assert row["routes"]["CORRECT"]["accuracy"] >= 0.95
    assert row["routes"]["WRONG"]["opposite_delivery"] >= 0.95
    assert row["routes"]["STATIC_0"]["accuracy"] <= 0.55
    assert row["routes"]["STATIC_1"]["accuracy"] <= 0.55
    assert row["routes"]["FULL"]["accuracy"] <= 0.55
    assert row["swap_parity"]
