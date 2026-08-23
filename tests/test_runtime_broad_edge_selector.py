import ast
import inspect
import textwrap

import pytest
import torch

from reality_stone.clarus.runtime_broad_edge_selector import (
    FORBIDDEN_EDGE_SELECTOR_NAMES,
    CountNormalizedEdgeField,
    EdgeFieldSnapshot,
    compile_edge_field_mask,
    run_broad_edge_selector_seed,
)


def _supports() -> tuple[torch.Tensor, torch.Tensor]:
    candidate = torch.zeros(20, 20, dtype=torch.bool)
    candidate[8:12, 0:8] = True
    trunk = torch.zeros(20, 20, dtype=torch.bool)
    for index in range(4):
        trunk[16 + index, 8 + index] = True
    return candidate, trunk


def _identifiers(function) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_edge_field_count_normalizes_unequal_cue_exposure() -> None:
    field = CountNormalizedEdgeField(32, 4, 1e-6)
    q0 = torch.tensor([1.0, 0.0])
    q1 = torch.tensor([0.0, 1.0])
    use0 = torch.zeros(32, dtype=torch.float64)
    use1 = torch.zeros(32, dtype=torch.float64)
    use0[:4] = 0.25
    use1[4:8] = 0.25
    for _ in range(8):
        field.observe(q0, use0)
    for _ in range(4):
        field.observe(q1, use1)
    frozen = field.snapshot()
    assert torch.equal(frozen.counts, torch.tensor([8.0, 4.0], dtype=torch.float64))
    assert torch.equal(frozen.theta, frozen.accumulator / frozen.counts.view(1, 2))
    torch.testing.assert_close(frozen.theta[:, 0], use0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(frozen.theta[:, 1], use1, rtol=0.0, atol=0.0)


def test_edge_compiler_is_weight_blind_and_tie_fails_closed() -> None:
    candidate, trunk = _supports()
    field = CountNormalizedEdgeField(32, 4, 1e-6)
    q0 = torch.tensor([1.0, 0.0])
    q1 = torch.tensor([0.0, 1.0])
    use0 = torch.zeros(32, dtype=torch.float64)
    use1 = torch.zeros(32, dtype=torch.float64)
    use0[:4] = 0.25
    use1[4:8] = 0.25
    field.observe(q0, use0)
    field.observe(q1, use1)
    frozen = field.snapshot()
    mask0, info0 = compile_edge_field_mask(frozen, q0, candidate, trunk)
    mask1, info1 = compile_edge_field_mask(frozen, q1, candidate, trunk)
    assert info0["entry_edges"] == info1["entry_edges"] == 4
    assert int(mask0.sum()) == int(mask1.sum()) == 8
    assert int((mask0 != mask1).sum()) == 8
    assert tuple(inspect.signature(compile_edge_field_mask).parameters) == (
        "gate_snapshot", "factor_cue", "candidate_support", "trunk_support",
    )
    assert compile_edge_field_mask.__closure__ is None
    identifiers = _identifiers(compile_edge_field_mask) | _identifiers(CountNormalizedEdgeField.observe)
    assert not identifiers.intersection(FORBIDDEN_EDGE_SELECTOR_NAMES)

    tied = EdgeFieldSnapshot(
        theta=torch.ones(32, 2),
        accumulator=torch.ones(32, 2),
        counts=torch.ones(2),
        update_count=2,
        selected_edges=4,
        min_boundary_gap=1e-6,
    )
    with pytest.raises(ValueError, match="boundary tie"):
        compile_edge_field_mask(tied, q0, candidate, trunk)


def test_actual_runtime_selects_edges_for_heldout_11() -> None:
    row = run_broad_edge_selector_seed(97801)
    assert row["preflight"]["all_pass"]
    for gate in (
        "exact_training_multiset",
        "heldout_absent",
        "episode_local_peak",
        "field_input_signature",
        "pair_mask_budget",
        "pair_mask_hamming",
        "joint_lookup_holdout_abstains",
    ):
        assert row["preflight"]["gates"][gate]
    for factor in ("factor_A", "factor_B"):
        assert row["preflight"][factor]["gates"]["source_uniform_nonzero"]
        assert row["preflight"][factor]["gates"]["weight_only_abstains"]
        assert row["preflight"][factor]["gates"]["pooled_static_abstains"]
    assert row["status"] == "BROAD_EDGE_SELECTOR_PASS"
    assert row["heldout_context"] == (1, 1)
    assert row["routes"]["EDGE_FIELD_LEARNED"]["joint_accuracy"] >= 0.95
    assert row["routes"]["ORACLE"]["joint_accuracy"] >= 0.95
    assert row["routes"]["A_FACTOR_SHUFFLE_TRAIN"]["A_opposite_delivery"] >= 0.95
    assert row["routes"]["A_FACTOR_SHUFFLE_TRAIN"]["B_accuracy"] >= 0.95
    assert row["routes"]["B_FACTOR_SHUFFLE_TRAIN"]["B_opposite_delivery"] >= 0.95
    assert row["routes"]["B_FACTOR_SHUFFLE_TRAIN"]["A_accuracy"] >= 0.95
    assert row["routes"]["EDGE_FIELD_LEARNED"]["cartesian_trial_count"] == 144
    assert row["all_frozen_after_evaluation"]
