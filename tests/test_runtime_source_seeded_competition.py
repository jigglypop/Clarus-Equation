import torch

from reality_stone.clarus.experiments.runtime_source_seeded_competition import (
    allocate_source_bindings,
    run_seeded_source_competition_seed,
    seeded_edge_code,
)


def test_uniform_competition_abstains_but_seeded_capacity_allocates() -> None:
    row = run_seeded_source_competition_seed(98001)
    assert row["status"] == "SEEDED_SOURCE_ALLOCATION_PASS"
    assert not row["endpoint_opened"]
    assert row["output_identity_status"] == "NONIDENTIFIED_ENDPOINT_CLOSED"
    assert row["gates"]["uniform_no_capacity_abstains"]
    assert row["gates"]["competition_only_uniform_abstains"]
    assert row["gates"]["seeded_capacity_bijection"]
    assert row["gates"]["source_independent_bias_collapses"]
    assert row["gates"]["hidden_row_permutation_covariant"]
    assert row["controls"]["uniform_capacity"]["status"] == "ABSTAIN_BOUNDARY_TIE"
    assert row["controls"]["seeded_capacity"]["is_bijection"]
    assert row["capacity_collision_fraction"] == 0.0
    assert row["representative_episodes"][0]["observed_ticks"] == [0, 1, 2, 3]


def test_seed_code_is_balanced_and_allocator_rejects_ties() -> None:
    code = seeded_edge_code(71998004)
    torch.testing.assert_close(code.sum(dim=0), torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(code.norm(dim=0), torch.ones(4, dtype=torch.float64))
    tied = torch.full((4, 4), 0.25, dtype=torch.float64)
    result = allocate_source_bindings(
        tied, (0, 1, 2, 3), 1.1, 1e-6, use_capacity=True,
    )
    assert result["status"] == "ABSTAIN_BOUNDARY_TIE"
    assert not result["is_bijection"]
