from reality_stone.clarus.experiments.runtime_source_only_symmetry_nogo import (
    run_source_only_symmetry_seed,
)


def test_source_only_uniform_substrate_is_symmetric_at_true_arrival() -> None:
    row = run_source_only_symmetry_seed(97901)
    assert row["status"] == "SOURCE_ONLY_SYMMETRY_NO_GO"
    assert not row["endpoint_opened"]
    for gate in (
        "zero_through_tick_L",
        "nonzero_at_tick_L_plus_1",
        "hidden_row_symmetry",
        "per_payload_four_equal_edges",
        "cue_field_has_sixteen_tied_edges",
        "top4_boundary_tie",
        "compiler_abstains",
        "threshold_permutation_first_arrival_invariant",
        "no_hidden_target_decoder_endpoint_reads",
    ):
        assert row["gates"][gate]
    assert row["representative_arrival"]["observed_ticks"] == [0, 1, 2, 3]
    assert row["representative_arrival"]["positive_candidate_count"] == 4
    assert row["factor_A_symmetry"]["rows"][0]["positive_support_count"] == 16
    assert row["factor_A_symmetry"]["rows"][0]["top4_boundary_gap"] == 0.0


def test_source_only_result_never_opens_a_decoder_endpoint() -> None:
    row = run_source_only_symmetry_seed(97902)
    assert row["status"] == "SOURCE_ONLY_SYMMETRY_NO_GO"
    assert row["gates"]["sources_immutable"]
    assert row["gates"]["exact_training_multiset"]
    assert not row["endpoint_opened"]
