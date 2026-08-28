from examples.physics.zerod_causal_growth_no_go import (
    zerod_causal_growth_no_go,
)


def test_singleton_allows_two_normalized_covariant_growth_kernels() -> None:
    audit = zerod_causal_growth_no_go()

    assert audit.seed == (0,)
    assert audit.maximum_kernel_relations == frozenset(
        ((0, 1), (1, 2), (0, 2))
    )
    assert audit.minimal_past_kernel_relations == frozenset(((0, 1), (0, 2)))
    assert audit.both_deterministic_normalized
    assert audit.both_relabel_covariant
    assert audit.both_preserve_old_order
    assert audit.all_births_have_nonempty_past


def test_connected_outputs_are_nonisomorphic() -> None:
    audit = zerod_causal_growth_no_go()

    assert audit.maximum_relation_count == 3
    assert audit.minimal_past_relation_count == 2
    assert audit.maximum_height == 3
    assert audit.minimal_past_height == 2
    assert audit.maximum_width == 1
    assert audit.minimal_past_width == 2
    assert audit.maximum_order_dimension == 1
    assert audit.minimal_past_order_dimension == 2
    assert not audit.outputs_isomorphic
    assert not audit.unique_causal_growth_follows


def test_continuum_dimension_claim_is_untyped_without_readout() -> None:
    audit = zerod_causal_growth_no_go()

    assert not audit.continuum_dimension_is_defined_by_input
    assert audit.status == "SINGLETON_TO_UNIQUE_CAUSAL_GROWTH_DISPROVED"
    assert audit.claim_ceiling == "COMPLETE_FINITE_MODEL_THEORETIC_COUNTEREXAMPLE"
