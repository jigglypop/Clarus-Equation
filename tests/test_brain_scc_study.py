import pytest

from reality_stone.clarus.brain_scc_study import (
    CrossScaleNodeMap,
    DirectedScaleGraph,
    audit_scale_indexed_scc_study,
)


def _fine() -> DirectedScaleGraph:
    return DirectedScaleGraph(
        scale_id="cells",
        scale_rank=0,
        node_semantics="identified cells",
        edge_semantics="directed synapses above a frozen threshold",
        direction_source="anatomical polarity",
        nodes=("a0", "a1", "b0", "b1"),
        edges=(("a0", "a1"), ("a1", "a0"), ("b0", "b1"), ("b1", "b0")),
    )


def _coarse() -> DirectedScaleGraph:
    return DirectedScaleGraph(
        scale_id="regions",
        scale_rank=1,
        node_semantics="registered regions",
        edge_semantics="directed aggregate synapses under the same snapshot",
        direction_source="aggregated anatomical polarity",
        nodes=("A", "B"),
        edges=(("A", "A"), ("B", "B")),
    )


def test_typed_cross_scale_map_can_be_compatible_without_nested_maximal_claim() -> None:
    mapping = CrossScaleNodeMap(
        fine_scale_id="cells",
        coarse_scale_id="regions",
        mapping=(("a0", "A"), ("a1", "A"), ("b0", "B"), ("b1", "B")),
        mapping_semantics="atlas membership",
    )
    audit = audit_scale_indexed_scc_study((_fine(), _coarse()), (mapping,))
    assert audit.scale_compatible
    assert audit.cross_scale_component_violations == 0
    assert not audit.fixed_graph_nested_maximal_claim_allowed
    assert not audit.biological_identity_established


def test_split_image_of_one_fine_scc_is_a_cross_scale_violation() -> None:
    mapping = CrossScaleNodeMap(
        fine_scale_id="cells",
        coarse_scale_id="regions",
        mapping=(("a0", "A"), ("a1", "B"), ("b0", "B"), ("b1", "B")),
        mapping_semantics="atlas membership",
    )
    audit = audit_scale_indexed_scc_study((_fine(), _coarse()), (mapping,))
    assert not audit.scale_compatible
    assert audit.cross_scale_component_violations == 1


def test_incomplete_reverse_or_untyped_study_fails_closed() -> None:
    incomplete = CrossScaleNodeMap(
        fine_scale_id="cells",
        coarse_scale_id="regions",
        mapping=(("a0", "A"),),
        mapping_semantics="atlas membership",
    )
    assert not audit_scale_indexed_scc_study((_fine(), _coarse()), (incomplete,)).complete_node_maps
    reverse = CrossScaleNodeMap(
        fine_scale_id="regions",
        coarse_scale_id="cells",
        mapping=(("A", "a0"), ("B", "b0")),
        mapping_semantics="invalid reverse",
    )
    with pytest.raises(ValueError, match="finer"):
        audit_scale_indexed_scc_study((_fine(), _coarse()), (reverse,))
