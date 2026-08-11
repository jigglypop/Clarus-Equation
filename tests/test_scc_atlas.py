from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from reality_stone.clarus.scc_atlas import (
    certify_dag_block_gain,
    construct_arch1,
    decoder_f_contraction_error_bound,
    decompose_scc,
    encoder_phi_contraction_error_bound,
    forward_time_unroll,
    project_time_coordinate,
    threshold_scc_filtration,
    validate_arch1,
)


def _loopless_edge_universe(size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (source, target) for source in range(size) for target in range(size) if source != target
    )


def _edges_from_mask(
    universe: tuple[tuple[int, int], ...],
    mask: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(edge for bit, edge in enumerate(universe) if mask & (1 << bit))


def _independent_reachability_partition(
    size: int,
    edges: tuple[tuple[int, int], ...],
) -> set[frozenset[int]]:
    reach = [[source == target for target in range(size)] for source in range(size)]
    for source, target in edges:
        reach[source][target] = True
    for middle in range(size):
        for source in range(size):
            if reach[source][middle]:
                for target in range(size):
                    reach[source][target] = reach[source][target] or reach[middle][target]
    remaining = set(range(size))
    partition: set[frozenset[int]] = set()
    while remaining:
        source = min(remaining)
        component = frozenset(
            target for target in range(size) if reach[source][target] and reach[target][source]
        )
        partition.add(component)
        remaining.difference_update(component)
    return partition


def _assert_topological_order(
    order: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
) -> None:
    position = {node: index for index, node in enumerate(order)}
    assert len(position) == len(order)
    assert all(position[source] < position[target] for source, target in edges)


def test_scc_matches_independent_reachability_for_all_loopless_graphs_n_le_4() -> None:
    graph_count = 0
    for size in range(1, 5):
        nodes = tuple(range(size))
        universe = _loopless_edge_universe(size)
        for mask in range(1 << len(universe)):
            edges = _edges_from_mask(universe, mask)
            result = decompose_scc(nodes, edges)
            expected = _independent_reachability_partition(size, edges)
            assert {frozenset(component) for component in result.components} == expected
            assert set(result.component_of) == set(nodes)
            assert all(
                result.component_of[node] == component_id
                for component_id, component in enumerate(result.components)
                for node in component
            )
            _assert_topological_order(result.topological_order, result.condensation_edges)
            graph_count += 1
    assert graph_count == 4_165


def test_self_loops_do_not_change_partition_but_mark_singletons_recurrent() -> None:
    without_loops = decompose_scc((0, 1), ((0, 1),))
    with_loops = decompose_scc((0, 1), ((0, 0), (0, 1), (1, 1)))
    assert with_loops.components == without_loops.components
    assert without_loops.component_is_recurrent == (False, False)
    assert with_loops.component_is_recurrent == (True, True)


def test_every_single_edge_addition_only_coarsens_for_all_graphs_n_le_3() -> None:
    checked = 0
    for size in range(1, 4):
        nodes = tuple(range(size))
        universe = _loopless_edge_universe(size)
        for mask in range(1 << len(universe)):
            edges = _edges_from_mask(universe, mask)
            before = decompose_scc(nodes, edges)
            for bit, edge in enumerate(universe):
                if mask & (1 << bit):
                    continue
                after = decompose_scc(nodes, (*edges, edge))
                for component in before.components:
                    parent = after.component_of[component[0]]
                    assert all(after.component_of[node] == parent for node in component)
                checked += 1
    assert checked == 196


def test_reciprocal_paths_merge_declared_blocks_into_one_maximal_scc() -> None:
    result = decompose_scc(
        ("a0", "a1", "b0", "b1"),
        (
            ("a0", "a1"),
            ("a1", "a0"),
            ("b0", "b1"),
            ("b1", "b0"),
            ("a0", "b0"),
            ("b1", "a1"),
        ),
    )
    assert result.components == (("a0", "a1", "b0", "b1"),)
    assert result.component_is_recurrent == (True,)


def test_topology_does_not_certify_dynamics_and_divergent_self_map_is_rejected() -> None:
    topology = decompose_scc((0,), ((0, 0),))
    state = 1.0
    for _ in range(10):
        state = 2.0 * state
    assert topology.components == ((0,),)
    assert topology.component_is_recurrent == (True,)
    assert state == 1_024.0
    with pytest.raises(ValueError, match="self-gain"):
        certify_dag_block_gain(
            ((2.0,),),
            normalization_scales=(1.0,),
            schedule="simultaneous",
        )


def test_threshold_filtration_records_merge_only_parent_links() -> None:
    filtration = threshold_scc_filtration(
        ("a", "b", "c"),
        (
            ("a", "b", 0.90),
            ("b", "a", 0.80),
            ("c", "a", 0.70),
            ("b", "c", 0.60),
        ),
        (0.85, 0.75, 0.65, 0.55),
        edge_semantics="dimensionless effective gain",
        layer="fast",
        score_name="absolute normalized gain",
    )
    assert [len(level.decomposition.components) for level in filtration.levels] == [3, 2, 2, 1]
    assert filtration.levels[0].parent_of_previous is None
    assert filtration.levels[1].parent_of_previous == (0, 0, 1)
    assert filtration.levels[2].parent_of_previous == (0, 1)
    assert filtration.levels[3].parent_of_previous == (0, 0)
    filtration.assert_compatible(
        nodes=("a", "b", "c"),
        edge_semantics="dimensionless effective gain",
        layer="fast",
        score_name="absolute normalized gain",
    )
    with pytest.raises(ValueError, match="node set"):
        filtration.assert_compatible(
            nodes=("a", "b", "c", "d"),
            edge_semantics="dimensionless effective gain",
            layer="fast",
            score_name="absolute normalized gain",
        )
    with pytest.raises(ValueError, match="semantics"):
        filtration.assert_compatible(
            nodes=("a", "b", "c"),
            edge_semantics="anatomical synapse",
            layer="fast",
            score_name="absolute normalized gain",
        )


def test_threshold_filtration_rejects_ambiguous_or_invalid_inputs() -> None:
    common = {
        "edge_semantics": "effective",
        "layer": "union",
        "score_name": "confidence",
    }
    with pytest.raises(ValueError, match="strictly decreasing"):
        threshold_scc_filtration((0, 1), ((0, 1, 0.5),), (0.4, 0.5), **common)
    with pytest.raises(ValueError, match="unique"):
        threshold_scc_filtration(
            (0, 1),
            ((0, 1, 0.5), (0, 1, 0.4)),
            (0.5,),
            **common,
        )
    with pytest.raises(ValueError, match="tie_rule"):
        threshold_scc_filtration(
            (0, 1),
            ((0, 1, 0.5),),
            (0.5,),
            tie_rule="approximately",
            **common,
        )


def test_positive_delay_forward_unroll_is_a_singleton_scc_dag() -> None:
    unroll = forward_time_unroll(
        ("a", "b"),
        (("a", "b", 2), ("b", "a", 1), ("a", "a", 3)),
        horizon=5,
    )
    assert len(unroll.event_nodes) == 12
    assert len(unroll.decomposition.components) == len(unroll.event_nodes)
    assert not any(unroll.decomposition.component_is_recurrent)
    assert all(source[1] < target[1] for source, target in unroll.event_edges)
    _assert_topological_order(
        unroll.decomposition.topological_order,
        unroll.decomposition.condensation_edges,
    )
    assert unroll.projected_template_edges == (("a", "a"), ("a", "b"), ("b", "a"))
    assert project_time_coordinate(("a", 5)) == "a"
    with pytest.raises(ValueError, match="positive integer"):
        forward_time_unroll(("a",), (("a", "a", 0),), horizon=2)


def test_arch1_constructor_realizes_exact_dag_condensation() -> None:
    construction = construct_arch1(
        (("a0", "a1"), ("b0",), ("c0", "c1", "c2")),
        ((0, 1), (0, 2), (1, 2)),
    )
    validation = construction.validation
    assert validation.valid
    assert validation.errors == ()
    assert validation.target_topological_order == (0, 1, 2)
    assert {frozenset(component) for component in validation.decomposition.components} == {
        frozenset(module) for module in construction.modules
    }
    mapped_condensation = {
        (validation.module_component_ids[source], validation.module_component_ids[target])
        for source, target in construction.target_edges
    }
    assert mapped_condensation == set(validation.decomposition.condensation_edges)


def test_arch1_rejects_reciprocal_target_and_validator_detects_extra_cross_edge() -> None:
    with pytest.raises(ValueError, match="must be a DAG"):
        construct_arch1((("a",), ("b",)), ((0, 1), (1, 0)))

    construction = construct_arch1((("a0", "a1"), ("b0", "b1")), ((0, 1),))
    invalid = validate_arch1(
        construction.modules,
        (*construction.edges, ("b0", "a0")),
        construction.target_edges,
    )
    assert not invalid.valid
    assert "cross-module edge relation does not exactly match the target graph" in invalid.errors
    assert "declared modules are not exactly the maximal SCCs" in invalid.errors


def test_finite_dag_block_gain_certificate_uses_target_source_orientation() -> None:
    matrix = np.asarray(
        (
            (0.20, 0.00, 0.00),
            (2.50, 0.30, 0.00),
            (0.10, 4.00, 0.40),
        ),
        dtype=np.float64,
    )
    certificate = certify_dag_block_gain(
        matrix,
        normalization_scales=(2.0, 3.0, 5.0),
        schedule="simultaneous",
    )
    assert certificate.certified
    assert certificate.gain_orientation == "M[target, source]"
    assert certificate.topological_order == (0, 1, 2)
    assert certificate.spectral_radius == pytest.approx(0.40)
    assert 0.0 <= certificate.contraction_factor < 1.0
    weights = np.asarray(certificate.neumann_weights)
    assert np.allclose((np.eye(3) - matrix) @ weights, np.ones(3), atol=1e-12)
    assert certificate.solve_relative_residual <= 1e-12
    residual_bound = certificate.residual_bound((0.01, 0.02, 0.03))
    assert len(residual_bound) == 3
    assert all(value >= 0.0 for value in residual_bound)


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (((0.2, 0.1), (0.1, 0.2)), "finite DAG"),
        (((0.2, -0.1), (0.0, 0.2)), "nonnegative"),
        (((math.nan, 0.0), (0.0, 0.2)), "finite"),
    ),
)
def test_block_gain_rejects_cyclic_or_invalid_matrices(
    matrix: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        certify_dag_block_gain(
            matrix,
            normalization_scales=(1.0, 1.0),
            schedule="simultaneous",
        )


def test_block_gain_rejects_schedule_mismatch_and_unusable_conditioning() -> None:
    with pytest.raises(ValueError, match="simultaneous schedule"):
        certify_dag_block_gain(
            ((0.2, 0.0), (0.3, 0.2)),
            normalization_scales=(1.0, 1.0),
            schedule="gauss-seidel",
        )
    with pytest.raises(ValueError, match="normalization_scales"):
        certify_dag_block_gain(
            ((0.2, 0.0), (0.3, 0.2)),
            normalization_scales=(1.0, 0.0),
            schedule="simultaneous",
        )
    with pytest.raises(FloatingPointError, match="condition limit"):
        certify_dag_block_gain(
            ((0.90, 0.00), (1e10, 0.90)),
            normalization_scales=(1.0, 1.0),
            schedule="simultaneous",
            condition_limit=1e6,
        )


def test_encoder_phi_and_decoder_f_error_bounds_remain_separately_typed() -> None:
    decoder = decoder_f_contraction_error_bound(
        initial_decoder_error=0.4,
        decoder_defect=0.02,
        f_contraction=0.5,
        steps=4,
    )
    encoder = encoder_phi_contraction_error_bound(
        initial_encoder_error=0.4,
        encoder_defect=0.02,
        phi_contraction=0.5,
        steps=4,
    )
    expected = 0.5**4 * 0.4 + (1.0 - 0.5**4) * 0.02 / (1.0 - 0.5)
    assert decoder.finite_horizon_bound == pytest.approx(expected)
    assert encoder.finite_horizon_bound == pytest.approx(expected)
    assert decoder.premise == "decoder defect with F contraction"
    assert encoder.premise == "encoder defect with Phi contraction"
    assert decoder.premise != encoder.premise


def test_graph_and_error_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unique"):
        decompose_scc(("a", "a"), ())
    with pytest.raises(ValueError, match="outside"):
        decompose_scc(("a",), (("a", "b"),))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        decoder_f_contraction_error_bound(
            initial_decoder_error=0.0,
            decoder_defect=0.1,
            f_contraction=1.0,
            steps=1,
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        encoder_phi_contraction_error_bound(
            initial_encoder_error=0.0,
            encoder_defect=0.1,
            phi_contraction=0.5,
            steps=-1,
        )


def test_component_order_is_independent_of_edge_iteration_order() -> None:
    nodes = ("x", "y", "z", "w")
    edges = (("x", "y"), ("y", "x"), ("y", "z"), ("z", "w"))
    expected = decompose_scc(nodes, edges)
    for permutation in itertools.permutations(edges):
        result = decompose_scc(nodes, permutation)
        assert result.components == expected.components
        assert result.condensation_edges == expected.condensation_edges
        assert result.topological_order == expected.topological_order
