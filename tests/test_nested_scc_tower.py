import math

import numpy as np
import pytest

from reality_stone.clarus.nested_scc_tower import (
    NestedTowerGenerator,
    TowerSpec,
    strongly_connected_components,
)


class EqualInt(int):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


class EqualStr(str):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


class LyingTuple(tuple):
    def __len__(self):
        return 3


class ShortLyingTuple(tuple):
    def __len__(self):
        return 1


class ShortLyingList(list):
    def __len__(self):
        return 1


class EvilNode(tuple):
    def __len__(self):
        return 2

    def __getitem__(self, _index):
        return 0

    def __iter__(self):
        return iter((0, 0))


def _reachable(vertices, edges, source):
    adjacency = {node: set() for node in vertices}
    for left, right in edges:
        adjacency[left].add(right)
    found = {source}
    pending = [source]
    while pending:
        node = pending.pop()
        for target in adjacency[node]:
            if target not in found:
                found.add(target)
                pending.append(target)
    return found


def _manual_predecessors(node, width):
    level, index = node
    result = {node, (level + 1, index)}
    if level > 0:
        result.add((level - 1, index))
    if index > 0:
        result.add((level, index - 1))
    if index + 1 < width:
        result.add((level, index + 1))
    return result


def test_generated_prefixes_are_properly_nested_and_independently_strong() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=3, maximum_depth=6))
    previous_vertices = set()
    previous_edges = set()
    for depth in range(7):
        prefix = generator.prefix(depth)
        audit = generator.audit_prefix(depth)
        vertices = set(prefix.vertices)
        edges = set(prefix.edges)
        assert audit.is_strongly_connected
        assert audit.component_count == 1
        assert audit.nested_with_previous
        if depth:
            assert previous_vertices < vertices
            assert previous_edges <= edges
        for node in vertices:
            assert _reachable(vertices, edges, node) == vertices
        previous_vertices = vertices
        previous_edges = edges


def test_finite_level_is_not_a_second_maximal_scc_of_the_larger_fixed_graph() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=3))
    lower = generator.prefix(1)
    upper = generator.prefix(3)
    components = strongly_connected_components(upper.vertices, upper.edges)
    assert len(components) == 1
    assert set(lower.vertices) < set(components[0])
    assert tuple(lower.vertices) not in components


def test_manifest_and_queries_are_deterministic_without_cache_state() -> None:
    spec = TowerSpec(shell_width=4, maximum_depth=5, observation_scales=(1, 2, 3, 4))
    left = NestedTowerGenerator(spec)
    right = NestedTowerGenerator(spec)
    assert left.manifest == right.manifest
    assert left.prefix(3) == right.prefix(3)
    assert left.predecessors((20, 2)) == right.predecessors((20, 2))
    assert left.recurrence_operator() is not left.recurrence_operator()
    assert np.array_equal(left.upward_operator(1), right.upward_operator(1))


def test_default_generator_specs_are_independent_and_poisoning_is_local() -> None:
    left = NestedTowerGenerator()
    right = NestedTowerGenerator()
    assert left.spec is not right.spec
    poisoned = left.spec
    object.__setattr__(poisoned, "shell_width", 99)
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        left.vertices(0)
    assert right.spec.shell_width == 3
    assert len(right.vertices(0)) == 3


def test_generator_spec_and_operator_storage_are_read_only_at_public_boundaries() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=3, maximum_depth=2))
    with pytest.raises(AttributeError):
        generator.spec = TowerSpec(shell_width=3, maximum_depth=2)
    with pytest.raises(AttributeError):
        object.__setattr__(generator, "assert_integrity", lambda: None)
    with pytest.raises(ValueError, match="read-only"):
        generator._within_base[0, 0] = 100.0
    with pytest.raises(ValueError, match="read-only"):
        generator._identity[0, 0] = 0.0

    public_operator = generator.recurrence_operator()
    public_operator[0, 0] = 100.0
    assert generator.recurrence_operator()[0, 0] != 100.0
    generator.assert_integrity()


@pytest.mark.parametrize(
    "query",
    [
        lambda generator: generator.vertices(0),
        lambda generator: generator.edges(0),
        lambda generator: generator.prefix(0),
        lambda generator: generator.audit_prefix(0),
        lambda generator: generator.predecessors((0, 0)),
        lambda generator: generator.backward_causal_cone(((0, 0),), 1),
        lambda generator: generator.forward_unroll(0, 1),
        lambda generator: generator.normalize_observation((1.0, 2.0, 3.0)),
    ],
)
def test_every_graph_and_normalization_query_rejects_same_width_scale_swap(query) -> None:
    generator = NestedTowerGenerator(
        TowerSpec(shell_width=3, maximum_depth=2, observation_scales=(1.0, 2.0, 3.0))
    )
    object.__setattr__(
        generator,
        "_spec",
        TowerSpec(shell_width=3, maximum_depth=2, observation_scales=(3.0, 2.0, 1.0)),
    )
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        query(generator)


@pytest.mark.parametrize(
    "original,replacement",
    [
        (
            TowerSpec(shell_width=2, maximum_depth=3, observation_scales=(1.0, 1.0)),
            TowerSpec(shell_width=4, maximum_depth=1, observation_scales=(1.0,) * 4),
        ),
        (
            TowerSpec(shell_width=4, maximum_depth=1, observation_scales=(1.0,) * 4),
            TowerSpec(shell_width=2, maximum_depth=3, observation_scales=(1.0, 1.0)),
        ),
    ],
)
def test_equal_capacity_width_depth_swaps_cannot_change_a_sealed_graph(
    original, replacement
) -> None:
    generator = NestedTowerGenerator(original)
    object.__setattr__(generator, "_spec", replacement)
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        generator.prefix(replacement.maximum_depth)


def test_private_operator_mutation_is_detected_even_after_write_flag_is_resealed() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=1))
    generator._within_base.setflags(write=True)
    generator._within_base[0, 0] = 100.0
    generator._within_base.setflags(write=False)
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        generator.certify_prefix(1)


def test_frozen_manifest_metadata_is_covered_by_the_live_integrity_seal() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=1))
    manifest = generator.manifest
    object.__setattr__(manifest, "maximum_in_degree", 999)
    with pytest.raises(ValueError, match="integrity seal mismatch"):
        generator.manifest


def test_forward_event_unroll_is_a_dag_with_only_singleton_components() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=2))
    template = generator.audit_prefix(2)
    unroll = generator.forward_unroll(2, horizon=7)
    assert template.is_strongly_connected
    assert unroll.acyclic
    assert unroll.singleton_component_count == unroll.vertex_count
    assert unroll.edge_count > 0


def test_complete_predecessor_rule_yields_a_finite_causal_cone() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=3, maximum_depth=2))
    certificate = generator.backward_causal_cone(((0, 1),), horizon=4)
    expected_distances = {(0, 1): 0}
    frontier = {(0, 1)}
    for distance in range(4):
        next_frontier = set()
        for node in frontier:
            predecessors = _manual_predecessors(node, width=3)
            assert set(generator.predecessors(node)) == predecessors
            for predecessor in predecessors:
                if predecessor not in expected_distances:
                    expected_distances[predecessor] = distance + 1
                    next_frontier.add(predecessor)
        frontier = next_frontier
    assert certificate.predecessor_complete
    assert set(certificate.nodes) == set(expected_distances)
    assert (4, 1) in certificate.nodes
    assert certificate.maximum_birth_depth == 4
    assert len(certificate.nodes) <= certificate.cardinality_bound
    for node in certificate.nodes:
        assert len(generator.predecessors(node)) <= certificate.maximum_in_degree
    repeat = generator.backward_causal_cone(((0, 1),), horizon=4)
    assert repeat == certificate


def test_zero_fixture_is_exact_but_generic_append_zero_image_is_rejected() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=3))
    zero = generator.compatibility_certificate(1, domain="zero_state_zero_input")
    generic = generator.compatibility_certificate(1, domain="append_zero_unit_cube")
    assert zero.certified
    assert zero.witness_defect == 0.0
    assert not generic.certified
    assert generic.witness_defect > 0.0
    assert "refused" in generic.reason


def test_append_zero_generic_fixture_is_exact_when_boundary_activation_is_removed() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=2, upward_gain=0.0))
    certificate = generator.compatibility_certificate(0, domain="append_zero_unit_cube")
    assert certificate.certified
    assert certificate.witness_defect == 0.0
    assert not generator.requires_extension(0)


@pytest.mark.parametrize("upward_gain", [1e-15, 1e-16, 1e-20])
def test_tiny_nonzero_boundary_gain_is_never_promoted_to_exact(upward_gain) -> None:
    generator = NestedTowerGenerator(
        TowerSpec(shell_width=2, maximum_depth=2, upward_gain=upward_gain)
    )
    certificate = generator.compatibility_certificate(0, domain="append_zero_unit_cube")
    assert generator.spec.upward_gain > 0.0
    assert not certificate.certified
    assert "nonzero upward" in certificate.reason
    assert generator.requires_extension(0)


def test_level_independent_contraction_bound_matches_jacobi_update() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=3, maximum_depth=5))
    expected_q = (
        generator.spec.recurrence_gain + generator.spec.upward_gain + generator.spec.downward_gain
    )
    rng = np.random.default_rng(17)
    for depth in range(6):
        certificate = generator.certify_prefix(depth)
        assert certificate.certified
        assert certificate.metric == "global_coordinate_sup"
        assert "global coordinate sup norm" in certificate.reason
        assert certificate.level_independent_bound == pytest.approx(expected_q)
        left = tuple(rng.uniform(-1, 1, size=3) for _ in range(depth + 1))
        right = tuple(rng.uniform(-1, 1, size=3) for _ in range(depth + 1))
        observation = np.zeros(3)
        next_left = generator.step(left, observation)
        next_right = generator.step(right, observation)
        before = max(float(np.max(np.abs(a - b))) for a, b in zip(left, right))
        after = max(float(np.max(np.abs(a - b))) for a, b in zip(next_left, next_right))
        assert after <= expected_q * before + 1e-14


def test_sup_norm_certificate_does_not_imply_euclidean_contraction() -> None:
    generator = NestedTowerGenerator(
        TowerSpec(
            shell_width=3,
            maximum_depth=0,
            recurrence_gain=0.98,
            upward_gain=0.0,
            downward_gain=0.0,
            contraction_cap=0.99,
        )
    )
    certificate = generator.certify_prefix(0)
    zero_state_jacobian = generator.recurrence_operator()
    assert certificate.certified
    assert certificate.metric == "global_coordinate_sup"
    assert certificate.level_independent_bound == pytest.approx(0.98)
    assert np.linalg.norm(zero_state_jacobian, ord=2) > 1.0


def test_topology_does_not_issue_stability_for_an_unstable_dynamic_rule() -> None:
    generator = NestedTowerGenerator(
        TowerSpec(
            shell_width=2,
            maximum_depth=3,
            recurrence_gain=0.8,
            upward_gain=0.2,
            downward_gain=0.1,
        )
    )
    assert generator.audit_prefix(3).is_strongly_connected
    certificate = generator.certify_prefix(3)
    assert certificate.level_independent_bound > 1.0
    assert not certificate.certified
    assert "not strict" in certificate.reason


def test_jacobi_certificate_refuses_a_schedule_mismatch() -> None:
    generator = NestedTowerGenerator()
    certificate = generator.certify_prefix(2, schedule="gauss_seidel")
    assert not certificate.certified
    assert "different update schedule" in certificate.reason
    with pytest.raises(ValueError, match="exact string"):
        generator.certify_prefix(2, schedule=EqualStr("previous_tick_jacobi"))
    with pytest.raises(ValueError, match="exact string"):
        generator.compatibility_certificate(1, domain=EqualStr("zero_state_zero_input"))


def test_observations_are_normalized_before_the_dimensionless_core() -> None:
    generator = NestedTowerGenerator(
        TowerSpec(
            shell_width=3,
            maximum_depth=1,
            observation_scales=(2.0, 4.0, 8.0),
        )
    )
    normalized = generator.normalize_observation((2.0, -8.0, 4.0))
    assert np.array_equal(normalized, np.asarray((1.0, -2.0, 0.5)))
    next_state = generator.step((np.zeros(3),), normalized)
    assert np.all(np.isfinite(next_state[0]))
    assert np.max(np.abs(next_state[0])) <= 1.0
    with pytest.raises(ValueError, match="finite"):
        generator.normalize_observation((0.0, math.inf, 0.0))
    with pytest.raises(ValueError, match="real number"):
        generator.normalize_observation((0.0, "1.0", 0.0))
    with pytest.raises(ValueError, match="real number"):
        generator.step(((False, 0.0, 0.0),), np.zeros(3))


@pytest.mark.parametrize(
    "outside",
    [
        1.0 + 5e-13,
        -1.0 - 5e-13,
        np.nextafter(1.0, math.inf),
        np.nextafter(-1.0, -math.inf),
    ],
)
def test_external_state_domain_is_the_exact_closed_interval(outside) -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=1, maximum_depth=0))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        generator.step(((outside,),), (0.0,))
    assert len(generator.step(((-1.0,),), (0.0,))) == 1
    assert len(generator.step(((1.0,),), (0.0,))) == 1


@pytest.mark.parametrize(
    "field",
    [
        "recurrence_gain",
        "upward_gain",
        "downward_gain",
        "input_gain",
        "level_decay",
        "contraction_cap",
    ],
)
@pytest.mark.parametrize("invalid", [True, "0.2"])
def test_every_numeric_config_rejects_bool_and_encoded_text(field, invalid) -> None:
    with pytest.raises(ValueError, match="real number"):
        TowerSpec(**{field: invalid})


def test_real_numeric_config_is_canonicalized_to_finite_float() -> None:
    spec = TowerSpec(
        recurrence_gain=0,
        upward_gain=0,
        downward_gain=0,
        input_gain=1,
        level_decay=1,
        contraction_cap=0.9,
        observation_scales=(1, 2, 3),
    )
    fields = (
        "recurrence_gain",
        "upward_gain",
        "downward_gain",
        "input_gain",
        "level_decay",
        "contraction_cap",
    )
    assert all(isinstance(getattr(spec, field), float) for field in fields)
    assert all(isinstance(scale, float) for scale in spec.observation_scales)


def test_manifest_names_serialized_metadata_without_capacity_claim() -> None:
    manifest = NestedTowerGenerator(TowerSpec(shell_width=3)).manifest
    assert manifest.serialized_operator_scalar_count == 27
    assert not hasattr(manifest, "generated_parameter_count")
    assert not hasattr(TowerSpec(), "depth_error_tolerance")
    assert not hasattr(TowerSpec(), "hysteresis_ticks")


def test_infinite_tail_certificate_closes_approximation_without_exact_compatibility() -> None:
    generator = NestedTowerGenerator(TowerSpec(maximum_depth=8))
    certificate = generator.infinite_tail_certificate(4)
    assert certificate.certified
    assert certificate.metric == "global_coordinate_sup"
    assert certificate.uniform_contraction_bound == pytest.approx(0.54)
    assert certificate.boundary_defect_bound == pytest.approx(0.16 * 0.72**5)
    assert certificate.fixed_point_error_bound == pytest.approx(
        certificate.boundary_defect_bound / 0.46
    )
    assert not generator.compatibility_certificate(4, domain="append_zero_unit_cube").certified


def test_infinite_tail_bound_dominates_deeper_prefix_fixed_point_difference() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=2, maximum_depth=10))
    observation = np.asarray((0.7, -0.4))

    def fixed_point(depth: int) -> tuple[np.ndarray, ...]:
        state = tuple(np.zeros(2) for _ in range(depth + 1))
        for _ in range(500):
            updated = generator.step(state, observation)
            error = max(float(np.max(np.abs(a - b))) for a, b in zip(updated, state))
            state = updated
            if error < 1e-14:
                break
        return state

    shallow = fixed_point(3)
    deep = fixed_point(10)
    embedded = (*shallow, *(np.zeros(2) for _ in range(7)))
    observed_error = max(float(np.max(np.abs(left - right))) for left, right in zip(embedded, deep))
    assert observed_error <= generator.infinite_tail_certificate(3).fixed_point_error_bound


def test_rollout_tail_bound_is_recursive_and_refuses_nondecaying_levels() -> None:
    generator = NestedTowerGenerator(TowerSpec(maximum_depth=6))
    certificate = generator.rollout_tail_certificate(2, 5, initial_error_bound=0.1)
    expected = 0.54**5 * 0.1 + (1.0 - 0.54**5) / 0.46 * (0.16 * 0.72**3)
    assert certificate.certified
    assert certificate.rollout_error_bound == pytest.approx(expected)

    nondecaying = NestedTowerGenerator(TowerSpec(maximum_depth=2, level_decay=1.0))
    refused = nondecaying.infinite_tail_certificate(1)
    assert not refused.certified
    assert "strictly below one" in refused.reason


@pytest.mark.parametrize(
    "poison",
    [
        False,
        0,
        None,
        "",
        "1.0",
        [],
        [1.0],
        LyingTuple((2.0,)),
        (value for value in (1.0,)),
    ],
)
def test_only_exact_empty_tuple_activates_default_observation_scales(poison) -> None:
    with pytest.raises(ValueError, match="exact empty tuple"):
        TowerSpec(shell_width=1, observation_scales=poison)
    assert TowerSpec(shell_width=1, observation_scales=()).observation_scales == (1.0,)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shell_width": 2.5}, "positive integer"),
        ({"shell_width": True}, "positive integer"),
        ({"shell_width": EqualInt(3)}, "positive integer"),
        ({"maximum_depth": 1.5}, "nonnegative integer"),
        ({"maximum_depth": False}, "nonnegative integer"),
        ({"maximum_depth": EqualInt(2)}, "nonnegative integer"),
        ({"update_schedule": EqualStr("previous_tick_jacobi")}, "only"),
        ({"observation_scales": (1.0, 0.0, 1.0)}, "positive"),
        ({"observation_scales": (1.0, True, 1.0)}, "real number"),
        ({"observation_scales": (1.0, "2.0", 1.0)}, "real number"),
        ({"level_decay": 0.0}, "level_decay"),
        ({"contraction_cap": 1.0}, "contraction_cap"),
    ],
)
def test_spec_rejects_untyped_or_dimensionally_unresolved_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        TowerSpec(**kwargs)


def test_depth_and_node_indices_fail_closed() -> None:
    generator = NestedTowerGenerator(TowerSpec(maximum_depth=2))
    for depth in (True, 1.5, EqualInt(0), -1, 3):
        with pytest.raises((TypeError, ValueError)):
            generator.prefix(depth)
    for node in (
        (-1, 0),
        (0, -1),
        (0, 3),
        (0.0, 1),
        (True, 1),
        (EqualInt(0), 1),
    ):
        with pytest.raises((TypeError, ValueError)):
            generator.predecessors(node)
    with pytest.raises((TypeError, ValueError)):
        generator.upward_operator(EqualInt(0))
    with pytest.raises(ValueError, match="nonnegative integer"):
        generator.backward_causal_cone(((0, 0),), EqualInt(1))
    with pytest.raises(ValueError, match="nonnegative integer"):
        generator.forward_unroll(0, EqualInt(1))


@pytest.mark.parametrize(
    "states",
    [
        [(0.0,), (0.0,)],
        ShortLyingList([(0.0,), (0.0,)]),
        ShortLyingTuple(((0.0,), (0.0,))),
    ],
)
def test_outer_state_sequences_cannot_lie_about_registered_depth(states) -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=1, maximum_depth=0))
    with pytest.raises(ValueError, match=r"depth must lie in \[0, 0\]"):
        generator.step_with_messages(states, (0.0,), (), ())


@pytest.mark.parametrize(
    "messages",
    [
        [(0.0,), (0.0,)],
        ShortLyingList([(0.0,), (0.0,)]),
        ShortLyingTuple(((0.0,), (0.0,))),
    ],
)
def test_outer_message_sequences_are_canonicalized_before_length_checks(messages) -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=1, maximum_depth=1))
    states = ((0.0,), (0.0,))
    valid = ((0.0,),)
    with pytest.raises(ValueError, match="active bridge count"):
        generator.step_with_messages(states, (0.0,), messages, valid)
    with pytest.raises(ValueError, match="active bridge count"):
        generator.step_with_messages(states, (0.0,), valid, messages)


def test_node_tuple_subclasses_cannot_forge_predecessor_or_cone_certificates() -> None:
    generator = NestedTowerGenerator(TowerSpec(shell_width=1, maximum_depth=1))
    evil = EvilNode((999, 999))
    with pytest.raises(TypeError, match="integer pair"):
        generator.predecessors(evil)
    with pytest.raises(TypeError, match="integer pair"):
        generator.backward_causal_cone((evil,), horizon=1)
