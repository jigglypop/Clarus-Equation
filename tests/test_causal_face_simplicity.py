from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_face_simplicity import (
    CompositionFace,
    composition_faces,
    cross_simplicity_residual,
    face_incidence_audit,
    face_simplicity_verdict,
    fan_euler_characteristic,
    geometric_self_dual_triple,
    maximum_poisson_exact_valence_probability,
    minimum_block_depth,
    simplicity_block_audit,
    simplicity_residual,
    soft_block_simplicity_weight,
)


D = 3.1777584234


def test_composition_face_is_canonical_factorization_triangle() -> None:
    fine = {("u", "a"), ("a", "v"), ("u", "b"), ("b", "v")}
    coarse = {("u", "v")}

    faces = composition_faces(fine, coarse)

    assert faces == (
        CompositionFace("u", "a", "v"),
        CompositionFace("u", "b", "v"),
    )
    assert all(face.oriented_boundary[-1] == ("v", "u") for face in faces)
    assert fan_euler_characteristic(len(faces)) == 1


def test_composition_faces_are_equivariant_under_relabeling() -> None:
    fine = {(0, 1), (1, 3), (0, 2), (2, 3)}
    coarse = {(0, 3)}
    relabel = {0: "z", 1: "x", 2: "y", 3: "w"}

    original = composition_faces(fine, coarse)
    mapped = {
        CompositionFace(
            relabel[face.source],
            relabel[face.middle],
            relabel[face.target],
        )
        for face in original
    }
    relabeled_faces = set(
        composition_faces(
            {(relabel[u], relabel[v]) for u, v in fine},
            {(relabel[u], relabel[v]) for u, v in coarse},
        )
    )

    assert mapped == relabeled_faces


def test_causal_cycles_are_rejected() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        composition_faces({(0, 1), (1, 0)}, {(0, 1)})


def test_ce_face_incidence_requires_a_coarse_block() -> None:
    one = face_incidence_audit(D, 1)
    four = face_incidence_audit(D, 4)

    assert one.expected_faces == pytest.approx(D - 1.0)
    assert one.probability_at_least_minimum == pytest.approx(
        0.17629188622245506,
        rel=1.0e-12,
    )
    assert four.probability_at_least_minimum == pytest.approx(
        0.9739978016710592,
        rel=1.0e-12,
    )
    assert minimum_block_depth(D, confidence=0.95) == 4
    assert minimum_block_depth(D, confidence=0.99) == 5


def test_raw_poisson_cannot_concentrate_on_exact_tetrahedral_valence() -> None:
    maximum = maximum_poisson_exact_valence_probability(4)

    assert maximum == pytest.approx(
        math.exp(-4.0) * 4.0**4 / math.factorial(4),
        rel=1.0e-15,
    )
    assert maximum < 0.20


def test_geometric_self_dual_triple_satisfies_plebanski_simplicity() -> None:
    triple = geometric_self_dual_triple(np.eye(4))

    assert simplicity_residual(triple) < 1.0e-14


def test_conformal_aligned_cells_remain_simple_after_blocking() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    second = geometric_self_dual_triple(1.7 * np.eye(4))
    audit = simplicity_block_audit(first, second)

    assert audit.first_local_residual < 1.0e-14
    assert audit.second_local_residual < 1.0e-14
    assert audit.cross_residual < 1.0e-14
    assert audit.blocked_residual < 1.0e-14
    assert audit.local_simplicity_sufficient


def test_individually_simple_mismatched_cells_fail_block_simplicity() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    mismatched_tetrad = np.array(
        [
            [0.74341184, 0.80016662, 0.30490388, -0.47971556],
            [0.02980649, 1.23067583, -0.07551285, 0.27316411],
            [-0.02660693, 0.26689902, 1.57540904, -0.27026490],
            [0.08125544, -0.18532303, 0.05090736, 0.52512219],
        ]
    )
    second = geometric_self_dual_triple(mismatched_tetrad)
    audit = simplicity_block_audit(first, second)

    assert audit.first_local_residual < 1.0e-12
    assert audit.second_local_residual < 1.0e-12
    assert audit.cross_residual > 0.03
    assert audit.blocked_residual > 0.10
    assert not audit.local_simplicity_sufficient


def test_soft_simplicity_weight_penalizes_shape_mismatch() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    aligned = geometric_self_dual_triple(1.2 * np.eye(4))
    mismatch = geometric_self_dual_triple(
        np.array(
            [
                [1.0, 0.4, 0.2, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, 0.7],
            ]
        )
    )

    assert soft_block_simplicity_weight(first, aligned, width=0.05) == pytest.approx(1.0)
    assert soft_block_simplicity_weight(first, mismatch, width=0.05) < 0.95


def test_cross_residual_is_exact_obstruction_for_two_simple_cells() -> None:
    first = geometric_self_dual_triple(np.eye(4))
    second = geometric_self_dual_triple(
        np.array(
            [
                [1.0, 0.3, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 0.0, 0.8],
            ]
        )
    )

    assert simplicity_residual(first) < 1.0e-14
    assert simplicity_residual(second) < 1.0e-14
    assert cross_simplicity_residual(first, second) > 0.0
    assert simplicity_residual(first + second) > 0.0


def test_verdict_records_both_topology_and_simplicity_obstructions() -> None:
    verdict = face_simplicity_verdict(D)

    assert verdict.canonical_face_attachment
    assert verdict.block_depth_95_percent == 4
    assert verdict.block_depth_99_percent == 5
    assert not verdict.raw_poisson_simplicial_concentration_possible
    assert not verdict.local_simplicity_closed_under_blocking
    assert "shape-matching" in verdict.remaining_obligation
