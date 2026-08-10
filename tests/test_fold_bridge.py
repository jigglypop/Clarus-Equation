import numpy as np

from reality_stone.clarus.fold_bridge import (
    _synthetic_strip,
    _vertex_normals,
    detect_fold_bridges,
)


SETTINGS = {
    "pial_distance_max_mm": 6.0,
    "minimum_topological_hops": 4,
    "normal_opposition_cosine_min": 0.5,
    "mutual_facing_cosine_min": 0.35,
    "surface_search_cutoff_mm": 80.0,
    "minimum_endpoint_depth_mm": 0.5,
}


def test_flat_strip_has_no_bridge_but_folded_strip_does() -> None:
    flat = detect_fold_bridges(*_synthetic_strip(False), SETTINGS)
    folded = detect_fold_bridges(*_synthetic_strip(True), SETTINGS)
    assert flat == []
    assert len(folded) >= 4


def test_normals_and_bridge_ratios_are_rigid_invariant() -> None:
    pial, white, faces = _synthetic_strip(True)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original = detect_fold_bridges(pial, white, faces, SETTINGS)
    moved = detect_fold_bridges(
        pial @ rotation.T + np.array([4.0, 2.0, -3.0]),
        white @ rotation.T + np.array([4.0, 2.0, -3.0]),
        faces,
        SETTINGS,
    )
    assert np.allclose(_vertex_normals(pial, faces) @ rotation.T, _vertex_normals(pial @ rotation.T, faces))
    assert len(original) == len(moved)
    assert np.allclose(
        sorted(row["surface_to_white_route_ratio"] for row in original),
        sorted(row["surface_to_white_route_ratio"] for row in moved),
    )
