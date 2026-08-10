"""Geometry-only detection of opposing cortical-bank bridge opportunities."""

from __future__ import annotations

import argparse
import heapq
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .fsaverage_geometry import read_gifti


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _vertex_normals(
    coordinates: np.ndarray, faces: np.ndarray, white: np.ndarray | None = None
) -> np.ndarray:
    triangles = coordinates[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(coordinates, dtype=float)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    orientation = (
        coordinates - white
        if white is not None
        else coordinates - coordinates.mean(axis=0)
    )
    if np.median(np.einsum("ij,ij->i", normals, orientation)) < 0.0:
        normals *= -1.0
    return normals


def _spatial_pairs(coordinates: np.ndarray, radius: float) -> np.ndarray:
    cell = np.floor(coordinates / radius).astype(int)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(map(tuple, cell)):
        buckets[key].append(index)
    offsets = [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ]
    pairs = []
    for first, key_array in enumerate(cell):
        key = tuple(key_array)
        for offset in offsets:
            neighbor_key = tuple(key[axis] + offset[axis] for axis in range(3))
            for second in buckets.get(neighbor_key, []):
                if second <= first:
                    continue
                if np.linalg.norm(coordinates[second] - coordinates[first]) <= radius:
                    pairs.append((first, second))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def _adjacency(
    coordinates: np.ndarray, faces: np.ndarray
) -> tuple[list[list[int]], list[list[tuple[int, float]]]]:
    topology = [[] for _ in coordinates]
    weighted = [[] for _ in coordinates]
    for first, second in _mesh_edges(faces):
        first, second = int(first), int(second)
        length = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        topology[first].append(second)
        topology[second].append(first)
        weighted[first].append((second, length))
        weighted[second].append((first, length))
    return topology, weighted


def _within_hops(topology: list[list[int]], source: int, maximum_hops: int) -> set[int]:
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        vertex, depth = queue.popleft()
        if depth == maximum_hops:
            continue
        for neighbor in topology[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return visited


def _target_distances(
    weighted: list[list[tuple[int, float]]],
    source: int,
    targets: set[int],
    cutoff: float,
) -> dict[int, float]:
    distances = {source: 0.0}
    queue = [(0.0, source)]
    found: dict[int, float] = {}
    while queue and len(found) < len(targets):
        distance, vertex = heapq.heappop(queue)
        if distance != distances.get(vertex) or distance > cutoff:
            continue
        if vertex in targets:
            found[vertex] = distance
        for neighbor, length in weighted[vertex]:
            candidate = distance + length
            if candidate <= cutoff and candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return found


def detect_fold_bridges(
    pial: np.ndarray,
    white: np.ndarray,
    faces: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, float | int]]:
    radius = float(settings["pial_distance_max_mm"])
    normals = _vertex_normals(pial, faces, white)
    pairs = _spatial_pairs(pial, radius)
    if len(pairs) == 0:
        return []
    delta = pial[pairs[:, 1]] - pial[pairs[:, 0]]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / (distance[:, None] + 1e-12)
    opposition = -np.einsum("ij,ij->i", normals[pairs[:, 0]], normals[pairs[:, 1]])
    face_first = np.einsum("ij,ij->i", normals[pairs[:, 0]], direction)
    face_second = np.einsum("ij,ij->i", normals[pairs[:, 1]], -direction)
    depth = np.linalg.norm(pial - white, axis=1)
    keep = (
        (opposition >= float(settings["normal_opposition_cosine_min"]))
        & (face_first >= float(settings["mutual_facing_cosine_min"]))
        & (face_second >= float(settings["mutual_facing_cosine_min"]))
        & (depth[pairs[:, 0]] >= float(settings["minimum_endpoint_depth_mm"]))
        & (depth[pairs[:, 1]] >= float(settings["minimum_endpoint_depth_mm"]))
    )
    pairs, distance, opposition = pairs[keep], distance[keep], opposition[keep]
    topology, weighted = _adjacency(pial, faces)
    grouped: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for (first, second), pial_distance, normal_opposition in zip(
        pairs, distance, opposition, strict=True
    ):
        grouped[int(first)].append((int(second), float(pial_distance), float(normal_opposition)))

    minimum_hops = int(settings["minimum_topological_hops"])
    cutoff = float(settings["surface_search_cutoff_mm"])
    bridges = []
    for first, candidates in grouped.items():
        local = _within_hops(topology, first, minimum_hops - 1)
        retained = [(second, dist, opp) for second, dist, opp in candidates if second not in local]
        if not retained:
            continue
        surface = _target_distances(
            weighted, first, {second for second, _, _ in retained}, cutoff
        )
        for second, pial_distance, normal_opposition in retained:
            if second not in surface:
                continue
            white_route = (
                depth[first]
                + np.linalg.norm(white[first] - white[second])
                + depth[second]
            )
            bridges.append(
                {
                    "first": first,
                    "second": second,
                    "pial_distance": pial_distance,
                    "surface_distance": surface[second],
                    "white_route_proxy": float(white_route),
                    "surface_to_white_route_ratio": float(surface[second] / white_route),
                    "normal_opposition": normal_opposition,
                }
            )
    return bridges


def _synthetic_strip(folded: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    across, along = 15, 5
    curve = []
    if folded:
        for index in range(across):
            if index <= 5:
                curve.append((-2.5, 12.5 - 2.5 * index))
            elif index <= 9:
                curve.append((-2.5 + 1.25 * (index - 5), 0.0))
            else:
                curve.append((2.5, 2.5 * (index - 9)))
    else:
        curve = [(2.5 * index, 0.0) for index in range(across)]
    pial = np.array(
        [[x, 3.0 * row, z] for x, z in curve for row in range(along)], dtype=float
    )
    faces = []
    for column in range(across - 1):
        for row in range(along - 1):
            first = column * along + row
            faces.extend([(first, first + along, first + 1), (first + 1, first + along, first + along + 1)])
    faces_array = np.asarray(faces, dtype=int)
    normals = _vertex_normals(pial, faces_array)
    white = pial + 2.0 * normals if folded else pial - 2.0 * normals
    return pial, white, faces_array


def _summary(
    bridges: list[dict[str, float | int]], strong_ratio: float = 1.0
) -> dict[str, float | int]:
    ratios = np.array([row["surface_to_white_route_ratio"] for row in bridges], dtype=float)
    strong_count = int(np.sum(ratios >= strong_ratio))
    return {
        "bridge_pair_count": len(bridges),
        "median_surface_to_white_route_ratio": float(np.median(ratios)) if len(ratios) else 0.0,
        "white_route_shorter_fraction": float(np.mean(ratios > 1.0)) if len(ratios) else 0.0,
        "maximum_surface_to_white_route_ratio": float(np.max(ratios)) if len(ratios) else 0.0,
        "strong_bridge_ratio_threshold": strong_ratio,
        "strong_bridge_pair_count": strong_count,
        "strong_bridge_fraction": strong_count / len(ratios) if len(ratios) else 0.0,
        "ratio_quantile_90": float(np.quantile(ratios, 0.9)) if len(ratios) else 0.0,
    }


def run_fold_bridge_gate(config_path: Path | str) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    files = prereg["source"]["files"]
    pial, faces = read_gifti(root / files["pial"]["path"], files["pial"])
    white, white_faces = read_gifti(root / files["white"]["path"], files["white"])
    if not np.array_equal(faces, white_faces):
        raise ValueError("pial and white topology mismatch")
    bridges = detect_fold_bridges(
        pial.astype(float), white.astype(float), faces, prereg["candidate"]
    )
    strong_ratio = float(prereg["candidate"].get("strong_bridge_ratio_min", 1.0))
    summary = _summary(bridges, strong_ratio)
    criteria = prereg["success_criteria"]
    checks = {
        "source_integrity": True,
        "bridge_population": summary["bridge_pair_count"] >= criteria["bridge_pair_count_min"],
        "download_budget": prereg["source"]["download_bytes"]
        <= criteria["download_bytes_max"],
    }
    if "strong_bridge_pair_count_min" in criteria:
        checks["strong_population"] = summary["strong_bridge_pair_count"] >= criteria[
            "strong_bridge_pair_count_min"
        ]
        checks["strong_fraction"] = summary["strong_bridge_fraction"] >= criteria[
            "strong_bridge_fraction_min"
        ]
        checks["tail_advantage"] = summary["ratio_quantile_90"] >= criteria[
            "ratio_quantile_90_min"
        ]
    else:
        checks["route_advantage"] = summary["median_surface_to_white_route_ratio"] >= criteria[
            "median_surface_to_white_route_ratio_min"
        ]
        checks["route_consistency"] = summary["white_route_shorter_fraction"] >= criteria[
            "white_route_shorter_fraction_min"
        ]
    synthetic = None
    if "reference" not in prereg:
        flat = _summary(
            detect_fold_bridges(*_synthetic_strip(False), prereg["candidate"]), strong_ratio
        )
        folded = _summary(
            detect_fold_bridges(*_synthetic_strip(True), prereg["candidate"]), strong_ratio
        )
        synthetic = {"flat": flat, "folded": folded}
        checks["flat_null"] = flat["bridge_pair_count"] <= criteria["synthetic_flat_pair_count_max"]
        if "synthetic_fold_strong_pair_count_min" in criteria:
            checks["fold_positive"] = folded["strong_bridge_pair_count"] >= criteria[
                "synthetic_fold_strong_pair_count_min"
            ]
        else:
            checks["fold_positive"] = folded["bridge_pair_count"] >= criteria[
                "synthetic_fold_pair_count_min"
            ]
    else:
        reference = json.loads((root / prereg["reference"]["artifact"]).read_text(encoding="utf-8"))
        if "left_to_right_strong_count_ratio_min" in criteria:
            count_ratio = reference["summary"]["strong_bridge_pair_count"] / max(
                summary["strong_bridge_pair_count"], 1
            )
            summary["left_to_right_strong_count_ratio"] = count_ratio
            checks["bilateral_count"] = (
                criteria["left_to_right_strong_count_ratio_min"]
                <= count_ratio
                <= criteria["left_to_right_strong_count_ratio_max"]
            )
        else:
            count_ratio = reference["summary"]["bridge_pair_count"] / max(
                summary["bridge_pair_count"], 1
            )
            summary["left_to_right_pair_count_ratio"] = count_ratio
            checks["bilateral_count"] = (
                criteria["left_to_right_pair_count_ratio_min"]
                <= count_ratio
                <= criteria["left_to_right_pair_count_ratio_max"]
            )
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    ranked = sorted(
        bridges, key=lambda row: float(row["surface_to_white_route_ratio"]), reverse=True
    )[:100]
    return {
        "experiment": prereg["experiment"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary,
        "synthetic_controls": synthetic,
        "top_bridges": ranked,
        "resource_usage": {
            "downloaded_bytes": prereg["source"]["download_bytes"],
            "elapsed_seconds": elapsed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_fold_bridge_gate(args.config)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
