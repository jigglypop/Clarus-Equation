"""Spatial-holdout prediction of functional-atlas boundaries from manifold geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .fsaverage_geometry import (
    _apply_laplace_beltrami,
    _cotangent_operator,
    _curvature_guided_laplace_beltrami_features,
    _laplace_beltrami_features,
    read_gifti,
)


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _heat_landmark_coordinates(
    coordinates: np.ndarray,
    faces: np.ndarray,
    landmark_count: int,
    heat_cfl: float,
    heat_steps: list[int],
) -> np.ndarray:
    """Build global intrinsic coordinates by diffusing geometry-only landmark impulses."""
    centered = coordinates - coordinates.mean(axis=0)
    first = int(np.argmax(np.einsum("ij,ij->i", centered, centered)))
    landmarks = [first]
    minimum_distance = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    for _ in range(1, landmark_count):
        selected = int(np.argmax(minimum_distance))
        landmarks.append(selected)
        distance = np.sum((coordinates - coordinates[selected]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)

    mass, left, right, weight = _cotangent_operator(coordinates, faces)
    degree = np.zeros(len(coordinates), dtype=float)
    np.add.at(degree, left, weight)
    np.add.at(degree, right, weight)
    time_step = heat_cfl / np.max(degree / mass)
    fields = np.zeros((len(coordinates), landmark_count), dtype=float)
    fields[np.asarray(landmarks), np.arange(landmark_count)] = 1.0
    requested = set(int(step) for step in heat_steps)
    snapshots = []
    for step in range(1, max(requested) + 1):
        fields += time_step * _apply_laplace_beltrami(
            fields, mass, left, right, weight
        )
        if step in requested:
            snapshots.append(fields.copy())
    return np.column_stack(snapshots)


def _edge_features(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    centered = coordinates - coordinates.mean(axis=0)
    midpoint = 0.5 * (centered[edges[:, 0]] + centered[edges[:, 1]])
    edge_length = np.linalg.norm(
        coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1
    )
    first_curvature, second_curvature = curvature[edges[:, 0]], curvature[edges[:, 1]]
    baseline = np.column_stack(
        [
            midpoint,
            np.linalg.norm(midpoint, axis=1),
            edge_length,
            0.5 * (first_curvature + second_curvature),
            0.5 * (np.abs(first_curvature) + np.abs(second_curvature)),
            np.abs(first_curvature - second_curvature),
        ]
    )
    if model.get("feature_tier") == "pointwise_geometry_only":
        return baseline, baseline
    heat_steps = [int(step) for step in model["heat_steps"]]
    isotropic = _laplace_beltrami_features(
        coordinates,
        curvature,
        faces,
        float(model["heat_cfl"]),
        heat_steps,
    )
    guided = _curvature_guided_laplace_beltrami_features(
        coordinates,
        curvature,
        faces,
        float(model["conductivity_floor"]),
        [float(scale) for scale in model["contrast_scales"]],
        float(model["heat_cfl"]),
        heat_steps,
    )
    node_fields = np.column_stack([isotropic, guided])
    edge_fields = np.column_stack(
        [
            0.5 * (node_fields[edges[:, 0]] + node_fields[edges[:, 1]]),
            np.abs(node_fields[edges[:, 0]] - node_fields[edges[:, 1]]),
        ]
    )
    local_candidate = np.column_stack([baseline, edge_fields])
    if "landmark_count" not in model:
        return baseline, local_candidate
    landmarks = _heat_landmark_coordinates(
        coordinates,
        faces,
        int(model["landmark_count"]),
        float(model["heat_cfl"]),
        [int(step) for step in model["landmark_heat_steps"]],
    )
    landmark_edges = np.column_stack(
        [
            0.5 * (landmarks[edges[:, 0]] + landmarks[edges[:, 1]]),
            np.abs(landmarks[edges[:, 0]] - landmarks[edges[:, 1]]),
        ]
    )
    return local_candidate, np.column_stack([local_candidate, landmark_edges])


def _weighted_ridge_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    penalty: float,
) -> np.ndarray:
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-8
    design = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    query = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    positive = train_y > 0
    positive_count, negative_count = int(positive.sum()), int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("both boundary classes are required in every training fold")
    sample_weight = np.where(
        positive,
        len(train_y) / (2.0 * positive_count),
        len(train_y) / (2.0 * negative_count),
    )
    weighted_design = design * np.sqrt(sample_weight)[:, None]
    weighted_target = train_y * np.sqrt(sample_weight)
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )
    return query @ weights


def _roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target > 0
    positive_count, negative_count = int(positive.sum()), int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = ranks[positive].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _fold_rows(
    baseline_x: np.ndarray,
    candidate_x: np.ndarray,
    target: np.ndarray,
    sectors: np.ndarray,
    held_out: list[int],
    penalty: float,
) -> list[dict[str, float | int]]:
    rows = []
    signed_target = np.where(target, 1.0, -1.0)
    for sector in held_out:
        test = sectors == sector
        train = ~test
        baseline_score = _weighted_ridge_scores(
            baseline_x[train], signed_target[train], baseline_x[test], penalty
        )
        candidate_score = _weighted_ridge_scores(
            candidate_x[train], signed_target[train], candidate_x[test], penalty
        )
        rows.append(
            {
                "sector": sector,
                "edges": int(test.sum()),
                "boundary_edges": int(target[test].sum()),
                "baseline_auc": _roc_auc(target[test], baseline_score),
                "candidate_auc": _roc_auc(target[test], candidate_score),
            }
        )
    return rows


def run_functional_boundary_gate(
    config_path: Path | str, split: str = "validation"
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    files = prereg["source"]["files"]
    coordinates, faces = read_gifti(root / files["pial"]["path"], files["pial"])
    curvature = read_gifti(root / files["curv"]["path"], files["curv"])[0].astype(float)
    labels = read_gifti(root / files["labels"]["path"], files["labels"])[0]
    edges = _mesh_edges(faces)
    eligible = (labels[edges[:, 0]] > 0) & (labels[edges[:, 1]] > 0)
    edges = edges[eligible]
    target = labels[edges[:, 0]] != labels[edges[:, 1]]
    baseline_x, candidate_x = _edge_features(
        coordinates.astype(float), curvature, faces, edges, prereg["model"]
    )
    centered = coordinates - coordinates.mean(axis=0)
    midpoint = 0.5 * (centered[edges[:, 0]] + centered[edges[:, 1]])
    angle = (np.arctan2(midpoint[:, 1], midpoint[:, 0]) + 2 * np.pi) % (2 * np.pi)
    sectors = np.floor(8 * angle / (2 * np.pi)).astype(int)
    held_out = [int(value) for value in prereg["split"][f"{split}_sectors"]]
    rows = _fold_rows(
        baseline_x,
        candidate_x,
        target,
        sectors,
        held_out,
        float(prereg["model"]["ridge"]),
    )
    baseline_auc = float(np.mean([row["baseline_auc"] for row in rows]))
    candidate_auc = float(np.mean([row["candidate_auc"] for row in rows]))

    control = prereg["negative_control"]
    generator = np.random.default_rng(int(control["seed"]))
    permutation_aucs = []
    for _ in range(int(control["permutations"])):
        shuffled = target.copy()
        for sector in range(8):
            indices = np.flatnonzero(sectors == sector)
            shuffled[indices] = generator.permutation(shuffled[indices])
        shuffled_rows = _fold_rows(
            baseline_x,
            candidate_x,
            shuffled,
            sectors,
            held_out,
            float(prereg["model"]["ridge"]),
        )
        permutation_aucs.append(
            float(np.mean([row["candidate_auc"] for row in shuffled_rows]))
        )

    summary = {
        "eligible_edges": len(edges),
        "boundary_edges": int(target.sum()),
        "boundary_prevalence": float(target.mean()),
        "baseline_auc": baseline_auc,
        "candidate_auc": candidate_auc,
        "candidate_auc_gain": candidate_auc - baseline_auc,
        "auc_improved_sector_count": sum(
            row["candidate_auc"] > row["baseline_auc"] for row in rows
        ),
        "permutation_mean_auc": float(np.mean(permutation_aucs)),
    }
    criteria = prereg["success_criteria"]
    checks = {
        "source_integrity": True,
        "candidate_auc": candidate_auc >= criteria["candidate_auc_min"],
        "auc_gain": summary["candidate_auc_gain"] >= criteria["candidate_auc_gain_min"],
        "sector_consistency": summary["auc_improved_sector_count"]
        >= criteria["auc_improved_sector_count_min"],
        "permutation_control": abs(summary["permutation_mean_auc"] - 0.5)
        <= criteria["permutation_mean_auc_deviation_max"],
        "download_budget": prereg["source"]["download_bytes"]
        <= criteria["download_bytes_max"],
    }
    if "candidate_auc_max" in criteria:
        checks["candidate_auc_ceiling"] = (
            candidate_auc <= criteria["candidate_auc_max"]
        )
    if "desikan_minus_candidate_auc_min" in criteria:
        reference_auc = float(prereg["reference"]["desikan_auc"])
        summary["desikan_minus_candidate_auc"] = reference_auc - candidate_auc
        checks["anatomy_function_gap"] = (
            summary["desikan_minus_candidate_auc"]
            >= criteria["desikan_minus_candidate_auc_min"]
        )
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    return {
        "experiment": prereg["experiment"],
        "split": split,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary,
        "per_sector": rows,
        "negative_control_aucs": permutation_aucs,
        "resource_usage": {
            "downloaded_bytes": prereg["source"]["download_bytes"],
            "elapsed_seconds": elapsed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_functional_boundary_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
