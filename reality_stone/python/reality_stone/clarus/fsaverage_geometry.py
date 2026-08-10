"""Small-dependency GIFTI parser and fsaverage local-geometry holdout gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import numpy as np


def read_gifti(path: Path, expected: dict[str, Any]) -> list[np.ndarray]:
    payload = path.read_bytes()
    if len(payload) != int(expected["bytes"]) or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise ValueError(f"integrity mismatch: {path}")
    root = ET.fromstring(payload)
    arrays = []
    for data_array in root.findall("DataArray"):
        dtype = np.dtype("<f4" if data_array.attrib["DataType"] == "NIFTI_TYPE_FLOAT32" else "<i4")
        encoded = base64.b64decode("".join(data_array.findtext("Data", "").split()))
        try:
            raw = zlib.decompress(encoded)
        except zlib.error:
            raw = zlib.decompress(encoded, 16 + zlib.MAX_WBITS)
        dimensions = [int(data_array.attrib[f"Dim{index}"]) for index in range(int(data_array.attrib["Dimensionality"]))]
        values = np.frombuffer(raw, dtype=dtype)
        if values.size != int(np.prod(dimensions)):
            raise ValueError(f"decoded element count mismatch: {path}")
        arrays.append(values.reshape(dimensions))
    return arrays


def _neighbors(vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    sets = [set() for _ in range(vertices)]
    for a, b, c in faces:
        sets[int(a)].update((int(b), int(c)))
        sets[int(b)].update((int(a), int(c)))
        sets[int(c)].update((int(a), int(b)))
    return [np.fromiter(sorted(values), dtype=int) for values in sets]


def _cotangent_operator(
    coordinates: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lumped masses and positive cotangent weights for undirected edges."""
    triangles = coordinates[faces]
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(twice_area <= 1e-12):
        raise ValueError("degenerate triangle in cortical mesh")

    mass = np.zeros(len(coordinates), dtype=float)
    for corner in range(3):
        np.add.at(mass, faces[:, corner], twice_area / 6.0)

    cotangent = []
    for corner, first, second in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        edge_a = triangles[:, first] - triangles[:, corner]
        edge_b = triangles[:, second] - triangles[:, corner]
        cotangent.append(np.einsum("ij,ij->i", edge_a, edge_b) / twice_area)

    left = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    right = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    raw_weight = 0.5 * np.concatenate(cotangent)
    low, high = np.minimum(left, right), np.maximum(left, right)
    key = low.astype(np.int64) * len(coordinates) + high
    unique_key, inverse = np.unique(key, return_inverse=True)
    weight = np.zeros(len(unique_key), dtype=float)
    np.add.at(weight, inverse, raw_weight)
    # Obtuse triangles can create negative cotangent edges. Clipping gives the
    # preregistered positive-semidefinite diffusion operator.
    weight = np.maximum(weight, 0.0)
    keep = weight > 1e-12
    return mass, unique_key[keep] // len(coordinates), unique_key[keep] % len(coordinates), weight[keep]


def _apply_laplace_beltrami(
    values: np.ndarray,
    mass: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Apply Δf = M^-1 W(f_j-f_i) without constructing a dense matrix."""
    array = np.asarray(values, dtype=float)
    vector = array.ndim == 1
    fields = array[:, None] if vector else array
    result = np.zeros_like(fields)
    flux = weight[:, None] * (fields[right] - fields[left])
    np.add.at(result, left, flux)
    np.add.at(result, right, -flux)
    result /= mass[:, None]
    return result[:, 0] if vector else result


def _laplace_beltrami_features(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    faces: np.ndarray,
    heat_cfl: float,
    heat_steps: list[int],
) -> np.ndarray:
    """Compute fixed-scale differential and heat features on the cortical manifold."""
    mass, left, right, weight = _cotangent_operator(coordinates, faces)
    return _operator_features(
        coordinates, curvature, mass, left, right, weight, heat_cfl, heat_steps
    )


def _operator_features(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    mass: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    weight: np.ndarray,
    heat_cfl: float,
    heat_steps: list[int],
) -> np.ndarray:
    laplace_curvature = _apply_laplace_beltrami(curvature, mass, left, right, weight)
    laplace_square = _apply_laplace_beltrami(curvature**2, mass, left, right, weight)
    gradient_energy = np.maximum(
        0.5 * (laplace_square - 2.0 * curvature * laplace_curvature), 0.0
    )
    mean_curvature = 0.5 * np.linalg.norm(
        _apply_laplace_beltrami(coordinates, mass, left, right, weight), axis=1
    )

    degree = np.zeros(len(coordinates), dtype=float)
    np.add.at(degree, left, weight)
    np.add.at(degree, right, weight)
    time_step = heat_cfl / np.max(degree / mass)
    requested = set(int(step) for step in heat_steps)
    diffused = curvature.copy()
    heat_fields = []
    for step in range(1, max(requested) + 1):
        diffused += time_step * _apply_laplace_beltrami(
            diffused, mass, left, right, weight
        )
        if step in requested:
            heat_fields.append(diffused.copy())
    return np.column_stack(
        [laplace_curvature, gradient_energy, mean_curvature, *heat_fields]
    )


def _principal_tangent_directions(
    coordinates: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Estimate an unoriented principal direction in every vertex tangent plane."""
    triangles = coordinates[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(coordinates, dtype=float)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    directions = np.empty_like(coordinates, dtype=float)
    for index, adjacent in enumerate(_neighbors(len(coordinates), faces)):
        offsets = coordinates[adjacent] - coordinates[index]
        tangent = offsets - np.outer(offsets @ normals[index], normals[index])
        covariance = tangent.T @ tangent / max(len(tangent), 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        direction -= np.dot(direction, normals[index]) * normals[index]
        directions[index] = direction / (np.linalg.norm(direction) + 1e-12)
    return directions


def _anisotropic_laplace_beltrami_features(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    faces: np.ndarray,
    anisotropy: float,
    heat_cfl: float,
    heat_steps: list[int],
) -> np.ndarray:
    """Diffuse preferentially along an unoriented local principal tangent field."""
    if not 0.0 <= anisotropy < 1.0:
        raise ValueError("anisotropy must be in [0, 1)")
    mass, left, right, weight = _cotangent_operator(coordinates, faces)
    directions = _principal_tangent_directions(coordinates, faces)
    edge = coordinates[right] - coordinates[left]
    edge /= np.linalg.norm(edge, axis=1, keepdims=True) + 1e-12
    alignment = 0.5 * (
        np.einsum("ij,ij->i", edge, directions[left]) ** 2
        + np.einsum("ij,ij->i", edge, directions[right]) ** 2
    )
    directional_weight = weight * (1.0 + anisotropy * (2.0 * alignment - 1.0))
    return _operator_features(
        coordinates,
        curvature,
        mass,
        left,
        right,
        directional_weight,
        heat_cfl,
        heat_steps,
    )


def _curvature_guided_laplace_beltrami_features(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    faces: np.ndarray,
    conductivity_floor: float,
    contrast_scales: list[float],
    heat_cfl: float,
    heat_steps: list[int],
) -> np.ndarray:
    """Apply edge-preserving manifold diffusion driven by curvature contrast."""
    if not 0.0 <= conductivity_floor <= 1.0:
        raise ValueError("conductivity_floor must be in [0, 1]")
    mass, left, right, weight = _cotangent_operator(coordinates, faces)
    edge_contrast = np.abs(curvature[right] - curvature[left])
    robust_scale = np.median(edge_contrast) + 1e-12
    feature_groups = []
    for contrast_scale in contrast_scales:
        if contrast_scale <= 0.0:
            raise ValueError("contrast scales must be positive")
        normalized = edge_contrast / (contrast_scale * robust_scale)
        conductivity = conductivity_floor + (1.0 - conductivity_floor) * np.exp(
            -(normalized**2)
        )
        feature_groups.append(
            _operator_features(
                coordinates,
                curvature,
                mass,
                left,
                right,
                weight * conductivity,
                heat_cfl,
                heat_steps,
            )
        )
    return np.column_stack(feature_groups)


def _features(coordinates: np.ndarray, curvature: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = coordinates - coordinates.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1)
    global_features = np.column_stack([centered, radius])
    scalar_features = np.column_stack([global_features, curvature, curvature**2])
    local = np.empty((len(coordinates), 6))
    for index, adjacent in enumerate(_neighbors(len(coordinates), faces)):
        offsets = coordinates[adjacent] - coordinates[index]
        lengths = np.linalg.norm(offsets, axis=1)
        laplacian = offsets.mean(axis=0)
        covariance = offsets.T @ offsets / max(len(offsets), 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        radial = centered[index] / (radius[index] + 1e-8)
        local[index] = (
            lengths.mean(), lengths.std(), np.linalg.norm(laplacian), np.dot(laplacian, radial),
            eigenvalues[-1] / (eigenvalues.sum() + 1e-8), eigenvalues[0] / (eigenvalues[-1] + 1e-8),
        )
    return global_features, scalar_features, np.column_stack([scalar_features, local])


def _feature_tiers(
    coordinates: np.ndarray,
    curvature: np.ndarray,
    faces: np.ndarray,
    model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_features, scalar_features, legacy_candidate = _features(
        coordinates, curvature, faces
    )
    if model.get("feature_tier") not in {
        "laplace_beltrami_v1",
        "anisotropic_laplace_beltrami_v1",
        "curvature_guided_lb_v1",
    }:
        return global_features, scalar_features, scalar_features, legacy_candidate
    intrinsic = _laplace_beltrami_features(
        coordinates,
        curvature,
        faces,
        float(model["heat_cfl"]),
        [int(step) for step in model["heat_steps"]],
    )
    isotropic = np.column_stack([scalar_features, intrinsic])
    if model.get("feature_tier") == "laplace_beltrami_v1":
        return global_features, scalar_features, isotropic, isotropic
    if model.get("feature_tier") == "anisotropic_laplace_beltrami_v1":
        anisotropic = _anisotropic_laplace_beltrami_features(
            coordinates,
            curvature,
            faces,
            float(model["anisotropy"]),
            float(model["heat_cfl"]),
            [int(step) for step in model["heat_steps"]],
        )
    else:
        anisotropic = _curvature_guided_laplace_beltrami_features(
            coordinates,
            curvature,
            faces,
            float(model["conductivity_floor"]),
            [float(scale) for scale in model["contrast_scales"]],
            float(model["heat_cfl"]),
            [int(step) for step in model["heat_steps"]],
        )
    return global_features, scalar_features, isotropic, np.column_stack(
        [isotropic, anisotropic]
    )


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, penalty: float) -> np.ndarray:
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-8
    design = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    query = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
    return query @ weights


def run_fsaverage_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    files = prereg["source"]["files"]
    coordinates, faces = read_gifti(root / files["pial"]["path"], files["pial"])
    sulc = read_gifti(root / files["sulc"]["path"], files["sulc"])[0].astype(float)
    curvature = read_gifti(root / files["curv"]["path"], files["curv"])[0].astype(float)
    if coordinates.shape != (10242, 3) or faces.shape != (20480, 3) or sulc.shape != (10242,):
        raise ValueError("unexpected fsaverage shape")
    global_x, scalar_x, isotropic_x, candidate_x = _feature_tiers(
        coordinates.astype(float), curvature, faces, prereg["model"]
    )
    baseline_tier = prereg["model"].get("baseline_tier")
    if baseline_tier == "global_position":
        baseline_x = global_x
    elif baseline_tier == "laplace_beltrami_v1":
        baseline_x = isotropic_x
    else:
        baseline_x = scalar_x
    centered = coordinates - coordinates.mean(axis=0)
    angle = (np.arctan2(centered[:, 1], centered[:, 0]) + 2 * np.pi) % (2 * np.pi)
    sectors = np.floor(8 * angle / (2 * np.pi)).astype(int)
    rows = []
    for sector in prereg["split"][f"{split}_sectors"]:
        test = sectors == int(sector)
        train = ~test
        baseline = _ridge_predict(baseline_x[train], sulc[train], baseline_x[test], prereg["model"]["ridge"])
        candidate = _ridge_predict(candidate_x[train], sulc[train], candidate_x[test], prereg["model"]["ridge"])
        scalar = _ridge_predict(scalar_x[train], sulc[train], scalar_x[test], prereg["model"]["ridge"])
        truth = sulc[test]
        baseline_sign = baseline
        candidate_sign = candidate
        scalar_sign = scalar
        if prereg["model"].get("readout") == "separate_ridge_depth_and_sign":
            sign_target = np.where(sulc[train] >= 0.0, 1.0, -1.0)
            baseline_sign = _ridge_predict(
                baseline_x[train], sign_target, baseline_x[test], prereg["model"]["ridge"]
            )
            candidate_sign = _ridge_predict(
                candidate_x[train], sign_target, candidate_x[test], prereg["model"]["ridge"]
            )
            scalar_sign = _ridge_predict(
                scalar_x[train], sign_target, scalar_x[test], prereg["model"]["ridge"]
            )
        rows.append({"sector": sector, "baseline_rmse": float(np.sqrt(np.mean((truth - baseline) ** 2))),
                     "candidate_rmse": float(np.sqrt(np.mean((truth - candidate) ** 2))),
                     "scalar_rmse": float(np.sqrt(np.mean((truth - scalar) ** 2))),
                     "baseline_sign_accuracy": float(np.mean(np.sign(baseline_sign) == np.sign(truth))),
                     "candidate_sign_accuracy": float(np.mean(np.sign(candidate_sign) == np.sign(truth))),
                     "scalar_sign_accuracy": float(np.mean(np.sign(scalar_sign) == np.sign(truth)))})
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "sector"}
    summary = {**mean, "rmse_reduction": 1 - mean["candidate_rmse"] / mean["baseline_rmse"],
               "sign_accuracy_gain": mean["candidate_sign_accuracy"] - mean["baseline_sign_accuracy"],
               "local_over_scalar_rmse_reduction": 1 - mean["candidate_rmse"] / mean["scalar_rmse"],
               "local_over_scalar_sign_gain": mean["candidate_sign_accuracy"] - mean["scalar_sign_accuracy"],
               "vertices": len(coordinates), "faces": len(faces)}
    summary["rmse_improved_sector_count"] = sum(
        row["candidate_rmse"] < row["baseline_rmse"] for row in rows
    )
    criteria = prereg["success_criteria"]
    checks = {"source_integrity": True,
              "rmse_improvement": summary["rmse_reduction"] >= criteria["candidate_rmse_reduction_min"],
              "sign_improvement": summary["sign_accuracy_gain"] >= criteria["candidate_sign_accuracy_gain_min"],
              "download_budget": prereg["source"]["download_bytes"] <= criteria["download_bytes_max"]}
    if "rmse_improved_sector_count_min" in criteria:
        checks["sector_consistency"] = (
            summary["rmse_improved_sector_count"]
            >= criteria["rmse_improved_sector_count_min"]
        )
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    return {"experiment": prereg["experiment"], "split": split, "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "summary": summary, "per_sector": rows,
            "resource_usage": {"downloaded_bytes": prereg["source"]["download_bytes"], "elapsed_seconds": elapsed}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_fsaverage_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
