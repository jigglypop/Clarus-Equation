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
    global_x, scalar_x, candidate_x = _features(coordinates.astype(float), curvature, faces)
    baseline_x = global_x if prereg["model"].get("baseline_tier") == "global_position" else scalar_x
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
        rows.append({"sector": sector, "baseline_rmse": float(np.sqrt(np.mean((truth - baseline) ** 2))),
                     "candidate_rmse": float(np.sqrt(np.mean((truth - candidate) ** 2))),
                     "scalar_rmse": float(np.sqrt(np.mean((truth - scalar) ** 2))),
                     "baseline_sign_accuracy": float(np.mean(np.sign(baseline) == np.sign(truth))),
                     "candidate_sign_accuracy": float(np.mean(np.sign(candidate) == np.sign(truth))),
                     "scalar_sign_accuracy": float(np.mean(np.sign(scalar) == np.sign(truth)))})
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "sector"}
    summary = {**mean, "rmse_reduction": 1 - mean["candidate_rmse"] / mean["baseline_rmse"],
               "sign_accuracy_gain": mean["candidate_sign_accuracy"] - mean["baseline_sign_accuracy"],
               "local_over_scalar_rmse_reduction": 1 - mean["candidate_rmse"] / mean["scalar_rmse"],
               "local_over_scalar_sign_gain": mean["candidate_sign_accuracy"] - mean["scalar_sign_accuracy"],
               "vertices": len(coordinates), "faces": len(faces)}
    criteria = prereg["success_criteria"]
    checks = {"source_integrity": True,
              "rmse_improvement": summary["rmse_reduction"] >= criteria["candidate_rmse_reduction_min"],
              "sign_improvement": summary["sign_accuracy_gain"] >= criteria["candidate_sign_accuracy_gain_min"],
              "download_budget": prereg["source"]["download_bytes"] <= criteria["download_bytes_max"]}
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
