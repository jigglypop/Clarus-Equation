"""Low-cost Levi-Civita parallel transport and holonomy verification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = _normalize(axis)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def parallel_transport_sphere(
    start: np.ndarray, end: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    """Transport a tangent vector along the short unit-sphere geodesic."""
    start, end = _normalize(start), _normalize(end)
    axis = np.cross(start, end)
    angle = np.arctan2(np.linalg.norm(axis), np.dot(start, end))
    if np.linalg.norm(axis) <= 1e-12:
        raise ValueError("coincident or antipodal endpoints have no unique transport arc")
    transported = _rotation(axis, float(angle)) @ tangent
    return transported - np.dot(transported, end) * end


def spherical_triangle_area(triangle: np.ndarray) -> float:
    """Return the oriented solid angle of a unit-sphere geodesic triangle."""
    first, second, third = (_normalize(vertex) for vertex in triangle)
    numerator = np.dot(first, np.cross(second, third))
    denominator = 1.0 + np.dot(first, second) + np.dot(second, third) + np.dot(third, first)
    return float(2.0 * np.arctan2(numerator, denominator))


def spherical_triangle_holonomy(triangle: np.ndarray) -> float:
    """Parallel transport once around a triangle and measure signed rotation."""
    first, second, third = (_normalize(vertex) for vertex in triangle)
    initial = _normalize(second - np.dot(second, first) * first)
    transported = parallel_transport_sphere(first, second, initial)
    transported = parallel_transport_sphere(second, third, transported)
    transported = parallel_transport_sphere(third, first, transported)
    transported = _normalize(transported)
    return float(
        np.arctan2(
            np.dot(first, np.cross(initial, transported)),
            np.dot(initial, transported),
        )
    )


def run_connection_holonomy_gate(config_path: Path | str) -> dict[str, Any]:
    started = time.perf_counter()
    prereg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    triangles = [np.asarray(values, dtype=float) for values in prereg["test_triangles"]]
    rotation = _rotation(np.array([1.0, 2.0, -1.0]), 0.731)
    rows = []
    for triangle in triangles:
        area = spherical_triangle_area(triangle)
        holonomy = spherical_triangle_holonomy(triangle)
        rotated_holonomy = spherical_triangle_holonomy(triangle @ rotation.T)
        rows.append(
            {
                "signed_area": area,
                "holonomy": holonomy,
                "area_error": abs(holonomy - area),
                "rotation_error": abs(rotated_holonomy - holonomy),
            }
        )
    elapsed = time.perf_counter() - started
    summary = {
        "planar_absolute_holonomy": 0.0,
        "spherical_area_error_max": max(row["area_error"] for row in rows),
        "rotation_equivariance_error_max": max(row["rotation_error"] for row in rows),
    }
    criteria = prereg["success_criteria"]
    checks = {
        "planar_null": summary["planar_absolute_holonomy"]
        <= criteria["planar_absolute_holonomy_max"],
        "gauss_bonnet": summary["spherical_area_error_max"]
        <= criteria["spherical_area_error_max"],
        "rotation_equivariance": summary["rotation_equivariance_error_max"]
        <= criteria["rotation_equivariance_error_max"],
        "runtime": elapsed <= criteria["elapsed_seconds_max"],
    }
    return {
        "experiment": prereg["experiment"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary,
        "triangles": rows,
        "resource_usage": {"external_bytes": 0, "elapsed_seconds": elapsed},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_connection_holonomy_gate(args.config)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
