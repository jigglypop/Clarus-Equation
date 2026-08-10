"""Two-dimensional tensor-growth folding solver with OBJ export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _smooth2(rng: np.random.Generator, size: int, modes: int = 5) -> np.ndarray:
    x, y = np.meshgrid(2 * np.pi * np.arange(size) / size, 2 * np.pi * np.arange(size) / size, indexing="ij")
    field = np.zeros((size, size))
    for _ in range(modes):
        kx, ky = rng.integers(1, 4, size=2)
        field += rng.normal() * np.sin(kx * x + ky * y + rng.uniform(0, 2 * np.pi))
    return field / (field.std() + 1e-12)


def _world(seed: int, size: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    scalar = _smooth2(rng, size)
    strength = _smooth2(rng, size)
    angle = np.pi * (0.5 + 0.18 * _smooth2(rng, size, 3))
    hidden_strength = _smooth2(rng, size)
    hidden_angle = rng.uniform(0, np.pi, (size, size))
    initial = rng.normal(0.0, 0.012, (size, size))
    return scalar, strength, angle, hidden_strength, hidden_angle, initial


def _lap(field: np.ndarray) -> np.ndarray:
    return np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field


def _gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.roll(field, -1, 0) - np.roll(field, 1, 0)) / 2, (np.roll(field, -1, 1) - np.roll(field, 1, 1)) / 2


def _divergence(x_flux: np.ndarray, y_flux: np.ndarray) -> np.ndarray:
    return (np.roll(x_flux, -1, 0) - np.roll(x_flux, 1, 0)) / 2 + (np.roll(y_flux, -1, 1) - np.roll(y_flux, 1, 1)) / 2


def _tensor(strength: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cosine, sine = np.cos(angle), np.sin(angle)
    return strength * (cosine**2 - 0.5), strength * cosine * sine, strength * (sine**2 - 0.5)


def solve_surface(seed: int, config: dict[str, Any], coefficient: float, hidden: bool) -> np.ndarray:
    scalar, strength, angle, hidden_strength, hidden_angle, height = _world(seed, int(config["size"]))
    qxx, qxy, qyy = _tensor(strength, angle)
    hxx, hxy, hyy = _tensor(hidden_strength, hidden_angle)
    gamma = float(config["base_growth"]) + float(config["scalar_growth_scale"]) * scalar
    scale = coefficient * float(config["tensor_scale"])
    hidden_scale = float(config["hidden_tensor_scale"]) if hidden else 0.0
    gxx, gxy, gyy = gamma + scale * qxx + hidden_scale * hxx, scale * qxy + hidden_scale * hxy, gamma + scale * qyy + hidden_scale * hyy
    for _ in range(int(config["steps"])):
        gx, gy = _gradient(height)
        norm2 = gx * gx + gy * gy
        velocity = (
            -float(config["bending"]) * _lap(_lap(height))
            -float(config["foundation"]) * height
            -_divergence(gxx * gx + gxy * gy, gxy * gx + gyy * gy)
            +float(config["quartic"]) * _divergence(norm2 * gx, norm2 * gy)
        )
        height += float(config["dt"]) * velocity
        height -= height.mean()
    return height


def _ridge_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted = prediction >= np.quantile(prediction, 0.8)
    actual = truth >= np.quantile(truth, 0.8)
    return float(np.logical_and(predicted, actual).sum() / np.logical_or(predicted, actual).sum())


def _select(seeds: list[int], config: dict[str, Any], truth_coefficient: float) -> float:
    truth = {seed: solve_surface(seed, config, truth_coefficient, True) for seed in seeds}
    losses = [
        np.mean([np.mean((truth[seed] - solve_surface(seed, config, float(candidate), False)) ** 2) for seed in seeds])
        for candidate in config["candidate_coefficients"]
    ]
    return float(config["candidate_coefficients"][int(np.argmin(losses))])


def _equivariance_error(seed: int, config: dict[str, Any]) -> float:
    # The isotropic null must commute exactly with grid rotation.
    original = solve_surface(seed, config, 0.0, False)
    scalar, strength, angle, hidden_strength, hidden_angle, initial = _world(seed, int(config["size"]))
    # Re-run the same null solver locally with rotated scalar and initial fields.
    gamma = float(config["base_growth"]) + float(config["scalar_growth_scale"]) * np.rot90(scalar)
    height = np.rot90(initial)
    for _ in range(int(config["steps"])):
        gx, gy = _gradient(height)
        norm2 = gx * gx + gy * gy
        height += float(config["dt"]) * (
            -float(config["bending"]) * _lap(_lap(height)) - float(config["foundation"]) * height
            -_divergence(gamma * gx, gamma * gy) + float(config["quartic"]) * _divergence(norm2 * gx, norm2 * gy)
        )
        height -= height.mean()
    return float(np.sqrt(np.mean((height - np.rot90(original)) ** 2)))


def write_obj(surface: np.ndarray, path: Path, vertical_scale: float = 8.0) -> None:
    size = surface.shape[0]
    lines = [f"v {i} {j} {vertical_scale * surface[i, j]:.8f}" for i in range(size) for j in range(size)]
    for i in range(size - 1):
        for j in range(size - 1):
            a, b, c, d = i * size + j + 1, (i + 1) * size + j + 1, (i + 1) * size + j + 2, i * size + j + 2
            lines.extend((f"f {a} {b} {c}", f"f {a} {c} {d}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run_surface_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    prereg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config, true_coefficient = prereg["simulation"], float(prereg["simulation"]["true_tensor_coefficient"])
    null_selected = _select(prereg["training_seeds"], config, 0.0)
    selected = _select(prereg["training_seeds"], config, true_coefficient)
    rows = []
    for seed in prereg[f"{split}_seeds"]:
        truth = solve_surface(seed, config, true_coefficient, True)
        baseline = solve_surface(seed, config, 0.0, False)
        candidate = solve_surface(seed, config, selected, False)
        rows.append({"seed": seed, "baseline_rmse": float(np.sqrt(np.mean((truth - baseline) ** 2))),
                     "candidate_rmse": float(np.sqrt(np.mean((truth - candidate) ** 2))),
                     "baseline_ridge_iou": _ridge_iou(baseline, truth), "candidate_ridge_iou": _ridge_iou(candidate, truth),
                     "rotation_equivariance_rmse": _equivariance_error(seed, config)})
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "seed"}
    summary = {"null_selected_coefficient": null_selected, "selected_coefficient": selected, **mean,
               "rmse_reduction": 1 - mean["candidate_rmse"] / mean["baseline_rmse"],
               "ridge_iou_gain": mean["candidate_ridge_iou"] - mean["baseline_ridge_iou"]}
    summary["ridge_mismatch_reduction"] = 1.0 - (1.0 - mean["candidate_ridge_iou"]) / max(1.0 - mean["baseline_ridge_iou"], 1e-12)
    criteria = prereg["success_criteria"]
    checks = {"null_guard": abs(null_selected) <= criteria["null_selected_coefficient_abs_max"],
              "coefficient_recovery": abs(selected - true_coefficient) <= criteria["alternative_selected_coefficient_error_max"],
              "surface_improvement": summary["rmse_reduction"] >= criteria["surface_rmse_reduction_min"],
              "ridge_improvement": (
                  summary["ridge_mismatch_reduction"] >= criteria["ridge_mismatch_reduction_min"]
                  if "ridge_mismatch_reduction_min" in criteria
                  else summary["ridge_iou_gain"] >= criteria["ridge_iou_gain_min"]
              ),
              "rotation_equivariance": summary["rotation_equivariance_rmse"] <= criteria["rotation_equivariance_rmse_max"],
              "no_external_data": criteria["external_download_bytes"] == 0}
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    return {"experiment": prereg["experiment"], "split": split, "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "summary": summary, "per_seed": rows,
            "resource_usage": {"external_download_bytes": 0, "elapsed_seconds": elapsed}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--obj", type=Path)
    args = parser.parse_args()
    report = run_surface_gate(args.config, args.split)
    if args.obj:
        prereg = json.loads(args.config.read_text(encoding="utf-8"))
        write_obj(solve_surface(prereg[f"{args.split}_seeds"][0], prereg["simulation"], prereg["simulation"]["true_tensor_coefficient"], True), args.obj)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
