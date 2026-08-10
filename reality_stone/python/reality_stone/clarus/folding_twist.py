"""Differential-growth folding null versus a local coupling/twist term."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _smooth_field(rng: np.random.Generator, points: int, harmonics: tuple[int, ...]) -> np.ndarray:
    coordinate = 2.0 * np.pi * np.arange(points) / points
    field = sum(rng.normal() * np.sin(k * coordinate + rng.uniform(0, 2 * np.pi)) for k in harmonics)
    return np.asarray(field / (np.std(field) + 1e-12), dtype=float)


def _divergence(flux: np.ndarray) -> np.ndarray:
    return (np.roll(flux, -1) - np.roll(flux, 1)) / 2.0


def simulate_fold(seed: int, config: dict[str, Any], twist: float, hidden: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = int(config["points"])
    base = _smooth_field(rng, points, (1, 2, 3))
    coupling = _smooth_field(rng, points, (2, 4, 5))
    hidden_draw = _smooth_field(rng, points, (3, 6, 7))
    hidden_field = hidden_draw if hidden else np.zeros(points)
    growth = (
        float(config["base_growth"])
        + float(config["growth_field_scale"]) * base
        + twist * float(config["growth_field_scale"]) * coupling
        + float(config["hidden_growth_scale"]) * hidden_field
    )
    height = rng.normal(0.0, 0.015, points)
    dt = float(config["dt"])
    for _ in range(int(config["steps"])):
        slope = (np.roll(height, -1) - np.roll(height, 1)) / 2.0
        laplacian = np.roll(height, -1) - 2.0 * height + np.roll(height, 1)
        biharmonic = np.roll(laplacian, -1) - 2.0 * laplacian + np.roll(laplacian, 1)
        velocity = (
            -float(config["bending"]) * biharmonic
            -float(config["foundation"]) * height
            -_divergence(growth * slope)
            +float(config["quartic"]) * _divergence(slope**3)
        )
        height += dt * velocity
        height -= height.mean()
    return height


def _peaks(field: np.ndarray) -> np.ndarray:
    return (field > np.roll(field, 1)) & (field > np.roll(field, -1)) & (field > np.quantile(field, 0.65))


def _peak_alignment(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted_peaks, true_peaks = np.flatnonzero(_peaks(prediction)), np.flatnonzero(_peaks(truth))
    if not len(predicted_peaks) or not len(true_peaks):
        return 0.0
    points = len(truth)
    distances = [min(min(abs(p - q), points - abs(p - q)) for p in predicted_peaks) for q in true_peaks]
    return float(np.mean(np.asarray(distances) <= 2))


def _peak_distance(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted_peaks, true_peaks = np.flatnonzero(_peaks(prediction)), np.flatnonzero(_peaks(truth))
    if not len(predicted_peaks) or not len(true_peaks):
        return float(len(truth) / 2)
    points = len(truth)
    distances = [min(min(abs(p - q), points - abs(p - q)) for p in predicted_peaks) for q in true_peaks]
    return float(np.mean(distances))


def _select(seeds: list[int], config: dict[str, Any], truth_twist: float) -> float:
    truths = {seed: simulate_fold(seed, config, truth_twist, True) for seed in seeds}
    losses = []
    for candidate in config["candidate_coefficients"]:
        error = []
        for seed in seeds:
            truth = truths[seed]
            prediction = simulate_fold(seed, config, float(candidate), False)
            error.append(np.mean((truth - prediction) ** 2))
        losses.append(float(np.mean(error)))
    return float(config["candidate_coefficients"][int(np.argmin(losses))])


def run_folding_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    config = prereg["simulation"]
    true_twist = float(config["true_twist_coefficient"])
    null_selected = _select(prereg["training_seeds"], config, 0.0)
    alternative_selected = _select(prereg["training_seeds"], config, true_twist)
    rows = []
    for seed in prereg[f"{split}_seeds"]:
        null_truth = simulate_fold(seed, config, 0.0, True)
        null_base = simulate_fold(seed, config, 0.0, False)
        null_candidate = null_base if null_selected == 0.0 else simulate_fold(seed, config, null_selected, False)
        truth = simulate_fold(seed, config, true_twist, True)
        baseline = null_base
        candidate = simulate_fold(seed, config, alternative_selected, False)
        rows.append({
            "seed": seed,
            "null_baseline_rmse": float(np.sqrt(np.mean((null_truth - null_base) ** 2))),
            "null_candidate_rmse": float(np.sqrt(np.mean((null_truth - null_candidate) ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean((truth - baseline) ** 2))),
            "candidate_rmse": float(np.sqrt(np.mean((truth - candidate) ** 2))),
            "baseline_peak_alignment": _peak_alignment(baseline, truth),
            "candidate_peak_alignment": _peak_alignment(candidate, truth),
            "baseline_peak_distance": _peak_distance(baseline, truth),
            "candidate_peak_distance": _peak_distance(candidate, truth),
        })
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    baseline_rmse, candidate_rmse = mean("baseline_rmse"), mean("candidate_rmse")
    summary = {
        "null_selected_coefficient": null_selected,
        "alternative_selected_coefficient": alternative_selected,
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_reduction": 1.0 - candidate_rmse / baseline_rmse,
        "baseline_peak_alignment": mean("baseline_peak_alignment"),
        "candidate_peak_alignment": mean("candidate_peak_alignment"),
        "peak_alignment_gain": mean("candidate_peak_alignment") - mean("baseline_peak_alignment"),
        "baseline_peak_distance": mean("baseline_peak_distance"),
        "candidate_peak_distance": mean("candidate_peak_distance"),
        "null_rmse_increase": mean("null_candidate_rmse") - mean("null_baseline_rmse"),
    }
    summary["peak_distance_reduction"] = 1.0 - summary["candidate_peak_distance"] / max(summary["baseline_peak_distance"], 1e-12)
    criteria = prereg["success_criteria"]
    checks = {
        "null_guard": abs(null_selected) <= criteria["null_selected_coefficient_abs_max"],
        "coefficient_recovery": abs(alternative_selected - true_twist) <= criteria["alternative_selected_coefficient_error_max"],
        "profile_improvement": summary["rmse_reduction"] >= criteria["alternative_rmse_reduction_min"],
        "peak_improvement": (
            summary["peak_distance_reduction"] >= criteria["alternative_peak_distance_reduction_min"]
            if "alternative_peak_distance_reduction_min" in criteria
            else summary["peak_alignment_gain"] >= criteria["alternative_peak_alignment_gain_min"]
        ),
        "null_no_harm": summary["null_rmse_increase"] <= criteria["null_candidate_rmse_increase_max"],
        "no_external_data": criteria["external_download_bytes"] == 0,
    }
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
    args = parser.parse_args()
    report = run_folding_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
