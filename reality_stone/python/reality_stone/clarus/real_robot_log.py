"""Offline gate for the small public SCITOS G5 ultrasonic sensor log."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_robot_log(path: Path, source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    if len(payload) != int(source["bytes"]):
        raise ValueError("source byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError("source sha256 mismatch")
    rows = [line.split(",") for line in payload.decode("ascii").splitlines() if line]
    features = np.asarray([[float(value) for value in row[:-1]] for row in rows], dtype=float)
    labels_text = [row[-1] for row in rows]
    names = sorted(set(labels_text))
    mapping = {name: index for index, name in enumerate(names)}
    labels = np.asarray([mapping[name] for name in labels_text], dtype=int)
    if features.shape != (int(source["rows"]), int(source["channels"])):
        raise ValueError("source shape mismatch")
    return features, labels


def _knn(
    train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.empty(len(query), dtype=int)
    margins = np.empty(len(query), dtype=float)
    neighbor_distance = np.empty(len(query), dtype=float)
    classes = int(train_y.max()) + 1
    for start in range(0, len(query), 128):
        block = query[start : start + 128]
        distance = ((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argpartition(distance, k - 1, axis=1)[:, :k]
        votes = np.stack([(train_y[neighbors] == label).sum(axis=1) for label in range(classes)], axis=1)
        order = np.sort(votes, axis=1)
        predictions[start : start + len(block)] = votes.argmax(axis=1)
        margins[start : start + len(block)] = (order[:, -1] - order[:, -2]) / k
        selected_distance = np.take_along_axis(distance, neighbors, axis=1)
        neighbor_distance[start : start + len(block)] = np.sqrt(selected_distance).mean(axis=1) / np.sqrt(train_x.shape[1])
    return predictions, margins, neighbor_distance


def _corrupt_and_reconstruct(
    features: np.ndarray, indices: np.ndarray, rng: np.random.Generator, delay_range: list[int], dropout: float,
    median: np.ndarray, lower: np.ndarray, upper: np.ndarray, trend_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.empty((len(indices), features.shape[1]))
    candidate = np.empty_like(baseline)
    uncertainty = np.empty(len(indices))
    last_valid = median.copy()
    for out_index, index in enumerate(indices):
        delay = int(rng.integers(delay_range[0], delay_range[1] + 1))
        source = max(0, index - delay)
        observed = features[source].copy()
        mask = rng.random(features.shape[1]) < dropout
        baseline[out_index] = np.where(mask, median, observed)
        start = max(1, source - trend_window + 1)
        deltas = np.diff(features[start - 1 : source + 1], axis=0)
        trend = np.median(deltas, axis=0) if len(deltas) else np.zeros(features.shape[1])
        projected = np.clip(observed + delay * trend, lower, upper)
        reconstructed = np.where(mask, last_valid, projected)
        last_valid = np.where(mask, last_valid, projected)
        candidate[out_index] = reconstructed
        uncertainty[out_index] = mask.mean() + delay / max(delay_range[1], 1) + np.mean(np.abs(trend) / (upper - lower + 1e-6))
    return baseline, candidate, uncertainty


def _selection_features(
    uncertainty: np.ndarray, margin: np.ndarray, distance: np.ndarray, prediction: np.ndarray, classes: int
) -> np.ndarray:
    one_hot = np.eye(classes)[prediction]
    return np.column_stack([uncertainty, margin, distance, one_hot])


def _fit_error_logistic(features: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-8
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    weights = np.zeros(design.shape[1])
    for iteration in range(800):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))
        gradient = design.T @ (probability - errors) / len(errors)
        weights -= 0.2 / (1.0 + iteration / 400.0) * gradient
    return weights, mean, scale


def _error_probability(model: tuple[np.ndarray, np.ndarray, np.ndarray], features: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))


def run_real_robot_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[2]
    features, labels = load_robot_log(root / prereg["source"]["relative_path"], prereg["source"])
    count = len(features)
    train_stop = int(count * prereg["split"]["train_fraction"])
    validation_stop = int(count * (prereg["split"]["train_fraction"] + prereg["split"]["validation_fraction"]))
    validation_midpoint = (train_stop + validation_stop) // 2
    learned_selection = prereg["model"].get("selection_model") == "prior_logistic"
    if split == "validation":
        indices = np.arange(validation_midpoint if learned_selection else train_stop, validation_stop)
        selection_indices = np.arange(train_stop, validation_midpoint)
    else:
        indices = np.arange(validation_stop, count)
        selection_indices = np.arange(train_stop, validation_stop)
    train_x, train_y = features[:train_stop], labels[:train_stop]
    median, lower, upper = np.median(train_x, axis=0), train_x.min(axis=0), train_x.max(axis=0)
    scale = train_x.std(axis=0) + 1e-6
    seeds = prereg[f"{split}_seeds"]
    results = []
    for seed in seeds:
        faults = prereg["faults"]
        baseline, candidate, uncertainty = _corrupt_and_reconstruct(
            features, indices, np.random.default_rng(seed), faults[f"{split}_delay_range"],
            float(faults[f"{split}_dropout_probability"]), median, lower, upper, int(prereg["model"]["trend_window"]),
        )
        base_prediction, _, _ = _knn((train_x - median) / scale, train_y, (baseline - median) / scale, int(prereg["model"]["knn_k"]))
        candidate_prediction, margin, neighbor_distance = _knn((train_x - median) / scale, train_y, (candidate - median) / scale, int(prereg["model"]["knn_k"]))
        truth = labels[indices]
        coverage = float(prereg["model"]["accepted_coverage"])
        if learned_selection:
            _, selection_candidate, selection_uncertainty = _corrupt_and_reconstruct(
                features, selection_indices, np.random.default_rng(seed + 100000), faults[f"{split}_delay_range"],
                float(faults[f"{split}_dropout_probability"]), median, lower, upper, int(prereg["model"]["trend_window"]),
            )
            selection_prediction, selection_margin, selection_distance = _knn(
                (train_x - median) / scale, train_y, (selection_candidate - median) / scale, int(prereg["model"]["knn_k"])
            )
            selection_features = _selection_features(
                selection_uncertainty, selection_margin, selection_distance, selection_prediction, int(labels.max()) + 1
            )
            selection_errors = (selection_prediction != labels[selection_indices]).astype(float)
            selector = _fit_error_logistic(selection_features, selection_errors)
            score = _error_probability(
                selector, _selection_features(uncertainty, margin, neighbor_distance, candidate_prediction, int(labels.max()) + 1)
            )
        else:
            score = (
                uncertainty
                - float(prereg["model"].get("margin_weight", 0.2)) * margin
                + float(prereg["model"].get("distance_weight", 0.0)) * neighbor_distance
            )
        accepted = score <= np.quantile(score, coverage)
        results.append({
            "seed": seed,
            "baseline_accuracy": float(np.mean(base_prediction == truth)),
            "candidate_accuracy": float(np.mean(candidate_prediction == truth)),
            "accepted_error_rate": float(np.mean(candidate_prediction[accepted] != truth[accepted])),
            "coverage": float(accepted.mean()),
        })
    summary = {key: float(np.mean([item[key] for item in results])) for key in results[0] if key != "seed"}
    summary["accuracy_gain"] = summary["candidate_accuracy"] - summary["baseline_accuracy"]
    criteria = prereg["success_criteria"]
    checks = {
        "source_integrity": True,
        "candidate_accuracy": summary["candidate_accuracy"] >= criteria["candidate_accuracy_min"],
        "accuracy_gain": summary["accuracy_gain"] >= criteria["candidate_accuracy_gain_min"],
        "selective_error": summary["accepted_error_rate"] <= criteria["accepted_error_rate_max"],
        "useful_coverage": summary["coverage"] >= criteria["accepted_coverage_min"],
        "download_budget": prereg["source"]["bytes"] <= criteria["download_bytes_max"],
    }
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    return {"experiment": prereg["experiment"], "split": split, "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "summary": summary, "per_seed": results,
            "source": prereg["source"], "resource_usage": {"elapsed_seconds": elapsed, "downloaded_bytes": 954168}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_real_robot_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
