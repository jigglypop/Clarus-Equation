"""NumPy-only calibration gate for short-horizon actuator-fault risk."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _make_windows(count: int, seed: int, spec: dict[str, Any], ood: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gain_range = spec["test_gain_range"] if ood else spec["train_gain_range"]
    delay_range = spec["test_delay_range"] if ood else spec["train_delay_range"]
    flip_probability = spec["test_flip_probability"] if ood else spec["train_flip_probability"]
    dt = float(spec["dt"])
    horizon = int(spec["horizon"])
    bound = float(spec["safe_bound"])
    observed_x = rng.uniform(-0.98, 0.98, count)
    observed_v = rng.uniform(-0.8, 0.8, count)
    target = rng.choice(np.array([-0.72, 0.72]), count)
    command = np.clip(6.0 * (target - observed_x) - 2.2 * observed_v, -1.0, 1.0)
    gain = rng.uniform(float(gain_range[0]), float(gain_range[1]), count)
    delay = rng.integers(int(delay_range[0]), int(delay_range[1]) + 1, count)
    dropout = rng.random(count) < (0.04 + 0.025 * delay)
    sign = np.where(rng.random(count) < float(flip_probability), -1.0, 1.0)
    uncertainty = 0.006 + 0.012 * delay + 0.07 * dropout
    true_x = observed_x + rng.normal(0.0, uncertainty)
    true_v = observed_v + rng.normal(0.0, 1.4 * uncertainty)
    innovation = np.abs((gain * sign - 1.0) * command) * dt + np.abs(rng.normal(0.0, 0.006, count))

    violated = np.zeros(count, dtype=bool)
    position = true_x.copy()
    velocity = true_v.copy()
    for _ in range(horizon):
        velocity = 0.992 * velocity + dt * gain * sign * command
        position = position + dt * velocity
        violated |= np.abs(position) >= bound

    outward_velocity = np.sign(observed_x) * observed_v
    features = np.column_stack(
        [np.abs(observed_x) / bound, outward_velocity, np.abs(command), uncertainty, innovation]
    )
    return features, violated.astype(float)


def _fit_logistic(
    features: np.ndarray, labels: np.ndarray, class_weighted: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-8
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    weights = np.zeros(design.shape[1])
    positive_weight = max(1.0, (len(labels) - labels.sum()) / max(labels.sum(), 1.0)) if class_weighted else 1.0
    sample_weight = np.where(labels > 0.5, positive_weight, 1.0)
    for iteration in range(1400):
        probability = _sigmoid(design @ weights)
        gradient = design.T @ ((probability - labels) * sample_weight) / sample_weight.sum()
        gradient[1:] += 2e-4 * weights[1:]
        weights -= 0.18 / (1.0 + iteration / 700.0) * gradient
    return weights, mean, scale


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], features: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    return _sigmoid(design @ weights)


def _candidate_features(features: np.ndarray, version: int) -> np.ndarray:
    if version < 2:
        return features
    position, outward_velocity, command, uncertainty, innovation = features.T
    outward_risk = position + np.maximum(outward_velocity, 0.0) + 0.25 * command
    return np.column_stack(
        [
            features,
            innovation * outward_risk,
            uncertainty * position,
            position**2,
        ]
    )


def _threshold(probability: np.ndarray, labels: np.ndarray, risk_limit: float = 0.01) -> float:
    order = np.argsort(probability)
    sorted_labels = labels[order]
    cumulative_risk = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    eligible = np.flatnonzero(cumulative_risk <= risk_limit)
    if not len(eligible):
        return 0.0
    final = int(eligible[-1])
    return float(np.nextafter(probability[order[final]], np.inf))


def _ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right if right < 1.0 else probability <= right)
        if np.any(mask):
            result += mask.mean() * abs(float(probability[mask].mean() - labels[mask].mean()))
    return float(result)


def _metrics(probability: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    accepted = probability < threshold
    accepted_count = int(accepted.sum())
    failures = float(labels[accepted].sum()) if accepted_count else 1.0
    return {
        "brier": float(np.mean((probability - labels) ** 2)),
        "ece": _ece(probability, labels),
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accepted_violation_rate": float(labels[accepted].mean()) if np.any(accepted) else 1.0,
        "accepted_count": float(accepted_count),
        "accepted_failures": failures,
        "prevalence": float(labels.mean()),
    }


def run_fault_ood_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    spec = prereg["data"]
    class_weighted = bool(prereg.get("model", {}).get("class_weighted", True))
    feature_version = int(prereg.get("model", {}).get("candidate_feature_version", 1))
    calibration_risk_limit = float(prereg.get("calibration_risk_limit", 0.01))
    seeds = prereg[f"{split}_seeds"]
    per_seed = []
    for seed in seeds:
        train_x, train_y = _make_windows(int(spec["train_samples"]), seed, spec, False)
        cal_x, cal_y = _make_windows(int(spec["calibration_samples"]), seed + 101, spec, False)
        test_x, test_y = _make_windows(int(spec["test_samples"]), seed + 202, spec, True)
        baseline_model = _fit_logistic(train_x[:, :3], train_y, class_weighted)
        train_candidate = _candidate_features(train_x, feature_version)
        cal_candidate = _candidate_features(cal_x, feature_version)
        test_candidate = _candidate_features(test_x, feature_version)
        candidate_model = _fit_logistic(train_candidate, train_y, class_weighted)
        baseline_cal = _predict(baseline_model, cal_x[:, :3])
        candidate_cal = _predict(candidate_model, cal_candidate)
        baseline_threshold = _threshold(baseline_cal, cal_y, calibration_risk_limit)
        candidate_threshold = _threshold(candidate_cal, cal_y, calibration_risk_limit)
        baseline = _metrics(_predict(baseline_model, test_x[:, :3]), test_y, baseline_threshold)
        candidate = _metrics(_predict(candidate_model, test_candidate), test_y, candidate_threshold)
        per_seed.append({"seed": seed, "baseline": baseline, "candidate": candidate})

    def avg(model: str, metric: str) -> float:
        return float(np.mean([item[model][metric] for item in per_seed]))

    summary = {
        model: {metric: avg(model, metric) for metric in per_seed[0][model]}
        for model in ("baseline", "candidate")
    }
    summary["candidate"]["worst_seed_accepted_violation_rate"] = float(
        max(item["candidate"]["accepted_violation_rate"] for item in per_seed)
    )
    summary["candidate"]["minimum_seed_coverage"] = float(
        min(item["candidate"]["coverage"] for item in per_seed)
    )
    confidence = prereg.get("confidence")
    if confidence:
        z = float(confidence["z"])
        uppers = []
        for item in per_seed:
            count = item["candidate"]["accepted_count"]
            rate = item["candidate"]["accepted_violation_rate"]
            z2 = z * z
            center = rate + z2 / (2.0 * count)
            radius = z * np.sqrt(rate * (1.0 - rate) / count + z2 / (4.0 * count * count))
            uppers.append(float((center + radius) / (1.0 + z2 / count)))
        summary["candidate"]["worst_seed_wilson_upper"] = max(uppers)
    criteria = prereg["success_criteria"]
    checks = {
        "brier_improves": summary["candidate"]["brier"] <= criteria["candidate_brier_max_fraction_of_baseline"] * summary["baseline"]["brier"],
        "calibrated": summary["candidate"]["ece"] <= criteria["candidate_ece_max"],
        "selective_safety": summary["candidate"]["accepted_violation_rate"] <= criteria["accepted_violation_rate_max"],
        "useful_coverage": summary["candidate"]["coverage"] >= criteria["accepted_coverage_min"],
        "safer_than_baseline": summary["candidate"]["accepted_violation_rate"] <= criteria["accepted_violation_rate_max_fraction_of_baseline"] * summary["baseline"]["accepted_violation_rate"],
        "no_external_data": criteria["external_download_bytes"] == 0,
    }
    if "worst_seed_wilson_upper_max" in criteria:
        checks["finite_sample_risk"] = summary["candidate"]["worst_seed_wilson_upper"] <= criteria["worst_seed_wilson_upper_max"]
    elapsed = time.perf_counter() - started
    checks["runtime"] = elapsed <= criteria["elapsed_seconds_max"]
    return {
        "experiment": prereg["experiment"], "split": split, "seeds": seeds,
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "summary": summary, "per_seed": per_seed,
        "resource_usage": {"external_download_bytes": 0, "elapsed_seconds": elapsed},
        "config": str(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_fault_ood_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
