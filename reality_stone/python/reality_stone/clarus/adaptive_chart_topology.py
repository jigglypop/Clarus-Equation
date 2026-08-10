"""Online chart growth, reuse, merge, and local repair gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .compositional_causal_world import (
    ForceCoefficients,
    LocalBasisModel,
    _episode,
    _rmse,
    fit_local_coefficients,
)
from .nonlinear_object_world import ObjectWorldConfig, rollout


@dataclass
class Chart:
    center: np.ndarray
    count: int = 1


class AdaptiveChartBank:
    def __init__(self, scale: Sequence[float], create: float, merge: float, maximum: int) -> None:
        self.scale = np.asarray(scale, dtype=float)
        self.create = create
        self.merge = merge
        self.maximum = maximum
        self.charts: list[Chart] = []

    def distance(self, value: np.ndarray, center: np.ndarray) -> float:
        return float(np.linalg.norm((value - center) / self.scale))

    def assign(self, value: np.ndarray) -> tuple[int, float]:
        distances = [self.distance(value, chart.center) for chart in self.charts]
        index = int(np.argmin(distances))
        return index, distances[index]

    def observe(self, value: np.ndarray) -> int:
        if not self.charts:
            self.charts.append(Chart(value.copy()))
            return 0
        index, distance = self.assign(value)
        if distance > self.create and len(self.charts) < self.maximum:
            self.charts.append(Chart(value.copy()))
            index = len(self.charts) - 1
        else:
            chart = self.charts[index]
            chart.center = (chart.center * chart.count + value) / (chart.count + 1)
            chart.count += 1
        self._merge()
        return min(index, len(self.charts) - 1)

    def _merge(self) -> None:
        for left in range(len(self.charts)):
            for right in range(left + 1, len(self.charts)):
                if self.distance(self.charts[left].center, self.charts[right].center) < self.merge:
                    a, b = self.charts[left], self.charts[right]
                    a.center = (a.center * a.count + b.center * b.count) / (a.count + b.count)
                    a.count += b.count
                    self.charts.pop(right)
                    return


def _fit(episode, cfg: dict) -> np.ndarray:
    return fit_local_coefficients(
        [episode], cfg["calibration_steps"],
        robust_trim_quantile=cfg["robust_trim_quantile"],
        coefficient_bounds=cfg["coefficient_bounds"],
    ).array()


def _make_episode(seed: int, label: str, cfg: dict):
    return _episode(
        seed, cfg["regimes"][label][:4], objects=4,
        noise=cfg["velocity_process_noise_std"],
        probe_steps=cfg["calibration_probe_steps"],
        visible_prefix_steps=cfg["visible_prefix_steps"],
    )


def _load_config(config_path: Path) -> tuple[dict, bytes]:
    raw = config_path.read_bytes()
    cfg = json.loads(raw)
    if "extends" not in cfg:
        return cfg, raw
    base, base_raw = _load_config(config_path.parent / cfg["extends"])
    merged = dict(base)
    merged.update(cfg.get("overrides", {}))
    for key in ("schema_version", "status", "registered_on", "supersedes", "change_reason", "experiment"):
        if key in cfg:
            merged[key] = cfg[key]
    return merged, base_raw + raw


def run_chart_topology_gate(config_path: Path) -> dict:
    started = time.perf_counter()
    cfg, raw = _load_config(config_path)
    bank = AdaptiveChartBank(
        cfg["coefficient_scale"], cfg["creation_threshold"],
        cfg["merge_threshold"], cfg["max_charts"],
    )
    fitted_train = []
    for seed, label in zip(cfg["train"]["seeds"], cfg["train"]["labels"]):
        value = _fit(_make_episode(seed, label, cfg), cfg)
        fitted_train.append(value)
        bank.observe(value)
    pooled = np.mean(fitted_train, axis=0)
    truths = {key: np.asarray(value) for key, value in cfg["regimes"].items()}
    chart_labels = [min(truths, key=lambda key: bank.distance(c.center, truths[key])) for c in bank.charts]
    correct = 0
    chart_errors, pooled_errors = [], []
    for seed, label in zip(cfg["test"]["seeds"], cfg["test"]["labels"]):
        episode = _make_episode(seed, label, cfg)
        fitted = _fit(episode, cfg)
        index, _ = bank.assign(fitted)
        correct += chart_labels[index] == label
        start = cfg["calibration_steps"]
        actions = episode.actions[start : start + 100]
        target = episode.states[start + 100, :, :4]
        chart_model = LocalBasisModel(ObjectWorldConfig(), ForceCoefficients(*bank.charts[index].center))
        pooled_model = LocalBasisModel(ObjectWorldConfig(), ForceCoefficients(*pooled))
        chart_errors.append(_rmse(target, rollout(chart_model, episode.states[start], actions)[100, :, :4]))
        pooled_errors.append(_rmse(target, rollout(pooled_model, episode.states[start], actions)[100, :, :4]))
    accuracy = correct / len(cfg["test"]["seeds"])
    before_centers = [chart.center.copy() for chart in bank.charts]
    corrupt_index = 0
    true_label = chart_labels[corrupt_index]
    bank.charts[corrupt_index].center += np.asarray(cfg["corruption"])
    before_distance = bank.distance(bank.charts[corrupt_index].center, truths[true_label])
    repair_value = _fit(_make_episode(17000, true_label, cfg), cfg)
    bank.charts[corrupt_index].center = repair_value
    after_distance = bank.distance(bank.charts[corrupt_index].center, truths[true_label])
    untouched = max(
        (float(np.linalg.norm(bank.charts[i].center - before_centers[i]))
         for i in range(1, len(bank.charts))), default=0.0
    )
    reduction = 1.0 - float(np.mean(chart_errors)) / float(np.mean(pooled_errors))
    repair_reduction = 1.0 - after_distance / before_distance
    gate = cfg["gate"]
    performance = bool(
        len(bank.charts) == gate["required_chart_count"]
        and accuracy >= gate["assignment_accuracy_min"]
        and reduction >= gate["rmse_reduction_vs_pooled"]
        and repair_reduction >= gate["repair_distance_reduction_min"]
        and untouched <= gate["untouched_chart_change_max"]
    )
    elapsed = time.perf_counter() - started
    return {
        "experiment": cfg["experiment"], "config_sha256": hashlib.sha256(raw).hexdigest(),
        "chart_count": len(bank.charts), "chart_labels": chart_labels,
        "chart_centers": [chart.center.tolist() for chart in bank.charts],
        "assignment_accuracy": accuracy, "mean_chart_rmse": float(np.mean(chart_errors)),
        "mean_pooled_rmse": float(np.mean(pooled_errors)), "rmse_reduction": reduction,
        "repair_distance_before": before_distance, "repair_distance_after": after_distance,
        "repair_distance_reduction": repair_reduction, "untouched_chart_change": untouched,
        "resource_usage": {"external_download_bytes": 0, "trajectory_files_written": 0, "elapsed_wall_seconds": elapsed},
        "performance_passed": performance, "passed": performance and elapsed <= 15.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/preregistration/adaptive_chart_topology_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/agi/adaptive_chart_topology_v1.json"))
    args = parser.parse_args(argv)
    report = run_chart_topology_gate(args.config)
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
