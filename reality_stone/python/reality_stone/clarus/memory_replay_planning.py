"""Bounded prototype memory, priority replay, and macro-action planning gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .compositional_causal_world import ForceCoefficients, LocalBasisModel
from .nonlinear_object_world import ObjectWorldConfig, physics_step, rollout


@dataclass
class MemoryItem:
    center: np.ndarray
    count: int
    priority: float


class PriorityReplayMemory:
    def __init__(self, capacity: int, scale: Sequence[float], merge: float, rarity: float) -> None:
        self.capacity = capacity
        self.scale = np.asarray(scale, dtype=float)
        self.merge = merge
        self.rarity = rarity
        self.items: list[MemoryItem] = []

    def distance(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm((left - right) / self.scale))

    def observe(self, value: np.ndarray) -> None:
        if self.items:
            distances = [self.distance(value, item.center) for item in self.items]
            index = int(np.argmin(distances))
            novelty = distances[index]
            if novelty <= self.merge:
                item = self.items[index]
                item.center = (item.center * item.count + value) / (item.count + 1)
                item.count += 1
                item.priority = novelty + self.rarity / item.count
                return
        novelty = min((self.distance(value, item.center) for item in self.items), default=1.0)
        candidate = MemoryItem(value.copy(), 1, novelty + self.rarity)
        if len(self.items) < self.capacity:
            self.items.append(candidate)
        else:
            replace = int(np.argmin([item.priority for item in self.items]))
            if candidate.priority > self.items[replace].priority:
                self.items[replace] = candidate

    def replay(self) -> None:
        for item in self.items:
            item.priority += self.rarity / item.count

    def recall(self, query: np.ndarray) -> np.ndarray:
        return min(self.items, key=lambda item: self.distance(query, item.center)).center.copy()


def _recency(stream: Sequence[np.ndarray], capacity: int) -> list[np.ndarray]:
    return [value.copy() for value in stream[-capacity:]]


def _reservoir(stream: Sequence[np.ndarray], capacity: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for index, value in enumerate(stream):
        if index < capacity:
            result.append(value.copy())
        else:
            target = int(rng.integers(0, index + 1))
            if target < capacity:
                result[target] = value.copy()
    return result


def _nearest(prototypes: Sequence[np.ndarray], query: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return min(prototypes, key=lambda value: np.linalg.norm((value - query) / scale)).copy()


def _macro_plan_cost(coefficients: np.ndarray, true_coefficients: np.ndarray, cfg: dict) -> float:
    state = np.array([[0.72, -0.62, 0.0, 0.0, 0.07, 1.0, 0.5]])
    target = state[:, :2].copy() if cfg.get("planning_target") == "hold_initial" else np.zeros((1, 2))
    action_cost = cfg.get("action_cost", 0.02)
    model = LocalBasisModel(ObjectWorldConfig(), ForceCoefficients(*coefficients))
    true_config = ObjectWorldConfig(
        linear_strength=true_coefficients[0], nonlinear_strength=true_coefficients[1],
        drag=true_coefficients[2], swirl_strength=true_coefficients[3],
    )
    actions = [np.array(value) for value in itertools.product(cfg["action_values"], repeat=2)]
    total = 0.0
    for _ in range(0, cfg["planning_steps"], cfg["macro_horizon"]):
        scored = []
        for action in actions:
            predicted = rollout(model, state, np.repeat(action[None, :], cfg["macro_horizon"], axis=0))
            cost = float(np.sum((predicted[1:, :, :2] - target) ** 2) + 0.1 * np.sum(predicted[1:, :, 2:4] ** 2))
            cost += action_cost * cfg["macro_horizon"] * float(action @ action)
            scored.append(cost)
        chosen = actions[int(np.argmin(scored))]
        for _ in range(cfg["macro_horizon"]):
            state, _ = physics_step(state, chosen, true_config)
            total += float(np.sum((state[:, :2] - target) ** 2) + 0.1 * np.sum(state[:, 2:4] ** 2) + action_cost * chosen @ chosen)
    return total


def _zero_cost(true_coefficients: np.ndarray, cfg: dict) -> float:
    state = np.array([[0.72, -0.62, 0.0, 0.0, 0.07, 1.0, 0.5]])
    target = state[:, :2].copy() if cfg.get("planning_target") == "hold_initial" else np.zeros((1, 2))
    true_config = ObjectWorldConfig(
        linear_strength=true_coefficients[0], nonlinear_strength=true_coefficients[1],
        drag=true_coefficients[2], swirl_strength=true_coefficients[3],
    )
    total = 0.0
    for _ in range(cfg["planning_steps"]):
        state, _ = physics_step(state, np.zeros(2), true_config)
        total += float(np.sum((state[:, :2] - target) ** 2) + 0.1 * np.sum(state[:, 2:4] ** 2))
    return total


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


def _run_seed(cfg: dict, seed: int) -> dict:
    truths = {key: np.asarray(value, dtype=float) for key, value in cfg["regimes"].items()}
    scale = np.asarray(cfg["coefficient_scale"], dtype=float)
    rng = np.random.default_rng(seed)
    stream = [truths[label] + rng.normal(scale=0.025 * scale) for label in cfg["stream_labels"]]
    memory = PriorityReplayMemory(cfg["capacity"], scale, cfg["merge_threshold"], cfg["rarity_weight"])
    for index, value in enumerate(stream):
        memory.observe(value)
        if (index + 1) % 4 == 0:
            memory.replay()
    recency = _recency(stream, cfg["capacity"])
    reservoir = _reservoir(stream, cfg["capacity"], seed + 1)
    recalled = {label: memory.recall(value) for label, value in truths.items()}
    accuracy = sum(
        min(truths, key=lambda key: np.linalg.norm((recalled[label] - truths[key]) / scale)) == label
        for label in truths
    ) / len(truths)
    rare = truths["B"]
    priority_rare_error = float(np.linalg.norm((memory.recall(rare) - rare) / scale))
    recency_rare_error = float(np.linalg.norm((_nearest(recency, rare, scale) - rare) / scale))
    planning_cost = _macro_plan_cost(memory.recall(rare), rare, cfg)
    recency_cost = _macro_plan_cost(_nearest(recency, rare, scale), rare, cfg)
    zero_cost = _zero_cost(rare, cfg)
    prototype_bytes = sum(item.center.nbytes for item in memory.items)
    return {
        "memory_items": len(memory.items), "prototype_bytes": prototype_bytes,
        "recall_accuracy": accuracy, "priority_rare_error": priority_rare_error,
        "recency_rare_error": recency_rare_error, "planning_cost": planning_cost,
        "recency_planning_cost": recency_cost, "zero_action_cost": zero_cost,
        "reservoir_items": len(reservoir),
    }


def run_memory_replay_gate(config_path: Path, *, split: str = "validation") -> dict:
    started = time.perf_counter()
    cfg, raw = _load_config(config_path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    results = [_run_seed(cfg, seed) for seed in cfg[f"{split}_seeds"]]
    keys = ("memory_items", "prototype_bytes", "recall_accuracy", "priority_rare_error", "recency_rare_error", "planning_cost", "recency_planning_cost", "zero_action_cost")
    means = {key: float(np.mean([result[key] for result in results])) for key in keys}
    gate = cfg["gate"]
    passed = bool(
        means["recall_accuracy"] >= gate["recall_accuracy_min"]
        and means["priority_rare_error"] <= means["recency_rare_error"] * (1.0 - gate["rare_error_reduction_vs_recency"])
        and means["planning_cost"] <= means["zero_action_cost"] * (1.0 - gate["cost_reduction_vs_zero"])
        and means["planning_cost"] <= means["recency_planning_cost"] * (1.0 - gate["cost_reduction_vs_recency"])
        and means["memory_items"] <= cfg["capacity"]
        and means["prototype_bytes"] <= cfg["resource_limits"]["max_prototype_bytes"]
    )
    elapsed = time.perf_counter() - started
    return {
        "experiment": cfg["experiment"], "split": split, "config_sha256": hashlib.sha256(raw).hexdigest(),
        **means, "seed_results": results,
        "resource_usage": {"external_download_bytes": 0, "trajectory_files_written": 0, "elapsed_wall_seconds": elapsed},
        "performance_passed": passed, "passed": passed and elapsed <= cfg["resource_limits"]["max_cpu_seconds_target"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/preregistration/memory_replay_planning_v1.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/agi/memory_replay_planning_v1.json"))
    args = parser.parse_args(argv)
    report = run_memory_replay_gate(args.config, split=args.split)
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
