"""Low-cost active sensing gate built on the nonlinear object world."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .nonlinear_object_world import (
    LocalChartModel,
    ObjectEpisode,
    ObjectWorldConfig,
    _initial_state,
    generate_object_episode,
    match_observation,
)


PolicyName = Literal[
    "no_sensor",
    "fixed_round_robin",
    "random_hidden",
    "max_information_gain",
    "oracle_error",
]


def covariance_predict(
    covariance: np.ndarray, process_variance: float | np.ndarray
) -> np.ndarray:
    variance = np.asarray(process_variance, dtype=float)
    if np.any(variance < 0.0):
        raise ValueError("process_variance must be nonnegative")
    return np.asarray(covariance, dtype=float) + variance


def covariance_observe(covariance: np.ndarray, observation_variance: float) -> np.ndarray:
    if observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    value = np.asarray(covariance, dtype=float)
    return value * observation_variance / (value + observation_variance)


def information_gain(covariance: np.ndarray, observation_variance: float) -> np.ndarray:
    if observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    return 0.5 * np.log1p(np.asarray(covariance, dtype=float) / observation_variance)


def _select_focus(
    policy: PolicyName,
    hidden: np.ndarray,
    covariance: np.ndarray,
    estimate: np.ndarray,
    oracle: np.ndarray,
    time_index: int,
    query_interval: int,
    rng: np.random.Generator,
) -> int | None:
    candidates = np.flatnonzero(hidden)
    if policy == "no_sensor" or not len(candidates):
        return None
    if policy == "fixed_round_robin":
        return int((time_index // query_interval) % len(hidden))
    if policy == "random_hidden":
        return int(rng.choice(candidates))
    if policy == "max_information_gain":
        return int(candidates[np.argmax(covariance[candidates])])
    if policy == "oracle_error":
        errors = np.linalg.norm(estimate[candidates, :4] - oracle[candidates, :4], axis=1)
        return int(candidates[np.argmax(errors)])
    raise ValueError(f"unknown policy: {policy}")


def evaluate_policy(
    model: LocalChartModel,
    episodes: Sequence[ObjectEpisode],
    policy: PolicyName,
    *,
    query_interval: int,
    process_variance: float,
    observation_variance: float,
    query_cost: float,
    mass_scaled_covariance: bool = False,
    random_seed: int = 9100,
) -> dict:
    if query_interval <= 0:
        raise ValueError("query_interval must be positive")
    rng = np.random.default_rng(random_seed)
    episode_costs: list[float] = []
    episode_task_losses: list[float] = []
    episode_query_rates: list[float] = []
    for episode in episodes:
        template = _initial_state(episode)
        estimate = template.copy()
        covariance = np.full(len(template), observation_variance)
        process_noise = np.full(len(template), process_variance)
        if mass_scaled_covariance:
            process_noise /= template[:, 5] ** 2
        task_losses: list[float] = []
        queries = 0
        for time_index, action in enumerate(episode.actions):
            matched, visible = match_observation(episode.observations[time_index], template)
            estimate[visible, :4] = matched[visible, :4]
            covariance[visible] = covariance_observe(
                covariance[visible], observation_variance
            )
            hidden = ~visible
            if time_index % query_interval == 0 and policy != "no_sensor":
                focus = _select_focus(
                    policy,
                    hidden,
                    covariance,
                    estimate,
                    episode.states[time_index],
                    time_index,
                    query_interval,
                    rng,
                )
                if focus is not None:
                    queries += 1
                    if hidden[focus]:
                        estimate[focus, :4] = episode.states[time_index, focus, :4]
                        covariance[focus] = covariance_observe(
                            covariance[focus], observation_variance
                        )
            if np.any(hidden):
                error = estimate[hidden, :4] - episode.states[time_index, hidden, :4]
                task_losses.append(float(np.mean(error**2)))
            else:
                task_losses.append(0.0)
            estimate = model.step(estimate, action)
            covariance = covariance_predict(covariance, process_noise)
        task_loss = float(np.mean(task_losses))
        query_rate = queries / len(episode.actions)
        episode_task_losses.append(task_loss)
        episode_query_rates.append(query_rate)
        episode_costs.append(task_loss + query_cost * query_rate)
    return {
        "mean_cost": float(np.mean(episode_costs)),
        "mean_task_loss": float(np.mean(episode_task_losses)),
        "mean_query_rate": float(np.mean(episode_query_rates)),
        "seed_costs": episode_costs,
    }


def _load_registration(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _training_episodes(config: ObjectWorldConfig) -> list[ObjectEpisode]:
    return [
        generate_object_episode(seed, objects=2 + seed % 2, config=config)
        for seed in range(1000, 1020)
    ]


def run_active_perception_gate(config_path: Path, *, split: str = "test") -> dict:
    started = time.perf_counter()
    registration, config_hash = _load_registration(config_path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    settings = registration["world"]
    limits = registration["resource_limits"]
    config = ObjectWorldConfig()
    model = LocalChartModel(config)
    train = _training_episodes(config)
    model.fit(train)
    seeds = registration["splits"][f"{split}_seeds"]
    episodes = [
        generate_object_episode(
            seed,
            objects=settings["objects"],
            occlusion=tuple(settings["occlusion_steps"]),
            velocity_process_noise_std=settings["velocity_process_noise_std"],
            mass_scaled_velocity_noise=settings.get("mass_scaled_process_noise", False),
            config=config,
        )
        for seed in seeds
    ]
    metrics = {
        policy: evaluate_policy(
            model,
            episodes,
            policy,
            query_interval=settings["query_interval"],
            process_variance=settings["process_variance"],
            observation_variance=settings["observation_variance"],
            query_cost=settings["query_cost"],
            mass_scaled_covariance=settings.get("mass_scaled_covariance", False),
            random_seed=9100,
        )
        for policy in registration["policies"]
    }
    active = metrics["max_information_gain"]
    no_sensor = metrics["no_sensor"]
    random_hidden = metrics["random_hidden"]
    fixed = metrics["fixed_round_robin"]
    gate = registration["gate"]
    wins = sum(
        left < right
        for left, right in zip(active["seed_costs"], random_hidden["seed_costs"])
    )
    paired_improvements = np.asarray(random_hidden["seed_costs"]) - np.asarray(
        active["seed_costs"]
    )
    paired_mean = float(np.mean(paired_improvements))
    paired_standard_error = float(
        np.std(paired_improvements, ddof=1) / np.sqrt(len(paired_improvements))
    )
    paired_ci95_lower = paired_mean - 1.96 * paired_standard_error
    if "minimum_seed_wins_vs_random_out_of_5" in gate:
        robustness_passed = wins >= gate["minimum_seed_wins_vs_random_out_of_5"]
    else:
        robustness_passed = (
            paired_ci95_lower >= gate["paired_mean_improvement_ci95_lower_min"]
        )
    passed = bool(
        active["mean_cost"]
        <= no_sensor["mean_cost"] * (1.0 - gate["cost_reduction_vs_no_sensor"])
        and active["mean_cost"]
        <= random_hidden["mean_cost"]
        * (1.0 - gate["cost_reduction_vs_random_hidden"])
        and active["mean_cost"]
        <= fixed["mean_cost"]
        * (1.0 - gate["cost_reduction_vs_fixed_round_robin"])
        and robustness_passed
        and active["mean_query_rate"] <= gate["max_query_rate"]
    )
    elapsed = time.perf_counter() - started
    resource_passed = bool(
        len(train) <= limits["max_training_episodes"]
        and len(episodes)
        <= limits[f"max_{split}_episodes"]
        and elapsed <= limits["max_cpu_seconds_target"]
        and not limits["write_trajectory_files"]
    )
    return {
        "experiment": registration["experiment"],
        "split": split,
        "config_sha256": config_hash,
        "resource_usage": {
            "external_download_bytes": 0,
            "backend": "numpy",
            "training_episodes": len(train),
            "evaluation_episodes": len(episodes),
            "trajectory_files_written": 0,
            "elapsed_cpu_wall_seconds": elapsed,
        },
        "policies": metrics,
        "seed_wins_vs_random": wins,
        "paired_improvement": {
            "mean": paired_mean,
            "standard_error": paired_standard_error,
            "ci95_lower": paired_ci95_lower,
        },
        "performance_passed": passed,
        "resource_passed": resource_passed,
        "passed": passed and resource_passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/preregistration/active_perception_v1.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_active_perception_gate(args.config, split=args.split)
    if args.output is None:
        experiment = json.loads(args.config.read_text(encoding="utf-8"))["experiment"]
        output = Path(f"artifacts/agi/{experiment}_{args.split}.json")
    else:
        output = args.output
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
