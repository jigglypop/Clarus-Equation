"""Low-cost compositional causal OOD gate for shared local force laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .nonlinear_object_world import (
    MASS,
    GlobalAccelerationModel,
    ObjectEpisode,
    ObjectWorldConfig,
    _initial_state,
    generate_object_episode,
    match_observation,
    physics_step,
    rollout,
)


@dataclass(frozen=True)
class ForceCoefficients:
    linear: float
    cubic: float
    drag: float
    swirl: float
    action_gain: float = 1.0

    def array(self) -> np.ndarray:
        return np.array(
            [self.linear, self.cubic, self.drag, self.swirl, self.action_gain]
        )


class LocalBasisModel:
    name = "adaptive_local_basis"

    def __init__(self, base_config: ObjectWorldConfig, coefficients: ForceCoefficients) -> None:
        self.base_config = base_config
        self.coefficients = coefficients

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        config = ObjectWorldConfig(
            dt=self.base_config.dt,
            steps=self.base_config.steps,
            bounds=self.base_config.bounds,
            nonlinear_strength=self.coefficients.cubic,
            linear_strength=self.coefficients.linear,
            drag=self.coefficients.drag,
            swirl_strength=self.coefficients.swirl,
            restitution=self.base_config.restitution,
            max_objects=self.base_config.max_objects,
        )
        result, _ = physics_step(
            state,
            np.asarray(action) * self.coefficients.action_gain,
            config,
        )
        return result


def fit_local_coefficients(
    episodes: Sequence[ObjectEpisode],
    calibration_steps: int,
    *,
    robust_trim_quantile: float = 1.0,
    coefficient_bounds: Sequence[Sequence[float]] | None = None,
    robust_group_trim: bool = False,
) -> ForceCoefficients:
    design_rows: list[np.ndarray] = []
    targets: list[float] = []
    groups: list[int] = []
    group_index = 0
    dt = 0.02
    for episode in episodes:
        template = _initial_state(episode)
        previous, previous_visible = match_observation(episode.observations[0], template)
        for time_index in range(calibration_steps):
            current, current_visible = match_observation(
                episode.observations[time_index + 1], template
            )
            for index in np.flatnonzero(previous_visible & current_visible):
                position = previous[index, :2]
                velocity = previous[index, 2:4]
                rotated = np.array([-position[1], position[0]])
                action_feature = episode.actions[time_index] / previous[index, MASS]
                acceleration = (current[index, 2:4] - velocity) / dt
                for axis in range(2):
                    design_rows.append(
                        np.array(
                            [
                                -position[axis],
                                -position[axis] * float(position @ position),
                                -velocity[axis],
                                rotated[axis],
                                action_feature[axis],
                            ]
                        )
                    )
                    targets.append(float(acceleration[axis]))
                    groups.append(group_index)
                group_index += 1
            previous, previous_visible = current, current_visible
    design = np.asarray(design_rows)
    target = np.asarray(targets)
    keep = np.ones(len(target), dtype=bool)
    coefficients = np.zeros(design.shape[1])
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        if robust_trim_quantile >= 1.0:
            break
        residual = np.abs(target - design @ coefficients)
        if robust_group_trim:
            group_values = np.zeros(group_index)
            np.maximum.at(group_values, np.asarray(groups), residual)
            group_keep = group_values <= np.quantile(group_values, robust_trim_quantile)
            keep = group_keep[np.asarray(groups)]
        else:
            keep = residual <= np.quantile(residual, robust_trim_quantile)
    if coefficient_bounds is not None:
        bounds = np.asarray(coefficient_bounds, dtype=float)
        coefficients = np.clip(coefficients, bounds[:, 0], bounds[:, 1])
    return ForceCoefficients(*map(float, coefficients))


def _episode(
    seed: int,
    coefficients: Sequence[float],
    *,
    objects: int,
    noise: float,
    probe_steps: int = 0,
    visible_prefix_steps: int = 10,
) -> ObjectEpisode:
    linear, cubic, drag, swirl = map(float, coefficients)
    config = ObjectWorldConfig(
        linear_strength=linear,
        nonlinear_strength=cubic,
        drag=drag,
        swirl_strength=swirl,
    )
    return generate_object_episode(
        seed,
        objects=objects,
        occlusion=(6, 14),
        velocity_process_noise_std=noise,
        calibration_probe_steps=probe_steps,
        visible_prefix_steps=visible_prefix_steps,
        config=config,
    )


def _adaptive_monolithic(
    episode: ObjectEpisode, calibration_steps: int, config: ObjectWorldConfig
) -> GlobalAccelerationModel:
    truncated = ObjectEpisode(
        states=episode.states[: calibration_steps + 1],
        observations=episode.observations[: calibration_steps + 1],
        actions=episode.actions[:calibration_steps],
        visibility=episode.visibility[: calibration_steps + 1],
        collision=episode.collision[:calibration_steps],
    )
    model = GlobalAccelerationModel(config, nonlinear=True)
    model.fit([truncated])
    model.name = "adaptive_monolithic"
    return model


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _paired_ci95(improvements: Sequence[float]) -> float:
    values = np.asarray(improvements, dtype=float)
    return float(np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def evaluate_split(
    episodes: Sequence[ObjectEpisode],
    true_coefficients: Sequence[Sequence[float]],
    pooled: ForceCoefficients,
    calibration_steps: int,
    horizons: Sequence[int],
    robust_trim_quantile: float = 1.0,
    coefficient_bounds: Sequence[Sequence[float]] | None = None,
    robust_group_trim: bool = False,
) -> dict:
    base = ObjectWorldConfig()
    names = ("pooled_local", "adaptive_monolithic", "adaptive_local_basis", "oracle_coefficients")
    metrics = {name: {str(h): [] for h in horizons} for name in names}
    coefficient_errors: list[float] = []
    intervention_errors: list[float] = []
    for episode, truth_values in zip(episodes, true_coefficients):
        fitted = fit_local_coefficients(
            [episode],
            calibration_steps,
            robust_trim_quantile=robust_trim_quantile,
            coefficient_bounds=coefficient_bounds,
            robust_group_trim=robust_group_trim,
        )
        truth = ForceCoefficients(*map(float, truth_values), action_gain=1.0)
        models = {
            "pooled_local": LocalBasisModel(base, pooled),
            "adaptive_monolithic": _adaptive_monolithic(episode, calibration_steps, base),
            "adaptive_local_basis": LocalBasisModel(base, fitted),
            "oracle_coefficients": LocalBasisModel(base, truth),
        }
        initial = episode.states[calibration_steps].copy()
        actions = episode.actions[calibration_steps : calibration_steps + max(horizons)]
        for name, model in models.items():
            prediction = rollout(model, initial, actions)
            for horizon in horizons:
                target = episode.states[calibration_steps + horizon, :, :4]
                metrics[name][str(horizon)].append(
                    _rmse(target, prediction[horizon, :, :4])
                )
        relative = np.abs(fitted.array() - truth.array()) / np.maximum(
            np.abs(truth.array()), 0.05
        )
        coefficient_errors.append(float(np.median(relative)))
        state = initial
        action_a = np.array([0.18, -0.12])
        action_b = np.array([-0.14, 0.16])
        predicted_effect = models["adaptive_local_basis"].step(
            state, action_a
        )[:, 2:4] - models["adaptive_local_basis"].step(state, action_b)[:, 2:4]
        true_effect = models["oracle_coefficients"].step(
            state, action_a
        )[:, 2:4] - models["oracle_coefficients"].step(state, action_b)[:, 2:4]
        intervention_errors.append(_rmse(true_effect, predicted_effect))
    summarized = {
        name: {
            "mean_rmse": {key: float(np.mean(values)) for key, values in data.items()},
            "seed_rmse_100": data[str(max(horizons))],
        }
        for name, data in metrics.items()
    }
    local_100 = metrics["adaptive_local_basis"][str(max(horizons))]
    pooled_improvement = np.asarray(metrics["pooled_local"][str(max(horizons))]) - local_100
    mono_improvement = (
        np.asarray(metrics["adaptive_monolithic"][str(max(horizons))]) - local_100
    )
    return {
        "models": summarized,
        "median_coefficient_relative_error": float(np.median(coefficient_errors)),
        "mean_intervention_effect_error": float(np.mean(intervention_errors)),
        "paired_ci95_lower_vs_pooled": _paired_ci95(pooled_improvement),
        "paired_ci95_lower_vs_monolithic": _paired_ci95(mono_improvement),
    }


def _load_registration(config_path: Path) -> tuple[dict, bytes]:
    raw = config_path.read_bytes()
    registration = json.loads(raw)
    if "extends" not in registration:
        return registration, raw
    base_path = config_path.parent / registration["extends"]
    base, base_raw = _load_registration(base_path)
    merged = dict(base)
    merged.update(registration.get("overrides", {}))
    for key in ("schema_version", "status", "registered_on", "supersedes", "change_reason", "experiment"):
        if key in registration:
            merged[key] = registration[key]
    return merged, base_raw + raw


def _generated_coefficients(seeds: Sequence[int], specification: dict) -> list[list[float]]:
    keys = ("linear", "cubic", "drag", "swirl")
    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed + specification["rng_offset"])
        result.append(
            [float(rng.uniform(*specification[key])) for key in keys]
        )
    return result


def run_compositional_gate(config_path: Path, *, split: str = "test") -> dict:
    started = time.perf_counter()
    registration, raw = _load_registration(config_path)
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    noise = registration["velocity_process_noise_std"]
    probe_steps = registration.get("calibration_probe_steps", 0)
    visible_prefix_steps = registration.get("visible_prefix_steps", 10)
    train_episodes = [
        _episode(
            12000 + index,
            coefficients,
            objects=2 + index % 2,
            noise=noise,
            probe_steps=probe_steps,
            visible_prefix_steps=visible_prefix_steps,
        )
        for index, coefficients in enumerate(registration["train"])
    ]
    pooled = fit_local_coefficients(
        train_episodes,
        registration["calibration_steps"],
        robust_trim_quantile=registration.get("robust_trim_quantile", 1.0),
        coefficient_bounds=registration.get("coefficient_bounds"),
        robust_group_trim=registration.get("robust_group_trim", False),
    )
    split_config = registration[split]
    split_coefficients = split_config.get("coefficients")
    if "coefficient_generator" in registration:
        split_coefficients = _generated_coefficients(
            split_config["seeds"], registration["coefficient_generator"]
        )
    if split_coefficients is None:
        raise ValueError("split must provide coefficients or coefficient_generator")
    episodes = [
        _episode(
            seed,
            coefficients,
            objects=4,
            noise=noise,
            probe_steps=probe_steps,
            visible_prefix_steps=visible_prefix_steps,
        )
        for seed, coefficients in zip(split_config["seeds"], split_coefficients)
    ]
    metrics = evaluate_split(
        episodes,
        split_coefficients,
        pooled,
        registration["calibration_steps"],
        registration["rollout_horizons"],
        robust_trim_quantile=registration.get("robust_trim_quantile", 1.0),
        coefficient_bounds=registration.get("coefficient_bounds"),
        robust_group_trim=registration.get("robust_group_trim", False),
    )
    gate = registration["gate"]
    local = metrics["models"]["adaptive_local_basis"]["mean_rmse"]["100"]
    pooled_rmse = metrics["models"]["pooled_local"]["mean_rmse"]["100"]
    monolithic = metrics["models"]["adaptive_monolithic"]["mean_rmse"]["100"]
    performance_passed = bool(
        local <= pooled_rmse * (1.0 - gate["rmse_reduction_vs_pooled"])
        and local <= monolithic * (1.0 - gate["rmse_reduction_vs_monolithic"])
        and metrics["paired_ci95_lower_vs_pooled"] > gate["paired_ci95_lower_min"]
        and metrics["paired_ci95_lower_vs_monolithic"] > gate["paired_ci95_lower_min"]
        and metrics["mean_intervention_effect_error"]
        <= gate["intervention_effect_error_max"]
        and metrics["median_coefficient_relative_error"]
        <= gate["median_coefficient_relative_error_max"]
    )
    elapsed = time.perf_counter() - started
    limits = registration["resource_limits"]
    resource_passed = bool(
        elapsed <= limits["max_cpu_seconds_target"]
        and limits["external_download_bytes"] == 0
        and not limits["write_trajectory_files"]
    )
    return {
        "experiment": registration["experiment"],
        "split": split,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "resource_usage": {
            "external_download_bytes": 0,
            "trajectory_files_written": 0,
            "elapsed_wall_seconds": elapsed,
            "training_episodes": len(train_episodes),
            "evaluation_episodes": len(episodes),
        },
        "pooled_coefficients": pooled.array().tolist(),
        **metrics,
        "performance_passed": performance_passed,
        "resource_passed": resource_passed,
        "passed": performance_passed and resource_passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/preregistration/compositional_causal_ood_v1.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_compositional_gate(args.config, split=args.split)
    output = args.output or Path(f"artifacts/agi/compositional_causal_{args.split}_v1.json")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
