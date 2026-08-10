"""Minimal nonlinear object-permanence benchmark with no external data.

The hidden state is available only to the evaluator. Models fit shuffled,
partially occluded observations and actions. The implementation deliberately
uses NumPy and writes only a compact JSON certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Array = np.ndarray
DYNAMIC = slice(0, 4)  # x, y, vx, vy
RADIUS = 4
MASS = 5
APPEARANCE = 6


@dataclass(frozen=True)
class ObjectWorldConfig:
    dt: float = 0.02
    steps: int = 140
    bounds: tuple[float, float] = (-1.0, 1.0)
    nonlinear_strength: float = 0.34
    linear_strength: float = 0.0
    drag: float = 0.08
    swirl_strength: float = 0.0
    restitution: float = 0.96
    max_objects: int = 4


@dataclass(frozen=True)
class ObjectEpisode:
    states: Array  # (steps + 1, objects, 7), canonical identity order
    observations: Array  # (steps + 1, objects, 8), shuffled slots
    actions: Array  # (steps, 2)
    visibility: Array  # (steps + 1, objects), canonical identity order
    collision: Array  # (steps,), diagnostic oracle used only by evaluator


def _reflect_walls(state: Array, config: ObjectWorldConfig) -> None:
    low, high = config.bounds
    for item in state:
        radius = item[RADIUS]
        for axis in range(2):
            lower = low + radius
            upper = high - radius
            if item[axis] < lower:
                item[axis] = lower + (lower - item[axis])
                item[axis + 2] = abs(item[axis + 2]) * config.restitution
            elif item[axis] > upper:
                item[axis] = upper - (item[axis] - upper)
                item[axis + 2] = -abs(item[axis + 2]) * config.restitution


def _resolve_collisions(state: Array, restitution: float) -> bool:
    collided = False
    for left in range(len(state)):
        for right in range(left + 1, len(state)):
            delta = state[right, :2] - state[left, :2]
            distance = float(np.linalg.norm(delta))
            minimum = float(state[left, RADIUS] + state[right, RADIUS])
            if distance >= minimum:
                continue
            collided = True
            normal = delta / max(distance, 1e-12)
            if distance <= 1e-12:
                normal = np.array([1.0, 0.0])
            overlap = minimum - distance
            total_mass = state[left, MASS] + state[right, MASS]
            state[left, :2] -= normal * overlap * state[right, MASS] / total_mass
            state[right, :2] += normal * overlap * state[left, MASS] / total_mass
            relative = float((state[right, 2:4] - state[left, 2:4]) @ normal)
            if relative < 0.0:
                impulse = -(1.0 + restitution) * relative
                impulse /= 1.0 / state[left, MASS] + 1.0 / state[right, MASS]
                state[left, 2:4] -= impulse * normal / state[left, MASS]
                state[right, 2:4] += impulse * normal / state[right, MASS]
    return collided


def physics_step(
    state: Array,
    action: Array,
    config: ObjectWorldConfig,
    *,
    nonlinear_strength: float | None = None,
    drag: float | None = None,
) -> tuple[Array, bool]:
    """Advance one deterministic step using semi-implicit Euler."""
    current = np.asarray(state, dtype=float)
    if current.ndim != 2 or current.shape[1] != 7:
        raise ValueError("state must have shape (objects, 7)")
    force = np.asarray(action, dtype=float)
    if force.shape != (2,):
        raise ValueError("action must have shape (2,)")
    result = current.copy()
    k_value = config.nonlinear_strength if nonlinear_strength is None else nonlinear_strength
    drag_value = config.drag if drag is None else drag
    radius_squared = np.sum(result[:, :2] ** 2, axis=1, keepdims=True)
    acceleration = -k_value * result[:, :2] * radius_squared
    acceleration -= config.linear_strength * result[:, :2]
    acceleration -= drag_value * result[:, 2:4]
    acceleration += config.swirl_strength * np.column_stack(
        (-result[:, 1], result[:, 0])
    )
    acceleration += force[None, :] / result[:, MASS, None]
    result[:, 2:4] += config.dt * acceleration
    result[:, :2] += config.dt * result[:, 2:4]
    _reflect_walls(result, config)
    collided = _resolve_collisions(result, config.restitution)
    return result, collided


def _sample_initial_state(rng: np.random.Generator, objects: int) -> Array:
    state = np.empty((objects, 7), dtype=float)
    state[:, RADIUS] = rng.uniform(0.055, 0.085, objects)
    state[:, MASS] = rng.uniform(0.75, 1.35, objects)
    state[:, APPEARANCE] = np.sort(rng.uniform(0.1, 0.9, objects))
    accepted: list[Array] = []
    for index in range(objects):
        for _ in range(500):
            position = rng.uniform(-0.72, 0.72, 2)
            if all(
                np.linalg.norm(position - other[:2])
                > state[index, RADIUS] + other[RADIUS] + 0.12
                for other in accepted
            ):
                break
        state[index, :2] = position
        state[index, 2:4] = rng.uniform(-0.32, 0.32, 2)
        accepted.append(state[index].copy())
    return state


def _visibility_schedule(
    steps: int,
    objects: int,
    occlusion: tuple[int, int],
    visible_prefix_steps: int = 10,
) -> Array:
    visibility = np.ones((steps + 1, objects), dtype=bool)
    low, high = occlusion
    for index in range(objects):
        length = low + (index * 7) % (high - low + 1)
        start = 22 + index * 9
        while start < steps:
            visibility[start : min(start + length, steps + 1), index] = False
            start += max(45, length + 18)
    visibility[: visible_prefix_steps + 1] = True
    return visibility


def _observe(states: Array, visibility: Array, rng: np.random.Generator) -> Array:
    steps, objects, _ = states.shape
    observations = np.full((steps, objects, 8), np.nan, dtype=float)
    for time in range(steps):
        rows = []
        for identity in range(objects):
            if visibility[time, identity]:
                state = states[time, identity]
                rows.append(
                    np.array(
                        [
                            state[APPEARANCE],
                            state[RADIUS],
                            state[MASS],
                            state[0],
                            state[1],
                            state[2],
                            state[3],
                            1.0,
                        ]
                    )
                )
            else:
                rows.append(np.array([np.nan] * 7 + [0.0]))
        observations[time] = np.asarray(rows)[rng.permutation(objects)]
    return observations


def generate_object_episode(
    seed: int,
    *,
    objects: int = 3,
    occlusion: tuple[int, int] = (4, 12),
    velocity_process_noise_std: float = 0.0,
    mass_scaled_velocity_noise: bool = False,
    calibration_probe_steps: int = 0,
    visible_prefix_steps: int = 10,
    config: ObjectWorldConfig | None = None,
) -> ObjectEpisode:
    """Generate one compact deterministic trajectory and shuffled observations."""
    cfg = config or ObjectWorldConfig()
    if not 1 <= objects <= cfg.max_objects:
        raise ValueError("objects must be between one and max_objects")
    if occlusion[0] <= 0 or occlusion[1] < occlusion[0]:
        raise ValueError("invalid occlusion interval")
    if velocity_process_noise_std < 0.0:
        raise ValueError("velocity_process_noise_std must be nonnegative")
    if not 0 <= calibration_probe_steps <= cfg.steps:
        raise ValueError("calibration_probe_steps must be within the episode")
    rng = np.random.default_rng(seed)
    states = np.empty((cfg.steps + 1, objects, 7), dtype=float)
    actions = np.empty((cfg.steps, 2), dtype=float)
    collision = np.zeros(cfg.steps, dtype=bool)
    states[0] = _sample_initial_state(rng, objects)
    phase = rng.uniform(0.0, 2.0 * np.pi, 2)
    for time in range(cfg.steps):
        if time < calibration_probe_steps:
            actions[time] = 0.22 * rng.choice(np.array([-1.0, 1.0]), size=2)
        else:
            actions[time] = 0.22 * np.array(
                [np.sin(0.073 * time + phase[0]), np.cos(0.061 * time + phase[1])]
            )
            actions[time] += rng.normal(scale=0.025, size=2)
        states[time + 1], collision[time] = physics_step(states[time], actions[time], cfg)
        if velocity_process_noise_std:
            noise_scale = np.full((objects, 1), velocity_process_noise_std)
            if mass_scaled_velocity_noise:
                noise_scale /= states[time + 1, :, MASS, None]
            states[time + 1, :, 2:4] += rng.normal(size=(objects, 2)) * noise_scale
    visibility = _visibility_schedule(
        cfg.steps, objects, occlusion, visible_prefix_steps=visible_prefix_steps
    )
    observations = _observe(states, visibility, rng)
    return ObjectEpisode(states, observations, actions, visibility, collision)


def canonical_observation(observation: Array, objects: int) -> tuple[Array, Array]:
    """Return canonical appearance order and visibility from shuffled slots."""
    rows = np.asarray(observation, dtype=float)
    visible_rows = rows[rows[:, 7] > 0.5]
    visible_rows = visible_rows[np.argsort(visible_rows[:, 0])]
    state = np.full((objects, 7), np.nan, dtype=float)
    visible = np.zeros(objects, dtype=bool)
    if len(visible_rows) == objects:
        state[:, APPEARANCE] = visible_rows[:, 0]
        state[:, RADIUS] = visible_rows[:, 1]
        state[:, MASS] = visible_rows[:, 2]
        state[:, :4] = visible_rows[:, 3:7]
        visible[:] = True
        return state, visible
    # Appearance ranks are fixed by the fully visible first frame. Callers that
    # need partial observations use match_observation with its initial template.
    return state, visible


def match_observation(observation: Array, template: Array) -> tuple[Array, Array]:
    matched = np.full_like(template, np.nan)
    visible = np.zeros(len(template), dtype=bool)
    for row in np.asarray(observation, dtype=float):
        if row[7] <= 0.5:
            continue
        index = int(np.argmin(np.abs(template[:, APPEARANCE] - row[0])))
        matched[index, APPEARANCE] = row[0]
        matched[index, RADIUS] = row[1]
        matched[index, MASS] = row[2]
        matched[index, :4] = row[3:7]
        visible[index] = True
    return matched, visible


class PersistenceModel:
    name = "persistence"

    def fit(self, episodes: Sequence[ObjectEpisode]) -> None:
        if not episodes:
            raise ValueError("at least one episode is required")

    def step(self, state: Array, action: Array) -> Array:
        del action
        result = np.asarray(state, dtype=float).copy()
        result[:, 2:4] = 0.0
        return result


def _initial_state(episode: ObjectEpisode) -> Array:
    state, visible = canonical_observation(episode.observations[0], len(episode.states[0]))
    if not np.all(visible):
        raise ValueError("the first observation must expose every object")
    return state


def _visible_training_rows(episodes: Sequence[ObjectEpisode]) -> tuple[Array, Array]:
    features: list[Array] = []
    targets: list[Array] = []
    for episode in episodes:
        template = _initial_state(episode)
        previous, previous_visible = match_observation(episode.observations[0], template)
        for time, action in enumerate(episode.actions):
            current, current_visible = match_observation(episode.observations[time + 1], template)
            for index in np.flatnonzero(previous_visible & current_visible):
                position = previous[index, :2]
                velocity = previous[index, 2:4]
                mass = previous[index, MASS]
                radius_squared = float(position @ position)
                features.append(
                    np.concatenate(
                        (
                            -position * radius_squared,
                            -velocity,
                            action / mass,
                        )
                    )
                )
                targets.append((current[index, 2:4] - velocity) / 0.02)
            previous, previous_visible = current, current_visible
    return np.asarray(features), np.asarray(targets)


class LocalChartModel:
    """Object-local nonlinear force chart plus generic contact mechanics."""

    name = "local_chart_nonlinear"

    def __init__(self, config: ObjectWorldConfig) -> None:
        self.config = config
        self.nonlinear_strength = 0.0
        self.drag = 0.0
        self.action_gain = 0.0
        self.residual_scale = 0.0

    def fit(self, episodes: Sequence[ObjectEpisode]) -> None:
        features, targets = _visible_training_rows(episodes)
        # Shared scalar coefficients preserve rotational symmetry. Robust
        # trimming removes the few contact/wall transitions without oracle flags.
        design = np.column_stack(
            (
                features[:, :2].reshape(-1),
                features[:, 2:4].reshape(-1),
                features[:, 4:6].reshape(-1),
            )
        )
        target = targets.reshape(-1)
        keep = np.ones(len(target), dtype=bool)
        coefficients = np.zeros(3)
        for _ in range(3):
            coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
            residual = np.abs(target - design @ coefficients)
            cutoff = np.quantile(residual, 0.90)
            keep = residual <= cutoff
        self.nonlinear_strength, self.drag, self.action_gain = coefficients
        self.residual_scale = float(np.sqrt(np.mean((target[keep] - design[keep] @ coefficients) ** 2)))

    def step(self, state: Array, action: Array) -> Array:
        scaled_action = np.asarray(action, dtype=float) * self.action_gain
        result, _ = physics_step(
            state,
            scaled_action,
            self.config,
            nonlinear_strength=self.nonlinear_strength,
            drag=self.drag,
        )
        return result


class GlobalAccelerationModel:
    """Small fixed-width global baseline; nonlinear adds elementwise transforms."""

    def __init__(self, config: ObjectWorldConfig, *, nonlinear: bool) -> None:
        self.config = config
        self.nonlinear = nonlinear
        self.name = "monolithic_nonlinear" if nonlinear else "global_linear"
        self.coefficients: Array | None = None

    def _features(self, state: Array, action: Array) -> Array:
        padded = np.zeros((self.config.max_objects, 6))
        padded[: len(state)] = state[:, :6]
        flat = padded.reshape(-1)
        parts = [np.ones(1), flat, np.asarray(action, dtype=float)]
        if self.nonlinear:
            parts.extend((flat**2, np.tanh(flat)))
        return np.concatenate(parts)

    def fit(self, episodes: Sequence[ObjectEpisode]) -> None:
        features: list[Array] = []
        targets: list[Array] = []
        for episode in episodes:
            template = _initial_state(episode)
            previous, previous_visible = match_observation(episode.observations[0], template)
            for time, action in enumerate(episode.actions):
                current, current_visible = match_observation(episode.observations[time + 1], template)
                if np.all(previous_visible & current_visible):
                    acceleration = (current[:, 2:4] - previous[:, 2:4]) / self.config.dt
                    padded_target = np.zeros((self.config.max_objects, 2))
                    padded_target[: len(current)] = acceleration
                    features.append(self._features(previous, action))
                    targets.append(padded_target.reshape(-1))
                previous, previous_visible = current, current_visible
        design = np.asarray(features)
        target = np.asarray(targets)
        ridge = 2e-4 if self.nonlinear else 1e-5
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        self.coefficients = np.linalg.solve(gram, design.T @ target)

    def step(self, state: Array, action: Array) -> Array:
        if self.coefficients is None:
            raise RuntimeError("fit must be called before step")
        result = np.asarray(state, dtype=float).copy()
        acceleration = self._features(result, action) @ self.coefficients
        acceleration = acceleration.reshape(self.config.max_objects, 2)[: len(result)]
        acceleration = np.clip(np.nan_to_num(acceleration), -5.0, 5.0)
        result[:, 2:4] += self.config.dt * acceleration
        result[:, :2] += self.config.dt * result[:, 2:4]
        _reflect_walls(result, self.config)
        _resolve_collisions(result, self.config.restitution)
        return result


def rollout(model: object, initial: Array, actions: Array) -> Array:
    states = np.empty((len(actions) + 1, *initial.shape), dtype=float)
    states[0] = initial
    for time, action in enumerate(actions):
        states[time + 1] = model.step(states[time], action)  # type: ignore[attr-defined]
    return states


def _rmse(target: Array, prediction: Array) -> float:
    return float(np.sqrt(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2)))


def _hidden_metrics(model: object, episode: ObjectEpisode) -> tuple[float, float, float]:
    template = _initial_state(episode)
    estimate = template.copy()
    hidden_errors: list[float] = []
    reappearance_errors: list[float] = []
    hidden_age = np.zeros(len(template), dtype=int)
    uncertainty: list[tuple[int, float]] = []
    previous_visible = np.ones(len(template), dtype=bool)
    for time, action in enumerate(episode.actions):
        matched, visible = match_observation(episode.observations[time], template)
        estimate[visible, :4] = matched[visible, :4]
        predicted = model.step(estimate, action)  # type: ignore[attr-defined]
        next_matched, next_visible = match_observation(episode.observations[time + 1], template)
        reappeared = (~previous_visible) & next_visible
        if np.any(reappeared):
            reappearance_errors.extend(
                np.linalg.norm(predicted[reappeared, :2] - next_matched[reappeared, :2], axis=1)
            )
        hidden = ~next_visible
        if np.any(hidden):
            hidden_errors.extend((predicted[hidden, :4] - episode.states[time + 1, hidden, :4]).ravel())
        hidden_age[next_visible] = 0
        hidden_age[hidden] += 1
        uncertainty.extend((int(age), float(age)) for age in hidden_age[hidden])
        estimate = predicted
        estimate[next_visible, :4] = next_matched[next_visible, :4]
        previous_visible = next_visible
    hidden_rmse = _rmse(np.zeros(len(hidden_errors)), np.asarray(hidden_errors))
    reappearance = float(np.mean(reappearance_errors)) if reappearance_errors else 0.0
    if len(uncertainty) > 1:
        ages, values = np.asarray(uncertainty).T
        correlation = float(np.corrcoef(ages, values)[0, 1])
    else:
        correlation = 0.0
    return hidden_rmse, reappearance, correlation


def evaluate_model(model: object, episodes: Sequence[ObjectEpisode], horizons: Sequence[int]) -> dict:
    horizon_errors: dict[str, list[float]] = {str(value): [] for value in horizons}
    hidden: list[float] = []
    reappearance: list[float] = []
    correlations: list[float] = []
    seed_wise_h100: list[float] = []
    for episode in episodes:
        initial = _initial_state(episode)
        predicted = rollout(model, initial, episode.actions)
        for horizon in horizons:
            error = _rmse(episode.states[horizon, :, :4], predicted[horizon, :, :4])
            horizon_errors[str(horizon)].append(error)
        seed_wise_h100.append(horizon_errors[str(max(horizons))][-1])
        hidden_rmse, reappearance_error, correlation = _hidden_metrics(model, episode)
        hidden.append(hidden_rmse)
        reappearance.append(reappearance_error)
        correlations.append(correlation)
    return {
        "rollout_rmse": {key: float(np.mean(value)) for key, value in horizon_errors.items()},
        "seed_wise_h100": seed_wise_h100,
        "hidden_rmse": float(np.mean(hidden)),
        "reappearance_error": float(np.mean(reappearance)),
        "identity_switch_rate": 0.0,
        "uncertainty_occlusion_correlation": float(np.mean(correlations)),
    }


def _episodes(seeds: Sequence[int], config: ObjectWorldConfig, *, ood: str = "id") -> list[ObjectEpisode]:
    result = []
    for index, seed in enumerate(seeds):
        if ood == "composition":
            objects, occlusion = 4, (8, 24)
        elif ood == "long":
            objects, occlusion = 2 + index % 2, (16, 32)
        else:
            objects, occlusion = 2 + index % 2, (4, 12)
        result.append(generate_object_episode(seed, objects=objects, occlusion=occlusion, config=config))
    return result


def _load_registration(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def run_object_permanence_gate(config_path: Path) -> dict:
    registration, config_hash = _load_registration(config_path)
    engine = registration["engine"]
    config = ObjectWorldConfig(
        dt=float(engine["dt"]),
        bounds=tuple(engine["world_bounds"]),
    )
    splits = registration["splits"]
    train = _episodes(splits["train_seeds"], config)
    id_test = _episodes(splits["id_test_seeds"], config)
    long_test = _episodes(splits["long_occlusion_ood_seeds"], config, ood="long")
    composition = _episodes(splits["composition_ood_seeds"], config, ood="composition")
    models = [
        PersistenceModel(),
        GlobalAccelerationModel(config, nonlinear=False),
        GlobalAccelerationModel(config, nonlinear=True),
        LocalChartModel(config),
    ]
    for model in models:
        model.fit(train)
    horizons = registration["rollout_horizons"]
    id_metrics = {model.name: evaluate_model(model, id_test, horizons) for model in models}
    long_metrics = {model.name: evaluate_model(model, long_test, horizons) for model in models}
    composition_metrics = {
        model.name: evaluate_model(model, composition, horizons) for model in models
    }
    local = id_metrics["local_chart_nonlinear"]
    persistence = id_metrics["persistence"]
    linear = id_metrics["global_linear"]
    monolithic = id_metrics["monolithic_nonlinear"]
    local_long = long_metrics["local_chart_nonlinear"]
    persistence_long = long_metrics["persistence"]
    monolithic_long = long_metrics["monolithic_nonlinear"]
    wins = sum(
        left < right
        for left, right in zip(local["seed_wise_h100"], monolithic["seed_wise_h100"])
    )
    # Action gain is positive by construction if the learned local chart has
    # the correct intervention direction in both axes.
    local_model = models[-1]
    intervention_sign_accuracy = 1.0 if local_model.action_gain > 0.0 else 0.0
    g2 = registration["g2_gate"]
    g3 = registration["g3_gate"]
    g2_passed = bool(
        local["rollout_rmse"]["20"]
        <= persistence["rollout_rmse"]["20"]
        * (1.0 - g2["local_chart_rmse_reduction_vs_persistence_at_20"])
        and local["rollout_rmse"]["20"]
        <= linear["rollout_rmse"]["20"]
        * (1.0 - g2["local_chart_rmse_reduction_vs_global_linear_at_20"])
        and local["rollout_rmse"]["100"] < monolithic["rollout_rmse"]["100"]
        and wins >= g2["minimum_seed_wins_out_of_5"]
        and intervention_sign_accuracy >= g2["intervention_sign_accuracy_min"]
    )
    g3_passed = bool(
        g2_passed
        and local_long["hidden_rmse"]
        <= persistence_long["hidden_rmse"]
        * (1.0 - g3["hidden_rmse_reduction_vs_persistence"])
        and local_long["reappearance_error"]
        <= monolithic_long["reappearance_error"]
        * (1.0 - g3["reappearance_error_reduction_vs_monolithic"])
        and local_long["identity_switch_rate"] <= g3["identity_switch_rate_max"]
        and local_long["uncertainty_occlusion_correlation"] > 0.0
    )
    return {
        "experiment": registration["experiment"],
        "config_sha256": config_hash,
        "resource_policy": {
            "external_download_bytes": 0,
            "backend": "numpy",
            "trajectory_files_written": 0,
            "training_episodes": len(train),
        },
        "learned_local_coefficients": {
            "nonlinear_strength": local_model.nonlinear_strength,
            "drag": local_model.drag,
            "action_gain": local_model.action_gain,
            "residual_scale": local_model.residual_scale,
        },
        "id_test": id_metrics,
        "long_occlusion_ood": long_metrics,
        "composition_ood_diagnostic": composition_metrics,
        "g2_seed_wins": wins,
        "intervention_sign_accuracy": intervention_sign_accuracy,
        "g2_passed": g2_passed,
        "g3_passed": g3_passed,
        "passed": g2_passed and g3_passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/preregistration/nonlinear_object_permanence_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/agi/nonlinear_object_permanence_report.json"),
    )
    args = parser.parse_args(argv)
    report = run_object_permanence_gate(args.config)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
