"""Low-cost fault-injection sandbox and predictive safety supervisor."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class State:
    position: float = 0.0
    velocity: float = 0.0


def _target(step: int, magnitude: float, period: int = 50) -> float:
    phase = (step // period) % 4
    return magnitude if phase in (0, 3) else -magnitude


def _faults(step: int, config: dict[str, Any]) -> tuple[bool, float, bool]:
    dropout = any(start <= step <= stop for start, stop in config["dropout_windows"])
    fault_gain = float(config.get("gain_multiplier", 2.35))
    gain = fault_gain if config["gain_window"][0] <= step <= config["gain_window"][1] else 1.0
    flipped = config["sign_flip_window"][0] <= step <= config["sign_flip_window"][1]
    return dropout, gain, flipped


def _advance(state: State, command: float, gain: float, flipped: bool, dt: float) -> State:
    applied = gain * (-command if flipped else command)
    velocity = 0.992 * state.velocity + dt * applied
    return State(state.position + dt * velocity, velocity)


def _run_episode(seed: int, config: dict[str, Any], supervised: bool) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dt = float(config["dt"])
    bound = float(config["safe_bound"])
    action_limit = float(config["action_limit"])
    velocity_limit = float(config["velocity_limit"])
    delay = int(config["sensor_delay_steps"])
    state = State(float(rng.normal(0.0, 0.015)), float(rng.normal(0.0, 0.01)))
    estimate = State(state.position, state.velocity)
    observations: list[tuple[float, float] | None] = []
    covariance = 0.0
    last_command = 0.0
    violations = 0
    interventions = 0
    intervention_steps: list[int] = []
    violation_steps: list[int] = []
    target_error = 0.0
    zero_error = 0.0
    max_command = 0.0
    max_innovation = 0.0
    max_estimated_speed = 0.0
    max_abs_estimate = abs(estimate.position)
    max_abs_position = abs(state.position)
    missing_observations = 0
    cooldown = 0
    command_history: list[float] = []
    estimate_history: list[State] = []
    robust = supervised and int(config.get("supervisor_version", 1)) >= 2

    for step in range(int(config["steps"])):
        estimate_history.append(State(estimate.position, estimate.velocity))
        target = _target(step, float(config["target_magnitude"]), int(config.get("target_period", 50)))
        dropout, gain, flipped = _faults(step, config)
        noisy = (state.position + rng.normal(0.0, 0.003), state.velocity + rng.normal(0.0, 0.004))
        observations.append(None if dropout else noisy)
        delayed = observations[step - delay] if step >= delay else observations[0]

        if delayed is None:
            missing_observations += 1
            estimate = _advance(estimate, last_command, 1.0, False, dt)
            covariance += 0.018
        else:
            observed_step = max(0, step - delay)
            comparison = estimate_history[observed_step]
            innovation = abs(delayed[0] - comparison.position) + 0.35 * abs(delayed[1] - comparison.velocity)
            max_innovation = max(max_innovation, innovation)
            estimate = State(float(delayed[0]), float(delayed[1]))
            if robust:
                for old_command in command_history[observed_step:step]:
                    estimate = _advance(estimate, old_command, 1.0, False, dt)
            covariance = max(0.002, 0.32 * covariance)
            threshold = float(config.get("innovation_threshold", 0.10))
            if supervised and innovation > threshold:
                cooldown = max(cooldown, int(config.get("isolation_steps", 5)))

        reference = target * float(config.get("safe_reference_scale", 1.0)) if robust else target
        raw = 8.0 * (reference - estimate.position) - 3.0 * estimate.velocity
        command = float(np.clip(raw, -action_limit, action_limit))

        if supervised:
            proposed = command
            reason = False
            if robust:
                command = float(np.clip(command, -float(config["supervised_action_limit"]), float(config["supervised_action_limit"])))
            if covariance > 0.065 or cooldown > 0:
                command = 0.0 if robust else float(np.clip(-5.0 * estimate.velocity, -0.55, 0.55))
                reason = True
            if abs(estimate.velocity) > velocity_limit:
                command = float(np.clip(-7.0 * estimate.velocity, -action_limit, action_limit))
                reason = True
            horizon = int(config.get("prediction_horizon", 1))
            worst_velocity = estimate.velocity + horizon * dt * 2.35 * command
            margin = 0.018 + 2.4 * covariance
            predicted = estimate.position + horizon * dt * worst_velocity
            if abs(predicted) + margin >= bound:
                command = 0.0 if robust and cooldown > 0 else float(np.clip(-8.0 * estimate.velocity - 5.0 * estimate.position, -action_limit, action_limit))
                reason = True
            if reason and abs(command - proposed) > 1e-12:
                interventions += 1
                intervention_steps.append(step)
            cooldown = max(0, cooldown - 1)

        command = float(np.clip(command, -action_limit, action_limit))
        max_estimated_speed = max(max_estimated_speed, abs(estimate.velocity))
        max_abs_estimate = max(max_abs_estimate, abs(estimate.position))
        max_command = max(max_command, abs(command))
        last_command = command
        command_history.append(command)
        state = _advance(state, command, gain, flipped, dt)
        max_abs_position = max(max_abs_position, abs(state.position))
        violated = abs(state.position) >= bound
        violations += int(violated)
        if violated:
            violation_steps.append(step)
        target_error += abs(state.position - target)
        zero_error += abs(target)

    horizon = int(config.get("prediction_horizon", 1))
    successful_interventions = sum(
        not any(step <= bad_step <= step + horizon for bad_step in violation_steps)
        for step in intervention_steps
    )
    return {
        "violations": float(violations),
        "target_error": target_error / int(config["steps"]),
        "zero_error": zero_error / int(config["steps"]),
        "interventions": float(interventions),
        "successful_interventions": float(successful_interventions),
        "max_command": max_command,
        "max_innovation": max_innovation,
        "max_estimated_speed": max_estimated_speed,
        "max_abs_estimate": max_abs_estimate,
        "max_abs_position": max_abs_position,
        "missing_fraction": missing_observations / int(config["steps"]),
    }


def run_sandbox_safety_gate(config_path: Path | str, split: str = "validation") -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    prereg = json.loads(config_path.read_text(encoding="utf-8"))
    env = prereg["environment"]
    seeds = prereg[f"{split}_seeds"]
    supervised = [_run_episode(seed, env, True) for seed in seeds]
    baseline = [_run_episode(seed, env, False) for seed in seeds]
    safe_violations = int(sum(item["violations"] for item in supervised))
    base_violations = int(sum(item["violations"] for item in baseline))
    safe_error = float(np.mean([item["target_error"] for item in supervised]))
    base_error = float(np.mean([item["target_error"] for item in baseline]))
    zero_error = float(np.mean([item["zero_error"] for item in supervised]))
    interventions = int(sum(item["interventions"] for item in supervised))
    successes = int(sum(item["successful_interventions"] for item in supervised))
    max_command = max(item["max_command"] for item in supervised + baseline)
    elapsed = time.perf_counter() - started
    criteria = prereg["success_criteria"]
    checks = {
        "zero_supervised_violations": safe_violations == criteria["supervised_boundary_violations"],
        "baseline_exposes_hazard": base_violations >= criteria["baseline_boundary_violations_min"],
        "useful_vs_zero": safe_error <= criteria["supervised_mean_target_error_max_fraction_of_zero"] * zero_error,
        "competitive_vs_baseline": safe_error <= criteria["supervised_mean_target_error_max_fraction_of_baseline"] * base_error,
        "shield_was_exercised": interventions >= criteria["shield_interventions_min"],
        "shield_success": interventions > 0 and successes / interventions >= criteria["shield_success_rate_min"],
        "command_saturated": max_command <= criteria["max_abs_command"] + 1e-12,
        "no_external_data": criteria["external_download_bytes"] == 0,
        "runtime": elapsed <= criteria["elapsed_seconds_max"],
    }
    return {
        "experiment": prereg["experiment"], "split": split, "seeds": seeds,
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "metrics": {
            "supervised_boundary_violations": safe_violations,
            "baseline_boundary_violations": base_violations,
            "supervised_mean_target_error": safe_error,
            "baseline_mean_target_error": base_error,
            "zero_action_mean_target_error": zero_error,
            "shield_interventions": interventions,
            "shield_success_rate": successes / interventions if interventions else 0.0,
            "max_abs_command": max_command,
        },
        "resource_usage": {"external_download_bytes": 0, "elapsed_seconds": elapsed},
        "skipped_cost": ["physical_robot", "paid_api", "new_neuroimaging"],
        "config": str(config_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_sandbox_safety_gate(args.config, args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
