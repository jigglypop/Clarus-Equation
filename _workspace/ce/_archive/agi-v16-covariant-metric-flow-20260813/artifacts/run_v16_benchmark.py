"""Development and sealed-confirmation evaluator for the V16 metric flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from reality_stone.clarus import covariant_metric_flow as metric_flow_module
from reality_stone.clarus.covariant_metric_flow import (
    CovariantMetricConfig,
    CovariantMetricFlow,
)


DEVELOPMENT_SEEDS = range(917_000, 917_064)
CONFIRMATION_SEEDS = range(918_000, 918_256)
RATES = (0.05, 0.1, 0.2, 0.4, 1.0)
TIE_MULTIPLIER = 64.0
REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_RELATIVE = PurePosixPath(
    "_workspace/ce/agi-v16-covariant-metric-flow-20260813"
)
MANIFEST_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-manifest.json"
RATES_RELATIVE = RUN_RELATIVE / "artifacts/selected-rates.json"
DEVELOPMENT_RESULT_RELATIVE = RUN_RELATIVE / "artifacts/development-results.json"
RECEIPT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-opened.json"
RESULT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-results.json"
REQUIRED_MANIFEST_PATHS = frozenset(
    {
        "reality_stone/python/reality_stone/clarus/covariant_metric_flow.py",
        "reality_stone/python/reality_stone/clarus/__init__.py",
        str(RUN_RELATIVE / "artifacts/run_v16_benchmark.py"),
        str(RUN_RELATIVE / "00-contract.md"),
        str(RATES_RELATIVE),
        str(DEVELOPMENT_RESULT_RELATIVE),
    }
)
PRODUCTION_RELATIVE = PurePosixPath(
    "reality_stone/python/reality_stone/clarus/covariant_metric_flow.py"
)
IMPORTED_PRODUCTION_PATH = Path(metric_flow_module.__file__).resolve(strict=True)


def tied_argmin(values: np.ndarray) -> int:
    minimum = float(np.min(values))
    tolerance = TIE_MULTIPLIER * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(values))),
    )
    return int(np.flatnonzero(values <= minimum + tolerance)[0])


def hidden_metric(rng: np.random.Generator) -> np.ndarray:
    basis, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    basis = basis * signs
    eigenvalues = np.exp(rng.uniform(math.log(0.25), math.log(4.0), 3))
    return basis @ np.diag(eigenvalues) @ basis.T


def unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.float64)
    flat = result.reshape((-1, shape[-1]))
    for row in flat:
        while True:
            candidate = rng.normal(size=shape[-1])
            norm = float(np.linalg.norm(candidate))
            if norm > 0.0 and math.isfinite(norm):
                row[:] = candidate / norm
                break
    return result


def episode_inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = hidden_metric(rng)
    candidates = unit_vectors(rng, (128, 4, 3))
    noise = rng.normal(size=128)
    routes = unit_vectors(rng, (64, 4, 2, 3))
    return target, candidates, noise, routes


def project_spd(metric: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    clipped = np.clip(eigenvalues, 1.0e-6, 1.0e6)
    return (eigenvectors * clipped) @ eigenvectors.T


def invariant_metric_error(metric: np.ndarray, target: np.ndarray) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(target)
    inverse_sqrt = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    generalized = np.linalg.eigvalsh(inverse_sqrt @ metric @ inverse_sqrt)
    return float(np.sqrt(np.mean(np.log(generalized) ** 2)))


def score_routes(metric: np.ndarray, target: np.ndarray, routes: np.ndarray) -> dict[str, float]:
    predicted = np.einsum("...i,ij,...j->...", routes, metric, routes).sum(axis=2)
    actual = np.einsum("...i,ij,...j->...", routes, target, routes).sum(axis=2)
    choices = np.asarray([tied_argmin(row) for row in predicted])
    optimum = np.asarray([tied_argmin(row) for row in actual])
    selected = actual[np.arange(len(routes)), choices]
    best = np.min(actual, axis=1)
    return {
        "correct": float(np.sum(choices == optimum)),
        "count": float(len(routes)),
        "regret_sum": float(np.sum((selected - best) / best)),
    }


def score_episode(seed: int, learner: str, rate: float) -> dict[str, float | bool]:
    target, candidates, noise, routes = episode_inputs(seed)
    flow = CovariantMetricFlow(
        3,
        CovariantMetricConfig(eta=rate if learner == "v16" else 0.4),
    )
    state = flow.identity_state()
    additive = np.eye(3)
    conformal = 1.0
    online_regret: list[float] = []

    try:
        for step, options in enumerate(candidates):
            if learner == "v16":
                predicted = np.asarray([flow.predict(state, option) for option in options])
            elif learner == "additive":
                predicted = np.einsum("ki,ij,kj->k", options, additive, options)
            elif learner == "conformal":
                predicted = conformal * np.einsum("ki,ki->k", options, options)
            elif learner == "identity":
                predicted = np.einsum("ki,ki->k", options, options)
            else:  # pragma: no cover - evaluator invariant
                raise ValueError(learner)

            action = (step // 4) % 4 if step % 4 == 0 else tied_argmin(predicted)
            vector = options[action]
            true_costs = np.einsum("ki,ij,kj->k", options, target, options)
            online_regret.append(
                float((true_costs[action] - np.min(true_costs)) / np.min(true_costs))
            )
            observed = float(true_costs[action] * math.exp(0.05 * noise[step]))

            if learner == "v16":
                state = flow.update(state, vector, observed)
            elif learner == "additive":
                prediction = float(vector @ additive @ vector)
                residual = math.log(prediction / observed)
                additive = project_spd(
                    additive - rate * (residual / prediction) * np.outer(vector, vector)
                )
            elif learner == "conformal":
                prediction = conformal * float(vector @ vector)
                residual = math.log(prediction / observed)
                conformal *= math.exp(-rate * residual)

        if learner == "v16":
            learned = flow.metric(state)
        elif learner == "additive":
            learned = additive
        elif learner == "conformal":
            learned = conformal * np.eye(3)
        else:
            learned = np.eye(3)
        route_score = score_routes(learned, target, routes)
        return {
            **route_score,
            "metric_error": invariant_metric_error(learned, target),
            "online_regret_after_32": float(np.mean(online_regret[32:])),
            "finite": bool(np.all(np.isfinite(learned))),
        }
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "correct": 0.0,
            "count": 64.0,
            "regret_sum": math.inf,
            "metric_error": math.inf,
            "online_regret_after_32": math.inf,
            "finite": False,
            "error": f"{type(error).__name__}: {error}",
        }


def aggregate(seeds: range, learner: str, rate: float) -> dict[str, float | int]:
    episodes = [score_episode(seed, learner, rate) for seed in seeds]
    count = sum(float(item["count"]) for item in episodes)
    return {
        "seeds": len(episodes),
        "finite_episode_rate": float(np.mean([bool(item["finite"]) for item in episodes])),
        "route_accuracy": sum(float(item["correct"]) for item in episodes) / count,
        "mean_normalized_regret": sum(float(item["regret_sum"]) for item in episodes) / count,
        "median_invariant_metric_error": float(
            np.median([float(item["metric_error"]) for item in episodes])
        ),
        "mean_online_regret_after_32": float(
            np.mean([float(item["online_regret_after_32"]) for item in episodes])
        ),
    }


def development() -> dict[str, Any]:
    learners: dict[str, Any] = {}
    for learner in ("v16", "additive", "conformal"):
        candidates = {str(rate): aggregate(DEVELOPMENT_SEEDS, learner, rate) for rate in RATES}
        selected = min(
            RATES,
            key=lambda rate: (
                float(candidates[str(rate)]["mean_normalized_regret"]),
                rate,
            ),
        )
        learners[learner] = {"rates": candidates, "selected_rate": selected}
    learners["identity"] = aggregate(DEVELOPMENT_SEEDS, "identity", 0.0)
    return {"mode": "development", "learners": learners}


def chart_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1_600_000_000)
    orthogonal: list[np.ndarray] = []
    for _ in range(2):
        basis, upper = np.linalg.qr(rng.normal(size=(3, 3)), mode="reduced")
        signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
        orthogonal.append(basis * signs)
    log_singular = rng.uniform(-math.log(10.0), math.log(10.0), size=3)
    result = orthogonal[0] @ np.diag(np.exp(log_singular)) @ orthogonal[1].T
    if not np.all(np.isfinite(result)) or np.linalg.matrix_rank(result) != 3:
        raise FloatingPointError("confirmation chart draw is nonfinite or singular")
    return result


def chart_episode(seed: int, rate: float) -> dict[str, float | bool]:
    target, candidates, noise, routes = episode_inputs(seed)
    jacobian = chart_matrix(seed)
    inverse = np.linalg.inv(jacobian)
    transformed_target = inverse.T @ target @ inverse
    transformed_candidates = np.einsum("ij,tkj->tki", jacobian, candidates)
    transformed_routes = np.einsum("ij,...j->...i", jacobian, routes)
    flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=rate))
    transformed_flow = CovariantMetricFlow(3, CovariantMetricConfig(eta=rate))
    state = flow.identity_state()
    transformed_state = transformed_flow.make_state_from_metric(inverse.T @ inverse)
    action_matches = 0
    prediction_errors: list[float] = []

    try:
        for step, (options, options_y) in enumerate(
            zip(candidates, transformed_candidates, strict=True)
        ):
            predictions = np.asarray([flow.predict(state, option) for option in options])
            predictions_y = np.asarray(
                [transformed_flow.predict(transformed_state, option) for option in options_y]
            )
            prediction_errors.extend(
                abs(float(left) - float(right))
                / max(1.0e-300, abs(float(left)), abs(float(right)))
                for left, right in zip(predictions, predictions_y, strict=True)
            )
            if step % 4 == 0:
                action = action_y = (step // 4) % 4
            else:
                action = tied_argmin(predictions)
                action_y = tied_argmin(predictions_y)
            action_matches += int(action == action_y)
            true_cost = float(options[action] @ target @ options[action])
            observed = true_cost * math.exp(0.05 * noise[step])
            state = flow.update(state, options[action], observed)
            # Keep the metamorphic state pair on corresponding observations even
            # if the independently computed policies disagree.  Action agreement
            # is scored separately, so a mismatch still fails G-CHART without
            # corrupting all later covariance diagnostics.
            transformed_state = transformed_flow.update(
                transformed_state,
                options_y[action],
                observed,
            )

        route_prediction = np.asarray(
            [flow.route_costs(state, route_set) for route_set in routes]
        )
        route_prediction_y = np.asarray(
            [
                transformed_flow.route_costs(transformed_state, route_set)
                for route_set in transformed_routes
            ]
        )
        prediction_errors.extend(
            abs(float(left) - float(right))
            / max(1.0e-300, abs(float(left)), abs(float(right)))
            for left, right in zip(
                route_prediction.flat,
                route_prediction_y.flat,
                strict=True,
            )
        )
        transported_metric = inverse.T @ flow.metric(state) @ inverse
        metric_y = transformed_flow.metric(transformed_state)
        metric_error = float(
            np.linalg.norm(metric_y - transported_metric)
            / max(1.0e-300, np.linalg.norm(transported_metric))
        )
        target_consistency = float(
            np.linalg.norm(transformed_target - inverse.T @ target @ inverse)
        )
        return {
            "finite": True,
            "action_matches": action_matches,
            "action_count": 128,
            "max_relative_prediction_error": max(prediction_errors),
            "relative_metric_transport_error": metric_error,
            "target_transport_identity_error": target_consistency,
        }
    except (
        FloatingPointError,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "finite": False,
            "action_matches": 0,
            "action_count": 128,
            "max_relative_prediction_error": math.inf,
            "relative_metric_transport_error": math.inf,
            "target_transport_identity_error": math.inf,
            "error": f"{type(error).__name__}: {error}",
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_root(root: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    if resolved_root != REPO_ROOT:
        raise ValueError(f"root must be evaluator repository root: {REPO_ROOT}")
    expected_evaluator = resolved_root.joinpath(
        *(RUN_RELATIVE / "artifacts/run_v16_benchmark.py").parts
    ).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_evaluator:
        raise ValueError("executed evaluator is not the repository evaluator")
    expected_production = resolved_root.joinpath(*PRODUCTION_RELATIVE.parts).resolve(
        strict=True
    )
    if IMPORTED_PRODUCTION_PATH != expected_production:
        raise ValueError("imported production module is not the repository module")
    return resolved_root


def _canonical_root_path(root: Path, relative: PurePosixPath) -> Path:
    resolved_root = _validated_root(root)
    candidate = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):  # pragma: no cover - fixed paths
        raise ValueError(f"path escapes repository root: {relative}")
    return candidate


def _manifest_target(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    return _canonical_root_path(root, parsed)


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    canonical_manifest = _canonical_root_path(root, MANIFEST_RELATIVE)
    if manifest_path.resolve(strict=True) != canonical_manifest:
        raise ValueError(f"manifest must be {MANIFEST_RELATIVE}")
    manifest = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest must be a nonempty path-to-SHA256 object")
    if not REQUIRED_MANIFEST_PATHS.issubset(manifest):
        missing = sorted(REQUIRED_MANIFEST_PATHS.difference(manifest))
        raise ValueError(f"manifest is missing required paths: {missing}")
    for relative, expected in manifest.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid SHA-256 for manifest path: {relative}")
        actual = sha256(_manifest_target(root, relative))
        if actual != expected:
            raise ValueError(f"manifest mismatch: {relative}")
    return {str(key): str(value) for key, value in manifest.items()}


def load_selected_rates(root: Path, rates_path: Path) -> dict[str, float]:
    canonical_rates = _canonical_root_path(root, RATES_RELATIVE)
    if rates_path.resolve(strict=True) != canonical_rates:
        raise ValueError(f"rates must be {RATES_RELATIVE}")
    selected = json.loads(canonical_rates.read_text(encoding="utf-8"))
    expected_keys = {"v16", "additive", "conformal"}
    if not isinstance(selected, dict) or set(selected) != expected_keys:
        raise ValueError(f"rates must have exactly these keys: {sorted(expected_keys)}")
    rates: dict[str, float] = {}
    for name in sorted(expected_keys):
        value = selected[name]
        if isinstance(value, bool):
            raise ValueError(f"rate for {name} must be a finite preregistered value")
        try:
            rate = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"rate for {name} must be a finite preregistered value"
            ) from error
        if not math.isfinite(rate) or rate not in RATES:
            raise ValueError(f"rate for {name} is outside the preregistered grid")
        rates[name] = rate
    return rates


def verify_development_provenance(root: Path, rates: dict[str, float]) -> None:
    path = _canonical_root_path(root, DEVELOPMENT_RESULT_RELATIVE)
    result = json.loads(path.read_text(encoding="utf-8"))
    try:
        if result["mode"] != "development":
            raise ValueError("development result has the wrong mode")
        learners = result["learners"]
        for name, rate in rates.items():
            selected = float(learners[name]["selected_rate"])
            if selected != rate:
                raise ValueError(
                    f"sealed rate for {name} does not match development argmin"
                )
            candidates = learners[name]["rates"]
            recomputed = min(
                RATES,
                key=lambda candidate: (
                    float(candidates[str(candidate)]["mean_normalized_regret"]),
                    candidate,
                ),
            )
            if recomputed != rate:
                raise ValueError(
                    f"development result does not reproduce selected rate for {name}"
                )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("development result", "sealed rate")
        ):
            raise
        raise ValueError("invalid development result schema") from error


def open_confirmation_block(
    root: Path,
    manifest_path: Path,
    rates: dict[str, float],
) -> Path:
    receipt = root.resolve(strict=True).joinpath(*RECEIPT_RELATIVE.parts)
    result = root.resolve(strict=True).joinpath(*RESULT_RELATIVE.parts)
    if result.exists():
        raise RuntimeError("confirmation result already exists; seed block is closed")
    payload = {
        "status": "opened-before-seed-access",
        "seed_start": CONFIRMATION_SEEDS.start,
        "seed_stop_exclusive": CONFIRMATION_SEEDS.stop,
        "manifest_path": str(MANIFEST_RELATIVE),
        "manifest_sha256": sha256(manifest_path.resolve(strict=True)),
        "selected_rates": rates,
    }
    try:
        with receipt.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError("confirmation seed block was already opened") from error
    return result


def confirmation(root: Path, manifest_path: Path, rates_path: Path) -> dict[str, Any]:
    manifest = verify_manifest(root, manifest_path)
    rates = load_selected_rates(root, rates_path)
    verify_development_provenance(root, rates)
    result_path = open_confirmation_block(root, manifest_path, rates)
    learners = {
        name: aggregate(CONFIRMATION_SEEDS, name, rate)
        for name, rate in rates.items()
    }
    learners["identity"] = aggregate(CONFIRMATION_SEEDS, "identity", 0.0)
    charts = [chart_episode(seed, rates["v16"]) for seed in CONFIRMATION_SEEDS]
    chart = {
        "finite_episode_rate": float(np.mean([bool(item["finite"]) for item in charts])),
        "action_agreement": sum(int(item["action_matches"]) for item in charts)
        / sum(int(item["action_count"]) for item in charts),
        "max_relative_prediction_error": max(
            float(item["max_relative_prediction_error"]) for item in charts
        ),
        "max_relative_metric_transport_error": max(
            float(item["relative_metric_transport_error"]) for item in charts
        ),
    }
    v16 = learners["v16"]
    identity = learners["identity"]
    conformal = learners["conformal"]
    additive = learners["additive"]
    gates = {
        "finite": v16["finite_episode_rate"] == 1.0,
        "accuracy": v16["route_accuracy"] >= 0.90,
        "regret": v16["mean_normalized_regret"] <= 0.05,
        "metric_error": v16["median_invariant_metric_error"] <= 0.25,
        "identity_improvement": identity["mean_normalized_regret"]
        - v16["mean_normalized_regret"]
        >= 0.10,
        "conformal_improvement": conformal["mean_normalized_regret"]
        - v16["mean_normalized_regret"]
        >= 0.05,
        "additive_noninferiority": v16["mean_normalized_regret"]
        - additive["mean_normalized_regret"]
        <= 0.02,
        "chart_actions": chart["action_agreement"] == 1.0,
        "chart_predictions": chart["max_relative_prediction_error"] <= 1.0e-10,
        "closed_loop": identity["mean_online_regret_after_32"]
        - v16["mean_online_regret_after_32"]
        >= 0.05,
    }
    result = {
        "mode": "confirmation",
        "manifest_verified": True,
        "manifest": manifest,
        "selected_rates": rates,
        "learners": learners,
        "chart": chart,
        "gates": gates,
        "learning_chart_closed_loop_pass": all(gates.values()),
    }
    try:
        with result_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:  # pragma: no cover - prechecked + one process
        raise RuntimeError("confirmation result path appeared after block opening") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rates", type=Path)
    args = parser.parse_args()
    if args.mode == "development":
        result = development()
        development_path = _validated_root(args.root).joinpath(
            *DEVELOPMENT_RESULT_RELATIVE.parts
        )
        with development_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        if args.manifest is None or args.rates is None:
            parser.error("confirmation requires --manifest and --rates")
        result = confirmation(args.root.resolve(), args.manifest, args.rates)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
