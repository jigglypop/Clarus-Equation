"""Run the frozen, retrospective E17 neural-geometry candidate tournament.

This script is deliberately self-contained.  It validates the frozen registry
hashes before loading outcomes, records every candidate's eligibility, chooses
hyperparameters on outer-train animals only, and evaluates one selected tuple on
the held-out animal.  E17 was already opened before the candidate universe was
frozen, so every result is discovery rather than confirmation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import linear_sum_assignment, minimize
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr


HORIZONS = (1, 5, 15, 30)
COVARIANCE_RIDGES = (0.0, 1e-6, 1e-3, 1e-1)
METRIC_RIDGES = (0.0, 1e-6, 1e-3, 1e-1)
S13_PENALTIES = (0.0, 1e-4, 1e-2, 1.0)
S14_TAUS = (0.1, 1.0, 10.0)
DECODER_PENALTIES = (0.0, 1e-4, 1e-2, 1.0)
DIFFUSION_TIMES = (1, 5, 15, 30)
GRAPH_SYMMETRIZATIONS = ("absolute_mean", "positive_mean")
GROUND_METRICS = ("identity", "fit_state_covariance_precision")

EPS_G = 1e-6
EPS_NUMERIC = 1e-12
MIN_TRANSITIONS_PER_PARAMETER = 10
MAX_PAIR_SAMPLES = 512
MAX_SCORE_SAMPLES = 256
MONTE_CARLO_DRAWS = 256
PERMUTATIONS = 256
SEED = 1729


@dataclass(frozen=True)
class TrialBlocks:
    fit: tuple[np.ndarray, ...]
    inner: tuple[np.ndarray, ...]
    test: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Session:
    session_id: str
    animal: str
    source_path: str
    source_sha256: str
    original_dimension: int
    dimension: int
    conditions: dict[str, TrialBlocks]


@dataclass(frozen=True)
class LinearModel:
    j: np.ndarray
    bias: np.ndarray
    q: np.ndarray
    fit_transitions: int


@dataclass(frozen=True)
class HorizonModel:
    j_h: np.ndarray
    bias_h: np.ndarray
    reachability: np.ndarray


class CandidateFailure(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(to_jsonable(payload), target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("refusing to serialize a nonfinite numeric result")
        return numeric
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def tuple_key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def load_mat(path: Path) -> dict[str, Any]:
    return {
        key: value
        for key, value in loadmat(path, simplify_cells=True).items()
        if not key.startswith("__")
    }


def branch_trials(condition: dict[str, Any]) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(trial, dtype=float) for trial in condition["branch"])
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("branch trials must be nonempty time-by-ROI matrices")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("branch trial shapes change within a condition")
    return arrays


def split_trials(trials: tuple[np.ndarray, ...]) -> TrialBlocks:
    count = len(trials)
    fit_count = int(math.floor(0.50 * count))
    inner_count = int(math.floor(0.25 * count))
    test_start = fit_count + inner_count
    if fit_count < 2 or inner_count < 2 or count - test_start < 2:
        raise ValueError(f"insufficient trials for frozen 50/25/remainder split: {count}")
    return TrialBlocks(
        fit=trials[:fit_count],
        inner=trials[fit_count:test_start],
        test=trials[test_start:],
    )


def prepare_session(path: Path, source_sha256: str | None = None) -> Session:
    cont_data = load_mat(path)["cont_data"]
    raw = {
        "saline": split_trials(branch_trials(cont_data["Sal"])),
        "dcz": split_trials(branch_trials(cont_data["DCZ"])),
    }
    pooled_fit = np.vstack(raw["saline"].fit + raw["dcz"].fit)
    mean = np.nanmean(pooled_fit, axis=0)
    scale = np.nanstd(pooled_fit, axis=0)
    keep = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-8)
    if not np.any(keep):
        raise ValueError("no ROI survives the frozen fit-only chart")

    def transform(items: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return tuple((item[:, keep] - mean[keep]) / scale[keep] for item in items)

    conditions = {
        label: TrialBlocks(
            fit=transform(blocks.fit),
            inner=transform(blocks.inner),
            test=transform(blocks.test),
        )
        for label, blocks in raw.items()
    }
    animal_match = re.match(r"(DCO\d+)", path.stem)
    if animal_match is None:
        raise ValueError(f"cannot parse animal from {path.stem}")
    return Session(
        session_id=path.stem,
        animal=animal_match.group(1),
        source_path=path.as_posix(),
        source_sha256=source_sha256 or sha256_file(path),
        original_dimension=int(keep.size),
        dimension=int(np.sum(keep)),
        conditions=conditions,
    )


def finite_rows(items: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(tuple(items))
    return matrix[np.isfinite(matrix).all(axis=1)]


def transition_pairs(
    trials: tuple[np.ndarray, ...], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for trial in trials:
        if trial.shape[0] <= horizon:
            continue
        left.append(trial[:-horizon])
        right.append(trial[horizon:])
    if not left:
        raise CandidateFailure("INSUFFICIENT_PAIRS", "no transition pair survives horizon")
    x = np.vstack(left)
    y = np.vstack(right)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if not np.any(keep):
        raise CandidateFailure("INSUFFICIENT_PAIRS", "no finite transition pair")
    return x[keep], y[keep]


def covariance_mle(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] < 2:
        raise CandidateFailure("INSUFFICIENT_PAIRS", "covariance needs at least two rows")
    centered = values - np.mean(values, axis=0)
    result = centered.T @ centered / centered.shape[0]
    return symmetrize(result)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0


def require_finite(matrix: np.ndarray, label: str) -> None:
    if not np.isfinite(matrix).all():
        raise CandidateFailure("NUMERICAL_FAILURE", f"{label} contains nonfinite values")


def require_spd(matrix: np.ndarray, label: str) -> np.ndarray:
    matrix = symmetrize(matrix)
    require_finite(matrix, label)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise CandidateFailure("INELIGIBLE_SINGULAR", f"{label} is not SPD") from error
    return matrix


def fit_linear_model(trials: tuple[np.ndarray, ...]) -> LinearModel:
    x, y = transition_pairs(trials, 1)
    dimension = x.shape[1]
    if x.shape[0] < MIN_TRANSITIONS_PER_PARAMETER * (dimension + 1):
        raise CandidateFailure(
            "INSUFFICIENT_PAIRS",
            "fit transitions fail the frozen per-parameter minimum",
        )
    design = np.column_stack([x, np.ones(x.shape[0])])
    coefficient, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    j = coefficient[:-1].T
    bias = coefficient[-1]
    residual = y - design @ coefficient
    q = symmetrize(residual.T @ residual / residual.shape[0])
    require_finite(j, "J")
    require_finite(q, "Q")
    return LinearModel(j=j, bias=bias, q=q, fit_transitions=int(x.shape[0]))


def horizon_model(model: LinearModel, horizon: int) -> HorizonModel:
    dimension = model.j.shape[0]
    power = np.eye(dimension)
    bias_h = np.zeros(dimension)
    covariance = np.zeros((dimension, dimension))
    for _ in range(horizon):
        covariance += power @ model.q @ power.T
        bias_h += power @ model.bias
        power = power @ model.j
    return HorizonModel(
        j_h=power,
        bias_h=bias_h,
        reachability=symmetrize(covariance),
    )


def residuals_for(
    trials: tuple[np.ndarray, ...],
    horizon: int,
    horizon_fit: HorizonModel,
    mean_kind: str,
) -> np.ndarray:
    x, y = transition_pairs(trials, horizon)
    if mean_kind == "direct":
        prediction = x @ horizon_fit.j_h.T + horizon_fit.bias_h
    elif mean_kind == "persistence":
        prediction = x
    else:
        raise ValueError(f"unknown mean kind {mean_kind}")
    return y - prediction


def gaussian_nlpd(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = require_spd(covariance, "predictive covariance")
    chol = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(chol, residual.T)
    quadratic = np.sum(standardized * standardized, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    dimension = covariance.shape[0]
    return float(
        np.mean(0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic))
    )


def evenly_spaced_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.floor(np.linspace(0, count - 1, maximum)).astype(int)


def energy_score(residual: np.ndarray, covariance: np.ndarray) -> float:
    selected = residual[evenly_spaced_indices(residual.shape[0], MAX_SCORE_SAMPLES)]
    covariance = require_spd(covariance, "energy covariance")
    chol = np.linalg.cholesky(covariance)
    rng = np.random.Generator(np.random.PCG64(SEED))
    first = rng.standard_normal((MONTE_CARLO_DRAWS, covariance.shape[0])) @ chol.T
    second = rng.standard_normal((MONTE_CARLO_DRAWS, covariance.shape[0])) @ chol.T
    term_one = np.mean(np.linalg.norm(first[None, :, :] - selected[:, None, :], axis=2))
    term_two = np.mean(np.linalg.norm(first - second, axis=1))
    return float(term_one - 0.5 * term_two)


def inverse_softplus(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value + np.log(-np.expm1(-value))


def fit_low_rank_precision(
    residual_covariance: np.ndarray, rank: int, penalty: float
) -> tuple[np.ndarray, dict[str, Any]]:
    dimension = residual_covariance.shape[0]
    if dimension < 2 or rank < 1 or rank > min(3, dimension - 1):
        raise CandidateFailure("INELIGIBLE_DIMENSION", "S13 rank is unavailable")
    diagonal_precision = np.maximum(
        1.0 / (np.diag(residual_covariance) + EPS_G), EPS_G
    )
    initial_d = inverse_softplus(
        np.maximum(diagonal_precision - EPS_G, EPS_G)
    )
    initial_u = np.fromfunction(
        lambda i, j: 1e-3 * np.sin((i + 1.0) * (j + 1.0)),
        (dimension, rank),
        dtype=float,
    )
    initial = np.concatenate([initial_d, initial_u.ravel()])

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = vector[:dimension]
        u = vector[dimension:].reshape(dimension, rank)
        diagonal = np.logaddexp(0.0, d) + EPS_G
        metric = symmetrize(np.diag(diagonal) + u @ u.T)
        return d, u, metric

    def objective(vector: np.ndarray) -> float:
        _, u, metric = unpack(vector)
        sign, logdet = np.linalg.slogdet(metric)
        if sign <= 0 or not math.isfinite(logdet):
            return float("inf")
        return float(
            0.5 * (np.trace(metric @ residual_covariance) - logdet)
            + 0.5 * penalty * np.sum(u * u)
        )

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac="2-point",
        bounds=None,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
            "finite_diff_rel_step": 1e-6,
        },
    )
    _, _, metric = unpack(result.x)
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or not math.isfinite(float(result.fun))
    ):
        raise CandidateFailure(
            "OPTIMIZER_FAILED",
            f"S13 optimizer failed: {result.message}",
        )
    metric = require_spd(metric, "S13 precision")
    return metric, {
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "parameter_count": int(dimension + dimension * rank),
    }


def scalar_covariance_calibration(residual: np.ndarray, shape: np.ndarray) -> float:
    shape = require_spd(shape, "covariance shape")
    solved = np.linalg.solve(shape, residual.T)
    quadratic_sum = float(np.sum(residual.T * solved))
    scale = quadratic_sum / (residual.shape[0] * residual.shape[1])
    return max(EPS_NUMERIC, scale)


def aggregate_equal(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def strict_animal_mean(
    session_values: dict[str, float], sessions: list[Session], animals: set[str]
) -> tuple[float | None, dict[str, float]]:
    by_animal: dict[str, list[float]] = defaultdict(list)
    expected: dict[str, int] = defaultdict(int)
    for session in sessions:
        if session.animal in animals:
            expected[session.animal] += 1
            if session.session_id in session_values:
                by_animal[session.animal].append(session_values[session.session_id])
    animal_means: dict[str, float] = {}
    for animal in sorted(animals):
        if len(by_animal[animal]) != expected[animal] or expected[animal] == 0:
            return None, {}
        animal_means[animal] = float(np.mean(by_animal[animal]))
    return aggregate_equal(animal_means.values()), animal_means


UNCERTAINTY_CANDIDATES = (
    "S0",
    "S1",
    "S2",
    "S3",
    "S4-H",
    "S5",
    "S12",
    "S13",
    "BASE_FULL",
    "BASE_DIAGONAL",
    "BASE_ISOTROPIC",
    "BASE_PERSISTENCE",
)


def uncertainty_parameter_grid(candidate: str, dimension: int) -> list[dict[str, Any]]:
    if candidate in {"S1", "S2", "S3", "S4-H", "S5"}:
        return [{"lambda_c": value} for value in COVARIANCE_RIDGES]
    if candidate == "S13":
        if dimension < 2:
            return []
        return [
            {"rank": rank, "eta": penalty}
            for rank in range(1, min(3, dimension - 1) + 1)
            for penalty in S13_PENALTIES
        ]
    return [{}]


def uncertainty_shape(
    candidate: str,
    parameters: dict[str, Any],
    trials: tuple[np.ndarray, ...],
    model: LinearModel,
    horizon_fit: HorizonModel,
    horizon: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    dimension = model.j.shape[0]
    fit_residual = residuals_for(trials, horizon, horizon_fit, "direct")
    residual_covariance = covariance_mle(fit_residual)
    metadata: dict[str, Any] = {}
    mean_kind = "direct"

    if candidate == "S0":
        shape = np.eye(dimension)
        metadata["parameter_count"] = 0
    elif candidate == "S1":
        states = finite_rows(trials)
        shape = covariance_mle(states)
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "S2":
        increments = []
        for trial in trials:
            delta = np.diff(trial, axis=0)
            increments.append(delta[np.isfinite(delta).all(axis=1)])
        shape = covariance_mle(np.vstack(increments))
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "S3":
        shape = model.q
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "S4-H":
        shape = horizon_fit.reachability
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "S5":
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(model.j))))
        metadata["spectral_radius"] = spectral_radius
        if not spectral_radius < 1.0:
            raise CandidateFailure(
                "INELIGIBLE_SINGULAR",
                f"S5 stability gate failed: spectral radius {spectral_radius}",
            )
        shape = symmetrize(solve_discrete_lyapunov(model.j, model.q))
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "S12":
        diagonal_metric = np.maximum(
            1.0 / (np.diag(residual_covariance) + EPS_G), EPS_G
        )
        metric = require_spd(np.diag(diagonal_metric), "S12 precision")
        shape = np.linalg.inv(metric)
        metadata["parameter_count"] = dimension
    elif candidate == "S13":
        metric, optimizer = fit_low_rank_precision(
            residual_covariance,
            rank=int(parameters["rank"]),
            penalty=float(parameters["eta"]),
        )
        shape = np.linalg.inv(metric)
        metadata.update(optimizer)
    elif candidate == "BASE_FULL":
        shape = residual_covariance + EPS_G * np.eye(dimension)
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    elif candidate == "BASE_DIAGONAL":
        shape = np.diag(np.diag(residual_covariance)) + EPS_G * np.eye(dimension)
        metadata["parameter_count"] = dimension
    elif candidate == "BASE_ISOTROPIC":
        variance = float(np.trace(residual_covariance) / dimension)
        shape = np.eye(dimension) * (variance + EPS_G)
        metadata["parameter_count"] = 1
    elif candidate == "BASE_PERSISTENCE":
        mean_kind = "persistence"
        persistence_residual = residuals_for(trials, horizon, horizon_fit, mean_kind)
        shape = covariance_mle(persistence_residual) + EPS_G * np.eye(dimension)
        metadata["parameter_count"] = dimension * (dimension + 1) // 2
    else:
        raise ValueError(f"unknown uncertainty candidate {candidate}")

    if "lambda_c" in parameters:
        shape = shape + float(parameters["lambda_c"]) * np.eye(dimension)
    shape = require_spd(shape, f"{candidate} covariance shape")
    return shape, mean_kind, metadata


def fit_uncertainty_tuple(
    candidate: str,
    parameters: dict[str, Any],
    blocks: TrialBlocks,
    model: LinearModel,
    horizon_fit: HorizonModel,
    horizon: int,
) -> dict[str, Any]:
    shape, mean_kind, metadata = uncertainty_shape(
        candidate,
        parameters,
        blocks.fit,
        model,
        horizon_fit,
        horizon,
    )
    fit_residual = residuals_for(blocks.fit, horizon, horizon_fit, mean_kind)
    scale = scalar_covariance_calibration(fit_residual, shape)
    covariance = require_spd(scale * shape, f"{candidate} calibrated covariance")
    inner_residual = residuals_for(blocks.inner, horizon, horizon_fit, mean_kind)
    return {
        "status": "ELIGIBLE",
        "parameters": parameters,
        "mean_kind": mean_kind,
        "scale": scale,
        "shape": shape,
        "covariance": covariance,
        "inner_nlpd": gaussian_nlpd(inner_residual, covariance),
        **metadata,
    }


def score_uncertainty_test(
    fitted: dict[str, Any],
    blocks: TrialBlocks,
    horizon_fit: HorizonModel,
    horizon: int,
) -> dict[str, Any]:
    residual = residuals_for(
        blocks.test,
        horizon,
        horizon_fit,
        str(fitted["mean_kind"]),
    )
    covariance = np.asarray(fitted["covariance"], dtype=float)
    return {
        "test_pairs": int(residual.shape[0]),
        "nlpd": gaussian_nlpd(residual, covariance),
        "energy_score": energy_score(residual, covariance),
    }


def build_models(
    sessions: list[Session],
) -> tuple[
    dict[tuple[str, str], LinearModel],
    dict[tuple[str, str, int], HorizonModel],
]:
    models: dict[tuple[str, str], LinearModel] = {}
    horizons: dict[tuple[str, str, int], HorizonModel] = {}
    for session in sessions:
        for condition, blocks in session.conditions.items():
            model = fit_linear_model(blocks.fit)
            models[(session.session_id, condition)] = model
            for horizon in HORIZONS:
                horizons[(session.session_id, condition, horizon)] = horizon_model(
                    model, horizon
                )
    return models, horizons


def run_uncertainty_inner(
    sessions: list[Session],
    models: dict[tuple[str, str], LinearModel],
    horizons: dict[tuple[str, str, int], HorizonModel],
) -> tuple[dict[str, Any], dict[tuple[str, str, int, str, str], dict[str, Any]]]:
    report: dict[str, Any] = {}
    cache: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for session in sessions:
        session_report: dict[str, Any] = {}
        for condition, blocks in session.conditions.items():
            condition_report: dict[str, Any] = {}
            model = models[(session.session_id, condition)]
            for horizon in HORIZONS:
                horizon_report: dict[str, Any] = {}
                horizon_fit = horizons[(session.session_id, condition, horizon)]
                for candidate in UNCERTAINTY_CANDIDATES:
                    tuples: dict[str, Any] = {}
                    parameters_list = uncertainty_parameter_grid(
                        candidate, session.dimension
                    )
                    if not parameters_list:
                        tuples["{}"] = {
                            "status": "INELIGIBLE_DIMENSION",
                            "detail": "no frozen S13 rank exists at this dimension",
                        }
                    for parameters in parameters_list:
                        key = tuple_key(parameters)
                        try:
                            fitted = fit_uncertainty_tuple(
                                candidate,
                                parameters,
                                blocks,
                                model,
                                horizon_fit,
                                horizon,
                            )
                            cache[
                                (
                                    session.session_id,
                                    condition,
                                    horizon,
                                    candidate,
                                    key,
                                )
                            ] = fitted
                            tuples[key] = {
                                field: value
                                for field, value in fitted.items()
                                if field not in {"shape", "covariance"}
                            }
                        except CandidateFailure as error:
                            tuples[key] = {
                                "status": error.code,
                                "detail": error.detail,
                                "parameters": parameters,
                            }
                    horizon_report[candidate] = tuples
                condition_report[str(horizon)] = horizon_report
            session_report[condition] = condition_report
        report[session.session_id] = session_report
    return report, cache


def available_tuple_keys(
    raw_report: dict[str, Any],
    sessions: list[Session],
    animals: set[str],
    horizon: int,
    candidate: str,
) -> set[str]:
    intersection: set[str] | None = None
    for session in sessions:
        if session.animal not in animals:
            continue
        for condition in session.conditions:
            tuples = raw_report[session.session_id][condition][str(horizon)][candidate]
            eligible = {
                key for key, item in tuples.items() if item.get("status") == "ELIGIBLE"
            }
            intersection = eligible if intersection is None else intersection & eligible
    return intersection or set()


def select_uncertainty_tuple(
    raw_report: dict[str, Any],
    sessions: list[Session],
    train_animals: set[str],
    horizon: int,
    candidate: str,
) -> dict[str, Any]:
    eligible_keys = available_tuple_keys(
        raw_report, sessions, train_animals, horizon, candidate
    )
    choices: list[tuple[float, int, float, str, dict[str, float]]] = []
    for key in sorted(eligible_keys):
        session_values: dict[str, float] = {}
        parameter_count = 0
        ridge = 0.0
        for session in sessions:
            if session.animal not in train_animals:
                continue
            condition_scores = []
            for condition in session.conditions:
                item = raw_report[session.session_id][condition][str(horizon)][candidate][key]
                condition_scores.append(float(item["inner_nlpd"]))
                parameter_count = int(item.get("parameter_count", 0))
                ridge = float(item.get("parameters", {}).get("lambda_c", 0.0))
            session_values[session.session_id] = float(np.mean(condition_scores))
        mean, animal_means = strict_animal_mean(
            session_values, sessions, train_animals
        )
        if mean is not None:
            choices.append((mean, parameter_count, ridge, key, animal_means))
    if not choices:
        return {
            "status": "NO_ELIGIBLE_TUPLE",
            "candidate": candidate,
            "horizon": horizon,
        }
    choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = choices[0]
    return {
        "status": "SELECTED",
        "candidate": candidate,
        "horizon": horizon,
        "tuple_key": best[3],
        "parameters": json.loads(best[3]),
        "outer_train_inner_nlpd": best[0],
        "outer_train_animal_means": best[4],
        "eligible_tuple_count": len(choices),
    }


def run_uncertainty_outer(
    sessions: list[Session],
    raw_report: dict[str, Any],
    cache: dict[tuple[str, str, int, str, str], dict[str, Any]],
    horizons: dict[tuple[str, str, int], HorizonModel],
) -> dict[str, Any]:
    animals = sorted({session.animal for session in sessions})
    folds: dict[str, Any] = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        fold: dict[str, Any] = {}
        for horizon in HORIZONS:
            horizon_report: dict[str, Any] = {}
            for candidate in UNCERTAINTY_CANDIDATES:
                selection = select_uncertainty_tuple(
                    raw_report,
                    sessions,
                    train_animals,
                    horizon,
                    candidate,
                )
                if selection["status"] != "SELECTED":
                    horizon_report[candidate] = selection
                    continue
                key = str(selection["tuple_key"])
                session_scores: dict[str, Any] = {}
                failures: list[dict[str, str]] = []
                for session in sessions:
                    if session.animal != heldout:
                        continue
                    condition_scores = []
                    condition_energy = []
                    for condition, blocks in session.conditions.items():
                        fitted = cache.get(
                            (session.session_id, condition, horizon, candidate, key)
                        )
                        if fitted is None:
                            failures.append(
                                {
                                    "session": session.session_id,
                                    "condition": condition,
                                    "reason": "selected tuple unavailable on held-out calibration",
                                }
                            )
                            continue
                        try:
                            score = score_uncertainty_test(
                                fitted,
                                blocks,
                                horizons[(session.session_id, condition, horizon)],
                                horizon,
                            )
                            condition_scores.append(float(score["nlpd"]))
                            condition_energy.append(float(score["energy_score"]))
                        except CandidateFailure as error:
                            failures.append(
                                {
                                    "session": session.session_id,
                                    "condition": condition,
                                    "reason": f"{error.code}: {error.detail}",
                                }
                            )
                    if len(condition_scores) == len(session.conditions):
                        session_scores[session.session_id] = {
                            "nlpd": float(np.mean(condition_scores)),
                            "energy_score": float(np.mean(condition_energy)),
                        }
                if len(session_scores) != sum(
                    session.animal == heldout for session in sessions
                ):
                    status = "HELDOUT_EVALUATION_INCOMPLETE"
                    animal_nlpd = None
                    animal_energy = None
                else:
                    status = "EVALUATED"
                    animal_nlpd = float(
                        np.mean([item["nlpd"] for item in session_scores.values()])
                    )
                    animal_energy = float(
                        np.mean(
                            [item["energy_score"] for item in session_scores.values()]
                        )
                    )
                horizon_report[candidate] = {
                    **selection,
                    "status": status,
                    "heldout_animal": heldout,
                    "heldout_animal_nlpd": animal_nlpd,
                    "heldout_animal_energy_score": animal_energy,
                    "session_scores": session_scores,
                    "failures": failures,
                }
            fold[str(horizon)] = horizon_report
        folds[heldout] = fold

    scoreboard: dict[str, Any] = {}
    for horizon in HORIZONS:
        entries = []
        for candidate in UNCERTAINTY_CANDIDATES:
            values = [
                folds[animal][str(horizon)][candidate].get("heldout_animal_nlpd")
                for animal in animals
            ]
            if all(value is not None for value in values):
                entries.append(
                    {
                        "candidate": candidate,
                        "animal_nlpd": dict(zip(animals, values, strict=True)),
                        "mean_animal_nlpd": float(np.mean(values)),
                    }
                )
        entries.sort(key=lambda item: (item["mean_animal_nlpd"], item["candidate"]))
        scoreboard[str(horizon)] = {
            "status": "RETROSPECTIVE_DISCOVERY_NO_WINNER",
            "ranking": entries,
        }
    return {"folds": folds, "scoreboard": scoreboard}


DEFORMATION_CANDIDATES = ("S6-H", "S7-H", "S14", "S15")


def deformation_parameter_grid(candidate: str) -> list[dict[str, Any]]:
    if candidate in {"S6-H", "S7-H", "S15"}:
        return [{"lambda_g": ridge} for ridge in METRIC_RIDGES]
    if candidate == "S14":
        return [
            {"tau": tau, "lambda_g": ridge}
            for tau in S14_TAUS
            for ridge in METRIC_RIDGES
        ]
    raise ValueError(candidate)


def deformation_static_ineligibility(candidate: str, horizon: int) -> str | None:
    if candidate == "S7-H" and horizon == 1:
        return "INELIGIBLE_TAUTOLOGY"
    return None


def deformation_metric(
    candidate: str,
    parameters: dict[str, Any],
    model: LinearModel,
    horizon_fit: HorizonModel,
    horizon: int,
) -> np.ndarray:
    dimension = model.j.shape[0]
    ridge = float(parameters.get("lambda_g", 0.0))
    identity = np.eye(dimension)
    if candidate == "S6-H":
        metric = horizon_fit.j_h.T @ horizon_fit.j_h + ridge * identity
    elif candidate == "S7-H":
        metric = np.zeros((dimension, dimension))
        power = np.eye(dimension)
        for _ in range(horizon):
            metric += power.T @ power
            power = power @ model.j
        metric += ridge * identity
    elif candidate == "S14":
        symmetric = symmetrize(model.j)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        tau = float(parameters["tau"])
        positive = tau * np.logaddexp(0.0, eigenvalues / tau) + ridge
        metric = eigenvectors @ np.diag(positive) @ eigenvectors.T
    elif candidate == "S15":
        residual_operator = identity - model.j
        metric = residual_operator.T @ residual_operator + ridge * identity
    else:
        raise ValueError(candidate)
    return require_spd(metric, f"{candidate} metric")


def paired_separations(
    trials: tuple[np.ndarray, ...], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_items: list[np.ndarray] = []
    endpoint_items: list[float] = []
    integrated_items: list[float] = []
    for first_index in range(0, len(trials) - 1, 2):
        first = trials[first_index]
        second = trials[first_index + 1]
        first_limit = first.shape[0] - horizon
        second_limit = second.shape[0] - horizon
        if first_limit <= 0 or second_limit <= 0:
            continue
        second_initial = second[:second_limit]
        second_valid = np.isfinite(second_initial).all(axis=1)
        candidate_times = np.flatnonzero(second_valid)
        if not candidate_times.size:
            continue
        candidate_states = second_initial[candidate_times]
        for time in range(first_limit):
            if not np.isfinite(first[time : time + horizon + 1]).all():
                continue
            differences = candidate_states - first[time]
            distances = np.sum(differences * differences, axis=1)
            partner = int(candidate_times[int(np.argmin(distances))])
            if not np.isfinite(second[partner : partner + horizon + 1]).all():
                continue
            delta = first[time] - second[partner]
            endpoint = first[time + horizon] - second[partner + horizon]
            integrated = first[time : time + horizon] - second[
                partner : partner + horizon
            ]
            delta_items.append(delta)
            endpoint_items.append(float(endpoint @ endpoint))
            integrated_items.append(float(np.sum(integrated * integrated)))
    if not delta_items:
        raise CandidateFailure("INSUFFICIENT_PAIRS", "no cross-trial pair exists")
    indices = evenly_spaced_indices(len(delta_items), MAX_PAIR_SAMPLES)
    return (
        np.asarray(delta_items)[indices],
        np.asarray(endpoint_items)[indices],
        np.asarray(integrated_items)[indices],
    )


def quadratic_features(delta: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ij,nj->n", delta, metric, delta)


def fit_nonnegative_scale(feature: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(feature * feature))
    if denominator <= 0 or not math.isfinite(denominator):
        raise CandidateFailure("INELIGIBLE_ZERO_FEATURE", "quadratic feature is zero")
    numerator = float(np.sum(feature * target))
    return max(0.0, numerator / denominator)


def separation_score(
    feature: np.ndarray, target: np.ndarray, scale: float
) -> dict[str, Any]:
    prediction = scale * feature
    denominator = float(np.sqrt(np.mean(target * target)) + EPS_NUMERIC)
    nrmse = float(np.sqrt(np.mean((prediction - target) ** 2)) / denominator)
    if feature.size < 3 or np.std(feature) == 0 or np.std(target) == 0:
        rho = None
    else:
        statistic = spearmanr(feature, target).statistic
        rho = float(statistic) if math.isfinite(float(statistic)) else None
    return {"nrmse": nrmse, "spearman_rho": rho, "pair_count": int(feature.size)}


def fit_deformation_tuple(
    candidate: str,
    parameters: dict[str, Any],
    model: LinearModel,
    horizon_fit: HorizonModel,
    horizon: int,
    fit_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    inner_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    metric = deformation_metric(candidate, parameters, model, horizon_fit, horizon)
    fit_delta, fit_endpoint, fit_integrated = fit_pairs
    inner_delta, inner_endpoint, inner_integrated = inner_pairs
    target_fit = fit_integrated if candidate == "S7-H" else fit_endpoint
    target_inner = inner_integrated if candidate == "S7-H" else inner_endpoint
    fit_feature = quadratic_features(fit_delta, metric)
    scale = fit_nonnegative_scale(fit_feature, target_fit)
    inner_feature = quadratic_features(inner_delta, metric)
    inner_score = separation_score(inner_feature, target_inner, scale)
    return {
        "status": "ELIGIBLE",
        "parameters": parameters,
        "metric": metric,
        "scale": scale,
        "inner_nrmse": inner_score["nrmse"],
        "inner_spearman_rho": inner_score["spearman_rho"],
        "inner_pair_count": inner_score["pair_count"],
    }


def score_deformation_test(
    candidate: str,
    fitted: dict[str, Any],
    test_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    delta, endpoint, integrated = test_pairs
    target = integrated if candidate == "S7-H" else endpoint
    feature = quadratic_features(delta, np.asarray(fitted["metric"]))
    return separation_score(feature, target, float(fitted["scale"]))


def run_deformation_inner(
    sessions: list[Session],
    models: dict[tuple[str, str], LinearModel],
    horizons: dict[tuple[str, str, int], HorizonModel],
) -> tuple[dict[str, Any], dict[tuple[str, str, int, str, str], dict[str, Any]]]:
    report: dict[str, Any] = {}
    cache: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for session in sessions:
        session_report: dict[str, Any] = {}
        for condition, blocks in session.conditions.items():
            condition_report: dict[str, Any] = {}
            model = models[(session.session_id, condition)]
            for horizon in HORIZONS:
                horizon_report: dict[str, Any] = {}
                horizon_fit = horizons[(session.session_id, condition, horizon)]
                try:
                    fit_pairs = paired_separations(blocks.fit, horizon)
                    inner_pairs = paired_separations(blocks.inner, horizon)
                    pair_error = None
                except CandidateFailure as error:
                    fit_pairs = None
                    inner_pairs = None
                    pair_error = error
                for candidate in DEFORMATION_CANDIDATES:
                    tuples: dict[str, Any] = {}
                    for parameters in deformation_parameter_grid(candidate):
                        key = tuple_key(parameters)
                        static_failure = deformation_static_ineligibility(
                            candidate, horizon
                        )
                        if static_failure is not None:
                            tuples[key] = {
                                "status": static_failure,
                                "detail": (
                                    "with identity observation at H=1, both the feature and "
                                    "registered integrated target equal squared initial separation"
                                ),
                                "parameters": parameters,
                            }
                            continue
                        if pair_error is not None or fit_pairs is None or inner_pairs is None:
                            tuples[key] = {
                                "status": pair_error.code if pair_error else "INSUFFICIENT_PAIRS",
                                "detail": pair_error.detail if pair_error else "pair cache unavailable",
                                "parameters": parameters,
                            }
                            continue
                        try:
                            fitted = fit_deformation_tuple(
                                candidate,
                                parameters,
                                model,
                                horizon_fit,
                                horizon,
                                fit_pairs,
                                inner_pairs,
                            )
                            cache[
                                (
                                    session.session_id,
                                    condition,
                                    horizon,
                                    candidate,
                                    key,
                                )
                            ] = fitted
                            tuples[key] = {
                                field: value
                                for field, value in fitted.items()
                                if field != "metric"
                            }
                        except CandidateFailure as error:
                            tuples[key] = {
                                "status": error.code,
                                "detail": error.detail,
                                "parameters": parameters,
                            }
                    horizon_report[candidate] = tuples
                condition_report[str(horizon)] = horizon_report
            session_report[condition] = condition_report
        report[session.session_id] = session_report
    return report, cache


def select_deformation_tuple(
    raw_report: dict[str, Any],
    sessions: list[Session],
    train_animals: set[str],
    horizon: int,
    candidate: str,
) -> dict[str, Any]:
    common: set[str] | None = None
    for session in sessions:
        if session.animal not in train_animals:
            continue
        for condition in session.conditions:
            tuples = raw_report[session.session_id][condition][str(horizon)][candidate]
            eligible = {
                key for key, item in tuples.items() if item.get("status") == "ELIGIBLE"
            }
            common = eligible if common is None else common & eligible
    choices = []
    for key in sorted(common or set()):
        session_values: dict[str, float] = {}
        for session in sessions:
            if session.animal not in train_animals:
                continue
            values = [
                float(
                    raw_report[session.session_id][condition][str(horizon)][candidate][
                        key
                    ]["inner_nrmse"]
                )
                for condition in session.conditions
            ]
            session_values[session.session_id] = float(np.mean(values))
        mean, animal_means = strict_animal_mean(
            session_values, sessions, train_animals
        )
        if mean is not None:
            parameters = json.loads(key)
            dof = 0
            ridge = float(parameters.get("lambda_g", 0.0))
            choices.append((mean, dof, ridge, key, animal_means))
    if not choices:
        return {"status": "NO_ELIGIBLE_TUPLE"}
    choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = choices[0]
    return {
        "status": "SELECTED",
        "tuple_key": best[3],
        "parameters": json.loads(best[3]),
        "outer_train_inner_nrmse": best[0],
        "outer_train_animal_means": best[4],
        "eligible_tuple_count": len(choices),
    }


def run_deformation_outer(
    sessions: list[Session],
    raw_report: dict[str, Any],
    cache: dict[tuple[str, str, int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    animals = sorted({session.animal for session in sessions})
    test_pair_cache: dict[
        tuple[str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray] | CandidateFailure
    ] = {}
    for session in sessions:
        for condition, blocks in session.conditions.items():
            for horizon in HORIZONS:
                try:
                    test_pair_cache[(session.session_id, condition, horizon)] = (
                        paired_separations(blocks.test, horizon)
                    )
                except CandidateFailure as error:
                    test_pair_cache[(session.session_id, condition, horizon)] = error
    folds: dict[str, Any] = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        fold: dict[str, Any] = {}
        for horizon in HORIZONS:
            horizon_report: dict[str, Any] = {}
            for candidate in DEFORMATION_CANDIDATES:
                selection = select_deformation_tuple(
                    raw_report, sessions, train_animals, horizon, candidate
                )
                if selection["status"] != "SELECTED":
                    horizon_report[candidate] = selection
                    continue
                key = str(selection["tuple_key"])
                session_scores: dict[str, Any] = {}
                failures: list[dict[str, str]] = []
                for session in sessions:
                    if session.animal != heldout:
                        continue
                    scores = []
                    rhos = []
                    for condition, blocks in session.conditions.items():
                        fitted = cache.get(
                            (session.session_id, condition, horizon, candidate, key)
                        )
                        if fitted is None:
                            failures.append(
                                {
                                    "session": session.session_id,
                                    "condition": condition,
                                    "reason": "selected tuple unavailable",
                                }
                            )
                            continue
                        try:
                            test_pairs = test_pair_cache[
                                (session.session_id, condition, horizon)
                            ]
                            if isinstance(test_pairs, CandidateFailure):
                                raise test_pairs
                            score = score_deformation_test(
                                candidate, fitted, test_pairs
                            )
                            scores.append(float(score["nrmse"]))
                            if score["spearman_rho"] is not None:
                                rhos.append(float(score["spearman_rho"]))
                        except CandidateFailure as error:
                            failures.append(
                                {
                                    "session": session.session_id,
                                    "condition": condition,
                                    "reason": f"{error.code}: {error.detail}",
                                }
                            )
                    if len(scores) == len(session.conditions):
                        session_scores[session.session_id] = {
                            "nrmse": float(np.mean(scores)),
                            "spearman_rho": aggregate_equal(rhos),
                        }
                complete = len(session_scores) == sum(
                    session.animal == heldout for session in sessions
                )
                horizon_report[candidate] = {
                    **selection,
                    "status": "EVALUATED" if complete else "HELDOUT_EVALUATION_INCOMPLETE",
                    "heldout_animal": heldout,
                    "heldout_animal_nrmse": (
                        float(np.mean([item["nrmse"] for item in session_scores.values()]))
                        if complete
                        else None
                    ),
                    "session_scores": session_scores,
                    "failures": failures,
                }
            fold[str(horizon)] = horizon_report
        folds[heldout] = fold
    return {"folds": folds, "status": "RETROSPECTIVE_DISCOVERY_NO_WINNER"}


def flatten_labeled_states(blocks: dict[str, tuple[np.ndarray, ...]]) -> tuple[np.ndarray, np.ndarray]:
    saline = finite_rows(blocks["saline"])
    dcz = finite_rows(blocks["dcz"])
    x = np.vstack([saline, dcz])
    y = np.concatenate([np.zeros(saline.shape[0]), np.ones(dcz.shape[0])])
    return x, y


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -value))


def balanced_log_loss(probability: np.ndarray, label: np.ndarray) -> float:
    losses = []
    for target in (0.0, 1.0):
        keep = label == target
        if not np.any(keep):
            raise CandidateFailure("INSUFFICIENT_PAIRS", "decoder class is empty")
        p = np.clip(probability[keep], EPS_NUMERIC, 1.0 - EPS_NUMERIC)
        if target == 1.0:
            losses.append(float(np.mean(-np.log(p))))
        else:
            losses.append(float(np.mean(-np.log(1.0 - p))))
    return float(np.mean(losses))


def fit_condition_decoder(
    x: np.ndarray, y: np.ndarray, penalty: float
) -> dict[str, Any]:
    dimension = x.shape[1]

    def objective(vector: np.ndarray) -> float:
        w = vector[:dimension]
        bias = vector[-1]
        probability = sigmoid(x @ w + bias)
        return balanced_log_loss(probability, y) + 0.5 * penalty * float(w @ w)

    result = minimize(
        objective,
        np.zeros(dimension + 1),
        method="L-BFGS-B",
        jac="2-point",
        bounds=None,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
            "finite_diff_rel_step": 1e-6,
        },
    )
    if not result.success or not np.isfinite(result.x).all():
        raise CandidateFailure(
            "OPTIMIZER_FAILED", f"decoder optimizer failed: {result.message}"
        )
    return {
        "w": result.x[:dimension],
        "bias": float(result.x[-1]),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def condition_field_gates(
    w: np.ndarray, probability: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Evaluate S8/S9 algebraic SPD gates on calibration states only."""
    w = np.asarray(w, dtype=float)
    probability = np.asarray(probability, dtype=float)
    require_finite(w, "decoder weights")
    require_finite(probability, "fit decoder probabilities")
    dimension = int(w.size)
    identity = np.eye(dimension)
    sampled = probability[evenly_spaced_indices(probability.size, MAX_SCORE_SAMPLES)]
    output: dict[str, dict[str, Any]] = {"S8": {}, "S9": {}}
    for ridge in METRIC_RIDGES:
        parameters = {"lambda_g": ridge}
        key = tuple_key(parameters)
        zero_ridge_singular = ridge == 0.0 and (
            dimension > 1 or not np.any(w != 0.0)
        )

        fisher_minimum = math.inf
        fisher_logdets: list[float] = []
        fisher_singular = zero_ridge_singular
        for p in sampled:
            coefficient = float(p * (1.0 - p))
            fisher = symmetrize(coefficient * np.outer(w, w) + ridge * identity)
            eigenvalues = np.linalg.eigvalsh(fisher)
            fisher_minimum = min(fisher_minimum, float(np.min(eigenvalues)))
            if ridge == 0.0 and coefficient <= 0.0:
                fisher_singular = True
            if not fisher_singular:
                sign, logdet = np.linalg.slogdet(fisher)
                if sign <= 0:
                    fisher_singular = True
                else:
                    fisher_logdets.append(float(logdet))
        output["S8"][key] = {
            "status": "INELIGIBLE_SINGULAR" if fisher_singular else "ELIGIBLE",
            "parameters": parameters,
            "minimum_eigenvalue": 0.0 if fisher_singular else fisher_minimum,
            "mean_logdet": None if fisher_singular else aggregate_equal(fisher_logdets),
            "source_block": "session_fit_only",
            "role": "FIELD_GATE_ONLY_NO_INDEPENDENT_PREDICTIVE_ENDPOINT",
        }

        pullback = symmetrize(np.outer(w, w) + ridge * identity)
        pullback_eigenvalues = np.linalg.eigvalsh(pullback)
        pullback_singular = zero_ridge_singular
        output["S9"][key] = {
            "status": "INELIGIBLE_SINGULAR" if pullback_singular else "ELIGIBLE",
            "parameters": parameters,
            "minimum_eigenvalue": (
                0.0
                if pullback_singular
                else float(np.min(pullback_eigenvalues))
            ),
            "source_block": "session_fit_only",
            "role": "FIELD_GATE_ONLY_NO_INDEPENDENT_PREDICTIVE_ENDPOINT",
        }
    return output


def failed_condition_field_gates(
    code: str, detail: str
) -> dict[str, dict[str, Any]]:
    return {
        candidate: {
            tuple_key({"lambda_g": ridge}): {
                "status": code,
                "detail": f"decoder unavailable: {detail}",
                "parameters": {"lambda_g": ridge},
                "source_block": "session_fit_only",
                "role": "FIELD_GATE_ONLY_NO_INDEPENDENT_PREDICTIVE_ENDPOINT",
            }
            for ridge in METRIC_RIDGES
        }
        for candidate in ("S8", "S9")
    }


def run_condition_information(sessions: list[Session]) -> dict[str, Any]:
    raw_inner: dict[str, Any] = {}
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for session in sessions:
        fit_x, fit_y = flatten_labeled_states(
            {
                condition: blocks.fit
                for condition, blocks in session.conditions.items()
            }
        )
        inner_x, inner_y = flatten_labeled_states(
            {
                condition: blocks.inner
                for condition, blocks in session.conditions.items()
            }
        )
        tuples = {}
        for penalty in DECODER_PENALTIES:
            parameters = {"decoder_l2": penalty}
            key = tuple_key(parameters)
            try:
                fitted = fit_condition_decoder(fit_x, fit_y, penalty)
                probability = sigmoid(inner_x @ fitted["w"] + fitted["bias"])
                score = balanced_log_loss(probability, inner_y)
                fit_probability = sigmoid(fit_x @ fitted["w"] + fitted["bias"])
                cache[(session.session_id, key)] = fitted
                tuples[key] = {
                    "status": "ELIGIBLE",
                    "parameters": parameters,
                    "inner_balanced_log_loss": score,
                    "iterations": fitted["iterations"],
                    "fit_field_gates": condition_field_gates(
                        np.asarray(fitted["w"]), fit_probability
                    ),
                }
            except CandidateFailure as error:
                tuples[key] = {
                    "status": error.code,
                    "detail": error.detail,
                    "parameters": parameters,
                    "fit_field_gates": failed_condition_field_gates(
                        error.code, error.detail
                    ),
                }
        raw_inner[session.session_id] = tuples

    animals = sorted({session.animal for session in sessions})
    folds: dict[str, Any] = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        common: set[str] | None = None
        for session in sessions:
            if session.animal not in train_animals:
                continue
            eligible = {
                key
                for key, item in raw_inner[session.session_id].items()
                if item["status"] == "ELIGIBLE"
            }
            common = eligible if common is None else common & eligible
        choices = []
        for key in sorted(common or set()):
            values = {
                session.session_id: float(
                    raw_inner[session.session_id][key]["inner_balanced_log_loss"]
                )
                for session in sessions
                if session.animal in train_animals
            }
            mean, animal_means = strict_animal_mean(values, sessions, train_animals)
            if mean is not None:
                choices.append((mean, key, animal_means))
        if not choices:
            folds[heldout] = {"status": "NO_ELIGIBLE_TUPLE"}
            continue
        choices.sort(key=lambda item: (item[0], item[1]))
        selected = choices[0]
        key = selected[1]
        session_scores: dict[str, Any] = {}
        field_gates: dict[str, Any] = {}
        for session in sessions:
            if session.animal != heldout:
                continue
            fitted = cache[(session.session_id, key)]
            test_x, test_y = flatten_labeled_states(
                {
                    condition: blocks.test
                    for condition, blocks in session.conditions.items()
                }
            )
            probability = sigmoid(test_x @ fitted["w"] + fitted["bias"])
            session_scores[session.session_id] = balanced_log_loss(probability, test_y)
            field_gates[session.session_id] = raw_inner[session.session_id][key][
                "fit_field_gates"
            ]
        folds[heldout] = {
            "status": "EVALUATED",
            "selected_decoder_tuple": json.loads(key),
            "outer_train_inner_balanced_log_loss": selected[0],
            "outer_train_animal_means": selected[2],
            "heldout_animal_balanced_log_loss": float(np.mean(list(session_scores.values()))),
            "session_scores": session_scores,
            "field_gates": field_gates,
            "shared_endpoint_warning": "S8 and S9 share one decoder log loss and are not independent successes.",
            "field_gate_role": "fit-only algebraic diagnostic; lambda_g is not a predictive tournament tuple",
        }
    return {
        "raw_inner": raw_inner,
        "folds": folds,
        "status": "CONDITION_LABEL_DECODER_WITH_FIT_ONLY_FIELD_GATES",
        "strict_metric_tournament_status": "NOT_EVALUATED_NO_INDEPENDENT_METRIC_ENDPOINT",
    }


GRAPH_CANDIDATES = ("G1", "G2", "G3a", "G3b")


def graph_parameter_grid(candidate: str) -> list[dict[str, Any]]:
    if candidate == "G1":
        return [{"symmetrization": name} for name in GRAPH_SYMMETRIZATIONS]
    if candidate == "G2":
        return [
            {"symmetrization": name, "diffusion_time": time}
            for name in GRAPH_SYMMETRIZATIONS
            for time in DIFFUSION_TIMES
        ]
    if candidate == "G3a":
        return [{"epsilon_p": 1e-6}]
    if candidate == "G3b":
        return [{"epsilon_w": 1e-3}]
    raise ValueError(candidate)


def symmetric_conductance(j: np.ndarray, rule: str) -> np.ndarray:
    if rule == "absolute_mean":
        conductance = (np.abs(j) + np.abs(j.T)) / 2.0
    elif rule == "positive_mean":
        conductance = (np.maximum(j, 0.0) + np.maximum(j.T, 0.0)) / 2.0
    else:
        raise ValueError(rule)
    conductance = symmetrize(conductance)
    np.fill_diagonal(conductance, 0.0)
    return conductance


def connected_support(conductance: np.ndarray) -> bool:
    dimension = conductance.shape[0]
    if dimension == 0:
        return False
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(conductance[node] > 0)
        for neighbor in neighbors:
            value = int(neighbor)
            if value not in seen:
                seen.add(value)
                stack.append(value)
    return len(seen) == dimension


def effective_resistance(conductance: np.ndarray) -> np.ndarray:
    degree = np.sum(conductance, axis=1)
    if np.any(degree <= 0) or not connected_support(conductance):
        raise CandidateFailure(
            "INELIGIBLE_GRAPH_DISCONNECTED", "G1 conductance support is disconnected"
        )
    laplacian = np.diag(degree) - conductance
    pseudoinverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(pseudoinverse)
    squared = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
    squared = np.maximum(symmetrize(squared), 0.0)
    return np.sqrt(squared)


def diffusion_distance(
    conductance: np.ndarray, diffusion_time: int
) -> np.ndarray:
    degree = np.sum(conductance, axis=1)
    if np.any(degree <= 0) or not connected_support(conductance):
        raise CandidateFailure(
            "INELIGIBLE_GRAPH_KERNEL", "G2 graph has zero degree or disconnected support"
        )
    total_degree = float(np.sum(degree))
    transition = conductance / degree[:, None]
    stationary = degree / total_degree
    balance_error = float(
        np.max(
            np.abs(
                stationary[:, None] * transition
                - stationary[None, :] * transition.T
            )
        )
    )
    if balance_error > 1e-10:
        raise CandidateFailure(
            "INELIGIBLE_GRAPH_KERNEL",
            f"G2 detailed-balance residual {balance_error}",
        )
    inverse_sqrt = np.diag(1.0 / np.sqrt(degree))
    symmetric = inverse_sqrt @ conductance @ inverse_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrize(symmetric))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    stationary_modes = np.flatnonzero(np.isclose(eigenvalues, 1.0, atol=1e-10, rtol=0.0))
    if stationary_modes.size != 1:
        raise CandidateFailure(
            "INELIGIBLE_GRAPH_KERNEL",
            f"G2 stationary multiplicity is {stationary_modes.size}",
        )
    stationary_index = int(stationary_modes[0])
    if np.sum(eigenvectors[:, stationary_index]) < 0:
        eigenvectors[:, stationary_index] *= -1.0
    psi = math.sqrt(total_degree) * inverse_sqrt @ eigenvectors
    keep = [index for index in range(eigenvalues.size) if index != stationary_index]
    coordinates = psi[:, keep] * (
        np.asarray(eigenvalues[keep]) ** int(diffusion_time)
    )[None, :]
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(differences, axis=2)


def directed_graph_distance(j: np.ndarray, candidate: str) -> np.ndarray:
    dimension = j.shape[0]
    cost = np.full((dimension, dimension), np.inf)
    np.fill_diagonal(cost, 0.0)
    off_diagonal = [
        abs(float(j[target, source]))
        for source in range(dimension)
        for target in range(dimension)
        if target != source and abs(float(j[target, source])) > 0
    ]
    if candidate == "G3b" and not off_diagonal:
        raise CandidateFailure(
            "INELIGIBLE_GRAPH_DISCONNECTED", "G3b has no positive off-diagonal scale"
        )
    scale = float(np.median(off_diagonal)) if off_diagonal else 0.0
    for source in range(dimension):
        strengths = np.array(
            [
                abs(float(j[target, source])) if target != source else 0.0
                for target in range(dimension)
            ]
        )
        total = float(np.sum(strengths))
        for target in range(dimension):
            strength = strengths[target]
            if target == source or strength <= 0:
                continue
            if candidate == "G3a":
                probability = strength / total if total > 0 else 0.0
                cost[source, target] = -math.log(max(probability, 1e-6))
            elif candidate == "G3b":
                cost[source, target] = scale / (strength + 1e-3 * scale)
            else:
                raise ValueError(candidate)
    return np.asarray(shortest_path(cost, directed=True, unweighted=False), dtype=float)


def zero_lag_association(trials: tuple[np.ndarray, ...]) -> np.ndarray:
    states = finite_rows(trials)
    return np.abs(np.corrcoef(states, rowvar=False))


def lag_one_association(trials: tuple[np.ndarray, ...]) -> np.ndarray:
    x, y = transition_pairs(trials, 1)
    dimension = x.shape[1]
    result = np.full((dimension, dimension), np.nan)
    for source in range(dimension):
        for target in range(dimension):
            if source == target:
                continue
            left = x[:, source]
            right = y[:, target]
            if np.std(left) > 0 and np.std(right) > 0:
                result[source, target] = abs(float(np.corrcoef(left, right)[0, 1]))
    return result


def graph_association_score(
    candidate: str, distance: np.ndarray, trials: tuple[np.ndarray, ...]
) -> dict[str, Any]:
    dimension = distance.shape[0]
    if candidate in {"G1", "G2"}:
        association = zero_lag_association(trials)
        pairs = [(left, right) for left in range(dimension) for right in range(left + 1, dimension)]
    else:
        association = lag_one_association(trials)
        pairs = [(source, target) for source in range(dimension) for target in range(dimension) if source != target]
    distances = []
    associations = []
    for left, right in pairs:
        value_d = float(distance[left, right])
        value_a = float(association[left, right])
        if math.isfinite(value_d) and math.isfinite(value_a):
            distances.append(value_d)
            associations.append(value_a)
    if (
        len(distances) < 3
        or np.std(distances) == 0
        or np.std(associations) == 0
    ):
        raise CandidateFailure(
            "INSUFFICIENT_PAIRS", "fewer than three finite nonconstant graph pairs"
        )
    test = spearmanr(distances, associations)
    if not math.isfinite(float(test.statistic)):
        raise CandidateFailure("INSUFFICIENT_PAIRS", "Spearman statistic is nonfinite")
    return {
        "spearman_rho": float(test.statistic),
        "pair_count": len(distances),
    }


def graph_distance_for(
    candidate: str, parameters: dict[str, Any], j: np.ndarray
) -> np.ndarray:
    if candidate in {"G1", "G2"}:
        conductance = symmetric_conductance(j, str(parameters["symmetrization"]))
        if candidate == "G1":
            return effective_resistance(conductance)
        return diffusion_distance(conductance, int(parameters["diffusion_time"]))
    return directed_graph_distance(j, candidate)


def run_graph_tournament(
    sessions: list[Session], models: dict[tuple[str, str], LinearModel]
) -> dict[str, Any]:
    raw_inner: dict[str, Any] = {}
    distance_cache: dict[tuple[str, str, str, str], np.ndarray] = {}
    for session in sessions:
        session_report: dict[str, Any] = {}
        for condition, blocks in session.conditions.items():
            condition_report: dict[str, Any] = {}
            j = models[(session.session_id, condition)].j
            for candidate in GRAPH_CANDIDATES:
                tuples = {}
                for parameters in graph_parameter_grid(candidate):
                    key = tuple_key(parameters)
                    try:
                        distance = graph_distance_for(candidate, parameters, j)
                        score = graph_association_score(candidate, distance, blocks.inner)
                        distance_cache[(session.session_id, condition, candidate, key)] = distance
                        tuples[key] = {
                            "status": "ELIGIBLE",
                            "parameters": parameters,
                            "inner_spearman_rho": score["spearman_rho"],
                            "inner_pair_count": score["pair_count"],
                        }
                    except CandidateFailure as error:
                        tuples[key] = {
                            "status": error.code,
                            "detail": error.detail,
                            "parameters": parameters,
                        }
                condition_report[candidate] = tuples
            session_report[condition] = condition_report
        raw_inner[session.session_id] = session_report

    animals = sorted({session.animal for session in sessions})
    folds: dict[str, Any] = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        fold = {}
        for candidate in GRAPH_CANDIDATES:
            common: set[str] | None = None
            for session in sessions:
                if session.animal not in train_animals:
                    continue
                for condition in session.conditions:
                    eligible = {
                        key
                        for key, item in raw_inner[session.session_id][condition][candidate].items()
                        if item["status"] == "ELIGIBLE"
                    }
                    common = eligible if common is None else common & eligible
            choices = []
            for key in sorted(common or set()):
                values: dict[str, float] = {}
                for session in sessions:
                    if session.animal not in train_animals:
                        continue
                    values[session.session_id] = float(
                        np.mean(
                            [
                                raw_inner[session.session_id][condition][candidate][key][
                                    "inner_spearman_rho"
                                ]
                                for condition in session.conditions
                            ]
                        )
                    )
                mean, animal_means = strict_animal_mean(values, sessions, train_animals)
                if mean is not None:
                    choices.append((mean, key, animal_means))
            if not choices:
                fold[candidate] = {"status": "NO_ELIGIBLE_TUPLE"}
                continue
            choices.sort(key=lambda item: (item[0], item[1]))
            selected = choices[0]
            key = selected[1]
            session_scores: dict[str, float] = {}
            failures = []
            for session in sessions:
                if session.animal != heldout:
                    continue
                scores = []
                for condition, blocks in session.conditions.items():
                    distance = distance_cache.get(
                        (session.session_id, condition, candidate, key)
                    )
                    if distance is None:
                        failures.append(
                            {
                                "session": session.session_id,
                                "condition": condition,
                                "reason": "selected graph tuple unavailable",
                            }
                        )
                        continue
                    try:
                        score = graph_association_score(candidate, distance, blocks.test)
                        scores.append(float(score["spearman_rho"]))
                    except CandidateFailure as error:
                        failures.append(
                            {
                                "session": session.session_id,
                                "condition": condition,
                                "reason": f"{error.code}: {error.detail}",
                            }
                        )
                if len(scores) == len(session.conditions):
                    session_scores[session.session_id] = float(np.mean(scores))
            complete = len(session_scores) == sum(
                session.animal == heldout for session in sessions
            )
            fold[candidate] = {
                "status": "EVALUATED" if complete else "HELDOUT_EVALUATION_INCOMPLETE",
                "selected_tuple": json.loads(key),
                "outer_train_inner_spearman_rho": selected[0],
                "outer_train_animal_means": selected[2],
                "heldout_animal_spearman_rho": (
                    float(np.mean(list(session_scores.values()))) if complete else None
                ),
                "session_scores": session_scores,
                "failures": failures,
            }
        folds[heldout] = fold
    return {
        "raw_inner": raw_inner,
        "folds": folds,
        "direction": "more negative Spearman rho is the preregistered direction",
        "status": "TECHNICAL_GRAPH_PROXY_NO_POPULATION_WINNER",
    }


def one_step_path_residuals(
    trials: tuple[np.ndarray, ...], model: LinearModel
) -> np.ndarray:
    x, y = transition_pairs(trials, 1)
    return y - (x @ model.j.T + model.bias)


def permute_trial_time(
    trials: tuple[np.ndarray, ...], mode: str
) -> tuple[np.ndarray, ...]:
    if mode == "reverse":
        return tuple(trial[::-1].copy() for trial in trials)
    if mode != "shuffle":
        raise ValueError(mode)
    rng = np.random.Generator(np.random.PCG64(SEED))
    shuffled = []
    for trial in trials:
        order = rng.permutation(trial.shape[0])
        shuffled.append(trial[order])
    return tuple(shuffled)


def run_directional_action(
    sessions: list[Session], models: dict[tuple[str, str], LinearModel]
) -> dict[str, Any]:
    raw_inner: dict[str, Any] = {}
    covariance_cache: dict[tuple[str, str, str], np.ndarray] = {}
    for session in sessions:
        session_report: dict[str, Any] = {}
        for condition, blocks in session.conditions.items():
            model = models[(session.session_id, condition)]
            tuples = {}
            inner_residual = one_step_path_residuals(blocks.inner, model)
            for ridge in COVARIANCE_RIDGES:
                parameters = {"lambda_c": ridge}
                key = tuple_key(parameters)
                try:
                    covariance = require_spd(
                        model.q + ridge * np.eye(session.dimension),
                        "D1 process covariance",
                    )
                    covariance_cache[(session.session_id, condition, key)] = covariance
                    tuples[key] = {
                        "status": "ELIGIBLE",
                        "parameters": parameters,
                        "inner_forward_nlpd": gaussian_nlpd(
                            inner_residual, covariance
                        ),
                    }
                except CandidateFailure as error:
                    tuples[key] = {
                        "status": error.code,
                        "detail": error.detail,
                        "parameters": parameters,
                    }
            session_report[condition] = tuples
        raw_inner[session.session_id] = session_report

    animals = sorted({session.animal for session in sessions})
    folds: dict[str, Any] = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        common: set[str] | None = None
        for session in sessions:
            if session.animal not in train_animals:
                continue
            for condition in session.conditions:
                eligible = {
                    key
                    for key, item in raw_inner[session.session_id][condition].items()
                    if item["status"] == "ELIGIBLE"
                }
                common = eligible if common is None else common & eligible
        choices = []
        for key in sorted(common or set()):
            session_values: dict[str, float] = {}
            for session in sessions:
                if session.animal not in train_animals:
                    continue
                session_values[session.session_id] = float(
                    np.mean(
                        [
                            raw_inner[session.session_id][condition][key][
                                "inner_forward_nlpd"
                            ]
                            for condition in session.conditions
                        ]
                    )
                )
            mean, animal_means = strict_animal_mean(
                session_values, sessions, train_animals
            )
            if mean is not None:
                ridge = float(json.loads(key)["lambda_c"])
                choices.append((mean, ridge, key, animal_means))
        if not choices:
            folds[heldout] = {"status": "NO_ELIGIBLE_TUPLE"}
            continue
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        selected = choices[0]
        key = selected[2]
        session_scores: dict[str, Any] = {}
        failures = []
        for session in sessions:
            if session.animal != heldout:
                continue
            condition_scores = []
            for condition, blocks in session.conditions.items():
                covariance = covariance_cache.get((session.session_id, condition, key))
                if covariance is None:
                    failures.append(
                        {
                            "session": session.session_id,
                            "condition": condition,
                            "reason": "selected D1 tuple unavailable",
                        }
                    )
                    continue
                model = models[(session.session_id, condition)]
                forward = gaussian_nlpd(
                    one_step_path_residuals(blocks.test, model), covariance
                )
                reverse = gaussian_nlpd(
                    one_step_path_residuals(
                        permute_trial_time(blocks.test, "reverse"), model
                    ),
                    covariance,
                )
                shuffled = gaussian_nlpd(
                    one_step_path_residuals(
                        permute_trial_time(blocks.test, "shuffle"), model
                    ),
                    covariance,
                )
                condition_scores.append(
                    {
                        "forward_nlpd": forward,
                        "reverse_nlpd": reverse,
                        "shuffle_nlpd": shuffled,
                        "reverse_minus_forward": reverse - forward,
                        "shuffle_minus_forward": shuffled - forward,
                    }
                )
            if len(condition_scores) == len(session.conditions):
                session_scores[session.session_id] = {
                    key_name: float(np.mean([item[key_name] for item in condition_scores]))
                    for key_name in condition_scores[0]
                }
        complete = len(session_scores) == sum(
            session.animal == heldout for session in sessions
        )
        folds[heldout] = {
            "status": "EVALUATED" if complete else "HELDOUT_EVALUATION_INCOMPLETE",
            "selected_tuple": json.loads(key),
            "outer_train_inner_forward_nlpd": selected[0],
            "outer_train_animal_means": selected[3],
            "heldout_animal_reverse_minus_forward": (
                float(
                    np.mean(
                        [item["reverse_minus_forward"] for item in session_scores.values()]
                    )
                )
                if complete
                else None
            ),
            "heldout_animal_shuffle_minus_forward": (
                float(
                    np.mean(
                        [item["shuffle_minus_forward"] for item in session_scores.values()]
                    )
                )
                if complete
                else None
            ),
            "session_scores": session_scores,
            "failures": failures,
        }
    return {
        "raw_inner": raw_inner,
        "folds": folds,
        "controls": "same frozen forward model; reversed and PCG64(1729) within-trial shuffle",
        "status": "DISCRETE_ACTION_RETROSPECTIVE_DISCOVERY",
    }


def ground_metric_for(session: Session, name: str) -> np.ndarray:
    if name == "identity":
        return np.eye(session.dimension)
    if name != "fit_state_covariance_precision":
        raise ValueError(name)
    fit_states = finite_rows(
        session.conditions["saline"].fit + session.conditions["dcz"].fit
    )
    covariance = covariance_mle(fit_states) + EPS_G * np.eye(session.dimension)
    return np.linalg.inv(require_spd(covariance, "W2 fit-state covariance"))


def sample_states(trials: tuple[np.ndarray, ...]) -> np.ndarray:
    states = finite_rows(trials)
    return states[evenly_spaced_indices(states.shape[0], MAX_SCORE_SAMPLES)]


def squared_ground_cost(
    left: np.ndarray, right: np.ndarray, ground_metric: np.ndarray
) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    cost = np.einsum("abi,ij,abj->ab", difference, ground_metric, difference)
    return np.maximum(cost, 0.0)


def empirical_w2(
    left_trials: tuple[np.ndarray, ...],
    right_trials: tuple[np.ndarray, ...],
    ground_metric: np.ndarray,
) -> float:
    left = sample_states(left_trials)
    right = sample_states(right_trials)
    sample_count = min(left.shape[0], right.shape[0])
    if sample_count < 2:
        raise CandidateFailure("INSUFFICIENT_PAIRS", "W2 needs two states per condition")
    left = left[evenly_spaced_indices(left.shape[0], sample_count)]
    right = right[evenly_spaced_indices(right.shape[0], sample_count)]
    cost = squared_ground_cost(left, right, ground_metric)
    row, column = linear_sum_assignment(cost)
    return float(np.sqrt(np.mean(cost[row, column])))


def wasserstein_permutation_score(
    saline: tuple[np.ndarray, ...],
    dcz: tuple[np.ndarray, ...],
    ground_metric: np.ndarray,
) -> dict[str, Any]:
    observed = empirical_w2(saline, dcz, ground_metric)
    pooled = list(saline + dcz)
    saline_count = len(saline)
    rng = np.random.Generator(np.random.PCG64(SEED))
    null = np.empty(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        order = rng.permutation(len(pooled))
        permuted_saline = tuple(pooled[int(item)] for item in order[:saline_count])
        permuted_dcz = tuple(pooled[int(item)] for item in order[saline_count:])
        null[index] = empirical_w2(permuted_saline, permuted_dcz, ground_metric)
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1))
    z_score = (observed - null_mean) / null_sd if null_sd > 0 else None
    p_value = float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1))
    return {
        "w2": observed,
        "permutation_mean": null_mean,
        "permutation_sd": null_sd,
        "permutation_z": z_score,
        "one_sided_p": p_value,
        "permutations": PERMUTATIONS,
    }


def run_distribution_tournament(sessions: list[Session]) -> dict[str, Any]:
    raw_inner: dict[str, Any] = {}
    for session in sessions:
        tuples = {}
        for name in GROUND_METRICS:
            parameters = {"ground_metric": name}
            key = tuple_key(parameters)
            try:
                metric = ground_metric_for(session, name)
                score = wasserstein_permutation_score(
                    session.conditions["saline"].inner,
                    session.conditions["dcz"].inner,
                    metric,
                )
                tuples[key] = {
                    "status": "ELIGIBLE",
                    "parameters": parameters,
                    **score,
                }
            except CandidateFailure as error:
                tuples[key] = {
                    "status": error.code,
                    "detail": error.detail,
                    "parameters": parameters,
                }
        raw_inner[session.session_id] = tuples

    animals = sorted({session.animal for session in sessions})
    folds = {}
    for heldout in animals:
        train_animals = set(animals) - {heldout}
        common: set[str] | None = None
        for session in sessions:
            if session.animal not in train_animals:
                continue
            eligible = {
                key
                for key, item in raw_inner[session.session_id].items()
                if item["status"] == "ELIGIBLE" and item["permutation_z"] is not None
            }
            common = eligible if common is None else common & eligible
        choices = []
        for key in sorted(common or set()):
            values = {
                session.session_id: -float(raw_inner[session.session_id][key]["permutation_z"])
                for session in sessions
                if session.animal in train_animals
            }
            mean, animal_means = strict_animal_mean(values, sessions, train_animals)
            if mean is not None:
                choices.append((mean, key, animal_means))
        if not choices:
            folds[heldout] = {"status": "NO_ELIGIBLE_TUPLE"}
            continue
        choices.sort(key=lambda item: (item[0], item[1]))
        selected = choices[0]
        key = selected[1]
        name = str(json.loads(key)["ground_metric"])
        session_scores = {}
        failures = []
        for session in sessions:
            if session.animal != heldout:
                continue
            try:
                score = wasserstein_permutation_score(
                    session.conditions["saline"].test,
                    session.conditions["dcz"].test,
                    ground_metric_for(session, name),
                )
                session_scores[session.session_id] = score
            except CandidateFailure as error:
                failures.append(
                    {
                        "session": session.session_id,
                        "reason": f"{error.code}: {error.detail}",
                    }
                )
        complete = len(session_scores) == sum(
            session.animal == heldout for session in sessions
        )
        folds[heldout] = {
            "status": "EVALUATED" if complete else "HELDOUT_EVALUATION_INCOMPLETE",
            "selected_tuple": json.loads(key),
            "outer_train_mean_negative_permutation_z": selected[0],
            "outer_train_animal_means": selected[2],
            "heldout_animal_mean_w2": (
                float(np.mean([item["w2"] for item in session_scores.values()]))
                if complete
                else None
            ),
            "heldout_animal_mean_permutation_z": (
                float(
                    np.mean(
                        [item["permutation_z"] for item in session_scores.values()]
                    )
                )
                if complete
                else None
            ),
            "session_scores": session_scores,
            "failures": failures,
        }
    return {
        "raw_inner": raw_inner,
        "folds": folds,
        "status": "DISTRIBUTION_SHIFT_ONLY_NOT_A_STATE_METRIC",
    }


def collect_statuses(value: Any) -> list[str]:
    statuses: list[str] = []
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str):
            statuses.append(status)
        for item in value.values():
            statuses.extend(collect_statuses(item))
    elif isinstance(value, list):
        for item in value:
            statuses.extend(collect_statuses(item))
    return statuses


def tuple_execution_audit(
    sessions: list[Session],
    uncertainty_raw: dict[str, Any],
    deformation_raw: dict[str, Any],
    condition_information: dict[str, Any],
    graph: dict[str, Any],
    directional: dict[str, Any],
    distribution: dict[str, Any],
) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cell_count": 0,
            "expected_tuple_count": 0,
            "observed_tuple_count": 0,
            "missing_tuple_count": 0,
            "extra_tuple_count": 0,
            "duplicate_expected_key_count": 0,
        }
    )
    failures: list[dict[str, Any]] = []

    def check(
        family: str,
        cell: str,
        expected_parameters: Iterable[dict[str, Any]],
        observed: dict[str, Any],
    ) -> None:
        expected_list = [tuple_key(item) for item in expected_parameters]
        expected = set(expected_list)
        observed_keys = set(observed)
        missing = sorted(expected - observed_keys)
        extra = sorted(observed_keys - expected)
        duplicate_count = len(expected_list) - len(expected)
        summary = families[family]
        summary["cell_count"] += 1
        summary["expected_tuple_count"] += len(expected_list)
        summary["observed_tuple_count"] += len(observed)
        summary["missing_tuple_count"] += len(missing)
        summary["extra_tuple_count"] += len(extra)
        summary["duplicate_expected_key_count"] += duplicate_count
        if missing or extra or duplicate_count:
            failures.append(
                {
                    "family": family,
                    "cell": cell,
                    "missing": missing,
                    "extra": extra,
                    "duplicate_expected_key_count": duplicate_count,
                }
            )

    for session in sessions:
        for condition in session.conditions:
            for horizon in HORIZONS:
                uncertainty_cell = uncertainty_raw[session.session_id][condition][str(horizon)]
                for candidate in UNCERTAINTY_CANDIDATES:
                    expected = uncertainty_parameter_grid(candidate, session.dimension)
                    check(
                        "uncertainty",
                        f"{session.session_id}/{condition}/H{horizon}/{candidate}",
                        expected or [{}],
                        uncertainty_cell[candidate],
                    )
                deformation_cell = deformation_raw[session.session_id][condition][str(horizon)]
                for candidate in DEFORMATION_CANDIDATES:
                    check(
                        "deformation",
                        f"{session.session_id}/{condition}/H{horizon}/{candidate}",
                        deformation_parameter_grid(candidate),
                        deformation_cell[candidate],
                    )

            graph_cell = graph["raw_inner"][session.session_id][condition]
            for candidate in GRAPH_CANDIDATES:
                check(
                    "graph",
                    f"{session.session_id}/{condition}/{candidate}",
                    graph_parameter_grid(candidate),
                    graph_cell[candidate],
                )
            check(
                "directional",
                f"{session.session_id}/{condition}/D1",
                ({"lambda_c": ridge} for ridge in COVARIANCE_RIDGES),
                directional["raw_inner"][session.session_id][condition],
            )

        decoder_cell = condition_information["raw_inner"][session.session_id]
        check(
            "condition_decoder",
            f"{session.session_id}/decoder",
            ({"decoder_l2": penalty} for penalty in DECODER_PENALTIES),
            decoder_cell,
        )
        for decoder_key in sorted(decoder_cell):
            fields = decoder_cell[decoder_key]["fit_field_gates"]
            for candidate in ("S8", "S9"):
                check(
                    "condition_field",
                    f"{session.session_id}/{decoder_key}/{candidate}",
                    ({"lambda_g": ridge} for ridge in METRIC_RIDGES),
                    fields[candidate],
                )
        check(
            "distribution",
            f"{session.session_id}/P1-P2",
            ({"ground_metric": name} for name in GROUND_METRICS),
            distribution["raw_inner"][session.session_id],
        )

    if failures:
        raise RuntimeError(
            "frozen tuple execution is incomplete: "
            + json.dumps(failures[:20], sort_keys=True)
        )
    return {
        "status": "PASS_EXACT_EXPECTED_TUPLE_KEYS",
        "all_cells_complete": True,
        "families": dict(sorted(families.items())),
        "failure_count": 0,
    }


def candidate_coverage(
    registry: dict[str, Any],
    uncertainty_raw: dict[str, Any],
    uncertainty_outer: dict[str, Any],
    deformation_raw: dict[str, Any],
    deformation_outer: dict[str, Any],
    condition_information: dict[str, Any],
    graph: dict[str, Any],
    directional: dict[str, Any],
    distribution: dict[str, Any],
) -> dict[str, Any]:
    def runtime_statuses(identifier: str) -> list[str]:
        values: list[str] = []
        if identifier in {"S0", "S1", "S2", "S3", "S4-H", "S5", "S12", "S13"}:
            for session_report in uncertainty_raw.values():
                for condition_report in session_report.values():
                    for horizon_report in condition_report.values():
                        values.extend(collect_statuses(horizon_report.get(identifier, {})))
        elif identifier in {"S6-H", "S7-H", "S14", "S15"}:
            for session_report in deformation_raw.values():
                for condition_report in session_report.values():
                    for horizon_report in condition_report.values():
                        values.extend(collect_statuses(horizon_report.get(identifier, {})))
        elif identifier in {"S8", "S9"}:
            for decoder_tuples in condition_information.get("raw_inner", {}).values():
                for decoder in decoder_tuples.values():
                    values.extend(
                        collect_statuses(
                            decoder.get("fit_field_gates", {}).get(identifier, {})
                        )
                    )
        elif identifier in {"G1", "G2", "G3a", "G3b"}:
            for session_report in graph.get("raw_inner", {}).values():
                for condition_report in session_report.values():
                    values.extend(collect_statuses(condition_report.get(identifier, {})))
        elif identifier == "D1":
            values.extend(collect_statuses(directional.get("raw_inner", {})))
        elif identifier == "P1/P2":
            values.extend(collect_statuses(distribution.get("raw_inner", {})))
        return values

    def outer_counts(identifier: str) -> tuple[int, int, int]:
        selected = 0
        evaluated = 0
        expected = 0
        if identifier in {"S0", "S1", "S2", "S3", "S4-H", "S5", "S12", "S13"}:
            expected = len(uncertainty_outer.get("folds", {})) * len(HORIZONS)
            for fold in uncertainty_outer.get("folds", {}).values():
                for horizon in HORIZONS:
                    item = fold[str(horizon)][identifier]
                    if "parameters" in item:
                        selected += 1
                    if item.get("status") == "EVALUATED":
                        evaluated += 1
        elif identifier in DEFORMATION_CANDIDATES:
            expected = len(deformation_outer.get("folds", {})) * len(HORIZONS)
            for fold in deformation_outer.get("folds", {}).values():
                for horizon in HORIZONS:
                    item = fold[str(horizon)][identifier]
                    if "parameters" in item:
                        selected += 1
                    if item.get("status") == "EVALUATED":
                        evaluated += 1
        elif identifier in GRAPH_CANDIDATES:
            expected = len(graph.get("folds", {}))
            for fold in graph.get("folds", {}).values():
                item = fold[identifier]
                if "selected_tuple" in item:
                    selected += 1
                if item.get("status") == "EVALUATED":
                    evaluated += 1
        elif identifier == "D1":
            expected = len(directional.get("folds", {}))
            for item in directional.get("folds", {}).values():
                if "selected_tuple" in item:
                    selected += 1
                if item.get("status") == "EVALUATED":
                    evaluated += 1
        elif identifier == "P1/P2":
            expected = len(distribution.get("folds", {}))
            for item in distribution.get("folds", {}).values():
                if "selected_tuple" in item:
                    selected += 1
                if item.get("status") == "EVALUATED":
                    evaluated += 1
        return selected, evaluated, expected

    coverage: dict[str, Any] = {}
    for candidate in registry["candidates"]:
        identifier = str(candidate["id"])
        static_status = str(candidate["e17_status"])
        statuses = runtime_statuses(identifier)
        selected, evaluated, expected_outer = outer_counts(identifier)
        status_counts: dict[str, int] = defaultdict(int)
        for status in statuses:
            status_counts[status] += 1
        if static_status == "UNTESTABLE_MISSING_INPUT":
            strict_status = "UNTESTABLE_MISSING_INPUT"
        elif identifier in {"S8", "S9"}:
            strict_status = "FIELD_GATE_ONLY_NO_INDEPENDENT_METRIC_ENDPOINT"
        elif identifier in GRAPH_CANDIDATES and evaluated == 0:
            strict_status = "UNTESTABLE_UNDER_FROZEN_LOAO_INTERSECTION"
        elif expected_outer > 0 and evaluated == expected_outer:
            strict_status = "EVALUATED_ALL_FROZEN_OUTER_FOLDS"
        elif evaluated > 0:
            strict_status = "PARTIAL_OUTER_EVALUATION"
        elif statuses:
            strict_status = "RAW_TUPLES_ONLY_NO_OUTER_EVALUATION"
        else:
            strict_status = "NOT_ATTEMPTED"
        coverage[identifier] = {
            "static_status": static_status,
            "raw_tuple_attempted": bool(statuses),
            "runtime_status_counts": dict(sorted(status_counts.items())),
            "outer_folds_expected": expected_outer,
            "outer_folds_selected": selected,
            "outer_folds_evaluated": evaluated,
            "strict_tournament_status": strict_status,
            "missing_inputs": candidate.get("missing_inputs", []),
            "endpoint": candidate.get("endpoint"),
        }
    expected = {str(item["id"]) for item in registry["candidates"]}
    if set(coverage) != expected or len(expected) != 27:
        raise RuntimeError("candidate coverage does not match the frozen 27-ID universe")
    return coverage


def validate_registry(
    contract_path: Path, registry_markdown: Path, registry_json: Path
) -> dict[str, Any]:
    contract = contract_path.read_text(encoding="utf-8")
    markdown_hash = sha256_file(registry_markdown)
    json_hash = sha256_file(registry_json)
    registry = load_json(registry_json)
    if registry.get("registry_markdown_sha256") != markdown_hash:
        raise RuntimeError("JSON ledger does not pin the actual Markdown registry hash")
    if markdown_hash not in contract or json_hash not in contract:
        raise RuntimeError("contract does not pin the actual registry hashes")
    identifiers = [str(item["id"]) for item in registry["candidates"]]
    if len(identifiers) != 27 or len(set(identifiers)) != 27:
        raise RuntimeError("registry must contain 27 unique candidate IDs")
    return {
        "registry": registry,
        "markdown_sha256": markdown_hash,
        "json_sha256": json_hash,
        "candidate_ids": identifiers,
    }


def validate_freeze(
    freeze_path: Path,
    runner_path: Path,
    output_path: Path,
    data_root: Path,
    registry_validation: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[Path, str]]]:
    freeze = load_json(freeze_path)
    observed_runner_hash = sha256_file(runner_path)
    checks = {
        "runner_sha256": observed_runner_hash,
        "registry_markdown_sha256": registry_validation["markdown_sha256"],
        "registry_json_sha256": registry_validation["json_sha256"],
    }
    for field, observed in checks.items():
        if freeze.get(field) != observed:
            raise RuntimeError(
                f"freeze mismatch for {field}: expected {freeze.get(field)}, observed {observed}"
            )
    if freeze.get("output") != output_path.name:
        raise RuntimeError(
            f"freeze output mismatch: expected {freeze.get('output')}, got {output_path.name}"
        )
    expected_inputs = freeze.get("input_files_sha256")
    if not isinstance(expected_inputs, dict) or len(expected_inputs) != 11:
        raise RuntimeError("freeze must pin exactly 11 E17 MAT inputs")
    observed_paths = {
        path.relative_to(data_root).as_posix(): path
        for path in sorted((data_root / "Figure2" / "Data").glob("DCO*_dff.mat"))
    }
    if set(observed_paths) != set(expected_inputs):
        raise RuntimeError(
            "E17 input file set differs from freeze: "
            + json.dumps(
                {
                    "missing": sorted(set(expected_inputs) - set(observed_paths)),
                    "extra": sorted(set(observed_paths) - set(expected_inputs)),
                },
                sort_keys=True,
            )
        )
    verified: list[tuple[Path, str]] = []
    for relative_path in sorted(expected_inputs):
        path = observed_paths[relative_path]
        observed_hash = sha256_file(path)
        expected_hash = str(expected_inputs[relative_path])
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"E17 input hash mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        verified.append((path, observed_hash))
    return (
        {
            "status": "PASS_BYTE_PINNED_INPUTS_AND_CODE",
            "freeze_sha256": sha256_file(freeze_path),
            "freeze_path": freeze_path.as_posix(),
            "runner_sha256": observed_runner_hash,
            "verified_input_count": len(verified),
        },
        verified,
    )


def runtime_environment() -> dict[str, Any]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.__config__.show()
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": package_version("numpy"),
        "scipy_version": package_version("scipy"),
        "numpy_build_configuration": buffer.getvalue(),
        "reproduction_policy": "numeric tolerance, not byte identity across numerical stacks",
    }


def session_manifest(sessions: list[Session]) -> dict[str, Any]:
    animals = sorted({session.animal for session in sessions})
    return {
        "session_count": len(sessions),
        "animal_count": len(animals),
        "animals": {
            animal: [
                session.session_id for session in sessions if session.animal == animal
            ]
            for animal in animals
        },
        "sessions": [
            {
                "session_id": session.session_id,
                "animal": session.animal,
                "source_path": session.source_path,
                "source_sha256": session.source_sha256,
                "original_dimension": session.original_dimension,
                "retained_dimension": session.dimension,
                "trial_counts": {
                    condition: {
                        "fit": len(blocks.fit),
                        "inner": len(blocks.inner),
                        "test": len(blocks.test),
                    }
                    for condition, blocks in session.conditions.items()
                },
            }
            for session in sessions
        ],
    }


def kill_test_ledger(
    uncertainty: dict[str, Any], directional: dict[str, Any]
) -> dict[str, Any]:
    k3_diagnostics: dict[str, Any] = {}
    for horizon, board in uncertainty.get("scoreboard", {}).items():
        ranking = board.get("ranking", [])
        geometry = [item for item in ranking if not item["candidate"].startswith("BASE_")]
        direct = [item for item in ranking if item["candidate"].startswith("BASE_")]
        if geometry and direct:
            k3_diagnostics[horizon] = {
                "best_geometry": geometry[0]["candidate"],
                "best_geometry_mean_animal_nlpd": geometry[0]["mean_animal_nlpd"],
                "best_direct_baseline": direct[0]["candidate"],
                "best_direct_baseline_mean_animal_nlpd": direct[0]["mean_animal_nlpd"],
                "geometry_minus_direct_nlpd": (
                    geometry[0]["mean_animal_nlpd"]
                    - direct[0]["mean_animal_nlpd"]
                ),
            }
    reverse_diagnostics = {
        animal: fold.get("heldout_animal_reverse_minus_forward")
        for animal, fold in directional.get("folds", {}).items()
    }
    return {
        "K1": {
            "status": "PARTIAL_SYNTHETIC_CHART_CHECK_ONLY",
            "detail": "No confirmatory cell-resampling stability claim is possible without longitudinal cell identity.",
        },
        "K2": {
            "status": "PASS_V2_DISCOVERY_PROTOCOL",
            "detail": "V2 selects predictive tuples on outer-train inner blocks and evaluates only the selected tuple on held-out test blocks.",
        },
        "K3": {
            "status": "TRIGGERED_OR_UNRESOLVED_NO_POPULATION_ADVANTAGE",
            "diagnostics": k3_diagnostics,
        },
        "K4": {
            "status": "UNTESTABLE_MISSING_REGISTERED_PATH_OR_HITTING_TIME_ENDPOINT"
        },
        "K5": {
            "status": "D1_DIRECTIONALITY_NOT_CLEARED",
            "reverse_minus_forward_by_animal": reverse_diagnostics,
            "detail": "Time reversal is close to the forward score; shuffle disruption alone is temporal structure, not a directed-geometry validation.",
        },
        "K6": {
            "status": "TRIGGERED_FOR_W_SPECIFIC_MECHANISM",
            "detail": "E17 lacks direct W and cannot distinguish fixed-W gain/noise alternatives from an effective-dynamics metric.",
        },
        "K7": {
            "status": "TRIGGERED_FOR_DELTA_W_TO_DELTA_G_TO_DELTA_X_CHAIN",
            "detail": "The released E17 tables do not link direct connectivity, metric and trajectories in the same units and time window.",
        },
        "K8": {
            "status": "TRIGGERED_FOR_LONGITUDINAL_CONNECTIVITY_CHANGE",
            "detail": "No cross-session cell-identity chain is available.",
        },
        "K10": {
            "status": "TRIGGERED_FOR_CONFIRMATION",
            "detail": "E17 was opened before the final universe and remains retrospective discovery only; confirmation requires a new locked cohort.",
        },
    }


def parse_args() -> argparse.Namespace:
    artifact_root = Path(__file__).resolve().parent
    run_root = artifact_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=artifact_root / "realdata" / "NRM-E17-extracted",
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=artifact_root / "candidate-equation-registry.md",
    )
    parser.add_argument(
        "--registry-json",
        type=Path,
        default=artifact_root / "candidate-equation-registry.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=run_root / "00-contract.md",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=artifact_root / "e17-candidate-tournament-freeze-v2.2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_root / "e17-candidate-tournament-results-v2.2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to reopen/overwrite tournament output: {args.output}"
        )
    validation = validate_registry(args.contract, args.registry_md, args.registry_json)
    print("registry validated", flush=True)

    freeze_validation, data_files = validate_freeze(
        args.freeze,
        Path(__file__).resolve(),
        args.output,
        args.data_root,
        validation,
    )
    print("freeze and input bytes validated", flush=True)
    sessions = [prepare_session(path, digest) for path, digest in data_files]
    if len(sessions) != 11 or len({session.animal for session in sessions}) != 3:
        raise RuntimeError("frozen E17 contract expects 11 sessions from 3 animals")
    print("sessions prepared", flush=True)

    models, horizons = build_models(sessions)
    print("fit-only dynamics prepared", flush=True)

    uncertainty_raw, uncertainty_cache = run_uncertainty_inner(
        sessions, models, horizons
    )
    uncertainty = run_uncertainty_outer(
        sessions, uncertainty_raw, uncertainty_cache, horizons
    )
    print("uncertainty tournament complete", flush=True)

    deformation_raw, deformation_cache = run_deformation_inner(
        sessions, models, horizons
    )
    deformation = run_deformation_outer(
        sessions, deformation_raw, deformation_cache
    )
    print("deformation tournament complete", flush=True)

    condition_information = run_condition_information(sessions)
    print("condition-information tournament complete", flush=True)

    graph = run_graph_tournament(sessions, models)
    print("graph tournament complete", flush=True)

    directional = run_directional_action(sessions, models)
    print("directional action tournament complete", flush=True)

    distribution = run_distribution_tournament(sessions)
    print("distribution tournament complete", flush=True)

    execution_audit = tuple_execution_audit(
        sessions,
        uncertainty_raw,
        deformation_raw,
        condition_information,
        graph,
        directional,
        distribution,
    )
    print("exact tuple execution audit passed", flush=True)

    coverage = candidate_coverage(
        validation["registry"],
        uncertainty_raw,
        uncertainty,
        deformation_raw,
        deformation,
        condition_information,
        graph,
        directional,
        distribution,
    )
    output = {
        "schema_version": "2.0.2",
        "analysis_status": "RETROSPECTIVE_DISCOVERY_ONLY",
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_validation": freeze_validation,
        "registry_markdown_sha256": validation["markdown_sha256"],
        "registry_json_sha256": validation["json_sha256"],
        "candidate_ids": validation["candidate_ids"],
        "frozen_settings": validation["registry"]["common_grid"],
        "runtime_environment": runtime_environment(),
        "data_manifest": session_manifest(sessions),
        "tuple_execution_audit": execution_audit,
        "candidate_coverage": coverage,
        "uncertainty": {
            "raw_outer_train_inner": uncertainty_raw,
            **uncertainty,
        },
        "deformation": {
            "raw_outer_train_inner": deformation_raw,
            **deformation,
        },
        "condition_information": condition_information,
        "graph": graph,
        "directional_action": directional,
        "distribution": {
            **distribution,
            "permutation_inference_boundary": (
                "Permutation positions are descriptive because trial-level treatment-label "
                "exchangeability is not established by the release metadata."
            ),
        },
        "kill_tests": kill_test_ledger(uncertainty, directional),
        "claim_status": {
            "NRM_E17D": "CALCULATED_IF_THIS_FILE_EXISTS_AND_VALIDATES",
            "NRM_H1A": "UNTESTABLE_MISSING_SAME_UNIT_DIRECT_W_CHAIN",
            "NRM_H1B": "EXPLORATORY_EFFECTIVE_DYNAMICS_ONLY",
            "NRM_H2": "NOT_CONFIRMABLE_WITH_THREE_OPENED_ANIMALS",
            "population_winner": "PROHIBITED",
            "locked_validation": "NOT_RUN_REQUIRES_NEW_COHORT",
        },
        "generalization_target": (
            "LOAO_HYPERPARAMETER_GENERALIZATION_WITH_HELDOUT_SESSION_CALIBRATION; "
            "not cross-animal transport of a learned metric"
        ),
        "supersedes": [
            {
                "artifact": "e17-candidate-tournament-results.json",
                "status": "INVALIDATED_BY_STRUCTURAL_AND_PROVENANCE_AUDIT",
            },
            {
                "artifact": "e17-candidate-tournament-results-v2.json",
                "status": "SUPERSEDED_FOR_BOOLEAN_SCHEMA_SERIALIZATION_ONLY",
            },
            {
                "artifact": "e17-candidate-tournament-results-v2.1.json",
                "status": "SUPERSEDED_FOR_EXCLUSIVE_CREATE_AND_FINITE_SCORE_ASSURANCE_ONLY",
            },
        ],
        "write_policy": "create-only; existing outputs are never replaced",
        "inference_boundary": (
            "Every E17 rank and score is retrospective discovery. Session, trial, "
            "window, ROI and node-pair counts are not independent population units."
        ),
    }
    dump_json(args.output, output)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
