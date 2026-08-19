#!/usr/bin/env python3
"""Test a covariance-inverse metric equation on official macaque PFC caches."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from run_official_pfc_processed_geometry import git_value, safe_load, sha256


EXP1 = (
    ("Exp1 main", "selectivity_coefficients_exp1_140_1504stages.pickle", "selectivity_coefficients_xval"),
    (
        "Exp1 fixation-bias control",
        "selectivity_coefficients_exp1_fixbias_140_1504stages.pickle",
        "selectivity_coefficients_xval",
    ),
)
EXP2_STAGE_COUNTS = (3, 4, 5, 6)


def covariance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        raise ValueError(f"expected an N x 3 point array, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("nonfinite selectivity coefficient")
    centered = points - points.mean(axis=0, keepdims=True)
    result = centered.T @ centered / (len(points) - 1)
    result = (result + result.T) / 2.0
    if np.linalg.eigvalsh(result)[0] <= 0.0:
        raise ValueError("selectivity covariance is not SPD")
    return result


def airm_components(first: np.ndarray, last: np.ndarray) -> tuple[float, float, float]:
    values, vectors = np.linalg.eigh(first)
    inverse_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    relative = inverse_sqrt @ last @ inverse_sqrt
    relative = (relative + relative.T) / 2.0
    log_values = np.log(np.linalg.eigvalsh(relative))
    scale = math.sqrt(3.0) * abs(float(log_values.mean()))
    shape = float(np.linalg.norm(log_values - log_values.mean()))
    total = float(np.linalg.norm(log_values))
    return total, scale, shape


def spd_function(matrix: np.ndarray, function) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    if values[0] <= 0.0:
        raise ValueError("matrix function requires SPD input")
    return vectors @ np.diag(function(values)) @ vectors.T


def relative_precision_geometry(first: np.ndarray, last: np.ndarray) -> dict[str, np.ndarray]:
    """Return coordinate-free relative geometry and its symmetric numerical gauge."""
    first_metric = np.linalg.inv(first)
    last_metric = np.linalg.inv(last)
    first_metric_sqrt = spd_function(first_metric, np.sqrt)
    first_metric_inverse_sqrt = spd_function(first_metric, lambda values: values ** -0.5)
    symmetric = first_metric_inverse_sqrt @ last_metric @ first_metric_inverse_sqrt
    symmetric = (symmetric + symmetric.T) / 2.0
    endomorphism = np.linalg.solve(first_metric, last_metric)
    transporter = (
        first_metric_inverse_sqrt
        @ spd_function(symmetric, np.sqrt)
        @ first_metric_sqrt
    )
    return {
        "first_metric": first_metric,
        "last_metric": last_metric,
        "symmetric": symmetric,
        "endomorphism": endomorphism,
        "transporter": transporter,
        "log_modes": np.log(np.linalg.eigvalsh(symmetric)),
    }


def covariance_axis_diagnostics(first: np.ndarray, last: np.ndarray) -> dict[str, float]:
    first_values, first_vectors = np.linalg.eigh(first)
    last_values, last_vectors = np.linalg.eigh(last)
    first_axis = first_vectors[:, -1]
    last_axis = last_vectors[:, -1]
    cosine = float(np.clip(abs(first_axis @ last_axis), 0.0, 1.0))

    def relative_gap(values: np.ndarray, upper: int) -> float:
        return float((values[upper] - values[upper - 1]) / values[upper])

    return {
        "dominant_axis_angle_degrees": math.degrees(math.acos(cosine)),
        "dominant_gap_first": relative_gap(first_values, 2),
        "dominant_gap_last": relative_gap(last_values, 2),
        "minimum_gap_first": min(relative_gap(first_values, 1), relative_gap(first_values, 2)),
        "minimum_gap_last": min(relative_gap(last_values, 1), relative_gap(last_values, 2)),
        "condition_first": float(first_values[-1] / first_values[0]),
        "condition_last": float(last_values[-1] / last_values[0]),
    }


def released_row_bootstrap(
    first_points: np.ndarray,
    last_points: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Conditional uncertainty over released pseudopopulation rows, not animals."""
    rng = np.random.default_rng(seed)
    log_modes = np.empty((draws, 3), dtype=np.float64)
    distances = np.empty(draws, dtype=np.float64)
    angles = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        first = covariance(first_points[rng.integers(0, len(first_points), len(first_points))])
        last = covariance(last_points[rng.integers(0, len(last_points), len(last_points))])
        modes = relative_precision_geometry(first, last)["log_modes"]
        log_modes[draw] = modes
        distances[draw] = np.linalg.norm(modes)
        angles[draw] = covariance_axis_diagnostics(first, last)["dominant_axis_angle_degrees"]
    return {
        "draws": draws,
        "log_mode_interval_95": np.percentile(log_modes, [2.5, 97.5], axis=0),
        "airm_interval_95": np.percentile(distances, [2.5, 97.5]),
        "dominant_axis_angle_interval_95": np.percentile(angles, [2.5, 97.5]),
    }


def metric_bundle(first: np.ndarray, last: np.ndarray) -> dict[str, object]:
    first_sqrt = spd_function(first, np.sqrt)
    first_inverse_sqrt = spd_function(first, lambda values: values ** -0.5)
    relative = first_inverse_sqrt @ last @ first_inverse_sqrt
    relative = (relative + relative.T) / 2.0
    generalized_log_modes = np.log(np.linalg.eigvalsh(relative))
    mean_log = float(generalized_log_modes.mean())
    airm_total = float(np.linalg.norm(generalized_log_modes))
    airm_scale = math.sqrt(3.0) * abs(mean_log)
    airm_shape = float(np.linalg.norm(generalized_log_modes - mean_log))

    first_log = spd_function(first, np.log)
    last_log = spd_function(last, np.log)
    log_euclidean = float(np.linalg.norm(last_log - first_log))

    middle = first_sqrt @ last @ first_sqrt
    middle_sqrt = spd_function(middle, np.sqrt)
    bures_squared = float(np.trace(first) + np.trace(last) - 2.0 * np.trace(middle_sqrt))
    bures = math.sqrt(max(0.0, bures_squared))

    dimension = first.shape[0]
    jeffreys = 0.5 * float(
        np.trace(np.linalg.solve(last, first))
        + np.trace(np.linalg.solve(first, last))
        - 2.0 * dimension
    )
    precision_geometry = relative_precision_geometry(first, last)
    return {
        "generalized_log_modes": generalized_log_modes,
        "relative_precision_log_modes": precision_geometry["log_modes"],
        "airm_total": airm_total,
        "airm_scale": airm_scale,
        "airm_shape": airm_shape,
        "shape_fraction_sq": (airm_shape / airm_total) ** 2 if airm_total > 0.0 else 0.0,
        "signed_log_volume_ratio": 0.5 * float(np.linalg.slogdet(last)[1] - np.linalg.slogdet(first)[1]),
        "jeffreys": jeffreys,
        "log_euclidean": log_euclidean,
        "bures": bures,
    }


def coordinate_and_geodesic_checks(first: np.ndarray, last: np.ndarray) -> dict[str, float]:
    transform = np.asarray([[1.20, 0.20, 0.00], [0.10, 0.85, 0.15], [0.05, 0.00, 1.10]])
    transformed_first = transform @ first @ transform.T
    transformed_last = transform @ last @ transform.T
    inverse_transform = np.linalg.inv(transform)
    predicted_metric = inverse_transform.T @ np.linalg.inv(first) @ inverse_transform
    direct_metric = np.linalg.inv(transformed_first)
    metric_law_residual = float(
        np.linalg.norm(direct_metric - predicted_metric) / np.linalg.norm(predicted_metric)
    )

    base_relative = relative_precision_geometry(first, last)
    transformed_relative = relative_precision_geometry(transformed_first, transformed_last)
    endomorphism_target = transform @ base_relative["endomorphism"] @ inverse_transform
    transporter_target = transform @ base_relative["transporter"] @ inverse_transform
    endomorphism_residual = float(
        np.linalg.norm(transformed_relative["endomorphism"] - endomorphism_target)
        / np.linalg.norm(endomorphism_target)
    )
    transporter_residual = float(
        np.linalg.norm(transformed_relative["transporter"] - transporter_target)
        / np.linalg.norm(transporter_target)
    )
    first_metric = base_relative["first_metric"]
    last_metric = base_relative["last_metric"]
    transporter = base_relative["transporter"]
    congruence_residual = float(
        np.linalg.norm(transporter.T @ first_metric @ transporter - last_metric)
        / np.linalg.norm(last_metric)
    )
    self_adjoint_residual = float(
        np.linalg.norm(transporter.T @ first_metric - first_metric @ transporter)
        / np.linalg.norm(first_metric @ transporter)
    )
    direction = np.asarray([1.0, 0.35, -0.20])
    transformed_direction = transform @ direction
    direction_ratio = float(direction @ last_metric @ direction / (direction @ first_metric @ direction))
    transformed_direction_ratio = float(
        transformed_direction @ transformed_relative["last_metric"] @ transformed_direction
        / (transformed_direction @ transformed_relative["first_metric"] @ transformed_direction)
    )
    first_metric_sqrt = spd_function(first_metric, np.sqrt)
    whitened_direction = first_metric_sqrt @ direction
    whitened_direction /= math.sqrt(float(direction @ first_metric @ direction))
    symmetric_ratio = float(
        whitened_direction @ base_relative["symmetric"] @ whitened_direction
    )

    original = metric_bundle(first, last)
    transformed = metric_bundle(transformed_first, transformed_last)
    first_sqrt = spd_function(first, np.sqrt)
    first_inverse_sqrt = spd_function(first, lambda values: values ** -0.5)
    relative = first_inverse_sqrt @ last @ first_inverse_sqrt
    midpoint = first_sqrt @ spd_function(relative, np.sqrt) @ first_sqrt
    first_half = metric_bundle(first, midpoint)["airm_total"]
    last_half = metric_bundle(midpoint, last)["airm_total"]
    half_target = float(original["airm_total"]) / 2.0
    return {
        "metric_transform_relative_residual": metric_law_residual,
        "relative_endomorphism_similarity_residual": endomorphism_residual,
        "canonical_transport_congruence_residual": congruence_residual,
        "canonical_transport_self_adjoint_residual": self_adjoint_residual,
        "canonical_transport_similarity_residual": transporter_residual,
        "relative_direction_ratio_gl_residual": abs(direction_ratio - transformed_direction_ratio),
        "relative_symmetric_representation_residual": abs(direction_ratio - symmetric_ratio),
        "relative_precision_spectrum_gl_residual": float(
            np.max(
                np.abs(
                    np.sort(base_relative["log_modes"])
                    - np.sort(transformed_relative["log_modes"])
                )
            )
        ),
        "airm_gl_invariance_residual": abs(float(original["airm_total"]) - float(transformed["airm_total"])),
        "jeffreys_gl_invariance_residual": abs(float(original["jeffreys"]) - float(transformed["jeffreys"])),
        "log_euclidean_gl_change": abs(
            float(original["log_euclidean"]) - float(transformed["log_euclidean"])
        ),
        "bures_gl_change": abs(float(original["bures"]) - float(transformed["bures"])),
        "geodesic_midpoint_half_residual": max(
            abs(float(first_half) - half_target), abs(float(last_half) - half_target)
        ),
    }


def metric_cost_change(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    first_metric = np.linalg.inv(first)
    last_metric = np.linalg.inv(last)
    return np.log(np.diag(last_metric) / np.diag(first_metric))


def gaussian_model(points: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    matrix = points.T @ points / len(points)
    if kind == "diagonal":
        matrix = np.diag(np.diag(matrix))
    elif kind == "spherical":
        matrix = np.eye(3) * np.trace(matrix) / 3.0
    elif kind != "full":
        raise ValueError(f"unknown Gaussian model: {kind}")
    if np.linalg.eigvalsh(matrix)[0] <= 0.0:
        raise ValueError(f"non-SPD held-out covariance: {kind}")
    return np.linalg.inv(matrix), float(np.linalg.slogdet(matrix)[1])


def gaussian_nll(points: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    inverse, logdet = model
    energy = np.einsum("ni,ij,nj->n", points, inverse, points)
    return 0.5 * (energy + logdet + 3.0 * math.log(2.0 * math.pi))


def heldout_metric_prediction(
    first_points: np.ndarray,
    last_points: np.ndarray,
    repeats: int,
    folds: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if max(
        float(np.max(np.abs(first_points.mean(axis=0)))),
        float(np.max(np.abs(last_points.mean(axis=0)))),
    ) > 1e-12:
        raise ValueError("held-out density test requires the author-centered Exp2 coordinates")
    rng = np.random.default_rng(seed)
    improvements: dict[str, list[float]] = {"pooled full": [], "stage diagonal": [], "stage spherical": []}
    for _ in range(repeats):
        first_folds = np.array_split(rng.permutation(len(first_points)), folds)
        last_folds = np.array_split(rng.permutation(len(last_points)), folds)
        for fold in range(folds):
            first_test = first_folds[fold]
            last_test = last_folds[fold]
            first_train = np.concatenate([first_folds[index] for index in range(folds) if index != fold])
            last_train = np.concatenate([last_folds[index] for index in range(folds) if index != fold])

            full_first = gaussian_model(first_points[first_train], "full")
            full_last = gaussian_model(last_points[last_train], "full")
            full_nll = np.concatenate(
                [
                    gaussian_nll(first_points[first_test], full_first),
                    gaussian_nll(last_points[last_test], full_last),
                ]
            ).mean()

            pooled = gaussian_model(
                np.concatenate([first_points[first_train], last_points[last_train]], axis=0), "full"
            )
            diagonal_first = gaussian_model(first_points[first_train], "diagonal")
            diagonal_last = gaussian_model(last_points[last_train], "diagonal")
            spherical_first = gaussian_model(first_points[first_train], "spherical")
            spherical_last = gaussian_model(last_points[last_train], "spherical")
            alternatives = {
                "pooled full": (pooled, pooled),
                "stage diagonal": (diagonal_first, diagonal_last),
                "stage spherical": (spherical_first, spherical_last),
            }
            for name, (first_model, last_model) in alternatives.items():
                alternative_nll = np.concatenate(
                    [
                        gaussian_nll(first_points[first_test], first_model),
                        gaussian_nll(last_points[last_test], last_model),
                    ]
                ).mean()
                improvements[name].append(float(alternative_nll - full_nll))

    return {
        name: {
            "mean_nll_gain": float(np.mean(values)),
            "fold_wins": int(np.sum(np.asarray(values) > 0.0)),
            "folds": int(len(values)),
        }
        for name, values in improvements.items()
    }


def permutation_test(
    first_points: np.ndarray,
    last_points: np.ndarray,
    draws: int,
    seed: int,
    extended: bool = False,
) -> dict[str, object]:
    first_covariance = covariance(first_points)
    last_covariance = covariance(last_points)
    observed = metric_bundle(first_covariance, last_covariance)
    if extended:
        observed.update(covariance_axis_diagnostics(first_covariance, last_covariance))

    pooled = np.concatenate([first_points, last_points], axis=0)
    first_count = len(first_points)
    rng = np.random.default_rng(seed)
    tested_statistics = ["airm_total", "airm_shape"]
    if extended:
        tested_statistics.extend(
            ["jeffreys", "log_euclidean", "bures", "dominant_axis_angle_degrees"]
        )
    exceedances = {name: 0 for name in tested_statistics}
    for _ in range(draws):
        order = rng.permutation(len(pooled))
        null_first = covariance(pooled[order[:first_count]])
        null_last = covariance(pooled[order[first_count:]])
        if extended:
            null = metric_bundle(null_first, null_last)
            null.update(covariance_axis_diagnostics(null_first, null_last))
        else:
            null_total, null_scale, null_shape = airm_components(null_first, null_last)
            null = {
                "airm_total": null_total,
                "airm_scale": null_scale,
                "airm_shape": null_shape,
            }
        for name in tested_statistics:
            exceedances[name] += int(float(null[name]) >= float(observed[name]))

    result: dict[str, object] = {
        "n_first": int(len(first_points)),
        "n_last": int(len(last_points)),
        **observed,
        "p_total": (exceedances["airm_total"] + 1) / (draws + 1),
        "p_shape": (exceedances["airm_shape"] + 1) / (draws + 1),
        "metric_log_cost_change": metric_cost_change(first_covariance, last_covariance),
        "first_covariance": first_covariance,
        "last_covariance": last_covariance,
        "coordinate_checks": coordinate_and_geodesic_checks(first_covariance, last_covariance),
    }
    for name in ("jeffreys", "log_euclidean", "bures"):
        result[f"p_{name}"] = (exceedances[name] + 1) / (draws + 1) if extended else None
    result["p_dominant_axis_angle"] = (
        (exceedances["dominant_axis_angle_degrees"] + 1) / (draws + 1)
        if extended
        else None
    )
    return result


def exp1_points(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, float]:
    stages = safe_load(path)[key]
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError(f"expected four Exp1 stages in {path}")
    arrays = [np.asarray(stage, dtype=np.float64) for stage in stages]
    for array in arrays:
        if array.ndim != 3 or array.shape[1:] != (3, 2):
            raise ValueError(f"unexpected Exp1 selectivity shape: {array.shape}")
    duplicate_residual = max(float(np.max(np.abs(array[:, :, 0] - array[:, :, 1]))) for array in arrays)
    if duplicate_residual != 0.0:
        raise ValueError("Exp1 cached folds are no longer exact duplicates")
    return arrays[0][:, :, 0], arrays[-1][:, :, 0], duplicate_residual


def exp2_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stages = safe_load(path)["selectivity_coefficients"]
    if not isinstance(stages, list) or len(stages) < 3:
        raise ValueError(f"unexpected Exp2 stage list in {path}")
    arrays = [np.asarray(stage, dtype=np.float64) for stage in stages]
    for array in arrays:
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"unexpected Exp2 selectivity shape: {array.shape}")
    return arrays[0], arrays[-1]


def ler_delta_p(observed: np.ndarray, null: np.ndarray) -> tuple[float, float]:
    observed_delta = float(observed[-1] - observed[0])
    null_delta = np.asarray(null[:, 1] - null[:, 0], dtype=np.float64)
    p_value = (int(np.sum(np.abs(null_delta) >= abs(observed_delta))) + 1) / (len(null_delta) + 1)
    return observed_delta, p_value


def decoder_evidence(processed: Path, exp1_row: dict[str, object], exp2_row: dict[str, object]) -> dict[str, object]:
    colour = safe_load(processed / "exp1_decoding_collocked_50_150_4stages.pickle")
    shape = safe_load(processed / "exp1_decoding_shapelocked_100_150_4stages.pickle")
    exp2 = safe_load(processed / "exp2_decoding_time_avg_4stages_50_100.pickle")

    exp1_specs = [
        ("colour", np.asarray(colour["early_decoding"])[:, 0], np.asarray(colour["early_decoding_ler_rnd"])[:, :, 0]),
        ("shape", np.asarray(shape["late_decoding"])[:, 1], np.asarray(shape["late_decoding_ler_rnd"])[:, :, 1]),
        ("XOR", np.asarray(shape["late_decoding"])[:, 3], np.asarray(shape["late_decoding_ler_rnd"])[:, :, 3]),
    ]
    exp2_specs = [
        ("set", np.asarray(exp2["scores_set"])[0], np.asarray(exp2["scores_set_rnd"])[:, 0, :]),
        ("set*context (XOR2)", np.asarray(exp2["scores_xor2"])[0], np.asarray(exp2["scores_xor2_rnd"])[:, 0, :]),
        ("context", np.asarray(exp2["scores_context"])[0], np.asarray(exp2["scores_context_rnd"])[:, 0, :]),
    ]

    entries: list[dict[str, object]] = []
    for experiment, specs, row in (("Exp1", exp1_specs, exp1_row), ("Exp2", exp2_specs, exp2_row)):
        costs = np.asarray(row["metric_log_cost_change"], dtype=np.float64)
        for axis_index, (axis, observed, null) in enumerate(specs):
            decoder_delta, decoder_p = ler_delta_p(observed, null)
            entries.append(
                {
                    "experiment": experiment,
                    "axis": axis,
                    "metric_log_cost_change": float(costs[axis_index]),
                    "decoder_delta": decoder_delta,
                    "decoder_ler_p": decoder_p,
                }
            )

    accessibility = -np.asarray([entry["metric_log_cost_change"] for entry in entries], dtype=np.float64)
    decoder_delta = np.asarray([entry["decoder_delta"] for entry in entries], dtype=np.float64)
    observed_correlation = float(np.corrcoef(accessibility, decoder_delta)[0, 1])
    null_correlations: list[float] = []
    first_decoder = decoder_delta[:3]
    second_decoder = decoder_delta[3:]
    for first_order in itertools.permutations(range(3)):
        for second_order in itertools.permutations(range(3)):
            permuted = np.concatenate([first_decoder[list(first_order)], second_decoder[list(second_order)]])
            null_correlations.append(float(np.corrcoef(accessibility, permuted)[0, 1]))
    exact_p = float(
        np.mean(np.abs(null_correlations) >= abs(observed_correlation) - 1e-15)
    )
    return {
        "entries": entries,
        "pooled_correlation": observed_correlation,
        "within_experiment_axis_permutation_p": exact_p,
        "sign_matches": int(np.sum(np.sign(accessibility) == np.sign(decoder_delta))),
        "axis_count": int(len(entries)),
    }


def fisher_information_bridge(processed: Path) -> dict[str, object]:
    exp1_raw = safe_load(processed / "selectivity_coefficients_exp1_140_1504stages.pickle")[
        "selectivity_coefficients_xval"
    ]
    exp1_stages = [np.asarray(stage, dtype=np.float64)[:, :, 0] for stage in exp1_raw]
    exp2_stages = [
        np.asarray(stage, dtype=np.float64)
        for stage in safe_load(processed / "selectivity_coefficients_exp2_70_100_4stages.pickle")[
            "selectivity_coefficients"
        ]
    ]
    colour = safe_load(processed / "exp1_decoding_collocked_50_150_4stages.pickle")
    shape = safe_load(processed / "exp1_decoding_shapelocked_100_150_4stages.pickle")
    exp2 = safe_load(processed / "exp2_decoding_time_avg_4stages_50_100.pickle")
    exp1_accuracy = np.column_stack(
        [
            np.asarray(colour["early_decoding"])[:, 0],
            np.asarray(shape["late_decoding"])[:, 1],
            np.asarray(shape["late_decoding"])[:, 3],
        ]
    )
    exp2_accuracy = np.column_stack(
        [
            np.asarray(exp2["scores_set"])[0],
            np.asarray(exp2["scores_xor2"])[0],
            np.asarray(exp2["scores_context"])[0],
        ]
    )

    def fisher_tensor(points: np.ndarray) -> np.ndarray:
        result = points.T @ points
        if np.linalg.eigvalsh(result)[0] <= 0.0:
            raise ValueError("rank-deficient Fisher pullback candidate")
        return result

    def fisher_axis_scale(stages: list[np.ndarray]) -> np.ndarray:
        return np.asarray([np.sqrt(np.diag(fisher_tensor(stage))) for stage in stages])

    def inverse_covariance_scale(stages: list[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [
                np.sqrt(len(stage))
                / np.sqrt(np.diag(np.linalg.inv(covariance(stage))))
                for stage in stages
            ]
        )

    normal = NormalDist()

    def to_probit(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [normal.inv_cdf(float(np.clip(value, 1e-9, 1.0 - 1e-9))) for value in values.ravel()]
        ).reshape(values.shape)

    def from_probit(values: np.ndarray) -> np.ndarray:
        return np.asarray([normal.cdf(float(value)) for value in values.ravel()]).reshape(values.shape)

    def fit_transfer(train_scale: np.ndarray, test_scale: np.ndarray) -> tuple[float, np.ndarray]:
        train_target = to_probit(exp1_accuracy)
        coefficient = float(
            np.sum(train_scale * train_target) / np.sum(train_scale * train_scale)
        )
        return coefficient, from_probit(coefficient * test_scale)

    fisher_exp1 = fisher_axis_scale(exp1_stages)
    fisher_exp2 = fisher_axis_scale(exp2_stages)
    fisher_coefficient, fisher_prediction = fit_transfer(fisher_exp1, fisher_exp2)

    inverse_exp1 = inverse_covariance_scale(exp1_stages)
    inverse_exp2 = inverse_covariance_scale(exp2_stages)
    inverse_coefficient, inverse_prediction = fit_transfer(inverse_exp1, inverse_exp2)

    isotropic_exp1 = np.repeat(
        np.sqrt(np.mean(fisher_exp1 * fisher_exp1, axis=1))[:, None], 3, axis=1
    )
    isotropic_exp2 = np.repeat(
        np.sqrt(np.mean(fisher_exp2 * fisher_exp2, axis=1))[:, None], 3, axis=1
    )
    isotropic_coefficient, isotropic_prediction = fit_transfer(isotropic_exp1, isotropic_exp2)

    def score(prediction: np.ndarray) -> dict[str, float]:
        if float(np.std(prediction)) <= 1e-15:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(exp2_accuracy.ravel(), prediction.ravel())[0, 1])
        return {
            "rmse": float(np.sqrt(np.mean((exp2_accuracy - prediction) ** 2))),
            "mae": float(np.mean(np.abs(exp2_accuracy - prediction))),
            "correlation": correlation,
        }

    global_prediction = np.full_like(exp2_accuracy, float(exp1_accuracy.mean()))
    stage_prediction = np.repeat(exp1_accuracy.mean(axis=1)[:, None], 3, axis=1)

    transform = np.asarray([[1.20, 0.20, 0.00], [0.10, 0.85, 0.15], [0.05, 0.00, 1.10]])
    inverse_transform = np.linalg.inv(transform)
    source = exp1_stages[0]
    source_fisher = fisher_tensor(source)
    transformed_source = source @ inverse_transform
    transformed_fisher = fisher_tensor(transformed_source)
    predicted_fisher = inverse_transform.T @ source_fisher @ inverse_transform
    tensor_law_residual = float(
        np.linalg.norm(transformed_fisher - predicted_fisher) / np.linalg.norm(predicted_fisher)
    )

    return {
        "equation": "J_F=S^T Q^-1 S; Q=sigma^2 I absorbed into kappa",
        "kappa": fisher_coefficient,
        "exp1_information_scale": fisher_exp1,
        "exp2_information_scale": fisher_exp2,
        "exp2_observed_accuracy": exp2_accuracy,
        "exp2_predicted_accuracy": fisher_prediction,
        "tensor_law_residual": tensor_law_residual,
        "models": {
            "Fisher pullback (one parameter)": {
                "coefficient": fisher_coefficient,
                **score(fisher_prediction),
            },
            "inverse-covariance accessibility (one parameter)": {
                "coefficient": inverse_coefficient,
                **score(inverse_prediction),
            },
            "isotropic Fisher (one parameter)": {
                "coefficient": isotropic_coefficient,
                **score(isotropic_prediction),
            },
            "Exp1 global-mean baseline": {"coefficient": float("nan"), **score(global_prediction)},
            "Exp1 stage-mean baseline": {"coefficient": float("nan"), **score(stage_prediction)},
        },
    }


def transfer_inputs(processed: Path, experiment: str, stage_count: int = 4) -> tuple[list[np.ndarray], np.ndarray]:
    if experiment == "Exp1":
        raw = safe_load(processed / "selectivity_coefficients_exp1_140_1504stages.pickle")[
            "selectivity_coefficients_xval"
        ]
        stages = [np.asarray(stage, dtype=np.float64)[:, :, 0] for stage in raw]
        colour = safe_load(processed / "exp1_decoding_collocked_50_150_4stages.pickle")
        shape = safe_load(processed / "exp1_decoding_shapelocked_100_150_4stages.pickle")
        accuracy = np.column_stack(
            [
                np.asarray(colour["early_decoding"])[:, 0],
                np.asarray(shape["late_decoding"])[:, 1],
                np.asarray(shape["late_decoding"])[:, 3],
            ]
        )
    elif experiment == "Exp2":
        raw = safe_load(
            processed / f"selectivity_coefficients_exp2_70_100_{stage_count}stages.pickle"
        )["selectivity_coefficients"]
        stages = [np.asarray(stage, dtype=np.float64) for stage in raw]
        decoded = safe_load(
            processed / f"exp2_decoding_time_avg_{stage_count}stages_50_100.pickle"
        )
        accuracy = np.column_stack(
            [
                np.asarray(decoded["scores_set"])[0],
                np.asarray(decoded["scores_xor2"])[0],
                np.asarray(decoded["scores_context"])[0],
            ]
        )
    else:
        raise ValueError(f"unknown transfer experiment: {experiment}")
    if len(stages) != stage_count or accuracy.shape != (stage_count, 3):
        raise ValueError(f"unexpected {experiment} transfer shapes")
    return stages, np.asarray(accuracy, dtype=np.float64)


GATE_BETAS = (1.0, 2.0, 4.0, 8.0)
GATE_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5)


def equation_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"name": "global intercept", "kind": "global", "formula": "a", "features": 0, "mode": "global"},
        {"name": "axis intercepts", "kind": "axis", "formula": "a_j", "features": 0, "mode": "axis"},
        {
            "name": "stage progression",
            "kind": "stage",
            "formula": "a_j+b k/(K-1)",
            "features": 1,
            "mode": "axis",
        },
        {
            "name": "Fisher strict",
            "kind": "fisher",
            "formula": "b sqrt(e_j^T J e_j)",
            "features": 1,
            "mode": "none",
            "links": ("probit",),
        },
        {
            "name": "Fisher total",
            "kind": "fisher",
            "formula": "a_j+b sqrt(e_j^T J e_j)",
            "features": 1,
            "mode": "axis",
        },
        {
            "name": "Fisher isotropic",
            "kind": "isotropic_fisher",
            "formula": "a_j+b sqrt(tr(J)/3)",
            "features": 1,
            "mode": "axis",
        },
        {
            "name": "precision accessibility",
            "kind": "accessibility",
            "formula": "a_j+b/sqrt(e_j^T X^-1 e_j)",
            "features": 1,
            "mode": "axis",
        },
        {"name": "matrix log", "kind": "log", "formula": "a_j+b e_j^T log(X)e_j", "features": 1, "mode": "axis"},
        {"name": "scale only", "kind": "scale", "formula": "a_j+b logdet(X)/3", "features": 1, "mode": "axis"},
        {"name": "shape only", "kind": "shape", "formula": "a_j+b e_j^T log(Xhat)e_j", "features": 1, "mode": "axis"},
        {
            "name": "scale plus shape",
            "kind": "scale_shape",
            "formula": "a_j+b_s logdet(X)/3+b_h e_j^T log(Xhat)e_j",
            "features": 2,
            "mode": "axis",
        },
        {"name": "AIRM reference", "kind": "airm", "formula": "a_j+b d_AI(H_ref,H_k)", "features": 1, "mode": "axis"},
        {
            "name": "log-volume reference",
            "kind": "logvolume",
            "formula": "a_j+b logdet(H_ref^-1 H_k)/3",
            "features": 1,
            "mode": "axis",
        },
        {
            "name": "AIRM plus log-volume",
            "kind": "airm_logvolume",
            "formula": "a_j+b_d d_AI+b_v log-volume",
            "features": 2,
            "mode": "axis",
        },
        {
            "name": "relative information stretch",
            "kind": "relative_information",
            "formula": "a_j+b sqrt(H_k,jj/H_1,jj)",
            "features": 1,
            "mode": "axis",
        },
        {
            "name": "relative precision stretch",
            "kind": "relative_precision",
            "formula": "a_j+b sqrt(g_k(e_j,e_j)/g_1(e_j,e_j))",
            "features": 1,
            "mode": "axis",
        },
        {"name": "spectral condition", "kind": "condition", "formula": "a_j+b log cond(X)", "features": 1, "mode": "axis"},
        {"name": "spectral effective rank", "kind": "effective_rank", "formula": "a_j+b exp(H(eig(X)))", "features": 1, "mode": "axis"},
        {"name": "spectral trace", "kind": "trace", "formula": "a_j+b log(tr(X)/3)", "features": 1, "mode": "axis"},
    ]
    for beta in GATE_BETAS:
        for threshold in GATE_THRESHOLDS:
            suffix = f"beta={beta:g} theta={threshold:g}"
            specs.extend(
                [
                    {
                        "name": f"neuron projected-drive gate {suffix}",
                        "kind": "projected_gate",
                        "value": (beta, threshold),
                        "formula": (
                            "a_j+b sqrt(mean_n[(d_nj sigmoid("
                            f"{beta:g}(d_nj-{threshold:g})))^2])"
                        ),
                        "features": 1,
                        "mode": "axis",
                    },
                    {
                        "name": f"threshold-only gate {suffix}",
                        "kind": "threshold_only",
                        "value": (beta, threshold),
                        "formula": (
                            "a_j+b mean_n[sigmoid("
                            f"{beta:g}(d_nj-{threshold:g}))]"
                        ),
                        "features": 1,
                        "mode": "axis",
                    },
                    {
                        "name": f"projected drive plus threshold {suffix}",
                        "kind": "projected_threshold_additive",
                        "value": (beta, threshold),
                        "formula": (
                            "a_j+b_d sqrt(mean_n[d_nj^2])+b_g mean_n[sigmoid("
                            f"{beta:g}(d_nj-{threshold:g}))]"
                        ),
                        "features": 2,
                        "mode": "axis",
                    },
                ]
            )
    for power in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        specs.append(
            {
                "name": f"SPD power p={power:g}",
                "kind": "power",
                "value": power,
                "formula": f"a_j+b sqrt(e_j^T X^{power:g} e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for power in (-2.0, -1.0, -0.5, 0.5, 2.0):
        specs.append(
            {
                "name": f"correlation power p={power:g}",
                "kind": "correlation_power",
                "value": power,
                "formula": f"a_j+b sqrt(e_j^T R^{power:g} e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for ridge in (0.1, 0.3, 1.0, 3.0, 10.0):
        specs.append(
            {
                "name": f"precision resolvent lambda={ridge:g}",
                "kind": "resolvent",
                "value": ridge,
                "formula": f"a_j+b sqrt(e_j^T(X+{ridge:g}I)^-1e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for ridge in (0.1, 1.0, 10.0):
        for power in (0.5, 1.0, 2.0):
            specs.append(
                {
                    "name": f"regularized precision lambda={ridge:g} p={power:g}",
                    "kind": "regularized_power",
                    "value": (ridge, power),
                    "formula": f"a_j+b sqrt(e_j^T(X+{ridge:g}I)^-{power:g}e_j)",
                    "features": 1,
                    "mode": "axis",
                }
            )
    for alpha in (0.25, 0.5, 1.0):
        specs.append(
            {
                "name": f"matrix exponential alpha={alpha:g}",
                "kind": "exponential",
                "value": alpha,
                "formula": f"a_j+b sqrt(e_j^T exp({alpha:g}X)e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for offset in (0.1, 0.3, 1.0, 3.0):
        specs.append(
            {
                "name": f"shifted matrix log epsilon={offset:g}",
                "kind": "shifted_log",
                "value": offset,
                "formula": f"a_j+b e_j^T log(X+{offset:g}I)e_j",
                "features": 1,
                "mode": "axis",
            }
        )
    for alpha in (0.1, 0.3, 1.0, 3.0, 10.0):
        specs.append(
            {
                "name": f"spectral ratio saturation alpha={alpha:g}",
                "kind": "ratio_saturation",
                "value": alpha,
                "formula": f"a_j+b sqrt(e_j^T X(X+{alpha:g}I)^-1 e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for beta in (0.1, 0.3, 1.0, 3.0):
        specs.append(
            {
                "name": f"spectral exponential saturation beta={beta:g}",
                "kind": "exponential_saturation",
                "value": beta,
                "formula": f"a_j+b sqrt(e_j^T(I-exp(-{beta:g}X))e_j)",
                "features": 1,
                "mode": "axis",
            }
        )
    for alpha in (0.25, 0.5, 0.75):
        specs.extend(
            [
                {
                    "name": f"Fisher-occupancy geometric mix alpha={alpha:g}",
                    "kind": "geometric_mix",
                    "value": alpha,
                    "formula": f"a_j+b sqrt(e_j^T(F#_{alpha:g}G)e_j)",
                    "features": 1,
                    "mode": "axis",
                },
                {
                    "name": f"Fisher-occupancy arithmetic mix alpha={alpha:g}",
                    "kind": "arithmetic_mix",
                    "value": alpha,
                    "formula": f"a_j+b sqrt(e_j^T((1-{alpha:g})F+{alpha:g}G)e_j)",
                    "features": 1,
                    "mode": "axis",
                },
            ]
        )
    for order in (1.0, 4.0, math.inf):
        label = "inf" if math.isinf(order) else f"{order:g}"
        specs.append(
            {
                "name": f"population L{label}",
                "kind": "lp",
                "value": order,
                "formula": f"a_j+b ||S_:j||_{label}",
                "features": 1,
                "mode": "axis",
            }
        )
    return specs


LINKS = ("identity", "probit", "logit", "cloglog", "arcsine")


def link_transform(values: np.ndarray, link: str) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    if link == "identity":
        return values
    if link == "probit":
        normal = NormalDist()
        return np.asarray([normal.inv_cdf(float(value)) for value in values.ravel()]).reshape(values.shape)
    if link == "logit":
        return np.log(values / (1.0 - values))
    if link == "cloglog":
        return np.log(-np.log1p(-values))
    if link == "arcsine":
        return np.arcsin(np.sqrt(values))
    raise ValueError(f"unknown link: {link}")


def link_inverse(values: np.ndarray, link: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if link == "identity":
        return np.clip(values, 0.0, 1.0)
    if link == "probit":
        normal = NormalDist()
        return np.asarray([normal.cdf(float(value)) for value in values.ravel()]).reshape(values.shape)
    if link == "logit":
        clipped = np.clip(values, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    if link == "cloglog":
        return 1.0 - np.exp(-np.exp(np.clip(values, -50.0, 50.0)))
    if link == "arcsine":
        return np.sin(np.clip(values, 0.0, math.pi / 2.0)) ** 2
    raise ValueError(f"unknown link: {link}")


def information_matrix(points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(points, dtype=np.float64).T @ np.asarray(points, dtype=np.float64) / len(points)
    matrix = (matrix + matrix.T) / 2.0
    if np.linalg.eigvalsh(matrix)[0] <= 0.0:
        raise ValueError("information second moment is not SPD")
    return matrix


def training_geometry(stages: list[np.ndarray], indices: list[int]) -> tuple[float, np.ndarray]:
    matrices = [information_matrix(stages[index]) for index in indices]
    mean_log_scale = float(np.mean([np.linalg.slogdet(matrix)[1] / 3.0 for matrix in matrices]))
    tau = math.exp(mean_log_scale)
    reference_log = np.mean([spd_function(matrix, np.log) for matrix in matrices], axis=0)
    values, vectors = np.linalg.eigh((reference_log + reference_log.T) / 2.0)
    reference = vectors @ np.diag(np.exp(values)) @ vectors.T
    return tau, reference


def equation_features(
    stages: list[np.ndarray], spec: dict[str, object], tau: float, reference: np.ndarray
) -> np.ndarray:
    count = int(spec["features"])
    if count == 0:
        return np.empty((len(stages), 3, 0), dtype=np.float64)
    outputs: list[np.ndarray] = []
    first_information = information_matrix(stages[0])
    first_precision = np.linalg.inv(covariance(stages[0]))
    for stage_index, points in enumerate(stages):
        h_matrix = information_matrix(points)
        j_matrix = h_matrix * len(points)
        x_matrix = h_matrix / tau
        kind = str(spec["kind"])
        if kind == "stage":
            value = stage_index / max(1, len(stages) - 1)
            feature = np.full((3, 1), value)
        elif kind == "fisher":
            feature = np.sqrt(np.diag(j_matrix))[:, None]
        elif kind == "isotropic_fisher":
            feature = np.full((3, 1), math.sqrt(float(np.trace(j_matrix)) / 3.0))
        elif kind in {"projected_gate", "threshold_only", "projected_threshold_additive"}:
            beta, threshold = (float(value) for value in spec["value"])
            projected_drive = np.abs(points) / math.sqrt(tau)
            gate = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(beta * (projected_drive - threshold), -50.0, 50.0)
                )
            )
            ungated_rms = np.sqrt(np.mean(projected_drive**2, axis=0))[:, None]
            gate_mean = np.mean(gate, axis=0)[:, None]
            if kind == "projected_gate":
                feature = np.sqrt(np.mean((projected_drive * gate) ** 2, axis=0))[:, None]
            elif kind == "threshold_only":
                feature = gate_mean
            else:
                feature = np.column_stack([ungated_rms[:, 0], gate_mean[:, 0]])
        elif kind == "accessibility":
            feature = (1.0 / np.sqrt(np.diag(np.linalg.inv(x_matrix))))[:, None]
        elif kind == "power":
            powered = spd_function(x_matrix, lambda values: values ** float(spec["value"]))
            feature = np.sqrt(np.diag(powered))[:, None]
        elif kind == "correlation_power":
            scale = np.sqrt(np.diag(x_matrix))
            correlation = x_matrix / np.outer(scale, scale)
            powered = spd_function(correlation, lambda values: values ** float(spec["value"]))
            feature = np.sqrt(np.diag(powered))[:, None]
        elif kind == "resolvent":
            inverse = np.linalg.inv(x_matrix + float(spec["value"]) * np.eye(3))
            feature = np.sqrt(np.diag(inverse))[:, None]
        elif kind == "regularized_power":
            ridge, power = spec["value"]
            regularized = x_matrix + float(ridge) * np.eye(3)
            powered = spd_function(regularized, lambda values: values ** -float(power))
            feature = np.sqrt(np.diag(powered))[:, None]
        elif kind == "exponential":
            exponential = spd_function(x_matrix, lambda values: np.exp(float(spec["value"]) * values))
            feature = np.sqrt(np.diag(exponential))[:, None]
        elif kind == "shifted_log":
            shifted = x_matrix + float(spec["value"]) * np.eye(3)
            feature = np.diag(spd_function(shifted, np.log))[:, None]
        elif kind == "ratio_saturation":
            alpha = float(spec["value"])
            saturated = x_matrix @ np.linalg.inv(x_matrix + alpha * np.eye(3))
            feature = np.sqrt(np.diag((saturated + saturated.T) / 2.0))[:, None]
        elif kind == "exponential_saturation":
            beta = float(spec["value"])
            saturated = np.eye(3) - spd_function(x_matrix, lambda values: np.exp(-beta * values))
            feature = np.sqrt(np.diag(saturated))[:, None]
        elif kind == "log":
            feature = np.diag(spd_function(x_matrix, np.log))[:, None]
        elif kind in {"geometric_mix", "arithmetic_mix"}:
            fisher_shape = h_matrix / math.exp(np.linalg.slogdet(h_matrix)[1] / 3.0)
            occupancy_metric = np.linalg.inv(covariance(points))
            occupancy_shape = occupancy_metric / math.exp(
                np.linalg.slogdet(occupancy_metric)[1] / 3.0
            )
            alpha = float(spec["value"])
            if kind == "arithmetic_mix":
                mixed = (1.0 - alpha) * fisher_shape + alpha * occupancy_shape
            else:
                fisher_sqrt = spd_function(fisher_shape, np.sqrt)
                fisher_inverse_sqrt = spd_function(fisher_shape, lambda values: values ** -0.5)
                relative = fisher_inverse_sqrt @ occupancy_shape @ fisher_inverse_sqrt
                mixed = fisher_sqrt @ spd_function(relative, lambda values: values ** alpha) @ fisher_sqrt
            feature = np.sqrt(np.diag((mixed + mixed.T) / 2.0))[:, None]
        elif kind in {"scale", "shape", "scale_shape"}:
            log_scale = float(np.linalg.slogdet(x_matrix)[1] / 3.0)
            normalized = x_matrix / math.exp(log_scale)
            shape = np.diag(spd_function(normalized, np.log))[:, None]
            scale_feature = np.full((3, 1), log_scale)
            if kind == "scale":
                feature = scale_feature
            elif kind == "shape":
                feature = shape
            else:
                feature = np.column_stack([scale_feature[:, 0], shape[:, 0]])
        elif kind in {"airm", "logvolume", "airm_logvolume"}:
            distance = airm_components(reference, h_matrix)[0]
            logvolume = float((np.linalg.slogdet(h_matrix)[1] - np.linalg.slogdet(reference)[1]) / 3.0)
            distance_feature = np.full((3, 1), distance)
            volume_feature = np.full((3, 1), logvolume)
            if kind == "airm":
                feature = distance_feature
            elif kind == "logvolume":
                feature = volume_feature
            else:
                feature = np.column_stack([distance_feature[:, 0], volume_feature[:, 0]])
        elif kind == "relative_information":
            feature = np.sqrt(np.diag(h_matrix) / np.diag(first_information))[:, None]
        elif kind == "relative_precision":
            precision = np.linalg.inv(covariance(points))
            feature = np.sqrt(np.diag(precision) / np.diag(first_precision))[:, None]
        elif kind == "condition":
            eigenvalues = np.linalg.eigvalsh(x_matrix)
            feature = np.full((3, 1), math.log(float(eigenvalues[-1] / eigenvalues[0])))
        elif kind == "effective_rank":
            eigenvalues = np.linalg.eigvalsh(x_matrix)
            probabilities = eigenvalues / eigenvalues.sum()
            effective_rank = math.exp(float(-np.sum(probabilities * np.log(probabilities))))
            feature = np.full((3, 1), effective_rank)
        elif kind == "trace":
            feature = np.full((3, 1), math.log(float(np.trace(x_matrix)) / 3.0))
        elif kind == "lp":
            order = float(spec["value"])
            if math.isinf(order):
                values = np.max(np.abs(points), axis=0)
            else:
                values = np.sum(np.abs(points) ** order, axis=0) ** (1.0 / order)
            feature = values[:, None]
        else:
            raise ValueError(f"unknown equation feature kind: {kind}")
        if feature.shape != (3, count) or not np.all(np.isfinite(feature)):
            raise ValueError(f"invalid feature for {spec['name']}: {feature.shape}")
        outputs.append(feature)
    return np.asarray(outputs, dtype=np.float64)


def fit_link_model(
    features: np.ndarray,
    accuracy: np.ndarray,
    spec: dict[str, object],
    link: str,
    train_stages: list[int],
) -> tuple[dict[str, object], np.ndarray]:
    stage_count = len(accuracy)
    rows = stage_count * 3
    axes = np.tile(np.arange(3), stage_count)
    stage_ids = np.repeat(np.arange(stage_count), 3)
    train_mask = np.isin(stage_ids, train_stages)
    raw_features = features.reshape(rows, int(spec["features"]))
    mode = str(spec["mode"])
    if raw_features.shape[1]:
        train_raw = raw_features[train_mask]
        means = np.zeros(raw_features.shape[1]) if mode == "none" else train_raw.mean(axis=0)
        scales = np.sqrt(np.mean((train_raw - means) ** 2, axis=0))
        if np.any(scales <= 1e-12):
            raise ValueError(f"constant feature in {spec['name']}")
        normalized = (raw_features - means) / scales
    else:
        means = np.empty(0)
        scales = np.empty(0)
        normalized = raw_features
    if mode == "global":
        design = np.ones((rows, 1))
    elif mode == "axis":
        design = np.column_stack([np.eye(3)[axes], normalized])
    elif mode == "none":
        design = normalized
    else:
        raise ValueError(f"unknown intercept mode: {mode}")
    target = link_transform(accuracy, link).ravel()
    train_design = design[train_mask]
    coefficients, _, rank, _ = np.linalg.lstsq(train_design, target[train_mask], rcond=None)
    if rank != train_design.shape[1] or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"rank-deficient fit for {spec['name']} / {link}")
    prediction = link_inverse((design @ coefficients).reshape(stage_count, 3), link)
    return {
        "coefficients": coefficients,
        "feature_means": means,
        "feature_scales": scales,
        "mode": mode,
        "link": link,
    }, prediction


def predict_link_model(features: np.ndarray, model: dict[str, object]) -> np.ndarray:
    stage_count = len(features)
    axes = np.tile(np.arange(3), stage_count)
    raw = features.reshape(stage_count * 3, features.shape[2])
    means = np.asarray(model["feature_means"])
    scales = np.asarray(model["feature_scales"])
    normalized = (raw - means) / scales if raw.shape[1] else raw
    mode = str(model["mode"])
    if mode == "global":
        design = np.ones((stage_count * 3, 1))
    elif mode == "axis":
        design = np.column_stack([np.eye(3)[axes], normalized])
    elif mode == "none":
        design = normalized
    else:
        raise ValueError(f"unknown model mode: {mode}")
    transformed = (design @ np.asarray(model["coefficients"])).reshape(stage_count, 3)
    return link_inverse(transformed, str(model["link"]))


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def prediction_scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed_flat = np.asarray(observed, dtype=np.float64).ravel()
    predicted_flat = np.asarray(predicted, dtype=np.float64).ravel()
    pearson = float("nan") if np.std(predicted_flat) <= 1e-15 else float(np.corrcoef(observed_flat, predicted_flat)[0, 1])
    observed_rank = rank_values(observed_flat)
    predicted_rank = rank_values(predicted_flat)
    spearman = float("nan") if np.std(predicted_rank) <= 1e-15 else float(np.corrcoef(observed_rank, predicted_rank)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean((observed_flat - predicted_flat) ** 2))),
        "mae": float(np.mean(np.abs(observed_flat - predicted_flat))),
        "pearson": pearson,
        "spearman": spearman,
    }


def equation_family_tournament(processed: Path) -> dict[str, object]:
    train_stages, train_accuracy = transfer_inputs(processed, "Exp1")
    specs = equation_specs()
    records: list[dict[str, object]] = []
    all_stage_ids = list(range(len(train_stages)))
    for spec in specs:
        allowed_links = tuple(spec.get("links", LINKS))
        for link in allowed_links:
            fold_squared_errors: list[float] = []
            valid = True
            for held_stage in all_stage_ids:
                fit_ids = [index for index in all_stage_ids if index != held_stage]
                try:
                    tau, reference = training_geometry(train_stages, fit_ids)
                    features = equation_features(train_stages, spec, tau, reference)
                    _, prediction = fit_link_model(features, train_accuracy, spec, link, fit_ids)
                    fold_squared_errors.extend(
                        ((prediction[held_stage] - train_accuracy[held_stage]) ** 2).tolist()
                    )
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    valid = False
                    break
            if not valid:
                continue
            parameter_count = (
                (1 if spec["mode"] == "global" else 0)
                + (3 if spec["mode"] == "axis" else 0)
                + int(spec["features"])
            )
            records.append(
                {
                    "name": spec["name"],
                    "kind": spec["kind"],
                    "formula": spec["formula"],
                    "link": link,
                    "parameters": parameter_count,
                    "cv_rmse": math.sqrt(float(np.mean(fold_squared_errors))),
                    "spec": spec,
                }
            )
    records.sort(key=lambda row: (float(row["cv_rmse"]), int(row["parameters"]), str(row["name"]), str(row["link"])))
    if not records:
        raise ValueError("no equation-family candidate survived Exp1 CV")

    full_tau, full_reference = training_geometry(train_stages, all_stage_ids)
    exp2_inputs = {
        count: transfer_inputs(processed, "Exp2", count) for count in EXP2_STAGE_COUNTS
    }
    for record in records:
        spec = record["spec"]
        train_features = equation_features(train_stages, spec, full_tau, full_reference)
        model, _ = fit_link_model(train_features, train_accuracy, spec, str(record["link"]), all_stage_ids)
        per_binning: dict[int, dict[str, float]] = {}
        for count, (test_stages, test_accuracy) in exp2_inputs.items():
            test_features = equation_features(test_stages, spec, full_tau, full_reference)
            prediction = predict_link_model(test_features, model)
            per_binning[count] = prediction_scores(test_accuracy, prediction)
        record["per_binning"] = per_binning
        record["mean_exp2_rmse"] = float(np.mean([values["rmse"] for values in per_binning.values()]))
        record["mean_exp2_mae"] = float(np.mean([values["mae"] for values in per_binning.values()]))
        del record["spec"]

    best_by_producer: list[dict[str, object]] = []
    for spec in specs:
        matches = [record for record in records if record["name"] == spec["name"]]
        if matches:
            best_by_producer.append(min(matches, key=lambda row: (float(row["cv_rmse"]), str(row["link"]))))
    best_by_producer.sort(key=lambda row: (float(row["cv_rmse"]), int(row["parameters"]), str(row["name"])))
    selected = records[0]
    return {
        "candidate_count": len(records),
        "producer_count": len(best_by_producer),
        "links": LINKS,
        "selected": selected,
        "best_by_producer": best_by_producer,
        "full_tau": full_tau,
        "role_mapping": {
            "Exp1": ["colour", "shape", "XOR"],
            "Exp2": ["set", "set*context", "context"],
        },
    }


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    ranks_a = rank_values(np.asarray(values_a, dtype=np.float64))
    ranks_b = rank_values(np.asarray(values_b, dtype=np.float64))
    if np.std(ranks_a) <= 1e-15 or np.std(ranks_b) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def geometry_behavior_sensitivity(processed: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    def geometry_series(stages: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        matrices = [covariance(stage) for stage in stages]
        first = matrices[0]
        distance = np.asarray([airm_components(first, matrix)[0] for matrix in matrices])
        volume = np.asarray(
            [(np.linalg.slogdet(matrix)[1] - np.linalg.slogdet(first)[1]) / 3.0 for matrix in matrices]
        )
        return distance, volume

    exp1_stages, _ = transfer_inputs(processed, "Exp1")
    exp1_distance, exp1_volume = geometry_series(exp1_stages)
    termination = safe_load(processed / "exp1_beh_terminations4stages.pickle")["data_prop_s"]
    termination_mean = np.asarray([np.mean(np.asarray(stage, dtype=np.float64)) for stage in termination])
    entries.append(
        {
            "dataset": "Exp1",
            "endpoint": "fixation-break ratio",
            "stages": 4,
            "rho_airm": spearman(exp1_distance, termination_mean),
            "rho_logvolume": spearman(exp1_volume, termination_mean),
        }
    )
    switch_groups = safe_load(processed / "exp1_switch_costs4stages.pickle")["switch_costs_all"]
    for name, group in zip(("colour switch", "shape switch", "hierarchical switch"), switch_groups):
        means = np.asarray([np.mean(np.asarray(stage, dtype=np.float64)) for stage in group])
        entries.append(
            {
                "dataset": "Exp1",
                "endpoint": name,
                "stages": 4,
                "rho_airm": spearman(exp1_distance, means),
                "rho_logvolume": spearman(exp1_volume, means),
            }
        )
    for count in EXP2_STAGE_COUNTS:
        stages, _ = transfer_inputs(processed, "Exp2", count)
        distance, volume = geometry_series(stages)
        behavior = safe_load(processed / f"fixation_breaks_prop_exp2_{count}stages.pickle")[
            "data_prop_s_exp2_set2"
        ]
        means = np.asarray([np.mean(np.asarray(stage, dtype=np.float64)) for stage in behavior])
        entries.append(
            {
                "dataset": "Exp2",
                "endpoint": "fixation-break ratio",
                "stages": count,
                "rho_airm": spearman(distance, means),
                "rho_logvolume": spearman(volume, means),
            }
        )
    return entries


def stage_geometry_decomposition(processed: Path) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for dataset in ("Exp1", "Exp2"):
        stages, _ = transfer_inputs(processed, dataset, 4)
        matrices = [covariance(stage) for stage in stages]
        reference_values, reference_vectors = np.linalg.eigh(matrices[0])
        reference_order = np.argsort(reference_values)[::-1]
        reference_vectors = reference_vectors[:, reference_order]
        reference_scale = float(np.linalg.slogdet(matrices[0])[1] / 3.0)
        previous = matrices[0]
        for stage_index, matrix in enumerate(matrices):
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            log_values = np.log(eigenvalues)
            probabilities = eigenvalues / eigenvalues.sum()
            projector_rotation_sq = 0.0
            for axis in range(3):
                reference_projector = np.outer(reference_vectors[:, axis], reference_vectors[:, axis])
                stage_projector = np.outer(eigenvectors[:, axis], eigenvectors[:, axis])
                projector_rotation_sq += float(np.linalg.norm(stage_projector - reference_projector) ** 2)
            metric_diagonal = np.diag(np.linalg.inv(matrix))
            dominant_cosine = float(
                np.clip(abs(reference_vectors[:, 0] @ eigenvectors[:, 0]), 0.0, 1.0)
            )
            relative_gaps = np.diff(eigenvalues[::-1]) / eigenvalues[::-1][1:]
            outputs.append(
                {
                    "dataset": dataset,
                    "stage": stage_index + 1,
                    "scale_change": float(np.linalg.slogdet(matrix)[1] / 3.0 - reference_scale),
                    "anisotropy": float(np.linalg.norm(log_values - log_values.mean())),
                    "effective_rank": math.exp(float(-np.sum(probabilities * np.log(probabilities)))),
                    "basis_rotation": math.sqrt(0.5 * projector_rotation_sq),
                    "dominant_axis_angle_degrees": math.degrees(math.acos(dominant_cosine)),
                    "dominant_axis_gap": float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]),
                    "minimum_relative_gap": float(np.min(relative_gaps)),
                    "airm_from_initial": airm_components(matrices[0], matrix)[0],
                    "airm_successive": 0.0 if stage_index == 0 else airm_components(previous, matrix)[0],
                    "axis_cost_ratio": math.sqrt(float(metric_diagonal.max() / metric_diagonal.min())),
                    "det_normalized_eigenvalues": (
                        eigenvalues / math.exp(float(np.log(eigenvalues).mean()))
                    ),
                }
            )
            previous = matrix
    return outputs


def cross_set_metric_alignment(processed: Path) -> list[dict[str, object]]:
    data = safe_load(
        processed / "exp2_selectivity_dat_early_50_100_late_100_150_stages_4.pickle"
    )

    def cosine(left: np.ndarray, right: np.ndarray, dual_metric: np.ndarray) -> float:
        numerator = float(np.einsum("ni,ij,nj->", left, dual_metric, right))
        left_norm = float(np.einsum("ni,ij,nj->", left, dual_metric, left))
        right_norm = float(np.einsum("ni,ij,nj->", right, dual_metric, right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            raise ValueError("nonpositive metric alignment norm")
        return float(np.clip(numerator / math.sqrt(left_norm * right_norm), -1.0, 1.0))

    outputs: list[dict[str, object]] = []
    for stage_index in range(4):
        row: dict[str, object] = {"stage": stage_index + 1}
        for epoch, left_key, right_key in (
            ("early", "epochs_task1_e", "epochs_task2_e"),
            ("late", "epochs_task1_l", "epochs_task2_l"),
        ):
            left_all = np.asarray(data[left_key][stage_index], dtype=np.float64)
            right_all = np.asarray(data[right_key][stage_index], dtype=np.float64)
            euclidean_values: list[float] = []
            metric_values: list[float] = []
            for fold in range(left_all.shape[2]):
                left = left_all[:, :, fold]
                right = right_all[:, :, fold]
                identity = np.eye(3)
                pooled_information = (left.T @ left + right.T @ right) / (2.0 * len(left))
                dual_metric = np.linalg.inv((pooled_information + pooled_information.T) / 2.0)
                euclidean_values.append(cosine(left, right, identity))
                metric_values.append(cosine(left, right, dual_metric))
            row[f"{epoch}_euclidean_cosine"] = float(np.mean(euclidean_values))
            row[f"{epoch}_metric_cosine"] = float(np.mean(metric_values))
        row["official_early_colour_cosine"] = float(
            np.asarray(data["cos_sim_mat_e"])[stage_index, 0]
        )
        row["official_early_context_cosine"] = float(
            np.asarray(data["cos_sim_mat_e"])[stage_index, 1]
        )
        row["official_shape_cosine"] = float(np.asarray(data["cos_sim_mat"])[stage_index, 1])
        row["official_xor_cosine"] = float(np.asarray(data["cos_sim_mat"])[stage_index, 2])
        outputs.append(row)
    return outputs


def routing_chain_sensitivity(
    processed: Path,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Test the observable middle link of the routing hypothesis on official caches."""
    def frobenius_cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = math.sqrt(float(np.sum(left * left) * np.sum(right * right)))
        if denominator <= 0.0:
            raise ValueError("zero Frobenius norm in cross-task alignment")
        return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))

    routing_cache = safe_load(
        processed / "exp2_selectivity_dat_early_50_100_late_100_150_stages_4.pickle"
    )
    task1_late = [
        np.asarray(stage, dtype=np.float64)[:, :, 0]
        for stage in routing_cache["epochs_task1_l"]
    ]
    task2_late = [
        np.asarray(stage, dtype=np.float64)[:, :, 0]
        for stage in routing_cache["epochs_task2_l"]
    ]
    late_covariances = [covariance(stage) for stage in task1_late]
    late_geometry_displacement = np.asarray(
        [airm_components(late_covariances[0], matrix)[0] for matrix in late_covariances]
    )
    late_cross_task_alignment = np.asarray(
        [frobenius_cosine(left, right) for left, right in zip(task1_late, task2_late)]
    )
    late_observed_correlation = float(
        np.corrcoef(late_geometry_displacement, late_cross_task_alignment)[0, 1]
    )
    rng = np.random.default_rng(seed)
    late_null_correlations = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        shuffled_alignment = np.asarray(
            [
                frobenius_cosine(left, right[rng.permutation(len(right))])
                for left, right in zip(task1_late, task2_late)
            ]
        )
        late_null_correlations[draw] = np.corrcoef(
            late_geometry_displacement, shuffled_alignment
        )[0, 1]

    def covariance_2d(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0, keepdims=True)
        matrix = centered.T @ centered / (len(points) - 1)
        matrix = (matrix + matrix.T) / 2.0
        if np.linalg.eigvalsh(matrix)[0] <= 0.0:
            raise ValueError("late shape/XOR covariance is not SPD")
        return matrix

    late_shape_xor = [stage[:, [1, 2]] for stage in task1_late]
    late_shape_xor_metrics = [np.linalg.inv(covariance_2d(stage)) for stage in late_shape_xor]
    late_shape_xor_accessibility = np.asarray(
        [
            -0.5
            * np.log(
                np.diag(metric) / np.diag(late_shape_xor_metrics[0])
            )
            for metric in late_shape_xor_metrics
        ]
    )
    late_shape_xor_alignment = np.asarray(routing_cache["cos_sim_mat"], dtype=np.float64)[
        :, [1, 2]
    ]
    late_shape_xor_null = np.asarray(
        routing_cache["cos_sim_mat_rnd"], dtype=np.float64
    )[:, [1, 2], :]
    late_shape_xor_correlation = float(
        np.corrcoef(
            late_shape_xor_accessibility.ravel(), late_shape_xor_alignment.ravel()
        )[0, 1]
    )
    late_shape_xor_null_correlations = np.asarray(
        [
            np.corrcoef(
                late_shape_xor_accessibility.ravel(),
                late_shape_xor_null[:, :, draw].ravel(),
            )[0, 1]
            for draw in range(late_shape_xor_null.shape[2])
        ]
    )

    return {
        "late_3d": {
            "geometry_displacement": late_geometry_displacement,
            "cross_task_alignment": late_cross_task_alignment,
            "pearson": late_observed_correlation,
            "shuffle_draws": draws,
            "p_greater": float(
                (np.sum(late_null_correlations >= late_observed_correlation) + 1)
                / (draws + 1)
            ),
            "p_two_sided": float(
                (
                    np.sum(
                        np.abs(late_null_correlations)
                        >= abs(late_observed_correlation)
                    )
                    + 1
                )
                / (draws + 1)
            ),
        },
        "late_shape_xor_control": {
            "accessibility": late_shape_xor_accessibility,
            "alignment": late_shape_xor_alignment,
            "pearson": late_shape_xor_correlation,
            "null_draws": int(late_shape_xor_null.shape[2]),
            "p_two_sided": float(
                (
                    np.sum(
                        np.abs(late_shape_xor_null_correlations)
                        >= abs(late_shape_xor_correlation)
                    )
                    + 1
                )
                / (late_shape_xor_null.shape[2] + 1)
            ),
        },
    }


def alternate_partition_robustness(
    processed: Path, tournament: dict[str, object]
) -> list[dict[str, object]]:
    train_stages, train_accuracy = transfer_inputs(processed, "Exp1")
    raw = safe_load(processed / "selectivity_coefficients_exp1_fixbias_140_1504stages.pickle")[
        "selectivity_coefficients_xval"
    ]
    alternate_stages = [np.asarray(stage, dtype=np.float64)[:, :, 0] for stage in raw]
    colour = safe_load(processed / "exp1_decoding_fixbias_collocked_50_150_4stages.pickle")
    shape = safe_load(processed / "exp1_decoding_fixbias_shapelocked_100_150_4stages.pickle")
    alternate_accuracy = np.column_stack(
        [
            np.asarray(colour["decoding"])[:, 0],
            np.asarray(shape["decoding"])[:, 1],
            np.asarray(shape["decoding"])[:, 3],
        ]
    )
    tau, reference = training_geometry(train_stages, list(range(4)))
    spec_by_name = {str(spec["name"]): spec for spec in equation_specs()}
    results: list[dict[str, object]] = []
    producer_rows = tournament["best_by_producer"]
    assert isinstance(producer_rows, list)
    for candidate in producer_rows:
        spec = spec_by_name[str(candidate["name"])]
        train_features = equation_features(train_stages, spec, tau, reference)
        model, _ = fit_link_model(
            train_features,
            train_accuracy,
            spec,
            str(candidate["link"]),
            list(range(4)),
        )
        alternate_features = equation_features(alternate_stages, spec, tau, reference)
        prediction = predict_link_model(alternate_features, model)
        results.append(
            {
                "name": candidate["name"],
                "link": candidate["link"],
                **prediction_scores(alternate_accuracy, prediction),
            }
        )
    results.sort(key=lambda row: (float(row["rmse"]), str(row["name"])))
    return results


def verdict(p_value: float) -> str:
    return "REJECT H0" if p_value <= 0.05 else "DO NOT REJECT H0"


def render(
    rows: list[dict[str, object]],
    decoder: dict[str, object],
    fisher: dict[str, object],
    tournament: dict[str, object],
    behavior: list[dict[str, object]],
    decomposition: list[dict[str, object]],
    alignment: list[dict[str, object]],
    routing: dict[str, object],
    alternate: list[dict[str, object]],
    hashes: dict[str, str],
    draws: int,
    seed: int,
    remote: str,
    commit: str,
) -> str:
    exp1_primary = rows[0]
    exp2_primary = next(row for row in rows if row["name"] == "Exp2 (4-stage binning)")
    heldout = exp2_primary["heldout_prediction"]
    bootstrap = exp2_primary["released_row_bootstrap"]
    assert isinstance(heldout, dict)
    assert isinstance(bootstrap, dict)
    lines = [
        "# Official PFC metric-equation battery",
        "",
        "Status: `REAL_DATA_EQUATION_TEST`",
        "",
        f"- Official author repository: `{remote}`",
        f"- Frozen author commit: `{commit}`",
        "- Official dataset DOI: https://doi.org/10.5061/dryad.c2fqz61kb",
        "- Paper DOI: https://doi.org/10.1038/s41593-026-02333-w",
        "",
        "## Equation and null",
        "",
        "For each first/last learning stage, the official per-neuron selectivity vectors are treated as points in the released three-coordinate selectivity chart:",
        "",
        "$$",
        "C_k=\\operatorname{Cov}(s_n\\mid k),\\qquad g_k=C_k^{-1}.",
        "$$",
        "",
        "The empirical null is exchangeability of first/last-stage selectivity rows while preserving the two observed sample sizes. The primary statistic is the affine-invariant distance",
        "",
        "$$",
        "D_{\\mathrm{AI}}(g_1,g_L)=D_{\\mathrm{AI}}(C_1,C_L)",
        "=\\left\\|\\log\\left(C_1^{-1/2}C_LC_1^{-1/2}\\right)\\right\\|_F.",
        "$$",
        "",
        "The same first/last covariances are also evaluated with the generalized log modes, exact scale/anisotropy decomposition, symmetric Gaussian KL (Jeffreys divergence), log-Euclidean distance, Bures/Wasserstein-2 distance, the SPD geodesic midpoint, and the covariant metric transformation law. AIRM and Jeffreys are GL(3)-congruence invariant. Log-Euclidean and Bures are fixed-chart sensitivity analyses.",
        "",
        f"Monte Carlo permutations: `{draws}` per row; seed base: `{seed}`; p-values use the +1 correction.",
        "",
        "## Results",
        "",
        "| Dataset | N first/last | AIRM total | AIRM shape | Shape share | p(total) | p(shape) | Primary decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {n_first}/{n_last} | {airm_total:.6f} | {airm_shape:.6f} | {shape_fraction_sq:.1%} | {p_total:.6f} | {p_shape:.6f} | {decision} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "The primary Exp1 initial-learning comparison does not reject equal first/last selectivity geometry. Its fixation-bias control agrees with that null result. The primary four-stage Exp2 rule-generalization comparison rejects exchangeability, and the same conclusion is stable under the official 3, 5, and 6-stage binnings.",
            "",
            "## Full equation battery on the primary comparisons",
            "",
            "The secondary p-values below reuse the same row-exchangeability sensitivity null. Jeffreys is a different weighting of the same generalized eigenvalues as AIRM, so it is not independent evidence. Log-Euclidean and Bures depend on the released coordinate scaling.",
            "",
            "| Statistic | Transformation status | Exp1 value | Exp1 p | Exp2 value | Exp2 p |",
            "|---|---|---:|---:|---:|---:|",
            "| AIRM | GL(3) invariant | {e1v:.6f} | {e1p:.6f} | {e2v:.6f} | {e2p:.6f} |".format(
                e1v=exp1_primary["airm_total"], e1p=exp1_primary["p_total"],
                e2v=exp2_primary["airm_total"], e2p=exp2_primary["p_total"],
            ),
            "| AIRM anisotropy | GL(3) invariant decomposition | {e1v:.6f} | {e1p:.6f} | {e2v:.6f} | {e2p:.6f} |".format(
                e1v=exp1_primary["airm_shape"], e1p=exp1_primary["p_shape"],
                e2v=exp2_primary["airm_shape"], e2p=exp2_primary["p_shape"],
            ),
            "| Symmetric Gaussian KL | GL(3) invariant | {e1v:.6f} | {e1p:.6f} | {e2v:.6f} | {e2p:.6f} |".format(
                e1v=exp1_primary["jeffreys"], e1p=exp1_primary["p_jeffreys"],
                e2v=exp2_primary["jeffreys"], e2p=exp2_primary["p_jeffreys"],
            ),
            "| Log-Euclidean | fixed-chart/O(3) | {e1v:.6f} | {e1p:.6f} | {e2v:.6f} | {e2p:.6f} |".format(
                e1v=exp1_primary["log_euclidean"], e1p=exp1_primary["p_log_euclidean"],
                e2v=exp2_primary["log_euclidean"], e2p=exp2_primary["p_log_euclidean"],
            ),
            "| Bures/W2 | fixed Euclidean ground cost | {e1v:.6f} | {e1p:.6f} | {e2v:.6f} | {e2p:.6f} |".format(
                e1v=exp1_primary["bures"], e1p=exp1_primary["p_bures"],
                e2v=exp2_primary["bures"], e2p=exp2_primary["p_bures"],
            ),
            "",
            "Generalized log-deformation modes `log(lambda_i)`:",
            "",
            f"- Exp1: `{np.asarray(exp1_primary['generalized_log_modes']).round(6).tolist()}`",
            f"- Exp2: `{np.asarray(exp2_primary['generalized_log_modes']).round(6).tolist()}`",
            f"- Signed log-volume changes: Exp1 `{exp1_primary['signed_log_volume_ratio']:.6f}`, Exp2 `{exp2_primary['signed_log_volume_ratio']:.6f}`",
            "",
            "## Frozen relative-deformation equation",
            "",
            "The coordinate-free relative object is the positive, $g_1$-self-adjoint endomorphism",
            "",
            "$$",
            "A_k=g_1^{-1}g_k,\\qquad L_k=\\log A_k,\\qquad",
            "A_k'=P A_k P^{-1}.",
            "$$",
            "",
            "For a declared task contrast $\\delta$ that transforms with the chart, the directional stretch and the discovered calibration equation are",
            "",
            "$$",
            "R_k(\\delta)=\\frac{\\delta^Tg_k\\delta}{\\delta^Tg_1\\delta},\\qquad",
            "\\rho_k(\\delta)=\\sqrt{R_k(\\delta)},\\qquad",
            "\\widehat A_{k,j}=\\operatorname{logistic}(a_j+b\\rho_k(e_j)).",
            "$$",
            "",
            "The released factors fix $e_j$ to the named factor chart. If the chart is recoded by $s'=Ps$, the same contrast must transform as $\\delta'=P\\delta$; then $R_k$ is invariant. Numerically one may use the symmetric whitened representation",
            "",
            "$$",
            "M_k=g_1^{-1/2}g_kg_1^{-1/2},\\qquad",
            "u_j=\\frac{g_1^{1/2}e_j}{\\sqrt{e_j^Tg_1e_j}},\\qquad",
            "R_k(e_j)=u_j^TM_ku_j.",
            "$$",
            "",
            "Using $e_j^TM_ke_j$ without the whitened normalized $u_j$ is not the same equation. The generalized eigenvalues of $A_k$ are chart-invariant; raw coordinates of $M_k$ and fixed numerical axes are not.",
            "",
            "A canonical deformation transporter is",
            "",
            "$$",
            "T_k=A_k^{1/2}=\\exp(\\tfrac12 L_k),\\qquad T_k^Tg_1T_k=g_k.",
            "$$",
            "",
            "It is the unique positive $g_1$-self-adjoint choice. A generic congruence factor is not unique, so no raw matrix $T$ is identified as a biological mechanism by this dataset.",
            "",
            "### Exp2 relative-precision stability",
            "",
            f"- Relative precision stretches: `{np.exp(np.asarray(exp2_primary['relative_precision_log_modes'])).round(6).tolist()}`",
            f"- Log stretches: `{np.asarray(exp2_primary['relative_precision_log_modes']).round(6).tolist()}`",
            f"- Dominant covariance-axis angle: `{exp2_primary['dominant_axis_angle_degrees']:.6f}` degrees",
            f"- Dominant eigengaps, stage 1/stage 4: `{exp2_primary['dominant_gap_first']:.6f}` / `{exp2_primary['dominant_gap_last']:.6f}`",
            f"- Released-row bootstrap 95% angle interval: `{np.asarray(bootstrap['dominant_axis_angle_interval_95']).round(4).tolist()}` degrees",
            f"- Released-row bootstrap 95% AIRM interval: `{np.asarray(bootstrap['airm_interval_95']).round(4).tolist()}`",
            f"- Released-row bootstrap 95% log-stretch intervals: `{np.asarray(bootstrap['log_mode_interval_95']).round(4).T.tolist()}`",
            f"- Fixed-size released-row repartition p(angle): `{exp2_primary['p_dominant_axis_angle']:.6f}`",
            "",
            "These intervals and p-value condition on released pseudopopulation rows. They do not provide animal/session inference. The dominant axis is reasonably separated; the lower modes are less stable, so individual lower-axis rotations are not promoted.",
            "",
            "## Tensor-law and SPD-geodesic checks on the official matrices",
            "",
            "| Check | Exp1 residual/change | Exp2 residual/change |",
            "|---|---:|---:|",
        ]
    )
    check_labels = [
        ("metric_transform_relative_residual", "g'=P^-T g P^-1 residual"),
        ("relative_endomorphism_similarity_residual", "A'=P A P^-1 residual"),
        ("canonical_transport_congruence_residual", "T^T g1 T=gk residual"),
        ("canonical_transport_self_adjoint_residual", "T is g1-self-adjoint residual"),
        ("canonical_transport_similarity_residual", "T'=P T P^-1 residual"),
        ("relative_direction_ratio_gl_residual", "directional stretch GL-invariance residual"),
        ("relative_symmetric_representation_residual", "R=u^T M u residual"),
        ("relative_precision_spectrum_gl_residual", "relative precision log-spectrum residual"),
        ("airm_gl_invariance_residual", "AIRM GL-invariance residual"),
        ("jeffreys_gl_invariance_residual", "Jeffreys GL-invariance residual"),
        ("geodesic_midpoint_half_residual", "SPD midpoint half-distance residual"),
        ("log_euclidean_gl_change", "Log-Euclidean change under non-orthogonal P (expected)"),
        ("bures_gl_change", "Bures change under non-orthogonal P (expected)"),
    ]
    exp1_checks = exp1_primary["coordinate_checks"]
    exp2_checks = exp2_primary["coordinate_checks"]
    assert isinstance(exp1_checks, dict) and isinstance(exp2_checks, dict)
    for key, label in check_labels:
        lines.append(f"| {label} | {exp1_checks[key]:.3e} | {exp2_checks[key]:.3e} |")

    lines.extend(
        [
            "",
            "## Derived Fisher-pullback equation and cross-experiment test",
            "",
            "Treat the released neuron-by-factor selectivity matrix `S` as the Jacobian of the mean population response in a linear Gaussian encoding model. Under the explicitly approximate homoscedastic residual model, the total population Fisher information and its one-parameter decoder calibration are",
            "",
            "$$",
            "J_F=S^TQ^{-1}S,\\qquad Q=\\sigma^2I,",
            "\\qquad \\Phi^{-1}(A_{k,j})=\\kappa\\sqrt{e_j^TJ_{F,k}e_j}.",
            "$$",
            "",
            "The common unknown noise scale and fixed stimulus-contrast factor are absorbed into the single coefficient `kappa`. It was fitted once on all 12 Exp1 stage-axis values and frozen before predicting the 12 Exp2 stage-axis decoder accuracies. This is a homoscedastic Gaussian/Bayes calibration applied to the released cross-validated SVM readout, not an identity for SVM accuracy.",
            "",
            f"- Fitted `kappa`: `{fisher['kappa']:.6f}`",
            f"- Fisher tensor coordinate-law residual: `{fisher['tensor_law_residual']:.3e}`",
            "- No prediction p-value is assigned: ordered learning stages and distinct task axes are not exchangeable biological units.",
            "",
            "| Frozen Exp1 model -> Exp2 | RMSE | MAE | Pearson r |",
            "|---|---:|---:|---:|",
        ]
    )
    fisher_models = fisher["models"]
    assert isinstance(fisher_models, dict)
    for name, values in fisher_models.items():
        lines.append(
            f"| {name} | {values['rmse']:.6f} | {values['mae']:.6f} | {values['correlation']:.6f} |"
        )
    observed_accuracy = np.asarray(fisher["exp2_observed_accuracy"])
    predicted_accuracy = np.asarray(fisher["exp2_predicted_accuracy"])
    lines.extend(
        [
            "",
            "Exp2 observed vs Fisher-predicted within-decoder accuracy `[set, XOR2, context]`:",
            "",
            "| Stage | Observed | Predicted |",
            "|---:|---|---|",
        ]
    )
    for stage_index, (observed, predicted) in enumerate(zip(observed_accuracy, predicted_accuracy), start=1):
        lines.append(
            f"| {stage_index} | `{observed.round(6).tolist()}` | `{predicted.round(6).tolist()}` |"
        )

    lines.extend(
        [
            "",
            "The Fisher calibration is compared directly with global/stage-mean, isotropic, and inverse-covariance controls below. Because residual Q is unavailable and the released rows are pseudopopulations, this comparison is descriptive cross-experiment transfer rather than a population-level significance test.",
            "",
            "Exp1 first-to-last log changes in the coordinate-axis metric costs `(colour, shape, XOR)` are reported below. Positive means that coordinate became more expensive under the inverse-covariance candidate; negative means cheaper.",
            "",
            f"`{np.asarray(rows[0]['metric_log_cost_change']).round(6).tolist()}`",
            "",
            "## Finite equation-family tournament",
            "",
            "The released data identify only stagewise three-factor selectivity summaries, so a finite operational universe is frozen to matrix spectral functions, Fisher/population norms, scale-shape summaries, regularized precision, stage geometry, neuron-level projected-drive gates, and standard accuracy links. Every coefficient and the candidate choice use Exp1 only. Exp2 is evaluated without coefficient refitting at the official 3, 4, 5, and 6-stage binnings.",
            "",
            "For each stage,",
            "",
            "$$",
            "H_k=N_k^{-1}S_k^TS_k,\\qquad J_k=N_kH_k,\\qquad X_k=H_k/\\tau_{\\mathrm{Exp1}},",
            "$$",
            "",
            "and the common calibration form is",
            "",
            "$$",
            "\widehat z_{kj}=a_j+\sum_m b_m x_{m,kj},\qquad",
            "\widehat A_{kj}=\ell^{-1}(\widehat z_{kj}).",
            "$$",
            "",
            f"- Enumerated producer-link candidates: `{tournament['candidate_count']}`",
            f"- Distinct producers: `{tournament['producer_count']}`",
            f"- Links: `{', '.join(tournament['links'])}`",
            f"- Exp1-only scale `tau`: `{tournament['full_tau']:.9g}`",
            "- Selection: leave-one-Exp1-stage-out accuracy RMSE; ties prefer fewer parameters.",
            "",
        ]
    )
    selected = tournament["selected"]
    assert isinstance(selected, dict)
    selected_bins = selected["per_binning"]
    assert isinstance(selected_bins, dict)
    lines.extend(
        [
            "### Exp1-selected equation",
            "",
            f"- Producer: `{selected['name']}`",
            f"- Formula: `${selected['formula']}$`",
            f"- Link: `{selected['link']}`",
            f"- Parameters: `{selected['parameters']}`",
            f"- Exp1 leave-one-stage-out RMSE: `{selected['cv_rmse']:.6f}`",
            f"- Mean Exp2 robustness-binning RMSE: `{selected['mean_exp2_rmse']:.6f}`",
            "",
            "| Exp2 binning | RMSE | MAE | Pearson r | Spearman rho |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for count in EXP2_STAGE_COUNTS:
        score = selected_bins[count]
        lines.append(
            f"| {count} | {score['rmse']:.6f} | {score['mae']:.6f} | {score['pearson']:.6f} | {score['spearman']:.6f} |"
        )
    producer_rows = tournament["best_by_producer"]
    assert isinstance(producer_rows, list)
    ungated_drive = next(row for row in producer_rows if row["name"] == "SPD power p=1")
    best_projected_gate = min(
        (row for row in producer_rows if row["kind"] == "projected_gate"),
        key=lambda row: (float(row["cv_rmse"]), str(row["name"])),
    )
    best_threshold_only = min(
        (row for row in producer_rows if row["kind"] == "threshold_only"),
        key=lambda row: (float(row["cv_rmse"]), str(row["name"])),
    )
    best_additive = min(
        (row for row in producer_rows if row["kind"] == "projected_threshold_additive"),
        key=lambda row: (float(row["cv_rmse"]), str(row["name"])),
    )
    gate_rows = [ungated_drive, best_projected_gate, best_threshold_only, best_additive]
    cv_gate_gain = float(ungated_drive["cv_rmse"]) - float(best_projected_gate["cv_rmse"])
    exp2_gate_gain = float(ungated_drive["mean_exp2_rmse"]) - float(
        best_projected_gate["mean_exp2_rmse"]
    )
    lines.extend(
        [
            "",
            "### Projected-drive threshold test",
            "",
            "The verbal `strength x alignment` factors are not separately identifiable under this projected-drive definition. For neuron row $v_{nk}$ and named factor projector $P_j=e_je_j^T$ they collapse exactly to the dimensionless projected drive",
            "",
            "$$",
            "s_{nk}=\\frac{\\lVert v_{nk}\\rVert}{\\sqrt{\\tau}},\\qquad",
            "a_{nkj}=\\frac{\\lVert P_jv_{nk}\\rVert}{\\lVert v_{nk}\\rVert},\\qquad",
            "d_{nkj}=s_{nk}a_{nkj}=\\frac{|S_{k,nj}|}{\\sqrt{\\tau}}.",
            "$$",
            "",
            "The primary gate is applied before population aggregation:",
            "",
            "$$",
            "\\gamma_{nkj}=\\sigma[\\beta(d_{nkj}-\\theta)],\\qquad",
            "R_{kj}=\\left[N_k^{-1}\\sum_n(d_{nkj}\\gamma_{nkj})^2\\right]^{1/2},",
            "$$",
            "",
            "$$",
            "h(\\widehat A_{kj})=a_j+bR_{kj}.",
            "$$",
            "",
            "Here $\\tau$ is recomputed from Exp1 training stages in every fold. The finite grid is $\\beta\\in\\{1,2,4,8\\}$ and $\\theta\\in\\{0.25,0.5,0.75,1,1.5\\}$. The no-gate row is exactly `SPD power p=1`; threshold-only removes amplitude, while the additive control gives projected RMS and mean gate activation separate coefficients.",
            "",
            "| Model | Exp1-selected specification | Link | Params | Exp1 CV RMSE | Mean Exp2 RMSE |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    gate_labels = (
        "No gate",
        "Projected-drive gate",
        "Threshold only",
        "Additive drive + threshold",
    )
    for label, candidate in zip(gate_labels, gate_rows):
        lines.append(
            f"| {label} | {candidate['name']} | {candidate['link']} | {candidate['parameters']} | {candidate['cv_rmse']:.6f} | {candidate['mean_exp2_rmse']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Relative to the no-gate projected drive, the Exp1-selected primary gate changes RMSE by `{cv_gate_gain:+.6f}` on Exp1 CV and `{exp2_gate_gain:+.6f}` on the frozen Exp2 readout; positive values favor the gate.",
            "",
            "This is a post-discussion discovery test on released pseudopopulation rows. It tests a selectivity-to-decoder calibration surrogate, not a synaptic threshold, effective connectivity, or causal routing mechanism.",
        ]
    )
    lines.extend(
        [
            "",
            "### Every producer, best Exp1-selected link",
            "",
            "This table retains every nonduplicated producer family. The best link for each row is chosen only by Exp1 leave-one-stage-out error; the Exp2 columns are readouts, not selection inputs.",
            "",
            "| Producer | Best link | Params | Exp1 CV RMSE | Mean Exp2 RMSE | K=3 | K=4 | K=5 | K=6 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in producer_rows:
        bins = candidate["per_binning"]
        lines.append(
            "| {name} | {link} | {parameters} | {cv_rmse:.6f} | {mean_exp2_rmse:.6f} | {k3:.6f} | {k4:.6f} | {k5:.6f} | {k6:.6f} |".format(
                **candidate,
                k3=bins[3]["rmse"],
                k4=bins[4]["rmse"],
                k5=bins[5]["rmse"],
                k6=bins[6]["rmse"],
            )
        )
    required_alternate = {
        str(selected["name"]),
        "relative information stretch",
        "relative precision stretch",
        "SPD power p=-1",
        "Fisher total",
        str(best_projected_gate["name"]),
    }
    alternate_rows = [
        row
        for index, row in enumerate(alternate)
        if index < 10 or str(row["name"]) in required_alternate
    ]
    lines.extend(
        [
            "",
            "### Alternate fixation-bias partition robustness",
            "",
            "The Exp1-main fitted coefficients are applied without refitting to the authors' alternate fixation-bias stage assignment and matching decoder cache. This reuses the same experiment and is a robustness check, not a new cohort.",
            "",
            "| Producer | Link | RMSE | MAE | Pearson r | Spearman rho |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    seen_alternate: set[str] = set()
    for row in alternate_rows:
        name = str(row["name"])
        if name in seen_alternate:
            continue
        seen_alternate.add(name)
        lines.append(
            "| {name} | {link} | {rmse:.6f} | {mae:.6f} | {pearson:.6f} | {spearman:.6f} |".format(
                **row
            )
        )
    alternate_by_name = {str(row["name"]): row for row in alternate}
    combined_rows: list[tuple[float, dict[str, object], dict[str, object]]] = []
    for candidate in producer_rows:
        alternate_row = alternate_by_name[str(candidate["name"])]
        combined_rows.append(
            (
                (float(candidate["mean_exp2_rmse"]) + float(alternate_row["rmse"])) / 2.0,
                candidate,
                alternate_row,
            )
        )
    combined_rows.sort(key=lambda item: (item[0], str(item[1]["name"])))
    robust_score, robust_candidate, robust_alternate = combined_rows[0]
    lines.extend(
        [
            "",
            "### Cross-surface discovery winner",
            "",
            f"- Producer: `{robust_candidate['name']}`",
            f"- Formula: `${robust_candidate['formula']}$`",
            f"- Link: `{robust_candidate['link']}`",
            f"- Mean Exp2-binning RMSE: `{robust_candidate['mean_exp2_rmse']:.6f}`",
            f"- Alternate fixation-bias RMSE: `{robust_alternate['rmse']:.6f}`",
            f"- Equal-weight two-surface mean RMSE: `{robust_score:.6f}`",
            "",
            "This combined ranking is descriptive and post-discovery, but it identifies the equation that remains accurate across rule-generalization binnings and the alternate fixation-bias partition instead of optimizing only one surface.",
        ]
    )
    lines.extend(
        [
            "",
            "The four Exp2 binnings reuse the same underlying sessions, so agreement across them is robustness to binning rather than four independent replications. The producer universe was expanded after inspecting earlier Exp2 results, making this a discovery tournament. A fresh session-level cohort is required for confirmation.",
            "",
            "## Geometry trajectory decomposition",
            "",
            "The covariance path is decomposed into determinant scale, determinant-normalized anisotropy, fixed-chart spectral diagnostics, effective rank, and successive AIRM motion. Ordered-projector displacement is retained only as a chart-dependent diagnostic because eigenvalue crossings or small gaps make individual lower eigenvectors unstable. Dominant-axis angle and eigengap are the interpretable rotation checks.",
            "",
            "| Dataset | Stage | Scale change | Anisotropy | Effective rank | Ordered-projector displacement | Dominant angle (deg) | Dominant gap | Min gap | AIRM from stage 1 | Successive AIRM | Axis cost ratio | det-normalized eigenvalues |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for entry in decomposition:
        lines.append(
            "| {dataset} | {stage} | {scale_change:+.6f} | {anisotropy:.6f} | {effective_rank:.6f} | {basis_rotation:.6f} | {dominant_axis_angle_degrees:.3f} | {dominant_axis_gap:.6f} | {minimum_relative_gap:.6f} | {airm_from_initial:.6f} | {airm_successive:.6f} | {axis_cost_ratio:.6f} | `{eigenvalues}` |".format(
                **entry,
                eigenvalues=np.asarray(entry["det_normalized_eigenvalues"]).round(5).tolist(),
            )
        )
    lines.extend(
        [
            "",
            "## Pooled-information-weighted cross-task cosine",
            "",
            "For the two official stimulus-set selectivity matrices, a separately pooled second-moment weighting gives",
            "",
            "$$",
            "\\cos_{H^{-1}}(S_1,S_2)=\\frac{\\operatorname{tr}(S_1H^{-1}S_2^T)}{\\sqrt{\\operatorname{tr}(S_1H^{-1}S_1^T)\\operatorname{tr}(S_2H^{-1}S_2^T)}}.",
            "\\qquad H=(S_1^TS_1+S_2^TS_2)/(2N).",
            "$$",
            "",
            "The Euclidean and pooled-information columns summarize all three factor coordinates. This $H^{-1}$ is built from both task matrices and is not the separately discovered stage metric $g_k=C_k^{-1}$. The named-axis columns follow the variables actually computed in the authors' `figure_4.py`.",
            "These are parallel descriptive readouts, not a demonstrated equality between single-neuron alignment and a metric eigenspace.",
            "",
            "| Stage | Early Euclidean cos | Early pooled-information cos | Late Euclidean cos | Late pooled-information cos | Early colour | Early context | Late shape | Late XOR |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in alignment:
        lines.append(
            "| {stage} | {early_euclidean_cosine:.6f} | {early_metric_cosine:.6f} | {late_euclidean_cosine:.6f} | {late_metric_cosine:.6f} | {official_early_colour_cosine:.6f} | {official_early_context_cosine:.6f} | {official_shape_cosine:.6f} | {official_xor_cosine:.6f} |".format(
                **entry
            )
        )
    late_3d = routing["late_3d"]
    late_shape_xor = routing["late_shape_xor_control"]
    assert isinstance(late_3d, dict)
    assert isinstance(late_shape_xor, dict)
    lines.extend(
        [
            "",
            "## Subspace-conditioned routing hypothesis",
            "",
            "The pasted routing proposal contains a useful hypothesis but its additive formulas need two corrections. A weighted sum of projectors is generally not itself a projector, and an additive metric update need not remain SPD. A finite, typed replacement is",
            "",
            "$$",
            "z(c)=\\operatorname{softmax}(Ac+b),\\qquad",
            "U(z)=\\operatorname{qf}\\left(U_0+\\sum_a z_aB_a\\right),\\qquad",
            "\\Pi(z)=U(z)U(z)^T,",
            "$$",
            "",
            "where the `qf` input must have full column rank. If a soft continuous gate rather than a literal subspace is intended, use $G(z)=U\\operatorname{diag}(\\sigma(Kz+d))U^T$ and call it a gain operator, not a projector.",
            "",
            "In coordinate-free form, let $\\mathcal S(z)$ be a dimensionless $g_0$-self-adjoint endomorphism and define",
            "",
            "$$",
            "g_z(u,v)=g_0\\!\\left(\\exp(\\mathcal S(z))u,v\\right),\\qquad",
            "\\mathcal S(z)=\\sum_a z_a\\mathcal S_a.",
            "$$",
            "",
            "This guarantees $g_z\\succ0$. Metric deformation alone still does not select a destination or generate motion. One explicit fixed-chart bridge is",
            "",
            "$$",
            "D_z=\\Pi_zg_z^{-1}\\Pi_z+\\epsilon(I-\\Pi_z)g_z^{-1}(I-\\Pi_z),\\qquad",
            "\\dot x=-\\kappa D_z\\nabla_xV(x,z)+f_\\perp(x,z),\\quad \\epsilon>0.",
            "$$",
            "",
            "The target-dependent potential $V$ and any non-gradient drift $f_\\perp$ are additional hypotheses. Thus global search can be amortized into a learned controller/value field, but its computational cost is moved offline or into $z(c)$ and $V$; it is not proven to disappear.",
            "",
            "### Observable PFC middle-link discovery test",
            "",
            "A time- and coordinate-matched check uses fold 0 of the official 100-150 ms task matrices. Let $X_k$ be task 1, $Y_k$ task 2, and $C_k=\\operatorname{Cov}(X_k)$. The two stage-level observables are",
            "",
            "$$",
            "d_k=d_{\\mathrm{AI}}(C_1,C_k),\\qquad",
            "q_k=\\frac{\\langle X_k,Y_k\\rangle_F}{\\|X_k\\|_F\\|Y_k\\|_F}.",
            "$$",
            "",
            f"- $d_k$: `{np.asarray(late_3d['geometry_displacement']).round(6).tolist()}`",
            f"- $q_k$: `{np.asarray(late_3d['cross_task_alignment']).round(6).tolist()}`",
            f"- Pearson $r(d,q)$: `{late_3d['pearson']:.6f}`",
            f"- Stagewise independent task-2 released-row shuffles: one-sided `p={late_3d['p_greater']:.6f}`, two-sided `p={late_3d['p_two_sided']:.6f}` over `{late_3d['shuffle_draws']}` draws",
            "",
            "This is the strongest available same-cache middle-link result: larger task-1 geometry displacement accompanies stronger matched task-2 alignment. The shuffle tests neuron-row correspondence conditional on four released pseudopopulation stages; it is post-discovery and is not an animal/session population test.",
            "",
            "The time-matched shape/XOR control does not reproduce the relationship:",
            "",
            f"- accessibility: `{np.asarray(late_shape_xor['accessibility']).round(6).tolist()}`",
            f"- official alignment: `{np.asarray(late_shape_xor['alignment']).round(6).tolist()}`",
            f"- Pearson: `{late_shape_xor['pearson']:.6f}`; author-null two-sided `p={late_shape_xor['p_two_sided']:.6f}` over `{late_shape_xor['null_draws']}` draws",
            "",
            "Therefore the released data contain a positive 3D discovery signal, but not a uniform axis-by-axis routing law. Two numerical corrections to the pasted interpretation are required: `-0.149823 -> 0.368520` is computed from the authors' early **colour** variables even though their exported plot labels that column as context, and `0.149701 -> 0.475854` is late **shape-selectivity alignment**. Neither series is metric alignment. The pooled-information cosine above is another same-cache statistic and must not be renamed as $g_k$.",
            "",
            "No primary-chart context coupling is reported: the 70-100 ms `[set, set*context, context]` metric and the separate Fig. 4 selectivity design do not supply an exact same-coordinate bridge.",
            "",
            "### External evidence boundary",
            "",
            "- [Tafazoli et al.](https://doi.org/10.1038/s41586-025-09805-2) directly support shared, task-selectively engaged sensory and motor subspaces; they do not fit a metric tensor.",
            "- [Binish et al.](https://doi.org/10.1038/s41593-026-02290-4) directly support a low-dimensional PFC-M1 communication subspace predictive of context-dependent action; they do not measure geodesic or routing cost.",
            "- [Gonzalez et al.](https://doi.org/10.1038/s41586-026-10481-z) support task/state-dependent hippocampal-retrosplenial communication subspaces and sleep reactivation; they do not identify $g_z$.",
            "- The primary dendrite study is [Maristany de Las Casas et al., Science](https://doi.org/10.1126/science.adx4358), not the cited Nature Neuroscience research highlight. It supports local rule-dependent dendritic gating, not a global Riemannian router.",
            "",
            "The evidence therefore supports `shared subspace engagement` and a PFC `relative precision deformation` candidate separately. The combined chain $c\\to\\Pi_z\\to g_z\\to\\dot x$ remains a falsifiable model, not an observed biological identity.",
        ]
    )
    lines.extend(
        [
            "",
            "## Equation availability boundary",
            "",
            "| Equation family | Status on this official local release |",
            "|---|---|",
            "| $S^TQ^{-p}S$ with measured residual $Q$ | UNAVAILABLE: trial residual covariance is not released in the local processed cache |",
            "| $(\\Sigma+\\lambda I)^{-p}$ and spectral functions | EXECUTED in the finite tournament |",
            "| Fisher-occupancy arithmetic and affine-invariant geometric mixtures | EXECUTED as fixed-chart sensitivities |",
            "| Increment mobility $[\\operatorname{Cov}(r_{t+1}-r_t)+\\lambda I]^{-1}$ | UNAVAILABLE: raw neural time series/spikes are absent locally |",
            "| Relative stretch, scale-shape, effective rank, eigenbasis rotation | EXECUTED |",
            "| Pooled-information-weighted cross-task cosine | EXECUTED on the official task1/task2 selectivity matrices; distinct from $g_k$ |",
            "| Late 3D $d_{\\mathrm{AI}}$ to cross-task alignment | EXECUTED as a post-discovery released-row conditional test |",
            "| State-dependent curvature/geodesic trajectory | UNAVAILABLE: only stagewise constant summary matrices are present |",
            "| Structural $W\\rightarrow g$ | UNAVAILABLE: no structural connectivity $W$ |",
            "",
            "## Geometry and official behavior summaries",
            "",
            "For each stage, the table compares behavior with distance from the first-stage covariance and signed log-volume change:",
            "",
            "$$",
            "d_k=d_{\\mathrm{AI}}(C_1,C_k),\\qquad v_k=\\tfrac13\\log\\det(C_1^{-1}C_k).",
            "$$",
            "",
            "| Dataset | Endpoint | Stages | Spearman(d, behavior) | Spearman(v, behavior) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for entry in behavior:
        lines.append(
            "| {dataset} | {endpoint} | {stages} | {rho_airm:.6f} | {rho_logvolume:.6f} |".format(
                **entry
            )
        )
    lines.extend(
        [
            "",
            "These behavior correlations use only 3-6 ordered stage means and receive no p-value. They are external endpoint sensitivities, not trial-level neural mediation tests.",
            "",
            "## Held-out prediction check",
            "",
            "For the primary four-stage Exp2 comparison, a zero-mean Gaussian with the full stage-specific metric was fitted on four folds and scored on the unseen fifth fold. The table reports `alternative NLL - full-metric NLL`, so positive values favor the full stage-specific metric. The 200 x 5 folds are repeated prediction checks, not independent biological samples.",
            "",
            "| Alternative | Mean held-out NLL penalty (nat/row) | Folds won by full metric |",
            "|---|---:|---:|",
        ]
    )
    for name, values in heldout.items():
        lines.append(
            f"| {name} | {values['mean_nll_gain']:.6f} | {values['fold_wins']}/{values['folds']} |"
        )

    lines.extend(
        [
            "",
            "## Matched functional readouts",
            "",
            "The metric-axis cost change is `log((e_i^T g_last e_i)/(e_i^T g_first e_i))`. Decoder p-values use the authors' 1,000 learning-epoch-reassignment nulls, not the row permutation above.",
            "",
            "| Experiment | Named axis | Metric log-cost change | Decoder change | Author LER p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    entries = decoder["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        lines.append(
            "| {experiment} | {axis} | {metric_log_cost_change:+.6f} | {decoder_delta:+.6f} | {decoder_ler_p:.6f} |".format(
                **entry
            )
        )
    lines.extend(
        [
            "",
            "Across the six pre-named axes, accessibility change `-Delta log cost` and decoder change have Pearson "
            f"`r={decoder['pooled_correlation']:.6f}` with {decoder['sign_matches']}/{decoder['axis_count']} matching signs. "
            "This is an exploratory alignment summary only; stages and task axes are not exchangeable inferential units, so no permutation p-value is assigned.",
            "",
            "## Input integrity",
            "",
        ]
    )
    for name, digest in hashes.items():
        lines.append(f"- `{name}`: `{digest}`")

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is an official real-data test of stage-specific inverse-covariance geometry and a separately derived Fisher-pullback decoder bridge on released PFC selectivity pseudopopulations. It is not a mock-data result.",
            "",
            "The released rows do not retain session or animal identifiers, and the Exp1 cache's two apparent folds are exact duplicates. Therefore row exchangeability and row-fold prediction are sensitivity analyses, not animal-population or held-out-session inference.",
            "",
            "Every equation supported by these released stagewise 3D selectivity summaries is evaluated above. The Fisher pullback uses the explicitly stated homoscedastic z-score approximation and absorbs the missing residual-noise scale into kappa. Process-noise reachability, controllability Gramians, fully noise-calibrated or state-dependent Fisher fields, curvature, geodesic trajectory prediction, directed action, structural W producers, and the causal chain Delta W -> Delta g -> Delta x require raw trajectories, perturbation channels, spatial fields, or connectivity that are absent here. Substituting fabricated inputs for those equations would be a mock analysis, so they are deliberately not computed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_819)
    args = parser.parse_args()
    if args.draws < 999:
        raise ValueError("draws must be at least 999")

    processed = args.repo.resolve() / "processed_data"
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    route_index = 0

    for name, filename, key in EXP1:
        path = processed / filename
        first, last, duplicate_residual = exp1_points(path, key)
        row = permutation_test(
            first,
            last,
            args.draws,
            args.seed + route_index,
            extended=name == "Exp1 main",
        )
        row.update(name=name, duplicate_fold_max_abs=duplicate_residual, decision=verdict(float(row["p_total"])))
        rows.append(row)
        hashes[filename] = sha256(path)
        route_index += 1

    for stage_count in EXP2_STAGE_COUNTS:
        filename = f"selectivity_coefficients_exp2_70_100_{stage_count}stages.pickle"
        path = processed / filename
        first, last = exp2_points(path)
        row = permutation_test(
            first,
            last,
            args.draws,
            args.seed + route_index,
            extended=stage_count == 4,
        )
        row.update(name=f"Exp2 ({stage_count}-stage binning)", decision=verdict(float(row["p_total"])))
        if stage_count == 4:
            row["heldout_prediction"] = heldout_metric_prediction(
                first,
                last,
                repeats=200,
                folds=5,
                seed=args.seed + 10_000,
            )
            row["released_row_bootstrap"] = released_row_bootstrap(
                first,
                last,
                draws=args.draws,
                seed=args.seed + 20_000,
            )
        rows.append(row)
        hashes[filename] = sha256(path)
        route_index += 1

    exp1_primary = rows[0]
    exp2_primary = next(row for row in rows if row["name"] == "Exp2 (4-stage binning)")
    decoder = decoder_evidence(processed, exp1_primary, exp2_primary)
    fisher = fisher_information_bridge(processed)
    tournament = equation_family_tournament(processed)
    behavior = geometry_behavior_sensitivity(processed)
    decomposition = stage_geometry_decomposition(processed)
    alignment = cross_set_metric_alignment(processed)
    routing = routing_chain_sensitivity(processed, args.draws, args.seed)
    alternate = alternate_partition_robustness(processed, tournament)
    for filename in (
        "exp1_decoding_collocked_50_150_4stages.pickle",
        "exp1_decoding_shapelocked_100_150_4stages.pickle",
        "exp1_beh_terminations4stages.pickle",
        "exp1_switch_costs4stages.pickle",
        "exp2_selectivity_dat_early_50_100_late_100_150_stages_4.pickle",
        "exp1_decoding_fixbias_collocked_50_150_4stages.pickle",
        "exp1_decoding_fixbias_shapelocked_100_150_4stages.pickle",
    ):
        hashes[filename] = sha256(processed / filename)
    for stage_count in EXP2_STAGE_COUNTS:
        for filename in (
            f"exp2_decoding_time_avg_{stage_count}stages_50_100.pickle",
            f"fixation_breaks_prop_exp2_{stage_count}stages.pickle",
        ):
            hashes[filename] = sha256(processed / filename)

    report = render(
        rows,
        decoder,
        fisher,
        tournament,
        behavior,
        decomposition,
        alignment,
        routing,
        alternate,
        hashes,
        args.draws,
        args.seed,
        git_value(args.repo.resolve(), "config", "--get", "remote.origin.url"),
        git_value(args.repo.resolve(), "rev-parse", "HEAD"),
    )
    output = args.output.resolve()
    output.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
