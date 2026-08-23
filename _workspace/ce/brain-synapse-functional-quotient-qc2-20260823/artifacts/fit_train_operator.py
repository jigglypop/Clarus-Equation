"""Fit the frozen BA-SRM3 train-only response operator and rank gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import scipy.linalg


VERSION = "BA-SRM3-TRAIN-OPERATOR-V1"
EXPECTED_DATASET_SHA256 = (
    "06201f5be8cb87b244af4d780b912c0676edf8224bc4c4feb467b3ac59f63b48"
)
EXPECTED_DATASET_RECEIPT_SHA256 = (
    "7264d46a6a8b7d2096ffe0b9f876267df372097589d8e2d91a5bd56362311ff0"
)
EXPECTED_MODEL_MODULE_SHA256 = (
    "9b0da87311a1b1d938e7b6a79d726e01bcec8b57c094aae0f7dd46458f9d018e"
)

OUTER_FOLD_SALT = "BA-SRM3-OUTER-FOLD-V1:"
INNER_R_FOLD_SALT_PREFIX = "BA-SRM3-INNER-R-V1:"
BOOTSTRAP_SALT = "BA-SRM3-RANK-BOOTSTRAP-V1:"
DIMENSIONS = (2, 4, 8, 16, 32)
ELLS = (0.5, 1.0, 2.0, 4.0)
RIDGES = (1e-6, 1e-4, 1e-2, 1.0)
GAMMAS = (0.25, 0.5, 0.75, 1.0)
FOLDS = 5
INNER_FOLDS = 4
EXPECTED_DATASET_KEYS = {
    "numeric",
    "categorical",
    "target",
    "sequence_key",
    "slice_ext_id",
    "synapse_type",
    "numeric_feature_names",
    "categorical_feature_names",
    "target_names",
}


class FitFailure(RuntimeError):
    """Raised when the frozen train operator cannot be fit safely."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model_module():
    path = Path(__file__).with_name("srm3_model.py")
    observed = sha256_file(path)
    if observed != EXPECTED_MODEL_MODULE_SHA256:
        raise RuntimeError("frozen srm3_model.py SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("ba_srm3_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load srm3_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODEL = _load_model_module()


def fold_assignments(groups: np.ndarray, salt: str, folds: int) -> np.ndarray:
    return np.asarray(
        [MODEL.deterministic_fold(str(group), salt, folds) for group in groups],
        dtype=int,
    )


def validate_group_folds(groups: np.ndarray, folds: np.ndarray, count: int) -> None:
    if set(folds.tolist()) != set(range(count)):
        raise FitFailure("one or more group folds are empty")
    observed: dict[str, int] = {}
    for group, fold in zip(groups, folds):
        key = str(group)
        if key in observed and observed[key] != int(fold):
            raise FitFailure("slice group crosses folds")
        observed[key] = int(fold)


def require_dimension(coordinates: np.ndarray, dimension: int, context: str) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] < int(dimension):
        raise FitFailure(
            f"{context} PCA provides {coordinates.shape[1] if coordinates.ndim == 2 else 0} "
            f"directions, below selected d={dimension}"
        )


def validate_dataset(data: dict[str, np.ndarray]) -> None:
    if set(data) != EXPECTED_DATASET_KEYS:
        missing = sorted(EXPECTED_DATASET_KEYS - set(data))
        extra = sorted(set(data) - EXPECTED_DATASET_KEYS)
        raise FitFailure(f"dataset key mismatch; missing={missing}, extra={extra}")
    rows = int(data["target"].shape[0]) if data["target"].ndim == 2 else -1
    expected_shapes = {
        "numeric": (rows, 98),
        "categorical": (rows, 5),
        "target": (rows, 16),
        "sequence_key": (rows,),
        "slice_ext_id": (rows,),
        "synapse_type": (rows,),
        "numeric_feature_names": (98,),
        "categorical_feature_names": (5,),
        "target_names": (16,),
    }
    for key, shape in expected_shapes.items():
        if data[key].shape != shape:
            raise FitFailure(f"dataset shape mismatch for {key}: {data[key].shape} != {shape}")
        if data[key].dtype == object:
            raise FitFailure(f"dataset contains object/pickle array: {key}")
    if rows < 1 or not np.all(np.isfinite(data["target"])):
        raise FitFailure("target is not a nonempty raw finite dimensionless matrix")
    labels = set(data["synapse_type"].astype(str).tolist())
    if labels != {"ex", "in"}:
        raise FitFailure(f"unexpected synapse strata: {sorted(labels)}")
    sequence_keys = data["sequence_key"].astype(str)
    groups = data["slice_ext_id"].astype(str)
    if np.any(sequence_keys == "") or np.unique(sequence_keys).size != rows:
        raise FitFailure("sequence keys are blank or nonunique")
    if np.any(groups == ""):
        raise FitFailure("slice group contains blank identifier")


def _kernel_eigendecomposition(coordinates: np.ndarray, ell: float):
    kernel = MODEL.rbf_kernel(coordinates, coordinates, ell)
    values, vectors = scipy.linalg.eigh(
        kernel, check_finite=False, driver="evr"
    )
    return values, vectors


def _ridge_predictions(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    train_target_std: np.ndarray,
    test_kernel: np.ndarray,
    ridge: float,
) -> np.ndarray:
    projected = eigenvectors.T @ train_target_std
    alpha = eigenvectors @ (projected / (eigenvalues[:, None] + ridge))
    return test_kernel @ alpha


def select_theta_cv(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    fold_salt: str = OUTER_FOLD_SALT,
    folds: int = FOLDS,
    dimensions: tuple[int, ...] = DIMENSIONS,
    ells: tuple[float, ...] = ELLS,
    ridges: tuple[float, ...] = RIDGES,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    assignments = fold_assignments(groups, fold_salt, folds)
    validate_group_folds(groups, assignments, folds)
    accum: dict[tuple[int, float, float], list[float]] = {}
    availability: list[set[int]] = []
    fold_receipts: list[dict[str, Any]] = []

    for fold in range(folds):
        train = assignments != fold
        test = assignments == fold
        params, train_max = MODEL.fit_preprocessor(
            numeric[train], categorical[train], max(dimensions)
        )
        available = {d for d in dimensions if d <= train_max.shape[1]}
        availability.append(available)
        output_scaler = MODEL.fit_output_scaler(target[train])
        y_train_std = MODEL.standardize_target(target[train], output_scaler)
        y_test_std = MODEL.standardize_target(target[test], output_scaler)
        fold_receipts.append(
            {
                "fold": fold,
                "train_rows": int(np.sum(train)),
                "test_rows": int(np.sum(test)),
                "train_slice_groups": int(np.unique(groups[train]).size),
                "test_slice_groups": int(np.unique(groups[test]).size),
                "available_dimensions": sorted(available),
                "pca_numeric_rank": int(train_max.shape[1]),
            }
        )
        for dimension in sorted(available):
            train_coordinates = train_max[:, :dimension]
            test_coordinates = MODEL.transform_preprocessor(
                params, numeric[test], categorical[test], dimension
            )
            for ell in ells:
                eigenvalues, eigenvectors = _kernel_eigendecomposition(
                    train_coordinates, ell
                )
                test_kernel = MODEL.rbf_kernel(
                    test_coordinates, train_coordinates, ell
                )
                for ridge in ridges:
                    prediction = _ridge_predictions(
                        eigenvalues,
                        eigenvectors,
                        y_train_std,
                        test_kernel,
                        ridge,
                    )
                    squared_error = float(np.sum((prediction - y_test_std) ** 2))
                    key = (dimension, float(ell), float(ridge))
                    values = accum.setdefault(key, [0.0, 0.0, 0.0])
                    values[0] += squared_error
                    values[1] += float(prediction.size)
                    values[2] += 1.0

    common_dimensions = set.intersection(*availability)
    if not any(d >= 2 for d in common_dimensions):
        raise FitFailure("no d>=2 dimension is available in every fold")
    table: list[dict[str, Any]] = []
    for (dimension, ell, ridge), (sse, elements, seen_folds) in accum.items():
        if dimension not in common_dimensions or int(seen_folds) != folds:
            continue
        table.append(
            {
                "dimension": dimension,
                "ell": ell,
                "ridge": ridge,
                "standardized_mse": sse / elements,
                "squared_error": sse,
                "elements": int(elements),
            }
        )
    if not table:
        raise FitFailure("no complete CV candidate")
    best_loss = min(row["standardized_mse"] for row in table)
    tied = [row for row in table if row["standardized_mse"] <= best_loss + 1e-12]
    selected = sorted(
        tied,
        key=lambda row: (
            row["dimension"],
            -row["ridge"],
            -row["ell"],
        ),
    )[0]
    receipt = {
        "fold_salt": fold_salt,
        "folds": folds,
        "common_available_dimensions": sorted(common_dimensions),
        "excluded_dimensions": sorted(set(dimensions) - common_dimensions),
        "fold_receipts": fold_receipts,
    }
    return selected, sorted(
        table,
        key=lambda row: (row["dimension"], row["ell"], row["ridge"]),
    ), receipt


def crossfit_fixed_theta(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    theta: dict[str, Any],
    fold_salt: str = OUTER_FOLD_SALT,
    folds: int = FOLDS,
) -> np.ndarray:
    assignments = fold_assignments(groups, fold_salt, folds)
    validate_group_folds(groups, assignments, folds)
    predictions = np.full_like(target, np.nan, dtype=float)
    dimension = int(theta["dimension"])
    for fold in range(folds):
        train = assignments != fold
        test = assignments == fold
        params, _ = MODEL.fit_preprocessor(
            numeric[train], categorical[train], max(DIMENSIONS)
        )
        train_coordinates = MODEL.transform_preprocessor(
            params, numeric[train], categorical[train], dimension
        )
        test_coordinates = MODEL.transform_preprocessor(
            params, numeric[test], categorical[test], dimension
        )
        output_scaler = MODEL.fit_output_scaler(target[train])
        train_std = MODEL.standardize_target(target[train], output_scaler)
        fitted = MODEL.fit_krr(
            train_coordinates,
            train_std,
            float(theta["ell"]),
            float(theta["ridge"]),
            output_scaler,
        )
        predictions[test] = MODEL.predict_krr(fitted, test_coordinates)
    if not np.all(np.isfinite(predictions)):
        raise FitFailure("crossfit produced nonfinite/unfilled predictions")
    return predictions


def fit_partition_predict(
    train_numeric: np.ndarray,
    train_categorical: np.ndarray,
    train_target: np.ndarray,
    test_numeric: np.ndarray,
    test_categorical: np.ndarray,
    theta: dict[str, Any],
) -> np.ndarray:
    """Fit exactly on one partition and predict a disjoint partition."""
    params, train_coordinates_max = MODEL.fit_preprocessor(
        train_numeric, train_categorical, max(DIMENSIONS)
    )
    dimension = int(theta["dimension"])
    require_dimension(train_coordinates_max, dimension, "partition fit")
    train_coordinates = train_coordinates_max[:, :dimension]
    test_coordinates = MODEL.transform_preprocessor(
        params, test_numeric, test_categorical, dimension
    )
    output_scaler = MODEL.fit_output_scaler(train_target)
    fitted = MODEL.fit_krr(
        train_coordinates,
        MODEL.standardize_target(train_target, output_scaler),
        float(theta["ell"]),
        float(theta["ridge"]),
        output_scaler,
    )
    return MODEL.predict_krr(fitted, test_coordinates)


def nested_outer_fit(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    dimensions: tuple[int, ...] = DIMENSIONS,
    ells: tuple[float, ...] = ELLS,
    ridges: tuple[float, ...] = RIDGES,
    outer_folds: int = FOLDS,
    inner_folds: int = INNER_FOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run the preregistered outer/inner train-only nesting.

    For outer fold f, neither its targets nor target-derived quantities enter
    theta, gamma, R, preprocessing, or M fitted on -f.
    """
    outer_assignment = fold_assignments(groups, OUTER_FOLD_SALT, outer_folds)
    validate_group_folds(groups, outer_assignment, outer_folds)
    outer_prediction = np.full_like(target, np.nan, dtype=float)
    covariance_raw = np.empty(
        (outer_folds, target.shape[1], target.shape[1]), dtype=float
    )
    covariance_effective = np.empty_like(covariance_raw)
    receipts: list[dict[str, Any]] = []

    for outer_fold in range(outer_folds):
        fit_mask = outer_assignment != outer_fold
        test_mask = outer_assignment == outer_fold
        inner_salt = f"{INNER_R_FOLD_SALT_PREFIX}{outer_fold}:"
        theta, theta_table, theta_receipt = select_theta_cv(
            numeric[fit_mask],
            categorical[fit_mask],
            target[fit_mask],
            groups[fit_mask],
            fold_salt=inner_salt,
            folds=inner_folds,
            dimensions=dimensions,
            ells=ells,
            ridges=ridges,
        )
        inner_prediction = crossfit_fixed_theta(
            numeric[fit_mask],
            categorical[fit_mask],
            target[fit_mask],
            groups[fit_mask],
            theta,
            fold_salt=inner_salt,
            folds=inner_folds,
        )
        inner_residual = target[fit_mask] - inner_prediction
        gamma, gamma_table = select_gamma(inner_residual)
        sample = MODEL.residual_covariance(inner_residual)
        raw = MODEL.shrink_covariance(sample, gamma["gamma"])
        effective, floor = MODEL.floored_covariance(raw)
        covariance_raw[outer_fold] = raw
        covariance_effective[outer_fold] = effective
        outer_prediction[test_mask] = fit_partition_predict(
            numeric[fit_mask],
            categorical[fit_mask],
            target[fit_mask],
            numeric[test_mask],
            categorical[test_mask],
            theta,
        )
        receipts.append(
            {
                "outer_fold": outer_fold,
                "outer_train_rows": int(np.sum(fit_mask)),
                "outer_test_rows": int(np.sum(test_mask)),
                "outer_train_slice_groups": int(np.unique(groups[fit_mask]).size),
                "outer_test_slice_groups": int(np.unique(groups[test_mask]).size),
                "inner_fold_salt": inner_salt,
                "inner_folds": inner_folds,
                "selected_theta": {
                    "dimension": int(theta["dimension"]),
                    "ell": float(theta["ell"]),
                    "ridge": float(theta["ridge"]),
                    "standardized_mse": float(theta["standardized_mse"]),
                },
                "theta_cv": theta_receipt,
                "theta_candidates": theta_table,
                "selected_gamma": float(gamma["gamma"]),
                "gamma_candidates": gamma_table,
                "covariance_floor": float(floor),
                "outer_target_used_in_fit": False,
            }
        )

    if not np.all(np.isfinite(outer_prediction)):
        raise FitFailure("nested outer fit produced nonfinite/unfilled predictions")
    return outer_prediction, covariance_raw, covariance_effective, receipts


def select_gamma(residuals: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = MODEL.residual_covariance(residuals)
    table = []
    for gamma in GAMMAS:
        raw = MODEL.shrink_covariance(sample, gamma)
        effective, floor = MODEL.floored_covariance(raw)
        score = MODEL.gaussian_log_score(
            residuals, np.zeros_like(residuals), effective
        )
        eigenvalues = np.linalg.eigvalsh(raw)
        table.append(
            {
                "gamma": gamma,
                "mean_training_oof_log_score": float(np.mean(score)),
                "floor": floor,
                "raw_min_eigenvalue": float(eigenvalues[0]),
                "raw_max_eigenvalue": float(eigenvalues[-1]),
                "raw_condition_number": (
                    float(eigenvalues[-1] / eigenvalues[0])
                    if eigenvalues[0] > 0.0
                    else None
                ),
            }
        )
    best_score = max(row["mean_training_oof_log_score"] for row in table)
    tied = [
        row
        for row in table
        if row["mean_training_oof_log_score"] >= best_score - 1e-12
    ]
    return sorted(tied, key=lambda row: -row["gamma"])[0], table


def rank_bootstrap(
    fifth_ratios: np.ndarray,
    groups: np.ndarray,
    synapse_type: str,
    replicates: int = 1000,
) -> dict[str, Any]:
    unique = np.unique(groups)
    seed = int.from_bytes(
        hashlib.sha256(f"{BOOTSTRAP_SALT}{synapse_type}".encode("utf-8")).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    medians = np.empty(replicates, dtype=float)
    by_group = {group: fifth_ratios[groups == group] for group in unique}
    for index in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        values = np.concatenate([by_group[group] for group in sampled])
        medians[index] = np.median(values)
    quantiles = np.quantile(medians, [0.025, 0.5, 0.975])
    return {
        "salt": BOOTSTRAP_SALT,
        "seed_uint64": seed,
        "replicates": replicates,
        "statistic": "slice-resampled median sigma5_over_sigma1; descriptive only",
        "q025": float(quantiles[0]),
        "q500": float(quantiles[1]),
        "q975": float(quantiles[2]),
    }


def serialize_preprocessor(prefix: str, params: Any, arrays: dict[str, np.ndarray]):
    arrays[f"{prefix}_numeric_median"] = params.numeric_median
    arrays[f"{prefix}_numeric_scale"] = params.numeric_scale
    arrays[f"{prefix}_keep_columns"] = params.keep_columns
    arrays[f"{prefix}_composite_center"] = params.composite_center
    arrays[f"{prefix}_pca_components"] = params.pca_components
    arrays[f"{prefix}_pca_eigenvalues"] = params.pca_eigenvalues


def fit_stratum(
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    sequence_keys: np.ndarray,
    synapse_type: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    (
        nested_outer_prediction,
        nested_outer_covariance_raw,
        nested_outer_covariance_effective,
        nested_outer_receipts,
    ) = nested_outer_fit(numeric, categorical, target, groups)

    # The final operator hyperparameters are selected on the full train cohort.
    # These OOF scores are tuning quantities, never reported as held-out evidence.
    selected, cv_table, cv_receipt = select_theta_cv(
        numeric, categorical, target, groups
    )
    oof_prediction = crossfit_fixed_theta(
        numeric, categorical, target, groups, selected
    )
    # Contract requires raw dimensionless Y residuals here, never standardized Y.
    residuals = target - oof_prediction
    gamma_selected, gamma_table = select_gamma(residuals)
    sample = MODEL.residual_covariance(residuals)
    covariance_raw = MODEL.shrink_covariance(sample, gamma_selected["gamma"])
    covariance_effective, covariance_floor = MODEL.floored_covariance(covariance_raw)

    params, full_coordinates_max = MODEL.fit_preprocessor(
        numeric, categorical, max(DIMENSIONS)
    )
    dimension = int(selected["dimension"])
    require_dimension(full_coordinates_max, dimension, "full-train fit")
    full_coordinates = full_coordinates_max[:, :dimension]
    output_scaler = MODEL.fit_output_scaler(target)
    target_std = MODEL.standardize_target(target, output_scaler)
    fitted = MODEL.fit_krr(
        full_coordinates,
        target_std,
        selected["ell"],
        selected["ridge"],
        output_scaler,
    )
    jacobians = MODEL.jacobian_krr(fitted, full_coordinates)
    ranks, fifth_ratios = MODEL.numerical_ranks(jacobians)
    rank_min = int(np.min(ranks))
    rank_max = int(np.max(ranks))
    constant_high_dimensional = rank_min == rank_max and rank_min >= 5
    bootstrap = rank_bootstrap(fifth_ratios, groups, synapse_type)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance_raw)

    arrays: dict[str, np.ndarray] = {}
    serialize_preprocessor(synapse_type, params, arrays)
    arrays[f"{synapse_type}_train_coordinates"] = full_coordinates
    arrays[f"{synapse_type}_krr_alpha"] = fitted.alpha
    arrays[f"{synapse_type}_output_median"] = fitted.output_median
    arrays[f"{synapse_type}_output_mad"] = fitted.output_mad
    arrays[f"{synapse_type}_covariance_raw"] = covariance_raw
    arrays[f"{synapse_type}_covariance_effective"] = covariance_effective
    arrays[f"{synapse_type}_train_target"] = target
    arrays[f"{synapse_type}_oof_prediction"] = oof_prediction
    arrays[f"{synapse_type}_nested_outer_prediction"] = nested_outer_prediction
    arrays[f"{synapse_type}_nested_outer_fold"] = fold_assignments(
        groups, OUTER_FOLD_SALT, FOLDS
    )
    arrays[f"{synapse_type}_nested_outer_covariance_raw"] = (
        nested_outer_covariance_raw
    )
    arrays[f"{synapse_type}_nested_outer_covariance_effective"] = (
        nested_outer_covariance_effective
    )
    arrays[f"{synapse_type}_rank"] = ranks
    arrays[f"{synapse_type}_sigma5_ratio"] = fifth_ratios
    arrays[f"{synapse_type}_slice_ext_id"] = groups.astype(str)
    arrays[f"{synapse_type}_sequence_key"] = sequence_keys.astype(str)

    metadata = {
        "category_levels": [list(levels) for levels in params.category_levels],
        "selected_theta": {
            "dimension": dimension,
            "ell": float(selected["ell"]),
            "ridge": float(selected["ridge"]),
            "standardized_mse": float(selected["standardized_mse"]),
        },
        "selected_gamma": float(gamma_selected["gamma"]),
        "covariance_floor": covariance_floor,
    }
    result = {
        "rows": int(target.shape[0]),
        "slice_groups": int(np.unique(groups).size),
        "cv": cv_receipt,
        "cv_candidates": cv_table,
        "nested_outer": {
            "purpose": "train-only tuning/controls; not held-out performance evidence",
            "outer_fold_salt": OUTER_FOLD_SALT,
            "inner_fold_salt_prefix": INNER_R_FOLD_SALT_PREFIX,
            "outer_folds": FOLDS,
            "inner_folds": INNER_FOLDS,
            "folds": nested_outer_receipts,
        },
        "selected_theta": metadata["selected_theta"],
        "gamma_candidates": gamma_table,
        "selected_gamma": float(gamma_selected["gamma"]),
        "covariance": {
            "floor": covariance_floor,
            "raw_eigenvalues": covariance_eigenvalues.tolist(),
            "raw_condition_number": (
                float(covariance_eigenvalues[-1] / covariance_eigenvalues[0])
                if covariance_eigenvalues[0] > 0.0
                else None
            ),
            "residual_source": "raw dimensionless Y minus raw dimensionless OOF prediction",
        },
        "rank": {
            "relative_threshold": MODEL.RANK_RELATIVE_TOL,
            "minimum": rank_min,
            "maximum": rank_max,
            "counts": {
                str(value): int(np.sum(ranks == value))
                for value in sorted(np.unique(ranks))
            },
            "constant_rank_at_train_anchors": rank_min == rank_max,
            "constant_rank_at_least_5": constant_high_dimensional,
            "sigma5_ratio_min": float(np.min(fifth_ratios)),
            "sigma5_ratio_median": float(np.median(fifth_ratios)),
            "bootstrap": bootstrap,
        },
        "operator_train_gate_pass": constant_high_dimensional,
    }
    return result, arrays, metadata


def verify_model_npz(path: Path, expected_keys: set[str]) -> None:
    with np.load(path, allow_pickle=False) as observed:
        if set(observed.files) != expected_keys:
            raise FitFailure("model NPZ key mismatch after write")
        for key in observed.files:
            if observed[key].dtype == object:
                raise FitFailure("model NPZ contains object/pickle array")


def write_text_partial(final_path: Path, text: str) -> Path:
    partial = final_path.with_name(final_path.name + ".partial")
    if partial.exists():
        raise FitFailure(f"stale partial text artifact exists: {partial.name}")
    partial.write_text(text, encoding="utf-8")
    if partial.read_text(encoding="utf-8") != text:
        raise FitFailure(f"partial text verification failed: {partial.name}")
    return partial


def operator_stage_gate(rank_pass: bool) -> dict[str, Any]:
    """Authorize only the next train-only stage; never authorize development."""
    return {
        "status": (
            "PASS_TRAIN_OPERATOR_RANK_GATE"
            if rank_pass
            else "STOP_TRAIN_OPERATOR_RANK"
        ),
        "train_geometry_controls_unlock": bool(rank_pass),
        "development_unlock": False,
        "development_unlock_requires": (
            "separate hash-pinned train-only geometry/rho/gauge/control receipt"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.model.exists() or args.metadata.exists() or args.output.exists():
            raise FitFailure("refusing to overwrite frozen model artifacts")
        if sha256_file(args.dataset) != EXPECTED_DATASET_SHA256:
            raise FitFailure("train dataset SHA-256 mismatch")
        if sha256_file(args.dataset_receipt) != EXPECTED_DATASET_RECEIPT_SHA256:
            raise FitFailure("train dataset receipt SHA-256 mismatch")
        dataset_receipt = json.loads(args.dataset_receipt.read_text(encoding="utf-8"))
        if dataset_receipt.get("status") != "PASS_TRAIN_DATASET":
            raise FitFailure("dataset receipt did not pass")
        if dataset_receipt.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
            raise FitFailure("dataset receipt hash mismatch")

        with np.load(args.dataset, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        validate_dataset(data)
        all_arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {"version": VERSION, "strata": {}}
        result: dict[str, Any] = {}
        for label in ("ex", "in"):
            mask = data["synapse_type"] == label
            stratum_result, arrays, stratum_metadata = fit_stratum(
                data["numeric"][mask],
                data["categorical"][mask],
                data["target"][mask],
                data["slice_ext_id"][mask],
                data["sequence_key"][mask],
                label,
            )
            result[label] = stratum_result
            all_arrays.update(arrays)
            metadata["strata"][label] = stratum_metadata

        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial_model = args.model.with_name(args.model.name + ".partial.npz")
        if partial_model.exists():
            raise FitFailure("stale partial model artifact exists")
        np.savez_compressed(partial_model, **all_arrays)
        verify_model_npz(partial_model, set(all_arrays))
        model_hash = sha256_file(partial_model)
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        partial_metadata = write_text_partial(args.metadata, metadata_text)
        metadata_hash = sha256_file(partial_metadata)
        both_pass = all(result[label]["operator_train_gate_pass"] for label in ("ex", "in"))
        stage_gate = operator_stage_gate(both_pass)
        receipt = {
            **stage_gate,
            "version": VERSION,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "dataset_receipt_sha256": EXPECTED_DATASET_RECEIPT_SHA256,
            "model_module_sha256": EXPECTED_MODEL_MODULE_SHA256,
            "model": str(args.model.resolve(strict=False)),
            "model_sha256": model_hash,
            "metadata": str(args.metadata.resolve(strict=False)),
            "metadata_sha256": metadata_hash,
            "fold_salt": OUTER_FOLD_SALT,
            "bootstrap_salt": BOOTSTRAP_SALT,
            "strata": result,
            "train_outcomes_read": True,
            "development_outcomes_read": False,
            "confirmation_outcomes_read": False,
            "waveform_blobs_read": False,
        }
        receipt_text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        partial_output = write_text_partial(args.output, receipt_text)

        # The receipt is the commit marker and is promoted last. Individual files
        # are atomically replaced; an interrupted run cannot leave a partial JSON.
        partial_metadata.replace(args.metadata)
        partial_model.replace(args.model)
        verify_model_npz(args.model, set(all_arrays))
        if sha256_file(args.model) != model_hash:
            raise FitFailure("model hash changed during promotion")
        if sha256_file(args.metadata) != metadata_hash:
            raise FitFailure("metadata hash changed during promotion")
        partial_output.replace(args.output)
    except (FitFailure, MODEL.ModelFailure, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_TRAIN_OPERATOR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS_TRAIN_OPERATOR_RANK_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
