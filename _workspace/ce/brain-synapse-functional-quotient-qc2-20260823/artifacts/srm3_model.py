"""Numerical primitives for the frozen BA-SRM3 train-only model.

The module is database-free.  It implements fold-local preprocessing, an exact
multi-output RBF kernel ridge operator, its analytic Jacobian, predictive
covariance, pullback tensors, and coordinate-transport checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg


MAD_TO_SIGMA = 1.482602218505602
IQR_TO_SIGMA = 1.0 / 1.3489795003921634
PCA_RELATIVE_RANK_TOL = 1e-12
RANK_RELATIVE_TOL = 1e-4


class ModelFailure(RuntimeError):
    """Raised when a frozen numerical gate cannot be instantiated."""


@dataclass(frozen=True)
class Preprocessor:
    numeric_median: np.ndarray
    numeric_scale: np.ndarray
    category_levels: tuple[tuple[str, ...], ...]
    keep_columns: np.ndarray
    composite_center: np.ndarray
    pca_components: np.ndarray
    pca_eigenvalues: np.ndarray


@dataclass(frozen=True)
class OutputScaler:
    median: np.ndarray
    mad: np.ndarray


@dataclass(frozen=True)
class KRRModel:
    train_coordinates: np.ndarray
    alpha: np.ndarray
    ell: float
    ridge: float
    output_median: np.ndarray
    output_mad: np.ndarray


def finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=float))


def _robust_location_scale(column: np.ndarray) -> tuple[float, float]:
    valid = column[np.isfinite(column)]
    if valid.size == 0:
        return 0.0, 1.0
    center = float(np.median(valid))
    mad = float(np.median(np.abs(valid - center))) * MAD_TO_SIGMA
    if math.isfinite(mad) and mad > 0.0:
        return center, mad
    q25, q75 = np.percentile(valid, [25.0, 75.0])
    iqr_scale = float(q75 - q25) * IQR_TO_SIGMA
    if math.isfinite(iqr_scale) and iqr_scale > 0.0:
        return center, iqr_scale
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    if math.isfinite(std) and std > 0.0:
        return center, std
    return center, 1.0


def _normalize_category(value: object) -> str:
    if value is None:
        return "__MISSING__"
    text = str(value)
    return text if text else "__EMPTY__"


def _fit_numeric(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ModelFailure("numeric input must be a matrix")
    stats = [_robust_location_scale(values[:, j]) for j in range(values.shape[1])]
    median = np.asarray([item[0] for item in stats], dtype=float)
    scale = np.asarray([item[1] for item in stats], dtype=float)
    return median, scale


def _transform_numeric(
    values: np.ndarray, median: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    filled = np.where(missing, median[None, :], values)
    standardized = (filled - median[None, :]) / scale[None, :]
    return np.concatenate([standardized, missing.astype(float)], axis=1)


def _fit_category_levels(values: np.ndarray) -> tuple[tuple[str, ...], ...]:
    values = np.asarray(values, dtype=object)
    if values.ndim != 2:
        raise ModelFailure("categorical input must be a matrix")
    levels = []
    for j in range(values.shape[1]):
        observed = sorted({_normalize_category(value) for value in values[:, j]})
        if "__UNK__" not in observed:
            observed.append("__UNK__")
        levels.append(tuple(observed))
    return tuple(levels)


def _transform_categories(
    values: np.ndarray, levels: tuple[tuple[str, ...], ...]
) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    columns: list[np.ndarray] = []
    for j, vocabulary in enumerate(levels):
        index = {level: idx for idx, level in enumerate(vocabulary)}
        unknown = index["__UNK__"]
        encoded = np.zeros((values.shape[0], len(vocabulary)), dtype=float)
        for i, value in enumerate(values[:, j]):
            label = _normalize_category(value)
            encoded[i, index.get(label, unknown)] = 1.0
        columns.append(encoded)
    if not columns:
        return np.empty((values.shape[0], 0), dtype=float)
    return np.concatenate(columns, axis=1)


def fit_preprocessor(
    numeric: np.ndarray,
    categorical: np.ndarray,
    max_dimension: int,
) -> tuple[Preprocessor, np.ndarray]:
    numeric = np.asarray(numeric, dtype=float)
    categorical = np.asarray(categorical, dtype=object)
    if numeric.shape[0] != categorical.shape[0]:
        raise ModelFailure("numeric/categorical row mismatch")
    median, scale = _fit_numeric(numeric)
    levels = _fit_category_levels(categorical)
    numeric_design = _transform_numeric(numeric, median, scale)
    category_design = _transform_categories(categorical, levels)
    design = np.concatenate([numeric_design, category_design], axis=1)
    variance = np.var(design, axis=0)
    keep = np.flatnonzero(np.isfinite(variance) & (variance > 0.0))
    if keep.size == 0:
        raise ModelFailure("all input channels are constant")
    design = design[:, keep]
    center = np.mean(design, axis=0)
    centered = design - center[None, :]
    _, singular, vt = scipy.linalg.svd(
        centered, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    if singular.size == 0 or singular[0] <= 0.0:
        raise ModelFailure("input design has zero rank")
    eigenvalues = singular**2 / max(1, centered.shape[0] - 1)
    positive = eigenvalues > eigenvalues[0] * PCA_RELATIVE_RANK_TOL
    available = int(np.sum(positive))
    retained = min(int(max_dimension), available)
    if retained < 1:
        raise ModelFailure("no numerically identified PCA directions")
    components = vt[:retained].copy()
    eigenvalues = eigenvalues[:retained].copy()
    # Resolve SVD sign ambiguity using the largest-magnitude loading.
    for j in range(retained):
        pivot = int(np.argmax(np.abs(components[j])))
        if components[j, pivot] < 0.0:
            components[j] *= -1.0
    scores = centered @ components.T
    coordinates = scores / np.sqrt(eigenvalues)[None, :]
    params = Preprocessor(
        numeric_median=median,
        numeric_scale=scale,
        category_levels=levels,
        keep_columns=keep,
        composite_center=center,
        pca_components=components,
        pca_eigenvalues=eigenvalues,
    )
    return params, coordinates


def transform_preprocessor(
    params: Preprocessor,
    numeric: np.ndarray,
    categorical: np.ndarray,
    dimension: int,
) -> np.ndarray:
    if dimension < 1 or dimension > params.pca_components.shape[0]:
        raise ModelFailure("requested unavailable PCA dimension")
    numeric_design = _transform_numeric(
        np.asarray(numeric, dtype=float),
        params.numeric_median,
        params.numeric_scale,
    )
    category_design = _transform_categories(
        np.asarray(categorical, dtype=object), params.category_levels
    )
    design = np.concatenate([numeric_design, category_design], axis=1)
    centered = design[:, params.keep_columns] - params.composite_center[None, :]
    scores = centered @ params.pca_components[:dimension].T
    return scores / np.sqrt(params.pca_eigenvalues[:dimension])[None, :]


def fit_output_scaler(target: np.ndarray) -> OutputScaler:
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or not np.all(np.isfinite(target)):
        raise ModelFailure("target must be a finite matrix")
    median = np.median(target, axis=0)
    mad = np.median(np.abs(target - median[None, :]), axis=0)
    if not np.all(np.isfinite(mad) & (mad > 0.0)):
        raise ModelFailure("target contains zero/nonfinite MAD coordinate")
    return OutputScaler(median=median, mad=mad)


def standardize_target(target: np.ndarray, scaler: OutputScaler) -> np.ndarray:
    return (np.asarray(target, dtype=float) - scaler.median[None, :]) / scaler.mad[
        None, :
    ]


def unstandardize_target(target: np.ndarray, scaler: OutputScaler) -> np.ndarray:
    return scaler.median[None, :] + np.asarray(target, dtype=float) * scaler.mad[
        None, :
    ]


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    values = (
        np.sum(left**2, axis=1)[:, None]
        + np.sum(right**2, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(values, 0.0)


def rbf_kernel(left: np.ndarray, right: np.ndarray, ell: float) -> np.ndarray:
    if not math.isfinite(ell) or ell <= 0.0:
        raise ModelFailure("ell must be positive and finite")
    return np.exp(-squared_distances(left, right) / (2.0 * ell**2))


def fit_krr(
    coordinates: np.ndarray,
    standardized_target: np.ndarray,
    ell: float,
    ridge: float,
    output_scaler: OutputScaler,
) -> KRRModel:
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ModelFailure("ridge must be positive and finite")
    coordinates = np.asarray(coordinates, dtype=float)
    target = np.asarray(standardized_target, dtype=float)
    kernel = rbf_kernel(coordinates, coordinates, ell)
    system = kernel + float(ridge) * np.eye(kernel.shape[0])
    alpha = scipy.linalg.solve(
        system, target, assume_a="pos", check_finite=False
    )
    return KRRModel(
        train_coordinates=coordinates.copy(),
        alpha=alpha,
        ell=float(ell),
        ridge=float(ridge),
        output_median=output_scaler.median.copy(),
        output_mad=output_scaler.mad.copy(),
    )


def predict_krr(model: KRRModel, query: np.ndarray) -> np.ndarray:
    kernel = rbf_kernel(np.asarray(query, dtype=float), model.train_coordinates, model.ell)
    standardized = kernel @ model.alpha
    return model.output_median[None, :] + standardized * model.output_mad[None, :]


def jacobian_krr(model: KRRModel, query: np.ndarray) -> np.ndarray:
    """Return Jacobians with shape (n_query, n_output, n_input)."""
    query = np.asarray(query, dtype=float)
    kernel = rbf_kernel(query, model.train_coordinates, model.ell)
    difference = model.train_coordinates[None, :, :] - query[:, None, :]
    weighted = kernel[:, :, None] * difference / (model.ell**2)
    # qid, io -> qod, then restore raw-output units via output MAD.
    jacobian_std = np.einsum("qid,io->qod", weighted, model.alpha, optimize=True)
    return jacobian_std * model.output_mad[None, :, None]


def residual_covariance(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[0] < 2:
        raise ModelFailure("at least two residual rows are required")
    centered = residuals - np.mean(residuals, axis=0, keepdims=True)
    return centered.T @ centered / (residuals.shape[0] - 1)


def shrink_covariance(sample: np.ndarray, gamma: float) -> np.ndarray:
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 2 or sample.shape[0] != sample.shape[1]:
        raise ModelFailure("sample covariance must be square")
    if not np.all(np.isfinite(sample)):
        raise ModelFailure("sample covariance must be finite")
    if gamma < 0.0 or gamma > 1.0:
        raise ModelFailure("gamma must be in [0,1]")
    result = (1.0 - gamma) * sample + gamma * np.diag(np.diag(sample))
    return (result + result.T) * 0.5


def floored_covariance(covariance: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ModelFailure("covariance must be square")
    if not np.all(np.isfinite(covariance)):
        raise ModelFailure("covariance must be finite")
    diagonal_median = float(np.median(np.diag(covariance)))
    if not math.isfinite(diagonal_median) or diagonal_median <= 0.0:
        raise ModelFailure("covariance diagonal median is not positive")
    floor = 1e-8 * diagonal_median
    return covariance + floor * np.eye(covariance.shape[0]), floor


def gaussian_log_score(
    target: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if target.shape != mean.shape or target.ndim != 2:
        raise ModelFailure("target/mean shape mismatch")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(mean)):
        raise ModelFailure("target and mean must be finite")
    chol = scipy.linalg.cholesky(covariance, lower=True, check_finite=False)
    residual = (target - mean).T
    whitened = scipy.linalg.solve_triangular(
        chol, residual, lower=True, check_finite=False
    )
    quadratic = np.sum(whitened**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    dimension = target.shape[1]
    return -0.5 * (quadratic + logdet + dimension * np.log(2.0 * np.pi))


def pullback_metrics(jacobians: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    jacobians = np.asarray(jacobians, dtype=float)
    chol = scipy.linalg.cholesky(covariance, lower=True, check_finite=False)
    metrics = np.empty(
        (jacobians.shape[0], jacobians.shape[2], jacobians.shape[2]), dtype=float
    )
    for idx, jacobian in enumerate(jacobians):
        whitened = scipy.linalg.solve_triangular(
            chol, jacobian, lower=True, check_finite=False
        )
        metric = whitened.T @ whitened
        metrics[idx] = (metric + metric.T) * 0.5
    return metrics


def numerical_ranks(
    jacobians: np.ndarray, relative_tolerance: float = RANK_RELATIVE_TOL
) -> tuple[np.ndarray, np.ndarray]:
    ranks = []
    fifth_ratios = []
    for jacobian in np.asarray(jacobians, dtype=float):
        singular = scipy.linalg.svdvals(jacobian, check_finite=False)
        if singular.size == 0 or singular[0] <= 0.0:
            ranks.append(0)
            fifth_ratios.append(0.0)
            continue
        ratios = singular / singular[0]
        ranks.append(int(np.sum(ratios >= relative_tolerance)))
        fifth_ratios.append(float(ratios[4]) if ratios.size >= 5 else 0.0)
    return np.asarray(ranks, dtype=int), np.asarray(fifth_ratios, dtype=float)


def secant_squared(
    query: np.ndarray,
    reference: np.ndarray,
    query_metrics: np.ndarray,
    reference_metrics: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    result = np.empty((query.shape[0], reference.shape[0]), dtype=float)
    for q_idx in range(query.shape[0]):
        difference = reference - query[q_idx][None, :]
        left = np.einsum(
            "ni,ij,nj->n",
            difference,
            query_metrics[q_idx],
            difference,
            optimize=True,
        )
        right = np.einsum(
            "ni,nij,nj->n",
            difference,
            reference_metrics,
            difference,
            optimize=True,
        )
        result[q_idx] = np.maximum(0.5 * (left + right), 0.0)
    return result


def neighbor_predict(
    distances_squared: np.ndarray,
    reference_target: np.ndarray,
    rho: float,
    minimum_effective_neighbors: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(rho) or rho <= 0.0:
        raise ModelFailure("rho must be positive and finite")
    weights = np.exp(-np.asarray(distances_squared, dtype=float) / (2.0 * rho**2))
    totals = np.sum(weights, axis=1)
    effective = totals**2 / np.sum(weights**2, axis=1)
    supported = (totals > 0.0) & (effective >= minimum_effective_neighbors)
    prediction = np.full(
        (weights.shape[0], np.asarray(reference_target).shape[1]), np.nan
    )
    prediction[supported] = (
        weights[supported] @ np.asarray(reference_target, dtype=float)
    ) / totals[supported, None]
    return prediction, effective, supported


def slice_equal_delta(
    candidate_score: np.ndarray,
    control_score: np.ndarray,
    slice_ids: Sequence[str],
) -> tuple[float, float, int]:
    delta = np.asarray(candidate_score) - np.asarray(control_score)
    grouped: dict[str, list[float]] = {}
    for value, group in zip(delta, slice_ids):
        grouped.setdefault(str(group), []).append(float(value))
    means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if means.size < 2:
        raise ModelFailure("at least two slice groups are required for SE")
    return float(np.mean(means)), float(np.std(means, ddof=1) / np.sqrt(means.size)), int(
        means.size
    )


def deterministic_fold(group_id: str, salt: str, folds: int) -> int:
    digest = hashlib.sha256(f"{salt}{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def generalized_spectrum(metric: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = scipy.linalg.eigvalsh(metric, reference, check_finite=False)
    return np.sort(np.real(values))


def transport_metric(metric: np.ndarray, transform: np.ndarray) -> np.ndarray:
    inverse = scipy.linalg.inv(transform, check_finite=False)
    return inverse.T @ metric @ inverse


def transport_jacobian(jacobian: np.ndarray, transform: np.ndarray) -> np.ndarray:
    inverse = scipy.linalg.inv(transform, check_finite=False)
    return jacobian @ inverse
