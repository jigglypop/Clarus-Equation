"""Numerical primitives for the BA-SRM1 preregistered analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist


ALPHAS = tuple(10.0**power for power in range(-6, 3))
K_VALUES = (8, 16, 32)
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
QUADRATIC_PAIRS = tuple((i, j) for i in range(4) for j in range(i, 4))
METRIC_KINDS = ("reference", "diagonal", "constant", "variable")


def stable_split(group_id: str) -> str:
    bucket = hashlib.sha256(group_id.encode("utf-8")).digest()[0] % 10
    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "development"
    return "confirmation"


def inner_fold(group_id: str, n_folds: int = 5) -> int:
    digest = hashlib.sha256(("BA-SRM1-inner:" + group_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def quadratic_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"expected n×4 input; got {x.shape}")
    quadratic = np.column_stack([x[:, i] * x[:, j] for i, j in QUADRATIC_PAIRS])
    return np.column_stack([np.ones(x.shape[0]), x, quadratic])


def quadratic_feature_jacobians(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    jac = np.zeros((x.shape[0], 15, 4), dtype=float)
    for i in range(4):
        jac[:, 1 + i, i] = 1.0
    offset = 5
    for feature, (i, j) in enumerate(QUADRATIC_PAIRS, start=offset):
        if i == j:
            jac[:, feature, i] = 2.0 * x[:, i]
        else:
            jac[:, feature, i] = x[:, j]
            jac[:, feature, j] = x[:, i]
    return jac


def linear_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(x.shape[0]), x])


def fit_ridge(design: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + float(alpha) * penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def grouped_mse(y: np.ndarray, pred: np.ndarray, groups: Sequence[str]) -> float:
    row_loss = np.mean((np.asarray(y) - np.asarray(pred)) ** 2, axis=1)
    values = []
    groups_arr = np.asarray(groups, dtype=object)
    for group in sorted(set(groups_arr.tolist())):
        values.append(float(np.mean(row_loss[groups_arr == group])))
    return float(np.mean(values))


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    feature_kind: str,
    alphas: Sequence[float] = ALPHAS,
) -> dict:
    feature_fn = quadratic_features if feature_kind == "quadratic" else linear_features
    groups_arr = np.asarray(groups, dtype=object)
    folds = np.array([inner_fold(str(group)) for group in groups_arr], dtype=int)
    losses: dict[str, float] = {}
    predictions: dict[float, np.ndarray] = {}
    for alpha in alphas:
        pred = np.full_like(y, np.nan, dtype=float)
        for fold in range(5):
            train = folds != fold
            valid = folds == fold
            if not np.any(valid) or not np.any(train):
                raise ValueError(f"empty grouped inner fold {fold}")
            coef = fit_ridge(feature_fn(x[train]), y[train], alpha)
            pred[valid] = feature_fn(x[valid]) @ coef
        if not np.all(np.isfinite(pred)):
            raise ValueError("nonfinite ridge OOF prediction")
        losses[f"{alpha:.0e}"] = grouped_mse(y, pred, groups_arr)
        predictions[float(alpha)] = pred
    best = min((float(losses[f"{a:.0e}"]), float(a)) for a in alphas)[1]
    return {
        "alpha": best,
        "losses": losses,
        "oof_prediction": predictions[best],
    }


@dataclass(frozen=True)
class ResponseModel:
    coefficient: np.ndarray
    residual_variance: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return quadratic_features(x) @ self.coefficient

    def jacobians(self, x: np.ndarray) -> np.ndarray:
        basis_jac = quadratic_feature_jacobians(x)
        return np.einsum("nfi,fo->noi", basis_jac, self.coefficient)

    def metrics(self, x: np.ndarray) -> np.ndarray:
        jac = self.jacobians(x)
        inv_var = 1.0 / self.residual_variance
        return np.einsum("noi,o,noj->nij", jac, inv_var, jac)


def fit_response_model(x: np.ndarray, y: np.ndarray, alpha: float) -> ResponseModel:
    coef = fit_ridge(quadratic_features(x), y, alpha)
    residual = y - quadratic_features(x) @ coef
    variance = np.maximum(np.mean(residual**2, axis=0), 1e-6)
    return ResponseModel(coef, variance)


def symmetric_knn_adjacency(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k={k} invalid for n={n}")
    distances = cdist(x, x, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    adjacency = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    adjacency[rows, neighbors.reshape(-1)] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    return adjacency


@dataclass(frozen=True)
class GraphGeometry:
    adjacency: np.ndarray
    shortest_paths: np.ndarray
    edge_scale: float
    edge_rows: np.ndarray
    edge_cols: np.ndarray
    edge_lengths: np.ndarray


def build_graph(
    x: np.ndarray,
    node_metrics: np.ndarray,
    k: int,
    *,
    adjacency: np.ndarray | None = None,
) -> GraphGeometry:
    x = np.asarray(x, dtype=float)
    metrics = np.asarray(node_metrics, dtype=float)
    if adjacency is None:
        adjacency = symmetric_knn_adjacency(x, k)
    rows, cols = np.where(np.triu(adjacency, k=1))
    delta = x[rows] - x[cols]
    avg_metric = 0.5 * (metrics[rows] + metrics[cols])
    squared = np.einsum("ni,nij,nj->n", delta, avg_metric, delta)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(squared), initial=0.0)))
    if np.any(squared < -tolerance):
        raise ValueError("metric produced negative squared edge length")
    lengths = np.sqrt(np.maximum(squared, 0.0))
    if len(lengths) == 0 or np.any(lengths <= 0.0):
        raise ValueError("metric graph has zero/nonexistent edge length")
    matrix = np.zeros((x.shape[0], x.shape[0]), dtype=float)
    matrix[rows, cols] = lengths
    matrix[cols, rows] = lengths
    shortest = dijkstra(csr_matrix(matrix), directed=False, unweighted=False)
    if not np.all(np.isfinite(shortest)):
        raise ValueError("train graph is disconnected")
    return GraphGeometry(
        adjacency=adjacency,
        shortest_paths=shortest,
        edge_scale=float(np.median(lengths)),
        edge_rows=rows,
        edge_cols=cols,
        edge_lengths=lengths,
    )


def query_distances(
    x_train: np.ndarray,
    train_metrics: np.ndarray,
    graph: GraphGeometry,
    x_query: np.ndarray,
    query_metrics: np.ndarray,
    k: int,
    *,
    neighbor_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if neighbor_indices is None:
        ref_distances = cdist(x_query, x_train, metric="euclidean")
        neighbor_indices = np.argpartition(ref_distances, kth=k - 1, axis=1)[:, :k]
    out = np.full((x_query.shape[0], x_train.shape[0]), np.inf, dtype=float)
    for row in range(x_query.shape[0]):
        neighbors = neighbor_indices[row]
        delta = x_query[row] - x_train[neighbors]
        avg_metric = 0.5 * (query_metrics[row][None, :, :] + train_metrics[neighbors])
        squared = np.einsum("ni,nij,nj->n", delta, avg_metric, delta)
        if np.any(squared <= 0.0) or np.any(~np.isfinite(squared)):
            raise ValueError("query attachment has nonpositive/nonfinite edge")
        attachment = np.sqrt(squared)
        out[row] = np.min(attachment[:, None] + graph.shortest_paths[neighbors], axis=0)
    if not np.all(np.isfinite(out)):
        raise ValueError("held-out query has no finite train path")
    return out, neighbor_indices


def kernel_predict(distances: np.ndarray, y_train: np.ndarray, bandwidth: float) -> np.ndarray:
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    log_weight = -0.5 * (distances / bandwidth) ** 2
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(log_weight)
    denominator = np.sum(weight, axis=1, keepdims=True)
    if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
        raise ValueError("kernel has zero/nonfinite weight")
    return weight @ y_train / denominator


def metric_stacks(
    train_metrics: np.ndarray,
    query_metrics: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "reference":
        constant = np.eye(4)
    else:
        mean_metric = np.mean(train_metrics, axis=0)
        constant = np.diag(np.diag(mean_metric)) if kind == "diagonal" else mean_metric
    if kind == "variable":
        return train_metrics, query_metrics
    return (
        np.repeat(constant[None, :, :], train_metrics.shape[0], axis=0),
        np.repeat(constant[None, :, :], query_metrics.shape[0], axis=0),
    )


def select_kernel_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    response_alpha: float,
) -> dict:
    groups_arr = np.asarray(groups, dtype=object)
    folds = np.array([inner_fold(str(group)) for group in groups_arr], dtype=int)
    candidates = {
        (kind, k, multiplier): np.full_like(y, np.nan, dtype=float)
        for kind in METRIC_KINDS
        for k in K_VALUES
        for multiplier in BANDWIDTH_MULTIPLIERS
    }
    invalid: CounterLike = CounterLike()
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        model = fit_response_model(x[train], y[train], response_alpha)
        train_metric = model.metrics(x[train])
        valid_metric = model.metrics(x[valid])
        for k in K_VALUES:
            try:
                adjacency = symmetric_knn_adjacency(x[train], k)
            except ValueError as exc:
                for kind in METRIC_KINDS:
                    invalid.add((kind, k), str(exc))
                continue
            for kind in METRIC_KINDS:
                metric_train, metric_valid = metric_stacks(train_metric, valid_metric, kind)
                try:
                    graph = build_graph(x[train], metric_train, k, adjacency=adjacency)
                    distances, _ = query_distances(
                        x[train], metric_train, graph, x[valid], metric_valid, k
                    )
                except ValueError as exc:
                    invalid.add((kind, k), str(exc))
                    continue
                for multiplier in BANDWIDTH_MULTIPLIERS:
                    pred = kernel_predict(distances, y[train], multiplier * graph.edge_scale)
                    candidates[kind, k, multiplier][valid] = pred

    chosen: dict[str, dict] = {}
    losses: dict[str, dict] = {kind: {} for kind in METRIC_KINDS}
    for kind in METRIC_KINDS:
        valid_options = []
        for k in K_VALUES:
            for multiplier in BANDWIDTH_MULTIPLIERS:
                pred = candidates[kind, k, multiplier]
                key = f"k={k},bw={multiplier:g}"
                if np.all(np.isfinite(pred)):
                    loss = grouped_mse(y, pred, groups_arr)
                    losses[kind][key] = loss
                    valid_options.append((loss, k, multiplier, pred))
                else:
                    losses[kind][key] = None
        if valid_options:
            loss, k, multiplier, pred = min(valid_options, key=lambda item: item[:3])
            chosen[kind] = {
                "k": int(k),
                "bandwidth_multiplier": float(multiplier),
                "grouped_mse": float(loss),
                "oof_prediction": pred,
            }
        else:
            chosen[kind] = {"status": "NO_CONNECTED_FINITE_CANDIDATE"}
    return {"chosen": chosen, "losses": losses, "invalid": invalid.to_dict()}


class CounterLike:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    def add(self, key: tuple, reason: str) -> None:
        name = f"{key}:{reason}"
        self._values[name] = self._values.get(name, 0) + 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self._values.items()))


def final_kernel_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    response_model: ResponseModel,
    kind: str,
    k: int,
    bandwidth_multiplier: float,
) -> tuple[np.ndarray, GraphGeometry, np.ndarray, np.ndarray]:
    train_metric_raw = response_model.metrics(x_train)
    query_metric_raw = response_model.metrics(x_query)
    train_metric, query_metric = metric_stacks(train_metric_raw, query_metric_raw, kind)
    adjacency = symmetric_knn_adjacency(x_train, k)
    graph = build_graph(x_train, train_metric, k, adjacency=adjacency)
    distances, neighbors = query_distances(
        x_train, train_metric, graph, x_query, query_metric, k
    )
    pred = kernel_predict(distances, y_train, bandwidth_multiplier * graph.edge_scale)
    return pred, graph, distances, neighbors


def predictive_variance(y: np.ndarray, oof_prediction: np.ndarray) -> np.ndarray:
    return np.maximum(np.mean((y - oof_prediction) ** 2, axis=0), 1e-6)


def gaussian_logpdf_diag(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    residual = y - mean
    return -0.5 * np.sum(
        np.log(2.0 * math.pi * variance)[None, :] + residual**2 / variance[None, :],
        axis=1,
    )


def worst_support_eigenvalue_ratio(eigenvalues: np.ndarray) -> float:
    """Return min_support(lambda_min / lambda_max) for an ordered spectrum."""

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 2 or eigenvalues.shape[1] < 2:
        raise ValueError("expected support-by-eigenvalue matrix")
    ratio = eigenvalues[:, 0] / np.maximum(
        eigenvalues[:, -1], np.finfo(float).tiny
    )
    return float(np.min(ratio))


def grouped_elpd_comparison(
    candidate_logpdf: np.ndarray,
    control_logpdf: np.ndarray,
    groups: Sequence[str],
) -> dict:
    groups_arr = np.asarray(groups, dtype=object)
    differences = []
    by_group = {}
    for group in sorted(set(groups_arr.tolist())):
        value = float(np.sum(candidate_logpdf[groups_arr == group] - control_logpdf[groups_arr == group]))
        differences.append(value)
        by_group[str(group)] = value
    delta = float(np.sum(differences))
    n = len(differences)
    se_total = float(np.std(differences, ddof=1) * math.sqrt(n)) if n > 1 else float("inf")
    nonzero = [value for value in differences if value != 0.0]
    direction = float(np.mean(np.asarray(nonzero) > 0.0)) if nonzero else 0.0
    return {
        "delta_elpd": delta,
        "se_total": se_total,
        "two_se": 2.0 * se_total,
        "positive_slice_fraction": direction,
        "slice_count": n,
        "by_slice": by_group,
    }


def bootstrap_rank_audit(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alpha: float,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    model = fit_response_model(x, y, alpha)
    metric = model.metrics(x)
    eigen = np.linalg.eigvalsh(metric)
    nominal_ratio = eigen[:, 0] / np.maximum(eigen[:, -1], np.finfo(float).tiny)
    nominal_full_rank = bool(np.all(eigen[:, 0] > 0.0))
    unique_groups = np.array(sorted(set(map(str, groups))), dtype=object)
    groups_arr = np.asarray(groups, dtype=object)
    group_indices = {group: np.flatnonzero(groups_arr == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    statistics = []
    full_rank = []
    r_conditions = []
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([group_indices[group] for group in sampled])
        boot_model = fit_response_model(x[index], y[index], alpha)
        boot_metric = boot_model.metrics(x)
        boot_eigen = np.linalg.eigvalsh(boot_metric)
        # Contract O2 binds the statistic to the worst support point in each
        # slice-cluster resample, not to a lower-tail summary of that support.
        statistics.append(worst_support_eigenvalue_ratio(boot_eigen))
        full_rank.append(bool(np.all(boot_eigen[:, 0] > 0.0)))
        r_conditions.append(float(np.max(boot_model.residual_variance) / np.min(boot_model.residual_variance)))
    lower = float(np.quantile(statistics, 0.025))
    r_condition = float(np.max(model.residual_variance) / np.min(model.residual_variance))
    passed = bool(
        nominal_full_rank
        and all(full_rank)
        and lower > 1e-4
        and r_condition <= 1e6
        and max(r_conditions) <= 1e6
    )
    return {
        "status": "PASS" if passed else "RANK_UNIDENTIFIED",
        "nominal_full_rank_all_support": nominal_full_rank,
        "nominal_ratio_min": float(np.min(nominal_ratio)),
        "nominal_ratio_median": float(np.median(nominal_ratio)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_full_rank_fraction": float(np.mean(full_rank)),
        "bootstrap_statistic_lower_2p5": lower,
        "bootstrap_statistic_median": float(np.median(statistics)),
        "r_condition_nominal": r_condition,
        "r_condition_bootstrap_max": float(max(r_conditions)),
        "threshold": 1e-4,
        "seed": seed,
    }


def fixed_gauge_transforms() -> list[tuple[str, np.ndarray, np.ndarray]]:
    transforms: list[tuple[str, np.ndarray, np.ndarray]] = []
    for seed in range(83101, 83117):
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.normal(size=(4, 4)))
        signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
        q = q @ np.diag(signs)
        transforms.append((f"orthogonal-{seed}", q, np.zeros(4)))
    for values in itertools.product((0.5, 2.0), repeat=4):
        transforms.append(("diagonal-" + "-".join(map(str, values)), np.diag(values), np.zeros(4)))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for shear in (-0.25, 0.25):
                a = np.eye(4)
                a[i, j] = shear
                transforms.append((f"shear-{i}-{j}-{shear:+g}", a, np.zeros(4)))
    for i in range(4):
        for sign in (-1.0, 1.0):
            b = np.zeros(4)
            b[i] = sign
            transforms.append((f"translation-{i}-{sign:+g}", np.eye(4), b))
    return transforms


def gauge_audit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    response_model: ResponseModel,
    *,
    k: int,
    bandwidth_multiplier: float,
) -> dict:
    raw_train_metric = response_model.metrics(x_train)
    raw_query_metric = response_model.metrics(x_query)
    adjacency = symmetric_knn_adjacency(x_train, k)
    original_graph = build_graph(x_train, raw_train_metric, k, adjacency=adjacency)
    original_distances, neighbors = query_distances(
        x_train, raw_train_metric, original_graph, x_query, raw_query_metric, k
    )
    bandwidth = bandwidth_multiplier * original_graph.edge_scale
    original_prediction = kernel_predict(original_distances, y_train, bandwidth)
    original_spectrum = np.stack([eigvalsh(g, np.eye(4)) for g in raw_train_metric])

    max_line_error = 0.0
    max_spectrum_error = 0.0
    max_prediction_error = 0.0
    tested = 0
    failures = []
    for name, a, b in fixed_gauge_transforms():
        condition = float(np.linalg.cond(a))
        if condition > 4.0 + 1e-12:
            continue
        inv = np.linalg.inv(a)
        metric_transform = lambda g: inv.T @ g @ inv
        x_train_new = x_train @ a.T + b
        x_query_new = x_query @ a.T + b
        train_metric_new = np.stack([metric_transform(g) for g in raw_train_metric])
        query_metric_new = np.stack([metric_transform(g) for g in raw_query_metric])
        gref_new = inv.T @ inv
        graph_new = build_graph(
            x_train_new, train_metric_new, k, adjacency=adjacency
        )
        distances_new, _ = query_distances(
            x_train_new,
            train_metric_new,
            graph_new,
            x_query_new,
            query_metric_new,
            k,
            neighbor_indices=neighbors,
        )
        prediction_new = kernel_predict(distances_new, y_train, bandwidth)
        spectrum_new = np.stack([eigvalsh(g, gref_new) for g in train_metric_new])
        line_den = np.maximum(np.abs(original_graph.edge_lengths), np.finfo(float).tiny)
        line_error = float(np.max(np.abs(graph_new.edge_lengths - original_graph.edge_lengths) / line_den))
        spectrum_den = np.maximum(np.abs(original_spectrum), np.finfo(float).tiny)
        spectrum_error = float(np.max(np.abs(spectrum_new - original_spectrum) / spectrum_den))
        pred_den = np.maximum(np.abs(original_prediction), 1.0)
        prediction_error = float(np.max(np.abs(prediction_new - original_prediction) / pred_den))
        max_line_error = max(max_line_error, line_error)
        max_spectrum_error = max(max_spectrum_error, spectrum_error)
        max_prediction_error = max(max_prediction_error, prediction_error)
        if line_error > 1e-8 or spectrum_error > 1e-8 or prediction_error > 1e-4:
            failures.append(name)
        tested += 1
    passed = not failures and tested == 64
    return {
        "status": "PASS" if passed else "FAIL",
        "transforms_tested": tested,
        "max_line_element_relative_error": max_line_error,
        "max_generalized_spectrum_relative_error": max_spectrum_error,
        "max_prediction_relative_error": max_prediction_error,
        "line_spectrum_tolerance": 1e-8,
        "prediction_tolerance": 1e-4,
        "failures": failures,
    }
