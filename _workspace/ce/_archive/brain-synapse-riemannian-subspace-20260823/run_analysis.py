"""Execute the fail-closed BA-SRM1 Allen-SynPhys analysis.

The confirmation target columns are queried only for strata that pass the
pre-registered rank, gauge, and development-survival gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import platform
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy

from srm1_analysis import (
    fit_response_model,
    fit_ridge,
    final_kernel_prediction,
    gaussian_logpdf_diag,
    gauge_audit,
    grouped_elpd_comparison,
    inner_fold,
    linear_features,
    predictive_variance,
    quadratic_features,
    select_kernel_hyperparameters,
    select_ridge_alpha,
    stable_split,
    bootstrap_rank_audit,
)


EXPECTED_DATABASE_SHA256 = (
    "7372499fdd874f057565080d5769baaf2659ef39d9f3bc3c7147dd1e1c280a53"
)
TARGET_COLUMNS = (
    "pulse_amp_stp_initial_50hz",
    "pulse_amp_stp_induction_50hz",
    "pulse_amp_stp_recovery_250ms",
    "variability_stp_induced_state_50hz",
)
INPUT_COLUMNS = (
    "abs_resting_psp_over_train_median",
    "soma_distance_over_1_m",
    "post_input_resistance_over_1_ohm",
    "post_membrane_tau_over_1_s",
)
SEEDS = {"ex": 83201, "in": 83202}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0}
    if not np.all(np.isfinite(array)):
        raise ValueError("summary received nonfinite values")
    q = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(array.size),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def grouped_mean_oof(y: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    groups_array = np.asarray(groups, dtype=object)
    folds = np.asarray([inner_fold(str(group)) for group in groups_array])
    prediction = np.full_like(y, np.nan, dtype=float)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        if not np.any(train) or not np.any(valid):
            raise ValueError(f"empty mean-control fold {fold}")
        prediction[valid] = np.mean(y[train], axis=0)
    return prediction


def category_key(row: dict) -> str:
    fields = (
        "pre_cre_type",
        "post_cre_type",
        "pre_layer",
        "post_layer",
        "pre_cell_class",
        "post_cell_class",
    )
    return "|".join(str(row.get(field) or "MISSING") for field in fields)


def category_fit_predict(
    y_train: np.ndarray,
    keys_train: Sequence[str],
    keys_query: Sequence[str],
) -> np.ndarray:
    global_mean = np.mean(y_train, axis=0)
    by_key: dict[str, np.ndarray] = {}
    keys_array = np.asarray(keys_train, dtype=object)
    for key in sorted(set(keys_array.tolist())):
        by_key[key] = np.mean(y_train[keys_array == key], axis=0)
    return np.stack([by_key.get(key, global_mean) for key in keys_query])


def category_oof(y: np.ndarray, keys: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    groups_array = np.asarray(groups, dtype=object)
    folds = np.asarray([inner_fold(str(group)) for group in groups_array])
    keys_array = np.asarray(keys, dtype=object)
    prediction = np.full_like(y, np.nan, dtype=float)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        prediction[valid] = category_fit_predict(
            y[train], keys_array[train].tolist(), keys_array[valid].tolist()
        )
    return prediction


@dataclass(frozen=True)
class Preprocessor:
    r_reference: float
    z_mean: np.ndarray
    z_covariance: np.ndarray
    z_covariance_shrunk: np.ndarray
    z_cholesky: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    def transform_z(self, rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        z = raw_z(rows, self.r_reference)
        x = np.linalg.solve(self.z_cholesky, (z - self.z_mean).T).T
        return z, x

    def transform_y(self, rows: Sequence[dict], targets: dict[int, np.ndarray], synapse_type: str) -> tuple[np.ndarray, np.ndarray]:
        y = raw_y(rows, targets, synapse_type, self.r_reference)
        return y, (y - self.y_mean) / self.y_scale


def raw_z(rows: Sequence[dict], r_reference: float) -> np.ndarray:
    values = np.asarray(
        [
            (
                abs(float(row["psp_amplitude"])) / r_reference,
                float(row["distance"]),
                float(row["input_resistance"]),
                float(row["tau"]),
            )
            for row in rows
        ],
        dtype=float,
    )
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("strict input domain contains nonpositive/nonfinite value")
    return np.log(values)


def raw_y(
    rows: Sequence[dict],
    targets: dict[int, np.ndarray],
    synapse_type: str,
    r_reference: float,
) -> np.ndarray:
    sign = 1.0 if synapse_type == "ex" else -1.0
    output = []
    for row in rows:
        target = targets[int(row["pair_id"])]
        output.append(
            (
                sign * target[0] / r_reference,
                sign * target[1] / r_reference,
                sign * target[2] / r_reference,
                target[3],
            )
        )
    array = np.asarray(output, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("target transform produced nonfinite value")
    return array


def fit_preprocessor(
    rows: Sequence[dict], targets: dict[int, np.ndarray], synapse_type: str
) -> tuple[Preprocessor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_reference = float(np.median([abs(float(row["psp_amplitude"])) for row in rows]))
    if not np.isfinite(r_reference) or r_reference <= 0.0:
        raise ValueError("invalid train resting-response reference")
    z = raw_z(rows, r_reference)
    z_mean = np.mean(z, axis=0)
    covariance = np.cov(z, rowvar=False, ddof=1)
    shrinkage = 1e-6 * float(np.trace(covariance)) / 4.0
    if not np.isfinite(shrinkage) or shrinkage <= 0.0:
        raise ValueError("invalid reference-covariance shrinkage")
    covariance_shrunk = covariance + shrinkage * np.eye(4)
    cholesky = np.linalg.cholesky(covariance_shrunk)
    x = np.linalg.solve(cholesky, (z - z_mean).T).T
    y = raw_y(rows, targets, synapse_type, r_reference)
    y_mean = np.mean(y, axis=0)
    y_scale = np.std(y, axis=0, ddof=1)
    if np.any(y_scale <= 0.0) or not np.all(np.isfinite(y_scale)):
        raise ValueError("target standardization is singular")
    y_standardized = (y - y_mean) / y_scale
    preprocessor = Preprocessor(
        r_reference,
        z_mean,
        covariance,
        covariance_shrunk,
        cholesky,
        y_mean,
        y_scale,
    )
    return preprocessor, z, x, y, y_standardized


def query_feature_rows(connection: sqlite3.Connection) -> list[dict]:
    sql = """
        SELECT sy.id AS synapse_id, pa.id AS pair_id,
               sl.ext_id AS slice_id, sl.species AS species,
               ex.project_name AS project_name, ex.target_region AS target_region,
               sy.synapse_type AS synapse_type,
               sy.psp_amplitude AS psp_amplitude,
               pa.distance AS distance,
               ipost.input_resistance AS input_resistance,
               ipost.tau AS tau,
               pre.cre_type AS pre_cre_type,
               post.cre_type AS post_cre_type,
               pre.target_layer AS pre_layer,
               post.target_layer AS post_layer,
               pre.cell_class AS pre_cell_class,
               post.cell_class AS post_cell_class
        FROM synapse AS sy
        JOIN pair AS pa ON pa.id = sy.pair_id
        JOIN experiment AS ex ON ex.id = pa.experiment_id
        JOIN slice AS sl ON sl.id = ex.slice_id
        JOIN dynamics AS dy ON dy.pair_id = pa.id
        JOIN intrinsic AS ipost ON ipost.cell_id = pa.post_cell_id
        JOIN cell AS pre ON pre.id = pa.pre_cell_id
        JOIN cell AS post ON post.id = pa.post_cell_id
        WHERE sl.species = 'mouse'
          AND ex.project_name IN ('mouse V1 coarse matrix', 'mouse V1 pre-production')
          AND sy.synapse_type IN ('ex', 'in')
          AND dy.qc_pass = 1
          AND sy.psp_amplitude IS NOT NULL AND abs(sy.psp_amplitude) > 0
          AND pa.distance IS NOT NULL AND pa.distance > 0
          AND ipost.input_resistance IS NOT NULL AND ipost.input_resistance > 0
          AND ipost.tau IS NOT NULL AND ipost.tau > 0
        ORDER BY sy.synapse_type, sl.ext_id, pa.id
    """
    rows = [dict(row) for row in connection.execute(sql)]
    pair_ids = [int(row["pair_id"]) for row in rows]
    synapse_ids = [int(row["synapse_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)) or len(synapse_ids) != len(set(synapse_ids)):
        raise RuntimeError("join uniqueness failure; no aggregation is allowed")
    for row in rows:
        values = np.asarray(
            [row["psp_amplitude"], row["distance"], row["input_resistance"], row["tau"]],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or values[1] <= 0 or values[2] <= 0 or values[3] <= 0 or values[0] == 0:
            raise RuntimeError("SQL strict-input predicate admitted an invalid row")
        row["split"] = stable_split(str(row["slice_id"]))
        row["category_key"] = category_key(row)
    return rows


def query_targets(
    connection: sqlite3.Connection, pair_ids: Sequence[int]
) -> tuple[dict[int, np.ndarray], dict]:
    if not pair_ids:
        return {}, {"requested": 0, "complete": 0, "excluded": 0}
    if len(pair_ids) > 900:
        raise ValueError("target query exceeds fixed SQLite parameter budget")
    placeholders = ",".join("?" for _ in pair_ids)
    sql = f"""
        SELECT pair_id, {', '.join(TARGET_COLUMNS)}
        FROM dynamics
        WHERE qc_pass = 1 AND pair_id IN ({placeholders})
        ORDER BY pair_id
    """
    result: dict[int, np.ndarray] = {}
    excluded = {"missing_row": 0, "null_or_nonfinite": 0}
    returned_ids = set()
    for row in connection.execute(sql, tuple(map(int, pair_ids))):
        pair_id = int(row[0])
        returned_ids.add(pair_id)
        if any(value is None for value in row[1:]):
            excluded["null_or_nonfinite"] += 1
            continue
        values = np.asarray(row[1:], dtype=float)
        if not np.all(np.isfinite(values)):
            excluded["null_or_nonfinite"] += 1
            continue
        result[pair_id] = values
    excluded["missing_row"] = len(set(map(int, pair_ids)) - returned_ids)
    receipt = {
        "requested": len(pair_ids),
        "complete": len(result),
        "excluded": excluded,
        "columns": list(TARGET_COLUMNS),
    }
    return result, receipt


def retain_target_complete(rows: Sequence[dict], targets: dict[int, np.ndarray]) -> list[dict]:
    return [row for row in rows if int(row["pair_id"]) in targets]


def linear_control(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Sequence[str],
    x_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    selected = select_ridge_alpha(
        x_train, y_train, groups_train, feature_kind="linear"
    )
    coefficient = fit_ridge(
        linear_features(x_train), y_train, float(selected["alpha"])
    )
    prediction = linear_features(x_query) @ coefficient
    return prediction, selected["oof_prediction"], {
        "alpha": selected["alpha"], "losses": selected["losses"]
    }


def evaluate_predictions(
    y_query: np.ndarray,
    query_groups: Sequence[str],
    predictions: dict[str, np.ndarray],
    variances: dict[str, np.ndarray],
    *,
    candidate: str,
    controls: Sequence[str],
) -> dict:
    logpdf = {
        name: gaussian_logpdf_diag(y_query, prediction, variances[name])
        for name, prediction in predictions.items()
    }
    totals = {name: float(np.sum(values)) for name, values in logpdf.items()}
    best_control = max(controls, key=lambda name: (totals[name], name))
    comparisons = {
        name: grouped_elpd_comparison(logpdf[candidate], logpdf[name], query_groups)
        for name in controls
    }
    primary = comparisons[best_control]
    survival = bool(
        primary["delta_elpd"] > primary["two_se"]
        and primary["positive_slice_fraction"] >= 0.75
    )
    return {
        "candidate": candidate,
        "best_control": best_control,
        "total_elpd": totals,
        "candidate_vs_controls": comparisons,
        "primary_comparison": primary,
        "survival": survival,
    }


def physical_stats(rows: Sequence[dict], targets: dict[int, np.ndarray] | None = None) -> dict:
    report = {
        "resting_psp_abs_mV": summary(abs(float(row["psp_amplitude"])) * 1e3 for row in rows),
        "soma_distance_um": summary(float(row["distance"]) * 1e6 for row in rows),
        "post_input_resistance_MOhm": summary(float(row["input_resistance"]) / 1e6 for row in rows),
        "post_membrane_tau_ms": summary(float(row["tau"]) * 1e3 for row in rows),
    }
    if targets is not None:
        complete = [row for row in rows if int(row["pair_id"]) in targets]
        for index, name in enumerate(TARGET_COLUMNS[:3]):
            report[name + "_mV"] = summary(
                float(targets[int(row["pair_id"])][index]) * 1e3 for row in complete
            )
        report[TARGET_COLUMNS[3]] = summary(
            float(targets[int(row["pair_id"])][3]) for row in complete
        )
    return report


def strip_kernel_selection(selection: dict) -> dict:
    output = {"losses": selection["losses"], "invalid": selection["invalid"], "chosen": {}}
    for name, chosen in selection["chosen"].items():
        output["chosen"][name] = {
            key: value for key, value in chosen.items() if key != "oof_prediction"
        }
    return output


def run_stratum_development(
    synapse_type: str,
    feature_rows: Sequence[dict],
    targets: dict[int, np.ndarray],
) -> tuple[dict, dict]:
    train_rows = retain_target_complete(
        [row for row in feature_rows if row["split"] == "train"], targets
    )
    development_rows = retain_target_complete(
        [row for row in feature_rows if row["split"] == "development"], targets
    )
    groups_train = [str(row["slice_id"]) for row in train_rows]
    groups_dev = [str(row["slice_id"]) for row in development_rows]
    support = {
        "train_pairs": len(train_rows),
        "train_slices": len(set(groups_train)),
        "development_pairs": len(development_rows),
        "development_slices": len(set(groups_dev)),
    }
    support_pass = bool(
        len(train_rows) + len(development_rows) >= 80
        and len(set(groups_train)) >= 10
        and len(set(groups_dev)) >= 5
    )
    if not support_pass:
        return {
            "status": "DIAGNOSTIC_ONLY_INSUFFICIENT_SUPPORT",
            "support": support,
            "confirmation_authorized": False,
        }, {}

    pre, z_train, x_train, y_train_raw, y_train = fit_preprocessor(
        train_rows, targets, synapse_type
    )
    z_dev, x_dev = pre.transform_z(development_rows)
    y_dev_raw, y_dev = pre.transform_y(development_rows, targets, synapse_type)

    response_selection = select_ridge_alpha(
        x_train, y_train, groups_train, feature_kind="quadratic"
    )
    response_alpha = float(response_selection["alpha"])
    response_model = fit_response_model(x_train, y_train, response_alpha)
    kernel_selection = select_kernel_hyperparameters(
        x_train, y_train, groups_train, response_alpha
    )
    if "status" in kernel_selection["chosen"]["variable"]:
        return {
            "status": "NO_VARIABLE_GRAPH_CANDIDATE",
            "support": support,
            "kernel_selection": strip_kernel_selection(kernel_selection),
            "confirmation_authorized": False,
        }, {}

    predictions: dict[str, np.ndarray] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    graph_receipts = {}
    for kind in ("reference", "diagonal", "constant", "variable"):
        chosen = kernel_selection["chosen"][kind]
        if "status" in chosen:
            continue
        prediction, graph, distances, neighbors = final_kernel_prediction(
            x_train,
            y_train,
            x_dev,
            response_model,
            kind,
            int(chosen["k"]),
            float(chosen["bandwidth_multiplier"]),
        )
        name = "metric_" + kind
        predictions[name] = prediction
        oof_predictions[name] = chosen["oof_prediction"]
        graph_receipts[name] = {
            "k": int(chosen["k"]),
            "bandwidth_multiplier": float(chosen["bandwidth_multiplier"]),
            "train_nodes": int(x_train.shape[0]),
            "undirected_edges": int(len(graph.edge_lengths)),
            "median_edge_length": float(graph.edge_scale),
            "query_nodes": int(x_dev.shape[0]),
            "query_neighbor_shape": list(neighbors.shape),
            "finite_query_distances": bool(np.all(np.isfinite(distances))),
        }

    predictions["direct_quadratic"] = response_model.predict(x_dev)
    oof_predictions["direct_quadratic"] = response_selection["oof_prediction"]
    linear_pred, linear_oof, linear_receipt = linear_control(
        x_train, y_train, groups_train, x_dev
    )
    predictions["raw_linear"] = linear_pred
    oof_predictions["raw_linear"] = linear_oof

    subset_receipts = {}
    for name, columns in {
        "strength_only": (0,),
        "distance_only": (1,),
        "membrane_only": (2, 3),
    }.items():
        prediction, oof, receipt = linear_control(
            x_train[:, columns], y_train, groups_train, x_dev[:, columns]
        )
        predictions[name] = prediction
        oof_predictions[name] = oof
        subset_receipts[name] = receipt

    mean_oof = grouped_mean_oof(y_train, groups_train)
    predictions["global_mean"] = np.repeat(
        np.mean(y_train, axis=0)[None, :], len(development_rows), axis=0
    )
    oof_predictions["global_mean"] = mean_oof
    keys_train = [row["category_key"] for row in train_rows]
    keys_dev = [row["category_key"] for row in development_rows]
    predictions["cell_type"] = category_fit_predict(
        y_train, keys_train, keys_dev
    )
    oof_predictions["cell_type"] = category_oof(
        y_train, keys_train, groups_train
    )

    variances = {
        name: predictive_variance(y_train, oof_predictions[name])
        for name in predictions
    }
    controls = sorted(name for name in predictions if name != "metric_variable")
    development_evaluation = evaluate_predictions(
        y_dev,
        groups_dev,
        predictions,
        variances,
        candidate="metric_variable",
        controls=controls,
    )

    rank = bootstrap_rank_audit(
        x_train,
        y_train,
        groups_train,
        response_alpha,
        repetitions=1000,
        seed=SEEDS[synapse_type],
    )
    variable_choice = kernel_selection["chosen"]["variable"]
    try:
        gauge = gauge_audit(
            x_train,
            y_train,
            x_dev,
            response_model,
            k=int(variable_choice["k"]),
            bandwidth_multiplier=float(variable_choice["bandwidth_multiplier"]),
        )
    except Exception as exc:
        gauge = {"status": "FAIL", "exception": f"{type(exc).__name__}: {exc}"}

    metrics = response_model.metrics(x_train)
    eigenvalues = np.linalg.eigvalsh(metrics)
    trace = np.trace(metrics, axis1=1, axis2=2)
    determinant = np.linalg.det(metrics)
    confirmation_authorized = bool(
        rank["status"] == "PASS"
        and gauge["status"] == "PASS"
        and development_evaluation["survival"]
    )
    result = {
        "status": "DEVELOPMENT_PASS" if confirmation_authorized else "DEVELOPMENT_STOP",
        "support": support,
        "preprocessor": {
            "input_coordinates": list(INPUT_COLUMNS),
            "r_reference_V": pre.r_reference,
            "r_reference_mV": pre.r_reference * 1e3,
            "z_mean": pre.z_mean,
            "z_covariance": pre.z_covariance,
            "z_covariance_shrunk": pre.z_covariance_shrunk,
            "z_cholesky": pre.z_cholesky,
            "whitened_reference_metric": "identity",
            "y_mean_raw": pre.y_mean,
            "y_scale_raw": pre.y_scale,
        },
        "physical_numbers": {
            "train": physical_stats(train_rows, targets),
            "development": physical_stats(development_rows, targets),
        },
        "response_map": {
            "basis_count": int(quadratic_features(x_train[:1]).shape[1]),
            "ridge_alpha": response_alpha,
            "ridge_losses": response_selection["losses"],
            "residual_variance_standardized_y": response_model.residual_variance,
            "coefficient": response_model.coefficient,
        },
        "metric_numerics_train": {
            "eigenvalue_min": float(np.min(eigenvalues)),
            "eigenvalue_median_by_axis": np.median(eigenvalues, axis=0),
            "eigenvalue_max": float(np.max(eigenvalues)),
            "trace": summary(trace),
            "determinant": summary(determinant),
        },
        "kernel_selection": strip_kernel_selection(kernel_selection),
        "graphs": graph_receipts,
        "linear_control": linear_receipt,
        "subset_controls": subset_receipts,
        "cell_type_control": {
            "train_categories": len(set(keys_train)),
            "development_unseen_categories": len(set(keys_dev) - set(keys_train)),
        },
        "missingness_control": "IDENTICAL_TO_GLOBAL_MEAN_AFTER_STRICT_COMPLETE_CASE_FILTER",
        "protocol_order_shuffle": "UNAVAILABLE_SMALL_DB_HAS_ZERO_EVENT_ROWS",
        "predictive_variance_standardized_y": variances,
        "development": development_evaluation,
        "rank_audit": rank,
        "gauge_audit": gauge,
        "confirmation_authorized": confirmation_authorized,
    }
    frozen = {
        "synapse_type": synapse_type,
        "response_alpha": response_alpha,
        "kernel_choices": {
            kind: {
                key: value
                for key, value in chosen.items()
                if key in ("k", "bandwidth_multiplier")
            }
            for kind, chosen in kernel_selection["chosen"].items()
            if "status" not in chosen
        },
        "linear_alpha": linear_receipt["alpha"],
        "subset_alpha": {
            name: receipt["alpha"] for name, receipt in subset_receipts.items()
        },
        "best_development_control": development_evaluation["best_control"],
        "predictive_variance": variances,
        "preprocessor": pre,
        "response_model": response_model,
        "train_rows": train_rows,
        "groups_train": groups_train,
        "x_train": x_train,
        "y_train": y_train,
        "oof_predictions": oof_predictions,
    }
    return result, frozen


def confirmation_predictions(
    frozen: dict,
    rows: Sequence[dict],
    targets: dict[int, np.ndarray],
) -> tuple[dict, np.ndarray, list[str]]:
    pre: Preprocessor = frozen["preprocessor"]
    _, x_query = pre.transform_z(rows)
    _, y_query = pre.transform_y(rows, targets, frozen["synapse_type"])
    x_train = frozen["x_train"]
    y_train = frozen["y_train"]
    groups_train = frozen["groups_train"]
    response_model = frozen["response_model"]
    predictions = {}
    for kind, choice in frozen["kernel_choices"].items():
        prediction, _, _, _ = final_kernel_prediction(
            x_train,
            y_train,
            x_query,
            response_model,
            kind,
            int(choice["k"]),
            float(choice["bandwidth_multiplier"]),
        )
        predictions["metric_" + kind] = prediction
    predictions["direct_quadratic"] = response_model.predict(x_query)
    linear_coef = fit_ridge(
        linear_features(x_train), y_train, float(frozen["linear_alpha"])
    )
    predictions["raw_linear"] = linear_features(x_query) @ linear_coef
    for name, columns in {
        "strength_only": (0,),
        "distance_only": (1,),
        "membrane_only": (2, 3),
    }.items():
        coefficient = fit_ridge(
            linear_features(x_train[:, columns]),
            y_train,
            float(frozen["subset_alpha"][name]),
        )
        predictions[name] = linear_features(x_query[:, columns]) @ coefficient
    predictions["global_mean"] = np.repeat(
        np.mean(y_train, axis=0)[None, :], len(rows), axis=0
    )
    train_keys = [row["category_key"] for row in frozen["train_rows"]]
    query_keys = [row["category_key"] for row in rows]
    predictions["cell_type"] = category_fit_predict(y_train, train_keys, query_keys)
    return predictions, y_query, [str(row["slice_id"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    database = args.database.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    database_hash = sha256(database)
    if database_hash != EXPECTED_DATABASE_SHA256:
        raise RuntimeError(f"database SHA-256 mismatch: {database_hash}")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity failure: {integrity}")

    feature_rows = query_feature_rows(connection)
    result = {
        "run": "BA-SRM1",
        "status": "RUNNING",
        "database": str(database),
        "database_bytes": database.stat().st_size,
        "database_sha256": database_hash,
        "sqlite_integrity": integrity,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "formula": {
            "input": "z=(log(|r1|/r_ref),log(L_soma/1m),log(Rin_post/1ohm),log(tau_m_post/1s))",
            "target": "y=(s*a2/r_ref,s*a6:8/r_ref,s*a9:12_250ms/r_ref,v5:8)",
            "response": "quadratic H2 with 15 basis terms",
            "metric": "g_resp=J^T R^-1 J",
            "reference": "g_ref=Sigma_shrunk^-1; represented as I after frozen whitening",
            "graph": "train-only symmetric-union kNN, trapezoid line element, Dijkstra",
        },
        "feature_query": {
            "outcome_columns_touched": False,
            "eligible_input_rows": len(feature_rows),
            "by_stratum_split": {},
        },
        "outcome_access": {
            "train_development_contact": True,
            "confirmation_contact": False,
            "confirmation_strata": [],
        },
        "strata": {},
    }
    for synapse_type in ("ex", "in"):
        rows = [row for row in feature_rows if row["synapse_type"] == synapse_type]
        result["feature_query"]["by_stratum_split"][synapse_type] = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "development", "confirmation")
        }
        opened_rows = [row for row in rows if row["split"] in ("train", "development")]
        targets, target_receipt = query_targets(
            connection, [int(row["pair_id"]) for row in opened_rows]
        )
        stratum_result, frozen = run_stratum_development(
            synapse_type, rows, targets
        )
        stratum_result["train_development_target_receipt"] = target_receipt
        result["strata"][synapse_type] = stratum_result
        if not stratum_result.get("confirmation_authorized", False):
            stratum_result["confirmation"] = {
                "status": "NOT_CONTACTED",
                "reason": "rank, gauge, or development-survival gate did not pass",
            }
            continue

        confirmation_input_rows = [
            row for row in rows if row["split"] == "confirmation"
        ]
        confirmation_targets, receipt = query_targets(
            connection, [int(row["pair_id"]) for row in confirmation_input_rows]
        )
        confirmation_rows = retain_target_complete(
            confirmation_input_rows, confirmation_targets
        )
        result["outcome_access"]["confirmation_contact"] = True
        result["outcome_access"]["confirmation_strata"].append(synapse_type)
        predictions, y_confirmation, groups_confirmation = confirmation_predictions(
            frozen, confirmation_rows, confirmation_targets
        )
        best_control = frozen["best_development_control"]
        confirmation_eval = evaluate_predictions(
            y_confirmation,
            groups_confirmation,
            predictions,
            frozen["predictive_variance"],
            candidate="metric_variable",
            controls=(best_control,),
        )
        confirmation_eval["target_receipt"] = receipt
        confirmation_eval["physical_numbers"] = physical_stats(
            confirmation_rows, confirmation_targets
        )
        confirmation_eval["frozen_control_from_development"] = best_control
        stratum_result["confirmation"] = confirmation_eval

    connection.close()
    statuses = [item.get("status") for item in result["strata"].values()]
    result["status"] = (
        "CONFIRMATION_EXECUTED"
        if result["outcome_access"]["confirmation_contact"]
        else "DEVELOPMENT_STOP_CONFIRMATION_UNTOUCHED"
    )
    result["claim_boundary"] = {
        "conductance": "UNOBSERVED_NOT_FIT",
        "release_probability_Npq": "UNOBSERVED_NOT_FIT",
        "directed_delay": "EXCLUDED_FROM_RIEMANNIAN_CHART",
        "STDP_eligibility_homeostasis": "UNOBSERVED_NOT_FIT",
        "morphology_contact_count": "UNOBSERVED_NOT_FIT",
        "curvature_memory": "REJECTED_NOT_TESTED",
        "CE_core_change": "NONE_DELTA_F_CE_EQUALS_ZERO",
    }
    result["stratum_statuses"] = statuses

    script_path = Path(__file__).resolve()
    library_path = script_path.with_name("srm1_analysis.py")
    result["code_sha256"] = {
        script_path.name: sha256(script_path),
        library_path.name: sha256(library_path),
    }
    output_path = output / "results.json"
    output_path.write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "confirmation_contact": result["outcome_access"]["confirmation_contact"],
        "stratum_statuses": statuses,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
