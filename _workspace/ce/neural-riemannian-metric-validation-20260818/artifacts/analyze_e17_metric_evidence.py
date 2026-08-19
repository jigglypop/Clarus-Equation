"""Run the frozen E17 descriptive reproduction and effective-dynamics feasibility check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon


HORIZON = 5
TRAIN_FRACTION = 0.60
Q_RIDGE_FRACTION = 1e-6
METRIC_RIDGE_LAMBDA = 0.0
MIN_TRANSITIONS_PER_PARAMETER = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def load_data(path: Path) -> dict[str, Any]:
    return {
        key: value
        for key, value in loadmat(path, simplify_cells=True).items()
        if not key.startswith("__")
    }


def spatial_clustering(root: Path) -> dict[str, Any]:
    path = root / "Figure3/FunctionalClustering/Data/SpatialCorr_RuleARuleB.mat"
    data = load_data(path)
    result: dict[str, Any] = {
        "source_file": path.as_posix(),
        "source_sha256": sha256_file(path),
        "distance_scale": 3.0,
        "maximum_distance_um": 20.0,
        "comparisons": {},
    }
    for correlation_type in ("noise", "signal"):
        for rule in ("RuleA", "RuleB"):
            key = f"branch_{correlation_type}_{rule}"
            values = np.asarray(data[key], dtype=float).copy()
            values[:, 0] *= 3.0
            values = values[np.isfinite(values).all(axis=1) & (values[:, 0] <= 20.0)]
            rho, pvalue = spearmanr(values[:, 0], values[:, 1])
            result["comparisons"][key] = {
                "pair_count": int(values.shape[0]),
                "spearman_rho": finite_float(rho),
                "spearman_p": finite_float(pvalue),
                "mean_correlation": finite_float(np.mean(values[:, 1])),
            }
    result["interpretation"] = (
        "Pair-level descriptive reproduction only; pair rows and synapses are not "
        "independent animals and this is not a Riemannian-metric test."
    )
    return result


def pearson_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    correlations = np.full(left.shape[1], np.nan, dtype=float)
    for column in range(left.shape[1]):
        x = left[:, column]
        y = right[:, column]
        if np.std(x) > 0 and np.std(y) > 0:
            correlations[column] = np.corrcoef(x, y)[0, 1]
    return correlations


def drift_reproduction(
    type_sum: list[dict[str, np.ndarray]],
    significance: dict[str, np.ndarray],
    x_df: np.ndarray,
    aligned_day: int,
    tested_days: tuple[int, ...],
) -> dict[str, Any]:
    all_zero = np.zeros(type_sum[0]["sel_types"].shape[1], dtype=bool)
    for day in type_sum:
        all_zero |= np.all(np.asarray(day["sel_types"]) == 0, axis=0)
    keep = ~all_zero

    aligned_selectivity = np.asarray(type_sum[aligned_day - 1]["sel_types"])[:, keep]
    order = np.argsort(np.argmax(aligned_selectivity, axis=0))
    choice_flags = np.asarray(significance["Choice"])[..., keep][:, order]
    coding = np.sum(choice_flags[np.asarray(tested_days) - 1], axis=0) > 0

    start = int(np.flatnonzero(x_df > 0)[0])
    end = int(np.flatnonzero(x_df > 2)[0])
    epoch = np.concatenate(
        [np.arange(start, end + 1), len(x_df) + np.arange(start, end + 1)]
    )
    trajectories: list[np.ndarray] = []
    for day in type_sum:
        type1 = np.asarray(day["nu_type1"], dtype=float)[:, keep][:, order]
        type2 = np.asarray(day["nu_type2"], dtype=float)[:, keep][:, order]
        trajectories.append(np.vstack([type1, type2])[epoch][:, coding])

    pairs = {
        "AA": ((1, 2), (1, 3), (2, 3)),
        "AB": ((1, 4), (4, 3), (2, 4)),
        "AA_prime": ((1, 5), (5, 3), (2, 5)),
    }
    grouped: dict[str, np.ndarray] = {}
    for label, day_pairs in pairs.items():
        matrix = np.vstack(
            [pearson_columns(trajectories[a - 1], trajectories[b - 1]) for a, b in day_pairs]
        )
        grouped[label] = matrix.reshape(-1, order="F")

    combined = np.column_stack([grouped["AA"], grouped["AB"], grouped["AA_prime"]])
    combined = combined[np.isfinite(combined).all(axis=1)]
    if combined.shape[0]:
        friedman = friedmanchisquare(combined[:, 0], combined[:, 1], combined[:, 2])
        pair_tests = {
            "AA_vs_AB": wilcoxon(combined[:, 0], combined[:, 1]),
            "AA_vs_AA_prime": wilcoxon(combined[:, 0], combined[:, 2]),
            "AB_vs_AA_prime": wilcoxon(combined[:, 1], combined[:, 2]),
        }
    else:
        friedman = None
        pair_tests = {}

    summaries = {}
    for column, label in enumerate(("AA", "AB", "AA_prime")):
        values = combined[:, column]
        summaries[label] = {
            "mean": finite_float(np.mean(values)) if values.size else None,
            "median": finite_float(np.median(values)) if values.size else None,
            "sd": finite_float(np.std(values, ddof=1)) if values.size > 1 else None,
        }

    return {
        "tracked_dendrites_before_zero_filter": int(keep.size),
        "tracked_dendrites_after_zero_filter": int(np.sum(keep)),
        "choice_coding_dendrites": int(np.sum(coding)),
        "paired_dendrite_comparison_rows": int(combined.shape[0]),
        "correlation_summary": summaries,
        "released_code_reimplementation_statistics": {
            "friedman_statistic": finite_float(friedman.statistic) if friedman else None,
            "friedman_p": finite_float(friedman.pvalue) if friedman else None,
            "wilcoxon": {
                label: {
                    "statistic": finite_float(test.statistic),
                    "p": finite_float(test.pvalue),
                }
                for label, test in pair_tests.items()
            },
        },
        "inference_boundary": (
            "Dendrite-level p-values reproduce the released code but are not population "
            "evidence because the MAT file has no animal identifier."
        ),
    }


def figure4_analysis(root: Path) -> dict[str, Any]:
    path = root / "Figure4/Data/DataRepDrift_CaImagingDendrites.mat"
    data = load_data(path)
    x_df = np.asarray(data["x_df"], dtype=float)
    return {
        "source_file": path.as_posix(),
        "source_sha256": sha256_file(path),
        "test_A_A_A_B_A": drift_reproduction(
            data["type_sum_test"], data["sig_test"], x_df, 3, (1, 2, 3)
        ),
        "control_A_A_A_A_A": drift_reproduction(
            data["type_sum_ctr"], data["sig_ctr"], x_df, 5, (3, 4, 5)
        ),
    }


def synapse_release_eligibility(root: Path) -> dict[str, Any]:
    path = root / "Figure3/Selectivity/Data/SelectivityStatsGluSNFR.mat"
    data = load_data(path)
    counts = {
        "RuleA": int(np.asarray(data["ConA_Summ"]["selec"]).shape[1]),
        "RuleB": int(np.asarray(data["ConB_Summ"]["selec"]).shape[1]),
        "RuleA_prime": int(np.asarray(data["ConA2_Summ"]["selec"]).shape[1]),
    }
    return {
        "source_file": path.as_posix(),
        "source_sha256": sha256_file(path),
        "synapse_columns": counts,
        "top_level_identity_fields": [],
        "same_synapse_pre_post_map_in_release": False,
        "future_single_trial_trajectories_in_release": False,
        "eligibility": "FAIL_TIER_A",
        "reason": (
            "The three rules contain unequal condition-mean synapse columns and no "
            "synapse-ID map or single-trial future trajectory."
        ),
    }


def trial_arrays(condition: dict[str, Any]) -> list[np.ndarray]:
    arrays = [np.asarray(trial, dtype=float) for trial in condition["branch"]]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("branch trials must be nonempty time-by-ROI matrices")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("branch trial shapes change within a condition")
    return arrays


def chart_trials(
    saline: list[np.ndarray], dcz: list[np.ndarray]
) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    split_sal = int(math.floor(TRAIN_FRACTION * len(saline)))
    split_dcz = int(math.floor(TRAIN_FRACTION * len(dcz)))
    if split_sal < 1 or split_dcz < 1 or split_sal == len(saline) or split_dcz == len(dcz):
        raise ValueError("condition has too few trials for the fixed split")

    calibration = np.vstack(saline[:split_sal] + dcz[:split_dcz])
    mean = np.nanmean(calibration, axis=0)
    scale = np.nanstd(calibration, axis=0)
    valid = np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-8)
    if not np.any(valid):
        raise ValueError("no variable ROI survives the frozen chart rule")

    def transform(items: list[np.ndarray]) -> list[np.ndarray]:
        return [(item[:, valid] - mean[valid]) / scale[valid] for item in items]

    return (
        {
            "sal_train": transform(saline[:split_sal]),
            "sal_test": transform(saline[split_sal:]),
            "dcz_train": transform(dcz[:split_dcz]),
            "dcz_test": transform(dcz[split_dcz:]),
        },
        {
            "saline_trials": len(saline),
            "dcz_trials": len(dcz),
            "saline_train_trials": split_sal,
            "dcz_train_trials": split_dcz,
            "retained_rois": int(np.sum(valid)),
            "original_rois": int(valid.size),
        },
    )


def transition_pairs(trials: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for trial in trials:
        left.append(trial[:-horizon])
        right.append(trial[horizon:])
    x = np.vstack(left)
    y = np.vstack(right)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[finite], y[finite]


def fit_dynamics(trials: list[np.ndarray]) -> dict[str, np.ndarray | float | int]:
    x, y = transition_pairs(trials, 1)
    dimension = x.shape[1]
    if x.shape[0] < MIN_TRANSITIONS_PER_PARAMETER * (dimension + 1):
        raise ValueError("too few calibration transitions for the frozen rule")
    design = np.column_stack([x, np.ones(x.shape[0])])
    coefficient, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    j = coefficient[:-1].T
    bias = coefficient[-1]
    residual = y - design @ coefficient
    q = residual.T @ residual / residual.shape[0]
    q_scale = float(np.trace(q) / dimension)
    q = q + max(Q_RIDGE_FRACTION * q_scale, np.finfo(float).eps) * np.eye(dimension)

    j_power = np.eye(dimension)
    covariance = np.zeros((dimension, dimension))
    bias_h = np.zeros(dimension)
    for _ in range(HORIZON):
        covariance += j_power @ q @ j_power.T
        bias_h += j_power @ bias
        j_power = j_power @ j
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    metric = np.linalg.inv(covariance)
    np.linalg.cholesky(metric)
    return {
        "j": j,
        "bias": bias,
        "q": q,
        "j_h": j_power,
        "bias_h": bias_h,
        "c_h": covariance,
        "g_h": metric,
        "calibration_transitions": int(x.shape[0]),
    }


def gaussian_nlpd(residual: np.ndarray, covariance: np.ndarray) -> float:
    chol = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(chol, residual.T)
    quadratic = np.sum(standardized * standardized, axis=0)
    logdet = 2 * np.sum(np.log(np.diag(chol)))
    dimension = covariance.shape[0]
    return float(np.mean(0.5 * (dimension * np.log(2 * np.pi) + logdet + quadratic)))


def model_residual(
    model: dict[str, np.ndarray | float | int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    return y - (x @ np.asarray(model["j_h"]).T + np.asarray(model["bias_h"]))


def evaluate_model(
    own: dict[str, np.ndarray | float | int],
    other: dict[str, np.ndarray | float | int],
    train_trials: list[np.ndarray],
    test_trials: list[np.ndarray],
) -> dict[str, Any]:
    x, y = transition_pairs(test_trials, HORIZON)
    own_cov = np.asarray(own["c_h"])
    own_residual = model_residual(own, x, y)
    other_residual = model_residual(other, x, y)
    other_cov = np.asarray(other["c_h"])
    diagonal = np.diag(np.diag(own_cov))
    isotropic = np.eye(own_cov.shape[0]) * np.trace(own_cov) / own_cov.shape[0]

    train_x, train_y = transition_pairs(train_trials, HORIZON)
    persistence_variance = float(np.mean((train_y - train_x) ** 2))
    persistence_covariance = np.eye(own_cov.shape[0]) * max(
        persistence_variance, np.finfo(float).eps
    )
    empirical_covariance = own_residual.T @ own_residual / own_residual.shape[0]
    empirical_scale = float(np.trace(empirical_covariance) / own_cov.shape[0])
    empirical_covariance += max(
        Q_RIDGE_FRACTION * empirical_scale, np.finfo(float).eps
    ) * np.eye(own_cov.shape[0])

    full = gaussian_nlpd(own_residual, own_cov)
    wrong = gaussian_nlpd(other_residual, other_cov)
    diagonal_score = gaussian_nlpd(own_residual, diagonal)
    isotropic_score = gaussian_nlpd(own_residual, isotropic)
    persistence_score = gaussian_nlpd(y - x, persistence_covariance)
    return {
        "test_horizon_pairs": int(x.shape[0]),
        "own_full_nlpd": full,
        "wrong_condition_nlpd": wrong,
        "diagonal_nlpd": diagonal_score,
        "isotropic_nlpd": isotropic_score,
        "persistence_nlpd": persistence_score,
        "own_advantage_over_wrong": wrong - full,
        "full_advantage_over_diagonal": diagonal_score - full,
        "full_advantage_over_isotropic": isotropic_score - full,
        "full_advantage_over_persistence": persistence_score - full,
        "empirical_residual_covariance": empirical_covariance,
    }


def logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("expected positive determinant")
    return float(value)


def affine_spd_distance(left: np.ndarray, right: np.ndarray) -> float:
    chol = np.linalg.cholesky(left)
    intermediate = np.linalg.solve(chol, right)
    whitened = np.linalg.solve(chol, intermediate.T).T
    whitened = (whitened + whitened.T) / 2
    eigenvalues = np.linalg.eigvalsh(whitened)
    if np.any(eigenvalues <= 0):
        raise ValueError("generalized covariance eigenvalue is not positive")
    return float(np.linalg.norm(np.log(eigenvalues)))


def session_feasibility(path: Path) -> dict[str, Any]:
    data = load_data(path)["cont_data"]
    saline = trial_arrays(data["Sal"])
    dcz = trial_arrays(data["DCZ"])
    charted, counts = chart_trials(saline, dcz)
    sal_model = fit_dynamics(charted["sal_train"])
    dcz_model = fit_dynamics(charted["dcz_train"])
    sal_eval = evaluate_model(
        sal_model, dcz_model, charted["sal_train"], charted["sal_test"]
    )
    dcz_eval = evaluate_model(
        dcz_model, sal_model, charted["dcz_train"], charted["dcz_test"]
    )

    sal_cov = np.asarray(sal_model["c_h"])
    dcz_cov = np.asarray(dcz_model["c_h"])
    sal_metric = np.asarray(sal_model["g_h"])
    dcz_metric = np.asarray(dcz_model["g_h"])
    predicted_change = logdet(dcz_cov) - logdet(sal_cov)
    observed_change = logdet(np.asarray(dcz_eval["empirical_residual_covariance"])) - logdet(
        np.asarray(sal_eval["empirical_residual_covariance"])
    )

    def public_scores(evaluation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in evaluation.items()
            if key != "empirical_residual_covariance"
        }

    return {
        "session": path.stem,
        "animal": re.match(r"(DCO\d+)", path.stem).group(1),
        "source_sha256": sha256_file(path),
        **counts,
        "saline_calibration_transitions": sal_model["calibration_transitions"],
        "dcz_calibration_transitions": dcz_model["calibration_transitions"],
        "saline_test": public_scores(sal_eval),
        "dcz_test": public_scores(dcz_eval),
        "metric_affine_spd_distance": affine_spd_distance(sal_metric, dcz_metric),
        "predicted_logdet_change_dcz_minus_saline": predicted_change,
        "observed_logdet_change_dcz_minus_saline": observed_change,
        "logdet_change_sign_match": bool(np.sign(predicted_change) == np.sign(observed_change)),
    }


def aggregate_animals(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[session["animal"]].append(session)

    paths = (
        ("saline_own_advantage_over_wrong", "saline_test", "own_advantage_over_wrong"),
        ("dcz_own_advantage_over_wrong", "dcz_test", "own_advantage_over_wrong"),
        ("saline_full_advantage_over_isotropic", "saline_test", "full_advantage_over_isotropic"),
        ("dcz_full_advantage_over_isotropic", "dcz_test", "full_advantage_over_isotropic"),
        ("metric_affine_spd_distance", None, "metric_affine_spd_distance"),
    )
    animals: dict[str, Any] = {}
    for animal, items in sorted(grouped.items()):
        summary: dict[str, Any] = {"session_count": len(items)}
        for output_key, section, metric in paths:
            values = [item[section][metric] if section else item[metric] for item in items]
            summary[output_key] = float(np.mean(values))
        summary["logdet_sign_match_fraction"] = float(
            np.mean([item["logdet_change_sign_match"] for item in items])
        )
        summary["mean_predicted_logdet_change"] = float(
            np.mean([item["predicted_logdet_change_dcz_minus_saline"] for item in items])
        )
        summary["mean_observed_logdet_change"] = float(
            np.mean([item["observed_logdet_change_dcz_minus_saline"] for item in items])
        )
        summary["animal_mean_logdet_sign_match"] = bool(
            np.sign(summary["mean_predicted_logdet_change"])
            == np.sign(summary["mean_observed_logdet_change"])
        )
        animals[animal] = summary
    return {
        "independent_animal_count": len(animals),
        "animals": animals,
        "population_inference": "NOT_RUN_N_EQUALS_3",
    }


def figure2_feasibility(root: Path) -> dict[str, Any]:
    files = sorted((root / "Figure2/Data").glob("DCO*_dff.mat"))
    sessions = []
    failures = []
    for path in files:
        try:
            sessions.append(session_feasibility(path))
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append({"session": path.stem, "error": str(error)})
    direction_summary = {
        "logdet_change_sign_matches": int(
            sum(session["logdet_change_sign_match"] for session in sessions)
        ),
        "saline_own_model_better_than_wrong": int(
            sum(session["saline_test"]["own_advantage_over_wrong"] > 0 for session in sessions)
        ),
        "dcz_own_model_better_than_wrong": int(
            sum(session["dcz_test"]["own_advantage_over_wrong"] > 0 for session in sessions)
        ),
        "saline_full_covariance_better_than_isotropic": int(
            sum(
                session["saline_test"]["full_advantage_over_isotropic"] > 0
                for session in sessions
            )
        ),
        "dcz_full_covariance_better_than_isotropic": int(
            sum(
                session["dcz_test"]["full_advantage_over_isotropic"] > 0
                for session in sessions
            )
        ),
        "denominator_sessions": len(sessions),
    }
    return {
        "protocol": {
            "train_fraction": TRAIN_FRACTION,
            "horizon_frames": HORIZON,
            "q_ridge_fraction": Q_RIDGE_FRACTION,
            "metric_ridge_lambda": METRIC_RIDGE_LAMBDA,
            "q_ridge_role": "process-noise estimator stabilization, not metric regularization",
            "minimum_transitions_per_parameter": MIN_TRANSITIONS_PER_PARAMETER,
            "split": "first 60% versus last 40% in released array order",
            "temporal_order_provenance": "UNVERIFIED",
            "chart_invariance": "NOT_CLAIMED; fixed chart and chart-dependent isotropic Q ridge",
        },
        "session_count": len(files),
        "successful_sessions": len(sessions),
        "failed_sessions": failures,
        "direction_summary": direction_summary,
        "direction_summary_unit_warning": (
            "The 11 sessions are repeated measurements nested in 3 animals, and overlapping "
            "H-step pairs are not independent corroborations."
        ),
        "sessions": sessions,
        "animal_summary": aggregate_animals(sessions),
        "inference_boundary": (
            "Exploratory effective-dynamics/H4 feasibility. It contains no direct W^s "
            "change and cannot establish H1A, H2, or causal metric mediation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("acquisition_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    acquisition = json.loads(args.acquisition_manifest.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "dataset": "NRM-E17",
        "archive_sha256": acquisition["archive"]["sha256"],
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "eligibility": {
            "NRM-H1A": "UNTESTABLE",
            "NRM-H1B": "EXPLORATORY_FEASIBILITY_ONLY",
            "NRM-H2": "UNTESTABLE_BY_CONSTRUCTION",
            "NRM-H4": "EXPLORATORY_FEASIBILITY_ONLY",
            "reason": (
                "Synaptic input, longitudinal dendrites, and NDNF manipulation are "
                "released in separate figure datasets without one same-unit chain."
            ),
        },
        "figure3_synapse_release": synapse_release_eligibility(root),
        "figure3_spatial_clustering": spatial_clustering(root),
        "figure4_longitudinal_drift": figure4_analysis(root),
        "figure2_effective_dynamics": figure2_feasibility(root),
        "overall_verdict": (
            "E17 supports descriptive rule-dependent synaptic clustering, same-dendrite "
            "representational change, and an exploratory gain/inhibition dynamics test, "
            "but it does not test Delta W^s -> Delta g -> future Delta x in the same units."
        ),
    }
    args.output.resolve().write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "H1A": result["eligibility"]["NRM-H1A"],
                "figure2_sessions": result["figure2_effective_dynamics"]["successful_sessions"],
                "independent_animals": result["figure2_effective_dynamics"]["animal_summary"][
                    "independent_animal_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
