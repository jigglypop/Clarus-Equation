"""Second-stage, fail-closed improvement loops for the rejected-claim audit.

This executable does not promote a rejected CE claim merely because an added
parameter or an external benchmark fits.  It records the strongest supported
descendant of each attempted repair, propagates fitted-parameter uncertainty
into held-out tests, and keeps all theory-level promotion gates explicit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import rejection_loop_engineering as base


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FORWARD_MODEL_PATH = REPO_ROOT / "examples" / "physics" / "ce_residual_forward_model.py"

EXPECTED_BASE_SOURCE_SHA256 = base.EXPECTED_SOURCE_SHA256
EXPECTED_BASE_SEMANTIC_SHA256 = base.EXPECTED_SEMANTIC_MANIFEST_SHA256
EXPECTED_BASE_REGRESSION_WITNESS_SHA256 = (
    base.EXPECTED_REGRESSION_WITNESS_REGISTRY_SHA256
)
EXPECTED_EMBEDDED_DESI_SHA256 = (
    "e8f24d5ef0ce808f9c1f67d52fead2eede5cf9265e31aed41db94ecb30c324f5"
)
DESI_UPSTREAM_COMMIT = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
DESI_UPSTREAM_PINS = {
    "repository_commit": DESI_UPSTREAM_COMMIT,
    "mean_url": (
        "https://raw.githubusercontent.com/CobayaSampler/bao_data/"
        f"{DESI_UPSTREAM_COMMIT}/"
        "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt"
    ),
    "mean_bytes": 472,
    "mean_sha256": "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585",
    "covariance_url": (
        "https://raw.githubusercontent.com/CobayaSampler/bao_data/"
        f"{DESI_UPSTREAM_COMMIT}/"
        "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt"
    ),
    "covariance_bytes": 2547,
    "covariance_sha256": "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
}
PORTAL_ARTIFACT_PINS = {
    "paper_url": "https://arxiv.org/pdf/2410.21089v2",
    "paper_bytes": 3668031,
    "paper_sha256": "dc31c67d61457679c4a642dba42377183b96dbdfe2dd04cb9b61dd284f98a145",
    "figure_image_sha256": "4c2a4fa670e92cef0c208e90d49b666a146583090904e541b2e439aa21b4aace",
    "figure_image_size": (10615, 4507),
    "lz_hepdata_url": "https://www.hepdata.net/record/158592?format=json",
    "lz_hepdata_doi": "10.17182/hepdata.155182.v2/t1",
}
PORTAL_FIGURE_CALIBRATION = {
    "panel_bounds_pixels": {"x_left": 830, "x_right": 5198, "y_top": 171, "y_bottom": 3867},
    "x_axis_u_limits": (0.0, 7.0),
    "candidate_mass_GeV": 62.0,
    "higgs_mass_GeV": 125.0,
    "candidate_column": 2138,
    "column_tolerance_pixels": 0.5,
}

ALLOWED_STATUSES = {
    "REJECT",
    "AUDIT_PASS",
    "SELECTION_NEGATIVE_CONTROL",
    "BRIDGE_PASS",
    "EXTERNAL_FIGURE_BRIDGE_PASS",
    "NOT_REACHED",
}


@dataclass(frozen=True)
class ImprovementIteration:
    index: int
    candidate: str
    status: str
    mutation: str
    added_inputs: tuple[str, ...]
    gate: str
    metrics: dict[str, Any]
    limitation: str


@dataclass(frozen=True)
class ImprovementBranch:
    branch_id: str
    source_loop_id: str
    parent_claim_still_rejected: bool
    iterations: tuple[ImprovementIteration, ...]
    maximum_supported_stage: str
    original_claim_promoted: bool
    ce_specific_physical_claim_closed: bool
    next_required_gate: str


@dataclass(frozen=True)
class BAOHoldoutFold:
    fold_id: str
    train_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    q_train: float
    q_variance: float
    train_fixed_chi2: float
    train_fitted_chi2: float
    train_fitted_dof: int
    train_fitted_p_value: float
    holdout_fixed_chi2: float
    holdout_fixed_p_value: float
    holdout_plugin_chi2: float
    holdout_plugin_p_value: float
    holdout_predictive_chi2: float
    holdout_predictive_p_value: float
    predictive_score_improvement: float


@dataclass(frozen=True)
class SelfCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ImprovementReport:
    schema_version: str
    base_source_sha256: str
    base_semantic_sha256: str
    base_rejected_occurrences: int
    base_regression_witnesses: int
    base_regression_witness_registry_sha256: str
    base_original_claims_promoted: int
    source_pins: dict[str, Any]
    branches: tuple[ImprovementBranch, ...]
    evidence: dict[str, Any]
    original_claims_promoted: int
    ce_specific_physical_claims_closed: int


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def dot(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if not left or not right:
        return []
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(len(right)))
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def submatrix(
    matrix: list[list[float]],
    row_indices: tuple[int, ...],
    column_indices: tuple[int, ...],
) -> list[list[float]]:
    return [[matrix[i][j] for j in column_indices] for i in row_indices]


def add_rank_one(
    matrix: list[list[float]],
    scale: float,
    vector: list[float],
) -> list[list[float]]:
    return [
        [matrix[i][j] + scale * vector[i] * vector[j] for j in range(len(vector))]
        for i in range(len(vector))
    ]


def log_determinant_positive(matrix: list[list[float]]) -> float:
    """Cholesky log determinant for a symmetric positive-definite matrix."""

    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            residual = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual <= 0.0:
                    raise ValueError("matrix is not positive definite")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    return 2.0 * sum(math.log(lower[i][i]) for i in range(size))


@lru_cache(maxsize=1)
def load_forward_model() -> Any:
    if not FORWARD_MODEL_PATH.exists():
        raise FileNotFoundError(FORWARD_MODEL_PATH)
    specification = importlib.util.spec_from_file_location(
        "ce_residual_forward_model_improvement_audit",
        FORWARD_MODEL_PATH,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {FORWARD_MODEL_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def embedded_desi_digest(dataset: Any) -> str:
    payload = {
        "data": [[point.z, point.kind, point.value] for point in dataset.data],
        "covariance": [list(row) for row in dataset.covariance],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_true_runs(flags: list[bool], offset: int) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        coordinate = offset + index
        if flag and start is None:
            start = coordinate
        elif not flag and start is not None:
            runs.append((start, coordinate - 1))
            start = None
    if start is not None:
        runs.append((start, offset + len(flags) - 1))
    return tuple(runs)


def portal_figure_live_checks(path: Path) -> tuple[SelfCheck, ...]:
    """Verify the pinned raster and reproduce the four Fig. 3 curve reads."""

    try:
        from PIL import Image
    except ImportError:
        return (
            SelfCheck(
                "portal-figure-live-verification",
                False,
                "Pillow is required only for optional live raster verification",
            ),
        )
    if not path.is_file():
        return (
            SelfCheck(
                "portal-figure-live-verification",
                False,
                f"missing portal figure image: {path}",
            ),
        )
    image = Image.open(path).convert("RGB")
    image_hash = file_sha256(path)
    integrity_passed = (
        image.size == PORTAL_ARTIFACT_PINS["figure_image_size"]
        and image_hash == PORTAL_ARTIFACT_PINS["figure_image_sha256"]
    )
    calibration = PORTAL_FIGURE_CALIBRATION
    x_left = calibration["panel_bounds_pixels"]["x_left"]
    x_right = calibration["panel_bounds_pixels"]["x_right"]
    u_min, u_max = calibration["x_axis_u_limits"]
    delta = (
        2.0 * calibration["candidate_mass_GeV"] - calibration["higgs_mass_GeV"]
    ) / calibration["higgs_mass_GeV"]
    coordinate_u = -math.log10(-delta)
    column_float = x_left + (coordinate_u - u_min) * (x_right - x_left) / (u_max - u_min)
    column = round(column_float)
    x_calibration_passed = (
        column == calibration["candidate_column"]
        and abs(column - column_float) <= calibration["column_tolerance_pixels"]
    )
    specifications = {
        "cyan": (
            1500,
            1800,
            lambda rgb: rgb[0] < 100 and rgb[1] > 220 and rgb[2] > 220,
            (1658, 1671),
            2.4191233235598564e-4,
        ),
        "lz": (
            1000,
            1800,
            lambda rgb: max(rgb) < 30,
            (1392, 1403),
            4.705643778724436e-4,
        ),
        "fermi": (
            200,
            1000,
            lambda rgb: rgb[0] < 100 and rgb[1] > 140 and rgb[2] < 120,
            (437, 459),
            5.014370851167121e-3,
        ),
        "dampe": (
            200,
            1000,
            lambda rgb: rgb[0] < 80 and rgb[1] < 120 and rgb[2] > 150,
            (350, 371),
            6.236105013365809e-3,
        ),
    }
    curve_details: list[str] = []
    curves_passed = True
    y_top = calibration["panel_bounds_pixels"]["y_top"]
    y_bottom = calibration["panel_bounds_pixels"]["y_bottom"]
    for name, (lower, upper, predicate, expected_run, expected_value) in specifications.items():
        flags = [predicate(image.getpixel((column, y))) for y in range(lower, upper + 1)]
        runs = contiguous_true_runs(flags, lower)
        if expected_run not in runs:
            curves_passed = False
            curve_details.append(f"{name}: expected {expected_run}, got {runs}")
            continue
        midpoint = 0.5 * (expected_run[0] + expected_run[1])
        value = 10.0 ** (-2.0 - 4.0 * (midpoint - y_top) / (y_bottom - y_top))
        value_passed = abs(math.log10(value / expected_value)) <= 0.01
        curves_passed = curves_passed and value_passed
        curve_details.append(f"{name}={value:.12e}, run={expected_run}")
    return (
        SelfCheck(
            "portal-figure-live-integrity",
            integrity_passed,
            f"size={image.size}, sha256={image_hash}",
        ),
        SelfCheck(
            "portal-figure-live-x-calibration",
            integrity_passed and x_calibration_passed,
            (
                f"u={coordinate_u:.12f}, axis=[{u_min:.1f},{u_max:.1f}], "
                f"column_float={column_float:.6f}, column={column}, "
                f"residual={abs(column - column_float):.6f}px"
            ),
        ),
        SelfCheck(
            "portal-figure-live-digitization",
            integrity_passed and x_calibration_passed and curves_passed,
            "; ".join(curve_details),
        ),
    )


def pinned_file_check(
    name: str,
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> SelfCheck:
    if not path.is_file():
        return SelfCheck(name, False, f"missing artifact: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = file_sha256(path)
    return SelfCheck(
        name,
        actual_bytes == expected_bytes and actual_sha256 == expected_sha256,
        f"bytes={actual_bytes}, sha256={actual_sha256}",
    )


def covariance_redshift_blocks(dataset: Any) -> tuple[tuple[int, ...], ...]:
    """Keep equal-z observations and every nonzero covariance edge together."""

    count = len(dataset.data)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(count):
        for j in range(i + 1, count):
            if (
                dataset.data[i].z == dataset.data[j].z
                or abs(dataset.covariance[i][j]) > 0.0
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return tuple(
        tuple(group)
        for group in sorted(groups.values(), key=lambda item: dataset.data[item[0]].z)
    )


def invert(module: Any, matrix: list[list[float]]) -> list[list[float]]:
    result = module.invert_matrix(tuple(tuple(value for value in row) for row in matrix))
    return [list(row) for row in result]


def conditional_scale_fold(
    module: Any,
    observations: list[float],
    predictions: list[float],
    covariance: list[list[float]],
    train_indices: tuple[int, ...],
    holdout_indices: tuple[int, ...],
    fold_id: str,
) -> BAOHoldoutFold:
    c_tt = submatrix(covariance, train_indices, train_indices)
    c_hh = submatrix(covariance, holdout_indices, holdout_indices)
    c_ht = submatrix(covariance, holdout_indices, train_indices)
    c_th = submatrix(covariance, train_indices, holdout_indices)
    inv_tt = invert(module, c_tt)

    y_t = [predictions[i] for i in train_indices]
    d_t = [observations[i] for i in train_indices]
    y_h = [predictions[i] for i in holdout_indices]
    d_h = [observations[i] for i in holdout_indices]
    precision_y_t = matvec(inv_tt, y_t)
    q_variance = 1.0 / dot(y_t, precision_y_t)
    q_train = q_variance * dot(y_t, matvec(inv_tt, d_t))

    c_ht_inv_tt = matmul(c_ht, inv_tt)
    conditional_covariance_product = matmul(c_ht_inv_tt, c_th)
    conditional_covariance = [
        [
            c_hh[i][j] - conditional_covariance_product[i][j]
            for j in range(len(holdout_indices))
        ]
        for i in range(len(holdout_indices))
    ]
    inv_conditional = invert(module, conditional_covariance)

    train_fixed_residual = [predictions[i] - observations[i] for i in train_indices]
    train_fitted_residual = [
        q_train * predictions[i] - observations[i] for i in train_indices
    ]
    conditional_design = [
        y_h[i] - matvec(c_ht_inv_tt, y_t)[i]
        for i in range(len(holdout_indices))
    ]

    def conditional_residual(q_value: float) -> list[float]:
        residual_t = [q_value * y_t[i] - d_t[i] for i in range(len(train_indices))]
        conditional_shift = matvec(c_ht_inv_tt, residual_t)
        return [
            q_value * y_h[i] - d_h[i] - conditional_shift[i]
            for i in range(len(holdout_indices))
        ]

    fixed_residual = conditional_residual(1.0)
    plugin_residual = conditional_residual(q_train)
    predictive_covariance = add_rank_one(
        conditional_covariance,
        q_variance,
        conditional_design,
    )
    inv_predictive = invert(module, predictive_covariance)

    train_fixed_chi2 = dot(train_fixed_residual, matvec(inv_tt, train_fixed_residual))
    train_fitted_chi2 = dot(train_fitted_residual, matvec(inv_tt, train_fitted_residual))
    holdout_fixed_chi2 = dot(fixed_residual, matvec(inv_conditional, fixed_residual))
    holdout_plugin_chi2 = dot(plugin_residual, matvec(inv_conditional, plugin_residual))
    holdout_predictive_chi2 = dot(
        plugin_residual,
        matvec(inv_predictive, plugin_residual),
    )
    holdout_dof = len(holdout_indices)
    train_fitted_dof = len(train_indices) - 1
    score_improvement = (
        holdout_fixed_chi2
        + log_determinant_positive(conditional_covariance)
        - holdout_predictive_chi2
        - log_determinant_positive(predictive_covariance)
    )
    return BAOHoldoutFold(
        fold_id=fold_id,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        q_train=q_train,
        q_variance=q_variance,
        train_fixed_chi2=train_fixed_chi2,
        train_fitted_chi2=train_fitted_chi2,
        train_fitted_dof=train_fitted_dof,
        train_fitted_p_value=module.chi_square_survival(
            train_fitted_chi2,
            train_fitted_dof,
        ),
        holdout_fixed_chi2=holdout_fixed_chi2,
        holdout_fixed_p_value=module.chi_square_survival(
            holdout_fixed_chi2,
            holdout_dof,
        ),
        holdout_plugin_chi2=holdout_plugin_chi2,
        holdout_plugin_p_value=module.chi_square_survival(
            holdout_plugin_chi2,
            holdout_dof,
        ),
        holdout_predictive_chi2=holdout_predictive_chi2,
        holdout_predictive_p_value=module.chi_square_survival(
            holdout_predictive_chi2,
            holdout_dof,
        ),
        predictive_score_improvement=score_improvement,
    )


@lru_cache(maxsize=1)
def bao_evidence() -> dict[str, Any]:
    module = load_forward_model()
    dataset = module.named_bao_dataset("desi-dr2-all")
    fixed = module.assess_bao_fit(
        dataset.data,
        module.CEForwardParams(),
        covariance=dataset.covariance,
    )
    if fixed.scale_fit_diagnostic is None:
        raise AssertionError("global scale diagnostic is unavailable")
    observations = [point.value for point in dataset.data]
    predictions = [item.predicted for item in fixed.contributions]
    covariance = [list(row) for row in dataset.covariance]
    covariance_inverse = invert(module, covariance)
    global_q_variance = 1.0 / dot(
        predictions,
        matvec(covariance_inverse, predictions),
    )
    blocks = covariance_redshift_blocks(dataset)

    even_train = tuple(index for rank, block in enumerate(blocks) if rank % 2 == 0 for index in block)
    odd_train = tuple(index for rank, block in enumerate(blocks) if rank % 2 == 1 for index in block)
    two_fold = (
        conditional_scale_fold(
            module,
            observations,
            predictions,
            covariance,
            even_train,
            odd_train,
            "parity-A-even-blocks-train",
        ),
        conditional_scale_fold(
            module,
            observations,
            predictions,
            covariance,
            odd_train,
            even_train,
            "parity-B-odd-blocks-train",
        ),
    )
    all_indices = tuple(range(len(observations)))
    lobo = tuple(
        conditional_scale_fold(
            module,
            observations,
            predictions,
            covariance,
            tuple(index for index in all_indices if index not in block),
            block,
            f"hold-z-{dataset.data[block[0]].z:.3f}",
        )
        for block in blocks
    )

    scale = fixed.scale_fit_diagnostic
    n_observations = len(observations)
    scale_aicc = scale.aic + 2.0 * 1.0 * 2.0 / (n_observations - 1.0 - 1.0)
    return {
        "dataset_name": dataset.name,
        "embedded_digest": embedded_desi_digest(dataset),
        "blocks": blocks,
        "fixed": {
            "chi2": fixed.chi2,
            "dof": fixed.dof,
            "p_value": fixed.survival_p_value,
            "verdict": fixed.verdict,
            "aic": fixed.aic,
            "aicc": fixed.aic,
            "bic": fixed.bic,
        },
        "global_same_data_scale": {
            "q": scale.scale_factor,
            "q_variance": global_q_variance,
            "chi2": scale.chi2,
            "dof": scale.dof,
            "p_value": scale.survival_p_value,
            "aic": scale.aic,
            "aicc": scale_aicc,
            "bic": scale.bic,
            "aic_improvement": fixed.aic - scale.aic,
            "aicc_improvement": fixed.aic - scale_aicc,
            "bic_improvement": fixed.bic - scale.bic,
            "equivalent_rd_Mpc": scale.equivalent_rd_mpc_at_fixed_h0,
            "equivalent_H0_km_s_Mpc": scale.equivalent_h0_at_fixed_rd,
        },
        "two_fold": two_fold,
        "two_fold_totals": {
            "fixed_chi2": sum(fold.holdout_fixed_chi2 for fold in two_fold),
            "plugin_chi2": sum(fold.holdout_plugin_chi2 for fold in two_fold),
            "predictive_chi2": sum(fold.holdout_predictive_chi2 for fold in two_fold),
            "predictive_score_improvement": sum(
                fold.predictive_score_improvement for fold in two_fold
            ),
            "q_difference_sigma": abs(two_fold[0].q_train - two_fold[1].q_train)
            / math.sqrt(two_fold[0].q_variance + two_fold[1].q_variance),
        },
        "leave_one_block_out": lobo,
        "leave_one_block_out_totals": {
            "plugin_chi2": sum(fold.holdout_plugin_chi2 for fold in lobo),
            "predictive_chi2": sum(fold.holdout_predictive_chi2 for fold in lobo),
            "minimum_predictive_p_value": min(
                fold.holdout_predictive_p_value for fold in lobo
            ),
            "positive_score_folds": sum(
                fold.predictive_score_improvement > 0.0 for fold in lobo
            ),
            "predictive_score_improvement": sum(
                fold.predictive_score_improvement for fold in lobo
            ),
        },
        "external_nonoverlapping_release_tested": False,
        "aggregate_cv_p_value_reported": False,
    }


def portal_evidence() -> dict[str, Any]:
    mass_gev = 62.0
    arbitrary_coupling = 3.0e-4
    coupling = 2.4191233235598564e-4
    relic_interval = (2.380254399901916e-4, 2.458626966442090e-4)
    lz_figure_limit = 4.705643778724436e-4
    fermi_figure_limit = 5.014370851167121e-3
    dampe_figure_limit = 6.236105013365809e-3
    higgs_mass = 125.0
    vev = 246.0
    visible_width_gev = 0.00407
    nucleon_mass = 0.938
    nucleon_form_factor = 1.0 / 3.0
    gev_minus_two_to_cm2 = 0.389379e-27
    phase = math.sqrt(1.0 - 4.0 * mass_gev**2 / higgs_mass**2)
    invisible_width = (
        coupling**2 * vev**2 * phase / (8.0 * math.pi * higgs_mass)
    )
    invisible_branching = invisible_width / (visible_width_gev + invisible_width)
    bare_mass = math.sqrt(mass_gev**2 - coupling * vev**2)
    sigma_si = (
        coupling**2
        * nucleon_mass**4
        * nucleon_form_factor**2
        / (math.pi * mass_gev**2 * higgs_mass**4)
        * gev_minus_two_to_cm2
    )
    lz_low_mass, lz_low_limit = 46.0, 2.2681993499922688e-48
    lz_high_mass, lz_high_limit = 65.0, 2.546041480288214e-48
    interpolation_fraction = math.log(mass_gev / lz_low_mass) / math.log(
        lz_high_mass / lz_low_mass
    )
    lz_v2_interpolated_limit = math.exp(
        math.log(lz_low_limit)
        + interpolation_fraction * math.log(lz_high_limit / lz_low_limit)
    )
    delta = (2.0 * mass_gev - higgs_mass) / higgs_mass
    coordinate = -math.log10(-delta)
    calibration = PORTAL_FIGURE_CALIBRATION
    x_left = calibration["panel_bounds_pixels"]["x_left"]
    x_right = calibration["panel_bounds_pixels"]["x_right"]
    u_min, u_max = calibration["x_axis_u_limits"]
    expected_column_float = x_left + (coordinate - u_min) * (x_right - x_left) / (u_max - u_min)
    return {
        "normalization": {
            "paper_operator": "-a2*S^2*Hdagger*H",
            "manuscript_operator": "-lambda_HP*phi^2*Hdagger*H",
            "lambda_HP_equals_a2": True,
        },
        "figure_digitization": {
            "page_zero_based": 5,
            "image_index": 0,
            "image_size": (10615, 4507),
            "panel_bounds_pixels": calibration["panel_bounds_pixels"],
            "x_axis_u_limits": calibration["x_axis_u_limits"],
            "coordinate_u": coordinate,
            "expected_column_float": expected_column_float,
            "candidate_column": calibration["candidate_column"],
            "column_rounding_residual_pixels": abs(
                calibration["candidate_column"] - expected_column_float
            ),
            "column_tolerance_pixels": calibration["column_tolerance_pixels"],
            "cyan_mask": "R<100, G>220, B>220",
            "cyan_run_pixels": (1658, 1671),
            "cyan_midpoint_pixel": 1664.5,
            "relic_coupling_center": coupling,
            "relic_line_thickness_interval": relic_interval,
            "conservative_relative_digitization_error": 0.02,
            "lz_coupling_limit": lz_figure_limit,
            "fermi_coupling_limit": fermi_figure_limit,
            "dampe_coupling_limit": dampe_figure_limit,
        },
        "arbitrary_point": {
            "mass_GeV": mass_gev,
            "coupling": arbitrary_coupling,
            "fraction_above_relic_center": arbitrary_coupling / coupling - 1.0,
            "on_digitized_relic_curve": relic_interval[0] <= arbitrary_coupling <= relic_interval[1],
        },
        "replacement_point": {
            "mass_GeV": mass_gev,
            "coupling": coupling,
            "bare_mass_GeV": bare_mass,
            "Gamma_invisible_MeV": 1.0e3 * invisible_width,
            "BR_invisible": invisible_branching,
            "sigma_SI_cm2_paper_equation": sigma_si,
            "lz_v2_limit_cm2_log_interpolated": lz_v2_interpolated_limit,
            "sigma_over_lz_v2": sigma_si / lz_v2_interpolated_limit,
            "on_digitized_relic_curve": relic_interval[0] <= coupling <= relic_interval[1],
            "below_lz_figure_limit": coupling < lz_figure_limit,
            "below_fermi_figure_limit": coupling < fermi_figure_limit,
            "below_dampe_figure_limit": coupling < dampe_figure_limit,
            "below_strict_invisible_bound": invisible_branching < 0.107,
        },
        "lz_v2_interpolation": {
            "anchors": (
                (lz_low_mass, lz_low_limit),
                (lz_high_mass, lz_high_limit),
            ),
            "log_interpolation_fraction": interpolation_fraction,
        },
        "raw_boltzmann_grid_available": False,
        "joint_likelihood_grid_available": False,
        "parameters_derived_from_ce": False,
        "live_artifact_files_required_for_reverification": True,
        "live_artifact_reverification_in_default_self_test": False,
    }


def anomaly_evidence() -> dict[str, Any]:
    fields = (
        ("Q", 3, 2, Fraction(1, 6), Fraction(1, 3)),
        ("u^c", 3, 1, Fraction(-2, 3), Fraction(-1, 3)),
        ("d^c", 3, 1, Fraction(1, 3), Fraction(-1, 3)),
        ("L", 1, 2, Fraction(-1, 2), Fraction(-1, 1)),
        ("e^c", 1, 1, Fraction(1, 1), Fraction(1, 1)),
        ("nu^c", 1, 1, Fraction(0, 1), Fraction(1, 1)),
    )
    coefficients = {
        "SU3^2_BminusL_without_common_T": (
            2 * fields[0][4] + fields[1][4] + fields[2][4]
        ),
        "SU2^2_BminusL_without_common_T": 3 * fields[0][4] + fields[3][4],
        "Y^2_BminusL": sum(
            color * weak * hypercharge**2 * charge
            for _, color, weak, hypercharge, charge in fields
        ),
        "Y_BminusL^2": sum(
            color * weak * hypercharge * charge**2
            for _, color, weak, hypercharge, charge in fields
        ),
        "BminusL^3": sum(
            color * weak * charge**3
            for _, color, weak, _, charge in fields
        ),
        "gravity^2_BminusL": sum(
            color * weak * charge
            for _, color, weak, _, charge in fields
        ),
    }
    without_nu = fields[:-1]
    cubic_without_nu = sum(
        color * weak * charge**3 for _, color, weak, _, charge in without_nu
    )
    gravity_without_nu = sum(
        color * weak * charge for _, color, weak, _, charge in without_nu
    )
    hypercharge_trace = sum(
        color * weak * hypercharge * charge
        for _, color, weak, hypercharge, charge in fields
    )
    branches = {
        "D_UNBROKEN_BL": {
            "description_level": "UV and EFT",
            "exact_BL_unbroken": True,
            "MR_nonzero": False,
            "explicit_C5_nonzero": False,
            "matched_low_energy_C5_nonzero": False,
            "complex_heavy_Yukawa": False,
        },
        "M_WEINBERG_EFT": {
            "description_level": "low-energy EFT",
            "exact_BL_unbroken": False,
            "MR_nonzero": False,
            "explicit_C5_nonzero": True,
            "matched_low_energy_C5_nonzero": True,
            "complex_heavy_Yukawa": False,
        },
        "M_TYPEI_UV": {
            "description_level": "UV completion",
            "exact_BL_unbroken": False,
            "MR_nonzero": True,
            "explicit_C5_nonzero": False,
            "matched_low_energy_C5_nonzero": True,
            "complex_heavy_Yukawa": True,
            "matching_relation": "C5 = Y^T M_R^{-1} Y up to convention",
        },
    }
    return {
        "field_basis": "left-handed Weyl, one generation",
        "coefficients": {key: str(value) for key, value in coefficients.items()},
        "all_local_anomalies_zero": all(value == 0 for value in coefficients.values()),
        "without_nu_c": {
            "BminusL^3": str(cubic_without_nu),
            "gravity^2_BminusL": str(gravity_without_nu),
        },
        "SU2_doublet_count_with_color": 4,
        "SU2_Witten_parity_even": True,
        "Tr_Y_BminusL_per_generation": str(hypercharge_trace),
        "kinetic_mixing_generically_generated": hypercharge_trace != 0,
        "operator_charges": {
            "bar_L_tildeH_nuR": "0",
            "nuR_Majorana": "-2",
            "Weinberg_LH_squared": "-2",
        },
        "real_portal_scalar_continuous_BL_charge": "0",
        "branches": branches,
        "gauged_unbroken_BL_massless_vector_gate_open": True,
    }


def modified_bessel_k(order: int, z_value: float, panels: int = 200) -> float:
    """Integral K_nu(z), sufficient for the explicitly labelled toy gate."""

    if z_value <= 0.0 or panels <= 0 or panels % 2:
        raise ValueError("z must be positive and Simpson panels positive/even")
    upper = 14.0
    step = upper / panels

    def integrand(argument: float) -> float:
        cosh_value = math.cosh(argument)
        return math.exp(-z_value * cosh_value) * math.cosh(order * argument)

    total = integrand(0.0) + integrand(upper)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * integrand(index * step)
    return total * step / 3.0


@lru_cache(maxsize=8)
def solve_decay_inverse_decay(step: float, initial_abundance: str = "thermal") -> float:
    z_initial = 0.001
    z_final = 30.001
    steps = int(round((z_final - z_initial) / step))
    actual_step = (z_final - z_initial) / steps
    g_star = 106.75
    electroweak_vev = 246.22
    dirac_vev = electroweak_vev / math.sqrt(2.0)
    planck_mass = 1.2209e19
    m_star_ev = (
        8.0
        * math.pi
        * dirac_vev**2
        * 1.66
        * math.sqrt(g_star)
        / planck_mass
        * 1.0e9
    )
    decay_parameter = 0.049986 / m_star_ev

    @lru_cache(maxsize=None)
    def coefficients(index: int) -> tuple[float, float, float]:
        z_value = z_initial + index * actual_step
        k1 = modified_bessel_k(1, z_value)
        k2 = modified_bessel_k(2, z_value)
        equilibrium = 0.5 * z_value**2 * k2
        decay = decay_parameter * z_value * k1 / k2
        washout = 0.25 * decay_parameter * z_value**3 * k1
        return equilibrium, decay, washout

    abundance = 1.0 if initial_abundance == "thermal" else 0.0
    asymmetry = 0.0
    for index in range(steps):
        equilibrium_0, decay_0, washout_0 = coefficients(index)
        equilibrium_1, decay_1, washout_1 = coefficients(index + 1)
        source_abundance_0 = decay_0 * equilibrium_0
        source_abundance_1 = decay_1 * equilibrium_1

        right_abundance = (
            (1.0 - 0.5 * actual_step * decay_0) * abundance
            + 0.5 * actual_step * (source_abundance_0 + source_abundance_1)
        )
        abundance_next = right_abundance / (1.0 + 0.5 * actual_step * decay_1)

        derivative_source_0 = -decay_0 * abundance + source_abundance_0
        derivative_source_1 = -decay_1 * abundance_next + source_abundance_1
        right_asymmetry = (
            (1.0 - 0.5 * actual_step * washout_0) * asymmetry
            + 0.5 * actual_step * (derivative_source_0 + derivative_source_1)
        )
        asymmetry_next = right_asymmetry / (1.0 + 0.5 * actual_step * washout_1)
        abundance, asymmetry = abundance_next, asymmetry_next
    return -asymmetry


@lru_cache(maxsize=1)
def leptogenesis_evidence() -> dict[str, Any]:
    g_star = 106.75
    electroweak_vev = 246.22
    dirac_vev = electroweak_vev / math.sqrt(2.0)
    planck_mass = 1.2209e19
    m_tilde_ev = 0.049986
    light_mass_difference_gev = (49.986 - 0.306) * 1.0e-12
    m_star_ev = (
        8.0
        * math.pi
        * dirac_vev**2
        * 1.66
        * math.sqrt(g_star)
        / planck_mass
        * 1.0e9
    )
    decay_parameter = m_tilde_ev / m_star_ev
    efficiency_coarse = solve_decay_inverse_decay(0.02, "thermal")
    efficiency_fine = solve_decay_inverse_decay(0.01, "thermal")
    efficiency_zero = solve_decay_inverse_decay(0.01, "zero")
    equilibrium_yield_coefficient = (
        135.0 * 1.202056903159594 / (4.0 * math.pi**4)
    )
    target_yb = 8.7e-11
    sphaleron = 28.0 / 79.0
    required_epsilon = (
        target_yb
        * g_star
        / (sphaleron * equilibrium_yield_coefficient * efficiency_fine)
    )
    formal_m1_min = (
        required_epsilon
        * 16.0
        * math.pi
        * dirac_vev**2
        / (3.0 * light_mass_difference_gev)
    )
    benchmark_m1 = 2.0e12
    epsilon_di_max = (
        3.0
        * benchmark_m1
        * light_mass_difference_gev
        / (16.0 * math.pi * dirac_vev**2)
    )
    hubble_at_m1 = 1.66 * math.sqrt(g_star) * benchmark_m1**2 / planck_mass
    ydagger_y_11 = decay_parameter * 8.0 * math.pi * hubble_at_m1 / benchmark_m1
    width_1 = ydagger_y_11 * benchmark_m1 / (8.0 * math.pi)
    return {
        "surrogate": "one-flavor decays plus inverse decays only",
        "z_interval": (0.001, 30.001),
        "bessel_integral_upper": 14.0,
        "bessel_simpson_panels": 200,
        "integrator": "implicit trapezoid",
        "m_star_eV": m_star_ev,
        "K": decay_parameter,
        "efficiency_h_0p02": efficiency_coarse,
        "efficiency_h_0p01": efficiency_fine,
        "efficiency_step_difference": abs(efficiency_fine - efficiency_coarse),
        "efficiency_zero_initial_h_0p01": efficiency_zero,
        "initial_condition_difference": abs(efficiency_fine - efficiency_zero),
        "equilibrium_yield_coefficient": equilibrium_yield_coefficient,
        "required_epsilon_set_from_target_YB": required_epsilon,
        "target_used_to_set_epsilon": True,
        "formal_DI_minimum_M1_GeV": formal_m1_min,
        "DI_bound_convention": "non-SUSY one-Higgs 3/(16*pi)",
        "unflavored_threshold_GeV": 1.0e12,
        "formal_minimum_inside_one_flavor_regime": formal_m1_min >= 1.0e12,
        "bridge_benchmark_M1_GeV": benchmark_m1,
        "epsilon_DI_max_at_benchmark": epsilon_di_max,
        "capacity_margin": epsilon_di_max / required_epsilon,
        "YdaggerY_11": ydagger_y_11,
        "sqrt_YdaggerY_11": math.sqrt(ydagger_y_11),
        "Gamma1_GeV": width_1,
        "H_at_M1_GeV": hubble_at_m1,
        "Gamma_over_H": width_1 / hubble_at_m1,
        "complex_yukawa_matrix_supplied": False,
        "flavored_or_density_matrix_transport_solved": False,
        "delta_L1_scatterings_included": False,
        "delta_L2_washout_included": False,
        "spectator_thermal_effects_included": False,
        "reheating_gate_passed": False,
        "YB_is_prediction": False,
    }


def build_branches(evidence: dict[str, Any]) -> tuple[ImprovementBranch, ...]:
    bao = evidence["bao"]
    portal = evidence["portal"]
    anomaly = evidence["neutrino_anomaly"]
    lepto = evidence["leptogenesis"]
    return (
        ImprovementBranch(
            branch_id="bao_grouped_cross_validation",
            source_loop_id="bootstrap_cosmology",
            parent_claim_still_rejected=True,
            iterations=(
                ImprovementIteration(
                    0,
                    "fixed CE fractions plus external H0 and rd",
                    "REJECT",
                    "retain the original common parameter package",
                    (),
                    "full 13-point DESI DR2 covariance goodness of fit",
                    bao["fixed"],
                    "the rejection is specific to this fixed package, not flat LambdaCDM in general",
                ),
                ImprovementIteration(
                    1,
                    "one H0*rd scale fitted to all DR2 points",
                    "SELECTION_NEGATIVE_CONTROL",
                    "add one external-normalization nuisance",
                    ("one scale fitted to the tested data",),
                    "apply AIC, AICc, and BIC penalties",
                    bao["global_same_data_scale"],
                    "same-data fit is exploratory and cannot be a prediction",
                ),
                ImprovementIteration(
                    2,
                    "deterministic redshift/covariance-block two-fold validation",
                    "SELECTION_NEGATIVE_CONTROL",
                    "fit the scale on alternating intact blocks and integrate its uncertainty in the other blocks",
                    ("deterministic block construction", "training posterior scale variance"),
                    "both directional predictive p-values >=0.05 and mutually consistent q estimates",
                    bao["two_fold_totals"],
                    "all DR2 data were already known; this is internal validation, not an external forecast",
                ),
                ImprovementIteration(
                    3,
                    "leave-one-redshift-block-out sensitivity",
                    "SELECTION_NEGATIVE_CONTROL",
                    "repeat the predictive check for every intact covariance block",
                    ("seven overlapping training fits",),
                    "all predictive p-values >=0.05 while reporting adverse scores",
                    bao["leave_one_block_out_totals"],
                    "two of seven blocks worsen predictive score and overlapping folds have no aggregate p-value",
                ),
                ImprovementIteration(
                    4,
                    "nonoverlapping external BAO/CMB/SN prediction",
                    "NOT_REACHED",
                    "freeze the nuisance or derive it before a new release",
                    ("future independent data", "covariant stress tensor and perturbations"),
                    "pass an untouched external likelihood with one frozen parameter set",
                    {"external_nonoverlapping_release_tested": False},
                    "the present exercise cannot manufacture temporal independence",
                ),
            ),
            maximum_supported_stage="INTERNAL_GROUPED_CV_SHAPE_SELECTION_PASS",
            original_claim_promoted=False,
            ce_specific_physical_claim_closed=False,
            next_required_gate="derive the normalization and pass a nonoverlapping external BAO/CMB/SN release",
        ),
        ImprovementBranch(
            branch_id="portal_relic_curve_replacement",
            source_loop_id="scalar_portal_benchmark",
            parent_claim_still_rejected=True,
            iterations=(
                ImprovementIteration(
                    0,
                    "arbitrary 62 GeV, a2=3e-4 rectangle point",
                    "REJECT",
                    "test the prior illustrative point against the digitized relic equality curve",
                    ("arXiv:2410.21089v2 Fig. 3",),
                    "require membership in the full cyan line-thickness interval",
                    portal["arbitrary_point"],
                    "marginal one-dimensional ranges do not establish correlated relic equality",
                ),
                ImprovementIteration(
                    1,
                    "62 GeV, a2=2.419123e-4 digitized relic point",
                    "EXTERNAL_FIGURE_BRIDGE_PASS",
                    "move only the coupling to the center of the published all-DM curve",
                    ("digitized external relic curve", "independent bare mass", "LZ v2 table"),
                    "pass relic-curve, invisible, direct, and plotted indirect bounds at one point",
                    portal["replacement_point"],
                    "the paper supplies no raw Boltzmann or likelihood grid and the point is not CE-derived",
                ),
                ImprovementIteration(
                    2,
                    "CE-derived portal global likelihood",
                    "NOT_REACHED",
                    "derive mS and a2 before evaluating a raw global likelihood",
                    ("raw Boltzmann grid", "joint likelihood", "CE matching"),
                    "pass a reproducible computational likelihood without selecting the point from its target curve",
                    {
                        "raw_boltzmann_grid_available": False,
                        "joint_likelihood_grid_available": False,
                        "parameters_derived_from_ce": False,
                    },
                    "figure digitization is evidence for an external model point, not a new theory prediction",
                ),
            ),
            maximum_supported_stage="EXTERNAL_FIGURE_BRIDGE_PASS",
            original_claim_promoted=False,
            ce_specific_physical_claim_closed=False,
            next_required_gate="publish/reproduce the raw relic and likelihood grids, then derive the point from CE",
        ),
        ImprovementBranch(
            branch_id="neutrino_symmetry_consistency",
            source_loop_id="neutrino_mass",
            parent_claim_still_rejected=True,
            iterations=(
                ImprovementIteration(
                    0,
                    "three-nuR exact unbroken B-L Dirac branch",
                    "AUDIT_PASS",
                    "compute every local anomaly exactly in a left-handed Weyl basis",
                    ("three right-handed neutrinos", "exact unbroken B-L"),
                    "all six local anomaly coefficients vanish and nuR removal fails",
                    {
                        "coefficients": anomaly["coefficients"],
                        "all_local_anomalies_zero": anomaly["all_local_anomalies_zero"],
                        "without_nu_c": anomaly["without_nu_c"],
                    },
                    "gauged unbroken B-L opens a massless-vector/long-range-force gate; Yukawas remain fitted",
                ),
                ImprovementIteration(
                    1,
                    "Dirac branch and Weinberg-EFT/type-I-UV descendants",
                    "AUDIT_PASS",
                    "separate exact-symmetry incompatibility from UV-to-EFT matching",
                    (),
                    "forbid exact-unbroken B-L with LNV, while requiring type-I to match a low-energy C5",
                    {"branches": anomaly["branches"], "operator_charges": anomaly["operator_charges"]},
                    "Weinberg EFT and type-I UV are compatible descriptions at different scales; consistency does not predict flavor",
                ),
                ImprovementIteration(
                    2,
                    "fully specified neutrino flavor model",
                    "NOT_REACHED",
                    "supply the protected flavor action and held-out predictions",
                    ("flavor symmetry", "full Yukawa/Wilson matrix", "RG prescription"),
                    "pass masses, PMNS observables, phases, and RG stability without target fitting",
                    {"held_out_flavor_likelihood_present": False},
                    "the audit only proves that the proposed completions are internally distinct and anomaly consistent",
                ),
            ),
            maximum_supported_stage="EXACT_SYMMETRY_AND_UV_EFT_MATCHING_AUDIT_PASS",
            original_claim_promoted=False,
            ce_specific_physical_claim_closed=False,
            next_required_gate="choose one symmetry realization and derive a held-out flavor texture",
        ),
        ImprovementBranch(
            branch_id="leptogenesis_transport_surrogate",
            source_loop_id="baryogenesis",
            parent_claim_still_rejected=True,
            iterations=(
                ImprovementIteration(
                    0,
                    "one-flavor decay/inverse-decay ODE",
                    "BRIDGE_PASS",
                    "replace the assumed kappa=0.1 with a converged strong-washout surrogate",
                    ("mtilde1=0.049986 eV", "M1=2e12 GeV benchmark", "thermal history assumptions"),
                    "step convergence, initial-abundance convergence, Gamma/H=K, and DI capacity",
                    lepto,
                    "epsilon is set from the observed YB and the surrogate omits scattering, flavor, thermal, and spectator terms",
                ),
                ImprovementIteration(
                    1,
                    "three-flavor or density-matrix prediction",
                    "NOT_REACHED",
                    "derive complex heavy-neutrino parameters and solve the full transport system",
                    (
                        "M1,2,3 and complex 3x3 Yukawa or Casas-Ibarra R",
                        "TR>=M1",
                        "Delta-L=1/2 and spectator/thermal effects",
                    ),
                    "compute epsilon from the action and hold out YB",
                    {"YB_is_prediction": False, "full_transport_solved": False},
                    "a capacity margin is not a baryogenesis prediction",
                ),
            ),
            maximum_supported_stage="ONE_FLAVOR_DECAY_INVERSE_DECAY_SURROGATE_BRIDGE_PASS",
            original_claim_promoted=False,
            ce_specific_physical_claim_closed=False,
            next_required_gate="fix the heavy sector independently and solve converged flavored transport with YB held out",
        ),
    )


def _build_report_uncached() -> ImprovementReport:
    base_report = base.build_report()
    inventory = base.rejected_inventory()
    semantic_digest = base.semantic_manifest_digest(inventory, base.semantic_route_index())
    evidence = copy.deepcopy(
        {
            "bao": bao_evidence(),
            "portal": portal_evidence(),
            "neutrino_anomaly": anomaly_evidence(),
            "leptogenesis": leptogenesis_evidence(),
        }
    )
    branches = build_branches(evidence)
    return ImprovementReport(
        schema_version="3.0.0",
        base_source_sha256=base_report.source_sha256,
        base_semantic_sha256=semantic_digest,
        base_rejected_occurrences=base_report.source_rejected_occurrences,
        base_regression_witnesses=len(
            base_report.deleted_parent_regression_witnesses
        ),
        base_regression_witness_registry_sha256=(
            base_report.regression_witness_registry_sha256
        ),
        base_original_claims_promoted=base_report.original_claims_promoted,
        source_pins=copy.deepcopy(
            {
                "desi_upstream": DESI_UPSTREAM_PINS,
                "portal_artifacts": PORTAL_ARTIFACT_PINS,
            }
        ),
        branches=branches,
        evidence=evidence,
        original_claims_promoted=sum(branch.original_claim_promoted for branch in branches),
        ce_specific_physical_claims_closed=sum(
            branch.ce_specific_physical_claim_closed for branch in branches
        ),
    )


@lru_cache(maxsize=1)
def build_report() -> ImprovementReport:
    return _build_report_uncached()


def validate_report(report: ImprovementReport) -> tuple[SelfCheck, ...]:
    checks: list[SelfCheck] = []
    canonical_report = _build_report_uncached()
    base_report = base.build_report()
    base_loop_ids = {loop.loop_id for loop in base_report.loops}
    canonical_branch_ids = tuple(
        branch.branch_id for branch in canonical_report.branches
    )
    canonical_branches_by_id = {
        branch.branch_id: branch for branch in canonical_report.branches
    }
    actual_branch_ids = tuple(branch.branch_id for branch in report.branches)
    derived_original_claims_promoted = sum(
        branch.original_claim_promoted for branch in report.branches
    )
    derived_ce_specific_physical_claims_closed = sum(
        branch.ce_specific_physical_claim_closed for branch in report.branches
    )
    branch_static_metadata_match = all(
        branch.branch_id in canonical_branches_by_id
        and branch.source_loop_id in base_loop_ids
        and branch.source_loop_id
        == canonical_branches_by_id[branch.branch_id].source_loop_id
        and branch.maximum_supported_stage
        == canonical_branches_by_id[branch.branch_id].maximum_supported_stage
        and branch.next_required_gate
        == canonical_branches_by_id[branch.branch_id].next_required_gate
        and branch.original_claim_promoted
        == canonical_branches_by_id[branch.branch_id].original_claim_promoted
        and branch.ce_specific_physical_claim_closed
        == canonical_branches_by_id[
            branch.branch_id
        ].ce_specific_physical_claim_closed
        for branch in report.branches
    )
    branch_identity_and_aggregate_consistent = (
        report.schema_version == canonical_report.schema_version
        and actual_branch_ids == canonical_branch_ids
        and len(set(actual_branch_ids)) == len(actual_branch_ids)
        and branch_static_metadata_match
        and report.original_claims_promoted
        == derived_original_claims_promoted
        and report.ce_specific_physical_claims_closed
        == derived_ce_specific_physical_claims_closed
    )
    checks.append(
        SelfCheck(
            "canonical-report-rebuild",
            report == canonical_report,
            "every report field matches a fresh deterministic rebuild",
        )
    )
    checks.append(
        SelfCheck(
            "source-pin-integrity",
            report.source_pins == canonical_report.source_pins,
            "DESI and portal provenance pins match the deterministic builder",
        )
    )
    checks.append(
        SelfCheck(
            "branch-identity-stage-and-aggregate-consistency",
            branch_identity_and_aggregate_consistent,
            f"branch ids={actual_branch_ids}; expected={canonical_branch_ids}; "
            f"derived/report promotions={derived_original_claims_promoted}/"
            f"{report.original_claims_promoted}, CE closures="
            f"{derived_ce_specific_physical_claims_closed}/"
            f"{report.ce_specific_physical_claims_closed}",
        )
    )
    base_checks = base.validate_report(base_report)
    checks.append(
        SelfCheck(
            "base-audit-lock",
            len(base_checks) == 22
            and all(check.passed for check in base_checks)
            and report.base_source_sha256 == EXPECTED_BASE_SOURCE_SHA256
            and report.base_semantic_sha256 == EXPECTED_BASE_SEMANTIC_SHA256
            and report.base_rejected_occurrences
            == base_report.source_rejected_occurrences
            == 0
            and base_report.routed_rejected_occurrences
            == base_report.source_rejected_occurrences
            == 0
            and base_report.occurrence_routes == ()
            and report.base_regression_witnesses
            == len(base_report.deleted_parent_regression_witnesses)
            == 18
            and report.base_regression_witness_registry_sha256
            == base_report.regression_witness_registry_sha256
            == EXPECTED_BASE_REGRESSION_WITNESS_SHA256
            and report.base_original_claims_promoted == 0,
            "base rejection audit remains 22/22 with "
            "zero deleted-parent markers in canonical prose, 18 immutable "
            "internal witnesses, and zero promotions",
        )
    )
    checks.append(
        SelfCheck(
            "branch-shape-and-status",
            len(report.branches) == 4
            and all(branch.parent_claim_still_rejected for branch in report.branches)
            and all(
                iteration.status in ALLOWED_STATUSES
                for branch in report.branches
                for iteration in branch.iterations
            ),
            "four improvement branches use only the fail-closed status vocabulary",
        )
    )
    checks.append(
        SelfCheck(
            "no-claim-promotion",
            report.original_claims_promoted == 0
            and report.ce_specific_physical_claims_closed == 0,
            "original promotions=0 and CE-specific physical closures=0",
        )
    )

    # Numerical golden checks use the deterministic builder.  The supplied
    # report is checked for exact and structural equality above, so malformed
    # evidence or missing/extra branches fail without causing lookup errors.
    bao = canonical_report.evidence["bao"]
    expected_blocks = ((0,), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
    checks.append(
        SelfCheck(
            "desi-source-and-block-lock",
            bao["embedded_digest"] == EXPECTED_EMBEDDED_DESI_SHA256
            and bao["blocks"] == expected_blocks
            and DESI_UPSTREAM_PINS["repository_commit"]
            == "bb0c1c9009dc76d1391300e169e8df38fd1096db"
            and DESI_UPSTREAM_PINS["mean_url"]
            == (
                "https://raw.githubusercontent.com/CobayaSampler/bao_data/"
                "bb0c1c9009dc76d1391300e169e8df38fd1096db/"
                "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt"
            )
            and DESI_UPSTREAM_PINS["mean_bytes"] == 472
            and DESI_UPSTREAM_PINS["mean_sha256"]
            == "9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585"
            and DESI_UPSTREAM_PINS["covariance_url"]
            == (
                "https://raw.githubusercontent.com/CobayaSampler/bao_data/"
                "bb0c1c9009dc76d1391300e169e8df38fd1096db/"
                "desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt"
            )
            and DESI_UPSTREAM_PINS["covariance_bytes"] == 2547
            and DESI_UPSTREAM_PINS["covariance_sha256"]
            == "252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509",
            f"commit={DESI_UPSTREAM_PINS['repository_commit']}; "
            f"embedded={bao['embedded_digest']}; blocks={bao['blocks']}",
        )
    )
    checks.append(
        SelfCheck(
            "desi-fixed-rejection",
            close(bao["fixed"]["chi2"], 37.100260857153614, 5e-10)
            and bao["fixed"]["dof"] == 13
            and close(bao["fixed"]["p_value"], 0.000399573259824, 5e-15)
            and bao["fixed"]["verdict"] == "REJECT",
            f"chi2={bao['fixed']['chi2']:.12f}/13, p={bao['fixed']['p_value']:.12g}",
        )
    )
    global_scale = bao["global_same_data_scale"]
    checks.append(
        SelfCheck(
            "desi-global-scale-penalties",
            close(global_scale["q"], 0.98647693346963, 5e-13)
            and close(global_scale["q_variance"], 7.466681796400377e-06, 5e-18)
            and close(global_scale["chi2"], 12.608346862241673, 5e-10)
            and close(global_scale["aic_improvement"], 22.49191399491194, 5e-10)
            and close(global_scale["aicc_improvement"], 22.12827763127558, 5e-10)
            and close(global_scale["bic_improvement"], 21.926964637450403, 5e-10),
            f"q={global_scale['q']:.12f}; AICc improvement={global_scale['aicc_improvement']:.6f}",
        )
    )
    two_fold = bao["two_fold"]
    checks.append(
        SelfCheck(
            "desi-two-fold-predictive-pass",
            len(two_fold) == 2
            and all(fold.holdout_predictive_p_value >= 0.05 for fold in two_fold)
            and close(two_fold[0].q_train, 0.9853264986128911, 5e-13)
            and close(two_fold[1].q_train, 0.9878951903165132, 5e-13)
            and close(
                bao["two_fold_totals"]["predictive_score_improvement"],
                22.87617010135335,
                5e-9,
            )
            and bao["two_fold_totals"]["q_difference_sigma"] < 1.0,
            "both intact-block directions pass after integrating training-scale uncertainty",
        )
    )
    lobo = bao["leave_one_block_out"]
    checks.append(
        SelfCheck(
            "desi-lobo-sensitivity",
            len(lobo) == 7
            and bao["leave_one_block_out_totals"]["minimum_predictive_p_value"] >= 0.05
            and bao["leave_one_block_out_totals"]["positive_score_folds"] == 5
            and close(
                bao["leave_one_block_out_totals"]["predictive_chi2"],
                13.666311200396625,
                5e-9,
            ),
            "all seven predictive p-values pass, with two adverse predictive-score folds retained",
        )
    )
    checks.append(
        SelfCheck(
            "desi-no-external-promotion",
            bao["external_nonoverlapping_release_tested"] is False
            and bao["aggregate_cv_p_value_reported"] is False
            and canonical_branches_by_id[
                "bao_grouped_cross_validation"
            ].maximum_supported_stage
            == "INTERNAL_GROUPED_CV_SHAPE_SELECTION_PASS",
            "internal overlapping folds are not relabeled as an external prediction",
        )
    )

    portal = canonical_report.evidence["portal"]
    arbitrary = portal["arbitrary_point"]
    replacement = portal["replacement_point"]
    digitization = portal["figure_digitization"]
    checks.append(
        SelfCheck(
            "portal-provenance-pins-and-normalization",
            PORTAL_ARTIFACT_PINS["paper_sha256"]
            == "dc31c67d61457679c4a642dba42377183b96dbdfe2dd04cb9b61dd284f98a145"
            and PORTAL_ARTIFACT_PINS["figure_image_sha256"]
            == "4c2a4fa670e92cef0c208e90d49b666a146583090904e541b2e439aa21b4aace"
            and portal["normalization"]["lambda_HP_equals_a2"] is True
            and digitization["x_axis_u_limits"] == (0.0, 7.0)
            and close(digitization["coordinate_u"], 2.0969100130080562, 5e-15)
            and close(digitization["expected_column_float"], 2138.471848117027, 5e-10)
            and digitization["column_rounding_residual_pixels"]
            <= digitization["column_tolerance_pixels"]
            and portal["live_artifact_reverification_in_default_self_test"] is False,
            (
                "paper/image pins and u-to-column calibration are locked; "
                "live raster verification is an explicit optional gate"
            ),
        )
    )
    checks.append(
        SelfCheck(
            "portal-arbitrary-point-rejected",
            arbitrary["on_digitized_relic_curve"] is False
            and arbitrary["fraction_above_relic_center"] > 0.20,
            f"3e-4 is {100.0 * arbitrary['fraction_above_relic_center']:.2f}% above the relic center",
        )
    )
    checks.append(
        SelfCheck(
            "portal-replacement-common-point-pass",
            replacement["on_digitized_relic_curve"] is True
            and replacement["below_lz_figure_limit"] is True
            and replacement["below_fermi_figure_limit"] is True
            and replacement["below_dampe_figure_limit"] is True
            and replacement["below_strict_invisible_bound"] is True
            and replacement["sigma_over_lz_v2"] < 1.0
            and close(replacement["sigma_over_lz_v2"], 0.2652633564, 5e-10)
            and close(replacement["coupling"], 2.4191233235598564e-4, 5e-16)
            and close(replacement["bare_mass_GeV"], 61.88182636, 5e-8),
            f"a2={replacement['coupling']:.10e}; sigma/LZ={replacement['sigma_over_lz_v2']:.6f}",
        )
    )
    checks.append(
        SelfCheck(
            "portal-figure-only-ce-guard",
            portal["raw_boltzmann_grid_available"] is False
            and portal["joint_likelihood_grid_available"] is False
            and portal["parameters_derived_from_ce"] is False
            and canonical_branches_by_id[
                "portal_relic_curve_replacement"
            ].maximum_supported_stage
            == "EXTERNAL_FIGURE_BRIDGE_PASS",
            "external figure evidence passes, while computational-global and CE-derived promotion remain closed",
        )
    )

    anomaly = canonical_report.evidence["neutrino_anomaly"]
    checks.append(
        SelfCheck(
            "bl-anomaly-exact-arithmetic",
            anomaly["all_local_anomalies_zero"] is True
            and anomaly["without_nu_c"]["BminusL^3"] == "-1"
            and anomaly["without_nu_c"]["gravity^2_BminusL"] == "-1"
            and anomaly["SU2_Witten_parity_even"] is True,
            f"coefficients={anomaly['coefficients']}; removing nu^c gives cubic/gravity=-1",
        )
    )
    checks.append(
        SelfCheck(
            "bl-operator-and-mixing-audit",
            anomaly["operator_charges"]["bar_L_tildeH_nuR"] == "0"
            and anomaly["operator_charges"]["nuR_Majorana"] == "-2"
            and anomaly["operator_charges"]["Weinberg_LH_squared"] == "-2"
            and anomaly["Tr_Y_BminusL_per_generation"] == "8/3"
            and anomaly["kinetic_mixing_generically_generated"] is True
            and anomaly["gauged_unbroken_BL_massless_vector_gate_open"] is True,
            "Dirac is allowed, Majorana/Weinberg are forbidden, and gauge-sector follow-up stays open",
        )
    )
    branch_assignments = anomaly["branches"]
    checks.append(
        SelfCheck(
            "neutrino-symmetry-and-uv-eft-matching",
            branch_assignments["D_UNBROKEN_BL"]["exact_BL_unbroken"] is True
            and not branch_assignments["D_UNBROKEN_BL"]["MR_nonzero"]
            and not branch_assignments["D_UNBROKEN_BL"]["matched_low_energy_C5_nonzero"]
            and branch_assignments["M_WEINBERG_EFT"]["explicit_C5_nonzero"] is True
            and branch_assignments["M_WEINBERG_EFT"]["matched_low_energy_C5_nonzero"] is True
            and not branch_assignments["M_WEINBERG_EFT"]["exact_BL_unbroken"]
            and branch_assignments["M_TYPEI_UV"]["MR_nonzero"] is True
            and branch_assignments["M_TYPEI_UV"]["matched_low_energy_C5_nonzero"] is True
            and not branch_assignments["M_TYPEI_UV"]["exact_BL_unbroken"],
            "exact Dirac is incompatible with LNV; type-I UV consistently matches the Weinberg EFT",
        )
    )

    lepto = canonical_report.evidence["leptogenesis"]
    checks.append(
        SelfCheck(
            "leptogenesis-ode-convergence",
            close(lepto["efficiency_h_0p02"], 0.00489307846656, 2e-12)
            and close(lepto["efficiency_h_0p01"], 0.00489316466135, 2e-12)
            and lepto["efficiency_step_difference"] < 1.0e-7
            and lepto["initial_condition_difference"] < 1.0e-11,
            f"kappa_f={lepto['efficiency_h_0p01']:.12g}; step delta={lepto['efficiency_step_difference']:.3e}",
        )
    )
    checks.append(
        SelfCheck(
            "leptogenesis-capacity-consistency",
            close(lepto["K"], 46.70678032185461, 5e-11)
            and close(lepto["Gamma_over_H"], lepto["K"], 5e-11)
            and close(lepto["required_epsilon_set_from_target_YB"], 1.2857811437e-5, 5e-15)
            and close(lepto["formal_DI_minimum_M1_GeV"], 1.3144701377e11, 5e3)
            and lepto["DI_bound_convention"] == "non-SUSY one-Higgs 3/(16*pi)"
            and lepto["formal_minimum_inside_one_flavor_regime"] is False
            and lepto["capacity_margin"] > 15.0,
            f"epsilon_req={lepto['required_epsilon_set_from_target_YB']:.8e}; margin={lepto['capacity_margin']:.4f}",
        )
    )
    checks.append(
        SelfCheck(
            "leptogenesis-no-target-leakage-promotion",
            lepto["target_used_to_set_epsilon"] is True
            and lepto["complex_yukawa_matrix_supplied"] is False
            and lepto["flavored_or_density_matrix_transport_solved"] is False
            and lepto["YB_is_prediction"] is False
            and canonical_branches_by_id[
                "leptogenesis_transport_surrogate"
            ].iterations[-1].status
            == "NOT_REACHED",
            "the converged surrogate is retained as capacity only; YB prediction remains NOT_REACHED",
        )
    )
    return tuple(checks)


def json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable report")
    parser.add_argument("--self-test", action="store_true", help="run fail-closed regression gates")
    parser.add_argument("--verify-portal-pdf", type=Path, help="verify a downloaded arXiv v2 PDF")
    parser.add_argument(
        "--verify-portal-image",
        type=Path,
        help="verify and redigitize pypdf pages[5].images[0]",
    )
    parser.add_argument("--verify-desi-mean", type=Path, help="verify the upstream DR2 mean file")
    parser.add_argument(
        "--verify-desi-covariance",
        type=Path,
        help="verify the upstream DR2 covariance file",
    )
    parser.add_argument(
        "--require-live-artifacts",
        action="store_true",
        help=(
            "release gate: require portal PDF/image and DESI mean/covariance "
            "paths, then include every live verification in the exit status"
        ),
    )
    args = parser.parse_args()
    report = build_report()
    checks = list(validate_report(report))
    artifact_paths = {
        "portal_pdf": args.verify_portal_pdf,
        "portal_image": args.verify_portal_image,
        "desi_mean": args.verify_desi_mean,
        "desi_covariance": args.verify_desi_covariance,
    }
    provided_artifacts = tuple(
        name for name, path in artifact_paths.items() if path is not None
    )
    missing_artifacts = tuple(
        name for name, path in artifact_paths.items() if path is None
    )
    live_checks: list[SelfCheck] = []
    if args.require_live_artifacts:
        live_checks.append(
            SelfCheck(
                "live-artifact-requirements",
                not missing_artifacts,
                "all four live artifact paths supplied"
                if not missing_artifacts
                else f"missing required artifact paths: {missing_artifacts}",
            )
        )
    if args.verify_portal_pdf is not None:
        live_checks.append(
            pinned_file_check(
                "portal-pdf-live-integrity",
                args.verify_portal_pdf,
                PORTAL_ARTIFACT_PINS["paper_bytes"],
                PORTAL_ARTIFACT_PINS["paper_sha256"],
            )
        )
    if args.verify_portal_image is not None:
        live_checks.extend(portal_figure_live_checks(args.verify_portal_image))
    if args.verify_desi_mean is not None:
        live_checks.append(
            pinned_file_check(
                "desi-mean-live-integrity",
                args.verify_desi_mean,
                DESI_UPSTREAM_PINS["mean_bytes"],
                DESI_UPSTREAM_PINS["mean_sha256"],
            )
        )
    if args.verify_desi_covariance is not None:
        live_checks.append(
            pinned_file_check(
                "desi-covariance-live-integrity",
                args.verify_desi_covariance,
                DESI_UPSTREAM_PINS["covariance_bytes"],
                DESI_UPSTREAM_PINS["covariance_sha256"],
            )
        )
    checks.extend(live_checks)
    if not provided_artifacts and not args.require_live_artifacts:
        live_status = "NOT_RUN"
    elif missing_artifacts and not args.require_live_artifacts:
        live_status = (
            "PARTIAL_PASS"
            if all(check.passed for check in live_checks)
            else "PARTIAL_FAIL"
        )
    else:
        live_status = (
            "PASS" if live_checks and all(check.passed for check in live_checks) else "FAIL"
        )
    live_artifact_verification = {
        "status": live_status,
        "required": args.require_live_artifacts,
        "provided": {
            name: str(path) if path is not None else None
            for name, path in artifact_paths.items()
        },
        "missing": missing_artifacts,
        "checks": json_ready(live_checks),
    }
    passed = all(check.passed for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": passed,
                    "report": json_ready(report),
                    "self_checks": json_ready(checks),
                    "live_artifact_verification": live_artifact_verification,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if passed else 1
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    if live_status == "NOT_RUN":
        print(
            "[NOT_RUN] live-artifact-verification: portal PDF/image and DESI "
            "mean/covariance were not supplied; use --require-live-artifacts "
            "for the release gate"
        )
    else:
        print(
            f"[{live_status}] live-artifact-verification: "
            f"provided={provided_artifacts}, missing={missing_artifacts}"
        )
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} ({sum(c.passed for c in checks)}/{len(checks)})")
    if args.self_test:
        return 0 if passed else 1
    print("Use --json for the complete evidence ledger.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
