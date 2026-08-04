"""Adversarially validated crossed-holdout spatiotemporal response-mask gate.

The caller supplies a frozen design tensor ``M`` whose probe, resonance,
spatial-support, temporal-support, and command factors were calibrated outside
this audit.  One global amplitude is fitted on training cells, then disjoint
held-out cells are predicted from paired matched-minus-sham observations.

The final stage is deliberately *not* a causality or matter claim.  A supplied
``prearrival_mask`` is only a preregistered early-time control; relativistic
causality must be tested by a separate spacelike/no-signalling protocol.
Individual factors of ``M`` remain scale-gauge nonidentifiable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
import hashlib
import hmac
import math
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


class ResonantMaskStage(str, Enum):
    """Monotone control stages; none is a physical-field claim."""

    PAIRED_RESPONSE_CONTROL = "PAIRED_RESPONSE_CONTROL"
    FROZEN_MANIFEST_CONTROL = "FROZEN_MANIFEST_CONTROL"
    JOINT_MASK_GLS_CONTROL = "JOINT_MASK_GLS_CONTROL"
    CROSSED_HELDOUT_PREDICTION_CONTROL = "CROSSED_HELDOUT_PREDICTION_CONTROL"
    CONDITIONAL_SPATIOTEMPORAL_RESPONSE_MASK = (
        "CONDITIONAL_SPATIOTEMPORAL_RESPONSE_MASK"
    )


@dataclass(frozen=True)
class ResonantMaskClaimLocks:
    individual_factors_physically_identified: bool = False
    ce_coupling_derived: bool = False
    public_scaffold_derived: bool = False
    relativistic_causality_derived: bool = False
    causal_broadband_boundary_derived: bool = False
    material_phase_derived: bool = False
    new_matter_derived: bool = False
    observer_relative_reality_derived: bool = False
    renormalized_stress_tensor_derived: bool = False
    externally_timestamped_manifest_verified: bool = False


@dataclass(frozen=True)
class ResonantMaskRawInputs:
    """Canonical immutable inputs retained so a certificate can be recomputed."""

    cell_shape: tuple[int, ...]
    matched_response_flat: tuple[tuple[float, ...], ...]
    sham_response_flat: tuple[tuple[float, ...], ...]
    design_flat: tuple[float, ...]
    training_mask_flat: tuple[bool, ...]
    heldout_mask_flat: tuple[bool, ...]
    prearrival_mask_flat: tuple[bool, ...]
    off_support_mask_flat: tuple[bool, ...]
    target_mask_flat: tuple[bool, ...]
    matched_block_ids: tuple[str, ...]
    sham_block_ids: tuple[str, ...]
    preprocessing_artifact_sha256: str
    design_calibration_artifact_sha256: str
    declared_manifest_sha256: str
    manifest_frozen_before_data: bool
    masks_fixed_before_holdout: bool
    observations_are_independent_blocks: bool
    gaussian_mean_model_declared: bool
    expected_response_sign: int
    familywise_alpha: float
    equivalence_bound: float
    minimum_target_response: float
    maximum_training_reduced_chi_square: float
    maximum_covariance_condition_number: float
    covariance_rank_relative_tolerance: float
    minimum_paired_covariance_eigenvalue: float
    minimum_residual_mean_variance: float
    minimum_trials: int


@dataclass(frozen=True)
class ResonantSpatiotemporalMaskAudit:
    schema_version: str
    raw_inputs: ResonantMaskRawInputs
    cell_shape: tuple[int, ...]
    trial_count: int
    independent_block_count: int
    training_cell_count: int
    heldout_cell_count: int
    training_model_degrees_of_freedom: int
    simultaneous_comparison_count: int
    simultaneous_confidence_multiplier: float
    manifest_sha256: str
    computed_manifest_sha256: str
    manifest_hash_matches: bool
    manifest_frozen_before_data: bool
    masks_fixed_before_holdout: bool
    train_holdout_disjoint_and_complete: bool
    protected_masks_pairwise_disjoint: bool
    protected_masks_cover_exactly_heldout: bool
    paired_block_ids_aligned: bool
    paired_block_ids_unique: bool
    minimum_independent_blocks_met: bool
    independent_block_model_declared: bool
    gaussian_mean_model_declared: bool
    training_design_non_saturated: bool
    paired_covariance_rank: int
    paired_covariance_condition_number: float | None
    paired_covariance_minimum_eigenvalue: float
    training_covariance_rank: int
    training_covariance_condition_number: float | None
    training_residual_covariance_rank: int
    training_residual_covariance_condition_number: float | None
    heldout_residual_covariance_rank: int
    heldout_residual_covariance_condition_number: float | None
    training_covariance_nonvacuous: bool
    heldout_covariance_nonvacuous: bool
    covariance_nonvacuous: bool
    fitted_global_amplitude: float | None
    fitted_global_amplitude_standard_error: float | None
    training_reduced_chi_square: float | None
    maximum_training_absolute_residual: float | None
    maximum_training_residual_upper_bound: float | None
    maximum_heldout_residual_upper_bound: float | None
    maximum_prearrival_response_upper_bound: float | None
    maximum_off_support_response_upper_bound: float | None
    minimum_target_response_lower_bound: float | None
    heldout_localization_margin: float | None
    joint_mask_gls_pass: bool
    heldout_prediction_pass: bool
    prearrival_equivalence_pass: bool
    off_support_equivalence_pass: bool
    target_response_pass: bool
    heldout_localization_pass: bool
    factor_rescaling_counterexample_exact: bool
    individual_factor_normalizations_identifiable: bool
    conditional_spatiotemporal_response_mask: bool
    maximum_supported_stage: ResonantMaskStage
    first_blocker: str
    blockers: tuple[str, ...]
    claim_locks: ResonantMaskClaimLocks

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maximum_supported_stage"] = self.maximum_supported_stage.value
        return payload


@dataclass(frozen=True)
class _CovarianceDiagnostics:
    rank: int
    condition_number: float | None
    minimum_eigenvalue: float
    valid: bool


def _continued_beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized incomplete beta."""

    maximum_iterations = 256
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        numerator = iteration * (b - iteration) * x
        denominator = (qam + doubled) * (a + doubled)
        d = 1.0 + numerator * d / denominator
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / (denominator * c)
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x
        denominator = (a + doubled) * (qap + doubled)
        d = 1.0 + numerator * d / denominator
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / (denominator * c)
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ValueError("regularized incomplete beta fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x < 0.0 or x > 1.0:
        raise ValueError("regularized incomplete beta argument must lie in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta_fraction(a, b, x) / a
    return 1.0 - front * _continued_beta_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail_twice = _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    if value > 0.0:
        return 1.0 - 0.5 * tail_twice
    return 0.5 * tail_twice


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Dependency-free inverse CDF for the positive Student-t tail."""

    if not 0.5 < probability < 1.0:
        raise ValueError("Student-t quantile probability must lie in (0.5, 1)")
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ValueError("Student-t quantile could not be bracketed")
    for _ in range(96):
        midpoint = 0.5 * (lower + upper)
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _strict_integer(value: Integral, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(value: bool, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _numeric_array(value: ArrayLike, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "c", "O", "S", "U", "V"}:
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _bool_mask(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return np.asarray(result, dtype=bool)


def _hex_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value.lower()


def _block_ids(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of block identifiers")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(item, str) or not item or len(item) > 256 for item in result):
        raise ValueError(f"{name} must contain non-empty strings of at most 256 characters")
    return result


def _hash_text(digest: Any, label: str, value: str) -> None:
    encoded_label = label.encode("utf-8")
    encoded_value = value.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "little"))
    digest.update(encoded_label)
    digest.update(len(encoded_value).to_bytes(8, "little"))
    digest.update(encoded_value)


def _manifest_values(
    *,
    expected_response_sign: Integral,
    familywise_alpha: Real,
    equivalence_bound: Real,
    minimum_target_response: Real,
    maximum_training_reduced_chi_square: Real,
    maximum_covariance_condition_number: Real,
    covariance_rank_relative_tolerance: Real,
    minimum_paired_covariance_eigenvalue: Real,
    minimum_residual_mean_variance: Real,
    minimum_trials: Integral,
    observations_are_independent_blocks: bool,
    gaussian_mean_model_declared: bool,
) -> tuple[int, float, float, float, float, float, float, float, float, int, bool, bool]:
    sign = _strict_integer(expected_response_sign, name="expected_response_sign", minimum=-1)
    if sign not in {-1, 1}:
        raise ValueError("expected_response_sign must be -1 or +1")
    alpha = _probability(familywise_alpha, name="familywise_alpha")
    equivalence = _finite_positive(equivalence_bound, name="equivalence_bound")
    target_minimum = _finite_positive(
        minimum_target_response, name="minimum_target_response"
    )
    if equivalence >= target_minimum:
        raise ValueError("equivalence_bound must be smaller than minimum_target_response")
    chi_limit = _finite_positive(
        maximum_training_reduced_chi_square,
        name="maximum_training_reduced_chi_square",
    )
    condition_limit = _finite_positive(
        maximum_covariance_condition_number,
        name="maximum_covariance_condition_number",
    )
    if condition_limit < 1.0:
        raise ValueError("maximum_covariance_condition_number must be at least one")
    rank_tolerance = _probability(
        covariance_rank_relative_tolerance,
        name="covariance_rank_relative_tolerance",
    )
    paired_eigen_floor = _finite_positive(
        minimum_paired_covariance_eigenvalue,
        name="minimum_paired_covariance_eigenvalue",
    )
    residual_variance_floor = _finite_positive(
        minimum_residual_mean_variance,
        name="minimum_residual_mean_variance",
    )
    min_trials = _strict_integer(minimum_trials, name="minimum_trials", minimum=64)
    independent = _strict_bool(
        observations_are_independent_blocks,
        name="observations_are_independent_blocks",
    )
    gaussian = _strict_bool(
        gaussian_mean_model_declared,
        name="gaussian_mean_model_declared",
    )
    return (
        sign,
        alpha,
        equivalence,
        target_minimum,
        chi_limit,
        condition_limit,
        rank_tolerance,
        paired_eigen_floor,
        residual_variance_floor,
        min_trials,
        independent,
        gaussian,
    )


def resonant_mask_manifest_sha256(
    *,
    design_tensor: ArrayLike,
    training_mask: object,
    heldout_mask: object,
    prearrival_mask: object,
    off_support_mask: object,
    target_mask: object,
    matched_block_ids: Sequence[str],
    sham_block_ids: Sequence[str],
    preprocessing_artifact_sha256: str,
    design_calibration_artifact_sha256: str,
    observations_are_independent_blocks: bool,
    gaussian_mean_model_declared: bool,
    expected_response_sign: Integral = 1,
    familywise_alpha: Real = 0.05,
    equivalence_bound: Real = 0.05,
    minimum_target_response: Real = 0.5,
    maximum_training_reduced_chi_square: Real = 4.0,
    maximum_covariance_condition_number: Real = 1.0e8,
    covariance_rank_relative_tolerance: Real = 1.0e-10,
    minimum_paired_covariance_eigenvalue: Real = 1.0e-12,
    minimum_residual_mean_variance: Real = 1.0e-12,
    minimum_trials: Integral = 64,
) -> str:
    """Hash every pass-affecting design, provenance, and inference setting."""

    design = _numeric_array(design_tensor, name="design_tensor")
    if design.ndim < 2 or design.size < 4:
        raise ValueError("design_tensor must have at least two axes and four cells")
    shape = design.shape
    masks = (
        ("training", _bool_mask(training_mask, name="training_mask", shape=shape)),
        ("heldout", _bool_mask(heldout_mask, name="heldout_mask", shape=shape)),
        ("prearrival", _bool_mask(prearrival_mask, name="prearrival_mask", shape=shape)),
        ("off_support", _bool_mask(off_support_mask, name="off_support_mask", shape=shape)),
        ("target", _bool_mask(target_mask, name="target_mask", shape=shape)),
    )
    matched_ids = _block_ids(matched_block_ids, name="matched_block_ids")
    sham_ids = _block_ids(sham_block_ids, name="sham_block_ids")
    if len(matched_ids) != len(sham_ids):
        raise ValueError("matched_block_ids and sham_block_ids must have equal length")
    preprocessing_hash = _hex_digest(
        preprocessing_artifact_sha256, name="preprocessing_artifact_sha256"
    )
    calibration_hash = _hex_digest(
        design_calibration_artifact_sha256,
        name="design_calibration_artifact_sha256",
    )
    values = _manifest_values(
        expected_response_sign=expected_response_sign,
        familywise_alpha=familywise_alpha,
        equivalence_bound=equivalence_bound,
        minimum_target_response=minimum_target_response,
        maximum_training_reduced_chi_square=maximum_training_reduced_chi_square,
        maximum_covariance_condition_number=maximum_covariance_condition_number,
        covariance_rank_relative_tolerance=covariance_rank_relative_tolerance,
        minimum_paired_covariance_eigenvalue=minimum_paired_covariance_eigenvalue,
        minimum_residual_mean_variance=minimum_residual_mean_variance,
        minimum_trials=minimum_trials,
        observations_are_independent_blocks=observations_are_independent_blocks,
        gaussian_mean_model_declared=gaussian_mean_model_declared,
    )

    digest = hashlib.sha256()
    digest.update(b"resonant-spatiotemporal-mask-manifest/v2\0")
    _hash_text(digest, "cell_shape", repr(shape))
    digest.update(np.asarray(design, dtype="<f8", order="C").tobytes())
    for name, mask in masks:
        _hash_text(digest, "mask_name", name)
        digest.update(np.asarray(mask, dtype=np.uint8, order="C").tobytes())
    for name, identifiers in (
        ("matched_block_ids", matched_ids),
        ("sham_block_ids", sham_ids),
    ):
        _hash_text(digest, "identifier_sequence", name)
        for identifier in identifiers:
            _hash_text(digest, "block_id", identifier)
    _hash_text(digest, "preprocessing_artifact_sha256", preprocessing_hash)
    _hash_text(digest, "design_calibration_artifact_sha256", calibration_hash)
    value_names = (
        "expected_response_sign",
        "familywise_alpha",
        "equivalence_bound",
        "minimum_target_response",
        "maximum_training_reduced_chi_square",
        "maximum_covariance_condition_number",
        "covariance_rank_relative_tolerance",
        "minimum_paired_covariance_eigenvalue",
        "minimum_residual_mean_variance",
        "minimum_trials",
        "observations_are_independent_blocks",
        "gaussian_mean_model_declared",
    )
    for name, value in zip(value_names, values, strict=True):
        _hash_text(digest, name, repr(value))
    return digest.hexdigest()


def _covariance_diagnostics(
    matrix: np.ndarray,
    *,
    expected_rank: int,
    relative_tolerance: float,
    minimum_positive_eigenvalue: float,
    minimum_diagonal: float,
    maximum_condition_number: float,
) -> _CovarianceDiagnostics:
    matrix = np.asarray(matrix, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        return _CovarianceDiagnostics(0, None, 0.0, False)
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    symmetry_tolerance = relative_tolerance * scale
    symmetric = bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=symmetry_tolerance))
    symmetrized = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetrized)
    minimum_eigenvalue = float(np.min(eigenvalues))
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-300)
    rank_threshold = max(
        minimum_positive_eigenvalue,
        relative_tolerance * spectral_scale,
    )
    negative_tolerance = max(1.0e-15, relative_tolerance * spectral_scale)
    positive = eigenvalues[eigenvalues > rank_threshold]
    rank = int(positive.size)
    condition = None
    if positive.size:
        condition = float(np.max(positive) / np.min(positive))
    diagonal = np.diag(symmetrized)
    valid = bool(
        symmetric
        and minimum_eigenvalue >= -negative_tolerance
        and rank == expected_rank
        and condition is not None
        and math.isfinite(condition)
        and condition <= maximum_condition_number
        and np.all(np.isfinite(diagonal))
        and np.all(diagonal >= minimum_diagonal)
    )
    return _CovarianceDiagnostics(rank, condition, minimum_eigenvalue, valid)


def _standard_errors(
    covariance: np.ndarray,
    *,
    minimum_variance: float,
) -> np.ndarray | None:
    diagonal = np.diag(np.asarray(covariance, dtype=float))
    if (
        not np.all(np.isfinite(diagonal))
        or np.any(diagonal < minimum_variance)
    ):
        return None
    return np.sqrt(diagonal)


def _stage(
    *,
    manifest_pass: bool,
    gls_pass: bool,
    heldout_pass: bool,
    conditional_pass: bool,
) -> ResonantMaskStage:
    if not manifest_pass:
        return ResonantMaskStage.PAIRED_RESPONSE_CONTROL
    if not gls_pass:
        return ResonantMaskStage.FROZEN_MANIFEST_CONTROL
    if not heldout_pass:
        return ResonantMaskStage.JOINT_MASK_GLS_CONTROL
    if not conditional_pass:
        return ResonantMaskStage.CROSSED_HELDOUT_PREDICTION_CONTROL
    return ResonantMaskStage.CONDITIONAL_SPATIOTEMPORAL_RESPONSE_MASK


def _raw_arrays(
    raw: ResonantMaskRawInputs,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not isinstance(raw, ResonantMaskRawInputs):
        raise ValueError("raw_inputs must be ResonantMaskRawInputs")
    shape = raw.cell_shape
    if (
        not isinstance(shape, tuple)
        or len(shape) < 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
        or math.prod(shape) < 4
    ):
        raise ValueError("raw cell_shape must contain at least two positive axes and four cells")
    cell_count = math.prod(shape)
    design_flat = _numeric_array(raw.design_flat, name="raw design_flat")
    if design_flat.shape != (cell_count,):
        raise ValueError("raw design_flat does not match cell_shape")
    design = design_flat.reshape(shape)
    matched = _numeric_array(raw.matched_response_flat, name="raw matched_response_flat")
    sham = _numeric_array(raw.sham_response_flat, name="raw sham_response_flat")
    if (
        matched.ndim != 2
        or matched.shape != sham.shape
        or matched.shape[1] != cell_count
        or matched.shape[0] < 2
    ):
        raise ValueError("raw response tensors must contain at least two paired rows")
    masks = tuple(
        _bool_mask(value, name=name, shape=(cell_count,)).reshape(shape)
        for name, value in (
            ("raw training_mask_flat", raw.training_mask_flat),
            ("raw heldout_mask_flat", raw.heldout_mask_flat),
            ("raw prearrival_mask_flat", raw.prearrival_mask_flat),
            ("raw off_support_mask_flat", raw.off_support_mask_flat),
            ("raw target_mask_flat", raw.target_mask_flat),
        )
    )
    return (matched, sham, design, *masks)


def _build_report(raw: ResonantMaskRawInputs) -> ResonantSpatiotemporalMaskAudit:
    matched, sham, design, training, heldout, prearrival, off_support, target = (
        _raw_arrays(raw)
    )
    shape = design.shape
    cell_count = design.size
    trial_count = matched.shape[0]
    matched_ids = _block_ids(raw.matched_block_ids, name="raw matched_block_ids")
    sham_ids = _block_ids(raw.sham_block_ids, name="raw sham_block_ids")
    if len(matched_ids) != trial_count or len(sham_ids) != trial_count:
        raise ValueError("raw block identifiers must match the response trial count")
    preprocessing_hash = _hex_digest(
        raw.preprocessing_artifact_sha256,
        name="raw preprocessing_artifact_sha256",
    )
    calibration_hash = _hex_digest(
        raw.design_calibration_artifact_sha256,
        name="raw design_calibration_artifact_sha256",
    )
    declared_hash = _hex_digest(
        raw.declared_manifest_sha256,
        name="raw declared_manifest_sha256",
    )
    frozen = _strict_bool(
        raw.manifest_frozen_before_data,
        name="raw manifest_frozen_before_data",
    )
    fixed = _strict_bool(
        raw.masks_fixed_before_holdout,
        name="raw masks_fixed_before_holdout",
    )
    values = _manifest_values(
        expected_response_sign=raw.expected_response_sign,
        familywise_alpha=raw.familywise_alpha,
        equivalence_bound=raw.equivalence_bound,
        minimum_target_response=raw.minimum_target_response,
        maximum_training_reduced_chi_square=raw.maximum_training_reduced_chi_square,
        maximum_covariance_condition_number=raw.maximum_covariance_condition_number,
        covariance_rank_relative_tolerance=raw.covariance_rank_relative_tolerance,
        minimum_paired_covariance_eigenvalue=raw.minimum_paired_covariance_eigenvalue,
        minimum_residual_mean_variance=raw.minimum_residual_mean_variance,
        minimum_trials=raw.minimum_trials,
        observations_are_independent_blocks=raw.observations_are_independent_blocks,
        gaussian_mean_model_declared=raw.gaussian_mean_model_declared,
    )
    (
        sign,
        alpha,
        equivalence,
        target_minimum,
        chi_limit,
        condition_limit,
        rank_tolerance,
        paired_eigen_floor,
        residual_variance_floor,
        min_trials,
        independent_declared,
        gaussian_declared,
    ) = values

    if not np.any(design != 0.0):
        raise ValueError("zero-product design is a null control, not a mask candidate")
    if not np.any(training) or not np.any(heldout):
        raise ValueError("training_mask and heldout_mask must both be non-empty")
    if not np.any(prearrival) or not np.any(off_support) or not np.any(target):
        raise ValueError("prearrival, off_support, and target masks must be non-empty")
    pairwise_protected = not (
        np.any(prearrival & off_support)
        or np.any(prearrival & target)
        or np.any(off_support & target)
    )
    if np.any(np.abs(design[prearrival | off_support]) > 0.0):
        raise ValueError("frozen design must be zero on prearrival and off-support cells")
    if np.any(design[target] == 0.0):
        raise ValueError("frozen design must be nonzero on every target cell")

    computed_hash = resonant_mask_manifest_sha256(
        design_tensor=design,
        training_mask=training,
        heldout_mask=heldout,
        prearrival_mask=prearrival,
        off_support_mask=off_support,
        target_mask=target,
        matched_block_ids=matched_ids,
        sham_block_ids=sham_ids,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        observations_are_independent_blocks=independent_declared,
        gaussian_mean_model_declared=gaussian_declared,
        expected_response_sign=sign,
        familywise_alpha=alpha,
        equivalence_bound=equivalence,
        minimum_target_response=target_minimum,
        maximum_training_reduced_chi_square=chi_limit,
        maximum_covariance_condition_number=condition_limit,
        covariance_rank_relative_tolerance=rank_tolerance,
        minimum_paired_covariance_eigenvalue=paired_eigen_floor,
        minimum_residual_mean_variance=residual_variance_floor,
        minimum_trials=min_trials,
    )
    hash_matches = hmac.compare_digest(declared_hash, computed_hash)
    disjoint_complete = bool(not np.any(training & heldout) and np.all(training | heldout))
    protected = prearrival | off_support | target
    protected_cover = bool(np.array_equal(protected, heldout))
    manifest_pass = bool(
        hash_matches
        and frozen
        and fixed
        and disjoint_complete
        and pairwise_protected
        and protected_cover
    )

    ids_aligned = matched_ids == sham_ids
    ids_unique = len(set(matched_ids)) == trial_count and len(set(sham_ids)) == trial_count
    independent_count = len(set(matched_ids) & set(sham_ids))
    minimum_blocks_met = independent_count >= min_trials
    block_control = bool(
        ids_aligned
        and ids_unique
        and minimum_blocks_met
        and independent_declared
    )

    flat_training = training.reshape(-1)
    flat_heldout = heldout.reshape(-1)
    flat_design = design.reshape(-1)
    train_count = int(np.count_nonzero(flat_training))
    heldout_count = int(np.count_nonzero(flat_heldout))
    train_design = flat_design[flat_training]
    train_dof = train_count - 1
    training_design_non_saturated = bool(
        train_count >= 3
        and train_dof >= 2
        and np.count_nonzero(train_design) >= 2
    )

    comparison_count = int(
        train_count
        + heldout_count
        + np.count_nonzero(prearrival)
        + np.count_nonzero(off_support)
        + np.count_nonzero(target)
    )
    critical_degrees = max(independent_count - 1, 1)
    critical = _student_t_quantile(
        1.0 - alpha / (2.0 * comparison_count),
        critical_degrees,
    )
    if not math.isfinite(critical) or critical <= 0.0:
        raise ValueError("simultaneous Student-t critical value is not finite")

    paired = matched - sham
    cell_mean = np.mean(paired, axis=0)
    sample_covariance = np.asarray(np.cov(paired, rowvar=False, ddof=1), dtype=float)
    sample_covariance = np.atleast_2d(sample_covariance)
    mean_covariance = sample_covariance / trial_count
    full_diagnostics = _covariance_diagnostics(
        sample_covariance,
        expected_rank=cell_count,
        relative_tolerance=rank_tolerance,
        minimum_positive_eigenvalue=paired_eigen_floor,
        minimum_diagonal=paired_eigen_floor,
        maximum_condition_number=condition_limit,
    )

    train_covariance = mean_covariance[np.ix_(flat_training, flat_training)]
    train_diagnostics = _covariance_diagnostics(
        train_covariance,
        expected_rank=train_count,
        relative_tolerance=rank_tolerance,
        minimum_positive_eigenvalue=paired_eigen_floor / trial_count,
        minimum_diagonal=paired_eigen_floor / trial_count,
        maximum_condition_number=condition_limit,
    )
    train_mean = cell_mean[flat_training]
    amplitude: float | None = None
    amplitude_error: float | None = None
    train_reduced_chi: float | None = None
    maximum_train_residual: float | None = None
    maximum_train_upper: float | None = None
    train_residual_diagnostics = _CovarianceDiagnostics(0, None, 0.0, False)
    heldout_residual_diagnostics = _CovarianceDiagnostics(0, None, 0.0, False)
    maximum_heldout_upper: float | None = None
    weights: np.ndarray | None = None
    train_residual: np.ndarray | None = None
    train_residual_covariance: np.ndarray | None = None
    heldout_residual_covariance: np.ndarray | None = None

    if train_diagnostics.valid and training_design_non_saturated:
        precision_design = np.linalg.solve(train_covariance, train_design)
        denominator = float(train_design @ precision_design)
        if math.isfinite(denominator) and denominator > 0.0:
            weights = precision_design / denominator
            amplitude = float(weights @ train_mean)
            amplitude_error = math.sqrt(1.0 / denominator)
            train_residual = train_mean - amplitude * train_design
            train_operator = np.eye(train_count) - np.outer(train_design, weights)
            train_residual_covariance = (
                train_operator @ train_covariance @ train_operator.T
            )
            train_residual_diagnostics = _covariance_diagnostics(
                train_residual_covariance,
                expected_rank=train_dof,
                relative_tolerance=rank_tolerance,
                minimum_positive_eigenvalue=residual_variance_floor,
                minimum_diagonal=residual_variance_floor,
                maximum_condition_number=condition_limit,
            )
            train_reduced_chi = float(
                train_residual
                @ np.linalg.solve(train_covariance, train_residual)
                / train_dof
            )
            maximum_train_residual = float(np.max(np.abs(train_residual)))
            train_error = _standard_errors(
                train_residual_covariance,
                minimum_variance=residual_variance_floor,
            )
            if train_error is not None:
                maximum_train_upper = float(
                    np.max(np.abs(train_residual) + critical * train_error)
                )

            heldout_mean = cell_mean[flat_heldout]
            heldout_design = flat_design[flat_heldout]
            heldout_residual = heldout_mean - amplitude * heldout_design
            covariance_hh = mean_covariance[np.ix_(flat_heldout, flat_heldout)]
            covariance_ht = mean_covariance[np.ix_(flat_heldout, flat_training)]
            covariance_heldout_alpha = covariance_ht @ weights
            amplitude_variance = float(weights @ train_covariance @ weights)
            heldout_residual_covariance = (
                covariance_hh
                + np.outer(heldout_design, heldout_design) * amplitude_variance
                - np.outer(covariance_heldout_alpha, heldout_design)
                - np.outer(heldout_design, covariance_heldout_alpha)
            )
            heldout_residual_diagnostics = _covariance_diagnostics(
                heldout_residual_covariance,
                expected_rank=heldout_count,
                relative_tolerance=rank_tolerance,
                minimum_positive_eigenvalue=residual_variance_floor,
                minimum_diagonal=residual_variance_floor,
                maximum_condition_number=condition_limit,
            )
            heldout_error = _standard_errors(
                heldout_residual_covariance,
                minimum_variance=residual_variance_floor,
            )
            if heldout_error is not None:
                maximum_heldout_upper = float(
                    np.max(np.abs(heldout_residual) + critical * heldout_error)
                )

    training_covariance_nonvacuous = bool(
        train_diagnostics.valid and train_residual_diagnostics.valid
    )
    heldout_covariance_nonvacuous = bool(
        full_diagnostics.valid and heldout_residual_diagnostics.valid
    )
    covariance_nonvacuous = bool(
        training_covariance_nonvacuous and heldout_covariance_nonvacuous
    )
    model_assumptions = bool(block_control and gaussian_declared)
    gls_pass = bool(
        manifest_pass
        and model_assumptions
        and training_design_non_saturated
        and training_covariance_nonvacuous
        and train_reduced_chi is not None
        and train_reduced_chi <= chi_limit
        and maximum_train_upper is not None
        and maximum_train_upper <= equivalence
    )
    heldout_prediction_pass = bool(
        gls_pass
        and heldout_covariance_nonvacuous
        and maximum_heldout_upper is not None
        and maximum_heldout_upper <= equivalence
    )

    raw_error = _standard_errors(
        mean_covariance,
        minimum_variance=paired_eigen_floor / trial_count,
    ) if full_diagnostics.valid else None
    prearrival_upper: float | None = None
    off_support_upper: float | None = None
    target_lower: float | None = None
    localization_margin: float | None = None
    if raw_error is not None:
        shaped_mean = cell_mean.reshape(shape)
        shaped_error = raw_error.reshape(shape)
        prearrival_upper = float(
            np.max(np.abs(shaped_mean[prearrival]) + critical * shaped_error[prearrival])
        )
        off_support_upper = float(
            np.max(np.abs(shaped_mean[off_support]) + critical * shaped_error[off_support])
        )
        target_cell_signs = sign * np.sign(design[target])
        target_lower = float(
            np.min(
                target_cell_signs * shaped_mean[target]
                - critical * shaped_error[target]
            )
        )
        localization_margin = target_lower - off_support_upper

    prearrival_pass = bool(
        full_diagnostics.valid
        and prearrival_upper is not None
        and prearrival_upper <= equivalence
    )
    off_support_pass = bool(
        full_diagnostics.valid
        and off_support_upper is not None
        and off_support_upper <= equivalence
    )
    target_pass = bool(
        full_diagnostics.valid
        and target_lower is not None
        and target_lower >= target_minimum
    )
    localization_pass = bool(
        target_pass
        and off_support_pass
        and localization_margin is not None
        and localization_margin > 0.0
    )
    conditional_pass = bool(
        heldout_prediction_pass
        and prearrival_pass
        and off_support_pass
        and target_pass
        and localization_pass
    )

    blockers: list[str] = []
    if not hash_matches:
        blockers.append("computed design/provenance/inference hash does not match the manifest")
    if not frozen:
        blockers.append("manifest was not declared frozen before data collection")
    if not fixed:
        blockers.append("training and heldout masks were not fixed before unblinding")
    if not disjoint_complete:
        blockers.append("training and heldout masks are not a disjoint complete partition")
    if not pairwise_protected:
        blockers.append("prearrival, off-support, and target masks overlap")
    if not protected_cover:
        blockers.append("protected masks do not cover exactly the heldout cells")
    if not ids_aligned:
        blockers.append("matched and sham block identifiers are not pairwise aligned")
    if not ids_unique:
        blockers.append("block identifiers are duplicated; rows are not independent blocks")
    if not minimum_blocks_met:
        blockers.append("the preregistered minimum number of independent blocks is not met")
    if not independent_declared:
        blockers.append("independent or preblocked observation model was not declared")
    if not gaussian_declared:
        blockers.append("the conditional Gaussian mean model was not declared")
    if not training_design_non_saturated:
        blockers.append("training design is saturated or has fewer than two signal cells")
    if not training_covariance_nonvacuous:
        blockers.append("training or training-residual covariance is vacuous or ill-conditioned")
    if not full_diagnostics.valid:
        blockers.append("full paired covariance is rank deficient, vacuous, or ill-conditioned")
    if not heldout_covariance_nonvacuous:
        blockers.append("heldout residual covariance is rank deficient or vacuous")
    if (
        train_reduced_chi is None
        or maximum_train_upper is None
        or train_reduced_chi > chi_limit
        or maximum_train_upper > equivalence
    ):
        blockers.append("one-amplitude design fails its simultaneous training GLS gate")
    if maximum_heldout_upper is None or maximum_heldout_upper > equivalence:
        blockers.append("crossed heldout response is not predicted within simultaneous bounds")
    if not prearrival_pass:
        blockers.append("early-time control response exceeds the equivalence bound")
    if not off_support_pass:
        blockers.append("off-support response exceeds the spatial leakage bound")
    if not target_pass:
        blockers.append("target response lower bound does not reach the fixed minimum")
    if localization_margin is None or localization_margin <= 0.0:
        blockers.append("target lower bound does not exceed off-support upper bound")
    blockers.append(
        "response-mask control does not establish relativistic causality, identify individual "
        "factors, derive CE coupling, create matter, or verify an external manifest timestamp"
    )

    return ResonantSpatiotemporalMaskAudit(
        schema_version="resonant-spatiotemporal-mask/v2",
        raw_inputs=raw,
        cell_shape=shape,
        trial_count=trial_count,
        independent_block_count=independent_count,
        training_cell_count=train_count,
        heldout_cell_count=heldout_count,
        training_model_degrees_of_freedom=train_dof,
        simultaneous_comparison_count=comparison_count,
        simultaneous_confidence_multiplier=critical,
        manifest_sha256=declared_hash,
        computed_manifest_sha256=computed_hash,
        manifest_hash_matches=hash_matches,
        manifest_frozen_before_data=frozen,
        masks_fixed_before_holdout=fixed,
        train_holdout_disjoint_and_complete=disjoint_complete,
        protected_masks_pairwise_disjoint=pairwise_protected,
        protected_masks_cover_exactly_heldout=protected_cover,
        paired_block_ids_aligned=ids_aligned,
        paired_block_ids_unique=ids_unique,
        minimum_independent_blocks_met=minimum_blocks_met,
        independent_block_model_declared=independent_declared,
        gaussian_mean_model_declared=gaussian_declared,
        training_design_non_saturated=training_design_non_saturated,
        paired_covariance_rank=full_diagnostics.rank,
        paired_covariance_condition_number=full_diagnostics.condition_number,
        paired_covariance_minimum_eigenvalue=full_diagnostics.minimum_eigenvalue,
        training_covariance_rank=train_diagnostics.rank,
        training_covariance_condition_number=train_diagnostics.condition_number,
        training_residual_covariance_rank=train_residual_diagnostics.rank,
        training_residual_covariance_condition_number=(
            train_residual_diagnostics.condition_number
        ),
        heldout_residual_covariance_rank=heldout_residual_diagnostics.rank,
        heldout_residual_covariance_condition_number=(
            heldout_residual_diagnostics.condition_number
        ),
        training_covariance_nonvacuous=training_covariance_nonvacuous,
        heldout_covariance_nonvacuous=heldout_covariance_nonvacuous,
        covariance_nonvacuous=covariance_nonvacuous,
        fitted_global_amplitude=amplitude,
        fitted_global_amplitude_standard_error=amplitude_error,
        training_reduced_chi_square=train_reduced_chi,
        maximum_training_absolute_residual=maximum_train_residual,
        maximum_training_residual_upper_bound=maximum_train_upper,
        maximum_heldout_residual_upper_bound=maximum_heldout_upper,
        maximum_prearrival_response_upper_bound=prearrival_upper,
        maximum_off_support_response_upper_bound=off_support_upper,
        minimum_target_response_lower_bound=target_lower,
        heldout_localization_margin=localization_margin,
        joint_mask_gls_pass=gls_pass,
        heldout_prediction_pass=heldout_prediction_pass,
        prearrival_equivalence_pass=prearrival_pass,
        off_support_equivalence_pass=off_support_pass,
        target_response_pass=target_pass,
        heldout_localization_pass=localization_pass,
        factor_rescaling_counterexample_exact=True,
        individual_factor_normalizations_identifiable=False,
        conditional_spatiotemporal_response_mask=conditional_pass,
        maximum_supported_stage=_stage(
            manifest_pass=manifest_pass,
            gls_pass=gls_pass,
            heldout_pass=heldout_prediction_pass,
            conditional_pass=conditional_pass,
        ),
        first_blocker=blockers[0],
        blockers=tuple(blockers),
        claim_locks=ResonantMaskClaimLocks(),
    )


def resonant_spatiotemporal_mask_audit(
    *,
    matched_response: ArrayLike,
    sham_response: ArrayLike,
    design_tensor: ArrayLike,
    training_mask: object,
    heldout_mask: object,
    prearrival_mask: object,
    off_support_mask: object,
    target_mask: object,
    matched_block_ids: Sequence[str],
    sham_block_ids: Sequence[str],
    preprocessing_artifact_sha256: str,
    design_calibration_artifact_sha256: str,
    declared_manifest_sha256: str,
    manifest_frozen_before_data: bool,
    masks_fixed_before_holdout: bool,
    observations_are_independent_blocks: bool,
    gaussian_mean_model_declared: bool,
    expected_response_sign: Integral = 1,
    familywise_alpha: Real = 0.05,
    equivalence_bound: Real = 0.05,
    minimum_target_response: Real = 0.5,
    maximum_training_reduced_chi_square: Real = 4.0,
    maximum_covariance_condition_number: Real = 1.0e8,
    covariance_rank_relative_tolerance: Real = 1.0e-10,
    minimum_paired_covariance_eigenvalue: Real = 1.0e-12,
    minimum_residual_mean_variance: Real = 1.0e-12,
    minimum_trials: Integral = 64,
) -> ResonantSpatiotemporalMaskAudit:
    """Fit one amplitude and audit frozen crossed holdouts, failing closed."""

    design = _numeric_array(design_tensor, name="design_tensor")
    if design.ndim < 2 or design.size < 4:
        raise ValueError("design_tensor must have at least two axes and four cells")
    shape = design.shape
    matched = _numeric_array(matched_response, name="matched_response")
    sham = _numeric_array(sham_response, name="sham_response")
    if matched.shape != sham.shape or matched.ndim != design.ndim + 1:
        raise ValueError(
            "matched_response and sham_response must have shape (trial, *design_shape)"
        )
    if matched.shape[1:] != shape:
        raise ValueError("response cell shape must match design_tensor")
    if matched.shape[0] < 2:
        raise ValueError("at least two paired response rows are required")
    masks = tuple(
        _bool_mask(value, name=name, shape=shape)
        for name, value in (
            ("training_mask", training_mask),
            ("heldout_mask", heldout_mask),
            ("prearrival_mask", prearrival_mask),
            ("off_support_mask", off_support_mask),
            ("target_mask", target_mask),
        )
    )
    matched_ids = _block_ids(matched_block_ids, name="matched_block_ids")
    sham_ids = _block_ids(sham_block_ids, name="sham_block_ids")
    if len(matched_ids) != matched.shape[0] or len(sham_ids) != matched.shape[0]:
        raise ValueError("block identifier sequences must match response trial count")
    preprocessing_hash = _hex_digest(
        preprocessing_artifact_sha256, name="preprocessing_artifact_sha256"
    )
    calibration_hash = _hex_digest(
        design_calibration_artifact_sha256,
        name="design_calibration_artifact_sha256",
    )
    declared_hash = _hex_digest(
        declared_manifest_sha256, name="declared_manifest_sha256"
    )
    frozen = _strict_bool(
        manifest_frozen_before_data, name="manifest_frozen_before_data"
    )
    fixed = _strict_bool(masks_fixed_before_holdout, name="masks_fixed_before_holdout")
    values = _manifest_values(
        expected_response_sign=expected_response_sign,
        familywise_alpha=familywise_alpha,
        equivalence_bound=equivalence_bound,
        minimum_target_response=minimum_target_response,
        maximum_training_reduced_chi_square=maximum_training_reduced_chi_square,
        maximum_covariance_condition_number=maximum_covariance_condition_number,
        covariance_rank_relative_tolerance=covariance_rank_relative_tolerance,
        minimum_paired_covariance_eigenvalue=minimum_paired_covariance_eigenvalue,
        minimum_residual_mean_variance=minimum_residual_mean_variance,
        minimum_trials=minimum_trials,
        observations_are_independent_blocks=observations_are_independent_blocks,
        gaussian_mean_model_declared=gaussian_mean_model_declared,
    )
    (
        sign,
        alpha,
        equivalence,
        target_minimum,
        chi_limit,
        condition_limit,
        rank_tolerance,
        paired_eigen_floor,
        residual_variance_floor,
        min_trials,
        independent,
        gaussian,
    ) = values
    raw = ResonantMaskRawInputs(
        cell_shape=shape,
        matched_response_flat=tuple(
            tuple(float(value) for value in row) for row in matched.reshape(matched.shape[0], -1)
        ),
        sham_response_flat=tuple(
            tuple(float(value) for value in row) for row in sham.reshape(sham.shape[0], -1)
        ),
        design_flat=tuple(float(value) for value in design.reshape(-1)),
        training_mask_flat=tuple(bool(value) for value in masks[0].reshape(-1)),
        heldout_mask_flat=tuple(bool(value) for value in masks[1].reshape(-1)),
        prearrival_mask_flat=tuple(bool(value) for value in masks[2].reshape(-1)),
        off_support_mask_flat=tuple(bool(value) for value in masks[3].reshape(-1)),
        target_mask_flat=tuple(bool(value) for value in masks[4].reshape(-1)),
        matched_block_ids=matched_ids,
        sham_block_ids=sham_ids,
        preprocessing_artifact_sha256=preprocessing_hash,
        design_calibration_artifact_sha256=calibration_hash,
        declared_manifest_sha256=declared_hash,
        manifest_frozen_before_data=frozen,
        masks_fixed_before_holdout=fixed,
        observations_are_independent_blocks=independent,
        gaussian_mean_model_declared=gaussian,
        expected_response_sign=sign,
        familywise_alpha=alpha,
        equivalence_bound=equivalence,
        minimum_target_response=target_minimum,
        maximum_training_reduced_chi_square=chi_limit,
        maximum_covariance_condition_number=condition_limit,
        covariance_rank_relative_tolerance=rank_tolerance,
        minimum_paired_covariance_eigenvalue=paired_eigen_floor,
        minimum_residual_mean_variance=residual_variance_floor,
        minimum_trials=min_trials,
    )
    return validate_resonant_spatiotemporal_mask_audit(_build_report(raw))


def validate_resonant_spatiotemporal_mask_audit(
    report: ResonantSpatiotemporalMaskAudit,
) -> ResonantSpatiotemporalMaskAudit:
    """Recompute the complete certificate and reject any field-level tampering."""

    if not isinstance(report, ResonantSpatiotemporalMaskAudit):
        raise ValueError("report must be ResonantSpatiotemporalMaskAudit")
    if any(asdict(report.claim_locks).values()):
        raise ValueError("resonant-mask physical claim locks must remain false")
    expected = _build_report(report.raw_inputs)
    if report != expected:
        mismatches = tuple(
            item.name
            for item in fields(report)
            if getattr(report, item.name) != getattr(expected, item.name)
        )
        detail = ", ".join(mismatches[:4]) or "unknown field"
        raise ValueError(f"resonant-mask report differs from canonical recomputation: {detail}")
    return report


__all__ = [
    "ResonantMaskClaimLocks",
    "ResonantMaskRawInputs",
    "ResonantMaskStage",
    "ResonantSpatiotemporalMaskAudit",
    "resonant_mask_manifest_sha256",
    "resonant_spatiotemporal_mask_audit",
    "validate_resonant_spatiotemporal_mask_audit",
]
