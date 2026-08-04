"""Fail-closed controls for a Euclidean CE correlator.

The certificate starts from paired raw configurations ``O(t)`` and ``O(0)``.
It computes the unbiased connected correlator and its delete-one jackknife
estimator covariance,
checks numerical necessary conditions for a nonnegative Euclidean spectrum,
and audits single-exponential effective-mass stability.  A fixed exponential
kernel grid is also decomposed by SVD to exhibit two distinct nonnegative
spectra with the same sampled correlator whenever the inverse problem has a
null space.

These are Euclidean screening controls only.  Analytic continuation, a
Minkowski pole and residue, LSZ, and identification with the CE field remain
locked false.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from numbers import Real
from typing import Any

import numpy as np


CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV = 29.64757
EUCLIDEAN_KERNEL_ASSUMPTION = "vacuum_nonperiodic_bosonic_scalar_forward_exponential"
ENSEMBLE_SAMPLING_ASSUMPTION = "independent_configurations_or_preblocked_independent_bins"


class EuclideanCorrelatorStage(str, Enum):
    """Monotone stages below any Minkowski-particle claim."""

    REGISTERED_SCALE = "REGISTERED_SCALE"
    CONNECTED_CORRELATOR_CONTROL = "CONNECTED_CORRELATOR_CONTROL"
    POSITIVE_SPECTRUM_NECESSARY_CONTROL = "POSITIVE_SPECTRUM_NECESSARY_CONTROL"
    EUCLIDEAN_SCREENING_CONTROL = "EUCLIDEAN_SCREENING_CONTROL"


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _strict_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _real_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain real non-boolean values")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical") from error
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _as_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_matrix_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


@dataclass(frozen=True)
class PairedEuclideanEnsemble:
    """Raw paired configurations within the narrow forward-kernel scope.

    Rows must already be independent configurations or independent preblocked
    bins.  This array-only scaffold does not estimate MCMC autocorrelation.
    Times must exclude ``t=0`` contact terms.  Finite-temperature/periodic
    wraparound and fermionic or tensor reflection factors are out of scope.
    """

    euclidean_time_mev_inv: np.ndarray
    operator_t: np.ndarray
    operator_zero: np.ndarray
    kernel_assumption: str = field(
        default=EUCLIDEAN_KERNEL_ASSUMPTION,
        init=False,
    )
    sampling_assumption: str = field(
        default=ENSEMBLE_SAMPLING_ASSUMPTION,
        init=False,
    )

    def __post_init__(self) -> None:
        times = _real_array(
            self.euclidean_time_mev_inv,
            name="euclidean_time_mev_inv",
            ndim=1,
        )
        operator_t = _real_array(self.operator_t, name="operator_t", ndim=2)
        operator_zero = _real_array(
            self.operator_zero,
            name="operator_zero",
            ndim=1,
        )
        if times.size < 4:
            raise ValueError("euclidean_time_mev_inv must contain at least four times")
        if times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "euclidean_time_mev_inv must exclude t=0 contact terms and be strictly increasing"
            )
        if operator_t.shape[1] != times.size:
            raise ValueError("operator_t time dimension must match euclidean times")
        if operator_t.shape[0] != operator_zero.size:
            raise ValueError("operator_t and operator_zero configuration counts must match")
        if operator_zero.size < 2:
            raise ValueError("paired ensemble must contain at least two configurations")
        object.__setattr__(self, "euclidean_time_mev_inv", times)
        object.__setattr__(self, "operator_t", operator_t)
        object.__setattr__(self, "operator_zero", operator_zero)


@dataclass(frozen=True)
class EuclideanCorrelatorTolerances:
    """Predeclared numerical and sampling thresholds."""

    minimum_configuration_count: int = 8
    fit_window_points: int = 4
    minimum_positive_correlation: float = 1.0e-12
    maximum_subtraction_identity_relative_residual: float = 1.0e-10
    covariance_psd_relative_tolerance: float = 1.0e-10
    monotonicity_relative_tolerance: float = 1.0e-10
    log_convex_slope_tolerance: float = 1.0e-10
    hankel_psd_relative_tolerance: float = 1.0e-10
    time_uniformity_relative_tolerance: float = 1.0e-10
    maximum_window_mass_relative_drift: float = 1.0e-6
    maximum_window_log_residual: float = 1.0e-10
    covariance_rank_relative_tolerance: float = 1.0e-10
    maximum_reduced_chi_squared: float = 5.0
    maximum_window_mass_pull: float = 3.0
    maximum_registered_mass_relative_error: float = 1.0e-6
    svd_rank_relative_tolerance: float = 1.0e-12
    spectral_counterexample_relative_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_configuration_count",
            _strict_integer(
                self.minimum_configuration_count,
                name="minimum_configuration_count",
                minimum=4,
            ),
        )
        object.__setattr__(
            self,
            "fit_window_points",
            _strict_integer(
                self.fit_window_points,
                name="fit_window_points",
                minimum=3,
            ),
        )
        for field_name in self.__dataclass_fields__:
            if field_name in {"minimum_configuration_count", "fit_window_points"}:
                continue
            object.__setattr__(
                self,
                field_name,
                _positive(getattr(self, field_name), name=field_name),
            )
        bounded_relative_fields = (
            "monotonicity_relative_tolerance",
            "maximum_subtraction_identity_relative_residual",
            "covariance_psd_relative_tolerance",
            "hankel_psd_relative_tolerance",
            "time_uniformity_relative_tolerance",
            "maximum_window_mass_relative_drift",
            "covariance_rank_relative_tolerance",
            "maximum_registered_mass_relative_error",
            "svd_rank_relative_tolerance",
            "spectral_counterexample_relative_tolerance",
        )
        for field_name in bounded_relative_fields:
            if getattr(self, field_name) >= 1.0:
                raise ValueError(f"{field_name} must be less than one")


@dataclass(frozen=True)
class ConnectedCorrelatorAudit:
    configuration_count: int
    time_count: int
    mean_operator_t: tuple[float, ...]
    mean_operator_zero: float
    raw_two_point: tuple[float, ...]
    disconnected_product: tuple[float, ...]
    connected_correlator: tuple[float, ...]
    configuration_sample_covariance: tuple[tuple[float, ...], ...]
    connected_mean_covariance: tuple[tuple[float, ...], ...]
    subtraction_identity_max_residual: float
    subtraction_identity_relative_residual: float
    configuration_covariance_symmetry_relative_residual: float
    minimum_configuration_covariance_eigenvalue: float
    configuration_covariance_positive_semidefinite: bool
    connected_correlator_control_pass: bool


@dataclass(frozen=True)
class PositiveSpectrumNecessaryAudit:
    minimum_connected_correlation: float
    maximum_forward_difference: float
    minimum_log_slope_increment: float | None
    time_grid_uniform: bool
    signed_finite_difference_minima: tuple[float, ...]
    complete_monotonicity_test_available: bool
    complete_monotonicity: bool | None
    hankel_test_available: bool
    hankel_h0_order: int
    hankel_h1_order: int
    hankel_h1_minus_h2_order: int
    minimum_hankel_h0_eigenvalue: float | None
    minimum_hankel_h1_eigenvalue: float | None
    minimum_hankel_h0_minus_h1_eigenvalue: float | None
    minimum_hankel_h1_minus_h2_eigenvalue: float | None
    hankel_h0_positive_semidefinite: bool | None
    hankel_h1_positive_semidefinite: bool | None
    hankel_h0_minus_h1_positive_semidefinite: bool | None
    hankel_h1_minus_h2_positive_semidefinite: bool | None
    truncated_hausdorff_parity_condition_pass: bool | None
    connected_strictly_positive: bool
    connected_nonincreasing: bool
    log_convex: bool
    necessary_conditions_pass: bool


@dataclass(frozen=True)
class EffectiveMassWindowAudit:
    adjacent_effective_mass_mev: tuple[float, ...]
    window_mass_mev: tuple[float, ...]
    window_mass_standard_error_mev: tuple[float | None, ...]
    window_reduced_chi_squared: tuple[float | None, ...]
    window_covariance_rank: tuple[int, ...]
    window_effective_degrees_of_freedom: tuple[int, ...]
    window_covariance_positive_semidefinite: tuple[bool, ...]
    window_gls_identifiable: tuple[bool, ...]
    maximum_window_log_residual: float | None
    maximum_window_reduced_chi_squared: float | None
    maximum_window_mass_pull: float | None
    window_mass_relative_drift: float | None
    mean_window_mass_mev: float | None
    registered_mass_relative_error: float | None
    all_effective_masses_positive: bool
    window_stability_pass: bool
    registered_scale_match: bool
    covariance_aware_fit_pass: bool
    single_exponential_screening_pass: bool


@dataclass(frozen=True)
class SpectralNonuniquenessAudit:
    mass_grid_mev: tuple[float, ...]
    kernel_shape: tuple[int, int]
    singular_values: tuple[float, ...]
    numerical_rank: int
    nullity: int
    nonzero_condition_number: float
    normalization_augmented_singular_values: tuple[float, ...]
    normalization_augmented_numerical_rank: int
    normalization_augmented_nullity: int
    null_vector: tuple[float, ...]
    null_vector_kernel_relative_residual: float | None
    null_vector_normalization_residual: float | None
    base_discrete_atom_weights: tuple[float, ...]
    counterexample_epsilon: float
    minus_discrete_atom_weights: tuple[float, ...]
    plus_discrete_atom_weights: tuple[float, ...]
    minimum_counterexample_weight: float | None
    correlator_pair_max_residual: float | None
    correlator_pair_relative_residual: float | None
    total_weight_pair_residual: float | None
    two_distinct_nonnegative_normalized_weight_vectors_constructed: bool
    fixed_grid_discrete_weights_injective: bool
    normalization_constrained_fixed_grid_weights_injective: bool


@dataclass(frozen=True)
class EuclideanCorrelatorCertificate:
    schema_version: str
    registered_inverse_correlation_scale_mev: float
    euclidean_kernel_assumption: str
    ensemble_sampling_assumption: str
    tolerances: EuclideanCorrelatorTolerances
    maximum_supported_stage: EuclideanCorrelatorStage
    connected: ConnectedCorrelatorAudit | None
    positive_spectrum_necessary: PositiveSpectrumNecessaryAudit | None
    effective_mass: EffectiveMassWindowAudit | None
    spectral_nonuniqueness: SpectralNonuniquenessAudit | None
    raw_paired_ensemble_present: bool
    connected_correlator_control_pass: bool
    positive_spectrum_necessary_control_pass: bool
    euclidean_screening_control_pass: bool
    minkowski_pole_derived: bool
    positive_minkowski_residue_derived: bool
    spectral_density_uniquely_identified: bool
    physical_lsz_particle_derived: bool
    ce_field_identity_derived: bool
    first_blocker: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maximum_supported_stage"] = self.maximum_supported_stage.value
        return payload


def _connected_correlator(
    ensemble: PairedEuclideanEnsemble,
    tolerances: EuclideanCorrelatorTolerances,
) -> ConnectedCorrelatorAudit:
    operator_t = ensemble.operator_t
    operator_zero = ensemble.operator_zero
    configuration_count = operator_zero.size
    with np.errstate(over="ignore", invalid="ignore"):
        mean_t = np.mean(operator_t, axis=0)
        mean_zero = float(np.mean(operator_zero))
        raw_two_point = np.mean(operator_t * operator_zero[:, np.newaxis], axis=0)
        disconnected = mean_t * mean_zero
        centered_cross_products = (operator_t - mean_t) * (operator_zero[:, np.newaxis] - mean_zero)
        connected = np.sum(centered_cross_products, axis=0) / (configuration_count - 1)
    initial_derived = (
        mean_t,
        np.asarray(mean_zero),
        raw_two_point,
        disconnected,
        centered_cross_products,
        connected,
    )
    if not all(np.all(np.isfinite(values)) for values in initial_derived):
        raise ValueError("derived connected-correlator quantities must be finite")

    # Delete-one estimates use the same unbiased sample-covariance convention.
    # Their pseudo-values give a configuration-level sample covariance; dividing
    # it by N is the standard delete-one jackknife estimator covariance.
    leave_one_out = np.empty(
        (configuration_count, ensemble.euclidean_time_mev_inv.size),
        dtype=np.float64,
    )
    for omitted in range(configuration_count):
        retained = np.arange(configuration_count) != omitted
        retained_t = operator_t[retained]
        retained_zero = operator_zero[retained]
        retained_centered_t = retained_t - np.mean(retained_t, axis=0)
        retained_centered_zero = retained_zero - np.mean(retained_zero)
        with np.errstate(over="ignore", invalid="ignore"):
            leave_one_out[omitted] = np.sum(
                retained_centered_t * retained_centered_zero[:, np.newaxis],
                axis=0,
            ) / (configuration_count - 2)
    with np.errstate(over="ignore", invalid="ignore"):
        pseudo_values = configuration_count * connected - (configuration_count - 1) * leave_one_out
        sample_covariance = np.atleast_2d(np.cov(pseudo_values, rowvar=False, ddof=1))
        mean_covariance = sample_covariance / configuration_count
        corrected_subtraction = (configuration_count / (configuration_count - 1)) * (
            raw_two_point - disconnected
        )
    final_derived = (
        leave_one_out,
        pseudo_values,
        sample_covariance,
        mean_covariance,
        corrected_subtraction,
    )
    if not all(np.all(np.isfinite(values)) for values in final_derived):
        raise ValueError("derived jackknife quantities must be finite")

    covariance_entry_scale = max(
        float(np.max(np.abs(sample_covariance))),
        1.0e-300,
    )
    covariance_symmetry_relative_residual = float(
        np.max(np.abs(sample_covariance - sample_covariance.T)) / covariance_entry_scale
    )
    sample_covariance = 0.5 * (sample_covariance + sample_covariance.T)
    mean_covariance = 0.5 * (mean_covariance + mean_covariance.T)
    covariance_eigenvalues = np.linalg.eigvalsh(sample_covariance)
    minimum_covariance_eigenvalue = float(np.min(covariance_eigenvalues))
    covariance_scale = max(
        float(np.max(np.abs(covariance_eigenvalues))),
        1.0e-300,
    )
    covariance_psd = minimum_covariance_eigenvalue >= (
        -tolerances.covariance_psd_relative_tolerance * covariance_scale
    )
    subtraction_residual = float(np.max(np.abs(connected - corrected_subtraction)))
    connected_scale = max(float(np.max(np.abs(connected))), 1.0e-300)
    subtraction_relative_residual = subtraction_residual / connected_scale
    control_pass = (
        covariance_psd
        and covariance_symmetry_relative_residual <= tolerances.covariance_psd_relative_tolerance
        and subtraction_relative_residual
        <= tolerances.maximum_subtraction_identity_relative_residual
    )
    return ConnectedCorrelatorAudit(
        configuration_count=configuration_count,
        time_count=ensemble.euclidean_time_mev_inv.size,
        mean_operator_t=_as_tuple(mean_t),
        mean_operator_zero=mean_zero,
        raw_two_point=_as_tuple(raw_two_point),
        disconnected_product=_as_tuple(disconnected),
        connected_correlator=_as_tuple(connected),
        configuration_sample_covariance=_as_matrix_tuple(sample_covariance),
        connected_mean_covariance=_as_matrix_tuple(mean_covariance),
        subtraction_identity_max_residual=subtraction_residual,
        subtraction_identity_relative_residual=subtraction_relative_residual,
        configuration_covariance_symmetry_relative_residual=(covariance_symmetry_relative_residual),
        minimum_configuration_covariance_eigenvalue=minimum_covariance_eigenvalue,
        configuration_covariance_positive_semidefinite=covariance_psd,
        connected_correlator_control_pass=control_pass,
    )


def _positive_spectrum_audit(
    times: np.ndarray,
    connected: np.ndarray,
    tolerances: EuclideanCorrelatorTolerances,
) -> PositiveSpectrumNecessaryAudit:
    scale = max(float(np.max(np.abs(connected))), 1.0e-300)
    strictly_positive = bool(np.all(connected > tolerances.minimum_positive_correlation))
    forward_difference = np.diff(connected)
    maximum_forward = float(np.max(forward_difference))
    nonincreasing = maximum_forward <= (tolerances.monotonicity_relative_tolerance * scale)

    if strictly_positive:
        log_slopes = np.diff(np.log(connected)) / np.diff(times)
        slope_increments = np.diff(log_slopes)
        minimum_slope_increment = float(np.min(slope_increments)) if slope_increments.size else 0.0
        log_convex = minimum_slope_increment >= -tolerances.log_convex_slope_tolerance
    else:
        minimum_slope_increment = None
        log_convex = False

    time_steps = np.diff(times)
    time_scale = max(float(np.max(np.abs(time_steps))), 1.0e-300)
    uniform = bool(
        np.max(np.abs(time_steps - time_steps[0]))
        <= tolerances.time_uniformity_relative_tolerance * time_scale
    )

    if uniform:
        signed_difference = np.array(connected, copy=True)
        finite_difference_minima: list[float] = []
        complete_monotonicity = True
        while signed_difference.size:
            minimum_signed_difference = float(np.min(signed_difference))
            finite_difference_minima.append(minimum_signed_difference)
            if minimum_signed_difference < (-tolerances.monotonicity_relative_tolerance * scale):
                complete_monotonicity = False
            signed_difference = -np.diff(signed_difference)

        h0_order = (connected.size + 1) // 2
        h1_order = connected.size // 2
        h0 = np.fromfunction(
            lambda row, column: connected[(row + column).astype(int)],
            (h0_order, h0_order),
            dtype=int,
        )
        h1 = np.fromfunction(
            lambda row, column: connected[(row + column + 1).astype(int)],
            (h1_order, h1_order),
            dtype=int,
        )
        h0_for_difference = np.fromfunction(
            lambda row, column: connected[(row + column).astype(int)],
            (h1_order, h1_order),
            dtype=int,
        )
        h0_minus_h1 = h0_for_difference - h1
        h1_minus_h2_order = (connected.size - 1) // 2
        h1_minus_h2 = np.fromfunction(
            lambda row, column: (
                connected[(row + column + 1).astype(int)]
                - connected[(row + column + 2).astype(int)]
            ),
            (h1_minus_h2_order, h1_minus_h2_order),
            dtype=int,
        )

        def minimum_eigenvalue_and_psd(matrix: np.ndarray) -> tuple[float, bool]:
            eigenvalues = np.linalg.eigvalsh(matrix)
            minimum = float(np.min(eigenvalues))
            eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-300)
            positive_semidefinite = minimum >= (
                -tolerances.hankel_psd_relative_tolerance * eigenvalue_scale
            )
            return minimum, positive_semidefinite

        minimum_h0, h0_psd = minimum_eigenvalue_and_psd(h0)
        minimum_h1, h1_psd = minimum_eigenvalue_and_psd(h1)
        minimum_h0_minus_h1, h0_minus_h1_psd = minimum_eigenvalue_and_psd(h0_minus_h1)
        minimum_h1_minus_h2, h1_minus_h2_psd = minimum_eigenvalue_and_psd(h1_minus_h2)
        if connected.size % 2:
            truncated_hausdorff_pass = h0_psd and h1_minus_h2_psd
        else:
            truncated_hausdorff_pass = h1_psd and h0_minus_h1_psd
    else:
        finite_difference_minima = []
        complete_monotonicity = None
        h0_order = 0
        h1_order = 0
        h1_minus_h2_order = 0
        minimum_h0 = None
        minimum_h1 = None
        minimum_h0_minus_h1 = None
        minimum_h1_minus_h2 = None
        h0_psd = None
        h1_psd = None
        h0_minus_h1_psd = None
        h1_minus_h2_psd = None
        truncated_hausdorff_pass = None
    necessary_pass = (
        strictly_positive
        and nonincreasing
        and log_convex
        and uniform
        and complete_monotonicity is True
        and truncated_hausdorff_pass is True
    )
    return PositiveSpectrumNecessaryAudit(
        minimum_connected_correlation=float(np.min(connected)),
        maximum_forward_difference=maximum_forward,
        minimum_log_slope_increment=minimum_slope_increment,
        time_grid_uniform=uniform,
        signed_finite_difference_minima=tuple(finite_difference_minima),
        complete_monotonicity_test_available=uniform,
        complete_monotonicity=complete_monotonicity,
        hankel_test_available=uniform,
        hankel_h0_order=h0_order,
        hankel_h1_order=h1_order,
        hankel_h1_minus_h2_order=h1_minus_h2_order,
        minimum_hankel_h0_eigenvalue=minimum_h0,
        minimum_hankel_h1_eigenvalue=minimum_h1,
        minimum_hankel_h0_minus_h1_eigenvalue=minimum_h0_minus_h1,
        minimum_hankel_h1_minus_h2_eigenvalue=minimum_h1_minus_h2,
        hankel_h0_positive_semidefinite=h0_psd,
        hankel_h1_positive_semidefinite=h1_psd,
        hankel_h0_minus_h1_positive_semidefinite=h0_minus_h1_psd,
        hankel_h1_minus_h2_positive_semidefinite=h1_minus_h2_psd,
        truncated_hausdorff_parity_condition_pass=truncated_hausdorff_pass,
        connected_strictly_positive=strictly_positive,
        connected_nonincreasing=nonincreasing,
        log_convex=log_convex,
        necessary_conditions_pass=necessary_pass,
    )


def _effective_mass_audit(
    times: np.ndarray,
    connected: np.ndarray,
    connected_estimator_covariance: np.ndarray,
    *,
    registered_mass_mev: float,
    tolerances: EuclideanCorrelatorTolerances,
) -> EffectiveMassWindowAudit:
    if np.any(connected <= tolerances.minimum_positive_correlation):
        return EffectiveMassWindowAudit(
            adjacent_effective_mass_mev=(),
            window_mass_mev=(),
            window_mass_standard_error_mev=(),
            window_reduced_chi_squared=(),
            window_covariance_rank=(),
            window_effective_degrees_of_freedom=(),
            window_covariance_positive_semidefinite=(),
            window_gls_identifiable=(),
            maximum_window_log_residual=None,
            maximum_window_reduced_chi_squared=None,
            maximum_window_mass_pull=None,
            window_mass_relative_drift=None,
            mean_window_mass_mev=None,
            registered_mass_relative_error=None,
            all_effective_masses_positive=False,
            window_stability_pass=False,
            registered_scale_match=False,
            covariance_aware_fit_pass=False,
            single_exponential_screening_pass=False,
        )

    log_connected = np.log(connected)
    log_covariance = connected_estimator_covariance / np.outer(
        connected,
        connected,
    )
    log_covariance = 0.5 * (log_covariance + log_covariance.T)
    if not np.all(np.isfinite(log_covariance)):
        raise ValueError("delta-method log-correlator covariance must be finite")
    adjacent_mass = -np.diff(log_connected) / np.diff(times)
    window_masses: list[float] = []
    window_mass_errors: list[float | None] = []
    window_chi_squared: list[float | None] = []
    window_covariance_ranks: list[int] = []
    window_effective_dof: list[int] = []
    window_covariance_psd: list[bool] = []
    window_gls_identifiable: list[bool] = []
    window_screening_available: list[bool] = []
    mass_influence_rows: list[np.ndarray] = []
    window_residuals: list[float] = []
    width = tolerances.fit_window_points
    design_columns = 2
    for start in range(0, times.size - width + 1):
        time_window = times[start : start + width]
        log_window = log_connected[start : start + width]
        design = np.column_stack((np.ones(width), time_window))
        covariance_window = log_covariance[
            start : start + width,
            start : start + width,
        ]
        covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance_window)
        covariance_scale = float(np.max(np.abs(covariance_eigenvalues)))
        if covariance_scale == 0.0:
            covariance_psd = True
            supported = np.zeros(covariance_eigenvalues.size, dtype=bool)
        else:
            covariance_psd = bool(
                np.min(covariance_eigenvalues)
                >= -tolerances.covariance_psd_relative_tolerance * covariance_scale
            )
            supported = covariance_eigenvalues > (
                tolerances.covariance_rank_relative_tolerance * covariance_scale
            )
        covariance_rank = int(np.count_nonzero(supported))
        coefficient: np.ndarray
        influence_window: np.ndarray
        reduced_chi_squared: float | None
        if covariance_psd and covariance_rank >= design_columns:
            support_vectors = covariance_eigenvectors[:, supported]
            support_values = covariance_eigenvalues[supported]
            whitening = support_vectors.T / np.sqrt(support_values)[:, np.newaxis]
            whitened_design = whitening @ design
            whitened_log = whitening @ log_window
            design_singular_values = np.linalg.svd(
                whitened_design,
                compute_uv=False,
            )
            design_rank = int(
                np.count_nonzero(
                    design_singular_values
                    > tolerances.covariance_rank_relative_tolerance * design_singular_values[0]
                )
            )
        else:
            whitening = np.empty((0, width), dtype=np.float64)
            whitened_design = np.empty((0, design_columns), dtype=np.float64)
            whitened_log = np.empty(0, dtype=np.float64)
            design_rank = 0
        gls_identifiable = design_rank == design_columns
        effective_dof = covariance_rank - design_rank
        if gls_identifiable:
            whitened_pseudoinverse = np.linalg.pinv(
                whitened_design,
                rcond=tolerances.covariance_rank_relative_tolerance,
            )
            coefficient_map = whitened_pseudoinverse @ whitening
            coefficient = coefficient_map @ log_window
            influence_window = -coefficient_map[1]
            whitened_residual = whitened_log - whitened_design @ coefficient
            chi_squared = float(whitened_residual @ whitened_residual)
            reduced_chi_squared = chi_squared / effective_dof if effective_dof > 0 else None
            mass_variance = float(influence_window @ covariance_window @ influence_window)
            mass_error = math.sqrt(max(mass_variance, 0.0))
        else:
            coefficient, _, _, _ = np.linalg.lstsq(
                design,
                log_window,
                rcond=None,
            )
            influence_window = np.zeros(width, dtype=np.float64)
            reduced_chi_squared = None
            mass_error = None
        prediction = design @ coefficient
        residual = log_window - prediction
        window_masses.append(float(-coefficient[1]))
        window_mass_errors.append(mass_error)
        window_chi_squared.append(reduced_chi_squared)
        window_covariance_ranks.append(covariance_rank)
        window_effective_dof.append(effective_dof)
        window_covariance_psd.append(bool(covariance_psd))
        window_gls_identifiable.append(bool(gls_identifiable))
        window_screening_available.append(
            bool(covariance_psd and gls_identifiable and effective_dof > 0)
        )
        full_influence = np.zeros(times.size, dtype=np.float64)
        full_influence[start : start + width] = influence_window
        mass_influence_rows.append(full_influence)
        window_residuals.append(float(np.max(np.abs(residual))))

    mass_values = np.asarray(window_masses, dtype=np.float64)
    mean_mass = float(np.mean(mass_values))
    mass_scale = max(abs(mean_mass), 1.0e-300)
    relative_drift = float((np.max(mass_values) - np.min(mass_values)) / mass_scale)
    maximum_log_residual = max(window_residuals)
    if all(value is not None for value in window_chi_squared):
        maximum_reduced_chi_squared = max(
            value for value in window_chi_squared if value is not None
        )
    else:
        maximum_reduced_chi_squared = None

    maximum_mass_pull: float | None = None
    if all(window_screening_available):
        influence_matrix = np.vstack(mass_influence_rows)
        mass_covariance = influence_matrix @ log_covariance @ influence_matrix.T
        mass_covariance = 0.5 * (mass_covariance + mass_covariance.T)
        maximum_mass_pull = 0.0
        for first in range(mass_values.size):
            for second in range(first + 1, mass_values.size):
                difference_variance = float(
                    mass_covariance[first, first]
                    + mass_covariance[second, second]
                    - 2.0 * mass_covariance[first, second]
                )
                mass_difference = abs(mass_values[first] - mass_values[second])
                variance_scale = max(
                    abs(float(mass_covariance[first, first])),
                    abs(float(mass_covariance[second, second])),
                    1.0e-300,
                )
                if difference_variance < (
                    -tolerances.covariance_psd_relative_tolerance * variance_scale
                ):
                    maximum_mass_pull = None
                    break
                if difference_variance <= 0.0:
                    zero_variance_mass_tolerance = (
                        tolerances.maximum_window_mass_relative_drift
                        * max(
                            abs(mass_values[first]),
                            abs(mass_values[second]),
                            registered_mass_mev,
                        )
                    )
                    if mass_difference > zero_variance_mass_tolerance:
                        maximum_mass_pull = None
                        break
                    continue
                maximum_mass_pull = max(
                    maximum_mass_pull,
                    mass_difference / math.sqrt(difference_variance),
                )
            if maximum_mass_pull is None:
                break
    registered_error = abs(mean_mass - registered_mass_mev) / registered_mass_mev
    masses_positive = bool(np.all(adjacent_mass > 0.0) and np.all(mass_values > 0.0))
    covariance_fit_pass = bool(
        all(window_screening_available)
        and maximum_reduced_chi_squared is not None
        and maximum_reduced_chi_squared <= tolerances.maximum_reduced_chi_squared
        and maximum_mass_pull is not None
        and maximum_mass_pull <= tolerances.maximum_window_mass_pull
    )
    window_stable = bool(
        masses_positive
        and relative_drift <= tolerances.maximum_window_mass_relative_drift
        and maximum_log_residual <= tolerances.maximum_window_log_residual
        and covariance_fit_pass
    )
    registered_match = bool(registered_error <= tolerances.maximum_registered_mass_relative_error)
    return EffectiveMassWindowAudit(
        adjacent_effective_mass_mev=_as_tuple(adjacent_mass),
        window_mass_mev=tuple(window_masses),
        window_mass_standard_error_mev=tuple(window_mass_errors),
        window_reduced_chi_squared=tuple(window_chi_squared),
        window_covariance_rank=tuple(window_covariance_ranks),
        window_effective_degrees_of_freedom=tuple(window_effective_dof),
        window_covariance_positive_semidefinite=tuple(window_covariance_psd),
        window_gls_identifiable=tuple(window_gls_identifiable),
        maximum_window_log_residual=maximum_log_residual,
        maximum_window_reduced_chi_squared=maximum_reduced_chi_squared,
        maximum_window_mass_pull=maximum_mass_pull,
        window_mass_relative_drift=relative_drift,
        mean_window_mass_mev=mean_mass,
        registered_mass_relative_error=registered_error,
        all_effective_masses_positive=masses_positive,
        window_stability_pass=window_stable,
        registered_scale_match=registered_match,
        covariance_aware_fit_pass=covariance_fit_pass,
        single_exponential_screening_pass=bool(window_stable and registered_match),
    )


def _mass_grid(value: object) -> np.ndarray:
    masses = _real_array(value, name="mass_grid_mev", ndim=1)
    if masses.size < 2:
        raise ValueError("mass_grid_mev must contain at least two masses")
    if masses[0] < 0.0 or np.any(np.diff(masses) <= 0.0):
        raise ValueError("mass_grid_mev must be nonnegative and strictly increasing")
    return masses


def _spectral_nonuniqueness_audit(
    times: np.ndarray,
    masses: np.ndarray,
    tolerances: EuclideanCorrelatorTolerances,
) -> SpectralNonuniquenessAudit:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        kernel = np.exp(-np.outer(times, masses))
    if not np.all(np.isfinite(kernel)):
        raise ValueError("fixed exponential mass-grid kernel must be finite")
    _, singular_values, _ = np.linalg.svd(kernel, full_matrices=True)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        raise ValueError("fixed exponential mass-grid kernel is numerically zero")
    rank_threshold = singular_values[0] * tolerances.svd_rank_relative_tolerance
    rank = int(np.count_nonzero(singular_values > rank_threshold))
    if rank == 0:
        raise ValueError("fixed exponential mass-grid kernel has zero numerical rank")
    nullity = int(masses.size - rank)
    condition_number = float(singular_values[0] / singular_values[rank - 1])
    normalization_augmented_kernel = np.vstack((np.ones(masses.size, dtype=np.float64), kernel))
    _, augmented_singular_values, augmented_right = np.linalg.svd(
        normalization_augmented_kernel,
        full_matrices=True,
    )
    augmented_rank_threshold = augmented_singular_values[0] * tolerances.svd_rank_relative_tolerance
    augmented_rank = int(np.count_nonzero(augmented_singular_values > augmented_rank_threshold))
    augmented_nullity = int(masses.size - augmented_rank)
    base = np.full(masses.size, 1.0 / masses.size, dtype=np.float64)

    if augmented_nullity <= 0:
        null_vector = np.empty(0, dtype=np.float64)
        null_vector_residual = None
        normalization_residual = None
        epsilon = 0.0
        weights_minus = np.empty(0, dtype=np.float64)
        weights_plus = np.empty(0, dtype=np.float64)
        minimum_weight = None
        maximum_residual = None
        relative_residual = None
        total_weight_residual = None
        constructed = False
    else:
        null_vector = augmented_right[augmented_rank]
        kernel_null_vector = kernel @ null_vector
        kernel_scale = max(float(np.max(np.abs(kernel))), 1.0e-300)
        null_vector_residual = float(np.max(np.abs(kernel_null_vector)) / kernel_scale)
        normalization_residual = abs(float(np.sum(null_vector)))
        support = np.abs(null_vector) > 1.0e-15
        epsilon_limit = float(np.min(base[support] / np.abs(null_vector[support])))
        epsilon = 0.25 * epsilon_limit
        weights_minus = base - epsilon * null_vector
        weights_plus = base + epsilon * null_vector
        correlator_minus = kernel @ weights_minus
        correlator_plus = kernel @ weights_plus
        maximum_residual = float(np.max(np.abs(correlator_plus - correlator_minus)))
        correlator_scale = max(
            float(np.max(np.abs(correlator_plus))),
            float(np.max(np.abs(correlator_minus))),
            1.0e-300,
        )
        relative_residual = maximum_residual / correlator_scale
        minimum_weight = float(min(np.min(weights_minus), np.min(weights_plus)))
        total_weight_residual = abs(float(np.sum(weights_plus) - np.sum(weights_minus)))
        constructed = (
            masses.size > rank
            and masses.size > augmented_rank
            and epsilon > 0.0
            and minimum_weight >= 0.0
            and null_vector_residual <= tolerances.spectral_counterexample_relative_tolerance
            and normalization_residual <= tolerances.spectral_counterexample_relative_tolerance
            and relative_residual <= tolerances.spectral_counterexample_relative_tolerance
            and total_weight_residual <= tolerances.spectral_counterexample_relative_tolerance
            and not np.array_equal(weights_minus, weights_plus)
        )
    return SpectralNonuniquenessAudit(
        mass_grid_mev=_as_tuple(masses),
        kernel_shape=(int(kernel.shape[0]), int(kernel.shape[1])),
        singular_values=_as_tuple(singular_values),
        numerical_rank=rank,
        nullity=nullity,
        nonzero_condition_number=condition_number,
        normalization_augmented_singular_values=_as_tuple(augmented_singular_values),
        normalization_augmented_numerical_rank=augmented_rank,
        normalization_augmented_nullity=augmented_nullity,
        null_vector=_as_tuple(null_vector),
        null_vector_kernel_relative_residual=null_vector_residual,
        null_vector_normalization_residual=normalization_residual,
        base_discrete_atom_weights=_as_tuple(base),
        counterexample_epsilon=epsilon,
        minus_discrete_atom_weights=_as_tuple(weights_minus),
        plus_discrete_atom_weights=_as_tuple(weights_plus),
        minimum_counterexample_weight=minimum_weight,
        correlator_pair_max_residual=maximum_residual,
        correlator_pair_relative_residual=relative_residual,
        total_weight_pair_residual=total_weight_residual,
        two_distinct_nonnegative_normalized_weight_vectors_constructed=constructed,
        fixed_grid_discrete_weights_injective=rank == masses.size,
        normalization_constrained_fixed_grid_weights_injective=(augmented_rank == masses.size),
    )


def euclidean_correlator_certificate(
    *,
    registered_inverse_correlation_scale_mev: Real = (CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV),
    ensemble: PairedEuclideanEnsemble | None = None,
    mass_grid_mev: object | None = None,
    tolerances: EuclideanCorrelatorTolerances | None = None,
) -> EuclideanCorrelatorCertificate:
    """Compute the highest Euclidean control stage from raw paired samples."""

    registered_mass = _positive(
        registered_inverse_correlation_scale_mev,
        name="registered_inverse_correlation_scale_mev",
    )
    thresholds = tolerances or EuclideanCorrelatorTolerances()
    if not isinstance(thresholds, EuclideanCorrelatorTolerances):
        raise ValueError("tolerances must be EuclideanCorrelatorTolerances")
    if ensemble is not None and not isinstance(ensemble, PairedEuclideanEnsemble):
        raise ValueError("ensemble must be PairedEuclideanEnsemble or None")
    if ensemble is None and mass_grid_mev is not None:
        raise ValueError("mass_grid_mev cannot be audited without an ensemble")

    if ensemble is None:
        blockers = (
            "raw paired O(t), O(0) ensemble is absent",
            "delete-one covariance assumes independent configurations or preblocked bins",
            "periodic wraparound, contact terms, and non-scalar reflection factors are out of scope",
            "Kallen-Lehmann spectral positivity is not derived",
            "analytic continuation and a Minkowski pole are not derived",
            "an asymptotic state and LSZ reduction are not derived",
            "the sampled operator is not identified with the CE field",
        )
        return EuclideanCorrelatorCertificate(
            schema_version="1.0",
            registered_inverse_correlation_scale_mev=registered_mass,
            euclidean_kernel_assumption=EUCLIDEAN_KERNEL_ASSUMPTION,
            ensemble_sampling_assumption=ENSEMBLE_SAMPLING_ASSUMPTION,
            tolerances=thresholds,
            maximum_supported_stage=EuclideanCorrelatorStage.REGISTERED_SCALE,
            connected=None,
            positive_spectrum_necessary=None,
            effective_mass=None,
            spectral_nonuniqueness=None,
            raw_paired_ensemble_present=False,
            connected_correlator_control_pass=False,
            positive_spectrum_necessary_control_pass=False,
            euclidean_screening_control_pass=False,
            minkowski_pole_derived=False,
            positive_minkowski_residue_derived=False,
            spectral_density_uniquely_identified=False,
            physical_lsz_particle_derived=False,
            ce_field_identity_derived=False,
            first_blocker=blockers[0],
            blockers=blockers,
        )

    if ensemble.operator_zero.size < thresholds.minimum_configuration_count:
        raise ValueError("paired ensemble configuration count is below the predeclared minimum")
    if thresholds.fit_window_points > ensemble.euclidean_time_mev_inv.size:
        raise ValueError("fit_window_points exceeds the Euclidean time count")
    if mass_grid_mev is None:
        raise ValueError("mass_grid_mev is required when an ensemble is supplied")
    masses = _mass_grid(mass_grid_mev)

    connected_audit = _connected_correlator(ensemble, thresholds)
    connected_values = np.asarray(connected_audit.connected_correlator)
    positive_audit = _positive_spectrum_audit(
        ensemble.euclidean_time_mev_inv,
        connected_values,
        thresholds,
    )
    effective_audit = _effective_mass_audit(
        ensemble.euclidean_time_mev_inv,
        connected_values,
        np.asarray(connected_audit.connected_mean_covariance),
        registered_mass_mev=registered_mass,
        tolerances=thresholds,
    )
    spectral_audit = _spectral_nonuniqueness_audit(
        ensemble.euclidean_time_mev_inv,
        masses,
        thresholds,
    )

    connected_pass = connected_audit.connected_correlator_control_pass
    positive_pass = connected_pass and positive_audit.necessary_conditions_pass
    screening_pass = positive_pass and effective_audit.single_exponential_screening_pass
    stage = EuclideanCorrelatorStage.REGISTERED_SCALE
    if connected_pass:
        stage = EuclideanCorrelatorStage.CONNECTED_CORRELATOR_CONTROL
    if connected_pass and positive_pass:
        stage = EuclideanCorrelatorStage.POSITIVE_SPECTRUM_NECESSARY_CONTROL
    if screening_pass:
        stage = EuclideanCorrelatorStage.EUCLIDEAN_SCREENING_CONTROL

    blockers: list[str] = []
    if not connected_pass:
        blockers.append("connected subtraction identity or jackknife covariance control failed")
    elif not positive_pass:
        blockers.append("connected correlator fails a positive-spectrum necessary condition")
    elif not effective_audit.window_stability_pass:
        blockers.append("single-exponential effective mass is not window stable")
    elif not effective_audit.registered_scale_match:
        blockers.append("screening mass does not match the registered inverse scale")
    blockers.extend(
        (
            "Kallen-Lehmann spectral positivity is not derived",
            "delete-one covariance assumes independent configurations or preblocked bins",
            "periodic wraparound, contact terms, and non-scalar reflection factors are out of scope",
            "analytic continuation and a Minkowski pole are not derived",
            "an asymptotic state and LSZ reduction are not derived",
            "the sampled operator is not identified with the CE field",
        )
    )
    return EuclideanCorrelatorCertificate(
        schema_version="1.0",
        registered_inverse_correlation_scale_mev=registered_mass,
        euclidean_kernel_assumption=EUCLIDEAN_KERNEL_ASSUMPTION,
        ensemble_sampling_assumption=ENSEMBLE_SAMPLING_ASSUMPTION,
        tolerances=thresholds,
        maximum_supported_stage=stage,
        connected=connected_audit,
        positive_spectrum_necessary=positive_audit,
        effective_mass=effective_audit,
        spectral_nonuniqueness=spectral_audit,
        raw_paired_ensemble_present=True,
        connected_correlator_control_pass=connected_pass,
        positive_spectrum_necessary_control_pass=positive_pass,
        euclidean_screening_control_pass=screening_pass,
        minkowski_pole_derived=False,
        positive_minkowski_residue_derived=False,
        spectral_density_uniquely_identified=False,
        physical_lsz_particle_derived=False,
        ce_field_identity_derived=False,
        first_blocker=blockers[0],
        blockers=tuple(blockers),
    )


def current_ce_euclidean_correlator_certificate() -> EuclideanCorrelatorCertificate:
    """Return the current CE state: a registered scale without raw ensemble."""

    return euclidean_correlator_certificate()
