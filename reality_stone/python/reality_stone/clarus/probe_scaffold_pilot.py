"""Strict pilot gates for probe-relative dressing versus a public scaffold.

The module operationalizes a deliberately modest question:

* does a controller produce a response specific to probe A while a calibrated
  reference probe remains equivalent to its null control;
* does that selectivity track independently measured phase locking under a
  held-out-designated noise sweep;
* after the pump is off, is there a common environmental response kernel that
  predicts a held-out third probe; and
* is the measured energy budget closed without vacuous uncertainties?

Passing every gate yields only a conditional public-response-kernel candidate,
not a public scaffold.  It does not derive new matter, a boundary condition, a
covariant stress tensor, or a wormhole source.  In particular, narrative or
observer language never enters the numerical pass conditions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real


SCHEMA_VERSION = "clarus-probe-scaffold-pilot/v1"
MIN_REQUESTED_CONFIDENCE_MULTIPLIER = 1.96
MIN_CONFIDENCE_MULTIPLIER = 3.182446305284263
MIN_SAMPLE_MEAN_MULTIPLIER = MIN_CONFIDENCE_MULTIPLIER
MIN_REGRESSION_MULTIPLIER = 4.302652729911275


@dataclass(frozen=True)
class EffectInterval:
    observation_count: int
    mean_effect: float
    standard_error: float
    lower_bound: float
    upper_bound: float
    confidence_multiplier: float


@dataclass(frozen=True)
class PhaseLockAudit:
    phase_sample_count: int
    effective_sample_size: float
    raw_resultant_length: float
    bias_corrected_resultant_length: float
    circular_mean_phase_rad: float
    sufficient_effective_samples: bool
    effective_sample_size_method: str
    time_autocorrelation_corrected: bool


@dataclass(frozen=True)
class ProbeSelectivityAudit:
    probe_a_effect: EffectInterval
    reference_probe_effect: EffectInterval
    difference_in_differences: EffectInterval
    expected_response_sign: int
    minimum_probe_a_effect: float
    reference_equivalence_bound: float
    minimum_selective_effect: float
    probe_a_response_detected: bool
    reference_equivalent_to_zero: bool
    selective_contrast_detected: bool
    private_dressing_conditionally_supported: bool
    public_environment_change_implied: bool


@dataclass(frozen=True)
class PhaseNoiseSweepPoint:
    noise_strength: float
    phase_offsets_rad: tuple[float, ...]
    probe_a_pump_on_matched: tuple[float, ...]
    probe_a_pump_on_sham: tuple[float, ...]
    probe_a_pump_off_matched: tuple[float, ...]
    probe_a_pump_off_sham: tuple[float, ...]
    reference_pump_on_matched: tuple[float, ...]
    reference_pump_on_sham: tuple[float, ...]
    reference_pump_off_matched: tuple[float, ...]
    reference_pump_off_sham: tuple[float, ...]
    held_out: bool = False


@dataclass(frozen=True)
class PhaseNoiseSweepPointAudit:
    raw_point: PhaseNoiseSweepPoint
    noise_strength: float
    held_out: bool
    phase_lock: PhaseLockAudit
    selectivity: ProbeSelectivityAudit


@dataclass(frozen=True)
class PhaseNoiseSweepAudit:
    points: tuple[PhaseNoiseSweepPointAudit, ...]
    expected_response_sign: int
    minimum_probe_a_effect: float
    reference_equivalence_bound: float
    minimum_selective_effect: float
    requested_confidence_multiplier: float
    noise_phase_correlation: float | None
    phase_selectivity_correlation: float | None
    phase_lock_span: float
    selective_response_span: float
    conservative_high_to_low_response_drop: float
    minimum_absolute_correlation: float
    minimum_phase_lock_span: float
    minimum_selective_response_span: float
    minimum_conservative_response_drop: float
    heldout_prediction_equivalence_bound: float
    heldout_noise_strength: float
    heldout_predicted_selective_response: float
    heldout_prediction_residual: EffectInterval
    noise_values_unique: bool
    phase_lock_decreases_with_noise: bool
    selective_response_tracks_phase_lock: bool
    reference_equivalent_at_every_point: bool
    highest_coherence_selectivity_passes: bool
    high_coherence_response_exceeds_low_coherence_response: bool
    heldout_response_matches_designated_prediction: bool
    phase_lock_dependence_conditionally_supported: bool
    noise_to_phase_lock_dynamics_derived: bool
    causation_by_phase_lock_derived: bool


@dataclass(frozen=True)
class CommonKernelProbeReadout:
    probe_id: str
    calibrated_response_gain: float
    post_pump_response: tuple[float, ...]
    post_pump_sham: tuple[float, ...]
    pre_pump_response: tuple[float, ...]
    pre_pump_sham: tuple[float, ...]
    held_out: bool = False


@dataclass(frozen=True)
class CommonKernelProbeAudit:
    raw_readout: CommonKernelProbeReadout
    probe_id: str
    calibrated_response_gain: float
    held_out: bool
    raw_post_effect: EffectInterval
    normalized_post_kernel: EffectInterval
    normalized_pre_kernel: EffectInterval


@dataclass(frozen=True)
class PostPumpCommonKernelAudit:
    probe_audits: tuple[CommonKernelProbeAudit, ...]
    fitted_training_kernel: EffectInterval
    heldout_kernel_residual: EffectInterval
    residual_drive_monitor: EffectInterval
    nuisance_monitor: EffectInterval
    raw_residual_drive_monitor_post: tuple[float, ...]
    raw_residual_drive_monitor_sham: tuple[float, ...]
    raw_nuisance_monitor_post: tuple[float, ...]
    raw_nuisance_monitor_sham: tuple[float, ...]
    pump_start_time_s: float
    pump_off_time_s: float
    post_readout_start_time_s: float
    post_readout_end_time_s: float
    pump_off_dwell_s: float
    minimum_pump_off_dwell_s: float
    minimum_common_kernel: float
    kernel_factorization_equivalence_bound: float
    pre_response_equivalence_bound: float
    monitor_equivalence_bound: float
    residual_drive_to_kernel_gain_upper_bound: float
    residual_drive_kernel_explanation_upper_bound: float
    nuisance_to_kernel_gain_upper_bound: float
    nuisance_kernel_explanation_upper_bound: float
    apparatus_memory_kernel_upper_bound: float
    minimum_unexplained_kernel_margin: float
    probe_correlation_model: str
    probe_covariance_measured: bool
    time_ordering_valid: bool
    minimum_dwell_met: bool
    all_pre_responses_equivalent_to_zero: bool
    residual_drive_equivalent_to_zero: bool
    nuisance_monitor_equivalent_to_zero: bool
    monitors_and_apparatus_memory_cannot_explain_kernel: bool
    training_probes_factorize_common_kernel: bool
    heldout_probe_matches_common_kernel: bool
    common_post_pump_kernel_nonzero: bool
    heldout_probe_designation_declared: bool
    calibration_fixed_before_pump: bool
    blind_analysis_declared: bool
    separate_heldout_readout_chain_declared: bool
    independence_metadata_declared_complete: bool
    post_pump_persistence_conditionally_supported: bool
    heldout_separate_chain_response_conditionally_supported: bool
    physical_material_phase_derived: bool


@dataclass(frozen=True)
class EnergyLedgerAudit:
    trial_count: int
    mean_pump_work_j: float
    mean_controller_work_j: float
    mean_probe_work_j: float
    mean_transfer_work_j: float
    mean_preexisting_reservoir_release_j: float
    mean_candidate_decoupled_energy_j: float
    mean_radiated_energy_j: float
    mean_thermal_mechanical_energy_j: float
    mean_reservoir_storage_j: float
    mean_recovered_work_j: float
    mean_balance_residual_j: float
    balance_residual_interval: EffectInterval
    absolute_closure_tolerance_j: float
    signed_mean_energy_vector_j: tuple[float, ...]
    signed_ledger_signs: tuple[int, ...]
    energy_channel_values_j: tuple[tuple[float, ...], ...]
    energy_covariance_j2: tuple[tuple[float, ...], ...]
    minimum_channel_values_j: tuple[float, ...]
    declared_covariance_balance_sigma_j: float
    total_balance_sigma_j: float
    relative_balance_residual: float
    relative_balance_uncertainty: float
    maximum_relative_balance_residual: float
    maximum_relative_uncertainty: float
    all_channels_nonnegative: bool
    covariance_symmetric_positive_semidefinite: bool
    balance_statistically_consistent_with_zero: bool
    balance_residual_small: bool
    uncertainty_nonvacuous: bool
    pump_and_controller_decoupled_at_endpoint_declared: bool
    energy_ledger_closed_conditionally: bool
    microscopic_energy_transfer_mechanism_derived: bool


@dataclass(frozen=True)
class ProbeScaffoldClaimLocks:
    religious_narrative_is_physical_evidence: bool = False
    consciousness_changes_physics_derived: bool = False
    new_material_derived: bool = False
    boundary_condition_derived: bool = False
    transferable_sample_scaffold_derived: bool = False
    covariant_stress_tensor_derived: bool = False
    wormhole_source_derived: bool = False


@dataclass(frozen=True)
class ProbeScaffoldPilotReport:
    schema_version: str
    phase_noise_sweep: PhaseNoiseSweepAudit
    post_pump_common_kernel: PostPumpCommonKernelAudit
    energy_ledger: EnergyLedgerAudit
    maximum_private_branch_stage: str
    maximum_public_branch_stage: str
    conditional_public_response_kernel_candidate: bool
    conditional_public_scaffold_candidate: bool
    physical_public_scaffold_derived: bool
    claim_locks: ProbeScaffoldClaimLocks


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_positive(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonzero(value: Real, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be nonzero")
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


def _strict_bool_fields(value: object, names: Sequence[str], *, prefix: str) -> None:
    for name in names:
        _strict_bool(getattr(value, name), name=f"{prefix}.{name}")


def _strict_sign(value: Integral) -> int:
    sign = _strict_integer(value, name="expected_response_sign", minimum=-1)
    if sign not in {-1, 1}:
        raise ValueError("expected_response_sign must be -1 or +1")
    return sign


def _finite_series(
    values: Iterable[Real],
    *,
    name: str,
    minimum_count: int = 4,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of real scalars")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of real scalars") from error
    if len(raw) < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} observations")
    return tuple(_finite_real(item, name=f"{name}[{index}]") for index, item in enumerate(raw))


def _paired_differences(
    treatment: Iterable[Real],
    control: Iterable[Real],
    *,
    treatment_name: str,
    control_name: str,
    minimum_count: int = 4,
) -> tuple[float, ...]:
    treated = _finite_series(treatment, name=treatment_name, minimum_count=minimum_count)
    baseline = _finite_series(control, name=control_name, minimum_count=minimum_count)
    if len(treated) != len(baseline):
        raise ValueError(f"{treatment_name} and {control_name} must have equal length")
    return tuple(on - off for on, off in zip(treated, baseline, strict=True))


def _interval_from_mean(
    *,
    observation_count: int,
    mean_effect: float,
    standard_error: float,
    confidence_multiplier: float,
) -> EffectInterval:
    count = _strict_integer(observation_count, name="observation_count", minimum=1)
    mean = _finite_real(mean_effect, name="mean_effect")
    error = _finite_nonnegative(standard_error, name="standard_error")
    multiplier = _finite_positive(confidence_multiplier, name="confidence_multiplier")
    if multiplier < MIN_CONFIDENCE_MULTIPLIER:
        raise ValueError(
            f"confidence_multiplier must be at least {MIN_CONFIDENCE_MULTIPLIER}"
        )
    radius = multiplier * error
    return EffectInterval(
        observation_count=count,
        mean_effect=mean,
        standard_error=error,
        lower_bound=mean - radius,
        upper_bound=mean + radius,
        confidence_multiplier=multiplier,
    )


def _effect_interval(
    effects: Iterable[Real],
    *,
    confidence_multiplier: float,
    name: str,
    additional_standard_error: float = 0.0,
) -> EffectInterval:
    samples = _finite_series(effects, name=name)
    requested_multiplier = _finite_positive(
        confidence_multiplier,
        name="confidence_multiplier",
    )
    if requested_multiplier < MIN_REQUESTED_CONFIDENCE_MULTIPLIER:
        raise ValueError(
            "confidence_multiplier must request at least a two-sided 95% interval"
        )
    # Every raw series has at least four observations.  The df=3 two-sided 95%
    # Student-t critical value is a conservative floor for all longer series.
    multiplier = max(requested_multiplier, MIN_SAMPLE_MEAN_MULTIPLIER)
    extra_error = _finite_nonnegative(
        additional_standard_error,
        name="additional_standard_error",
    )
    mean = math.fsum(samples) / len(samples)
    squared = math.fsum((item - mean) ** 2 for item in samples)
    sample_variance = squared / (len(samples) - 1)
    sampling_error = math.sqrt(sample_variance / len(samples))
    # Independence between trial scatter and calibration error is not encoded.
    # The triangle bound is therefore the safe default.
    total_error = sampling_error + extra_error
    return _interval_from_mean(
        observation_count=len(samples),
        mean_effect=mean,
        standard_error=total_error,
        confidence_multiplier=multiplier,
    )


def paired_effect_audit(
    treatment: Iterable[Real],
    control: Iterable[Real],
    *,
    confidence_multiplier: float = 1.96,
) -> EffectInterval:
    """Return a paired mean-effect interval from raw treatment/control readings."""

    differences = _paired_differences(
        treatment,
        control,
        treatment_name="treatment",
        control_name="control",
    )
    return _effect_interval(
        differences,
        confidence_multiplier=confidence_multiplier,
        name="paired_effect",
    )


def _equivalent_to_zero(interval: EffectInterval, bound: float) -> bool:
    equivalence = _finite_nonnegative(bound, name="equivalence_bound")
    return interval.lower_bound >= -equivalence and interval.upper_bound <= equivalence


def phase_lock_order_parameter(
    phase_offsets_rad: Iterable[Real],
    *,
    weights: Iterable[Real] | None = None,
    minimum_effective_samples: int = 4,
) -> PhaseLockAudit:
    """Compute the weighted circular resultant `|sum w exp(i*phase)|/sum w`."""

    phases = _finite_series(phase_offsets_rad, name="phase_offsets_rad")
    required = _strict_integer(
        minimum_effective_samples,
        name="minimum_effective_samples",
        minimum=2,
    )
    if weights is None:
        normalized_weights = (1.0,) * len(phases)
    else:
        normalized_weights = _finite_series(weights, name="weights")
        if len(normalized_weights) != len(phases):
            raise ValueError("weights and phase_offsets_rad must have equal length")
        if any(weight < 0.0 for weight in normalized_weights):
            raise ValueError("weights must be nonnegative")
    weight_sum = math.fsum(normalized_weights)
    squared_weight_sum = math.fsum(weight * weight for weight in normalized_weights)
    if weight_sum <= 0.0 or squared_weight_sum <= 0.0:
        raise ValueError("weights must have positive total weight")
    real_part = math.fsum(
        weight * math.cos(phase)
        for phase, weight in zip(phases, normalized_weights, strict=True)
    )
    imaginary_part = math.fsum(
        weight * math.sin(phase)
        for phase, weight in zip(phases, normalized_weights, strict=True)
    )
    resultant = min(1.0, math.hypot(real_part, imaginary_part) / weight_sum)
    effective_count = weight_sum * weight_sum / squared_weight_sum
    if effective_count <= 1.0:
        bias_corrected = 0.0
    else:
        bias_corrected_squared = max(
            0.0,
            (effective_count * resultant * resultant - 1.0) / (effective_count - 1.0),
        )
        bias_corrected = math.sqrt(bias_corrected_squared)
    return PhaseLockAudit(
        phase_sample_count=len(phases),
        effective_sample_size=effective_count,
        raw_resultant_length=resultant,
        bias_corrected_resultant_length=bias_corrected,
        circular_mean_phase_rad=math.atan2(imaginary_part, real_part),
        sufficient_effective_samples=effective_count >= required,
        effective_sample_size_method="KISH_WEIGHT_ONLY",
        time_autocorrelation_corrected=False,
    )


def _factorial_interaction(
    pump_on_matched: Iterable[Real],
    pump_on_sham: Iterable[Real],
    pump_off_matched: Iterable[Real],
    pump_off_sham: Iterable[Real],
    *,
    name: str,
) -> tuple[float, ...]:
    on_contrast = _paired_differences(
        pump_on_matched,
        pump_on_sham,
        treatment_name=f"{name}.pump_on_matched",
        control_name=f"{name}.pump_on_sham",
    )
    off_contrast = _paired_differences(
        pump_off_matched,
        pump_off_sham,
        treatment_name=f"{name}.pump_off_matched",
        control_name=f"{name}.pump_off_sham",
    )
    if len(on_contrast) != len(off_contrast):
        raise ValueError(f"{name} pump-on and pump-off contrasts must have equal length")
    return tuple(
        on - off for on, off in zip(on_contrast, off_contrast, strict=True)
    )


def probe_selectivity_audit(
    probe_a_pump_on_matched: Iterable[Real],
    probe_a_pump_on_sham: Iterable[Real],
    probe_a_pump_off_matched: Iterable[Real],
    probe_a_pump_off_sham: Iterable[Real],
    reference_pump_on_matched: Iterable[Real],
    reference_pump_on_sham: Iterable[Real],
    reference_pump_off_matched: Iterable[Real],
    reference_pump_off_sham: Iterable[Real],
    *,
    expected_response_sign: int = 1,
    minimum_probe_a_effect: float = 0.5,
    reference_equivalence_bound: float = 0.2,
    minimum_selective_effect: float = 0.5,
    confidence_multiplier: float = 1.96,
) -> ProbeSelectivityAudit:
    """Audit a pump x controller interaction and its A-minus-reference contrast."""

    sign = _strict_sign(expected_response_sign)
    minimum_a = _finite_positive(minimum_probe_a_effect, name="minimum_probe_a_effect")
    reference_bound = _finite_nonnegative(
        reference_equivalence_bound,
        name="reference_equivalence_bound",
    )
    minimum_selectivity = _finite_positive(
        minimum_selective_effect,
        name="minimum_selective_effect",
    )
    if 2.0 * reference_bound > min(minimum_a, minimum_selectivity):
        raise ValueError(
            "reference_equivalence_bound must not exceed half the smaller "
            "minimum response effect"
        )
    multiplier = _finite_positive(confidence_multiplier, name="confidence_multiplier")
    a_raw = _factorial_interaction(
        probe_a_pump_on_matched,
        probe_a_pump_on_sham,
        probe_a_pump_off_matched,
        probe_a_pump_off_sham,
        name="probe_a",
    )
    b_raw = _factorial_interaction(
        reference_pump_on_matched,
        reference_pump_on_sham,
        reference_pump_off_matched,
        reference_pump_off_sham,
        name="reference_probe",
    )
    if len(a_raw) != len(b_raw):
        raise ValueError("probe A and reference probe must have equal paired trial counts")
    oriented_a = tuple(sign * item for item in a_raw)
    oriented_b = tuple(sign * item for item in b_raw)
    selective = tuple(a - b for a, b in zip(oriented_a, oriented_b, strict=True))
    a_effect = _effect_interval(
        oriented_a,
        confidence_multiplier=multiplier,
        name="probe_a_effect",
    )
    b_effect = _effect_interval(
        oriented_b,
        confidence_multiplier=multiplier,
        name="reference_probe_effect",
    )
    difference = _effect_interval(
        selective,
        confidence_multiplier=multiplier,
        name="difference_in_differences",
    )
    a_detected = a_effect.lower_bound >= minimum_a
    reference_null = _equivalent_to_zero(b_effect, reference_bound)
    selective_detected = difference.lower_bound >= minimum_selectivity
    conditional = a_detected and reference_null and selective_detected
    return ProbeSelectivityAudit(
        probe_a_effect=a_effect,
        reference_probe_effect=b_effect,
        difference_in_differences=difference,
        expected_response_sign=sign,
        minimum_probe_a_effect=minimum_a,
        reference_equivalence_bound=reference_bound,
        minimum_selective_effect=minimum_selectivity,
        probe_a_response_detected=a_detected,
        reference_equivalent_to_zero=reference_null,
        selective_contrast_detected=selective_detected,
        private_dressing_conditionally_supported=conditional,
        public_environment_change_implied=False,
    )


def _pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation inputs must have equal length of at least two")
    first_mean = math.fsum(first) / len(first)
    second_mean = math.fsum(second) / len(second)
    first_deviation = tuple(item - first_mean for item in first)
    second_deviation = tuple(item - second_mean for item in second)
    denominator = math.sqrt(
        math.fsum(item * item for item in first_deviation)
        * math.fsum(item * item for item in second_deviation)
    )
    if denominator == 0.0:
        return None
    numerator = math.fsum(
        left * right
        for left, right in zip(first_deviation, second_deviation, strict=True)
    )
    return max(-1.0, min(1.0, numerator / denominator))


def _linear_heldout_prediction(
    training_phase: Sequence[float],
    training_response: Sequence[float],
    training_standard_errors: Sequence[float],
    *,
    heldout_phase: float,
    heldout_response: EffectInterval,
    confidence_multiplier: float,
) -> tuple[float, EffectInterval]:
    if (
        len(training_phase) != len(training_response)
        or len(training_phase) != len(training_standard_errors)
        or len(training_phase) < 4
    ):
        raise ValueError("linear heldout prediction requires at least four training points")
    phase_mean = math.fsum(training_phase) / len(training_phase)
    response_mean = math.fsum(training_response) / len(training_response)
    centred_phase = tuple(item - phase_mean for item in training_phase)
    sxx = math.fsum(item * item for item in centred_phase)
    if sxx <= 0.0:
        raise ValueError("training phase-lock values do not identify a response slope")
    if not min(training_phase) <= heldout_phase <= max(training_phase):
        raise ValueError("heldout phase-lock value must lie within the training range")
    slope = math.fsum(
        phase_offset * (response - response_mean)
        for phase_offset, response in zip(
            centred_phase,
            training_response,
            strict=True,
        )
    ) / sxx
    intercept = response_mean - slope * phase_mean
    predicted = intercept + slope * heldout_phase
    residual_sum = math.fsum(
        (response - (intercept + slope * phase)) ** 2
        for phase, response in zip(training_phase, training_response, strict=True)
    )
    residual_variance = residual_sum / (len(training_phase) - 2)
    residual_model_error = math.sqrt(max(0.0, residual_variance)) * math.sqrt(
        1.0
        + 1.0 / len(training_phase)
        + (heldout_phase - phase_mean) ** 2 / sxx
    )
    prediction_coefficients = tuple(
        1.0 / len(training_phase)
        + (heldout_phase - phase_mean) * phase_offset / sxx
        for phase_offset in centred_phase
    )
    # With unmeasured cross-level covariance, the triangle bound is the largest
    # possible propagated SE from the training response means.
    training_measurement_error = math.fsum(
        abs(coefficient) * error
        for coefficient, error in zip(
            prediction_coefficients,
            training_standard_errors,
            strict=True,
        )
    )
    # No cross-level covariance has been measured, including covariance with
    # the held-out response.  Add every SE component by the triangle bound.
    total_error = math.fsum(
        (
            residual_model_error,
            training_measurement_error,
            heldout_response.standard_error,
        )
    )
    residual_interval = _interval_from_mean(
        observation_count=heldout_response.observation_count,
        mean_effect=heldout_response.mean_effect - predicted,
        standard_error=total_error,
        confidence_multiplier=max(
            confidence_multiplier,
            MIN_REGRESSION_MULTIPLIER,
            heldout_response.confidence_multiplier,
        ),
    )
    return predicted, residual_interval


def phase_noise_sweep_audit(
    points: Iterable[PhaseNoiseSweepPoint],
    *,
    expected_response_sign: int = 1,
    minimum_probe_a_effect: float = 0.5,
    reference_equivalence_bound: float = 0.2,
    minimum_selective_effect: float = 0.5,
    confidence_multiplier: float = 1.96,
    minimum_absolute_correlation: float = 0.8,
    minimum_phase_lock_span: float = 0.25,
    minimum_selective_response_span: float = 0.5,
    minimum_conservative_response_drop: float = 0.2,
    heldout_prediction_equivalence_bound: float = 0.2,
) -> PhaseNoiseSweepAudit:
    """Fit phase/selectivity on training noise levels and predict one held-out level."""

    raw_points = tuple(points)
    if len(raw_points) < 5:
        raise ValueError("phase/noise sweep requires at least five points")
    response_sign = _strict_sign(expected_response_sign)
    minimum_a = _finite_positive(
        minimum_probe_a_effect,
        name="minimum_probe_a_effect",
    )
    reference_bound = _finite_nonnegative(
        reference_equivalence_bound,
        name="reference_equivalence_bound",
    )
    minimum_selectivity = _finite_positive(
        minimum_selective_effect,
        name="minimum_selective_effect",
    )
    if 2.0 * reference_bound > min(minimum_a, minimum_selectivity):
        raise ValueError(
            "reference_equivalence_bound must not exceed half the smaller "
            "minimum response effect"
        )
    requested_multiplier = _finite_positive(
        confidence_multiplier,
        name="confidence_multiplier",
    )
    if requested_multiplier < MIN_REQUESTED_CONFIDENCE_MULTIPLIER:
        raise ValueError(
            "confidence_multiplier must request at least a two-sided 95% interval"
        )
    minimum_correlation = _finite_positive(
        minimum_absolute_correlation,
        name="minimum_absolute_correlation",
    )
    if minimum_correlation > 1.0:
        raise ValueError("minimum_absolute_correlation cannot exceed one")
    minimum_phase_span = _finite_positive(
        minimum_phase_lock_span,
        name="minimum_phase_lock_span",
    )
    minimum_response_span = _finite_positive(
        minimum_selective_response_span,
        name="minimum_selective_response_span",
    )
    minimum_drop = _finite_positive(
        minimum_conservative_response_drop,
        name="minimum_conservative_response_drop",
    )
    prediction_bound = _finite_nonnegative(
        heldout_prediction_equivalence_bound,
        name="heldout_prediction_equivalence_bound",
    )
    if 2.0 * prediction_bound > minimum_selectivity:
        raise ValueError(
            "heldout_prediction_equivalence_bound must not exceed half "
            "minimum_selective_effect"
        )
    audited: list[PhaseNoiseSweepPointAudit] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, PhaseNoiseSweepPoint):
            raise ValueError(f"points[{index}] must be a PhaseNoiseSweepPoint")
        noise = _finite_nonnegative(point.noise_strength, name=f"points[{index}].noise")
        held_out = _strict_bool(point.held_out, name=f"points[{index}].held_out")
        phase_offsets = _finite_series(
            point.phase_offsets_rad,
            name=f"points[{index}].phase_offsets_rad",
        )
        response_names = (
            "probe_a_pump_on_matched",
            "probe_a_pump_on_sham",
            "probe_a_pump_off_matched",
            "probe_a_pump_off_sham",
            "reference_pump_on_matched",
            "reference_pump_on_sham",
            "reference_pump_off_matched",
            "reference_pump_off_sham",
        )
        responses = {
            name: _finite_series(
                getattr(point, name),
                name=f"points[{index}].{name}",
            )
            for name in response_names
        }
        canonical_point = PhaseNoiseSweepPoint(
            noise_strength=noise,
            phase_offsets_rad=phase_offsets,
            probe_a_pump_on_matched=responses["probe_a_pump_on_matched"],
            probe_a_pump_on_sham=responses["probe_a_pump_on_sham"],
            probe_a_pump_off_matched=responses["probe_a_pump_off_matched"],
            probe_a_pump_off_sham=responses["probe_a_pump_off_sham"],
            reference_pump_on_matched=responses["reference_pump_on_matched"],
            reference_pump_on_sham=responses["reference_pump_on_sham"],
            reference_pump_off_matched=responses["reference_pump_off_matched"],
            reference_pump_off_sham=responses["reference_pump_off_sham"],
            held_out=held_out,
        )
        phase = phase_lock_order_parameter(canonical_point.phase_offsets_rad)
        selectivity = probe_selectivity_audit(
            canonical_point.probe_a_pump_on_matched,
            canonical_point.probe_a_pump_on_sham,
            canonical_point.probe_a_pump_off_matched,
            canonical_point.probe_a_pump_off_sham,
            canonical_point.reference_pump_on_matched,
            canonical_point.reference_pump_on_sham,
            canonical_point.reference_pump_off_matched,
            canonical_point.reference_pump_off_sham,
            expected_response_sign=response_sign,
            minimum_probe_a_effect=minimum_a,
            reference_equivalence_bound=reference_bound,
            minimum_selective_effect=minimum_selectivity,
            confidence_multiplier=requested_multiplier,
        )
        audited.append(
            PhaseNoiseSweepPointAudit(
                raw_point=canonical_point,
                noise_strength=noise,
                held_out=held_out,
                phase_lock=phase,
                selectivity=selectivity,
            )
        )
    audited.sort(key=lambda item: item.noise_strength)
    noise_values = tuple(item.noise_strength for item in audited)
    unique_noise = len(set(noise_values)) == len(noise_values)
    if not unique_noise:
        raise ValueError("noise strengths must be unique")
    heldout_points = tuple(item for item in audited if item.held_out)
    training_points = tuple(item for item in audited if not item.held_out)
    if len(heldout_points) != 1 or len(training_points) < 4:
        raise ValueError("exactly one noise point must be held out and at least four must train")
    training_noise = tuple(item.noise_strength for item in training_points)
    training_phase = tuple(
        item.phase_lock.bias_corrected_resultant_length for item in training_points
    )
    training_response = tuple(
        item.selectivity.difference_in_differences.mean_effect
        for item in training_points
    )
    training_response_errors = tuple(
        item.selectivity.difference_in_differences.standard_error
        for item in training_points
    )
    noise_phase = _pearson_correlation(training_noise, training_phase)
    phase_response = _pearson_correlation(training_phase, training_response)
    phase_span = max(training_phase) - min(training_phase)
    response_span = max(training_response) - min(training_response)
    high_point = max(
        training_points,
        key=lambda item: item.phase_lock.bias_corrected_resultant_length,
    )
    low_point = min(
        training_points,
        key=lambda item: item.phase_lock.bias_corrected_resultant_length,
    )
    conservative_drop = (
        high_point.selectivity.difference_in_differences.lower_bound
        - low_point.selectivity.difference_in_differences.upper_bound
    )
    phase_decreases = (
        noise_phase is not None
        and noise_phase <= -minimum_correlation
        and phase_span >= minimum_phase_span
        and all(item.phase_lock.sufficient_effective_samples for item in training_points)
    )
    response_tracks = (
        phase_response is not None
        and phase_response >= minimum_correlation
        and response_span >= minimum_response_span
    )
    reference_null = all(
        item.selectivity.reference_equivalent_to_zero for item in audited
    )
    high_passes = high_point.selectivity.private_dressing_conditionally_supported
    conservative_separation = conservative_drop >= minimum_drop
    heldout_point = heldout_points[0]
    heldout_prediction, heldout_residual = _linear_heldout_prediction(
        training_phase,
        training_response,
        training_response_errors,
        heldout_phase=heldout_point.phase_lock.bias_corrected_resultant_length,
        heldout_response=heldout_point.selectivity.difference_in_differences,
        confidence_multiplier=requested_multiplier,
    )
    heldout_matches = _equivalent_to_zero(heldout_residual, prediction_bound)
    conditional = (
        phase_decreases
        and response_tracks
        and reference_null
        and high_passes
        and conservative_separation
        and heldout_matches
    )
    return PhaseNoiseSweepAudit(
        points=tuple(audited),
        expected_response_sign=response_sign,
        minimum_probe_a_effect=minimum_a,
        reference_equivalence_bound=reference_bound,
        minimum_selective_effect=minimum_selectivity,
        requested_confidence_multiplier=requested_multiplier,
        noise_phase_correlation=noise_phase,
        phase_selectivity_correlation=phase_response,
        phase_lock_span=phase_span,
        selective_response_span=response_span,
        conservative_high_to_low_response_drop=conservative_drop,
        minimum_absolute_correlation=minimum_correlation,
        minimum_phase_lock_span=minimum_phase_span,
        minimum_selective_response_span=minimum_response_span,
        minimum_conservative_response_drop=minimum_drop,
        heldout_prediction_equivalence_bound=prediction_bound,
        heldout_noise_strength=heldout_point.noise_strength,
        heldout_predicted_selective_response=heldout_prediction,
        heldout_prediction_residual=heldout_residual,
        noise_values_unique=unique_noise,
        phase_lock_decreases_with_noise=phase_decreases,
        selective_response_tracks_phase_lock=response_tracks,
        reference_equivalent_at_every_point=reference_null,
        highest_coherence_selectivity_passes=high_passes,
        high_coherence_response_exceeds_low_coherence_response=conservative_separation,
        heldout_response_matches_designated_prediction=heldout_matches,
        phase_lock_dependence_conditionally_supported=conditional,
        noise_to_phase_lock_dynamics_derived=False,
        causation_by_phase_lock_derived=False,
    )


def _worst_case_difference_interval(
    first: EffectInterval,
    second: EffectInterval,
    *,
    confidence_multiplier: float,
) -> EffectInterval:
    return _interval_from_mean(
        observation_count=min(first.observation_count, second.observation_count),
        mean_effect=first.mean_effect - second.mean_effect,
        standard_error=first.standard_error + second.standard_error,
        confidence_multiplier=max(
            confidence_multiplier,
            first.confidence_multiplier,
            second.confidence_multiplier,
        ),
    )


def post_pump_common_kernel_audit(
    probes: Iterable[CommonKernelProbeReadout],
    *,
    residual_drive_monitor_post: Iterable[Real],
    residual_drive_monitor_sham: Iterable[Real],
    nuisance_monitor_post: Iterable[Real],
    nuisance_monitor_sham: Iterable[Real],
    pump_start_time_s: float,
    pump_off_time_s: float,
    post_readout_start_time_s: float,
    post_readout_end_time_s: float,
    minimum_pump_off_dwell_s: float,
    minimum_common_kernel: float = 0.5,
    kernel_factorization_equivalence_bound: float = 0.2,
    pre_response_equivalence_bound: float = 0.2,
    monitor_equivalence_bound: float = 0.1,
    residual_drive_to_kernel_gain_upper_bound: float = 1.0,
    nuisance_to_kernel_gain_upper_bound: float = 1.0,
    apparatus_memory_kernel_upper_bound: float = 0.1,
    minimum_unexplained_kernel_margin: float = 0.3,
    confidence_multiplier: float = 1.96,
    heldout_probe_designation_declared: bool,
    calibration_fixed_before_pump: bool,
    blind_analysis_declared: bool,
    separate_heldout_readout_chain_declared: bool,
) -> PostPumpCommonKernelAudit:
    """Fit a common post-pump kernel and predict one designated held-out probe."""

    raw_probes = tuple(probes)
    if len(raw_probes) < 3:
        raise ValueError("common-kernel audit requires at least three calibrated probes")
    multiplier = _finite_positive(confidence_multiplier, name="confidence_multiplier")
    minimum_kernel = _finite_positive(minimum_common_kernel, name="minimum_common_kernel")
    factor_bound = _finite_nonnegative(
        kernel_factorization_equivalence_bound,
        name="kernel_factorization_equivalence_bound",
    )
    pre_bound = _finite_nonnegative(
        pre_response_equivalence_bound,
        name="pre_response_equivalence_bound",
    )
    monitor_bound = _finite_nonnegative(
        monitor_equivalence_bound,
        name="monitor_equivalence_bound",
    )
    if 2.0 * factor_bound > minimum_kernel:
        raise ValueError(
            "kernel_factorization_equivalence_bound must not exceed half "
            "minimum_common_kernel"
        )
    if 2.0 * pre_bound > minimum_kernel:
        raise ValueError(
            "pre_response_equivalence_bound must not exceed half minimum_common_kernel"
        )
    residual_gain_bound = _finite_positive(
        residual_drive_to_kernel_gain_upper_bound,
        name="residual_drive_to_kernel_gain_upper_bound",
    )
    nuisance_gain_bound = _finite_positive(
        nuisance_to_kernel_gain_upper_bound,
        name="nuisance_to_kernel_gain_upper_bound",
    )
    apparatus_memory_bound = _finite_nonnegative(
        apparatus_memory_kernel_upper_bound,
        name="apparatus_memory_kernel_upper_bound",
    )
    unexplained_margin_required = _finite_positive(
        minimum_unexplained_kernel_margin,
        name="minimum_unexplained_kernel_margin",
    )
    if 2.0 * monitor_bound > min(minimum_kernel, unexplained_margin_required):
        raise ValueError(
            "monitor_equivalence_bound must not exceed half the smaller "
            "common-kernel scale"
        )
    audited: list[CommonKernelProbeAudit] = []
    normalized_post_samples: dict[str, tuple[float, ...]] = {}
    seen_ids: set[str] = set()
    for index, probe in enumerate(raw_probes):
        if not isinstance(probe, CommonKernelProbeReadout):
            raise ValueError(f"probes[{index}] must be a CommonKernelProbeReadout")
        probe_id = probe.probe_id.strip()
        if not probe_id or probe_id in seen_ids:
            raise ValueError("probe_id values must be nonempty and unique")
        seen_ids.add(probe_id)
        gain = _finite_nonzero(
            probe.calibrated_response_gain,
            name=f"probes[{index}].calibrated_response_gain",
        )
        held_out = _strict_bool(probe.held_out, name=f"probes[{index}].held_out")
        readout_names = (
            "post_pump_response",
            "post_pump_sham",
            "pre_pump_response",
            "pre_pump_sham",
        )
        readouts = {
            name: _finite_series(
                getattr(probe, name),
                name=f"probes[{index}].{name}",
            )
            for name in readout_names
        }
        canonical_readout = CommonKernelProbeReadout(
            probe_id=probe_id,
            calibrated_response_gain=gain,
            post_pump_response=readouts["post_pump_response"],
            post_pump_sham=readouts["post_pump_sham"],
            pre_pump_response=readouts["pre_pump_response"],
            pre_pump_sham=readouts["pre_pump_sham"],
            held_out=held_out,
        )
        post_differences = _paired_differences(
            canonical_readout.post_pump_response,
            canonical_readout.post_pump_sham,
            treatment_name=f"probes[{index}].post_pump_response",
            control_name=f"probes[{index}].post_pump_sham",
        )
        pre_differences = _paired_differences(
            canonical_readout.pre_pump_response,
            canonical_readout.pre_pump_sham,
            treatment_name=f"probes[{index}].pre_pump_response",
            control_name=f"probes[{index}].pre_pump_sham",
        )
        normalized_post = tuple(item / gain for item in post_differences)
        normalized_pre = tuple(item / gain for item in pre_differences)
        normalized_post_samples[probe_id] = normalized_post
        audited.append(
            CommonKernelProbeAudit(
                raw_readout=canonical_readout,
                probe_id=probe_id,
                calibrated_response_gain=gain,
                held_out=held_out,
                raw_post_effect=_effect_interval(
                    post_differences,
                    confidence_multiplier=multiplier,
                    name=f"{probe_id}.raw_post_effect",
                ),
                normalized_post_kernel=_effect_interval(
                    normalized_post,
                    confidence_multiplier=multiplier,
                    name=f"{probe_id}.normalized_post_kernel",
                ),
                normalized_pre_kernel=_effect_interval(
                    normalized_pre,
                    confidence_multiplier=multiplier,
                    name=f"{probe_id}.normalized_pre_kernel",
                ),
            )
        )
    heldout = tuple(item for item in audited if item.held_out)
    training = tuple(item for item in audited if not item.held_out)
    if len(heldout) != 1 or len(training) < 2:
        raise ValueError("exactly one probe must be held out and at least two must train")
    pooled_training = tuple(
        sample
        for item in training
        for sample in normalized_post_samples[item.probe_id]
    )
    pooled_kernel = _effect_interval(
        pooled_training,
        confidence_multiplier=multiplier,
        name="pooled_training_kernel",
    )
    # Probe streams can share clocks, pumps, or drift.  Until their covariance is
    # measured, pooling must not report a smaller SE than any training probe.
    conservative_training_error = max(
        pooled_kernel.standard_error,
        *(item.normalized_post_kernel.standard_error for item in training),
    )
    fitted_kernel = _interval_from_mean(
        observation_count=pooled_kernel.observation_count,
        mean_effect=pooled_kernel.mean_effect,
        standard_error=conservative_training_error,
        confidence_multiplier=max(
            multiplier,
            *(item.normalized_post_kernel.confidence_multiplier for item in training),
        ),
    )
    training_pair_intervals = tuple(
        _worst_case_difference_interval(
            training[left].normalized_post_kernel,
            training[right].normalized_post_kernel,
            confidence_multiplier=multiplier,
        )
        for left in range(len(training))
        for right in range(left + 1, len(training))
    )
    training_factorizes = all(
        _equivalent_to_zero(interval, factor_bound)
        for interval in training_pair_intervals
    )
    heldout_interval = heldout[0].normalized_post_kernel
    heldout_residual = _worst_case_difference_interval(
        heldout_interval,
        fitted_kernel,
        confidence_multiplier=multiplier,
    )
    heldout_matches = _equivalent_to_zero(heldout_residual, factor_bound)
    all_pre_null = all(
        _equivalent_to_zero(item.normalized_pre_kernel, pre_bound)
        for item in audited
    )
    all_post_nonzero = (
        fitted_kernel.lower_bound >= minimum_kernel
        and all(
            item.normalized_post_kernel.lower_bound >= minimum_kernel
            for item in audited
        )
    )
    raw_residual_post = _finite_series(
        residual_drive_monitor_post,
        name="residual_drive_monitor_post",
    )
    raw_residual_sham = _finite_series(
        residual_drive_monitor_sham,
        name="residual_drive_monitor_sham",
    )
    raw_nuisance_post = _finite_series(
        nuisance_monitor_post,
        name="nuisance_monitor_post",
    )
    raw_nuisance_sham = _finite_series(
        nuisance_monitor_sham,
        name="nuisance_monitor_sham",
    )
    residual_drive = paired_effect_audit(
        raw_residual_post,
        raw_residual_sham,
        confidence_multiplier=multiplier,
    )
    nuisance = paired_effect_audit(
        raw_nuisance_post,
        raw_nuisance_sham,
        confidence_multiplier=multiplier,
    )
    drive_null = _equivalent_to_zero(residual_drive, monitor_bound)
    nuisance_null = _equivalent_to_zero(nuisance, monitor_bound)
    residual_drive_explanation = residual_gain_bound * max(
        abs(residual_drive.lower_bound),
        abs(residual_drive.upper_bound),
    )
    nuisance_explanation = nuisance_gain_bound * max(
        abs(nuisance.lower_bound),
        abs(nuisance.upper_bound),
    )
    memory_excluded = (
        fitted_kernel.lower_bound
        - residual_drive_explanation
        - nuisance_explanation
        - apparatus_memory_bound
        >= unexplained_margin_required
    )
    pump_start = _finite_real(pump_start_time_s, name="pump_start_time_s")
    pump_off = _finite_real(pump_off_time_s, name="pump_off_time_s")
    readout_start = _finite_real(
        post_readout_start_time_s,
        name="post_readout_start_time_s",
    )
    readout_end = _finite_real(post_readout_end_time_s, name="post_readout_end_time_s")
    minimum_dwell = _finite_positive(
        minimum_pump_off_dwell_s,
        name="minimum_pump_off_dwell_s",
    )
    ordering = pump_start < pump_off <= readout_start < readout_end
    dwell = readout_start - pump_off
    dwell_met = ordering and dwell >= minimum_dwell
    designation_declared = _strict_bool(
        heldout_probe_designation_declared,
        name="heldout_probe_designation_declared",
    )
    calibration_fixed = _strict_bool(
        calibration_fixed_before_pump,
        name="calibration_fixed_before_pump",
    )
    blinded = _strict_bool(blind_analysis_declared, name="blind_analysis_declared")
    separate_chain = _strict_bool(
        separate_heldout_readout_chain_declared,
        name="separate_heldout_readout_chain_declared",
    )
    metadata_complete = (
        designation_declared and calibration_fixed and blinded and separate_chain
    )
    persistence = (
        dwell_met
        and calibration_fixed
        and all_pre_null
        and drive_null
        and nuisance_null
        and memory_excluded
        and training_factorizes
        and all_post_nonzero
    )
    transfer = persistence and heldout_matches and metadata_complete
    return PostPumpCommonKernelAudit(
        probe_audits=tuple(audited),
        fitted_training_kernel=fitted_kernel,
        heldout_kernel_residual=heldout_residual,
        residual_drive_monitor=residual_drive,
        nuisance_monitor=nuisance,
        raw_residual_drive_monitor_post=raw_residual_post,
        raw_residual_drive_monitor_sham=raw_residual_sham,
        raw_nuisance_monitor_post=raw_nuisance_post,
        raw_nuisance_monitor_sham=raw_nuisance_sham,
        pump_start_time_s=pump_start,
        pump_off_time_s=pump_off,
        post_readout_start_time_s=readout_start,
        post_readout_end_time_s=readout_end,
        pump_off_dwell_s=dwell,
        minimum_pump_off_dwell_s=minimum_dwell,
        minimum_common_kernel=minimum_kernel,
        kernel_factorization_equivalence_bound=factor_bound,
        pre_response_equivalence_bound=pre_bound,
        monitor_equivalence_bound=monitor_bound,
        residual_drive_to_kernel_gain_upper_bound=residual_gain_bound,
        residual_drive_kernel_explanation_upper_bound=residual_drive_explanation,
        nuisance_to_kernel_gain_upper_bound=nuisance_gain_bound,
        nuisance_kernel_explanation_upper_bound=nuisance_explanation,
        apparatus_memory_kernel_upper_bound=apparatus_memory_bound,
        minimum_unexplained_kernel_margin=unexplained_margin_required,
        probe_correlation_model="UNMEASURED_WORST_CASE_CORRELATION",
        probe_covariance_measured=False,
        time_ordering_valid=ordering,
        minimum_dwell_met=dwell_met,
        all_pre_responses_equivalent_to_zero=all_pre_null,
        residual_drive_equivalent_to_zero=drive_null,
        nuisance_monitor_equivalent_to_zero=nuisance_null,
        monitors_and_apparatus_memory_cannot_explain_kernel=memory_excluded,
        training_probes_factorize_common_kernel=training_factorizes,
        heldout_probe_matches_common_kernel=heldout_matches,
        common_post_pump_kernel_nonzero=all_post_nonzero,
        heldout_probe_designation_declared=designation_declared,
        calibration_fixed_before_pump=calibration_fixed,
        blind_analysis_declared=blinded,
        separate_heldout_readout_chain_declared=separate_chain,
        independence_metadata_declared_complete=metadata_complete,
        post_pump_persistence_conditionally_supported=persistence,
        heldout_separate_chain_response_conditionally_supported=transfer,
        physical_material_phase_derived=False,
    )


def _validated_covariance(
    covariance: Iterable[Iterable[Real]],
    *,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(row) for row in covariance)
    if len(rows) != dimension or any(len(row) != dimension for row in rows):
        raise ValueError(f"energy covariance must have shape {dimension}x{dimension}")
    matrix = tuple(
        tuple(
            _finite_real(value, name=f"energy_covariance_j2[{row}][{column}]")
            for column, value in enumerate(values)
        )
        for row, values in enumerate(rows)
    )
    for row in range(dimension):
        if matrix[row][row] < 0.0:
            raise ValueError("energy covariance must be positive semidefinite")
        for column in range(dimension):
            pair_scale = max(abs(matrix[row][column]), abs(matrix[column][row]))
            pair_tolerance = 0.0 if pair_scale == 0.0 else 64.0 * math.ulp(pair_scale)
            if abs(matrix[row][column] - matrix[column][row]) > pair_tolerance:
                raise ValueError("energy covariance must be symmetric")
    lower = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for row in range(dimension):
        squared_sum = math.fsum(
            lower[row][item] * lower[row][item] for item in range(row)
        )
        diagonal = matrix[row][row] - squared_sum
        pivot_scale = max(abs(matrix[row][row]), abs(squared_sum))
        pivot_tolerance = (
            0.0 if pivot_scale == 0.0 else 64.0 * math.ulp(pivot_scale)
        )
        if diagonal < -pivot_tolerance:
            raise ValueError("energy covariance must be positive semidefinite")
        if diagonal <= pivot_tolerance:
            lower[row][row] = 0.0
            for target in range(row + 1, dimension):
                cross_sum = math.fsum(
                    lower[target][item] * lower[row][item] for item in range(row)
                )
                cross = matrix[target][row] - cross_sum
                cross_scale = max(abs(matrix[target][row]), abs(cross_sum))
                cross_tolerance = (
                    0.0 if cross_scale == 0.0 else 64.0 * math.ulp(cross_scale)
                )
                if abs(cross) > cross_tolerance:
                    raise ValueError("energy covariance must be positive semidefinite")
        else:
            lower[row][row] = math.sqrt(diagonal)
            for target in range(row + 1, dimension):
                cross = matrix[target][row] - math.fsum(
                    lower[target][item] * lower[row][item] for item in range(row)
                )
                lower[target][row] = cross / lower[row][row]
    return matrix


def energy_ledger_audit(
    *,
    pump_work_j: Iterable[Real],
    controller_work_j: Iterable[Real],
    probe_work_j: Iterable[Real],
    transfer_work_j: Iterable[Real],
    preexisting_reservoir_release_j: Iterable[Real],
    candidate_decoupled_energy_j: Iterable[Real],
    radiated_energy_j: Iterable[Real],
    thermal_mechanical_energy_j: Iterable[Real],
    reservoir_storage_j: Iterable[Real],
    recovered_work_j: Iterable[Real],
    energy_covariance_j2: Iterable[Iterable[Real]],
    pump_and_controller_decoupled_at_endpoint_declared: bool,
    absolute_closure_tolerance_j: float = 0.0,
    maximum_relative_balance_residual: float = 0.02,
    maximum_relative_uncertainty: float = 0.1,
    confidence_multiplier: float = 1.96,
) -> EnergyLedgerAudit:
    """Audit a fixed signed ledger with covariance and no fitted catch-all channel."""

    channel_names = (
        "pump_work_j",
        "controller_work_j",
        "probe_work_j",
        "transfer_work_j",
        "preexisting_reservoir_release_j",
        "candidate_decoupled_energy_j",
        "radiated_energy_j",
        "thermal_mechanical_energy_j",
        "reservoir_storage_j",
        "recovered_work_j",
    )
    raw_channels = (
        pump_work_j,
        controller_work_j,
        probe_work_j,
        transfer_work_j,
        preexisting_reservoir_release_j,
        candidate_decoupled_energy_j,
        radiated_energy_j,
        thermal_mechanical_energy_j,
        reservoir_storage_j,
        recovered_work_j,
    )
    channels = tuple(
        _finite_series(values, name=name)
        for name, values in zip(channel_names, raw_channels, strict=True)
    )
    if len({len(channel) for channel in channels}) != 1:
        raise ValueError("all energy-ledger channels must have equal trial counts")
    signs = (1, 1, 1, 1, 1, -1, -1, -1, -1, -1)
    covariance = _validated_covariance(energy_covariance_j2, dimension=len(channels))
    absolute_tolerance = _finite_nonnegative(
        absolute_closure_tolerance_j,
        name="absolute_closure_tolerance_j",
    )
    relative_balance_limit = _finite_nonnegative(
        maximum_relative_balance_residual,
        name="maximum_relative_balance_residual",
    )
    relative_uncertainty_limit = _finite_positive(
        maximum_relative_uncertainty,
        name="maximum_relative_uncertainty",
    )
    if relative_balance_limit > 0.1:
        raise ValueError("maximum_relative_balance_residual cannot exceed 0.1")
    if relative_uncertainty_limit > 0.25:
        raise ValueError("maximum_relative_uncertainty cannot exceed 0.25")
    multiplier = _finite_positive(confidence_multiplier, name="confidence_multiplier")
    endpoint_decoupled = _strict_bool(
        pump_and_controller_decoupled_at_endpoint_declared,
        name="pump_and_controller_decoupled_at_endpoint_declared",
    )
    all_nonnegative = all(item >= 0.0 for channel in channels for item in channel)
    means = tuple(math.fsum(channel) / len(channel) for channel in channels)
    total_input = math.fsum(means[index] for index in range(5))
    candidate_claim = means[5]
    if total_input <= 0.0:
        raise ValueError("mean total supplied work must be positive")
    residuals = tuple(
        math.fsum(
            sign * channels[channel_index][trial]
            for channel_index, sign in enumerate(signs)
        )
        for trial in range(len(channels[0]))
    )
    declared_variance = math.fsum(
        signs[row] * covariance[row][column] * signs[column]
        for row in range(len(signs))
        for column in range(len(signs))
    )
    declared_sigma = math.sqrt(max(0.0, declared_variance))
    residual_interval = _effect_interval(
        residuals,
        confidence_multiplier=multiplier,
        name="energy_balance_residual",
        additional_standard_error=declared_sigma,
    )
    mean_residual = math.fsum(residuals) / len(residuals)
    total_sigma = residual_interval.standard_error
    energy_claim_scale = candidate_claim
    relative_residual = (
        math.inf if energy_claim_scale <= 0.0 else abs(mean_residual) / energy_claim_scale
    )
    relative_uncertainty = (
        math.inf if energy_claim_scale <= 0.0 else total_sigma / energy_claim_scale
    )
    statistical_consistency = (
        abs(mean_residual)
        <= absolute_tolerance
        + residual_interval.confidence_multiplier * total_sigma
    )
    residual_small = (
        energy_claim_scale > 0.0
        and abs(mean_residual)
        <= relative_balance_limit * energy_claim_scale
    )
    uncertainty_nonvacuous = (
        energy_claim_scale > 0.0
        and total_sigma <= relative_uncertainty_limit * energy_claim_scale
        and absolute_tolerance <= relative_uncertainty_limit * energy_claim_scale
    )
    closed = (
        all_nonnegative
        and endpoint_decoupled
        and statistical_consistency
        and residual_small
        and uncertainty_nonvacuous
    )
    signed_means = tuple(sign * value for sign, value in zip(signs, means, strict=True))
    minimum_values = tuple(min(channel) for channel in channels)
    return EnergyLedgerAudit(
        trial_count=len(channels[0]),
        mean_pump_work_j=means[0],
        mean_controller_work_j=means[1],
        mean_probe_work_j=means[2],
        mean_transfer_work_j=means[3],
        mean_preexisting_reservoir_release_j=means[4],
        mean_candidate_decoupled_energy_j=means[5],
        mean_radiated_energy_j=means[6],
        mean_thermal_mechanical_energy_j=means[7],
        mean_reservoir_storage_j=means[8],
        mean_recovered_work_j=means[9],
        mean_balance_residual_j=mean_residual,
        balance_residual_interval=residual_interval,
        absolute_closure_tolerance_j=absolute_tolerance,
        signed_mean_energy_vector_j=signed_means,
        signed_ledger_signs=signs,
        energy_channel_values_j=channels,
        energy_covariance_j2=covariance,
        minimum_channel_values_j=minimum_values,
        declared_covariance_balance_sigma_j=declared_sigma,
        total_balance_sigma_j=total_sigma,
        relative_balance_residual=relative_residual,
        relative_balance_uncertainty=relative_uncertainty,
        maximum_relative_balance_residual=relative_balance_limit,
        maximum_relative_uncertainty=relative_uncertainty_limit,
        all_channels_nonnegative=all_nonnegative,
        covariance_symmetric_positive_semidefinite=True,
        balance_statistically_consistent_with_zero=statistical_consistency,
        balance_residual_small=residual_small,
        uncertainty_nonvacuous=uncertainty_nonvacuous,
        pump_and_controller_decoupled_at_endpoint_declared=endpoint_decoupled,
        energy_ledger_closed_conditionally=closed,
        microscopic_energy_transfer_mechanism_derived=False,
    )


def _same_summary_number(first: float, second: float) -> bool:
    if first == second:
        return True
    if not math.isfinite(first) or not math.isfinite(second):
        return False
    return math.isclose(first, second, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _validate_effect_interval(interval: EffectInterval, *, name: str) -> None:
    if not isinstance(interval, EffectInterval):
        raise ValueError(f"{name} must be an EffectInterval")
    if interval.observation_count < 1:
        raise ValueError(f"{name} observation count must be positive")
    if (
        interval.standard_error < 0.0
        or interval.confidence_multiplier < MIN_CONFIDENCE_MULTIPLIER
    ):
        raise ValueError(f"{name} has an invalid uncertainty")
    expected_radius = interval.standard_error * interval.confidence_multiplier
    if not _same_summary_number(
        interval.lower_bound,
        interval.mean_effect - expected_radius,
    ) or not _same_summary_number(
        interval.upper_bound,
        interval.mean_effect + expected_radius,
    ):
        raise ValueError(f"{name} bounds are inconsistent with its mean and error")


def _validate_selectivity_summary(audit: ProbeSelectivityAudit) -> None:
    if not isinstance(audit, ProbeSelectivityAudit):
        raise ValueError("selectivity summary has an unexpected type")
    _validate_effect_interval(audit.probe_a_effect, name="probe A effect")
    _validate_effect_interval(audit.reference_probe_effect, name="reference effect")
    _validate_effect_interval(
        audit.difference_in_differences,
        name="difference in differences",
    )
    _strict_bool_fields(
        audit,
        (
            "probe_a_response_detected",
            "reference_equivalent_to_zero",
            "selective_contrast_detected",
            "private_dressing_conditionally_supported",
            "public_environment_change_implied",
        ),
        prefix="selectivity",
    )
    _strict_sign(audit.expected_response_sign)
    _finite_positive(audit.minimum_probe_a_effect, name="minimum_probe_a_effect")
    _finite_nonnegative(
        audit.reference_equivalence_bound,
        name="reference_equivalence_bound",
    )
    _finite_positive(audit.minimum_selective_effect, name="minimum_selective_effect")
    if 2.0 * audit.reference_equivalence_bound > min(
        audit.minimum_probe_a_effect,
        audit.minimum_selective_effect,
    ):
        raise ValueError("reference equivalence bound exceeds half the response effect")
    expected_flags = (
        audit.probe_a_effect.lower_bound >= audit.minimum_probe_a_effect,
        _equivalent_to_zero(
            audit.reference_probe_effect,
            audit.reference_equivalence_bound,
        ),
        audit.difference_in_differences.lower_bound >= audit.minimum_selective_effect,
    )
    observed_flags = (
        audit.probe_a_response_detected,
        audit.reference_equivalent_to_zero,
        audit.selective_contrast_detected,
    )
    if audit.minimum_probe_a_effect <= 0.0 or audit.minimum_selective_effect <= 0.0:
        raise ValueError("selectivity minimum effects must remain positive")
    if observed_flags != expected_flags:
        raise ValueError("selectivity pass flags are inconsistent with effect intervals")
    if audit.private_dressing_conditionally_supported != all(expected_flags):
        raise ValueError("private dressing flag is inconsistent with selectivity gates")
    if audit.public_environment_change_implied:
        raise ValueError("probe selectivity cannot imply a public environment change")


def _validate_phase_lock_summary(audit: PhaseLockAudit) -> None:
    if not isinstance(audit, PhaseLockAudit):
        raise ValueError("phase-lock summary has an unexpected type")
    if audit.phase_sample_count < 4:
        raise ValueError("phase-lock summary has too few samples")
    _strict_bool_fields(
        audit,
        ("sufficient_effective_samples", "time_autocorrelation_corrected"),
        prefix="phase_lock",
    )
    if not (0.0 < audit.effective_sample_size <= audit.phase_sample_count):
        raise ValueError("phase-lock effective sample size is invalid")
    if not (0.0 <= audit.raw_resultant_length <= 1.0):
        raise ValueError("raw phase resultant is outside [0, 1]")
    if not (0.0 <= audit.bias_corrected_resultant_length <= 1.0):
        raise ValueError("bias-corrected phase resultant is outside [0, 1]")
    expected_squared = 0.0
    if audit.effective_sample_size > 1.0:
        expected_squared = max(
            0.0,
            (
                audit.effective_sample_size * audit.raw_resultant_length**2 - 1.0
            )
            / (audit.effective_sample_size - 1.0),
        )
    if not _same_summary_number(
        audit.bias_corrected_resultant_length,
        math.sqrt(expected_squared),
    ):
        raise ValueError("phase resultant bias correction is inconsistent")
    if audit.sufficient_effective_samples != (audit.effective_sample_size >= 4.0):
        raise ValueError("phase effective-sample pass flag is inconsistent")
    if (
        audit.effective_sample_size_method != "KISH_WEIGHT_ONLY"
        or audit.time_autocorrelation_corrected
    ):
        raise ValueError("phase-lock scope metadata was altered")


def _validate_phase_noise_sweep_summary(audit: PhaseNoiseSweepAudit) -> None:
    if not isinstance(audit, PhaseNoiseSweepAudit):
        raise ValueError("phase_noise_sweep must be a PhaseNoiseSweepAudit")
    if len(audit.points) < 5:
        raise ValueError("phase/noise summary has too few points")
    _strict_bool_fields(
        audit,
        (
            "noise_values_unique",
            "phase_lock_decreases_with_noise",
            "selective_response_tracks_phase_lock",
            "reference_equivalent_at_every_point",
            "highest_coherence_selectivity_passes",
            "high_coherence_response_exceeds_low_coherence_response",
            "heldout_response_matches_designated_prediction",
            "phase_lock_dependence_conditionally_supported",
            "noise_to_phase_lock_dynamics_derived",
            "causation_by_phase_lock_derived",
        ),
        prefix="phase_noise_sweep",
    )
    _strict_sign(audit.expected_response_sign)
    _finite_positive(audit.minimum_probe_a_effect, name="minimum_probe_a_effect")
    _finite_nonnegative(
        audit.reference_equivalence_bound,
        name="reference_equivalence_bound",
    )
    _finite_positive(audit.minimum_selective_effect, name="minimum_selective_effect")
    requested_multiplier = _finite_positive(
        audit.requested_confidence_multiplier,
        name="requested_confidence_multiplier",
    )
    if requested_multiplier < MIN_REQUESTED_CONFIDENCE_MULTIPLIER:
        raise ValueError("phase/noise requested confidence is below 95%")
    if 2.0 * audit.reference_equivalence_bound > min(
        audit.minimum_probe_a_effect,
        audit.minimum_selective_effect,
    ):
        raise ValueError("phase/noise reference bound is vacuous")
    for point in audit.points:
        if not isinstance(point.raw_point, PhaseNoiseSweepPoint):
            raise ValueError("phase/noise raw point has an unexpected type")
        _finite_nonnegative(point.noise_strength, name="noise_strength")
        _strict_bool(point.held_out, name="held_out")
        _validate_phase_lock_summary(point.phase_lock)
        _validate_selectivity_summary(point.selectivity)
        if (
            point.selectivity.expected_response_sign != audit.expected_response_sign
            or not _same_summary_number(
                point.selectivity.minimum_probe_a_effect,
                audit.minimum_probe_a_effect,
            )
            or not _same_summary_number(
                point.selectivity.reference_equivalence_bound,
                audit.reference_equivalence_bound,
            )
            or not _same_summary_number(
                point.selectivity.minimum_selective_effect,
                audit.minimum_selective_effect,
            )
        ):
            raise ValueError("phase/noise point configuration is inconsistent")
        if (
            not _same_summary_number(
                point.raw_point.noise_strength,
                point.noise_strength,
            )
            or point.raw_point.held_out is not point.held_out
        ):
            raise ValueError("phase/noise raw point metadata is inconsistent")
        expected_phase_lock = phase_lock_order_parameter(
            point.raw_point.phase_offsets_rad
        )
        expected_selectivity = probe_selectivity_audit(
            point.raw_point.probe_a_pump_on_matched,
            point.raw_point.probe_a_pump_on_sham,
            point.raw_point.probe_a_pump_off_matched,
            point.raw_point.probe_a_pump_off_sham,
            point.raw_point.reference_pump_on_matched,
            point.raw_point.reference_pump_on_sham,
            point.raw_point.reference_pump_off_matched,
            point.raw_point.reference_pump_off_sham,
            expected_response_sign=audit.expected_response_sign,
            minimum_probe_a_effect=audit.minimum_probe_a_effect,
            reference_equivalence_bound=audit.reference_equivalence_bound,
            minimum_selective_effect=audit.minimum_selective_effect,
            confidence_multiplier=audit.requested_confidence_multiplier,
        )
        if expected_phase_lock != point.phase_lock:
            raise ValueError("phase-lock summary is inconsistent with raw phases")
        if expected_selectivity != point.selectivity:
            raise ValueError("selectivity summary is inconsistent with raw responses")
    minimum_correlation = _finite_positive(
        audit.minimum_absolute_correlation,
        name="minimum_absolute_correlation",
    )
    if minimum_correlation > 1.0:
        raise ValueError("minimum_absolute_correlation cannot exceed one")
    _finite_positive(audit.minimum_phase_lock_span, name="minimum_phase_lock_span")
    _finite_positive(
        audit.minimum_selective_response_span,
        name="minimum_selective_response_span",
    )
    _finite_positive(
        audit.minimum_conservative_response_drop,
        name="minimum_conservative_response_drop",
    )
    prediction_bound = _finite_nonnegative(
        audit.heldout_prediction_equivalence_bound,
        name="heldout_prediction_equivalence_bound",
    )
    minimum_selective_effect = min(
        point.selectivity.minimum_selective_effect for point in audit.points
    )
    if 2.0 * prediction_bound > minimum_selective_effect:
        raise ValueError(
            "heldout prediction equivalence bound exceeds half the selective effect"
        )
    noise_values = tuple(point.noise_strength for point in audit.points)
    unique_noise = len(set(noise_values)) == len(noise_values)
    if audit.noise_values_unique != unique_noise or not unique_noise:
        raise ValueError("phase/noise uniqueness flag is inconsistent")
    heldout = tuple(point for point in audit.points if point.held_out)
    training = tuple(point for point in audit.points if not point.held_out)
    if len(heldout) != 1 or len(training) < 4:
        raise ValueError("phase/noise summary must retain one held-out point")
    training_noise = tuple(point.noise_strength for point in training)
    training_phase = tuple(
        point.phase_lock.bias_corrected_resultant_length for point in training
    )
    training_response = tuple(
        point.selectivity.difference_in_differences.mean_effect for point in training
    )
    training_errors = tuple(
        point.selectivity.difference_in_differences.standard_error for point in training
    )
    expected_noise_phase = _pearson_correlation(training_noise, training_phase)
    expected_phase_response = _pearson_correlation(training_phase, training_response)
    for observed, expected, label in (
        (audit.noise_phase_correlation, expected_noise_phase, "noise-phase correlation"),
        (
            audit.phase_selectivity_correlation,
            expected_phase_response,
            "phase-response correlation",
        ),
    ):
        if observed is None or expected is None:
            if observed is not expected:
                raise ValueError(f"{label} is inconsistent")
        elif not _same_summary_number(observed, expected):
            raise ValueError(f"{label} is inconsistent")
    phase_span = max(training_phase) - min(training_phase)
    response_span = max(training_response) - min(training_response)
    high = max(
        training,
        key=lambda point: point.phase_lock.bias_corrected_resultant_length,
    )
    low = min(
        training,
        key=lambda point: point.phase_lock.bias_corrected_resultant_length,
    )
    conservative_drop = (
        high.selectivity.difference_in_differences.lower_bound
        - low.selectivity.difference_in_differences.upper_bound
    )
    for observed, expected, label in (
        (audit.phase_lock_span, phase_span, "phase-lock span"),
        (audit.selective_response_span, response_span, "response span"),
        (
            audit.conservative_high_to_low_response_drop,
            conservative_drop,
            "conservative response drop",
        ),
    ):
        if not _same_summary_number(observed, expected):
            raise ValueError(f"{label} is inconsistent")
    phase_decreases = (
        expected_noise_phase is not None
        and expected_noise_phase <= -audit.minimum_absolute_correlation
        and phase_span >= audit.minimum_phase_lock_span
        and all(point.phase_lock.sufficient_effective_samples for point in training)
    )
    response_tracks = (
        expected_phase_response is not None
        and expected_phase_response >= audit.minimum_absolute_correlation
        and response_span >= audit.minimum_selective_response_span
    )
    reference_null = all(
        point.selectivity.reference_equivalent_to_zero for point in audit.points
    )
    high_passes = high.selectivity.private_dressing_conditionally_supported
    conservative_separation = (
        conservative_drop >= audit.minimum_conservative_response_drop
    )
    heldout_point = heldout[0]
    prediction, residual = _linear_heldout_prediction(
        training_phase,
        training_response,
        training_errors,
        heldout_phase=heldout_point.phase_lock.bias_corrected_resultant_length,
        heldout_response=heldout_point.selectivity.difference_in_differences,
        confidence_multiplier=audit.heldout_prediction_residual.confidence_multiplier,
    )
    _validate_effect_interval(
        audit.heldout_prediction_residual,
        name="heldout phase-response residual",
    )
    if (
        not _same_summary_number(audit.heldout_noise_strength, heldout_point.noise_strength)
        or not _same_summary_number(
            audit.heldout_predicted_selective_response,
            prediction,
        )
        or any(
            not _same_summary_number(observed, expected)
            for observed, expected in (
                (
                    audit.heldout_prediction_residual.mean_effect,
                    residual.mean_effect,
                ),
                (
                    audit.heldout_prediction_residual.standard_error,
                    residual.standard_error,
                ),
            )
        )
    ):
        raise ValueError("heldout phase-response prediction is inconsistent")
    heldout_matches = _equivalent_to_zero(
        audit.heldout_prediction_residual,
        audit.heldout_prediction_equivalence_bound,
    )
    expected_conditional = (
        phase_decreases
        and response_tracks
        and reference_null
        and high_passes
        and conservative_separation
        and heldout_matches
    )
    observed_flags = (
        audit.phase_lock_decreases_with_noise,
        audit.selective_response_tracks_phase_lock,
        audit.reference_equivalent_at_every_point,
        audit.highest_coherence_selectivity_passes,
        audit.high_coherence_response_exceeds_low_coherence_response,
        audit.heldout_response_matches_designated_prediction,
        audit.phase_lock_dependence_conditionally_supported,
    )
    expected_flags = (
        phase_decreases,
        response_tracks,
        reference_null,
        high_passes,
        conservative_separation,
        heldout_matches,
        expected_conditional,
    )
    if observed_flags != expected_flags:
        raise ValueError("phase/noise pass flags are inconsistent with summaries")
    if audit.noise_to_phase_lock_dynamics_derived:
        raise ValueError("measured phase lock does not derive noise-to-phase dynamics")
    if audit.causation_by_phase_lock_derived:
        raise ValueError("phase/noise association cannot derive phase causation")


def _pooled_effect_interval_from_summaries(
    intervals: Sequence[EffectInterval],
    *,
    confidence_multiplier: float,
) -> EffectInterval:
    total_count = sum(interval.observation_count for interval in intervals)
    pooled_mean = math.fsum(
        interval.observation_count * interval.mean_effect for interval in intervals
    ) / total_count
    pooled_sse = math.fsum(
        (interval.observation_count - 1)
        * interval.observation_count
        * interval.standard_error**2
        + interval.observation_count * (interval.mean_effect - pooled_mean) ** 2
        for interval in intervals
    )
    pooled_error = math.sqrt((pooled_sse / (total_count - 1)) / total_count)
    conservative_error = max(
        pooled_error,
        *(interval.standard_error for interval in intervals),
    )
    return _interval_from_mean(
        observation_count=total_count,
        mean_effect=pooled_mean,
        standard_error=conservative_error,
        confidence_multiplier=max(
            confidence_multiplier,
            *(interval.confidence_multiplier for interval in intervals),
        ),
    )


def _validate_common_kernel_summary(audit: PostPumpCommonKernelAudit) -> None:
    if not isinstance(audit, PostPumpCommonKernelAudit):
        raise ValueError("post_pump_common_kernel has an unexpected type")
    if len(audit.probe_audits) < 3:
        raise ValueError("common-kernel summary has too few probes")
    _strict_bool_fields(
        audit,
        (
            "probe_covariance_measured",
            "time_ordering_valid",
            "minimum_dwell_met",
            "all_pre_responses_equivalent_to_zero",
            "residual_drive_equivalent_to_zero",
            "nuisance_monitor_equivalent_to_zero",
            "monitors_and_apparatus_memory_cannot_explain_kernel",
            "training_probes_factorize_common_kernel",
            "heldout_probe_matches_common_kernel",
            "common_post_pump_kernel_nonzero",
            "heldout_probe_designation_declared",
            "calibration_fixed_before_pump",
            "blind_analysis_declared",
            "separate_heldout_readout_chain_declared",
            "independence_metadata_declared_complete",
            "post_pump_persistence_conditionally_supported",
            "heldout_separate_chain_response_conditionally_supported",
            "physical_material_phase_derived",
        ),
        prefix="common_kernel",
    )
    _finite_positive(audit.minimum_common_kernel, name="minimum_common_kernel")
    factor_bound = _finite_nonnegative(
        audit.kernel_factorization_equivalence_bound,
        name="kernel_factorization_equivalence_bound",
    )
    pre_bound = _finite_nonnegative(
        audit.pre_response_equivalence_bound,
        name="pre_response_equivalence_bound",
    )
    if 2.0 * factor_bound > audit.minimum_common_kernel:
        raise ValueError("common-kernel factor bound exceeds half the minimum kernel")
    if 2.0 * pre_bound > audit.minimum_common_kernel:
        raise ValueError("pre-response bound exceeds half the minimum kernel")
    _finite_nonnegative(audit.monitor_equivalence_bound, name="monitor_equivalence_bound")
    _finite_positive(
        audit.residual_drive_to_kernel_gain_upper_bound,
        name="residual_drive_to_kernel_gain_upper_bound",
    )
    _finite_positive(
        audit.nuisance_to_kernel_gain_upper_bound,
        name="nuisance_to_kernel_gain_upper_bound",
    )
    _finite_nonnegative(
        audit.apparatus_memory_kernel_upper_bound,
        name="apparatus_memory_kernel_upper_bound",
    )
    _finite_positive(
        audit.minimum_unexplained_kernel_margin,
        name="minimum_unexplained_kernel_margin",
    )
    if 2.0 * audit.monitor_equivalence_bound > min(
        audit.minimum_common_kernel,
        audit.minimum_unexplained_kernel_margin,
    ):
        raise ValueError("monitor equivalence bound exceeds half the kernel scale")
    _finite_positive(audit.minimum_pump_off_dwell_s, name="minimum_pump_off_dwell_s")
    if (
        audit.probe_correlation_model != "UNMEASURED_WORST_CASE_CORRELATION"
        or audit.probe_covariance_measured
    ):
        raise ValueError("training-probe correlation scope metadata was altered")
    identifiers = tuple(probe.probe_id for probe in audit.probe_audits)
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("common-kernel probe identifiers are invalid")
    for probe in audit.probe_audits:
        _strict_bool(probe.held_out, name=f"{probe.probe_id}.held_out")
        if not isinstance(probe.raw_readout, CommonKernelProbeReadout):
            raise ValueError("common-kernel raw readout has an unexpected type")
        if probe.calibrated_response_gain == 0.0:
            raise ValueError("common-kernel probe gain must be nonzero")
        _validate_effect_interval(probe.raw_post_effect, name=f"{probe.probe_id} raw post")
        _validate_effect_interval(
            probe.normalized_post_kernel,
            name=f"{probe.probe_id} normalized post",
        )
        _validate_effect_interval(
            probe.normalized_pre_kernel,
            name=f"{probe.probe_id} normalized pre",
        )
        if not _same_summary_number(
            probe.raw_post_effect.mean_effect,
            probe.calibrated_response_gain * probe.normalized_post_kernel.mean_effect,
        ) or not _same_summary_number(
            probe.raw_post_effect.standard_error,
            abs(probe.calibrated_response_gain)
            * probe.normalized_post_kernel.standard_error,
        ):
            raise ValueError("raw and calibrated common-kernel summaries disagree")
        raw = probe.raw_readout
        if (
            raw.probe_id != probe.probe_id
            or not _same_summary_number(
                raw.calibrated_response_gain,
                probe.calibrated_response_gain,
            )
            or raw.held_out is not probe.held_out
        ):
            raise ValueError("common-kernel raw readout metadata is inconsistent")
        post_differences = _paired_differences(
            raw.post_pump_response,
            raw.post_pump_sham,
            treatment_name=f"{probe.probe_id}.raw_post_response",
            control_name=f"{probe.probe_id}.raw_post_sham",
        )
        pre_differences = _paired_differences(
            raw.pre_pump_response,
            raw.pre_pump_sham,
            treatment_name=f"{probe.probe_id}.raw_pre_response",
            control_name=f"{probe.probe_id}.raw_pre_sham",
        )
        normalized_post = tuple(
            value / probe.calibrated_response_gain for value in post_differences
        )
        normalized_pre = tuple(
            value / probe.calibrated_response_gain for value in pre_differences
        )
        expected_raw_post = _effect_interval(
            post_differences,
            confidence_multiplier=probe.raw_post_effect.confidence_multiplier,
            name=f"{probe.probe_id}.raw_post_validation",
        )
        expected_normalized_post = _effect_interval(
            normalized_post,
            confidence_multiplier=probe.normalized_post_kernel.confidence_multiplier,
            name=f"{probe.probe_id}.normalized_post_validation",
        )
        expected_normalized_pre = _effect_interval(
            normalized_pre,
            confidence_multiplier=probe.normalized_pre_kernel.confidence_multiplier,
            name=f"{probe.probe_id}.normalized_pre_validation",
        )
        if (
            expected_raw_post != probe.raw_post_effect
            or expected_normalized_post != probe.normalized_post_kernel
            or expected_normalized_pre != probe.normalized_pre_kernel
        ):
            raise ValueError("common-kernel summaries are inconsistent with raw readouts")
    heldout = tuple(probe for probe in audit.probe_audits if probe.held_out)
    training = tuple(probe for probe in audit.probe_audits if not probe.held_out)
    if len(heldout) != 1 or len(training) < 2:
        raise ValueError("common-kernel summary must contain one held-out probe")
    _validate_effect_interval(audit.fitted_training_kernel, name="fitted common kernel")
    expected_fitted = _pooled_effect_interval_from_summaries(
        tuple(probe.normalized_post_kernel for probe in training),
        confidence_multiplier=audit.fitted_training_kernel.confidence_multiplier,
    )
    if any(
        not _same_summary_number(observed, expected)
        for observed, expected in (
            (audit.fitted_training_kernel.mean_effect, expected_fitted.mean_effect),
            (audit.fitted_training_kernel.standard_error, expected_fitted.standard_error),
        )
    ):
        raise ValueError("fitted common-kernel summary is inconsistent")
    training_factorizes = all(
        _equivalent_to_zero(
            _worst_case_difference_interval(
                training[left].normalized_post_kernel,
                training[right].normalized_post_kernel,
                confidence_multiplier=audit.fitted_training_kernel.confidence_multiplier,
            ),
            audit.kernel_factorization_equivalence_bound,
        )
        for left in range(len(training))
        for right in range(left + 1, len(training))
    )
    _validate_effect_interval(audit.heldout_kernel_residual, name="heldout kernel residual")
    expected_heldout_residual = _worst_case_difference_interval(
        heldout[0].normalized_post_kernel,
        audit.fitted_training_kernel,
        confidence_multiplier=audit.heldout_kernel_residual.confidence_multiplier,
    )
    if any(
        not _same_summary_number(observed, expected)
        for observed, expected in (
            (audit.heldout_kernel_residual.mean_effect, expected_heldout_residual.mean_effect),
            (
                audit.heldout_kernel_residual.standard_error,
                expected_heldout_residual.standard_error,
            ),
        )
    ):
        raise ValueError("heldout common-kernel residual is inconsistent")
    heldout_matches = _equivalent_to_zero(
        audit.heldout_kernel_residual,
        audit.kernel_factorization_equivalence_bound,
    )
    all_pre_null = all(
        _equivalent_to_zero(
            probe.normalized_pre_kernel,
            audit.pre_response_equivalence_bound,
        )
        for probe in audit.probe_audits
    )
    all_post_nonzero = (
        audit.fitted_training_kernel.lower_bound >= audit.minimum_common_kernel
        and all(
            probe.normalized_post_kernel.lower_bound >= audit.minimum_common_kernel
            for probe in audit.probe_audits
        )
    )
    _validate_effect_interval(audit.residual_drive_monitor, name="residual-drive monitor")
    _validate_effect_interval(audit.nuisance_monitor, name="nuisance monitor")
    expected_residual_monitor = paired_effect_audit(
        audit.raw_residual_drive_monitor_post,
        audit.raw_residual_drive_monitor_sham,
        confidence_multiplier=audit.residual_drive_monitor.confidence_multiplier,
    )
    expected_nuisance_monitor = paired_effect_audit(
        audit.raw_nuisance_monitor_post,
        audit.raw_nuisance_monitor_sham,
        confidence_multiplier=audit.nuisance_monitor.confidence_multiplier,
    )
    if expected_residual_monitor != audit.residual_drive_monitor:
        raise ValueError("residual-drive summary is inconsistent with raw monitor data")
    if expected_nuisance_monitor != audit.nuisance_monitor:
        raise ValueError("nuisance summary is inconsistent with raw monitor data")
    drive_null = _equivalent_to_zero(
        audit.residual_drive_monitor,
        audit.monitor_equivalence_bound,
    )
    nuisance_null = _equivalent_to_zero(
        audit.nuisance_monitor,
        audit.monitor_equivalence_bound,
    )
    residual_explanation = audit.residual_drive_to_kernel_gain_upper_bound * max(
        abs(audit.residual_drive_monitor.lower_bound),
        abs(audit.residual_drive_monitor.upper_bound),
    )
    nuisance_explanation = audit.nuisance_to_kernel_gain_upper_bound * max(
        abs(audit.nuisance_monitor.lower_bound),
        abs(audit.nuisance_monitor.upper_bound),
    )
    memory_excluded = (
        audit.fitted_training_kernel.lower_bound
        - residual_explanation
        - nuisance_explanation
        - audit.apparatus_memory_kernel_upper_bound
        >= audit.minimum_unexplained_kernel_margin
    )
    ordering = (
        audit.pump_start_time_s
        < audit.pump_off_time_s
        <= audit.post_readout_start_time_s
        < audit.post_readout_end_time_s
    )
    dwell = audit.post_readout_start_time_s - audit.pump_off_time_s
    dwell_met = ordering and dwell >= audit.minimum_pump_off_dwell_s
    metadata_complete = (
        audit.heldout_probe_designation_declared
        and audit.calibration_fixed_before_pump
        and audit.blind_analysis_declared
        and audit.separate_heldout_readout_chain_declared
    )
    persistence = (
        dwell_met
        and audit.calibration_fixed_before_pump
        and all_pre_null
        and drive_null
        and nuisance_null
        and memory_excluded
        and training_factorizes
        and all_post_nonzero
    )
    heldout_response = persistence and heldout_matches and metadata_complete
    expected_flags = (
        ordering,
        dwell_met,
        all_pre_null,
        drive_null,
        nuisance_null,
        memory_excluded,
        training_factorizes,
        heldout_matches,
        all_post_nonzero,
        metadata_complete,
        persistence,
        heldout_response,
    )
    observed_flags = (
        audit.time_ordering_valid,
        audit.minimum_dwell_met,
        audit.all_pre_responses_equivalent_to_zero,
        audit.residual_drive_equivalent_to_zero,
        audit.nuisance_monitor_equivalent_to_zero,
        audit.monitors_and_apparatus_memory_cannot_explain_kernel,
        audit.training_probes_factorize_common_kernel,
        audit.heldout_probe_matches_common_kernel,
        audit.common_post_pump_kernel_nonzero,
        audit.independence_metadata_declared_complete,
        audit.post_pump_persistence_conditionally_supported,
        audit.heldout_separate_chain_response_conditionally_supported,
    )
    if observed_flags != expected_flags:
        raise ValueError("common-kernel pass flags are inconsistent with summaries")
    if (
        not _same_summary_number(audit.pump_off_dwell_s, dwell)
        or not _same_summary_number(
            audit.residual_drive_kernel_explanation_upper_bound,
            residual_explanation,
        )
        or not _same_summary_number(
            audit.nuisance_kernel_explanation_upper_bound,
            nuisance_explanation,
        )
    ):
        raise ValueError("common-kernel dwell or residual-drive bound is inconsistent")
    if audit.physical_material_phase_derived:
        raise ValueError("common-kernel control cannot derive a physical material phase")


def _validate_energy_ledger_summary(audit: EnergyLedgerAudit) -> None:
    if not isinstance(audit, EnergyLedgerAudit):
        raise ValueError("energy_ledger must be an EnergyLedgerAudit")
    _strict_bool_fields(
        audit,
        (
            "all_channels_nonnegative",
            "covariance_symmetric_positive_semidefinite",
            "balance_statistically_consistent_with_zero",
            "balance_residual_small",
            "uncertainty_nonvacuous",
            "pump_and_controller_decoupled_at_endpoint_declared",
            "energy_ledger_closed_conditionally",
            "microscopic_energy_transfer_mechanism_derived",
        ),
        prefix="energy_ledger",
    )
    _validate_effect_interval(audit.balance_residual_interval, name="energy balance interval")
    _finite_nonnegative(
        audit.maximum_relative_balance_residual,
        name="maximum_relative_balance_residual",
    )
    _finite_positive(
        audit.maximum_relative_uncertainty,
        name="maximum_relative_uncertainty",
    )
    if audit.maximum_relative_balance_residual > 0.1:
        raise ValueError("maximum_relative_balance_residual cannot exceed 0.1")
    if audit.maximum_relative_uncertainty > 0.25:
        raise ValueError("maximum_relative_uncertainty cannot exceed 0.25")
    _finite_nonnegative(
        audit.absolute_closure_tolerance_j,
        name="absolute_closure_tolerance_j",
    )
    reported_means = (
        audit.mean_pump_work_j,
        audit.mean_controller_work_j,
        audit.mean_probe_work_j,
        audit.mean_transfer_work_j,
        audit.mean_preexisting_reservoir_release_j,
        audit.mean_candidate_decoupled_energy_j,
        audit.mean_radiated_energy_j,
        audit.mean_thermal_mechanical_energy_j,
        audit.mean_reservoir_storage_j,
        audit.mean_recovered_work_j,
    )
    expected_signs = (1, 1, 1, 1, 1, -1, -1, -1, -1, -1)
    channel_names = (
        "pump_work_j",
        "controller_work_j",
        "probe_work_j",
        "transfer_work_j",
        "preexisting_reservoir_release_j",
        "candidate_decoupled_energy_j",
        "radiated_energy_j",
        "thermal_mechanical_energy_j",
        "reservoir_storage_j",
        "recovered_work_j",
    )
    if len(audit.energy_channel_values_j) != len(expected_signs):
        raise ValueError("energy ledger raw channels are incomplete")
    channels = tuple(
        _finite_series(values, name=name)
        for name, values in zip(
            channel_names,
            audit.energy_channel_values_j,
            strict=True,
        )
    )
    if len({len(channel) for channel in channels}) != 1:
        raise ValueError("energy ledger raw channels have unequal trial counts")
    means = tuple(math.fsum(channel) / len(channel) for channel in channels)
    if any(
        not _same_summary_number(observed, expected)
        for observed, expected in zip(reported_means, means, strict=True)
    ):
        raise ValueError("energy ledger means are inconsistent with raw channels")
    if audit.signed_ledger_signs != expected_signs:
        raise ValueError("energy ledger sign convention was altered")
    if len(audit.minimum_channel_values_j) != len(means):
        raise ValueError("energy ledger channel minima are incomplete")
    expected_minima = tuple(min(channel) for channel in channels)
    if any(
        not _same_summary_number(observed, expected)
        for observed, expected in zip(
            audit.minimum_channel_values_j,
            expected_minima,
            strict=True,
        )
    ):
        raise ValueError("energy ledger minima are inconsistent with raw channels")
    covariance = _validated_covariance(
        audit.energy_covariance_j2,
        dimension=len(expected_signs),
    )
    declared_variance = math.fsum(
        expected_signs[row] * covariance[row][column] * expected_signs[column]
        for row in range(len(expected_signs))
        for column in range(len(expected_signs))
    )
    expected_declared_sigma = math.sqrt(max(0.0, declared_variance))
    _finite_nonnegative(
        audit.declared_covariance_balance_sigma_j,
        name="declared_covariance_balance_sigma_j",
    )
    _finite_nonnegative(audit.total_balance_sigma_j, name="total_balance_sigma_j")
    if not _same_summary_number(
        audit.declared_covariance_balance_sigma_j,
        expected_declared_sigma,
    ):
        raise ValueError("declared covariance sigma is inconsistent with covariance")
    if audit.total_balance_sigma_j < audit.declared_covariance_balance_sigma_j:
        raise ValueError("total balance sigma cannot be below covariance sigma")
    if audit.covariance_symmetric_positive_semidefinite is not True:
        raise ValueError("validated energy covariance flag must remain true")
    residuals = tuple(
        math.fsum(
            sign * channels[channel_index][trial]
            for channel_index, sign in enumerate(expected_signs)
        )
        for trial in range(len(channels[0]))
    )
    expected_interval = _effect_interval(
        residuals,
        confidence_multiplier=audit.balance_residual_interval.confidence_multiplier,
        name="energy_balance_residual_validation",
        additional_standard_error=expected_declared_sigma,
    )
    if (
        audit.balance_residual_interval.observation_count
        != expected_interval.observation_count
        or any(
            not _same_summary_number(observed, expected)
            for observed, expected in (
                (
                    audit.balance_residual_interval.mean_effect,
                    expected_interval.mean_effect,
                ),
                (
                    audit.balance_residual_interval.standard_error,
                    expected_interval.standard_error,
                ),
                (
                    audit.balance_residual_interval.lower_bound,
                    expected_interval.lower_bound,
                ),
                (
                    audit.balance_residual_interval.upper_bound,
                    expected_interval.upper_bound,
                ),
            )
        )
    ):
        raise ValueError("energy balance interval is inconsistent with raw trials")
    if not _same_summary_number(
        audit.total_balance_sigma_j,
        expected_interval.standard_error,
    ):
        raise ValueError("total balance sigma is inconsistent with raw trials")
    expected_signed_means = tuple(
        sign * value for sign, value in zip(expected_signs, means, strict=True)
    )
    if len(audit.signed_mean_energy_vector_j) != len(means) or any(
        not _same_summary_number(observed, expected)
        for observed, expected in zip(
            audit.signed_mean_energy_vector_j,
            expected_signed_means,
            strict=True,
        )
    ):
        raise ValueError("energy ledger signed means are inconsistent")
    expected_residual = math.fsum(expected_signed_means)
    if not _same_summary_number(audit.mean_balance_residual_j, expected_residual):
        raise ValueError("energy ledger residual is inconsistent with channel means")
    if audit.trial_count != len(channels[0]):
        raise ValueError("energy trial count is inconsistent")
    expected_nonnegative = all(value >= 0.0 for channel in channels for value in channel)
    if audit.all_channels_nonnegative != expected_nonnegative:
        raise ValueError("energy channel nonnegativity flag is inconsistent")
    candidate = audit.mean_candidate_decoupled_energy_j
    expected_relative_residual = (
        math.inf if candidate <= 0.0 else abs(expected_residual) / candidate
    )
    expected_relative_uncertainty = (
        math.inf if candidate <= 0.0 else audit.total_balance_sigma_j / candidate
    )
    if not _same_summary_number(
        audit.relative_balance_residual,
        expected_relative_residual,
    ) or not _same_summary_number(
        audit.relative_balance_uncertainty,
        expected_relative_uncertainty,
    ):
        raise ValueError("energy relative residual or uncertainty is inconsistent")
    expected_statistical = (
        abs(expected_residual)
        <= audit.absolute_closure_tolerance_j
        + audit.balance_residual_interval.confidence_multiplier
        * audit.total_balance_sigma_j
    )
    expected_small = (
        candidate > 0.0
        and abs(expected_residual)
        <= audit.maximum_relative_balance_residual * candidate
    )
    expected_informative = (
        candidate > 0.0
        and audit.total_balance_sigma_j <= audit.maximum_relative_uncertainty * candidate
        and audit.absolute_closure_tolerance_j
        <= audit.maximum_relative_uncertainty * candidate
    )
    expected_closed = (
        expected_nonnegative
        and audit.covariance_symmetric_positive_semidefinite
        and audit.pump_and_controller_decoupled_at_endpoint_declared
        and expected_statistical
        and expected_small
        and expected_informative
    )
    observed_flags = (
        audit.balance_statistically_consistent_with_zero,
        audit.balance_residual_small,
        audit.uncertainty_nonvacuous,
        audit.energy_ledger_closed_conditionally,
    )
    expected_flags = (
        expected_statistical,
        expected_small,
        expected_informative,
        expected_closed,
    )
    if observed_flags != expected_flags:
        raise ValueError("energy ledger pass flags are inconsistent with numeric summaries")
    if audit.microscopic_energy_transfer_mechanism_derived:
        raise ValueError("energy closure cannot derive a microscopic transfer mechanism")


def probe_scaffold_pilot_report(
    *,
    phase_noise_sweep: PhaseNoiseSweepAudit,
    post_pump_common_kernel: PostPumpCommonKernelAudit,
    energy_ledger: EnergyLedgerAudit,
) -> ProbeScaffoldPilotReport:
    """Combine the independent gates without promoting them to new physics."""

    if not isinstance(phase_noise_sweep, PhaseNoiseSweepAudit):
        raise ValueError("phase_noise_sweep must be a PhaseNoiseSweepAudit")
    if not isinstance(post_pump_common_kernel, PostPumpCommonKernelAudit):
        raise ValueError("post_pump_common_kernel must be a PostPumpCommonKernelAudit")
    _validate_phase_noise_sweep_summary(phase_noise_sweep)
    _validate_common_kernel_summary(post_pump_common_kernel)
    _validate_energy_ledger_summary(energy_ledger)
    high_private = phase_noise_sweep.highest_coherence_selectivity_passes
    phase_locked = phase_noise_sweep.phase_lock_dependence_conditionally_supported
    persistence = post_pump_common_kernel.post_pump_persistence_conditionally_supported
    transfer = (
        post_pump_common_kernel.heldout_separate_chain_response_conditionally_supported
    )
    energy_closed = energy_ledger.energy_ledger_closed_conditionally
    private_stage = "PRIVATE_NULL_CONTROL_ONLY"
    if high_private:
        private_stage = "CONDITIONAL_PRIVATE_DRESSING"
    if high_private and phase_locked:
        private_stage = "CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING"
    public_stage = "PUBLIC_NULL_CONTROL_ONLY"
    if persistence:
        public_stage = "CONDITIONAL_POST_PUMP_COMMON_KERNEL"
    if persistence and transfer:
        public_stage = "CONDITIONAL_DECLARED_SEPARATE_CHAIN_HELDOUT_RESPONSE"
    response_candidate = persistence and transfer and energy_closed
    if response_candidate:
        public_stage = "CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE"
    report = ProbeScaffoldPilotReport(
        schema_version=SCHEMA_VERSION,
        phase_noise_sweep=phase_noise_sweep,
        post_pump_common_kernel=post_pump_common_kernel,
        energy_ledger=energy_ledger,
        maximum_private_branch_stage=private_stage,
        maximum_public_branch_stage=public_stage,
        conditional_public_response_kernel_candidate=response_candidate,
        conditional_public_scaffold_candidate=False,
        physical_public_scaffold_derived=False,
        claim_locks=ProbeScaffoldClaimLocks(),
    )
    validate_probe_scaffold_pilot_report(report)
    return report


def validate_probe_scaffold_pilot_report(report: ProbeScaffoldPilotReport) -> None:
    """Reject scope or claim-lock tampering in a pilot report."""

    if not isinstance(report, ProbeScaffoldPilotReport):
        raise ValueError("report must be a ProbeScaffoldPilotReport")
    if report.schema_version != SCHEMA_VERSION:
        raise ValueError("unexpected probe-scaffold schema version")
    _strict_bool_fields(
        report,
        (
            "conditional_public_response_kernel_candidate",
            "conditional_public_scaffold_candidate",
            "physical_public_scaffold_derived",
        ),
        prefix="report",
    )
    if not isinstance(report.claim_locks, ProbeScaffoldClaimLocks):
        raise ValueError("report claim_locks has an unexpected type")
    _strict_bool_fields(
        report.claim_locks,
        tuple(vars(report.claim_locks)),
        prefix="claim_locks",
    )
    _validate_phase_noise_sweep_summary(report.phase_noise_sweep)
    _validate_common_kernel_summary(report.post_pump_common_kernel)
    _validate_energy_ledger_summary(report.energy_ledger)
    lock_values = tuple(vars(report.claim_locks).values())
    if any(lock_values) or report.physical_public_scaffold_derived:
        raise ValueError("probe-scaffold physical claim locks must remain false")
    expected_response_candidate = (
        report.post_pump_common_kernel.post_pump_persistence_conditionally_supported
        and report.post_pump_common_kernel.heldout_separate_chain_response_conditionally_supported
        and report.energy_ledger.energy_ledger_closed_conditionally
    )
    if (
        report.conditional_public_response_kernel_candidate
        != expected_response_candidate
    ):
        raise ValueError("public-response candidate flag is inconsistent with gates")
    if report.conditional_public_scaffold_candidate:
        raise ValueError("heldout response is not a transferable public scaffold")
    expected_private = "PRIVATE_NULL_CONTROL_ONLY"
    if report.phase_noise_sweep.highest_coherence_selectivity_passes:
        expected_private = "CONDITIONAL_PRIVATE_DRESSING"
    if (
        report.phase_noise_sweep.highest_coherence_selectivity_passes
        and report.phase_noise_sweep.phase_lock_dependence_conditionally_supported
    ):
        expected_private = "CONDITIONAL_PHASE_LOCKED_PRIVATE_DRESSING"
    expected_public = "PUBLIC_NULL_CONTROL_ONLY"
    if report.post_pump_common_kernel.post_pump_persistence_conditionally_supported:
        expected_public = "CONDITIONAL_POST_PUMP_COMMON_KERNEL"
    if report.post_pump_common_kernel.heldout_separate_chain_response_conditionally_supported:
        expected_public = "CONDITIONAL_DECLARED_SEPARATE_CHAIN_HELDOUT_RESPONSE"
    if expected_response_candidate:
        expected_public = "CONDITIONAL_PUBLIC_RESPONSE_KERNEL_CANDIDATE"
    if report.maximum_private_branch_stage != expected_private:
        raise ValueError("maximum private branch stage is inconsistent with gates")
    if report.maximum_public_branch_stage != expected_public:
        raise ValueError("maximum public branch stage is inconsistent with gates")


__all__ = [
    "SCHEMA_VERSION",
    "CommonKernelProbeAudit",
    "CommonKernelProbeReadout",
    "EffectInterval",
    "EnergyLedgerAudit",
    "PhaseLockAudit",
    "PhaseNoiseSweepAudit",
    "PhaseNoiseSweepPoint",
    "PhaseNoiseSweepPointAudit",
    "PostPumpCommonKernelAudit",
    "ProbeScaffoldClaimLocks",
    "ProbeScaffoldPilotReport",
    "ProbeSelectivityAudit",
    "energy_ledger_audit",
    "paired_effect_audit",
    "phase_lock_order_parameter",
    "phase_noise_sweep_audit",
    "post_pump_common_kernel_audit",
    "probe_scaffold_pilot_report",
    "probe_selectivity_audit",
    "validate_probe_scaffold_pilot_report",
]
