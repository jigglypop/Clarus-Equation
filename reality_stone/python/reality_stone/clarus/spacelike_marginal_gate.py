"""Fail-closed finite-sample gate for selector-conditioned spacelike marginals.

The input is a randomized binary selector ``S in {0, 1}``, one local detector
``A``, and one or more detectors ``B`` declared spacelike to the selector
region.  Every detector has a predeclared finite outcome alphabet and two
multinomial count vectors.  For a ``k``-bin empirical distribution based on
``n`` observations, this module uses the Weissman L1 concentration inequality

    P(||p_hat - p||_1 >= epsilon)
        <= (2**k - 2) exp(-n epsilon**2 / 2).

The familywise error probability is split by Bonferroni over all
``2 * (1 + number_of_B_detectors)`` selector-conditioned marginals.  The
triangle inequality then gives a simultaneous radius for each total-variation
distance,

    |TV(p_1, p_0) - TV(p_hat_1, p_hat_0)| <= (epsilon_0 + epsilon_1) / 2.

This is deliberately conservative.  A conditional gate passes only when the
local lower confidence bound is strictly above ``delta_min`` and every
spacelike upper confidence bound is at most ``delta_ns``.  Low counts and a
fully vacuous ``[0, 1]`` TV interval fail closed.  Passing is a statistical
no-signalling control only; physical response, new matter, observer-selective
reality, and CE-field identity claims remain hard false.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any


WEISSMAN_L1_BOUND = (
    "P(||p_hat-p||_1>=epsilon)<=(2^k-2)exp(-n*epsilon^2/2)"
)
SIMULTANEOUS_BOUND_METHOD = (
    "Weissman multinomial L1 concentration; Bonferroni over every detector-selector "
    "marginal; TV triangle inequality"
)


class SpacelikeMarginalStage(str, Enum):
    """Monotone statistical stages, none of which is a physical-field claim."""

    RAW_FINITE_BIN_COUNTS = "RAW_FINITE_BIN_COUNTS"
    RANDOMIZED_PREDECLARED_COUNTS = "RANDOMIZED_PREDECLARED_COUNTS"
    SIMULTANEOUS_TV_BOUND_CONTROL = "SIMULTANEOUS_TV_BOUND_CONTROL"
    LOCAL_SELECTOR_RESPONSE_CONTROL = "LOCAL_SELECTOR_RESPONSE_CONTROL"
    CONDITIONAL_SPACELIKE_MARGINAL_CONTROL = (
        "CONDITIONAL_SPACELIKE_MARGINAL_CONTROL"
    )


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(
    value: Real,
    *,
    name: str,
    lower_open: bool,
    upper_open: bool,
) -> float:
    result = _finite_real(value, name=name)
    lower_invalid = result <= 0.0 if lower_open else result < 0.0
    upper_invalid = result >= 1.0 if upper_open else result > 1.0
    if lower_invalid or upper_invalid:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise ValueError(f"{name} must lie in {left}0, 1{right}")
    return result


def _strict_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _nonempty_string(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of bin labels")
    result = tuple(
        _nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two finite bins")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique bin labels")
    return result


def _counts(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of integer counts")
    result = tuple(
        _strict_integer(value, name=f"{name}[{index}]", minimum=0)
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if sum(result) <= 0:
        raise ValueError(f"{name} must have a positive total count")
    return result


@dataclass(frozen=True)
class BinnedSelectorCounts:
    """Predeclared finite-bin counts for one detector under ``S=0`` and ``S=1``."""

    detector_id: str
    bin_labels: Sequence[str]
    selector_0_counts: Sequence[int]
    selector_1_counts: Sequence[int]

    def __post_init__(self) -> None:
        detector_id = _nonempty_string(self.detector_id, name="detector_id")
        labels = _labels(self.bin_labels, name="bin_labels")
        selector_0 = _counts(self.selector_0_counts, name="selector_0_counts")
        selector_1 = _counts(self.selector_1_counts, name="selector_1_counts")
        if len(selector_0) != len(labels) or len(selector_1) != len(labels):
            raise ValueError("selector count vectors must match the predeclared bin count")
        object.__setattr__(self, "detector_id", detector_id)
        object.__setattr__(self, "bin_labels", labels)
        object.__setattr__(self, "selector_0_counts", selector_0)
        object.__setattr__(self, "selector_1_counts", selector_1)


@dataclass(frozen=True)
class MarginalTVAudit:
    """One detector's empirical TV distance and simultaneous confidence interval."""

    detector_id: str
    bin_labels: tuple[str, ...]
    selector_0_counts: tuple[int, ...]
    selector_1_counts: tuple[int, ...]
    selector_0_total: int
    selector_1_total: int
    selector_0_empirical_probability: tuple[float, ...]
    selector_1_empirical_probability: tuple[float, ...]
    empirical_tv: float
    bonferroni_alpha_per_marginal: float
    selector_0_weissman_l1_radius: float
    selector_1_weissman_l1_radius: float
    tv_confidence_radius: float
    tv_lower_confidence_bound: float
    tv_upper_confidence_bound: float
    minimum_count_pass: bool
    confidence_interval_nonvacuous: bool


@dataclass(frozen=True)
class SpacelikeMarginalGateReport:
    """Serializable fail-closed report for the local/spacelike TV gate."""

    schema_version: str
    confidence_method: str
    weissman_l1_bound: str
    selector_randomized: bool
    bins_predeclared_before_unblinding: bool
    delta_min: float
    delta_ns: float
    familywise_alpha: float
    simultaneous_coverage_lower_bound: float
    bonferroni_marginal_count: int
    bonferroni_alpha_per_marginal: float
    minimum_count_per_selector: int
    local_a: MarginalTVAudit
    spacelike_b: tuple[MarginalTVAudit, ...]
    all_minimum_counts_pass: bool
    all_confidence_intervals_nonvacuous: bool
    local_effect_control_pass: bool
    spacelike_no_signalling_control_pass: bool
    conditional_gate_pass: bool
    maximum_spacelike_empirical_tv: float
    maximum_spacelike_upper_confidence_bound: float
    signalling_or_leakage_counterexample_detected: bool
    maximum_supported_stage: SpacelikeMarginalStage
    physical_response_derived: bool
    new_matter_derived: bool
    observer_selective_reality_derived: bool
    ce_field_identity_derived: bool
    first_blocker: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maximum_supported_stage"] = self.maximum_supported_stage.value
        return payload


def _log_weissman_prefactor(bin_count: int) -> float:
    """Return ``log(2**k - 2)`` without constructing a potentially huge integer."""

    log_two = math.log(2.0)
    return bin_count * log_two + math.log1p(-math.exp((1 - bin_count) * log_two))


def _weissman_l1_radius(
    *,
    bin_count: int,
    sample_count: int,
    alpha_per_marginal: float,
) -> float:
    log_ratio = _log_weissman_prefactor(bin_count) - math.log(alpha_per_marginal)
    radius = math.sqrt(2.0 * log_ratio / sample_count)
    # L1 distance between probability vectors is deterministically at most two.
    return min(2.0, radius)


def _marginal_audit(
    counts: BinnedSelectorCounts,
    *,
    alpha_per_marginal: float,
    minimum_count_per_selector: int,
) -> MarginalTVAudit:
    selector_0_total = sum(counts.selector_0_counts)
    selector_1_total = sum(counts.selector_1_counts)
    probability_0 = tuple(value / selector_0_total for value in counts.selector_0_counts)
    probability_1 = tuple(value / selector_1_total for value in counts.selector_1_counts)
    empirical_tv = 0.5 * sum(
        abs(value_1 - value_0) for value_0, value_1 in zip(probability_0, probability_1)
    )
    radius_0 = _weissman_l1_radius(
        bin_count=len(counts.bin_labels),
        sample_count=selector_0_total,
        alpha_per_marginal=alpha_per_marginal,
    )
    radius_1 = _weissman_l1_radius(
        bin_count=len(counts.bin_labels),
        sample_count=selector_1_total,
        alpha_per_marginal=alpha_per_marginal,
    )
    tv_radius = min(1.0, 0.5 * (radius_0 + radius_1))
    lower_bound = max(0.0, empirical_tv - tv_radius)
    upper_bound = min(1.0, empirical_tv + tv_radius)
    minimum_count_pass = (
        selector_0_total >= minimum_count_per_selector
        and selector_1_total >= minimum_count_per_selector
    )
    # A clipped [0, 1] interval contains no information about TV at all.
    confidence_interval_nonvacuous = lower_bound > 0.0 or upper_bound < 1.0
    return MarginalTVAudit(
        detector_id=counts.detector_id,
        bin_labels=tuple(counts.bin_labels),
        selector_0_counts=tuple(counts.selector_0_counts),
        selector_1_counts=tuple(counts.selector_1_counts),
        selector_0_total=selector_0_total,
        selector_1_total=selector_1_total,
        selector_0_empirical_probability=probability_0,
        selector_1_empirical_probability=probability_1,
        empirical_tv=empirical_tv,
        bonferroni_alpha_per_marginal=alpha_per_marginal,
        selector_0_weissman_l1_radius=radius_0,
        selector_1_weissman_l1_radius=radius_1,
        tv_confidence_radius=tv_radius,
        tv_lower_confidence_bound=lower_bound,
        tv_upper_confidence_bound=upper_bound,
        minimum_count_pass=minimum_count_pass,
        confidence_interval_nonvacuous=confidence_interval_nonvacuous,
    )


def _stage(
    *,
    declarations_pass: bool,
    simultaneous_bound_control_pass: bool,
    local_effect_control_pass: bool,
    conditional_gate_pass: bool,
) -> SpacelikeMarginalStage:
    stage = SpacelikeMarginalStage.RAW_FINITE_BIN_COUNTS
    if declarations_pass:
        stage = SpacelikeMarginalStage.RANDOMIZED_PREDECLARED_COUNTS
    if simultaneous_bound_control_pass:
        stage = SpacelikeMarginalStage.SIMULTANEOUS_TV_BOUND_CONTROL
    if local_effect_control_pass:
        stage = SpacelikeMarginalStage.LOCAL_SELECTOR_RESPONSE_CONTROL
    if conditional_gate_pass:
        stage = SpacelikeMarginalStage.CONDITIONAL_SPACELIKE_MARGINAL_CONTROL
    return stage


def _blockers(
    *,
    selector_randomized: bool,
    bins_predeclared_before_unblinding: bool,
    local_a: MarginalTVAudit,
    spacelike_b: tuple[MarginalTVAudit, ...],
    delta_min: float,
    delta_ns: float,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not selector_randomized:
        blockers.append("binary selector randomization is not declared")
    if not bins_predeclared_before_unblinding:
        blockers.append("finite outcome bins were not declared before unblinding")

    for audit in (local_a, *spacelike_b):
        if not audit.minimum_count_pass:
            blockers.append(
                f"{audit.detector_id} has a selector arm below the predeclared minimum count"
            )
        if not audit.confidence_interval_nonvacuous:
            blockers.append(
                f"{audit.detector_id} has a vacuous [0, 1] simultaneous TV interval"
            )

    if not local_a.tv_lower_confidence_bound > delta_min:
        blockers.append(
            "local A TV lower confidence bound is not strictly above delta_min"
        )
    for audit in spacelike_b:
        if audit.tv_upper_confidence_bound > delta_ns:
            if audit.tv_lower_confidence_bound > delta_ns:
                blockers.append(
                    f"{audit.detector_id} spacelike TV lower bound exceeds delta_ns; "
                    "this is a signalling/leakage counterexample to the gate"
                )
            else:
                blockers.append(
                    f"{audit.detector_id} spacelike TV upper confidence bound exceeds delta_ns"
                )

    blockers.append(
        "this statistical gate cannot derive a physical response, new matter, "
        "observer-selective reality, or CE field identity"
    )
    return tuple(blockers)


def spacelike_marginal_gate(
    *,
    local_a: BinnedSelectorCounts,
    spacelike_b: Sequence[BinnedSelectorCounts],
    delta_min: Real,
    delta_ns: Real,
    selector_randomized: bool,
    bins_predeclared_before_unblinding: bool,
    familywise_alpha: Real = 0.05,
    minimum_count_per_selector: int = 100,
) -> SpacelikeMarginalGateReport:
    """Compute simultaneous local-effect and spacelike-no-signalling controls.

    ``delta_min`` and ``delta_ns`` are TV-distance thresholds in ``[0, 1)``.
    The returned conditional pass is meaningful only under the two explicit
    randomization/preregistration declarations; both are strict booleans.
    """

    if not isinstance(local_a, BinnedSelectorCounts):
        raise ValueError("local_a must be BinnedSelectorCounts")
    if isinstance(spacelike_b, (str, bytes)) or not isinstance(spacelike_b, Sequence):
        raise ValueError("spacelike_b must be a non-empty sequence of BinnedSelectorCounts")
    spacelike_counts = tuple(spacelike_b)
    if not spacelike_counts:
        raise ValueError("spacelike_b must contain at least one detector")
    if not all(isinstance(value, BinnedSelectorCounts) for value in spacelike_counts):
        raise ValueError("spacelike_b must contain only BinnedSelectorCounts")

    detector_ids = (local_a.detector_id, *(value.detector_id for value in spacelike_counts))
    if len(set(detector_ids)) != len(detector_ids):
        raise ValueError("detector_id values must be unique across local A and spacelike B")

    delta_min_value = _unit_interval(
        delta_min,
        name="delta_min",
        lower_open=False,
        upper_open=True,
    )
    delta_ns_value = _unit_interval(
        delta_ns,
        name="delta_ns",
        lower_open=False,
        upper_open=True,
    )
    alpha = _unit_interval(
        familywise_alpha,
        name="familywise_alpha",
        lower_open=True,
        upper_open=True,
    )
    minimum_count = _strict_integer(
        minimum_count_per_selector,
        name="minimum_count_per_selector",
        minimum=1,
    )
    randomized = _strict_bool(selector_randomized, name="selector_randomized")
    predeclared = _strict_bool(
        bins_predeclared_before_unblinding,
        name="bins_predeclared_before_unblinding",
    )

    bonferroni_marginal_count = 2 * (1 + len(spacelike_counts))
    alpha_per_marginal = alpha / bonferroni_marginal_count
    local_audit = _marginal_audit(
        local_a,
        alpha_per_marginal=alpha_per_marginal,
        minimum_count_per_selector=minimum_count,
    )
    spacelike_audits = tuple(
        _marginal_audit(
            value,
            alpha_per_marginal=alpha_per_marginal,
            minimum_count_per_selector=minimum_count,
        )
        for value in spacelike_counts
    )
    all_audits = (local_audit, *spacelike_audits)
    all_minimum_counts_pass = all(value.minimum_count_pass for value in all_audits)
    all_intervals_nonvacuous = all(
        value.confidence_interval_nonvacuous for value in all_audits
    )
    declarations_pass = randomized and predeclared
    simultaneous_bound_control_pass = (
        declarations_pass and all_minimum_counts_pass and all_intervals_nonvacuous
    )
    local_effect_control_pass = (
        simultaneous_bound_control_pass
        and local_audit.tv_lower_confidence_bound > delta_min_value
    )
    spacelike_no_signalling_control_pass = (
        simultaneous_bound_control_pass
        and all(
            value.tv_upper_confidence_bound <= delta_ns_value
            for value in spacelike_audits
        )
    )
    conditional_gate_pass = (
        local_effect_control_pass and spacelike_no_signalling_control_pass
    )
    signalling_counterexample = any(
        value.tv_lower_confidence_bound > delta_ns_value
        for value in spacelike_audits
    )
    maximum_spacelike_empirical_tv = max(
        value.empirical_tv for value in spacelike_audits
    )
    maximum_spacelike_ucb = max(
        value.tv_upper_confidence_bound for value in spacelike_audits
    )
    blockers = _blockers(
        selector_randomized=randomized,
        bins_predeclared_before_unblinding=predeclared,
        local_a=local_audit,
        spacelike_b=spacelike_audits,
        delta_min=delta_min_value,
        delta_ns=delta_ns_value,
    )
    report = SpacelikeMarginalGateReport(
        schema_version="spacelike-marginal-gate-v1",
        confidence_method=SIMULTANEOUS_BOUND_METHOD,
        weissman_l1_bound=WEISSMAN_L1_BOUND,
        selector_randomized=randomized,
        bins_predeclared_before_unblinding=predeclared,
        delta_min=delta_min_value,
        delta_ns=delta_ns_value,
        familywise_alpha=alpha,
        simultaneous_coverage_lower_bound=1.0 - alpha,
        bonferroni_marginal_count=bonferroni_marginal_count,
        bonferroni_alpha_per_marginal=alpha_per_marginal,
        minimum_count_per_selector=minimum_count,
        local_a=local_audit,
        spacelike_b=spacelike_audits,
        all_minimum_counts_pass=all_minimum_counts_pass,
        all_confidence_intervals_nonvacuous=all_intervals_nonvacuous,
        local_effect_control_pass=local_effect_control_pass,
        spacelike_no_signalling_control_pass=spacelike_no_signalling_control_pass,
        conditional_gate_pass=conditional_gate_pass,
        maximum_spacelike_empirical_tv=maximum_spacelike_empirical_tv,
        maximum_spacelike_upper_confidence_bound=maximum_spacelike_ucb,
        signalling_or_leakage_counterexample_detected=signalling_counterexample,
        maximum_supported_stage=_stage(
            declarations_pass=declarations_pass,
            simultaneous_bound_control_pass=simultaneous_bound_control_pass,
            local_effect_control_pass=local_effect_control_pass,
            conditional_gate_pass=conditional_gate_pass,
        ),
        physical_response_derived=False,
        new_matter_derived=False,
        observer_selective_reality_derived=False,
        ce_field_identity_derived=False,
        first_blocker=blockers[0],
        blockers=blockers,
    )
    return validate_spacelike_marginal_gate_report(report)


def validate_spacelike_marginal_gate_report(
    report: SpacelikeMarginalGateReport,
) -> SpacelikeMarginalGateReport:
    """Reject tampered stages, confidence summaries, or physical claim locks."""

    if not isinstance(report, SpacelikeMarginalGateReport):
        raise ValueError("report must be SpacelikeMarginalGateReport")
    if any(
        (
            report.physical_response_derived,
            report.new_matter_derived,
            report.observer_selective_reality_derived,
            report.ce_field_identity_derived,
        )
    ):
        raise ValueError("spacelike marginal physical/new-matter/CE claim locks must remain false")
    if report.bonferroni_marginal_count != 2 * (1 + len(report.spacelike_b)):
        raise ValueError("Bonferroni marginal count is inconsistent with detector count")
    expected_alpha_per_marginal = (
        report.familywise_alpha / report.bonferroni_marginal_count
    )
    if report.bonferroni_alpha_per_marginal != expected_alpha_per_marginal:
        raise ValueError("Bonferroni alpha allocation is inconsistent")

    expected_local = _marginal_audit(
        BinnedSelectorCounts(
            report.local_a.detector_id,
            report.local_a.bin_labels,
            report.local_a.selector_0_counts,
            report.local_a.selector_1_counts,
        ),
        alpha_per_marginal=expected_alpha_per_marginal,
        minimum_count_per_selector=report.minimum_count_per_selector,
    )
    expected_spacelike = tuple(
        _marginal_audit(
            BinnedSelectorCounts(
                value.detector_id,
                value.bin_labels,
                value.selector_0_counts,
                value.selector_1_counts,
            ),
            alpha_per_marginal=expected_alpha_per_marginal,
            minimum_count_per_selector=report.minimum_count_per_selector,
        )
        for value in report.spacelike_b
    )
    if report.local_a != expected_local or report.spacelike_b != expected_spacelike:
        raise ValueError("marginal TV audit is inconsistent with raw finite-bin counts")

    all_audits = (expected_local, *expected_spacelike)
    expected_counts_pass = all(value.minimum_count_pass for value in all_audits)
    expected_nonvacuous = all(
        value.confidence_interval_nonvacuous for value in all_audits
    )
    declarations_pass = (
        report.selector_randomized and report.bins_predeclared_before_unblinding
    )
    simultaneous_pass = (
        declarations_pass and expected_counts_pass and expected_nonvacuous
    )
    expected_local_pass = (
        simultaneous_pass
        and expected_local.tv_lower_confidence_bound > report.delta_min
    )
    expected_spacelike_pass = (
        simultaneous_pass
        and all(
            value.tv_upper_confidence_bound <= report.delta_ns
            for value in expected_spacelike
        )
    )
    expected_gate_pass = expected_local_pass and expected_spacelike_pass
    expected_counterexample = any(
        value.tv_lower_confidence_bound > report.delta_ns
        for value in expected_spacelike
    )
    expected_stage = _stage(
        declarations_pass=declarations_pass,
        simultaneous_bound_control_pass=simultaneous_pass,
        local_effect_control_pass=expected_local_pass,
        conditional_gate_pass=expected_gate_pass,
    )
    expected_blockers = _blockers(
        selector_randomized=report.selector_randomized,
        bins_predeclared_before_unblinding=report.bins_predeclared_before_unblinding,
        local_a=expected_local,
        spacelike_b=expected_spacelike,
        delta_min=report.delta_min,
        delta_ns=report.delta_ns,
    )
    expected_summary = (
        expected_counts_pass,
        expected_nonvacuous,
        expected_local_pass,
        expected_spacelike_pass,
        expected_gate_pass,
        max(value.empirical_tv for value in expected_spacelike),
        max(value.tv_upper_confidence_bound for value in expected_spacelike),
        expected_counterexample,
        expected_stage,
        expected_blockers[0],
        expected_blockers,
    )
    actual_summary = (
        report.all_minimum_counts_pass,
        report.all_confidence_intervals_nonvacuous,
        report.local_effect_control_pass,
        report.spacelike_no_signalling_control_pass,
        report.conditional_gate_pass,
        report.maximum_spacelike_empirical_tv,
        report.maximum_spacelike_upper_confidence_bound,
        report.signalling_or_leakage_counterexample_detected,
        report.maximum_supported_stage,
        report.first_blocker,
        report.blockers,
    )
    if actual_summary != expected_summary:
        raise ValueError("spacelike marginal stage or gate summary is inconsistent")
    return report


__all__ = [
    "BinnedSelectorCounts",
    "MarginalTVAudit",
    "SIMULTANEOUS_BOUND_METHOD",
    "SpacelikeMarginalGateReport",
    "SpacelikeMarginalStage",
    "WEISSMAN_L1_BOUND",
    "spacelike_marginal_gate",
    "validate_spacelike_marginal_gate_report",
]
