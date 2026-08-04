from __future__ import annotations

from dataclasses import replace
import math

import pytest

from reality_stone.clarus.spacelike_marginal_gate import (
    BinnedSelectorCounts,
    SpacelikeMarginalStage,
    spacelike_marginal_gate,
    validate_spacelike_marginal_gate_report,
)


ALPHA = 0.01
DELTA_MIN = 0.10
DELTA_NS = 0.004


def _counts(
    detector_id: str,
    selector_0: tuple[int, ...],
    selector_1: tuple[int, ...],
) -> BinnedSelectorCounts:
    return BinnedSelectorCounts(
        detector_id=detector_id,
        bin_labels=tuple(f"bin_{index}" for index in range(len(selector_0))),
        selector_0_counts=selector_0,
        selector_1_counts=selector_1,
    )


def _local_effect() -> BinnedSelectorCounts:
    # p(.|S=0)=(0.4, 0.6), p(.|S=1)=(0.6, 0.4), so empirical TV=0.2.
    return _counts("A_local", (800_000, 1_200_000), (1_200_000, 800_000))


def _large_exact_nulls() -> tuple[BinnedSelectorCounts, ...]:
    return (
        _counts("B_spacelike_1", (1_000_000, 1_000_000), (1_000_000, 1_000_000)),
        _counts(
            "B_spacelike_2",
            (500_000, 500_000, 500_000, 500_000),
            (500_000, 500_000, 500_000, 500_000),
        ),
    )


def _gate(
    *,
    local_a: BinnedSelectorCounts | None = None,
    spacelike_b: tuple[BinnedSelectorCounts, ...] | None = None,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "local_a": _local_effect() if local_a is None else local_a,
        "spacelike_b": _large_exact_nulls() if spacelike_b is None else spacelike_b,
        "delta_min": DELTA_MIN,
        "delta_ns": DELTA_NS,
        "selector_randomized": True,
        "bins_predeclared_before_unblinding": True,
        "familywise_alpha": ALPHA,
        "minimum_count_per_selector": 100,
    }
    arguments.update(overrides)
    return spacelike_marginal_gate(**arguments)  # type: ignore[arg-type]


def test_large_count_exact_spacelike_null_passes_only_conditional_gate() -> None:
    report = _gate()

    assert report.bonferroni_marginal_count == 6
    assert report.bonferroni_alpha_per_marginal == pytest.approx(ALPHA / 6.0)
    assert report.simultaneous_coverage_lower_bound == pytest.approx(0.99)
    assert report.local_a.empirical_tv == pytest.approx(0.2)
    assert report.local_a.tv_lower_confidence_bound > DELTA_MIN
    assert all(audit.empirical_tv == 0.0 for audit in report.spacelike_b)
    assert all(audit.tv_upper_confidence_bound <= DELTA_NS for audit in report.spacelike_b)
    assert report.maximum_spacelike_empirical_tv == 0.0
    assert report.local_effect_control_pass
    assert report.spacelike_no_signalling_control_pass
    assert report.conditional_gate_pass
    assert not report.signalling_or_leakage_counterexample_detected
    assert report.maximum_supported_stage is (
        SpacelikeMarginalStage.CONDITIONAL_SPACELIKE_MARGINAL_CONTROL
    )

    # A pass is statistical control, not physical/CE evidence.
    assert not report.physical_response_derived
    assert not report.new_matter_derived
    assert not report.observer_selective_reality_derived
    assert not report.ce_field_identity_derived
    assert report.first_blocker.startswith("this statistical gate cannot derive")
    assert report.to_dict()["maximum_supported_stage"] == (
        "CONDITIONAL_SPACELIKE_MARGINAL_CONTROL"
    )


def test_weissman_radius_and_tv_triangle_radius_match_declared_formula() -> None:
    report = _gate()
    audit = report.local_a
    sample_count = 2_000_000
    expected_l1_radius = math.sqrt(
        2.0
        / sample_count
        * math.log((2**2 - 2) * report.bonferroni_marginal_count / ALPHA)
    )

    assert audit.selector_0_weissman_l1_radius == pytest.approx(expected_l1_radius)
    assert audit.selector_1_weissman_l1_radius == pytest.approx(expected_l1_radius)
    assert audit.tv_confidence_radius == pytest.approx(expected_l1_radius)
    assert audit.tv_lower_confidence_bound == pytest.approx(0.2 - expected_l1_radius)
    assert audit.tv_upper_confidence_bound == pytest.approx(0.2 + expected_l1_radius)


def test_spacelike_difference_is_signalling_or_leakage_counterexample() -> None:
    leaking_b = _counts(
        "B_spacelike_leak",
        (1_000_000, 1_000_000),
        (1_600_000, 400_000),
    )
    report = _gate(spacelike_b=(leaking_b,))

    assert report.local_effect_control_pass
    assert report.spacelike_b[0].empirical_tv == pytest.approx(0.3)
    assert report.spacelike_b[0].tv_lower_confidence_bound > DELTA_NS
    assert not report.spacelike_no_signalling_control_pass
    assert not report.conditional_gate_pass
    assert report.signalling_or_leakage_counterexample_detected
    assert report.maximum_supported_stage is (
        SpacelikeMarginalStage.LOCAL_SELECTOR_RESPONSE_CONTROL
    )
    assert "signalling/leakage counterexample" in report.first_blocker
    assert not report.physical_response_derived


def test_low_counts_and_vacuous_bounds_fail_closed() -> None:
    local = _counts("A_local", (6, 4), (4, 6))
    remote = _counts("B_spacelike", (5, 5), (5, 5))
    low_count = _gate(local_a=local, spacelike_b=(remote,))

    assert not low_count.all_minimum_counts_pass
    assert not low_count.all_confidence_intervals_nonvacuous
    assert low_count.local_a.tv_lower_confidence_bound == 0.0
    assert low_count.local_a.tv_upper_confidence_bound == 1.0
    assert not low_count.conditional_gate_pass
    assert low_count.maximum_supported_stage is (
        SpacelikeMarginalStage.RANDOMIZED_PREDECLARED_COUNTS
    )
    assert "minimum count" in low_count.first_blocker

    vacuous_only = _gate(
        local_a=local,
        spacelike_b=(remote,),
        minimum_count_per_selector=1,
    )
    assert vacuous_only.all_minimum_counts_pass
    assert not vacuous_only.all_confidence_intervals_nonvacuous
    assert "vacuous" in vacuous_only.first_blocker
    assert not vacuous_only.local_effect_control_pass


def test_large_count_local_null_cannot_pass_as_an_empty_experiment() -> None:
    null_local = _counts(
        "A_local",
        (1_000_000, 1_000_000),
        (1_000_000, 1_000_000),
    )
    report = _gate(local_a=null_local)

    assert report.spacelike_no_signalling_control_pass
    assert not report.local_effect_control_pass
    assert not report.conditional_gate_pass
    assert report.maximum_supported_stage is (
        SpacelikeMarginalStage.SIMULTANEOUS_TV_BOUND_CONTROL
    )
    assert report.first_blocker == (
        "local A TV lower confidence bound is not strictly above delta_min"
    )


def test_randomization_and_predeclaration_are_fail_closed_stage_prerequisites() -> None:
    not_randomized = _gate(selector_randomized=False)
    assert not not_randomized.conditional_gate_pass
    assert not_randomized.maximum_supported_stage is (
        SpacelikeMarginalStage.RAW_FINITE_BIN_COUNTS
    )
    assert not_randomized.first_blocker == "binary selector randomization is not declared"

    not_predeclared = _gate(bins_predeclared_before_unblinding=False)
    assert not not_predeclared.conditional_gate_pass
    assert not_predeclared.maximum_supported_stage is (
        SpacelikeMarginalStage.RAW_FINITE_BIN_COUNTS
    )
    assert "not declared before unblinding" in not_predeclared.first_blocker


def test_count_input_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="at least two"):
        BinnedSelectorCounts("A", ("only",), (1,), (1,))
    with pytest.raises(ValueError, match="unique"):
        BinnedSelectorCounts("A", ("same", "same"), (1, 1), (1, 1))
    with pytest.raises(ValueError, match="match"):
        BinnedSelectorCounts("A", ("x", "y"), (1,), (1, 1))
    with pytest.raises(ValueError, match="at least 0"):
        BinnedSelectorCounts("A", ("x", "y"), (-1, 2), (1, 1))
    with pytest.raises(ValueError, match="integer"):
        BinnedSelectorCounts("A", ("x", "y"), (True, 1), (1, 1))
    with pytest.raises(ValueError, match="positive total"):
        BinnedSelectorCounts("A", ("x", "y"), (0, 0), (1, 1))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"delta_min": -0.1}, "delta_min"),
        ({"delta_ns": 1.0}, "delta_ns"),
        ({"familywise_alpha": 0.0}, "familywise_alpha"),
        ({"minimum_count_per_selector": True}, "integer"),
        ({"selector_randomized": 1}, "bool"),
        ({"bins_predeclared_before_unblinding": 1}, "bool"),
    ],
)
def test_gate_threshold_and_declaration_validation_is_strict(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _gate(**override)


def test_detector_collection_requires_multiple_roles_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _gate(spacelike_b=())
    duplicate = _counts("A_local", (1_000_000, 1_000_000), (1_000_000, 1_000_000))
    with pytest.raises(ValueError, match="unique"):
        _gate(spacelike_b=(duplicate,))
    with pytest.raises(ValueError, match="only BinnedSelectorCounts"):
        _gate(spacelike_b=(object(),))  # type: ignore[arg-type]


def test_report_validator_rejects_claim_stage_and_audit_tampering() -> None:
    report = _gate()
    with pytest.raises(ValueError, match="claim locks"):
        validate_spacelike_marginal_gate_report(
            replace(report, new_matter_derived=True)
        )
    with pytest.raises(ValueError, match="stage or gate summary"):
        validate_spacelike_marginal_gate_report(
            replace(
                report,
                maximum_supported_stage=SpacelikeMarginalStage.RAW_FINITE_BIN_COUNTS,
            )
        )
    with pytest.raises(ValueError, match="marginal TV audit"):
        validate_spacelike_marginal_gate_report(
            replace(
                report,
                local_a=replace(report.local_a, empirical_tv=0.9),
            )
        )
