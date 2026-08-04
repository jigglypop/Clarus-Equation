from __future__ import annotations

import math

import pytest

from reality_stone.clarus.time_travel_causality import (
    global_time_function_audit,
    periodic_time_ctc_audit,
)


def test_local_ce_inequalities_do_not_exclude_a_global_ctc() -> None:
    audit = periodic_time_ctc_audit(period=2.0, steps=256)

    assert audit.positive_lapse
    assert audit.determinant_bound
    assert audit.locally_future_timelike
    assert audit.closes_in_quotient
    assert audit.accumulated_proper_time > 0.0
    assert audit.refutes_local_no_go
    assert math.isclose(audit.lifted_time_change, audit.period)


def test_global_real_time_function_excludes_a_closed_causal_curve() -> None:
    audit = global_time_function_audit((0.1, 0.2, 0.3, 0.4))

    assert audit.strictly_increasing
    assert audit.total_time_change > 0.0
    assert not audit.closure_requires_zero_change
    assert audit.ctc_excluded


def test_non_monotone_time_does_not_trigger_global_exclusion_theorem() -> None:
    audit = global_time_function_audit((0.5, -0.5))

    assert not audit.strictly_increasing
    assert audit.closure_requires_zero_change
    assert not audit.ctc_excluded


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"period": 0.0}, "period"),
        ({"steps": 0}, "steps"),
        ({"alpha_total": 0.0}, "alpha_total"),
    ],
)
def test_periodic_ctc_gate_rejects_invalid_inputs(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        periodic_time_ctc_audit(**kwargs)
