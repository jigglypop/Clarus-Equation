"""Logical gates for the CE closed-timelike-curve (CTC) claim.

The module deliberately separates two statements that are conflated in the
current documentation:

1. Positive local lapse and a contractive transition determinant.
2. Existence of a global real-valued time function that increases on every
   allowed causal curve.

The first statement is local and does not exclude non-trivial global time
topology.  The second excludes CTCs directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class PeriodicTimeCTCAudit:
    """Audit of a future-directed timelike loop on ``S1 x R^3``."""

    period: float
    steps: int
    lapse: float
    proper_time_per_step: float
    determinant_squared_per_step: float
    accumulated_proper_time: float
    lifted_time_change: float
    quotient_start_time: float
    quotient_end_time: float
    positive_lapse: bool
    determinant_bound: bool
    locally_future_timelike: bool
    closes_in_quotient: bool

    @property
    def refutes_local_no_go(self) -> bool:
        """Whether all local CE gates hold although the causal curve closes."""

        return (
            self.positive_lapse
            and self.determinant_bound
            and self.locally_future_timelike
            and self.closes_in_quotient
            and self.accumulated_proper_time > 0.0
        )


@dataclass(frozen=True)
class GlobalTimeFunctionAudit:
    """Finite-partition form of the global-time-function contradiction."""

    increments: tuple[float, ...]
    total_time_change: float
    strictly_increasing: bool
    closure_requires_zero_change: bool
    ctc_excluded: bool


def periodic_time_ctc_audit(
    *,
    period: float = 1.0,
    steps: int = 128,
    alpha_total: float = 0.1592,
) -> PeriodicTimeCTCAudit:
    """Construct a CTC countermodel satisfying the local CE inequalities.

    Take the flat metric ``ds^2 = -dt^2 + dx^2 + dy^2 + dz^2``, identify
    ``t ~ t + period``, and hold the spatial coordinates fixed.  Every step is
    future-directed and timelike with lapse one.  After one period the curve
    returns to the same spacetime event in the quotient manifold.

    This is a logical countermodel, not a claim that periodic time is physical.
    It tests whether the *local* inequalities alone imply absence of CTCs.
    """

    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not math.isfinite(alpha_total) or alpha_total <= 0.0:
        raise ValueError("alpha_total must be finite and positive")

    proper_time_per_step = period / steps
    lapse = 1.0
    determinant_squared = math.exp(
        -alpha_total * lapse * proper_time_per_step
    )
    lifted_time_change = steps * proper_time_per_step
    quotient_start = 0.0
    quotient_end = math.fmod(lifted_time_change, period)
    closure_tolerance = 32.0 * math.ulp(period)
    closes = min(abs(quotient_end), abs(period - quotient_end)) <= closure_tolerance

    return PeriodicTimeCTCAudit(
        period=period,
        steps=steps,
        lapse=lapse,
        proper_time_per_step=proper_time_per_step,
        determinant_squared_per_step=determinant_squared,
        accumulated_proper_time=lifted_time_change,
        lifted_time_change=lifted_time_change,
        quotient_start_time=quotient_start,
        quotient_end_time=quotient_end,
        positive_lapse=lapse > 0.0,
        determinant_bound=0.0 < determinant_squared <= 1.0,
        locally_future_timelike=proper_time_per_step > 0.0,
        closes_in_quotient=closes,
    )


def global_time_function_audit(
    increments: Iterable[float],
    *,
    tolerance: float = 1e-12,
) -> GlobalTimeFunctionAudit:
    """Check the contradiction between strict time monotonicity and closure.

    If a real-valued global time function strictly increases along every
    future-directed causal segment, every finite partition has positive time
    increments.  Their sum is positive, while a closed curve would require the
    same function to have identical endpoint values and hence a zero sum.
    """

    values = tuple(float(value) for value in increments)
    if not values:
        raise ValueError("at least one time increment is required")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("time increments must be finite")

    total = math.fsum(values)
    increasing = all(value > tolerance for value in values)
    closure_requires_zero = abs(total) <= tolerance

    return GlobalTimeFunctionAudit(
        increments=values,
        total_time_change=total,
        strictly_increasing=increasing,
        closure_requires_zero_change=closure_requires_zero,
        ctc_excluded=increasing and not closure_requires_zero,
    )
