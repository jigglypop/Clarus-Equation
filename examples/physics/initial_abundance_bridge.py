"""Typed bridge from zero-dimensional composition to cosmological abundance.

The zero-dimensional entries are event/record fractions.  They become an
initial energy density only after an event density, energy scale, matching
efficiency, and a spatial coarse-graining hypersurface are supplied.  Cosmic
dilution or interaction then maps that initial density to a later abundance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MatchingChannel:
    composition: float
    relative_event_energy: float
    matching_efficiency: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.composition <= 1.0:
            raise ValueError("composition must lie in [0, 1]")
        if self.relative_event_energy < 0.0 or self.matching_efficiency < 0.0:
            raise ValueError("energy weight and matching efficiency must be non-negative")

    @property
    def dimensionless_weight(self) -> float:
        return self.composition * self.relative_event_energy * self.matching_efficiency


def matched_density_ratio(left: MatchingChannel, right: MatchingChannel) -> float:
    """Initial physical-density ratio after the common n_event*E scale cancels."""

    if right.dimensionless_weight <= 0.0:
        raise ZeroDivisionError("right matching channel has zero physical weight")
    return left.dimensionless_weight / right.dimensionless_weight


def noninteracting_ratio_evolution(
    initial_ratio: float,
    *,
    a_initial: float,
    a_final: float,
    w_left: float,
    w_right: float,
) -> float:
    """Exact constant-w ratio evolution for separately conserved fluids."""

    if initial_ratio < 0.0 or a_initial <= 0.0 or a_final <= 0.0:
        raise ValueError("ratio must be non-negative and scale factors positive")
    return initial_ratio * (a_final / a_initial) ** (-3.0 * (w_left - w_right))


def required_initial_matter_to_vacuum_ratio(
    present_ratio: float, *, a_initial: float
) -> float:
    """Inverse map for separately conserved w_m=0 and w_v=-1 components."""

    return noninteracting_ratio_evolution(
        present_ratio,
        a_initial=1.0,
        a_final=a_initial,
        w_left=0.0,
        w_right=-1.0,
    )


def kinetic_vacuum_transfer_rate_shape(
    *, gamma: float, tau: float, tau_dot: float, amplitude: float
) -> float:
    """Dimensionless-Hubble-unit shape Q=A*gamma*exp(-gamma*tau)*tau_dot.

    ``gamma*tau`` is dimensionless.  The caller is responsible for using a
    consistent time normalization for gamma and tau_dot.
    """

    if gamma <= 0.0 or amplitude < 0.0:
        raise ValueError("gamma must be positive and amplitude non-negative")
    argument = gamma * tau
    if not math.isfinite(argument):
        raise ValueError("gamma*tau must be finite")
    return amplitude * gamma * math.exp(-argument) * tau_dot

