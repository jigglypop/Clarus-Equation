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


@dataclass(frozen=True)
class LogRatioIdentifiability:
    """Local log-Jacobian audit for one matched density ratio.

    The parameter order is ``q_left, q_right, epsilon_left, epsilon_right,
    eta_left, eta_right``.  One observed ratio supplies one independent row,
    so the unconstrained local map has rank one and a five-dimensional null
    space.  Composition normalization or a microscopic matching law can reduce
    that null space, but neither is supplied by the ratio itself.
    """

    density_ratio: float
    parameter_order: tuple[str, ...]
    log_jacobian: tuple[float, ...]
    jacobian_rank: int
    nullity: int
    energy_efficiency_null_direction: tuple[float, ...]
    role: str = "CONDITIONAL_IDENTIFIABILITY_THEOREM_NOT_ABUNDANCE_PREDICTION"


def log_density_ratio_identifiability(
    left: MatchingChannel,
    right: MatchingChannel,
) -> LogRatioIdentifiability:
    """Return the exact local log-Jacobian and a matching degeneracy.

    For positive channel entries,

        log(rho_left/rho_right)
        = log(q_left/q_right)
        + log(epsilon_left/epsilon_right)
        + log(eta_left/eta_right).

    The returned null direction rescales the left relative event energy and
    matching efficiency oppositely.  It therefore changes microscopic
    matching inputs while leaving the physical-density ratio unchanged.
    """

    entries = (
        left.composition,
        right.composition,
        left.relative_event_energy,
        right.relative_event_energy,
        left.matching_efficiency,
        right.matching_efficiency,
    )
    if any(value <= 0.0 for value in entries):
        raise ValueError("log-identifiability requires strictly positive entries")
    return LogRatioIdentifiability(
        density_ratio=matched_density_ratio(left, right),
        parameter_order=(
            "log_q_left",
            "log_q_right",
            "log_epsilon_left",
            "log_epsilon_right",
            "log_eta_left",
            "log_eta_right",
        ),
        log_jacobian=(1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        jacobian_rank=1,
        nullity=5,
        energy_efficiency_null_direction=(0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
    )


def matching_degenerate_channel(
    channel: MatchingChannel,
    *,
    energy_rescaling: float,
) -> MatchingChannel:
    """Move along an exact energy-efficiency degeneracy of the bridge."""

    if not math.isfinite(energy_rescaling) or energy_rescaling <= 0.0:
        raise ValueError("energy_rescaling must be finite and positive")
    return MatchingChannel(
        composition=channel.composition,
        relative_event_energy=(
            channel.relative_event_energy * energy_rescaling
        ),
        matching_efficiency=(
            channel.matching_efficiency / energy_rescaling
        ),
    )


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
