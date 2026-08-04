"""Gates from a boundary target label to localized spatial actuation.

The audits separate three logically different requirements: a target-dependent
actuator, causal delivery of the target information, and a throat scale that
simultaneously meets the local density and CE coherence requirements.  Passing
these necessary gates would still not derive a conserved stress tensor or a
stable wormhole solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Iterable

import numpy as np

from .spatial_folding import NEWTON_G_M3_KG_S2, SPEED_OF_LIGHT_M_S


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class TargetLocalizationAudit:
    response_rank: int
    command_count: int
    location_count: int
    selected_locations: tuple[int, ...]
    all_commands_localized: bool
    all_commands_meet_required_density: bool
    target_dependent_actuator_required: bool
    actuator_map_derived_from_ce: bool


@dataclass(frozen=True)
class CausalTargetDeliveryAudit:
    distance_m: float
    target_information_bits: float
    earliest_delivery_s: float
    requested_activation_s: float
    deadline_satisfied: bool
    instantaneous_adaptive_activation: bool
    preinstalled_receiver_required: bool


@dataclass(frozen=True)
class ThroatScaleWindowAudit:
    minimum_radius_from_density_m: float
    maximum_radius_from_coherence_m: float
    feasible_radius_window_exists: bool
    scale_gap_ratio: float
    conserved_stress_tensor_derived: bool
    stable_wormhole_established: bool


def target_localization_audit(
    actuator_response_j_m3: ArrayLike,
    *,
    required_density_j_m3: float,
) -> TargetLocalizationAudit:
    """Audit whether command ``a`` peaks strongly enough at location ``a``.

    Rows are spatial locations and columns are target commands.  Entries are
    magnitudes of negative null-projected stress.  A command is localized only
    when its diagonal response is strictly larger than every off-target response.
    Merely supplying such a matrix is an actuator assumption, not a CE derivation.
    """

    response = np.asarray(actuator_response_j_m3, dtype=float)
    if response.ndim != 2 or response.shape[0] < 1 or response.shape[0] != response.shape[1]:
        raise ValueError("actuator_response_j_m3 must be a non-empty square matrix")
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("actuator_response_j_m3 must be finite and non-negative")
    required = float(required_density_j_m3)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("required_density_j_m3 must be finite and positive")

    count = response.shape[0]
    selected = tuple(int(np.argmax(response[:, command])) for command in range(count))
    localized = all(
        selected[command] == command
        and np.count_nonzero(response[:, command] == response[command, command]) == 1
        for command in range(count)
    )
    sufficient = all(response[command, command] >= required for command in range(count))
    return TargetLocalizationAudit(
        response_rank=int(np.linalg.matrix_rank(response)),
        command_count=count,
        location_count=count,
        selected_locations=selected,
        all_commands_localized=localized,
        all_commands_meet_required_density=sufficient,
        target_dependent_actuator_required=count > 1,
        actuator_map_derived_from_ce=False,
    )


def causal_target_delivery_audit(
    *,
    distance_m: float,
    candidate_count: int,
    requested_activation_s: float,
    signal_speed_fraction_c: float = 1.0,
) -> CausalTargetDeliveryAudit:
    """Apply the light-cone lower bound to an adaptive remote target command."""

    distance = float(distance_m)
    deadline = float(requested_activation_s)
    beta = float(signal_speed_fraction_c)
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, Integral):
        raise ValueError("candidate_count must be an integer")
    count = int(candidate_count)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    if not math.isfinite(deadline) or deadline < 0.0:
        raise ValueError("requested_activation_s must be finite and non-negative")
    if count < 1:
        raise ValueError("candidate_count must be positive")
    if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("signal_speed_fraction_c must lie in (0, 1]")

    earliest = distance / (beta * SPEED_OF_LIGHT_M_S)
    bits = math.log2(count)
    deadline_satisfied = deadline >= earliest
    return CausalTargetDeliveryAudit(
        distance_m=distance,
        target_information_bits=bits,
        earliest_delivery_s=earliest,
        requested_activation_s=deadline,
        deadline_satisfied=deadline_satisfied,
        instantaneous_adaptive_activation=distance == 0.0 and deadline_satisfied,
        preinstalled_receiver_required=distance > 0.0,
    )


def throat_scale_window_audit(
    *,
    candidate_negative_density_j_m3: float,
    ce_correlation_length_m: float,
    shape_derivative: float = -1.0,
) -> ThroatScaleWindowAudit:
    """Intersect the Morris--Thorne density and CE coherence radius bounds."""

    density = float(candidate_negative_density_j_m3)
    correlation = float(ce_correlation_length_m)
    b_prime = float(shape_derivative)
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("candidate_negative_density_j_m3 must be finite and positive")
    if not math.isfinite(correlation) or correlation <= 0.0:
        raise ValueError("ce_correlation_length_m must be finite and positive")
    if not math.isfinite(b_prime) or b_prime >= 1.0:
        raise ValueError("shape_derivative must be finite and below one")

    coefficient = (
        SPEED_OF_LIGHT_M_S**4
        * (1.0 - b_prime)
        / (8.0 * math.pi * NEWTON_G_M3_KG_S2)
    )
    minimum_radius = math.sqrt(coefficient / density)
    maximum_radius = correlation
    return ThroatScaleWindowAudit(
        minimum_radius_from_density_m=minimum_radius,
        maximum_radius_from_coherence_m=maximum_radius,
        feasible_radius_window_exists=minimum_radius <= maximum_radius,
        scale_gap_ratio=minimum_radius / maximum_radius,
        conserved_stress_tensor_derived=False,
        stable_wormhole_established=False,
    )
