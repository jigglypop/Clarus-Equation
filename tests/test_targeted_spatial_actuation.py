from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.spatial_folding import casimir_cell_conversion_audit
from reality_stone.clarus.targeted_spatial_actuation import (
    causal_target_delivery_audit,
    target_localization_audit,
    throat_scale_window_audit,
)


def test_broadcast_response_cannot_encode_a_target_location() -> None:
    audit = target_localization_audit(np.ones((3, 3)), required_density_j_m3=0.5)

    assert audit.response_rank == 1
    assert not audit.all_commands_localized
    assert audit.all_commands_meet_required_density
    assert not audit.actuator_map_derived_from_ce


def test_diagonal_response_localizes_only_by_assuming_an_actuator_map() -> None:
    audit = target_localization_audit(np.eye(3) * 2.0, required_density_j_m3=1.0)

    assert audit.response_rank == 3
    assert audit.selected_locations == (0, 1, 2)
    assert audit.all_commands_localized
    assert audit.all_commands_meet_required_density
    assert not audit.actuator_map_derived_from_ce


def test_adaptive_remote_command_obeys_light_cone_delay() -> None:
    light_year_m = 9.4607304725808e15
    audit = causal_target_delivery_audit(
        distance_m=light_year_m,
        candidate_count=8,
        requested_activation_s=0.0,
    )

    assert math.isclose(audit.target_information_bits, 3.0)
    assert 31_557_599.9 < audit.earliest_delivery_s < 31_557_600.1
    assert not audit.deadline_satisfied
    assert not audit.instantaneous_adaptive_activation
    assert audit.preinstalled_receiver_required


def test_ce_cell_density_and_coherence_radius_bounds_do_not_overlap() -> None:
    density = casimir_cell_conversion_audit().energy_density_j_m3
    audit = throat_scale_window_audit(
        candidate_negative_density_j_m3=density,
        ce_correlation_length_m=6.65e-15,
    )

    assert 1.68e8 < audit.minimum_radius_from_density_m < 1.70e8
    assert audit.maximum_radius_from_coherence_m == 6.65e-15
    assert audit.scale_gap_ratio > 2.5e22
    assert not audit.feasible_radius_window_exists
    assert not audit.conserved_stress_tensor_derived
    assert not audit.stable_wormhole_established


@pytest.mark.parametrize(
    ("density", "correlation"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_scale_window_rejects_nonpositive_inputs(
    density: float,
    correlation: float,
) -> None:
    with pytest.raises(ValueError):
        throat_scale_window_audit(
            candidate_negative_density_j_m3=density,
            ce_correlation_length_m=correlation,
        )
