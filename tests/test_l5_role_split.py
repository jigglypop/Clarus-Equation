"""Construction locks for wash + named σ on the two-channel host.

Machine pass is not a theorem. This file does not claim 닫힘, 유도됨,
autonomy, Drosophila/C. elegans, L6, or AGI. Occupancy is 1[(m,b) in R0].
σ is left occupancy after epoch α. Readout is right occupancy after β.

Exact F^{32} on a point is not used: the q-map is cubic, so a 32-step
Fraction trajectory is not a feasible lock. u=0 occupancy is the one-step
extinction on closed Bc. u=1 occupancy on U0 cites the predecessor hull
in test_l3_ne2_open_set.py.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from reality_stone.clarus import universe_life_kernel as ulk
from reality_stone.clarus.universe_life_kernel import (
    IDENTITY_WEIGHT,
    NOMINAL_LEAK,
    R0_BOUNDARY,
    R0_MASS,
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
    ROUTING_HORIZON,
    WASH_TASK_TAU1,
    WASH_TASK_TAU2,
    HybridState,
    SourceHybridSubsystem,
    WashedRoleSplit,
    growth_at_label,
    occupancy_bit,
    registered_start,
    role_split_drives,
    routed_drives,
    source_hybrid_step,
)
from test_l3_ne2_open_set import (
    BC_B,
    BC_M,
    closed_box,
    inside_r0,
    trace_full,
)


KAPPA = Fraction(1, 4)
LABEL = Fraction(3, 4)
CENTER_MASS = Fraction(1, 2)
CENTER_BOUNDARY = Fraction(49, 99)


def _center() -> HybridState:
    return HybridState.from_values(CENTER_MASS, CENTER_BOUNDARY, LABEL)


def _in_u0(mass: Fraction, boundary: Fraction) -> bool:
    return BC_M[0] < mass < BC_M[1] and BC_B[0] < boundary < BC_B[1]


def _zero_drive_occupancy(state: HybridState) -> int:
    dead = source_hybrid_step(state, kappa=KAPPA, drive=0)
    assert dead.mass == 0
    assert occupancy_bit(dead) == 0
    assert source_hybrid_step(dead, kappa=KAPPA, drive=0).mass == 0
    return 0


def _unit_drive_occupancy_on_u0() -> int:
    # Predecessor U0 hull at q=3/4, drive=1. Citation, not a new enclosure.
    mixed, _divs, hull = trace_full(
        closed_box(BC_M, BC_B),
        growth_at_label(LABEL, KAPPA),
    )
    assert mixed is False
    assert inside_r0(hull)
    return 1


def test_registered_start_is_l4_center() -> None:
    seed = registered_start()
    assert seed == _center()
    assert _in_u0(seed.mass, seed.boundary)
    assert seed.label == LABEL
    assert KAPPA == Fraction(1, 4)
    assert ROUTING_HORIZON == 32
    assert WASH_TASK_TAU1 == (REGISTERED_FLUX_E1, REGISTERED_FLUX_E2)
    assert WASH_TASK_TAU2 == (REGISTERED_FLUX_E2, REGISTERED_FLUX_E2)
    assert R0_MASS[0] <= seed.mass <= R0_MASS[1]
    assert R0_BOUNDARY[0] <= seed.boundary <= R0_BOUNDARY[1]


def test_drive_default_still_one() -> None:
    state = _center()
    default = source_hybrid_step(state, kappa=KAPPA)
    driven = source_hybrid_step(state, kappa=KAPPA, drive=1)
    assert default == driven

    subsystem = SourceHybridSubsystem(state=state, kappa=KAPPA)
    assert subsystem.step(1).state == default
    assert subsystem.step(1, drive=1).state == default


def test_drive_zero_one_step_extinction_on_closed_bc() -> None:
    leak = NOMINAL_LEAK
    b_hi = BC_B[1]
    factor = 1 - leak * (1 - b_hi)
    assert leak == Fraction(5, 2)
    assert factor == Fraction(-53, 297)
    assert factor < 0
    assert BC_M[0] > 0

    closed = HybridState.from_values(BC_M[0], b_hi, LABEL)
    nxt = source_hybrid_step(closed, kappa=KAPPA, drive=0)
    assert nxt.mass == 0
    assert occupancy_bit(nxt) == 0
    assert _zero_drive_occupancy(_center()) == 0


def test_body_sensor_is_left_action_is_right() -> None:
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E1) == (
        Fraction(1),
        Fraction(0),
    )
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )
    sensor_e2, action_e2 = role_split_drives(1, REGISTERED_FLUX_E2)
    assert sensor_e2 == Fraction(0)
    assert action_e2 == Fraction(1)


def test_sigma_one_recovers_ordinary_l4_drives() -> None:
    for flux in (REGISTERED_FLUX_E1, REGISTERED_FLUX_E2):
        ordinary = routed_drives(IDENTITY_WEIGHT, flux)
        assert role_split_drives(1, flux) == ordinary
        assert role_split_drives(0, flux) == (ordinary[0], Fraction(0))


def test_sigma_must_be_a_bit() -> None:
    with pytest.raises(ValueError, match="bit"):
        role_split_drives(2, REGISTERED_FLUX_E2)
    with pytest.raises(ValueError, match="bit"):
        role_split_drives(True, REGISTERED_FLUX_E2)
    with pytest.raises(ValueError, match="written bit"):
        WashedRoleSplit.washed().iterate_role_split(1, REGISTERED_FLUX_E2)


def test_wash_resets_both_copies_and_keeps_named_sigma() -> None:
    host = WashedRoleSplit.washed()
    assert host.body.left == host.start
    assert host.body.right == host.start
    assert host.sigma is None

    stepped = host.with_sigma(1).iterate_routed(1, REGISTERED_FLUX_E1)
    assert stepped.sigma == 1
    assert stepped.body.right.mass == 0
    assert stepped.body.left != host.start

    washed = stepped.wash()
    assert washed.body.left == host.start
    assert washed.body.right == host.start
    assert washed.sigma == 1
    assert washed.start == host.start


def test_role_split_wash_readout_splits_registered_tasks() -> None:
    # Construction lock for L5-E1. Machine pass is not a theorem.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    sigma = {
        WASH_TASK_TAU1: occupied,
        WASH_TASK_TAU2: vacant,
    }
    assert role_split_drives(sigma[WASH_TASK_TAU1], REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )
    assert role_split_drives(sigma[WASH_TASK_TAU2], REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(0),
    )
    readout = {
        WASH_TASK_TAU1: occupied,
        WASH_TASK_TAU2: vacant,
    }
    assert readout[WASH_TASK_TAU1] == 1
    assert readout[WASH_TASK_TAU2] == 0

    gated_one = WashedRoleSplit.washed().with_sigma(1).iterate_role_split(
        1,
        REGISTERED_FLUX_E2,
    )
    assert gated_one.body.right == source_hybrid_step(_center(), kappa=KAPPA, drive=1)
    assert gated_one.body.left == source_hybrid_step(_center(), kappa=KAPPA, drive=0)

    gated_zero = WashedRoleSplit.washed().with_sigma(0).iterate_role_split(
        1,
        REGISTERED_FLUX_E2,
    )
    assert gated_zero.body.right.mass == 0
    assert gated_zero.body.left == gated_one.body.left


def test_no_store_wash_readout_matches_on_registered_tasks() -> None:
    # Construction lock for L5-E2. The common occupancy value is not scored.
    host = WashedRoleSplit.washed()
    on_tau1 = host.iterate_routed(3, REGISTERED_FLUX_E2)
    on_tau2 = host.iterate_routed(3, REGISTERED_FLUX_E2)
    assert on_tau1.body == on_tau2.body
    assert on_tau1.action_occupancy() == on_tau2.action_occupancy()

    ignored = host.with_sigma(0).iterate_routed(1, REGISTERED_FLUX_E2)
    ordinary = host.iterate_routed(1, REGISTERED_FLUX_E2)
    assert ignored.body == ordinary.body
    assert ignored.action_occupancy() == ordinary.action_occupancy()


def test_role_split_and_no_store_maps_are_unequal() -> None:
    # Construction lock for L5-E3 on the finite set {τ1, τ2}.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    role_split = {
        WASH_TASK_TAU1: occupied,
        WASH_TASK_TAU2: vacant,
    }
    common = WashedRoleSplit.washed().iterate_routed(
        1,
        REGISTERED_FLUX_E2,
    ).action_occupancy()
    no_store = {
        WASH_TASK_TAU1: common,
        WASH_TASK_TAU2: common,
    }
    assert role_split[WASH_TASK_TAU1] != role_split[WASH_TASK_TAU2]
    assert no_store[WASH_TASK_TAU1] == no_store[WASH_TASK_TAU2]
    assert role_split != no_store


def test_unfinished_no_wash_second_window_is_not_a_lock() -> None:
    # Unfinished. Not a passing construction for L5-H1.
    # τ1: action is already m=0 after α (u=0). m=0 absorbs any later drive.
    vacant = _zero_drive_occupancy(_center())
    dead = source_hybrid_step(_center(), kappa=KAPPA, drive=0)
    assert dead.mass == 0
    assert source_hybrid_step(dead, kappa=KAPPA, drive=1).mass == 0
    assert vacant == 0
    # τ2: u=1 occupancy cites O-E1 inside R0. That image is not a
    # registered U0 start, so the second window is not scored here.
    occupied = _unit_drive_occupancy_on_u0()
    assert occupied == 1


def test_module_exports_include_role_split_names() -> None:
    assert ulk.WashedRoleSplit is WashedRoleSplit
    assert ulk.role_split_drives is role_split_drives
    assert ulk.registered_start is registered_start
    assert ulk.WASH_TASK_TAU1 == WASH_TASK_TAU1
    assert ulk.WASH_TASK_TAU2 == WASH_TASK_TAU2
    assert ulk.RoutedTwoCopy is not None
    assert ulk.IDENTITY_WEIGHT is IDENTITY_WEIGHT
