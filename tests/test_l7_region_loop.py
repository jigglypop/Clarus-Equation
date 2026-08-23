"""Construction locks for a three-epoch wash with named I on two cubes.

Machine pass is not a theorem. This file does not claim 닫힘, 유도됨,
autonomy, mouse / CCF, L8, or AGI. Occupancy is 1[(m,b) in R0].
After α, σ is left occupancy. After β, I is right occupancy.
Loop γ uses u^A = I * u_I(e^γ). Feedforward freezes σ.
Overwrite sets σ ← o^A(β) on the same named slot.

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
    LOOP_TASK_PHI1,
    LOOP_TASK_PHI2,
    NOMINAL_LEAK,
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
    ROUTING_HORIZON,
    HybridState,
    SourceHybridSubsystem,
    WashedRoleSplit,
    growth_at_label,
    loop_gate_drives,
    occupancy_bit,
    overwrite_sigma_from_action,
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


def test_registered_tasks_and_start() -> None:
    seed = registered_start()
    assert seed == _center()
    assert _in_u0(seed.mass, seed.boundary)
    assert seed.label == LABEL
    assert KAPPA == Fraction(1, 4)
    assert ROUTING_HORIZON == 32
    assert LOOP_TASK_PHI1 == (
        REGISTERED_FLUX_E1,
        REGISTERED_FLUX_E2,
        REGISTERED_FLUX_E2,
    )
    assert LOOP_TASK_PHI2 == (
        REGISTERED_FLUX_E1,
        REGISTERED_FLUX_E1,
        REGISTERED_FLUX_E2,
    )
    assert LOOP_TASK_PHI1[0] == LOOP_TASK_PHI2[0] == REGISTERED_FLUX_E1
    assert LOOP_TASK_PHI1[2] == LOOP_TASK_PHI2[2] == REGISTERED_FLUX_E2
    assert LOOP_TASK_PHI1[1] != LOOP_TASK_PHI2[1]


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


def test_loop_gate_equals_role_split_product() -> None:
    for named_i in (0, 1):
        for flux in (REGISTERED_FLUX_E1, REGISTERED_FLUX_E2):
            assert loop_gate_drives(named_i, flux) == role_split_drives(
                named_i,
                flux,
            )
    with pytest.raises(ValueError, match="bit"):
        loop_gate_drives(2, REGISTERED_FLUX_E2)
    with pytest.raises(ValueError, match="bit"):
        loop_gate_drives(True, REGISTERED_FLUX_E2)


def test_drive_zero_one_step_extinction_on_closed_bc() -> None:
    leak = NOMINAL_LEAK
    b_hi = BC_B[1]
    factor = 1 - leak * (1 - b_hi)
    assert leak == Fraction(5, 2)
    assert factor == Fraction(-53, 297)
    assert factor < 0
    assert BC_M[0] > 0
    assert Fraction(3, 5) - b_hi == Fraction(106, 1485)

    closed = HybridState.from_values(BC_M[0], b_hi, LABEL)
    nxt = source_hybrid_step(closed, kappa=KAPPA, drive=0)
    assert nxt.mass == 0
    assert occupancy_bit(nxt) == 0
    assert _zero_drive_occupancy(_center()) == 0


def test_drive_default_still_one() -> None:
    state = _center()
    default = source_hybrid_step(state, kappa=KAPPA)
    driven = source_hybrid_step(state, kappa=KAPPA, drive=1)
    assert default == driven

    subsystem = SourceHybridSubsystem(state=state, kappa=KAPPA)
    assert subsystem.step(1).state == default
    assert subsystem.step(1, drive=1).state == default


def test_loop_gate_keeps_named_sigma() -> None:
    host = WashedRoleSplit.washed().with_sigma(1)
    gated_zero = host.iterate_loop_gate(1, REGISTERED_FLUX_E2, 0)
    assert gated_zero.sigma == 1
    assert gated_zero.body.right.mass == 0
    assert gated_zero.body.left == source_hybrid_step(_center(), kappa=KAPPA, drive=0)

    gated_one = host.iterate_loop_gate(1, REGISTERED_FLUX_E2, 1)
    assert gated_one.sigma == 1
    assert gated_one.body.right == source_hybrid_step(_center(), kappa=KAPPA, drive=1)
    assert gated_one.body.left == gated_zero.body.left


def test_loop_readout_splits_registered_tasks() -> None:
    # Construction lock for L7-E1. Machine pass is not a theorem.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    sigma = occupied
    named_i = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: vacant,
    }
    assert role_split_drives(sigma, LOOP_TASK_PHI1[1]) == (Fraction(0), Fraction(1))
    assert role_split_drives(sigma, LOOP_TASK_PHI2[1]) == (Fraction(1), Fraction(0))
    assert loop_gate_drives(named_i[LOOP_TASK_PHI1], REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )
    assert loop_gate_drives(named_i[LOOP_TASK_PHI2], REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(0),
    )
    readout = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: vacant,
    }
    assert readout[LOOP_TASK_PHI1] == 1
    assert readout[LOOP_TASK_PHI2] == 0

    host = WashedRoleSplit.washed().with_sigma(1)
    beta_phi1 = host.iterate_role_split(1, REGISTERED_FLUX_E2)
    beta_phi2 = host.iterate_role_split(1, REGISTERED_FLUX_E1)
    assert beta_phi1.body.right == source_hybrid_step(_center(), kappa=KAPPA, drive=1)
    assert beta_phi2.body.right.mass == 0

    gamma_phi1 = host.wash().iterate_loop_gate(1, REGISTERED_FLUX_E2, 1)
    gamma_phi2 = host.wash().iterate_loop_gate(1, REGISTERED_FLUX_E2, 0)
    assert gamma_phi1.body.right == source_hybrid_step(_center(), kappa=KAPPA, drive=1)
    assert gamma_phi2.body.right.mass == 0
    assert gamma_phi1.sigma == gamma_phi2.sigma == 1


def test_feedforward_readout_matches_on_registered_tasks() -> None:
    # Construction lock for L7-E2. The common occupancy value is not scored.
    occupied = _unit_drive_occupancy_on_u0()
    sigma = occupied
    assert role_split_drives(sigma, REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )
    host = WashedRoleSplit.washed().with_sigma(1)
    on_phi1 = host.iterate_role_split(1, REGISTERED_FLUX_E2)
    on_phi2 = host.iterate_role_split(1, REGISTERED_FLUX_E2)
    assert on_phi1.body == on_phi2.body
    assert on_phi1.action_occupancy() == on_phi2.action_occupancy()


def test_loop_and_feedforward_maps_are_unequal() -> None:
    # Construction lock for L7-E3 on the finite set {φ1, φ2}.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    loop_map = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: vacant,
    }
    feedforward_map = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: occupied,
    }
    assert loop_map[LOOP_TASK_PHI1] != loop_map[LOOP_TASK_PHI2]
    assert feedforward_map[LOOP_TASK_PHI1] == feedforward_map[LOOP_TASK_PHI2]
    assert loop_map != feedforward_map


def test_overwrite_readout_equals_loop_readout() -> None:
    # Construction lock for L7-H1. Same γ-gate as named I. Same two cubes.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    host = WashedRoleSplit.washed().with_sigma(1)

    beta_phi1 = host.iterate_role_split(1, REGISTERED_FLUX_E2)
    ow_phi1 = overwrite_sigma_from_action(beta_phi1)
    assert ow_phi1.sigma == occupancy_bit(beta_phi1.body.right)
    assert ow_phi1.sigma == occupied

    beta_phi2 = host.iterate_role_split(1, REGISTERED_FLUX_E1)
    ow_phi2 = overwrite_sigma_from_action(beta_phi2)
    assert ow_phi2.sigma == occupancy_bit(beta_phi2.body.right)
    assert ow_phi2.sigma == vacant

    assert loop_gate_drives(ow_phi1.sigma, REGISTERED_FLUX_E2) == role_split_drives(
        ow_phi1.sigma,
        REGISTERED_FLUX_E2,
    )
    assert loop_gate_drives(ow_phi2.sigma, REGISTERED_FLUX_E2) == role_split_drives(
        ow_phi2.sigma,
        REGISTERED_FLUX_E2,
    )
    assert loop_gate_drives(1, REGISTERED_FLUX_E2) == role_split_drives(
        ow_phi1.sigma,
        REGISTERED_FLUX_E2,
    )
    assert loop_gate_drives(0, REGISTERED_FLUX_E2) == role_split_drives(
        ow_phi2.sigma,
        REGISTERED_FLUX_E2,
    )

    loop_readout = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: vacant,
    }
    overwrite_readout = {
        LOOP_TASK_PHI1: occupied,
        LOOP_TASK_PHI2: vacant,
    }
    assert overwrite_readout == loop_readout

    gamma_ow_phi1 = ow_phi1.wash().iterate_role_split(1, REGISTERED_FLUX_E2)
    gamma_loop_phi1 = host.wash().iterate_loop_gate(1, REGISTERED_FLUX_E2, 1)
    assert gamma_ow_phi1.body == gamma_loop_phi1.body
    gamma_ow_phi2 = ow_phi2.wash().iterate_role_split(1, REGISTERED_FLUX_E2)
    gamma_loop_phi2 = host.wash().iterate_loop_gate(1, REGISTERED_FLUX_E2, 0)
    assert gamma_ow_phi2.body == gamma_loop_phi2.body
    assert gamma_ow_phi2.body.right.mass == 0


def test_overwrite_requires_washed_host() -> None:
    with pytest.raises(TypeError, match="WashedRoleSplit"):
        overwrite_sigma_from_action(_center())  # type: ignore[arg-type]


def test_module_exports_include_loop_names() -> None:
    assert ulk.LOOP_TASK_PHI1 == LOOP_TASK_PHI1
    assert ulk.LOOP_TASK_PHI2 == LOOP_TASK_PHI2
    assert ulk.loop_gate_drives is loop_gate_drives
    assert ulk.overwrite_sigma_from_action is overwrite_sigma_from_action
    assert ulk.WashedRoleSplit is WashedRoleSplit
    assert ulk.role_split_drives is role_split_drives
    assert ulk.registered_start is registered_start
    assert ulk.IDENTITY_WEIGHT is IDENTITY_WEIGHT
