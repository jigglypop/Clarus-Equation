"""Construction locks for two-channel routed drive on the boxed host.

Machine pass is not a theorem. This file does not claim 닫힘, 유도됨,
autonomy, C. elegans identity, L5, or AGI. Occupancy is 1[(m,b) in R0]
at T=32. Label is not an observable.

Exact F^{32} on a point is not used: the q-map is cubic, so a 32-step
Fraction trajectory is not a feasible lock. u=0 occupancy is the one-step
extinction on closed Bc. u=1 occupancy on U0 cites the predecessor hull
in test_l3_ne2_open_set.py.
"""

from __future__ import annotations

from fractions import Fraction

from reality_stone.clarus import universe_life_kernel as ulk
from reality_stone.clarus.universe_life_kernel import (
    COMPLETE_BINARY_ROUTER,
    IDENTITY_WEIGHT,
    NOMINAL_LEAK,
    R0_BOUNDARY,
    R0_MASS,
    REGISTERED_FLUX_E1,
    REGISTERED_FLUX_E2,
    ROUTING_HORIZON,
    HybridState,
    RoutedTwoCopy,
    SourceHybridSubsystem,
    growth_at_label,
    occupancy_bit,
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
SWAP_WEIGHT = (
    (Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(0)),
)


def _center() -> HybridState:
    return HybridState.from_values(CENTER_MASS, CENTER_BOUNDARY, LABEL)


def _body() -> RoutedTwoCopy:
    seed = _center()
    return RoutedTwoCopy(left=seed, right=seed, kappa=KAPPA)


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


def test_registered_center_is_interior_of_u0() -> None:
    assert _in_u0(CENTER_MASS, CENTER_BOUNDARY)
    assert (BC_M[0] + BC_M[1]) / 2 == CENTER_MASS
    assert (BC_B[0] + BC_B[1]) / 2 == CENTER_BOUNDARY
    assert R0_MASS[0] <= CENTER_MASS <= R0_MASS[1]
    assert R0_BOUNDARY[0] <= CENTER_BOUNDARY <= R0_BOUNDARY[1]
    assert ROUTING_HORIZON == 32
    assert LABEL == Fraction(3, 4)
    assert KAPPA == Fraction(1, 4)


def test_drive_one_recovers_current_growth_bracket() -> None:
    state = _center()
    default = source_hybrid_step(state, kappa=KAPPA)
    driven = source_hybrid_step(state, kappa=KAPPA, drive=1)
    assert default == driven

    subsystem = SourceHybridSubsystem(state=state, kappa=KAPPA)
    assert subsystem.step(1).state == default
    assert subsystem.step(1, drive=1).state == default
    assert subsystem.step(0, drive=1).state == default


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


def test_q_maps_stay_uncoupled_from_drive() -> None:
    state = _center()
    full = source_hybrid_step(state, kappa=KAPPA, drive=1)
    none = source_hybrid_step(state, kappa=KAPPA, drive=0)
    assert full.label == none.label
    assert full.mass != none.mass

    body = _body().step(1, 0)
    assert body.left.label == body.right.label
    assert body.left.mass != body.right.mass


def test_identity_routes_registered_fluxes_to_standard_basis() -> None:
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E1) == (
        Fraction(1),
        Fraction(0),
    )
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )


def test_complete_binary_router_sends_both_fluxes_to_half() -> None:
    half = (Fraction(1, 2), Fraction(1, 2))
    assert routed_drives(COMPLETE_BINARY_ROUTER, REGISTERED_FLUX_E1) == half
    assert routed_drives(COMPLETE_BINARY_ROUTER, REGISTERED_FLUX_E2) == half


def test_identity_occupancy_pair_splits_registered_fluxes() -> None:
    # Construction lock for L4-E1. Machine pass is not a theorem.
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E1) == (
        Fraction(1),
        Fraction(0),
    )
    assert routed_drives(IDENTITY_WEIGHT, REGISTERED_FLUX_E2) == (
        Fraction(0),
        Fraction(1),
    )
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    assert (occupied, vacant) == (1, 0)
    assert (vacant, occupied) == (0, 1)

    stepped = _body().iterate_routed(1, IDENTITY_WEIGHT, REGISTERED_FLUX_E1)
    assert stepped.left == source_hybrid_step(_center(), kappa=KAPPA, drive=1)
    assert stepped.right.mass == 0


def test_complete_binary_occupancy_pairs_match_on_registered_fluxes() -> None:
    # Construction lock for L4-E2. The common occupancy value is not scored.
    drive_e1 = routed_drives(COMPLETE_BINARY_ROUTER, REGISTERED_FLUX_E1)
    drive_e2 = routed_drives(COMPLETE_BINARY_ROUTER, REGISTERED_FLUX_E2)
    assert drive_e1 == drive_e2 == (Fraction(1, 2), Fraction(1, 2))

    body = _body()
    on_e1 = body.iterate(3, *drive_e1)
    on_e2 = body.iterate(3, *drive_e2)
    assert on_e1 == on_e2
    assert on_e1.left == on_e1.right
    assert on_e1.occupancy_pair() == on_e2.occupancy_pair()


def test_identity_and_complete_binary_maps_are_unequal() -> None:
    # Construction lock for L4-E3 on the finite set {e1, e2}.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    identity = {
        REGISTERED_FLUX_E1: (occupied, vacant),
        REGISTERED_FLUX_E2: (vacant, occupied),
    }
    complete_pair = _body().iterate(
        1,
        *routed_drives(COMPLETE_BINARY_ROUTER, REGISTERED_FLUX_E1),
    ).occupancy_pair()
    complete = {
        REGISTERED_FLUX_E1: complete_pair,
        REGISTERED_FLUX_E2: complete_pair,
    }
    assert identity[REGISTERED_FLUX_E1] != identity[REGISTERED_FLUX_E2]
    assert complete[REGISTERED_FLUX_E1] == complete[REGISTERED_FLUX_E2]
    assert identity != complete


def test_killing_identity_and_swap_both_separate() -> None:
    # Killing test for the rejected universal on symmetric-support binaries.
    # Not a theorem. Documents that I and the swap matrix both separate.
    occupied = _unit_drive_occupancy_on_u0()
    vacant = _zero_drive_occupancy(_center())
    assert routed_drives(SWAP_WEIGHT, REGISTERED_FLUX_E1) == (
        Fraction(0),
        Fraction(1),
    )
    assert routed_drives(SWAP_WEIGHT, REGISTERED_FLUX_E2) == (
        Fraction(1),
        Fraction(0),
    )
    identity_e1 = (occupied, vacant)
    identity_e2 = (vacant, occupied)
    swap_e1 = (vacant, occupied)
    swap_e2 = (occupied, vacant)
    assert identity_e1 != identity_e2
    assert swap_e1 != swap_e2
    assert identity_e1 == (1, 0)
    assert identity_e2 == (0, 1)
    assert swap_e1 == (0, 1)
    assert swap_e2 == (1, 0)


def test_module_exports_include_routing_names() -> None:
    assert ulk.RoutedTwoCopy is RoutedTwoCopy
    assert ulk.routed_drives is routed_drives
    assert ulk.occupancy_bit is occupancy_bit
    assert ulk.IDENTITY_WEIGHT is IDENTITY_WEIGHT
    assert ulk.COMPLETE_BINARY_ROUTER is COMPLETE_BINARY_ROUTER
    assert ulk.REGISTERED_FLUX_E1 == REGISTERED_FLUX_E1
    assert ulk.REGISTERED_FLUX_E2 == REGISTERED_FLUX_E2
    assert ulk.ROUTING_HORIZON == ROUTING_HORIZON
