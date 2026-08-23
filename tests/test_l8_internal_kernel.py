"""Construction locks for a one-step host kernel on a registered pair.

Machine pass is not a theorem. This file does not claim 닫힘, 유도됨,
autonomy, BrainRuntime, a third cube, or AGI. The registered kernel
is the one-step host map typed H → H. Occupancy is 1[(m, b) in R0].

Exact F^{32} Fraction trajectories are not used. Sensor u=0 is the
one-step extinction. Action (m', b') cites the L6 one-step locks.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from reality_stone.clarus import universe_life_kernel as ulk
from reality_stone.clarus.universe_life_kernel import (
    ACTIVITY_DELTA_BOUNDARY,
    ACTIVITY_DELTA_MASS,
    ACTIVITY_NEXT_CIRC,
    ACTIVITY_NEXT_STAR,
    ACTIVITY_PAIR_SIGMA,
    IDENTITY_WEIGHT,
    NOMINAL_LEAK,
    REGISTERED_FLUX_E2,
    HybridState,
    HostTuple,
    activity_readout,
    internal_kernel,
    loop_gate_drives,
    occupancy_bit,
    registered_activity_pair,
    registered_host_pair,
    registered_start,
    role_split_drives,
    source_hybrid_step,
)
from test_l3_ne2_open_set import BC_B, BC_M


KAPPA = Fraction(1, 4)
LABEL = Fraction(3, 4)


def _in_u0(mass: Fraction, boundary: Fraction) -> bool:
    return BC_M[0] < mass < BC_M[1] and BC_B[0] < boundary < BC_B[1]


def _phi_on_registered_drives(host: HostTuple) -> HostTuple:
    """Independent one-step assembly on S: sensor u=0, action u=1."""

    return HostTuple(
        tick=host.tick + 1,
        flux=host.flux,
        sensor=source_hybrid_step(host.sensor, kappa=KAPPA, drive=0),
        action=source_hybrid_step(host.action, kappa=KAPPA, drive=1),
        sigma=host.sigma,
        named_i=host.named_i,
    )


def test_registered_hosts_are_l81() -> None:
    star_state, circ_state = registered_activity_pair()
    h_star, h_circ = registered_host_pair()
    assert star_state == registered_start()
    assert h_star.tick == h_circ.tick == 0
    assert h_star.flux == h_circ.flux == REGISTERED_FLUX_E2
    assert REGISTERED_FLUX_E2 == (Fraction(0), Fraction(1))
    assert h_star.sensor == h_star.action == star_state
    assert h_circ.sensor == h_circ.action == circ_state
    assert h_star.sigma == h_circ.sigma == ACTIVITY_PAIR_SIGMA == 1
    assert h_star.named_i == h_circ.named_i == 1
    assert _in_u0(h_star.action.mass, h_star.action.boundary)
    assert _in_u0(h_circ.action.mass, h_circ.action.boundary)
    assert h_star.sensor.label == h_circ.sensor.label == LABEL
    assert KAPPA == Fraction(1, 4)


def test_drives_on_s_are_action_one_sensor_zero() -> None:
    h_star, h_circ = registered_host_pair()
    for host in (h_star, h_circ):
        gated = loop_gate_drives(host.named_i, host.flux)
        split = role_split_drives(host.sigma, host.flux)
        assert gated == split == (Fraction(0), Fraction(1))
        assert gated == loop_gate_drives(1, REGISTERED_FLUX_E2)
    assert IDENTITY_WEIGHT == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))


def test_host_tuple_rejects_bad_slots() -> None:
    star, _circ = registered_activity_pair()
    with pytest.raises(TypeError, match="sensor"):
        HostTuple(
            tick=0,
            flux=REGISTERED_FLUX_E2,
            sensor=0,  # type: ignore[arg-type]
            action=star,
            sigma=1,
            named_i=1,
        )
    with pytest.raises(ValueError, match="bit"):
        HostTuple(
            tick=0,
            flux=REGISTERED_FLUX_E2,
            sensor=star,
            action=star,
            sigma=2,
            named_i=1,
        )
    with pytest.raises(TypeError, match="HostTuple"):
        internal_kernel(star)  # type: ignore[arg-type]


def test_k_equals_phi_on_s() -> None:
    # Construction lock for L8-E1. Machine pass is not a theorem.
    h_star, h_circ = registered_host_pair()
    for host in (h_star, h_circ):
        image = internal_kernel(host)
        assembled = _phi_on_registered_drives(host)
        assert image == assembled
        assert isinstance(image, HostTuple)
        assert image.tick == 1
        assert image.flux == host.flux == REGISTERED_FLUX_E2
        assert image.sigma == host.sigma == 1
        assert image.named_i == host.named_i == 1
        assert image.sensor.mass == 0
        assert image.sensor.label == LABEL
        assert image.action.label == LABEL

    k_star = internal_kernel(h_star)
    k_circ = internal_kernel(h_circ)
    assert (k_star.action.mass, k_star.action.boundary) == ACTIVITY_NEXT_STAR
    assert (k_circ.action.mass, k_circ.action.boundary) == ACTIVITY_NEXT_CIRC
    assert ACTIVITY_NEXT_STAR == (Fraction(7187, 12672), Fraction(491, 990))
    assert ACTIVITY_NEXT_CIRC == (Fraction(16891, 29700), Fraction(133, 270))
    assert k_star.sensor == HybridState.from_values(0, Fraction(491, 990), LABEL)
    assert k_circ.sensor == HybridState.from_values(0, Fraction(133, 270), LABEL)
    assert k_star.sensor.mass == 0
    assert k_circ.sensor.mass == 0


def test_sensor_one_step_extinction_and_action_l6_lock() -> None:
    leak = NOMINAL_LEAK
    factor = 1 - leak * (1 - Fraction(49, 99))
    assert leak == Fraction(5, 2)
    assert factor == Fraction(-26, 99)
    assert factor < 0

    h_star, h_circ = registered_host_pair()
    for host, locked in (
        (h_star, ACTIVITY_NEXT_STAR),
        (h_circ, ACTIVITY_NEXT_CIRC),
    ):
        dead = source_hybrid_step(host.sensor, kappa=KAPPA, drive=0)
        assert dead.mass == 0
        assert occupancy_bit(dead) == 0
        assert activity_readout(host.action) == locked
        stepped = internal_kernel(host)
        assert stepped.sensor == dead
        assert (stepped.action.mass, stepped.action.boundary) == locked


def test_oa_same_action_next_differs() -> None:
    # Construction lock for L8-E2. Machine pass is not a theorem.
    h_star, h_circ = registered_host_pair()
    o_star = occupancy_bit(h_star.action)
    o_circ = occupancy_bit(h_circ.action)
    assert o_star == o_circ == 1

    k_star = internal_kernel(h_star)
    k_circ = internal_kernel(h_circ)
    next_star = (k_star.action.mass, k_star.action.boundary)
    next_circ = (k_circ.action.mass, k_circ.action.boundary)
    assert next_star == ACTIVITY_NEXT_STAR
    assert next_circ == ACTIVITY_NEXT_CIRC
    assert next_star != next_circ
    assert next_star[0] - next_circ[0] == ACTIVITY_DELTA_MASS
    assert next_star[1] - next_circ[1] == ACTIVITY_DELTA_BOUNDARY
    assert ACTIVITY_DELTA_MASS == Fraction(-1487, 950400)
    assert ACTIVITY_DELTA_BOUNDARY == Fraction(1, 297)
    assert ACTIVITY_DELTA_MASS != 0
    assert ACTIVITY_DELTA_BOUNDARY != 0


def test_k_and_oa_unequal_as_maps() -> None:
    # Construction lock for L8-E3 on the finite pair.
    h_star, h_circ = registered_host_pair()
    kernel_map = {
        h_star: internal_kernel(h_star),
        h_circ: internal_kernel(h_circ),
    }
    occupancy_map = {
        h_star: occupancy_bit(h_star.action),
        h_circ: occupancy_bit(h_circ.action),
    }
    assert kernel_map[h_star] != kernel_map[h_circ]
    assert occupancy_map[h_star] == occupancy_map[h_circ] == 1
    assert kernel_map != occupancy_map
    assert type(kernel_map[h_star]) is HostTuple
    assert type(occupancy_map[h_star]) is int


def test_bit_valued_map_cannot_equal_phi() -> None:
    # Construction lock for L8-H1. Type / codomain check.
    # A map S → {0, 1} cannot equal Φ(H), whose slots are a HostTuple.
    h_star, h_circ = registered_host_pair()
    for host in (h_star, h_circ):
        image = internal_kernel(host)
        assert isinstance(image, HostTuple)
        assert type(image) is not int
        assert image not in (0, 1)
        for bit in (0, 1):
            assert bit != image
            assert type(bit) is not type(image)


def test_module_exports_include_host_kernel_names() -> None:
    assert ulk.HostTuple is HostTuple
    assert ulk.registered_host_pair is registered_host_pair
    assert ulk.internal_kernel is internal_kernel
    assert ulk.activity_readout is activity_readout
    assert ulk.loop_gate_drives is loop_gate_drives
    assert ulk.REGISTERED_FLUX_E2 == REGISTERED_FLUX_E2
