"""Construction locks for one-step activity readout on a registered pair.

Machine pass is not a theorem. This file does not claim 닫힘, 유도됨,
autonomy, zebrafish, L7, or AGI. Readout is one-step (m', b').
Label q' is discarded. U0 membership cites predecessor geometry.
Exact F^{32} Fraction trajectories are not used.
"""

from __future__ import annotations

from fractions import Fraction

import reality_stone.clarus as clarus
from reality_stone.clarus.universe_life_kernel import (
    ACTIVITY_DELTA_BOUNDARY,
    ACTIVITY_DELTA_MASS,
    ACTIVITY_NEXT_CIRC,
    ACTIVITY_NEXT_STAR,
    ACTIVITY_PAIR_CIRC_MASS,
    ACTIVITY_PAIR_SIGMA,
    HybridState,
    SourceHybridSubsystem,
    activity_readout,
    registered_activity_pair,
    registered_start,
    source_hybrid_step,
)
from test_l3_ne2_open_set import BC_B, BC_M


KAPPA = Fraction(1, 4)
LABEL = Fraction(3, 4)


def _in_u0(mass: Fraction, boundary: Fraction) -> bool:
    return BC_M[0] < mass < BC_M[1] and BC_B[0] < boundary < BC_B[1]


def test_registered_pair_is_l61() -> None:
    star, circ = registered_activity_pair()
    assert star == registered_start()
    assert star == HybridState.from_values(Fraction(1, 2), Fraction(49, 99), LABEL)
    assert circ == HybridState.from_values(
        ACTIVITY_PAIR_CIRC_MASS,
        Fraction(49, 99),
        LABEL,
    )
    assert ACTIVITY_PAIR_CIRC_MASS == Fraction(7, 15)
    assert ACTIVITY_PAIR_SIGMA == 1
    assert KAPPA == Fraction(1, 4)
    assert _in_u0(star.mass, star.boundary)
    assert _in_u0(circ.mass, circ.boundary)
    assert star.label == LABEL
    assert circ.label == LABEL
    assert Fraction(13, 30) < Fraction(7, 15) < Fraction(1, 2) < Fraction(17, 30)
    assert Fraction(137, 297) < Fraction(49, 99) < Fraction(157, 297)


def test_drive_default_still_one() -> None:
    star, circ = registered_activity_pair()
    for state in (star, circ):
        default = source_hybrid_step(state, kappa=KAPPA)
        driven = source_hybrid_step(state, kappa=KAPPA, drive=1)
        assert default == driven
        assert activity_readout(state) == activity_readout(state, drive=1)
    subsystem = SourceHybridSubsystem(state=star, kappa=KAPPA)
    assert subsystem.step(1).state == source_hybrid_step(star, kappa=KAPPA)
    assert subsystem.step(1, drive=1).state == source_hybrid_step(star, kappa=KAPPA)


def test_one_step_readout_differs_on_the_pair() -> None:
    # Construction lock for L6-E1. Machine pass is not a theorem.
    star, circ = registered_activity_pair()
    next_star = activity_readout(star)
    next_circ = activity_readout(circ)
    assert next_star == ACTIVITY_NEXT_STAR
    assert next_circ == ACTIVITY_NEXT_CIRC
    assert ACTIVITY_NEXT_STAR == (Fraction(7187, 12672), Fraction(491, 990))
    assert ACTIVITY_NEXT_CIRC == (Fraction(16891, 29700), Fraction(133, 270))
    assert next_star != next_circ
    assert next_star[0] - next_circ[0] == ACTIVITY_DELTA_MASS
    assert next_star[1] - next_circ[1] == ACTIVITY_DELTA_BOUNDARY
    assert ACTIVITY_DELTA_MASS == Fraction(-1487, 950400)
    assert ACTIVITY_DELTA_BOUNDARY == Fraction(1, 297)
    assert ACTIVITY_DELTA_MASS != 0
    assert ACTIVITY_DELTA_BOUNDARY != 0
    stepped_star = source_hybrid_step(star, kappa=KAPPA, drive=1)
    stepped_circ = source_hybrid_step(circ, kappa=KAPPA, drive=1)
    assert (stepped_star.mass, stepped_star.boundary) == next_star
    assert (stepped_circ.mass, stepped_circ.boundary) == next_circ


def test_bit_predictor_cannot_match_both_next_pairs() -> None:
    # Construction lock for L6-E2. A bit predictor is a function of σ only.
    star, circ = registered_activity_pair()
    next_star = activity_readout(star)
    next_circ = activity_readout(circ)
    sigma_of = {star: ACTIVITY_PAIR_SIGMA, circ: ACTIVITY_PAIR_SIGMA}
    assert sigma_of[star] == sigma_of[circ] == 1

    def bit_predictor(sigma: int) -> tuple[Fraction, Fraction]:
        return ACTIVITY_NEXT_STAR if sigma == 1 else ACTIVITY_NEXT_CIRC

    predicted_star = bit_predictor(sigma_of[star])
    predicted_circ = bit_predictor(sigma_of[circ])
    assert predicted_star == predicted_circ
    assert not (predicted_star == next_star and predicted_circ == next_circ)


def test_activity_map_and_bit_map_are_unequal() -> None:
    # Construction lock for L6-E3 on the finite pair.
    star, circ = registered_activity_pair()
    activity_map = {
        star: activity_readout(star),
        circ: activity_readout(circ),
    }
    bit_map = {
        star: ACTIVITY_NEXT_STAR,
        circ: ACTIVITY_NEXT_STAR,
    }
    assert activity_map[star] != activity_map[circ]
    assert bit_map[star] == bit_map[circ]
    assert activity_map != bit_map


def test_h1_citation_both_points_in_u0() -> None:
    # Citation: both points lie in U0 so O-E1 applies.
    # Not a new enclosure. No T=32 Fraction trajectory.
    star, circ = registered_activity_pair()
    assert _in_u0(star.mass, star.boundary)
    assert _in_u0(circ.mass, circ.boundary)
    assert star.label == LABEL
    assert circ.label == LABEL


def test_public_exports_include_activity_pair_names() -> None:
    assert clarus.registered_activity_pair is registered_activity_pair
    assert clarus.activity_readout is activity_readout
    assert clarus.ACTIVITY_NEXT_STAR == ACTIVITY_NEXT_STAR
    assert clarus.ACTIVITY_NEXT_CIRC == ACTIVITY_NEXT_CIRC
    assert clarus.ACTIVITY_DELTA_MASS == ACTIVITY_DELTA_MASS
    assert clarus.ACTIVITY_DELTA_BOUNDARY == ACTIVITY_DELTA_BOUNDARY
    assert clarus.ACTIVITY_PAIR_CIRC_MASS == ACTIVITY_PAIR_CIRC_MASS
    assert clarus.ACTIVITY_PAIR_SIGMA == ACTIVITY_PAIR_SIGMA
