from __future__ import annotations

from fractions import Fraction
import inspect

import pytest

import reality_stone.clarus as clarus
from reality_stone.clarus.universe_life_kernel import (
    CUBE_CORNERS,
    DIVIDING_DISCRIMINANT_AT_HALF,
    DIVIDING_DISCRIMINANT_AT_ONE,
    EXTINCTION_AREA_FLOOR,
    KAPPA_OPEN_RIGHT,
    NOMINAL_GROWTH,
    NOMINAL_PARAMETERS,
    SOURCE_EXTINCTION_AREA,
    SOURCE_FIXED_POINTS,
    HybridState,
    SourceHybridSubsystem,
    UniverseKernel,
    UniverseState,
    admissible_kappa,
    dividing_mass_discriminant,
    growth_at_label,
    hosted_states,
    iterate_source,
    killing_low_label_growth,
    registered_grid,
    source_hybrid_step,
    source_one_step_extinction_area,
)

HOST_TICKS = 8
HOST_TOLERANCE = Fraction(1, 10**15)


def _same_state(left: HybridState, right: HybridState) -> None:
    assert left == right
    assert abs(left.mass - right.mass) <= HOST_TOLERANCE
    assert abs(left.boundary - right.boundary) <= HOST_TOLERANCE
    assert abs(left.label - right.label) <= HOST_TOLERANCE


@pytest.mark.parametrize("state", [*CUBE_CORNERS, *SOURCE_FIXED_POINTS])
def test_g_host_kernel_matches_local_f0_for_eight_ticks(state: HybridState) -> None:
    kernel = UniverseKernel()
    universe = kernel.iterate(kernel.host(state, flux=1), HOST_TICKS)
    local = iterate_source(state, HOST_TICKS)

    assert universe.tick == HOST_TICKS
    assert universe.flux == 1
    assert universe.residual == 0
    assert len(universe.subsystems) == 1
    _same_state(hosted_states(universe)[0], local)


def test_g_host_registered_grid_covers_corners_and_fixed_points() -> None:
    grid = registered_grid()
    assert len(CUBE_CORNERS) == 8
    assert len(SOURCE_FIXED_POINTS) == 6
    assert set(CUBE_CORNERS).issubset(set(grid))
    assert set(SOURCE_FIXED_POINTS).issubset(set(grid))


def test_source_fixed_points_are_fixed_under_local_f0() -> None:
    for state in SOURCE_FIXED_POINTS:
        _same_state(source_hybrid_step(state), state)


def test_g_couple_kappa_zero_label_does_not_change_mass_boundary() -> None:
    mass = Fraction(2, 5)
    boundary = Fraction(1, 3)
    low = source_hybrid_step(HybridState.from_values(mass, boundary, Fraction(1, 4)))
    high = source_hybrid_step(HybridState.from_values(mass, boundary, Fraction(3, 4)))
    mid = source_hybrid_step(HybridState.from_values(mass, boundary, Fraction(1, 2)))

    assert (low.mass, low.boundary) == (high.mass, high.boundary) == (mid.mass, mid.boundary)
    assert low.label != high.label


def test_g_couple_kappa_quarter_label_enters_predivision_mass() -> None:
    seed = (Fraction(1, 2), Fraction(1, 2))
    kappa = Fraction(1, 4)
    low = source_hybrid_step(
        HybridState.from_values(*seed, Fraction(1, 4)),
        kappa=kappa,
    )
    high = source_hybrid_step(
        HybridState.from_values(*seed, Fraction(3, 4)),
        kappa=kappa,
    )
    uncoupled = source_hybrid_step(HybridState.from_values(*seed, Fraction(1, 4)))

    assert (low.mass, low.boundary) != (high.mass, high.boundary)
    assert (low.mass, low.boundary) != (uncoupled.mass, uncoupled.boundary)
    assert growth_at_label(Fraction(1, 4), kappa) != growth_at_label(Fraction(3, 4), kappa)


@pytest.mark.parametrize("kappa", [Fraction(1, 2), Fraction(1)])
def test_g_couple_killing_kappa_is_refused(kappa: Fraction) -> None:
    with pytest.raises(ValueError, match="killing tests"):
        admissible_kappa(kappa)
    with pytest.raises(ValueError, match="I_r"):
        source_hybrid_step(
            HybridState.from_values(Fraction(1, 2), Fraction(1, 2), Fraction(1, 4)),
            kappa=kappa,
        )
    with pytest.raises(ValueError, match="I_r"):
        UniverseKernel(kappa=kappa)


def test_g_couple_killing_tests_document_low_label_dividing_root_failure() -> None:
    # Parent-reading killing tests, not members of I_r.  Algebra cited from
    # the math lane: at q=1/4, r=(9/2)(1-kappa/2) and Δ_r=9r^2-32r+4.
    assert KAPPA_OPEN_RIGHT == Fraction(86, 315)
    assert Fraction(1, 4) < KAPPA_OPEN_RIGHT
    assert Fraction(1, 2) > KAPPA_OPEN_RIGHT

    growth_half = killing_low_label_growth(Fraction(1, 2))
    growth_one = killing_low_label_growth(Fraction(1))
    assert growth_half == Fraction(27, 8)
    assert growth_one == Fraction(9, 4)
    assert dividing_mass_discriminant(growth_half) == DIVIDING_DISCRIMINANT_AT_HALF
    assert dividing_mass_discriminant(growth_one) == DIVIDING_DISCRIMINANT_AT_ONE
    assert DIVIDING_DISCRIMINANT_AT_HALF < 0
    assert DIVIDING_DISCRIMINANT_AT_ONE < 0


def test_g_couple_half_label_extinction_area_stays_at_source_wedge() -> None:
    assert source_one_step_extinction_area() == SOURCE_EXTINCTION_AREA
    assert SOURCE_EXTINCTION_AREA == Fraction(1, 10)
    assert SOURCE_EXTINCTION_AREA >= EXTINCTION_AREA_FLOOR
    assert growth_at_label(Fraction(1, 2), Fraction(0)) == NOMINAL_GROWTH
    assert growth_at_label(Fraction(1, 2), Fraction(1, 4)) == NOMINAL_GROWTH

    # Source one-step wedge includes high-m / low-b points; (0,0) is already extinct.
    collapsed = source_hybrid_step(HybridState.from_values(1, 0, Fraction(1, 2)))
    assert collapsed.mass == 0
    already_extinct = source_hybrid_step(
        HybridState.from_values(0, 0, Fraction(1, 2))
    )
    assert already_extinct.mass == 0
    already_extinct_small_b = source_hybrid_step(
        HybridState.from_values(0, Fraction(1, 20), Fraction(1, 2))
    )
    assert already_extinct_small_b.mass == 0


def test_kernel_applies_the_same_flux_and_rejects_reward_arguments() -> None:
    kernel = UniverseKernel()
    left = HybridState.from_values(Fraction(1, 2), Fraction(1, 2), Fraction(1, 4))
    right = HybridState.from_values(0, 0, Fraction(3, 4))
    universe = kernel.host((left, right), flux=1)
    stepped = kernel.step(universe)

    assert inspect.signature(kernel.step).parameters.keys() == {"universe"}
    assert "reward" not in inspect.signature(SourceHybridSubsystem.step).parameters
    assert hosted_states(stepped) == (
        source_hybrid_step(left),
        source_hybrid_step(right),
    )
    assert stepped.flux == 1
    assert stepped.residual == 0


def test_default_chemostat_holds_unit_flux() -> None:
    kernel = UniverseKernel()
    assert kernel.leakage == 0
    assert kernel.target_flux == 1
    assert kernel.next_flux(Fraction(1)) == 1
    assert kernel.next_flux(Fraction(2, 5)) == Fraction(2, 5)


def test_public_exports_match_module_names() -> None:
    assert clarus.HybridState is HybridState
    assert clarus.UniverseState is UniverseState
    assert clarus.UniverseKernel is UniverseKernel
    assert clarus.SourceHybridSubsystem is SourceHybridSubsystem
    assert clarus.NOMINAL_PARAMETERS is NOMINAL_PARAMETERS
    assert clarus.source_hybrid_step is source_hybrid_step
    assert clarus.iterate_source is iterate_source
    assert clarus.admissible_kappa is admissible_kappa
    assert clarus.SOURCE_EXTINCTION_AREA == SOURCE_EXTINCTION_AREA
    assert clarus.KAPPA_OPEN_RIGHT == KAPPA_OPEN_RIGHT
