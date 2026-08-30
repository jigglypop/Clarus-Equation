"""Focused tests for the two-component-only flat-GR background contract."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(*, w_reservoir: float = 0.1) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=w_reservoir,
            w_open=2.1767e-4,
        )
    )


@pytest.mark.parametrize("n", [-5.0, -4.0, -3.0, 0.0])
def test_two_fluid_flat_gr_background_constraints_hold(n: float) -> None:
    receipt = FiniteQuenchTwoFluidFlatGRBackground(_bridge()).construct(n)
    assert receipt.total_density > 0.0
    assert receipt.source_pair_residual == 0.0
    assert receipt.source_pair_cancels
    assert receipt.total_continuity_holds
    assert receipt.friedmann_normalization_holds
    assert receipt.raychaudhuri_normalization_holds
    assert receipt.all_background_constraints_hold


def test_gravity_coupling_and_friedmann_unit_are_exactly_related() -> None:
    receipt = FiniteQuenchTwoFluidFlatGRBackground(_bridge()).construct(-4.0)
    assert receipt.omega_density_unit == pytest.approx(
        1.0 / receipt.total_density
    )
    assert receipt.gravity_constraint_coupling == pytest.approx(
        1.5 * receipt.omega_density_unit
    )
    assert receipt.hubble_squared_over_eight_pi_g_rho_unit_over_three == (
        receipt.total_density
    )


def test_raychaudhuri_h_matches_independent_log_density_difference() -> None:
    bridge = _bridge()
    background = FiniteQuenchTwoFluidFlatGRBackground(bridge)
    n = -4.0
    step = 1.0e-5

    def log_hubble(node: float) -> float:
        rho_total = (
            bridge.production_density(node) + bridge.reservoir_density(node)
        )
        return 0.5 * math.log(rho_total)

    finite_difference = (log_hubble(n + step) - log_hubble(n - step)) / (
        2.0 * step
    )
    receipt = background.construct(n)
    assert finite_difference == pytest.approx(
        receipt.hubble_log_derivative,
        rel=3.0e-10,
    )


def test_fixed_comoving_kappa_log_derivative_includes_expansion() -> None:
    receipt = FiniteQuenchTwoFluidFlatGRBackground(_bridge()).construct(-4.0)
    assert receipt.kappa_log_derivative_at_fixed_comoving_k == pytest.approx(
        -1.0 - receipt.hubble_log_derivative
    )


def test_species_manifest_makes_the_two_component_only_axiom_explicit() -> None:
    receipt = FiniteQuenchTwoFluidFlatGRBackground(_bridge()).construct(-4.0)
    assert receipt.species_manifest == ("produced", "reservoir")
    assert receipt.external_background_species_assumed_absent
    assert receipt.role.endswith("NOT_OBSERVED_COSMOLOGY")


def test_vacuum_like_reservoir_background_remains_algebraically_regular() -> None:
    receipt = FiniteQuenchTwoFluidFlatGRBackground(
        _bridge(w_reservoir=-1.0)
    ).construct(-5.0)
    assert receipt.total_density > 0.0
    assert receipt.total_enthalpy == 0.0
    assert receipt.hubble_log_derivative == 0.0
    assert receipt.all_background_constraints_hold


def test_wrong_hubble_squared_candidate_fails_friedmann_only() -> None:
    background = FiniteQuenchTwoFluidFlatGRBackground(_bridge())
    expected = background.construct(-4.0)
    receipt = background.audit(
        -4.0,
        normalized_hubble_squared=(
            1.1
            * expected.hubble_squared_over_eight_pi_g_rho_unit_over_three
        ),
        hubble_log_derivative=expected.hubble_log_derivative,
    )
    assert not receipt.friedmann_normalization_holds
    assert receipt.raychaudhuri_normalization_holds
    assert not receipt.all_background_constraints_hold


def test_wrong_hubble_log_derivative_fails_raychaudhuri_only() -> None:
    background = FiniteQuenchTwoFluidFlatGRBackground(_bridge())
    expected = background.construct(-4.0)
    receipt = background.audit(
        -4.0,
        normalized_hubble_squared=(
            expected.hubble_squared_over_eight_pi_g_rho_unit_over_three
        ),
        hubble_log_derivative=expected.hubble_log_derivative + 0.5,
    )
    assert receipt.friedmann_normalization_holds
    assert not receipt.raychaudhuri_normalization_holds
    assert not receipt.all_background_constraints_hold


@pytest.mark.parametrize("n", [True, math.nan, math.inf, 0.1, -20.0])
def test_background_inputs_fail_closed(n: object) -> None:
    with pytest.raises(ValueError):
        FiniteQuenchTwoFluidFlatGRBackground(_bridge()).construct(n)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(normalized_hubble_squared=0.0, hubble_log_derivative=-1.0),
        dict(normalized_hubble_squared=math.inf, hubble_log_derivative=-1.0),
        dict(normalized_hubble_squared=1.0, hubble_log_derivative=math.nan),
        dict(normalized_hubble_squared=True, hubble_log_derivative=-1.0),
    ],
)
def test_external_background_candidates_fail_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FiniteQuenchTwoFluidFlatGRBackground(_bridge()).audit(-4.0, **kwargs)
