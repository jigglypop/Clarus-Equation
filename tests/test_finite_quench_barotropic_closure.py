"""Focused tests for the strict causal barotropic constitutive branch."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_barotropic_closure import (
    FiniteQuenchStrictBarotropicClosure,
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


def test_constructed_strict_barotrope_closes_pressure_and_stress() -> None:
    receipt = FiniteQuenchStrictBarotropicClosure(_bridge()).construct(
        n=-4.0,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
    )
    assert receipt.produced_pressure_perturbation == 0.0
    assert receipt.reservoir_pressure_perturbation == pytest.approx(-0.008)
    assert receipt.produced_normalized_anisotropic_stress == 0.0
    assert receipt.reservoir_normalized_anisotropic_stress == 0.0
    assert receipt.pressure_closure_holds
    assert receipt.zero_anisotropic_stress_holds
    assert receipt.all_strict_barotropic_constraints_hold
    assert receipt.role.endswith("NOT_MICROPHYSICAL_QUENCH_DERIVATION")


def test_wrong_reservoir_pressure_is_detected() -> None:
    closure = FiniteQuenchStrictBarotropicClosure(_bridge())
    receipt = closure.audit(
        n=-4.0,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=0.02,
        produced_normalized_anisotropic_stress=0.0,
        reservoir_normalized_anisotropic_stress=0.0,
        produced_background_pressure_derivative=0.0,
        reservoir_background_pressure_derivative=(
            0.1 * closure.bridge.reservoir_derivative(-4.0)
        ),
    )
    assert not receipt.pressure_closure_holds
    assert not receipt.all_strict_barotropic_constraints_hold


def test_nonzero_anisotropic_stress_is_detected_independently() -> None:
    closure = FiniteQuenchStrictBarotropicClosure(_bridge())
    receipt = closure.audit(
        n=-4.0,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=-0.008,
        produced_normalized_anisotropic_stress=0.0,
        reservoir_normalized_anisotropic_stress=0.1,
        produced_background_pressure_derivative=0.0,
        reservoir_background_pressure_derivative=(
            0.1 * closure.bridge.reservoir_derivative(-4.0)
        ),
    )
    assert receipt.pressure_closure_holds
    assert not receipt.zero_anisotropic_stress_holds
    assert not receipt.all_strict_barotropic_constraints_hold


def test_interaction_nonadiabatic_coefficient_vanishes_on_source_support() -> None:
    bridge = _bridge()
    assert bridge.source(-4.0) > 0.0
    receipt = FiniteQuenchStrictBarotropicClosure(bridge).construct(
        n=-4.0,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
    )
    assert receipt.nonadiabatic_interaction_coefficient_produced == 0.0
    assert receipt.nonadiabatic_interaction_coefficient_reservoir == 0.0
    assert receipt.reservoir_rest_frame_sound_speed_squared == (
        receipt.reservoir_adiabatic_sound_speed_squared
    )


def test_zero_produced_background_before_source_requires_no_division() -> None:
    bridge = _bridge()
    assert bridge.production_density(-5.0) == 0.0
    receipt = FiniteQuenchStrictBarotropicClosure(bridge).construct(
        n=-5.0,
        produced_density_perturbation=0.0,
        reservoir_density_perturbation=0.02,
    )
    assert receipt.all_strict_barotropic_constraints_hold


def test_background_barotrope_derivative_matches_independent_difference() -> None:
    bridge = _bridge()
    w_r = bridge.config.w_reservoir
    n = -4.0
    step = 1.0e-5
    pressure_plus = w_r * bridge.reservoir_density(n + step)
    pressure_minus = w_r * bridge.reservoir_density(n - step)
    finite_difference = (pressure_plus - pressure_minus) / (2.0 * step)
    analytic = w_r * bridge.reservoir_derivative(n)
    assert finite_difference == pytest.approx(analytic, rel=2.0e-9)


def test_wrong_background_pressure_derivative_is_detected() -> None:
    closure = FiniteQuenchStrictBarotropicClosure(_bridge())
    receipt = closure.audit(
        n=-4.0,
        produced_density_perturbation=0.12,
        reservoir_density_perturbation=-0.08,
        produced_pressure_perturbation=0.0,
        reservoir_pressure_perturbation=-0.008,
        produced_normalized_anisotropic_stress=0.0,
        reservoir_normalized_anisotropic_stress=0.0,
        produced_background_pressure_derivative=0.0,
        reservoir_background_pressure_derivative=123.0,
    )
    assert receipt.pressure_closure_holds
    assert not receipt.background_barotrope_derivative_holds
    assert not receipt.all_strict_barotropic_constraints_hold


@pytest.mark.parametrize("w_reservoir", [-1.0, -0.1, 1.1])
def test_noncausal_or_gradient_unstable_strict_barotrope_is_rejected(
    w_reservoir: float,
) -> None:
    with pytest.raises(ValueError, match="0 <= w_reservoir <= 1"):
        FiniteQuenchStrictBarotropicClosure(
            _bridge(w_reservoir=w_reservoir)
        )


@pytest.mark.parametrize("w_reservoir", [0.0, 1.0])
def test_causal_endpoint_barotropes_are_admitted(w_reservoir: float) -> None:
    receipt = FiniteQuenchStrictBarotropicClosure(
        _bridge(w_reservoir=w_reservoir)
    ).construct(
        n=-4.0,
        produced_density_perturbation=0.1,
        reservoir_density_perturbation=0.2,
    )
    assert receipt.causal_nonnegative_sound_speed_branch
    assert receipt.all_strict_barotropic_constraints_hold


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            n=0.1,
            produced_density_perturbation=0.1,
            reservoir_density_perturbation=0.2,
        ),
        dict(
            n=-4.0,
            produced_density_perturbation=math.nan,
            reservoir_density_perturbation=0.2,
        ),
        dict(
            n=-4.0,
            produced_density_perturbation=True,
            reservoir_density_perturbation=0.2,
        ),
    ],
)
def test_closure_inputs_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FiniteQuenchStrictBarotropicClosure(_bridge()).construct(**kwargs)
