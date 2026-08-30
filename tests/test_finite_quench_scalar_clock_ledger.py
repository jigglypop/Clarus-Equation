"""Focused tests for the singularity-free common scalar-clock ledger."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_perturbation_contract import (
    FiniteQuenchPerturbationAxioms,
    FiniteQuenchPerturbationContract,
)
from examples.physics.finite_quench_scalar_clock_ledger import (
    FiniteQuenchScalarClockLedger,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge() -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=0.12,
            reservoir_present_density=0.21,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


def _clock_ledger() -> FiniteQuenchScalarClockLedger:
    return FiniteQuenchScalarClockLedger(_bridge())


def _constant_bias_contract() -> FiniteQuenchPerturbationContract:
    return FiniteQuenchPerturbationContract(
        _bridge(),
        FiniteQuenchPerturbationAxioms(
            gauge="newtonian",
            transfer_frame="declared_common_scalar_frame",
            initial_mode="density_equal_time_shift_seed",
            anisotropic_stress_model="zero",
            energy_transfer_bias=0.75,
            momentum_drag_bias=0.4,
            produced_equation_of_state=0.0,
            produced_sound_speed_squared=0.0,
            reservoir_sound_speed_squared=0.2,
        ),
    )


def test_constructed_clock_ledger_closes_all_algebraic_residuals() -> None:
    receipt = _clock_ledger().construct(n=-4.0, scalar_clock_shift=0.25)
    assert receipt.produced_density_clock_residual == 0.0
    assert receipt.reservoir_density_clock_residual == 0.0
    assert receipt.source_clock_residual == 0.0
    assert receipt.paired_source_residual == 0.0
    assert receipt.all_declared_clock_constraints_hold
    assert receipt.model == "declared_common_scalar_clock_ledger"
    assert receipt.role.endswith("NOT_COVARIANT_QMU_OR_DYNAMICAL_SOLUTION")


def test_audit_rejects_a_perturbed_source_without_dividing_by_q_prime() -> None:
    ledger = _clock_ledger()
    exact = ledger.construct(n=-4.0, scalar_clock_shift=0.25)
    audited = ledger.audit(
        n=exact.n,
        scalar_clock_shift=exact.scalar_clock_shift,
        produced_density_perturbation=exact.produced_density_perturbation,
        reservoir_density_perturbation=exact.reservoir_density_perturbation,
        produced_energy_transfer_perturbation=(
            exact.produced_energy_transfer_perturbation + 0.125
        ),
        reservoir_energy_transfer_perturbation=(
            exact.reservoir_energy_transfer_perturbation - 0.125
        ),
    )
    assert audited.produced_density_clock_holds
    assert audited.reservoir_density_clock_holds
    assert not audited.source_clock_holds
    assert audited.paired_source_cancels
    assert not audited.all_declared_clock_constraints_hold


def test_source_endpoint_is_nonsingular_but_does_not_identify_clock() -> None:
    receipt = _clock_ledger().construct(n=-3.5, scalar_clock_shift=0.75)
    assert receipt.background_source == 0.0
    assert receipt.background_source_derivative == 0.0
    assert receipt.produced_energy_transfer_perturbation == 0.0
    assert receipt.source_clock_residual == 0.0
    assert receipt.source_clock_holds
    assert receipt.source_derivative_exact_float_zero


def test_zero_produced_background_uses_absolute_perturbation_not_contrast() -> None:
    receipt = _clock_ledger().construct(n=-5.0, scalar_clock_shift=0.75)
    assert receipt.produced_density == 0.0
    assert receipt.produced_density_derivative == 0.0
    assert receipt.produced_density_perturbation == 0.0
    assert receipt.produced_density_contrast_or_none is None
    assert receipt.produced_density_contrast_status == (
        "UNDEFINED_ZERO_BACKGROUND_DENSITY"
    )
    assert not receipt.produced_clock_identifiable_from_density_derivative
    assert receipt.all_declared_clock_constraints_hold


def test_arbitrary_density_and_unpaired_source_fail_independent_checks() -> None:
    exact = _clock_ledger().construct(n=-4.0, scalar_clock_shift=0.25)
    audited = _clock_ledger().audit(
        n=exact.n,
        scalar_clock_shift=exact.scalar_clock_shift,
        produced_density_perturbation=(
            exact.produced_density_perturbation + 0.25
        ),
        reservoir_density_perturbation=exact.reservoir_density_perturbation,
        produced_energy_transfer_perturbation=(
            exact.produced_energy_transfer_perturbation
        ),
        reservoir_energy_transfer_perturbation=0.0,
    )
    assert not audited.produced_density_clock_holds
    assert audited.reservoir_density_clock_holds
    assert audited.source_clock_holds
    assert not audited.paired_source_cancels
    assert not audited.all_declared_clock_constraints_hold


def test_constant_density_bias_fails_the_common_clock_at_source_centre() -> None:
    contract = _constant_bias_contract()
    ledger = FiniteQuenchScalarClockLedger(contract.bridge)
    n = -4.0
    delta_p = 0.2
    delta_r = contract.reservoir_delta_for_density_equal_time_shift(n, delta_p)
    old = contract.receipt(
        n=n,
        k_over_a_h=3.0,
        produced_delta=delta_p,
        reservoir_delta=delta_r,
        produced_velocity_divergence=0.4,
        reservoir_velocity_divergence=-0.1,
    )
    rho_p = contract.bridge.production_density(n)
    rho_r = contract.bridge.reservoir_density(n)
    clock = rho_p * delta_p / contract.bridge.production_derivative(n)
    audited = ledger.audit(
        n=n,
        scalar_clock_shift=clock,
        produced_density_perturbation=rho_p * delta_p,
        reservoir_density_perturbation=rho_r * delta_r,
        produced_energy_transfer_perturbation=old.produced_energy_transfer,
        reservoir_energy_transfer_perturbation=old.reservoir_energy_transfer,
    )
    assert audited.produced_density_clock_holds
    assert audited.reservoir_density_clock_holds
    assert not audited.source_clock_holds
    assert audited.paired_source_cancels


def test_in_support_q_prime_root_exposes_constant_bias_obstruction() -> None:
    bridge = _bridge()
    ledger = FiniteQuenchScalarClockLedger(bridge)
    n_root = -4.0 - 1.0 / 6.0
    source = bridge.source(n_root)
    source_prime = bridge.source_derivative(n_root)
    clock_receipt = ledger.construct(n=n_root, scalar_clock_shift=0.3)
    constant_bias_delta_q = 0.75 * source * 0.2
    assert source > 0.0
    assert abs(source_prime) <= 1.0e-12 * source
    assert abs(clock_receipt.produced_energy_transfer_perturbation) <= (
        1.0e-12 * source
    )
    assert abs(constant_bias_delta_q) > 1.0e-3 * source


def test_no_single_constant_density_bias_matches_two_source_nodes() -> None:
    bridge = _bridge()
    delta_p = 0.2

    def required_bias(n: float) -> float:
        rho_p = bridge.production_density(n)
        rho_p_prime = bridge.production_derivative(n)
        clock = rho_p * delta_p / rho_p_prime
        return (
            bridge.source_derivative(n)
            * clock
            / (bridge.source(n) * delta_p)
        )

    centre_bias = required_bias(-4.0)
    root_bias = required_bias(-4.0 - 1.0 / 6.0)
    assert centre_bias == pytest.approx(-4.0, rel=1.0e-14, abs=1.0e-14)
    assert root_bias == pytest.approx(0.0, abs=1.0e-12)
    assert abs(centre_bias - root_bias) > 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=0.1, scalar_clock_shift=0.2),
        dict(n=-4.0, scalar_clock_shift=math.nan),
        dict(n=-4.0, scalar_clock_shift=True),
        dict(n=-4.0, scalar_clock_shift=1.0e308),
    ],
)
def test_construct_inputs_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _clock_ledger().construct(**kwargs)
