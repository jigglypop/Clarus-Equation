"""Focused tests for the explicit finite-quench perturbation source ledger."""

from __future__ import annotations

import math

import pytest

from examples.physics.finite_quench_perturbation_contract import (
    FiniteQuenchPerturbationAxioms,
    FiniteQuenchPerturbationContract,
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


def _axioms(**overrides: object) -> FiniteQuenchPerturbationAxioms:
    values = dict(
        gauge="newtonian",
        transfer_frame="declared_common_scalar_frame",
        initial_mode="density_equal_time_shift_seed",
        anisotropic_stress_model="zero",
        energy_transfer_bias=0.75,
        momentum_drag_bias=0.4,
        produced_equation_of_state=0.0,
        produced_sound_speed_squared=0.0,
        reservoir_sound_speed_squared=0.2,
    )
    values.update(overrides)
    return FiniteQuenchPerturbationAxioms(**values)


def _receipt(
    contract: FiniteQuenchPerturbationContract,
    **overrides: object,
):
    values = dict(
        n=-4.0,
        k_over_a_h=3.0,
        produced_delta=0.2,
        reservoir_delta=0.1,
        produced_velocity_divergence=0.4,
        reservoir_velocity_divergence=-0.1,
    )
    values.update(overrides)
    return contract.receipt(**values)


def test_energy_and_momentum_exchange_pairs_cancel_exactly() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    receipt = _receipt(contract)
    assert receipt.background_source > 0.0
    assert receipt.produced_energy_transfer == -receipt.reservoir_energy_transfer
    assert receipt.energy_transfer_sum_residual == 0.0
    assert receipt.produced_momentum_potential == (
        -receipt.reservoir_momentum_potential
    )
    assert receipt.momentum_potential_sum_residual == 0.0
    assert receipt.normalized_gradient_sum_residual == 0.0
    assert receipt.nonnegative_drag_quadratic_proxy > 0.0
    assert receipt.declared_sound_speeds_in_unit_interval
    assert receipt.role.endswith("NOT_EINSTEIN_BOLTZMANN_SOLUTION")


def test_zero_wavenumber_has_no_physical_spatial_transfer() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    receipt = _receipt(contract, k_over_a_h=0.0)
    assert receipt.produced_momentum_potential != 0.0
    assert receipt.produced_normalized_gradient_proxy == 0.0
    assert receipt.reservoir_normalized_gradient_proxy == 0.0
    assert receipt.normalized_gradient_sum_residual == 0.0


def test_no_background_source_means_no_exchange() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    receipt = _receipt(contract, n=-5.0)
    assert receipt.background_source == 0.0
    assert receipt.produced_energy_transfer == 0.0
    assert receipt.reservoir_energy_transfer == 0.0
    assert receipt.produced_momentum_potential == 0.0
    assert receipt.nonnegative_drag_quadratic_proxy == 0.0
    assert receipt.density_equal_time_shift_residual is None
    assert receipt.density_equal_time_shift_at_this_node is None
    assert receipt.declared_source_time_shift_residual is None
    assert receipt.density_and_source_equal_time_shift_at_this_node is None
    assert receipt.source_time_shift_diagnostic_status == "DENSITY_CLOCK_UNAVAILABLE"


def test_interacting_equal_time_shift_pair_is_constructed_not_inferred() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    produced_delta = 0.2
    reservoir_delta = contract.reservoir_delta_for_density_equal_time_shift(
        -4.0,
        produced_delta
    )
    equal_pair = _receipt(
        contract,
        produced_delta=produced_delta,
        reservoir_delta=reservoir_delta,
    )
    unequal_pair = _receipt(
        contract,
        produced_delta=produced_delta,
        reservoir_delta=reservoir_delta + 0.1,
    )
    assert abs(equal_pair.density_equal_time_shift_residual) <= 1.0e-16
    assert equal_pair.density_equal_time_shift_at_this_node
    assert abs(equal_pair.declared_source_time_shift_residual) > 1.0e-6
    assert not equal_pair.density_and_source_equal_time_shift_at_this_node
    assert equal_pair.source_time_shift_diagnostic_status == (
        "NONDEGENERATE_NODE_DIAGNOSTIC"
    )
    assert not unequal_pair.density_equal_time_shift_at_this_node
    assert abs(equal_pair.noninteracting_barotropic_entropy_proxy) > 1.0e-6


def test_post_source_equal_time_shift_reduces_to_barotropic_proxy() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    produced_delta = 0.2
    n_post = -3.0
    reservoir_delta = contract.reservoir_delta_for_density_equal_time_shift(
        n_post,
        produced_delta,
    )
    receipt = _receipt(
        contract,
        n=n_post,
        produced_delta=produced_delta,
        reservoir_delta=reservoir_delta,
    )
    assert receipt.background_source == 0.0
    assert receipt.density_equal_time_shift_at_this_node
    assert receipt.declared_source_time_shift_residual == 0.0
    assert receipt.density_and_source_equal_time_shift_at_this_node
    assert receipt.source_time_shift_diagnostic_status == "VACUOUS_NO_SOURCE"
    assert abs(receipt.noninteracting_barotropic_entropy_proxy) <= 1.0e-16


def test_equal_velocities_remove_the_declared_drag_channel() -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    receipt = _receipt(
        contract,
        produced_velocity_divergence=0.3,
        reservoir_velocity_divergence=0.3,
    )
    assert receipt.produced_momentum_potential == 0.0
    assert receipt.nonnegative_drag_quadratic_proxy == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(gauge="synchronous"),
        dict(transfer_frame="total_energy_frame"),
        dict(initial_mode="adiabatic"),
        dict(anisotropic_stress_model="free"),
        dict(momentum_drag_bias=-0.1),
        dict(produced_equation_of_state=-1.0),
        dict(produced_equation_of_state=0.1),
        dict(produced_sound_speed_squared=-0.1),
        dict(reservoir_sound_speed_squared=1.1),
        dict(energy_transfer_bias=math.nan),
    ],
)
def test_perturbation_axioms_fail_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _axioms(**overrides)


def test_vacuum_reservoir_has_no_fluid_rest_frame_in_this_branch() -> None:
    with pytest.raises(ValueError):
        FiniteQuenchPerturbationContract(
            _bridge(w_reservoir=-1.0),
            _axioms(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(n=0.1),
        dict(n=-20.0),
        dict(k_over_a_h=-0.1),
        dict(produced_delta=True),
        dict(reservoir_velocity_divergence=math.inf),
        dict(
            produced_velocity_divergence=1.0e308,
            reservoir_velocity_divergence=-1.0e308,
        ),
    ],
)
def test_receipt_inputs_fail_closed(overrides: dict[str, object]) -> None:
    contract = FiniteQuenchPerturbationContract(_bridge(), _axioms())
    with pytest.raises(ValueError):
        _receipt(contract, **overrides)
