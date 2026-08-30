"""Focused integration and falsifier tests for the common-clock GR node."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_gr_linear_node import (
    FiniteQuenchGRLinearNode,
    GRLinearNodeReceipt,
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


def _construct(
    *,
    n: float = -4.0,
    kappa: float = 2.0,
    clock: float = 0.25,
    momentum_p: float = 0.01,
    momentum_r: float = -0.005,
    intrinsic_f_p: float = 0.003,
) -> GRLinearNodeReceipt:
    return FiniteQuenchGRLinearNode(_bridge()).construct(
        n=n,
        k_over_a_h=kappa,
        scalar_clock_shift=clock,
        produced_momentum_density=momentum_p,
        reservoir_momentum_density=momentum_r,
        produced_intrinsic_momentum_potential=intrinsic_f_p,
    )


def _reaudit(
    node: GRLinearNodeReceipt,
    **overrides: object,
) -> GRLinearNodeReceipt:
    fields: dict[str, object] = dict(
        background=node.background,
        scalar_clock=node.scalar_clock,
        closure=node.closure,
        einstein_constraint=node.einstein_constraint,
        transfer_projection=node.transfer_projection,
        energy_equation=node.energy_equation,
        momentum_equation=node.momentum_equation,
    )
    fields.update(overrides)
    return FiniteQuenchGRLinearNode(_bridge()).audit(**fields)


def test_constructed_common_clock_gr_node_closes_every_raw_gate() -> None:
    node = _construct()
    assert node.background_holds
    assert node.common_scalar_clock_holds
    assert node.strict_barotropic_closure_holds
    assert node.scalar_einstein_constraints_hold
    assert node.lower_qmu_projection_holds
    assert node.energy_equations_hold
    assert node.momentum_equations_hold
    assert node.prior_linear_system_gate_holds
    assert node.all_cross_receipt_state_identifications_hold
    assert node.full_declared_gr_linear_node_holds
    assert not node.failure_reasons
    assert all(residual == 0.0 for _, residual in node.cross_residuals)


def test_normalized_q_clock_and_physical_q_clock_differ_by_hq_t() -> None:
    bridge = _bridge()
    node = _construct()
    clock = node.scalar_clock
    projection = node.transfer_projection
    expected_correction = (
        node.background.hubble_log_derivative
        * bridge.source(clock.n)
        * clock.scalar_clock_shift
    )
    assert (
        projection.produced_physical_energy_perturbation
        - clock.produced_energy_transfer_perturbation
    ) == pytest.approx(expected_correction)


def test_internal_q_prime_root_keeps_required_hubble_clock_correction() -> None:
    bridge = _bridge()
    n_root = -4.0 - 1.0 / 6.0
    node = _construct(n=n_root)
    clock_source = node.scalar_clock.produced_energy_transfer_perturbation
    physical_source = (
        node.transfer_projection.produced_physical_energy_perturbation
    )
    scale = bridge.source(n_root)
    assert scale > 0.0
    assert abs(clock_source) <= 1.0e-12 * scale
    assert abs(physical_source) > 1.0e-3 * scale
    assert node.full_declared_gr_linear_node_holds


@pytest.mark.parametrize(
    "n,momentum_p",
    [(-5.0, 0.0), (-4.0, 0.01), (-3.0, 0.01), (0.0, 0.01)],
)
def test_source_on_and_off_nodes_close(n: float, momentum_p: float) -> None:
    node = _construct(n=n, momentum_p=momentum_p, momentum_r=0.005)
    assert node.full_declared_gr_linear_node_holds


def test_zero_produced_enthalpy_requires_zero_produced_momentum() -> None:
    bridge = _bridge()
    assert bridge.production_density(-5.0) == 0.0
    with pytest.raises(ValueError, match="momentum must vanish"):
        _construct(n=-5.0, momentum_p=0.01, momentum_r=0.005)


def test_total_velocity_is_recovered_from_total_momentum_and_enthalpy() -> None:
    node = _construct()
    expected = (
        node.einstein_constraint.total_momentum_density
        / node.background.total_enthalpy
    )
    assert (
        node.transfer_projection.normalized_total_velocity_potential
        == pytest.approx(expected)
    )
    assert node.momentum_equation.total_energy_frame_relation_holds


def test_intrinsic_momentum_transfer_is_free_but_must_be_paired() -> None:
    node = _construct(intrinsic_f_p=0.37)
    projection = node.transfer_projection
    assert projection.produced_intrinsic_momentum_potential == pytest.approx(0.37)
    assert projection.reservoir_intrinsic_momentum_potential == pytest.approx(-0.37)
    assert projection.intrinsic_momentum_pair_cancels
    assert node.full_declared_gr_linear_node_holds


def test_forged_clock_boolean_cannot_hide_wrong_density_clock() -> None:
    node = _construct()
    forged = replace(
        node.scalar_clock,
        produced_density_perturbation=(
            node.scalar_clock.produced_density_perturbation + 0.1
        ),
        all_declared_clock_constraints_hold=True,
    )
    audited = _reaudit(node, scalar_clock=forged)
    assert not audited.common_scalar_clock_holds
    assert not audited.all_cross_receipt_state_identifications_hold
    assert not audited.full_declared_gr_linear_node_holds


def test_forged_einstein_boolean_cannot_hide_wrong_curvature() -> None:
    node = _construct()
    forged = replace(
        node.einstein_constraint,
        curvature_potential=(
            node.einstein_constraint.curvature_potential + 0.1
        ),
        all_declared_scalar_constraints_hold=True,
    )
    audited = _reaudit(node, einstein_constraint=forged)
    assert not audited.scalar_einstein_constraints_hold
    assert not audited.full_declared_gr_linear_node_holds


def test_forged_qmu_boolean_cannot_hide_wrong_physical_clock_source() -> None:
    node = _construct()
    projection = node.transfer_projection
    forged = replace(
        projection,
        produced_physical_energy_perturbation=(
            projection.produced_physical_energy_perturbation + 0.1
        ),
        reservoir_physical_energy_perturbation=(
            projection.reservoir_physical_energy_perturbation - 0.1
        ),
        common_clock_physical_source_holds=True,
        all_declared_lower_component_constraints_hold=True,
    )
    audited = _reaudit(node, transfer_projection=forged)
    assert not audited.lower_qmu_projection_holds
    assert not audited.all_cross_receipt_state_identifications_hold
    assert not audited.full_declared_gr_linear_node_holds


def test_wrong_energy_derivative_is_recomputed_and_rejected() -> None:
    node = _construct()
    energy = node.energy_equation
    forged = replace(
        energy,
        provided_produced_density_perturbation_derivative=(
            energy.provided_produced_density_perturbation_derivative + 0.1
        ),
        energy_equations_and_exchange_hold=True,
        common_clock_energy_branch_holds=True,
    )
    audited = _reaudit(node, energy_equation=forged)
    assert not audited.energy_equations_hold
    assert not audited.prior_linear_system_gate_holds
    assert not audited.full_declared_gr_linear_node_holds


def test_wrong_momentum_derivative_is_recomputed_and_rejected() -> None:
    node = _construct()
    momentum = node.momentum_equation
    forged = replace(
        momentum,
        provided_reservoir_momentum_density_derivative=(
            momentum.provided_reservoir_momentum_density_derivative + 0.1
        ),
        momentum_equations_and_exchange_hold=True,
        total_energy_frame_momentum_branch_holds=True,
    )
    audited = _reaudit(node, momentum_equation=forged)
    assert not audited.momentum_equations_hold
    assert not audited.prior_linear_system_gate_holds
    assert not audited.full_declared_gr_linear_node_holds


def test_pressure_state_mismatch_between_closure_and_fluid_equations_fails() -> None:
    node = _construct()
    forged = replace(
        node.closure,
        reservoir_pressure_perturbation=(
            node.closure.reservoir_pressure_perturbation + 0.1
        ),
        pressure_closure_holds=True,
        all_strict_barotropic_constraints_hold=True,
    )
    audited = _reaudit(node, closure=forged)
    assert not audited.strict_barotropic_closure_holds
    assert not audited.all_cross_receipt_state_identifications_hold
    assert not audited.full_declared_gr_linear_node_holds


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kappa=0.0),
        dict(kappa=math.inf),
        dict(clock=math.nan),
        dict(momentum_p=True),
        dict(intrinsic_f_p=math.inf),
    ],
)
def test_construct_inputs_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _construct(**kwargs)


def test_receipt_names_free_inputs_and_denies_time_integration() -> None:
    node = _construct()
    assert "scalar_clock_shift" in node.free_declared_inputs
    assert "produced_intrinsic_momentum_potential" in node.free_declared_inputs
    assert not node.common_clock_tangent_preservation_proven
    assert not node.finite_step_constraint_propagation_proven
    assert node.role.endswith(
        "NOT_TIME_INTEGRATED_MICROPHYSICAL_OR_OBSERVABLE_SOLUTION"
    )
