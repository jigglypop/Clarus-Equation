"""Focused falsifier tests for the kappa>0 scalar Einstein constraints."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from examples.physics.finite_quench_barotropic_closure import (
    FiniteQuenchStrictBarotropicClosure,
)
from examples.physics.finite_quench_einstein_constraint import (
    FiniteQuenchScalarEinsteinConstraint,
)
from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
    FiniteQuenchBridgeConfig,
)


def _bridge(
    *,
    omega_prod0: float = 0.12,
    reservoir_present_density: float = 0.21,
) -> FiniteQuenchBridge:
    return FiniteQuenchBridge(
        FiniteQuenchBridgeConfig(
            n_star=-4.0,
            half_width=0.5,
            omega_prod0=omega_prod0,
            reservoir_present_density=reservoir_present_density,
            w_reservoir=0.1,
            w_open=2.1767e-4,
        )
    )


def _receipts(
    bridge: FiniteQuenchBridge,
    *,
    n: float = -4.0,
    delta_p: float = 0.12,
    delta_r: float = -0.08,
):
    background = FiniteQuenchTwoFluidFlatGRBackground(bridge).construct(n)
    closure = FiniteQuenchStrictBarotropicClosure(bridge).construct(
        n=n,
        produced_density_perturbation=delta_p,
        reservoir_density_perturbation=delta_r,
    )
    return background, closure


def test_constructed_generic_node_satisfies_every_scalar_constraint() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    receipt = FiniteQuenchScalarEinsteinConstraint(bridge).construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    expected = (
        receipt.gravity_constraint_coupling
        * (
            3.0 * receipt.total_momentum_density
            - receipt.total_density_perturbation
        )
        / receipt.k_over_a_h_squared
    )
    assert receipt.curvature_potential == pytest.approx(expected)
    assert receipt.lapse_potential == receipt.curvature_potential
    assert receipt.energy_constraint_holds
    assert receipt.momentum_constraint_holds
    assert receipt.zero_stress_traceless_spatial_constraint_holds
    assert receipt.combined_constraint_holds
    assert receipt.all_declared_scalar_constraints_hold
    assert not receipt.failure_reasons


def test_pure_positive_momentum_has_positive_curvature_in_vmm_convention() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge, delta_p=0.0, delta_r=0.0)
    receipt = FiniteQuenchScalarEinsteinConstraint(bridge).construct(
        background=background,
        closure=closure,
        k_over_a_h=3.0,
        produced_momentum_density=0.02,
        reservoir_momentum_density=0.01,
    )
    assert receipt.total_density_perturbation == 0.0
    assert receipt.total_momentum_density > 0.0
    assert receipt.curvature_potential > 0.0


def test_zero_momentum_positive_overdensity_has_negative_potential() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge, delta_p=0.2, delta_r=0.1)
    receipt = FiniteQuenchScalarEinsteinConstraint(bridge).construct(
        background=background,
        closure=closure,
        k_over_a_h=100.0,
        produced_momentum_density=0.0,
        reservoir_momentum_density=0.0,
    )
    assert receipt.total_density_perturbation > 0.0
    assert receipt.curvature_potential < 0.0


def test_old_wrong_u_sign_is_rejected_by_zero_i_and_combined_constraints() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge, delta_p=0.0, delta_r=0.0)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    kappa = 2.0
    momentum_p = 0.02
    momentum_r = 0.01
    total_u = momentum_p + momentum_r
    coupling = background.gravity_constraint_coupling
    wrong_psi = -coupling * (3.0 * total_u) / (kappa**2)
    wrong_phi = wrong_psi
    wrong_psi_prime = coupling * total_u - wrong_phi
    receipt = solver.audit(
        background=background,
        closure=closure,
        k_over_a_h=kappa,
        produced_momentum_density=momentum_p,
        reservoir_momentum_density=momentum_r,
        lapse_potential=wrong_phi,
        curvature_potential=wrong_psi,
        curvature_potential_log_derivative=wrong_psi_prime,
    )
    assert receipt.energy_constraint_holds
    assert not receipt.momentum_constraint_holds
    assert not receipt.combined_constraint_holds
    assert not receipt.all_declared_scalar_constraints_hold


def test_curvature_offset_can_keep_zero_i_but_fails_zero_zero() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    base = solver.construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    shifted_psi = base.curvature_potential + 0.1
    shifted_phi = shifted_psi
    shifted_psi_prime = (
        -base.gravity_constraint_coupling * base.total_momentum_density
        - shifted_phi
    )
    receipt = solver.audit(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
        lapse_potential=shifted_phi,
        curvature_potential=shifted_psi,
        curvature_potential_log_derivative=shifted_psi_prime,
    )
    assert receipt.momentum_constraint_holds
    assert receipt.zero_stress_traceless_spatial_constraint_holds
    assert not receipt.energy_constraint_holds
    assert not receipt.combined_constraint_holds


def test_unequal_metric_potentials_fail_only_zero_stress_traceless_gate() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    base = solver.construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    wrong_phi = base.lapse_potential + 0.1
    adjusted_psi_prime = (
        -base.gravity_constraint_coupling * base.total_momentum_density
        - wrong_phi
    )
    receipt = solver.audit(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
        lapse_potential=wrong_phi,
        curvature_potential=base.curvature_potential,
        curvature_potential_log_derivative=adjusted_psi_prime,
    )
    assert receipt.energy_constraint_holds
    assert receipt.momentum_constraint_holds
    assert receipt.combined_constraint_holds
    assert not receipt.zero_stress_traceless_spatial_constraint_holds


def test_forged_background_coupling_is_recomputed_not_trusted() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    base = solver.construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    forged = replace(
        background,
        gravity_constraint_coupling=(
            2.0 * background.gravity_constraint_coupling
        ),
        all_background_constraints_hold=True,
    )
    receipt = solver.audit(
        background=forged,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
        lapse_potential=base.lapse_potential,
        curvature_potential=base.curvature_potential,
        curvature_potential_log_derivative=(
            base.curvature_potential_log_derivative
        ),
    )
    assert not receipt.background_receipt_matches_bridge
    assert not receipt.all_declared_scalar_constraints_hold
    assert "BACKGROUND_RECEIPT_MISMATCH" in receipt.failure_reasons


def test_forged_closure_success_boolean_cannot_hide_wrong_pressure() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    base = solver.construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    forged = replace(
        closure,
        reservoir_pressure_perturbation=(
            closure.reservoir_pressure_perturbation + 0.5
        ),
        pressure_closure_holds=True,
        all_strict_barotropic_constraints_hold=True,
    )
    receipt = solver.audit(
        background=background,
        closure=forged,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
        lapse_potential=base.lapse_potential,
        curvature_potential=base.curvature_potential,
        curvature_potential_log_derivative=(
            base.curvature_potential_log_derivative
        ),
    )
    assert not receipt.closure_receipt_matches_bridge
    assert not receipt.all_declared_scalar_constraints_hold


def test_nonzero_anisotropic_stress_cannot_pass_zero_stress_metric_gate() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    solver = FiniteQuenchScalarEinsteinConstraint(bridge)
    base = solver.construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    nonzero_stress = replace(
        closure,
        reservoir_normalized_anisotropic_stress=0.1,
        zero_anisotropic_stress_holds=True,
        all_strict_barotropic_constraints_hold=True,
    )
    receipt = solver.audit(
        background=background,
        closure=nonzero_stress,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
        lapse_potential=base.lapse_potential,
        curvature_potential=base.curvature_potential,
        curvature_potential_log_derivative=(
            base.curvature_potential_log_derivative
        ),
    )
    assert receipt.total_normalized_anisotropic_stress == pytest.approx(0.1)
    assert not receipt.zero_stress_traceless_spatial_constraint_holds
    assert not receipt.all_declared_scalar_constraints_hold


def test_two_species_manifest_cannot_be_silently_extended() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    forged = replace(
        background,
        species_manifest=("produced", "reservoir", "omitted_species"),
        external_background_species_assumed_absent=False,
    )
    with pytest.raises(ValueError, match="background receipt"):
        FiniteQuenchScalarEinsteinConstraint(bridge).construct(
            background=forged,
            closure=closure,
            k_over_a_h=2.0,
            produced_momentum_density=0.03,
            reservoir_momentum_density=-0.01,
        )


def test_receipts_from_a_different_bridge_are_rejected() -> None:
    bridge = _bridge()
    other = _bridge(omega_prod0=0.2, reservoir_present_density=0.3)
    background, closure = _receipts(other)
    with pytest.raises(ValueError):
        FiniteQuenchScalarEinsteinConstraint(bridge).construct(
            background=background,
            closure=closure,
            k_over_a_h=2.0,
            produced_momentum_density=0.03,
            reservoir_momentum_density=-0.01,
        )


@pytest.mark.parametrize("n", [-5.0, -4.0, -3.0, 0.0])
def test_source_on_and_off_nodes_share_the_same_constraint_identity(n: float) -> None:
    bridge = _bridge()
    background, closure = _receipts(
        bridge,
        n=n,
        delta_p=0.01,
        delta_r=-0.02,
    )
    receipt = FiniteQuenchScalarEinsteinConstraint(bridge).construct(
        background=background,
        closure=closure,
        k_over_a_h=1.5,
        produced_momentum_density=0.01,
        reservoir_momentum_density=0.02,
    )
    assert receipt.all_declared_scalar_constraints_hold


@pytest.mark.parametrize(
    "kappa",
    [0.0, -1.0, True, math.nan, math.inf, 1.0e-308, 1.0e308],
)
def test_zero_or_nonfinite_or_nonrepresentable_kappa_is_rejected(
    kappa: object,
) -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    with pytest.raises(ValueError):
        FiniteQuenchScalarEinsteinConstraint(bridge).construct(
            background=background,
            closure=closure,
            k_over_a_h=kappa,
            produced_momentum_density=0.03,
            reservoir_momentum_density=-0.01,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("produced_momentum_density", math.nan),
        ("reservoir_momentum_density", math.inf),
        ("produced_momentum_density", True),
    ],
)
def test_momentum_inputs_fail_closed(field: str, value: object) -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    kwargs = dict(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        FiniteQuenchScalarEinsteinConstraint(bridge).construct(**kwargs)


def test_overflowing_total_perturbation_fails_closed() -> None:
    bridge = _bridge()
    background, closure = _receipts(
        bridge,
        delta_p=1.0e308,
        delta_r=1.0e308,
    )
    with pytest.raises(ValueError):
        FiniteQuenchScalarEinsteinConstraint(bridge).construct(
            background=background,
            closure=closure,
            k_over_a_h=2.0,
            produced_momentum_density=0.0,
            reservoir_momentum_density=0.0,
        )


def test_receipt_states_dimensionless_scope_and_nonintegration_role() -> None:
    bridge = _bridge()
    background, closure = _receipts(bridge)
    receipt = FiniteQuenchScalarEinsteinConstraint(bridge).construct(
        background=background,
        closure=closure,
        k_over_a_h=2.0,
        produced_momentum_density=0.03,
        reservoir_momentum_density=-0.01,
    )
    roles = dict(receipt.dimensionless_roles)
    assert roles["k_over_a_h"] == "dimensionless_wavenumber"
    assert roles["C"] == "four_pi_G_rho_unit_over_H_squared"
    assert receipt.velocity_convention.endswith("minus_k_squared_v")
    assert receipt.combined_constraint_is_derived_crosscheck
    assert receipt.role.endswith("NOT_PROPAGATED_OR_INTEGRATED_SOLUTION")
