"""covariant_energy_bridge 의 세계관 정합·두 스칼라 교환 전류·물질 격자 비용 회귀검사."""

from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal.causal_domino import certify_causal_quantum_domino
from examples.physics.record.covariant_energy_bridge import (
    audit_closed_branch_worldtube,
    audit_e9d_outcome_closed_branch_worldtube,
    audit_flat_quantum_kick_worldtube_receipt,
    certify_allocation_nonidentifiability,
    certify_covariant_material_lattice_cost,
    construct_flat_charge_stress_kernel_counterexample,
    construct_flat_receipt_current_counterexample,
    two_scalar_exchange_receipt,
)
from examples.physics.record.finite_ctp_diagonal_source_obstruction import (
    audit_quantum_kick_conservation,
)


MINKOWSKI = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
REST_OBSERVER = (1.0, 0.0, 0.0, 0.0)
ZERO_TENSOR = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
)


def _diagonal_stress(energy_density: float) -> tuple[tuple[float, ...], ...]:
    return (
        (energy_density, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    )


def _base_inputs() -> dict[str, object]:
    return {
        "source_receipt_id": "QNB-E9-D:00",
        "branch_probability": 0.16,
        "conditional_trace": 1.0,
        "receipt_energy": 6.0,
        "metrics_covariant": (MINKOWSKI,),
        "orientation_observers_contravariant": (REST_OBSERVER,),
        "time_flows_contravariant": (REST_OBSERVER,),
        "exchange_currents_system_covariant": ((-6.0, 0.0, 0.0, 0.0),),
        "exchange_currents_battery_covariant": ((6.0, 0.0, 0.0, 0.0),),
        "system_stresses_contravariant": (_diagonal_stress(9.0),),
        "battery_stresses_contravariant": (_diagonal_stress(3.0),),
        "symmetrized_time_flow_gradients_covariant": (ZERO_TENSOR,),
        "proper_four_volume_weights": (1.0,),
        "system_initial_surface_energy": 3.0,
        "system_final_surface_energy": 9.0,
        "battery_initial_surface_energy": 6.0,
        "battery_final_surface_energy": 0.0,
        "system_lateral_outward_energy_flux": 0.0,
        "battery_lateral_outward_energy_flux": 0.0,
        "reference_mass_scale": 1.0,
    }


def test_positive_domino_outcome_matches_supplied_killing_worldtube() -> None:
    domino = certify_causal_quantum_domino(
        site_count=4,
        depth=2,
        theta=math.asin(math.sqrt(0.4)),
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        energy_gap=3.0,
    )
    outcome = {
        item.basis_label: item for item in domino.battery_outcomes
    }["00"]
    inputs = _base_inputs()
    for key in (
        "source_receipt_id",
        "branch_probability",
        "conditional_trace",
        "receipt_energy",
        "system_initial_surface_energy",
        "system_final_surface_energy",
        "battery_initial_surface_energy",
        "battery_final_surface_energy",
    ):
        del inputs[key]

    receipt = audit_e9d_outcome_closed_branch_worldtube(
        outcome=outcome,
        initial_system_energy=domino.initial_system_energy,
        **inputs,
    )

    assert receipt.source_receipt_id == "QNB-E9-D:00"
    assert receipt.branch_probability == pytest.approx(0.16)
    assert receipt.receipt_energy == pytest.approx(6.0)
    assert receipt.system_source_injection_energy == pytest.approx(6.0)
    assert receipt.battery_source_injection_energy == pytest.approx(-6.0)
    assert receipt.system_surface_energy_change == pytest.approx(6.0)
    assert receipt.battery_surface_energy_change == pytest.approx(-6.0)
    assert receipt.system_deformation_energy == pytest.approx(0.0)
    assert receipt.battery_deformation_energy == pytest.approx(0.0)
    assert receipt.dimensionless_system_balance_residual < 1.0e-12
    assert receipt.dimensionless_battery_balance_residual < 1.0e-12
    assert receipt.dimensionless_total_balance_residual < 1.0e-12
    assert receipt.dimensionless_exchange_cancellation_residual < 1.0e-12
    assert receipt.maximum_dimensionless_opposite_current_residual < 1.0e-12
    assert receipt.dimensionless_receipt_worldtube_residual < 1.0e-12
    assert receipt.maximum_dimensionless_killing_equation_residual < 1.0e-12
    assert receipt.dimensions_pass
    assert receipt.current_mass_dimension == 5
    assert receipt.stress_mass_dimension == 4
    assert receipt.four_volume_mass_dimension == -4
    assert receipt.energy_mass_dimension == 1
    assert receipt.normalized_residual_mass_dimension == 0
    assert receipt.positive_probability_outcome
    assert receipt.conditional_branch_normalized
    assert receipt.supplied_time_flow_future_timelike
    assert receipt.supplied_killing_flow_on_samples
    assert receipt.supplied_zero_lateral_flux
    assert receipt.opposite_exchange_current_cancels
    assert receipt.supplied_sector_balances_hold
    assert receipt.supplied_total_energy_balance_holds
    assert receipt.supplied_total_energy_and_exchange_closure_holds
    assert receipt.exclusive_branch_receipt_matches_both_sectors
    assert receipt.killing_zero_flux_receipt_matching_holds
    assert not receipt.source_receipt_id_is_provenance_label_only
    assert receipt.typed_e9d_outcome_consistency_verified
    assert not receipt.source_receipt_provenance_authenticated
    assert not receipt.e9d_receipt_to_worldtube_derived
    assert receipt.quadrature_worldtube_supplied
    assert receipt.opposite_sector_current_supplied
    assert not receipt.branch_stress_from_domino_derived
    assert not receipt.battery_to_covariant_action_derived
    assert not receipt.continuum_worldtube_derived
    assert not receipt.physical_pointer_derived
    assert not receipt.record_to_gravity_source_derived


def test_lateral_flux_keeps_general_balance_but_blocks_simple_bulk_matching() -> None:
    inputs = _base_inputs()
    inputs.update(
        {
            "exchange_currents_system_covariant": ((-10.0, 0.0, 0.0, 0.0),),
            "exchange_currents_battery_covariant": ((10.0, 0.0, 0.0, 0.0),),
            "system_initial_surface_energy": 0.0,
            "system_final_surface_energy": 6.0,
            "battery_initial_surface_energy": 10.0,
            "battery_final_surface_energy": 4.0,
            "system_lateral_outward_energy_flux": 4.0,
            "battery_lateral_outward_energy_flux": -4.0,
        }
    )

    receipt = audit_closed_branch_worldtube(**inputs)

    assert receipt.system_source_injection_energy == pytest.approx(10.0)
    assert receipt.system_predicted_surface_energy_change == pytest.approx(6.0)
    assert receipt.battery_predicted_surface_energy_change == pytest.approx(-6.0)
    assert receipt.dimensionless_system_balance_residual < 1.0e-12
    assert receipt.dimensionless_battery_balance_residual < 1.0e-12
    assert receipt.dimensionless_receipt_worldtube_residual < 1.0e-12
    assert receipt.exclusive_branch_receipt_matches_both_sectors
    assert not receipt.supplied_zero_lateral_flux
    assert not receipt.killing_zero_flux_receipt_matching_holds


def test_non_killing_deformation_is_not_mislabelled_as_exchange_energy() -> None:
    deformation = np.zeros((4, 4), dtype=float)
    deformation[0, 0] = -0.2
    inputs = _base_inputs()
    inputs.update(
        {
            "branch_probability": 1.0,
            "receipt_energy": 0.0,
            "time_flows_contravariant": ((1.2, 0.0, 0.0, 0.0),),
            "exchange_currents_system_covariant": ((0.0, 0.0, 0.0, 0.0),),
            "exchange_currents_battery_covariant": ((0.0, 0.0, 0.0, 0.0),),
            "system_stresses_contravariant": (_diagonal_stress(2.0),),
            "battery_stresses_contravariant": (ZERO_TENSOR,),
            "symmetrized_time_flow_gradients_covariant": (deformation,),
            "proper_four_volume_weights": (6.0,),
            "system_initial_surface_energy": 6.0,
            "system_final_surface_energy": 8.4,
            "battery_initial_surface_energy": 0.0,
            "battery_final_surface_energy": 0.0,
        }
    )

    receipt = audit_closed_branch_worldtube(**inputs)

    assert receipt.system_source_injection_energy == pytest.approx(0.0)
    assert receipt.system_deformation_energy == pytest.approx(2.4)
    assert receipt.system_surface_energy_change == pytest.approx(2.4)
    assert receipt.dimensionless_system_balance_residual < 1.0e-12
    assert receipt.dimensionless_total_balance_residual < 1.0e-12
    assert receipt.maximum_dimensionless_killing_equation_residual == pytest.approx(0.2)
    assert receipt.supplied_sector_balances_hold
    assert receipt.supplied_total_energy_balance_holds
    assert receipt.supplied_total_energy_and_exchange_closure_holds
    assert not receipt.supplied_killing_flow_on_samples
    assert not receipt.exclusive_branch_receipt_matches_both_sectors
    assert not receipt.killing_zero_flux_receipt_matching_holds


def test_nonopposite_momentum_current_fails_total_exchange_closure() -> None:
    inputs = _base_inputs()
    inputs["exchange_currents_battery_covariant"] = ((6.0, 1.0, 0.0, 0.0),)

    receipt = audit_closed_branch_worldtube(**inputs)

    assert receipt.dimensionless_exchange_cancellation_residual == pytest.approx(0.0)
    assert receipt.maximum_dimensionless_opposite_current_residual == pytest.approx(1.0)
    assert receipt.supplied_sector_balances_hold
    assert receipt.supplied_total_energy_balance_holds
    assert not receipt.supplied_total_energy_and_exchange_closure_holds
    assert not receipt.opposite_exchange_current_cancels
    assert not receipt.killing_zero_flux_receipt_matching_holds


def test_conserved_quantum_four_kick_matches_a_supplied_ward_worldtube() -> None:
    identity = np.eye(2, dtype=complex)
    occupation = np.diag((0.0, 1.0)).astype(complex)
    system_energy = np.kron(occupation, identity)
    environment_energy = np.kron(identity, occupation)
    zero_operator = np.zeros_like(system_energy)
    sector_operators = np.stack(
        (
            np.stack((system_energy, zero_operator, zero_operator, zero_operator)),
            np.stack((environment_energy, zero_operator, zero_operator, zero_operator)),
        )
    )
    swap = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=complex,
    )
    initial_vector = np.array((0.0, 0.0, 1.0, 0.0), dtype=complex)
    quantum = audit_quantum_kick_conservation(
        swap,
        np.outer(initial_vector, initial_vector.conj()),
        sector_operators,
        all_receivers_included=True,
    )

    bridge = audit_flat_quantum_kick_worldtube_receipt(
        kick_audit=quantum,
        exchange_currents_covariant=(
            ((1.0, 0.0, 0.0, 0.0),),
            ((-1.0, 0.0, 0.0, 0.0),),
        ),
        proper_four_volume_weights=(1.0,),
        lateral_outward_four_momentum_fluxes=(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        same_local_action_identification_supplied=True,
        shared_inertial_four_vector_basis_supplied=True,
        reference_mass_scale=1.0,
    )

    assert np.asarray(bridge.quantum_mean_kicks) == pytest.approx(
        np.asarray(((-1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    )
    assert np.asarray(bridge.integrated_exchange_four_momenta) == pytest.approx(
        np.asarray(bridge.quantum_mean_kicks)
    )
    assert np.asarray(bridge.predicted_worldtube_kicks) == pytest.approx(
        np.asarray(bridge.quantum_mean_kicks)
    )
    assert bridge.maximum_dimensionless_sector_matching_residual < 1.0e-12
    assert bridge.maximum_dimensionless_local_exchange_residual < 1.0e-12
    assert bridge.dimensionless_integrated_exchange_residual < 1.0e-12
    assert bridge.dimensionless_total_quantum_kick_residual < 1.0e-12
    assert bridge.dimensions_pass
    assert bridge.quantum_operator_conservation_certified
    assert bridge.all_receivers_included
    assert bridge.local_exchange_currents_cancel
    assert bridge.integrated_exchange_cancels
    assert bridge.numerical_integrated_worldtube_matching_holds
    assert bridge.conditional_quantum_to_worldtube_bridge_holds
    assert bridge.worldtube_localization_supplied
    assert bridge.lateral_flux_supplied
    assert not bridge.operator_components_as_physical_four_vector_derived
    assert not bridge.exchange_current_from_quantum_dynamics_derived
    assert not bridge.local_stress_from_quantum_kick_derived
    assert not bridge.general_curved_spacetime_transport_derived
    assert not bridge.physical_clarus_source_derived

    ward = audit_closed_branch_worldtube(
        source_receipt_id="E41-kick:conditional-flat-worldtube",
        branch_probability=1.0,
        conditional_trace=1.0,
        receipt_energy=1.0,
        metrics_covariant=(MINKOWSKI,),
        orientation_observers_contravariant=(REST_OBSERVER,),
        time_flows_contravariant=(REST_OBSERVER,),
        exchange_currents_system_covariant=((-1.0, 0.0, 0.0, 0.0),),
        exchange_currents_battery_covariant=((1.0, 0.0, 0.0, 0.0),),
        system_stresses_contravariant=(_diagonal_stress(1.0),),
        battery_stresses_contravariant=(_diagonal_stress(0.0),),
        symmetrized_time_flow_gradients_covariant=(ZERO_TENSOR,),
        proper_four_volume_weights=(1.0,),
        system_initial_surface_energy=0.0,
        system_final_surface_energy=1.0,
        battery_initial_surface_energy=1.0,
        battery_final_surface_energy=0.0,
        system_lateral_outward_energy_flux=0.0,
        battery_lateral_outward_energy_flux=0.0,
        reference_mass_scale=1.0,
    )
    assert ward.killing_zero_flux_receipt_matching_holds
    assert not ward.branch_stress_from_domino_derived
    assert not ward.record_to_gravity_source_derived

    contract_withheld = audit_flat_quantum_kick_worldtube_receipt(
        kick_audit=quantum,
        exchange_currents_covariant=(
            ((1.0, 0.0, 0.0, 0.0),),
            ((-1.0, 0.0, 0.0, 0.0),),
        ),
        proper_four_volume_weights=(1.0,),
        lateral_outward_four_momentum_fluxes=np.zeros((2, 4)),
        same_local_action_identification_supplied=False,
        shared_inertial_four_vector_basis_supplied=True,
        reference_mass_scale=1.0,
    )
    assert contract_withheld.numerical_integrated_worldtube_matching_holds
    assert not contract_withheld.conditional_quantum_to_worldtube_bridge_holds


def test_complete_four_charges_have_a_nontrivial_local_stress_kernel() -> None:
    certificate = construct_flat_charge_stress_kernel_counterexample(
        spatial_volume=3.0,
        energy_density=2.0,
        shear_amplitude=0.5,
        reference_mass_scale=1.0,
    )

    assert certificate.profile_a_four_momentum == pytest.approx((6.0, 0.0, 0.0, 0.0))
    assert certificate.profile_b_four_momentum == pytest.approx(
        certificate.profile_a_four_momentum
    )
    assert certificate.profile_a_stress_contravariant != (
        certificate.profile_b_stress_contravariant
    )
    assert certificate.dimensionless_four_momentum_residual == 0.0
    assert certificate.dimensionless_local_stress_difference == pytest.approx(0.5)
    assert certificate.maximum_dimensionless_ward_residual == 0.0
    assert certificate.dimensions_pass
    assert certificate.both_profiles_symmetric
    assert certificate.both_profiles_divergence_free
    assert certificate.both_profiles_satisfy_dominant_energy_condition
    assert certificate.same_complete_four_momentum
    assert certificate.local_stresses_distinct
    assert certificate.finite_charge_to_local_stress_nonuniqueness_certified
    assert certificate.periodic_spatial_cell_supplied
    assert not certificate.local_action_for_profiles_derived
    assert not certificate.local_stress_selected_by_finite_charges
    assert not certificate.cosmological_perturbations_selected_by_background_charges


def test_one_scalar_receipt_does_not_select_the_momentum_current() -> None:
    certificate = construct_flat_receipt_current_counterexample(
        receipt_energy=6.0,
        duration=2.0,
        spatial_volume=3.0,
        momentum_source_a=0.0,
        momentum_source_b=4.0,
        reference_mass_scale=1.0,
    )

    assert certificate.four_volume == pytest.approx(6.0)
    assert certificate.energy_source_density == pytest.approx(1.0)
    assert certificate.profile_a_current_covector == pytest.approx((-1.0, 0.0, 0.0, 0.0))
    assert certificate.profile_b_current_covector == pytest.approx((-1.0, 4.0, 0.0, 0.0))
    assert certificate.profile_a_battery_current_covector == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert certificate.profile_b_battery_current_covector == pytest.approx((1.0, -4.0, 0.0, 0.0))
    assert certificate.profile_a_computed_system_divergence_covector == pytest.approx(
        certificate.profile_a_current_covector
    )
    assert certificate.profile_b_computed_system_divergence_covector == pytest.approx(
        certificate.profile_b_current_covector
    )
    assert certificate.profile_a_computed_battery_divergence_covector == pytest.approx(
        certificate.profile_a_battery_current_covector
    )
    assert certificate.profile_b_computed_battery_divergence_covector == pytest.approx(
        certificate.profile_b_battery_current_covector
    )
    assert certificate.profile_a_integrated_energy == pytest.approx(6.0)
    assert certificate.profile_b_integrated_energy == pytest.approx(6.0)
    assert certificate.dimensionless_current_difference == pytest.approx(4.0)
    assert certificate.complement_constant_energy_density == pytest.approx(4.0)
    assert certificate.minimum_complement_energy_density == pytest.approx(2.0)
    assert certificate.maximum_dimensionless_divergence_identity_residual < 1.0e-12
    assert certificate.maximum_dimensionless_total_divergence_residual < 1.0e-12
    assert certificate.maximum_dimensionless_lateral_energy_flux_density < 1.0e-12
    assert certificate.dimensions_pass
    assert certificate.same_flat_worldtube
    assert certificate.same_scalar_receipt
    assert certificate.current_profiles_distinct
    assert certificate.lateral_energy_flux_zero
    assert certificate.opposite_sector_closes_total_stress
    assert certificate.unique_current_from_receipt_claim_refuted
    assert not certificate.worldtube_selected_by_receipt
    assert not certificate.branch_stress_from_receipt_derived
    assert not certificate.covariant_action_from_receipt_derived
    assert not certificate.record_to_gravity_source_derived


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("branch_probability", 1.1, "branch_probability"),
        ("metrics_covariant", (np.eye(4),), "signature"),
        (
            "time_flows_contravariant",
            ((0.0, 1.0, 0.0, 0.0),),
            "future timelike",
        ),
        ("proper_four_volume_weights", (0.0,), "positive"),
        (
            "system_stresses_contravariant",
            (
                (
                    (1.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                ),
            ),
            "symmetric",
        ),
    ],
)
def test_invalid_worldtube_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = _base_inputs()
    inputs[field] = value

    with pytest.raises((ValueError, ArithmeticError), match=message):
        audit_closed_branch_worldtube(**inputs)


def _on_shell_inputs(coupling: float = 0.3) -> dict[str, object]:
    phi = 0.8
    psi = -0.4
    mass_phi = 1.2
    mass_psi = 0.7
    interaction_d_phi = coupling * phi * psi * psi
    interaction_d_psi = coupling * phi * phi * psi
    return {
        "metric_covariant": MINKOWSKI,
        "observer_contravariant": REST_OBSERVER,
        "phi": phi,
        "psi": psi,
        "gradient_phi_covector": (0.5, 0.0, 0.0, 0.0),
        "gradient_psi_covector": (-0.6, 0.0, 0.0, 0.0),
        "box_phi": mass_phi * mass_phi * phi + interaction_d_phi,
        "box_psi": mass_psi * mass_psi * psi + interaction_d_psi,
        "mass_phi": mass_phi,
        "mass_psi": mass_psi,
        "coupling": coupling,
        "reference_mass_scale": 1.0,
    }


def test_on_shell_split_current_closes_total_ward_identity() -> None:
    receipt = two_scalar_exchange_receipt(
        **_on_shell_inputs(),
        allocation_fraction=0.37,
    )

    assert receipt.interaction_energy_density == pytest.approx(0.01536)
    assert receipt.exchange_current_phi_covector == pytest.approx(
        (-0.0049536, 0.0, 0.0, 0.0)
    )
    assert receipt.exchange_current_psi_covector == pytest.approx(
        (0.0049536, 0.0, 0.0, 0.0)
    )
    assert receipt.phi_sector_divergence_covector == pytest.approx(
        receipt.exchange_current_phi_covector
    )
    assert receipt.psi_sector_divergence_covector == pytest.approx(
        receipt.exchange_current_psi_covector
    )
    assert receipt.dimensionless_eom_residual < 1.0e-12
    assert receipt.dimensionless_interaction_allocation_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence < 1.0e-12
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_complementarity_residual < 1.0e-12
    assert receipt.metric_signature == (-1, 1, 1, 1)
    assert receipt.dimensions_pass
    assert receipt.interaction_energy_counted_once
    assert receipt.field_mass_dimension == 1
    assert receipt.interaction_mass_dimension == 4
    assert receipt.current_mass_dimension == 5
    assert receipt.normalized_residual_mass_dimension == 0
    assert receipt.on_shell_within_tolerance
    assert receipt.total_stress_conserved_on_shell
    assert receipt.covariant_action_exchange_current_derived
    assert not receipt.interaction_allocation_dynamically_selected
    assert not receipt.domino_receipt_to_action_derived
    assert not receipt.covariant_matching_current_derived
    assert not receipt.record_to_gravity_source_derived


def test_same_action_and_interaction_density_do_not_select_a_unique_current() -> None:
    certificate = certify_allocation_nonidentifiability(**_on_shell_inputs())

    assert certificate.alpha_zero_receipt.interaction_energy_density == pytest.approx(
        certificate.alpha_one_receipt.interaction_energy_density
    )
    assert certificate.alpha_zero_receipt.exchange_current_phi_covector == pytest.approx(
        (0.0192, 0.0, 0.0, 0.0)
    )
    assert certificate.alpha_one_receipt.exchange_current_phi_covector == pytest.approx(
        (-0.04608, 0.0, 0.0, 0.0)
    )
    assert certificate.dimensionless_interaction_density_difference == pytest.approx(0.0)
    assert certificate.dimensionless_current_difference == pytest.approx(0.06528)
    assert (
        certificate.dimensionless_total_interaction_allocation_difference
        == pytest.approx(0.0)
    )
    assert certificate.same_action_and_interaction_density
    assert certificate.currents_distinct
    assert certificate.total_stress_alpha_invariant
    assert certificate.unique_current_claim_refuted
    assert certificate.supplied_allocation_required
    assert not certificate.domino_receipt_to_action_derived
    assert not certificate.physical_source_derived


def test_zero_coupling_recovers_decoupled_exchange_limit() -> None:
    receipt = two_scalar_exchange_receipt(
        **_on_shell_inputs(coupling=0.0),
        allocation_fraction=0.61,
    )

    assert receipt.interaction_energy_density == 0.0
    assert receipt.exchange_current_phi_covector == (0.0, 0.0, 0.0, 0.0)
    assert receipt.exchange_current_psi_covector == (0.0, 0.0, 0.0, 0.0)
    assert receipt.zero_coupling_exchange_vanishes
    assert receipt.total_stress_conserved_on_shell


def test_off_shell_input_keeps_ward_identity_but_not_conservation_claim() -> None:
    inputs = _on_shell_inputs()
    inputs["box_phi"] = float(inputs["box_phi"]) + 0.1
    receipt = two_scalar_exchange_receipt(
        **inputs,
        allocation_fraction=0.37,
    )

    assert receipt.dimensionless_eom_residual == pytest.approx(0.1)
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence > 0.0
    assert not receipt.on_shell_within_tolerance
    assert not receipt.total_stress_conserved_on_shell
    with pytest.raises(ValueError, match="on-shell"):
        certify_allocation_nonidentifiability(**inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allocation_fraction", 1.1, "allocation_fraction"),
        ("coupling", -0.1, "coupling"),
        ("observer_contravariant", (2.0, 0.0, 0.0, 0.0), "unit timelike"),
        ("metric_covariant", np.eye(4), "Lorentzian signature"),
    ],
)
def test_invalid_action_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = _on_shell_inputs()
    inputs[field] = value
    inputs.setdefault("allocation_fraction", 0.5)

    with pytest.raises((ValueError, ArithmeticError), match=message):
        two_scalar_exchange_receipt(**inputs)


def _certificate(**overrides):
    arguments = dict(
        cells_per_axis=4,
        proper_cell_spacing=2.0,
        rod_scale=1.5,
        battery_energy_per_cell=0.75,
        carrier_mass=2.0,
        carrier_momentum=1.25,
        onsite_exchange_coupling=-0.4,
        quartic_coupling=0.4,
        guide_well_mass_squared=0.2,
        cell_well_mass_squared=0.0,
    )
    arguments.update(overrides)
    return certify_covariant_material_lattice_cost(**arguments)


def test_dimensions_geometry_and_supplied_winding_are_closed():
    certificate = _certificate()
    q = math.pi

    assert certificate.action_terms_have_mass_dimension_four
    assert certificate.dimensionless_core_arguments == (
        ("q a = 2 pi", "compact phase winding"),
        ("q L / (2 pi) = N", "supplied integer winding"),
        ("v_g = |k| / sqrt(|k|^2 + m_H^2)", "free-particle sample"),
    )
    assert certificate.compact_phase_period_is_two_pi
    assert certificate.clock_field_used is False
    assert certificate.wave_number == pytest.approx(q)
    assert certificate.material_gram_diagonal == pytest.approx((q**2,) * 3)
    assert certificate.material_gram_determinant == pytest.approx(q**6)
    assert certificate.normalized_gram_determinant == pytest.approx(1.0)
    assert certificate.proper_cell_volume == pytest.approx(2.0**3)
    assert certificate.winding_per_axis == pytest.approx(4.0)
    assert not certificate.spacing_action_winding_derived


def test_free_rod_stress_and_finite_energy_receipt_are_separate():
    certificate = _certificate()
    expected_rho = 1.5 * 1.5**2 * math.pi**2

    assert certificate.rod_energy_density == pytest.approx(expected_rho)
    assert certificate.rod_pressure == pytest.approx(-expected_rho / 3.0)
    assert certificate.rod_equation_of_state == pytest.approx(-1.0 / 3.0)
    assert certificate.finite_rod_energy == pytest.approx(expected_rho * 8.0**3)
    assert certificate.finite_rod_receipt
    assert certificate.supplied_finite_free_rod_background_bookkeeping


def test_guide_and_volume_battery_capacities_are_not_the_rod_ledger():
    certificate = _certificate()

    assert certificate.guide_all_success_battery_count == 4
    assert certificate.guide_battery_capacity == pytest.approx(4 * 0.75)
    assert certificate.full_volume_cell_count == 4**3
    assert certificate.full_volume_battery_capacity == pytest.approx(4**3 * 0.75)
    assert certificate.rod_and_battery_ledgers_kept_separate
    assert certificate.finite_rod_energy == pytest.approx(
        1.5 * 1.5**2 * math.pi**2 * 8.0**3
    )


def test_exact_onsite_quartic_bound_and_extremal_witness():
    certificate = _certificate()

    assert certificate.quartic_lower_bound_coefficient == pytest.approx(0.0)
    assert certificate.extremal_quartic_potential == pytest.approx(0.0)
    assert certificate.quartic_saturation_residual < 1.0e-14
    strict = _certificate(quartic_coupling=0.9)
    assert strict.quartic_lower_bound_coefficient == pytest.approx((0.9 - 0.4) / 4.0)
    assert strict.extremal_quartic_potential == pytest.approx(
        strict.quartic_lower_bound_coefficient
    )
    assert strict.quartic_saturation_residual < 1.0e-14


def test_below_quartic_bound_fails_closed():
    with pytest.raises(ValueError, match="quartic stability"):
        _certificate(quartic_coupling=0.399)


def test_free_group_speed_has_a_ceiling_but_is_not_a_front_proof():
    certificate = _certificate()

    assert 0.0 <= certificate.carrier_group_velocity < 1.0
    assert certificate.canonical_fixed_background_classical_principal_symbol
    assert certificate.fixed_background_classical_domain_of_dependence
    assert not certificate.band_or_front_speed_derived


def test_unproved_claims_remain_false_and_invalid_inputs_fail():
    certificate = _certificate()
    assert certificate.diffeomorphism_covariant_scalar_candidate_by_construction
    assert certificate.static_common_coupling_without_coordinate_time_schedule
    assert not any(
        (
            certificate.interacting_backreacted_theta_solution_derived,
            certificate.background_stability_or_caustic_freedom_derived,
            certificate.periodic_well_localized_modes_derived,
            certificate.action_to_projected_rates_or_resonance_derived,
            certificate.scattering_energy_transfer_receipt_derived,
            certificate.durable_record_or_selection_derived,
            certificate.repeated_cptp_fresh_ancilla_derived,
            certificate.qft_microcausality_or_no_signalling_derived,
            certificate.coupled_gr_source_derived,
            certificate.infinite_isolated_lattice_finite_total_energy_derived,
            certificate.gates_five_to_eight_derived,
        )
    )

    for overrides in (
        {"cells_per_axis": 0},
        {"cells_per_axis": 1.5},
        {"proper_cell_spacing": 0.0},
        {"onsite_exchange_coupling": 0.0},
        {"guide_well_mass_squared": -0.1},
    ):
        with pytest.raises(ValueError):
            _certificate(**overrides)
