from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.causal_quantum_domino import certify_causal_quantum_domino
from examples.physics.covariant_worldtube_energy_bridge import (
    audit_closed_branch_worldtube,
    audit_e9d_outcome_closed_branch_worldtube,
    audit_flat_quantum_kick_worldtube_receipt,
    construct_flat_charge_stress_kernel_counterexample,
    construct_flat_receipt_current_counterexample,
)
from examples.physics.finite_ctp_diagonal_source_obstruction import (
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
