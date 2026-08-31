from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.quantum_selection_dimension_dark_bridge import (
    audit_dimension_selection,
    audit_folded_opportunity_dark_sector,
    audit_mutual_execution,
    certify_quantum_selection_dimension_dark_bridge,
    main,
)


def test_reciprocal_execution_is_interventional_cptp_and_energy_conserving() -> None:
    angle = math.pi / 6.0
    audit = audit_mutual_execution(interaction_angle=angle, energy_gap=2.0)
    expected = math.sin(angle) ** 2

    assert audit.activation_probability == pytest.approx(expected)
    assert audit.nonactivation_probability == pytest.approx(1.0 - expected)
    assert audit.a_to_b_intervention_effect == pytest.approx(expected)
    assert audit.b_to_a_intervention_effect == pytest.approx(expected)
    assert audit.inactive_system_activations == pytest.approx((0.0, 0.0))
    assert audit.forward_system_activations == pytest.approx((1.0, expected))
    assert audit.reverse_system_activations == pytest.approx((expected, 1.0))
    assert audit.global_unitary
    assert audit.reduced_channel_cptp
    assert audit.energy_conserved
    assert audit.reciprocal_execution_certified
    assert audit.forward_reverse_output_overlap == pytest.approx(0.0, abs=1.0e-12)
    assert not audit.arbitrary_unknown_state_cloned


def test_no_interaction_has_no_mutual_execution_or_spontaneous_seed() -> None:
    audit = audit_mutual_execution(interaction_angle=0.0)

    assert audit.global_unitary
    assert audit.reduced_channel_cptp
    assert audit.energy_conserved
    assert audit.activation_probability == 0.0
    assert audit.a_to_b_intervention_effect == 0.0
    assert audit.b_to_a_intervention_effect == 0.0
    assert not audit.reciprocal_execution_certified
    assert audit.seed_and_battery_required


def test_full_exchange_activates_the_other_pointer_and_leaves_a_record() -> None:
    audit = audit_mutual_execution(interaction_angle=math.pi / 2.0)

    assert audit.forward_system_activations == pytest.approx((1.0, 1.0))
    assert audit.reverse_system_activations == pytest.approx((1.0, 1.0))
    assert audit.forward_battery_activations == pytest.approx((1.0, 0.0))
    assert audit.reverse_battery_activations == pytest.approx((0.0, 1.0))
    assert audit.forward_reverse_output_overlap == pytest.approx(0.0, abs=1.0e-12)


def test_dimension_selection_builds_nested_zero_one_two_three_filtration() -> None:
    audit = audit_dimension_selection()

    assert audit.projector_ranks == (0, 1, 2, 3)
    assert audit.rank_increments == (1, 1, 1)
    assert audit.gram_determinants == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert audit.exterior_norms_squared == audit.gram_determinants
    assert audit.selected_probabilities == pytest.approx((0.0, 0.2, 0.5, 1.0))
    assert audit.incremental_probabilities == pytest.approx((0.2, 0.3, 0.5))
    assert audit.maximum_projector_nesting_residual == pytest.approx(0.0)
    assert audit.independent_direction_records
    assert audit.cumulative_selection_monotone
    assert audit.dimensions_zero_through_three_certified
    assert audit.geometry_readout_is_adopted_axiom
    assert not audit.hilbert_rank_equals_spatial_dimension_without_readout_axiom


def test_dimension_certificate_is_invariant_under_common_orthogonal_rotation() -> None:
    angle = 0.37
    rotation = np.array(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    density = np.diag((0.2, 0.3, 0.5))
    rotated = audit_dimension_selection(
        direction_vectors=np.eye(3) @ rotation.T,
        direction_density=rotation @ density @ rotation.T,
    )

    assert rotated.gram_determinants == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert rotated.selected_probabilities == pytest.approx((0.0, 0.2, 0.5, 1.0))
    assert rotated.dimensions_zero_through_three_certified


def test_dependent_direction_or_nonstate_density_fails_closed() -> None:
    with pytest.raises(ValueError, match='linearly independent'):
        audit_dimension_selection(
            direction_vectors=((1.0, 0.0, 0.0),) * 3,
        )
    with pytest.raises(ValueError, match='unit trace'):
        audit_dimension_selection(direction_density=np.eye(3))


def test_folded_opportunity_routes_one_physical_receipt_without_double_counting() -> None:
    angle = math.pi / 6.0
    audit = audit_folded_opportunity_dark_sector(
        interaction_angle=angle,
        energy_gap=2.0,
    )
    mobile = math.sin(angle) ** 2
    locked = math.cos(angle) ** 2

    assert audit.mobile_probability == pytest.approx(mobile)
    assert audit.locked_probability == pytest.approx(locked)
    assert audit.nonselected_weighted_surprisal == pytest.approx(-locked * math.log(locked))
    assert audit.natural_cell_volume == pytest.approx(1.0 / 8.0)
    assert audit.dimensionless_volume_combination == pytest.approx(1.0)
    assert audit.total_folded_energy == pytest.approx(2.0)
    assert audit.mobile_dust_energy == pytest.approx(2.0 * mobile)
    assert audit.locked_vacuum_energy == pytest.approx(2.0 * locked)
    assert audit.physical_opportunity_energy == pytest.approx(2.0 * locked)
    assert audit.partition.dust_fraction == pytest.approx(mobile)
    assert audit.partition.vacuum_fraction == pytest.approx(locked)
    assert audit.partition.unassigned_energy == pytest.approx(0.0)
    assert audit.probability_normalization_residual == pytest.approx(0.0, abs=1.0e-12)
    assert audit.shared_energy_partition_residual == pytest.approx(0.0, abs=1.0e-12)
    assert not audit.probability_used_as_energy
    assert audit.physical_energy_receipt_supplies_scale
    assert audit.opportunity_cost_is_dimensionless_allocation_diagnostic
    assert audit.allocation_probabilities_derived_from_execution_unitary
    assert audit.one_receipt_no_double_counting_closed
    assert audit.conditional_dm_de_stress_forms_closed
    assert audit.ensemble_receipt_partition
    assert not audit.counterfactual_quantum_branch_dynamics_derived
    assert audit.flrw.continuity_equation_residual == pytest.approx(0.0, abs=1.0e-12)
    assert audit.flrw.friedmann_equation_residual == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize('angle', (0.0, math.pi / 2.0))
def test_two_component_dark_partition_requires_an_interior_probability(angle: float) -> None:
    with pytest.raises(ValueError, match=r'\(0, pi/2\)'):
        audit_folded_opportunity_dark_sector(interaction_angle=angle)


def test_joint_certificate_closes_only_the_declared_conditional_model() -> None:
    certificate = certify_quantum_selection_dimension_dark_bridge()

    assert certificate.all_three_conditional_claims_closed
    assert certificate.all_three_finite_witnesses_closed
    assert certificate.status == (
        'THREE_FINITE_CONDITIONAL_WITNESSES_CLOSED_PHYSICAL_MAPS_OPEN'
    )
    assert certificate.dimension_selection_controls_dark_partition
    assert certificate.interaction_angle_derived_from_dimension_selection
    assert certificate.dimension_to_dark_probability_residual == pytest.approx(0.0)
    assert certificate.dimension_derived_mobile_probability == pytest.approx(0.5)
    assert certificate.dimension_derived_locked_probability == pytest.approx(0.5)
    assert certificate.claim_1_nonselected_dark_status.startswith('CONDITIONAL')
    assert certificate.claim_2_dimension_implementation_status.startswith('CONDITIONAL')
    assert certificate.claim_3_mutual_execution_status.startswith('FINITE')
    assert not certificate.unconditional_standard_qm_to_real_cosmology_proved
    assert not certificate.all_three_user_claims_unconditionally_proved
    assert len(certificate.required_adopted_axioms) == 5
    assert all(reason for _, reason in certificate.dimensionless_arguments)


def test_dimension_weights_drive_the_same_execution_and_dark_probabilities() -> None:
    certificate = certify_quantum_selection_dimension_dark_bridge(
        direction_density=np.diag((0.1, 0.2, 0.7)),
    )

    assert certificate.dimension_selection.selected_probabilities == pytest.approx(
        (0.0, 0.1, 0.3, 1.0)
    )
    assert certificate.dimension_derived_mobile_probability == pytest.approx(0.3)
    assert certificate.dimension_derived_locked_probability == pytest.approx(0.7)
    assert certificate.mutual_execution.activation_probability == pytest.approx(0.3)
    assert certificate.dark_readout.mobile_probability == pytest.approx(0.3)
    assert certificate.dark_readout.locked_probability == pytest.approx(0.7)
    assert certificate.dark_readout.partition.dust_fraction == pytest.approx(0.3)
    assert certificate.dark_readout.partition.vacuum_fraction == pytest.approx(0.7)
    assert certificate.dimension_selection_controls_dark_partition


def test_explicit_mismatched_execution_angle_fails_the_joint_link() -> None:
    certificate = certify_quantum_selection_dimension_dark_bridge(
        interaction_angle=math.pi / 6.0,
    )

    assert certificate.mutual_execution.reciprocal_execution_certified
    assert certificate.dark_readout.conditional_dm_de_stress_forms_closed
    assert not certificate.dimension_selection_controls_dark_partition
    assert not certificate.all_three_conditional_claims_closed
    assert certificate.status == 'JOINT_FINITE_WITNESS_OR_PROBABILITY_LINK_AUDIT_FAILED'


def test_cli_emits_machine_readable_certificate(capsys) -> None:
    assert main(['--pretty']) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload['all_three_conditional_claims_closed']
    assert payload['mutual_execution']['reduced_channel_cptp']
    assert payload['dimension_selection']['projector_ranks'] == [0, 1, 2, 3]
    assert payload['dark_readout']['one_receipt_no_double_counting_closed']
