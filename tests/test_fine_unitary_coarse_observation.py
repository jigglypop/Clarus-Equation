from __future__ import annotations

import json

import numpy as np
import pytest

from examples.physics.fine_unitary_coarse_observation import (
    apparatus_zero_embedding,
    certificate,
    coarse_visibility_labels,
    controlled_record_inverse,
    controlled_record_unitary,
    fine_visibility_labels,
    partial_trace_apparatus,
    projective_dephasing,
    record_isometry,
    record_kraus_operators,
    run,
    selective_update,
)


@pytest.mark.parametrize("dimension", [3, 4])
def test_controlled_shift_is_a_bijective_unitary_with_explicit_inverse(
    dimension: int,
) -> None:
    unitary = controlled_record_unitary(dimension)
    inverse = controlled_record_inverse(dimension)
    identity = np.eye(dimension * dimension)
    assert np.allclose(unitary.conj().T @ unitary, identity)
    assert np.allclose(unitary @ unitary.conj().T, identity)
    assert np.allclose(inverse, unitary.conj().T)
    assert np.allclose(inverse @ unitary, identity)


def test_record_restriction_is_isometric_and_bijective_only_onto_its_image() -> None:
    dimension = 3
    isometry = record_isometry(dimension)
    assert isometry.shape == (dimension * dimension, dimension)
    assert np.linalg.matrix_rank(isometry) == dimension
    assert np.allclose(isometry.conj().T @ isometry, np.eye(dimension))
    assert np.allclose(
        isometry,
        controlled_record_unitary(dimension) @ apparatus_zero_embedding(dimension),
    )
    result = certificate(dimension=dimension)
    assert result.record_output_residual < 1.0e-12
    assert result.inverse_recovery_residual < 1.0e-12
    assert result.status["record_isometry_bijective_onto_its_image"]
    assert not result.status["record_isometry_surjective_onto_full_joint_space"]


def test_partial_trace_is_the_cptp_dephasing_channel() -> None:
    result = certificate()
    assert result.reduced_system_residual < 1.0e-12
    assert result.reduced_apparatus_residual < 1.0e-12
    assert result.kraus_completeness_residual < 1.0e-12
    assert result.kraus_channel_residual < 1.0e-12
    assert result.choi_minimum_eigenvalue >= -1.0e-12
    assert result.status["explicit_projective_record_channel_cptp"]

    operators = record_kraus_operators(3)
    assert len(operators) == 3
    assert all(np.allclose(operator, np.diag(np.eye(3)[index])) for index, operator in enumerate(operators))


def test_fine_sort_is_bijective_while_coarse_readout_forgets_hidden_identity() -> None:
    fine = fine_visibility_labels(3, 1)
    coarse = coarse_visibility_labels(3, 1)
    assert fine == (("hidden", 0), ("visible", 1), ("hidden", 2))
    assert len(set(fine)) == 3
    assert len(set(coarse)) == 2

    # Boundary case: with only one hidden label the binary readout happens to
    # remain injective; this does not make selective state update injective.
    assert len(set(coarse_visibility_labels(2, 0))) == 2
    assert certificate(dimension=2, selected=0).status[
        "coarse_visibility_readout_injective"
    ]
    assert not certificate().status["coarse_visibility_readout_injective"]


def test_selective_and_nonselective_state_updates_are_many_to_one() -> None:
    result = certificate()
    assert result.distinct_input_residual > 1.0e-3
    assert result.nonselective_collision_residual < 1.0e-12
    assert result.selective_operation_collision_residual < 1.0e-12
    assert result.selective_posterior_collision_residual < 1.0e-12
    assert result.status["nonselective_dephasing_many_to_one_witness"]
    assert result.status["selective_update_many_to_one_witness"]


def test_probability_accounting_and_claim_ceiling_are_explicit() -> None:
    result = certificate()
    assert result.status["declared_finite_controlled_unitary_bijective"]
    assert result.status["fine_discrete_label_sort_bijective_onto_declared_image"]
    assert result.status[
        "fine_discrete_label_bijection_onto_image_is_homeomorphism"
    ]
    assert result.status["premeasurement_components_preserved_by_fine_unitary"]
    assert all(result.dimensions.values())
    assert result.accounting["branch_probabilities_sum_to_one"]
    assert result.accounting["visible_plus_hidden_probability_sum_to_one"]
    assert result.accounting[
        "hidden_labels_retained_individually_in_fine_label_map"
    ]
    assert not result.accounting["coarse_and_fine_probabilities_added_as_separate_energy"]
    assert result.boundaries["selected_label_is_not_an_input_to_controlled_unitary"]
    assert result.boundaries[
        "finite_w_is_declared_model_not_actual_universe_dynamics"
    ]
    assert result.boundaries["fine_sort_is_label_only_not_physical_branch_dynamics"]
    assert result.boundaries[
        "fine_sort_codomain_is_declared_image_not_full_cartesian_product"
    ]
    assert result.boundaries["finite_label_topology_declared_discrete"]
    assert result.boundaries[
        "finite_dimension_is_hilbert_label_dimension_not_spacetime_dimension"
    ]
    assert result.boundaries["cptp_claim_is_for_the_explicit_projective_instrument"]
    assert not result.status["unitary_selects_one_unique_actual_outcome"]
    assert not result.status["durable_physical_pointer_derived"]
    assert not result.status["energy_hamiltonian_or_transfer_derived"]
    assert not result.status["spacetime_homeomorphism_derived"]
    assert not result.status["spacetime_metric_or_curvature_derived"]
    assert not result.status["fold_stress_or_gravity_derived"]
    assert not result.status["success_gates_5_to_8_complete"]


def test_public_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        controlled_record_unitary(1)
    with pytest.raises(ValueError, match="at least two"):
        controlled_record_inverse(True)
    with pytest.raises(ValueError, match="record range"):
        fine_visibility_labels(3, 3)
    with pytest.raises(ValueError, match="positive probability"):
        selective_update(np.diag([1.0, 0.0]), 1)
    with pytest.raises(ValueError, match="unit trace"):
        projective_dephasing(np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        partial_trace_apparatus(np.eye(3) / 3.0, 2)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)


def test_run_payload_is_json_serializable_and_keeps_status_ceiling() -> None:
    payload = run()
    json.dumps(payload)
    assert payload["status"]["declared_finite_controlled_unitary_bijective"]
    assert not payload["status"]["unitary_selects_one_unique_actual_outcome"]
