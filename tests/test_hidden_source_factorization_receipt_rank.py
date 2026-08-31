from __future__ import annotations

import json

import numpy as np
import pytest

from examples.physics.contextual_global_section_obstruction import (
    QUANTUM_ETA,
    marginal_incidence_matrix,
    quantum_kernel_perturbed_extension,
    symmetric_signed_global_extension,
)
from examples.physics.hidden_source_factorization_receipt_rank import (
    certificate,
    combined_readout_rank,
    factor_linear_source,
    linear_source_factorization_residual,
    linear_source_kernel_residual,
    receipt_kernel_rank,
    reconstruct_from_visible_and_walsh,
    run,
    visible_and_walsh_receipt,
    walsh_receipt_matrix,
)
from examples.physics.representation_invariant_measure_bridge import (
    normalized_atom_tangent_basis,
)


def test_walsh_rows_are_an_exact_orthogonal_basis_of_the_hidden_kernel() -> None:
    incidence = marginal_incidence_matrix()
    walsh = walsh_receipt_matrix()

    assert incidence.shape == (16, 16)
    assert walsh.shape == (7, 16)
    assert np.array_equal(incidence @ walsh.T, np.zeros((16, 7), dtype=int))
    assert np.array_equal(walsh @ walsh.T, 16 * np.eye(7, dtype=int))
    result = certificate()
    assert result.incidence_rank == 9
    assert result.incidence_nullity == 7
    assert result.walsh_rank == 7


def test_one_six_and_seven_receipts_give_the_exact_rank_ladder() -> None:
    walsh = walsh_receipt_matrix()
    result = certificate()

    assert combined_readout_rank(walsh[:1]) == 10
    assert combined_readout_rank(walsh[:6]) == 15
    assert combined_readout_rank(walsh) == 16
    assert receipt_kernel_rank(walsh[:1]) == 1
    assert receipt_kernel_rank(walsh[:6]) == 6
    assert receipt_kernel_rank(walsh) == 7
    assert result.minimum_receipt_rows_for_full_recovery == 7
    assert result.status["receipt_rank_lower_bound_witness_certified"]


def test_normalized_tangent_has_eight_visible_plus_seven_hidden_directions() -> None:
    incidence = marginal_incidence_matrix()
    walsh = walsh_receipt_matrix()
    tangent = normalized_atom_tangent_basis()

    assert tangent.shape == (16, 15)
    assert np.allclose(np.sum(tangent, axis=0), 0.0)
    assert np.linalg.matrix_rank(incidence @ tangent) == 8
    assert combined_readout_rank(walsh[:1], normalized_tangent=True) == 9
    assert combined_readout_rank(walsh[:6], normalized_tangent=True) == 14
    assert combined_readout_rank(walsh, normalized_tangent=True) == 15


def test_linear_source_factors_exactly_when_it_annihilates_the_hidden_kernel() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    visible_source = incidence[[0, 5]]
    factor = factor_linear_source(visible_source)

    assert linear_source_kernel_residual(visible_source) == pytest.approx(0.0)
    assert linear_source_factorization_residual(visible_source) < 1.0e-12
    assert np.allclose(factor @ incidence, visible_source, atol=1.0e-12, rtol=0.0)

    hidden_source = walsh_receipt_matrix()[[0]]
    assert linear_source_kernel_residual(hidden_source) == pytest.approx(16.0)
    assert linear_source_factorization_residual(hidden_source) > 0.5
    with pytest.raises(ValueError, match="not constant"):
        factor_linear_source(hidden_source)


def test_ambient_factor_extension_is_not_unique() -> None:
    result = certificate()

    assert result.visible_source_factorization_residual < 1.0e-12
    assert result.alternative_ambient_factor_residual < 1.0e-12
    assert result.ambient_factor_extension_difference > 0.1
    assert result.status["ambient_factor_extension_nonuniqueness_certified"]


def test_permutation_invariant_norm_is_not_constant_on_visible_fibres() -> None:
    base = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    shifted = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_visible, _ = visible_and_walsh_receipt(base)
    shifted_visible, _ = visible_and_walsh_receipt(shifted)
    result = certificate()

    assert np.allclose(base_visible, shifted_visible, atol=1.0e-12, rtol=0.0)
    assert float(np.dot(shifted, shifted)) - float(np.dot(base, base)) == pytest.approx(
        0.000625
    )
    assert result.permutation_norm_residual < 1.0e-12
    assert result.same_fibre_norm_square_difference == pytest.approx(0.000625)
    assert result.status["permutation_covariance_not_fibre_invariance_certified"]


def test_walsh_receipts_detect_q_delta_and_reconstruct_the_full_coordinate() -> None:
    coordinates = np.asarray(quantum_kernel_perturbed_extension(0.1))
    visible, receipt = visible_and_walsh_receipt(coordinates)
    reconstructed = reconstruct_from_visible_and_walsh(visible, receipt)
    result = certificate()

    assert np.allclose(reconstructed, coordinates, atol=1.0e-12, rtol=0.0)
    assert result.q_delta_visible_residual < 1.0e-12
    assert result.q_delta_first_walsh_change == pytest.approx(0.1)
    assert result.q_delta_other_walsh_residual < 1.0e-12
    assert result.status["full_walsh_coordinate_reconstruction_certified"]


def test_visible_duplicate_receipts_add_no_hidden_rank_or_new_source() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    duplicate = incidence[:7]
    result = certificate()

    assert receipt_kernel_rank(duplicate) == 0
    assert combined_readout_rank(duplicate) == 9
    assert result.accounting["duplicate_receipt_factors_through_visible_map"]
    assert result.accounting["duplicate_receipt_not_added_as_new_source"]
    assert result.status["duplicate_visible_receipt_adds_no_rank_certified"]


def test_general_relabel_covariance_is_not_fixed_map_automorphism() -> None:
    result = certificate()

    assert result.relabel_visible_residual < 1.0e-12
    assert result.relabel_receipt_residual < 1.0e-12
    assert result.relabel_combined_rank == 16
    assert result.relabel_fixed_incidence_residual == pytest.approx(1.0)
    assert result.relabel_fixed_receipt_residual == pytest.approx(2.0)
    assert result.status["general_relabel_covariance_certified"]
    assert not result.status["chosen_general_relabel_is_fixed_map_automorphism"]


def test_dimension_accounting_alternatives_and_physical_claim_ceiling() -> None:
    result = certificate()

    assert all(result.dimensions.values())
    assert not result.accounting["receipt_probability_energy_or_volume_double_counted"]
    assert all(result.alternatives.values())
    assert result.boundaries["seven_rows_are_necessary_only_for_full_linear_q_recovery"]
    assert result.boundaries["seven_is_not_a_gravity_component_field_or_boson_count"]
    assert not result.status["physical_walsh_receipt_derived"]
    assert not result.status["hidden_signed_coordinate_is_physical_state_derived"]
    assert not result.status["local_covariant_action_or_stress_derived"]
    assert not result.status["spacetime_metric_curvature_or_gravity_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed_and_serializes() -> None:
    incidence = marginal_incidence_matrix().astype(np.float64)
    visible, receipt = visible_and_walsh_receipt(np.full(16, 1.0 / 16.0))

    with pytest.raises(ValueError, match="sixteen"):
        linear_source_kernel_residual(np.ones(15))
    with pytest.raises(ValueError, match="finite"):
        receipt_kernel_rank(np.full((1, 16), np.nan))
    with pytest.raises(ValueError, match="seven"):
        reconstruct_from_visible_and_walsh(visible, receipt[:6])
    inconsistent = np.array(visible, copy=True)
    inconsistent[0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        reconstruct_from_visible_and_walsh(inconsistent, receipt)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)

    payload = run()
    json.dumps(payload)
    json.loads(certificate().to_json())
    assert payload["combined_rank_seven_receipts"] == 16
    assert np.array_equal(incidence @ walsh_receipt_matrix().T, np.zeros((16, 7)))
