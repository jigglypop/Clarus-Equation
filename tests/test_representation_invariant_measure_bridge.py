from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.chsh_local_seed_obstruction import isotropic_chsh_box
from examples.physics.contextual_global_section_obstruction import (
    QUANTUM_ETA,
    marginal_incidence_matrix,
    marginalize_global_weights,
    quantum_kernel_perturbed_extension,
    swap_opposite_score_weights,
    symmetric_signed_global_extension,
    walsh_kernel_vectors,
)
from examples.physics.representation_invariant_measure_bridge import (
    UNIFORM_CONTEXT_WEIGHTS,
    atom_permutation_matrix,
    certificate,
    conditional_conformal_metric,
    conditional_fisher_quadratic,
    context_block_permutation,
    fisher_pullback_metric,
    hellinger_coordinates,
    hellinger_tangent,
    high_frequency_volume_witness,
    isotropic_fisher_component,
    isotropic_fisher_distance,
    lorentzian_signature,
    matrix_inertia,
    metric_volume_ratio,
    normalized_atom_tangent_basis,
    product_fisher_rao_distance,
    run,
)


def _single_context_tangent() -> np.ndarray:
    tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    tangent[0, 0] = np.asarray(((0.25, -0.25), (-0.25, 0.25)))
    return tangent


def test_context_weighted_fisher_and_hellinger_factor_count_each_context_once() -> None:
    box = isotropic_chsh_box(0.0)
    tangent = _single_context_tangent()
    coordinates = hellinger_coordinates(box)
    differential = hellinger_tangent(box, tangent)
    quadratic = conditional_fisher_quadratic(box, tangent)

    assert float(np.sum(coordinates * coordinates)) == pytest.approx(4.0)
    assert quadratic == pytest.approx(0.25)
    assert float(np.sum(differential * differential)) == pytest.approx(quadratic)


def test_pullback_has_exact_incidence_rank_and_same_seven_kernel_directions() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    incidence = marginal_incidence_matrix().astype(np.float64)
    metric = fisher_pullback_metric(target, incidence=incidence)

    assert matrix_inertia(metric) == (9, 0, 7)
    assert np.allclose(metric, metric.T, atol=1.0e-12, rtol=0.0)
    for vector in walsh_kernel_vectors().values():
        direction = np.asarray(vector, dtype=np.float64)
        assert np.array_equal(incidence @ direction, np.zeros(16))
        assert np.max(np.abs(metric @ direction)) < 1.0e-12


def test_normalized_global_tangent_quotient_has_rank_eight() -> None:
    metric = fisher_pullback_metric(isotropic_chsh_box(QUANTUM_ETA))
    basis = normalized_atom_tangent_basis()
    restricted = basis.T @ metric @ basis

    assert basis.shape == (16, 15)
    assert np.allclose(np.sum(basis, axis=0), 0.0)
    assert matrix_inertia(restricted) == (8, 0, 7)


def test_signed_kernel_lifts_have_identical_visible_box_and_fisher_pullback() -> None:
    base_q = symmetric_signed_global_extension(QUANTUM_ETA)
    shifted_q = quantum_kernel_perturbed_extension(0.1)
    base_box = marginalize_global_weights(base_q)
    shifted_box = marginalize_global_weights(shifted_q)

    assert base_q != shifted_q
    assert np.allclose(base_box, shifted_box, atol=1.0e-12, rtol=0.0)
    assert np.allclose(
        fisher_pullback_metric(base_box),
        fisher_pullback_metric(shifted_box),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_simultaneous_coordinate_relabel_is_a_metric_congruence() -> None:
    q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    incidence = marginal_incidence_matrix().astype(np.float64)
    box = marginalize_global_weights(q)
    metric = fisher_pullback_metric(box, incidence=incidence)
    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    atom_permutation = atom_permutation_matrix(tuple(reversed(range(16))))

    relabelled_q = atom_permutation @ q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_box = (row_permutation @ box.reshape(-1)).reshape(2, 2, 2, 2)
    relabelled_weights = tuple(
        UNIFORM_CONTEXT_WEIGHTS[index] for index in context_order
    )
    relabelled_metric = fisher_pullback_metric(
        relabelled_box,
        incidence=relabelled_incidence,
        context_weights=relabelled_weights,
    )

    assert np.allclose(
        relabelled_incidence @ relabelled_q,
        relabelled_box.reshape(-1),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert np.allclose(
        relabelled_metric,
        atom_permutation @ metric @ atom_permutation.T,
        atol=1.0e-12,
        rtol=0.0,
    )
    result = certificate()
    assert result.general_relabel_fixed_incidence_residual > 0.5
    assert result.status["chosen_general_relabel_is_not_fixed_incidence_automorphism"]


def test_context_design_must_cotransform_unless_uniform_symmetry_is_assumed() -> None:
    result = certificate()
    target = isotropic_chsh_box(QUANTUM_ETA)
    atom_only = marginalize_global_weights(
        swap_opposite_score_weights(symmetric_signed_global_extension(QUANTUM_ETA))
    )

    assert result.fixed_nonuniform_context_swap_residual > 0.1
    assert result.co_transformed_context_swap_residual < 1.0e-12
    assert result.uniform_context_swap_residual < 1.0e-12
    assert not np.allclose(atom_only, target, atol=1.0e-12, rtol=0.0)
    assert result.atom_only_probability_residual > 0.1
    assert result.atom_only_fixed_incidence_residual > 0.5
    assert result.status["atom_only_fixed_incidence_automorphism_excluded"]


def test_isotropic_line_has_arcsine_coordinate_and_completion_boundary() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    origin = isotropic_chsh_box(0.0)

    assert isotropic_fisher_component(QUANTUM_ETA) == pytest.approx(2.0)
    assert isotropic_fisher_distance(0.0, QUANTUM_ETA) == pytest.approx(math.pi / 4.0)
    assert product_fisher_rao_distance(origin, target) == pytest.approx(math.pi / 4.0)
    assert isotropic_fisher_component(0.999) > 500.0
    with pytest.raises(ValueError, match="strict chart"):
        isotropic_fisher_component(1.0)
    with pytest.raises(ValueError, match="positive"):
        product_fisher_rao_distance(origin, isotropic_chsh_box(1.0))


def test_fisher_psd_does_not_supply_a_lorentzian_signature() -> None:
    result = certificate()

    assert result.pullback_inertia == (9, 0, 7)
    assert result.normalized_tangent_inertia == (8, 0, 7)
    assert result.status["fisher_form_positive_semidefinite"]
    assert not result.status["fisher_metric_is_spacetime_lorentz_metric_derived"]
    assert not result.status["lorentzian_signature_or_lightcone_derived_from_fisher"]
    assert result.boundaries["fisher_psd_no_go_is_not_a_general_lorentz_geometry_no_go"]


def test_supplied_lorentz_metric_and_volume_ratio_only_fix_conformal_control() -> None:
    reference = np.diag((-1.0, 1.0, 1.0, 1.0))
    metric = conditional_conformal_metric(reference, 16.0)
    null = np.asarray((1.0, 1.0, 0.0, 0.0))

    assert np.allclose(metric, 4.0 * reference)
    assert lorentzian_signature(metric) == lorentzian_signature(reference) == (3, 1, 0)
    assert metric_volume_ratio(metric, reference) == pytest.approx(16.0)
    assert float(null @ reference @ null) == pytest.approx(0.0)
    assert float(null @ metric @ null) == pytest.approx(0.0)
    assert np.allclose(conditional_conformal_metric(reference, 1.0), reference)


def test_uniform_volume_convergence_does_not_control_second_derivatives() -> None:
    small = high_frequency_volume_witness(10)
    large = high_frequency_volume_witness(100)

    assert small.minimum_volume_ratio > 0.0
    assert large.uniform_value_residual_bound < small.uniform_value_residual_bound
    assert large.probe_value_residual == pytest.approx(1.0e-4)
    assert abs(large.probe_first_derivative) < 1.0e-12
    assert large.probe_second_derivative == pytest.approx(-10000.0)


def test_dimension_accounting_alternatives_and_claim_ceiling_are_explicit() -> None:
    result = certificate()

    assert all(result.dimensions.values())
    assert result.accounting["context_weights_sum_to_one"]
    assert result.accounting["each_context_counted_once_not_once_per_outcome_cell"]
    assert not result.accounting["probability_energy_or_volume_double_counted"]
    assert all(result.alternatives.values())
    assert not result.status["physical_volume_law_derived"]
    assert not result.status["curvature_einstein_dynamics_or_gravity_derived"]
    assert not result.status["gr_c2_limit_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed_and_serializes() -> None:
    box = isotropic_chsh_box(0.0)
    tangent = _single_context_tangent()

    with pytest.raises(ValueError, match="four finite positive"):
        hellinger_coordinates(box, context_weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        hellinger_coordinates(box, context_weights=(0.4, 0.3, 0.2, 0.2))
    with pytest.raises(ValueError, match="positive"):
        hellinger_coordinates(isotropic_chsh_box(1.0))
    with pytest.raises(ValueError, match="zero sum"):
        conditional_fisher_quadratic(box, np.ones_like(tangent))
    with pytest.raises(ValueError, match="sixteen"):
        fisher_pullback_metric(box, incidence=np.zeros((15, 16)))
    with pytest.raises(ValueError, match="symmetric"):
        matrix_inertia(np.asarray(((1.0, 1.0), (0.0, 1.0))))
    with pytest.raises(ValueError, match="permutation"):
        context_block_permutation((0, 0, 2, 3))
    with pytest.raises(ValueError, match="permutation"):
        atom_permutation_matrix(tuple(range(15)))
    with pytest.raises(ValueError, match="Lorentzian"):
        conditional_conformal_metric(np.eye(4), 1.0)
    with pytest.raises(ValueError, match="positive"):
        conditional_conformal_metric(np.diag((-1.0, 1.0)), 0.0)
    with pytest.raises(ValueError, match="determinant"):
        metric_volume_ratio(np.eye(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        metric_volume_ratio(np.asarray(((1.0, 1.0), (0.0, 1.0))), np.eye(2))
    with pytest.raises(ValueError, match="determinant"):
        metric_volume_ratio(np.diag((1.0, 0.0)), np.eye(2))
    with pytest.raises(ValueError, match="integer"):
        high_frequency_volume_witness(True)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)

    result = certificate()
    payload = run()
    json.dumps(payload)
    json.loads(result.to_json())
    assert payload["incidence_rank"] == 9
    assert payload["normalized_tangent_rank"] == 8
