from collections import Counter
import json
import math

import numpy as np
import pytest

from examples.physics.chsh_local_seed_obstruction import (
    audit_probability_box,
    box_correlations,
    chsh_scores,
    isotropic_chsh_box,
)
from examples.physics.contextual_global_section_obstruction import (
    ContextLedgerCoordinate,
    QUANTUM_ETA,
    certificate,
    context_cells,
    deterministic_oriented_scores,
    exact_rational_rank,
    global_assignments,
    invert_context_ledger,
    lift_context_ledger,
    marginal_incidence_matrix,
    marginalize_global_weights,
    negative_mass,
    normalized_absolute_weights,
    quantum_kernel_perturbed_extension,
    run,
    swap_opposite_score_weights,
    symmetric_signed_global_extension,
    total_variation_norm,
    walsh_kernel_vectors,
)


def test_full_visible_plus_hidden_ledger_is_bijective_but_readout_is_not() -> None:
    assignments = global_assignments()

    assert len(assignments) == 16
    for x in (0, 1):
        for y in (0, 1):
            coordinates = tuple(
                lift_context_ledger(assignment, x, y) for assignment in assignments
            )
            assert len(set(coordinates)) == 16
            assert tuple(map(invert_context_ledger, coordinates)) == assignments
            visible_counts = Counter(item.visible_outcomes for item in coordinates)
            assert set(visible_counts.values()) == {4}
            assert len(visible_counts) == 4

    result = certificate()
    assert result.status["full_context_ledger_set_bijection_certified"]
    assert result.status["visible_projection_many_to_one_certified"]
    assert result.boundaries[
        "finite_discrete_full_ledger_bijection_is_homeomorphism"
    ]
    assert result.boundaries["positive_global_measure_failure_is_not_bijection_failure"]


def test_symmetric_signed_extension_has_exact_two_weight_levels() -> None:
    scores = deterministic_oriented_scores()
    weights = symmetric_signed_global_extension(QUANTUM_ETA)
    expected_negative = (1.0 - math.sqrt(2.0)) / 16.0
    expected_positive = (1.0 + math.sqrt(2.0)) / 16.0

    assert set(scores) == {-2, 2}
    assert scores.count(-2) == scores.count(2) == 8
    assert math.fsum(weights) == pytest.approx(1.0)
    assert min(weights) == pytest.approx(expected_negative)
    assert max(weights) == pytest.approx(expected_positive)
    assert sum(value < 0.0 for value in weights) == 8
    assert sum(value > 0.0 for value in weights) == 8

    for eta in (0.0, 0.3, 0.5, QUANTUM_ETA, 1.0):
        swept = symmetric_signed_global_extension(eta)
        assert np.allclose(
            marginalize_global_weights(swept),
            isotropic_chsh_box(eta),
            atol=1.0e-12,
            rtol=0.0,
        )
        assert total_variation_norm(swept) == pytest.approx(max(1.0, 2.0 * eta))
        assert all(value >= -1.0e-15 for value in swept) is (eta <= 0.5)


def test_signed_extension_reconstructs_every_positive_born_context() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    reconstructed = marginalize_global_weights(
        symmetric_signed_global_extension(QUANTUM_ETA)
    )
    target_audit = audit_probability_box(target)
    result = certificate()

    assert np.allclose(reconstructed, target, atol=1.0e-12, rtol=0.0)
    assert target_audit.minimum_probability > 0.0
    assert target_audit.maximum_normalization_residual < 1.0e-12
    assert target_audit.maximum_no_signalling_residual < 1.0e-12
    assert result.parent_instrument_probability_residual < 1.0e-12
    assert result.status["fixed_context_parent_instruments_remain_cptp"]
    assert result.status["finite_target_operational_no_signalling_certified"]


def test_positive_global_probability_is_excluded_without_excluding_bijection() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    oriented_score, absolute_score = chsh_scores(target)
    result = certificate()

    assert all(abs(score) == 2 for score in deterministic_oriented_scores())
    assert oriented_score == pytest.approx(2.0 * math.sqrt(2.0))
    assert absolute_score == pytest.approx(2.0 * math.sqrt(2.0))
    assert oriented_score > 2.0
    assert result.status[
        "positive_setting_independent_global_probability_excluded_for_target"
    ]
    assert result.status["full_context_ledger_set_bijection_certified"]
    assert not result.status["physical_hidden_path_or_seed_law_derived"]


def test_incidence_rank_nine_has_seven_exact_walsh_kernel_directions() -> None:
    incidence = marginal_incidence_matrix()
    kernel = walsh_kernel_vectors()
    result = certificate()

    assert incidence.shape == (16, 16)
    assert len(context_cells()) == 16
    assert exact_rational_rank(incidence) == 9
    assert len(kernel) == 7
    for vector in kernel.values():
        assert np.array_equal(incidence @ np.asarray(vector), np.zeros(16, dtype=int))
    assert result.incidence_rank == 9
    assert result.incidence_nullity == 7
    assert result.maximum_walsh_kernel_residual == 0


def test_kernel_perturbation_proves_minimizer_is_not_unique() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    base = symmetric_signed_global_extension(QUANTUM_ETA)
    half_width = math.sqrt(2.0) - 1.0

    for delta in (-half_width, -0.5 * half_width, 0.5 * half_width, half_width):
        perturbed = quantum_kernel_perturbed_extension(delta)
        assert perturbed != base
        assert np.allclose(
            marginalize_global_weights(perturbed), target, atol=1.0e-12, rtol=0.0
        )
        assert total_variation_norm(perturbed) == pytest.approx(math.sqrt(2.0))
        assert negative_mass(perturbed) == pytest.approx(
            (math.sqrt(2.0) - 1.0) / 2.0
        )

    endpoints = (
        quantum_kernel_perturbed_extension(-half_width),
        quantum_kernel_perturbed_extension(half_width),
    )
    assert min(abs(value) for item in endpoints for value in item) < 1.0e-15
    for sign in (-1.0, 1.0):
        assert total_variation_norm(
            quantum_kernel_perturbed_extension(sign * 1.1 * half_width)
        ) > math.sqrt(2.0)
    result = certificate()
    assert result.minimum_beyond_interval_l1_excess > 0.0
    assert result.status["minimum_signed_extension_is_nonunique"]


def test_chsh_duality_bound_is_saturated_by_minimum_signed_negativity() -> None:
    weights = symmetric_signed_global_extension(QUANTUM_ETA)
    scores = deterministic_oriented_scores()
    signed_expectation = math.fsum(
        weight * score for weight, score in zip(weights, scores)
    )
    norm = total_variation_norm(weights)
    result = certificate()

    assert signed_expectation == pytest.approx(2.0 * math.sqrt(2.0))
    assert abs(signed_expectation) <= 2.0 * norm + 1.0e-12
    assert abs(signed_expectation) == pytest.approx(2.0 * norm)
    assert norm == pytest.approx(math.sqrt(2.0))
    assert negative_mass(weights) == pytest.approx((math.sqrt(2.0) - 1.0) / 2.0)
    assert result.signed_l1_saturation_residual < 1.0e-12
    assert result.signed_negative_mass == pytest.approx(
        result.signed_negative_mass_lower_bound
    )
    assert result.status["minimum_signed_l1_and_negativity_certified"]


def test_normalized_absolute_replacement_changes_born_marginals() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    signed = symmetric_signed_global_extension(QUANTUM_ETA)
    positive = normalized_absolute_weights(signed)
    replacement = marginalize_global_weights(positive)
    target_correlations = np.asarray(box_correlations(target))
    replacement_correlations = np.asarray(box_correlations(replacement))
    replacement_score, _ = chsh_scores(replacement)
    result = certificate()

    assert total_variation_norm(signed) == pytest.approx(math.sqrt(2.0))
    assert math.fsum(positive) == pytest.approx(1.0)
    assert min(positive) > 0.0
    assert np.allclose(
        replacement_correlations, 0.5 * target_correlations, atol=1.0e-12
    )
    assert replacement_score == pytest.approx(math.sqrt(2.0))
    assert not np.allclose(replacement, target, atol=1.0e-12, rtol=0.0)
    assert result.normalized_absolute_target_residual > 1.0e-3
    assert result.status["symmetric_absolute_replacement_changes_born_marginals"]
    assert result.boundaries[
        "absolute_value_result_uses_symmetric_delta_zero_representative"
    ]


def test_arbitrary_atom_bijection_preserves_norms_but_not_incidence() -> None:
    target = isotropic_chsh_box(QUANTUM_ETA)
    signed = symmetric_signed_global_extension(QUANTUM_ETA)
    permuted = swap_opposite_score_weights(signed)
    result = certificate()

    assert math.fsum(permuted) == pytest.approx(math.fsum(signed))
    assert total_variation_norm(permuted) == pytest.approx(total_variation_norm(signed))
    assert negative_mass(permuted) == pytest.approx(negative_mass(signed))
    assert not np.allclose(
        marginalize_global_weights(permuted), target, atol=1.0e-12, rtol=0.0
    )
    assert result.status[
        "arbitrary_atom_bijection_need_not_preserve_physical_marginals"
    ]
    assert result.boundaries[
        "atom_permutation_preserves_marginals_only_if_incidence_is_respected"
    ]


def test_dimension_accounting_claim_ceiling_and_alternatives_remain_explicit() -> None:
    result = certificate()

    assert all(result.dimensions.values())
    assert result.accounting["each_context_born_box_normalized_once"]
    assert result.accounting["full_ledger_relabels_each_atom_once"]
    assert result.accounting["signed_extension_is_an_alternative_linear_representation"]
    assert result.accounting["signed_and_absolute_models_are_not_added_together"]
    assert result.accounting["signed_weight_not_added_as_energy_or_stress"]
    assert not result.accounting["signed_or_hidden_atom_carries_energy"]
    assert result.boundaries["signed_weight_is_not_observed_probability_or_frequency"]
    assert result.boundaries["signed_weight_is_not_negative_energy_or_stress"]
    assert result.boundaries[
        "absolute_value_result_is_not_general_metric_measure_or_gravity_no_go"
    ]
    assert result.boundaries[
        "finite_discrete_zero_dimensionality_is_not_spacetime_dimension"
    ]
    assert all(result.alternatives.values())
    assert not result.status["objective_single_outcome_selection_derived"]
    assert not result.status["relativistic_qft_microcausality_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["spacetime_metric_volume_or_gravity_derived"]
    assert not result.status["mass_dependent_probability_deformation_derived"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed_and_run_is_json_serializable() -> None:
    with pytest.raises(ValueError, match="assignment"):
        lift_context_ledger((1, 1, 1), 0, 0)
    with pytest.raises(ValueError, match="x"):
        lift_context_ledger((1, 1, 1, 1), 2, 0)
    with pytest.raises(TypeError, match="ContextLedgerCoordinate"):
        invert_context_ledger(((0, 0), (1, 1), (-1, -1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eta"):
        symmetric_signed_global_extension(1.1)
    with pytest.raises(ValueError, match="sixteen"):
        marginalize_global_weights((1.0,))
    with pytest.raises(ValueError, match="nonempty"):
        exact_rational_rank(np.empty((0, 0)))
    with pytest.raises(ValueError, match="positive total mass"):
        normalized_absolute_weights((0.0,) * 16)
    with pytest.raises(ValueError, match="delta"):
        quantum_kernel_perturbed_extension(math.nan)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)

    coordinate = ContextLedgerCoordinate((0, 1), (1, -1), (-1, 1))
    assert lift_context_ledger(invert_context_ledger(coordinate), 0, 1) == coordinate
    payload = run()
    json.dumps(payload)
    assert payload["atom_count"] == 16
    assert payload["incidence_rank"] == 9
    assert payload["status"]["full_context_ledger_set_bijection_certified"]
