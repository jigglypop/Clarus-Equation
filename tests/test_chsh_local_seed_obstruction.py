from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.chsh_local_seed_obstruction import (
    OUTCOMES,
    audit_probability_box,
    box_correlations,
    certificate,
    chsh_scores,
    deterministic_facet_score,
    deterministic_local_strategies,
    deterministic_mixture_box,
    invert_seed_coordinate,
    isotropic_chsh_box,
    lift_seed_coordinate,
    local_boundary_strategies,
    quantum_projective_instrument_audit,
    run,
    usual_coproduct_seed_lift_is_homeomorphism,
)


def test_singlet_projectors_reproduce_the_pr_oriented_quantum_box() -> None:
    audit = quantum_projective_instrument_audit()
    expected = isotropic_chsh_box(1.0 / math.sqrt(2.0))

    assert np.allclose(audit.probabilities, expected, atol=1.0e-12)
    assert box_correlations(audit.probabilities) == pytest.approx(
        (-1.0 / math.sqrt(2.0),) * 3 + (1.0 / math.sqrt(2.0),)
    )
    facet_score, absolute_chsh = chsh_scores(audit.probabilities)
    assert facet_score == pytest.approx(2.0 * math.sqrt(2.0))
    assert absolute_chsh == pytest.approx(2.0 * math.sqrt(2.0))


def test_each_fixed_setting_instrument_is_cptp_and_posteriors_normalize() -> None:
    audit = quantum_projective_instrument_audit()

    assert audit.maximum_projector_residual < 1.0e-12
    assert audit.maximum_completeness_residual < 1.0e-12
    assert audit.minimum_choi_eigenvalue > -1.0e-12
    assert audit.maximum_posterior_trace_residual < 1.0e-12


def test_quantum_and_pr_boxes_are_operationally_no_signalling() -> None:
    quantum = audit_probability_box(isotropic_chsh_box(1.0 / math.sqrt(2.0)))
    pr = audit_probability_box(isotropic_chsh_box(1.0))

    for audit in (quantum, pr):
        assert audit.minimum_probability >= 0.0
        assert audit.maximum_normalization_residual < 1.0e-12
        assert audit.maximum_no_signalling_residual < 1.0e-12
        assert audit.maximum_unbiased_marginal_residual < 1.0e-12


def test_all_sixteen_local_deterministic_strategies_obey_chsh_two() -> None:
    strategies = deterministic_local_strategies()
    scores = tuple(deterministic_facet_score(strategy) for strategy in strategies)

    assert len(strategies) == 16
    assert set(scores) == {-2, 2}
    assert all(abs(score) == 2 for score in scores)
    assert 2.0 * math.sqrt(2.0) > max(abs(score) for score in scores)


def test_eight_facet_vertices_uniformly_reconstruct_the_local_boundary() -> None:
    strategies = local_boundary_strategies()
    local_box = deterministic_mixture_box(strategies)

    assert len(strategies) == 8
    assert all(deterministic_facet_score(strategy) == 2 for strategy in strategies)
    assert np.allclose(local_box, isotropic_chsh_box(0.5), atol=1.0e-12)
    assert box_correlations(local_box) == pytest.approx((-0.5, -0.5, -0.5, 0.5))
    assert audit_probability_box(local_box).maximum_unbiased_marginal_residual < 1.0e-12


def test_nonsignalling_remainder_local_fraction_is_exact() -> None:
    local_weight = 2.0 - math.sqrt(2.0)
    nonlocal_weight = math.sqrt(2.0) - 1.0
    local_box = isotropic_chsh_box(0.5)
    pr_box = isotropic_chsh_box(1.0)
    quantum_box = isotropic_chsh_box(1.0 / math.sqrt(2.0))
    reconstructed = local_weight * local_box + nonlocal_weight * pr_box

    assert local_weight + nonlocal_weight == pytest.approx(1.0)
    assert np.allclose(reconstructed, quantum_box, atol=1.0e-12)
    _, quantum_chsh = chsh_scores(quantum_box)
    chsh_upper_bound = (4.0 - quantum_chsh) / 2.0
    assert chsh_upper_bound == pytest.approx(local_weight)


def test_fine_seed_lift_is_bijective_while_coarse_readout_is_many_to_one() -> None:
    probabilities = (0.0, 0.2, 0.3, 0.5)
    first = lift_seed_coordinate(probabilities, 0.05)
    second = lift_seed_coordinate(probabilities, 0.15)

    assert first.outcome_index == second.outcome_index == 1
    assert first.residual_coordinate != second.residual_coordinate
    assert invert_seed_coordinate(
        probabilities, first.outcome_index, first.residual_coordinate
    ) == pytest.approx(0.05)
    assert invert_seed_coordinate(
        probabilities, second.outcome_index, second.residual_coordinate
    ) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="zero-probability"):
        invert_seed_coordinate(probabilities, 0, 0.5)


def test_measure_bijection_is_not_a_usual_topology_homeomorphism() -> None:
    assert not usual_coproduct_seed_lift_is_homeomorphism((0.5, 0.5))
    assert usual_coproduct_seed_lift_is_homeomorphism((0.0, 1.0, 0.0))

    result = certificate()
    assert result.status["fine_seed_weighted_measure_bijection_formula_certified"]
    assert result.status["usual_topology_homeomorphism_counterexample"]
    assert not result.status["usual_topology_homeomorphism_derived"]
    assert result.coarse_seed_readout_many_to_one
    assert result.boundaries[
        "finite_discrete_observation_label_space_is_zero_dimensional"
    ]
    assert result.boundaries["zero_dimensional_readout_is_not_spacetime_dimension"]
    assert result.boundaries["measure_bijection_does_not_earn_metric_pullback"]


def test_certificate_keeps_bell_assumptions_and_claim_ceiling_explicit() -> None:
    result = certificate()

    assert result.status["fixed_setting_quantum_projective_instruments_cptp"]
    assert result.status["finite_singlet_chsh_box_certified"]
    assert result.status["finite_box_operational_no_signalling_certified"]
    assert result.status[
        "setting_independent_local_factorization_excluded_for_box"
    ]
    assert result.status["nonsignalling_remainder_local_fraction_certified"]
    assert result.boundaries["bell_assumes_setting_independent_seed_distribution"]
    assert result.boundaries["bell_assumes_factorized_local_response"]
    assert result.boundaries["global_or_contextual_fine_bijection_not_excluded"]
    assert result.boundaries["operational_no_signalling_is_not_qft_microcausality"]
    assert all(result.alternatives.values())

    assert not result.status["physical_seed_law_derived"]
    assert not result.status["objective_single_outcome_selection_derived"]
    assert not result.status["relativistic_qft_microcausality_derived"]
    assert not result.status["full_lightcone_no_controllable_influence_gate_complete"]
    assert not result.status["spacetime_topology_metric_or_curvature_derived"]
    assert not result.status["fold_stress_or_gravity_derived"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_dimension_and_probability_accounting_do_not_create_energy() -> None:
    result = certificate()

    assert all(result.dimensions.values())
    assert result.accounting["each_setting_probability_box_normalized_once"]
    assert result.accounting["local_and_pr_mixture_weights_sum_to_one"]
    assert result.accounting["weighted_fibre_measure_uses_born_probability_once"]
    assert result.accounting["coarse_and_fine_seed_probabilities_not_double_counted"]
    assert result.accounting["unselected_probabilities_not_added_as_energy_or_stress"]
    assert not result.accounting["seed_or_hidden_coordinate_carries_energy"]
    assert not result.status["mass_dependent_probability_deformation_derived"]


def test_public_contract_fails_closed_and_run_is_json_serializable() -> None:
    with pytest.raises(ValueError, match="eta"):
        isotropic_chsh_box(-0.1)
    with pytest.raises(ValueError, match="eta"):
        isotropic_chsh_box(1.1)
    with pytest.raises(ValueError, match="four outcomes"):
        deterministic_facet_score((1, 1, 1))
    with pytest.raises(ValueError, match="at least one"):
        deterministic_mixture_box(())
    with pytest.raises(ValueError, match="residual_coordinate"):
        invert_seed_coordinate((0.5, 0.5), 0, 1.0)
    with pytest.raises(ValueError, match="outcome_index"):
        invert_seed_coordinate((0.5, 0.5), True, 0.5)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)

    payload = run()
    json.dumps(payload)
    assert payload["deterministic_strategy_count"] == 16
    assert payload["local_boundary_strategy_count"] == 8
    assert payload["status"]["setting_independent_local_factorization_excluded_for_box"]
    assert tuple(OUTCOMES) == (-1, 1)
