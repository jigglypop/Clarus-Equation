from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.physics.local_born_selection_kernel import (
    CoarseOutcomeOperation,
    apply_nonselective_instrument,
    born_probabilities,
    build_seed_partition,
    certificate,
    equal_copy_internal_refinement,
    run,
    select_outcome,
    select_partition_cell,
)


P0 = np.diag([1.0, 0.0]).astype(np.complex128)
P1 = np.diag([0.0, 1.0]).astype(np.complex128)
PROJECTIVE_OUTCOMES = (
    CoarseOutcomeOperation("zero", (P0,)),
    CoarseOutcomeOperation("one", (P1,)),
)


def test_half_open_seed_partition_handles_zero_cells_and_exact_boundaries() -> None:
    partition = build_seed_partition((0.0, 0.4, 0.6))
    assert partition.input_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert partition.cell_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert partition.intervals[0] == (0.0, 0.0)
    assert partition.intervals[1] == pytest.approx((0.0, 0.4))
    assert partition.intervals[2] == pytest.approx((0.4, 1.0))
    assert select_partition_cell(partition, 0.0) == 1
    assert select_partition_cell(partition, math.nextafter(0.4, 0.0)) == 1
    assert select_partition_cell(partition, 0.4) == 2
    assert select_partition_cell(partition, math.nextafter(1.0, 0.0)) == 2

    nonleading_zeros = build_seed_partition(
        (0.1, 0.0, 0.2, 0.3000000000002, 0.4000000000003, 0.0)
    )
    first_cell_end = nonleading_zeros.intervals[0][1]
    assert nonleading_zeros.intervals[1] == pytest.approx(
        (first_cell_end, first_cell_end)
    )
    assert nonleading_zeros.intervals[-1] == pytest.approx((1.0, 1.0))


def test_supplied_seed_returns_one_born_outcome_and_normalized_posterior() -> None:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())
    probabilities = born_probabilities(PROJECTIVE_OUTCOMES, state)
    assert probabilities == pytest.approx((0.5, 0.5))

    selected_zero = select_outcome(PROJECTIVE_OUTCOMES, state, 0.25)
    selected_one = select_outcome(PROJECTIVE_OUTCOMES, state, 0.75)
    assert selected_zero.label == "zero"
    assert selected_one.label == "one"
    assert selected_zero.raw_born_probability == pytest.approx(0.5)
    assert selected_zero.partition_probability == pytest.approx(0.5)
    assert np.allclose(selected_zero.posterior, P0)
    assert np.allclose(selected_one.posterior, P1)
    assert np.allclose(
        0.5 * selected_zero.posterior + 0.5 * selected_one.posterior,
        apply_nonselective_instrument(PROJECTIVE_OUTCOMES, state),
    )


def test_internal_equal_copy_refinement_preserves_coarse_selection() -> None:
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())
    refined = (
        equal_copy_internal_refinement(PROJECTIVE_OUTCOMES[0], 11),
        PROJECTIVE_OUTCOMES[1],
    )
    assert born_probabilities(refined, state) == pytest.approx((0.5, 0.5))
    # Exact cumulative boundaries are measure-zero and may move by roundoff
    # under a numerically equivalent Kraus refinement, so compare interiors.
    for seed in (0.0, 0.25, 0.5001, math.nextafter(1.0, 0.0)):
        base = select_outcome(PROJECTIVE_OUTCOMES, state, seed)
        changed = select_outcome(refined, state, seed)
        assert changed.label == base.label
        assert changed.raw_born_probability == pytest.approx(base.raw_born_probability)
        assert changed.partition_probability == pytest.approx(
            base.partition_probability
        )
        assert np.allclose(changed.subnormalized_state, base.subnormalized_state)
        assert np.allclose(changed.posterior, base.posterior)


def test_raw_born_and_numerically_normalized_partition_probabilities_are_explicit() -> None:
    delta = 4.0e-13
    near_complete_outcomes = (
        CoarseOutcomeOperation("zero", (math.sqrt(1.0 + delta) * P0,)),
        CoarseOutcomeOperation("one", (P1,)),
    )
    plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    state = np.outer(plus, plus.conj())

    selected = select_outcome(near_complete_outcomes, state, 0.25)
    interval_length = selected.interval[1] - selected.interval[0]
    assert abs(selected.raw_born_probability - selected.partition_probability) > 1.0e-14
    assert interval_length == pytest.approx(selected.partition_probability, abs=1.0e-15)
    assert np.trace(selected.posterior).real == pytest.approx(1.0, abs=1.0e-14)


def test_collision_selection_keeps_zero_branch_out_and_closes_energy_receipts() -> None:
    result = certificate()
    assert result.outcome_labels == ("silent", "left", "right")
    assert result.raw_born_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert result.partition_probabilities == pytest.approx((0.0, 0.4, 0.6))
    assert result.seed_intervals[0] == (0.0, 0.0)
    assert result.probe_labels == ("left", "left", "right", "right")
    assert result.collision_operator_energy_ledger_residual < 1.0e-12
    assert result.maximum_supported_branch_relative_energy_residual < 1.0e-12
    assert result.maximum_supported_branch_dimensionless_energy_variance < 1.0e-12
    assert result.status["supplied_collision_operator_energy_ledger_certified"]
    assert result.status["sharp_supported_branch_energy_receipts_certified"]
    assert result.accounting["all_zero_probability_outcomes_not_conditioned"]

    scaled = certificate(energy_scale=7.0)
    assert scaled.energy_scale == pytest.approx(7.0)
    assert scaled.maximum_supported_branch_relative_energy_residual < 1.0e-12
    assert scaled.maximum_supported_branch_dimensionless_energy_variance < 1.0e-12
    assert scaled.status["sharp_supported_branch_energy_receipts_certified"]


def test_uniform_seed_average_is_cptp_and_coarse_refinement_invariant() -> None:
    result = certificate()
    assert result.probability_normalization_residual < 1.0e-12
    assert result.maximum_interval_probability_residual < 1.0e-12
    assert result.maximum_posterior_trace_residual < 1.0e-12
    assert result.seed_average_channel_residual < 1.0e-12
    assert result.completeness_residual < 1.0e-12
    assert result.refinement_operation_residual < 1.0e-12
    assert result.refinement_probability_residual < 1.0e-12
    assert result.refinement_posterior_residual < 1.0e-12
    assert result.refinement_interval_residual < 1.0e-12
    assert result.refinement_same_seed_label_mismatches == 0
    assert result.status["inverse_cdf_born_partition_certified"]
    assert result.status["explicit_collision_instrument_cptp"]
    assert result.status["coarse_selection_internal_refinement_invariant"]


def test_bell_witness_separates_nonselective_locality_from_forced_seed_signal() -> None:
    result = certificate()
    assert result.remote_nonselective_marginal_residual < 1.0e-12
    assert result.forced_seed_remote_trace_distance == pytest.approx(1.0)
    assert result.status["single_local_nonselective_marginal_witness"]
    assert result.status["controllable_seed_signalling_counterexample"]
    assert result.boundaries["forced_seed_is_prohibited_external_intervention"]
    assert not result.status["relativistic_no_signalling_derived"]


def test_fixed_seed_and_scalar_energy_receipt_counterexamples_are_locked() -> None:
    result = certificate()
    assert result.fixed_seed_born_frequency_error == pytest.approx(0.5)
    assert result.status["fixed_seed_born_frequency_counterexample"]
    assert result.x_measurement_best_scalar_receipts == pytest.approx((0.0, 0.0), abs=1.0e-12)
    assert result.x_measurement_relative_frobenius_receipt_residual == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert result.x_measurement_relative_operator_receipt_residual == pytest.approx(0.5)
    assert result.status["general_scalar_energy_receipt_counterexample"]
    assert not result.status["general_measurement_energy_conservation_derived"]


def test_dimension_accounting_axiom_and_claim_ceiling_are_explicit() -> None:
    result = certificate()
    assert all(result.dimensions.values())
    assert result.accounting["probabilities_partition_seed_measure_once"]
    assert result.accounting["weighted_posteriors_equal_nonselective_channel_once"]
    assert result.accounting["unselected_probabilities_not_added_as_energy"]
    assert result.accounting["selected_record_energy_receipt_counted_once"]
    assert not result.accounting["seed_carries_energy"]
    assert result.boundaries[
        "uniform_independent_uncontrollable_seed_is_explicit_axiom"
    ]
    assert result.boundaries["unitary_or_stinespring_does_not_derive_seed"]
    assert result.boundaries["internal_kraus_labels_do_not_enter_seed_partition"]
    assert result.boundaries[
        "finite_refinement_probe_set_excludes_boundary_neighborhoods"
    ]
    assert result.boundaries["same_seed_refinement_claim_limited_to_declared_probe_set"]
    assert all(result.alternatives.values())
    assert not result.status["physical_uniform_seed_law_derived"]
    assert not result.status["objective_single_outcome_selection_derived"]
    assert not result.status["durable_physical_pointer_derived"]
    assert not result.status["spacetime_metric_curvature_or_gravity_derived"]
    assert not result.status["independent_holdout_complete"]
    assert not result.status["success_gates_1_to_8_complete"]


def test_public_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        build_seed_partition((0.2, 0.2))
    with pytest.raises(ValueError, match="nonnegative"):
        build_seed_partition((-0.1, 1.1))
    partition = build_seed_partition((0.5, 0.5))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        select_partition_cell(partition, 1.0)
    with pytest.raises(ValueError, match="positive integer"):
        equal_copy_internal_refinement(PROJECTIVE_OUTCOMES[0], True)
    with pytest.raises(ValueError, match="complete"):
        born_probabilities((CoarseOutcomeOperation("only", (0.5 * P0,)),), P0)
    with pytest.raises(ValueError, match="left_probability"):
        certificate(left_probability=1.0)
    with pytest.raises(ValueError, match="energy_scale"):
        certificate(energy_scale=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        certificate(tolerance=0.0)


def test_run_payload_is_json_serializable_without_promoting_physical_selection() -> None:
    payload = run()
    json.dumps(payload)
    assert payload["status"]["inverse_cdf_born_partition_certified"]
    assert not payload["status"]["objective_single_outcome_selection_derived"]
