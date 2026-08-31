from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.kraus_refinement_metric_obstruction import (
    I2,
    P0,
    P1,
    apply_cp_map,
    certificate,
    choi_matrix,
    duplicate_operation,
    isometric_refinement,
    raw_count_conformal_factor,
)


def test_nonzero_duplication_preserves_operation_probability_and_posterior() -> None:
    result = certificate()
    assert result.hidden_multiplicities == (1, 2, 16, 37)
    assert result.maximum_operation_residual < 1.0e-12
    assert result.maximum_coarse_probability_residual < 1.0e-12
    assert result.maximum_posterior_residual < 1.0e-12
    assert all(math.isclose(value, result.outcome_probability) for value in result.sublabel_probability_sums)
    assert result.status["outcome_operation_isometry_invariant"]
    assert result.status["coarse_probability_invariant"]
    assert result.status["posterior_invariant"]


def test_full_instrument_remains_cptp_and_choi_matrix_is_invariant() -> None:
    result = certificate()
    assert result.maximum_full_completeness_residual < 1.0e-12
    assert result.maximum_total_probability_residual < 1.0e-12
    assert result.maximum_choi_residual < 1.0e-12
    assert result.numerical_choi_ranks == (1, 1, 1, 1)
    assert result.status["cptp_completeness_preserved"]
    assert result.status["choi_matrix_invariant"]
    assert result.status["choi_rank_numerically_invariant"]
    assert not result.status["minimal_kraus_rank_theorem_proved_by_finite_regression"]


def test_general_isometric_mixing_preserves_a_nontrivial_dephasing_channel() -> None:
    isometry = 0.5 * np.array(
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0j], [1.0, -1.0j]],
        dtype=np.complex128,
    )
    refined = isometric_refinement((P0, P1), isometry)
    state = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=np.complex128)
    assert len(refined) == 4
    assert np.allclose(apply_cp_map(refined, state), apply_cp_map((P0, P1), state))
    assert np.allclose(choi_matrix(refined), choi_matrix((P0, P1)))
    result = certificate()
    assert result.general_isometry_shape == (4, 2)
    assert result.general_isometry_residual < 1.0e-12
    assert result.general_channel_residual < 1.0e-12


def test_raw_count_metric_candidate_fails_representation_invariance() -> None:
    result = certificate(spacetime_dimension=4)
    assert result.raw_conformal_factors[0] == 1.0
    assert result.raw_conformal_factors[2] == 2.0
    assert result.raw_metric_coefficient_ratios[0] == 1.0
    assert result.raw_metric_coefficient_ratios[2] == 4.0
    assert not result.status["raw_hidden_count_invariant"]
    assert result.status["raw_count_metric_changes_for_same_instrument"]
    assert not result.status["raw_count_defines_physical_volume_or_metric"]


def test_dimension_accounting_boundary_and_status_ceiling() -> None:
    result = certificate()
    assert all(result.dimensions.values())
    assert result.accounting["refined_sublabel_probabilities_sum_to_coarse_probability"]
    assert result.accounting["coarse_plus_refined_probability_double_counting_forbidden"]
    assert not result.accounting["representation_only_sublabel_adds_energy_or_stress"]
    assert not result.accounting["energy_receipt_or_stress_used"]
    assert result.boundaries["sublabel_is_unobserved"]
    assert not result.boundaries["physical_pointer_record_derived"]
    assert all(result.alternatives.values())
    assert not result.status["local_volume_measure_derived"]
    assert not result.status["metric_or_curvature_derived"]
    assert not result.status["gr_lensing_backreaction_derived"]
    assert not result.status["success_gates_5_to_8_complete"]


def test_public_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        duplicate_operation(I2, 0)
    with pytest.raises(ValueError, match="positive integer"):
        duplicate_operation(I2, True)
    with pytest.raises(ValueError, match="isometry columns"):
        isometric_refinement((P0, P1), np.eye(3))
    with pytest.raises(ValueError, match=r"u\^dagger u"):
        isometric_refinement((P0, P1), np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least two"):
        raw_count_conformal_factor(1, spacetime_dimension=1)
    with pytest.raises(ValueError, match="outcome_probability"):
        certificate(outcome_probability=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        certificate(hidden_multiplicities=())
    with pytest.raises(ValueError, match="positive integer"):
        certificate(hidden_multiplicities=(1, False))


def test_relative_choi_rank_remains_one_for_a_tiny_nonzero_outcome() -> None:
    result = certificate(outcome_probability=1.0e-20)
    assert result.numerical_choi_ranks == (1, 1, 1, 1)
    assert result.maximum_total_probability_residual < 1.0e-12
