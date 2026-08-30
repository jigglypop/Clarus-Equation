from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.proper_vertex_single_cell_haar import (
    certify_proper_vertex_truncated_haar_estimate,
    certify_proper_vertex_truncated_importance_estimate,
    inverse_truncated_sinh_squared_cdf,
    shifted_halton_points,
    sl2c_polar_element,
    su2_from_unit_cube,
    truncated_sinh_squared_radial_volume,
)


def test_polar_coordinate_factors_and_haar_jacobian() -> None:
    cutoff = 1.2
    expected = math.sinh(2.0 * cutoff) / 4.0 - cutoff / 2.0
    assert truncated_sinh_squared_radial_volume(cutoff) == pytest.approx(expected)
    probabilities = np.asarray((0.0, 0.1, 0.5, 0.9, 0.999999))
    radial = inverse_truncated_sinh_squared_cdf(probabilities, cutoff)
    recovered = (
        np.sinh(2.0 * radial) / 4.0 - radial / 2.0
    ) / expected
    assert np.allclose(recovered, probabilities, atol=2.0e-14, rtol=0.0)

    rotation = su2_from_unit_cube(np.asarray((0.2, 0.3, 0.4)))
    assert np.allclose(rotation.conj().T @ rotation, np.eye(2), atol=1.0e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-14)
    element = sl2c_polar_element(
        0.7, np.asarray((0.0, 0.0, 1.0)), rotation
    )
    assert np.linalg.det(element) == pytest.approx(1.0, abs=1.0e-13)


def test_shifted_halton_is_reproducible_and_twenty_four_dimensional() -> None:
    first = shifted_halton_points(16, 24, seed=17)
    second = shifted_halton_points(16, 24, seed=17)
    third = shifted_halton_points(16, 24, seed=18)
    assert first.shape == (16, 24)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert np.all((first >= 0.0) & (first < 1.0))


def test_truncated_haar_estimate_materializes_without_claiming_tail() -> None:
    certificate = certify_proper_vertex_truncated_haar_estimate(
        radial_cutoff=0.8,
        sample_count_per_replicate=8,
        replicate_count=2,
        cp1_shape=(6, 12),
    )
    assert certificate.polar_haar_measure_materialized
    assert certificate.empirical_truncated_integral_estimated
    assert certificate.all_samples_recomputed_full_eq53_projectors
    assert certificate.minimum_absolute_normalized_orientation_determinant > 0.0
    assert certificate.replicate_count == 2
    assert certificate.sample_count_per_replicate == 8
    assert certificate.four_group_truncated_haar_volume == pytest.approx(
        certificate.one_group_truncated_haar_volume**4
    )
    assert np.isfinite(certificate.truncated_integral_estimate)
    assert np.isfinite(certificate.real_standard_error_across_replicates)
    assert np.isfinite(certificate.imaginary_standard_error_across_replicates)
    assert 0.0 < certificate.magnitude_effective_sample_size <= 16.0
    assert 0.0 < certificate.largest_magnitude_fraction <= 1.0
    assert certificate.empirical_coefficient_second_moment > 0.0
    assert not certificate.rigorous_qmc_error_bound_proved
    assert not certificate.finite_importance_variance_proved
    assert not certificate.radial_tail_bound_proved
    assert not certificate.noncompact_haar_integral_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert certificate.claim_ceiling.endswith('HAAR_ESTIMATE_ONLY')


def test_critical_rotation_importance_estimate_tracks_proposal_and_ceiling() -> None:
    certificate = certify_proper_vertex_truncated_importance_estimate(
        radial_cutoff=0.8,
        sample_count_per_replicate=8,
        replicate_count=2,
        cp1_shape=(6, 12),
    )
    assert certificate.proposal_density_exactly_accounted_in_weights
    assert certificate.empirical_truncated_importance_estimate_materialized
    assert len(certificate.critical_rotation_centers) == 4
    assert len(certificate.replicate_estimates) == 2
    assert certificate.minimum_absolute_normalized_orientation_determinant > 0.0
    assert 0.0 < certificate.contribution_magnitude_effective_sample_size <= 16.0
    assert 0.0 < certificate.largest_contribution_magnitude_fraction <= 1.0
    assert not certificate.finite_importance_variance_proved
    assert not certificate.radial_tail_bound_proved
    assert not certificate.noncompact_haar_integral_evaluated
    assert certificate.claim_ceiling.endswith('IMPORTANCE_ONLY')
