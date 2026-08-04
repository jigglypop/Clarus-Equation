from __future__ import annotations

import math

import numpy as np
import pytest

from reality_stone.clarus.euclidean_correlator_certificate import (
    CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV,
    ENSEMBLE_SAMPLING_ASSUMPTION,
    EUCLIDEAN_KERNEL_ASSUMPTION,
    EuclideanCorrelatorStage,
    EuclideanCorrelatorTolerances,
    PairedEuclideanEnsemble,
    current_ce_euclidean_correlator_certificate,
    euclidean_correlator_certificate,
)


CONTROL_MASS_MEV = 0.35


def _ensemble_from_connected_shape(
    shape: np.ndarray,
    *,
    times: np.ndarray | None = None,
    configuration_count: int = 48,
    offset_t: float = 0.0,
    offset_zero: float = 0.0,
    orthogonal_noise_scale: float = 0.2,
) -> PairedEuclideanEnsemble:
    values = np.asarray(shape, dtype=np.float64)
    if times is None:
        times = np.arange(1, values.size + 1, dtype=np.float64)
    rng = np.random.default_rng(20260804)
    operator_zero_centered = rng.normal(size=configuration_count)
    operator_zero_centered -= np.mean(operator_zero_centered)
    operator_zero_centered /= np.std(operator_zero_centered, ddof=1)

    orthogonal_noise = rng.normal(size=(configuration_count, values.size))
    orthogonal_noise -= np.mean(orthogonal_noise, axis=0)
    projection = (operator_zero_centered @ orthogonal_noise) / (
        operator_zero_centered @ operator_zero_centered
    )
    orthogonal_noise -= operator_zero_centered[:, np.newaxis] * projection

    operator_t = (
        offset_t
        + operator_zero_centered[:, np.newaxis] * values[np.newaxis, :]
        + orthogonal_noise_scale * orthogonal_noise
    )
    operator_zero = offset_zero + operator_zero_centered
    return PairedEuclideanEnsemble(times, operator_t, operator_zero)


def _mass_grid() -> np.ndarray:
    return np.linspace(0.02, 1.8, 12, dtype=np.float64)


def _certificate(
    shape: np.ndarray,
    *,
    times: np.ndarray | None = None,
    registered_mass_mev: float = CONTROL_MASS_MEV,
    orthogonal_noise_scale: float = 0.2,
):
    ensemble = _ensemble_from_connected_shape(
        shape,
        times=times,
        orthogonal_noise_scale=orthogonal_noise_scale,
    )
    return euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=registered_mass_mev,
        ensemble=ensemble,
        mass_grid_mev=_mass_grid(),
    )


def test_current_ce_stays_at_registered_scale_without_raw_ensemble() -> None:
    certificate = current_ce_euclidean_correlator_certificate()
    payload = certificate.to_dict()

    assert certificate.registered_inverse_correlation_scale_mev == (
        CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV
    )
    assert certificate.maximum_supported_stage is EuclideanCorrelatorStage.REGISTERED_SCALE
    assert certificate.euclidean_kernel_assumption == EUCLIDEAN_KERNEL_ASSUMPTION
    assert certificate.ensemble_sampling_assumption == ENSEMBLE_SAMPLING_ASSUMPTION
    assert not certificate.raw_paired_ensemble_present
    assert not certificate.euclidean_screening_control_pass
    assert not certificate.minkowski_pole_derived
    assert not certificate.physical_lsz_particle_derived
    assert not certificate.ce_field_identity_derived
    assert certificate.first_blocker == "raw paired O(t), O(0) ensemble is absent"
    assert payload["maximum_supported_stage"] == "REGISTERED_SCALE"
    assert payload["tolerances"]["minimum_configuration_count"] == 8


def test_connected_correlator_is_unbiased_and_covariance_is_delete_one_jackknife() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    ensemble = _ensemble_from_connected_shape(shape)
    certificate = euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
        ensemble=ensemble,
        mass_grid_mev=_mass_grid(),
    )
    audit = certificate.connected
    assert audit is not None

    centered_t = ensemble.operator_t - np.mean(ensemble.operator_t, axis=0)
    centered_zero = ensemble.operator_zero - np.mean(ensemble.operator_zero)
    expected_connected = np.sum(centered_t * centered_zero[:, np.newaxis], axis=0) / (
        ensemble.operator_zero.size - 1
    )
    np.testing.assert_allclose(audit.connected_correlator, expected_connected, rtol=1e-13)
    np.testing.assert_allclose(audit.connected_correlator, shape, rtol=1e-13)

    leave_one_out = []
    for omitted in range(ensemble.operator_zero.size):
        retained = np.arange(ensemble.operator_zero.size) != omitted
        leave_one_out.append(
            np.array(
                [
                    np.cov(
                        ensemble.operator_t[retained, time_index],
                        ensemble.operator_zero[retained],
                        ddof=1,
                    )[0, 1]
                    for time_index in range(times.size)
                ]
            )
        )
    leave_one_out_array = np.asarray(leave_one_out)
    centered_leave_one_out = leave_one_out_array - np.mean(leave_one_out_array, axis=0)
    expected_jackknife_covariance = (
        (ensemble.operator_zero.size - 1)
        / ensemble.operator_zero.size
        * centered_leave_one_out.T
        @ centered_leave_one_out
    )
    np.testing.assert_allclose(
        audit.connected_mean_covariance,
        expected_jackknife_covariance,
        rtol=2e-12,
        atol=1e-14,
    )
    assert audit.connected_correlator_control_pass


def test_disconnected_offsets_subtract_without_changing_connected_result() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    baseline = _ensemble_from_connected_shape(shape)
    shifted = _ensemble_from_connected_shape(
        shape,
        offset_t=12.0,
        offset_zero=-7.0,
    )
    baseline_certificate = euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
        ensemble=baseline,
        mass_grid_mev=_mass_grid(),
    )
    shifted_certificate = euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
        ensemble=shifted,
        mass_grid_mev=_mass_grid(),
    )
    assert baseline_certificate.connected is not None
    assert shifted_certificate.connected is not None
    np.testing.assert_allclose(
        shifted_certificate.connected.connected_correlator,
        baseline_certificate.connected.connected_correlator,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted_certificate.connected.connected_mean_covariance,
        baseline_certificate.connected.connected_mean_covariance,
        rtol=1e-11,
        atol=1e-13,
    )


def test_synthetic_forward_exponential_reaches_only_euclidean_screening() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = 1.7 * np.exp(-CONTROL_MASS_MEV * times)
    certificate = _certificate(shape, times=times)

    assert certificate.connected_correlator_control_pass
    assert certificate.positive_spectrum_necessary_control_pass
    assert certificate.euclidean_screening_control_pass
    assert certificate.maximum_supported_stage is (
        EuclideanCorrelatorStage.EUCLIDEAN_SCREENING_CONTROL
    )
    assert certificate.positive_spectrum_necessary is not None
    assert certificate.positive_spectrum_necessary.complete_monotonicity
    assert certificate.positive_spectrum_necessary.truncated_hausdorff_parity_condition_pass
    assert certificate.effective_mass is not None
    assert certificate.effective_mass.covariance_aware_fit_pass
    assert all(certificate.effective_mass.window_gls_identifiable)
    assert certificate.effective_mass.mean_window_mass_mev == pytest.approx(
        CONTROL_MASS_MEV,
        rel=1e-10,
    )
    assert not certificate.minkowski_pole_derived
    assert not certificate.positive_minkowski_residue_derived
    assert not certificate.spectral_density_uniquely_identified
    assert not certificate.physical_lsz_particle_derived
    assert not certificate.ce_field_identity_derived


@pytest.mark.parametrize(
    "shape",
    [
        np.array([1.0, 0.8, -0.1, 0.4, 0.3, 0.2, 0.1]),
        np.array([1.0, 0.8, 0.82, 0.6, 0.45, 0.3, 0.2]),
    ],
)
def test_negative_or_nonmonotone_connected_data_fail_necessary_control(
    shape: np.ndarray,
) -> None:
    certificate = _certificate(shape)
    audit = certificate.positive_spectrum_necessary

    assert audit is not None
    assert not audit.necessary_conditions_pass
    assert not certificate.positive_spectrum_necessary_control_pass
    assert certificate.maximum_supported_stage is (
        EuclideanCorrelatorStage.CONNECTED_CORRELATOR_CONTROL
    )


def test_positive_decreasing_but_log_concave_data_fail() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-0.08 * times**2)
    certificate = _certificate(shape, times=times)
    audit = certificate.positive_spectrum_necessary

    assert audit is not None
    assert audit.connected_strictly_positive
    assert audit.connected_nonincreasing
    assert not audit.log_convex
    assert not audit.necessary_conditions_pass


def test_odd_length_truncated_hausdorff_localizer_catches_false_positive() -> None:
    shape = np.array([1.0, 0.5, 0.25, 0.125, 0.1])
    certificate = _certificate(shape)
    audit = certificate.positive_spectrum_necessary

    assert audit is not None
    assert audit.connected_strictly_positive
    assert audit.connected_nonincreasing
    assert audit.log_convex
    assert audit.complete_monotonicity
    assert audit.hankel_h0_positive_semidefinite
    assert audit.minimum_hankel_h1_minus_h2_eigenvalue is not None
    assert audit.minimum_hankel_h1_minus_h2_eigenvalue < 0.0
    assert not audit.hankel_h1_minus_h2_positive_semidefinite
    assert not audit.truncated_hausdorff_parity_condition_pass
    assert not audit.necessary_conditions_pass


def test_nonuniform_grid_fails_closed_when_moment_tests_are_unavailable() -> None:
    times = np.array([1.0, 2.0, 3.1, 4.1, 5.2, 6.2, 7.3])
    shape = np.exp(-CONTROL_MASS_MEV * times)
    certificate = _certificate(shape, times=times)
    audit = certificate.positive_spectrum_necessary

    assert audit is not None
    assert audit.log_convex
    assert not audit.time_grid_uniform
    assert not audit.complete_monotonicity_test_available
    assert not audit.hankel_test_available
    assert not audit.necessary_conditions_pass


def test_rank_deficient_jackknife_covariance_cannot_be_regularized_into_evidence() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    certificate = _certificate(
        shape,
        times=times,
        orthogonal_noise_scale=0.0,
    )

    assert certificate.positive_spectrum_necessary_control_pass
    assert certificate.effective_mass is not None
    assert not certificate.effective_mass.covariance_aware_fit_pass
    assert not certificate.euclidean_screening_control_pass
    assert certificate.maximum_supported_stage is (
        EuclideanCorrelatorStage.POSITIVE_SPECTRUM_NECESSARY_CONTROL
    )


def test_positive_mixture_passes_moment_tests_but_has_window_mass_drift() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = 0.6 * np.exp(-0.2 * times) + 0.4 * np.exp(-1.0 * times)
    certificate = _certificate(shape, times=times)

    assert certificate.positive_spectrum_necessary_control_pass
    assert certificate.effective_mass is not None
    assert certificate.effective_mass.window_mass_relative_drift is not None
    assert certificate.effective_mass.window_mass_relative_drift > (
        certificate.tolerances.maximum_window_mass_relative_drift
    )
    assert not certificate.effective_mass.single_exponential_screening_pass
    assert not certificate.euclidean_screening_control_pass


def test_augmented_nullspace_gives_two_nonnegative_equal_weight_spectra() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    certificate = _certificate(shape, times=times)
    audit = certificate.spectral_nonuniqueness
    assert audit is not None

    assert audit.nullity > 0
    assert audit.normalization_augmented_nullity > 0
    assert not audit.fixed_grid_discrete_weights_injective
    assert not audit.normalization_constrained_fixed_grid_weights_injective
    assert audit.two_distinct_nonnegative_normalized_weight_vectors_constructed
    minus = np.asarray(audit.minus_discrete_atom_weights)
    plus = np.asarray(audit.plus_discrete_atom_weights)
    masses = np.asarray(audit.mass_grid_mev)
    kernel = np.exp(-np.outer(times, masses))
    assert np.all(minus >= 0.0)
    assert np.all(plus >= 0.0)
    assert not np.array_equal(minus, plus)
    np.testing.assert_allclose(kernel @ minus, kernel @ plus, rtol=1e-11, atol=1e-13)
    assert np.sum(minus) == pytest.approx(np.sum(plus), abs=1e-13)
    assert audit.total_weight_pair_residual == pytest.approx(0.0, abs=1e-13)


def test_invalid_shapes_nonfinite_contact_time_and_low_configuration_count_reject() -> None:
    times = np.arange(1, 5, dtype=np.float64)
    operator_t = np.ones((8, 4))
    operator_zero = np.ones(8)
    with pytest.raises(ValueError, match="time dimension"):
        PairedEuclideanEnsemble(times, np.ones((8, 3)), operator_zero)
    with pytest.raises(ValueError, match="configuration counts"):
        PairedEuclideanEnsemble(times, operator_t, np.ones(7))
    with pytest.raises(ValueError, match="finite"):
        PairedEuclideanEnsemble(times, np.full((8, 4), math.nan), operator_zero)
    with pytest.raises(ValueError, match="t=0"):
        PairedEuclideanEnsemble(np.arange(4, dtype=np.float64), operator_t, operator_zero)

    small_ensemble = PairedEuclideanEnsemble(times, np.ones((4, 4)), np.ones(4))
    with pytest.raises(ValueError, match="predeclared minimum"):
        euclidean_correlator_certificate(
            ensemble=small_ensemble,
            mass_grid_mev=_mass_grid(),
        )


def test_finite_raw_values_that_overflow_derived_products_reject() -> None:
    times = np.arange(1, 5, dtype=np.float64)
    signs = np.where(np.arange(8) % 2 == 0, 1.0, -1.0)
    ensemble = PairedEuclideanEnsemble(
        times,
        signs[:, np.newaxis] * np.full((8, 4), 1.0e308),
        signs * 1.0e308,
    )
    with pytest.raises(ValueError, match="derived"):
        euclidean_correlator_certificate(
            ensemble=ensemble,
            mass_grid_mev=_mass_grid(),
        )


def test_catastrophic_disconnected_subtraction_cannot_promote_stage() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    ensemble = _ensemble_from_connected_shape(
        shape,
        times=times,
        offset_t=1.0e12,
        offset_zero=1.0e12,
    )
    certificate = euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
        ensemble=ensemble,
        mass_grid_mev=_mass_grid(),
    )

    assert certificate.connected is not None
    assert not certificate.connected.connected_correlator_control_pass
    assert not certificate.connected_correlator_control_pass
    assert certificate.maximum_supported_stage is EuclideanCorrelatorStage.REGISTERED_SCALE


def test_fully_underflowed_exponential_kernel_rejects() -> None:
    times = np.arange(1, 8, dtype=np.float64)
    shape = np.exp(-CONTROL_MASS_MEV * times)
    ensemble = _ensemble_from_connected_shape(shape, times=times)
    underflowed_mass_grid = np.linspace(800.0, 900.0, 12, dtype=np.float64)

    with pytest.raises(ValueError, match="numerically zero|zero numerical rank"):
        euclidean_correlator_certificate(
            registered_inverse_correlation_scale_mev=CONTROL_MASS_MEV,
            ensemble=ensemble,
            mass_grid_mev=underflowed_mass_grid,
        )


@pytest.mark.parametrize("value", [True, 0.0, math.nan, math.inf])
def test_registered_scale_rejects_nonphysical_values(value: object) -> None:
    with pytest.raises(ValueError):
        euclidean_correlator_certificate(
            registered_inverse_correlation_scale_mev=value,  # type: ignore[arg-type]
        )


def test_relative_tolerances_cannot_be_loosened_beyond_one() -> None:
    with pytest.raises(ValueError, match="less than one"):
        EuclideanCorrelatorTolerances(svd_rank_relative_tolerance=1.0)
