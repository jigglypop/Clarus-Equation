"""Run the current CE Euclidean gate and a labeled synthetic control."""

from __future__ import annotations

import json

import numpy as np

from reality_stone.clarus.euclidean_correlator_certificate import (
    CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV,
    PairedEuclideanEnsemble,
    current_ce_euclidean_correlator_certificate,
    euclidean_correlator_certificate,
)


def _synthetic_single_exponential_control():
    """Build independent deterministic bins with a known connected shape."""

    mass_mev = CURRENT_CE_INVERSE_CORRELATION_SCALE_MEV
    times = 0.01 * np.arange(1, 8, dtype=np.float64)
    connected_shape = 1.7 * np.exp(-mass_mev * times)
    configuration_count = 64
    rng = np.random.default_rng(20260804)

    operator_zero_centered = rng.normal(size=configuration_count)
    operator_zero_centered -= np.mean(operator_zero_centered)
    operator_zero_centered /= np.std(operator_zero_centered, ddof=1)

    orthogonal_noise = rng.normal(size=(configuration_count, times.size))
    orthogonal_noise -= np.mean(orthogonal_noise, axis=0)
    projection = (operator_zero_centered @ orthogonal_noise) / (
        operator_zero_centered @ operator_zero_centered
    )
    orthogonal_noise -= operator_zero_centered[:, np.newaxis] * projection

    ensemble = PairedEuclideanEnsemble(
        euclidean_time_mev_inv=times,
        operator_t=(
            3.0
            + operator_zero_centered[:, np.newaxis] * connected_shape[np.newaxis, :]
            + 0.2 * orthogonal_noise
        ),
        operator_zero=-2.0 + operator_zero_centered,
    )
    return euclidean_correlator_certificate(
        registered_inverse_correlation_scale_mev=mass_mev,
        ensemble=ensemble,
        mass_grid_mev=np.linspace(2.0, 100.0, 16, dtype=np.float64),
    )


def main() -> None:
    """Print the data-absent CE result beside the synthetic method control."""

    current = current_ce_euclidean_correlator_certificate()
    synthetic = _synthetic_single_exponential_control()
    effective = synthetic.effective_mass
    spectral = synthetic.spectral_nonuniqueness
    if effective is None or spectral is None:
        raise RuntimeError("synthetic control did not produce the expected audits")
    if not synthetic.euclidean_screening_control_pass:
        raise RuntimeError("synthetic screening control failed")

    payload = {
        "current_ce": {
            "maximum_stage": current.maximum_supported_stage.value,
            "first_blocker": current.first_blocker,
            "raw_paired_ensemble_present": current.raw_paired_ensemble_present,
            "minkowski_pole_derived": current.minkowski_pole_derived,
            "physical_lsz_particle_derived": current.physical_lsz_particle_derived,
            "ce_field_identity_derived": current.ce_field_identity_derived,
        },
        "synthetic_method_control": {
            "maximum_stage": synthetic.maximum_supported_stage.value,
            "mean_screening_mass_mev": effective.mean_window_mass_mev,
            "registered_mass_relative_error": (effective.registered_mass_relative_error),
            "covariance_aware_fit_pass": bool(effective.covariance_aware_fit_pass),
            "normalization_augmented_nullity": (spectral.normalization_augmented_nullity),
            "two_equal_weight_nonnegative_spectra": (
                spectral.two_distinct_nonnegative_normalized_weight_vectors_constructed
            ),
            "correlator_pair_relative_residual": (spectral.correlator_pair_relative_residual),
            "total_weight_pair_residual": spectral.total_weight_pair_residual,
            "spectral_density_uniquely_identified": (
                synthetic.spectral_density_uniquely_identified
            ),
            "minkowski_pole_derived": synthetic.minkowski_pole_derived,
            "physical_lsz_particle_derived": (synthetic.physical_lsz_particle_derived),
            "ce_field_identity_derived": synthetic.ce_field_identity_derived,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
