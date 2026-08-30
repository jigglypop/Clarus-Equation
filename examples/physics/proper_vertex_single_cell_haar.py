'''Truncated empirical Haar integration for one Lorentzian proper vertex.

Each unfixed SL(2,C) element is parameterized by the polar decomposition

    X = U exp(r n.sigma / 2),

with normalized SU(2) Haar measure, ordinary area measure on S2, and the
declared convention ``C_H = 1``:

    dX = sinh(r)^2 dr dOmega(n) dU.

The four radial variables are truncated to ``0 <= r <= R``.  Randomly shifted
Halton replicates estimate the resulting compact 24-real-dimensional
integral.  Every sample recomputes all beta signs and Eq.-53 target
projectors.  The output is an empirical truncated estimate, not the full
noncompact Haar integral and not a rigorous tail/error bound.
'''

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.proper_vertex_single_cell_integrand import (
    evaluate_proper_vertex_coefficient_at_frames,
)
from examples.physics.proper_vertex_single_cell_kernel import (
    certify_proper_vertex_single_cell_kernel,
)


_FIRST_24_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
)


@dataclass(frozen=True)
class TruncatedHaarReplicate:
    replicate_index: int
    random_shift_seed: int
    sample_count: int
    orientation_degenerate_sample_count: int
    coefficient_mean: complex
    absolute_coefficient_mean: float
    truncated_integral_estimate: complex


@dataclass(frozen=True)
class ProperVertexTruncatedHaarCertificate:
    cell: tuple[int, int, int, int, int]
    root_omitted_vertex: int
    haar_normalization_c_h: float
    su2_haar_normalized_to_one: bool
    sphere_area_normalization: float
    radial_cutoff: float
    one_group_radial_volume: float
    one_group_truncated_haar_volume: float
    four_group_truncated_haar_volume: float
    sample_count_per_replicate: int
    replicate_count: int
    cp1_quadrature_shape: tuple[int, int]
    replicates: tuple[TruncatedHaarReplicate, ...]
    total_orientation_degenerate_sample_count: int
    minimum_absolute_normalized_orientation_determinant: float
    truncated_integral_estimate: complex
    real_standard_error_across_replicates: float
    imaginary_standard_error_across_replicates: float
    mean_absolute_coefficient: float
    average_phase_ratio: complex
    empirical_coefficient_second_moment: float
    magnitude_effective_sample_size: float
    largest_magnitude_fraction: float
    polar_haar_measure_materialized: bool
    all_samples_recomputed_full_eq53_projectors: bool
    empirical_truncated_integral_estimated: bool
    rigorous_qmc_error_bound_proved: bool
    finite_importance_variance_proved: bool
    radial_tail_bound_proved: bool
    noncompact_haar_integral_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = 'TRUNCATED_EMPIRICAL_SHIFTED_HALTON_HAAR_ESTIMATE_ONLY'


@dataclass(frozen=True)
class ProperVertexTruncatedImportanceCertificate:
    cell: tuple[int, int, int, int, int]
    radial_cutoff: float
    haar_normalization_c_h: float
    radial_near_origin_mixture_weight: float
    radial_origin_gamma_rate: float
    rotation_critical_mixture_weight: float
    rotation_beta_concentration: float
    sample_count_per_replicate: int
    replicate_count: int
    cp1_quadrature_shape: tuple[int, int]
    critical_rotation_centers: tuple[tuple[int, tuple[float, float, float]], ...]
    replicate_estimates: tuple[complex, ...]
    truncated_integral_estimate: complex
    real_standard_error_across_replicates: float
    imaginary_standard_error_across_replicates: float
    contribution_magnitude_effective_sample_size: float
    largest_contribution_magnitude_fraction: float
    average_phase_ratio: complex
    orientation_degenerate_sample_count: int
    minimum_absolute_normalized_orientation_determinant: float
    proposal_density_exactly_accounted_in_weights: bool
    empirical_truncated_importance_estimate_materialized: bool
    finite_importance_variance_proved: bool
    radial_tail_bound_proved: bool
    noncompact_haar_integral_evaluated: bool
    status: str
    claim_ceiling: str = 'TRUNCATED_EMPIRICAL_CRITICAL_ROTATION_IMPORTANCE_ONLY'


def truncated_sinh_squared_radial_volume(radial_cutoff: float) -> float:
    '''Return integral_0^R sinh(r)^2 dr.'''

    if not math.isfinite(radial_cutoff) or radial_cutoff <= 0.0:
        raise ValueError('radial_cutoff must be finite and positive')
    return math.sinh(2.0 * radial_cutoff) / 4.0 - radial_cutoff / 2.0


def _radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(indices.shape, dtype=float)
    remainder = indices.astype(np.int64, copy=True)
    factor = 1.0 / base
    while np.any(remainder > 0):
        values += factor * (remainder % base)
        remainder //= base
        factor /= base
    return values


def shifted_halton_points(
    sample_count: int,
    dimension: int,
    *,
    seed: int,
) -> np.ndarray:
    '''Return one Cranley--Patterson shifted Halton point set.'''

    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError('sample_count must be a positive integer')
    if type(dimension) is not int or not 1 <= dimension <= len(_FIRST_24_PRIMES):
        raise ValueError('dimension must be between one and twenty-four')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    indices = np.arange(1, sample_count + 1, dtype=np.int64)
    base_points = np.column_stack(
        [_radical_inverse(indices, base) for base in _FIRST_24_PRIMES[:dimension]]
    )
    shifts = np.random.default_rng(seed).random(dimension)
    return np.mod(base_points + shifts, 1.0)


def inverse_truncated_sinh_squared_cdf(
    probabilities: np.ndarray,
    radial_cutoff: float,
) -> np.ndarray:
    '''Invert the normalized sinh(r)^2 density on [0,R] by bisection.'''

    probability = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability >= 1.0)
    ):
        raise ValueError('probabilities must lie in [0,1)')
    normalization = truncated_sinh_squared_radial_volume(radial_cutoff)
    lower = np.zeros_like(probability)
    upper = np.full_like(probability, radial_cutoff)
    target = probability * normalization
    for _ in range(64):
        middle = (lower + upper) / 2.0
        value = np.sinh(2.0 * middle) / 4.0 - middle / 2.0
        lower = np.where(value < target, middle, lower)
        upper = np.where(value >= target, middle, upper)
    return (lower + upper) / 2.0


def su2_from_unit_cube(coordinates: np.ndarray) -> np.ndarray:
    '''Map three uniform coordinates to normalized SU(2) Haar matrices.'''

    point = np.asarray(coordinates, dtype=float)
    if point.shape != (3,) or np.any(~np.isfinite(point)) or np.any(
        (point < 0.0) | (point >= 1.0)
    ):
        raise ValueError('coordinates must be three values in [0,1)')
    first, second, third = point
    q0 = math.sqrt(1.0 - first) * math.sin(2.0 * math.pi * second)
    q1 = math.sqrt(1.0 - first) * math.cos(2.0 * math.pi * second)
    q2 = math.sqrt(first) * math.sin(2.0 * math.pi * third)
    q3 = math.sqrt(first) * math.cos(2.0 * math.pi * third)
    a_value = complex(q0, q1)
    b_value = complex(q2, q3)
    return np.asarray(
        ((a_value, b_value), (-np.conjugate(b_value), np.conjugate(a_value))),
        dtype=complex,
    )


def sl2c_polar_element(
    radial: float,
    direction: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    '''Return U exp(r n.sigma/2) in SL(2,C).'''

    n_value = np.asarray(direction, dtype=float)
    u_value = np.asarray(rotation, dtype=complex)
    if not math.isfinite(radial) or radial < 0.0:
        raise ValueError('radial must be finite and nonnegative')
    if n_value.shape != (3,) or abs(float(n_value @ n_value) - 1.0) > 1.0e-10:
        raise ValueError('direction must be a unit three-vector')
    if u_value.shape != (2, 2) or abs(complex(np.linalg.det(u_value)) - 1.0) > 1.0e-10:
        raise ValueError('rotation must be an SU(2) matrix')
    n_sigma = np.asarray(
        (
            (n_value[2], complex(n_value[0], -n_value[1])),
            (complex(n_value[0], n_value[1]), -n_value[2]),
        ),
        dtype=complex,
    )
    boost = (
        math.cosh(radial / 2.0) * np.eye(2)
        + math.sinh(radial / 2.0) * n_sigma
    )
    return u_value @ boost


def _su2_cube_coordinates(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=complex)
    a_value = complex(matrix[0, 0])
    b_value = complex(matrix[0, 1])
    q0, q1 = a_value.real, a_value.imag
    q2, q3 = b_value.real, b_value.imag
    first = min(1.0, max(0.0, q2**2 + q3**2))
    second = math.atan2(q0, q1) / (2.0 * math.pi) % 1.0
    third = math.atan2(q2, q3) / (2.0 * math.pi) % 1.0
    return np.asarray((first, second, third))


def _right_polar_rotation(element: np.ndarray) -> np.ndarray:
    matrix = np.asarray(element, dtype=complex)
    positive_squared = np.conjugate(matrix.T) @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(positive_squared)
    positive = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ np.conjugate(eigenvectors.T)
    )
    rotation = matrix @ np.linalg.inv(positive)
    return rotation / np.sqrt(complex(np.linalg.det(rotation)))


def _beta_product_density(
    point: np.ndarray,
    center: np.ndarray,
    concentration: float,
) -> float:
    log_density = 0.0
    for value, mean in zip(point, center):
        alpha = 1.0 + concentration * mean
        beta = 1.0 + concentration * (1.0 - mean)
        if value <= 0.0:
            if alpha > 1.0:
                return 0.0
            log_value_term = 0.0
        else:
            log_value_term = (alpha - 1.0) * math.log(value)
        if value >= 1.0:
            if beta > 1.0:
                return 0.0
            log_complement_term = 0.0
        else:
            log_complement_term = (beta - 1.0) * math.log1p(-value)
        log_density += (
            log_value_term
            + log_complement_term
            - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
        )
    return math.exp(log_density)


def _sample_rotation_importance(
    rng: np.random.Generator,
    center: np.ndarray,
    mixture_weight: float,
    concentration: float,
) -> tuple[np.ndarray, float]:
    if rng.random() < mixture_weight:
        alpha = 1.0 + concentration * center
        beta = 1.0 + concentration * (1.0 - center)
        point = rng.beta(alpha, beta)
    else:
        point = rng.random(3)
    local_density = _beta_product_density(point, center, concentration)
    mixture_density = (1.0 - mixture_weight) + mixture_weight * local_density
    return su2_from_unit_cube(point), mixture_density


def _sample_radial_importance(
    rng: np.random.Generator,
    radial_cutoff: float,
    mixture_weight: float,
    origin_gamma_rate: float,
) -> tuple[float, float]:
    radial_volume = truncated_sinh_squared_radial_volume(radial_cutoff)
    scaled_cutoff = origin_gamma_rate * radial_cutoff
    if scaled_cutoff < 0.1:
        term = scaled_cutoff**3 / 6.0
        tail_series = term
        order = 3
        while abs(term) > 1.0e-18 * max(tail_series, 1.0e-300):
            order += 1
            term *= scaled_cutoff / order
            tail_series += term
        origin_normalization = math.exp(-scaled_cutoff) * tail_series
    else:
        origin_normalization = 1.0 - math.exp(-scaled_cutoff) * (
            1.0 + scaled_cutoff + scaled_cutoff**2 / 2.0
        )
    if rng.random() < mixture_weight:
        while True:
            radial = float(rng.gamma(shape=3.0, scale=1.0 / origin_gamma_rate))
            if radial <= radial_cutoff:
                break
    else:
        radial = float(
            inverse_truncated_sinh_squared_cdf(
                np.asarray((rng.random(),)), radial_cutoff
            )[0]
        )
    haar_density = math.sinh(radial) ** 2
    origin_density = (
        origin_gamma_rate**3
        * radial**2
        * math.exp(-origin_gamma_rate * radial)
        / 2.0
        / origin_normalization
    )
    normalized_haar_density = haar_density / radial_volume
    proposal_density = (
        (1.0 - mixture_weight) * normalized_haar_density
        + mixture_weight * origin_density
    )
    return radial, proposal_density


def _frames_from_unit_point(
    point: np.ndarray,
    labels: tuple[int, ...],
    root: int,
    radial_cutoff: float,
) -> dict[int, np.ndarray]:
    frames = {root: np.eye(2, dtype=complex)}
    radial_probabilities = point[0::6]
    radial_values = inverse_truncated_sinh_squared_cdf(
        radial_probabilities, radial_cutoff
    )
    for index, label in enumerate(item for item in labels if item != root):
        offset = 6 * index
        cos_theta = 2.0 * point[offset + 1] - 1.0
        azimuth = 2.0 * math.pi * point[offset + 2]
        sine_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
        direction = np.asarray(
            (
                sine_theta * math.cos(azimuth),
                sine_theta * math.sin(azimuth),
                cos_theta,
            )
        )
        rotation = su2_from_unit_cube(point[offset + 3 : offset + 6])
        frames[label] = sl2c_polar_element(
            float(radial_values[index]), direction, rotation
        )
    return frames


def certify_proper_vertex_truncated_haar_estimate(
    *,
    cell_index: int = 0,
    level: int = 3,
    gamma: float = 0.274,
    radial_cutoff: float = 1.0,
    sample_count_per_replicate: int = 32,
    replicate_count: int = 4,
    base_seed: int = 20260829,
    cp1_shape: tuple[int, int] = (8, 16),
    haar_normalization_c_h: float = 1.0,
) -> ProperVertexTruncatedHaarCertificate:
    '''Estimate the compact radial truncation of the gauge-fixed Haar integral.'''

    if type(replicate_count) is not int or replicate_count < 2:
        raise ValueError('replicate_count must be an integer at least two')
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError('base_seed must be a nonnegative integer')
    if len(cp1_shape) != 2:
        raise ValueError('cp1_shape must contain two grid dimensions')
    if not math.isfinite(haar_normalization_c_h) or haar_normalization_c_h <= 0.0:
        raise ValueError('haar_normalization_c_h must be finite and positive')
    kernel = certify_proper_vertex_single_cell_kernel(
        cell_index=cell_index, level=level, gamma=gamma
    )
    labels = tuple(sorted(label for label, _ in kernel.critical_point.gauge_fixed_frames))
    root = kernel.critical_point.root_omitted_vertex
    radial_volume = truncated_sinh_squared_radial_volume(radial_cutoff)
    one_group_volume = 4.0 * math.pi * haar_normalization_c_h * radial_volume
    four_group_volume = one_group_volume**4
    replicate_records: list[TruncatedHaarReplicate] = []
    all_coefficients: list[complex] = []
    total_degenerate = 0
    minimum_orientation_margin = math.inf
    all_recomputed = True
    for replicate in range(replicate_count):
        seed = base_seed + 104729 * replicate
        points = shifted_halton_points(
            sample_count_per_replicate, 24, seed=seed
        )
        coefficients: list[complex] = []
        degenerate_samples = 0
        for point in points:
            frames = _frames_from_unit_point(point, labels, root, radial_cutoff)
            evaluation = evaluate_proper_vertex_coefficient_at_frames(
                frames=frames,
                kernel_contract=kernel,
                number_u=cp1_shape[0],
                number_phi=cp1_shape[1],
            )
            coefficients.append(evaluation.coefficient_product)
            minimum_orientation_margin = min(
                minimum_orientation_margin,
                evaluation.minimum_absolute_normalized_orientation_determinant,
            )
            if evaluation.orientation_degenerate_face_count > 0:
                degenerate_samples += 1
            all_recomputed = all_recomputed and (
                evaluation.orientation_degenerate_face_count == 0
                or evaluation.coefficient_product == 0.0j
            )
        coefficient_array = np.asarray(coefficients)
        coefficient_mean = complex(np.mean(coefficient_array))
        absolute_mean = float(np.mean(np.abs(coefficient_array)))
        replicate_records.append(
            TruncatedHaarReplicate(
                replicate_index=replicate,
                random_shift_seed=seed,
                sample_count=sample_count_per_replicate,
                orientation_degenerate_sample_count=degenerate_samples,
                coefficient_mean=coefficient_mean,
                absolute_coefficient_mean=absolute_mean,
                truncated_integral_estimate=four_group_volume * coefficient_mean,
            )
        )
        all_coefficients.extend(coefficients)
        total_degenerate += degenerate_samples

    replicate_estimates = np.asarray(
        [item.truncated_integral_estimate for item in replicate_records]
    )
    estimate = complex(np.mean(replicate_estimates))
    real_error = float(
        np.std(replicate_estimates.real, ddof=1) / math.sqrt(replicate_count)
    )
    imaginary_error = float(
        np.std(replicate_estimates.imag, ddof=1) / math.sqrt(replicate_count)
    )
    coefficient_array = np.asarray(all_coefficients)
    absolute_mean = float(np.mean(np.abs(coefficient_array)))
    coefficient_mean = complex(np.mean(coefficient_array))
    average_phase = (
        coefficient_mean / absolute_mean if absolute_mean > 0.0 else 0.0j
    )
    magnitudes = np.abs(coefficient_array)
    magnitude_sum = float(np.sum(magnitudes))
    squared_magnitude_sum = float(np.sum(magnitudes**2))
    magnitude_ess = (
        magnitude_sum**2 / squared_magnitude_sum
        if squared_magnitude_sum > 0.0
        else 0.0
    )
    largest_fraction = (
        float(np.max(magnitudes)) / magnitude_sum
        if magnitude_sum > 0.0
        else 0.0
    )
    finite_output = all(
        math.isfinite(value)
        for value in (
            estimate.real,
            estimate.imag,
            real_error,
            imaginary_error,
            absolute_mean,
        )
    )
    return ProperVertexTruncatedHaarCertificate(
        cell=kernel.cell,
        root_omitted_vertex=root,
        haar_normalization_c_h=haar_normalization_c_h,
        su2_haar_normalized_to_one=True,
        sphere_area_normalization=4.0 * math.pi,
        radial_cutoff=radial_cutoff,
        one_group_radial_volume=radial_volume,
        one_group_truncated_haar_volume=one_group_volume,
        four_group_truncated_haar_volume=four_group_volume,
        sample_count_per_replicate=sample_count_per_replicate,
        replicate_count=replicate_count,
        cp1_quadrature_shape=cp1_shape,
        replicates=tuple(replicate_records),
        total_orientation_degenerate_sample_count=total_degenerate,
        minimum_absolute_normalized_orientation_determinant=(
            minimum_orientation_margin
        ),
        truncated_integral_estimate=estimate,
        real_standard_error_across_replicates=real_error,
        imaginary_standard_error_across_replicates=imaginary_error,
        mean_absolute_coefficient=absolute_mean,
        average_phase_ratio=average_phase,
        empirical_coefficient_second_moment=float(
            np.mean(magnitudes**2)
        ),
        magnitude_effective_sample_size=magnitude_ess,
        largest_magnitude_fraction=largest_fraction,
        polar_haar_measure_materialized=True,
        all_samples_recomputed_full_eq53_projectors=all_recomputed,
        empirical_truncated_integral_estimated=finite_output,
        rigorous_qmc_error_bound_proved=False,
        finite_importance_variance_proved=False,
        radial_tail_bound_proved=False,
        noncompact_haar_integral_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'TRUNCATED_PROPER_VERTEX_HAAR_ESTIMATE_MATERIALIZED'
            if finite_output
            else 'TRUNCATED_PROPER_VERTEX_HAAR_ESTIMATE_FAILED'
        ),
    )


def certify_proper_vertex_truncated_importance_estimate(
    *,
    cell_index: int = 0,
    level: int = 3,
    gamma: float = 0.274,
    radial_cutoff: float = 1.0,
    sample_count_per_replicate: int = 64,
    replicate_count: int = 4,
    base_seed: int = 20260829,
    cp1_shape: tuple[int, int] = (8, 16),
    radial_near_origin_mixture_weight: float = 0.9,
    radial_origin_gamma_rate: float = 12.0,
    rotation_critical_mixture_weight: float = 0.9,
    rotation_beta_concentration: float = 40.0,
    haar_normalization_c_h: float = 1.0,
) -> ProperVertexTruncatedImportanceCertificate:
    '''Importance-sample the same radial truncation around critical rotations.'''

    if type(sample_count_per_replicate) is not int or sample_count_per_replicate <= 0:
        raise ValueError('sample_count_per_replicate must be positive')
    if type(replicate_count) is not int or replicate_count < 2:
        raise ValueError('replicate_count must be at least two')
    for name, value in (
        ('radial_near_origin_mixture_weight', radial_near_origin_mixture_weight),
        ('rotation_critical_mixture_weight', rotation_critical_mixture_weight),
    ):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f'{name} must lie in [0,1)')
    if not math.isfinite(rotation_beta_concentration) or rotation_beta_concentration <= 0.0:
        raise ValueError('rotation_beta_concentration must be finite and positive')
    if not math.isfinite(radial_origin_gamma_rate) or radial_origin_gamma_rate <= 0.0:
        raise ValueError('radial_origin_gamma_rate must be finite and positive')
    kernel = certify_proper_vertex_single_cell_kernel(
        cell_index=cell_index, level=level, gamma=gamma
    )
    critical_frames = dict(kernel.critical_point.gauge_fixed_frames)
    labels = tuple(sorted(critical_frames))
    root = kernel.critical_point.root_omitted_vertex
    nonroot = tuple(label for label in labels if label != root)
    rotation_centers = {
        label: _su2_cube_coordinates(_right_polar_rotation(critical_frames[label]))
        for label in nonroot
    }
    all_contributions: list[complex] = []
    replicate_estimates: list[complex] = []
    degenerate_samples = 0
    minimum_orientation_margin = math.inf
    for replicate in range(replicate_count):
        rng = np.random.default_rng(base_seed + 104729 * replicate)
        contributions: list[complex] = []
        for _ in range(sample_count_per_replicate):
            frames = {root: np.eye(2, dtype=complex)}
            weight = 1.0
            for label in nonroot:
                radial, radial_proposal = _sample_radial_importance(
                    rng,
                    radial_cutoff,
                    radial_near_origin_mixture_weight,
                    radial_origin_gamma_rate,
                )
                cos_theta = 2.0 * rng.random() - 1.0
                azimuth = 2.0 * math.pi * rng.random()
                sine_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
                direction = np.asarray(
                    (
                        sine_theta * math.cos(azimuth),
                        sine_theta * math.sin(azimuth),
                        cos_theta,
                    )
                )
                rotation, rotation_proposal = _sample_rotation_importance(
                    rng,
                    rotation_centers[label],
                    rotation_critical_mixture_weight,
                    rotation_beta_concentration,
                )
                frames[label] = sl2c_polar_element(radial, direction, rotation)
                weight *= (
                    haar_normalization_c_h
                    * 4.0
                    * math.pi
                    * math.sinh(radial) ** 2
                    / radial_proposal
                    / rotation_proposal
                )
            evaluation = evaluate_proper_vertex_coefficient_at_frames(
                frames=frames,
                kernel_contract=kernel,
                number_u=cp1_shape[0],
                number_phi=cp1_shape[1],
            )
            minimum_orientation_margin = min(
                minimum_orientation_margin,
                evaluation.minimum_absolute_normalized_orientation_determinant,
            )
            if evaluation.orientation_degenerate_face_count > 0:
                degenerate_samples += 1
            contributions.append(weight * evaluation.coefficient_product)
        estimate = complex(np.mean(contributions))
        replicate_estimates.append(estimate)
        all_contributions.extend(contributions)

    estimates = np.asarray(replicate_estimates)
    contributions_array = np.asarray(all_contributions)
    estimate = complex(np.mean(estimates))
    magnitudes = np.abs(contributions_array)
    magnitude_sum = float(np.sum(magnitudes))
    squared_sum = float(np.sum(magnitudes**2))
    ess = magnitude_sum**2 / squared_sum if squared_sum > 0.0 else 0.0
    largest_fraction = (
        float(np.max(magnitudes)) / magnitude_sum if magnitude_sum > 0.0 else 0.0
    )
    mean_abs = float(np.mean(magnitudes))
    phase_ratio = (
        complex(np.mean(contributions_array)) / mean_abs if mean_abs > 0.0 else 0.0j
    )
    finite = np.all(np.isfinite(contributions_array))
    return ProperVertexTruncatedImportanceCertificate(
        cell=kernel.cell,
        radial_cutoff=radial_cutoff,
        haar_normalization_c_h=haar_normalization_c_h,
        radial_near_origin_mixture_weight=radial_near_origin_mixture_weight,
        radial_origin_gamma_rate=radial_origin_gamma_rate,
        rotation_critical_mixture_weight=rotation_critical_mixture_weight,
        rotation_beta_concentration=rotation_beta_concentration,
        sample_count_per_replicate=sample_count_per_replicate,
        replicate_count=replicate_count,
        cp1_quadrature_shape=cp1_shape,
        critical_rotation_centers=tuple(
            (label, tuple(float(value) for value in rotation_centers[label]))
            for label in nonroot
        ),
        replicate_estimates=tuple(replicate_estimates),
        truncated_integral_estimate=estimate,
        real_standard_error_across_replicates=float(
            np.std(estimates.real, ddof=1) / math.sqrt(replicate_count)
        ),
        imaginary_standard_error_across_replicates=float(
            np.std(estimates.imag, ddof=1) / math.sqrt(replicate_count)
        ),
        contribution_magnitude_effective_sample_size=ess,
        largest_contribution_magnitude_fraction=largest_fraction,
        average_phase_ratio=phase_ratio,
        orientation_degenerate_sample_count=degenerate_samples,
        minimum_absolute_normalized_orientation_determinant=(
            minimum_orientation_margin
        ),
        proposal_density_exactly_accounted_in_weights=True,
        empirical_truncated_importance_estimate_materialized=bool(finite),
        finite_importance_variance_proved=False,
        radial_tail_bound_proved=False,
        noncompact_haar_integral_evaluated=False,
        status=(
            'TRUNCATED_PROPER_VERTEX_IMPORTANCE_ESTIMATE_MATERIALIZED'
            if finite
            else 'TRUNCATED_PROPER_VERTEX_IMPORTANCE_ESTIMATE_FAILED'
        ),
    )
