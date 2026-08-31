"""Observable Fisher geometry and its spacetime-metric boundary.

This E30 certificate starts from the strictly positive conditional
probabilities visible in the E29 2x2 binary scenario.  A fixed context design
``pi_xy`` is counted once per context, not once per outcome cell, and defines

``ds^2 = sum_xy pi_xy sum_ab (dP_ab|xy)^2 / P_ab|xy``.

Pulling this Fisher--Rao form back through the E29 incidence map ``P=Mq``
annihilates exactly the seven signed-lift kernel directions.  The resulting
form is positive semidefinite on the global coordinates and positive
definite only after quotienting by that kernel and normalization.  It is an
information metric, not a Lorentzian spacetime metric.

The final conformal calculation is explicitly conditional.  If a Lorentzian
reference metric and an independent positive dimensionless volume ratio are
supplied, the E24 algebra fixes a conformal representative while preserving
its null cone.  This does not derive the volume law, curvature, Einstein
dynamics, gravity, or a physical meaning for signed weights.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.chsh_local_seed_obstruction import (
    CHSH_PATTERN,
    OUTCOMES,
    SETTINGS,
    isotropic_chsh_box,
)
from examples.physics.contextual_global_section_obstruction import (
    QUANTUM_ETA,
    deterministic_oriented_scores,
    exact_rational_rank,
    marginal_incidence_matrix,
    marginalize_global_weights,
    quantum_kernel_perturbed_extension,
    swap_opposite_score_weights,
    symmetric_signed_global_extension,
    walsh_kernel_vectors,
)
from examples.physics.quotient_causal_metric_reconstruction import volume_recovery


DEFAULT_TOLERANCE = 1.0e-12
UNIFORM_CONTEXT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def _positive_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def _context_weights(
    values: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[float, float, float, float]:
    tol = _positive_tolerance(tolerance)
    weights = tuple(float(value) for value in values)
    if len(weights) != 4 or not all(
        math.isfinite(value) and value > 0.0 for value in weights
    ):
        raise ValueError("context weights must be four finite positive values")
    if abs(math.fsum(weights) - 1.0) > tol:
        raise ValueError("context weights must sum to one")
    return weights  # type: ignore[return-value]


def _strict_probability_box(
    probabilities: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    tol = _positive_tolerance(tolerance)
    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    if float(box.min()) <= 0.0:
        raise ValueError("Fisher chart requires every probability cell to be positive")
    residual = max(
        abs(float(np.sum(box[x, y])) - 1.0) for x in SETTINGS for y in SETTINGS
    )
    if residual > tol:
        raise ValueError("each context probability box must sum to one")
    return np.array(box, copy=True)


def _conditional_tangent(
    differential: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    tol = _positive_tolerance(tolerance)
    tangent = np.asarray(differential, dtype=np.float64)
    if tangent.shape != (2, 2, 2, 2) or not np.isfinite(tangent).all():
        raise ValueError("differential must be finite with shape (2, 2, 2, 2)")
    residual = max(
        abs(float(np.sum(tangent[x, y]))) for x in SETTINGS for y in SETTINGS
    )
    if residual > tol:
        raise ValueError("each context differential must have zero sum")
    return np.array(tangent, copy=True)


def _expanded_context_weights(
    context_weights: Sequence[float], *, tolerance: float = DEFAULT_TOLERANCE
) -> np.ndarray:
    weights = _context_weights(context_weights, tolerance=tolerance)
    return np.repeat(np.asarray(weights, dtype=np.float64), 4)


def hellinger_coordinates(
    probabilities: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return ``Psi_xyab=2*sqrt(pi_xy*P_xyab)`` in the E30 convention."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    expanded = np.asarray(weights, dtype=np.float64).reshape(2, 2, 1, 1)
    return 2.0 * np.sqrt(expanded * box)


def hellinger_tangent(
    probabilities: np.ndarray,
    differential: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return ``dPsi=sqrt(pi_xy/P_xyab)*dP_xyab``."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    tangent = _conditional_tangent(differential, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    expanded = np.asarray(weights, dtype=np.float64).reshape(2, 2, 1, 1)
    return np.sqrt(expanded / box) * tangent


def conditional_fisher_quadratic(
    probabilities: np.ndarray,
    differential: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Evaluate the declared conditional Fisher--Rao quadratic form."""

    tangent_coordinates = hellinger_tangent(
        probabilities,
        differential,
        context_weights=context_weights,
        tolerance=tolerance,
    )
    return float(np.sum(tangent_coordinates * tangent_coordinates))


def product_fisher_rao_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return the product Fisher distance for fixed context weights.

    Each context simplex has distance ``2 acos(sum_ab sqrt(P_ab Q_ab))``;
    the four context distances are combined with the declared ``pi_xy``.
    """

    left = _strict_probability_box(first, tolerance=tolerance)
    right = _strict_probability_box(second, tolerance=tolerance)
    weights = _context_weights(context_weights, tolerance=tolerance)
    squared = 0.0
    for context_index, (x, y) in enumerate(
        (pair for pair in ((0, 0), (0, 1), (1, 0), (1, 1)))
    ):
        coefficient = float(np.sum(np.sqrt(left[x, y] * right[x, y])))
        coefficient = min(1.0, max(0.0, coefficient))
        context_distance = 2.0 * math.acos(coefficient)
        squared += weights[context_index] * context_distance * context_distance
    return math.sqrt(max(0.0, squared))


def fisher_weight_matrix(
    probabilities: np.ndarray,
    *,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return ``diag(pi_xy/P_xyab)`` in fixed row order."""

    box = _strict_probability_box(probabilities, tolerance=tolerance)
    expanded = _expanded_context_weights(context_weights, tolerance=tolerance)
    return np.diag(expanded / box.reshape(-1))


def fisher_pullback_metric(
    probabilities: np.ndarray,
    *,
    incidence: np.ndarray | None = None,
    context_weights: Sequence[float] = UNIFORM_CONTEXT_WEIGHTS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return ``M.T @ diag(pi/P) @ M`` for a declared linear map.

    Shape validation makes the algebra fail closed, but does not certify that
    a caller-supplied matrix is a physical marginal-incidence map.  The exact
    E29 rank and kernel claims in :func:`certificate` use only the canonical
    ``marginal_incidence_matrix()``.
    """

    matrix = (
        marginal_incidence_matrix()
        if incidence is None
        else np.asarray(incidence, dtype=np.float64)
    )
    if matrix.ndim != 2 or matrix.shape[0] != 16 or not np.isfinite(matrix).all():
        raise ValueError("incidence must be finite with sixteen context-cell rows")
    weight = fisher_weight_matrix(
        probabilities, context_weights=context_weights, tolerance=tolerance
    )
    metric = matrix.T @ weight @ matrix
    return 0.5 * (metric + metric.T)


def normalized_atom_tangent_basis() -> np.ndarray:
    """Return a 16x15 basis whose columns sum to zero."""

    basis = np.zeros((16, 15), dtype=np.float64)
    for column in range(15):
        basis[column, column] = 1.0
        basis[15, column] = -1.0
    return basis


def matrix_inertia(
    matrix: np.ndarray, *, tolerance: float = 1.0e-10
) -> tuple[int, int, int]:
    """Return counts of positive, negative, and zero eigenvalues."""

    tol = _positive_tolerance(tolerance)
    values = np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("matrix must be finite, nonempty, and square")
    if not np.allclose(values, values.T, atol=tol, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(values)
    positive = int(np.sum(eigenvalues > tol))
    negative = int(np.sum(eigenvalues < -tol))
    zero = len(eigenvalues) - positive - negative
    return positive, negative, zero


def context_block_permutation(order: Sequence[int]) -> np.ndarray:
    """Return a row permutation for four contiguous context blocks."""

    declared = tuple(order)
    if (
        len(declared) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in declared)
        or set(declared) != {0, 1, 2, 3}
    ):
        raise ValueError("context order must be a permutation of (0,1,2,3)")
    permutation = np.zeros((16, 16), dtype=np.float64)
    for new_context, old_context in enumerate(declared):
        for cell in range(4):
            permutation[4 * new_context + cell, 4 * old_context + cell] = 1.0
    return permutation


def atom_permutation_matrix(order: Sequence[int]) -> np.ndarray:
    """Return a 16x16 atom-coordinate permutation matrix."""

    declared = tuple(order)
    if (
        len(declared) != 16
        or any(isinstance(value, bool) or not isinstance(value, int) for value in declared)
        or set(declared) != set(range(16))
    ):
        raise ValueError("atom order must be a permutation of range(16)")
    permutation = np.zeros((16, 16), dtype=np.float64)
    for new_index, old_index in enumerate(declared):
        permutation[new_index, old_index] = 1.0
    return permutation


def isotropic_fisher_component(eta: float) -> float:
    """Return ``g_etaeta=1/(1-eta^2)`` on ``0 <= eta < 1``."""

    parameter = float(eta)
    if not math.isfinite(parameter) or not 0.0 <= parameter < 1.0:
        raise ValueError("eta must be finite and lie in [0, 1) for the strict chart")
    return 1.0 / (1.0 - parameter * parameter)


def isotropic_fisher_distance(first_eta: float, second_eta: float) -> float:
    """Return the exact isotropic distance ``|asin(eta2)-asin(eta1)|``."""

    first = float(first_eta)
    second = float(second_eta)
    isotropic_fisher_component(first)
    isotropic_fisher_component(second)
    return abs(math.asin(second) - math.asin(first))


def _isotropic_tangent() -> np.ndarray:
    tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for x in SETTINGS:
        for y in SETTINGS:
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    tangent[x, y, a_index, b_index] = (
                        0.25 * a * b * CHSH_PATTERN[x, y]
                    )
    return tangent


def lorentzian_signature(
    metric: np.ndarray, *, tolerance: float = 1.0e-10
) -> tuple[int, int, int]:
    """Return inertia after symmetric validation; convention is one negative."""

    return matrix_inertia(metric, tolerance=tolerance)


def conditional_conformal_metric(
    reference_metric: np.ndarray,
    volume_ratio: float,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Apply the supplied E24 conformal-volume algebra.

    ``g = v^(2/d) g0`` is returned only after ``g0`` is checked to have one
    negative and ``d-1`` positive eigenvalues.  The function does not derive
    either input.
    """

    tol = _positive_tolerance(tolerance)
    reference = np.asarray(reference_metric, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[0] != reference.shape[1]
        or reference.shape[0] < 2
        or not np.isfinite(reference).all()
    ):
        raise ValueError("reference metric must be a finite square matrix of dimension >=2")
    if not np.allclose(reference, reference.T, atol=tol, rtol=0.0):
        raise ValueError("reference metric must be symmetric")
    dimension = reference.shape[0]
    positive, negative, zero = lorentzian_signature(reference, tolerance=tol)
    if (positive, negative, zero) != (dimension - 1, 1, 0):
        raise ValueError("reference metric must have one-negative Lorentzian signature")
    ratio = float(volume_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("volume_ratio must be finite and positive")
    conformal_factor = volume_recovery(ratio, n=dimension)
    return conformal_factor * conformal_factor * reference


def metric_volume_ratio(
    metric: np.ndarray,
    reference_metric: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> float:
    """Return ``sqrt(|det g|/|det g0|)`` for symmetric nondegenerate metrics."""

    tol = _positive_tolerance(tolerance)
    current = np.asarray(metric, dtype=np.float64)
    reference = np.asarray(reference_metric, dtype=np.float64)
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape != reference.shape
        or not np.isfinite(current).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("metric and reference_metric must be finite square matrices of equal shape")
    if not np.allclose(current, current.T, atol=tol, rtol=0.0) or not np.allclose(
        reference, reference.T, atol=tol, rtol=0.0
    ):
        raise ValueError("metric and reference_metric must be symmetric")
    denominator = abs(float(np.linalg.det(reference)))
    numerator = abs(float(np.linalg.det(current)))
    if denominator <= tol or numerator <= tol:
        raise ValueError("metric determinants must be nonzero")
    return math.sqrt(numerator / denominator)


@dataclass(frozen=True)
class HighFrequencyVolumeWitness:
    n: int
    minimum_volume_ratio: float
    uniform_value_residual_bound: float
    probe_time: float
    probe_value_residual: float
    probe_first_derivative: float
    probe_second_derivative: float


def high_frequency_volume_witness(n: int) -> HighFrequencyVolumeWitness:
    """Return a ``v_n`` witness for uniform convergence without C2 convergence."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer of at least two")
    frequency = float(n * n)
    amplitude = 1.0 / frequency
    probe_time = math.pi / (2.0 * frequency)
    phase = frequency * probe_time
    return HighFrequencyVolumeWitness(
        n=n,
        minimum_volume_ratio=1.0 - amplitude,
        uniform_value_residual_bound=amplitude,
        probe_time=probe_time,
        probe_value_residual=amplitude * math.sin(phase),
        probe_first_derivative=math.cos(phase),
        probe_second_derivative=-frequency * math.sin(phase),
    )


@dataclass(frozen=True)
class RepresentationInvariantMeasureCertificate:
    context_weights: tuple[float, float, float, float]
    target_minimum_probability: float
    context_normalization_residual: float
    hellinger_coordinate_norm_squared: float
    hellinger_quadratic_residual: float
    incidence_rank: int
    incidence_nullity: int
    pullback_rank: int
    pullback_inertia: tuple[int, int, int]
    normalized_tangent_rank: int
    normalized_tangent_inertia: tuple[int, int, int]
    maximum_incidence_kernel_residual: float
    maximum_pullback_kernel_residual: float
    q_delta_probability_residual: float
    q_delta_pullback_residual: float
    simultaneous_relabel_probability_residual: float
    simultaneous_relabel_congruence_residual: float
    general_relabel_fixed_incidence_residual: float
    fixed_nonuniform_context_swap_residual: float
    co_transformed_context_swap_residual: float
    uniform_context_swap_residual: float
    atom_only_probability_residual: float
    atom_only_fixed_incidence_residual: float
    quantum_isotropic_component: float
    quantum_isotropic_coordinate: float
    analytic_quantum_distance: float
    product_quantum_distance: float
    isotropic_distance_residual: float
    reference_signature: tuple[int, int, int]
    conformal_signature: tuple[int, int, int]
    supplied_volume_ratio: float
    recovered_volume_ratio: float
    conformal_volume_residual: float
    null_vector_reference_residual: float
    null_vector_conformal_residual: float
    unit_volume_reference_residual: float
    high_frequency_uniform_residual_bound: float
    high_frequency_second_derivative_magnitude: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the certificate without adding physical interpretation."""

        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> RepresentationInvariantMeasureCertificate:
    """Build the E30 observable-information and conformal-boundary certificate."""

    tol = _positive_tolerance(tolerance)
    weights = _context_weights(UNIFORM_CONTEXT_WEIGHTS, tolerance=tol)
    target = isotropic_chsh_box(QUANTUM_ETA)
    target = _strict_probability_box(target, tolerance=tol)
    normalization_residual = max(
        abs(float(np.sum(target[x, y])) - 1.0)
        for x in SETTINGS
        for y in SETTINGS
    )

    tangent = _isotropic_tangent()
    quadratic = conditional_fisher_quadratic(
        target, tangent, context_weights=weights, tolerance=tol
    )
    dpsi = hellinger_tangent(
        target, tangent, context_weights=weights, tolerance=tol
    )
    psi = hellinger_coordinates(target, context_weights=weights, tolerance=tol)

    incidence = marginal_incidence_matrix().astype(np.float64)
    exact_rank = exact_rational_rank(incidence.astype(np.int64))
    pullback = fisher_pullback_metric(
        target, incidence=incidence, context_weights=weights, tolerance=tol
    )
    inertia = matrix_inertia(pullback)
    pullback_rank = inertia[0] + inertia[1]
    tangent_basis = normalized_atom_tangent_basis()
    normalized_metric = tangent_basis.T @ pullback @ tangent_basis
    normalized_inertia = matrix_inertia(normalized_metric)
    normalized_rank = normalized_inertia[0] + normalized_inertia[1]
    kernel_vectors = walsh_kernel_vectors()
    incidence_kernel_residual = max(
        float(np.max(np.abs(incidence @ np.asarray(vector, dtype=np.float64))))
        for vector in kernel_vectors.values()
    )
    pullback_kernel_residual = max(
        float(np.max(np.abs(pullback @ np.asarray(vector, dtype=np.float64))))
        for vector in kernel_vectors.values()
    )

    base_q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    delta_q = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_box = marginalize_global_weights(base_q)
    delta_box = marginalize_global_weights(delta_q)
    base_metric = fisher_pullback_metric(base_box, tolerance=tol)
    delta_metric = fisher_pullback_metric(delta_box, tolerance=tol)

    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    atom_permutation = atom_permutation_matrix(tuple(reversed(range(16))))
    relabelled_q = atom_permutation @ base_q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_vector = relabelled_incidence @ relabelled_q
    expected_relabelled_vector = row_permutation @ base_box.reshape(-1)
    relabelled_box = expected_relabelled_vector.reshape(2, 2, 2, 2)
    relabelled_weights = tuple(weights[index] for index in context_order)
    relabelled_metric = fisher_pullback_metric(
        relabelled_box,
        incidence=relabelled_incidence,
        context_weights=relabelled_weights,
        tolerance=tol,
    )
    expected_congruence = atom_permutation @ base_metric @ atom_permutation.T
    general_relabel_fixed_incidence_residual = float(
        np.max(np.abs(relabelled_incidence - incidence))
    )

    nonuniform_weights = (0.4, 0.3, 0.2, 0.1)
    one_context_tangent = np.zeros((2, 2, 2, 2), dtype=np.float64)
    one_context_tangent[0, 0] = tangent[0, 0]
    swapped_box = (row_permutation @ target.reshape(-1)).reshape(2, 2, 2, 2)
    swapped_tangent = (
        row_permutation @ one_context_tangent.reshape(-1)
    ).reshape(2, 2, 2, 2)
    original_nonuniform_quadratic = conditional_fisher_quadratic(
        target, one_context_tangent, context_weights=nonuniform_weights, tolerance=tol
    )
    fixed_nonuniform_quadratic = conditional_fisher_quadratic(
        swapped_box,
        swapped_tangent,
        context_weights=nonuniform_weights,
        tolerance=tol,
    )
    co_transformed_weights = tuple(
        nonuniform_weights[index] for index in context_order
    )
    co_transformed_quadratic = conditional_fisher_quadratic(
        swapped_box,
        swapped_tangent,
        context_weights=co_transformed_weights,
        tolerance=tol,
    )
    original_uniform_quadratic = conditional_fisher_quadratic(
        target, one_context_tangent, context_weights=weights, tolerance=tol
    )
    swapped_uniform_quadratic = conditional_fisher_quadratic(
        swapped_box, swapped_tangent, context_weights=weights, tolerance=tol
    )

    scores = deterministic_oriented_scores()
    atom_only_order = list(range(16))
    negative_index = scores.index(-2)
    positive_index = scores.index(2)
    atom_only_order[negative_index], atom_only_order[positive_index] = (
        atom_only_order[positive_index],
        atom_only_order[negative_index],
    )
    atom_only_permutation = atom_permutation_matrix(atom_only_order)
    permuted_q = swap_opposite_score_weights(base_q)
    if not np.array_equal(atom_only_permutation @ base_q, np.asarray(permuted_q)):
        raise AssertionError("declared atom-only permutation disagrees with E29 witness")
    atom_only_box = marginalize_global_weights(permuted_q)
    atom_only_residual = float(np.max(np.abs(atom_only_box - target)))
    atom_only_fixed_incidence_residual = float(
        np.max(np.abs(incidence @ atom_only_permutation.T - incidence))
    )

    zero_box = isotropic_chsh_box(0.0)
    analytic_distance = isotropic_fisher_distance(0.0, QUANTUM_ETA)
    product_distance = product_fisher_rao_distance(
        zero_box, target, context_weights=weights, tolerance=tol
    )

    reference_metric = np.diag((-1.0, 1.0, 1.0, 1.0))
    supplied_ratio = 16.0
    conformal_metric = conditional_conformal_metric(reference_metric, supplied_ratio)
    recovered_ratio = metric_volume_ratio(conformal_metric, reference_metric)
    null_vector = np.array((1.0, 1.0, 0.0, 0.0), dtype=np.float64)
    null_reference = float(null_vector @ reference_metric @ null_vector)
    null_conformal = float(null_vector @ conformal_metric @ null_vector)
    unit_metric = conditional_conformal_metric(reference_metric, 1.0)
    high_frequency = high_frequency_volume_witness(100)

    numerical_limit = 100.0 * tol
    pullback_kernel_certified = (
        exact_rank == 9
        and inertia == (9, 0, 7)
        and incidence_kernel_residual <= numerical_limit
        and pullback_kernel_residual <= numerical_limit
    )
    normalized_quotient_certified = normalized_inertia == (8, 0, 7)
    q_delta_invariant = (
        float(np.max(np.abs(delta_box - base_box))) <= numerical_limit
        and float(np.max(np.abs(delta_metric - base_metric))) <= numerical_limit
    )
    simultaneous_relabel_certified = (
        float(np.max(np.abs(relabelled_vector - expected_relabelled_vector)))
        <= numerical_limit
        and float(np.max(np.abs(relabelled_metric - expected_congruence)))
        <= numerical_limit
    )
    context_weight_boundary_certified = (
        abs(fixed_nonuniform_quadratic - original_nonuniform_quadratic)
        > numerical_limit
        and abs(co_transformed_quadratic - original_nonuniform_quadratic)
        <= numerical_limit
        and abs(swapped_uniform_quadratic - original_uniform_quadratic)
        <= numerical_limit
    )
    isotropic_certified = (
        abs(quadratic - isotropic_fisher_component(QUANTUM_ETA))
        <= numerical_limit
        and abs(analytic_distance - math.pi / 4.0) <= numerical_limit
        and abs(product_distance - analytic_distance) <= numerical_limit
    )
    conformal_control_certified = (
        lorentzian_signature(reference_metric) == (3, 1, 0)
        and lorentzian_signature(conformal_metric) == (3, 1, 0)
        and abs(recovered_ratio - supplied_ratio) <= numerical_limit
        and abs(null_reference) <= numerical_limit
        and abs(null_conformal) <= numerical_limit
        and float(np.max(np.abs(unit_metric - reference_metric))) <= numerical_limit
    )

    dimensions = {
        "probabilities_signed_coordinates_and_context_weights_dimensionless": True,
        "fisher_line_element_and_metric_coefficients_dimensionless_here": True,
        "eta_and_chi_dimensionless": True,
        "volume_ratio_and_conformal_factor_dimensionless": True,
        "reference_metric_supplies_any_physical_length_convention": True,
        "dimensionless_information_distance_is_not_spacetime_length": True,
    }
    accounting = {
        "context_weights_sum_to_one": abs(math.fsum(weights) - 1.0) <= tol,
        "each_context_counted_once_not_once_per_outcome_cell": True,
        "fisher_metric_uses_positive_visible_probabilities_only": True,
        "signed_q_is_not_inserted_as_a_probability": True,
        "signed_q_absolute_q_and_fisher_not_added_as_energy_or_stress": True,
        "supplied_volume_ratio_is_not_derived_from_fisher_or_q": True,
        "probability_energy_or_volume_double_counted": False,
    }
    boundaries = {
        "uniform_context_weights_are_a_symmetry_axiom": True,
        "nonuniform_weights_must_cotransform_with_context_labels": True,
        "zero_probability_cells_are_outside_strict_fisher_chart": True,
        "eta_one_is_a_metric_completion_boundary_not_an_interior_point": True,
        "kernel_rank_eight_is_not_spacetime_dimension": True,
        "same_fisher_metric_does_not_select_a_hidden_signed_lift": True,
        "general_coordinate_relabel_changes_incidence_unless_automorphism": True,
        "fixed_incidence_automorphism_requires_rmc_inverse_equals_m": True,
        "caller_supplied_incidence_shape_is_not_physical_validation": True,
        "fisher_psd_no_go_is_not_a_general_lorentz_geometry_no_go": True,
        "conformal_control_reuses_supplied_e24_inputs": True,
        "pointwise_or_uniform_v_to_one_does_not_control_curvature": True,
        "c2_source_action_and_field_equations_still_required": True,
    }
    alternatives = {
        "operational_fisher_geometry_with_cotransformed_design": True,
        "monotone_or_quantum_fisher_operational_state_metric": True,
        "kraus_refinement_invariant_record_algebra": True,
        "independent_lorentz_metric_volume_and_covariant_action": True,
        "causal_set_or_eps_continuum_bridge": True,
    }
    status = {
        "hellinger_factor_convention_certified": (
            abs(quadratic - float(np.sum(dpsi * dpsi))) <= numerical_limit
        ),
        "pullback_rank_kernel_certified": pullback_kernel_certified,
        "normalized_quotient_rank_eight_certified": normalized_quotient_certified,
        "signed_lift_kernel_fisher_invariance_certified": q_delta_invariant,
        "simultaneous_relabel_congruence_certified": simultaneous_relabel_certified,
        "chosen_general_relabel_is_not_fixed_incidence_automorphism": (
            general_relabel_fixed_incidence_residual > numerical_limit
        ),
        "context_weight_symmetry_boundary_certified": (
            context_weight_boundary_certified
        ),
        "atom_only_fixed_incidence_automorphism_excluded": (
            atom_only_residual > numerical_limit
            and atom_only_fixed_incidence_residual > numerical_limit
        ),
        "isotropic_fisher_chart_certified": isotropic_certified,
        "fisher_form_positive_semidefinite": inertia[1] == 0,
        "fisher_metric_is_spacetime_lorentz_metric_derived": False,
        "lorentzian_signature_or_lightcone_derived_from_fisher": False,
        "supplied_conformal_volume_algebra_certified": conformal_control_certified,
        "physical_volume_law_derived": False,
        "curvature_einstein_dynamics_or_gravity_derived": False,
        "gr_c2_limit_derived": False,
        "parent_fixed_context_cptp_modified": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return RepresentationInvariantMeasureCertificate(
        context_weights=weights,
        target_minimum_probability=float(target.min()),
        context_normalization_residual=normalization_residual,
        hellinger_coordinate_norm_squared=float(np.sum(psi * psi)),
        hellinger_quadratic_residual=abs(quadratic - float(np.sum(dpsi * dpsi))),
        incidence_rank=exact_rank,
        incidence_nullity=16 - exact_rank,
        pullback_rank=pullback_rank,
        pullback_inertia=inertia,
        normalized_tangent_rank=normalized_rank,
        normalized_tangent_inertia=normalized_inertia,
        maximum_incidence_kernel_residual=incidence_kernel_residual,
        maximum_pullback_kernel_residual=pullback_kernel_residual,
        q_delta_probability_residual=float(np.max(np.abs(delta_box - base_box))),
        q_delta_pullback_residual=float(np.max(np.abs(delta_metric - base_metric))),
        simultaneous_relabel_probability_residual=float(
            np.max(np.abs(relabelled_vector - expected_relabelled_vector))
        ),
        simultaneous_relabel_congruence_residual=float(
            np.max(np.abs(relabelled_metric - expected_congruence))
        ),
        general_relabel_fixed_incidence_residual=(
            general_relabel_fixed_incidence_residual
        ),
        fixed_nonuniform_context_swap_residual=abs(
            fixed_nonuniform_quadratic - original_nonuniform_quadratic
        ),
        co_transformed_context_swap_residual=abs(
            co_transformed_quadratic - original_nonuniform_quadratic
        ),
        uniform_context_swap_residual=abs(
            swapped_uniform_quadratic - original_uniform_quadratic
        ),
        atom_only_probability_residual=atom_only_residual,
        atom_only_fixed_incidence_residual=atom_only_fixed_incidence_residual,
        quantum_isotropic_component=isotropic_fisher_component(QUANTUM_ETA),
        quantum_isotropic_coordinate=math.asin(QUANTUM_ETA),
        analytic_quantum_distance=analytic_distance,
        product_quantum_distance=product_distance,
        isotropic_distance_residual=abs(product_distance - analytic_distance),
        reference_signature=lorentzian_signature(reference_metric),
        conformal_signature=lorentzian_signature(conformal_metric),
        supplied_volume_ratio=supplied_ratio,
        recovered_volume_ratio=recovered_ratio,
        conformal_volume_residual=abs(recovered_ratio - supplied_ratio),
        null_vector_reference_residual=abs(null_reference),
        null_vector_conformal_residual=abs(null_conformal),
        unit_volume_reference_residual=float(
            np.max(np.abs(unit_metric - reference_metric))
        ),
        high_frequency_uniform_residual_bound=(
            high_frequency.uniform_value_residual_bound
        ),
        high_frequency_second_derivative_magnitude=abs(
            high_frequency.probe_second_derivative
        ),
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def run() -> dict[str, object]:
    """Return a JSON-serializable E30 certificate."""

    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
