"""Finite obstruction to reading raw Kraus multiplicity as metric volume.

An outcome operation is unchanged when its Kraus family is mixed by an
isometry.  In particular, one non-zero Kraus operator can be replaced by any
number of equal non-zero copies.  The hidden label count is therefore a
property of the representation, not of the quantum instrument.

This module is a finite certificate for that obstruction.  It does not derive
a physical record, spacetime volume, metric, stress tensor, or gravity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np


DEFAULT_TOLERANCE = 1.0e-12
I2 = np.eye(2, dtype=np.complex128)
P0 = np.diag([1.0, 0.0]).astype(np.complex128)
P1 = np.diag([0.0, 1.0]).astype(np.complex128)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_density(state: np.ndarray, *, tolerance: float) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.shape != (2, 2):
        raise ValueError("the finite witness requires a 2 by 2 density matrix")
    if not np.isfinite(density).all():
        raise ValueError("density matrix entries must be finite")
    if np.linalg.norm(density - density.conj().T, ord="fro") > tolerance:
        raise ValueError("density matrix must be Hermitian")
    if abs(float(np.trace(density).real) - 1.0) > tolerance:
        raise ValueError("density matrix must have unit trace")
    if abs(float(np.trace(density).imag)) > tolerance:
        raise ValueError("density matrix trace must be real")
    if float(np.linalg.eigvalsh(density).min()) < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")
    return density


def apply_cp_map(kraus: Sequence[np.ndarray], state: np.ndarray) -> np.ndarray:
    """Apply a finite Kraus family without exposing its internal label."""

    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    density = np.asarray(state, dtype=np.complex128)
    shape = density.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("state must be a square matrix")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must match the state dimension")
    if any(not np.isfinite(operator).all() for operator in operators):
        raise ValueError("Kraus operator entries must be finite")
    return sum(
        (operator @ density @ operator.conj().T for operator in operators),
        np.zeros(shape, dtype=np.complex128),
    )


def choi_matrix(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """Return ``sum |K>><<K|`` using column-major vectorisation."""

    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    shape = operators[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Kraus operators must be square matrices")
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must have one common shape")
    if any(not np.isfinite(operator).all() for operator in operators):
        raise ValueError("Kraus operator entries must be finite")
    size = shape[0] * shape[1]
    result = np.zeros((size, size), dtype=np.complex128)
    for operator in operators:
        vector = operator.reshape(-1, order="F")
        result += np.outer(vector, vector.conj())
    return result


def isometric_refinement(
    kraus: Sequence[np.ndarray],
    isometry: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, ...]:
    """Mix a Kraus family by ``u`` after checking ``u^dagger u = I``."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not kraus:
        raise ValueError("a Kraus family must be non-empty")
    operators = tuple(np.asarray(operator, dtype=np.complex128) for operator in kraus)
    shape = operators[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Kraus operators must be square matrices")
    if any(operator.shape != shape for operator in operators):
        raise ValueError("all Kraus operators must have one common shape")
    mixing = np.asarray(isometry, dtype=np.complex128)
    if mixing.ndim != 2 or mixing.shape[1] != len(operators):
        raise ValueError("isometry columns must equal the original Kraus count")
    if mixing.shape[0] < mixing.shape[1]:
        raise ValueError("isometry cannot have fewer rows than columns")
    if not np.isfinite(mixing).all():
        raise ValueError("isometry entries must be finite")
    identity = np.eye(mixing.shape[1], dtype=np.complex128)
    residual = float(np.linalg.norm(mixing.conj().T @ mixing - identity, ord=2))
    if residual > tolerance:
        raise ValueError("mixing matrix must satisfy u^dagger u = I")
    return tuple(
        sum(
            (mixing[row, column] * operators[column] for column in range(len(operators))),
            np.zeros(shape, dtype=np.complex128),
        )
        for row in range(mixing.shape[0])
    )


def duplicate_operation(operator: np.ndarray, multiplicity: int) -> tuple[np.ndarray, ...]:
    """Return ``k`` non-zero copies ``K/sqrt(k)`` of one operator."""

    count = _positive_integer(multiplicity, "multiplicity")
    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("operator entries must be finite")
    return tuple(matrix / math.sqrt(count) for _ in range(count))


def raw_count_conformal_factor(
    raw_count: int,
    *,
    reference_count: int = 1,
    spacetime_dimension: int = 4,
) -> float:
    """The deliberately naive dimensionless scale ``(N/N*)**(1/D)``."""

    count = _positive_integer(raw_count, "raw_count")
    reference = _positive_integer(reference_count, "reference_count")
    dimension = _positive_integer(spacetime_dimension, "spacetime_dimension")
    if dimension < 2:
        raise ValueError("spacetime_dimension must be at least two")
    return (count / reference) ** (1.0 / dimension)


@dataclass(frozen=True)
class KrausRefinementCertificate:
    outcome_probability: float
    hidden_multiplicities: tuple[int, ...]
    maximum_operation_residual: float
    maximum_full_completeness_residual: float
    maximum_coarse_probability_residual: float
    maximum_total_probability_residual: float
    maximum_posterior_residual: float
    maximum_choi_residual: float
    numerical_choi_ranks: tuple[int, ...]
    sublabel_probability_sums: tuple[float, ...]
    raw_conformal_factors: tuple[float, ...]
    raw_metric_coefficient_ratios: tuple[float, ...]
    general_isometry_shape: tuple[int, int]
    general_isometry_residual: float
    general_channel_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *,
    outcome_probability: float = 0.3,
    hidden_multiplicities: tuple[int, ...] = (1, 2, 16, 37),
    spacetime_dimension: int = 4,
    tolerance: float = DEFAULT_TOLERANCE,
) -> KrausRefinementCertificate:
    """Build the deterministic outcome-wise refinement obstruction."""

    if not math.isfinite(outcome_probability) or not 0.0 < outcome_probability < 1.0:
        raise ValueError("outcome_probability must be finite and lie in (0, 1)")
    if not hidden_multiplicities:
        raise ValueError("hidden_multiplicities must be non-empty")
    multiplicities = tuple(
        _positive_integer(value, "hidden multiplicity") for value in hidden_multiplicities
    )
    dimension = _positive_integer(spacetime_dimension, "spacetime_dimension")
    if dimension < 2:
        raise ValueError("spacetime_dimension must be at least two")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    p = outcome_probability
    outcome_operator = math.sqrt(p) * I2
    complement_operator = math.sqrt(1.0 - p) * I2
    states = (
        P0,
        P1,
        0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128),
        0.5 * np.array([[1.0, -1.0j], [1.0j, 1.0]], dtype=np.complex128),
        np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=np.complex128),
    )
    states = tuple(_validated_density(state, tolerance=tolerance) for state in states)
    base_family = (outcome_operator,)
    base_choi = choi_matrix(base_family)

    operation_residuals: list[float] = []
    completeness_residuals: list[float] = []
    probability_residuals: list[float] = []
    total_probability_residuals: list[float] = []
    posterior_residuals: list[float] = []
    choi_residuals: list[float] = []
    numerical_choi_ranks: list[int] = []
    sublabel_sums: list[float] = []

    for count in multiplicities:
        refined = duplicate_operation(outcome_operator, count)
        full_family = refined + (complement_operator,)
        completeness = sum(
            (operator.conj().T @ operator for operator in full_family),
            np.zeros_like(I2),
        )
        completeness_residuals.append(float(np.linalg.norm(completeness - I2, ord=2)))
        refined_choi = choi_matrix(refined)
        choi_residuals.append(float(np.linalg.norm(refined_choi - base_choi, ord=2)))
        choi_singular_values = np.linalg.svd(refined_choi, compute_uv=False)
        choi_scale = float(choi_singular_values[0])
        numerical_choi_ranks.append(
            int(np.count_nonzero(choi_singular_values > tolerance * choi_scale))
        )
        sublabel_sums.append(
            sum(
                float(np.trace(operator @ states[-1] @ operator.conj().T).real)
                for operator in refined
            )
        )
        for state in states:
            base_output = apply_cp_map(base_family, state)
            refined_output = apply_cp_map(refined, state)
            operation_residuals.append(float(np.linalg.norm(refined_output - base_output, ord=2)))
            base_probability = float(np.trace(base_output).real)
            refined_probability = float(np.trace(refined_output).real)
            complement_probability = float(
                np.trace(complement_operator @ state @ complement_operator.conj().T).real
            )
            probability_residuals.append(abs(refined_probability - base_probability))
            total_probability_residuals.append(
                max(
                    abs(complement_probability - (1.0 - p)),
                    abs(refined_probability + complement_probability - 1.0),
                )
            )
            posterior_residuals.append(
                float(
                    np.linalg.norm(
                        refined_output / refined_probability - base_output / base_probability,
                        ord=2,
                    )
                )
            )

    # A nontrivial 4 by 2 isometry verifies the general mixing formula on the
    # qubit dephasing channel, independently of the equal-copy construction.
    general_isometry = 0.5 * np.array(
        [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0j], [1.0, -1.0j]],
        dtype=np.complex128,
    )
    isometry_identity = np.eye(2, dtype=np.complex128)
    general_isometry_residual = float(
        np.linalg.norm(general_isometry.conj().T @ general_isometry - isometry_identity, ord=2)
    )
    dephasing = (P0, P1)
    mixed_dephasing = isometric_refinement(dephasing, general_isometry, tolerance=tolerance)
    general_channel_residual = max(
        float(np.linalg.norm(apply_cp_map(mixed_dephasing, state) - apply_cp_map(dephasing, state), ord=2))
        for state in states
    )

    omega = tuple(
        raw_count_conformal_factor(count, spacetime_dimension=dimension)
        for count in multiplicities
    )
    metric_ratios = tuple(value * value for value in omega)
    maximum_operation_residual = max(operation_residuals)
    maximum_completeness_residual = max(completeness_residuals)
    maximum_probability_residual = max(probability_residuals)
    maximum_total_probability_residual = max(total_probability_residuals)
    maximum_posterior_residual = max(posterior_residuals)
    maximum_choi_residual = max(choi_residuals)
    quantum_invariant = max(
        maximum_operation_residual,
        maximum_completeness_residual,
        maximum_probability_residual,
        maximum_total_probability_residual,
        maximum_posterior_residual,
        maximum_choi_residual,
        general_isometry_residual,
        general_channel_residual,
    ) <= 10.0 * tolerance
    raw_metric_changes = len({round(value, 12) for value in metric_ratios}) > 1

    return KrausRefinementCertificate(
        outcome_probability=p,
        hidden_multiplicities=multiplicities,
        maximum_operation_residual=maximum_operation_residual,
        maximum_full_completeness_residual=maximum_completeness_residual,
        maximum_coarse_probability_residual=maximum_probability_residual,
        maximum_total_probability_residual=maximum_total_probability_residual,
        maximum_posterior_residual=maximum_posterior_residual,
        maximum_choi_residual=maximum_choi_residual,
        numerical_choi_ranks=tuple(numerical_choi_ranks),
        sublabel_probability_sums=tuple(sublabel_sums),
        raw_conformal_factors=omega,
        raw_metric_coefficient_ratios=metric_ratios,
        general_isometry_shape=general_isometry.shape,
        general_isometry_residual=general_isometry_residual,
        general_channel_residual=general_channel_residual,
        dimensions={
            "raw_count_dimensionless": True,
            "count_ratio_dimensionless": True,
            "conformal_factor_dimensionless": True,
            "absolute_volume_requires_independent_reference_scale": True,
            "dimension_consistency_does_not_make_count_physical": True,
        },
        accounting={
            "refined_sublabel_probabilities_sum_to_coarse_probability": all(
                math.isclose(value, p, abs_tol=10.0 * tolerance) for value in sublabel_sums
            ),
            "coarse_plus_refined_probability_double_counting_forbidden": True,
            "representation_only_sublabel_adds_energy_or_stress": False,
            "energy_receipt_or_stress_used": False,
        },
        boundaries={
            "sublabel_is_unobserved": True,
            "physical_pointer_record_derived": False,
            "zero_probability_posterior_excluded": True,
            "finite_dimensional_only": True,
        },
        alternatives={
            "physical_recorded_refinement_open": True,
            "choi_invariant_route_open": True,
            "independent_causal_order_and_volume_route_open": True,
        },
        status={
            "outcome_operation_isometry_invariant": quantum_invariant,
            "coarse_probability_invariant": maximum_probability_residual <= 10.0 * tolerance,
            "posterior_invariant": maximum_posterior_residual <= 10.0 * tolerance,
            "cptp_completeness_preserved": maximum_completeness_residual <= 10.0 * tolerance,
            "choi_matrix_invariant": maximum_choi_residual <= 10.0 * tolerance,
            "raw_hidden_count_invariant": False,
            "raw_count_metric_changes_for_same_instrument": raw_metric_changes and quantum_invariant,
            "raw_count_defines_physical_volume_or_metric": False,
            "choi_rank_numerically_invariant": len(set(numerical_choi_ranks)) == 1,
            "minimal_kraus_rank_theorem_proved_by_finite_regression": False,
            "physical_record_algebra_derived": False,
            "local_volume_measure_derived": False,
            "metric_or_curvature_derived": False,
            "fold_stress_derived": False,
            "gr_lensing_backreaction_derived": False,
            "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-probability", type=float, default=0.3)
    args = parser.parse_args()
    print(json.dumps(asdict(certificate(outcome_probability=args.outcome_probability)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
