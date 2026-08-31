"""Finite source-factorization and independent-receipt rank boundary.

This E31 certificate starts from the canonical E29 marginal map ``P=Mq``.
If a proposed source is constant on every fibre of ``M``, it contains no
hidden-lift information and factors through the visible image.  Conversely,
an extra linear receipt ``E q`` distinguishes every hidden direction only if
``ker(M)`` and ``ker(E)`` intersect trivially.

The seven Walsh rows used below are dimensionless coordinate diagnostics.
They prove a rank lower bound for reconstructing the complete finite signed
coordinate ``q``; they are not physical records, energy receipts, stress
components, spacetime volume, fields, or bosons.  No metric or gravity law is
constructed here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.contextual_global_section_obstruction import (
    QUANTUM_ETA,
    exact_rational_rank,
    marginal_incidence_matrix,
    quantum_kernel_perturbed_extension,
    symmetric_signed_global_extension,
    walsh_kernel_vectors,
)
from examples.physics.representation_invariant_measure_bridge import (
    atom_permutation_matrix,
    context_block_permutation,
    normalized_atom_tangent_basis,
)


ATOM_COUNT = 16
DEFAULT_TOLERANCE = 1.0e-12


def _positive_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def _coordinate_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (ATOM_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be sixteen finite coordinates")
    return np.array(vector, copy=True)


def _row_map(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] != ATOM_COUNT
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite row map with sixteen columns")
    return np.array(matrix, copy=True)


def _numerical_rank(matrix: np.ndarray, *, tolerance: float = 1.0e-10) -> int:
    tol = _positive_tolerance(tolerance)
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("matrix must be a nonempty finite two-dimensional array")
    singular_values = np.linalg.svd(values, compute_uv=False)
    return int(np.sum(singular_values > tol))


def walsh_receipt_matrix() -> np.ndarray:
    """Return the seven canonical dimensionless E29 kernel rows."""

    return np.asarray(list(walsh_kernel_vectors().values()), dtype=np.int64)


def linear_source_kernel_residual(source: np.ndarray | Sequence[float]) -> float:
    """Return the largest response of a linear source to a Walsh kernel row."""

    source_map = _row_map(source, name="source")
    walsh = walsh_receipt_matrix().astype(np.float64)
    return float(np.max(np.abs(source_map @ walsh.T)))


def linear_source_factorization_residual(
    source: np.ndarray | Sequence[float],
) -> float:
    """Return the residual of the minimum-norm ambient factor ``L=A M``.

    The returned ambient ``A`` is only one extension of the unique map on
    ``im(M)``.  A zero residual does not make that ambient extension unique.
    """

    source_map = _row_map(source, name="source")
    incidence = marginal_incidence_matrix().astype(np.float64)
    ambient_factor = source_map @ np.linalg.pinv(incidence)
    return float(np.max(np.abs(source_map - ambient_factor @ incidence)))


def factor_linear_source(
    source: np.ndarray | Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return one ambient factor for a source that descends through ``M``."""

    tol = _positive_tolerance(tolerance)
    source_map = _row_map(source, name="source")
    incidence = marginal_incidence_matrix().astype(np.float64)
    ambient_factor = source_map @ np.linalg.pinv(incidence)
    residual = float(np.max(np.abs(source_map - ambient_factor @ incidence)))
    if residual > tol:
        raise ValueError("source is not constant on the canonical incidence fibres")
    return ambient_factor


def receipt_kernel_rank(
    receipt: np.ndarray | Sequence[float],
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """Return ``rank(E|ker M)`` in the canonical Walsh basis."""

    receipt_map = _row_map(receipt, name="receipt")
    walsh = walsh_receipt_matrix().astype(np.float64)
    return _numerical_rank(receipt_map @ walsh.T, tolerance=tolerance)


def combined_readout_rank(
    receipt: np.ndarray | Sequence[float],
    *,
    normalized_tangent: bool = False,
) -> int:
    """Return the rank of ``[M;E]``, optionally on the normalized tangent."""

    receipt_map = _row_map(receipt, name="receipt")
    incidence = marginal_incidence_matrix().astype(np.float64)
    combined = np.vstack((incidence, receipt_map))
    if normalized_tangent:
        combined = combined @ normalized_atom_tangent_basis()
    return exact_rational_rank(combined)


def visible_and_walsh_receipt(
    coordinates: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the visible vector and seven diagnostic Walsh coordinates."""

    vector = _coordinate_vector(coordinates, name="coordinates")
    incidence = marginal_incidence_matrix().astype(np.float64)
    walsh = walsh_receipt_matrix().astype(np.float64)
    return incidence @ vector, walsh @ vector


def reconstruct_from_visible_and_walsh(
    visible: Sequence[float],
    receipt: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Recover ``q`` from consistent canonical ``(Mq, Wq)`` coordinates.

    The Moore--Penrose section is a coordinate convention, not a physical
    selection law.  Walsh orthogonality supplies the kernel component.
    """

    tol = _positive_tolerance(tolerance)
    visible_vector = _coordinate_vector(visible, name="visible")
    receipt_vector = np.asarray(receipt, dtype=np.float64)
    if receipt_vector.shape != (7,) or not np.isfinite(receipt_vector).all():
        raise ValueError("receipt must be seven finite Walsh coordinates")
    incidence = marginal_incidence_matrix().astype(np.float64)
    walsh = walsh_receipt_matrix().astype(np.float64)
    coordinates = (
        np.linalg.pinv(incidence) @ visible_vector
        + walsh.T @ receipt_vector / 16.0
    )
    visible_residual = float(
        np.max(np.abs(incidence @ coordinates - visible_vector))
    )
    receipt_residual = float(np.max(np.abs(walsh @ coordinates - receipt_vector)))
    if visible_residual > tol or receipt_residual > tol:
        raise ValueError("visible and receipt coordinates are inconsistent with the canonical maps")
    return coordinates


@dataclass(frozen=True)
class HiddenSourceFactorizationCertificate:
    incidence_rank: int
    incidence_nullity: int
    walsh_rank: int
    maximum_incidence_walsh_residual: float
    maximum_walsh_gram_residual: float
    combined_rank_one_receipt: int
    combined_rank_six_receipts: int
    combined_rank_seven_receipts: int
    normalized_visible_rank: int
    normalized_combined_rank_one_receipt: int
    normalized_combined_rank_six_receipts: int
    normalized_combined_rank_seven_receipts: int
    minimum_receipt_rows_for_full_recovery: int
    duplicate_receipt_kernel_rank: int
    duplicate_receipt_combined_rank: int
    visible_source_kernel_residual: float
    visible_source_factorization_residual: float
    hidden_source_kernel_residual: float
    hidden_source_factorization_residual: float
    ambient_factor_extension_difference: float
    alternative_ambient_factor_residual: float
    q_delta_visible_residual: float
    q_delta_first_walsh_change: float
    q_delta_other_walsh_residual: float
    permutation_norm_residual: float
    same_fibre_norm_square_difference: float
    reconstruction_coordinate_residual: float
    reconstruction_visible_residual: float
    reconstruction_receipt_residual: float
    relabel_visible_residual: float
    relabel_receipt_residual: float
    relabel_combined_rank: int
    relabel_fixed_incidence_residual: float
    relabel_fixed_receipt_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> HiddenSourceFactorizationCertificate:
    """Build the E31 finite factorization and receipt-rank certificate."""

    tol = _positive_tolerance(tolerance)
    numerical_limit = 100.0 * tol
    incidence_integer = marginal_incidence_matrix()
    incidence = incidence_integer.astype(np.float64)
    walsh_integer = walsh_receipt_matrix()
    walsh = walsh_integer.astype(np.float64)

    incidence_rank = exact_rational_rank(incidence_integer)
    walsh_rank = exact_rational_rank(walsh_integer)
    incidence_walsh_residual = float(np.max(np.abs(incidence @ walsh.T)))
    walsh_gram_residual = float(
        np.max(np.abs(walsh @ walsh.T - 16.0 * np.eye(7)))
    )
    combined_ranks = {
        count: exact_rational_rank(np.vstack((incidence_integer, walsh_integer[:count])))
        for count in (1, 6, 7)
    }
    tangent = normalized_atom_tangent_basis()
    normalized_visible_rank = exact_rational_rank(incidence_integer @ tangent)
    normalized_combined_ranks = {
        count: exact_rational_rank(
            np.vstack((incidence_integer, walsh_integer[:count])) @ tangent
        )
        for count in (1, 6, 7)
    }

    duplicate_receipt = incidence[:7]
    duplicate_kernel_rank = receipt_kernel_rank(duplicate_receipt)
    duplicate_combined_rank = combined_readout_rank(duplicate_receipt)

    visible_source = incidence[[0, 5]]
    visible_kernel_residual = linear_source_kernel_residual(visible_source)
    visible_factorization_residual = linear_source_factorization_residual(
        visible_source
    )
    minimum_norm_factor = factor_linear_source(visible_source, tolerance=tol)
    _, _, transpose_vh = np.linalg.svd(incidence.T)
    left_null_row = transpose_vh[incidence_rank]
    alternative_factor = np.array(minimum_norm_factor, copy=True)
    alternative_factor[0] += left_null_row
    alternative_factor_residual = float(
        np.max(np.abs(visible_source - alternative_factor @ incidence))
    )
    ambient_factor_difference = float(
        np.max(np.abs(alternative_factor - minimum_norm_factor))
    )

    hidden_source = walsh[[0]]
    hidden_kernel_residual = linear_source_kernel_residual(hidden_source)
    hidden_factorization_residual = linear_source_factorization_residual(hidden_source)

    base_q = np.asarray(symmetric_signed_global_extension(QUANTUM_ETA))
    delta_q = np.asarray(quantum_kernel_perturbed_extension(0.1))
    base_visible, base_receipt = visible_and_walsh_receipt(base_q)
    delta_visible, delta_receipt = visible_and_walsh_receipt(delta_q)
    q_delta_visible_residual = float(np.max(np.abs(delta_visible - base_visible)))
    receipt_change = delta_receipt - base_receipt
    q_delta_first_walsh_change = float(receipt_change[0])
    q_delta_other_walsh_residual = float(np.max(np.abs(receipt_change[1:])))

    atom_permutation = atom_permutation_matrix(tuple(reversed(range(ATOM_COUNT))))
    permutation_norm_residual = abs(
        float(np.dot(atom_permutation @ base_q, atom_permutation @ base_q))
        - float(np.dot(base_q, base_q))
    )
    fibre_norm_difference = abs(
        float(np.dot(delta_q, delta_q)) - float(np.dot(base_q, base_q))
    )

    reconstructed = reconstruct_from_visible_and_walsh(
        delta_visible, delta_receipt, tolerance=tol
    )
    reconstructed_visible, reconstructed_receipt = visible_and_walsh_receipt(
        reconstructed
    )
    reconstruction_coordinate_residual = float(
        np.max(np.abs(reconstructed - delta_q))
    )
    reconstruction_visible_residual = float(
        np.max(np.abs(reconstructed_visible - delta_visible))
    )
    reconstruction_receipt_residual = float(
        np.max(np.abs(reconstructed_receipt - delta_receipt))
    )

    context_order = (1, 0, 2, 3)
    row_permutation = context_block_permutation(context_order)
    receipt_permutation = np.eye(7, dtype=np.float64)[::-1]
    relabelled_q = atom_permutation @ delta_q
    relabelled_incidence = row_permutation @ incidence @ atom_permutation.T
    relabelled_receipt_map = receipt_permutation @ walsh @ atom_permutation.T
    relabel_visible_residual = float(
        np.max(
            np.abs(
                relabelled_incidence @ relabelled_q
                - row_permutation @ delta_visible
            )
        )
    )
    relabel_receipt_residual = float(
        np.max(
            np.abs(
                relabelled_receipt_map @ relabelled_q
                - receipt_permutation @ delta_receipt
            )
        )
    )
    relabel_combined_rank = exact_rational_rank(
        np.vstack((relabelled_incidence, relabelled_receipt_map))
    )
    relabel_fixed_incidence_residual = float(
        np.max(np.abs(relabelled_incidence - incidence))
    )
    relabel_fixed_receipt_residual = float(
        np.max(np.abs(relabelled_receipt_map - walsh))
    )

    dimensions = {
        "q_visible_probabilities_and_walsh_coordinates_dimensionless": True,
        "norms_ranks_nullities_and_residuals_dimensionless": True,
        "dimensionful_receipts_require_fixed_reference_scales_before_rank_comparison": True,
        "finite_coordinate_rank_is_not_spacetime_dimension": True,
    }
    accounting = {
        "duplicate_receipt_factors_through_visible_map": duplicate_combined_rank == 9,
        "duplicate_receipt_not_added_as_new_source": True,
        "signed_q_and_walsh_coordinates_not_booked_as_energy_or_stress": True,
        "receipt_probability_energy_or_volume_double_counted": False,
    }
    boundaries = {
        "fibre_constancy_is_required_for_visible_source_factorization": True,
        "permutation_covariance_does_not_imply_fibre_constancy": True,
        "seven_rows_are_necessary_only_for_full_linear_q_recovery": True,
        "seven_rows_are_not_sufficient_without_kernel_rank_seven": True,
        "seven_is_not_a_gravity_component_field_or_boson_count": True,
        "ambient_factor_extension_is_not_unique": True,
        "walsh_receipts_are_canonical_coordinate_witnesses_only": True,
        "general_relabel_covariance_is_not_fixed_map_automorphism": True,
        "physical_receipt_provenance_and_dynamics_are_not_supplied": True,
        "full_finite_signed_coordinate_is_not_a_physical_ontology": True,
    }
    alternatives = {
        "visible_quotient_source_only": True,
        "independent_operational_receipt_with_provenance": True,
        "hidden_ontology_with_local_covariant_action": True,
        "entanglement_linearized_einstein_dictionary": True,
        "causal_order_plus_independent_volume_dictionary": True,
    }
    status = {
        "canonical_rank_nine_nullity_seven_reused": (
            incidence_rank == 9 and ATOM_COUNT - incidence_rank == 7
        ),
        "walsh_kernel_orthogonality_certified": (
            walsh_rank == 7
            and incidence_walsh_residual <= numerical_limit
            and walsh_gram_residual <= numerical_limit
        ),
        "receipt_rank_lower_bound_witness_certified": (
            combined_ranks == {1: 10, 6: 15, 7: 16}
            and normalized_combined_ranks == {1: 9, 6: 14, 7: 15}
        ),
        "full_walsh_coordinate_reconstruction_certified": (
            reconstruction_coordinate_residual <= numerical_limit
            and reconstruction_visible_residual <= numerical_limit
            and reconstruction_receipt_residual <= numerical_limit
        ),
        "linear_visible_source_factorization_certified": (
            visible_kernel_residual <= numerical_limit
            and visible_factorization_residual <= numerical_limit
        ),
        "ambient_factor_extension_nonuniqueness_certified": (
            alternative_factor_residual <= numerical_limit
            and ambient_factor_difference > numerical_limit
        ),
        "permutation_covariance_not_fibre_invariance_certified": (
            q_delta_visible_residual <= numerical_limit
            and permutation_norm_residual <= numerical_limit
            and fibre_norm_difference > numerical_limit
        ),
        "duplicate_visible_receipt_adds_no_rank_certified": (
            duplicate_kernel_rank == 0 and duplicate_combined_rank == 9
        ),
        "general_relabel_covariance_certified": (
            relabel_visible_residual <= numerical_limit
            and relabel_receipt_residual <= numerical_limit
            and relabel_combined_rank == 16
        ),
        "chosen_general_relabel_is_fixed_map_automorphism": (
            relabel_fixed_incidence_residual <= numerical_limit
            and relabel_fixed_receipt_residual <= numerical_limit
        ),
        "physical_walsh_receipt_derived": False,
        "hidden_signed_coordinate_is_physical_state_derived": False,
        "local_covariant_action_or_stress_derived": False,
        "spacetime_metric_curvature_or_gravity_derived": False,
        "objective_selection_derived": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return HiddenSourceFactorizationCertificate(
        incidence_rank=incidence_rank,
        incidence_nullity=ATOM_COUNT - incidence_rank,
        walsh_rank=walsh_rank,
        maximum_incidence_walsh_residual=incidence_walsh_residual,
        maximum_walsh_gram_residual=walsh_gram_residual,
        combined_rank_one_receipt=combined_ranks[1],
        combined_rank_six_receipts=combined_ranks[6],
        combined_rank_seven_receipts=combined_ranks[7],
        normalized_visible_rank=normalized_visible_rank,
        normalized_combined_rank_one_receipt=normalized_combined_ranks[1],
        normalized_combined_rank_six_receipts=normalized_combined_ranks[6],
        normalized_combined_rank_seven_receipts=normalized_combined_ranks[7],
        minimum_receipt_rows_for_full_recovery=ATOM_COUNT - incidence_rank,
        duplicate_receipt_kernel_rank=duplicate_kernel_rank,
        duplicate_receipt_combined_rank=duplicate_combined_rank,
        visible_source_kernel_residual=visible_kernel_residual,
        visible_source_factorization_residual=visible_factorization_residual,
        hidden_source_kernel_residual=hidden_kernel_residual,
        hidden_source_factorization_residual=hidden_factorization_residual,
        ambient_factor_extension_difference=ambient_factor_difference,
        alternative_ambient_factor_residual=alternative_factor_residual,
        q_delta_visible_residual=q_delta_visible_residual,
        q_delta_first_walsh_change=q_delta_first_walsh_change,
        q_delta_other_walsh_residual=q_delta_other_walsh_residual,
        permutation_norm_residual=permutation_norm_residual,
        same_fibre_norm_square_difference=fibre_norm_difference,
        reconstruction_coordinate_residual=reconstruction_coordinate_residual,
        reconstruction_visible_residual=reconstruction_visible_residual,
        reconstruction_receipt_residual=reconstruction_receipt_residual,
        relabel_visible_residual=relabel_visible_residual,
        relabel_receipt_residual=relabel_receipt_residual,
        relabel_combined_rank=relabel_combined_rank,
        relabel_fixed_incidence_residual=relabel_fixed_incidence_residual,
        relabel_fixed_receipt_residual=relabel_fixed_receipt_residual,
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def run() -> dict[str, object]:
    """Return a JSON-serializable E31 certificate."""

    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
