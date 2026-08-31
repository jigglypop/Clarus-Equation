"""Separate bijective fine evolution from a many-to-one observation readout.

For an ``n``-level system and an ``n``-level record register, the controlled
shift

    W |i,j> = |i, j+i mod n>

is a permutation unitary.  With the register prepared in ``|0>``, its
restriction is the record isometry

    V |psi> = sum_i c_i |i,i>.

Thus this declared finite system-plus-register update loses no component:
``W`` is bijective, and ``V`` is bijective onto its image.  A separate
finite-label map is bijective
when it retains every hidden label and both finite label sets carry the
discrete topology.  That label fact does not derive physical hidden-branch
dynamics.  Only a later readout that forgets the individual hidden labels is
many-to-one (when there are at least two hidden labels).  A selected label is
supplied to that conditional readout; it is not produced as a unique actual
outcome by the unitary.

This finite certificate derives no durable physical pointer, selection law,
energy ledger, spacetime topology, metric, curvature, or gravity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math

import numpy as np


DEFAULT_TOLERANCE = 1.0e-12


def _record_dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("dimension must be an integer of at least two")
    return value


def _selected_label(value: int, dimension: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("selected label must be an integer")
    if not 0 <= value < dimension:
        raise ValueError("selected label must lie in the record range")
    return value


def _positive_tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _validated_density(
    state: np.ndarray,
    dimension: int,
    *,
    tolerance: float,
) -> np.ndarray:
    density = np.asarray(state, dtype=np.complex128)
    if density.shape != (dimension, dimension):
        raise ValueError(f"density matrix must have shape ({dimension}, {dimension})")
    if not np.isfinite(density).all():
        raise ValueError("density matrix entries must be finite")
    if np.linalg.norm(density - density.conj().T, ord="fro") > tolerance:
        raise ValueError("density matrix must be Hermitian")
    trace = np.trace(density)
    if abs(float(trace.real) - 1.0) > tolerance or abs(float(trace.imag)) > tolerance:
        raise ValueError("density matrix must have unit trace")
    if float(np.linalg.eigvalsh(density).min()) < -tolerance:
        raise ValueError("density matrix must be positive semidefinite")
    return density


def controlled_record_unitary(dimension: int) -> np.ndarray:
    """Return the permutation unitary ``|i,j> -> |i,j+i mod n>``."""

    size = _record_dimension(dimension)
    unitary = np.zeros((size * size, size * size), dtype=np.complex128)
    for system_label in range(size):
        for apparatus_label in range(size):
            source = system_label * size + apparatus_label
            target = system_label * size + (apparatus_label + system_label) % size
            unitary[target, source] = 1.0
    return unitary


def controlled_record_inverse(dimension: int) -> np.ndarray:
    """Return the explicit inverse ``|i,j> -> |i,j-i mod n>``."""

    size = _record_dimension(dimension)
    inverse = np.zeros((size * size, size * size), dtype=np.complex128)
    for system_label in range(size):
        for apparatus_label in range(size):
            source = system_label * size + apparatus_label
            target = system_label * size + (apparatus_label - system_label) % size
            inverse[target, source] = 1.0
    return inverse


def apparatus_zero_embedding(dimension: int) -> np.ndarray:
    """Embed ``|psi>`` as ``|psi> tensor |0>``."""

    size = _record_dimension(dimension)
    embedding = np.zeros((size * size, size), dtype=np.complex128)
    for system_label in range(size):
        embedding[system_label * size, system_label] = 1.0
    return embedding


def record_isometry(dimension: int) -> np.ndarray:
    """Return ``V = W (I tensor |0>)`` with ``V|i> = |i,i>``."""

    size = _record_dimension(dimension)
    return controlled_record_unitary(size) @ apparatus_zero_embedding(size)


def record_kraus_operators(dimension: int) -> tuple[np.ndarray, ...]:
    """Extract ``K_a = <a|_A V`` from the record isometry."""

    size = _record_dimension(dimension)
    tensor = record_isometry(size).reshape(size, size, size)
    return tuple(tensor[:, label, :].copy() for label in range(size))


def projective_dephasing(state: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """Apply the nonselective channel ``rho -> sum_i P_i rho P_i``."""

    tol = _positive_tolerance(tolerance)
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("state must be a square density matrix")
    size = _record_dimension(density.shape[0])
    density = _validated_density(density, size, tolerance=tol)
    return np.diag(np.diag(density)).astype(np.complex128)


def selective_update(
    state: np.ndarray,
    selected: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return probability, subnormalized operation, and posterior for ``selected``."""

    tol = _positive_tolerance(tolerance)
    density = np.asarray(state, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("state must be a square density matrix")
    size = _record_dimension(density.shape[0])
    density = _validated_density(density, size, tolerance=tol)
    label = _selected_label(selected, size)
    projector = np.zeros((size, size), dtype=np.complex128)
    projector[label, label] = 1.0
    operation = projector @ density @ projector
    probability = float(np.trace(operation).real)
    if probability <= tol:
        raise ValueError("selected outcome must have positive probability")
    return probability, operation, operation / probability


def partial_trace_apparatus(
    joint_state: np.ndarray,
    dimension: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Trace the apparatus from a system-first joint density matrix."""

    size = _record_dimension(dimension)
    tol = _positive_tolerance(tolerance)
    joint = _validated_density(joint_state, size * size, tolerance=tol)
    tensor = joint.reshape(size, size, size, size)
    return np.trace(tensor, axis1=1, axis2=3)


def partial_trace_system(
    joint_state: np.ndarray,
    dimension: int,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Trace the system from a system-first joint density matrix."""

    size = _record_dimension(dimension)
    tol = _positive_tolerance(tolerance)
    joint = _validated_density(joint_state, size * size, tolerance=tol)
    tensor = joint.reshape(size, size, size, size)
    return np.trace(tensor, axis1=0, axis2=2)


def fine_visibility_labels(dimension: int, selected: int) -> tuple[tuple[str, int], ...]:
    """Map labels bijectively onto the returned declared fine-label image."""

    size = _record_dimension(dimension)
    visible = _selected_label(selected, size)
    return tuple(
        ("visible" if label == visible else "hidden", label)
        for label in range(size)
    )


def coarse_visibility_labels(dimension: int, selected: int) -> tuple[str, ...]:
    """Forget individual hidden identities and retain only visible/hidden."""

    return tuple(sector for sector, _ in fine_visibility_labels(dimension, selected))


def _apply_kraus_channel(
    operators: tuple[np.ndarray, ...],
    state: np.ndarray,
) -> np.ndarray:
    return sum(
        (operator @ state @ operator.conj().T for operator in operators),
        np.zeros_like(state),
    )


def _choi_matrix(operators: tuple[np.ndarray, ...]) -> np.ndarray:
    size = operators[0].shape[0]
    choi = np.zeros((size * size, size * size), dtype=np.complex128)
    for operator in operators:
        vector = operator.reshape(-1, order="F")
        choi += np.outer(vector, vector.conj())
    return choi


@dataclass(frozen=True)
class FineUnitaryCoarseObservationCertificate:
    dimension: int
    selected_label: int
    branch_probabilities: tuple[float, ...]
    visible_probability: float
    hidden_probability: float
    fine_labels: tuple[tuple[str, int], ...]
    coarse_labels: tuple[str, ...]
    unitary_left_residual: float
    unitary_right_residual: float
    explicit_inverse_residual: float
    record_isometry_residual: float
    record_output_residual: float
    inverse_recovery_residual: float
    reduced_system_residual: float
    reduced_apparatus_residual: float
    kraus_completeness_residual: float
    kraus_channel_residual: float
    choi_minimum_eigenvalue: float
    distinct_input_residual: float
    nonselective_collision_residual: float
    selective_operation_collision_residual: float
    selective_posterior_collision_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *,
    dimension: int = 3,
    selected: int = 1,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FineUnitaryCoarseObservationCertificate:
    """Build a deterministic finite certificate for the two-map distinction."""

    size = _record_dimension(dimension)
    label = _selected_label(selected, size)
    tol = _positive_tolerance(tolerance)
    identity_joint = np.eye(size * size, dtype=np.complex128)
    identity_system = np.eye(size, dtype=np.complex128)

    unitary = controlled_record_unitary(size)
    inverse = controlled_record_inverse(size)
    embedding = apparatus_zero_embedding(size)
    isometry = unitary @ embedding
    unitary_left_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity_joint, ord=2)
    )
    unitary_right_residual = float(
        np.linalg.norm(unitary @ unitary.conj().T - identity_joint, ord=2)
    )
    explicit_inverse_residual = max(
        float(np.linalg.norm(inverse @ unitary - identity_joint, ord=2)),
        float(np.linalg.norm(unitary @ inverse - identity_joint, ord=2)),
        float(np.linalg.norm(inverse - unitary.conj().T, ord=2)),
    )
    record_isometry_residual = float(
        np.linalg.norm(isometry.conj().T @ isometry - identity_system, ord=2)
    )

    phases = np.exp(2.0j * np.pi * np.arange(size) / size)
    amplitudes = phases / math.sqrt(size)
    apparatus_zero = np.zeros(size, dtype=np.complex128)
    apparatus_zero[0] = 1.0
    initial_joint_vector = np.kron(amplitudes, apparatus_zero)
    recorded_vector = unitary @ initial_joint_vector
    expected_recorded_vector = np.zeros(size * size, dtype=np.complex128)
    for branch, amplitude in enumerate(amplitudes):
        expected_recorded_vector[branch * size + branch] = amplitude
    record_output_residual = float(
        np.linalg.norm(recorded_vector - expected_recorded_vector)
    )
    inverse_recovery_residual = float(
        np.linalg.norm(inverse @ recorded_vector - initial_joint_vector)
    )

    pure_state = np.outer(amplitudes, amplitudes.conj())
    probabilities = tuple(float(abs(amplitude) ** 2) for amplitude in amplitudes)
    diagonal_state = np.diag(probabilities).astype(np.complex128)
    recorded_density = np.outer(recorded_vector, recorded_vector.conj())
    reduced_system = partial_trace_apparatus(recorded_density, size, tolerance=tol)
    reduced_apparatus = partial_trace_system(recorded_density, size, tolerance=tol)
    dephased = projective_dephasing(pure_state, tolerance=tol)
    reduced_system_residual = float(np.linalg.norm(reduced_system - dephased, ord=2))
    reduced_apparatus_residual = float(np.linalg.norm(reduced_apparatus - dephased, ord=2))

    kraus = record_kraus_operators(size)
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        np.zeros_like(identity_system),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(completeness - identity_system, ord=2)
    )
    kraus_channel_residual = float(
        np.linalg.norm(_apply_kraus_channel(kraus, pure_state) - dephased, ord=2)
    )
    choi_minimum_eigenvalue = float(np.linalg.eigvalsh(_choi_matrix(kraus)).min())

    pure_probability, pure_operation, pure_posterior = selective_update(
        pure_state, label, tolerance=tol
    )
    diagonal_probability, diagonal_operation, diagonal_posterior = selective_update(
        diagonal_state, label, tolerance=tol
    )
    distinct_input_residual = float(np.linalg.norm(pure_state - diagonal_state, ord=2))
    nonselective_collision_residual = float(
        np.linalg.norm(
            projective_dephasing(pure_state, tolerance=tol)
            - projective_dephasing(diagonal_state, tolerance=tol),
            ord=2,
        )
    )
    selective_operation_collision_residual = max(
        abs(pure_probability - diagonal_probability),
        float(np.linalg.norm(pure_operation - diagonal_operation, ord=2)),
    )
    selective_posterior_collision_residual = float(
        np.linalg.norm(pure_posterior - diagonal_posterior, ord=2)
    )

    fine_labels = fine_visibility_labels(size, label)
    coarse_labels = coarse_visibility_labels(size, label)
    # The declared codomain is exactly F_{a,n} = {(visible, a)} union
    # {(hidden, i): i != a}, represented by ``set(fine_labels)``.  Equality of
    # its cardinality with the domain proves bijectivity onto that codomain,
    # not onto the larger Cartesian product {visible, hidden} x L_n.
    fine_bijective_onto_declared_image = len(set(fine_labels)) == size
    coarse_injective = len(set(coarse_labels)) == size
    visible_probability = probabilities[label]
    hidden_probability = sum(
        probability
        for branch, probability in enumerate(probabilities)
        if branch != label
    )
    unitary_certified = max(
        unitary_left_residual,
        unitary_right_residual,
        explicit_inverse_residual,
        record_isometry_residual,
        record_output_residual,
        inverse_recovery_residual,
    ) <= 10.0 * tol
    cptp_certified = (
        kraus_completeness_residual <= 10.0 * tol
        and kraus_channel_residual <= 10.0 * tol
        and choi_minimum_eigenvalue >= -10.0 * tol
    )

    return FineUnitaryCoarseObservationCertificate(
        dimension=size,
        selected_label=label,
        branch_probabilities=probabilities,
        visible_probability=visible_probability,
        hidden_probability=hidden_probability,
        fine_labels=fine_labels,
        coarse_labels=coarse_labels,
        unitary_left_residual=unitary_left_residual,
        unitary_right_residual=unitary_right_residual,
        explicit_inverse_residual=explicit_inverse_residual,
        record_isometry_residual=record_isometry_residual,
        record_output_residual=record_output_residual,
        inverse_recovery_residual=inverse_recovery_residual,
        reduced_system_residual=reduced_system_residual,
        reduced_apparatus_residual=reduced_apparatus_residual,
        kraus_completeness_residual=kraus_completeness_residual,
        kraus_channel_residual=kraus_channel_residual,
        choi_minimum_eigenvalue=choi_minimum_eigenvalue,
        distinct_input_residual=distinct_input_residual,
        nonselective_collision_residual=nonselective_collision_residual,
        selective_operation_collision_residual=selective_operation_collision_residual,
        selective_posterior_collision_residual=selective_posterior_collision_residual,
        dimensions={
            "basis_labels_dimensionless": True,
            "unitary_and_density_entries_dimensionless": True,
            "branch_probabilities_dimensionless": True,
            "energy_requires_independent_hamiltonian_scale": True,
            "physical_duration_requires_independent_time_scale": True,
        },
        accounting={
            "branch_probabilities_sum_to_one": math.isclose(
                sum(probabilities), 1.0, abs_tol=10.0 * tol
            ),
            "visible_plus_hidden_probability_sum_to_one": math.isclose(
                visible_probability + hidden_probability, 1.0, abs_tol=10.0 * tol
            ),
            "hidden_labels_retained_individually_in_fine_label_map": (
                fine_bijective_onto_declared_image
            ),
            "coarse_and_fine_probabilities_added_as_separate_energy": False,
            "energy_or_stress_assigned_without_ledger": False,
        },
        boundaries={
            "selected_label_is_supplied_to_conditional_readout": True,
            "selected_label_is_not_an_input_to_controlled_unitary": True,
            "finite_w_is_declared_model_not_actual_universe_dynamics": True,
            "record_register_is_abstract_not_a_durable_pointer": True,
            "fine_sort_is_label_only_not_physical_branch_dynamics": True,
            "fine_sort_codomain_is_declared_image_not_full_cartesian_product": True,
            "finite_label_topology_declared_discrete": True,
            "coarse_label_map_many_to_one_only_for_dimension_at_least_three": True,
            "finite_dimension_is_hilbert_label_dimension_not_spacetime_dimension": True,
            "hilbert_state_map_uses_norm_topology_only": True,
            "cptp_claim_is_for_the_explicit_projective_instrument": True,
        },
        alternatives={
            "local_decoherence_instrument_route_open": True,
            "actual_selection_law_route_open": True,
            "representation_invariant_geometry_route_open": True,
        },
        status={
            "declared_finite_controlled_unitary_bijective": unitary_certified,
            "record_isometry_bijective_onto_its_image": record_isometry_residual <= 10.0 * tol,
            "record_isometry_surjective_onto_full_joint_space": False,
            "fine_discrete_label_sort_bijective_onto_declared_image": (
                fine_bijective_onto_declared_image
            ),
            "fine_discrete_label_bijection_onto_image_is_homeomorphism": (
                fine_bijective_onto_declared_image
            ),
            "coarse_visibility_readout_injective": coarse_injective,
            "nonselective_dephasing_many_to_one_witness": (
                distinct_input_residual > 10.0 * tol
                and nonselective_collision_residual <= 10.0 * tol
            ),
            "selective_update_many_to_one_witness": (
                distinct_input_residual > 10.0 * tol
                and selective_operation_collision_residual <= 10.0 * tol
                and selective_posterior_collision_residual <= 10.0 * tol
            ),
            "explicit_projective_record_channel_cptp": cptp_certified,
            "premeasurement_components_preserved_by_fine_unitary": unitary_certified,
            "unitary_selects_one_unique_actual_outcome": False,
            "durable_physical_pointer_derived": False,
            "energy_hamiltonian_or_transfer_derived": False,
            "spacetime_homeomorphism_derived": False,
            "spacetime_metric_or_curvature_derived": False,
            "fold_stress_or_gravity_derived": False,
            "relativistic_no_signalling_derived": False,
            "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--selected", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            asdict(certificate(dimension=args.dimension, selected=args.selected)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
