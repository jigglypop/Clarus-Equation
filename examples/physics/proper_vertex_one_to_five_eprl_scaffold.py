'''Cellwise Y_gamma evaluations and proper-projector sector diagnostics.

This module combines the incidence-oriented spinor section, a small integer
spin witness, the homogeneous EPRL embedding formula, and the finite proper
projector.  It materializes fifty cell/face projector matrices.

For the chosen deterministic tetrahedron frames and oriented incidence
normals, every Engle--Zipfel sector scalar q_ab is strictly negative.  The
positive projector consequently removes the chosen coherent state and keeps
its J-dual direction.  Applying the Cartan involution
``X -> (X^dagger)^-1`` gives a gauge-inequivalent, projector-positive branch;
an independent rank-six continuum reconstruction places that branch in the
``(omega, nu, mu) = (+1, +1, +1)`` Einstein--Hilbert orientation/Plebanski
sector for each of the five fixed fine 4-simplices.  This remains a local
sector certificate: it is neither an Einstein--Hilbert dynamics result nor a
proper boundary state, vertex integral, or five-vertex amplitude.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from examples.physics.eprl_y_gamma_and_proper_projector import (
    evaluate_y_gamma_coherent_state,
    proper_orientation_sign,
    proper_positive_spectral_projector,
    relative_sl2c,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    FINE_SIMPLICES,
    lorentzian_one_to_five_coordinates,
    triangle_area_squared,
)
from examples.physics.proper_vertex_one_to_five_bivectors import (
    certify_lorentzian_one_to_five_bivectors,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    j_dual_spinor,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    MINKOWSKI_METRIC,
    PAULI_MATRICES,
    sl2c_lorentz_matrix,
)
from examples.physics.proper_vertex_one_to_five_incidence_spinors import (
    NodeId,
    certify_lorentzian_one_to_five_incidence_spinors,
)
from examples.physics.proper_vertex_one_to_five_ls_intertwiners import (
    spin_coherent_state_coefficients,
)
from examples.physics.lorentzian_bivector_reconstruction import hodge_dual
from examples.physics.proper_vertex_one_to_five_regge_faces import (
    _tetrahedron,
)
from examples.physics.proper_vertex_one_to_five_spin_assignment import (
    certify_lorentzian_one_to_five_integer_spin_assignment,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
)


@dataclass(frozen=True)
class CellFaceEPRLProjectorData:
    cell: tuple[int, int, int, int, int]
    source_omitted_vertex: int
    target_omitted_vertex: int
    triangle: tuple[int, int, int]
    spin: int
    gamma: float
    principal_series_p: float
    beta: int
    beta_first_normalized_determinant: float
    beta_second_normalized_determinant: float
    sector_scalar_q: float
    relative_sl2c: np.ndarray
    trace_vector: np.ndarray
    projector: np.ndarray
    projector_rank: int
    projector_zero_eigenvalue_count: int
    chosen_simple_bivector_residual: float
    antipodal_negated_bivector_residual: float
    original_critical_equation_residual: float
    cartan_dual_critical_equation_residual: float
    cartan_dual_orientation_equation_residual: float
    cartan_dual_parity_bivector_residual: float
    cartan_dual_beta: int
    cartan_dual_sector_scalar_q: float
    cartan_dual_projector: np.ndarray
    cartan_dual_projector_rank: int
    cartan_dual_coherent_projected_norm: float
    chosen_coherent_projected_norm: float
    j_dual_coherent_projected_norm: float
    y_gamma_sample_value: complex
    y_gamma_sample_expected_value: float
    y_gamma_sample_residual: float
    projector_residual: float


@dataclass(frozen=True)
class CellContinuumSectorAudit:
    cell: tuple[int, int, int, int, int]
    coordinate_face_matrix_rank: int
    original_continuum_reconstruction_residual: float
    cartan_dual_continuum_reconstruction_residual: float
    original_hodge_tetrad_residual: float
    cartan_dual_parity_hodge_tetrad_residual: float
    original_dynamical_orientation_scalar: float
    cartan_dual_dynamical_orientation_scalar: float
    original_omega: int
    cartan_dual_omega: int
    original_plebanski_sector_nu: int
    cartan_dual_plebanski_sector_nu: int
    original_mu: int
    cartan_dual_mu: int


@dataclass(frozen=True)
class LorentzianOneToFiveEPRLScaffoldCertificate:
    level: int
    gamma: float
    face_record_count: int
    face_records: tuple[CellFaceEPRLProjectorData, ...]
    continuum_sector_audits: tuple[CellContinuumSectorAudit, ...]
    all_beta_diagnostics_nondegenerate: bool
    all_projector_generators_nontrivial: bool
    all_fifty_y_gamma_samples_materialized: bool
    all_fifty_proper_projector_matrices_materialized: bool
    all_chosen_incidence_sector_scalars_negative: bool
    all_chosen_coherent_states_removed_by_positive_projector: bool
    all_j_dual_coherent_states_preserved_by_positive_projector: bool
    all_chosen_spinor_bivectors_match_classical_branch: bool
    all_antipodal_spinor_bivectors_match_globally_negated_branch: bool
    all_cartan_dual_frames_proper_orthochronous: bool
    all_cartan_dual_beta_signs_match_original: bool
    all_cartan_dual_orientation_equations_verified: bool
    all_original_and_cartan_dual_critical_equations_verified: bool
    all_cartan_dual_bivectors_match_parity_transform: bool
    all_cartan_dual_sector_scalars_positive: bool
    all_cartan_dual_projectors_preserve_chosen_coherent_states: bool
    all_five_cartan_dual_solutions_inequivalent_to_original: bool
    cartan_dual_solution_inequivalence_residuals: tuple[float, ...]
    max_projector_residual: float
    max_y_gamma_sample_residual: float
    minimum_absolute_sector_scalar: float
    local_y_gamma_evaluation_scaffold_constructed: bool
    local_proper_projector_scaffold_constructed: bool
    chosen_boundary_data_in_positive_einstein_hilbert_sector: bool
    cartan_dual_boundary_data_in_positive_einstein_hilbert_sector: bool
    globally_negated_bivector_boundary_candidate_constructed: bool
    parity_related_projector_positive_reconstruction_candidate_constructed: bool
    independent_mu_omega_einstein_hilbert_gate_verified: bool
    proper_sector_boundary_state_constructed: bool
    full_principal_series_representations_materialized: bool
    gauge_fixed_single_vertex_integral_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'FIXED_CELL_CARTAN_DUAL_EH_BRANCH_AND_PROJECTOR_SCAFFOLD_ONLY'
    )


def _simple_bivector(normal: np.ndarray) -> np.ndarray:
    time_axis = np.asarray((1.0, 0.0, 0.0, 0.0))
    local_normal = np.concatenate(((0.0,), normal))
    return -(
        np.outer(time_axis, local_normal)
        - np.outer(local_normal, time_axis)
    )


def _projective_spinor_residual(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = left / np.linalg.norm(left)
    right_unit = right / np.linalg.norm(right)
    overlap = complex(np.vdot(right_unit, left_unit))
    if abs(overlap) <= 1.0e-14:
        return math.inf
    return float(
        np.linalg.norm(left_unit - (overlap / abs(overlap)) * right_unit)
    )


_ANTISYMMETRIC_PAIRS = tuple(combinations(range(4), 2))


def _permutation_sign(indices: tuple[int, int, int, int]) -> int:
    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


def _continuum_tensor_from_faces(
    coordinate_faces: np.ndarray,
    internal_faces: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, float]:
    tensor = np.zeros((4, 4, 4, 4), dtype=float)
    residual = 0.0
    for internal_first, internal_second in _ANTISYMMETRIC_PAIRS:
        values = np.asarray(
            [face[internal_first, internal_second] for face in internal_faces]
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            coordinate_faces, values, rcond=None
        )
        residual = max(
            residual,
            float(np.linalg.norm(coordinate_faces @ coefficients - values)),
        )
        for coefficient, (spacetime_first, spacetime_second) in zip(
            coefficients, _ANTISYMMETRIC_PAIRS
        ):
            tensor[
                spacetime_first,
                spacetime_second,
                internal_first,
                internal_second,
            ] = coefficient
            tensor[
                spacetime_second,
                spacetime_first,
                internal_first,
                internal_second,
            ] = -coefficient
            tensor[
                spacetime_first,
                spacetime_second,
                internal_second,
                internal_first,
            ] = -coefficient
            tensor[
                spacetime_second,
                spacetime_first,
                internal_second,
                internal_first,
            ] = coefficient
    return tensor, residual


def _dynamical_orientation_scalar(tensor: np.ndarray) -> float:
    value = 0.0
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    spacetime_epsilon = _permutation_sign((mu, nu, rho, sigma))
                    if spacetime_epsilon == 0:
                        continue
                    for first in range(4):
                        for second in range(4):
                            for third in range(4):
                                for fourth in range(4):
                                    internal_epsilon = _permutation_sign(
                                        (first, second, third, fourth)
                                    )
                                    if internal_epsilon:
                                        value += (
                                            spacetime_epsilon
                                            * internal_epsilon
                                            * tensor[mu, nu, first, second]
                                            * tensor[rho, sigma, third, fourth]
                                        )
    return value


def _hodge_tetrad_tensor(parity_tetrad: bool) -> np.ndarray:
    parity = np.diag((1.0, -1.0, -1.0, -1.0))
    tensor = np.zeros((4, 4, 4, 4), dtype=float)
    for mu, nu in _ANTISYMMETRIC_PAIRS:
        first = parity[:, mu] if parity_tetrad else np.eye(4)[:, mu]
        second = parity[:, nu] if parity_tetrad else np.eye(4)[:, nu]
        internal_wedge = np.outer(first, second) - np.outer(second, first)
        dual = hodge_dual(internal_wedge)
        tensor[mu, nu] = dual
        tensor[nu, mu] = -dual
    return tensor


def _independent_continuum_sector_audits(
    placement: dict[int, tuple[Fraction, Fraction, Fraction, Fraction]],
    bivector_by_wedge: dict[tuple[tuple[int, ...], int, int], object],
) -> tuple[CellContinuumSectorAudit, ...]:
    original_expected = _hodge_tetrad_tensor(False)
    dual_expected = _hodge_tetrad_tensor(True)
    parity = np.diag((1.0, -1.0, -1.0, -1.0))
    audits: list[CellContinuumSectorAudit] = []
    for cell in FINE_SIMPLICES:
        coordinate_rows: list[list[float]] = []
        original_faces: list[np.ndarray] = []
        dual_faces: list[np.ndarray] = []
        for omitted_left, omitted_right in combinations(sorted(cell), 2):
            record = bivector_by_wedge[(cell, omitted_left, omitted_right)]
            triangle = record.triangle
            base = placement[triangle[0]]
            first_edge = np.asarray(
                [
                    float(value - origin)
                    for value, origin in zip(placement[triangle[1]], base)
                ]
            )
            second_edge = np.asarray(
                [
                    float(value - origin)
                    for value, origin in zip(placement[triangle[2]], base)
                ]
            )
            coordinate_bivector = (
                0.5
                * record.cell_orientation_sign
                * (
                    np.outer(first_edge, second_edge)
                    - np.outer(second_edge, first_edge)
                )
            )
            coordinate_rows.append(
                [
                    coordinate_bivector[first, second]
                    for first, second in _ANTISYMMETRIC_PAIRS
                ]
            )
            area = math.sqrt(
                float(triangle_area_squared(triangle, placement))
            )
            original = area * record.cell_oriented_unit_bivector
            original_faces.append(original)
            dual_faces.append(-parity @ original @ parity)
        coordinate_matrix = np.asarray(coordinate_rows)
        original_tensor, original_residual = _continuum_tensor_from_faces(
            coordinate_matrix, tuple(original_faces)
        )
        dual_tensor, dual_residual = _continuum_tensor_from_faces(
            coordinate_matrix, tuple(dual_faces)
        )
        original_q = _dynamical_orientation_scalar(original_tensor)
        dual_q = _dynamical_orientation_scalar(dual_tensor)
        original_nu = (
            1
            if np.linalg.norm(original_tensor - original_expected)
            <= np.linalg.norm(original_tensor + original_expected)
            else -1
        )
        dual_nu = (
            1
            if np.linalg.norm(dual_tensor - dual_expected)
            <= np.linalg.norm(dual_tensor + dual_expected)
            else -1
        )
        original_omega = 1 if original_q > 0.0 else -1
        dual_omega = 1 if dual_q > 0.0 else -1
        audits.append(
            CellContinuumSectorAudit(
                cell=cell,
                coordinate_face_matrix_rank=int(
                    np.linalg.matrix_rank(coordinate_matrix)
                ),
                original_continuum_reconstruction_residual=original_residual,
                cartan_dual_continuum_reconstruction_residual=dual_residual,
                original_hodge_tetrad_residual=float(
                    np.linalg.norm(original_tensor - original_expected)
                ),
                cartan_dual_parity_hodge_tetrad_residual=float(
                    np.linalg.norm(dual_tensor - dual_expected)
                ),
                original_dynamical_orientation_scalar=original_q,
                cartan_dual_dynamical_orientation_scalar=dual_q,
                original_omega=original_omega,
                cartan_dual_omega=dual_omega,
                original_plebanski_sector_nu=original_nu,
                cartan_dual_plebanski_sector_nu=dual_nu,
                original_mu=original_omega * original_nu,
                cartan_dual_mu=dual_omega * dual_nu,
            )
        )
    return tuple(audits)


def certify_lorentzian_one_to_five_eprl_scaffold(
    *,
    level: int = 3,
    gamma: float = 0.274,
    tolerance: float = 1.0e-11,
) -> LorentzianOneToFiveEPRLScaffoldCertificate:
    '''Materialize cellwise EPRL values/projectors and diagnose sector sign.'''

    if type(level) is not int or level <= 0:
        raise ValueError('level must be a positive integer')
    if not math.isfinite(gamma):
        raise ValueError('gamma must be finite and real')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')
    placement = lorentzian_one_to_five_coordinates()
    incidence = certify_lorentzian_one_to_five_incidence_spinors()
    if not incidence.incidence_oriented_spinor_line_section_constructed:
        raise ValueError('incidence-oriented spinor section must be closed')
    spin_assignment = certify_lorentzian_one_to_five_integer_spin_assignment(
        level=level
    )
    bivectors = certify_lorentzian_one_to_five_bivectors()
    bivector_by_wedge = {
        (record.cell, record.omitted_left, record.omitted_right): record
        for record in bivectors.wedge_data
    }
    continuum_audits = _independent_continuum_sector_audits(
        placement, bivector_by_wedge
    )
    spins = {
        record.triangle: record.rounded_su2_spin_j
        for record in spin_assignment.face_data
    }
    incidences = {
        (record.cell, record.tetrahedron, record.triangle): record
        for record in incidence.incidences
    }

    records: list[CellFaceEPRLProjectorData] = []
    beta_nondegenerate: list[bool] = []
    generator_nontrivial: list[bool] = []
    dual_frame_residuals: list[float] = []
    solution_inequivalence_residuals: list[float] = []
    parity_matrix = np.diag((1.0, -1.0, -1.0, -1.0))
    for cell in FINE_SIMPLICES:
        labels = tuple(sorted(cell))
        tetrahedra = {label: _tetrahedron(cell, label) for label in labels}
        frames = {
            label: oriented_tetrahedron_tangent_frame(
                tetrahedra[label], placement
            ).full_sl2c_frame
            for label in labels
        }
        dual_frames = {
            label: np.linalg.inv(np.conjugate(frame.T))
            for label, frame in frames.items()
        }
        anchor = labels[0]
        reference_equivalence = (
            dual_frames[anchor] @ np.linalg.inv(frames[anchor])
        )
        inequivalence_mismatches = tuple(
            min(
                float(
                    np.linalg.norm(
                        dual_frames[label] @ np.linalg.inv(frames[label])
                        - reference_equivalence
                    )
                ),
                float(
                    np.linalg.norm(
                        dual_frames[label] @ np.linalg.inv(frames[label])
                        + reference_equivalence
                    )
                ),
            )
            for label in labels
        )
        solution_inequivalence_residuals.append(max(inequivalence_mismatches))
        for label in labels:
            lorentz = sl2c_lorentz_matrix(frames[label])
            dual_lorentz = sl2c_lorentz_matrix(dual_frames[label])
            dual_frame_residuals.append(
                max(
                    float(
                        np.linalg.norm(
                            dual_lorentz.T @ MINKOWSKI_METRIC @ dual_lorentz
                            - MINKOWSKI_METRIC
                        )
                    ),
                    abs(float(np.linalg.det(dual_lorentz)) - 1.0),
                    max(0.0, 1.0 - float(dual_lorentz[0, 0])),
                    float(
                        np.linalg.norm(
                            dual_lorentz - parity_matrix @ lorentz @ parity_matrix
                        )
                    ),
                )
            )
        for source_label, target_label in combinations(labels, 2):
            triangle = tuple(
                sorted(
                    set(tetrahedra[source_label]).intersection(
                        tetrahedra[target_label]
                    )
                )
            )
            node: NodeId = (
                cell,
                tetrahedra[source_label],
                triangle,  # type: ignore[arg-type]
            )
            source_incidence = incidences[node]
            spin = spins[triangle]  # type: ignore[index]
            orientation = proper_orientation_sign(
                frames, source_label, target_label
            )
            relative = relative_sl2c(
                frames[source_label], frames[target_label]
            )
            projector = proper_positive_spectral_projector(
                Fraction(spin), orientation.beta, relative
            )
            dual_orientation = proper_orientation_sign(
                dual_frames, source_label, target_label
            )
            dual_relative = relative_sl2c(
                dual_frames[source_label], dual_frames[target_label]
            )
            dual_projector = proper_positive_spectral_projector(
                Fraction(spin), dual_orientation.beta, dual_relative
            )
            positive_matrix = relative @ np.conjugate(relative.T)
            trace_vector = np.asarray(
                [
                    float(np.trace(pauli @ positive_matrix).real)
                    for pauli in PAULI_MATRICES
                ]
            )
            q_value = float(
                orientation.beta
                * trace_vector
                @ source_incidence.oriented_face_normal
            )
            dual_positive_matrix = dual_relative @ np.conjugate(dual_relative.T)
            dual_trace_vector = np.asarray(
                [
                    float(np.trace(pauli @ dual_positive_matrix).real)
                    for pauli in PAULI_MATRICES
                ]
            )
            dual_q_value = float(
                dual_orientation.beta
                * dual_trace_vector
                @ source_incidence.oriented_face_normal
            )
            coherent = np.asarray(
                spin_coherent_state_coefficients(
                    spin, source_incidence.canonical_spinor
                ),
                dtype=complex,
            )
            dual_coherent = np.asarray(
                spin_coherent_state_coefficients(
                    spin, j_dual_spinor(source_incidence.canonical_spinor)
                ),
                dtype=complex,
            )
            chosen_norm = float(np.linalg.norm(projector.projector @ coherent))
            dual_norm = float(np.linalg.norm(projector.projector @ dual_coherent))
            dual_selected_norm = float(
                np.linalg.norm(dual_projector.projector @ coherent)
            )
            local_simple = _simple_bivector(
                source_incidence.oriented_face_normal
            )
            source_lorentz = oriented_tetrahedron_tangent_frame(
                tetrahedra[source_label], placement
            ).full_lorentz_frame
            global_simple = source_lorentz @ local_simple @ source_lorentz.T
            classical_bivector = bivector_by_wedge[
                (cell, source_label, target_label)
            ].cell_oriented_unit_bivector
            chosen_bivector_residual = float(
                np.linalg.norm(global_simple - classical_bivector)
            )
            antipodal_bivector_residual = float(
                np.linalg.norm(-global_simple + classical_bivector)
            )
            target_incidence = incidences[
                (
                    cell,
                    tetrahedra[target_label],
                    triangle,  # type: ignore[arg-type]
                )
            ]
            source_dual_lorentz = sl2c_lorentz_matrix(
                dual_frames[source_label]
            )
            target_dual_lorentz = sl2c_lorentz_matrix(
                dual_frames[target_label]
            )
            dual_source_bivector = (
                source_dual_lorentz
                @ local_simple
                @ source_dual_lorentz.T
            )
            dual_target_bivector = (
                target_dual_lorentz
                @ _simple_bivector(target_incidence.oriented_face_normal)
                @ target_dual_lorentz.T
            )
            dual_orientation_residual = float(
                np.linalg.norm(dual_source_bivector + dual_target_bivector)
            )
            dual_parity_residual = float(
                np.linalg.norm(
                    dual_source_bivector
                    + parity_matrix @ classical_bivector @ parity_matrix
                )
            )
            original_critical_residual = _projective_spinor_residual(
                frames[source_label] @ source_incidence.canonical_spinor,
                frames[target_label]
                @ j_dual_spinor(target_incidence.canonical_spinor),
            )
            dual_critical_residual = _projective_spinor_residual(
                dual_frames[source_label] @ source_incidence.canonical_spinor,
                dual_frames[target_label]
                @ j_dual_spinor(target_incidence.canonical_spinor),
            )
            sample = evaluate_y_gamma_coherent_state(
                Fraction(spin),
                gamma,
                source_incidence.canonical_spinor,
                np.conjugate(source_incidence.canonical_spinor),
            )
            expected_sample = math.sqrt((2 * spin + 1) / math.pi)
            projector_residual = max(
                projector.hermiticity_residual,
                projector.projector_idempotence_residual,
                projector.projector_hermiticity_residual,
            )
            records.append(
                CellFaceEPRLProjectorData(
                    cell=cell,
                    source_omitted_vertex=source_label,
                    target_omitted_vertex=target_label,
                    triangle=triangle,  # type: ignore[arg-type]
                    spin=spin,
                    gamma=gamma,
                    principal_series_p=gamma * spin,
                    beta=orientation.beta,
                    beta_first_normalized_determinant=(
                        orientation.first_normalized_determinant
                    ),
                    beta_second_normalized_determinant=(
                        orientation.second_normalized_determinant
                    ),
                    sector_scalar_q=q_value,
                    relative_sl2c=relative,
                    trace_vector=trace_vector,
                    projector=projector.projector,
                    projector_rank=projector.positive_eigenvalue_count,
                    projector_zero_eigenvalue_count=(
                        projector.zero_eigenvalue_count
                    ),
                    chosen_simple_bivector_residual=chosen_bivector_residual,
                    antipodal_negated_bivector_residual=(
                        antipodal_bivector_residual
                    ),
                    original_critical_equation_residual=(
                        original_critical_residual
                    ),
                    cartan_dual_critical_equation_residual=(
                        dual_critical_residual
                    ),
                    cartan_dual_orientation_equation_residual=(
                        dual_orientation_residual
                    ),
                    cartan_dual_parity_bivector_residual=dual_parity_residual,
                    cartan_dual_beta=dual_orientation.beta,
                    cartan_dual_sector_scalar_q=dual_q_value,
                    cartan_dual_projector=dual_projector.projector,
                    cartan_dual_projector_rank=(
                        dual_projector.positive_eigenvalue_count
                    ),
                    cartan_dual_coherent_projected_norm=dual_selected_norm,
                    chosen_coherent_projected_norm=chosen_norm,
                    j_dual_coherent_projected_norm=dual_norm,
                    y_gamma_sample_value=sample.embedded_value,
                    y_gamma_sample_expected_value=expected_sample,
                    y_gamma_sample_residual=abs(
                        sample.embedded_value - expected_sample
                    ),
                    projector_residual=projector_residual,
                )
            )
            beta_nondegenerate.append(orientation.nondegenerate)
            generator_nontrivial.append(projector.nontrivial_generator)

    max_projector_residual = max(record.projector_residual for record in records)
    max_y_gamma_residual = max(record.y_gamma_sample_residual for record in records)
    all_negative = all(record.sector_scalar_q < -tolerance for record in records)
    chosen_removed = all(
        record.chosen_coherent_projected_norm <= tolerance for record in records
    )
    dual_preserved = all(
        abs(record.j_dual_coherent_projected_norm - 1.0) <= tolerance
        for record in records
    )
    chosen_bivectors_match = all(
        record.chosen_simple_bivector_residual <= tolerance for record in records
    )
    antipodal_bivectors_match_negative = all(
        record.antipodal_negated_bivector_residual <= tolerance
        for record in records
    )
    dual_beta_matches = all(
        record.cartan_dual_beta == record.beta for record in records
    )
    dual_orientation_verified = all(
        record.cartan_dual_orientation_equation_residual <= tolerance
        for record in records
    )
    critical_equations_verified = all(
        max(
            record.original_critical_equation_residual,
            record.cartan_dual_critical_equation_residual,
        )
        <= tolerance
        for record in records
    )
    dual_parity_bivectors_verified = all(
        record.cartan_dual_parity_bivector_residual <= tolerance
        for record in records
    )
    dual_q_positive = all(
        record.cartan_dual_sector_scalar_q > tolerance for record in records
    )
    dual_projectors_preserve = all(
        abs(record.cartan_dual_coherent_projected_norm - 1.0) <= tolerance
        for record in records
    )
    solutions_inequivalent = all(
        residual > 1.0e-5 for residual in solution_inequivalence_residuals
    )
    independent_mu_omega_gate = all(
        audit.coordinate_face_matrix_rank == 6
        and max(
            audit.original_continuum_reconstruction_residual,
            audit.cartan_dual_continuum_reconstruction_residual,
            audit.original_hodge_tetrad_residual,
            audit.cartan_dual_parity_hodge_tetrad_residual,
        )
        <= tolerance
        and audit.original_omega == -1
        and audit.cartan_dual_omega == 1
        and audit.original_plebanski_sector_nu == 1
        and audit.cartan_dual_plebanski_sector_nu == 1
        and audit.original_mu == -1
        and audit.cartan_dual_mu == 1
        for audit in continuum_audits
    )
    closed = (
        len(records) == 50
        and all(beta_nondegenerate)
        and all(generator_nontrivial)
        and max_projector_residual <= tolerance
        and max_y_gamma_residual <= tolerance
        and all_negative
        and chosen_removed
        and dual_preserved
        and chosen_bivectors_match
        and antipodal_bivectors_match_negative
        and max(dual_frame_residuals) <= tolerance
        and dual_beta_matches
        and dual_orientation_verified
        and critical_equations_verified
        and dual_parity_bivectors_verified
        and dual_q_positive
        and dual_projectors_preserve
        and solutions_inequivalent
        and independent_mu_omega_gate
    )
    return LorentzianOneToFiveEPRLScaffoldCertificate(
        level=level,
        gamma=gamma,
        face_record_count=len(records),
        face_records=tuple(records),
        continuum_sector_audits=continuum_audits,
        all_beta_diagnostics_nondegenerate=all(beta_nondegenerate),
        all_projector_generators_nontrivial=all(generator_nontrivial),
        all_fifty_y_gamma_samples_materialized=(
            len(records) == 50 and max_y_gamma_residual <= tolerance
        ),
        all_fifty_proper_projector_matrices_materialized=(
            len(records) == 50 and max_projector_residual <= tolerance
        ),
        all_chosen_incidence_sector_scalars_negative=all_negative,
        all_chosen_coherent_states_removed_by_positive_projector=chosen_removed,
        all_j_dual_coherent_states_preserved_by_positive_projector=dual_preserved,
        all_chosen_spinor_bivectors_match_classical_branch=(
            chosen_bivectors_match
        ),
        all_antipodal_spinor_bivectors_match_globally_negated_branch=(
            antipodal_bivectors_match_negative
        ),
        all_cartan_dual_frames_proper_orthochronous=(
            max(dual_frame_residuals) <= tolerance
        ),
        all_cartan_dual_beta_signs_match_original=dual_beta_matches,
        all_cartan_dual_orientation_equations_verified=(
            dual_orientation_verified
        ),
        all_original_and_cartan_dual_critical_equations_verified=(
            critical_equations_verified
        ),
        all_cartan_dual_bivectors_match_parity_transform=(
            dual_parity_bivectors_verified
        ),
        all_cartan_dual_sector_scalars_positive=dual_q_positive,
        all_cartan_dual_projectors_preserve_chosen_coherent_states=(
            dual_projectors_preserve
        ),
        all_five_cartan_dual_solutions_inequivalent_to_original=(
            solutions_inequivalent
        ),
        cartan_dual_solution_inequivalence_residuals=tuple(
            solution_inequivalence_residuals
        ),
        max_projector_residual=max_projector_residual,
        max_y_gamma_sample_residual=max_y_gamma_residual,
        minimum_absolute_sector_scalar=min(
            abs(record.sector_scalar_q) for record in records
        ),
        local_y_gamma_evaluation_scaffold_constructed=closed,
        local_proper_projector_scaffold_constructed=closed,
        chosen_boundary_data_in_positive_einstein_hilbert_sector=False,
        cartan_dual_boundary_data_in_positive_einstein_hilbert_sector=(
            independent_mu_omega_gate and dual_q_positive
        ),
        globally_negated_bivector_boundary_candidate_constructed=(
            antipodal_bivectors_match_negative
        ),
        parity_related_projector_positive_reconstruction_candidate_constructed=(
            max(dual_frame_residuals) <= tolerance
            and dual_orientation_verified
            and critical_equations_verified
            and dual_q_positive
            and dual_projectors_preserve
            and solutions_inequivalent
        ),
        independent_mu_omega_einstein_hilbert_gate_verified=(
            independent_mu_omega_gate
        ),
        proper_sector_boundary_state_constructed=False,
        full_principal_series_representations_materialized=False,
        gauge_fixed_single_vertex_integral_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'LORENTZIAN_1_TO_5_CARTAN_DUAL_EINSTEIN_HILBERT_BRANCH_CLOSED'
            if closed
            else 'LORENTZIAN_1_TO_5_EPRL_PROJECTOR_SCAFFOLD_FAILED'
        ),
    )
