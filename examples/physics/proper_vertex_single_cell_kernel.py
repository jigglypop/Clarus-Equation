'''Typed proper-EPRL single-cell kernel and critical-point certificate.

The finite part of a Lorentzian proper vertex consists of five SU(2)
Livine--Speziale intertwiners, ten face labels, and ten finite proper
projectors.  This module extracts those objects for one fine 4-simplex of the
fixed 1-to-5 witness and root-gauge-fixes the Cartan-dual critical frames.

It deliberately stops at a kernel contract.  The principal-series matrix
coefficients, pointwise Lorentzian face pairings, non-compact Haar integral,
five-vertex contraction, and Hessian are not evaluated.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

import numpy as np

from examples.physics.eprl_y_gamma_and_proper_projector import (
    proper_orientation_sign,
    proper_positive_spectral_projector,
    relative_sl2c,
)
from examples.physics.proper_vertex_one_to_five_boundary import (
    FINE_SIMPLICES,
    lorentzian_one_to_five_coordinates,
)
from examples.physics.proper_vertex_one_to_five_bra_ket_gluing import (
    j_dual_spinor,
)
from examples.physics.proper_vertex_one_to_five_incidence_spinors import (
    certify_lorentzian_one_to_five_incidence_spinors,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import PAULI_MATRICES
from examples.physics.proper_vertex_one_to_five_ls_intertwiners import (
    four_valent_ls_recoupling_coefficients,
    spin_coherent_state_coefficients,
)
from examples.physics.proper_vertex_one_to_five_regge_faces import _tetrahedron
from examples.physics.proper_vertex_one_to_five_spin_assignment import (
    certify_lorentzian_one_to_five_integer_spin_assignment,
)
from examples.physics.proper_vertex_one_to_five_tangent_frames import (
    oriented_tetrahedron_tangent_frame,
)


Cell = tuple[int, int, int, int, int]
Tetrahedron = tuple[int, int, int, int]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class SingleCellLSIntertwiner:
    omitted_vertex: int
    tetrahedron: Tetrahedron
    opposite_tetrahedron_labels: tuple[int, int, int, int]
    triangles: tuple[Triangle, Triangle, Triangle, Triangle]
    spins: tuple[int, int, int, int]
    recoupling_channels: tuple[int, ...]
    normalized_coefficients: tuple[complex, ...]
    unnormalized_group_average_norm: float
    normalized_coefficient_residual: float
    spin_weighted_closure_defect: float


@dataclass(frozen=True)
class ProperFaceKernelContract:
    source_omitted_vertex: int
    target_omitted_vertex: int
    triangle: Triangle
    spin: int
    source_endpoint_spin: int
    target_endpoint_spin: int
    endpoint_spin_match: bool
    spin_space_dimension: int
    principal_series_k: Fraction
    principal_series_p: float
    projector_endpoint: str
    beta_at_cartan_dual_critical_point: int
    sector_scalar_q: float
    target_projector: np.ndarray
    source_coherent_state: np.ndarray
    target_coherent_state: np.ndarray
    target_projected_norm: float
    target_projector_residual: float
    critical_spinor_equation_residual: float
    epsilon_j_square_sign: int


@dataclass(frozen=True)
class GaugeFixedProperVertexCriticalPoint:
    root_omitted_vertex: int
    cartan_dual_frames: tuple[tuple[int, np.ndarray], ...]
    gauge_fixed_frames: tuple[tuple[int, np.ndarray], ...]
    root_identity_residual: float
    relative_element_gauge_invariance_residual: float
    common_left_projector_invariance_residual: float
    common_left_beta_invariant: bool


@dataclass(frozen=True)
class ProperVertexSingleCellKernelCertificate:
    cell: Cell
    level: int
    gamma: float
    tetrahedron_count: int
    unoriented_face_count: int
    directed_incidence_count: int
    integration_group_count_before_gauge_fixing: int
    integration_group_count_after_gauge_fixing: int
    integration_real_dimension_after_gauge_fixing: int
    ls_intertwiners: tuple[SingleCellLSIntertwiner, ...]
    face_kernel_contracts: tuple[ProperFaceKernelContract, ...]
    critical_point: GaugeFixedProperVertexCriticalPoint
    all_five_ls_intertwiners_nonzero_and_normalized: bool
    all_ten_face_spins_match_at_endpoints: bool
    all_ten_cartan_dual_projectors_preserve_eq53_target_kets: bool
    all_ten_cartan_dual_sector_scalars_positive: bool
    all_ten_critical_spinor_equations_verified: bool
    single_cell_ls_boundary_tensor_constructed: bool
    proper_face_kernel_types_and_endpoint_policy_defined: bool
    gauge_fixed_cartan_dual_critical_point_constructed: bool
    finite_level_spin_weighted_closure_exact: bool
    physical_regge_state_phase_constructed: bool
    full_principal_series_matrix_coefficients_materialized: bool
    pointwise_proper_vertex_integrand_evaluated: bool
    noncompact_haar_integral_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'SINGLE_CELL_BOUNDARY_TENSOR_KERNEL_CONTRACT_AND_CRITICAL_POINT_ONLY'
    )


def spin_j_anti_linear_dual(vector: np.ndarray, spin: int) -> np.ndarray:
    '''Apply J_j in the ascending magnetic basis m=-j,...,j.'''

    state = np.asarray(vector, dtype=complex)
    if type(spin) is not int or spin < 0 or state.shape != (2 * spin + 1,):
        raise ValueError('state must be a spin-j coefficient vector')
    return np.asarray(
        [
            (-1) ** (spin + magnetic) * np.conjugate(state[spin - magnetic])
            for magnetic in range(-spin, spin + 1)
        ],
        dtype=complex,
    )


def _projective_residual(left: np.ndarray, right: np.ndarray) -> float:
    left_unit = left / np.linalg.norm(left)
    right_unit = right / np.linalg.norm(right)
    overlap = complex(np.vdot(right_unit, left_unit))
    if abs(overlap) <= 1.0e-14:
        return math.inf
    return float(
        np.linalg.norm(left_unit - overlap / abs(overlap) * right_unit)
    )


def certify_proper_vertex_single_cell_kernel(
    *,
    cell_index: int = 0,
    level: int = 3,
    gamma: float = 0.274,
    tolerance: float = 1.0e-11,
) -> ProperVertexSingleCellKernelCertificate:
    '''Build one gauge-fixed proper-vertex kernel contract, without integrating.'''

    if type(cell_index) is not int or not 0 <= cell_index < len(FINE_SIMPLICES):
        raise ValueError('cell_index must select one fine 4-simplex')
    if type(level) is not int or level <= 0:
        raise ValueError('level must be a positive integer')
    if not math.isfinite(gamma):
        raise ValueError('gamma must be finite')
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('tolerance must be finite and positive')

    cell = FINE_SIMPLICES[cell_index]
    labels = tuple(sorted(cell))
    tetrahedra = {label: _tetrahedron(cell, label) for label in labels}
    placement = lorentzian_one_to_five_coordinates()
    incidence_certificate = certify_lorentzian_one_to_five_incidence_spinors()
    incidences = {
        (item.cell, item.tetrahedron, item.triangle): item
        for item in incidence_certificate.incidences
    }
    spin_certificate = certify_lorentzian_one_to_five_integer_spin_assignment(
        level=level
    )
    spins = {
        item.triangle: item.rounded_su2_spin_j
        for item in spin_certificate.face_data
    }
    closure_defects = {
        item.tetrahedron: item.rescaled_closure_defect_norm
        for item in spin_certificate.tetrahedron_data
    }

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

    ls_records: list[SingleCellLSIntertwiner] = []
    for label in labels:
        other_labels = tuple(item for item in labels if item != label)
        triangles = tuple(
            tuple(
                sorted(
                    set(tetrahedra[label]).intersection(tetrahedra[other])
                )
            )
            for other in other_labels
        )
        local_spins = tuple(spins[triangle] for triangle in triangles)
        local_spinors = tuple(
            incidences[(cell, tetrahedra[label], triangle)].canonical_spinor
            for triangle in triangles
        )
        channels, coefficients = four_valent_ls_recoupling_coefficients(
            local_spins, local_spinors
        )
        norm = math.sqrt(sum(abs(value) ** 2 for value in coefficients))
        normalized = tuple(value / norm for value in coefficients)
        ls_records.append(
            SingleCellLSIntertwiner(
                omitted_vertex=label,
                tetrahedron=tetrahedra[label],
                opposite_tetrahedron_labels=other_labels,
                triangles=triangles,  # type: ignore[arg-type]
                spins=local_spins,  # type: ignore[arg-type]
                recoupling_channels=channels,
                normalized_coefficients=normalized,
                unnormalized_group_average_norm=norm,
                normalized_coefficient_residual=abs(
                    sum(abs(value) ** 2 for value in normalized) - 1.0
                ),
                spin_weighted_closure_defect=closure_defects[
                    tetrahedra[label]
                ],
            )
        )

    face_contracts: list[ProperFaceKernelContract] = []
    for source, target in combinations(labels, 2):
        source_triangle = tuple(
            vertex for vertex in tetrahedra[source] if vertex != target
        )
        target_triangle = tuple(
            vertex for vertex in tetrahedra[target] if vertex != source
        )
        triangle = tuple(sorted(source_triangle))
        source_spin = spins[tuple(sorted(source_triangle))]
        target_spin = spins[tuple(sorted(target_triangle))]
        spin = target_spin
        target_incidence = incidences[(cell, tetrahedra[target], triangle)]
        orientation = proper_orientation_sign(dual_frames, target, source)
        relative = relative_sl2c(dual_frames[target], dual_frames[source])
        projector = proper_positive_spectral_projector(
            Fraction(spin), orientation.beta, relative
        )
        coherent = np.asarray(
            spin_coherent_state_coefficients(
                spin, target_incidence.canonical_spinor
            ),
            dtype=complex,
        )
        projected_norm = float(np.linalg.norm(projector.projector @ coherent))
        source_incidence = incidences[(cell, tetrahedra[source], triangle)]
        source_coherent = np.asarray(
            spin_coherent_state_coefficients(
                spin, source_incidence.canonical_spinor
            ),
            dtype=complex,
        )
        critical_residual = _projective_residual(
            dual_frames[source] @ source_incidence.canonical_spinor,
            dual_frames[target] @ j_dual_spinor(target_incidence.canonical_spinor),
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
            @ target_incidence.oriented_face_normal
        )
        face_contracts.append(
            ProperFaceKernelContract(
                source_omitted_vertex=source,
                target_omitted_vertex=target,
                triangle=triangle,  # type: ignore[arg-type]
                spin=spin,
                source_endpoint_spin=source_spin,
                target_endpoint_spin=target_spin,
                endpoint_spin_match=(
                    tuple(sorted(source_triangle))
                    == tuple(sorted(target_triangle))
                    and source_spin == target_spin
                ),
                spin_space_dimension=2 * spin + 1,
                principal_series_k=Fraction(spin),
                principal_series_p=gamma * spin,
                projector_endpoint='target_ba_as_in_Engle_Zipfel_Eq_53',
                beta_at_cartan_dual_critical_point=orientation.beta,
                sector_scalar_q=q_value,
                target_projector=projector.projector,
                source_coherent_state=source_coherent,
                target_coherent_state=coherent,
                target_projected_norm=projected_norm,
                target_projector_residual=max(
                    projector.projector_idempotence_residual,
                    projector.projector_hermiticity_residual,
                ),
                critical_spinor_equation_residual=critical_residual,
                epsilon_j_square_sign=(-1) ** (2 * spin),
            )
        )

    root = labels[-1]
    gauge_fixed = {
        label: np.linalg.inv(dual_frames[root]) @ dual_frames[label]
        for label in labels
    }
    common_left = np.asarray(
        ((math.exp(0.17), 0.0), (0.0, math.exp(-0.17))), dtype=complex
    )
    transformed = {
        label: common_left @ frame for label, frame in dual_frames.items()
    }
    transformed_fixed = {
        label: np.linalg.inv(transformed[root]) @ transformed[label]
        for label in labels
    }
    relative_residual = max(
        float(np.linalg.norm(gauge_fixed[label] - transformed_fixed[label]))
        for label in labels
    )
    projector_residuals: list[float] = []
    beta_matches: list[bool] = []
    for source, target in combinations(labels, 2):
        original_orientation = proper_orientation_sign(
            dual_frames, target, source
        )
        transformed_orientation = proper_orientation_sign(
            transformed, target, source
        )
        spin = spins[
            tuple(
                sorted(
                    set(tetrahedra[source]).intersection(tetrahedra[target])
                )
            )
        ]
        original_projector = proper_positive_spectral_projector(
            Fraction(spin),
            original_orientation.beta,
            relative_sl2c(dual_frames[target], dual_frames[source]),
        ).projector
        transformed_projector = proper_positive_spectral_projector(
            Fraction(spin),
            transformed_orientation.beta,
            relative_sl2c(transformed[target], transformed[source]),
        ).projector
        projector_residuals.append(
            float(np.linalg.norm(original_projector - transformed_projector))
        )
        beta_matches.append(
            original_orientation.beta == transformed_orientation.beta
        )

    critical_point = GaugeFixedProperVertexCriticalPoint(
        root_omitted_vertex=root,
        cartan_dual_frames=tuple((label, dual_frames[label]) for label in labels),
        gauge_fixed_frames=tuple((label, gauge_fixed[label]) for label in labels),
        root_identity_residual=float(
            np.linalg.norm(gauge_fixed[root] - np.eye(2))
        ),
        relative_element_gauge_invariance_residual=relative_residual,
        common_left_projector_invariance_residual=max(projector_residuals),
        common_left_beta_invariant=all(beta_matches),
    )

    ls_closed = all(
        item.unnormalized_group_average_norm > 1.0e-14
        and item.normalized_coefficient_residual <= tolerance
        for item in ls_records
    )
    projectors_preserve = all(
        abs(item.target_projected_norm - 1.0) <= tolerance
        and item.target_projector_residual <= tolerance
        for item in face_contracts
    )
    all_positive = all(
        item.sector_scalar_q > tolerance for item in face_contracts
    )
    endpoint_spins_match = all(
        item.endpoint_spin_match for item in face_contracts
    )
    critical_verified = all(
        item.critical_spinor_equation_residual <= tolerance
        for item in face_contracts
    )
    gauge_fixed_closed = (
        critical_point.root_identity_residual <= tolerance
        and critical_point.relative_element_gauge_invariance_residual <= tolerance
        and critical_point.common_left_projector_invariance_residual <= tolerance
        and critical_point.common_left_beta_invariant
    )
    closed = (
        len(ls_records) == 5
        and len(face_contracts) == 10
        and ls_closed
        and endpoint_spins_match
        and projectors_preserve
        and all_positive
        and critical_verified
        and gauge_fixed_closed
    )
    return ProperVertexSingleCellKernelCertificate(
        cell=cell,
        level=level,
        gamma=gamma,
        tetrahedron_count=len(ls_records),
        unoriented_face_count=len(face_contracts),
        directed_incidence_count=2 * len(face_contracts),
        integration_group_count_before_gauge_fixing=5,
        integration_group_count_after_gauge_fixing=4,
        integration_real_dimension_after_gauge_fixing=24,
        ls_intertwiners=tuple(ls_records),
        face_kernel_contracts=tuple(face_contracts),
        critical_point=critical_point,
        all_five_ls_intertwiners_nonzero_and_normalized=ls_closed,
        all_ten_face_spins_match_at_endpoints=endpoint_spins_match,
        all_ten_cartan_dual_projectors_preserve_eq53_target_kets=(
            projectors_preserve
        ),
        all_ten_cartan_dual_sector_scalars_positive=all_positive,
        all_ten_critical_spinor_equations_verified=critical_verified,
        single_cell_ls_boundary_tensor_constructed=ls_closed,
        proper_face_kernel_types_and_endpoint_policy_defined=True,
        gauge_fixed_cartan_dual_critical_point_constructed=gauge_fixed_closed,
        finite_level_spin_weighted_closure_exact=False,
        physical_regge_state_phase_constructed=False,
        full_principal_series_matrix_coefficients_materialized=False,
        pointwise_proper_vertex_integrand_evaluated=False,
        noncompact_haar_integral_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'SINGLE_CELL_PROPER_VERTEX_KERNEL_CONTRACT_AND_CRITICAL_POINT_CLOSED'
            if closed
            else 'SINGLE_CELL_PROPER_VERTEX_KERNEL_CONTRACT_FAILED'
        ),
    )
