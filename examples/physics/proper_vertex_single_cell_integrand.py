'''Numerical CP1 evaluation of one fixed proper-vertex integrand.

For the Cartan-dual classical frame tuple certified by
``proper_vertex_single_cell_kernel``, every Eq.-53 target projector preserves
its coherent ket.  The Lorentzian principal-series pairing can therefore be
evaluated in its published homogeneous-function/CP1 realization.  This
module performs that compact CP1 quadrature for the ten face propagators and
multiplies them to obtain a quadrature estimate of the pointwise coefficient
function relative to product Haar measure at the supplied group variables.
The rounded level-3 spins do not close exactly, so this is not a finite-spin
stationary-point certificate.

This is not the non-compact SL(2,C)^4 Haar integral.  It also does not provide
a five-vertex contraction, stationary-phase Hessian, or continuum dynamics.
'''

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.eprl_y_gamma_and_proper_projector import (
    proper_orientation_sign,
    proper_positive_spectral_projector,
    relative_sl2c,
)
from examples.physics.proper_vertex_single_cell_kernel import (
    ProperFaceKernelContract,
    ProperVertexSingleCellKernelCertificate,
    certify_proper_vertex_single_cell_kernel,
)


@dataclass(frozen=True)
class CP1FacePropagatorEvaluation:
    source_omitted_vertex: int
    target_omitted_vertex: int
    triangle: tuple[int, int, int]
    spin: int
    gamma: float
    principal_series_p: float
    coarse_quadrature_shape: tuple[int, int]
    fine_quadrature_shape: tuple[int, int]
    coherent_target_value: complex
    proper_projected_target_value: complex
    quadrature_absolute_difference: float
    quadrature_relative_difference: float
    projector_insertion_value_difference: float


@dataclass(frozen=True)
class ProperVertexSingleCellIntegrandCertificate:
    cell: tuple[int, int, int, int, int]
    root_omitted_vertex: int
    face_propagator_count: int
    cp1_real_dimension_per_face: int
    gauge_fixed_sl2c_real_dimension: int
    face_evaluations: tuple[CP1FacePropagatorEvaluation, ...]
    graphical_ordering_sign: int
    pointwise_proper_vertex_coefficient_function: complex
    max_face_quadrature_absolute_difference: float
    max_face_quadrature_relative_difference: float
    max_projector_insertion_value_difference: float
    all_ten_cp1_face_pairings_numerically_evaluated: bool
    all_ten_proper_projectors_inserted_on_eq53_target_endpoint: bool
    all_integer_spin_crossing_signs_trivial: bool
    cartan_dual_classical_frame_tuple_used: bool
    finite_spin_stationary_point_certified: bool
    pointwise_root_gauge_fixed_coefficient_function_quadrature_estimated: bool
    product_haar_density_factor_included: bool
    noncompact_sl2c_haar_measure_materialized: bool
    noncompact_sl2c_haar_integral_evaluated: bool
    proper_eprl_five_vertex_amplitude_derived: bool
    proper_eprl_multicell_hessian_computed: bool
    status: str
    claim_ceiling: str = (
        'POINTWISE_SINGLE_VERTEX_COEFFICIENT_AT_CARTAN_DUAL_CLASSICAL_FRAMES_ONLY'
    )


@dataclass(frozen=True)
class ArbitraryFrameProperCoefficientEvaluation:
    frame_labels: tuple[int, int, int, int, int]
    face_count: int
    cp1_quadrature_shape: tuple[int, int]
    orientation_degenerate_face_count: int
    minimum_absolute_normalized_orientation_determinant: float
    beta_signs: tuple[int, ...]
    projector_ranks: tuple[int, ...]
    face_coefficients: tuple[complex, ...]
    coefficient_product: complex
    all_frames_sl2c: bool
    all_nondegenerate_projectors_recomputed_from_supplied_frames: bool


def spin_j_linear_epsilon_dual(vector: np.ndarray, spin: int) -> np.ndarray:
    '''Apply the linear epsilon_j map used inside the bilinear alpha pairing.'''

    state = np.asarray(vector, dtype=complex)
    if type(spin) is not int or spin < 0 or state.shape != (2 * spin + 1,):
        raise ValueError('state must be a spin-j coefficient vector')
    return np.asarray(
        [
            (-1) ** (spin + magnetic) * state[spin - magnetic]
            for magnetic in range(-spin, spin + 1)
        ],
        dtype=complex,
    )


def spin_state_homogeneous_polynomial(
    vector: np.ndarray,
    spin: int,
    arguments: np.ndarray,
) -> np.ndarray:
    '''Evaluate the degree-2j polynomial without the sqrt((2j+1)/pi) factor.'''

    state = np.asarray(vector, dtype=complex)
    z = np.asarray(arguments, dtype=complex)
    if type(spin) is not int or spin < 0 or state.shape != (2 * spin + 1,):
        raise ValueError('state must be a spin-j coefficient vector')
    if z.ndim < 1 or z.shape[0] != 2 or not np.all(np.isfinite(z)):
        raise ValueError('arguments must have leading dimension two')
    value = np.zeros(z.shape[1:], dtype=complex)
    for index, magnetic in enumerate(range(-spin, spin + 1)):
        value += (
            state[index]
            * math.sqrt(math.comb(2 * spin, spin + magnetic))
            * z[0] ** (spin + magnetic)
            * z[1] ** (spin - magnetic)
        )
    return value


def _cp1_nodes(number_u: int, number_phi: int) -> tuple[np.ndarray, np.ndarray]:
    if type(number_u) is not int or number_u < 4:
        raise ValueError('number_u must be an integer at least four')
    if type(number_phi) is not int or number_phi < 8:
        raise ValueError('number_phi must be an integer at least eight')
    u, u_weights = np.polynomial.legendre.leggauss(number_u)
    phi = 2.0 * math.pi * (np.arange(number_phi) + 0.5) / number_phi
    z = np.stack(
        np.broadcast_arrays(
            np.sqrt((1.0 + u[:, None]) / 2.0),
            np.exp(1.0j * phi[None, :])
            * np.sqrt((1.0 - u[:, None]) / 2.0),
        )
    )
    # Omega_CP1 = (1/4) d(cos(theta)) d(phi), hence integral Omega = pi.
    weights = u_weights[:, None] * (2.0 * math.pi / number_phi) / 4.0
    return z, weights


def evaluate_cp1_proper_face_pairing(
    *,
    spin: int,
    gamma: float,
    source_state: np.ndarray,
    target_state_after_projector: np.ndarray,
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    number_u: int,
    number_phi: int,
) -> complex:
    '''Evaluate Engle--Zipfel's alpha pairing in the one-CP1 realization.'''

    if type(spin) is not int or spin <= 0:
        raise ValueError('spin must be a positive integer')
    if not math.isfinite(gamma):
        raise ValueError('gamma must be finite')
    source = np.asarray(source_state, dtype=complex)
    target = np.asarray(target_state_after_projector, dtype=complex)
    if source.shape != (2 * spin + 1,) or target.shape != source.shape:
        raise ValueError('source and target must be matching spin-j states')
    first = np.asarray(source_frame, dtype=complex)
    second = np.asarray(target_frame, dtype=complex)
    if first.shape != (2, 2) or second.shape != (2, 2):
        raise ValueError('frames must be two by two matrices')

    z, weights = _cp1_nodes(number_u, number_phi)
    flattened = z.reshape(2, -1)
    first_inverse_z = (np.linalg.inv(first) @ flattened).reshape(z.shape)
    second_inverse_z = (np.linalg.inv(second) @ flattened).reshape(z.shape)
    first_norm = np.sum(np.abs(first_inverse_z) ** 2, axis=0)
    second_norm = np.sum(np.abs(second_inverse_z) ** 2, axis=0)
    p_value = gamma * spin
    radial = np.exp(
        complex(-1 - spin, -p_value) * np.log(first_norm)
        + complex(-1 - spin, p_value) * np.log(second_norm)
    )
    source_polynomial = spin_state_homogeneous_polynomial(
        source, spin, np.conjugate(first_inverse_z)
    )
    target_polynomial = spin_state_homogeneous_polynomial(
        spin_j_linear_epsilon_dual(target, spin),
        spin,
        second_inverse_z,
    )
    c_value = math.sqrt(spin**2 + p_value**2) / (
        math.pi * complex(spin, -p_value)
    )
    return complex(
        c_value
        * (2 * spin + 1)
        * np.sum(weights * radial * source_polynomial * target_polynomial)
    )


def evaluate_proper_vertex_coefficient_at_frames(
    *,
    frames: dict[int, np.ndarray],
    kernel_contract: ProperVertexSingleCellKernelCertificate,
    number_u: int = 20,
    number_phi: int = 40,
    orientation_tolerance: float = 1.0e-14,
) -> ArbitraryFrameProperCoefficientEvaluation:
    '''Recompute all Eq.-53 projectors and alpha pairings at supplied frames.

    Unlike the fixed classical-frame certificate, this function never reuses
    the stored critical projectors.  The orientation sign, relative element,
    and target spectral projector are recomputed for every face.  Exact beta
    degeneracy is assigned the zero face coefficient; it is a Haar-measure
    zero boundary between orientation sectors.
    '''

    if not isinstance(kernel_contract, ProperVertexSingleCellKernelCertificate):
        raise ValueError('kernel_contract must be a single-cell certificate')
    kernel = kernel_contract
    labels = tuple(sorted(frames))
    expected_labels = tuple(
        sorted(label for label, _ in kernel.critical_point.gauge_fixed_frames)
    )
    if labels != expected_labels:
        raise ValueError('frames must contain exactly the five kernel labels')
    checked_frames: dict[int, np.ndarray] = {}
    for label in labels:
        frame = np.asarray(frames[label], dtype=complex)
        if (
            frame.shape != (2, 2)
            or not np.all(np.isfinite(frame))
            or abs(complex(np.linalg.det(frame)) - 1.0) > 1.0e-9
        ):
            raise ValueError('each supplied frame must lie in SL(2,C)')
        checked_frames[label] = frame
    if not math.isfinite(orientation_tolerance) or orientation_tolerance <= 0.0:
        raise ValueError('orientation_tolerance must be finite and positive')

    coefficients: list[complex] = []
    beta_signs: list[int] = []
    projector_ranks: list[int] = []
    degenerate_count = 0
    normalized_margins: list[float] = []
    for face in kernel.face_kernel_contracts:
        source = face.source_omitted_vertex
        target = face.target_omitted_vertex
        orientation = proper_orientation_sign(
            checked_frames,
            target,
            source,
            tolerance=orientation_tolerance,
        )
        normalized_margins.extend(
            (
                abs(orientation.first_normalized_determinant),
                abs(orientation.second_normalized_determinant),
            )
        )
        if (
            orientation.first_spatial_determinant == 0.0
            or orientation.second_spatial_determinant == 0.0
        ):
            degenerate_count += 1
            beta_signs.append(0)
            projector_ranks.append(0)
            coefficients.append(0.0j)
            continue
        beta = (
            1
            if (orientation.first_spatial_determinant > 0.0)
            == (orientation.second_spatial_determinant > 0.0)
            else -1
        )
        beta_signs.append(beta)
        projector = proper_positive_spectral_projector(
            face.principal_series_k,
            beta,
            relative_sl2c(checked_frames[target], checked_frames[source]),
        )
        projector_ranks.append(projector.positive_eigenvalue_count)
        projected_target = projector.projector @ face.target_coherent_state
        coefficients.append(
            evaluate_cp1_proper_face_pairing(
                spin=face.spin,
                gamma=kernel.gamma,
                source_state=face.source_coherent_state,
                target_state_after_projector=projected_target,
                source_frame=checked_frames[source],
                target_frame=checked_frames[target],
                number_u=number_u,
                number_phi=number_phi,
            )
        )
    product = complex(np.prod(coefficients))
    return ArbitraryFrameProperCoefficientEvaluation(
        frame_labels=labels,  # type: ignore[arg-type]
        face_count=len(coefficients),
        cp1_quadrature_shape=(number_u, number_phi),
        orientation_degenerate_face_count=degenerate_count,
        minimum_absolute_normalized_orientation_determinant=min(
            normalized_margins
        ),
        beta_signs=tuple(beta_signs),
        projector_ranks=tuple(projector_ranks),
        face_coefficients=tuple(coefficients),
        coefficient_product=product,
        all_frames_sl2c=True,
        all_nondegenerate_projectors_recomputed_from_supplied_frames=(
            degenerate_count == 0
        ),
    )


def _evaluate_face(
    face: ProperFaceKernelContract,
    frames: dict[int, np.ndarray],
    gamma: float,
    coarse_shape: tuple[int, int],
    fine_shape: tuple[int, int],
) -> CP1FacePropagatorEvaluation:
    projected_target = face.target_projector @ face.target_coherent_state
    coarse = evaluate_cp1_proper_face_pairing(
        spin=face.spin,
        gamma=gamma,
        source_state=face.source_coherent_state,
        target_state_after_projector=projected_target,
        source_frame=frames[face.source_omitted_vertex],
        target_frame=frames[face.target_omitted_vertex],
        number_u=coarse_shape[0],
        number_phi=coarse_shape[1],
    )
    fine = evaluate_cp1_proper_face_pairing(
        spin=face.spin,
        gamma=gamma,
        source_state=face.source_coherent_state,
        target_state_after_projector=projected_target,
        source_frame=frames[face.source_omitted_vertex],
        target_frame=frames[face.target_omitted_vertex],
        number_u=fine_shape[0],
        number_phi=fine_shape[1],
    )
    coherent = evaluate_cp1_proper_face_pairing(
        spin=face.spin,
        gamma=gamma,
        source_state=face.source_coherent_state,
        target_state_after_projector=face.target_coherent_state,
        source_frame=frames[face.source_omitted_vertex],
        target_frame=frames[face.target_omitted_vertex],
        number_u=fine_shape[0],
        number_phi=fine_shape[1],
    )
    difference = abs(fine - coarse)
    return CP1FacePropagatorEvaluation(
        source_omitted_vertex=face.source_omitted_vertex,
        target_omitted_vertex=face.target_omitted_vertex,
        triangle=face.triangle,
        spin=face.spin,
        gamma=gamma,
        principal_series_p=gamma * face.spin,
        coarse_quadrature_shape=coarse_shape,
        fine_quadrature_shape=fine_shape,
        coherent_target_value=coherent,
        proper_projected_target_value=fine,
        quadrature_absolute_difference=difference,
        quadrature_relative_difference=difference / max(abs(fine), 1.0e-14),
        projector_insertion_value_difference=abs(fine - coherent),
    )


def certify_proper_vertex_single_cell_integrand(
    *,
    cell_index: int = 0,
    level: int = 3,
    gamma: float = 0.274,
    coarse_shape: tuple[int, int] = (28, 56),
    fine_shape: tuple[int, int] = (44, 88),
    quadrature_tolerance: float = 2.0e-8,
) -> ProperVertexSingleCellIntegrandCertificate:
    '''Evaluate ten CP1 pairings and their product at one critical X tuple.'''

    if (
        len(coarse_shape) != 2
        or len(fine_shape) != 2
        or fine_shape[0] <= coarse_shape[0]
        or fine_shape[1] <= coarse_shape[1]
    ):
        raise ValueError('fine_shape must refine both coarse_shape dimensions')
    if not math.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0.0:
        raise ValueError('quadrature_tolerance must be finite and positive')
    kernel = certify_proper_vertex_single_cell_kernel(
        cell_index=cell_index, level=level, gamma=gamma
    )
    if not kernel.gauge_fixed_cartan_dual_critical_point_constructed:
        raise ValueError('the gauge-fixed critical kernel contract must close')
    frames = dict(kernel.critical_point.gauge_fixed_frames)
    evaluations = tuple(
        _evaluate_face(face, frames, gamma, coarse_shape, fine_shape)
        for face in kernel.face_kernel_contracts
    )
    maximum_absolute = max(
        item.quadrature_absolute_difference for item in evaluations
    )
    maximum_relative = max(
        item.quadrature_relative_difference for item in evaluations
    )
    maximum_projector = max(
        item.projector_insertion_value_difference for item in evaluations
    )
    all_evaluated = (
        len(evaluations) == 10
        and all(
            np.isfinite(item.proper_projected_target_value.real)
            and np.isfinite(item.proper_projected_target_value.imag)
            for item in evaluations
        )
        and maximum_absolute <= quadrature_tolerance
    )
    product = complex(np.prod(
        [item.proper_projected_target_value for item in evaluations]
    ))
    all_integer_spins = all(
        type(item.spin) is int and item.spin >= 0 for item in evaluations
    )
    graphical_sign = 1 if all_integer_spins else 0
    closed = (
        all_evaluated
        and maximum_projector <= 1.0e-10
        and all_integer_spins
        and product != 0.0j
    )
    return ProperVertexSingleCellIntegrandCertificate(
        cell=kernel.cell,
        root_omitted_vertex=kernel.critical_point.root_omitted_vertex,
        face_propagator_count=len(evaluations),
        cp1_real_dimension_per_face=2,
        gauge_fixed_sl2c_real_dimension=24,
        face_evaluations=evaluations,
        graphical_ordering_sign=graphical_sign,
        pointwise_proper_vertex_coefficient_function=product,
        max_face_quadrature_absolute_difference=maximum_absolute,
        max_face_quadrature_relative_difference=maximum_relative,
        max_projector_insertion_value_difference=maximum_projector,
        all_ten_cp1_face_pairings_numerically_evaluated=all_evaluated,
        all_ten_proper_projectors_inserted_on_eq53_target_endpoint=True,
        all_integer_spin_crossing_signs_trivial=all_integer_spins,
        cartan_dual_classical_frame_tuple_used=True,
        finite_spin_stationary_point_certified=False,
        pointwise_root_gauge_fixed_coefficient_function_quadrature_estimated=(
            closed
        ),
        product_haar_density_factor_included=False,
        noncompact_sl2c_haar_measure_materialized=False,
        noncompact_sl2c_haar_integral_evaluated=False,
        proper_eprl_five_vertex_amplitude_derived=False,
        proper_eprl_multicell_hessian_computed=False,
        status=(
            'POINTWISE_SINGLE_CELL_PROPER_VERTEX_COEFFICIENT_QUADRATURE_ESTIMATED'
            if closed
            else 'POINTWISE_SINGLE_CELL_PROPER_VERTEX_COEFFICIENT_QUADRATURE_FAILED'
        ),
    )
