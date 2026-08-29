"""Hessian and Ward inheritance across a refinement embedding.

Let a coarse action be the pullback of a fine action,

``Gamma_c(x) = Gamma_f(F(x))``.

For a nonlinear refinement map ``F`` the ordinary Hessian obeys

``H_c = J.T H_f J + sum_A (grad Gamma_f)_A (d2 F^A)``,

where ``J=dF``.  Hence the bilinear pullback is exact either for a fixed
linear embedding or at a fine stationary point.  If the refinement is linear,
the gauge generators are field independent, they intertwine as
``J G_c = G_f R``, and the fine Hessian has the Ward null directions
``H_f G_f=0``, then

``H_c G_c = J.T H_f G_f R = 0``.

The theorem is finite-dimensional linear algebra and is also accompanied by
an exact residual decomposition for approximate data.  The concrete audit
uses a dimensionless momentum and the already supplied Fierz--Pauli kernel to
build a nontrivial duplicated refinement witness.  It proves the bridge once
its action/Hessian inputs are supplied.  It does not derive those inputs from
EPRL, a proper spin-foam state sum, or the distributional rigging pairing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from examples.physics.lattice_fierz_pauli_refinement import (
    linearized_gauge_direction_matrix,
)
from examples.physics.two_derivative_spin2_uniqueness import (
    general_two_derivative_spin2_symbol,
)


_ETA_DIAGONAL = np.asarray((-1.0, 1.0, 1.0, 1.0))
_COMPONENTS = tuple(
    (first, second) for first in range(4) for second in range(first, 4)
)


def _finite_matrix(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return matrix


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite nonempty vector")
    return vector


def composed_action_hessian(
    fine_gradient: Sequence[float],
    fine_hessian: Sequence[Sequence[float]],
    embedding_jacobian: Sequence[Sequence[float]],
    embedding_second_derivatives: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Return the exact ordinary Hessian of ``Gamma_f(F(x))`` at one point.

    ``embedding_second_derivatives[A, i, j]`` is ``d_i d_j F^A``.
    The second term vanishes for a linear embedding or a stationary fine
    action, making the ordinary Hessian equal to the bilinear pullback.
    """

    gradient = _finite_vector("fine_gradient", fine_gradient)
    hessian = _finite_matrix("fine_hessian", fine_hessian)
    jacobian = _finite_matrix("embedding_jacobian", embedding_jacobian)
    second_derivatives = np.asarray(embedding_second_derivatives, dtype=float)
    fine_dimension = gradient.size
    if hessian.shape != (fine_dimension, fine_dimension):
        raise ValueError("fine_hessian shape must match fine_gradient")
    if jacobian.shape[0] != fine_dimension:
        raise ValueError("embedding_jacobian row count must match fine dimension")
    coarse_dimension = jacobian.shape[1]
    if (
        second_derivatives.shape
        != (fine_dimension, coarse_dimension, coarse_dimension)
        or not np.all(np.isfinite(second_derivatives))
    ):
        raise ValueError(
            "embedding_second_derivatives must have shape fine-by-coarse-by-coarse"
        )
    pullback = jacobian.T @ hessian @ jacobian
    chain_rule_extra = np.tensordot(
        gradient, second_derivatives, axes=(0, 0)
    )
    return np.asarray(pullback + chain_rule_extra, dtype=float)


@dataclass(frozen=True)
class WardPullbackCertificate:
    coarse_dimension: int
    fine_dimension: int
    coarse_gauge_parameter_count: int
    fine_gauge_parameter_count: int
    fine_self_adjoint_residual: float
    coarse_self_adjoint_residual: float
    hessian_pullback_residual: float
    gauge_intertwining_residual: float
    fine_ward_residual: float
    fine_left_ward_residual: float
    coarse_ward_residual: float
    coarse_left_ward_residual: float
    exact_residual_decomposition_error: float
    coarse_ward_triangle_bound: float
    coarse_ward_within_triangle_bound: bool
    exact_hessian_pullback: bool
    exact_gauge_intertwining: bool
    exact_fine_ward_identity: bool
    exact_fine_left_ward_identity: bool
    exact_coarse_ward_identity: bool
    exact_coarse_left_ward_identity: bool


def certify_linear_refinement_ward_pullback(
    coarse_hessian: Sequence[Sequence[float]],
    fine_hessian: Sequence[Sequence[float]],
    embedding_jacobian: Sequence[Sequence[float]],
    coarse_gauge_generators: Sequence[Sequence[float]],
    fine_gauge_generators: Sequence[Sequence[float]],
    gauge_parameter_refinement: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-10,
) -> WardPullbackCertificate:
    """Certify exact or approximate Hessian/Ward inheritance.

    With residuals

    ``R_H=H_c-J.T H_f J``, ``R_I=J G_c-G_f R``, ``R_W=H_f G_f``,

    the following identity is checked directly:

    ``H_c G_c = R_H G_c + J.T H_f R_I + J.T R_W R``.
    """

    coarse = _finite_matrix("coarse_hessian", coarse_hessian)
    fine = _finite_matrix("fine_hessian", fine_hessian)
    embedding = _finite_matrix("embedding_jacobian", embedding_jacobian)
    coarse_gauge = _finite_matrix(
        "coarse_gauge_generators", coarse_gauge_generators
    )
    fine_gauge = _finite_matrix("fine_gauge_generators", fine_gauge_generators)
    parameter_refinement = _finite_matrix(
        "gauge_parameter_refinement", gauge_parameter_refinement
    )
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    coarse_dimension = coarse.shape[0]
    fine_dimension = fine.shape[0]
    if coarse.shape != (coarse_dimension, coarse_dimension):
        raise ValueError("coarse_hessian must be square")
    if fine.shape != (fine_dimension, fine_dimension):
        raise ValueError("fine_hessian must be square")
    if embedding.shape != (fine_dimension, coarse_dimension):
        raise ValueError("embedding_jacobian has incompatible shape")
    if coarse_gauge.shape[0] != coarse_dimension:
        raise ValueError("coarse gauge generators have incompatible shape")
    if fine_gauge.shape[0] != fine_dimension:
        raise ValueError("fine gauge generators have incompatible shape")
    if parameter_refinement.shape != (
        fine_gauge.shape[1],
        coarse_gauge.shape[1],
    ):
        raise ValueError("gauge_parameter_refinement has incompatible shape")

    hessian_residual = coarse - embedding.T @ fine @ embedding
    intertwining_residual = (
        embedding @ coarse_gauge - fine_gauge @ parameter_refinement
    )
    fine_ward = fine @ fine_gauge
    fine_left_ward = fine_gauge.T @ fine
    coarse_ward = coarse @ coarse_gauge
    coarse_left_ward = coarse_gauge.T @ coarse
    first_term = hessian_residual @ coarse_gauge
    second_term = embedding.T @ fine @ intertwining_residual
    third_term = embedding.T @ fine_ward @ parameter_refinement
    reconstructed = first_term + second_term + third_term
    decomposition_error = float(np.linalg.norm(coarse_ward - reconstructed))
    triangle_bound = float(
        np.linalg.norm(first_term)
        + np.linalg.norm(second_term)
        + np.linalg.norm(third_term)
    )
    numerical_slack = tolerance * max(1.0, triangle_bound)
    coarse_ward_norm = float(np.linalg.norm(coarse_ward))

    return WardPullbackCertificate(
        coarse_dimension=coarse_dimension,
        fine_dimension=fine_dimension,
        coarse_gauge_parameter_count=coarse_gauge.shape[1],
        fine_gauge_parameter_count=fine_gauge.shape[1],
        fine_self_adjoint_residual=float(np.linalg.norm(fine - fine.T)),
        coarse_self_adjoint_residual=float(np.linalg.norm(coarse - coarse.T)),
        hessian_pullback_residual=float(np.linalg.norm(hessian_residual)),
        gauge_intertwining_residual=float(np.linalg.norm(intertwining_residual)),
        fine_ward_residual=float(np.linalg.norm(fine_ward)),
        fine_left_ward_residual=float(np.linalg.norm(fine_left_ward)),
        coarse_ward_residual=coarse_ward_norm,
        coarse_left_ward_residual=float(np.linalg.norm(coarse_left_ward)),
        exact_residual_decomposition_error=decomposition_error,
        coarse_ward_triangle_bound=triangle_bound,
        coarse_ward_within_triangle_bound=(
            coarse_ward_norm <= triangle_bound + numerical_slack
        ),
        exact_hessian_pullback=np.linalg.norm(hessian_residual) <= tolerance,
        exact_gauge_intertwining=(
            np.linalg.norm(intertwining_residual) <= tolerance
        ),
        exact_fine_ward_identity=np.linalg.norm(fine_ward) <= tolerance,
        exact_fine_left_ward_identity=(
            np.linalg.norm(fine_left_ward) <= tolerance
        ),
        exact_coarse_ward_identity=coarse_ward_norm <= tolerance,
        exact_coarse_left_ward_identity=(
            np.linalg.norm(coarse_left_ward) <= tolerance
        ),
    )


@dataclass(frozen=True)
class StationaryRefinementWardAudit:
    dimensionless_momentum_up: tuple[float, float, float, float]
    coarse_field_dimension: int
    fine_field_dimension: int
    refinement_relative_mode_scale: float
    embedding_is_linear: bool
    gauge_generators_are_field_independent: bool
    embedding_isometry_residual: float
    average_projector_idempotence_residual: float
    relative_projector_idempotence_residual: float
    relative_projector_embedding_residual: float
    constructed_quadratic_action_cylindrical_residual: float
    certificate: WardPullbackCertificate
    nonlinear_off_stationary_pullback_residual: float
    nonlinear_stationary_pullback_residual: float
    linear_off_stationary_pullback_residual: float
    exact_linear_refinement_hessian_pullback_closed: bool
    conditional_nonlinear_pullback_at_stationary_point_closed: bool
    fine_ward_identity_inherited_by_coarse_kernel: bool
    nonlinear_off_stationary_counterexample_closed: bool
    rigging_pairing_cylindricity_implies_action_hessian_consistency: bool
    microscopic_spin_foam_effective_action_supplied: bool
    proper_vertex_multicell_hessian_refinement_derived: bool
    ce_effective_kernel_proved_to_lie_in_two_derivative_ansatz: bool
    status: str
    claim_ceiling: str = (
        "EXACT_CONDITIONAL_HESSIAN_WARD_PULLBACK_NOT_MICROSCOPIC_KERNEL_DERIVATION"
    )


def audit_stationary_refinement_ward_bridge(
    dimensionless_momentum_up: Sequence[float] = (1.2, 0.3, -0.4, 0.8),
    *,
    refinement_relative_mode_scale: float = 3.0,
    tolerance: float = 1.0e-10,
) -> StationaryRefinementWardAudit:
    """Build and audit a duplicated quadratic refinement witness."""

    momentum = _finite_vector("dimensionless_momentum_up", dimensionless_momentum_up)
    if momentum.shape != (4,):
        raise ValueError("dimensionless_momentum_up must contain four values")
    relative_scale = float(refinement_relative_mode_scale)
    tolerance = float(tolerance)
    if not math.isfinite(relative_scale) or relative_scale <= 0.0:
        raise ValueError("refinement_relative_mode_scale must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    kernel = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -1.0)
    )
    weights = np.asarray(
        [
            (1.0 if mu == nu else 2.0)
            * _ETA_DIAGONAL[mu]
            * _ETA_DIAGONAL[nu]
            for mu, nu in _COMPONENTS
        ]
    )
    coarse_hessian = np.diag(weights) @ kernel
    coarse_dimension = coarse_hessian.shape[0]
    embedding = np.vstack((np.eye(coarse_dimension), np.eye(coarse_dimension)))
    embedding /= math.sqrt(2.0)
    average_projector = embedding @ embedding.T
    relative_projector = np.eye(2 * coarse_dimension) - average_projector
    fine_hessian = (
        embedding @ coarse_hessian @ embedding.T
        + relative_scale * relative_projector
    )
    coarse_gauge = linearized_gauge_direction_matrix(momentum)
    fine_gauge = embedding @ coarse_gauge
    parameter_refinement = np.eye(coarse_gauge.shape[1])
    certificate = certify_linear_refinement_ward_pullback(
        coarse_hessian,
        fine_hessian,
        embedding,
        coarse_gauge,
        fine_gauge,
        parameter_refinement,
        tolerance=tolerance,
    )

    embedding_isometry_residual = float(
        np.linalg.norm(embedding.T @ embedding - np.eye(coarse_dimension))
    )
    average_projector_residual = float(
        np.linalg.norm(average_projector @ average_projector - average_projector)
    )
    relative_projector_residual = float(
        np.linalg.norm(relative_projector @ relative_projector - relative_projector)
    )
    relative_embedding_residual = float(
        np.linalg.norm(relative_projector @ embedding)
    )
    probes = (
        np.arange(1.0, coarse_dimension + 1.0),
        np.linspace(-1.0, 1.0, coarse_dimension),
        np.eye(coarse_dimension)[0],
        np.eye(coarse_dimension)[-1],
    )
    action_residuals = []
    for probe in probes:
        coarse_action = 0.5 * float(probe @ coarse_hessian @ probe)
        embedded_probe = embedding @ probe
        fine_action = 0.5 * float(embedded_probe @ fine_hessian @ embedded_probe)
        action_residuals.append(abs(coarse_action - fine_action))
    action_residual = max(action_residuals)

    nonlinear_pullback = np.asarray([[2.0]]) @ np.asarray([[5.0]]) @ np.asarray(
        [[2.0]]
    )
    nonlinear_off_stationary = composed_action_hessian(
        (3.0,), ((5.0,),), ((2.0,),), (((4.0,),),)
    )
    nonlinear_stationary = composed_action_hessian(
        (0.0,), ((5.0,),), ((2.0,),), (((4.0,),),)
    )
    linear_off_stationary = composed_action_hessian(
        (3.0,), ((5.0,),), ((2.0,),), (((0.0,),),)
    )
    nonlinear_off_residual = float(
        np.linalg.norm(nonlinear_off_stationary - nonlinear_pullback)
    )
    nonlinear_stationary_residual = float(
        np.linalg.norm(nonlinear_stationary - nonlinear_pullback)
    )
    linear_off_residual = float(
        np.linalg.norm(linear_off_stationary - nonlinear_pullback)
    )

    linear_closed = (
        action_residual <= tolerance
        and embedding_isometry_residual <= tolerance
        and average_projector_residual <= tolerance
        and relative_projector_residual <= tolerance
        and relative_embedding_residual <= tolerance
        and certificate.exact_hessian_pullback
        and certificate.fine_self_adjoint_residual <= tolerance
        and certificate.coarse_self_adjoint_residual <= tolerance
    )
    ward_inherited = (
        certificate.exact_gauge_intertwining
        and certificate.exact_fine_ward_identity
        and certificate.exact_fine_left_ward_identity
        and certificate.exact_coarse_ward_identity
        and certificate.exact_coarse_left_ward_identity
        and certificate.exact_residual_decomposition_error <= tolerance
        and certificate.coarse_ward_within_triangle_bound
    )
    nonlinear_closed = (
        nonlinear_off_residual > tolerance
        and nonlinear_stationary_residual <= tolerance
        and linear_off_residual <= tolerance
    )
    closed = linear_closed and ward_inherited and nonlinear_closed

    return StationaryRefinementWardAudit(
        dimensionless_momentum_up=tuple(float(value) for value in momentum),
        coarse_field_dimension=coarse_dimension,
        fine_field_dimension=2 * coarse_dimension,
        refinement_relative_mode_scale=relative_scale,
        embedding_is_linear=True,
        gauge_generators_are_field_independent=True,
        embedding_isometry_residual=embedding_isometry_residual,
        average_projector_idempotence_residual=average_projector_residual,
        relative_projector_idempotence_residual=relative_projector_residual,
        relative_projector_embedding_residual=relative_embedding_residual,
        constructed_quadratic_action_cylindrical_residual=action_residual,
        certificate=certificate,
        nonlinear_off_stationary_pullback_residual=nonlinear_off_residual,
        nonlinear_stationary_pullback_residual=nonlinear_stationary_residual,
        linear_off_stationary_pullback_residual=linear_off_residual,
        exact_linear_refinement_hessian_pullback_closed=linear_closed,
        conditional_nonlinear_pullback_at_stationary_point_closed=(
            nonlinear_stationary_residual <= tolerance
        ),
        fine_ward_identity_inherited_by_coarse_kernel=ward_inherited,
        nonlinear_off_stationary_counterexample_closed=nonlinear_closed,
        rigging_pairing_cylindricity_implies_action_hessian_consistency=False,
        microscopic_spin_foam_effective_action_supplied=False,
        proper_vertex_multicell_hessian_refinement_derived=False,
        ce_effective_kernel_proved_to_lie_in_two_derivative_ansatz=False,
        status=(
            "CONDITIONAL_REFINEMENT_HESSIAN_WARD_BRIDGE_CLOSED"
            if closed
            else "REFINEMENT_HESSIAN_WARD_BRIDGE_AUDIT_FAILED"
        ),
    )
