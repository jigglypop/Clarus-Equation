"""Normalization-invariant audit for observation-selected Schur rendering.

For a real Euclidean quadratic kernel ``H=[[A,B],[B.T,C]]``, eliminating the
unobserved block gives ``K_eff=A-B C^-1 B.T``.  The raw kernel changes when the
retained variable is rescaled.  Its invariant content is the generalized
spectrum of ``(K_eff,A)``; in one dimension this is
``chi=B^2/(A C)`` and ``Z=K_eff/A=1-chi``.

This is an algebraic audit, not a dark-energy prediction.  Lorentzian gravity
still requires gauge projection, a retarded/CTP kernel, physical pole residues,
and a background metric variation.  The probability-to-kernel, temporal versus
spatial channel, and event-depth-to-redshift maps remain open physical bridges.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from examples.physics.gaussian_refinement_schur_kernel import (
    schur_complement_effective_hessian,
)


DEFAULT_DELTA = 0.17775842340997383
DEFAULT_SPATIAL_DIMENSION = 3.0
DEFAULT_D_EFF = 3.1777584234099736
DEFAULT_Q_LOW = 0.048646719644028225


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_matrix(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return matrix


def _inverse_square_root_spd(
    matrix: np.ndarray, *, name: str, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if np.linalg.norm(matrix - matrix.T) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) <= tolerance:
        raise ValueError(f"{name} must be positive definite")
    inverse_square_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    return np.asarray(inverse_square_root, dtype=float), eigenvalues


@dataclass(frozen=True)
class ProbabilityFlowCertificate:
    d_eff: float
    q: float
    update: float
    fixed_point_residual: float
    local_derivative: float
    locally_attracting: bool
    unit_branch_derivative: float
    unit_branch_repelling: bool
    event_depth_to_scale_factor_derived: bool = False
    probability_to_kernel_derived: bool = False
    probability_is_energy_density: bool = False
    prediction: bool = False
    status: str = "ZERO_D_PROBABILITY_FLOW_ONLY_PHYSICAL_MAPS_OPEN"


def poisson_probability_step(q: float, d_eff: float) -> float:
    """Return the dimensionless zero-dimensional update exp[-D(1-q)]."""

    q = _finite(q, "q")
    d_eff = _positive(d_eff, "d_eff")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    return math.exp(-d_eff * (1.0 - q))


def audit_probability_flow(
    d_eff: float = DEFAULT_D_EFF,
    q: float = DEFAULT_Q_LOW,
    *,
    tolerance: float = 1.0e-12,
) -> ProbabilityFlowCertificate:
    """Audit a supplied fixed-point branch without identifying it with energy."""

    tolerance = _positive(tolerance, "tolerance")
    d_eff = _positive(d_eff, "d_eff")
    q = _finite(q, "q")
    update = poisson_probability_step(q, d_eff)
    derivative = d_eff * update
    return ProbabilityFlowCertificate(
        d_eff=d_eff,
        q=q,
        update=update,
        fixed_point_residual=update - q,
        local_derivative=derivative,
        locally_attracting=(abs(update - q) <= tolerance and derivative < 1.0),
        unit_branch_derivative=d_eff,
        unit_branch_repelling=d_eff > 1.0,
    )


def continuous_event_flow_beta(q: float, d_eff: float, event_rate: float) -> float:
    """Return beta_q=nu(a)[F_D(q)-q] for a supplied event-rate map.

    This relaxation shares the discrete fixed points but is not a unique
    continuous embedding of the discrete recursion.
    """

    event_rate = _finite(event_rate, "event_rate")
    if event_rate < 0.0:
        raise ValueError("event_rate must be non-negative")
    q = _finite(q, "q")
    return event_rate * (poisson_probability_step(q, d_eff) - q)


def required_event_rate_for_target_flow(
    q: float,
    d_eff: float,
    target_beta: float,
    *,
    tolerance: float = 1.0e-12,
) -> float:
    """Inverse-reconstruct ``nu`` in beta=nu(F_D(q)-q) off a fixed point.

    The function makes the continuous-embedding non-uniqueness executable: an
    arbitrary compatible target beta fixes a different supplied event rate.
    At a fixed point the rate is unidentifiable and a nonzero beta is
    inconsistent, so no value is returned there.
    """

    q = _finite(q, "q")
    target_beta = _finite(target_beta, "target_beta")
    tolerance = _positive(tolerance, "tolerance")
    gap = poisson_probability_step(q, d_eff) - q
    if abs(gap) <= tolerance:
        raise ValueError("event rate is not identifiable at a fixed point")
    event_rate = target_beta / gap
    if event_rate < 0.0:
        raise ValueError("target_beta requires a negative event rate")
    return event_rate


@dataclass(frozen=True)
class ControlledDepthRenderingSequence:
    steps: int
    d_eff: float
    geometric_loss: float
    initial_q: float
    low_branch_upper: float
    image_upper: float
    contraction_bound: float
    minimum_spatial_radicand: float
    q_values: tuple[float, ...]
    spatial_scale_factors: tuple[float, ...]
    interval_invariant: bool
    contraction_certified: bool
    monotone_low_branch: bool
    unique_within_supplied_discrete_map: bool = True
    lambda_step_relation_closed: bool = True
    dimensionless_core_arguments: tuple[tuple[str, str], ...] = (
        ("D * (1 - q_n)", "D and q_n are dimensionless"),
        ("1 - x - q_n**2", "x=delta/d and q_n are dimensionless"),
    )
    physical_protocol_depth_derived: bool = False
    probability_to_spatial_residue_derived: bool = False
    event_depth_to_scale_factor_derived: bool = False
    absolute_dark_energy_density_derived: bool = False
    prediction: bool = False
    status: str = (
        "CONDITIONAL_CONTROLLED_DEPTH_SEQUENCE_NOT_COSMOLOGICAL_PREDICTION"
    )


def controlled_depth_rendering_sequence(
    steps: int,
    *,
    d_eff: float = DEFAULT_D_EFF,
    delta: float = DEFAULT_DELTA,
    spatial_dimension: float = DEFAULT_SPATIAL_DIMENSION,
    initial_q: float = 0.0,
    tolerance: float = 1.0e-12,
) -> ControlledDepthRenderingSequence:
    """Audit the low-branch discrete sequence at controlled protocol depth.

    This deliberately keeps the event-depth label discrete.  It does not map
    depth to scale factor or redshift.  The scale sequence additionally assumes
    the supplied spatial-only map ``Z_s=1-delta/d-q^2`` and ``Z_t=1``.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    d_eff = _positive(d_eff, "d_eff")
    if d_eff <= 1.0:
        raise ValueError("d_eff must exceed one for the selected low branch")
    delta = _finite(delta, "delta")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    spatial_dimension = _positive(spatial_dimension, "spatial_dimension")
    initial_q = _finite(initial_q, "initial_q")
    tolerance = _positive(tolerance, "tolerance")

    low_branch_upper = 1.0 / d_eff
    if not 0.0 <= initial_q <= low_branch_upper:
        raise ValueError("initial_q must lie in the certified low-branch interval")
    geometric_loss = delta / spatial_dimension
    minimum_radicand = 1.0 - geometric_loss - low_branch_upper**2
    if minimum_radicand <= tolerance:
        raise ValueError("spatial scale is not positive on the low-branch interval")

    image_upper = math.exp(-(d_eff - 1.0))
    contraction_bound = d_eff * image_upper
    q_values = [initial_q]
    for _ in range(steps):
        q_values.append(poisson_probability_step(q_values[-1], d_eff))
    spatial_scales = [
        math.sqrt(1.0 - geometric_loss - q_value**2)
        for q_value in q_values
    ]
    return ControlledDepthRenderingSequence(
        steps=steps,
        d_eff=d_eff,
        geometric_loss=geometric_loss,
        initial_q=initial_q,
        low_branch_upper=low_branch_upper,
        image_upper=image_upper,
        contraction_bound=contraction_bound,
        minimum_spatial_radicand=minimum_radicand,
        q_values=tuple(q_values),
        spatial_scale_factors=tuple(spatial_scales),
        interval_invariant=(image_upper <= low_branch_upper + tolerance),
        contraction_certified=(contraction_bound < 1.0),
        monotone_low_branch=all(
            right >= left - tolerance
            for left, right in zip(q_values[:-1], q_values[1:], strict=True)
        ),
    )


def next_controlled_depth_spatial_scale(
    previous_scale: float,
    *,
    d_eff: float = DEFAULT_D_EFF,
    delta: float = DEFAULT_DELTA,
    spatial_dimension: float = DEFAULT_SPATIAL_DIMENSION,
    tolerance: float = 1.0e-12,
) -> float:
    """Return the next conditional scale using only the previous scale.

    Implements the dimensionless relation
    lambda[n+1]^2 = 1-x-exp(-2D(1-sqrt(1-x-lambda[n]^2))).
    """

    previous_scale = _positive(previous_scale, "previous_scale")
    d_eff = _positive(d_eff, "d_eff")
    delta = _finite(delta, "delta")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    spatial_dimension = _positive(spatial_dimension, "spatial_dimension")
    tolerance = _positive(tolerance, "tolerance")
    geometric_loss = delta / spatial_dimension
    q_squared = 1.0 - geometric_loss - previous_scale**2
    if q_squared < -tolerance:
        raise ValueError("previous_scale is outside the supplied rendering map")
    q = math.sqrt(max(0.0, q_squared))
    if d_eff <= 1.0 or q > 1.0 / d_eff + tolerance:
        raise ValueError("previous_scale does not encode the certified low branch")
    q_next = poisson_probability_step(q, d_eff)
    next_radicand = 1.0 - geometric_loss - q_next**2
    if next_radicand <= tolerance:
        raise ValueError("next spatial scale is not positive")
    return math.sqrt(next_radicand)


@dataclass(frozen=True)
class ScalarNormalizedSchurCertificate:
    boundary: float
    mixing: float
    internal: float
    raw_effective_kernel: float
    normalized_loss: float
    retained_factor: float
    strictly_positive: bool
    p_and_q_rescaling_invariant: bool = True
    raw_effective_kernel_is_normalization_invariant: bool = False
    dimensionless_by_quadratic_kernel_homogeneity: bool = True
    prediction: bool = False
    status: str = "SCALAR_NORMALIZED_SCHUR_INVARIANT"


def normalized_scalar_schur_loss(
    boundary: float,
    mixing: float,
    internal: float,
    *,
    tolerance: float = 1.0e-12,
) -> ScalarNormalizedSchurCertificate:
    """Return chi=B^2/(A C) and Z=(A-B^2/C)/A for positive A,C."""

    boundary = _positive(boundary, "boundary")
    mixing = _finite(mixing, "mixing")
    internal = _positive(internal, "internal")
    tolerance = _positive(tolerance, "tolerance")
    normalized_loss = mixing * mixing / (boundary * internal)
    retained_factor = 1.0 - normalized_loss
    raw_effective = boundary - mixing * mixing / internal
    return ScalarNormalizedSchurCertificate(
        boundary=boundary,
        mixing=mixing,
        internal=internal,
        raw_effective_kernel=raw_effective,
        normalized_loss=normalized_loss,
        retained_factor=retained_factor,
        strictly_positive=retained_factor > tolerance,
    )


@dataclass(frozen=True)
class MatrixNormalizedSchurCertificate:
    boundary_dimension: int
    internal_dimension: int
    boundary_minimum_eigenvalue: float
    internal_minimum_eigenvalue: float
    whitened_coupling_singular_values: tuple[float, ...]
    normalized_loss_eigenvalues: tuple[float, ...]
    retained_generalized_eigenvalues: tuple[float, ...]
    determinant_ratio: float
    determinant_ratio_from_spectrum: float
    strictly_positive: bool
    coordinate_redefinition_invariant: bool = True
    gauge_projection_performed: bool = False
    retarded_ctp_kernel_derived: bool = False
    lorentzian_stability_derived: bool = False
    prediction: bool = False
    status: str = "EUCLIDEAN_PHYSICAL_SUBSPACE_SCHUR_INVARIANTS"


def normalized_matrix_schur_spectrum(
    boundary_hessian_block: Sequence[Sequence[float]],
    boundary_internal_mixing: Sequence[Sequence[float]],
    internal_hessian_block: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-12,
) -> MatrixNormalizedSchurCertificate:
    """Return basis-invariant generalized Schur loss and retained spectra."""

    tolerance = _positive(tolerance, "tolerance")
    boundary = _finite_matrix("boundary_hessian_block", boundary_hessian_block)
    mixing = _finite_matrix("boundary_internal_mixing", boundary_internal_mixing)
    internal = _finite_matrix("internal_hessian_block", internal_hessian_block)
    boundary_inverse_sqrt, boundary_eigenvalues = _inverse_square_root_spd(
        boundary, name="boundary_hessian_block", tolerance=tolerance
    )
    internal_inverse_sqrt, internal_eigenvalues = _inverse_square_root_spd(
        internal, name="internal_hessian_block", tolerance=tolerance
    )
    if mixing.shape != (boundary.shape[0], internal.shape[0]):
        raise ValueError("boundary_internal_mixing has incompatible shape")

    effective = schur_complement_effective_hessian(
        boundary, mixing, internal, tolerance=tolerance
    )
    whitened_coupling = boundary_inverse_sqrt @ mixing @ internal_inverse_sqrt
    normalized_loss = whitened_coupling @ whitened_coupling.T
    retained = np.eye(boundary.shape[0]) - normalized_loss
    loss_eigenvalues = np.linalg.eigvalsh(normalized_loss)
    retained_eigenvalues = np.linalg.eigvalsh(retained)
    singular_values = np.linalg.svd(whitened_coupling, compute_uv=False)

    sign_effective, logdet_effective = np.linalg.slogdet(effective)
    sign_boundary, logdet_boundary = np.linalg.slogdet(boundary)
    if sign_boundary <= 0.0:
        raise ValueError("boundary_hessian_block must have positive determinant")
    determinant_ratio = (
        0.0
        if sign_effective == 0.0
        else float(sign_effective * math.exp(logdet_effective - logdet_boundary))
    )
    determinant_ratio_from_spectrum = float(np.prod(retained_eigenvalues))

    return MatrixNormalizedSchurCertificate(
        boundary_dimension=boundary.shape[0],
        internal_dimension=internal.shape[0],
        boundary_minimum_eigenvalue=float(np.min(boundary_eigenvalues)),
        internal_minimum_eigenvalue=float(np.min(internal_eigenvalues)),
        whitened_coupling_singular_values=tuple(float(v) for v in singular_values),
        normalized_loss_eigenvalues=tuple(float(v) for v in loss_eigenvalues),
        retained_generalized_eigenvalues=tuple(float(v) for v in retained_eigenvalues),
        determinant_ratio=determinant_ratio,
        determinant_ratio_from_spectrum=determinant_ratio_from_spectrum,
        strictly_positive=float(np.min(retained_eigenvalues)) > tolerance,
    )


def _positive_square_root_or_none(value: float) -> float | None:
    return math.sqrt(value) if value > 0.0 else None


@dataclass(frozen=True)
class CompositionCandidates:
    geometric_loss: float
    probability_loss: float
    simultaneous_retained_factor: float
    simultaneous_scale_factor: float | None
    sequential_retained_factor: float
    sequential_scale_factor: float | None
    exponential_retained_factor: float
    exponential_scale_factor: float
    sequential_cross_term: float
    simultaneous_is_exact_for_block_diagonal_q: bool = True
    sequential_requires_rewhitened_cascade_axiom: bool = True
    exponential_requires_log_semigroup_axiom: bool = True
    probability_to_coupling_map_derived: bool = False
    composition_selected: bool = False
    prediction: bool = False
    status: str = "COMPOSITION_RULE_UNDERDETERMINED"


def rendering_composition_candidates(
    delta: float, spatial_dimension: float, q: float
) -> CompositionCandidates:
    """Compare simultaneous, sequential, and log-semigroup completions."""

    delta = _finite(delta, "delta")
    spatial_dimension = _positive(spatial_dimension, "spatial_dimension")
    q = _finite(q, "q")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    geometric_loss = delta / spatial_dimension
    probability_loss = q * q
    simultaneous = 1.0 - geometric_loss - probability_loss
    sequential = (1.0 - geometric_loss) * (1.0 - probability_loss)
    exponential = math.exp(-(geometric_loss + probability_loss))
    return CompositionCandidates(
        geometric_loss=geometric_loss,
        probability_loss=probability_loss,
        simultaneous_retained_factor=simultaneous,
        simultaneous_scale_factor=_positive_square_root_or_none(simultaneous),
        sequential_retained_factor=sequential,
        sequential_scale_factor=_positive_square_root_or_none(sequential),
        exponential_retained_factor=exponential,
        exponential_scale_factor=math.sqrt(exponential),
        sequential_cross_term=geometric_loss * probability_loss,
    )


@dataclass(frozen=True)
class RelativePoleReadout:
    temporal_residue: float
    spatial_residue: float
    reference_temporal_residue: float
    reference_spatial_residue: float
    channel_speed_squared: float
    reference_speed_squared: float
    relative_clock_ruler_factor: float
    pure_conformal_cancellation: bool
    common_field_rescaling_invariant: bool = True
    separate_lapse_and_ruler_identified: bool = False
    absolute_hubble_readout_derived: bool = False
    prediction: bool = False
    status: str = "RELATIVE_POLE_RATIO_ONLY"


def relative_pole_readout(
    temporal_residue: float,
    spatial_residue: float,
    reference_temporal_residue: float = 1.0,
    reference_spatial_residue: float = 1.0,
    *,
    tolerance: float = 1.0e-12,
) -> RelativePoleReadout:
    """Return the field-normalization-invariant relative clock/ruler ratio."""

    temporal = _positive(temporal_residue, "temporal_residue")
    spatial = _positive(spatial_residue, "spatial_residue")
    reference_temporal = _positive(
        reference_temporal_residue, "reference_temporal_residue"
    )
    reference_spatial = _positive(
        reference_spatial_residue, "reference_spatial_residue"
    )
    tolerance = _positive(tolerance, "tolerance")
    speed_squared = spatial / temporal
    reference_speed_squared = reference_spatial / reference_temporal
    relative_factor = math.sqrt(speed_squared / reference_speed_squared)
    return RelativePoleReadout(
        temporal_residue=temporal,
        spatial_residue=spatial,
        reference_temporal_residue=reference_temporal,
        reference_spatial_residue=reference_spatial,
        channel_speed_squared=speed_squared,
        reference_speed_squared=reference_speed_squared,
        relative_clock_ruler_factor=relative_factor,
        pure_conformal_cancellation=math.isclose(
            speed_squared,
            reference_speed_squared,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ),
    )


def metric_representative_family(
    relative_clock_ruler_factor: float, conformal_factor: float
) -> tuple[float, float]:
    """Return one of infinitely many (lapse, ruler) pairs with A_R/N_R=lambda."""

    relative_factor = _positive(
        relative_clock_ruler_factor, "relative_clock_ruler_factor"
    )
    conformal_factor = _positive(conformal_factor, "conformal_factor")
    root = math.sqrt(relative_factor)
    return conformal_factor / root, conformal_factor * root


@dataclass(frozen=True)
class ConditionalHubbleReadout:
    temporal_residue_ratio: float
    spatial_residue_ratio: float
    d_log_spatial_residue_ratio_d_log_a: float
    lapse_ratio: float
    ruler_ratio: float
    h_rendered_over_h_reference: float
    same_field_normalization_axiom_required: bool = True
    protocol_clock_ruler_calibration_derived: bool = False
    unique_from_relative_pole_ratio: bool = False
    prediction: bool = False
    status: str = "CONDITIONAL_METRIC_REPRESENTATIVE_READOUT"


def conditional_hubble_readout(
    temporal_residue_ratio: float,
    spatial_residue_ratio: float,
    d_log_spatial_residue_ratio_d_log_a: float,
) -> ConditionalHubbleReadout:
    """Evaluate H_R/H=[1+1/2 dln(Z_s/Z_s0)/dlna]/sqrt(Z_t/Z_t0)."""

    temporal_ratio = _positive(temporal_residue_ratio, "temporal_residue_ratio")
    spatial_ratio = _positive(spatial_residue_ratio, "spatial_residue_ratio")
    derivative = _finite(
        d_log_spatial_residue_ratio_d_log_a,
        "d_log_spatial_residue_ratio_d_log_a",
    )
    lapse = math.sqrt(temporal_ratio)
    ruler = math.sqrt(spatial_ratio)
    return ConditionalHubbleReadout(
        temporal_residue_ratio=temporal_ratio,
        spatial_residue_ratio=spatial_ratio,
        d_log_spatial_residue_ratio_d_log_a=derivative,
        lapse_ratio=lapse,
        ruler_ratio=ruler,
        h_rendered_over_h_reference=(1.0 + 0.5 * derivative) / lapse,
    )


@dataclass(frozen=True)
class ConstructedObservationRenderingAudit:
    delta: float
    spatial_dimension: float
    probability_flow: ProbabilityFlowCertificate
    legacy_raw_witness: ScalarNormalizedSchurCertificate
    simultaneous_schur_witness: MatrixNormalizedSchurCertificate
    composition_candidates: CompositionCandidates
    universal_channel_readout: RelativePoleReadout
    spatial_only_channel_readout: RelativePoleReadout
    temporal_only_channel_readout: RelativePoleReadout
    beta_q_for_unit_event_rate: float
    spatial_log_residue_flow_at_supplied_q: float
    conditional_spatial_hubble_readout: ConditionalHubbleReadout
    controlled_depth_sequence: ControlledDepthRenderingSequence
    raw_legacy_kernel_numerically_equals_simultaneous_factor_only: bool
    q_to_kernel_map_derived: bool = False
    temporal_spatial_channel_assignment_derived: bool = False
    event_depth_to_redshift_map_derived: bool = False
    retarded_ctp_background_variation_derived: bool = False
    absolute_dark_energy_density_derived: bool = False
    prediction: bool = False
    status: str = "INVARIANT_FORMULATION_CLOSED_PHYSICAL_BRIDGES_OPEN"
    claim_ceiling: str = (
        "CONSTRUCTED_NORMALIZATION_INVARIANT_RENDERING_AUDIT_NOT_DARK_ENERGY_"
        "PREDICTION"
    )


def audit_constructed_observation_rendering(
    *,
    delta: float = DEFAULT_DELTA,
    spatial_dimension: float = DEFAULT_SPATIAL_DIMENSION,
    d_eff: float = DEFAULT_D_EFF,
    q: float = DEFAULT_Q_LOW,
    tolerance: float = 1.0e-12,
) -> ConstructedObservationRenderingAudit:
    """Audit all declared completions without selecting one from data proximity."""

    tolerance = _positive(tolerance, "tolerance")
    flow = audit_probability_flow(d_eff, q, tolerance=tolerance)
    candidates = rendering_composition_candidates(delta, spatial_dimension, q)
    geometric_loss = candidates.geometric_loss
    legacy = normalized_scalar_schur_loss(
        1.0 - geometric_loss, q, 1.0, tolerance=tolerance
    )
    simultaneous = normalized_matrix_schur_spectrum(
        ((1.0,),),
        ((math.sqrt(geometric_loss), q),),
        ((1.0, 0.0), (0.0, 1.0)),
        tolerance=tolerance,
    )
    retained = candidates.simultaneous_retained_factor
    if retained <= 0.0:
        raise ValueError("constructed simultaneous retained factor must be positive")

    universal = relative_pole_readout(retained, retained, tolerance=tolerance)
    spatial_only = relative_pole_readout(1.0, retained, tolerance=tolerance)
    temporal_only = relative_pole_readout(retained, 1.0, tolerance=tolerance)
    beta_q = continuous_event_flow_beta(q, d_eff, event_rate=1.0)
    spatial_log_flow = -2.0 * q * beta_q / retained
    hubble = conditional_hubble_readout(1.0, retained, spatial_log_flow)
    controlled_sequence = controlled_depth_rendering_sequence(
        8,
        d_eff=d_eff,
        delta=delta,
        spatial_dimension=spatial_dimension,
        tolerance=tolerance,
    )
    return ConstructedObservationRenderingAudit(
        delta=float(delta),
        spatial_dimension=float(spatial_dimension),
        probability_flow=flow,
        legacy_raw_witness=legacy,
        simultaneous_schur_witness=simultaneous,
        composition_candidates=candidates,
        universal_channel_readout=universal,
        spatial_only_channel_readout=spatial_only,
        temporal_only_channel_readout=temporal_only,
        beta_q_for_unit_event_rate=beta_q,
        spatial_log_residue_flow_at_supplied_q=spatial_log_flow,
        conditional_spatial_hubble_readout=hubble,
        controlled_depth_sequence=controlled_sequence,
        raw_legacy_kernel_numerically_equals_simultaneous_factor_only=math.isclose(
            legacy.raw_effective_kernel,
            retained,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="observation_rendering_invariant_schur")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--spatial-dimension", type=float, default=DEFAULT_SPATIAL_DIMENSION)
    parser.add_argument("--d-eff", type=float, default=DEFAULT_D_EFF)
    parser.add_argument("--q", type=float, default=DEFAULT_Q_LOW)
    args = parser.parse_args()
    audit = audit_constructed_observation_rendering(
        delta=args.delta,
        spatial_dimension=args.spatial_dimension,
        d_eff=args.d_eff,
        q=args.q,
    )
    print(json.dumps(asdict(audit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
