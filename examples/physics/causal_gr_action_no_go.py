"""Finite no-go for selecting gravity dynamics from bare face incidence.

Let a nonempty finite 2-complex supply only a set of faces and relabeling
symmetry.  Put one dimensionless real label x_f on each face and fix the same
normalized product Gaussian measure for every candidate action.  Both

    S_flat(x) = 0,
    S_quad(x) = (lambda / 2) * sum_f x_f^2

are dimensionless and invariant under every face permutation, but they have
different stationary sets, Hessian ranks, and partition amplitudes.  Therefore
bare incidence plus relabeling invariance cannot uniquely select an action,
Plebanski gravity, a saddle, or a unique degree-of-freedom count.

The theorem is an underdetermination result.  It does not rule out a declared
Plebanski/spin-foam action; it proves that such a choice is an additional
physical axiom rather than a consequence of the bare combinatorics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from examples.physics.lorentzian_bivector_reconstruction import (
    common_linear_simplicity_nullity,
)


@dataclass(frozen=True)
class ActionUnderdeterminationAudit:
    face_count: int
    coupling: float
    same_incidence: bool
    same_normalized_measure: bool
    both_relabel_invariant: bool
    flat_stationary_dimension: int
    quadratic_stationary_dimension: int
    flat_hessian_rank: int
    quadratic_hessian_rank: int
    flat_partition_amplitude: complex
    quadratic_partition_amplitude: complex
    unique_action_selected: bool
    status: str = "BARE_INCIDENCE_CANNOT_SELECT_ACTION"
    claim_ceiling: str = "FINITE_REGULATED_ACTION_UNDERDETERMINATION"


@dataclass(frozen=True)
class BFSimplicityCounterexampleAudit:
    face_count: int
    identity_holonomy_flatness_residual: float
    internal_edge_closure_residual: float
    common_simplicity_normal_nullity: int
    finite_bf_saddle_conditions: bool
    linear_simplicity_sector: bool
    same_incidence_selects_constrained_gravity: bool
    status: str = "BF_SADDLE_REJECTED_BY_LINEAR_SIMPLICITY"
    claim_ceiling: str = "FINITE_BF_VS_SIMPLICITY_COUNTEREXAMPLE"


@dataclass(frozen=True)
class MeasureUnderdeterminationAudit:
    face_count: int
    first_variance: float
    second_variance: float
    both_normalized: bool
    both_relabel_invariant: bool
    first_total_second_moment: float
    second_total_second_moment: float
    unique_measure_selected: bool
    status: str = "BARE_INCIDENCE_CANNOT_SELECT_MEASURE"
    claim_ceiling: str = "FINITE_NORMALIZED_MEASURE_UNDERDETERMINATION"


def _validated_labels(labels: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in labels)
    if not values:
        raise ValueError("labels must contain at least one face value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("face labels must be finite")
    return values


def flat_action(labels: tuple[float, ...]) -> float:
    """Return the identically zero dimensionless action."""

    _validated_labels(labels)
    return 0.0


def quadratic_action(labels: tuple[float, ...], *, coupling: float) -> float:
    """Return a relabel-invariant dimensionless quadratic action."""

    values = _validated_labels(labels)
    if not math.isfinite(coupling) or coupling == 0.0:
        raise ValueError("coupling must be finite and nonzero")
    return 0.5 * coupling * math.fsum(value * value for value in values)


def flat_gradient(labels: tuple[float, ...]) -> tuple[float, ...]:
    values = _validated_labels(labels)
    return (0.0,) * len(values)


def quadratic_gradient(
    labels: tuple[float, ...],
    *,
    coupling: float,
) -> tuple[float, ...]:
    values = _validated_labels(labels)
    if not math.isfinite(coupling) or coupling == 0.0:
        raise ValueError("coupling must be finite and nonzero")
    return tuple(coupling * value for value in values)


def action_underdetermination_audit(
    face_count: int,
    *,
    coupling: float = 1.0,
) -> ActionUnderdeterminationAudit:
    """Return the constructive countermodel on one fixed finite face set.

    The common measure is the normalized product Gaussian.  Its two exact
    amplitudes are

        Z_flat = 1,
        Z_quad = (1 - i*lambda)^(-F/2).

    The power uses the principal branch, equivalently analytic continuation
    from lambda=0; Re(1-i*lambda)=1 keeps that continuation unambiguous.
    Since the actions obey the same incidence/relabeling assumptions while
    producing inequivalent dynamics, those assumptions do not entail a unique
    action.
    """

    if isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 1:
        raise ValueError("face_count must be a positive integer")
    if not math.isfinite(coupling) or coupling == 0.0:
        raise ValueError("coupling must be finite and nonzero")

    witness = tuple(float(index + 1) / face_count for index in range(face_count))
    permuted = tuple(reversed(witness))
    both_relabel_invariant = (
        flat_action(witness) == flat_action(permuted)
        and math.isclose(
            quadratic_action(witness, coupling=coupling),
            quadratic_action(permuted, coupling=coupling),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )
    return ActionUnderdeterminationAudit(
        face_count=face_count,
        coupling=coupling,
        same_incidence=True,
        same_normalized_measure=True,
        both_relabel_invariant=both_relabel_invariant,
        flat_stationary_dimension=face_count,
        quadratic_stationary_dimension=0,
        flat_hessian_rank=0,
        quadratic_hessian_rank=face_count,
        flat_partition_amplitude=1.0 + 0.0j,
        quadratic_partition_amplitude=complex(1.0, -coupling)
        ** (-0.5 * face_count),
        unique_action_selected=False,
    )


def bf_simplicity_counterexample() -> BFSimplicityCounterexampleAudit:
    """Return a four-face flat-BF witness outside every simple-normal sector.

    The four faces meet one internal edge; all other edges are treated as fixed
    boundary data.  Identity holonomies solve the discrete flatness equation,
    while B1+B2+B3+B4=0 solves the internal-edge BF closure equation.  The
    common-normal linear system has nullity zero, so the same configuration is
    excluded by a Plebanski linear-simplicity constraint.
    """

    basis = np.eye(4)
    first = np.outer(basis[0], basis[1]) - np.outer(basis[1], basis[0])
    second = np.outer(basis[2], basis[3]) - np.outer(basis[3], basis[2])
    bivectors = np.asarray((first, second, -first, -second))
    scale = float(sum(np.linalg.norm(item) for item in bivectors))
    closure_residual = float(np.linalg.norm(np.sum(bivectors, axis=0))) / scale
    nullity = common_linear_simplicity_nullity(bivectors)
    bf_conditions = closure_residual <= 1.0e-15
    linear_simple = nullity > 0
    return BFSimplicityCounterexampleAudit(
        face_count=4,
        identity_holonomy_flatness_residual=0.0,
        internal_edge_closure_residual=closure_residual,
        common_simplicity_normal_nullity=nullity,
        finite_bf_saddle_conditions=bf_conditions,
        linear_simplicity_sector=linear_simple,
        same_incidence_selects_constrained_gravity=False,
    )


def measure_underdetermination_audit(
    face_count: int,
    *,
    first_variance: float = 1.0,
    second_variance: float = 2.0,
) -> MeasureUnderdeterminationAudit:
    """Show that incidence and relabeling do not select a unique measure.

    Product zero-mean Gaussians with two distinct positive variances are both
    normalized and invariant under every face permutation.  Their invariant
    observable E[sum_f x_f^2]=F*sigma^2 differs, so the measures are not equal.
    """

    if isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 1:
        raise ValueError("face_count must be a positive integer")
    for name, variance in (
        ("first_variance", first_variance),
        ("second_variance", second_variance),
    ):
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if first_variance == second_variance:
        raise ValueError("the two variances must be distinct")
    return MeasureUnderdeterminationAudit(
        face_count=face_count,
        first_variance=first_variance,
        second_variance=second_variance,
        both_normalized=True,
        both_relabel_invariant=True,
        first_total_second_moment=face_count * first_variance,
        second_total_second_moment=face_count * second_variance,
        unique_measure_selected=False,
    )
