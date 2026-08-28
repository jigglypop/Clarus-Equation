"""Minimal split/merge block-RG audit for the Planck-rendering bridge.

The microscopic event emits a Poisson(D) number of candidate continuations.
Each continuation is independently marked either as a new rendered continuation
or as a recombination/face event. The critical marking is chosen so that the
mean number of *independent* rendered continuations is one.

This module deliberately distinguishes three statements:

1. the critical first moment is exactly fixed under blocking;
2. the full Poisson offspring law is not fixed under blocking;
3. after conditioning on one persistent rendered lineage, the local spine
   environment is shift-invariant and supplies an exact minimal fixed object.

The construction is a probability/RG toy. It does not derive the simplicity
constraint, the Plebanski amplitude, or physical spacetime from the branching
law alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class CriticalSplitMerge:
    microscopic_branch_mean: float
    distinct_probability: float
    merge_probability: float
    distinct_intensity: float
    face_intensity: float
    status: str = "MEAN_CRITICAL_SPLIT_MERGE"


def critical_split_merge(branch_mean: float) -> CriticalSplitMerge:
    """Return the unique independent marking with mean rendered offspring one.

    If K~Poisson(D) and every candidate is independently marked rendered with
    probability s, Poisson thinning gives

        S~Poisson(D*s), F~Poisson(D*(1-s)), S independent of F.

    Critical visible capacity requires E[S]=1, hence s=1/D.
    """

    _require_finite("branch_mean", branch_mean)
    if branch_mean <= 1.0:
        raise ValueError("branch_mean must exceed one for a non-zero merge sector")
    distinct_probability = 1.0 / branch_mean
    merge_probability = 1.0 - distinct_probability
    return CriticalSplitMerge(
        microscopic_branch_mean=branch_mean,
        distinct_probability=distinct_probability,
        merge_probability=merge_probability,
        distinct_intensity=1.0,
        face_intensity=branch_mean - 1.0,
    )


def marked_joint_probability(
    *,
    branch_mean: float,
    distinct_probability: float,
    distinct_count: int,
    face_count: int,
) -> float:
    """Return P(S=s,F=f) for independently marked Poisson candidates."""

    _require_finite("branch_mean", branch_mean)
    _require_finite("distinct_probability", distinct_probability)
    if branch_mean < 0.0:
        raise ValueError("branch_mean must be non-negative")
    if not 0.0 <= distinct_probability <= 1.0:
        raise ValueError("distinct_probability must lie in [0, 1]")
    _require_nonnegative_integer("distinct_count", distinct_count)
    _require_nonnegative_integer("face_count", face_count)
    rendered_mean = branch_mean * distinct_probability
    face_mean = branch_mean * (1.0 - distinct_probability)
    return (
        math.exp(-rendered_mean)
        * rendered_mean**distinct_count
        / math.factorial(distinct_count)
        * math.exp(-face_mean)
        * face_mean**face_count
        / math.factorial(face_count)
    )


def blocked_rendered_mean(rendered_mean: float, depth: int) -> float:
    """Return the mean number of outputs after blocking ``depth`` generations."""

    _require_finite("rendered_mean", rendered_mean)
    if rendered_mean < 0.0:
        raise ValueError("rendered_mean must be non-negative")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    return rendered_mean**depth


@dataclass(frozen=True)
class CriticalBlockMoments:
    depth: int
    output_mean: float
    output_variance: float
    expected_parent_events: float
    expected_face_events: float
    face_event_variance: float
    poisson_family_closed: bool
    status: str = "CRITICAL_MEAN_FIXED_FULL_LAW_NOT_FIXED"


def critical_block_moments(branch_mean: float, depth: int) -> CriticalBlockMoments:
    """Return exact first/second moments for a depth-block at criticality.

    For a critical Poisson Galton--Watson process starting from one event,
    E[Z_n]=1 and Var(Z_n)=n. If every active parent independently emits
    Poisson(mu) face events with mu=D-1, the expected block face count is
    mu*depth. The full offspring law broadens with depth and is therefore not
    a Poisson fixed distribution.
    """

    params = critical_split_merge(branch_mean)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    mu = params.face_intensity
    parent_count_variance = (depth - 1) * depth * (2 * depth - 1) / 6.0
    face_variance = mu * depth + mu * mu * parent_count_variance
    return CriticalBlockMoments(
        depth=depth,
        output_mean=1.0,
        output_variance=float(depth),
        expected_parent_events=float(depth),
        expected_face_events=mu * depth,
        face_event_variance=face_variance,
        poisson_family_closed=(depth == 1),
    )


def q_spine_distinct_probability(distinct_count: int) -> float:
    """Return the size-biased critical Poisson offspring law on the spine.

    Ordinary rendered offspring is S~Poisson(1). Conditioning on a persistent
    lineage gives the Doob h-transform/size-biased law

        P_Q(S=k)=k P(S=k), k>=1,

    which is exactly S=1+Poisson(1).
    """

    _require_nonnegative_integer("distinct_count", distinct_count)
    if distinct_count < 1:
        return 0.0
    return math.exp(-1.0) / math.factorial(distinct_count - 1)


@dataclass(frozen=True)
class SpineFixedPoint:
    rendered_continuation_mean: float
    persistent_spine_count: int
    folded_side_branch_mean: float
    face_event_mean: float
    shift_invariant_local_law: bool
    status: str = "SPINE_CONDITIONED_LOCAL_RG_FIXED_POINT"


def spine_fixed_point(branch_mean: float) -> SpineFixedPoint:
    """Return the exact local law seen from one persistent rendered lineage."""

    params = critical_split_merge(branch_mean)
    return SpineFixedPoint(
        rendered_continuation_mean=2.0,
        persistent_spine_count=1,
        folded_side_branch_mean=1.0,
        face_event_mean=params.face_intensity,
        shift_invariant_local_law=True,
    )


def critical_side_tree_total_progeny_probability(total_vertices: int) -> float:
    """Return the Borel(1) law for an ordinary critical Poisson side tree.

    The tree is finite almost surely, but its mean total progeny diverges. The
    probability mass has the scale-free asymptotic n^(-3/2)/sqrt(2*pi).
    """

    if isinstance(total_vertices, bool) or not isinstance(total_vertices, int):
        raise ValueError("total_vertices must be a positive integer")
    if total_vertices < 1:
        raise ValueError("total_vertices must be a positive integer")
    n = total_vertices
    log_probability = -n + (n - 1) * math.log(n) - math.lgamma(n + 1)
    return math.exp(log_probability)


def critical_borel_asymptotic_ratio(total_vertices: int) -> float:
    """Return P(N=n)*sqrt(2*pi)*n^(3/2), which tends to one."""

    probability = critical_side_tree_total_progeny_probability(total_vertices)
    return probability * math.sqrt(2.0 * math.pi) * total_vertices**1.5


def heat_time_from_area(
    *,
    area: float,
    planck_area: float,
    normalization: float = 1.0,
) -> float:
    """Return the additive dimensionless heat time alpha*A/A_P."""

    for name, value in (
        ("area", area),
        ("planck_area", planck_area),
        ("normalization", normalization),
    ):
        _require_finite(name, value)
    if area < 0.0:
        raise ValueError("area must be non-negative")
    if planck_area <= 0.0:
        raise ValueError("planck_area must be positive")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive")
    return normalization * area / planck_area


@dataclass(frozen=True)
class BlockRGVerdict:
    critical_rendered_mean: float
    critical_face_intensity: float
    full_poisson_measure_fixed: bool
    spine_local_measure_fixed: bool
    side_sector_scale_free: bool
    remaining_obligation: str


def block_rg_verdict(branch_mean: float) -> BlockRGVerdict:
    """Summarize what the minimal finite block calculation closes."""

    params = critical_split_merge(branch_mean)
    return BlockRGVerdict(
        critical_rendered_mean=params.distinct_intensity,
        critical_face_intensity=params.face_intensity,
        full_poisson_measure_fixed=False,
        spine_local_measure_fixed=True,
        side_sector_scale_free=True,
        remaining_obligation=(
            "derive the face attachment topology and simplicity amplitude; "
            "the split/merge count law alone does not produce Plebanski gravity"
        ),
    )
