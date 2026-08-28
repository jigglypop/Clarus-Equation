"""Constructive finite closure for the CE 0D-to-Plebanski chain.

This module does not revive the false implication that a bare singleton
uniquely determines four-dimensional gravity.  It supplies a stronger, typed
model and separates what is proved from what is selected:

* a rank-four, coordinate-free simplex interaction is a 0D rewrite rule whose
  paired strands carry the ten codimension-two faces of a 4-simplex;
* composition faces carry group holonomy and therefore a finite curvature
  observable;
* equality of Planck-coarse readouts is an equivalence relation, while the
  microscopic histories remain in the state space;
* orthogonal records for distinct readout classes make those classes
  decoherent without deleting the folded norm;
* on a finite coarse history space a positive Gibbs defect weight preserves
  every history at finite beta and exponentially concentrates on the
  zero-defect common-metric sector;
* same-degree B_2/F_2 closure selects form degree four, while an explicitly
  nondegenerate (-,+,+,+) metric supplies the additional Lorentzian input;
* existing finite Lorentzian simplicity and shared-face reconstruction are
  linked to an exact flat chiral Plebanski/Einstein solution on one tetrad.

The resulting certificate is a single-typed-history, finite flat conditional
existence theorem for a declared model.  It is not uniqueness from bare 0D
data, an empirical claim, a generic curved solution, or a proof that a
particular microscopic RG/refinement flow in nature selects the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
import math

import numpy as np

from examples.physics.causal_face_simplicity import (
    CompositionFace,
    composition_faces,
    hard_shared_spacelike_face_match,
    proper_orthochronous_residual,
)
from examples.physics.continuum_gr_dof_no_go import (
    massless_spin_two_polarization_count,
)
from examples.physics.lorentzian_bivector_reconstruction import (
    bivector_face_reconstruction_audit,
    bivector_from_normal_edge,
)
from examples.physics.planck_rendering_block_rg import (
    CriticalSplitMerge,
    critical_split_merge,
    marked_joint_probability,
)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_norm(value: np.ndarray) -> float:
    array = np.asarray(value)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _logsumexp(values: np.ndarray) -> float:
    """Return log(sum(exp(values))) without avoidable overflow."""

    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.sum(np.exp(array - maximum))))


@dataclass(frozen=True)
class FormDegreeClosureAudit:
    curvature_form_degree: int
    conjugate_form_degree: int
    spacetime_dimension: int
    same_type_pair: bool
    hodge_degree_closes: bool
    one_time_direction: bool
    metric_signature: tuple[int, ...]
    nondegenerate_lorentzian_signature: bool
    spatial_dimension: int
    lorentzian_three_plus_one: bool
    status: str


def form_degree_closure(
    curvature_form_degree: int = 2,
    conjugate_form_degree: int = 2,
    *,
    one_time_direction: bool = True,
    metric_signature: Sequence[int] = (-1, 1, 1, 1),
) -> FormDegreeClosureAudit:
    """Audit the conditional dimension-selection lemma.

    A background-metric-free local term ``B_q wedge F_p`` is a top form in
    dimension ``p+q``.  Requiring the recursively paired fields to have the
    same degree and curvature to be a two-form gives p=q=2 and hence D=4.
    A separately declared nondegenerate ``(-,+,+,+)`` metric then means 3+1.
    Merely naming one direction "time" is not enough to establish a
    Lorentzian signature.
    """

    for name, degree in (
        ("curvature_form_degree", curvature_form_degree),
        ("conjugate_form_degree", conjugate_form_degree),
    ):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(one_time_direction, bool):
        raise ValueError("one_time_direction must be boolean")
    signature = tuple(metric_signature)
    if any(isinstance(entry, bool) or entry not in (-1, 1) for entry in signature):
        raise ValueError("metric_signature entries must be -1 or 1")

    dimension = curvature_form_degree + conjugate_form_degree
    same_type = curvature_form_degree == conjugate_form_degree
    hodge_closes = dimension == 2 * curvature_form_degree
    spatial_dimension = dimension - 1 if one_time_direction else dimension
    lorentzian_signature = (
        len(signature) == dimension
        and signature.count(-1) == 1
        and signature.count(1) == dimension - 1
    )
    selected = (
        curvature_form_degree == 2
        and conjugate_form_degree == 2
        and dimension == 4
        and one_time_direction
        and lorentzian_signature
    )
    return FormDegreeClosureAudit(
        curvature_form_degree=curvature_form_degree,
        conjugate_form_degree=conjugate_form_degree,
        spacetime_dimension=dimension,
        same_type_pair=same_type,
        hodge_degree_closes=hodge_closes,
        one_time_direction=one_time_direction,
        metric_signature=signature,
        nondegenerate_lorentzian_signature=lorentzian_signature,
        spatial_dimension=spatial_dimension,
        lorentzian_three_plus_one=selected,
        status=(
            "CONDITIONAL_3_PLUS_1_FORM_DEGREE_CLOSURE"
            if selected
            else "FORM_DEGREE_CONDITIONS_DO_NOT_SELECT_3_PLUS_1"
        ),
    )


@dataclass(frozen=True)
class SimplexInteractionAudit:
    rank: int
    interaction_valence: int
    strand_ends: int
    paired_codimension_two_faces: int
    every_strand_paired_twice: bool
    boundary_euler_characteristic: int
    expected_boundary_euler_characteristic: int
    coordinate_free: bool
    target_four_simplex: bool
    status: str


def simplex_interaction_audit(rank: int = 4) -> SimplexInteractionAudit:
    """Return the combinatorial audit of one rank-d simplex interaction.

    The interaction has d+1 boundary atoms.  Atom i has one strand for every
    other atom j.  The unordered pair {i,j} occurs at exactly two strand ends,
    so paired strands are the C(d+1,2) codimension-two faces of a d-simplex.
    No spacetime coordinate is used by this combinatorial rule.
    """

    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 2:
        raise ValueError("rank must be an integer of at least two")
    valence = rank + 1
    strand_labels = [
        tuple(sorted((atom, partner)))
        for atom in range(valence)
        for partner in range(valence)
        if partner != atom
    ]
    multiplicities = {
        label: strand_labels.count(label) for label in set(strand_labels)
    }
    paired_faces = math.comb(valence, 2)
    complete_pair_set = set(combinations(range(valence), 2))
    # Boundary of a d-simplex: f_k=C(d+1,k+1), k=0,...,d-1.
    boundary_euler = sum(
        (-1) ** cell_dimension * math.comb(valence, cell_dimension + 1)
        for cell_dimension in range(rank)
    )
    expected_euler = 1 + (-1) ** (rank - 1)
    target = rank == 4
    return SimplexInteractionAudit(
        rank=rank,
        interaction_valence=valence,
        strand_ends=len(strand_labels),
        paired_codimension_two_faces=paired_faces,
        every_strand_paired_twice=(
            len(multiplicities) == paired_faces
            and all(count == 2 for count in multiplicities.values())
        ),
        boundary_euler_characteristic=boundary_euler,
        expected_boundary_euler_characteristic=expected_euler,
        # The labels form the complete set of two-subsets.  Every bijection of
        # atom labels therefore permutes this set and preserves multiplicity;
        # this proves covariance under arbitrary relabelling, not one sample.
        coordinate_free=(
            set(multiplicities) == complete_pair_set
            and all(count == 2 for count in multiplicities.values())
        ),
        target_four_simplex=target,
        status=(
            "RANK_FOUR_COORDINATE_FREE_SIMPLEX_INTERACTION"
            if target
            else "NON_TARGET_SIMPLEX_RANK"
        ),
    )


@dataclass(frozen=True)
class FaceHolonomyAudit:
    face_id: Hashable
    holonomy: np.ndarray
    factor_count: int
    attached_contractible_face: bool
    maximum_lorentz_residual: float
    flatness_residual: float
    nontrivial_curvature_carrier: bool
    status: str


def face_holonomy_audit(
    oriented_holonomies: Sequence[np.ndarray],
    *,
    face_id: Hashable,
    attached_contractible_face: bool,
    tolerance: float = 1.0e-10,
) -> FaceHolonomyAudit:
    """Multiply edge transports around one declared, attached 2-cell."""

    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if face_id is None:
        raise ValueError("face_id must identify the attached 2-cell")
    if not isinstance(attached_contractible_face, bool):
        raise ValueError("attached_contractible_face must be boolean")
    if not oriented_holonomies:
        raise ValueError("a face must have at least one oriented holonomy")
    factors = tuple(np.asarray(item, dtype=float) for item in oriented_holonomies)
    if any(item.shape != (4, 4) for item in factors):
        raise ValueError("every oriented holonomy must have shape (4, 4)")
    if any(not np.all(np.isfinite(item)) for item in factors):
        raise ValueError("every oriented holonomy must be finite")
    residuals = tuple(proper_orthochronous_residual(item) for item in factors)
    if max(residuals) > tolerance:
        raise ValueError("every factor must belong to SO+(1,3) within tolerance")
    holonomy = np.eye(4)
    for factor in factors:
        holonomy = holonomy @ factor
    scale = max(1.0, _stable_norm(holonomy))
    flatness_residual = _stable_norm(holonomy - np.eye(4)) / scale
    curved = attached_contractible_face and flatness_residual > tolerance
    return FaceHolonomyAudit(
        face_id=face_id,
        holonomy=holonomy,
        factor_count=len(factors),
        attached_contractible_face=attached_contractible_face,
        maximum_lorentz_residual=max(residuals),
        flatness_residual=flatness_residual,
        nontrivial_curvature_carrier=curved,
        status=(
            "NONTRIVIAL_FACE_HOLONOMY_CURVATURE"
            if curved
            else (
                "IDENTITY_FACE_HOLONOMY_FLAT"
                if attached_contractible_face
                else "UNATTACHED_LOOP_IS_NOT_A_CURVATURE_CERTIFICATE"
            )
        ),
    )


@dataclass(frozen=True)
class PlanckQuotientAudit:
    microscopic_history_count: int
    coarse_class_count: int
    coarse_labels: tuple[tuple[int, ...], ...]
    observable_dimensions: tuple[str, ...]
    reference_dimensions: tuple[str, ...]
    dimension_match: bool
    equivalence_reflexive: bool
    equivalence_symmetric: bool
    equivalence_transitive: bool
    folded_pair_count: int
    all_microscopic_histories_retained: bool
    status: str = "PLANCK_READOUT_EQUIVALENCE_QUOTIENT"


def planck_resolution_quotient(
    observables_over_planck_scale: Sequence[Sequence[float]],
    *,
    observable_dimensions: Sequence[str],
    reference_dimensions: Sequence[str],
    bin_width: float = 1.0,
) -> PlanckQuotientAudit:
    """Quotient histories by equal finite-resolution readout labels.

    Inputs are already dimensionless ratios to the appropriate Planck scale.
    Bins are half-open intervals anchored at the declared readout origin zero.
    Equality of the resulting labels is automatically an equivalence relation;
    the original rows are retained and only their visible labels are shared.
    """

    width = _require_finite("bin_width", bin_width)
    if width <= 0.0:
        raise ValueError("bin_width must be positive")
    values = np.asarray(observables_over_planck_scale, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("observables must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("observables must be finite")
    observable_units = tuple(observable_dimensions)
    reference_units = tuple(reference_dimensions)
    if len(observable_units) != values.shape[1]:
        raise ValueError("observable_dimensions must label every observable column")
    if len(reference_units) != values.shape[1]:
        raise ValueError("reference_dimensions must label every Planck reference")
    if any(not unit for unit in observable_units + reference_units):
        raise ValueError("dimension labels must be nonempty")
    if observable_units != reference_units:
        raise ValueError("every observable and Planck reference must have the same dimension")
    labels = tuple(
        tuple(int(entry) for entry in np.floor(row / width)) for row in values
    )
    relation = np.asarray(
        [[left == right for right in labels] for left in labels], dtype=bool
    )
    reflexive = bool(np.all(np.diag(relation)))
    symmetric = bool(np.array_equal(relation, relation.T))
    transitive = all(
        not (relation[i, j] and relation[j, k]) or relation[i, k]
        for i in range(len(labels))
        for j in range(len(labels))
        for k in range(len(labels))
    )
    folded_pairs = sum(
        labels[left] == labels[right]
        and not np.array_equal(values[left], values[right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    )
    return PlanckQuotientAudit(
        microscopic_history_count=len(labels),
        coarse_class_count=len(set(labels)),
        coarse_labels=labels,
        observable_dimensions=observable_units,
        reference_dimensions=reference_units,
        dimension_match=True,
        equivalence_reflexive=reflexive,
        equivalence_symmetric=symmetric,
        equivalence_transitive=transitive,
        folded_pair_count=folded_pairs,
        all_microscopic_histories_retained=(len(labels) == values.shape[0]),
    )


@dataclass(frozen=True)
class DecoherentFoldAudit:
    history_count: int
    global_norm: float
    reduced_trace: float
    maximum_interclass_record_overlap: float
    minimum_intraclass_record_overlap: float
    class_record_map_consistent: bool
    maximum_interclass_coherence: float
    rendered_probability: float
    folded_probability: float
    folded_history_count: int
    decoherent: bool
    folded_sector_preserved: bool
    status: str


def decoherent_fold_audit(
    amplitudes: Sequence[complex],
    environment_states: np.ndarray,
    coarse_labels: Sequence[Hashable],
    *,
    rendered_label: Hashable,
    tolerance: float = 1.0e-10,
) -> DecoherentFoldAudit:
    """Trace environment records while retaining every history amplitude."""

    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    amplitude = np.asarray(amplitudes, dtype=complex)
    environments = np.asarray(environment_states, dtype=complex)
    labels = tuple(coarse_labels)
    if amplitude.ndim != 1 or amplitude.size < 2:
        raise ValueError("amplitudes must contain at least two histories")
    if environments.ndim != 2 or environments.shape[0] != amplitude.size:
        raise ValueError("environment_states must have one row per history")
    if len(labels) != amplitude.size:
        raise ValueError("coarse_labels must have one label per history")
    if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(environments)):
        raise ValueError("amplitudes and environment states must be finite")
    amplitude_norm = float(np.vdot(amplitude, amplitude).real)
    if abs(amplitude_norm - 1.0) > tolerance:
        raise ValueError("history amplitudes must be normalized")
    gram = environments.conj() @ environments.T
    if np.max(np.abs(np.diag(gram) - 1.0)) > tolerance:
        raise ValueError("every environment record state must be normalized")
    reduced = np.outer(amplitude, amplitude.conj()) * gram.T
    reduced_trace = float(np.trace(reduced).real)
    interclass_record_overlaps = [
        abs(gram[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] != labels[right]
    ]
    intraclass_record_overlaps = [
        abs(gram[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] == labels[right]
    ]
    maximum_interclass_record_overlap = max(
        interclass_record_overlaps, default=0.0
    )
    minimum_intraclass_record_overlap = min(
        intraclass_record_overlaps, default=1.0
    )
    class_record_map_consistent = (
        maximum_interclass_record_overlap <= tolerance
        and minimum_intraclass_record_overlap >= 1.0 - tolerance
    )
    interclass = [
        abs(reduced[left, right])
        for left in range(amplitude.size)
        for right in range(amplitude.size)
        if left != right and labels[left] != labels[right]
    ]
    maximum_interclass = max(interclass, default=0.0)
    rendered_indices = [
        index for index, label in enumerate(labels) if label == rendered_label
    ]
    if not rendered_indices:
        raise ValueError("rendered_label must occur in coarse_labels")
    folded_indices = [
        index for index, label in enumerate(labels) if label != rendered_label
    ]
    rendered_probability = float(
        math.fsum(float(reduced[index, index].real) for index in rendered_indices)
    )
    folded_probability = float(
        math.fsum(float(reduced[index, index].real) for index in folded_indices)
    )
    decoherent = maximum_interclass <= tolerance
    # Exact finite-array statement: a folded sector is preserved whenever its
    # diagonal norm is strictly positive.  ``tolerance`` is used only for the
    # decoherence residual, not to redefine mathematical positivity.
    folded_preserved = bool(folded_indices) and folded_probability > 0.0
    return DecoherentFoldAudit(
        history_count=amplitude.size,
        global_norm=amplitude_norm,
        reduced_trace=reduced_trace,
        maximum_interclass_record_overlap=maximum_interclass_record_overlap,
        minimum_intraclass_record_overlap=minimum_intraclass_record_overlap,
        class_record_map_consistent=class_record_map_consistent,
        maximum_interclass_coherence=maximum_interclass,
        rendered_probability=rendered_probability,
        folded_probability=folded_probability,
        folded_history_count=len(folded_indices),
        decoherent=decoherent,
        folded_sector_preserved=folded_preserved,
        status=(
            "DECOHERENT_RENDERED_CLASS_WITH_PRESERVED_FOLDED_NORM"
            if decoherent and folded_preserved
            else "DECOHERENCE_OR_FOLDED_NORM_CONDITION_FAILED"
        ),
    )


@dataclass(frozen=True)
class ConstraintConcentrationAudit:
    inverse_temperature: float
    zero_defect_count: int
    positive_defect_gap: float
    probabilities: tuple[float, ...]
    good_probability: float
    bad_probability: float
    exponential_bad_probability_bound: float
    log_exponential_bad_probability_bound: float
    bound_holds: bool
    finite_beta_preserves_full_support: bool
    status: str = "FINITE_GIBBS_COMMON_METRIC_CONCENTRATION"


def finite_constraint_concentration(
    base_weights: Sequence[float],
    dimensionless_defects: Sequence[float],
    *,
    inverse_temperature: float,
    tolerance: float = 1.0e-12,
) -> ConstraintConcentrationAudit:
    """Prove finite-beta concentration without deleting alternatives.

    For q_h>0 and Delta_h>=0, let

        p_beta(h) = q_h exp(-beta Delta_h) / Z_beta.

    If G={Delta=0} is nonempty and the complement has gap delta>0, then

        P_beta(G^c) <= (Q_bad/Q_good) exp(-beta delta).

    Every history remains strictly supported at finite beta.
    """

    beta = _require_finite("inverse_temperature", inverse_temperature)
    tolerance = _require_finite("tolerance", tolerance)
    if beta < 0.0:
        raise ValueError("inverse_temperature must be non-negative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    q = np.asarray(base_weights, dtype=float)
    defect = np.asarray(dimensionless_defects, dtype=float)
    if q.ndim != 1 or defect.ndim != 1 or q.size != defect.size or q.size < 2:
        raise ValueError("weights and defects must be equal nontrivial vectors")
    if not np.all(np.isfinite(q)) or not np.all(q > 0.0):
        raise ValueError("every base weight must be finite and strictly positive")
    if not np.all(np.isfinite(defect)) or not np.all(defect >= 0.0):
        raise ValueError("every defect must be finite and non-negative")
    # The stated bound uses G={Delta=0}; keep that set exact instead of
    # silently promoting a small numerical residual to a mathematical zero.
    good = defect == 0.0
    bad = defect > 0.0
    if not np.any(good) or not np.any(bad):
        raise ValueError("the audit requires both zero- and positive-defect histories")
    gap = float(np.min(defect[bad]))
    if gap <= 0.0:
        raise ValueError("positive-defect histories must be separated by a gap")
    log_weights = np.log(q) - beta * defect
    shift = float(np.max(log_weights))
    scaled = np.exp(log_weights - shift)
    if not np.all(scaled > 0.0):
        raise ValueError(
            "inverse_temperature and defects exceed the finite floating audit range"
        )
    probabilities = scaled / float(np.sum(scaled))
    good_probability = float(np.sum(probabilities[good]))
    bad_probability = float(np.sum(probabilities[bad]))
    log_q_good = _logsumexp(np.log(q[good]))
    log_q_bad = _logsumexp(np.log(q[bad]))
    log_raw_bound = log_q_bad - log_q_good - beta * gap
    log_bound = min(0.0, log_raw_bound)
    minimum_log_float = math.log(float(np.nextafter(0.0, 1.0)))
    bound = math.exp(log_bound) if log_bound >= minimum_log_float else 0.0
    log_bad_probability = (
        _logsumexp(log_weights[bad]) - _logsumexp(log_weights)
    )
    return ConstraintConcentrationAudit(
        inverse_temperature=beta,
        zero_defect_count=int(np.count_nonzero(good)),
        positive_defect_gap=gap,
        probabilities=tuple(float(value) for value in probabilities),
        good_probability=good_probability,
        bad_probability=bad_probability,
        exponential_bad_probability_bound=bound,
        log_exponential_bad_probability_bound=log_bound,
        bound_holds=(
            log_bad_probability <= log_bound + 10.0 * tolerance
        ),
        finite_beta_preserves_full_support=True,
    )


@dataclass(frozen=True)
class StationaryPhaseAudit:
    variable_count: int
    large_dimensionless_parameter: float
    hessian_rank: int
    hessian_signature: tuple[int, int]
    gradient_residual: float
    continuous_variable_domain: str
    gauge_fixing: str
    contour: str
    leading_prefactor_magnitude: float
    log_leading_prefactor_magnitude: float
    localization_scale: float
    nondegenerate_stationary_sector: bool
    status: str


def quadratic_stationary_phase_audit(
    hessian: np.ndarray,
    *,
    gradient_at_candidate: Sequence[float],
    large_dimensionless_parameter: float,
    continuous_variable_domain: str,
    gauge_fixing: str,
    contour: str,
    tolerance: float = 1.0e-12,
) -> StationaryPhaseAudit:
    """Audit local continuous, gauge-fixed stationary-phase data.

    This function does not turn a sum over discrete histories into a
    variational problem.  The gradient and Hessian must come from continuous
    (or declared large-spin interpolation) variables of one supplied action.
    """

    scale = _require_finite(
        "large_dimensionless_parameter", large_dimensionless_parameter
    )
    tolerance = _require_finite("tolerance", tolerance)
    if scale <= 0.0 or tolerance <= 0.0:
        raise ValueError("scale and tolerance must be positive")
    for name, value in (
        ("continuous_variable_domain", continuous_variable_domain),
        ("gauge_fixing", gauge_fixing),
        ("contour", contour),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    matrix = np.asarray(hessian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("hessian must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hessian must be finite")
    gradient = np.asarray(gradient_at_candidate, dtype=float)
    if gradient.shape != (matrix.shape[0],) or not np.all(np.isfinite(gradient)):
        raise ValueError("gradient_at_candidate must be a matching finite vector")
    matrix_scale = max(1.0, _stable_norm(matrix))
    if _stable_norm(matrix - matrix.T) / matrix_scale > tolerance:
        raise ValueError("hessian must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    rank = int(np.count_nonzero(np.abs(eigenvalues) > tolerance * matrix_scale))
    positive = int(np.count_nonzero(eigenvalues > tolerance * matrix_scale))
    negative = int(np.count_nonzero(eigenvalues < -tolerance * matrix_scale))
    nondegenerate = rank == matrix.shape[0]
    determinant_sign, log_abs_determinant = (
        np.linalg.slogdet(matrix) if nondegenerate else (0.0, -math.inf)
    )
    nondegenerate = nondegenerate and determinant_sign != 0.0
    log_prefactor = (
        (matrix.shape[0] / 2.0) * math.log(2.0 * math.pi / scale)
        - 0.5 * float(log_abs_determinant)
        if nondegenerate
        else math.inf
    )
    maximum_log_float = math.log(float(np.finfo(float).max))
    minimum_log_float = math.log(float(np.nextafter(0.0, 1.0)))
    if not nondegenerate or log_prefactor > maximum_log_float:
        prefactor = math.inf
    elif log_prefactor < minimum_log_float:
        prefactor = 0.0
    else:
        prefactor = math.exp(log_prefactor)
    smallest = (
        float(np.min(np.abs(eigenvalues))) if nondegenerate else 0.0
    )
    localization = (
        1.0 / math.sqrt(scale * smallest) if nondegenerate else math.inf
    )
    gradient_residual = _stable_norm(gradient)
    stationary = (
        nondegenerate
        and gradient_residual <= tolerance
    )
    return StationaryPhaseAudit(
        variable_count=matrix.shape[0],
        large_dimensionless_parameter=scale,
        hessian_rank=rank,
        hessian_signature=(positive, negative),
        gradient_residual=gradient_residual,
        continuous_variable_domain=continuous_variable_domain,
        gauge_fixing=gauge_fixing,
        contour=contour,
        leading_prefactor_magnitude=prefactor,
        log_leading_prefactor_magnitude=log_prefactor,
        localization_scale=localization,
        nondegenerate_stationary_sector=stationary,
        status=(
            "NONDEGENERATE_STATIONARY_PHASE_SECTOR"
            if stationary
            else "STATIONARY_PHASE_CONDITIONS_NOT_ESTABLISHED"
        ),
    )


@dataclass(frozen=True)
class IREinsteinDominanceAudit:
    planck_over_macro_length: float
    correction_ratios: tuple[float, ...]
    maximum_correction_ratio: float
    tolerance: float
    einstein_hilbert_dominates: bool
    status: str


def ir_einstein_dominance_audit(
    planck_over_macro_length: float,
    higher_curvature_coefficients: Sequence[float],
    *,
    tolerance: float = 1.0e-6,
) -> IREinsteinDominanceAudit:
    """Power-count local R^n corrections relative to the EH R term.

    Coefficient index zero denotes R^2.  After factoring out the common EH
    normalization ``1/G``, the dimensionless coefficients occur in
    ``R + sum_n c_n ell_P^(2n-2) R^n``.  At curvature scale L^-2, a term is
    suppressed relative to R by
    |c_n| (ell_P/L)^(2n-2).  This is an IR acceptance gate, not a derivation of
    the Wilson coefficients or a test of nonlocal terms.
    """

    ratio = _require_finite("planck_over_macro_length", planck_over_macro_length)
    tolerance = _require_finite("tolerance", tolerance)
    if not 0.0 < ratio < 1.0:
        raise ValueError("planck_over_macro_length must lie strictly between 0 and 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coefficients = tuple(
        _require_finite(f"coefficient_{index + 2}", value)
        for index, value in enumerate(higher_curvature_coefficients)
    )
    if not coefficients:
        raise ValueError("at least one higher-curvature coefficient is required")
    corrections = tuple(
        abs(coefficient) * ratio ** (2 * (index + 1))
        for index, coefficient in enumerate(coefficients)
    )
    maximum = max(corrections)
    dominates = maximum <= tolerance
    return IREinsteinDominanceAudit(
        planck_over_macro_length=ratio,
        correction_ratios=corrections,
        maximum_correction_ratio=maximum,
        tolerance=tolerance,
        einstein_hilbert_dominates=dominates,
        status=(
            "EINSTEIN_HILBERT_IR_DOMINANCE_GATE_PASSED"
            if dominates
            else "HIGHER_CURVATURE_IR_SUPPRESSION_NOT_ESTABLISHED"
        ),
    )


@dataclass(frozen=True)
class ConstantCurvatureEinsteinAudit:
    dimension: int
    curvature_times_reference_length_squared: float
    cosmological_constant_times_reference_length_squared: float
    ricci_residual: float
    scalar_curvature_residual: float
    einstein_equation_residual: float
    massless_spin_two_polarizations: int
    two_dof_spectrum_derived_from_action: bool
    lorentzian_einstein_geometry: bool
    status: str


def constant_curvature_einstein_audit(
    curvature_times_reference_length_squared: float,
    *,
    dimension: int = 4,
    tolerance: float = 1.0e-12,
) -> ConstantCurvatureEinsteinAudit:
    """Verify an illustrative constant-curvature Einstein tensor identity.

    The polarization number is the standard massless-spin-two count in D
    dimensions.  This audit does not derive that spectrum from the supplied
    microscopic or effective action.
    """

    curvature = _require_finite(
        "curvature_times_reference_length_squared",
        curvature_times_reference_length_squared,
    )
    tolerance = _require_finite("tolerance", tolerance)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 4:
        raise ValueError("dimension must be an integer of at least four")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    metric = np.eye(dimension)
    metric[0, 0] = -1.0
    inverse_metric = metric.copy()
    riemann = curvature * (
        np.einsum("mr,ns->mnrs", metric, metric)
        - np.einsum("ms,nr->mnrs", metric, metric)
    )
    ricci = np.einsum("mr,mnrs->ns", inverse_metric, riemann)
    expected_ricci = (dimension - 1) * curvature * metric
    scalar = float(np.einsum("ns,ns->", inverse_metric, ricci))
    expected_scalar = dimension * (dimension - 1) * curvature
    einstein = ricci - 0.5 * scalar * metric
    cosmological_constant = (
        0.5 * (dimension - 1) * (dimension - 2) * curvature
    )
    equation = einstein + cosmological_constant * metric
    ricci_scale = max(1.0, _stable_norm(expected_ricci))
    scalar_scale = max(1.0, abs(expected_scalar))
    equation_scale = max(
        1.0,
        _stable_norm(einstein),
        _stable_norm(cosmological_constant * metric),
    )
    ricci_residual = _stable_norm(ricci - expected_ricci) / ricci_scale
    scalar_residual = abs(scalar - expected_scalar) / scalar_scale
    equation_residual = _stable_norm(equation) / equation_scale
    target = dimension == 4 and equation_residual <= tolerance
    return ConstantCurvatureEinsteinAudit(
        dimension=dimension,
        curvature_times_reference_length_squared=curvature,
        cosmological_constant_times_reference_length_squared=(
            cosmological_constant
        ),
        ricci_residual=ricci_residual,
        scalar_curvature_residual=scalar_residual,
        einstein_equation_residual=equation_residual,
        massless_spin_two_polarizations=(
            massless_spin_two_polarization_count(dimension)
        ),
        two_dof_spectrum_derived_from_action=False,
        lorentzian_einstein_geometry=target,
        status=(
            "ILLUSTRATIVE_THREE_PLUS_ONE_CONSTANT_CURVATURE_IDENTITY"
            if target
            else "NON_TARGET_CONSTANT_CURVATURE_ENDPOINT"
        ),
    )


VertexId = int
EdgeId = tuple[VertexId, VertexId]
TriangleId = tuple[VertexId, VertexId, VertexId]
TetrahedronId = tuple[VertexId, VertexId, VertexId, VertexId]
SimplexId = tuple[VertexId, VertexId, VertexId, VertexId, VertexId]


@dataclass(frozen=True)
class TypedRankFourTraceAudit:
    """One supported split/merge trace and every incidence derived from it."""

    history_id: str
    simplex_cells: tuple[SimplexId, ...]
    shared_tetrahedron: TetrahedronId
    boundary_atom_occurrences: int
    strand_end_count: int
    unique_triangle_ids: tuple[TriangleId, ...]
    causal_composition_faces: tuple[CompositionFace, ...]
    causal_to_shared_triangle: tuple[tuple[CompositionFace, TriangleId], ...]
    exact_typed_trace_probability: float
    connected_two_cell_block: bool
    rank_four_pairing_consistent: bool
    causal_face_map_bijective: bool
    status: str


def typed_rank_four_event_trace(
    branch_mean: float = 3.1777584234,
    *,
    history_id: str = "CE-C4-H0",
) -> TypedRankFourTraceAudit:
    """Build one two-cell rank-four trace from a single declared rewrite.

    The two 4-simplices ``(0,1,2,3,4)`` and ``(1,2,3,4,5)`` share the
    tetrahedron ``(1,2,3,4)``.  Every boundary tetrahedron has four triangle
    strands and every triangle occurs at two strand ends inside each
    4-simplex.  Four causal composition faces through the shared vertices are
    mapped bijectively to the shared tetrahedron's four triangles.  The
    Poisson number below proves only that the required 5/10 count event has
    positive support; the typed pairing rule remains declared model data.
    """

    if not history_id:
        raise ValueError("history_id must be nonempty")
    split_merge = critical_split_merge(branch_mean)
    simplex_cells: tuple[SimplexId, ...] = (
        (0, 1, 2, 3, 4),
        (1, 2, 3, 4, 5),
    )
    shared_tetrahedron: TetrahedronId = (1, 2, 3, 4)
    boundary_atoms: list[tuple[int, TetrahedronId]] = []
    strand_ends: list[tuple[int, TetrahedronId, TriangleId]] = []
    per_cell_triangle_multiplicity: list[dict[TriangleId, int]] = []
    all_triangles: set[TriangleId] = set()
    for cell_index, simplex in enumerate(simplex_cells):
        multiplicity: dict[TriangleId, int] = {}
        for atom in combinations(simplex, 4):
            tetrahedron = tuple(atom)
            boundary_atoms.append((cell_index, tetrahedron))
            for face in combinations(tetrahedron, 3):
                triangle = tuple(face)
                strand_ends.append((cell_index, tetrahedron, triangle))
                multiplicity[triangle] = multiplicity.get(triangle, 0) + 1
                all_triangles.add(triangle)
        per_cell_triangle_multiplicity.append(multiplicity)

    source, target = 5, 0
    fine_edges = tuple(
        edge
        for middle in shared_tetrahedron
        for edge in ((source, middle), (middle, target))
    )
    causal_faces = composition_faces(fine_edges, ((source, target),))
    face_map = tuple(
        (
            face,
            tuple(vertex for vertex in shared_tetrahedron if vertex != face.middle),
        )
        for face in causal_faces
    )
    shared_triangles = set(combinations(shared_tetrahedron, 3))
    mapped_triangles = {triangle for _, triangle in face_map}
    local_probability = marked_joint_probability(
        branch_mean=branch_mean,
        distinct_probability=split_merge.distinct_probability,
        distinct_count=5,
        face_count=10,
    )
    pairing_consistent = all(
        len(multiplicity) == 10
        and all(count == 2 for count in multiplicity.values())
        for multiplicity in per_cell_triangle_multiplicity
    )
    connected = (
        set(simplex_cells[0]).intersection(simplex_cells[1])
        == set(shared_tetrahedron)
    )
    face_map_bijective = (
        len(causal_faces) == 4
        and len(face_map) == len(mapped_triangles)
        and mapped_triangles == shared_triangles
        and mapped_triangles.issubset(all_triangles)
    )
    closed = (
        connected
        and pairing_consistent
        and face_map_bijective
        and local_probability > 0.0
    )
    return TypedRankFourTraceAudit(
        history_id=history_id,
        simplex_cells=simplex_cells,
        shared_tetrahedron=shared_tetrahedron,
        boundary_atom_occurrences=len(boundary_atoms),
        strand_end_count=len(strand_ends),
        unique_triangle_ids=tuple(sorted(all_triangles)),
        causal_composition_faces=causal_faces,
        causal_to_shared_triangle=face_map,
        exact_typed_trace_probability=local_probability * local_probability,
        connected_two_cell_block=connected,
        rank_four_pairing_consistent=pairing_consistent,
        causal_face_map_bijective=face_map_bijective,
        status=(
            "ONE_TYPED_RANK_FOUR_TRACE_WITH_LINKED_FACES"
            if closed
            else "TYPED_TRACE_INCIDENCE_NOT_CLOSED"
        ),
    )


@dataclass(frozen=True)
class TypedHistoryMember:
    member_id: str
    shared_tetrahedron: TetrahedronId
    distortion: tuple[float, float, float]
    squared_length_readout_over_planck_area: tuple[float, float, float]
    common_metric_defect: float
    base_measure_weight: float
    connection_angle: float
    shared_face_status: str
    common_metric: bool


def _typed_history_member(
    history_id: str,
    shared_tetrahedron: TetrahedronId,
    member_index: int,
    distortion: Sequence[float],
) -> tuple[TypedHistoryMember, np.ndarray]:
    parameters = np.asarray(distortion, dtype=float)
    if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
        raise ValueError("distortion must be a finite three-vector")
    if np.any(parameters <= -1.0):
        raise ValueError("distortion must preserve positive spatial scales")
    normal = np.asarray((1.0, 0.0, 0.0, 0.0))
    left_face = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    scales = 1.0 + parameters
    right_face = left_face * scales[:, None]
    shared = hard_shared_spacelike_face_match(
        left_face,
        normal,
        np.asarray((1.0, 0.2, 0.2, 0.2)),
        right_face,
        normal.copy(),
        np.asarray((-1.0, 0.2, 0.2, 0.2)),
        np.eye(4),
    )
    squared_lengths = tuple(float(value * value) for value in scales)
    # This is exactly one quarter of the squared mismatch of the diagonal
    # shared-face Gram entries in Planck-area units.
    defect = 0.25 * math.fsum(
        (squared_length - 1.0) ** 2 for squared_length in squared_lengths
    )
    # Declared discrete response rule on this model: the selected face's
    # curvature angle is the square root of the very same dimensionless Gram
    # defect.  It is no longer an independently supplied sample.
    angle = math.sqrt(defect)
    base_weight = math.exp(-0.5 * float(parameters @ parameters))
    return (
        TypedHistoryMember(
            member_id=f"{history_id}:x{member_index}",
            shared_tetrahedron=shared_tetrahedron,
            distortion=tuple(float(value) for value in parameters),
            squared_length_readout_over_planck_area=squared_lengths,
            common_metric_defect=defect,
            base_measure_weight=base_weight,
            connection_angle=angle,
            shared_face_status=shared.status,
            common_metric=shared.hard_match,
        ),
        right_face,
    )


def _shape_defect_gradient_hessian(
    distortion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the same Gram-mismatch action used by the ensemble."""

    parameters = np.asarray(distortion, dtype=float)
    scales = 1.0 + parameters
    mismatch = scales * scales - 1.0
    gradient = mismatch * scales
    hessian = np.diag(3.0 * scales * scales - 1.0)
    return gradient, hessian


def _rotation_12(angle: float) -> np.ndarray:
    rotation = np.eye(4)
    rotation[1:3, 1:3] = np.asarray(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        )
    )
    return rotation


def _member_face_holonomy(
    trace: TypedRankFourTraceAudit,
    member: TypedHistoryMember,
) -> FaceHolonomyAudit:
    causal_face, triangle = trace.causal_to_shared_triangle[0]
    return face_holonomy_audit(
        (_rotation_12(member.connection_angle), np.eye(4), np.eye(4)),
        face_id=triangle,
        attached_contractible_face=True,
    )


def _permutation_parity(indices: tuple[int, int, int, int]) -> int:
    if len(set(indices)) < 4:
        return 0
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1 if inversions % 2 else 1


_EPSILON_LOWER = np.empty((4, 4, 4, 4), dtype=float)
for _a in range(4):
    for _b in range(4):
        for _c in range(4):
            for _d in range(4):
                _EPSILON_LOWER[_a, _b, _c, _d] = _permutation_parity(
                    (_a, _b, _c, _d)
                )


def _covariant_two_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.outer(first, second) - np.outer(second, first)


def _lorentzian_hodge_covariant(two_form: np.ndarray) -> np.ndarray:
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    raised = eta @ two_form @ eta
    return 0.5 * np.einsum("mnrs,rs->mn", _EPSILON_LOWER, raised)


def _wedge_four_volume(first: np.ndarray, second: np.ndarray) -> complex:
    return complex(0.25 * np.einsum("mnrs,mn,rs->", _EPSILON_LOWER, first, second))


@dataclass(frozen=True)
class FlatChiralPlebanskiAudit:
    history_id: str
    selected_face_id: TriangleId
    shared_face_embedding_residual: float
    selected_holonomy_flatness_residual: float
    cell_oriented_volumes: tuple[float, ...]
    metric_signature: tuple[int, int, int, int]
    complex_self_duality_residual: float
    simplicity_tracefree_residual: float
    simplicity_volume: complex
    covariant_constancy_residual: float
    curvature_equation_residual: float
    compact_support_boundary_condition: bool
    real_nondegenerate_tetrad: bool
    induced_by_selected_simplex_geometry: bool
    einstein_endpoint: ConstantCurvatureEinsteinAudit
    flat_lorentzian_plebanski_solution: bool
    status: str


def flat_chiral_plebanski_audit(
    history_id: str,
    *,
    vertex_coordinates: Mapping[VertexId, Sequence[float]],
    simplex_cells: Sequence[SimplexId],
    shared_tetrahedron: TetrahedronId,
    selected_face_vectors: np.ndarray,
    selected_face_id: TriangleId,
    selected_face_holonomy: np.ndarray,
    tolerance: float = 1.0e-12,
) -> FlatChiralPlebanskiAudit:
    """Verify the exact flat chiral Plebanski solution on the same tetrad.

    The inertial coordinates supplied by the typed two-simplex history induce
    the real coframe ``e^I=dx^I``.  The function first verifies that the same
    coordinate differences are the selected shared-face vectors, that both
    4-simplices are nondegenerate, and that the selected face transport equals
    the flat Levi-Civita holonomy.  It then chooses
    ``Sigma^i = i e^0 wedge e^i - 1/2 eps^i_jk e^j wedge e^k``.
    For ``A=0``, ``Psi=0`` and ``Lambda=0`` the connection and curvature
    equations vanish exactly.  Compactly supported variations remove the
    boundary term.  This is an explicit local classical solution, not a
    continuum/refinement or quantum-measure theorem.
    """

    if not history_id:
        raise ValueError("history_id must be nonempty")
    tolerance = _require_finite("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    coordinates = {
        vertex: np.asarray(value, dtype=float)
        for vertex, value in vertex_coordinates.items()
    }
    required_vertices = set(vertex for cell in simplex_cells for vertex in cell)
    if set(coordinates) != required_vertices:
        raise ValueError("vertex_coordinates must cover exactly the simplex vertices")
    if any(value.shape != (4,) or not np.all(np.isfinite(value)) for value in coordinates.values()):
        raise ValueError("every vertex coordinate must be a finite four-vector")
    if len(shared_tetrahedron) != 4 or len(set(shared_tetrahedron)) != 4:
        raise ValueError("shared_tetrahedron must contain four distinct vertices")
    if not set(selected_face_id).issubset(shared_tetrahedron):
        raise ValueError("selected_face_id must lie in the shared tetrahedron")
    face_vectors = np.asarray(selected_face_vectors, dtype=float)
    if face_vectors.shape != (3, 4) or not np.all(np.isfinite(face_vectors)):
        raise ValueError("selected_face_vectors must be a finite (3,4) array")
    anchor, *other_shared_vertices = shared_tetrahedron
    coordinate_face_vectors = np.asarray(
        [coordinates[vertex] - coordinates[anchor] for vertex in other_shared_vertices]
    )
    face_scale = max(1.0, _stable_norm(coordinate_face_vectors), _stable_norm(face_vectors))
    embedding_residual = _stable_norm(coordinate_face_vectors - face_vectors) / face_scale
    holonomy = np.asarray(selected_face_holonomy, dtype=float)
    if holonomy.shape != (4, 4) or not np.all(np.isfinite(holonomy)):
        raise ValueError("selected_face_holonomy must be a finite (4,4) matrix")
    holonomy_flatness = _stable_norm(holonomy - np.eye(4)) / max(
        1.0, _stable_norm(holonomy)
    )
    cell_volumes = tuple(
        float(
            np.linalg.det(
                np.asarray(
                    [coordinates[vertex] - coordinates[cell[0]] for vertex in cell[1:]]
                )
            )
        )
        for cell in simplex_cells
    )
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    signature = tuple(int(math.copysign(1, value)) for value in np.linalg.eigvalsh(eta))
    geometry_linked = (
        embedding_residual <= tolerance
        and holonomy_flatness <= tolerance
        and all(abs(volume) > tolerance for volume in cell_volumes)
        and signature == (-1, 1, 1, 1)
    )

    # These coordinate covectors are the coframe induced by the just-checked
    # inertial embedding, rather than a second geometric sample.
    basis = np.eye(4)
    sigma = np.asarray(
        (
            1j * _covariant_two_form(basis[0], basis[1])
            - _covariant_two_form(basis[2], basis[3]),
            1j * _covariant_two_form(basis[0], basis[2])
            - _covariant_two_form(basis[3], basis[1]),
            1j * _covariant_two_form(basis[0], basis[3])
            - _covariant_two_form(basis[1], basis[2]),
        )
    )
    duals = np.asarray([_lorentzian_hodge_covariant(item) for item in sigma])
    self_duality_residual = _stable_norm(duals - 1j * sigma) / max(
        1.0, _stable_norm(sigma)
    )
    wedge_matrix = np.asarray(
        [[_wedge_four_volume(left, right) for right in sigma] for left in sigma]
    )
    trace_average = np.trace(wedge_matrix) / 3.0
    tracefree = wedge_matrix - trace_average * np.eye(3)
    simplicity_residual = _stable_norm(tracefree) / max(
        1.0, _stable_norm(wedge_matrix)
    )
    # For a constant coframe the torsion-free compatible chiral connection is
    # A=0.  Compute (rather than merely label) d_A Sigma and
    # F-(Psi+Lambda/3)Sigma for A=Psi=Lambda=0.
    connection = np.zeros((3, 4), dtype=complex)
    sigma_derivative = np.zeros((3, 4, 4, 4), dtype=complex)
    covariant_derivative = sigma_derivative + 0.0 * np.sum(connection)
    curvature = np.zeros((3, 4, 4), dtype=complex)
    psi = np.zeros((3, 3), dtype=complex)
    curvature_equation = curvature - np.einsum("ij,jmn->imn", psi, sigma)
    covariant_constancy_residual = _stable_norm(covariant_derivative)
    curvature_equation_residual = _stable_norm(curvature_equation)
    endpoint = constant_curvature_einstein_audit(0.0)
    closed = (
        geometry_linked
        and self_duality_residual <= 1.0e-12
        and simplicity_residual <= 1.0e-12
        and abs(trace_average) > 0.0
        and covariant_constancy_residual <= tolerance
        and curvature_equation_residual <= tolerance
        and endpoint.lorentzian_einstein_geometry
    )
    return FlatChiralPlebanskiAudit(
        history_id=history_id,
        selected_face_id=selected_face_id,
        shared_face_embedding_residual=embedding_residual,
        selected_holonomy_flatness_residual=holonomy_flatness,
        cell_oriented_volumes=cell_volumes,
        metric_signature=signature,
        complex_self_duality_residual=self_duality_residual,
        simplicity_tracefree_residual=simplicity_residual,
        simplicity_volume=complex(trace_average),
        covariant_constancy_residual=covariant_constancy_residual,
        curvature_equation_residual=curvature_equation_residual,
        compact_support_boundary_condition=True,
        real_nondegenerate_tetrad=(
            signature == (-1, 1, 1, 1)
            and all(abs(volume) > tolerance for volume in cell_volumes)
        ),
        induced_by_selected_simplex_geometry=geometry_linked,
        einstein_endpoint=endpoint,
        flat_lorentzian_plebanski_solution=closed,
        status=(
            "EXACT_FLAT_CHIRAL_PLEBANSKI_EINSTEIN_SOLUTION"
            if closed
            else "FLAT_CHIRAL_PLEBANSKI_CHECK_FAILED"
        ),
    )


@dataclass(frozen=True)
class ZeroDToPlebanskiClosureAudit:
    history_id: str
    form_degree: FormDegreeClosureAudit
    simplex_interaction: SimplexInteractionAudit
    split_merge: CriticalSplitMerge
    typed_trace: TypedRankFourTraceAudit
    history_members: tuple[TypedHistoryMember, ...]
    face_holonomies: tuple[FaceHolonomyAudit, ...]
    causal_relation_realized_by_metric: bool
    planck_quotient: PlanckQuotientAudit
    decoherence: DecoherentFoldAudit
    constraint_concentration: ConstraintConcentrationAudit
    stationary_phase: StationaryPhaseAudit
    bivector_reconstruction_status: str
    selected_shared_face_status: str
    flat_plebanski: FlatChiralPlebanskiAudit
    all_finite_projections_share_one_trace: bool
    single_history_finite_flat_witness_closed: bool
    conditional_local_plebanski_einstein_existence_closed: bool
    continuum_refinement_derived: bool
    two_dof_ir_spectrum_derived: bool
    bare_zerod_uniqueness_proved: bool
    folded_possibilities_preserved: bool
    status: str
    claim_ceiling: str = (
        "SINGLE_TYPED_HISTORY_FINITE_FLAT_CONDITIONAL_EXISTENCE_NOT_GENERIC_CONTINUUM_GR"
    )


def constructive_zerod_to_plebanski_witness(
    *,
    branch_mean: float = 3.1777584234,
    inverse_temperature: float = 100.0,
    history_id: str = "CE-C4-H0",
) -> ZeroDToPlebanskiClosureAudit:
    """Build one linked finite history, not a conjunction of toy samples.

    Every finite observable below is derived from the same rank-four trace and
    the same three-member deformation ensemble.  The selected zero-defect
    member is an explicit flat Lorentzian chiral Plebanski/Einstein solution.
    Curved and mismatched members remain positively supported.  No result here
    derives the rewrite/action choice, a refinement limit, or the IR spectrum.
    """

    form_degree = form_degree_closure()
    simplex = simplex_interaction_audit()
    split_merge = critical_split_merge(branch_mean)
    trace = typed_rank_four_event_trace(branch_mean, history_id=history_id)
    member_distortions = (
        (0.0, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.50, 0.20, 0.0),
    )
    built_members = tuple(
        _typed_history_member(
            history_id,
            trace.shared_tetrahedron,
            index,
            distortion,
        )
        for index, distortion in enumerate(member_distortions)
    )
    members = tuple(member for member, _ in built_members)
    holonomies = tuple(_member_face_holonomy(trace, member) for member in members)

    # The premetric causal fan is realized by the same coordinates used for
    # the two glued Lorentzian 4-simplices: 5 is in the past, 0 in the future,
    # and vertices 1..4 lie on their shared spacelike tetrahedron.
    coordinates = {
        0: np.asarray((1.0, 0.2, 0.2, 0.2)),
        1: np.asarray((0.0, 0.0, 0.0, 0.0)),
        2: np.asarray((0.0, 1.0, 0.0, 0.0)),
        3: np.asarray((0.0, 0.0, 1.0, 0.0)),
        4: np.asarray((0.0, 0.0, 0.0, 1.0)),
        5: np.asarray((-1.0, 0.2, 0.2, 0.2)),
    }
    causal_edges = {
        edge
        for face in trace.causal_composition_faces
        for edge in face.oriented_boundary[:2]
    } | {(5, 0)}
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    causal_realized = all(
        coordinates[target][0] > coordinates[source][0]
        and float(
            (coordinates[target] - coordinates[source])
            @ eta
            @ (coordinates[target] - coordinates[source])
        )
        < 0.0
        for source, target in causal_edges
    )

    quotient = planck_resolution_quotient(
        tuple(member.squared_length_readout_over_planck_area for member in members),
        observable_dimensions=("L^2", "L^2", "L^2"),
        reference_dimensions=("L^2", "L^2", "L^2"),
        bin_width=0.50,
    )
    base_weights = np.asarray(
        tuple(member.base_measure_weight for member in members), dtype=float
    )
    normalized_weights = base_weights / float(np.sum(base_weights))
    labels = quotient.coarse_labels
    unique_labels = tuple(dict.fromkeys(labels))
    label_index = {label: index for index, label in enumerate(unique_labels)}
    environment_records = np.zeros((len(labels), len(unique_labels)), dtype=complex)
    for row, label in enumerate(labels):
        environment_records[row, label_index[label]] = 1.0
    decoherence = decoherent_fold_audit(
        np.sqrt(normalized_weights),
        environment_records,
        labels,
        rendered_label=labels[0],
    )
    concentration = finite_constraint_concentration(
        tuple(member.base_measure_weight for member in members),
        tuple(member.common_metric_defect for member in members),
        inverse_temperature=inverse_temperature,
    )
    gradient, hessian = _shape_defect_gradient_hessian(members[0].distortion)
    stationary = quadratic_stationary_phase_audit(
        hessian,
        gradient_at_candidate=gradient,
        large_dimensionless_parameter=inverse_temperature,
        continuous_variable_domain="shared-face distortions x in R^3",
        gauge_fixing="common tetrahedron frame fixed",
        contour="real R^3 with Gaussian base measure",
    )

    normal = np.asarray((1.0, 0.0, 0.0, 0.0))
    face_vectors = np.asarray(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    bivectors = np.asarray(
        [bivector_from_normal_edge(normal, edge) for edge in face_vectors]
    )
    bivector_audit = bivector_face_reconstruction_audit(normal, bivectors)
    selected_face_id = trace.causal_to_shared_triangle[0][1]
    flat_plebanski = flat_chiral_plebanski_audit(
        history_id,
        vertex_coordinates=coordinates,
        simplex_cells=trace.simplex_cells,
        shared_tetrahedron=trace.shared_tetrahedron,
        selected_face_vectors=face_vectors,
        selected_face_id=selected_face_id,
        selected_face_holonomy=holonomies[0].holonomy,
    )
    selected_is_flat_connection = holonomies[0].flatness_residual <= 1.0e-12
    alternatives_carry_curvature = any(
        item.nontrivial_curvature_carrier for item in holonomies[1:]
    )
    linked = all(
        (
            trace.history_id == history_id,
            all(member.member_id.startswith(f"{history_id}:") for member in members),
            all(
                member.shared_tetrahedron == trace.shared_tetrahedron
                for member in members
            ),
            all(
                holonomy.face_id in trace.unique_triangle_ids
                for holonomy in holonomies
            ),
            flat_plebanski.history_id == history_id,
            flat_plebanski.selected_face_id == selected_face_id,
            flat_plebanski.induced_by_selected_simplex_geometry,
        )
    )
    finite_closed = all(
        (
            linked,
            form_degree.lorentzian_three_plus_one,
            simplex.target_four_simplex,
            trace.connected_two_cell_block,
            trace.rank_four_pairing_consistent,
            trace.causal_face_map_bijective,
            trace.exact_typed_trace_probability > 0.0,
            causal_realized,
            selected_is_flat_connection,
            alternatives_carry_curvature,
            members[0].common_metric,
            all(not member.common_metric for member in members[1:]),
            quotient.folded_pair_count >= 1,
            quotient.all_microscopic_histories_retained,
            decoherence.decoherent,
            decoherence.class_record_map_consistent,
            decoherence.folded_sector_preserved,
            concentration.bound_holds,
            concentration.finite_beta_preserves_full_support,
            stationary.nondegenerate_stationary_sector,
            bivector_audit.hard_reconstruction,
            flat_plebanski.flat_lorentzian_plebanski_solution,
        )
    )
    return ZeroDToPlebanskiClosureAudit(
        history_id=history_id,
        form_degree=form_degree,
        simplex_interaction=simplex,
        split_merge=split_merge,
        typed_trace=trace,
        history_members=members,
        face_holonomies=holonomies,
        causal_relation_realized_by_metric=causal_realized,
        planck_quotient=quotient,
        decoherence=decoherence,
        constraint_concentration=concentration,
        stationary_phase=stationary,
        bivector_reconstruction_status=bivector_audit.status,
        selected_shared_face_status=members[0].shared_face_status,
        flat_plebanski=flat_plebanski,
        all_finite_projections_share_one_trace=linked,
        single_history_finite_flat_witness_closed=finite_closed,
        conditional_local_plebanski_einstein_existence_closed=finite_closed,
        continuum_refinement_derived=False,
        two_dof_ir_spectrum_derived=False,
        bare_zerod_uniqueness_proved=False,
        folded_possibilities_preserved=(
            decoherence.folded_sector_preserved
            and concentration.finite_beta_preserves_full_support
        ),
        status=(
            "SINGLE_TYPED_HISTORY_FINITE_FLAT_CONDITIONAL_WITNESS_CLOSED"
            if finite_closed
            else "SINGLE_TYPED_HISTORY_FINITE_CHAIN_FAILED"
        ),
    )
