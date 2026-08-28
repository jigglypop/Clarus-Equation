"""Causal composition faces and a finite Plebanski-simplicity audit.

This module advances the Planck-rendering bridge in two deliberately separated
steps.

1. Face topology:
   A fine two-step causal path u->m->v and its one-step coarse image u->v
   determine a canonical composition triangle.  The face therefore witnesses
   the equivalence between a microscopic factorization and one coarse
   continuation; no arbitrary endpoint-pairing kernel is needed.

2. Metric renderability:
   A Euclideanized self-dual Plebanski triple B^i is simple when
       B^i wedge B^j proportional to delta^{ij}.
   Local simplicity of two cells does *not* imply simplicity of their block
   sum.  The missing condition is a cross-cell/shape-matching constraint.

3. Finite Lorentzian gluing:
   Two nondegenerate cell wedges can share one labelled spacelike 3-face only
   when its induced Gram matrix matches, the face normals are related by a
   proper orthochronous Lorentz transport, and the apices lie on opposite
   sides.  This is a local hard-matching lemma, not a Plebanski-sector test.

The finite audits are not a full Lorentzian spin-foam amplitude, a continuum
limit, or a derivation of general relativity.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable
import math

import numpy as np


Node = Hashable
Edge = tuple[Node, Node]


@dataclass(frozen=True, order=True)
class CompositionFace:
    """One triangular 2-cell witnessing fine composition into a coarse edge."""

    source: Node
    middle: Node
    target: Node

    @property
    def oriented_boundary(self) -> tuple[Edge, Edge, Edge]:
        return (
            (self.source, self.middle),
            (self.middle, self.target),
            (self.target, self.source),
        )


def _validate_acyclic(edges: set[Edge]) -> None:
    vertices: set[Node] = set()
    indegree: dict[Node, int] = defaultdict(int)
    outgoing: dict[Node, set[Node]] = defaultdict(set)
    for source, target in edges:
        if source == target:
            raise ValueError("causal edges cannot be self-loops")
        vertices.update((source, target))
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
        indegree.setdefault(source, indegree.get(source, 0))

    queue = deque(vertex for vertex in vertices if indegree[vertex] == 0)
    visited = 0
    while queue:
        vertex = queue.popleft()
        visited += 1
        for target in outgoing.get(vertex, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(vertices):
        raise ValueError("causal relation must be acyclic")


def composition_faces(
    fine_edges: Iterable[Edge],
    coarse_edges: Iterable[Edge],
) -> tuple[CompositionFace, ...]:
    """Return the 2-simplices of one causal block quotient.

    A face exists exactly when fine edges u->m and m->v factor a declared
    coarse continuation u->v.  This is the 2-skeleton-of-the-nerve attachment
    rule restricted to one block step.
    """

    fine = set(fine_edges)
    coarse = set(coarse_edges)
    _validate_acyclic(fine | coarse)

    incoming: dict[Node, set[Node]] = defaultdict(set)
    outgoing: dict[Node, set[Node]] = defaultdict(set)
    for source, target in fine:
        outgoing[source].add(target)
        incoming[target].add(source)

    result: list[CompositionFace] = []
    for source, target in coarse:
        middles = outgoing.get(source, set()) & incoming.get(target, set())
        for middle in middles:
            if middle not in (source, target):
                result.append(CompositionFace(source, middle, target))

    return tuple(
        sorted(
            result,
            key=lambda face: tuple(
                map(repr, (face.source, face.middle, face.target))
            ),
        )
    )


def fan_euler_characteristic(face_count: int) -> int:
    """Return chi for a fan of composition triangles sharing one coarse edge."""

    if isinstance(face_count, bool) or not isinstance(face_count, int) or face_count < 0:
        raise ValueError("face_count must be a non-negative integer")
    return (face_count + 2) - (2 * face_count + 1) + face_count


@dataclass(frozen=True)
class FaceIncidenceAudit:
    branch_mean: float
    block_depth: int
    face_intensity: float
    expected_faces: float
    probability_at_least_minimum: float
    exact_simplicial_probability: float
    minimum_faces: int


def _poisson_probability(rate: float, count: int) -> float:
    if rate == 0.0:
        return 1.0 if count == 0 else 0.0
    return math.exp(-rate + count * math.log(rate) - math.lgamma(count + 1))


def face_incidence_audit(
    branch_mean: float,
    block_depth: int,
    *,
    minimum_faces: int = 4,
) -> FaceIncidenceAudit:
    """Audit face incidence along a persistent Q-spine block.

    At the critical split/merge point the face-event intensity per spine epoch
    is mu=D-1.  Independent spine epochs therefore give a Poisson block count
    with mean b*(D-1).
    """

    if not math.isfinite(branch_mean) or branch_mean <= 1.0:
        raise ValueError("branch_mean must be finite and greater than one")
    if isinstance(block_depth, bool) or not isinstance(block_depth, int) or block_depth < 1:
        raise ValueError("block_depth must be a positive integer")
    if isinstance(minimum_faces, bool) or not isinstance(minimum_faces, int) or minimum_faces < 1:
        raise ValueError("minimum_faces must be a positive integer")

    face_intensity = branch_mean - 1.0
    rate = block_depth * face_intensity
    below = math.fsum(_poisson_probability(rate, count) for count in range(minimum_faces))
    return FaceIncidenceAudit(
        branch_mean=branch_mean,
        block_depth=block_depth,
        face_intensity=face_intensity,
        expected_faces=rate,
        probability_at_least_minimum=max(0.0, 1.0 - below),
        exact_simplicial_probability=_poisson_probability(rate, minimum_faces),
        minimum_faces=minimum_faces,
    )


def minimum_block_depth(
    branch_mean: float,
    *,
    confidence: float,
    minimum_faces: int = 4,
    maximum_depth: int = 1_000,
) -> int:
    """Return the first Q-spine block depth reaching a face-incidence confidence."""

    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    for depth in range(1, maximum_depth + 1):
        if (
            face_incidence_audit(
                branch_mean,
                depth,
                minimum_faces=minimum_faces,
            ).probability_at_least_minimum
            >= confidence
        ):
            return depth
    raise RuntimeError("requested confidence was not reached within maximum_depth")


def maximum_poisson_exact_valence_probability(valence: int) -> float:
    """Return sup_lambda P(Poisson(lambda)=valence), attained at lambda=valence."""

    if isinstance(valence, bool) or not isinstance(valence, int) or valence < 1:
        raise ValueError("valence must be a positive integer")
    return _poisson_probability(float(valence), valence)


_PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _stable_frobenius_norm(value: np.ndarray) -> float:
    """Return a Frobenius norm without avoidable scale under/overflow."""

    array = np.asarray(value, dtype=float)
    maximum = float(np.max(np.abs(array))) if array.size else 0.0
    if maximum == 0.0:
        return 0.0
    return maximum * float(np.linalg.norm(array / maximum))


def _scaled_determinant(value: np.ndarray) -> tuple[float, float, float]:
    """Return determinant, sign, and log(abs(det)) using row normalization."""

    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scaled determinant requires a square matrix")
    row_scales = np.max(np.abs(matrix), axis=1)
    if np.any(row_scales == 0.0):
        return 0.0, 0.0, -math.inf
    sign, normalized_log_abs = np.linalg.slogdet(matrix / row_scales[:, None])
    if sign == 0.0:
        return 0.0, 0.0, -math.inf
    log_abs = float(normalized_log_abs + math.fsum(math.log(x) for x in row_scales))
    maximum_log = math.log(np.finfo(float).max)
    minimum_log = math.log(np.nextafter(0.0, 1.0))
    if log_abs > maximum_log:
        determinant = math.copysign(math.inf, float(sign))
    elif log_abs < minimum_log:
        determinant = math.copysign(0.0, float(sign))
    else:
        determinant = math.copysign(math.exp(log_abs), float(sign))
    return determinant, float(sign), log_abs


def two_form_from_vectors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the six independent components of first wedge second."""

    first = _require_shape("first", first, (4,))
    second = _require_shape("second", second, (4,))
    matrix = np.outer(first, second) - np.outer(second, first)
    return np.array([matrix[i, j] for i, j in _PAIR_INDEX], dtype=float)


def wedge_scalar(first: np.ndarray, second: np.ndarray) -> float:
    """Return the oriented coefficient of first wedge second in four dimensions."""

    first = _require_shape("first", first, (6,))
    second = _require_shape("second", second, (6,))
    return float(
        first[0] * second[3]
        + first[3] * second[0]
        + first[1] * second[4]
        + first[4] * second[1]
        + first[2] * second[5]
        + first[5] * second[2]
    )


def geometric_self_dual_triple(tetrad: np.ndarray) -> np.ndarray:
    """Return the Euclidean self-dual Sigma^i(e) triple for a nondegenerate tetrad."""

    tetrad = _require_shape("tetrad", tetrad, (4, 4))
    if abs(float(np.linalg.det(tetrad))) <= 1.0e-12:
        raise ValueError("tetrad must be nondegenerate")

    epsilon = np.zeros((3, 3, 3), dtype=int)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

    result: list[np.ndarray] = []
    for i in range(3):
        form = two_form_from_vectors(tetrad[0], tetrad[i + 1])
        for j in range(3):
            for k in range(3):
                if epsilon[i, j, k]:
                    form = form + (
                        0.5
                        * epsilon[i, j, k]
                        * two_form_from_vectors(tetrad[j + 1], tetrad[k + 1])
                    )
        result.append(form)
    return np.asarray(result)


def plebanski_gram(triple: np.ndarray) -> np.ndarray:
    """Return X_ij=B^i wedge B^j for one self-dual 2-form triple."""

    triple = _require_shape("triple", triple, (3, 6))
    return np.array(
        [
            [wedge_scalar(triple[i], triple[j]) for j in range(3)]
            for i in range(3)
        ],
        dtype=float,
    )


def simplicity_residual(triple: np.ndarray) -> float:
    """Return the normalized traceless Plebanski simplicity residual."""

    gram = plebanski_gram(triple)
    traceless = gram - np.trace(gram) / 3.0 * np.eye(3)
    denominator = float(np.linalg.norm(gram))
    if denominator <= 1.0e-15:
        return math.inf
    return float(np.linalg.norm(traceless) / denominator)


def cross_simplicity_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the cross term controlling simplicity of a two-cell block sum."""

    first = _require_shape("first", first, (3, 6))
    second = _require_shape("second", second, (3, 6))
    return np.array(
        [
            [
                wedge_scalar(first[i], second[j])
                + wedge_scalar(second[i], first[j])
                for j in range(3)
            ]
            for i in range(3)
        ],
        dtype=float,
    )


def cross_simplicity_residual(first: np.ndarray, second: np.ndarray) -> float:
    """Return the normalized traceless cross-cell simplicity obstruction."""

    cross = cross_simplicity_matrix(first, second)
    traceless = cross - np.trace(cross) / 3.0 * np.eye(3)
    denominator = (
        float(np.linalg.norm(plebanski_gram(first)))
        + float(np.linalg.norm(plebanski_gram(second)))
        + float(np.linalg.norm(cross))
    )
    if denominator <= 1.0e-15:
        return math.inf
    return float(np.linalg.norm(traceless) / denominator)


@dataclass(frozen=True)
class SimplicityBlockAudit:
    first_local_residual: float
    second_local_residual: float
    cross_residual: float
    blocked_residual: float
    local_simplicity_sufficient: bool
    status: str = "LOCAL_SIMPLICITY_NOT_CLOSED_UNDER_BLOCKING"


def simplicity_block_audit(first: np.ndarray, second: np.ndarray) -> SimplicityBlockAudit:
    """Audit whether two locally simple cells remain simple after coarse addition."""

    first_local = simplicity_residual(first)
    second_local = simplicity_residual(second)
    cross = cross_simplicity_residual(first, second)
    blocked = simplicity_residual(np.asarray(first, dtype=float) + np.asarray(second, dtype=float))
    local_good = first_local < 1.0e-10 and second_local < 1.0e-10
    return SimplicityBlockAudit(
        first_local_residual=first_local,
        second_local_residual=second_local,
        cross_residual=cross,
        blocked_residual=blocked,
        local_simplicity_sufficient=(local_good and blocked < 1.0e-10),
    )


def soft_block_simplicity_weight(
    first: np.ndarray,
    second: np.ndarray,
    *,
    width: float,
) -> float:
    """Return a soft projector penalizing block and cross simplicity failure."""

    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    audit = simplicity_block_audit(first, second)
    residual_squared = (
        audit.first_local_residual**2
        + audit.second_local_residual**2
        + audit.cross_residual**2
        + audit.blocked_residual**2
    )
    return math.exp(-0.5 * residual_squared / (width * width))


@dataclass(frozen=True)
class RandomBlockSimplicityAudit:
    sample_count: int
    perturbation: float
    seed: int
    minimum_residual: float
    median_residual: float
    ninety_percent_residual: float
    maximum_residual: float
    fraction_above_tolerance: float


def random_tetrad_block_audit(
    *,
    sample_count: int = 1_000,
    perturbation: float = 0.35,
    seed: int = 20_260_828,
    minimum_determinant: float = 0.2,
    tolerance: float = 1.0e-6,
) -> RandomBlockSimplicityAudit:
    """Return a deterministic diagnostic for generic block-simplicity failure."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    for name, value in (
        ("perturbation", perturbation),
        ("minimum_determinant", minimum_determinant),
        ("tolerance", tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    generator = np.random.default_rng(seed)
    reference = geometric_self_dual_triple(np.eye(4))
    residuals: list[float] = []
    attempts = 0
    maximum_attempts = sample_count * 1_000
    while len(residuals) < sample_count and attempts < maximum_attempts:
        attempts += 1
        tetrad = np.eye(4) + perturbation * generator.normal(size=(4, 4))
        if float(np.linalg.det(tetrad)) <= minimum_determinant:
            continue
        candidate = geometric_self_dual_triple(tetrad)
        residuals.append(simplicity_residual(reference + candidate))

    if len(residuals) != sample_count:
        raise RuntimeError("could not sample enough positive-orientation tetrads")

    values = np.asarray(residuals)
    return RandomBlockSimplicityAudit(
        sample_count=sample_count,
        perturbation=perturbation,
        seed=seed,
        minimum_residual=float(np.min(values)),
        median_residual=float(np.median(values)),
        ninety_percent_residual=float(np.quantile(values, 0.9)),
        maximum_residual=float(np.max(values)),
        fraction_above_tolerance=float(np.mean(values > tolerance)),
    )


@dataclass(frozen=True)
class FaceSimplicityVerdict:
    canonical_face_attachment: bool
    one_epoch_nondegenerate_probability: float
    block_depth_95_percent: int
    block_depth_99_percent: int
    raw_poisson_simplicial_concentration_possible: bool
    local_simplicity_closed_under_blocking: bool
    remaining_obligation: str


def face_simplicity_verdict(branch_mean: float) -> FaceSimplicityVerdict:
    """Summarize the finite topology and simplicity audit."""

    one = face_incidence_audit(branch_mean, 1)
    return FaceSimplicityVerdict(
        canonical_face_attachment=True,
        one_epoch_nondegenerate_probability=one.probability_at_least_minimum,
        block_depth_95_percent=minimum_block_depth(branch_mean, confidence=0.95),
        block_depth_99_percent=minimum_block_depth(branch_mean, confidence=0.99),
        raw_poisson_simplicial_concentration_possible=(
            maximum_poisson_exact_valence_probability(4) > 0.95
        ),
        local_simplicity_closed_under_blocking=False,
        remaining_obligation=(
            "derive a Lorentzian closure/simplicity/shape-matching amplitude "
            "that is stable under coarse graining; count and local simplicity "
            "projectors alone do not yield a Plebanski continuum"
        ),
    )


_MINKOWSKI_METRIC = np.diag((-1.0, 1.0, 1.0, 1.0))


def minkowski_inner(first: np.ndarray, second: np.ndarray) -> float:
    """Return the (-,+,+,+) inner product of two contravariant vectors."""

    first = _require_shape("first", first, (4,))
    second = _require_shape("second", second, (4,))
    return float(first @ _MINKOWSKI_METRIC @ second)


def induced_spatial_gram(face_vectors: np.ndarray) -> np.ndarray:
    """Return the labelled intrinsic Gram matrix of three face tangents."""

    face_vectors = _require_shape("face_vectors", face_vectors, (3, 4))
    return face_vectors @ _MINKOWSKI_METRIC @ face_vectors.T


def proper_orthochronous_residual(transport: np.ndarray) -> float:
    """Return a zero-only residual for membership in SO^+(1,3)."""

    transport = _require_shape("transport", transport, (4, 4))
    metric_residual = float(
        np.linalg.norm(
            transport.T @ _MINKOWSKI_METRIC @ transport - _MINKOWSKI_METRIC
        )
        / np.linalg.norm(_MINKOWSKI_METRIC)
    )
    determinant_residual = abs(float(np.linalg.det(transport)) - 1.0)
    future_cone_residual = max(0.0, 1.0 - float(transport[0, 0]))
    return max(metric_residual, determinant_residual, future_cone_residual)


@dataclass(frozen=True)
class LorentzianSharedFaceAudit:
    """Finite conditional audit of one shared spacelike face."""

    left_gram: np.ndarray
    right_gram: np.ndarray
    left_wedge_determinant: float
    right_wedge_determinant: float
    left_wedge_log_abs_determinant: float
    right_wedge_log_abs_determinant: float
    lorentz_residual: float
    normal_transport_residual: float
    gram_residual: float
    tangent_transport_residual: float
    left_oriented_face_volume: float
    right_oriented_face_volume: float
    left_oriented_face_log_abs_volume: float
    right_oriented_face_log_abs_volume: float
    left_lapse: float
    right_lapse: float
    hard_match: bool
    status: str
    plebanski_branch: str = "NOT_TESTED_BY_FACE_GRAM"
    claim_ceiling: str = "FINITE_CONDITIONAL_SHARED_SPACELIKE_FACE_ONLY"


def _normal_residual(
    face_vectors: np.ndarray,
    normal: np.ndarray,
    face_scale: float,
) -> float:
    if face_scale <= 0.0:
        return math.inf
    norm_residual = abs(minkowski_inner(normal, normal) + 1.0)
    orthogonality = face_vectors @ _MINKOWSKI_METRIC @ normal
    return max(
        norm_residual,
        _stable_frobenius_norm(orthogonality) / face_scale,
    )


def hard_shared_spacelike_face_match(
    left_face: np.ndarray,
    left_normal: np.ndarray,
    left_apex: np.ndarray,
    right_face: np.ndarray,
    right_normal: np.ndarray,
    right_apex: np.ndarray,
    right_to_left: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> LorentzianSharedFaceAudit:
    """Audit a finite Lorentzian two-wedge gluing across one shared 3-face.

    The three tangent vectors on each side have the same labels.  The hard
    match requires the declared proper orthochronous transport to map every
    labelled tangent and the future normal.  Equal positive Gram matrices then
    certify the induced intrinsic isometry; oriented faces, nondegenerate cell
    wedges, and opposite-side apices are required separately.

    This datum cannot select a Lorentzian Plebanski branch and says nothing
    about a spin-foam measure, Regge equations, coarse-graining stability,
    a continuum limit, or propagating graviton degrees of freedom.
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    left_face = _require_shape("left_face", left_face, (3, 4))
    left_normal = _require_shape("left_normal", left_normal, (4,))
    left_apex = _require_shape("left_apex", left_apex, (4,))
    right_face = _require_shape("right_face", right_face, (3, 4))
    right_normal = _require_shape("right_normal", right_normal, (4,))
    right_apex = _require_shape("right_apex", right_apex, (4,))
    right_to_left = _require_shape("right_to_left", right_to_left, (4, 4))

    left_face_scale = _stable_frobenius_norm(left_face)
    right_face_scale = _stable_frobenius_norm(right_face)
    common_face_scale = max(left_face_scale, right_face_scale)
    if common_face_scale > 0.0:
        normalized_left_face = left_face / common_face_scale
        normalized_right_face = right_face / common_face_scale
    else:
        normalized_left_face = left_face
        normalized_right_face = right_face
    normalized_left_gram = induced_spatial_gram(normalized_left_face)
    normalized_right_gram = induced_spatial_gram(normalized_right_face)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        common_face_scale_squared = (
            np.float64(common_face_scale) * np.float64(common_face_scale)
        )
        left_gram = common_face_scale_squared * normalized_left_gram
        right_gram = common_face_scale_squared * normalized_right_gram
    (
        left_wedge_determinant,
        _,
        left_wedge_log_abs_determinant,
    ) = _scaled_determinant(np.vstack((left_face, left_apex)))
    (
        right_wedge_determinant,
        _,
        right_wedge_log_abs_determinant,
    ) = _scaled_determinant(
        np.vstack((right_face, right_apex))
    )
    lorentz_residual = proper_orthochronous_residual(right_to_left)
    transported_normal = right_to_left @ right_normal
    normal_transport_residual = float(
        np.linalg.norm(left_normal - transported_normal)
        / max(
            1.0,
            float(np.linalg.norm(left_normal)),
            float(np.linalg.norm(transported_normal)),
        )
    )
    gram_scale = max(
        _stable_frobenius_norm(normalized_left_gram),
        _stable_frobenius_norm(normalized_right_gram),
    )
    gram_residual = (
        _stable_frobenius_norm(normalized_left_gram - normalized_right_gram)
        / gram_scale
        if gram_scale > 0.0
        else math.inf
    )
    transported_right_face = right_face @ right_to_left.T
    tangent_scale = max(
        _stable_frobenius_norm(left_face),
        _stable_frobenius_norm(transported_right_face),
    )
    tangent_transport_residual = (
        _stable_frobenius_norm(left_face - transported_right_face) / tangent_scale
        if tangent_scale > 0.0
        else math.inf
    )
    (
        left_oriented_face_volume,
        left_orientation_sign,
        left_oriented_face_log_abs_volume,
    ) = _scaled_determinant(
        np.vstack((left_normal, left_face))
    )
    (
        right_oriented_face_volume,
        right_orientation_sign,
        right_oriented_face_log_abs_volume,
    ) = _scaled_determinant(
        np.vstack((right_normal, right_face))
    )
    left_lapse = minkowski_inner(left_normal, left_apex)
    right_lapse = minkowski_inner(left_normal, right_to_left @ right_apex)

    left_unit_face = left_face / left_face_scale if left_face_scale > 0.0 else left_face
    right_unit_face = (
        right_face / right_face_scale if right_face_scale > 0.0 else right_face
    )
    left_eigenvalues = np.linalg.eigvalsh(induced_spatial_gram(left_unit_face))
    right_eigenvalues = np.linalg.eigvalsh(induced_spatial_gram(right_unit_face))
    left_maximum = float(np.max(left_eigenvalues))
    right_maximum = float(np.max(right_eigenvalues))
    left_spacelike = (
        left_maximum > 0.0
        and float(np.min(left_eigenvalues)) / left_maximum > tolerance
    )
    right_spacelike = (
        right_maximum > 0.0
        and float(np.min(right_eigenvalues)) / right_maximum > tolerance
    )
    normals_valid = (
        left_normal[0] > tolerance
        and right_normal[0] > tolerance
        and _normal_residual(left_face, left_normal, left_face_scale) <= tolerance
        and _normal_residual(right_face, right_normal, right_face_scale) <= tolerance
    )
    transport_valid = lorentz_residual <= tolerance
    normals_compatible = normal_transport_residual <= tolerance
    shape_matches = gram_residual <= tolerance
    tangents_compatible = tangent_transport_residual <= tolerance
    orientation_matches = (
        left_orientation_sign != 0.0
        and right_orientation_sign != 0.0
        and left_orientation_sign == right_orientation_sign
    )
    wedge_nondegenerate = (
        left_face_scale > 0.0
        and right_face_scale > 0.0
        and abs(left_lapse) / left_face_scale > tolerance
        and abs(right_lapse) / right_face_scale > tolerance
    )
    opposite_sides = (
        wedge_nondegenerate
        and (
            (left_lapse < 0.0 < right_lapse)
            or (right_lapse < 0.0 < left_lapse)
        )
    )

    if not (left_spacelike and right_spacelike):
        status = "NONSPACELIKE_OR_DEGENERATE_FACE"
    elif not normals_valid:
        status = "INVALID_FUTURE_FACE_NORMAL"
    elif not transport_valid:
        status = "NON_PROPER_OR_NON_ORTHOCHRONOUS_TRANSPORT"
    elif not normals_compatible:
        status = "INCOMPATIBLE_FACE_NORMALS"
    elif not shape_matches:
        status = "SHAPE_MISMATCH"
    elif not orientation_matches:
        status = "ORIENTATION_REVERSING_FACE_MAP"
    elif not tangents_compatible:
        status = "INCOMPATIBLE_FACE_TANGENT_TRANSPORT"
    elif not wedge_nondegenerate:
        status = "DEGENERATE_CELL_WEDGE"
    elif not opposite_sides:
        status = "SAME_SIDE_APEX_CONFIGURATION"
    else:
        status = "FINITE_SHARED_SPACELIKE_FACE_MATCH"

    hard_match = status == "FINITE_SHARED_SPACELIKE_FACE_MATCH"
    return LorentzianSharedFaceAudit(
        left_gram=left_gram,
        right_gram=right_gram,
        left_wedge_determinant=left_wedge_determinant,
        right_wedge_determinant=right_wedge_determinant,
        left_wedge_log_abs_determinant=left_wedge_log_abs_determinant,
        right_wedge_log_abs_determinant=right_wedge_log_abs_determinant,
        lorentz_residual=lorentz_residual,
        normal_transport_residual=normal_transport_residual,
        gram_residual=gram_residual,
        tangent_transport_residual=tangent_transport_residual,
        left_oriented_face_volume=left_oriented_face_volume,
        right_oriented_face_volume=right_oriented_face_volume,
        left_oriented_face_log_abs_volume=left_oriented_face_log_abs_volume,
        right_oriented_face_log_abs_volume=right_oriented_face_log_abs_volume,
        left_lapse=left_lapse,
        right_lapse=right_lapse,
        hard_match=hard_match,
        status=status,
    )
