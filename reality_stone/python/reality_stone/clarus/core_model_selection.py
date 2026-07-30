"""Generalized model-selection audit for the scalar CE recursion sector.

The comparison implemented here is deliberately narrower than the full vector
or matrix Clarus dynamics.  Every model carries the explicit recursion scope
``scalar_equal_row_sum`` and is evaluated through

    x = S(D_eff * K(x)).

Two kinds of evidence are kept separate:

* ``algebraic_status`` checks normalization, monotonicity, root completeness,
  residuals, and local fixed-point stability;
* ``selection_status`` scores only observations marked for model selection.

An algebraic pass is therefore never promoted to an empirical model-selection
claim.  Likewise, a single independent selection observable is reported as
``UNDERIDENTIFIED`` rather than being used to choose among model families.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RECURSION_SCOPE = "scalar_equal_row_sum"
ALGEBRAIC_PASS = "PASS"
ALGEBRAIC_FAIL = "FAIL"
NOT_TESTABLE = "NOT_TESTABLE"
UNDERIDENTIFIED = "UNDERIDENTIFIED"
COMPETITIVE = "COMPETITIVE"
DISFAVORED = "DISFAVORED"

_OBSERVATION_ROLES = {
    "input",
    "calibration",
    "selection",
    "confirmation",
    "prospective",
    "reference",
}


@dataclass(frozen=True)
class ModelContext:
    """Inputs shared by every candidate in one manifest."""

    spatial_dimension: float
    delta: float


@dataclass(frozen=True)
class CandidateSpec:
    """One preregistered structural model."""

    candidate_id: str
    survival_id: str
    feedback_id: str
    d_eff_id: str
    parameter_count: int
    recursion_scope: str = RECURSION_SCOPE


@dataclass(frozen=True)
class Observation:
    """One manifest observation with an explicit evidence role."""

    key: str
    adapter: str
    value: float
    sigma: float
    role: str
    correlation_group: str
    source: str


@dataclass(frozen=True)
class RootAudit:
    """One root and its local fixed-point properties."""

    value: float
    residual: float
    fixed_map_derivative: float
    stability_radius: float
    stable: bool
    branch_label: str
    eligible_for_selection: bool


@dataclass(frozen=True)
class AlgebraicAudit:
    """Implementation and mathematical-property checks for one candidate."""

    status: str
    normalization_error: float
    initial_slope_error: float
    survival_positive_and_bounded: bool
    survival_monotone: bool
    factorization_defect_max: float
    factorization_defect_rms: float
    factorization_compatible: bool
    feedback_at_zero_error: float
    feedback_at_one_error: float
    feedback_bounded: bool
    feedback_monotone: bool
    d_eff: float
    d_eff_valid: bool
    root_count: int
    max_root_residual: float
    roots: tuple[RootAudit, ...]


@dataclass(frozen=True)
class SelectionResidual:
    """Prediction residual for one selection observation."""

    key: str
    prediction: float
    observed: float
    sigma: float
    z_score: float
    correlation_group: str


@dataclass(frozen=True)
class SelectionScore:
    """Empirical score for one branch."""

    status: str
    n_observations: int
    n_independent_observations: int
    chi2: float | None
    max_abs_z: float | None
    rms_z: float | None
    residuals: tuple[SelectionResidual, ...]


@dataclass(frozen=True)
class BranchEvaluation:
    """A root together with its empirical diagnostic score."""

    root: RootAudit
    selection: SelectionScore


@dataclass(frozen=True)
class CandidateEvaluation:
    """Complete result for a preregistered model candidate."""

    spec: CandidateSpec
    algebraic_status: str
    selection_status: str
    algebraic: AlgebraicAudit
    branches: tuple[BranchEvaluation, ...]


@dataclass(frozen=True)
class HarnessReport:
    """Serializable result of evaluating the complete candidate universe."""

    schema_version: str
    recursion_scope: str
    manifest_hash: str
    candidate_count: int
    algebraic_status: str
    selection_status: str
    n_selection_observations: int
    n_independent_selection_observations: int
    candidates: tuple[CandidateEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SurvivalLaw:
    value: Callable[[float], float]
    derivative: Callable[[float], float]


@dataclass(frozen=True)
class _FeedbackLaw:
    value: Callable[[float], float]
    derivative: Callable[[float], float]


def _exp_value(z: float) -> float:
    return math.exp(-z)


def _exp_derivative(z: float) -> float:
    return -math.exp(-z)


def _rational_q1_value(z: float) -> float:
    return 1.0 / (1.0 + z)


def _rational_q1_derivative(z: float) -> float:
    return -1.0 / ((1.0 + z) ** 2)


def _rational_q2_value(z: float) -> float:
    return (1.0 + z / 2.0) ** -2.0


def _rational_q2_derivative(z: float) -> float:
    return -(1.0 + z / 2.0) ** -3.0


SURVIVAL_LAWS: Mapping[str, _SurvivalLaw] = {
    "exponential": _SurvivalLaw(_exp_value, _exp_derivative),
    "rational_q1": _SurvivalLaw(_rational_q1_value, _rational_q1_derivative),
    "rational_q2": _SurvivalLaw(_rational_q2_value, _rational_q2_derivative),
}


def _linear_feedback(x: float) -> float:
    return 1.0 - x


def _linear_feedback_derivative(_x: float) -> float:
    return -1.0


def _residual_square_feedback(x: float) -> float:
    return (1.0 - x) ** 2


def _residual_square_feedback_derivative(x: float) -> float:
    return -2.0 * (1.0 - x)


def _state_square_feedback(x: float) -> float:
    return 1.0 - x * x


def _state_square_feedback_derivative(x: float) -> float:
    return -2.0 * x


FEEDBACK_LAWS: Mapping[str, _FeedbackLaw] = {
    "linear": _FeedbackLaw(_linear_feedback, _linear_feedback_derivative),
    "residual_square": _FeedbackLaw(
        _residual_square_feedback,
        _residual_square_feedback_derivative,
    ),
    "state_square": _FeedbackLaw(
        _state_square_feedback,
        _state_square_feedback_derivative,
    ),
}


def _bare_d(context: ModelContext) -> float:
    return context.spatial_dimension


def _ce_delta(context: ModelContext) -> float:
    return context.spatial_dimension + context.delta


def _double_delta(context: ModelContext) -> float:
    return context.spatial_dimension + 2.0 * context.delta


D_EFF_MAPS: Mapping[str, Callable[[ModelContext], float]] = {
    "bare_d": _bare_d,
    "ce_delta": _ce_delta,
    "double_delta": _double_delta,
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for a manifest mapping."""

    return hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()


def default_manifest_path() -> Path:
    """Locate the repository-local core claims manifest."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "benchmarks" / "core_claims_v1.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("benchmarks/core_claims_v1.json was not found")


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a model-selection manifest."""

    manifest_path = default_manifest_path() if path is None else Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject ambiguous scopes, unknown candidates, or invalid evidence rows."""

    if manifest.get("recursion_scope") != RECURSION_SCOPE:
        raise ValueError(
            f"recursion_scope must be {RECURSION_SCOPE!r}; "
            f"got {manifest.get('recursion_scope')!r}"
        )
    model_spec = manifest.get("model_spec")
    if not isinstance(model_spec, Mapping):
        raise ValueError("manifest must contain a model_spec mapping")
    if model_spec.get("recursion_scope") != RECURSION_SCOPE:
        raise ValueError("model_spec recursion_scope must match the scalar invariant sector")

    grid = manifest.get("candidate_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("manifest must contain candidate_grid")
    registry_checks = (
        ("survival", SURVIVAL_LAWS),
        ("feedback", FEEDBACK_LAWS),
        ("d_eff", D_EFF_MAPS),
    )
    for key, registry in registry_checks:
        entries = grid.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"candidate_grid.{key} must be a non-empty list")
        ids = [entry.get("id") for entry in entries if isinstance(entry, Mapping)]
        if len(ids) != len(entries) or len(set(ids)) != len(ids):
            raise ValueError(f"candidate_grid.{key} IDs must be present and unique")
        unknown = sorted(set(ids) - set(registry))
        if unknown:
            raise ValueError(f"unknown {key} candidates: {unknown}")
        for entry in entries:
            count = entry.get("parameter_count", 0)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"{key} parameter_count must be a non-negative integer")

    observations = manifest.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("manifest must contain observations")
    keys: list[str] = []
    for row in observations:
        if not isinstance(row, Mapping):
            raise ValueError("every observation must be a mapping")
        key = row.get("key")
        role = row.get("role")
        if not isinstance(key, str) or not key:
            raise ValueError("every observation must have a non-empty key")
        if role not in _OBSERVATION_ROLES:
            raise ValueError(f"invalid observation role for {key}: {role!r}")
        sigma = row.get("sigma")
        if not isinstance(sigma, (int, float)) or not math.isfinite(float(sigma)):
            raise ValueError(f"observation {key} must have a finite sigma")
        if role in {"calibration", "selection", "confirmation"} and float(sigma) <= 0.0:
            raise ValueError(f"scored observation {key} must have sigma > 0")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("observation keys must be unique")

    scan = manifest.get("root_scan")
    if not isinstance(scan, Mapping):
        raise ValueError("manifest must contain root_scan")
    if int(scan.get("scan_points", 0)) < 32:
        raise ValueError("root_scan.scan_points must be at least 32")
    interval = scan.get("interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or float(interval[0]) >= float(interval[1])
    ):
        raise ValueError("root_scan.interval must be an increasing pair")


def observations_from_manifest(manifest: Mapping[str, Any]) -> tuple[Observation, ...]:
    rows: list[Observation] = []
    for raw in manifest["observations"]:
        rows.append(
            Observation(
                key=str(raw["key"]),
                adapter=str(raw["adapter"]),
                value=float(raw["value"]),
                sigma=float(raw["sigma"]),
                role=str(raw["role"]),
                correlation_group=str(raw.get("correlation_group", raw["key"])),
                source=str(raw.get("source", "")),
            )
        )
    return tuple(rows)


def context_from_observations(observations: Sequence[Observation]) -> ModelContext:
    inputs = {item.adapter: item.value for item in observations if item.role == "input"}
    missing = {"spatial_dimension", "delta"} - set(inputs)
    if missing:
        raise ValueError(f"missing required scalar-sector inputs: {sorted(missing)}")
    return ModelContext(
        spatial_dimension=float(inputs["spatial_dimension"]),
        delta=float(inputs["delta"]),
    )


def candidate_specs(manifest: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    """Expand the preregistered Cartesian candidate grid."""

    grid = manifest["candidate_grid"]
    specs: list[CandidateSpec] = []
    for survival in grid["survival"]:
        for feedback in grid["feedback"]:
            for d_eff in grid["d_eff"]:
                ids = (str(survival["id"]), str(feedback["id"]), str(d_eff["id"]))
                specs.append(
                    CandidateSpec(
                        candidate_id="__".join(ids),
                        survival_id=ids[0],
                        feedback_id=ids[1],
                        d_eff_id=ids[2],
                        parameter_count=(
                            int(survival.get("parameter_count", 0))
                            + int(feedback.get("parameter_count", 0))
                            + int(d_eff.get("parameter_count", 0))
                        ),
                        recursion_scope=str(manifest["recursion_scope"]),
                    )
                )
    return tuple(specs)


def effective_dimension(spec: CandidateSpec, context: ModelContext) -> float:
    return float(D_EFF_MAPS[spec.d_eff_id](context))


def fixed_map(spec: CandidateSpec, context: ModelContext, x: float) -> float:
    """Evaluate F(x)=S(D_eff*K(x)) for the scalar invariant sector."""

    d_eff = effective_dimension(spec, context)
    feedback = FEEDBACK_LAWS[spec.feedback_id].value(x)
    return SURVIVAL_LAWS[spec.survival_id].value(d_eff * feedback)


def fixed_map_derivative(spec: CandidateSpec, context: ModelContext, x: float) -> float:
    d_eff = effective_dimension(spec, context)
    feedback_law = FEEDBACK_LAWS[spec.feedback_id]
    survival_law = SURVIVAL_LAWS[spec.survival_id]
    z = d_eff * feedback_law.value(x)
    return survival_law.derivative(z) * d_eff * feedback_law.derivative(x)


def fixed_point_residual(spec: CandidateSpec, context: ModelContext, x: float) -> float:
    return x - fixed_map(spec, context, x)


def _bisect_root(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    x_tolerance: float,
    max_iterations: int = 200,
) -> float:
    f_lo = func(lo)
    f_hi = func(hi)
    if abs(f_lo) == 0.0:
        return lo
    if abs(f_hi) == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError("root is not bracketed")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if abs(f_mid) <= 1.0e-16 or abs(hi - lo) <= x_tolerance:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def _minimize_abs_residual(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    iterations: int = 80,
) -> float:
    """Golden-section refinement used to catch tangential grid roots."""

    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - golden * (hi - lo)
    right = lo + golden * (hi - lo)
    f_left = abs(func(left))
    f_right = abs(func(right))
    for _ in range(iterations):
        if f_left <= f_right:
            hi = right
            right = left
            f_right = f_left
            left = hi - golden * (hi - lo)
            f_left = abs(func(left))
        else:
            lo = left
            left = right
            f_left = f_right
            right = lo + golden * (hi - lo)
            f_right = abs(func(right))
    return 0.5 * (lo + hi)


def _deduplicate_roots(values: Iterable[float], merge_tolerance: float) -> list[float]:
    roots: list[float] = []
    for value in sorted(values):
        if not roots or abs(value - roots[-1]) > merge_tolerance:
            roots.append(value)
        elif abs(value - 1.0) < abs(roots[-1] - 1.0):
            roots[-1] = value
    return roots


def scan_roots(
    spec: CandidateSpec,
    context: ModelContext,
    root_scan: Mapping[str, Any],
) -> tuple[RootAudit, ...]:
    """Scan the complete configured interval, including endpoint/tangent roots."""

    lo, hi = (float(value) for value in root_scan["interval"])
    scan_points = int(root_scan["scan_points"])
    residual_tolerance = float(root_scan["residual_tolerance"])
    x_tolerance = float(root_scan["x_tolerance"])
    merge_tolerance = float(root_scan["merge_tolerance"])
    stability_tolerance = float(root_scan.get("stability_tolerance", 1.0e-10))
    selection_policy = str(root_scan.get("selection_policy", "stable_nontrivial"))

    def func(value: float) -> float:
        return fixed_point_residual(spec, context, value)
    xs = [lo + (hi - lo) * index / scan_points for index in range(scan_points + 1)]
    fs = [func(value) for value in xs]
    raw_roots: list[float] = []

    for value, residual in zip(xs, fs):
        if abs(residual) <= residual_tolerance:
            raw_roots.append(value)

    for index in range(scan_points):
        left, right = xs[index], xs[index + 1]
        f_left, f_right = fs[index], fs[index + 1]
        if f_left * f_right < 0.0:
            raw_roots.append(
                _bisect_root(
                    func,
                    left,
                    right,
                    x_tolerance=x_tolerance,
                )
            )

    for index in range(1, scan_points):
        current = abs(fs[index])
        if current < abs(fs[index - 1]) and current < abs(fs[index + 1]):
            candidate = _minimize_abs_residual(func, xs[index - 1], xs[index + 1])
            if abs(func(candidate)) <= residual_tolerance:
                raw_roots.append(candidate)

    values = _deduplicate_roots(raw_roots, merge_tolerance)
    root_rows: list[RootAudit] = []
    nontrivial_index = 0
    for value in values:
        residual = abs(func(value))
        derivative = fixed_map_derivative(spec, context, value)
        radius = abs(derivative)
        is_trivial = abs(value - 1.0) <= 10.0 * merge_tolerance
        if is_trivial:
            label = "trivial"
        else:
            nontrivial_index += 1
            label = f"nontrivial_{nontrivial_index}"
        stable = radius < 1.0 - stability_tolerance
        if selection_policy == "stable_nontrivial":
            eligible = stable and not is_trivial
        elif selection_policy == "all":
            eligible = True
        else:
            raise ValueError(f"unknown root selection policy: {selection_policy}")
        root_rows.append(
            RootAudit(
                value=value,
                residual=residual,
                fixed_map_derivative=derivative,
                stability_radius=radius,
                stable=stable,
                branch_label=label,
                eligible_for_selection=eligible,
            )
        )
    return tuple(root_rows)


def algebraic_audit(
    spec: CandidateSpec,
    context: ModelContext,
    manifest: Mapping[str, Any],
) -> AlgebraicAudit:
    """Check algebraic properties without using observational fit quality."""

    audit_config = manifest["algebraic_audit"]
    tolerance = float(audit_config["tolerance"])
    factorization_tolerance = float(audit_config["factorization_tolerance"])
    survival_grid = [float(value) for value in audit_config["survival_grid"]]
    feedback_grid = [float(value) for value in audit_config["feedback_grid"]]
    law = SURVIVAL_LAWS[spec.survival_id]
    feedback = FEEDBACK_LAWS[spec.feedback_id]

    survival_values = [law.value(value) for value in survival_grid]
    normalization_error = abs(law.value(0.0) - 1.0)
    initial_slope_error = abs(law.derivative(0.0) + 1.0)
    positive_and_bounded = all(
        math.isfinite(value) and 0.0 < value <= 1.0 + tolerance
        for value in survival_values
    )
    survival_monotone = all(
        right <= left + tolerance
        for left, right in zip(survival_values, survival_values[1:])
    )

    defects: list[float] = []
    for left in survival_grid:
        for right in survival_grid:
            s_left = law.value(left)
            s_right = law.value(right)
            s_sum = law.value(left + right)
            if min(s_left, s_right, s_sum) <= 0.0:
                defects.append(float("inf"))
            else:
                defects.append(abs(math.log(s_sum) - math.log(s_left) - math.log(s_right)))
    factorization_max = max(defects)
    factorization_rms = math.sqrt(sum(value * value for value in defects) / len(defects))

    feedback_values = [feedback.value(value) for value in feedback_grid]
    feedback_zero_error = abs(feedback.value(0.0) - 1.0)
    feedback_one_error = abs(feedback.value(1.0))
    feedback_bounded = all(
        math.isfinite(value) and -tolerance <= value <= 1.0 + tolerance
        for value in feedback_values
    )
    feedback_monotone = all(
        right <= left + tolerance
        for left, right in zip(feedback_values, feedback_values[1:])
    )

    d_eff = effective_dimension(spec, context)
    d_eff_valid = math.isfinite(d_eff) and d_eff > 1.0
    roots = scan_roots(spec, context, manifest["root_scan"])
    max_root_residual = max((root.residual for root in roots), default=float("inf"))
    residual_tolerance = float(manifest["root_scan"]["residual_tolerance"])

    required_checks = (
        spec.recursion_scope == RECURSION_SCOPE,
        normalization_error <= tolerance,
        initial_slope_error <= tolerance,
        positive_and_bounded,
        survival_monotone,
        feedback_zero_error <= tolerance,
        feedback_one_error <= tolerance,
        feedback_bounded,
        feedback_monotone,
        d_eff_valid,
        bool(roots),
        max_root_residual <= residual_tolerance,
    )
    status = ALGEBRAIC_PASS if all(required_checks) else ALGEBRAIC_FAIL
    return AlgebraicAudit(
        status=status,
        normalization_error=normalization_error,
        initial_slope_error=initial_slope_error,
        survival_positive_and_bounded=positive_and_bounded,
        survival_monotone=survival_monotone,
        factorization_defect_max=factorization_max,
        factorization_defect_rms=factorization_rms,
        factorization_compatible=factorization_max <= factorization_tolerance,
        feedback_at_zero_error=feedback_zero_error,
        feedback_at_one_error=feedback_one_error,
        feedback_bounded=feedback_bounded,
        feedback_monotone=feedback_monotone,
        d_eff=d_eff,
        d_eff_valid=d_eff_valid,
        root_count=len(roots),
        max_root_residual=max_root_residual,
        roots=roots,
    )


def _predict_observation(root: RootAudit, observation: Observation) -> float:
    if observation.adapter in {"fixed_point", "omega_b_bridge"}:
        return root.value
    if observation.adapter == "stability_radius":
        return root.stability_radius
    raise ValueError(f"unknown observation adapter: {observation.adapter}")


def _selection_observations(observations: Sequence[Observation]) -> tuple[Observation, ...]:
    return tuple(item for item in observations if item.role == "selection")


def _independent_observation_count(observations: Sequence[Observation]) -> int:
    return len({item.correlation_group for item in observations})


def score_root_selection(
    root: RootAudit,
    observations: Sequence[Observation],
    manifest: Mapping[str, Any],
    *,
    algebraic_status: str,
) -> SelectionScore:
    """Score a branch without allowing inputs or references into the denominator."""

    selected = _selection_observations(observations)
    independent_count = _independent_observation_count(selected)
    if algebraic_status != ALGEBRAIC_PASS or not root.eligible_for_selection:
        return SelectionScore(
            status=NOT_TESTABLE,
            n_observations=len(selected),
            n_independent_observations=independent_count,
            chi2=None,
            max_abs_z=None,
            rms_z=None,
            residuals=(),
        )
    if not selected:
        return SelectionScore(
            status=NOT_TESTABLE,
            n_observations=0,
            n_independent_observations=0,
            chi2=None,
            max_abs_z=None,
            rms_z=None,
            residuals=(),
        )

    residuals: list[SelectionResidual] = []
    for observation in selected:
        prediction = _predict_observation(root, observation)
        z_score = (prediction - observation.value) / observation.sigma
        residuals.append(
            SelectionResidual(
                key=observation.key,
                prediction=prediction,
                observed=observation.value,
                sigma=observation.sigma,
                z_score=z_score,
                correlation_group=observation.correlation_group,
            )
        )
    chi2 = sum(item.z_score * item.z_score for item in residuals)
    max_abs_z = max(abs(item.z_score) for item in residuals)
    rms_z = math.sqrt(chi2 / len(residuals))

    selection_config = manifest["selection"]
    minimum = int(selection_config["minimum_independent_observations"])
    if independent_count < minimum:
        status = UNDERIDENTIFIED
    elif max_abs_z <= float(selection_config["competitive_max_abs_z"]):
        status = COMPETITIVE
    else:
        status = DISFAVORED
    return SelectionScore(
        status=status,
        n_observations=len(selected),
        n_independent_observations=independent_count,
        chi2=chi2,
        max_abs_z=max_abs_z,
        rms_z=rms_z,
        residuals=tuple(residuals),
    )


def _candidate_selection_status(branches: Sequence[BranchEvaluation]) -> str:
    statuses = {
        branch.selection.status
        for branch in branches
        if branch.root.eligible_for_selection
    }
    if not statuses:
        return NOT_TESTABLE
    if UNDERIDENTIFIED in statuses:
        return UNDERIDENTIFIED
    if COMPETITIVE in statuses:
        return COMPETITIVE
    if statuses == {DISFAVORED}:
        return DISFAVORED
    return NOT_TESTABLE


def evaluate_candidate(
    spec: CandidateSpec,
    context: ModelContext,
    observations: Sequence[Observation],
    manifest: Mapping[str, Any],
) -> CandidateEvaluation:
    algebraic = algebraic_audit(spec, context, manifest)
    branches = tuple(
        BranchEvaluation(
            root=root,
            selection=score_root_selection(
                root,
                observations,
                manifest,
                algebraic_status=algebraic.status,
            ),
        )
        for root in algebraic.roots
    )
    return CandidateEvaluation(
        spec=spec,
        algebraic_status=algebraic.status,
        selection_status=_candidate_selection_status(branches),
        algebraic=algebraic,
        branches=branches,
    )


def _overall_selection_status(candidates: Sequence[CandidateEvaluation]) -> str:
    statuses = {candidate.selection_status for candidate in candidates}
    if UNDERIDENTIFIED in statuses:
        return UNDERIDENTIFIED
    if COMPETITIVE in statuses:
        return COMPETITIVE
    if DISFAVORED in statuses:
        return DISFAVORED
    return NOT_TESTABLE


def evaluate_manifest(manifest: Mapping[str, Any]) -> HarnessReport:
    """Evaluate every preregistered candidate and every discovered root."""

    validate_manifest(manifest)
    observations = observations_from_manifest(manifest)
    context = context_from_observations(observations)
    evaluations = tuple(
        evaluate_candidate(spec, context, observations, manifest)
        for spec in candidate_specs(manifest)
    )
    algebraic_status = (
        ALGEBRAIC_PASS
        if all(candidate.algebraic_status == ALGEBRAIC_PASS for candidate in evaluations)
        else ALGEBRAIC_FAIL
    )
    selection_observations = _selection_observations(observations)
    return HarnessReport(
        schema_version=str(manifest["schema_version"]),
        recursion_scope=str(manifest["recursion_scope"]),
        manifest_hash=manifest_hash(manifest),
        candidate_count=len(evaluations),
        algebraic_status=algebraic_status,
        selection_status=_overall_selection_status(evaluations),
        n_selection_observations=len(selection_observations),
        n_independent_selection_observations=_independent_observation_count(
            selection_observations
        ),
        candidates=evaluations,
    )


def candidate_by_id(
    report: HarnessReport,
    candidate_id: str,
) -> CandidateEvaluation:
    """Return one evaluated candidate by its stable manifest ID."""

    for candidate in report.candidates:
        if candidate.spec.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


__all__ = [
    "ALGEBRAIC_FAIL",
    "ALGEBRAIC_PASS",
    "COMPETITIVE",
    "DISFAVORED",
    "NOT_TESTABLE",
    "RECURSION_SCOPE",
    "UNDERIDENTIFIED",
    "AlgebraicAudit",
    "BranchEvaluation",
    "CandidateEvaluation",
    "CandidateSpec",
    "HarnessReport",
    "ModelContext",
    "Observation",
    "RootAudit",
    "SelectionScore",
    "algebraic_audit",
    "candidate_by_id",
    "candidate_specs",
    "context_from_observations",
    "default_manifest_path",
    "effective_dimension",
    "evaluate_candidate",
    "evaluate_manifest",
    "fixed_map",
    "fixed_map_derivative",
    "fixed_point_residual",
    "load_manifest",
    "manifest_hash",
    "observations_from_manifest",
    "scan_roots",
    "score_root_selection",
    "validate_manifest",
]
