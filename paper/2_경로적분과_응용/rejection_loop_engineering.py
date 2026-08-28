"""Executable regression registry for parent claims deleted from chapters 01--14.

Canonical prose contains none of the deleted parent formulations.  This module
keeps their fingerprints, executable predicates, and explicitly separated
descendant calculations so an invalid parent cannot silently re-enter the text.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.physics.primordial_spectrum_readout_gate import (  # noqa: E402
    OBS_AS_1E9 as CANONICAL_OBS_AS_1E9,
    OBS_AS_SIGMA_1E9 as CANONICAL_OBS_AS_SIGMA_1E9,
)

CANONICAL_OBS_AS = CANONICAL_OBS_AS_1E9 * 1.0e-9
REGRESSION_WITNESS_DIGEST_FORMAT = "ce-regression-witness-v2-float15"
CHAPTERS = tuple(
    sorted(path for path in HERE.glob("[0-9][0-9]_*.md") if not path.name.startswith("00_"))
)

TERMINAL_STATUSES = {
    "REJECT",
    "EXACT_COUNTEREXAMPLE",
    "SELECTION_PASS",
    "BRIDGE_PASS",
    "PHENOMENOLOGY_PASS",
    "AUDIT_PASS",
    "OPEN",
    "NOT_REACHED",
    "NEGATIVE_CONTROL",
}
CLAIM_STATUS_TOKENS = (
    "[정의]",
    "[정리]",
    "[공리]",
    "[산출]",
    "[경험식]",
    "[미완성]",
    "[예측]",
)

# These source rows are anaphoric continuations: their semantic subject lives
# immediately above the line containing the [Rejected] disposition.  The explicit counts are
# part of the reviewed occurrence manifest, rather than a broad +/-N fallback.
SEMANTIC_CONTEXT_PREPEND_LINES: dict[tuple[str, int], int] = {
    ("02", 103): 1,
    ("13", 33): 1,
    ("13", 146): 1,
    ("14", 205): 2,
}


@dataclass(frozen=True)
class Iteration:
    index: int
    candidate: str
    status: str
    mutation: str
    added_inputs: tuple[str, ...]
    gate: str
    metrics: dict[str, Any]
    limitation: str


@dataclass(frozen=True)
class RejectLoop:
    loop_id: str
    source_chapters: tuple[str, ...]
    parent_claim: str
    parent_status: str
    iterations: tuple[Iteration, ...]
    maximum_supported_stage: str
    ce_specific_physical_claim_closed: bool
    next_required_gate: str


@dataclass(frozen=True)
class SelfCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RejectedOccurrenceRoute:
    chapter: str
    line: int
    claim_text: str
    semantic_context: str
    semantic_gate_ids: tuple[str, ...]
    loop_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticGateDefinition:
    gate_id: str
    loop_ids: tuple[str, ...]
    kill_gate: str
    alternate_route: str
    next_gate: str


@dataclass(frozen=True)
class RemainingRejectedRouteRule:
    rule_id: str
    chapter: str
    required_fragments: tuple[str, ...]
    semantic_gate_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegressionWitness:
    witness_id: str
    parent_claim: str
    parent_fingerprint_sha256: str
    loop_ids: tuple[str, ...]
    predicate: str
    observed: dict[str, Any]
    passed: bool


@dataclass(frozen=True)
class LoopEngineeringReport:
    schema_version: str
    source_sha256: str
    source_rejected_literal_occurrences: int
    excluded_taxonomy_occurrences: int
    source_rejected_occurrences: int
    routed_rejected_occurrences: int
    original_claims_promoted: int
    descendant_stage_passes: int
    ce_specific_physical_claims_closed: int
    loops: tuple[RejectLoop, ...]
    routing_by_chapter: dict[str, tuple[str, ...]]
    semantic_gate_definitions: dict[str, SemanticGateDefinition]
    occurrence_routes: tuple[RejectedOccurrenceRoute, ...]
    deleted_parent_regression_witnesses: tuple[RegressionWitness, ...]
    regression_witness_registry_sha256: str


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def bisect(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    iterations: int = 200,
) -> float:
    f_lower = function(lower)
    f_upper = function(upper)
    if f_lower * f_upper > 0.0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = function(midpoint)
        if f_lower * f_midpoint <= 0.0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint
    return 0.5 * (lower + upper)


def fixed_point(d_eff: float) -> float:
    value = 0.05
    for _ in range(1000):
        next_value = math.exp(-d_eff * (1.0 - value))
        if abs(next_value - value) < 1e-15:
            return next_value
        value = next_value
    raise RuntimeError("fixed-point iteration did not converge")


def simpson_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int = 200_000,
) -> float:
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson intervals must be a positive even integer")
    step = (upper - lower) / intervals
    total = function(lower) + function(upper)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * function(
            lower + index * step
        )
    return total * step / 3.0


def benchmark_core() -> dict[str, float]:
    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = 3.0 + delta
    q = fixed_point(d_eff)
    return {
        "alpha_s": alpha_s,
        "sin2_theta_w": sin2_theta_w,
        "delta": delta,
        "d_eff": d_eff,
        "q": q,
        "sigma": 1.0 - q,
    }


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in CHAPTERS:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        canonical_text = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        digest.update(canonical_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def rejected_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in CHAPTERS:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if "[Rejected]" not in line:
                continue
            stripped = line.strip()
            taxonomy_legend = line_number <= 15 and (
                all(token in line for token in CLAIM_STATUS_TOKENS)
                or stripped.startswith("- **[Rejected]**")
                or stripped.startswith("| `[Rejected]` |")
            )
            if taxonomy_legend:
                continue
            stripped_without_status = (
                stripped.replace("**[Rejected]**", "")
                .replace("[Rejected]", "")
                .replace("[Open]", "")
                .strip(" `*:-|")
            )
            context_lines = [stripped]
            # A few manuscript rows put the status marker on its own line.
            # Only those incomplete markers receive forward context; looking
            # backwards or blindly taking +/-2 lines can absorb a neighbouring
            # rejected claim and make semantic routing ambiguous.
            marker_leads_line = stripped.startswith(
                (
                    "**[Open][Rejected]**",
                    "`[Open][Rejected]`",
                    "[Open][Rejected]",
                    "**[Rejected]**",
                    "`[Rejected]`",
                    "[Rejected]",
                )
            )
            if marker_leads_line and len(stripped_without_status) < 20:
                for following in lines[line_number : line_number + 5]:
                    following_stripped = following.strip()
                    if not following_stripped:
                        if len(context_lines) > 1:
                            break
                        continue
                    if (
                        following_stripped.startswith("#")
                        or "[Rejected]" in following_stripped
                    ):
                        break
                    context_lines.append(following_stripped)
            if len(context_lines) == 1 and len(stripped_without_status) < 20:
                preceding_context: list[str] = []
                start = max(0, line_number - 5)
                for preceding in reversed(lines[start : line_number - 1]):
                    preceding_stripped = preceding.strip()
                    if (
                        not preceding_stripped
                        or preceding_stripped.startswith("#")
                        or "[Rejected]" in preceding_stripped
                    ):
                        break
                    preceding_context.append(preceding_stripped)
                if preceding_context:
                    context_lines = list(reversed(preceding_context)) + context_lines
            prepend_count = SEMANTIC_CONTEXT_PREPEND_LINES.get(
                (path.name[:2], line_number), 0
            )
            if prepend_count:
                explicit_preceding = [
                    item.strip()
                    for item in lines[
                        max(0, line_number - 1 - prepend_count) : line_number - 1
                    ]
                    if item.strip()
                ]
                context_lines = explicit_preceding + [stripped]
            inventory.append(
                {
                    "chapter": path.name,
                    "line": line_number,
                    "text": line.strip(),
                    "context": " ".join(context_lines),
                }
            )
    return inventory


def rejected_literal_occurrence_count() -> int:
    return sum(
        path.read_text(encoding="utf-8").count("[Rejected]")
        for path in CHAPTERS
    )


def finite_expression_search(
    target: float,
    powers: tuple[int, ...],
    coefficient_bound: int = 10,
) -> tuple[tuple[int, ...], float, float, int]:
    best_coefficients: tuple[int, ...] | None = None
    best_value = math.nan
    best_error = math.inf
    trials = 0
    coefficient_range = range(-coefficient_bound, coefficient_bound + 1)
    for coefficients in itertools.product(coefficient_range, repeat=len(powers)):
        if all(coefficient == 0 for coefficient in coefficients):
            continue
        trials += 1
        value = sum(
            coefficient * math.pi**power
            for coefficient, power in zip(coefficients, powers)
        )
        error = abs(value - target)
        if error < best_error:
            best_coefficients = tuple(coefficients)
            best_value = value
            best_error = error
    if best_coefficients is None:
        raise RuntimeError("expression search produced no candidate")
    return best_coefficients, best_value, best_error, trials


def finite_sparse_expression_search(
    target: float,
    powers: tuple[int, ...],
    coefficient_bound: int = 6,
    max_nonzero_terms: int = 3,
) -> tuple[tuple[int, ...], float, float, int]:
    """Exhaust a preregistered sparse integer polynomial grammar."""

    if coefficient_bound <= 0:
        raise ValueError("coefficient_bound must be positive")
    if not 1 <= max_nonzero_terms <= len(powers):
        raise ValueError("max_nonzero_terms is outside the grammar")
    nonzero_coefficients = tuple(
        coefficient
        for coefficient in range(-coefficient_bound, coefficient_bound + 1)
        if coefficient != 0
    )
    best_coefficients: tuple[int, ...] | None = None
    best_value = math.nan
    best_error = math.inf
    trials = 0
    for support_size in range(1, max_nonzero_terms + 1):
        for support in itertools.combinations(range(len(powers)), support_size):
            for selected in itertools.product(
                nonzero_coefficients, repeat=support_size
            ):
                coefficients = [0] * len(powers)
                for index, coefficient in zip(support, selected):
                    coefficients[index] = coefficient
                trials += 1
                value = sum(
                    coefficient * math.pi**power
                    for coefficient, power in zip(coefficients, powers)
                )
                error = abs(value - target)
                if error < best_error:
                    best_coefficients = tuple(coefficients)
                    best_value = value
                    best_error = error
    if best_coefficients is None:
        raise RuntimeError("sparse expression search produced no candidate")
    return best_coefficients, best_value, best_error, trials


def nonminimal_quartic_solution(
    xi: float = 0.49,
    e_folds: float = 60.0,
    target_as: float = CANONICAL_OBS_AS,
) -> dict[str, float]:
    def field_metric(field: float) -> float:
        return (
            1.0 + xi * (1.0 + 6.0 * xi) * field**2
        ) / (1.0 + xi * field**2) ** 2

    def log_potential_prime(field: float) -> float:
        return 4.0 / (field * (1.0 + xi * field**2))

    def epsilon(field: float) -> float:
        return 0.5 * log_potential_prime(field) ** 2 / field_metric(field)

    field_end = bisect(lambda field: epsilon(field) - 1.0, 0.01, 10.0)

    def accumulated_e_folds(field: float) -> float:
        return (
            (1.0 + 6.0 * xi) * (field**2 - field_end**2) / 8.0
            - 0.75
            * math.log(
                (1.0 + xi * field**2) / (1.0 + xi * field_end**2)
            )
        )

    field_star = bisect(
        lambda field: accumulated_e_folds(field) - e_folds,
        field_end,
        50.0,
    )
    metric = field_metric(field_star)
    u_prime_over_u = log_potential_prime(field_star)
    derivative_u_prime_over_u = -4.0 * (
        1.0 + 3.0 * xi * field_star**2
    ) / (field_star**2 * (1.0 + xi * field_star**2) ** 2)
    a = xi * (1.0 + 6.0 * xi)
    derivative_log_metric = (
        2.0 * a * field_star / (1.0 + a * field_star**2)
        - 4.0 * xi * field_star / (1.0 + xi * field_star**2)
    )
    epsilon_star = epsilon(field_star)
    eta_star = (
        (derivative_u_prime_over_u + u_prime_over_u**2) / metric
        - u_prime_over_u * derivative_log_metric / (2.0 * metric)
    )
    n_s = 1.0 - 6.0 * epsilon_star + 2.0 * eta_star
    tensor_ratio = 16.0 * epsilon_star
    quartic_lambda = (
        target_as
        * 96.0
        * math.pi**2
        * epsilon_star
        * (1.0 + xi * field_star**2) ** 2
        / field_star**4
    )
    einstein_potential = (
        quartic_lambda
        * field_star**4
        / (4.0 * (1.0 + xi * field_star**2) ** 2)
    )
    reconstructed_as = einstein_potential / (
        24.0 * math.pi**2 * epsilon_star
    )
    return {
        "field_end": field_end,
        "field_star": field_star,
        "epsilon_star": epsilon_star,
        "n_s": n_s,
        "r": tensor_ratio,
        "lambda_4_required": quartic_lambda,
        "A_s_reconstructed": reconstructed_as,
    }


def relaxation_flow(
    d_eff: float,
    initial_value: float = 1.0 - 1e-6,
    final_time: float = 40.0,
    step: float = 0.002,
) -> dict[str, float]:
    target = fixed_point(d_eff)

    def derivative(value: float) -> float:
        return math.exp(-d_eff * (1.0 - value)) - value

    value = initial_value
    first_hit = math.nan
    steps = int(round(final_time / step))
    for index in range(steps):
        k1 = derivative(value)
        k2 = derivative(value + 0.5 * step * k1)
        k3 = derivative(value + 0.5 * step * k2)
        k4 = derivative(value + step * k3)
        value += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if math.isnan(first_hit) and abs(value - target) < 1e-6:
            first_hit = (index + 1) * step
    return {
        "initial_value": initial_value,
        "target": target,
        "final_value": value,
        "absolute_error": abs(value - target),
        "dimensionless_time_to_1e-6": first_hit,
        "relaxation_rate_at_target": 1.0 - d_eff * target,
    }


def dimension_loop() -> RejectLoop:
    roots = tuple(d for d in range(0, 12) if d == d * (d - 1) // 2)
    return RejectLoop(
        loop_id="dimension_hodge",
        source_chapters=("01_차원의_유일성.md",),
        parent_claim="2-form Hodge self-duality uniquely proves three spatial dimensions",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "literal 2-form self-duality",
                "EXACT_COUNTEREXAMPLE",
                "enforce equal form degree under the Hodge star",
                (),
                "solve 2=d-2",
                {"selected_dimension": 4},
                "the literal self-duality condition selects d=4, not d=3",
            ),
            Iteration(
                1,
                "one-form/two-form type closure",
                "SELECTION_PASS",
                "replace self-duality by dim Lambda^1 = dim Lambda^2",
                ("type-closure axiom", "positive integer d>1"),
                "solve d=binomial(d,2)",
                {"integer_roots": roots, "selected_positive_root": 3},
                "the new axiom is not derived from the CE action",
            ),
            Iteration(
                2,
                "dynamics and observation",
                "NOT_REACHED",
                "ask whether the closure axiom is dynamically selected",
                ("dimension-changing action", "measure", "boundary conditions"),
                "derive a probability concentrated at d=3",
                {"derived": False},
                "no dimension-selection dynamics is supplied",
            ),
        ),
        maximum_supported_stage="SELECTION_DIMENSION",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the type-closure axiom from a covariant action and measure",
    )


def gauge_coupling_loop(core: dict[str, float]) -> RejectLoop:
    alpha_em_mz = 1.0 / 127.95

    def residual(alpha_s: float) -> float:
        sin2 = 4.0 * alpha_s ** (4.0 / 3.0)
        return alpha_s + alpha_em_mz / sin2 + alpha_em_mz - 1.0 / (2.0 * math.pi)

    alpha_s = bisect(residual, 0.10, 0.15)
    sin2 = 4.0 * alpha_s ** (4.0 / 3.0)
    alpha2 = alpha_em_mz / sin2
    return RejectLoop(
        loop_id="gauge_structure_and_couplings",
        source_chapters=("02_에스컬레이터.md", "10_공리_정당화.md"),
        parent_claim="d=3 and alpha_total=1/(2pi) alone derive the SM gauge group and all couplings",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "single sum-rule closure",
                "REJECT",
                "leave three couplings unknown with too few equations",
                (),
                "unknown count must not exceed independent equation count",
                {"unknown_couplings": 3, "independent_equations": 1},
                "the system is underdetermined",
            ),
            Iteration(
                1,
                "scale-matched three-equation bridge",
                "PHENOMENOLOGY_PASS",
                "supply alpha_em(MZ) and two CE bridge equations",
                (
                    "alpha_em(MZ)=1/127.95",
                    "sin2=4 alpha_s^(4/3)",
                    "alpha_em=alpha2 sin2",
                ),
                "solve the three equations and require sub-percent benchmark residuals",
                {
                    "alpha_s": alpha_s,
                    "sin2_theta_w": sin2,
                    "alpha2": alpha2,
                    "sum": alpha_s + alpha2 + alpha_em_mz,
                },
                "the QED running and CE bridge equations are supplied rather than derived",
            ),
            Iteration(
                2,
                "explicit direct-product gauge action",
                "BRIDGE_PASS",
                "supply SU(3)xSU(2)xU(1), representations, and anomaly-free hypercharges",
                ("gauge group", "matter representations", "hypercharges"),
                "construct a gauge-invariant anomaly-free action",
                {"group_supplied": True, "derived_from_dimension": False},
                "this makes the theory consistent but does not derive the group from d",
            ),
        ),
        maximum_supported_stage="SCALE_MATCHED_PHENOMENOLOGICAL_CLOSURE",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the gauge action, matching scale, thresholds, and beta functions from CE",
    )


def bootstrap_cosmology_loop(core: dict[str, float]) -> RejectLoop:
    alpha_total = 1.0 / (2.0 * math.pi)
    coupling_ratio_sum = (
        0.01008 + 0.03353 + core["alpha_s"]
    ) / alpha_total
    ratio = core["alpha_s"] * (
        3.0 + core["q"] * coupling_ratio_sum
    ) + core["alpha_s"] * core["delta"] * (
        1.0 + core["q"] * core["delta"]
    )
    omega_lambda = core["sigma"] / (1.0 + ratio)
    omega_dm = core["sigma"] * ratio / (1.0 + ratio)
    omega_m = core["q"] + omega_dm
    return RejectLoop(
        loop_id="bootstrap_cosmology",
        source_chapters=("03_자유매개변수.md", "04_해결한_난제.md"),
        parent_claim="the fixed point alone is a zero-input prediction of baryon, DM, and DE densities",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "q equals d=0 and observed baryon density",
                "REJECT",
                "identify an algebraic branch with two unrelated physical quantities",
                (),
                "require an explicit state map and stress tensor",
                {"state_map_present": False, "stress_tensor_present": False},
                "the fixed-point variable is an extinction probability in the stated interpretation",
            ),
            Iteration(
                1,
                "three-layer static density readout",
                "PHENOMENOLOGY_PASS",
                "supply a coupling-ratio sum and identify q with Omega_b",
                (
                    "q=Omega_b bridge",
                    f"cSigma={coupling_ratio_sum:.10f}",
                    "flat three-component normalization",
                ),
                "recompute all three fractions and require their sum to equal one",
                {
                    "R": ratio,
                    "Omega_lambda": omega_lambda,
                    "Omega_dm": omega_dm,
                    "Omega_m": omega_m,
                    "density_sum": core["q"] + omega_dm + omega_lambda,
                },
                "the bridge and cSigma are supplied; no background or growth likelihood is solved",
            ),
            Iteration(
                2,
                "fixed-parameter DESI DR2 forward likelihood",
                "REJECT",
                "run the repository full-covariance compressed-BAO gate without refitting",
                (
                    "external H0=67.4",
                    "external rd=147.09 Mpc",
                    "DESI DR2 compressed likelihood",
                ),
                "require a non-rejected full-covariance forward likelihood",
                {
                    "chi2": 37.100260857,
                    "dof": 13,
                    "survival_p_value": 0.000399573259824,
                    "verdict": "REJECT",
                },
                "the static CE density package fails this observed-data gate",
            ),
            Iteration(
                3,
                "same-data H0-rd scale refit",
                "NEGATIVE_CONTROL",
                "fit one overall H0*rd scale to the same DR2 likelihood",
                ("one fitted scale q", "DESI DR2 target data"),
                "record fit quality but forbid it from counting as a CE prediction",
                {
                    "q_fit": 0.986476933470,
                    "chi2": 12.608346862,
                    "dof": 12,
                    "survival_p_value": 0.398138192515,
                    "equivalent_rd_Mpc": 149.106375435,
                    "equivalent_H0_km_s_Mpc": 68.323949312,
                },
                "the passing scale is selected by the tested data and is permanently exploratory",
            ),
            Iteration(
                4,
                "covariant cosmological completion and untouched holdout",
                "NOT_REACHED",
                "promote the readout to a stress tensor and perturbation model",
                (
                    "covariant action",
                    "initial conditions",
                    "Boltzmann hierarchy",
                    "future preregistered release",
                ),
                "pass background, CMB, BAO, lensing, and growth likelihoods with one parameter set",
                {"joint_likelihood_present": False},
                "DR2 is already exploratory and static fractions do not define full dynamics",
            ),
        ),
        maximum_supported_stage="STATIC_PHENOMENOLOGICAL_DENSITY_READOUT",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive T_munu and run a preregistered joint background-plus-growth likelihood",
    )


def horizon_claim_loop(core: dict[str, float]) -> RejectLoop:
    n_e = 1.5 * core["d_eff"] * 12.0
    log_entropy = (
        0.5 * math.pi**2 * n_e
        - math.pi * core["delta"] * core["sigma"]
    )
    t_planck_s = 5.391247e-44
    mpc_km = 3.0856775814913673e19
    h0 = (
        math.sqrt(math.pi)
        * math.exp(-0.5 * log_entropy)
        / t_planck_s
        * mpc_km
    )
    m_planck_ev = 1.220910e28
    rho_quarter = 1.0e3 * (
        0.6891
        * (3.0 / 8.0)
        * m_planck_ev**4
        / math.exp(log_entropy)
    ) ** 0.25
    return RejectLoop(
        loop_id="horizon_and_claim_accounting",
        source_chapters=("04_해결한_난제.md",),
        parent_claim="a numerical proximity table proves that sixteen physical problems are solved",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "percentage-of-problems score",
                "REJECT",
                "average Open, fitted, and exact rows into one success percentage",
                (),
                "each problem must have an explicit solution criterion",
                {"valid_common_denominator": False},
                "heterogeneous closure stages cannot be averaged into truth probability",
            ),
            Iteration(
                1,
                "phase-area horizon descendant",
                "BRIDGE_PASS",
                "replace the literal entropy equality with a phase-area readout",
                (
                    "12 gauge channels",
                    "pi^2/2 entropy per e-fold",
                    "pi delta sigma boundary subtraction",
                ),
                "recompute Ne, logS, H0, and rhoLambda^(1/4)",
                {
                    "N_e": n_e,
                    "log_entropy": log_entropy,
                    "H0_km_s_Mpc": h0,
                    "rho_lambda_quarter_meV": rho_quarter,
                },
                "the same holographic identity ties the readout to H0; it is not an independent solution",
            ),
            Iteration(
                2,
                "stage-resolved closure ledger",
                "AUDIT_PASS",
                "score obligations by stage instead of declaring problems solved",
                ("Exact/Selection/Bridge/Phenomenology/Open/Open test taxonomy; Rejected disposition",),
                "require every claim to expose its next missing gate",
                {"stage_taxonomy_present": True, "truth_probability_score_used": False},
                "this closes the audit process, not the underlying physics",
            ),
        ),
        maximum_supported_stage="BRIDGE_HORIZON_READOUT_AND_AUDIT_CLOSURE",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the entropy-counting rule independently of the H0 identity",
    )


def inflation_loop() -> RejectLoop:
    solution = nonminimal_quartic_solution()
    return RejectLoop(
        loop_id="finite_xi_inflation",
        source_chapters=("05_인플레이션.md", "12_전이구간.md"),
        parent_claim="phi=0 bootstrap iteration plus a large-xi asymptotic formula is a complete inflation model",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "zero-field start with large-xi formula at xi=0.49",
                "REJECT",
                "use a stable minimum as the inflationary initial condition and an invalid asymptotic limit",
                (),
                "solve the finite-xi canonical field equations",
                {"xi": 0.49, "large_xi_condition_satisfied": False},
                "the classical field remains at the minimum and xi is not much larger than one",
            ),
            Iteration(
                1,
                "finite-xi nonminimal quartic action",
                "BRIDGE_PASS",
                "solve the exact field-space metric and fit lambda_4 only to A_s",
                (
                    "xi=0.49",
                    "N=60",
                    f"Planck 2018 A_s x 1e9={CANONICAL_OBS_AS_1E9}"
                    f"+/-{CANONICAL_OBS_AS_SIGMA_1E9} normalization",
                ),
                "integrate to N=60 and reconstruct A_s from U/(24 pi^2 epsilon)",
                solution,
                "lambda_4 is fitted to A_s and the initial-condition distribution is not derived",
            ),
            Iteration(
                2,
                "reheating and primordial likelihood",
                "NOT_REACHED",
                "evolve perturbations and the post-inflationary plasma",
                ("vacuum choice", "reheating interaction", "thermal history"),
                "jointly predict A_s, n_s, r, non-Gaussianity, and reheating observables",
                {"joint_primordial_likelihood_present": False},
                "the background slow-roll solution is not a complete cosmology",
            ),
        ),
        maximum_supported_stage="FINITE_XI_BACKGROUND_WITH_FITTED_AMPLITUDE",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive xi, lambda_4, initial conditions, and reheating from one action",
    )


def portal_loop(core: dict[str, float]) -> RejectLoop:
    portal_lambda = core["delta"] ** 2
    v_ew = 246.22
    higgs_mass = 125.25
    sm_width = 0.00407
    original_mass = v_ew * core["delta"]
    original_phase = math.sqrt(
        1.0 - 4.0 * original_mass**2 / higgs_mass**2
    )
    original_width = (
        portal_lambda**2
        * v_ew**2
        * original_phase
        / (8.0 * math.pi * higgs_mass)
    )
    original_branching = original_width / (original_width + sm_width)

    fixed_mass = 43.7677
    bound = 0.107
    fixed_phase = math.sqrt(1.0 - 4.0 * fixed_mass**2 / higgs_mass**2)
    width_at_bound = bound / (1.0 - bound) * sm_width
    lambda_limit = math.sqrt(
        width_at_bound
        * 8.0
        * math.pi
        * higgs_mass
        / (v_ew**2 * fixed_phase)
    )
    m0_at_limit = math.sqrt(fixed_mass**2 - lambda_limit * v_ew**2)
    m0_close_channel = math.sqrt(
        (higgs_mass / 2.0) ** 2 - portal_lambda * v_ew**2
    )
    mass_at_70_m0 = math.sqrt(70.0**2 - portal_lambda * v_ew**2)
    resonance_mass = 62.0
    resonance_lambda = 3.0e-4
    resonance_phase = math.sqrt(
        1.0 - 4.0 * resonance_mass**2 / higgs_mass**2
    )
    resonance_width = (
        resonance_lambda**2
        * v_ew**2
        * resonance_phase
        / (8.0 * math.pi * higgs_mass)
    )
    resonance_branching = resonance_width / (resonance_width + sm_width)
    resonance_m0 = math.sqrt(
        resonance_mass**2 - resonance_lambda * v_ew**2
    )
    nucleon_mass = 0.938
    nucleon_scalar_form_factor = 0.30
    gev_minus_two_to_cm2 = 0.389379e-27

    def estimated_spin_independent_cross_section(
        mass_gev: float,
        coupling: float,
    ) -> float:
        reduced_mass = nucleon_mass * mass_gev / (nucleon_mass + mass_gev)
        return (
            coupling**2
            * nucleon_scalar_form_factor**2
            * reduced_mass**2
            * nucleon_mass**2
            / (math.pi * higgs_mass**4 * mass_gev**2)
            * gev_minus_two_to_cm2
        )

    fixed_mass_sigma_si = estimated_spin_independent_cross_section(
        fixed_mass,
        lambda_limit,
    )
    mass_at_70_sigma_si = estimated_spin_independent_cross_section(
        70.0,
        portal_lambda,
    )
    resonance_sigma_si = estimated_spin_independent_cross_section(
        resonance_mass,
        resonance_lambda,
    )
    lz_40_gev_limit = 2.2e-48
    return RejectLoop(
        loop_id="scalar_portal_benchmark",
        source_chapters=(
            "05_인플레이션.md",
            "07_중성미자_질량.md",
            "08_바리온_비대칭.md",
            "12_전이구간.md",
        ),
        parent_claim="m0=0, lambdaHP=delta^2, and mphi=43.77 GeV is a viable exact-Z2 portal package",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "portal-dominated legacy point",
                "REJECT",
                "evaluate the stated benchmark without changing parameters",
                (),
                "require BR(h->invisible)<0.107",
                {
                    "lambda_HP": portal_lambda,
                    "mass_GeV": original_mass,
                    "Gamma_invisible_MeV": 1.0e3 * original_width,
                    "BR_invisible": original_branching,
                },
                "the invisible branching fraction is far above the supplied limit",
            ),
            Iteration(
                1,
                "fixed-mass weak-coupling branch",
                "BRIDGE_PASS",
                "reduce lambdaHP and introduce a positive bare mass",
                ("lambdaHP chosen at collider bound", "independent m0"),
                "saturate the single invisible-width bound at fixed physical mass",
                {
                    "mass_GeV": fixed_mass,
                    "lambda_HP_limit": lambda_limit,
                    "m0_GeV": m0_at_limit,
                    "BR_invisible_limit": bound,
                },
                "relic density, direct detection, global Higgs likelihood, and RG stability remain open",
            ),
            Iteration(
                2,
                "fixed-mass all-DM global-constraint branch",
                "REJECT",
                "test the width-only 43.77 GeV mutation against an external all-DM scan",
                ("external relic/direct-detection likelihood",),
                "require the point to lie in the reported surviving all-DM region",
                {
                    "mass_GeV": fixed_mass,
                    "inside_reported_mass_window": False,
                    "reported_mass_window_GeV": (60.5, 62.5),
                    "reported_lambda_window": (1.7e-4, 4.7e-4),
                    "estimated_sigma_SI_cm2": fixed_mass_sigma_si,
                    "LZ_strongest_40_GeV_limit_cm2": lz_40_gev_limit,
                    "cross_section_over_LZ_40_GeV_control": (
                        fixed_mass_sigma_si / lz_40_gev_limit
                    ),
                    "maximum_DM_fraction_from_control": (
                        lz_40_gev_limit / fixed_mass_sigma_si
                    ),
                    "source": "arXiv:2410.21089",
                    "direct_detection_source": "arXiv:2410.17036",
                },
                "the collider-only mutation is outside the surviving window and its all-DM scattering estimate is far above LZ",
            ),
            Iteration(
                3,
                "kinematically closed CE-coupling branch",
                "BRIDGE_PASS",
                "retain lambdaHP=delta^2 but raise the pole above mh/2 with m0>0",
                ("independent positive m0",),
                "close h->phiphi kinematically",
                {
                    "minimum_m0_to_close_GeV": m0_close_channel,
                    "example_mass_GeV": 70.0,
                    "example_m0_GeV": mass_at_70_m0,
                    "channel_closed": True,
                    "estimated_sigma_SI_cm2": mass_at_70_sigma_si,
                },
                "closing one decay leaves a large all-DM direct-detection cross section and does not establish viability",
            ),
            Iteration(
                4,
                "unverified external Higgs-resonance point",
                "OPEN",
                "move to the independently scanned Higgs-resonance window",
                (
                    "independent portal mass",
                    "independent lambdaHP",
                    "external global singlet-DM likelihood",
                ),
                "reproduce local collider/direct-detection controls, then verify membership in the correlated two-dimensional relic band",
                {
                    "reported_mass_window_GeV": (60.5, 62.5),
                    "reported_lambda_window": (1.7e-4, 4.7e-4),
                    "example_mass_GeV": resonance_mass,
                    "example_lambda_HP": resonance_lambda,
                    "example_m0_GeV": resonance_m0,
                    "example_Gamma_invisible_MeV": 1.0e3 * resonance_width,
                    "example_BR_invisible": resonance_branching,
                    "estimated_sigma_SI_cm2_fN_0p30": resonance_sigma_si,
                    "joint_allowed_membership_verified": False,
                    "relic_density_calculated": False,
                    "indirect_detection_likelihood_calculated": False,
                    "source": "arXiv:2410.21089",
                },
                "the marginal mass and coupling intervals do not prove that this arbitrary point lies on the correlated relic-density band",
            ),
            Iteration(
                5,
                "CE-derived common-parameter portal likelihood",
                "NOT_REACHED",
                "derive the resonance parameters and combine every constraint",
                ("cosmological history", "nucleon matrix elements", "renormalization prescription"),
                "pass one common-parameter global likelihood",
                {"global_likelihood_present": False},
                "the viable resonance branch abandons the CE mass and coupling relations",
            ),
        ),
        maximum_supported_stage="EXTERNAL_WINDOW_REPORTED_POINT_MEMBERSHIP_OPEN",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the resonance mass and coupling, then rerun the common global likelihood",
    )


def light_scalar_precision_loop() -> RejectLoop:
    scalar_mass_mev = 29.64757
    muon_mass_mev = 105.6583755
    legacy_coupling = 6.27e-4
    mass_ratio_squared = (scalar_mass_mev / muon_mass_mev) ** 2

    def denominator(x_value: float) -> float:
        return x_value**2 + (1.0 - x_value) * mass_ratio_squared

    legacy_integral = simpson_integral(
        lambda x_value: x_value**2 * (1.0 - x_value) / denominator(x_value),
        0.0,
        1.0,
    )
    scalar_integral = simpson_integral(
        lambda x_value: x_value**2 * (2.0 - x_value) / denominator(x_value),
        0.0,
        1.0,
    )
    loop_prefactor = legacy_coupling**2 / (8.0 * math.pi**2)
    legacy_delta_a = loop_prefactor * legacy_integral
    scalar_delta_a = loop_prefactor * scalar_integral
    current_delta_a = 38.0e-11
    current_sigma_a = 63.0e-11
    scalar_pull = (scalar_delta_a - current_delta_a) / current_sigma_a
    fitted_coupling = math.sqrt(
        current_delta_a * 8.0 * math.pi**2 / scalar_integral
    )

    electronic_radius_fm = 0.8406
    electronic_sigma_fm = 0.0015
    muonic_radius_fm = 0.84060
    muonic_sigma_fm = 0.00039
    legacy_delta_radius_squared_fm2 = 0.0587
    delta_radius_squared_sigma_fm2 = math.sqrt(
        (2.0 * electronic_radius_fm * electronic_sigma_fm) ** 2
        + (2.0 * muonic_radius_fm * muonic_sigma_fm) ** 2
    )
    legacy_radius_pull = (
        legacy_delta_radius_squared_fm2 / delta_radius_squared_sigma_fm2
    )
    return RejectLoop(
        loop_id="light_scalar_precision_observables",
        source_chapters=("04_해결한_난제.md",),
        parent_claim=(
            "the legacy light-scalar benchmark simultaneously explains the muon anomaly "
            "and a proton-radius discrepancy"
        ),
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "legacy one-loop kernel and historical targets",
                "REJECT",
                "recompute the archived benchmark exactly as written",
                (),
                "require the CP-even scalar numerator and current experimental targets",
                {
                    "legacy_integral": legacy_integral,
                    "legacy_delta_a_mu": legacy_delta_a,
                    "legacy_delta_a_mu_e11": legacy_delta_a / 1.0e-11,
                },
                "x^2(1-x) is not the CP-even scalar numerator and the targets have changed",
            ),
            Iteration(
                1,
                "correct CP-even scalar one-loop kernel",
                "REJECT",
                "replace the numerator by x^2(2-x) and use the 2025 theory comparison",
                ("current Delta a_mu=38(63)e-11",),
                "require the fixed legacy benchmark to lie within the current uncertainty",
                {
                    "scalar_integral": scalar_integral,
                    "delta_a_mu": scalar_delta_a,
                    "delta_a_mu_e11": scalar_delta_a / 1.0e-11,
                    "current_target_e11": 38.0,
                    "current_sigma_e11": 63.0,
                    "pull_sigma": scalar_pull,
                    "source": "arXiv:2505.21476",
                },
                "the corrected benchmark overshoots the present difference by more than seven sigma",
            ),
            Iteration(
                2,
                "coupling refit to the current central residual",
                "PHENOMENOLOGY_PASS",
                "keep the mass but refit the Yukawa coupling after observing Delta a_mu",
                ("fitted scalar-muon coupling", "current Delta a_mu central value"),
                "reproduce the central residual with the corrected scalar kernel",
                {
                    "mass_MeV": scalar_mass_mev,
                    "fitted_coupling": fitted_coupling,
                    "target_delta_a_mu": current_delta_a,
                },
                "this is a same-observable fit and supplies a Yukawa absent from the exact-Z2 portal",
            ),
            Iteration(
                3,
                "2026 electronic-versus-muonic proton-radius gate",
                "REJECT",
                "replace the historical radius target by the new electronic-hydrogen result",
                ("electronic H radius", "muonic H radius"),
                "require the fixed legacy squared-radius shift to be compatible with both radii",
                {
                    "electronic_radius_fm": electronic_radius_fm,
                    "electronic_sigma_fm": electronic_sigma_fm,
                    "muonic_radius_fm": muonic_radius_fm,
                    "muonic_sigma_fm": muonic_sigma_fm,
                    "legacy_delta_r2_fm2": legacy_delta_radius_squared_fm2,
                    "delta_r2_sigma_fm2": delta_radius_squared_sigma_fm2,
                    "legacy_pull_sigma": legacy_radius_pull,
                    "source": "Nature s41586-026-10124-3",
                },
                "the two modern radii agree while the legacy fixed shift is excluded at about 22.5 sigma",
            ),
            Iteration(
                4,
                "exact-Z2 null precision branch",
                "BRIDGE_PASS",
                "enforce zero singlet vev, mixing, and single-scalar fermion Yukawas",
                ("exact unbroken Z2 at the precision-observable scale",),
                "set the direct one-scalar contributions to zero and compare with current null residuals",
                {
                    "direct_delta_a_mu": 0.0,
                    "g_minus_2_pull_sigma": -current_delta_a / current_sigma_a,
                    "direct_delta_r2_fm2": 0.0,
                    "electronic_minus_muonic_radius_fm": electronic_radius_fm - muonic_radius_fm,
                },
                "compatibility with a null result is not an explanation and does not prove every higher loop vanishes",
            ),
            Iteration(
                5,
                "common UV-complete precision likelihood",
                "NOT_REACHED",
                "derive every allowed portal and fermion operator and evaluate all precision constraints",
                ("UV matching", "operator mixing", "global precision likelihood"),
                "pass g-2, spectroscopy, scattering, Higgs, and flavor data with one parameter set",
                {"common_precision_likelihood_present": False},
                "the fitted branch and exact-Z2 null branch are different models",
            ),
        ),
        maximum_supported_stage="EXACT_Z2_NULL_PRECISION_BRANCH_BRIDGE_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the allowed fermion operators and test a preregistered common precision likelihood",
    )


def strong_cp_loop() -> RejectLoop:
    theta_input = 0.3
    axion_over_f = -theta_input
    residual = math.remainder(theta_input + axion_over_f, 2.0 * math.pi)
    return RejectLoop(
        loop_id="strong_cp",
        source_chapters=("06_강한_CP.md",),
        parent_claim="a real CP-even singlet or bootstrap contraction dynamically sets theta_bar to zero",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "CP-even portal contraction",
                "REJECT",
                "infer removal of an existing QCD angle from absence of a new phase",
                (),
                "exhibit a shift field or protecting symmetry in the action",
                {"shift_degree_of_freedom_present": False},
                "no equation of motion acts on theta_bar",
            ),
            Iteration(
                1,
                "Peccei-Quinn axion completion",
                "BRIDGE_PASS",
                "add a periodic pseudoscalar coupled to G G-tilde",
                ("axion field", "PQ symmetry", "decay constant f_a", "anomaly coupling"),
                "minimize V=chi[1-cos(theta_bar+a/f_a)]",
                {
                    "theta_input": theta_input,
                    "a_over_f_at_minimum": axion_over_f,
                    "residual_theta": residual,
                    "normalized_second_derivative_positive": True,
                },
                "this is a standard new-field completion, not a derivation from the current CE portal",
            ),
            Iteration(
                2,
                "CE-specific UV and phenomenology",
                "NOT_REACHED",
                "derive PQ quality, anomaly coefficient, and allowed f_a from CE",
                ("UV charge assignment", "cosmological history", "experimental likelihood"),
                "pass EDM, stellar, cosmological, and laboratory constraints with derived parameters",
                {"ce_embedding_present": False},
                "the cancellation mechanism is a Bridge that adds new physics",
            ),
        ),
        maximum_supported_stage="STANDARD_AXION_MECHANISM_BRIDGE_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the axion/PQ sector and its allowed parameter window from CE",
    )


def neutrino_loop() -> RejectLoop:
    masses_mev = (0.306, 8.566, 49.986)
    v_ew = 246.22
    dirac_yukawas = tuple(
        math.sqrt(2.0) * mass_mev * 1.0e-12 / v_ew
        for mass_mev in masses_mev
    )
    weinberg_scales = tuple(
        v_ew**2 / (2.0 * mass_mev * 1.0e-12)
        for mass_mev in masses_mev
    )
    return RejectLoop(
        loop_id="neutrino_mass",
        source_chapters=("07_중성미자_질량.md",),
        parent_claim="the exact-Z2 real-singlet portal generates the legacy two-loop Majorana spectrum",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "current portal two-loop mass",
                "REJECT",
                "evaluate the operator content of the stated action",
                (),
                "require a nonzero Delta-L=2 Wilson coefficient",
                {"C_ij": 0.0, "LNV_vertex_present": False},
                "the coefficient is symmetry-forced to zero",
            ),
            Iteration(
                1,
                "Dirac-neutrino completion",
                "BRIDGE_PASS",
                "add three right-handed neutrinos, an unbroken anomaly-free B-L symmetry, and tiny Yukawa couplings",
                (
                    "three nu_R fields with B-L=-1",
                    "unbroken anomaly-free U(1)_(B-L)",
                    "Dirac Yukawa matrix",
                ),
                "forbid every Majorana term by unbroken B-L and use m_i=y_i v/sqrt(2)",
                {
                    "masses_meV": masses_mev,
                    "diagonal_yukawas": dirac_yukawas,
                    "lepton_number_conserved": True,
                    "majorana_mass_forbidden_by_unbroken_B_minus_L": True,
                },
                "the gauge symmetry, charges, and fitted Yukawa matrix are added inputs; PMNS structure is not predicted",
            ),
            Iteration(
                2,
                "Weinberg-operator completion",
                "BRIDGE_PASS",
                "add a Delta-L=2 dimension-five operator",
                ("complex symmetric C_ij", "heavy scale Lambda"),
                "set C_i=1 and infer Lambda_i=v^2/(2m_i)",
                {
                    "masses_meV": masses_mev,
                    "unit_coefficient_scales_GeV": weinberg_scales,
                    "lightest_scale_for_m3_GeV": min(weinberg_scales),
                },
                "the flavor matrix and UV particles are supplied rather than derived",
            ),
            Iteration(
                3,
                "predictive flavor completion",
                "NOT_REACHED",
                "derive masses, ordering, PMNS angles, phases, and radiative running together",
                ("UV flavor symmetry", "matching", "RG evolution"),
                "fit no neutrino observable used to choose the texture and pass held-out observables",
                {"held_out_flavor_prediction_present": False},
                "both viable completions add independent flavor parameters",
            ),
        ),
        maximum_supported_stage="CONSISTENT_MASS_GENERATION_WITH_ADDED_OPERATORS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive a UV flavor/LNV or Dirac sector with held-out predictions",
    )


def baryogenesis_loop() -> RejectLoop:
    target_yb = 8.7e-11
    g_star = 106.75
    # Here kappa_total is defined to include the relativistic equilibrium-yield
    # normalization.  Publishing the number without this convention changes the
    # inferred epsilon and Davidson-Ibarra scale by a factor about 2.40.
    efficiency_total = 0.1
    equilibrium_yield_factor = (
        135.0 * 1.202056903159594 / (4.0 * math.pi**4)
    )
    sphaleron_conversion = 28.0 / 79.0
    required_epsilon = (
        target_yb * g_star / (sphaleron_conversion * efficiency_total)
    )
    required_epsilon_explicit_yield = required_epsilon / equilibrium_yield_factor
    higgs_vev_broken_convention = 174.0
    heaviest_neutrino_gev = 49.986e-12
    minimum_m1 = (
        required_epsilon
        * 16.0
        * math.pi
        * higgs_vev_broken_convention**2
        / (3.0 * heaviest_neutrino_gev)
    )
    minimum_m1_explicit_yield = minimum_m1 / equilibrium_yield_factor
    return RejectLoop(
        loop_id="baryogenesis",
        source_chapters=("08_바리온_비대칭.md",),
        parent_claim="the real CP-even portal and CKM J alone generate the observed baryon asymmetry",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "CP-even wall source",
                "REJECT",
                "multiply CKM J by an uncomputed wall coefficient",
                (),
                "require a CP-odd invariant and transport source from the action",
                {"new_CP_odd_invariant_present": False},
                "the current portal supplies no new CP source",
            ),
            Iteration(
                1,
                "type-I thermal leptogenesis capacity check",
                "BRIDGE_PASS",
                "reuse a heavy-neutrino completion with complex Yukawas",
                (
                    "heavy Majorana neutrinos",
                    "complex Yukawa matrix",
                    "total efficiency kappa_total=0.1 including equilibrium-yield normalization",
                    "hierarchical Davidson-Ibarra regime",
                ),
                "under the stated kappa_total convention, require the CP asymmetry and DI upper bound to overlap",
                {
                    "target_Y_B": target_yb,
                    "required_epsilon_1": required_epsilon,
                    "minimum_M1_GeV": minimum_m1,
                    "sphaleron_conversion": sphaleron_conversion,
                    "efficiency_convention": "kappa_total absorbs the relativistic equilibrium-yield factor",
                    "equilibrium_yield_factor_without_1_over_gstar": equilibrium_yield_factor,
                    "required_epsilon_if_kappa_excludes_equilibrium_yield": required_epsilon_explicit_yield,
                    "minimum_M1_if_kappa_excludes_equilibrium_yield_GeV": minimum_m1_explicit_yield,
                },
                "the normalization convention, efficiency, and flavor structure are supplied; this is not a universal bound or a Boltzmann solution",
            ),
            Iteration(
                2,
                "flavored Boltzmann or density-matrix evolution",
                "NOT_REACHED",
                "evolve production, washout, spectator effects, and reheating",
                ("initial abundance", "temperature history", "full Yukawa matrix"),
                "obtain Y_B without fitting it and pass neutrino plus cosmological constraints",
                {"transport_solution_present": False},
                "the Bridge-level capacity check does not compute the asymmetry",
            ),
        ),
        maximum_supported_stage="LEPTOGENESIS_CAPACITY_BRIDGE_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="solve the flavored transport system with a neutrino sector fixed independently",
    )


def fermion_loop() -> RejectLoop:
    masses_mev = (0.51099895, 105.6583755, 1776.86)
    koide_value = sum(masses_mev) / (
        sum(math.sqrt(mass) for mass in masses_mev) ** 2
    )
    return RejectLoop(
        loop_id="fermion_spectrum",
        source_chapters=("09_페르미온_질량.md",),
        parent_claim="mixed-scale mass ratios and a C3 Gaussian action derive the fermion spectrum with no assumptions",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "mixed-scale alpha_s power fit",
                "REJECT",
                "compare masses defined in incompatible schemes and scales",
                (),
                "run every mass to one scheme and scale before forming ratios",
                {"common_scale_used": False},
                "the quoted percent residuals are not physical mass-ratio tests",
            ),
            Iteration(
                1,
                "common-scale RG hygiene",
                "SELECTION_PASS",
                "supply a renormalization scheme, scale, and threshold prescription",
                ("common mu", "mass scheme", "RG and thresholds"),
                "form ratios only after common-scale evolution",
                {"comparison_well_defined": True, "spectrum_predicted": False},
                "this repairs the comparison but does not predict Yukawa couplings",
            ),
            Iteration(
                2,
                "charged-lepton Koide circle",
                "PHENOMENOLOGY_PASS",
                "treat k=1 in the phase parametrization as an explicit posit",
                ("Koide-circle posit k=1", "overall scale", "phase"),
                "require |Q_K-2/3|<1e-5 on the charged-lepton inputs",
                {
                    "Q_K": koide_value,
                    "two_thirds": 2.0 / 3.0,
                    "absolute_residual": abs(koide_value - 2.0 / 3.0),
                },
                "the posit and fitted phase are not a dynamical mass-generation mechanism",
            ),
            Iteration(
                3,
                "predictive Yukawa dynamics",
                "NOT_REACHED",
                "derive the Koide constraint and quark/lepton textures from one action",
                ("flavor symmetry", "vacuum alignment", "common-scale RG"),
                "predict held-out masses and mixing observables",
                {"held_out_prediction_present": False},
                "no common action fixes the required posit and flavor parameters",
            ),
        ),
        maximum_supported_stage="COMMON_SCALE_COMPARISON_PLUS_EMPIRICAL_KOIDE",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive a flavor action and test untouched masses or mixings",
    )


def axiom_bridge_loop() -> RejectLoop:
    return RejectLoop(
        loop_id="axiom_and_operator_bridges",
        source_chapters=("10_공리_정당화.md",),
        parent_claim="Z mixing alone derives an extra dimension, tree=DE, loop=DM, and a unique portal",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "semantic bridge chain",
                "REJECT",
                "rename gauge indices and loop orders as spacetime and cosmic components",
                (),
                "require fields, operators, dimensions, and matching coefficients",
                {"operator_basis_defined": False, "matching_defined": False},
                "the identifications are not equations in a common action",
            ),
            Iteration(
                1,
                "explicit EFT operator basis",
                "BRIDGE_PASS",
                "introduce independent Wilson coefficients for each proposed bridge",
                (
                    "covariant field content",
                    "operator basis",
                    "Wilson coefficients",
                    "matching scale",
                ),
                "check dimensions, symmetries, and renormalization closure",
                {
                    "operator_consistency_testable": True,
                    "coefficients_derived_from_CE": 0,
                },
                "an EFT can make the claims calculable but loses zero-parameter status",
            ),
            Iteration(
                2,
                "CE matching calculation",
                "NOT_REACHED",
                "integrate out specified CE degrees of freedom",
                ("UV CE action", "vacuum", "regularization", "matching prescription"),
                "derive coefficient signs and magnitudes without using target observables",
                {"matching_calculation_present": False},
                "no UV action is available for matching",
            ),
        ),
        maximum_supported_stage="CONSISTENT_EFT_SCAFFOLD_BRIDGE_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="supply the UV action and compute Wilson coefficients at a declared scale",
    )


def causality_loop() -> RejectLoop:
    return RejectLoop(
        loop_id="causality_and_no_signalling",
        source_chapters=("11_게이지_격자와_인과성.md",),
        parent_claim="determinants, lapse signs, and three damping factors prove no time travel, FTL, or teleportation",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "determinant/lapse/damping no-go",
                "REJECT",
                "infer global causal structure from local coordinates and attenuation",
                (),
                "state invariant causal assumptions and a global theorem",
                {"global_causal_assumptions_present": False},
                "local invertibility and small amplitudes do not forbid global causal curves or signalling",
            ),
            Iteration(
                1,
                "standard Bridge-level no-go package",
                "BRIDGE_PASS",
                "replace the invalid proof by established structural assumptions",
                (
                    "global hyperbolicity",
                    "local Lorentz-invariant microcausality",
                    "CPTP quantum operations",
                ),
                "check no closed causal curve, spacelike commutator zero, and no-signalling marginals",
                {
                    "closed_causal_curves_allowed": False,
                    "spacelike_signal_allowed": False,
                    "entanglement_only_signal_allowed": False,
                    "classical_channel_required_for_quantum_teleportation": True,
                },
                "the no-go results follow from supplied standard assumptions, not from CE damping",
            ),
            Iteration(
                2,
                "derive causal axioms from CE",
                "NOT_REACHED",
                "obtain the Lorentzian cone, global hyperbolicity, and microcausal algebra from the CE action",
                ("Lorentzian action", "constraint algebra", "quantization"),
                "prove the causal properties for every allowed CE solution",
                {"ce_causal_theorem_present": False},
                "the current manuscript has no such derivation",
            ),
        ),
        maximum_supported_stage="STANDARD_BRIDGE_CAUSALITY_THEOREMS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the causal cone and quantum observable algebra from CE",
    )


def transition_loop(core: dict[str, float]) -> RejectLoop:
    flow = relaxation_flow(core["d_eff"])
    return RejectLoop(
        loop_id="transition_dynamics",
        source_chapters=("12_전이구간.md", "14_자기재귀성_대칭.md"),
        parent_claim="fixed-point iteration count or Lambert branch switching is physical time and inflation",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "algorithmic iteration as time",
                "REJECT",
                "interpret a numerical solver index as a physical clock",
                (),
                "supply an action or differential equation with a dimensional rate",
                {"physical_rate_present": False},
                "the same fixed point can be solved by algorithms with different iteration counts",
            ),
            Iteration(
                1,
                "continuous relaxation ansatz",
                "SELECTION_PASS",
                "postulate dx/dtau=F_D(x)-x and integrate from x0<1",
                ("relaxation ODE", "initial displacement", "tau=Gamma t"),
                "converge to the nontrivial fixed point with a positive local relaxation rate",
                flow,
                "tau is dimensionless until Gamma is derived and the flow has no gravity or perturbation sector",
            ),
            Iteration(
                2,
                "inflationary spacetime completion",
                "NOT_REACHED",
                "couple the order parameter to FRW gravity and quantum perturbations",
                ("kinetic normalization", "potential", "Gamma or Hamiltonian", "initial state"),
                "produce accelerated expansion, graceful exit, and primordial spectra",
                {"friedmann_solution_present": False, "mode_solution_present": False},
                "a relaxation model is not automatically inflation",
            ),
        ),
        maximum_supported_stage="DIMENSIONLESS_RELAXATION_DYNAMICS_SELECTION_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the dimensional rate and stress tensor, then solve FRW plus perturbations",
    )


def precision_constant_loop() -> RejectLoop:
    alpha_target = 137.035999177
    alpha_uncertainty = 0.000000021
    mass_target = 1836.152673426
    mass_uncertainty = 0.000000032
    alpha_coefficients, alpha_value, alpha_error, alpha_trials = finite_expression_search(
        alpha_target, (3, 2, 1, 0)
    )
    mass_coefficients, mass_value, mass_error, mass_trials = finite_expression_search(
        mass_target, (5, 3, 1, 0)
    )
    sparse_powers = (6, 5, 4, 3, 2, 1, 0)
    (
        sparse_alpha_coefficients,
        sparse_alpha_value,
        sparse_alpha_error,
        sparse_alpha_trials,
    ) = finite_sparse_expression_search(alpha_target, sparse_powers)
    (
        sparse_mass_coefficients,
        sparse_mass_value,
        sparse_mass_error,
        sparse_mass_trials,
    ) = finite_sparse_expression_search(mass_target, sparse_powers)
    return RejectLoop(
        loop_id="precision_constant_formula_search",
        source_chapters=("13_위상공간.md",),
        parent_claim="small pi-polynomial expressions are exact zero-input predictions of precision constants",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "legacy alpha and proton/electron formulas",
                "REJECT",
                "compare the stated formulas to precision measurements",
                (),
                "require absolute pull below five sigma",
                {
                    "alpha_formula": 4.0 * math.pi**3 + math.pi**2 + math.pi,
                    "alpha_pull_sigma": (
                        4.0 * math.pi**3 + math.pi**2 + math.pi - alpha_target
                    )
                    / alpha_uncertainty,
                    "mass_formula": 6.0 * math.pi**5,
                    "mass_pull_sigma": (
                        6.0 * math.pi**5 - mass_target
                    )
                    / mass_uncertainty,
                },
                "both exact claims fail by many standard deviations",
            ),
            Iteration(
                1,
                "bounded integer pi-expression search",
                "NEGATIVE_CONTROL",
                "search a preregistered coefficient grammar instead of hand-selecting one formula",
                ("grammar powers and coefficient bound", "target constants"),
                "apply precision, trials-factor, complexity, and untouched-holdout gates",
                {
                    "coefficient_bound": 10,
                    "alpha_best_coefficients": alpha_coefficients,
                    "alpha_best_value": alpha_value,
                    "alpha_best_pull_sigma": alpha_error / alpha_uncertainty,
                    "alpha_trials": alpha_trials,
                    "mass_best_coefficients": mass_coefficients,
                    "mass_best_value": mass_value,
                    "mass_best_pull_sigma": mass_error / mass_uncertainty,
                    "mass_trials": mass_trials,
                    "sparse_powers": sparse_powers,
                    "sparse_coefficient_bound": 6,
                    "sparse_max_nonzero_terms": 3,
                    "sparse_alpha_best_coefficients": sparse_alpha_coefficients,
                    "sparse_alpha_best_value": sparse_alpha_value,
                    "sparse_alpha_best_pull_sigma": sparse_alpha_error
                    / alpha_uncertainty,
                    "sparse_alpha_trials": sparse_alpha_trials,
                    "sparse_mass_best_coefficients": sparse_mass_coefficients,
                    "sparse_mass_best_value": sparse_mass_value,
                    "sparse_mass_best_pull_sigma": sparse_mass_error
                    / mass_uncertainty,
                    "sparse_mass_trials": sparse_mass_trials,
                    "untouched_holdout_count": 0,
                },
                "the best in-grammar fits still fail precision and there is no held-out observable",
            ),
            Iteration(
                2,
                "matched EFT constants",
                "BRIDGE_PASS",
                "add one counterterm per measured constant",
                ("two fitted counterterms", "renormalization scheme and scale"),
                "reproduce both inputs while counting predictive degrees of freedom",
                {
                    "observables": 2,
                    "fitted_counterterms": 2,
                    "predictive_degrees_of_freedom": 0,
                },
                "matching is consistent bookkeeping, not a prediction",
            ),
        ),
        maximum_supported_stage="EFT_MATCHING_WITH_ZERO_PREDICTIVE_DEGREES",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive a common formula before assigning an untouched precision holdout",
    )


def recursive_symmetry_loop() -> RejectLoop:
    return RejectLoop(
        loop_id="recursive_residual_to_field",
        source_chapters=("14_자기재귀성_대칭.md",),
        parent_claim="d=0 or a bootstrap residual automatically creates a real Z2 scalar with lambdaHP=delta^2",
        parent_status="REJECT",
        iterations=(
            Iteration(
                0,
                "residual-to-field semantic chain",
                "REJECT",
                "infer field type, symmetry, and coupling value without an action",
                (),
                "write a local action and check every symmetry implication separately",
                {"local_action_present": False},
                "SO(0), reality, Z2, and a numerical coupling are logically independent",
            ),
            Iteration(
                1,
                "explicit Landau-Ginzburg Z2 scalar",
                "BRIDGE_PASS",
                "postulate a real scalar and independent internal Z2 symmetry",
                (
                    "real scalar field",
                    "Z2 symmetry",
                    "m2>0",
                    "lambda4>0",
                    "independent lambdaHP",
                ),
                "verify even potential, stable phi=0 vacuum, and allowed H-dagger-H phi^2 portal",
                {
                    "odd_operators_forbidden": True,
                    "stable_zero_vev": True,
                    "portal_operator_allowed": True,
                    "lambda_HP_fixed_by_Z2": False,
                },
                "the consistent action is supplied and does not derive lambdaHP=delta^2",
            ),
            Iteration(
                2,
                "bootstrap residual matching",
                "NOT_REACHED",
                "derive the scalar two-point function and Wilson coefficients from the recursive system",
                ("residual-field map", "kinetic term", "normalization", "matching scale"),
                "match poles, residues, symmetries, and lambdaHP without target-data fitting",
                {"spectral_match_present": False},
                "the algebraic residual has no demonstrated field-theory spectrum",
            ),
        ),
        maximum_supported_stage="CONSISTENT_Z2_SCALAR_ACTION_BRIDGE_PASS",
        ce_specific_physical_claim_closed=False,
        next_required_gate="derive the residual-to-field spectral map and portal matching",
    )


CHAPTER_LOOP_ROUTES: dict[str, tuple[str, ...]] = {
    "01_차원의_유일성.md": ("dimension_hodge",),
    "02_에스컬레이터.md": ("gauge_structure_and_couplings",),
    "03_자유매개변수.md": ("bootstrap_cosmology",),
    "04_해결한_난제.md": (
        "bootstrap_cosmology",
        "horizon_and_claim_accounting",
        "light_scalar_precision_observables",
    ),
    "05_인플레이션.md": (
        "finite_xi_inflation",
        "scalar_portal_benchmark",
    ),
    "06_강한_CP.md": ("strong_cp",),
    "07_중성미자_질량.md": (
        "neutrino_mass",
        "scalar_portal_benchmark",
    ),
    "08_바리온_비대칭.md": (
        "baryogenesis",
        "scalar_portal_benchmark",
    ),
    "09_페르미온_질량.md": ("fermion_spectrum",),
    "10_공리_정당화.md": (
        "gauge_structure_and_couplings",
        "axiom_and_operator_bridges",
    ),
    "11_게이지_격자와_인과성.md": ("causality_and_no_signalling",),
    "12_전이구간.md": (
        "transition_dynamics",
        "finite_xi_inflation",
        "scalar_portal_benchmark",
    ),
    "13_위상공간.md": ("precision_constant_formula_search",),
    "14_자기재귀성_대칭.md": (
        "transition_dynamics",
        "recursive_residual_to_field",
    ),
}


# These are occurrence-level micro-gates, not chapter fallbacks.  Each gate
# states why the original claim is killed, which extra structure could define a
# Selection/Bridge descendant, and the domain-specific terminal test still required.
SEMANTIC_NEXT_GATE_BY_LOOP: dict[str, str] = {
    "dimension_hodge": "Prove the typed theorem with all hypotheses and explicitly eliminate every lower/alternative-dimensional counterexample.",
    "gauge_structure_and_couplings": "Perform action-level representation matching and RG evolution, then compare normalized gauge observables at one declared scale.",
    "bootstrap_cosmology": "Freeze the forward model and nuisance parameters, then evaluate the released data vector with its full covariance and evidence penalty.",
    "horizon_and_claim_accounting": "Recompute the dimensionally consistent horizon quantity and audit it against a frozen claim/status ledger.",
    "finite_xi_inflation": "Integrate the finite-xi background and perturbations through reheating and test a joint primordial likelihood without refitting the target.",
    "scalar_portal_benchmark": "Run collider, relic-density, direct/indirect-detection, vacuum-stability, and RG gates at one identical parameter point.",
    "light_scalar_precision_observables": "Evaluate the correctly normalized observable and current covariance at parameters fixed independently of that residual.",
    "strong_cp": "Derive the theta-dependent vacuum and hadronic matching, then pass the neutron-EDM likelihood with all new fields constrained.",
    "neutrino_mass": "Diagonalize the complete protected/LNV mass sector and pass held-out masses, PMNS observables, phases, and RG consistency.",
    "baryogenesis": "Solve the normalized Boltzmann or density-matrix transport system and reproduce Y_B without choosing coefficients from Y_B.",
    "fermion_spectrum": "Run all masses to one scheme/scale and predict an unused mass or mixing observable from the frozen flavor action.",
    "axiom_and_operator_bridges": "Exhibit an explicit field/operator map and verify dimensions, symmetries, matching coefficients, and a shared observable on both sides.",
    "causality_and_no_signalling": "Construct the operational channel and verify microcausality, global hyperbolicity, and zero spacelike signalling capacity.",
    "transition_dynamics": "Specify the continuous evolution and clock, then prove existence, stability, branch selection, and parameter-robust convergence.",
    "precision_constant_formula_search": "Freeze the derivation or finite grammar, include the full trials factor, and predict an unused precision quantity within uncertainty.",
    "recursive_residual_to_field": "Construct the residual-to-operator map and verify locality, poles, residues, symmetry action, and spectral positivity.",
}


SEMANTIC_NEXT_GATE_OVERRIDES: dict[str, str] = {
    "hodge_form_degree": "Prove the exact Hodge/form-degree theorem being claimed and keep the d=3 binomial root distinct from the d=4 self-dual two-form condition.",
    "lower_dim_counterexamples": "Evaluate the explicit 1D, 2D, and d>=4 Hamiltonian counterexamples; a single valid counterexample kills each universal absence claim.",
    "kernel_curvature_type": "Construct a dimensionally and tensorially typed trace/spectral map from the Hessian kernel to R[g], or leave K_gamma=R[g] rejected.",
    "gravity_quantization_eft": "Compute a gauge-fixed low-energy quantum-gravity EFT amplitude with its counterterms; make no UV-completion claim beyond its power-counting range.",
    "instanton_charge_double_count": "Write S_inst(Q) once in terms of |Q| and verify that no additional outer |Q| converts the intended suppression into an unsupported Q^2 factor.",
    "generation_count_category": "Derive an anomaly-free chiral representation theorem fixing exactly three copies and test that result independently of the observed family count.",
    "singlet_ew_portal": "Separate the absent direct W-singlet gauge vertex from indirect Higgs-portal effects and compute the latter from the declared portal action.",
    "gauge_coupling_bridge": "Provide a UV gauge embedding, generator normalization, boundary matching, and RG evolution; exponential factorization alone must not alter g_s into cube-root couplings.",
    "cosmology_zero_parameter_accounting": "Freeze and enumerate every continuous, discrete, external, calibrated, and fitted input; the zero-parameter label passes only if the ledger is empty.",
    "gauge_sum_arithmetic": "Substitute the three stated couplings and verify that their sum is 3/(2*pi), not 1/(2*pi), before any unification interpretation.",
    "perturbative_order_diagram": "Compute the complete NLO/CKM amplitude with diagrams, counterterms, scheme/scale variation, and an observable not used to choose the truncation.",
    "light_scalar_precision": "Derive the zero/small contribution from exact symmetries and the complete operator basis, then evaluate every allowed precision contribution with current covariance.",
    "portal_vacuum_energy": "Renormalize the effective action in one declared prescription, fix the vacuum counterterm independently, and test the resulting Omega_Lambda plus radiative stability.",
    "portal_parameter_package_consistency": "At one identical parameter point, verify unique pole mass, independently derived lambda_HP, and the exact invisible-width value before any package claim.",
    "transition_monotonicity": "Evaluate both one-sided derivatives at D=1 and prove strict monotonicity on every interval where that property is claimed.",
    "contraction_rate": "Solve x_-(D) anew at the tested D and verify D*x_-(D), rather than reusing a root from another parameter value.",
    "flow_potential_sign": "Differentiate the displayed V(x) explicitly and verify that -dV/dx equals the stated ODE at generic x and at every fixed point.",
    "electroweak_energy_formula": "Derive F and the exponent independently, compute the finite-temperature effective action, and compare the crossover observable rather than the zero-temperature VEV.",
    "koide_precision": "Recompute Q_K from the exact declared mass inputs and uncertainties; distinguish a rounding discrepancy from a statistically significant exact identity.",
    "koide_universality": "Evaluate the claimed Q_K=2/N rule at N=1 and N=2; it may advance only if those counterexamples are removed by derived hypotheses.",
    "inconsistent_cosine_root": "Substitute every quoted cosine/root value into the displayed equation and retain only algebraically valid branches before any physical interpretation.",
    "gaussian_normalization_invariance": "Vary edge-weight and kappa normalization while holding graph topology fixed; mu^2=3 can pass only if it is invariant or normalization is independently derived.",
    "top_qcd_formula": "Substitute the declared alpha_s and mass convention into the displayed pole/running conversion and verify its sign and coefficient before any fit.",
    "qcd_koide_quantification": "Run every quark mass to one MSbar scale with covariance, then recompute Q_u and Q_d before assigning any QCD-breaking percentage.",
    "fermion_adjacent_ratio_bridge": "Prove the adjacent-ratio identity and a typed d-to-generation map, then predict an unused common-scale mass ratio.",
    "gaussian_measure_truncation": "Normalize the positive-orthant truncated measure and derive why an ensemble distribution represents a vacuum state before extracting masses.",
    "gauge_phase_space_measure": "Construct the SU(N) gauge-orbit quotient and normalized Haar/Faddeev-Popov measure, then compare it explicitly with the polydisc measure.",
    "lambert_d0_identity": "At fixed D, verify both Lambert branches and separately prove any D-to-d map; the x=1 branch alone cannot pass the d=0 claim.",
    "f0_constant_not_identity": "Substitute multiple x values into F_0(x)=1 and x->x; the maps must agree on the stated domain before identity language is allowed.",
    "residual_field_reality": "Give an internal conjugation/reality condition and a local kinetic operator; SO(0) alone must play no role in that proof.",
    "z2_not_automatic": "Exhibit the independent Z2 action on fields and verify action, measure, and vacuum invariance without using d=0 as a premise.",
    "branch_cosmogenesis": "Solve a regular time-dependent branch transition from specified initial/boundary data and derive cosmological observables; algebraic relabeling is insufficient.",
    "determinant_teleportation_proof": "Define an operational quantum channel and show that the determinant expression implies the relevant LOCC/no-signalling bound; otherwise the proof remains rejected.",
    "lapse_zero_freeze": "Analyze the full ADM metric including shift and foliation regularity; N=0 alone may pass only if it implies a gauge-invariant physical freeze observable.",
    "triple_damping_ftl_proof": "Evaluate the proposed damping ansatz at finite v>c and derive every coefficient from an action; a nonzero probability cannot establish a no-go theorem.",
    "gauge_causality_safeguard": "Test microcausality with and without each gauge sector in an explicit local action; causal behavior cannot be credited to an unproved sector-factorization safeguard.",
    "residual_matter_identity": "For a dark-matter branch, match a stable particle spectrum and abundance; for dark energy, derive a renormalized stress tensor, equation of state, and cosmological evolution.",
    "residual_field_chain": "Verify each arrow separately: residual-to-operator spectral matching, independent exact Z2 invariance, and UV/EFT matching of lambda_HP followed by portal likelihoods.",
}


def _semantic_gate(
    gate_id: str,
    loop_ids: tuple[str, ...],
    kill_gate: str,
    alternate_route: str,
) -> SemanticGateDefinition:
    return SemanticGateDefinition(
        gate_id=gate_id,
        loop_ids=loop_ids,
        kill_gate=kill_gate,
        alternate_route=alternate_route,
        next_gate=SEMANTIC_NEXT_GATE_OVERRIDES.get(
            gate_id,
            " ".join(
                dict.fromkeys(
                    SEMANTIC_NEXT_GATE_BY_LOOP[loop_id]
                    for loop_id in loop_ids
                )
            ),
        ),
    )


_SEMANTIC_GATE_SPECS: dict[
    str, tuple[tuple[str, ...], str, str]
] = {
    # Chapters 01--03: type, dimension, gauge, and cosmology bridges.
    "hodge_form_degree": (
        ("dimension_hodge",),
        "Hodge duality sends Lambda^2 to Lambda^(d-2): literal same-degree F=*F selects d=4, while dim(Lambda^2)=dim(Lambda^1) has the separate positive root d=3.",
        "State which typed equation is intended and add independent axioms before any uniqueness claim.",
    ),
    "lower_dim_counterexamples": (
        ("dimension_hodge",),
        "Explicit 1D scattering, 2D stable-orbit, and d>=4 bound-state models contradict the claimed universal absences.",
        "Specify the Hamiltonian, potential, boundary conditions, and extra hypotheses that exclude each counterexample.",
    ),
    "cp_source_nonuniqueness": (
        ("dimension_hodge", "baryogenesis"),
        "A dimension count neither creates nor uniquely fixes a CP-odd invariant.",
        "Supply a concrete action and compute its independent CP-odd invariants.",
    ),
    "state_space_bridge": (
        ("dimension_hodge", "axiom_and_operator_bridges"),
        "The compared symbols live in different typed state spaces and no map is defined.",
        "Define a typed, invertible map with units, domain, codomain, and observable preservation.",
    ),
    "kernel_curvature_type": (
        ("axiom_and_operator_bridges", "gauge_structure_and_couplings"),
        "A Hessian or bilocal kernel and the spacetime scalar curvature have different operator/tensor types and units; K_gamma=R[g] needs a trace or spectral map.",
        "Specify the action, background, boundary conditions, and an explicit contraction or spectral relation.",
    ),
    "gravity_quantization_eft": (
        ("axiom_and_operator_bridges", "gauge_structure_and_couplings"),
        "Gravity being unlike a Yang-Mills force does not make it unquantizable: it is a consistent low-energy quantum EFT, while UV completion remains open.",
        "Specify background, gauge fixing, ghosts, operator basis, regulator, counterterms, and EFT power counting.",
    ),
    "generation_count_category": (
        ("dimension_hodge", "gauge_structure_and_couplings"),
        "A spacetime or group dimension is not a proof of the number of fermion generations.",
        "Provide a chiral representation and anomaly-free flavor construction fixing the multiplicity.",
    ),
    "color_singlet_representation": (
        ("gauge_structure_and_couplings",),
        "A color singlet cannot be relabeled as the nontrivial representation used in the claim.",
        "Specify the actual group representation and recompute all invariant contractions.",
    ),
    "gauge_sum_underdetermined": (
        ("gauge_structure_and_couplings",),
        "Adding gauge-factor dimensions or couplings does not derive a unified group or coupling.",
        "Choose an embedding, normalization, threshold spectrum, and run the RG equations.",
    ),
    "gauge_sum_arithmetic": (
        ("gauge_structure_and_couplings",),
        "If each of three couplings equals 1/(2*pi), their sum is 3/(2*pi), contradicting the claimed total 1/(2*pi).",
        "Correct the arithmetic first, then supply an independently normalized embedding and RG/matching calculation.",
    ),
    "cosmology_posthoc_ratio": (
        ("bootstrap_cosmology",),
        "A ratio selected after seeing cosmological data is not an out-of-sample prediction.",
        "Freeze the formula and nuisance model before evaluating an untouched dataset.",
    ),
    "perturbative_order_diagram": (
        ("axiom_and_operator_bridges",),
        "A loop-order label or diagram count is not an NLO amplitude or CKM calculation.",
        "Supply the operator basis, diagrams, counterterms, scheme, and flavor matrices.",
    ),
    "cosmology_zero_parameter_accounting": (
        ("bootstrap_cosmology",),
        "Supplying c_Sigma, alpha_s, d, branch choices, and external calibration contradicts the claim of zero inputs or zero free parameters.",
        "Publish a complete parameter/input ledger and distinguish fitted, external, discrete-choice, and derived quantities.",
    ),
    "cosmology_likelihood_calibration": (
        ("bootstrap_cosmology",),
        "A calibrated parameter identity is not a likelihood-level cosmological prediction.",
        "Freeze calibration, covariance, sound horizon, and selection functions before testing data.",
    ),
    # Chapters 04--05: headline claims, inflation, and portal mutations.
    "gauge_unification_claim": (
        ("gauge_structure_and_couplings",),
        "Numerical proximity at one scale is not gauge unification.",
        "Specify a unified group and thresholds, then run all normalized couplings to one crossing.",
    ),
    "solved_claim_accounting": (
        ("horizon_and_claim_accounting",),
        "Counting fitted, assumed, open, or rejected statements as solved is category error.",
        "Use a frozen claim ledger and count only independently verified derived claims.",
    ),
    "horizon_entropy_equation": (
        ("horizon_and_claim_accounting",),
        "The quoted horizon/entropy equality lacks a derivation and consistent units.",
        "Derive both sides in one horizon convention with propagated parameter uncertainty.",
    ),
    "cosmology_data_model": (
        ("bootstrap_cosmology",),
        "A DESI best-fit coordinate is not a standalone observation or model-independent constant.",
        "Use the released data vector, covariance, model, nuisance parameters, and likelihood.",
    ),
    "light_scalar_precision": (
        ("light_scalar_precision_observables",),
        "Declaring CE BSM=0, even when compatible with a null residual, is not a derivation or a solution from the CE action.",
        "Impose the exact symmetry on a complete operator basis and compute every allowed contribution before testing current precision data.",
    ),
    "electromagnetic_coupling_identity": (
        ("gauge_structure_and_couplings",),
        "The proposed alpha_em, weak-angle, and alpha_s identity fails normalization and RG running.",
        "Choose one scheme and scale and derive the relation from a specified gauge embedding.",
    ),
    "cosmology_version_mixing": (
        ("bootstrap_cosmology",),
        "Parameters from incompatible dataset/model versions cannot form one prediction vector.",
        "Select one versioned likelihood and refit all correlated parameters together.",
    ),
    "neutrino_operator_claim": (
        ("neutrino_mass",),
        "The stated field content contains no lepton-number-violating mass operator.",
        "Add either a protected Dirac sector or an explicit Delta-L=2 UV completion.",
    ),
    "inflation_initial_condition": (
        ("finite_xi_inflation",),
        "The stable phi=0 vacuum is not the large-field inflationary initial state.",
        "Add and normalize an initial-condition measure in the Einstein-frame dynamics.",
    ),
    "finite_xi_approximation": (
        ("finite_xi_inflation",),
        "The large-xi asymptotic formula is uncontrolled at xi=0.49.",
        "Solve the finite-xi canonical field and slow-roll equations without the asymptotic limit.",
    ),
    "iteration_time_bridge": (
        ("transition_dynamics", "finite_xi_inflation"),
        "A fixed-point iteration index has no physical time or e-fold normalization.",
        "Provide a continuous equation of motion, kinetic term, clock, and initial condition.",
    ),
    "desi_xi_postfit": (
        ("finite_xi_inflation", "bootstrap_cosmology"),
        "Mapping a DESI central value to xi after the fit is not an independent derivation.",
        "Pre-register the map and test xi with a joint primordial/late-time likelihood.",
    ),
    "z2_single_decay": (
        ("scalar_portal_benchmark",),
        "An exact unbroken Z2 forbids decay of a single odd scalar.",
        "Either keep the scalar stable or specify and constrain an explicit Z2-breaking operator.",
    ),
    "portal_benchmark": (
        ("scalar_portal_benchmark",),
        "The legacy m0=0 portal point fails the Higgs invisible-width gate.",
        "Free m0 and lambda_HP, then run collider, relic, direct, indirect, and stability likelihoods jointly.",
    ),
    "reheating_baryon_identity": (
        ("baryogenesis",),
        "Reheating energy density is not a net baryon number without CP violation and transport.",
        "Supply a CP-odd source, number violation, nonequilibrium history, and washout evolution.",
    ),
    # Chapter 06: strong CP.
    "strong_cp_field_mechanism": (
        ("strong_cp",),
        "The CP-even singlet portal supplies no theta-dependent relaxation degree of freedom.",
        "Add a symmetry-protected axion/PQ or another explicit theta-relaxation mechanism.",
    ),
    "strong_cp_singlet_symmetry": (
        ("strong_cp",),
        "Gauge singlet status alone is not a chiral, parity, CP, or PQ protection symmetry.",
        "Specify the protecting symmetry, charges, anomalies, and breaking pattern.",
    ),
    "strong_cp_even_not_axion": (
        ("strong_cp",),
        "A CP-even scalar is not an axionic shift field coupled to G dual-G.",
        "Introduce the pseudoscalar shift symmetry and anomaly coefficient explicitly.",
    ),
    "instanton_measure": (
        ("strong_cp",),
        "The proposed topological weight is undefined or not derived from the path-integral measure.",
        "Fix the Q-sector action, determinant, zero modes, normalization, and measure once.",
    ),
    "instanton_charge_double_count": (
        ("strong_cp",),
        "S_inst(Q) already contains the |Q| dependence, so multiplying by another outer |Q| double-counts charge and produces an unsupported Q^2 suppression.",
        "Choose one sector-action convention and derive the Q dependence exactly once from the instanton measure.",
    ),
    "strong_cp_bootstrap_bound": (
        ("strong_cp",),
        "A contraction residual is not a physical bound on theta without a dynamical map.",
        "Derive theta dependence from the action and compare the resulting neutron EDM likelihood.",
    ),
    "neutron_edm_prediction": (
        ("strong_cp",),
        "Declaring theta or the neutron EDM zero is not a prediction of the current portal.",
        "Compute theta relaxation and hadronic matching with uncertainties from an added mechanism.",
    ),
    # Chapter 07: neutrino operator, flavor, and portal consistency.
    "neutrino_lnv_operator": (
        ("neutrino_mass",),
        "No Delta-L=2 coefficient or diagram exists in the stated exact-Z2 real-singlet action.",
        "Add explicit Dirac protection or a complete LNV field/operator sector.",
    ),
    "scotogenic_field_content": (
        ("neutrino_mass",),
        "A real singlet Higgs portal is not the inert-doublet/fermion scotogenic model.",
        "Add the required multiplets, charges, lambda5 term, and loop matching.",
    ),
    "neutrino_flavor_matrix": (
        ("neutrino_mass",),
        "One real coefficient cannot encode a complex symmetric three-flavor mass matrix.",
        "Supply a full flavor texture and test masses, mixings, phases, and running together.",
    ),
    "neutrino_dimensionality": (
        ("neutrino_mass",),
        "The displayed neutrino mass expression has inconsistent mass dimension.",
        "Write a dimensionally closed operator matching formula before numerical evaluation.",
    ),
    "neutrino_loop_factor": (
        ("neutrino_mass",),
        "The asserted loop/phase factor has no diagrammatic or symmetry-factor derivation.",
        "Evaluate the complete regulated loop integral with declared conventions.",
    ),
    "neutrino_suppression_chain": (
        ("neutrino_mass",),
        "Cosmological fractions and correlation lengths do not automatically suppress a microscopic amplitude.",
        "Derive the suppression from propagators and vertices in one EFT.",
    ),
    "generation_adjoint": (
        ("neutrino_mass", "gauge_structure_and_couplings"),
        "SM generations are not an SU(3) adjoint representation in the stated Lagrangian.",
        "Define a flavor gauge group, representations, anomaly cancellation, and breaking.",
    ),
    "neutrino_fractional_power": (
        ("neutrino_mass",),
        "The fractional power lacks a diagrammatic scaling or RG law.",
        "Derive its exponent from an operator dimension or anomalous dimension.",
    ),
    "weinberg_qcd_running": (
        ("neutrino_mass",),
        "A color-singlet Weinberg operator cannot use the asserted alpha_s factor without matching.",
        "Compute the operator anomalous dimensions and threshold evolution.",
    ),
    "neutrino_posthoc_exponent": (
        ("neutrino_mass", "precision_constant_formula_search"),
        "An exponent selected from the target spectrum is an unconstrained fit, not a prediction.",
        "Freeze the exponent from independent dynamics and predict an unused flavor observable.",
    ),
    "neutrino_mee_consistency": (
        ("neutrino_mass",),
        "The quoted effective Majorana mass violates the triangle bounds of the listed spectrum.",
        "Use one mass ordering and PMNS phase convention and recompute m_ee.",
    ),
    "neutrino_ordering_state": (
        ("neutrino_mass",),
        "Flavor labels are not mass eigenstate labels and cannot establish ordering.",
        "Diagonalize one mass matrix and identify ordering from its eigenvalues.",
    ),
    "portal_collider": (
        ("scalar_portal_benchmark",),
        "The exact-Z2 legacy portal point is excluded by invisible Higgs decay.",
        "Use independent portal parameters and rerun the full correlated constraint set.",
    ),
    "portal_mass_identity": (
        ("scalar_portal_benchmark", "recursive_residual_to_field"),
        "One elementary field cannot have two physical pole masses in one vacuum.",
        "Introduce distinct fields or derive a spectral function with separately identified poles.",
    ),
    # Chapter 08: baryogenesis and transport.
    "baryon_cp_source": (
        ("baryogenesis",),
        "Generation counting, CKM J, or a CP-even wall does not supply the claimed new CP source.",
        "Compute a nonzero CP-odd invariant and wall source from a complete action.",
    ),
    "baryon_normalization": (
        ("baryogenesis",),
        "eta_B and Y_B differ by the entropy-to-photon normalization and cannot be interchanged.",
        "Choose one convention and propagate the 7.04 conversion consistently.",
    ),
    "portal_ewpt_extrapolation": (
        ("scalar_portal_benchmark", "baryogenesis"),
        "Descendant EWPT predictions cannot survive an already excluded benchmark unchanged.",
        "Refit a viable portal point and recompute the thermal potential and transport.",
    ),
    "baryon_wall_factor": (
        ("baryogenesis",),
        "The wall factor range is asserted without a wall-profile calculation or citation.",
        "Solve the finite-temperature bounce and transport source for fixed parameters.",
    ),
    "baryon_transport_fit": (
        ("baryogenesis",),
        "A coefficient chosen to span the observed asymmetry is a fit, not transport dynamics.",
        "Compute diffusion, reaction, and washout rates from the Lagrangian.",
    ),
    "baryon_arithmetic": (
        ("baryogenesis",),
        "The stated baryogenesis range is inconsistent with its own factor ranges.",
        "Recompute the interval with one normalization and propagated bounds.",
    ),
    "baryon_ansatz": (
        ("baryogenesis",),
        "A multiplicative ansatz omitting source, diffusion, and washout is not a mechanism.",
        "Solve Boltzmann or density-matrix transport with a derived CP source.",
    ),
    "density_conversion_not_generation": (
        ("baryogenesis",),
        "Converting present densities does not explain primordial asymmetry generation.",
        "Evolve a microscopic number-generating process through freeze-out and washout.",
    ),
    "baryon_dimension_formula": (
        ("baryogenesis",),
        "The proposed eta_B expression is dimensionful and cannot equal a dimensionless ratio.",
        "Restore all scales and derive a dimensionless abundance from number and entropy densities.",
    ),
    "leptogenesis_missing_sector": (
        ("baryogenesis", "neutrino_mass"),
        "Majorana language alone does not provide heavy-state production, CP decay, or LNV interactions.",
        "Specify the heavy spectrum, complex Yukawas, reheating, and transport equations.",
    ),
    # Chapter 09: fermion spectrum.
    "fermion_compound_summary": (
        ("fermion_spectrum", "axiom_and_operator_bridges", "precision_constant_formula_search"),
        "The summary combines dimensionally invalid A~sqrt(v), missing top/QCD matching, mixed mass schemes, fitted exponents, and unproved flavor dynamics.",
        "Split the claims, define dimensionful operators and common-scale matching, and pre-register every exponent before testing held-out flavor data.",
    ),
    "mixed_scale_quark": (
        ("fermion_spectrum",),
        "Masses from incompatible renormalization schemes and scales cannot form precision ratios.",
        "Run all masses through declared thresholds to one scheme and scale.",
    ),
    "posthoc_exponent": (
        ("fermion_spectrum", "precision_constant_formula_search"),
        "An exponent chosen from target masses is a fit with hidden freedom.",
        "Derive it independently and predict an unused mass or mixing observable.",
    ),
    "fermion_adjacent_ratio_bridge": (
        ("fermion_spectrum", "axiom_and_operator_bridges"),
        "The adjacent-ratio algebra does not establish the quoted relation, and identifying spacetime d with the generation count has no typed map.",
        "Correct the ratio identity and separately derive a representation/flavor map fixing N_gen.",
    ),
    "koide_precision": (
        ("fermion_spectrum",),
        "The displayed rounded mass inputs do not reproduce the stated Koide arithmetic exactly.",
        "Declare full-precision masses, scheme, scale, and uncertainty before quoting the residual.",
    ),
    "koide_universality": (
        ("fermion_spectrum",),
        "The proposed Q_K=2/N generalization fails its N=1 and N=2 counterexamples.",
        "Restrict the relation to an explicitly justified domain or derive hypotheses excluding those cases.",
    ),
    "gaussian_koide_posit": (
        ("fermion_spectrum",),
        "A Gaussian/C3 parametrization posits rather than derives the Koide constraint.",
        "Derive the vacuum alignment and measure from a flavor action.",
    ),
    "gaussian_normalization_invariance": (
        ("fermion_spectrum", "axiom_and_operator_bridges"),
        "Edge-weight or kappa rescaling changes the Gaussian spectrum, so mu^2=3 is not a normalization-invariant prediction of three nodes.",
        "Fix normalization from an independent action or prove the eigenvalue relation is invariant under every allowed rescaling.",
    ),
    "gaussian_measure_interactions": (
        ("fermion_spectrum", "axiom_and_operator_bridges"),
        "Having three nodes does not forbid nonlinear operators such as psi^4; graph cardinality is not an interaction no-go theorem.",
        "Enumerate the symmetry-allowed operator basis and derive which nonlinear couplings vanish from the action.",
    ),
    "gaussian_measure_truncation": (
        ("fermion_spectrum", "axiom_and_operator_bridges"),
        "Truncating a Gaussian to the positive orthant changes its normalization, and an ensemble measure is not automatically one vacuum configuration.",
        "Normalize the truncated measure and derive an ensemble-to-vacuum/state-selection map.",
    ),
    "inconsistent_cosine_root": (
        ("fermion_spectrum",),
        "The quoted cosine value and root do not satisfy the displayed algebraic equation.",
        "Recompute and substitute every root before applying any independent physical branch criterion.",
    ),
    "dimensional_A_sqrt_v": (
        ("fermion_spectrum",),
        "The A-sqrt(v) identification has inconsistent dimensions or normalization.",
        "Define field normalization and a dimensionally complete Yukawa operator.",
    ),
    "qcd_koide_quantification": (
        ("fermion_spectrum",),
        "Mixed-scale values Q_u about 0.849 and Q_d about 0.729 cannot quantify QCD breaking as 27% and 9%.",
        "Perform common-MSbar-scale RG and threshold evolution with the full mass covariance before computing percentages.",
    ),
    "top_mass_scheme": (
        ("fermion_spectrum",),
        "Pole, Monte-Carlo, and running top masses are not interchangeable.",
        "Choose a short-distance mass scheme and propagate conversion uncertainty.",
    ),
    "top_qcd_formula": (
        ("fermion_spectrum",),
        "The displayed top pole/running QCD correction has the wrong sign or coefficient and fails direct substitution.",
        "Use a declared perturbative conversion formula, order, alpha_s scale, and uncertainty.",
    ),
    "retrodiction": (
        ("fermion_spectrum",),
        "Reconstructing an input spectrum is retrodiction, not held-out prediction.",
        "Freeze the model on a training subset and predict untouched observables.",
    ),
    # Chapters 10--11: operator bridges and causality.
    "internal_index_dimension": (
        ("axiom_and_operator_bridges", "dimension_hodge"),
        "An internal gauge index is not a spacetime coordinate.",
        "Provide a metric, locality, propagator, and compactification map for any added dimension.",
    ),
    "singlet_ew_portal": (
        ("axiom_and_operator_bridges", "scalar_portal_benchmark"),
        "Gauge neutrality removes a direct W-singlet vertex but does not forbid indirect W interactions through |H|^2 phi^2 after electroweak breaking; it also does not fix that portal coefficient.",
        "Write the symmetry-allowed portal EFT and compute direct versus Higgs-mediated amplitudes separately.",
    ),
    "coherent_mixture_probability": (
        ("axiom_and_operator_bridges",),
        "A coherent amplitude and a classical probability mixture are different objects.",
        "Specify the density matrix, measurement, and decoherence map.",
    ),
    "unique_portal_loop": (
        ("axiom_and_operator_bridges", "scalar_portal_benchmark"),
        "Loop order does not select a unique portal operator or coefficient.",
        "Enumerate the EFT basis and perform explicit matching.",
    ),
    "tree_loop_stress_tensor": (
        ("axiom_and_operator_bridges", "bootstrap_cosmology"),
        "Tree/loop labels are not dark-energy/dark-matter stress tensors.",
        "Compute the renormalized stress tensor and its cosmological evolution.",
    ),
    "gauge_coupling_bridge": (
        ("axiom_and_operator_bridges", "gauge_structure_and_couplings"),
        "Factorizing exp[-(S1+S2+S3)/g_s^2] leaves the same g_s in each factor; it does not derive alpha_1=alpha_s^(1/3) or sin(theta_W)=2 alpha_s^(2/3).",
        "Specify a UV gauge embedding and boundary normalization, then calculate threshold matching and RG evolution.",
    ),
    "gauge_factorization": (
        ("causality_and_no_signalling", "gauge_structure_and_couplings"),
        "Gauge-group factorization does not imply independent physical lattices or channels.",
        "Define gauge-invariant observables and their local correlation functions.",
    ),
    "determinant_lapse_time_travel": (
        ("causality_and_no_signalling",),
        "A determinant or lapse sign is not a traversable closed-timelike-curve solution.",
        "Supply a Lorentzian metric solving the field equations and test global hyperbolicity.",
    ),
    "lapse_zero_freeze": (
        ("causality_and_no_signalling",),
        "N=0 can mark a degenerate foliation or coordinate horizon, and a nonzero shift N^i can remain; physical freezing or Delta x=0 does not follow.",
        "Use the full ADM lapse, shift, spatial metric, and gauge-invariant worldline observables to test physical motion.",
    ),
    "phase_time_period": (
        ("causality_and_no_signalling",),
        "A dimensionless phase period has no physical time until an energy scale and clock are fixed.",
        "Provide the Hamiltonian and convert phase evolution to proper time.",
    ),
    "gauge_causality_safeguard": (
        ("causality_and_no_signalling", "gauge_structure_and_couplings"),
        "A free neutral scalar with no gauge interaction can still be microcausal, so remaining gauge sectors do not enforce causality; exact independent-channel factorization is also absent.",
        "Specify the local action and prove microcausality directly, without treating gauge-factor survival as a causal safeguard.",
    ),
    "triple_damping_ftl_proof": (
        ("causality_and_no_signalling",),
        "The triple-damping ansatz has no derived action or defined xi and remains nonzero at finite v>c, so it cannot prove FTL or teleportation impossible.",
        "Derive a normalized operational channel from the action and test microcausality or zero signalling capacity directly.",
    ),
    "determinant_teleportation_proof": (
        ("causality_and_no_signalling",),
        "A determinant identity does not define a teleportation protocol or prove an operational no-teleportation/no-signalling theorem.",
        "Specify states, local operations, classical communication, channel capacity, and the exact inequality implied by the determinant.",
    ),
    "causality_dimension_uniqueness": (
        ("causality_and_no_signalling", "dimension_hodge"),
        "Causality assumptions do not uniquely derive three spatial dimensions as stated.",
        "List the independent axioms and exclude all alternative-dimensional causal models.",
    ),
    # Chapters 12--14: transitions, precision constants, and residual fields.
    "transition_monotonicity": (
        ("transition_dynamics",),
        "The stated x_sel is constant on part of [0,1] and nondifferentiable at D=1, so it is not smooth and strictly decreasing for every D>0.",
        "State the correct piecewise domain and prove continuity, differentiability, and monotonicity on each claimed interval.",
    ),
    "lambert_branch_world": (
        ("transition_dynamics",),
        "Choosing a Lambert-W branch does not create a physical world or transition law.",
        "Derive branch selection and observable meaning from initial/boundary conditions.",
    ),
    "flow_potential_sign": (
        ("transition_dynamics",),
        "Differentiating the displayed V=-F_D/D-x^2/2 does not reproduce the stated ODE dot(x)=x-F_D(x); the sign/derivative chain is inconsistent.",
        "Derive the gradient flow from one correctly differentiated potential before analyzing stability.",
    ),
    "contraction_rate": (
        ("transition_dynamics",),
        "Holding x_-(D) fixed while changing D is inconsistent; at D=20.556 the branch gives x_- about 1.18e-9 and D*x about 2.43e-8.",
        "Recompute the branch at each D before evaluating its derivative or contraction factor.",
    ),
    "electroweak_energy_formula": (
        ("transition_dynamics", "gauge_structure_and_couplings"),
        "The coefficient F and exponent 12 are undetermined, so 244--298 GeV is not a prediction; the 246 GeV zero-temperature VEV is also not the finite-temperature crossover.",
        "Derive the coefficient and exponent from an action, then compute the finite-temperature effective potential and crossover observable.",
    ),
    "portal_vacuum_energy": (
        ("scalar_portal_benchmark", "bootstrap_cosmology"),
        "A Higgs-portal parameter does not automatically predict the observed vacuum energy.",
        "Calculate the renormalized vacuum energy and cosmological likelihood in one prescription.",
    ),
    "portal_parameter_package_consistency": (
        ("scalar_portal_benchmark", "recursive_residual_to_field"),
        "Z2 does not fix lambda_HP=delta^2, one field cannot have both 43.7 GeV and 29.6 MeV pole masses, and that same coupling/light-scalar package does not yield BR_inv=0.005.",
        "Use one field spectrum and parameter point, derive rather than posit lambda_HP, and recompute its pole and invisible width together.",
    ),
    "gauge_phase_space_measure": (
        ("precision_constant_formula_search", "gauge_structure_and_couplings"),
        "A complex polydisc C^N is neither the SU(N) group nor its gauge-orbit phase space; no Haar measure or gauge quotient is supplied.",
        "Define the physical configuration/orbit space, gauge fixing, Faddeev-Popov factor, Haar measure, and normalization.",
    ),
    "alpha_geometric_derivation": (
        ("precision_constant_formula_search",),
        "A close pi-based expression is not a geometric derivation of alpha.",
        "Derive the coefficient from a gauge-invariant action and normalized path-integral measure.",
    ),
    "precision_formula_residual": (
        ("precision_constant_formula_search",),
        "The formula residual exceeds modern precision or uses target-tuned coefficients.",
        "Propagate current uncertainty and penalize the complete search space.",
    ),
    "color_normalization_mass": (
        ("precision_constant_formula_search", "fermion_spectrum"),
        "Color normalization does not derive a pole or running mass without a mass operator.",
        "Specify the Yukawa/mass operator, scheme, RG scale, and thresholds.",
    ),
    "iteration_time_irreversibility": (
        ("transition_dynamics", "recursive_residual_to_field"),
        "Iteration ordering is not thermodynamic time or irreversibility.",
        "Add a continuous stochastic/dynamical law and derive entropy production.",
    ),
    "lambert_d0_identity": (
        ("recursive_residual_to_field",),
        "For D>1 the W_-1 solution x=1 is a branch at that same D; it does not imply D=0 or a physical d=0 world.",
        "Provide an explicit D-to-d and state-space map, then prove branch selection under fixed boundary data.",
    ),
    "f0_constant_not_identity": (
        ("recursive_residual_to_field",),
        "F_0(x)=exp(0)=1 is a constant map, not the identity x->x, and it has only the fixed point x=1.",
        "Keep parameter and argument notation distinct and derive any physical map from the action.",
    ),
    "residual_entropy": (
        ("recursive_residual_to_field",),
        "An algebraic residual is not thermodynamic entropy.",
        "Define microstates, measure, coarse graining, and an entropy functional.",
    ),
    "residual_field_reality": (
        ("recursive_residual_to_field",),
        "Triviality of the spatial rotation group SO(0) does not restrict internal complex structure or make a field real.",
        "Specify the field representation, conjugation/reality condition, spacetime dependence, and kinetic operator.",
    ),
    "z2_not_automatic": (
        ("recursive_residual_to_field",),
        "The label d=0 or a trivial spatial rotation group does not imply an independent internal Z2 symmetry.",
        "State the Z2 transformation on every field and verify invariance of the full action, vacuum, and measure.",
    ),
    "z2_coupling_nondetermination": (
        ("recursive_residual_to_field", "scalar_portal_benchmark"),
        "Z2 symmetry permits but does not determine the portal coupling.",
        "Match lambda_HP from a specified UV or residual-field dynamics.",
    ),
    "branch_cosmogenesis": (
        ("transition_dynamics", "bootstrap_cosmology", "recursive_residual_to_field"),
        "Switching algebraic Lambert-W branches is not a dynamical cosmological transition or creation of spacetime.",
        "Supply a time-dependent action, initial/boundary data, a regular branch-transition solution, and cosmological observables.",
    ),
    "residual_matter_identity": (
        ("recursive_residual_to_field", "axiom_and_operator_bridges", "bootstrap_cosmology"),
        "An abstract residual is neither dark matter nor dark energy without, respectively, a particle/state map or a stress-energy and cosmological map.",
        "Construct the Hilbert/operator and stress-tensor maps, then reproduce spectra, charges, abundance, and equation-of-state evolution.",
    ),
    "residual_field_chain": (
        ("recursive_residual_to_field", "scalar_portal_benchmark"),
        "The residual-to-field-to-Z2-to-lambda_HP chain lacks spectral matching, an independently proved internal symmetry, and a coupling derivation.",
        "Match poles, residues, and locality; verify Z2 on action/vacuum/measure; then derive and constrain lambda_HP.",
    ),
}


SEMANTIC_GATE_DEFINITIONS: dict[str, SemanticGateDefinition] = {
    gate_id: _semantic_gate(gate_id, *spec)
    for gate_id, spec in _SEMANTIC_GATE_SPECS.items()
}


# A complete counterexample removes its parent sentence from the manuscript.
# Therefore the live routing registry is deliberately a deletion-tolerant
# allow-list: every remaining [Rejected] disposition must match exactly one
# chapter-scoped semantic fingerprint, while a rule that no longer appears is
# legal.  A new, rewritten, or ambiguous occurrence has no route and fails
# closed until it receives an explicit review.  Line numbers are report data,
# never identity, because deleting a counterexample necessarily shifts them.
REMAINING_REJECTED_ROUTE_RULES: tuple[RemainingRejectedRouteRule, ...] = (
    RemainingRejectedRouteRule(
        "03-state-space-x1-not-d0",
        "03",
        (r"$x=1$", r"$d=0$", r"$D=3.177758"),
        ("state_space_bridge",),
    ),
    RemainingRejectedRouteRule(
        "03-posthoc-cosmology-ratio",
        "03",
        (r"R=0.38063", r"\Omega_\Lambda=0.6891", "blind observable"),
        ("cosmology_posthoc_ratio",),
    ),
    RemainingRejectedRouteRule(
        "03-loop-order-is-not-nlo",
        "03",
        (r"\alpha_w,\alpha_{\rm em}", "NLO", "재규격화 계산"),
        ("perturbative_order_diagram",),
    ),
    RemainingRejectedRouteRule(
        "04-gauge-unification-ledger-row",
        "04",
        ("| 게이지 결합 통일 |", r"$M_Z$", "통일군·동일 결합값"),
        ("gauge_unification_claim",),
    ),
    RemainingRejectedRouteRule(
        "04-solved-percentage-accounting",
        "04",
        ("해결 16개", "76% 해결", "Open 항목"),
        ("solved_claim_accounting",),
    ),
    RemainingRejectedRouteRule(
        "04-ce-bsm-zero-gminus2",
        "04",
        ("CE BSM", r"$g-2$", "CE 고유 진폭"),
        ("light_scalar_precision",),
    ),
    RemainingRejectedRouteRule(
        "04-sm-derived-parameter-count",
        "04",
        ("| 표준모형 매개변수의 유도 개수 |", r"\alpha_s", "외부 입력"),
        ("solved_claim_accounting",),
    ),
    RemainingRejectedRouteRule(
        "08-uncomputed-cp-wall-source",
        "08",
        (r"\Delta\theta_{CP}", r"J\kappa_{\rm wall}", "transport equation"),
        ("baryon_cp_source",),
    ),
    RemainingRejectedRouteRule(
        "08-uncomputed-wall-factor-range",
        "08",
        (r"\kappa_{\rm wall}=10^{-2}", "계산이나 인용"),
        ("baryon_wall_factor",),
    ),
    RemainingRejectedRouteRule(
        "08-ansatz-does-not-solve-baryogenesis",
        "08",
        ("계속 **[Open]**", "“해결” 주장"),
        ("baryon_ansatz",),
    ),
    RemainingRejectedRouteRule(
        "08-density-conversion-is-not-generation",
        "08",
        ("이 환산", "초기 universe", "asymmetry"),
        ("density_conversion_not_generation",),
    ),
    RemainingRejectedRouteRule(
        "08-majorana-language-is-not-leptogenesis",
        "08",
        ("Majorana", "leptogenesis"),
        ("leptogenesis_missing_sector",),
    ),
    RemainingRejectedRouteRule(
        "08-portal-lacks-leptogenesis-sector",
        "08",
        ("CP-asymmetric heavy-particle", r"\Delta L=2", "neutrino interaction"),
        ("leptogenesis_missing_sector",),
    ),
)


# Legacy line anchors remain as historical provenance for the pre-deletion
# review.  They are not used to pair live occurrences: zipping a shorter live
# list to this list silently assigns the wrong gate after an interior deletion.
SEMANTIC_ROUTE_GROUPS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("01", "hodge_form_degree", (43, 112)),
    ("01", "lower_dim_counterexamples", (71,)),
    ("01", "cp_source_nonuniqueness", (83,)),
    ("01", "state_space_bridge", (104,)),
    ("02", "kernel_curvature_type", (24,)),
    ("02", "gravity_quantization_eft", (54,)),
    ("02", "generation_count_category", (62,)),
    ("02", "color_singlet_representation", (63,)),
    ("02", "gauge_sum_underdetermined", (78, 103)),
    ("03", "state_space_bridge", (122,)),
    ("03", "cosmology_posthoc_ratio", (239,)),
    ("03", "perturbative_order_diagram", (241,)),
    ("03", "cosmology_zero_parameter_accounting", (276, 352)),
    ("03", "cosmology_likelihood_calibration", (348,)),
    ("04", "gauge_unification_claim", (22,)),
    ("04", "solved_claim_accounting", (28, 229)),
    ("04", "horizon_entropy_equation", (46,)),
    ("04", "cosmology_data_model", (83,)),
    ("04", "light_scalar_precision", (110,)),
    ("04", "electromagnetic_coupling_identity", (180,)),
    ("04", "cosmology_version_mixing", (200,)),
    ("04", "neutrino_operator_claim", (223,)),
    ("05", "inflation_initial_condition", (62,)),
    ("05", "state_space_bridge", (64,)),
    ("05", "finite_xi_approximation", (118,)),
    ("05", "iteration_time_bridge", (151, 409)),
    ("05", "desi_xi_postfit", (194,)),
    ("05", "z2_single_decay", (262,)),
    ("05", "portal_benchmark", (276,)),
    ("05", "reheating_baryon_identity", (372,)),
    ("06", "strong_cp_field_mechanism", (14, 69, 235)),
    ("06", "strong_cp_singlet_symmetry", (70,)),
    ("06", "strong_cp_even_not_axion", (71,)),
    ("06", "instanton_charge_double_count", (111,)),
    ("06", "instanton_measure", (112, 113)),
    ("06", "strong_cp_bootstrap_bound", (192, 236)),
    ("06", "neutron_edm_prediction", (199,)),
    ("07", "neutrino_lnv_operator", (14, 135, 202, 338, 339)),
    ("07", "scotogenic_field_content", (118,)),
    ("07", "neutrino_flavor_matrix", (136,)),
    ("07", "neutrino_dimensionality", (156,)),
    ("07", "neutrino_loop_factor", (157,)),
    ("07", "neutrino_suppression_chain", (158,)),
    ("07", "generation_adjoint", (172,)),
    ("07", "neutrino_fractional_power", (173,)),
    ("07", "weinberg_qcd_running", (174,)),
    ("07", "neutrino_posthoc_exponent", (175,)),
    ("07", "neutrino_mee_consistency", (301,)),
    ("07", "neutrino_ordering_state", (306,)),
    ("07", "portal_collider", (329,)),
    ("07", "portal_mass_identity", (330,)),
    ("08", "baryon_cp_source", (14, 69, 156, 315)),
    ("08", "portal_collider", (16, 314)),
    ("08", "baryon_normalization", (44,)),
    ("08", "portal_ewpt_extrapolation", (140,)),
    ("08", "baryon_wall_factor", (157,)),
    ("08", "baryon_transport_fit", (199,)),
    ("08", "baryon_arithmetic", (200,)),
    ("08", "baryon_ansatz", (233,)),
    ("08", "density_conversion_not_generation", (267,)),
    ("08", "baryon_dimension_formula", (268,)),
    ("08", "leptogenesis_missing_sector", (279, 280)),
    ("09", "fermion_compound_summary", (16,)),
    ("09", "mixed_scale_quark", (34, 35, 396)),
    ("09", "posthoc_exponent", (80,)),
    ("09", "fermion_adjacent_ratio_bridge", (81,)),
    ("09", "koide_precision", (112,)),
    ("09", "koide_universality", (235,)),
    ("09", "gaussian_koide_posit", (277, 394)),
    ("09", "gaussian_normalization_invariance", (278,)),
    ("09", "gaussian_measure_truncation", (279,)),
    ("09", "gaussian_measure_interactions", (280,)),
    ("09", "inconsistent_cosine_root", (305,)),
    ("09", "dimensional_A_sqrt_v", (325, 397)),
    ("09", "qcd_koide_quantification", (339,)),
    ("09", "top_qcd_formula", (372,)),
    ("09", "top_mass_scheme", (374, 398)),
    ("09", "retrodiction", (383,)),
    ("10", "internal_index_dimension", (133, 321)),
    ("10", "singlet_ew_portal", (134,)),
    ("10", "coherent_mixture_probability", (136,)),
    ("10", "unique_portal_loop", (137,)),
    ("10", "tree_loop_stress_tensor", (210, 322)),
    ("10", "gauge_coupling_bridge", (267, 324)),
    ("10", "portal_parameter_package_consistency", (306,)),
    ("11", "gauge_factorization", (97, 257)),
    ("11", "determinant_lapse_time_travel", (122, 153, 259)),
    ("11", "lapse_zero_freeze", (154,)),
    ("11", "phase_time_period", (176,)),
    ("11", "determinant_teleportation_proof", (220,)),
    ("11", "triple_damping_ftl_proof", (221, 262)),
    ("11", "gauge_causality_safeguard", (223,)),
    ("11", "causality_dimension_uniqueness", (240,)),
    ("11", "gauge_sum_underdetermined", (263,)),
    ("12", "transition_monotonicity", (64,)),
    ("12", "lambert_branch_world", (91, 380)),
    ("12", "iteration_time_bridge", (120, 381)),
    ("12", "flow_potential_sign", (150,)),
    ("12", "contraction_rate", (157,)),
    ("12", "electroweak_energy_formula", (285,)),
    ("12", "perturbative_order_diagram", (320, 386)),
    ("12", "gauge_sum_arithmetic", (337,)),
    ("12", "gauge_sum_underdetermined", (387,)),
    ("12", "portal_parameter_package_consistency", (367,)),
    ("12", "portal_vacuum_energy", (372,)),
    ("13", "gauge_phase_space_measure", (33, 204)),
    ("13", "alpha_geometric_derivation", (94,)),
    ("13", "precision_formula_residual", (95, 154, 206)),
    ("13", "color_normalization_mass", (146, 153, 208)),
    ("14", "iteration_time_irreversibility", (86, 299)),
    ("14", "lambert_d0_identity", (195, 312)),
    ("14", "f0_constant_not_identity", (205, 311)),
    ("14", "residual_entropy", (235,)),
    ("14", "residual_field_reality", (246,)),
    ("14", "z2_not_automatic", (247,)),
    ("14", "z2_coupling_nondetermination", (248,)),
    ("14", "branch_cosmogenesis", (298,)),
    ("14", "residual_matter_identity", (300,)),
    ("14", "residual_field_chain", (313,)),
)


EXPECTED_SEMANTIC_MANIFEST_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
EXPECTED_SOURCE_SHA256 = (
    "b61c907bd584efe5e49c6cfeb3b434c363ae88272d136d09aedbedcec1e99274"
)
EXPECTED_REGRESSION_WITNESS_REGISTRY_SHA256 = (
    "f0c9d64e2622e897e8699842897fc7657e33ae7eccb10b827a5ada5d38cf85a0"
)


def declared_semantic_route_index() -> dict[tuple[str, int], tuple[str, ...]]:
    """Return the reviewed legacy anchors, before line-drift normalization."""

    index: dict[tuple[str, int], tuple[str, ...]] = {}
    for chapter_number, gate_id, lines in SEMANTIC_ROUTE_GROUPS:
        if gate_id not in SEMANTIC_GATE_DEFINITIONS:
            raise ValueError(f"undefined semantic gate: {gate_id}")
        for line in lines:
            key = (chapter_number, line)
            if key in index:
                raise ValueError(f"duplicate semantic route: {key}")
            index[key] = (gate_id,)
    return index


def semantic_route_index() -> dict[tuple[str, int], tuple[str, ...]]:
    """Route each live rejected claim by one reviewed semantic fingerprint.

    Missing rules are allowed because a fully counterexampled parent is deleted
    from the prose and retained in the executable witness registry below.
    Missing or multiple matches for a *present* occurrence deliberately produce
    no index entry, which the validator treats as a fail-closed routing error.
    """

    index: dict[tuple[str, int], tuple[str, ...]] = {}
    for occurrence in rejected_inventory():
        chapter = occurrence["chapter"][:2]
        normalized_context = " ".join(occurrence["context"].split()).casefold()
        matches = tuple(
            rule
            for rule in REMAINING_REJECTED_ROUTE_RULES
            if rule.chapter == chapter
            and all(
                " ".join(fragment.split()).casefold() in normalized_context
                for fragment in rule.required_fragments
            )
        )
        if len(matches) == 1:
            index[(chapter, occurrence["line"])] = matches[0].semantic_gate_ids
    return index


def semantic_manifest_digest(
    inventory: list[dict[str, Any]],
    route_index: dict[tuple[str, int], tuple[str, ...]],
) -> str:
    payload = []
    for occurrence in inventory:
        key = (occurrence["chapter"][:2], occurrence["line"])
        gate_ids = route_index.get(key, ())
        payload.append(
            {
                "chapter": occurrence["chapter"],
                "line": occurrence["line"],
                "claim_text": occurrence["text"],
                "semantic_context": occurrence["context"],
                "semantic_gates": [
                    asdict(SEMANTIC_GATE_DEFINITIONS[gate_id])
                    for gate_id in gate_ids
                ],
            }
        )
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def parent_claim_fingerprint(parent_claim: str) -> str:
    normalized = " ".join(parent_claim.split()).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _matrix_product(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _kronecker_product(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            left[left_row][left_column] * right[right_row][right_column]
            for left_column in range(len(left[0]))
            for right_column in range(len(right[0]))
        )
        for left_row in range(len(left))
        for right_row in range(len(right))
    )


def regression_witness_registry(
    loops: tuple[RejectLoop, ...],
) -> tuple[RegressionWitness, ...]:
    """Execute the immutable counterexamples removed from the prose.

    These witnesses are independent of how many ``[Rejected]`` tokens remain.
    A repaired descendant may coexist with a witness, but it cannot mutate the
    falsified parent back into a pass.
    """

    loops_by_id = {loop.loop_id: loop for loop in loops}

    def witness(
        witness_id: str,
        parent_claim: str,
        loop_ids: tuple[str, ...],
        predicate: str,
        observed: dict[str, Any],
        passed: bool,
    ) -> RegressionWitness:
        return RegressionWitness(
            witness_id=witness_id,
            parent_claim=parent_claim,
            parent_fingerprint_sha256=parent_claim_fingerprint(parent_claim),
            loop_ids=loop_ids,
            predicate=predicate,
            observed=observed,
            passed=passed,
        )

    dimension_roots = tuple(
        dimension
        for dimension in range(0, 11)
        if 2 * dimension == dimension * (dimension - 1)
    )
    poisson_d = 2.0
    poisson_nontrivial_root = fixed_point(poisson_d)
    poisson_one_residual = math.exp(-poisson_d * (1.0 - 1.0)) - 1.0

    s12, s23, s13, delta_cp = 0.2, 0.3, 0.1, 0.0
    c12 = math.sqrt(1.0 - s12**2)
    c23 = math.sqrt(1.0 - s23**2)
    c13 = math.sqrt(1.0 - s13**2)
    jarlskog = s12 * s23 * s13 * c12 * c23 * c13**2 * math.sin(delta_cp)

    contraction_diagonal = (2.0, 0.25, 1.0)
    contraction_determinant = math.prod(contraction_diagonal)
    contraction_operator_norm = max(abs(value) for value in contraction_diagonal)

    det_h = 1.0
    lapse_determinants = {
        "N=-1": -(-1.0) ** 2 * det_h,
        "N=0": -(0.0) ** 2 * det_h,
        "N=1": -(1.0) ** 2 * det_h,
    }

    identity = ((1.0, 0.0), (0.0, 1.0))
    annihilation = ((0.0, 1.0), (0.0, 0.0))
    sigma_z = ((1.0, 0.0), (0.0, -1.0))
    fermion_mode_1 = _kronecker_product(annihilation, identity)
    fermion_mode_2 = _kronecker_product(sigma_z, annihilation)
    fermion_mode_2_creation = tuple(zip(*fermion_mode_2))
    a1_a2dagger = _matrix_product(fermion_mode_1, fermion_mode_2_creation)
    a2dagger_a1 = _matrix_product(fermion_mode_2_creation, fermion_mode_1)
    fermion_anticommutator = tuple(
        tuple(
            a1_a2dagger[row][column] + a2dagger_a1[row][column]
            for column in range(4)
        )
        for row in range(4)
    )
    fermion_commutator = tuple(
        tuple(
            a1_a2dagger[row][column] - a2dagger_a1[row][column]
            for column in range(4)
        )
        for row in range(4)
    )
    fermion_anticommutator_norm = max(
        abs(value) for row in fermion_anticommutator for value in row
    )
    fermion_commutator_norm = max(
        abs(value) for row in fermion_commutator for value in row
    )

    koide_components = tuple(
        1.0 + math.sqrt(2.0) * math.cos(math.pi / 3.0 + 2.0 * math.pi * index / 3.0)
        for index in range(3)
    )
    koide_numerator = sum(component**2 for component in koide_components)
    koide_signed = koide_numerator / sum(koide_components) ** 2
    koide_principal = koide_numerator / sum(abs(component) for component in koide_components) ** 2

    inverse_d, inverse_y = 2.0, 2.0
    inverse_x = 1.0 + math.log(inverse_y) / inverse_d
    derivative_d, derivative_x = 2.0, 0.5
    derivative_general = (1.0 - derivative_x) * math.exp(
        -derivative_d * (1.0 - derivative_x)
    )
    derivative_fixed_point_only = derivative_x * (1.0 - derivative_x)
    f0_probe = 0.25
    f0_value = math.exp(0.0 * (1.0 - f0_probe))

    position = ((0.0, 1.0), (1.0, 0.0))
    field_left = _kronecker_product(position, identity)
    field_right = _kronecker_product(identity, position)
    left_right = _matrix_product(field_left, field_right)
    right_left = _matrix_product(field_right, field_left)
    neutral_scalar_commutator_norm = max(
        abs(left_right[row][column] - right_left[row][column])
        for row in range(4)
        for column in range(4)
    )

    periodic_time_period = 1.0
    periodic_timelike_tangent_norm = -1.0
    periodic_start = 0.0
    periodic_end = periodic_start + periodic_time_period
    periodic_curve_closed = (
        periodic_end % periodic_time_period
        == periodic_start % periodic_time_period
    )

    phi_probe, higgs_probe = 2.0, 3.0
    z2_even_operator = phi_probe**2 * higgs_probe**2
    z2_transformed_operator = (-phi_probe) ** 2 * higgs_probe**2

    alpha_reference = 137.035999177
    alpha_uncertainty = 0.000000021
    alpha_formula = 4.0 * math.pi**3 + math.pi**2 + math.pi
    alpha_pull = (alpha_formula - alpha_reference) / alpha_uncertainty
    mass_ratio_reference = 1836.152673426
    mass_ratio_uncertainty = 0.000000032
    mass_ratio_formula = 6.0 * math.pi**5
    mass_ratio_pull = (
        mass_ratio_formula - mass_ratio_reference
    ) / mass_ratio_uncertainty

    portal_legacy = loops_by_id["scalar_portal_benchmark"].iterations[0].metrics
    portal_bound = 0.107
    legacy_baryogenesis_range = (2.96e-14, 2.96e-13)
    observed_baryon_asymmetry = 8.7e-11
    core = benchmark_core()
    legacy_alpha_relation = (
        core["sin2_theta_w"] * core["alpha_s"] ** (2.0 / 3.0)
    )
    alpha_em_mz = 1.0 / 127.95

    return (
        witness(
            "dimension-counting-roots",
            "d = binom(d,2)가 곧바로 d = 3을 유일하게 준다",
            ("dimension_hodge",),
            "enumerate nonnegative integer roots of 2d=d(d-1)",
            {"roots_in_0_through_10": dimension_roots},
            dimension_roots == (0, 3),
        ),
        witness(
            "poisson-extinction-is-minimal-fixed-point",
            "생성함수의 임의 고정점이 곧 소멸확률이다",
            ("bootstrap_cosmology",),
            "at D=2, x=1 is fixed while the minimal fixed point is below one",
            {
                "D": poisson_d,
                "x1_fixed_point_residual": poisson_one_residual,
                "minimal_fixed_point": poisson_nontrivial_root,
            },
            close(poisson_one_residual, 0.0, 1e-15)
            and close(poisson_nontrivial_root, 0.20318786997998, 5e-14)
            and poisson_nontrivial_root < 1.0,
        ),
        witness(
            "generation-count-does-not-force-cp",
            "세 세대가 CP 위반을 강제한다",
            ("baryogenesis",),
            "evaluate a three-generation CKM parameterization at delta=0",
            {"delta": delta_cp, "jarlskog": jarlskog},
            jarlskog == 0.0,
        ),
        witness(
            "determinant-does-not-bound-components",
            "abs(det K) <= 1이면 모든 성분이 수축한다",
            ("causality_and_no_signalling",),
            "evaluate K=diag(2,1/4,1)",
            {
                "diagonal": contraction_diagonal,
                "abs_determinant": abs(contraction_determinant),
                "operator_norm": contraction_operator_norm,
            },
            abs(contraction_determinant) <= 1.0
            and contraction_operator_norm > 1.0,
        ),
        witness(
            "lapse-sign-does-not-prove-global-causality",
            "lapse의 부호 또는 행렬식만으로 CTC가 배제된다",
            ("causality_and_no_signalling",),
            "compare ADM determinants for N=-1,0,+1 at det(h)=1",
            {"determinants": lapse_determinants},
            lapse_determinants["N=0"] == 0.0
            and lapse_determinants["N=-1"] == lapse_determinants["N=1"],
        ),
        witness(
            "fermion-microcausality-is-graded",
            "spacelike 분리에서 fermion의 보통 commutator가 0이어야 한다",
            ("causality_and_no_signalling",),
            "evaluate distinct equal-time Jordan-Wigner lattice modes",
            {
                "anticommutator": fermion_anticommutator,
                "commutator": fermion_commutator,
                "anticommutator_norm": fermion_anticommutator_norm,
                "commutator_norm": fermion_commutator_norm,
            },
            fermion_anticommutator_norm == 0.0
            and fermion_commutator_norm > 0.0,
        ),
        witness(
            "koide-principal-root-domain-counterexample",
            "Koide의 signed 제곱근 매개화가 항상 물리적 주근호 항등식이다",
            ("fermion_spectrum",),
            "evaluate N=3, r=sqrt(2), phi=pi/3 with signed and principal roots",
            {
                "components": koide_components,
                "signed_ratio": koide_signed,
                "principal_root_ratio": koide_principal,
            },
            min(koide_components) < 0.0
            and close(koide_signed, 2.0 / 3.0, 5e-15)
            and close(koide_principal, 0.4093647857764433, 5e-14),
        ),
        witness(
            "transition-inverse-domain-counterexample",
            "전이 역함수 식은 y > 0이면 항상 x in [0,1]을 준다",
            ("transition_dynamics",),
            "evaluate x=1+ln(y)/D at D=2,y=2",
            {"D": inverse_d, "y": inverse_y, "x": inverse_x},
            inverse_x > 1.0,
        ),
        witness(
            "transition-source-derivative-domain",
            "partial_D r = x(1-x)가 일반 항등식이다",
            ("transition_dynamics",),
            "compare the general partial derivative at D=2,x=1/2",
            {
                "general_partial": derivative_general,
                "fixed_point_expression": derivative_fixed_point_only,
            },
            close(derivative_general, 0.18393972058572117, 5e-16)
            and derivative_general != derivative_fixed_point_only,
        ),
        witness(
            "f0-constant-not-identity",
            "D = 0의 사상이 항등사상이다",
            ("recursive_residual_to_field",),
            "evaluate F_0(x)=exp(0) away from x=1",
            {"x": f0_probe, "F0_x": f0_value, "identity_x": f0_probe},
            f0_value == 1.0 and f0_value != f0_probe,
        ),
        witness(
            "gauge-sectors-not-required-for-microcausality",
            "세 게이지 섹터가 있어야 국소 인과성이 가능하다",
            ("causality_and_no_signalling",),
            "commute distinct tensor-factor fields in a zero-gauge neutral scalar lattice",
            {
                "gauge_sector_count": 0,
                "equal_time_spacelike_commutator_norm": neutral_scalar_commutator_norm,
            },
            neutral_scalar_commutator_norm == 0.0,
        ),
        witness(
            "local-metric-does-not-imply-global-causality",
            "국소적으로 정상적인 계량이면 전역 CTC가 없다",
            ("causality_and_no_signalling",),
            "identify Minkowski time modulo one period and test the time-orbit",
            {
                "period": periodic_time_period,
                "curve_closed": periodic_curve_closed,
                "tangent_norm": periodic_timelike_tangent_norm,
            },
            periodic_curve_closed and periodic_timelike_tangent_norm < 0.0,
        ),
        witness(
            "z2-does-not-remove-even-loop-effects",
            "정확한 Z2가 스칼라의 모든 효과를 없앤다",
            ("light_scalar_precision_observables", "scalar_portal_benchmark"),
            "apply phi->-phi to the allowed nonzero phi^2 H^dagger H operator",
            {
                "operator": z2_even_operator,
                "transformed_operator": z2_transformed_operator,
            },
            z2_even_operator != 0.0
            and z2_transformed_operator == z2_even_operator,
        ),
        witness(
            "alpha-inverse-residual-disclosed",
            "미세구조상수 역수 후보가 정확한 무입력 예측이다",
            ("precision_constant_formula_search",),
            "compare 4pi^3+pi^2+pi with the pinned reference and uncertainty",
            {
                "formula": alpha_formula,
                "reference": alpha_reference,
                "pull_sigma": alpha_pull,
            },
            close(alpha_formula, 137.036303775878, 5e-13)
            and abs(alpha_pull) > 14_000.0,
        ),
        witness(
            "proton-electron-residual-disclosed",
            "양성자/전자 질량비 후보가 정확한 무입력 예측이다",
            ("precision_constant_formula_search",),
            "compare 6pi^5 with the pinned reference and uncertainty",
            {
                "formula": mass_ratio_formula,
                "reference": mass_ratio_reference,
                "pull_sigma": mass_ratio_pull,
            },
            close(mass_ratio_formula, 1836.118108711688, 5e-12)
            and abs(mass_ratio_pull) > 1_000_000.0,
        ),
        witness(
            "portal-benchmark-exclusion",
            "원래 Higgs portal 벤치마크가 invisible-width 상한을 통과한다",
            ("scalar_portal_benchmark",),
            "recompute the unchanged legacy point and compare BR_inv with 0.107",
            {**portal_legacy, "BR_invisible_bound": portal_bound},
            close(portal_legacy["mass_GeV"], 43.768, 5e-4)
            and close(portal_legacy["Gamma_invisible_MeV"], 13.753, 5e-3)
            and portal_legacy["BR_invisible"] > portal_bound,
        ),
        witness(
            "legacy-baryogenesis-arithmetic",
            "기존 바리온 생성식이 관측 비대칭을 생성한다",
            ("baryogenesis",),
            "compare the legacy range with the observed baryon asymmetry",
            {
                "legacy_range": legacy_baryogenesis_range,
                "observed_YB": observed_baryon_asymmetry,
            },
            max(legacy_baryogenesis_range) < observed_baryon_asymmetry / 100.0,
        ),
        witness(
            "legacy-alpha-em-relation-rejected",
            "s_W^2 alpha_s^(2/3)이 alpha_em(M_Z)의 정확한 관계다",
            ("gauge_structure_and_couplings",),
            "evaluate both sides at the declared inputs",
            {
                "legacy_left_hand_side": legacy_alpha_relation,
                "alpha_em_MZ": alpha_em_mz,
            },
            close(legacy_alpha_relation, 0.055592, 5e-7)
            and abs(legacy_alpha_relation - alpha_em_mz) > 0.04,
        ),
    )


def canonical_digest_value(value: Any) -> Any:
    """Normalize report data for a Python-version-independent digest.

    Fifteen significant decimal digits retain the reviewed numerical evidence
    while removing irrelevant last-bit libm drift (observed between Python
    3.11 and 3.14).  Floats are tagged so they cannot collide with source
    strings, and non-finite values fail closed instead of entering JSON.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("regression witness digest rejects non-finite floats")
        if value == 0.0:
            value = 0.0
        return {"$float15": format(value, ".15g")}
    if isinstance(value, dict):
        return {key: canonical_digest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_digest_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported regression witness digest type: {type(value)!r}")


def regression_witness_registry_digest(
    witnesses: tuple[RegressionWitness, ...],
) -> str:
    payload = {
        "format": REGRESSION_WITNESS_DIGEST_FORMAT,
        "witnesses": canonical_digest_value(
            [asdict(witness) for witness in witnesses]
        ),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_report() -> LoopEngineeringReport:
    core = benchmark_core()
    loops = (
        dimension_loop(),
        gauge_coupling_loop(core),
        bootstrap_cosmology_loop(core),
        horizon_claim_loop(core),
        inflation_loop(),
        portal_loop(core),
        light_scalar_precision_loop(),
        strong_cp_loop(),
        neutrino_loop(),
        baryogenesis_loop(),
        fermion_loop(),
        axiom_bridge_loop(),
        causality_loop(),
        transition_loop(core),
        precision_constant_loop(),
        recursive_symmetry_loop(),
    )
    inventory = rejected_inventory()
    route_index = semantic_route_index()
    literal_occurrences = rejected_literal_occurrence_count()
    occurrence_routes_list: list[RejectedOccurrenceRoute] = []
    for occurrence in inventory:
        gate_ids = route_index.get(
            (occurrence["chapter"][:2], occurrence["line"]), ()
        )
        loop_ids = tuple(
            dict.fromkeys(
                loop_id
                for gate_id in gate_ids
                for loop_id in SEMANTIC_GATE_DEFINITIONS[gate_id].loop_ids
            )
        )
        occurrence_routes_list.append(
            RejectedOccurrenceRoute(
                chapter=occurrence["chapter"],
                line=occurrence["line"],
                claim_text=occurrence["text"],
                semantic_context=occurrence["context"],
                semantic_gate_ids=gate_ids,
                loop_ids=loop_ids,
            )
        )
    occurrence_routes = tuple(occurrence_routes_list)
    routed = sum(
        1
        for route in occurrence_routes
        if route.semantic_gate_ids and route.loop_ids
    )
    descendant_passes = sum(
        1
        for loop in loops
        if any(
            iteration.status
            in {"SELECTION_PASS", "BRIDGE_PASS", "PHENOMENOLOGY_PASS", "AUDIT_PASS"}
            for iteration in loop.iterations
        )
    )
    witnesses = regression_witness_registry(loops)
    return LoopEngineeringReport(
        schema_version="5.0",
        source_sha256=source_digest(),
        source_rejected_literal_occurrences=literal_occurrences,
        excluded_taxonomy_occurrences=literal_occurrences - len(inventory),
        source_rejected_occurrences=len(inventory),
        routed_rejected_occurrences=routed,
        original_claims_promoted=sum(
            1 for loop in loops if loop.parent_status != "REJECT"
        ),
        descendant_stage_passes=descendant_passes,
        ce_specific_physical_claims_closed=sum(
            1 for loop in loops if loop.ce_specific_physical_claim_closed
        ),
        loops=loops,
        routing_by_chapter=copy.deepcopy(CHAPTER_LOOP_ROUTES),
        semantic_gate_definitions=copy.deepcopy(SEMANTIC_GATE_DEFINITIONS),
        occurrence_routes=occurrence_routes,
        deleted_parent_regression_witnesses=witnesses,
        regression_witness_registry_sha256=(
            regression_witness_registry_digest(witnesses)
        ),
    )


def validate_report(report: LoopEngineeringReport) -> tuple[SelfCheck, ...]:
    canonical_report = build_report()
    canonical_loop_ids = tuple(loop.loop_id for loop in canonical_report.loops)
    canonical_loops_by_id = {
        loop.loop_id: loop for loop in canonical_report.loops
    }
    actual_loop_ids = tuple(loop.loop_id for loop in report.loops)
    derived_original_claims_promoted = sum(
        1 for loop in report.loops if loop.parent_status != "REJECT"
    )
    derived_descendant_stage_passes = sum(
        1
        for loop in report.loops
        if any(
            iteration.status
            in {"SELECTION_PASS", "BRIDGE_PASS", "PHENOMENOLOGY_PASS", "AUDIT_PASS"}
            for iteration in loop.iterations
        )
    )
    derived_ce_specific_physical_claims_closed = sum(
        1 for loop in report.loops if loop.ce_specific_physical_claim_closed
    )
    loop_static_metadata_match = all(
        loop.loop_id in canonical_loops_by_id
        and loop.source_chapters
        == canonical_loops_by_id[loop.loop_id].source_chapters
        and loop.parent_claim
        == canonical_loops_by_id[loop.loop_id].parent_claim
        and loop.maximum_supported_stage
        == canonical_loops_by_id[loop.loop_id].maximum_supported_stage
        and loop.next_required_gate
        == canonical_loops_by_id[loop.loop_id].next_required_gate
        for loop in report.loops
    )
    loop_identity_and_aggregate_consistent = (
        report.schema_version == canonical_report.schema_version
        and actual_loop_ids == canonical_loop_ids
        and len(set(actual_loop_ids)) == len(actual_loop_ids)
        and loop_static_metadata_match
        and report.original_claims_promoted
        == derived_original_claims_promoted
        and report.descendant_stage_passes
        == derived_descendant_stage_passes
        and report.ce_specific_physical_claims_closed
        == derived_ce_specific_physical_claims_closed
    )
    canonical_witness_ids = tuple(
        witness.witness_id
        for witness in canonical_report.deleted_parent_regression_witnesses
    )
    actual_witness_ids = tuple(
        witness.witness_id
        for witness in report.deleted_parent_regression_witnesses
    )
    try:
        reported_witness_digest = regression_witness_registry_digest(
            report.deleted_parent_regression_witnesses
        )
    except (TypeError, ValueError):
        reported_witness_digest = ""
    witness_registry_valid = (
        actual_witness_ids == canonical_witness_ids
        and len(set(actual_witness_ids)) == len(actual_witness_ids)
        and all(
            witness.parent_fingerprint_sha256
            == parent_claim_fingerprint(witness.parent_claim)
            and witness.loop_ids
            and set(witness.loop_ids) <= set(canonical_loop_ids)
            and witness.predicate
            and witness.passed
            for witness in report.deleted_parent_regression_witnesses
        )
        and report.deleted_parent_regression_witnesses
        == canonical_report.deleted_parent_regression_witnesses
        and reported_witness_digest
        == report.regression_witness_registry_sha256
        == canonical_report.regression_witness_registry_sha256
        == EXPECTED_REGRESSION_WITNESS_REGISTRY_SHA256
    )
    loop_ids = {loop.loop_id for loop in report.loops}
    routed_ids = {
        loop_id
        for routes in report.routing_by_chapter.values()
        for loop_id in routes
    }
    statuses_valid = all(
        iteration.status in TERMINAL_STATUSES
        for loop in report.loops
        for iteration in loop.iterations
    )
    all_parent_rejected = all(
        loop.parent_status == "REJECT" for loop in report.loops
    )
    mutations_disclosed = all(
        iteration.index == 0
        or iteration.added_inputs
        or iteration.status in {"REJECT", "NOT_REACHED", "EXACT_COUNTEREXAMPLE"}
        for loop in report.loops
        for iteration in loop.iterations
    )
    terminal_next_gates = all(
        loop.maximum_supported_stage and loop.next_required_gate
        for loop in report.loops
    )
    current_inventory = rejected_inventory()
    route_index = semantic_route_index()
    inventory_keys = {
        (occurrence["chapter"][:2], occurrence["line"])
        for occurrence in current_inventory
    }
    manifest_keys = set(route_index)
    used_gate_ids = {
        gate_id for gate_ids in route_index.values() for gate_id in gate_ids
    }
    route_rule_ids = tuple(
        rule.rule_id for rule in REMAINING_REJECTED_ROUTE_RULES
    )
    route_rule_registry_valid = (
        len(set(route_rule_ids)) == len(route_rule_ids)
        and all(
            rule.rule_id
            and rule.chapter in {path.name[:2] for path in CHAPTERS}
            and rule.required_fragments
            and all(rule.required_fragments)
            and rule.semantic_gate_ids
            and set(rule.semantic_gate_ids) <= set(SEMANTIC_GATE_DEFINITIONS)
            for rule in REMAINING_REJECTED_ROUTE_RULES
        )
    )
    expected_occurrence_routes = tuple(
        (
            occurrence["chapter"],
            occurrence["line"],
            occurrence["text"],
            occurrence["context"],
            route_index.get(
                (occurrence["chapter"][:2], occurrence["line"]), ()
            ),
            tuple(
                dict.fromkeys(
                    loop_id
                    for gate_id in route_index.get(
                        (occurrence["chapter"][:2], occurrence["line"]), ()
                    )
                    for loop_id in SEMANTIC_GATE_DEFINITIONS[gate_id].loop_ids
                )
            ),
        )
        for occurrence in current_inventory
    )
    actual_occurrence_routes = tuple(
        (
            route.chapter,
            route.line,
            route.claim_text,
            route.semantic_context,
            route.semantic_gate_ids,
            route.loop_ids,
        )
        for route in report.occurrence_routes
    )
    semantic_definitions_valid = all(
        definition.gate_id == gate_id
        and definition.loop_ids
        and set(definition.loop_ids) <= loop_ids
        and definition.kill_gate
        and definition.alternate_route
        and definition.next_gate
        for gate_id, definition in report.semantic_gate_definitions.items()
    )
    occurrence_routes_valid = (
        actual_occurrence_routes == expected_occurrence_routes
        and inventory_keys == manifest_keys
        and len({(route.chapter, route.line) for route in report.occurrence_routes})
        == len(report.occurrence_routes)
        and all(
            route.semantic_gate_ids
            and route.loop_ids
            and set(route.semantic_gate_ids) <= set(report.semantic_gate_definitions)
            and set(route.loop_ids) <= loop_ids
            for route in report.occurrence_routes
        )
        and used_gate_ids <= set(report.semantic_gate_definitions)
        and set(report.semantic_gate_definitions)
        == set(SEMANTIC_GATE_DEFINITIONS)
        and semantic_definitions_valid
        and route_rule_registry_valid
        and len(
            {
                definition.next_gate
                for definition in report.semantic_gate_definitions.values()
            }
        )
        >= len(SEMANTIC_NEXT_GATE_BY_LOOP)
    )
    semantic_digest = semantic_manifest_digest(current_inventory, route_index)
    semantic_digest_valid = (
        semantic_digest == EXPECTED_SEMANTIC_MANIFEST_SHA256
    )
    # Golden numerical checks exercise the deterministic builder.  The supplied
    # report is compared with that builder above and by the structural check
    # below.  Reading numerical fixtures from the canonical report keeps this
    # validator fail-closed, rather than raising KeyError/IndexError, when a
    # caller supplies a report with missing, extra, or truncated loops.
    by_id = canonical_loops_by_id
    inflation_metrics = by_id["finite_xi_inflation"].iterations[1].metrics
    portal_metrics = by_id["scalar_portal_benchmark"].iterations[1].metrics
    portal_candidate_iteration = by_id["scalar_portal_benchmark"].iterations[4]
    portal_global_metrics = portal_candidate_iteration.metrics
    precision_g2_metrics = by_id["light_scalar_precision_observables"].iterations[1].metrics
    precision_radius_metrics = by_id["light_scalar_precision_observables"].iterations[3].metrics
    precision_null_metrics = by_id["light_scalar_precision_observables"].iterations[4].metrics
    neutrino_iteration = by_id["neutrino_mass"].iterations[1]
    neutrino_metrics = neutrino_iteration.metrics
    leptogenesis_iteration = by_id["baryogenesis"].iterations[1]
    leptogenesis_metrics = leptogenesis_iteration.metrics
    transition_metrics = by_id["transition_dynamics"].iterations[1].metrics
    precision_search_metrics = by_id[
        "precision_constant_formula_search"
    ].iterations[1].metrics
    return (
        SelfCheck(
            "canonical-report-rebuild",
            report == canonical_report,
            "every report field matches a fresh deterministic rebuild",
        ),
        SelfCheck(
            "loop-identity-and-aggregate-consistency",
            loop_identity_and_aggregate_consistent,
            f"loop ids={actual_loop_ids}; expected={canonical_loop_ids}; "
            f"derived/report promotions={derived_original_claims_promoted}/"
            f"{report.original_claims_promoted}, descendant passes="
            f"{derived_descendant_stage_passes}/{report.descendant_stage_passes}, "
            f"CE closures={derived_ce_specific_physical_claims_closed}/"
            f"{report.ce_specific_physical_claims_closed}",
        ),
        SelfCheck(
            "deleted-parent-regression-witness-registry",
            witness_registry_valid,
            f"{len(report.deleted_parent_regression_witnesses)} immutable "
            f"counterexample predicates executed; digest="
            f"{report.regression_witness_registry_sha256}",
        ),
        SelfCheck(
            "all-rejected-occurrences-routed",
            report.source_rejected_literal_occurrences
            == report.source_rejected_occurrences
            == report.routed_rejected_occurrences
            == 0
            and report.excluded_taxonomy_occurrences == 0
            and report.occurrence_routes == (),
            "canonical prose has zero deleted-parent markers; executable witnesses remain internal",
        ),
        SelfCheck(
            "occurrence-semantic-gate-manifest",
            occurrence_routes_valid and semantic_digest_valid,
            f"{len(report.occurrence_routes)} line-addressed claims carry explicit "
            f"kill/alternate/next gates; digest={semantic_digest}",
        ),
        SelfCheck(
            "whole-manuscript-source-digest-lock",
            report.source_sha256 == EXPECTED_SOURCE_SHA256
            and report.source_sha256 == source_digest(),
            f"canonical 01--14 source digest={report.source_sha256}",
        ),
        SelfCheck(
            "no-chapter-fallback-routing",
            actual_occurrence_routes == expected_occurrence_routes
            and inventory_keys == manifest_keys
            and route_rule_registry_valid,
            "live occurrences match exactly one deletion-tolerant semantic fingerprint",
        ),
        SelfCheck(
            "all-chapters-routed",
            set(path.name for path in CHAPTERS)
            == set(report.routing_by_chapter),
            f"{len(report.routing_by_chapter)}/{len(CHAPTERS)} chapters routed",
        ),
        SelfCheck(
            "routing-targets-exist",
            routed_ids <= loop_ids,
            f"unknown loop ids={sorted(routed_ids-loop_ids)}",
        ),
        SelfCheck(
            "parent-rejects-never-promoted",
            all_parent_rejected and report.original_claims_promoted == 0,
            f"original promotions={report.original_claims_promoted}",
        ),
        SelfCheck(
            "status-vocabulary",
            statuses_valid,
            "every stage uses the fail-closed vocabulary",
        ),
        SelfCheck(
            "mutation-input-disclosure",
            mutations_disclosed,
            "every advancing mutation discloses added inputs",
        ),
        SelfCheck(
            "terminal-next-gates",
            terminal_next_gates,
            "every loop exposes its maximum stage and next required gate",
        ),
        SelfCheck(
            "finite-xi-numerics",
            close(inflation_metrics["n_s"], 0.9677172, 2e-7)
            and close(inflation_metrics["r"], 0.00396833, 5e-8)
            and close(
                inflation_metrics["lambda_4_required"],
                1.2220726573e-10,
                5e-20,
            )
            and close(
                inflation_metrics["A_s_reconstructed"],
                CANONICAL_OBS_AS,
                5e-18,
            )
            and close(CANONICAL_OBS_AS_1E9, 2.099, 5e-15)
            and close(CANONICAL_OBS_AS_SIGMA_1E9, 0.029, 5e-15),
            f"Planck A_s x 1e9={CANONICAL_OBS_AS_1E9:.3f}"
            f"+/-{CANONICAL_OBS_AS_SIGMA_1E9:.3f}; "
            f"n_s={inflation_metrics['n_s']:.9f}, "
            f"r={inflation_metrics['r']:.9f}, "
            f"lambda4={inflation_metrics['lambda_4_required']:.12e}",
        ),
        SelfCheck(
            "portal-mutation-numerics",
            close(portal_metrics["lambda_HP_limit"], 0.00595010827, 5e-12)
            and close(portal_metrics["m0_GeV"], 39.43209966, 5e-8)
            and close(portal_global_metrics["example_m0_GeV"], 61.8531544344, 5e-10)
            and close(portal_global_metrics["example_BR_invisible"], 6.00130848e-5, 5e-13)
            and close(
                portal_global_metrics["estimated_sigma_SI_cm2_fN_0p30"],
                7.97227389e-49,
                5e-57,
            ),
            f"lambda_limit={portal_metrics['lambda_HP_limit']:.10f}, "
            f"m0={portal_metrics['m0_GeV']:.8f} GeV, "
            f"resonance BRinv={portal_global_metrics['example_BR_invisible']:.8e}",
        ),
        SelfCheck(
            "portal-correlated-band-status-lock",
            portal_candidate_iteration.status == "OPEN"
            and portal_global_metrics["joint_allowed_membership_verified"] is False
            and portal_global_metrics["relic_density_calculated"] is False
            and portal_global_metrics[
                "indirect_detection_likelihood_calculated"
            ]
            is False
            and by_id["scalar_portal_benchmark"].maximum_supported_stage
            == "EXTERNAL_WINDOW_REPORTED_POINT_MEMBERSHIP_OPEN",
            "the 62 GeV illustration remains OPEN until correlated relic-band and indirect-likelihood membership are computed",
        ),
        SelfCheck(
            "light-scalar-current-precision-gates",
            close(precision_g2_metrics["scalar_integral"], 0.981535160747, 5e-12)
            and close(precision_g2_metrics["pull_sigma"], 7.15412672, 5e-8)
            and close(precision_radius_metrics["legacy_pull_sigma"], 22.52802805, 5e-8)
            and precision_null_metrics["direct_delta_a_mu"] == 0.0
            and precision_null_metrics["direct_delta_r2_fm2"] == 0.0,
            f"g-2 pull={precision_g2_metrics['pull_sigma']:.6f} sigma, "
            f"radius pull={precision_radius_metrics['legacy_pull_sigma']:.6f} sigma",
        ),
        SelfCheck(
            "neutrino-dirac-mutation",
            close(
                neutrino_metrics["diagonal_yukawas"][-1],
                2.8710453712e-13,
                5e-23,
            )
            and neutrino_iteration.status == "BRIDGE_PASS"
            and neutrino_iteration.added_inputs
            == (
                "three nu_R fields with B-L=-1",
                "unbroken anomaly-free U(1)_(B-L)",
                "Dirac Yukawa matrix",
            )
            and neutrino_metrics["lepton_number_conserved"] is True
            and neutrino_metrics[
                "majorana_mass_forbidden_by_unbroken_B_minus_L"
            ]
            is True,
            f"largest y_nu={neutrino_metrics['diagonal_yukawas'][-1]:.12e}",
        ),
        SelfCheck(
            "leptogenesis-normalization-conventions",
            leptogenesis_iteration.status == "BRIDGE_PASS"
            and "total efficiency kappa_total=0.1 including equilibrium-yield normalization"
            in leptogenesis_iteration.added_inputs
            and leptogenesis_metrics["efficiency_convention"]
            == "kappa_total absorbs the relativistic equilibrium-yield factor"
            and close(
                leptogenesis_metrics[
                    "equilibrium_yield_factor_without_1_over_gstar"
                ],
                135.0 * 1.202056903159594 / (4.0 * math.pi**4),
                5e-15,
            )
            and close(
                leptogenesis_metrics["required_epsilon_1"],
                2.62033125e-7,
                5e-16,
            )
            and close(
                leptogenesis_metrics[
                    "required_epsilon_if_kappa_excludes_equilibrium_yield"
                ],
                leptogenesis_metrics["required_epsilon_1"]
                / leptogenesis_metrics[
                    "equilibrium_yield_factor_without_1_over_gstar"
                ],
                5e-16,
            )
            and close(
                leptogenesis_metrics[
                    "minimum_M1_if_kappa_excludes_equilibrium_yield_GeV"
                ],
                leptogenesis_metrics["minimum_M1_GeV"]
                / leptogenesis_metrics[
                    "equilibrium_yield_factor_without_1_over_gstar"
                ],
                5.0,
            )
            and close(
                leptogenesis_metrics["minimum_M1_GeV"],
                2_659_223_919.7527304,
                0.1,
            )
            and by_id["baryogenesis"].iterations[2].status == "NOT_REACHED"
            and by_id["baryogenesis"].iterations[2].metrics[
                "transport_solution_present"
            ]
            is False,
            "kappa_total and explicit-equilibrium-yield conventions are both locked",
        ),
        SelfCheck(
            "continuous-transition-mutation",
            transition_metrics["absolute_error"] < 1e-12
            and transition_metrics["relaxation_rate_at_target"] > 0.0
            and math.isfinite(
                transition_metrics["dimensionless_time_to_1e-6"]
            ),
            f"final error={transition_metrics['absolute_error']:.3e}, "
            f"tau_hit={transition_metrics['dimensionless_time_to_1e-6']:.3f}",
        ),
        SelfCheck(
            "sparse-formula-negative-control",
            precision_search_metrics["sparse_alpha_trials"] == 63_588
            and precision_search_metrics["sparse_mass_trials"] == 63_588
            and precision_search_metrics["sparse_alpha_best_pull_sigma"]
            > 10_000.0
            and precision_search_metrics["sparse_mass_best_pull_sigma"]
            > 1_000_000.0
            and precision_search_metrics["untouched_holdout_count"] == 0,
            "both 63,588-candidate sparse grammars fail precision and have no untouched holdout",
        ),
        SelfCheck(
            "ce-specific-closure-lock",
            report.ce_specific_physical_claims_closed == 0,
            "no added-field or fitted descendant is mislabelled as a CE-specific closure",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    report = build_report()
    checks = validate_report(report)
    passed = all(check.passed for check in checks)

    if args.json:
        print(
            json.dumps(
                {
                    "passed": passed,
                    "report": asdict(report),
                    "self_checks": [asdict(check) for check in checks],
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    else:
        print("REJECTION LOOP ENGINEERING")
        print("=" * 96)
        print(
            f"rejected occurrences routed: "
            f"{report.routed_rejected_occurrences}/"
            f"{report.source_rejected_occurrences}"
        )
        print(
            "deleted-parent regression witnesses: "
            f"{sum(witness.passed for witness in report.deleted_parent_regression_witnesses)}/"
            f"{len(report.deleted_parent_regression_witnesses)}"
        )
        print(f"loop families: {len(report.loops)}")
        print(f"original claims promoted: {report.original_claims_promoted}")
        print(f"descendant stage passes: {report.descendant_stage_passes}")
        print(
            "CE-specific physical claims fully closed: "
            f"{report.ce_specific_physical_claims_closed}"
        )
        print("-" * 96)
        for loop in report.loops:
            print(
                f"{loop.loop_id:38s} -> "
                f"{loop.maximum_supported_stage}"
            )
        print("-" * 96)
        for check in checks:
            print(
                f"[{'PASS' if check.passed else 'FAIL'}] "
                f"{check.name}: {check.detail}"
            )
        print(
            f"\nRESULT: {'PASS' if passed else 'FAIL'} "
            f"({sum(check.passed for check in checks)}/{len(checks)})"
        )
        if not args.self_test:
            print(
                "PASS means the loops are complete and fail-closed; "
                "it does not mean every rejected physical claim became true."
            )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
