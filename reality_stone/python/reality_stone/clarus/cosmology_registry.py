"""Typed, non-destructive registry for CE cosmology quantities.

The registry deliberately keeps three different numerical layers separate:

* ``CE_CORE_EXACT_V1`` evaluates the declared CE formula chain in binary64;
* ``LEGACY_DELTA_5DP_V1`` reproduces the historical rounded-``delta`` solver;
* ``LEGACY_ROUNDED_RUNTIME_V1`` preserves the product/runtime target triplet.

These layers have different roles and must not be silently substituted for one
another.  In particular, ``q_ext`` is an extinction probability.  Survival is
always represented explicitly as ``1 - q_ext``.

This module is intentionally standard-library-only so audits can load it by
file path without importing the torch-heavy :mod:`reality_stone.clarus`
package facade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping


class RegistryRole(str, Enum):
    """How a registry entry may be used."""

    SCIENTIFIC_CORE = "scientific_core"
    CONDITIONAL_MODEL = "conditional_model"
    RESEARCH_CANDIDATE = "research_candidate"
    HISTORICAL_MODEL = "historical_model"
    COMPATIBILITY_BOUNDARY = "compatibility_boundary"
    RESEARCH_TARGET = "research_target"
    OBSERVATION_REFERENCE = "observation_reference"


class FormalStatus(str, Enum):
    """CE formal status; this is not an observational confidence label."""

    DEFINITION = "definition"
    THEOREM = "theorem"
    AXIOM = "axiom"
    DERIVATION = "derivation"
    EMPIRICAL = "empirical"
    INCOMPLETE = "incomplete"
    EXCLUDED = "excluded"


class ModelStatus(str, Enum):
    """Activation boundary for a model/configuration."""

    ACTIVE_MATHEMATICS = "active_mathematics"
    CONDITIONAL = "conditional"
    CANDIDATE = "candidate"
    HISTORICAL = "historical"
    COMPATIBILITY_ONLY = "compatibility_only"
    INCOMPLETE = "incomplete"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class Provenance:
    """Machine-readable provenance attached to a registry entry."""

    source_id: str
    source_kind: str
    source_path: str
    formula_version: str
    precision: str
    note: str = ""


@dataclass(frozen=True)
class FixedPointResult:
    """Certificate for the non-identity extinction fixed point."""

    model_id: str
    d_eff: float
    q_ext: float
    survival: float
    contraction: float
    signed_residual: float
    absolute_residual: float
    iterations: int
    precision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_residual(q_ext: float, d_eff: float) -> float:
    """Return ``q - exp(-D(1-q))`` for an extinction candidate."""

    if not math.isfinite(q_ext) or not math.isfinite(d_eff):
        raise ValueError("q_ext and d_eff must be finite")
    return q_ext - math.exp(-d_eff * (1.0 - q_ext))


def solve_low_extinction_root(
    d_eff: float,
    *,
    model_id: str,
    precision: str,
    max_iterations: int = 100,
) -> FixedPointResult:
    """Solve the stable non-identity extinction root for ``D > 1``.

    Newton steps are safeguarded inside ``(0, 1/D)``.  The upper endpoint is
    below the identity root, so the solver cannot accidentally certify
    ``q=1``.  The returned residual and precision label travel with the value.
    """

    if not math.isfinite(d_eff) or d_eff <= 1.0:
        raise ValueError("the stable non-identity extinction root requires finite D > 1")
    if not model_id:
        raise ValueError("model_id is required")

    lo = 0.0
    hi = 1.0 / d_eff
    q_ext = math.exp(-d_eff)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        residual = bootstrap_residual(q_ext, d_eff)
        if residual > 0.0:
            hi = q_ext
        else:
            lo = q_ext

        derivative = 1.0 - d_eff * math.exp(-d_eff * (1.0 - q_ext))
        candidate = q_ext - residual / derivative
        if not (lo < candidate < hi) or not math.isfinite(candidate):
            candidate = 0.5 * (lo + hi)

        if candidate == q_ext:
            break
        q_ext = candidate
    else:  # pragma: no cover - defensive branch for altered numerical backends
        raise RuntimeError(f"low-root solve did not converge after {max_iterations} iterations")

    signed_residual = bootstrap_residual(q_ext, d_eff)
    absolute_residual = abs(signed_residual)
    residual_limit = 8.0 * math.ulp(q_ext)
    if not 0.0 < q_ext < 1.0 / d_eff or absolute_residual > residual_limit:
        raise RuntimeError(
            "low-root certificate failed: "
            f"q={q_ext!r}, residual={signed_residual!r}, limit={residual_limit!r}"
        )

    return FixedPointResult(
        model_id=model_id,
        d_eff=d_eff,
        q_ext=q_ext,
        survival=1.0 - q_ext,
        contraction=d_eff * q_ext,
        signed_residual=signed_residual,
        absolute_residual=absolute_residual,
        iterations=iterations,
        precision=precision,
    )


@dataclass(frozen=True)
class CoreChain:
    """Versioned CE formula chain and its fixed-point certificate."""

    model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    alpha_s: float
    sin2_theta_w: float
    delta: float
    d_eff: float
    fixed_point: FixedPointResult
    provenance: Provenance

    @property
    def q_ext(self) -> float:
        return self.fixed_point.q_ext

    @property
    def survival(self) -> float:
        return self.fixed_point.survival

    @property
    def contraction(self) -> float:
        return self.fixed_point.contraction

    @property
    def residual(self) -> float:
        return self.fixed_point.absolute_residual

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_core_exact_v1() -> CoreChain:
    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = sin2_theta_w * (1.0 - sin2_theta_w)
    d_eff = 3.0 + delta
    fixed_point = solve_low_extinction_root(
        d_eff,
        model_id="CE_CORE_EXACT_V1",
        precision="binary64 formula evaluation from alpha_s=0.11789",
    )
    return CoreChain(
        model_id="CE_CORE_EXACT_V1",
        role=RegistryRole.SCIENTIFIC_CORE,
        formal_status=FormalStatus.DERIVATION,
        status=ModelStatus.ACTIVE_MATHEMATICS,
        alpha_s=alpha_s,
        sin2_theta_w=sin2_theta_w,
        delta=delta,
        d_eff=d_eff,
        fixed_point=fixed_point,
        provenance=Provenance(
            source_id="R-U1-EXACT",
            source_kind="ce_formula_chain",
            source_path="docs/검증_원장/상수_우주론_원장.md",
            formula_version="alpha_s_to_sin2_to_delta_to_D_to_q/v1",
            precision="binary64; no intermediate decimal rounding",
            note="alpha_s scale/scheme and the sin2 mapping remain empirical model inputs",
        ),
    )


def _build_legacy_delta_5dp_v1() -> CoreChain:
    delta = 0.17776
    d_eff = 3.0 + delta
    fixed_point = solve_low_extinction_root(
        d_eff,
        model_id="LEGACY_DELTA_5DP_V1",
        precision="delta rounded to five decimal places before root solve",
    )
    return CoreChain(
        model_id="LEGACY_DELTA_5DP_V1",
        role=RegistryRole.COMPATIBILITY_BOUNDARY,
        formal_status=FormalStatus.AXIOM,
        status=ModelStatus.COMPATIBILITY_ONLY,
        alpha_s=0.11789,
        sin2_theta_w=4.0 * 0.11789 ** (4.0 / 3.0),
        delta=delta,
        d_eff=d_eff,
        fixed_point=fixed_point,
        provenance=Provenance(
            source_id="R-U1-LEGACY",
            source_kind="historical_rounded_input",
            source_path="reality_stone/python/reality_stone/clarus/bootstrap_solver.py",
            formula_version="legacy_delta_5dp/v1",
            precision="delta=0.17776 supplied at five decimal places",
            note="preserved for numerical regression; not the exact-chain value",
        ),
    )


CE_CORE_EXACT_V1 = _build_core_exact_v1()
LEGACY_DELTA_5DP_V1 = _build_legacy_delta_5dp_v1()


def q_ext_exact(core: CoreChain = CE_CORE_EXACT_V1) -> FixedPointResult:
    """Return an independently recomputed exact-chain extinction certificate."""

    if core.model_id != "CE_CORE_EXACT_V1":
        raise ValueError("q_ext_exact requires CE_CORE_EXACT_V1")
    return solve_low_extinction_root(
        core.d_eff,
        model_id=core.model_id,
        precision=core.fixed_point.precision,
    )


@dataclass(frozen=True)
class RuntimeRatioConfig:
    """Rounded product defaults retained as an explicit compatibility layer."""

    model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    active_ratio: float
    struct_ratio: float
    background_ratio: float
    contraction_display: float
    normalization_policy: str
    provenance: Provenance

    @property
    def raw_sum(self) -> float:
        return self.active_ratio + self.struct_ratio + self.background_ratio

    @property
    def raw_omega_m(self) -> float:
        return self.active_ratio + self.struct_ratio


LEGACY_ROUNDED_RUNTIME_V1 = RuntimeRatioConfig(
    model_id="LEGACY_ROUNDED_RUNTIME_V1",
    role=RegistryRole.COMPATIBILITY_BOUNDARY,
    formal_status=FormalStatus.AXIOM,
    status=ModelStatus.COMPATIBILITY_ONLY,
    active_ratio=0.0487,
    struct_ratio=0.2623,
    background_ratio=0.6891,
    contraction_display=0.155,
    normalization_policy=(
        "raw product targets; raw_sum=1.0001; do not use as an exactly normalized flat tuple"
    ),
    provenance=Provenance(
        source_id="R-U1-LEGACY",
        source_kind="runtime_compatibility",
        source_path="reality_stone/python/reality_stone/clarus/constants.py",
        formula_version="legacy_runtime_rounded/v1",
        precision="four decimal display values; contraction to three decimals",
        note="operational defaults, not a CE observational prediction",
    ),
)


@dataclass(frozen=True)
class DensityConfig:
    """Named density partition with an explicit model and activation boundary."""

    model_id: str
    core_model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    omega_b: float
    omega_c: float
    omega_lambda: float
    partition_formula: str
    provenance: Provenance

    @property
    def omega_m(self) -> float:
        return self.omega_b + self.omega_c

    @property
    def total(self) -> float:
        return self.omega_m + self.omega_lambda


def _density_from_dark_ratio(
    *,
    model_id: str,
    dark_ratio: float,
    role: RegistryRole,
    formal_status: FormalStatus,
    status: ModelStatus,
    formula: str,
    source_id: str,
    source_path: str,
    note: str,
) -> DensityConfig:
    q_ext = CE_CORE_EXACT_V1.q_ext
    complement = 1.0 - q_ext
    omega_c = complement * dark_ratio / (1.0 + dark_ratio)
    omega_lambda = complement / (1.0 + dark_ratio)
    return DensityConfig(
        model_id=model_id,
        core_model_id=CE_CORE_EXACT_V1.model_id,
        role=role,
        formal_status=formal_status,
        status=status,
        omega_b=q_ext,
        omega_c=omega_c,
        omega_lambda=omega_lambda,
        partition_formula=formula,
        provenance=Provenance(
            source_id=source_id,
            source_kind="named_density_partition",
            source_path=source_path,
            formula_version=model_id,
            precision="binary64 formula evaluation",
            note=note,
        ),
    )


_R_LO = CE_CORE_EXACT_V1.alpha_s * CE_CORE_EXACT_V1.d_eff
CE_DENSITY_LO_V1 = _density_from_dark_ratio(
    model_id="CE_DENSITY_LO_V1",
    dark_ratio=_R_LO,
    role=RegistryRole.CONDITIONAL_MODEL,
    formal_status=FormalStatus.AXIOM,
    status=ModelStatus.CONDITIONAL,
    formula="R_LO=alpha_s*D; omega_c/omega_lambda=R_LO; omega_b=q_ext",
    source_id="R-U3-RLO",
    source_path="examples/physics/cosmology.py",
    note="named toy partition; not selected as the scientific default",
)

CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1 = DensityConfig(
    model_id="CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1",
    core_model_id=CE_CORE_EXACT_V1.model_id,
    role=RegistryRole.HISTORICAL_MODEL,
    formal_status=FormalStatus.EMPIRICAL,
    status=ModelStatus.HISTORICAL,
    omega_b=CE_CORE_EXACT_V1.q_ext,
    omega_c=0.2622797333,
    omega_lambda=0.6890735470,
    partition_formula="historical supplied coupling sum=1.0147344271",
    provenance=Provenance(
        source_id="R-U1-LEGACY",
        source_kind="historical_manuscript_witness",
        source_path="docs/2_경로적분과_응용/validate_manuscript.py",
        formula_version="density_three_layer_manuscript/v1",
        precision="supplied manuscript decimals",
        note="historical witness; active_parent=False",
    ),
)

CE_DENSITY_THREE_LAYER_APPROX_V1 = DensityConfig(
    model_id="CE_DENSITY_THREE_LAYER_APPROX_V1",
    core_model_id=CE_CORE_EXACT_V1.model_id,
    role=RegistryRole.HISTORICAL_MODEL,
    formal_status=FormalStatus.EMPIRICAL,
    status=ModelStatus.HISTORICAL,
    omega_b=CE_CORE_EXACT_V1.q_ext,
    omega_c=0.26228049346744653,
    omega_lambda=0.6890727868885254,
    partition_formula="historical rounded coupling sum=1.015",
    provenance=Provenance(
        source_id="R-U1-LEGACY",
        source_kind="historical_discrimination_witness",
        source_path="examples/physics/cosmology_discrimination_gates.py",
        formula_version="density_three_layer_approx/v1",
        precision="ratio_sum supplied as 1.015",
        note="historical approximation; excluded from active model selection",
    ),
)

_R_NLO = _R_LO + _R_LO * _R_LO / (4.0 * math.pi)
CE_DENSITY_NLO_CANDIDATE_V1 = _density_from_dark_ratio(
    model_id="CE_DENSITY_NLO_CANDIDATE_V1",
    dark_ratio=_R_NLO,
    role=RegistryRole.RESEARCH_CANDIDATE,
    formal_status=FormalStatus.EMPIRICAL,
    status=ModelStatus.CANDIDATE,
    formula="R_NLO=R_LO+R_LO^2/(4*pi); omega_c/omega_lambda=R_NLO; omega_b=q_ext",
    source_id="R-U3-RLO",
    source_path="examples/physics/cosmology_discrimination_gates.py",
    note="explicit opt-in candidate; no blind selection has been performed",
)

# Short historical spellings are aliases to the named entries, not copies.
CE_DENSITY_3L_MANUSCRIPT_V1 = CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1
CE_DENSITY_3L_APPROX_V1 = CE_DENSITY_THREE_LAYER_APPROX_V1


@dataclass(frozen=True)
class FlatBackgroundConfig:
    """Flat late-time normalization of the rounded runtime boundary."""

    model_id: str
    source_model_id: str
    role: RegistryRole
    formal_status: FormalStatus
    status: ModelStatus
    raw_omega_m: float
    raw_omega_lambda: float
    raw_sum: float
    omega_m: float
    omega_lambda: float
    normalization_policy: str
    provenance: Provenance


_RUNTIME_RAW_M = LEGACY_ROUNDED_RUNTIME_V1.raw_omega_m
_RUNTIME_RAW_SUM = _RUNTIME_RAW_M + LEGACY_ROUNDED_RUNTIME_V1.background_ratio
CE_RESIDUAL_FLAT_LCDM_GR_V1 = FlatBackgroundConfig(
    model_id="CE_RESIDUAL_FLAT_LCDM_GR_V1",
    source_model_id=LEGACY_ROUNDED_RUNTIME_V1.model_id,
    role=RegistryRole.CONDITIONAL_MODEL,
    formal_status=FormalStatus.DERIVATION,
    status=ModelStatus.CONDITIONAL,
    raw_omega_m=_RUNTIME_RAW_M,
    raw_omega_lambda=LEGACY_ROUNDED_RUNTIME_V1.background_ratio,
    raw_sum=_RUNTIME_RAW_SUM,
    omega_m=_RUNTIME_RAW_M / _RUNTIME_RAW_SUM,
    omega_lambda=LEGACY_ROUNDED_RUNTIME_V1.background_ratio / _RUNTIME_RAW_SUM,
    normalization_policy="normalize only the late-time matter+Lambda background to unity",
    provenance=Provenance(
        source_id="R-U8-ORDER",
        source_kind="conditional_forward_model_boundary",
        source_path="examples/physics/ce_residual_forward_model.py",
        formula_version="legacy_boundary_flat_normalized/v1",
        precision="binary64 normalization of the raw four-decimal runtime tuple",
        note="raw and normalized values are both serialized to prevent silent mixing",
    ),
)


@dataclass(frozen=True)
class RouteClaim:
    """Target/route status kept independently from numerical configurations."""

    claim_id: str
    kind: str
    role: RegistryRole
    formal_status: FormalStatus
    priority: str
    active_scientific_claim: bool
    summary: str
    boundary: str
    reopen_condition: str = ""


_ROUTE_CLAIMS = {
    "T-U1-CANON": RouteClaim(
        "T-U1-CANON",
        "TARGET-HYPOTHESIS",
        RegistryRole.RESEARCH_TARGET,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Unify canonical and historical cosmology versions without deleting witnesses.",
        "Registry implemented; full consumer migration remains incomplete.",
    ),
    "R-U1-Q-DEF": RouteClaim(
        "R-U1-Q-DEF",
        "ROUTE",
        RegistryRole.SCIENTIFIC_CORE,
        FormalStatus.DEFINITION,
        "P2",
        True,
        "q_ext is extinction; survival is 1-q_ext.",
        "A legacy symbol may alias q_ext but may never be labelled survival.",
    ),
    "R-U1-Q-THM": RouteClaim(
        "R-U1-Q-THM",
        "ROUTE",
        RegistryRole.SCIENTIFIC_CORE,
        FormalStatus.THEOREM,
        "PASS",
        True,
        "For D>1 the stable non-identity root lies in (0,1/D) and D*q_ext<1.",
        "This theorem does not identify a cosmological species or density.",
    ),
    "T-U2-ABS": RouteClaim(
        "T-U2-ABS",
        "TARGET-HYPOTHESIS",
        RegistryRole.RESEARCH_TARGET,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Derive absolute baryon abundance without inserting the observed density.",
        "Mass, total yield, entropy, freeze surface, and H_* remain open.",
    ),
    "R-U2-DIRECT": RouteClaim(
        "R-U2-DIRECT",
        "ROUTE",
        RegistryRole.HISTORICAL_MODEL,
        FormalStatus.AXIOM,
        "P0-CLOSED",
        False,
        "LEGACY_DIRECT_READOUT_V1 identifies q_ext with present Omega_b.",
        "Preserved only as a named historical axiom, not a theorem or prediction.",
        "Requires a new current/action/yield/critical-density bridge.",
    ),
    "R-U2-GW": RouteClaim(
        "R-U2-GW",
        "ROUTE",
        RegistryRole.SCIENTIFIC_CORE,
        FormalStatus.THEOREM,
        "PASS",
        True,
        "Conditioned Poisson offspring law has mean D*q_ext.",
        "No species or density bridge is included.",
    ),
    "R-U2-COMP": RouteClaim(
        "R-U2-COMP",
        "ROUTE",
        RegistryRole.CONDITIONAL_MODEL,
        FormalStatus.DERIVATION,
        "P1",
        True,
        "Aggregate equal-node-energy descendant fraction is D*q_ext.",
        "Requires the equal-energy conserved-relic and detector axioms.",
    ),
    "R-U2-REACT": RouteClaim(
        "R-U2-REACT",
        "ROUTE",
        RegistryRole.RESEARCH_CANDIDATE,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Reacting-current bridge candidate.",
        "Microscopic collision or Schwinger-Keldysh action is not fixed.",
    ),
    "R-U2-FREEZE": RouteClaim(
        "R-U2-FREEZE",
        "ROUTE",
        RegistryRole.RESEARCH_CANDIDATE,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Freeze-out abundance route.",
        "Yield, mass, stoichiometry, and dilution remain free.",
    ),
    "T-U7-PROV": RouteClaim(
        "T-U7-PROV",
        "TARGET-HYPOTHESIS",
        RegistryRole.RESEARCH_TARGET,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Version official releases, covariance assets, hashes, and blind roles.",
        "Manifest exists; locally hashed likelihood/covariance migration is incomplete.",
    ),
    "R-U7-HYBRID": RouteClaim(
        "R-U7-HYBRID",
        "ROUTE",
        RegistryRole.HISTORICAL_MODEL,
        FormalStatus.EXCLUDED,
        "P0-CLOSED",
        False,
        "The mixed Planck/ACT/SPT tuple is not a single official posterior.",
        "Historical display only; excluded from scientific scoring.",
    ),
    "R-U7-HOLDOUT": RouteClaim(
        "R-U7-HOLDOUT",
        "ROUTE",
        RegistryRole.RESEARCH_TARGET,
        FormalStatus.INCOMPLETE,
        "P1",
        True,
        "Obtain a preregistered independent confirmatory holdout.",
        "Qualifying holdout count is zero; gate is NOT_READY.",
    ),
    "R-U7-PRED": RouteClaim(
        "R-U7-PRED",
        "ROUTE",
        RegistryRole.HISTORICAL_MODEL,
        FormalStatus.EXCLUDED,
        "P0-CLOSED",
        False,
        "Current numerical proximity is a blind prediction.",
        "Target-aware/exploratory comparisons are retained without prediction status.",
    ),
}
ROUTE_CLAIMS: Mapping[str, RouteClaim] = MappingProxyType(_ROUTE_CLAIMS)

# Named route handles keep historical and active research routes addressable
# without merging their formal status.
LEGACY_DIRECT_READOUT_V1 = ROUTE_CLAIMS["R-U2-DIRECT"]
CONDITIONED_POISSON_COMPOSITION_V1 = ROUTE_CLAIMS["R-U2-COMP"]
REACTING_CURRENT_RESEARCH_V1 = ROUTE_CLAIMS["R-U2-REACT"]
FREEZE_OUT_RESEARCH_V1 = ROUTE_CLAIMS["R-U2-FREEZE"]

# The historical epsilon spelling is an extinction-root alias only.
EPSILON_SQUARED_LEGACY = LEGACY_DELTA_5DP_V1.q_ext


@dataclass(frozen=True)
class ObservationRef:
    """Pointer to the separate observation manifest; contains no observed value."""

    manifest_id: str
    manifest_path: str
    role: RegistryRole
    formal_status: FormalStatus
    qualifying_independent_holdout_count: int
    holdout_gate_status: str


COSMOLOGY_OBSERVATIONS_V1 = ObservationRef(
    manifest_id="CE_COSMOLOGY_OBSERVATIONS_V1",
    manifest_path="benchmarks/cosmology/observations_v1.json",
    role=RegistryRole.OBSERVATION_REFERENCE,
    formal_status=FormalStatus.INCOMPLETE,
    qualifying_independent_holdout_count=0,
    holdout_gate_status="NOT_READY",
)


CORE_MODELS: Mapping[str, CoreChain] = MappingProxyType(
    {
        CE_CORE_EXACT_V1.model_id: CE_CORE_EXACT_V1,
        LEGACY_DELTA_5DP_V1.model_id: LEGACY_DELTA_5DP_V1,
    }
)
DENSITY_MODELS: Mapping[str, DensityConfig] = MappingProxyType(
    {
        item.model_id: item
        for item in (
            CE_DENSITY_LO_V1,
            CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1,
            CE_DENSITY_THREE_LAYER_APPROX_V1,
            CE_DENSITY_NLO_CANDIDATE_V1,
        )
    }
)

# No density partition is promoted to a scientific default before blind model
# selection.  Product code continues through the explicit compatibility layer.
SCIENTIFIC_DENSITY_DEFAULT: None = None
RUNTIME_COMPATIBILITY_DEFAULT = LEGACY_ROUNDED_RUNTIME_V1


__all__ = [
    "CE_CORE_EXACT_V1",
    "LEGACY_DELTA_5DP_V1",
    "LEGACY_ROUNDED_RUNTIME_V1",
    "CE_DENSITY_LO_V1",
    "CE_DENSITY_THREE_LAYER_MANUSCRIPT_V1",
    "CE_DENSITY_THREE_LAYER_APPROX_V1",
    "CE_DENSITY_3L_MANUSCRIPT_V1",
    "CE_DENSITY_3L_APPROX_V1",
    "CE_DENSITY_NLO_CANDIDATE_V1",
    "CE_RESIDUAL_FLAT_LCDM_GR_V1",
    "COSMOLOGY_OBSERVATIONS_V1",
    "LEGACY_DIRECT_READOUT_V1",
    "CONDITIONED_POISSON_COMPOSITION_V1",
    "REACTING_CURRENT_RESEARCH_V1",
    "FREEZE_OUT_RESEARCH_V1",
    "EPSILON_SQUARED_LEGACY",
    "CORE_MODELS",
    "DENSITY_MODELS",
    "ROUTE_CLAIMS",
    "SCIENTIFIC_DENSITY_DEFAULT",
    "RUNTIME_COMPATIBILITY_DEFAULT",
    "CoreChain",
    "DensityConfig",
    "FixedPointResult",
    "FlatBackgroundConfig",
    "FormalStatus",
    "ModelStatus",
    "ObservationRef",
    "Provenance",
    "RegistryRole",
    "RouteClaim",
    "RuntimeRatioConfig",
    "bootstrap_residual",
    "q_ext_exact",
    "solve_low_extinction_root",
]
