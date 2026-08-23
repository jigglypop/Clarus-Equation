"""Typed, non-destructive registry for CE cosmology readouts.

This module now owns only the cosmology readout layer (density partitions,
route claims, observation references).  The fixed-point core moved to
:mod:`reality_stone.clarus.core_registry` and the runtime target tuple moved
to :mod:`reality_stone.clarus.runtime_targets`; both are re-exported here as
a compatibility shim.

The registry deliberately keeps different numerical layers separate:

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

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

def _load_sibling_by_path(filename: str, module_key: str):
    """Load a sibling stdlib-only module by file path (no package context).

    Audit tooling loads this registry via ``spec_from_file_location`` without
    a parent package, so relative imports are not always available.
    """

    import importlib.util
    import os
    import sys

    cached = sys.modules.get(module_key)
    if cached is not None:
        return cached
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling registry module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_key, None)
        raise
    return module


# compatibility shim — canonical: core_registry/runtime_targets (멀티레포 분리 시 제거)
try:
    from .core_registry import (
        CE_CORE_EXACT_V1,
        LEGACY_DELTA_5DP_V1,
        CoreChain,
        FixedPointResult,
        FormalStatus,
        ModelStatus,
        Provenance,
        RegistryRole,
        bootstrap_residual,
        q_ext_exact,
        solve_low_extinction_root,
    )
except ImportError:  # loaded by file path without a parent package
    _core = _load_sibling_by_path("core_registry.py", "_ce_core_registry_v1")
    CE_CORE_EXACT_V1 = _core.CE_CORE_EXACT_V1
    LEGACY_DELTA_5DP_V1 = _core.LEGACY_DELTA_5DP_V1
    CoreChain = _core.CoreChain
    FixedPointResult = _core.FixedPointResult
    FormalStatus = _core.FormalStatus
    ModelStatus = _core.ModelStatus
    Provenance = _core.Provenance
    RegistryRole = _core.RegistryRole
    bootstrap_residual = _core.bootstrap_residual
    q_ext_exact = _core.q_ext_exact
    solve_low_extinction_root = _core.solve_low_extinction_root

# compatibility shim — canonical: core_registry/runtime_targets (멀티레포 분리 시 제거)
try:
    from .runtime_targets import (
        LEGACY_ROUNDED_RUNTIME_V1,
        RuntimeRatioConfig,
    )
except ImportError:  # loaded by file path without a parent package
    _runtime_targets = _load_sibling_by_path("runtime_targets.py", "_ce_runtime_targets_v1")
    LEGACY_ROUNDED_RUNTIME_V1 = _runtime_targets.LEGACY_ROUNDED_RUNTIME_V1
    RuntimeRatioConfig = _runtime_targets.RuntimeRatioConfig


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
