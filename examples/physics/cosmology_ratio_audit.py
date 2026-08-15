"""Audit CE density ratios against recent cosmology constraints.

This is a deliberately thin checker. It tests the part the current code can
actually support: present-day density ratios. It also records which modern
cosmology/dark-matter claims are out of scope because the repo has no forward
model for them yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    ROOT
    / "reality_stone"
    / "python"
    / "reality_stone"
    / "clarus"
    / "cosmology_registry.py"
)
OBSERVATION_MANIFEST_PATH = ROOT / "benchmarks" / "cosmology" / "observations_v1.json"
_REGISTRY_MODULE_NAME = "_ce_cosmology_registry_v1"


def _load_registry_module() -> ModuleType:
    """Load the stdlib-only registry without importing the package facade."""

    cached = sys.modules.get(_REGISTRY_MODULE_NAME)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(_REGISTRY_MODULE_NAME, REGISTRY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CE cosmology registry: {REGISTRY_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # module is executed, so register the lightweight module before exec.
    sys.modules[_REGISTRY_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_REGISTRY_MODULE_NAME, None)
        raise
    return module


def load_ce_ratios_from_constants() -> dict[str, float]:
    """Read the named compatibility triplet without importing torch modules.

    The public function name is preserved for downstream callers.  Its source
    is now the typed registry rather than AST-parsing assignment literals.
    """

    runtime = _load_registry_module().LEGACY_ROUNDED_RUNTIME_V1
    return {
        "omega_b": float(runtime.active_ratio),
        "omega_c": float(runtime.struct_ratio),
        "omega_lambda": float(runtime.background_ratio),
    }


def load_observation_manifest() -> dict[str, Any]:
    """Load and minimally validate the versioned observation manifest."""

    payload = json.loads(OBSERVATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != "CE_COSMOLOGY_OBSERVATIONS_V1":
        raise ValueError("unexpected cosmology observation manifest_id")
    required = set(payload["provenance_policy"]["required_entry_fields"])
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("cosmology observation manifest requires an observations list")
    for entry in observations:
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f"observation {entry.get('observation_id', '<unknown>')} "
                f"is missing provenance fields: {sorted(missing)}"
            )
    return payload


@dataclass(frozen=True)
class DensityBaseline:
    name: str
    omega_b_h2: float
    omega_c_h2: float
    h0: float
    note: str
    validity_status: str = "VALID_REFERENCE"
    scientific_score_eligible: bool = False

    @property
    def h(self) -> float:
        return self.h0 / 100.0

    @property
    def omega_b(self) -> float:
        return self.omega_b_h2 / (self.h * self.h)

    @property
    def omega_c(self) -> float:
        return self.omega_c_h2 / (self.h * self.h)

    @property
    def omega_lambda_flat(self) -> float:
        return 1.0 - self.omega_b - self.omega_c


@dataclass(frozen=True)
class RatioComparison:
    baseline: str
    omega_b_diff: float
    omega_c_diff: float
    omega_lambda_diff: float
    omega_b_rel: float
    omega_c_rel: float
    omega_lambda_rel: float

    @property
    def max_abs_relative_error(self) -> float:
        return max(abs(self.omega_b_rel), abs(self.omega_c_rel), abs(self.omega_lambda_rel))


@dataclass(frozen=True)
class CoverageVerdict:
    density_ratios_close: bool
    has_background_expansion_model: bool
    has_growth_model_for_s8: bool
    has_particle_dark_matter_model: bool
    has_detector_likelihood: bool
    scientific_score_eligible: bool = False
    closure_role: str = "exploratory_ratio_diagnostic"

    @property
    def summary(self) -> str:
        if not self.density_ratios_close:
            return "density-ratio mismatch"
        if any((
            self.has_background_expansion_model,
            self.has_growth_model_for_s8,
            self.has_particle_dark_matter_model,
            self.has_detector_likelihood,
        )):
            return "partial physical forward model"
        return "density ratios match; modern likelihood physics not implemented"


CE_RATIOS = load_ce_ratios_from_constants()

_BASELINE_NOTES = {
    "Planck2018_base": "Planck base-LambdaCDM reference.",
    "Planck_ACT_SPT_combined": (
        "Historical mixed tuple; no single official posterior or covariance."
    ),
    "ACT_DR6_DESI_reported": "ACT DR6.02 tagged compressed reference.",
    "SPT3G_CMBSPA": "Corrected SPT-3G D1 CMB-SPA compressed reference.",
}


def _load_recent_baselines() -> tuple[DensityBaseline, ...]:
    manifest = load_observation_manifest()
    by_id = {entry["observation_id"]: entry for entry in manifest["observations"]}
    baselines: list[DensityBaseline] = []
    for observation_id in manifest["legacy_ratio_baseline_ids"]:
        try:
            entry = by_id[observation_id]
            values = entry["values"]
            validity = entry["validity"]
            baselines.append(
                DensityBaseline(
                    name=observation_id,
                    omega_b_h2=float(values["omega_b_h2"]),
                    omega_c_h2=float(values["omega_c_h2"]),
                    h0=float(values["H0"]),
                    note=_BASELINE_NOTES[observation_id],
                    validity_status=str(validity["status"]),
                    scientific_score_eligible=bool(validity["scientific_score_eligible"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"invalid ratio baseline manifest entry: {observation_id}") from exc
    return tuple(baselines)


RECENT_BASELINES = _load_recent_baselines()


def relative_error(predicted: float, observed: float) -> float:
    if observed == 0.0:
        return 0.0
    return predicted / observed - 1.0


def compare_density_ratios(baseline: DensityBaseline) -> RatioComparison:
    return RatioComparison(
        baseline=baseline.name,
        omega_b_diff=CE_RATIOS["omega_b"] - baseline.omega_b,
        omega_c_diff=CE_RATIOS["omega_c"] - baseline.omega_c,
        omega_lambda_diff=CE_RATIOS["omega_lambda"] - baseline.omega_lambda_flat,
        omega_b_rel=relative_error(CE_RATIOS["omega_b"], baseline.omega_b),
        omega_c_rel=relative_error(CE_RATIOS["omega_c"], baseline.omega_c),
        omega_lambda_rel=relative_error(CE_RATIOS["omega_lambda"], baseline.omega_lambda_flat),
    )


def compare_all_density_ratios(
    baselines: tuple[DensityBaseline, ...] = RECENT_BASELINES,
) -> tuple[RatioComparison, ...]:
    return tuple(compare_density_ratios(baseline) for baseline in baselines)


def coverage_verdict(max_relative_tolerance: float = 0.04) -> CoverageVerdict:
    valid_names = {
        baseline.name
        for baseline in RECENT_BASELINES
        if baseline.validity_status != "EXCLUDED_HISTORICAL"
    }
    comparisons = tuple(
        comparison
        for comparison in compare_all_density_ratios()
        if comparison.baseline in valid_names
    )
    if not comparisons:
        raise ValueError("no valid density reference remains after provenance filtering")
    density_close = all(c.max_abs_relative_error <= max_relative_tolerance for c in comparisons)
    return CoverageVerdict(
        density_ratios_close=density_close,
        has_background_expansion_model=False,
        has_growth_model_for_s8=False,
        has_particle_dark_matter_model=False,
        has_detector_likelihood=False,
    )


def print_report() -> None:
    print("# CE Cosmology Ratio Audit")
    print()
    print("CE ratios")
    print(f"  Omega_b      {CE_RATIOS['omega_b']:.6f}")
    print(f"  Omega_DM     {CE_RATIOS['omega_c']:.6f}")
    print(f"  Omega_Lambda {CE_RATIOS['omega_lambda']:.6f}")
    print(f"  Omega_m      {CE_RATIOS['omega_b'] + CE_RATIOS['omega_c']:.6f}")
    print()
    print("Historical/excluded rows are displayed for parity but omitted from the verdict.")
    print("| baseline | dOmega_b | rel_b | dOmega_DM | rel_DM | dOmega_Lambda | rel_Lambda |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    baseline_by_name = {baseline.name: baseline for baseline in RECENT_BASELINES}
    for comparison in compare_all_density_ratios():
        baseline = baseline_by_name[comparison.baseline]
        display_name = comparison.baseline
        if baseline.validity_status == "EXCLUDED_HISTORICAL":
            display_name += " [EXCLUDED_HISTORICAL]"
        print(
            f"| {display_name} "
            f"| {comparison.omega_b_diff:+.6f} | {100.0 * comparison.omega_b_rel:+.2f}% "
            f"| {comparison.omega_c_diff:+.6f} | {100.0 * comparison.omega_c_rel:+.2f}% "
            f"| {comparison.omega_lambda_diff:+.6f} | {100.0 * comparison.omega_lambda_rel:+.2f}% |"
        )
    print()
    verdict = coverage_verdict()
    print("verdict", verdict.summary)
    print("density_ratios_close", verdict.density_ratios_close)
    print("has_background_expansion_model", verdict.has_background_expansion_model)
    print("has_growth_model_for_s8", verdict.has_growth_model_for_s8)
    print("has_particle_dark_matter_model", verdict.has_particle_dark_matter_model)
    print("has_detector_likelihood", verdict.has_detector_likelihood)
    print("scientific_score_eligible", verdict.scientific_score_eligible)
    print("closure_role", verdict.closure_role)


def main() -> int:
    print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
