"""Audit CE density ratios against recent cosmology constraints.

This is a deliberately thin checker. It tests the part the current code can
actually support: present-day density ratios. It also records which modern
cosmology/dark-matter claims are out of scope because the repo has no forward
model for them yet.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONSTANTS_PATH = ROOT / "reality_stone" / "python" / "reality_stone" / "clarus" / "constants.py"


def load_ce_ratios_from_constants() -> dict[str, float]:
    """Read CE ratio constants without importing torch-heavy package modules."""
    tree = ast.parse(CONSTANTS_PATH.read_text(encoding="utf-8"))
    names = {"ACTIVE_RATIO", "STRUCT_RATIO", "BACKGROUND_RATIO"}
    values: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in names:
            continue
        literal = ast.literal_eval(node.value)
        values[node.target.id] = float(literal)
    missing = names - values.keys()
    if missing:
        raise ValueError(f"missing CE constants: {sorted(missing)}")
    return {
        "omega_b": values["ACTIVE_RATIO"],
        "omega_c": values["STRUCT_RATIO"],
        "omega_lambda": values["BACKGROUND_RATIO"],
    }


@dataclass(frozen=True)
class DensityBaseline:
    name: str
    omega_b_h2: float
    omega_c_h2: float
    h0: float
    note: str

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


RECENT_BASELINES = (
    DensityBaseline(
        name="Planck2018_base",
        omega_b_h2=0.0224,
        omega_c_h2=0.120,
        h0=67.4,
        note="Planck base-LambdaCDM reference.",
    ),
    DensityBaseline(
        name="Planck_ACT_SPT_combined",
        omega_b_h2=0.02228,
        omega_c_h2=0.1195,
        h0=68.43,
        note="Representative combined CMB constraint used in 2025-2026 comparisons.",
    ),
    DensityBaseline(
        name="ACT_DR6_DESI_reported",
        omega_b_h2=0.0226,
        omega_c_h2=0.118,
        h0=68.22,
        note="Representative ACT DR6 plus DESI compressed parameter set.",
    ),
    DensityBaseline(
        name="SPT3G_CMBSPA",
        omega_b_h2=0.022398,
        omega_c_h2=0.12028,
        h0=67.19,
        note="Representative SPT-3G/CMB-SPA-style compressed set.",
    ),
)


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
    comparisons = compare_all_density_ratios()
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
    print("| baseline | dOmega_b | rel_b | dOmega_DM | rel_DM | dOmega_Lambda | rel_Lambda |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for comparison in compare_all_density_ratios():
        print(
            f"| {comparison.baseline} "
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


def main() -> int:
    print_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
