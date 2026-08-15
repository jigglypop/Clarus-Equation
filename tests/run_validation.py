#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the bootstrap, canonical constants, and dimensionless CE validators."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reality_stone.clarus.dimensionless import (  # noqa: E402
    MASS,
    TEMPERATURE,
    GateResult,
    Quantity,
    audit_dimensionless,
    check_dimensionless,
    dim,
    exp_arguments,
    group_dimension,
)
from reality_stone.clarus.cosmology_registry import (  # noqa: E402
    LEGACY_DELTA_5DP_V1,
)
from tests.scorecard import ConstantsScorecard  # noqa: E402


class BootstrapValidator:
    """Pure-Python validator for the versioned legacy fixed-point route."""

    MODEL = LEGACY_DELTA_5DP_V1
    DELTA = MODEL.delta
    D_EFF = MODEL.d_eff
    LEGACY_OMEGA_B_DISPLAY = 0.04865

    def equation(self, eps: float) -> float:
        """Return f(eps) for eps = exp(-(1-eps) * D_eff)."""

        return eps - math.exp(-(1 - eps) * self.D_EFF)

    def derivative(self, eps: float) -> float:
        """Return f'(eps)."""

        return 1 - self.D_EFF * math.exp(-(1 - eps) * self.D_EFF)

    def newton_solve(
        self,
        x0: float = 0.05,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> tuple[float, int]:
        """Solve the fixed-point equation with Newton-Raphson."""

        x = x0
        for iteration in range(max_iter):
            x_new = x - self.equation(x) / self.derivative(x)
            if abs(x_new - x) < tol:
                return x_new, iteration
            x = x_new
        raise RuntimeError(f"Failed to converge after {max_iter} iterations")

    def validate(self) -> dict[str, Any]:
        """Certify the equation without promoting q_ext to a density prediction."""

        eps, iterations = self.newton_solve()
        lhs = eps
        rhs = math.exp(-(1 - eps) * self.D_EFF)
        residual = abs(lhs - rhs)
        equation_satisfied = residual < 1e-9

        return {
            "solution": eps,
            "q_ext": eps,
            "survival": 1.0 - eps,
            "iterations": iterations,
            "lhs": lhs,
            "rhs": rhs,
            "residual": residual,
            "equation_satisfied": equation_satisfied,
            "model_id": self.MODEL.model_id,
            "model_role": self.MODEL.role.value,
            "model_status": self.MODEL.status.value,
            "quantity_semantics": "branching_extinction_probability",
            # Compatibility display only: there is no covariance attached to
            # this rounded value, so a sigma score would be fabricated.
            "omega_b_obs": self.LEGACY_OMEGA_B_DISPLAY,
            "legacy_omega_b_display": self.LEGACY_OMEGA_B_DISPLAY,
            "sigma_offset": None,
            "observation_comparison_status": (
                "historical_display_only_no_covariance"
            ),
            "physical_bridge_complete": False,
            "scientific_density_prediction": False,
            # Keep the historical machine key, but make it an equation-only
            # result rather than an observational or physical verdict.
            "pass": equation_satisfied,
        }


class ConstantsValidator:
    """Adapt the canonical constants scorecard for the combined validation report."""

    def __init__(self, scorecard: ConstantsScorecard | None = None) -> None:
        self.scorecard = scorecard or ConstantsScorecard()

    def validate_all(self) -> dict[str, Any]:
        """Return canonical rows without copying observations or grading rules."""

        results = [
            {
                "name": constant.name,
                "symbol": constant.symbol,
                "ce": constant.ce_value,
                "obs": constant.obs_value,
                "obs_sigma": constant.obs_sigma,
                "sigma_offset": constant.sigma_offset,
                "rel_error": constant.relative_error,
                "grade": constant.grade,
                "role": constant.role,
                "status": constant.status,
                "scoreable": constant.scoreable,
                "is_scored": constant.is_scored,
                "source": constant.source,
                "notes": constant.notes,
            }
            for constant in self.scorecard.constants
        ]
        return {"results": results, "summary": self.scorecard.summary()}


class DimensionalValidator:
    """Evaluate CE formula dimensions through the dimensionless checker."""

    @staticmethod
    def _gate_info(result: GateResult[Any], success_note: str) -> dict[str, Any]:
        return {
            "dims_correct": result.passed,
            "note": success_note if result.passed else "; ".join(result.errors),
            "errors": result.errors,
        }

    @staticmethod
    def _expected_dimension_info(
        actual: tuple[Fraction, ...],
        expected: tuple[Fraction, ...],
        success_note: str,
    ) -> dict[str, Any]:
        passed = actual == expected
        error = () if passed else (f"expected dims={expected}, got dims={actual}",)
        return {
            "dims_correct": passed,
            "note": success_note if passed else error[0],
            "errors": error,
        }

    def validate_all(self) -> dict[str, Any]:
        """Run live dimension-vector checks for every reported formula."""

        delta = Quantity("delta", 0.17776)
        d_eff = Quantity("D_eff", 3.17776)
        q_ext = Quantity("q_ext", LEGACY_DELTA_5DP_V1.q_ext)
        alpha_s = Quantity("alpha_s", 0.11789)
        proton_mass = Quantity("m_p", 938.2720813, MASS)
        clarus_mass = Quantity("m_phi", 29.648, MASS)
        boltzmann = Quantity("k_B", 1.380649e-23, dim(1, 0, 0, -1))
        critical_temperature = Quantity("T_c", 300.0, TEMPERATURE)
        xi = Quantity("xi", 1.0)
        omega = Quantity("Omega", 1.0)

        formulas: dict[str, dict[str, Any]] = {}

        formulas["S(D) = exp(-D)"] = self._gate_info(
            exp_arguments((Quantity("D", 3.0),)),
            "exponential argument D is dimensionless",
        )

        formulas["D_eff = 3 + delta"] = self._gate_info(
            audit_dimensionless(
                (Quantity("3", 3.0), delta),
                context="D_eff additive terms",
            ),
            "all additive terms are dimensionless",
        )

        bootstrap_factors = (Quantity("1-q_ext", 1 - q_ext.value), d_eff)
        bootstrap_factor_gate = audit_dimensionless(
            bootstrap_factors,
            context="bootstrap exponential factors",
        )
        if bootstrap_factor_gate.passed:
            bootstrap_dims = group_dimension(
                bootstrap_factors,
                {"1-q_ext": Fraction(1), "D_eff": Fraction(1)},
            )
            bootstrap_gate = exp_arguments(
                (Quantity("(1-q_ext)*D_eff", 1.0, bootstrap_dims),)
            )
        else:
            bootstrap_gate = bootstrap_factor_gate
        formulas["q_ext = exp(-(1-q_ext)*D_eff)"] = self._gate_info(
            bootstrap_gate,
            "bootstrap exponential argument is dimensionless",
        )

        weak_angle_dims = group_dimension(
            (alpha_s,),
            {"alpha_s": Fraction(4, 3)},
        )
        formulas["sin^2(theta_W) = 4*alpha_s^(4/3)"] = self._gate_info(
            check_dimensionless(
                Quantity("4*alpha_s^(4/3)", 1.0, weak_angle_dims),
                context="weak-angle bridge",
            ),
            "fractional-power bridge is dimensionless",
        )

        mass_readout_dims = group_dimension(
            (proton_mass, delta),
            {"m_p": Fraction(1), "delta": Fraction(2)},
        )
        formulas["m_phi = m_p * delta^2"] = self._expected_dimension_info(
            mass_readout_dims,
            MASS,
            "right-hand side has mass dimension",
        )

        thermal_argument_gate = exp_arguments((Quantity("thermal exponent", 1.0),))
        thermal_rhs_dims = group_dimension(
            (Quantity("prefactor", 1.0), clarus_mass),
            {"prefactor": Fraction(1), "m_phi": Fraction(1)},
        )
        thermal_lhs_dims = group_dimension(
            (boltzmann, critical_temperature),
            {"k_B": Fraction(1), "T_c": Fraction(1)},
        )
        if thermal_argument_gate.passed:
            formulas["k_B*T_c = prefactor*m_phi*exp(...)"] = self._expected_dimension_info(
                thermal_rhs_dims,
                thermal_lhs_dims,
                "both sides have energy dimension and the exponent is dimensionless",
            )
        else:
            formulas["k_B*T_c = prefactor*m_phi*exp(...)"] = self._gate_info(
                thermal_argument_gate,
                "thermal exponential argument is dimensionless",
            )

        equation_of_state_dims = group_dimension(
            (xi, omega),
            {"xi": Fraction(2), "Omega": Fraction(-1)},
        )
        formulas["w_0 = -2*xi^2/(3*Omega)"] = self._gate_info(
            check_dimensionless(
                Quantity("xi^2/Omega", 1.0, equation_of_state_dims),
                context="equation-of-state readout",
            ),
            "equation-of-state ratio is dimensionless",
        )

        passed = sum(info["dims_correct"] for info in formulas.values())
        total = len(formulas)
        return {
            "formulas": formulas,
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
        }


def _overall_status(
    bootstrap_result: dict[str, Any],
    constants_summary: dict[str, Any],
    dimensional_result: dict[str, Any],
) -> str:
    if not bootstrap_result["pass"] or not dimensional_result["all_pass"]:
        return "FAIL"
    return str(constants_summary["status"])


def main() -> dict[str, Any]:
    """Run all validators and print an honesty-preserving combined report."""

    print("=" * 96)
    print("CLARUS EQUATION VALIDATION SUITE")
    print("=" * 96)

    print("\n[1] BOOTSTRAP FIXED-POINT SOLVER")
    print("-" * 96)
    validator = BootstrapValidator()
    result = validator.validate()
    print(f"Model = {result['model_id']} ({result['model_status']})")
    print(f"Extinction root q_ext = {result['q_ext']:.10f}")
    print(f"Survival 1-q_ext = {result['survival']:.10f}")
    print(f"Residual |LHS-RHS| = {result['residual']:.2e}")
    print(f"Equation satisfied? {result['equation_satisfied']}")
    print(
        "Historical Omega_b display = "
        f"{result['legacy_omega_b_display']:.5f} (comparison excluded)"
    )
    print("Sigma offset = not computed (no covariance attached)")
    print(f"Equation status: {'PASS' if result['pass'] else 'FAIL'}")
    print("Physical density bridge: INCOMPLETE")

    print("\n[2] CANONICAL CONSTANTS SCORECARD")
    print("-" * 96)
    const_result = ConstantsValidator().validate_all()
    summary = const_result["summary"]
    scored_total = summary["scored_total"]
    print(f"Total entries: {summary['total']}")
    print(f"Scored Selection/Bridge/Phenomenology rows: {scored_total}")
    print(
        f"  PASS (< 1 sigma):    {summary['passed']:3d}  "
        f"({100 * summary['passed'] / scored_total:.1f}% of scored rows)"
    )
    print(
        f"  CAUTION (1-2 sigma): {summary['caution']:3d}  "
        f"({100 * summary['caution'] / scored_total:.1f}% of scored rows)"
    )
    print(
        f"  WARN (2-3 sigma):    {summary['warn']:3d}  "
        f"({100 * summary['warn'] / scored_total:.1f}% of scored rows)"
    )
    print(
        f"  FAIL (> 3 sigma):    {summary['fail']:3d}  "
        f"({100 * summary['fail'] / scored_total:.1f}% of scored rows)"
    )
    print(f"  EXACT/reference:     {summary['exact']:3d}")
    print(f"  INPUT/excluded:      {summary['input']:3d}")
    print(f"  OPEN:                {summary['open']:3d}")
    print(f"  OPEN TEST:           {summary['test']:3d}")
    print(f"Scored pass rate: {summary['pass_rate']:.1f}%")
    print(f"Scorecard status: {summary['status']}")

    print("\nDetailed Results:")
    print(
        f"{'Symbol':24s} {'Grade':14s} {'CE Value':14s} {'Obs Value':14s} "
        f"{'delta/sigma':12s} {'Status':10s}"
    )
    print("-" * 96)
    for row in const_result["results"]:
        ce_str = f"{row['ce']:.6g}"
        obs_str = "N/A" if row["obs"] is None else f"{row['obs']:.6g}"
        sigma_str = (
            "N/A" if math.isnan(row["sigma_offset"]) else f"{row['sigma_offset']:+.2f}"
        )
        print(
            f"{row['symbol'][:24]:24s} {row['grade'][:14]:14s} {ce_str:14s} "
            f"{obs_str:14s} {sigma_str:12s} {row['status']:10s}"
        )

    print("\n[3] DIMENSIONAL CONSISTENCY CHECK")
    print("-" * 96)
    dim_result = DimensionalValidator().validate_all()
    print(f"Formulas checked: {dim_result['total']}")
    print(f"Dimensionally consistent: {dim_result['passed']} / {dim_result['total']}")
    print(f"Overall: {'PASS' if dim_result['all_pass'] else 'ISSUES DETECTED'}")
    for formula, info in dim_result["formulas"].items():
        status = "OK" if info["dims_correct"] else "ERROR"
        print(f"  {formula:52s} [{status}] {info['note']}")

    overall_status = _overall_status(result, summary, dim_result)
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"Bootstrap equation:    {'PASS' if result['pass'] else 'FAIL'}")
    print(
        f"Constants scorecard:   {summary['status']} "
        f"({summary['passed']}/{scored_total} PASS; {summary['pass_rate']:.1f}%)"
    )
    print(f"Dimensional analysis:  {'PASS' if dim_result['all_pass'] else 'FAIL'}")
    print("Physical closure:      INCOMPLETE")
    print("Release readiness:     NOT_READY")
    print(f"\nOVERALL: {overall_status}")
    print("=" * 96 + "\n")

    return {
        "bootstrap": result,
        "constants": const_result,
        "dimensional": dim_result,
        "overall_status": overall_status,
        "physical_closure": "INCOMPLETE",
        "release_readiness": "NOT_READY",
    }


if __name__ == "__main__":
    main()
