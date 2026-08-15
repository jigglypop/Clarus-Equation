"""Audit the mathematical density bridge without promoting it to a prediction.

The scalar action implemented here is an existence construction: its constant
Euler--Lagrange equation reproduces the CE fixed-point equation.  The module
also keeps the missing physical bridge explicit.  A stationary scalar value is
not, by itself, a conserved baryon current or a critical-density fraction.

All arguments of ``log`` and all probability/fraction inputs are dimensionless.
The overall field and potential scales from the covariant action do not enter
these dimensionless checks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


VARIATIONAL_STATUS = "THEOREM_EXISTENCE_CONSTRUCTION"
POTENTIAL_CHOICE_STATUS = "MODEL_AXIOM"
EXTERNAL_INPUT_STATUS = "EXTERNAL_INPUT"
LOCAL_STABILITY_STATUS = "THEOREM_LOCAL"
WEIGHTED_EVENT_STATUS = "THEOREM_CONDITIONAL_IFF"
MATTER_COMPOSITION_STATUS = "CONDITIONAL_CONSTRUCTION"
CRITICAL_DENSITY_STATUS = "INCOMPLETE"
PREDICTION_STATUS = "NONE"


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_probability(value: float, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _require_field_value(value: float) -> float:
    value = _require_finite(value, "x")
    if not 0.0 < value <= 1.0:
        raise ValueError("x must lie in the declared domain zero < x <= one")
    return value


def potential(x: float, d: float, offset: float = 0.0) -> float:
    """Return v_D(x) for dimensionless x, D, and additive offset C."""
    x = _require_field_value(x)
    d = _require_finite(d, "d")
    offset = _require_finite(offset, "offset")
    return x * math.log(x) - x + d * (x - 0.5 * x * x) + offset


def stationary_gradient(x: float, d: float) -> float:
    """Return v_D'(x) = log(x) + D(1-x)."""
    x = _require_field_value(x)
    d = _require_finite(d, "d")
    return math.log(x) + d * (1.0 - x)


def stationary_hessian(x: float, d: float) -> float:
    """Return v_D''(x) = 1/x - D."""
    x = _require_field_value(x)
    d = _require_finite(d, "d")
    return 1.0 / x - d


def unit_branch_hessian(d: float) -> float:
    """Return the one-sided Hessian at the allowed-domain endpoint x=1."""
    d = _require_finite(d, "d")
    return 1.0 - d


def small_fixed_point_root(d: float, *, iterations: int = 200) -> float:
    """Solve q=exp[-D(1-q)] for the non-unit branch when D>1.

    The bisection is performed in y=-log(q), on ``log(D) < y < D``.  This
    avoids the false convergence that a direct fixed-point iteration can have
    at the second root q=1.  Values whose small root is below the positive
    binary64 range are rejected instead of silently returning zero.
    """
    d = _require_finite(d, "d")
    if d <= 1.0:
        raise ValueError("d must be greater than one for a distinct small branch")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    lower = math.log(d)
    upper = d

    def equation(y: float) -> float:
        return y - d * (1.0 - math.exp(-y))

    # For D extremely close to one, floating-point endpoint signs can collapse.
    # Moving inward retains the mathematical bracket whenever it is representable.
    lower = math.nextafter(lower, upper)
    upper = math.nextafter(upper, lower)
    if equation(lower) > 0.0 or equation(upper) < 0.0:
        raise ValueError("the distinct small branch is not resolved at binary64 precision")

    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if midpoint == lower or midpoint == upper:
            break
        if equation(midpoint) <= 0.0:
            lower = midpoint
        else:
            upper = midpoint

    y = 0.5 * (lower + upper)
    q = math.exp(-y)
    if q == 0.0:
        raise ValueError("the small branch underflows binary64")
    return q


@dataclass(frozen=True)
class WeightedEventAudit:
    probability: float
    mean_weight_event: float
    mean_weight_complement: float
    total_mean_weight: float
    weighted_fraction: float
    covariance: float
    direct_difference: float
    covariance_difference: float
    equal_conditional_means: bool


def weighted_event_audit(
    probability: float,
    mean_weight_event: float,
    mean_weight_complement: float,
) -> WeightedEventAudit:
    """Evaluate the exact probability-to-energy covariance identity.

    Positive conditional means encode the assumption of a positive energy
    weight.  For 0<q<1, the weighted fraction equals q if and only if the two
    conditional means are equal.
    """
    q = _require_probability(probability, "probability")
    mean_event = _require_finite(mean_weight_event, "mean_weight_event")
    mean_complement = _require_finite(mean_weight_complement, "mean_weight_complement")
    if mean_event <= 0.0 or mean_complement <= 0.0:
        raise ValueError("conditional mean weights must be positive")

    total_mean = q * mean_event + (1.0 - q) * mean_complement
    fraction = q * mean_event / total_mean
    covariance = q * (1.0 - q) * (mean_event - mean_complement)
    return WeightedEventAudit(
        probability=q,
        mean_weight_event=mean_event,
        mean_weight_complement=mean_complement,
        total_mean_weight=total_mean,
        weighted_fraction=fraction,
        covariance=covariance,
        direct_difference=fraction - q,
        covariance_difference=covariance / total_mean,
        equal_conditional_means=mean_event == mean_complement,
    )


@dataclass(frozen=True)
class MatterCompositionAudit:
    branching_probability: float
    matter_composition_fraction: float
    total_matter_fraction: float
    critical_density_fraction: float
    equals_branching_probability: bool
    status: str = MATTER_COMPOSITION_STATUS
    critical_density_bridge_status: str = CRITICAL_DENSITY_STATUS


def matter_composition_audit(probability: float, total_matter_fraction: float) -> MatterCompositionAudit:
    """Keep f_b^(m)=q separate from Omega_b=q*Omega_m."""
    q = _require_probability(probability, "probability")
    omega_m = _require_finite(total_matter_fraction, "total_matter_fraction")
    if not 0.0 < omega_m <= 1.0:
        raise ValueError("total_matter_fraction must be positive and at most one")
    omega_b = q * omega_m
    return MatterCompositionAudit(
        branching_probability=q,
        matter_composition_fraction=q,
        total_matter_fraction=omega_m,
        critical_density_fraction=omega_b,
        equals_branching_probability=omega_m == 1.0,
    )


def conserved_dust_log_fraction_slope(w_total: float) -> float:
    """Return d log(Omega_b)/d log(a)=3*w_total for conserved fixed-mass dust."""
    return 3.0 * _require_finite(w_total, "w_total")


def required_tracker_transfer_ratio(w_total: float) -> float:
    """Return Q_b/(H*rho_b) required to keep Omega_b constant."""
    return -3.0 * _require_finite(w_total, "w_total")


@dataclass(frozen=True)
class StaticScalarStress:
    energy_density: float
    pressure: float
    equation_of_state: float


def static_scalar_stress(potential_energy: float) -> StaticScalarStress:
    """Return the stress of a nonzero homogeneous static canonical scalar."""
    energy = _require_finite(potential_energy, "potential_energy")
    if energy == 0.0:
        raise ValueError("equation of state is undefined at zero energy density")
    return StaticScalarStress(
        energy_density=energy,
        pressure=-energy,
        equation_of_state=-1.0,
    )


@dataclass(frozen=True)
class VacuumOffsetCounterexample:
    root: float
    root_residual: float
    hessian: float
    baseline_offset: float
    shifted_offset: float
    baseline_fraction: float
    shifted_fraction: float


def vacuum_offset_counterexample(
    d: float,
    *,
    other_energy: float = 1.0,
    offset_shift: float = 0.25,
) -> VacuumOffsetCounterexample:
    """Hold q and its Hessian fixed while changing its energy fraction with C."""
    other_energy = _require_finite(other_energy, "other_energy")
    offset_shift = _require_finite(offset_shift, "offset_shift")
    if other_energy <= 0.0 or offset_shift <= 0.0:
        raise ValueError("other_energy and offset_shift must be positive")

    q = small_fixed_point_root(d)
    baseline_offset = q - 0.5 * d * q * q
    baseline_energy = potential(q, d, baseline_offset)
    shifted_energy = potential(q, d, baseline_offset + offset_shift)
    baseline_fraction = baseline_energy / (other_energy + baseline_energy)
    shifted_fraction = shifted_energy / (other_energy + shifted_energy)
    return VacuumOffsetCounterexample(
        root=q,
        root_residual=stationary_gradient(q, d),
        hessian=stationary_hessian(q, d),
        baseline_offset=baseline_offset,
        shifted_offset=baseline_offset + offset_shift,
        baseline_fraction=baseline_fraction,
        shifted_fraction=shifted_fraction,
    )


def action_dimension_ledger() -> dict[str, int | bool]:
    """Return the natural-unit mass dimensions of the candidate scalar action."""
    x_dimension = 0
    d_dimension = 0
    potential_dimension = 0
    derivative_dimension = 1
    field_scale_dimension = 1
    potential_scale_dimension = 1
    kinetic_density_dimension = (
        2 * field_scale_dimension + 2 * (derivative_dimension + x_dimension)
    )
    potential_density_dimension = 4 * potential_scale_dimension + potential_dimension
    return {
        "x": x_dimension,
        "d": d_dimension,
        "log_argument": x_dimension,
        "potential": potential_dimension,
        "field_scale": field_scale_dimension,
        "potential_scale": potential_scale_dimension,
        "kinetic_action_density": kinetic_density_dimension,
        "potential_action_density": potential_density_dimension,
        "required_action_density": 4,
        "passes": (
            x_dimension == 0
            and d_dimension == 0
            and kinetic_density_dimension == 4
            and potential_density_dimension == 4
        ),
    }


def build_audit(d: float) -> dict[str, Any]:
    """Build a machine-readable audit of approved theorems and open bridges."""
    d = _require_finite(d, "d")
    q = small_fixed_point_root(d)
    residual = stationary_gradient(q, d)
    hessian = stationary_hessian(q, d)
    endpoint_hessian = unit_branch_hessian(d)
    offset_example = vacuum_offset_counterexample(d)
    unequal_weights = weighted_event_audit(q, 2.0, 1.0)
    composition_example = matter_composition_audit(q, 0.5)
    static_stress = static_scalar_stress(1.0)
    dimensions = action_dimension_ledger()

    checks = {
        "small_root_stationary": abs(residual) <= 1.0e-13,
        "small_root_locally_stable": hessian > 0.0,
        "unit_root_unstable": endpoint_hessian < 0.0,
        "offset_preserves_stationary_data": (
            abs(offset_example.root_residual) <= 1.0e-13
            and offset_example.hessian == hessian
        ),
        "offset_changes_density_fraction": not math.isclose(
            offset_example.baseline_fraction,
            offset_example.shifted_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "weighted_covariance_identity": math.isclose(
            unequal_weights.direct_difference,
            unequal_weights.covariance_difference,
            rel_tol=1.0e-13,
            abs_tol=1.0e-15,
        ),
        "matter_fraction_not_critical_fraction": not composition_example.equals_branching_probability,
        "static_scalar_is_not_dust": static_stress.equation_of_state == -1.0,
        "mixed_era_constant_fraction_requires_transfer": (
            conserved_dust_log_fraction_slope(-1.0) != 0.0
            and required_tracker_transfer_ratio(-1.0) != 0.0
        ),
        "action_is_dimensionally_consistent": dimensions["passes"] is True,
    }
    approved_checks_pass = all(checks.values())
    return {
        "input": {"d": d, "d_status": EXTERNAL_INPUT_STATUS},
        "dimension_ledger": dimensions,
        "stationary_branch": {
            "q": q,
            "residual": residual,
            "hessian": hessian,
            "unit_branch_hessian": endpoint_hessian,
        },
        "counterexamples": {
            "additive_offset": asdict(offset_example),
            "unequal_conditional_energy": asdict(unequal_weights),
            "matter_composition": asdict(composition_example),
        },
        "static_scalar_stress": asdict(static_stress),
        "claims": {
            "potential_choice": POTENTIAL_CHOICE_STATUS,
            "variational_embedding": VARIATIONAL_STATUS,
            "local_branch_stability": LOCAL_STABILITY_STATUS,
            "weighted_event_identity": WEIGHTED_EVENT_STATUS,
            "matter_composition": MATTER_COMPOSITION_STATUS,
            "critical_density_bridge": CRITICAL_DENSITY_STATUS,
            "physical_prediction": PREDICTION_STATUS,
        },
        "checks": checks,
        "approved_mathematical_checks_pass": approved_checks_pass,
        "physical_bridge_complete": False,
        "is_physical_prediction": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, default=3.1777584234099736)
    parser.add_argument(
        "--require-physical-bridge",
        action="store_true",
        help="fail closed unless the independent critical-density bridge is complete",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_audit(args.d)
    report["exit_policy"] = (
        "REQUIRE_PHYSICAL_BRIDGE" if args.require_physical_bridge else "APPROVED_MATH_ONLY"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["approved_mathematical_checks_pass"]:
        return 1
    if args.require_physical_bridge and not report["physical_bridge_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
