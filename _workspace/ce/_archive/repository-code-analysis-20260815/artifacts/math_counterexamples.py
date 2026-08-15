"""Independent boundary/counterexample probes for the repository math audit.

Run from the repository root with ``PYTHONPATH=reality_stone/python``.
The script is read-only: it prints one JSON object and writes no files.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np

from reality_stone.clarus.bootstrap_solver import BootstrapSolver
from reality_stone.clarus.covariant_metric_flow import CovariantMetricFlow
from reality_stone.clarus.dimensionless_checker import (
    Dimension,
    DimensionlessChecker,
    Formula,
)
from reality_stone.clarus.homogeneous_signed_cue import HomogeneousSignedCue
from reality_stone.clarus.unified_metric import UnifiedMetricCore
from reality_stone.clarus.universe_life_kernel import (
    HostTuple,
    HybridState,
    REGISTERED_FLUX_E2,
    internal_kernel,
    registered_start,
)


def fixed_point_residual(q: float, depth: float) -> float:
    return q - math.exp(-depth * (1.0 - q))


def independent_low_root(depth: float) -> float:
    """Bisection from the analytic low-branch bracket, independent of CE code."""

    low = 0.0
    high = 1.0 / depth
    for _ in range(200):
        middle = (low + high) / 2.0
        if fixed_point_residual(middle, depth) > 0.0:
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def main() -> None:
    results: dict[str, object] = {}

    alpha_s = 0.11789
    sin2_theta_w = 4.0 * alpha_s ** (4.0 / 3.0)
    canonical_delta = sin2_theta_w * (1.0 - sin2_theta_w)
    canonical_depth = 3.0 + canonical_delta
    code_map_ad_formula = 4.0 / (math.e ** (4.0 / 3.0) * math.pi ** (4.0 / 3.0))
    results["constant_recalculation"] = {
        "alpha_s": alpha_s,
        "sin2_theta_w": sin2_theta_w,
        "delta_from_declared_bridge": canonical_delta,
        "depth_from_declared_bridge": canonical_depth,
        "low_root_from_declared_bridge": independent_low_root(canonical_depth),
        "hardcoded_bootstrap_depth_difference": BootstrapSolver.D_EFF - canonical_depth,
        "code_map_ad_formula_value": code_map_ad_formula,
    }

    solver = BootstrapSolver()
    low_root = independent_low_root(solver.D_EFF)
    newton_identity = solver.solve_newton(initial_guess=1.0)
    results["bootstrap_branches"] = {
        "depth": solver.D_EFF,
        "independent_low_root": low_root,
        "low_residual": fixed_point_residual(low_root, solver.D_EFF),
        "identity_root": 1.0,
        "identity_residual": fixed_point_residual(1.0, solver.D_EFF),
        "public_newton_from_one": newton_identity,
        "verify_identity_equation_satisfied": bool(
            solver.verify_fixed_point(1.0)["equation_satisfied"]
        ),
    }

    checker = DimensionlessChecker()
    false_formula = Formula(
        name="deliberately dimensional quotient",
        symbol="bad_ratio",
        formula="mass/time",
        expected_dim=Dimension.DIMENSIONLESS,
        source="audit counterexample",
    )
    results["dimensionless_false_positive"] = checker.check_formula(false_formula)

    metric_core = UnifiedMetricCore(
        points=[[0.0, 0.0], [1.0, 0.0]],
        adjacency=[[0.0, 1.0], [1.0, 0.0]],
    )
    nonsymmetric = np.array(
        [
            [[1.0e308, 0.0], [0.0, 1.0e308]],
            [[1.0, 1.0e100], [-1.0e100, 1.0]],
        ],
        dtype=np.float64,
    )
    accepted_state = metric_core.make_state(nonsymmetric)
    accepted_metric = np.asarray(accepted_state.metric)
    results["unified_metric_global_symmetry_tolerance"] = {
        "input_node1_max_antisymmetry": float(
            np.max(np.abs(nonsymmetric[1] - nonsymmetric[1].T))
        ),
        "accepted": True,
        "stored_node1": accepted_metric[1].tolist(),
    }

    flow = CovariantMetricFlow(2)
    routes = (((1.0e-8, 0.0),), ((2.0e-8, 0.0),))
    small_choice = flow.choose_route(flow.identity_state(), routes)
    scaled_state = flow.make_state_from_metric(np.eye(2) * 1.0e16)
    scaled_choice = flow.choose_route(scaled_state, routes)
    results["route_tie_scale_dependence"] = {
        "small_costs": small_choice.costs,
        "small_minimizers": small_choice.minimizers,
        "scaled_costs": scaled_choice.costs,
        "scaled_minimizers": scaled_choice.minimizers,
        "cost_ratio_before": small_choice.costs[1] / small_choice.costs[0],
        "cost_ratio_after": scaled_choice.costs[1] / scaled_choice.costs[0],
    }

    signed = HomogeneousSignedCue(1)
    # In the orthonormal basis parallel/perpendicular to z=(1,1), this metric
    # has eigenvalues 1 and 100.  Thus z^T G z=2, so write_cue accepts it, but
    # the registered + cue is decoded as - because the perpendicular action
    # cost is 200 while the -z action is reset to cost 4.
    ambient_metric = np.array([[50.5, -49.5], [-49.5, 50.5]])
    state = signed.make_state_from_metric(ambient_metric)
    cue = signed.lift_cue([1.0], +1)
    prewrite_prediction = float(np.asarray(cue) @ ambient_metric @ np.asarray(cue))
    written = signed.write_cue(state, [1.0], +1)
    readout = signed.readout(written, [1.0])
    results["homogeneous_accepted_wrong_sign"] = {
        "prewrite_prediction": prewrite_prediction,
        "cue_sign": +1,
        "action_order": readout.action_signs,
        "terminal_costs": readout.costs,
        "selected_sign": readout.selected_sign,
    }

    start = registered_start()
    host_zero = HostTuple(0, REGISTERED_FLUX_E2, start, start, 0, 1)
    host_one = HostTuple(0, REGISTERED_FLUX_E2, start, start, 1, 1)
    next_zero = internal_kernel(host_zero)
    next_one = internal_kernel(host_one)
    results["host_sigma_dynamics"] = {
        "sigma_changed": host_zero.sigma != host_one.sigma,
        "sensor_dynamics_equal": next_zero.sensor == next_one.sensor,
        "action_dynamics_equal": next_zero.action == next_one.action,
    }

    invalid_cube_state = HybridState(Fraction(2), Fraction(0), Fraction(0))
    results["hybrid_direct_constructor_domain"] = {
        "accepted_mass": str(invalid_cube_state.mass),
        "declared_unit_interval_violated": invalid_cube_state.mass > 1,
    }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
