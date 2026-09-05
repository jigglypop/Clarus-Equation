"""공동 제약 에너지의 정확 반례와 실제 기하의 접선 흐름을 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_joint_energy_flow.py"
SPEC = importlib.util.spec_from_file_location("ce_joint_energy_flow", SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_exact_gram_certificate_proves_action_gradient_obstruction(report):
    certificate = report["fold_certificate"]
    assert certificate["symbolic_difference"] == "0"
    assert certificate["g_y_value"] < -7*math.sqrt(3)/30
    assert {tuple(row["triangle"]) for row in certificate["rows"]} == {
        (0,3,4), (1,3,4), (2,3,4), (3,4,5)}
    for point in report["flat_points"]:
        if point["height"] == 1.:
            assert point["candidates"]["g_y"] == pytest.approx(
                certificate["g_y_value"], abs=1e-8)
            action = point["candidates"]["cases"][1]
            assert action["constraint_obstruction"][0] > 3*point["beta"]
            assert action["multipliers"] is None


def test_linear_energy_has_nonzero_obstruction_at_valid_fold(report):
    for point in report["flat_points"]:
        if point["height"] != 1.:
            continue
        length = point["candidates"]["cases"][0]
        expected = -1/checks.previous.full.limit(1.)
        assert length["constraint_obstruction"] == pytest.approx([0.,expected])
        assert length["multipliers"] is None
        assert point["square_flow"]["minimum_gram"] > .15
    q = checks.reduction.flat_lengths(1.)
    actual = checks.reduction.reduction(q)
    assert actual["constraint_rank"] == 2
    assert actual["pullback_rank"] == 26


def test_actual_action_and_independent_analytic_derivatives(report):
    for point in report["flat_points"]:
        assert point["flat_a_error"] < 1e-8
        assert point["analytic_a_e_error"] < 1e-6
        assert point["derivative_step_difference"] < 3e-6
        for candidate in point["candidates"]["cases"]:
            if candidate["multipliers"] is not None:
                assert candidate["consistency_residual"] < 1e-7
    for point in report["deformed_points"]:
        assert point["independent_action_hessian_error"] < 1e-6
        assert point["square_flow"]["minimum_gram"] > 0


def test_square_flow_preserves_both_constraints_and_all_momentum_terms(report):
    points = report["flat_points"] + report["deformed_points"]
    for point in points:
        flow = point["square_flow"]
        assert flow["constraint_rate_residual"] < 1e-6
        assert flow["momentum_tangent_residual"] < 1e-6
        assert flow["shifted_rest_momentum_residual"] < 1e-6
        assert flow["energy_rate_residual"] < 1e-10
        assert flow["a_rate_residual"] < 1e-10
    # 나머지 운동량의 변화를 지우는 두 변수 절단은 이 실제 흐름과 다르다.
    assert max(p["square_flow"]["omitted_shifted_momentum_defect"] for p in points) > 1.


def test_simple_fold_has_nonzero_tangent_flow(report):
    for point in report["flat_points"]:
        if point["height"] != 1.:
            continue
        beta = point["beta"]
        flow = point["square_flow"]
        assert flow["a_e"] == pytest.approx(-6*math.sqrt(2)*beta, abs=1e-6)
        assert flow["qdot_e_y"][1] == pytest.approx(6*math.sqrt(2)*beta, abs=1e-6)
        assert abs(flow["a"]) < 1e-8
        assert flow["energy"] < 1e-15


def test_short_actual_trajectories_preserve_layer_and_energy(report):
    curves = report["configuration_trajectories"]
    assert curves[0]["initial_a"] > .1
    assert curves[2]["initial_a"] < -.1
    for curve in curves:
        assert curve["maximum_a_change"] < 1e-7
        assert curve["maximum_energy_change"] < 1e-7
        assert curve["endpoint_step_difference"] < 1e-7
        assert curve["minimum_gram"] > .1
        assert curve["final_e_y"][1] > curve["initial_e_y"][1]
        if abs(curve["initial_a"]) > .1:
            assert curve["initial_a"]*curve["final_a"] > 0
    assert abs(curves[1]["final_a"]) < 1e-7


def test_evidence_matches_current_source_and_dependencies():
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected

