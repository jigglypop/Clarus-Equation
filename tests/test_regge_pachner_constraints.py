"""실제 전방 병합의 사전 제약과 공급한 양자 제약의 정의역을 검증한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"verify/Q-0020/regge_pachner_constraints.py"
SPEC = importlib.util.spec_from_file_location("ce_pachner_constraint_checks", SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_forward_move_changes_three_boundary_tetrahedra_to_two(report):
    data = report["boundary_moves"]
    assert data["first_removed"] == [(0, 1, 2, 3), (0, 1, 2, 4)]
    assert data["second_removed"] == [(0, 1, 3, 4), (0, 1, 3, 5), (0, 1, 4, 5)]
    assert data["second_added"] == [(0, 3, 4, 5), (1, 3, 4, 5)]
    assert data["final_internal_edges"] == [(0, 1)]
    link = {tuple(sorted(set(cell)-{0, 1})) for cell in checks.FINAL.cells}
    assert link == {(2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5)}


def test_local_corner_terms_match_whole_actions_and_gradients(report):
    for row in report["cases"]:
        assert row["action_residual"] < 1e-13
        assert row["gradient_residual"] < 1e-13
        assert row["internal_equation_residual"] < 2e-14


def test_inverse_removal_does_not_replace_forward_preconstraint(report):
    for row in report["cases"]:
        assert abs(row["postconstraint"]) < 1e-13
        assert row["undo_residual"] < 1e-13
        if row["factor"] == 1:
            assert abs(row["preconstraint"]) < 1e-12
            assert row["forward_accepted"]
            assert max(abs(value) for value in row["curvature"]) < 1e-12
        else:
            assert abs(row["preconstraint"]) > .2
            assert not row["forward_accepted"]
            assert max(abs(value) for value in row["curvature"]) > .1


def test_exact_gram_derivative_certificate_agrees_with_independent_action_difference(report):
    exact = report["exact_certificate"]
    assert exact["area_derivative"] == "3/8"
    assert set(exact["deficit_derivatives"].values()) == {"-3*sqrt(2)/2"}
    assert exact["mixed"] == "-9*sqrt(2)/4"
    assert exact["poisson_bracket"] == "9*sqrt(2)/4"
    expected = -9*math.sqrt(2)/4
    assert report["cases"][2]["mixed_hessian"] == pytest.approx(expected, abs=2e-9)
    assert report["action_mixed_error"] < 2e-7
    for value in report["action_mixed_finite_difference"]:
        assert value == pytest.approx(expected, abs=2e-7)


def test_poisson_bracket_from_two_separate_move_constraints_is_nonzero():
    x = checks.FINAL.lengths(checks.reference_points())
    h = 2e-5
    de, dy = np.eye(len(x))[checks.E_ID]*h, np.eye(len(x))[checks.Y_ID]*h
    c_e = -(checks.actions(x+de)["first"]["gradient"][checks.Y_ID]
            -checks.actions(x-de)["first"]["gradient"][checks.Y_ID])/(2*h)
    f_y = (checks.actions(x+dy)["second"]["gradient"][checks.E_ID]
           -checks.actions(x-dy)["second"]["gradient"][checks.E_ID])/(2*h)
    assert c_e-f_y == pytest.approx(9*math.sqrt(2)/4, abs=2e-7)


def test_internal_equation_root_allows_forward_boundary_momenta(report):
    x = checks.FINAL.lengths(checks.reference_points())
    x[checks.Y_ID] = report["root"]
    old_lengths = x[checks.OLD_IDS]
    old_momenta = checks.MOVE.old.evaluate(old_lengths)["gradient"]
    q, p = checks.forward(old_lengths, old_momenta, report["root"])
    keep = np.arange(len(x)) != checks.E_ID
    np.testing.assert_allclose(q, x[keep], atol=1e-13)
    np.testing.assert_allclose(p, checks.FINAL.evaluate(x)["gradient"][keep], atol=2e-13)
    assert len(q) == 14
    assert report["root"] == pytest.approx(report["reference_new_length"], abs=2e-12)
    assert report["root_residual"] < 1e-12


def test_actual_wave_obeys_ordered_constraint_but_not_naive_constraint(report):
    for row in report["ordering"]:
        assert row["naive_residual"] > .4
        assert row["naive_formula_error"] < 1e-8
        assert row["ordered_residual"] < 1e-8
        assert row["relative_naive_residual"] == pytest.approx(1/(2*row["y"]), abs=1e-9)


def test_first_derivative_norm_diverges_at_the_unit_reference_endpoint():
    import sympy as sp
    y, eta, length = sp.symbols("y eta length", positive=True)
    amplitude = sp.sqrt(2*y)/length
    norm = sp.integrate(sp.diff(amplitude, y)**2, (y, eta*length, length))
    assert sp.simplify(norm+sp.log(eta)/(2*length**2)) == 0
    assert sp.limit(norm, eta, 0, dir="+") == sp.oo


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_forward_tolerance_is_rejected(tolerance):
    x = checks.FINAL.lengths(checks.reference_points())
    with pytest.raises(ValueError, match="허용 오차"):
        checks.forward(x[checks.OLD_IDS], np.zeros(14), x[checks.Y_ID], tolerance=tolerance)


def test_saved_artifact_matches_source_dependencies_and_current_constraints(report):
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, digest in saved["dependencies"].items():
        assert digest == hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()
    assert len(saved["cases"]) == len(report["cases"]) == 5
    for old, current in zip(saved["cases"], report["cases"]):
        assert old["preconstraint"] == pytest.approx(current["preconstraint"], abs=2e-12)
    assert saved["exact_certificate"]["mixed"] == report["exact_certificate"]["mixed"]
    assert "전체 제약" in " ".join(saved["unfinished"])
    assert "내부 코어" in saved["naive_domain"]["domain_boundary"]
