"""원래 제약의 축약과 계수 변화에 대한 독립 기하 대조."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_constraint_reduction.py"
SPEC = importlib.util.spec_from_file_location("ce_original_reduction_checks", SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_actual_separate_move_derivatives_match_flat_family_and_full_action(report):
    for row in report["cases"]:
        h = row["h"]
        expected = 3*math.sqrt(2)*(1-h*h)
        assert row["reduction"]["bracket"] == pytest.approx(expected, abs=1e-6)
        assert row["independent_hessian_error"] < 1e-6
        assert row["step_difference"] < 1e-6
        assert row["flat_hessian_error"] < 1e-6
        assert row["equation_residual"] < 1e-9
        assert row["curvature_residual"] < 1e-9


def test_second_class_dirac_inverse_and_liouville_density(report):
    for row in report["cases"]:
        data = row["reduction"]
        assert data["constraint_rank"] == 2
        assert data["tangent_residual"] < 1e-12
        if row["h"] == 1:
            continue
        assert data["second_class"]
        assert data["pullback_rank"] == 28
        assert data["inverse_residual"] < 1e-7
        assert data["dirac_ey"] == pytest.approx(1/row["expected_bracket"], abs=1e-7)
        assert data["liouville_density"] == pytest.approx(abs(row["expected_bracket"]), abs=1e-6)
        assert data["determinant_sign"] == 1
        assert data["density_error"] < 1e-10


def test_cospherical_flat_geometry_is_valid_but_pair_is_not_second_class(report):
    row = next(row for row in report["cases"] if row["h"] == 1)
    data = row["reduction"]
    assert row["minimum_gram_eigenvalue"] > .15
    assert data["constraint_rank"] == 2
    assert data["pullback_rank"] == 26
    assert not data["second_class"]
    assert data["dirac_ey"] is None
    assert data["liouville_density"] is None
    assert max(data["smallest_singular_values"]) < 1e-7


def test_squared_coordinates_preserve_same_measure_only_with_jacobian(report):
    for row in report["cases"]:
        data = row["reduction"]
        if not data["second_class"]:
            continue
        changed = data["coordinate_check"]
        factor = 4*row["e"]*row["y"]
        assert changed["canonical_residual"] < 1e-12
        assert changed["bracket_error"] < 1e-10
        assert changed["density_error"] < 1e-10
        assert changed["restored_density_error"] < 1e-10
        assert changed["density_squared"] == pytest.approx(data["liouville_density"]/factor)
        assert abs(changed["density_squared"]-data["liouville_density"]) > .1


def test_reduction_uses_actual_constraints_away_from_symmetric_flat_family():
    q = checks.flat_lengths(.9)
    q *= 1+np.linspace(-.003, .004, len(q))
    data = checks.reduction(q)
    hessian, _ = checks.moves.FINAL.hessian(q, [checks.E, checks.Y])
    assert data["second_class"]
    assert data["bracket"] == pytest.approx(-hessian[0, 1], abs=1e-6)
    assert data["inverse_residual"] < 1e-7
    assert data["density_error"] < 1e-10
    assert abs(data["bracket"]-3*math.sqrt(2)*(1-.9**2)) > 1e-3


@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf")])
def test_invalid_height_is_rejected(height):
    with pytest.raises(ValueError):
        checks.flat_lengths(height)


def test_invalid_derivative_step_is_rejected():
    with pytest.raises(ValueError):
        checks.reduction(checks.flat_lengths(.5), step=0)


def test_saved_evidence_matches_current_source_and_dependencies():
    saved = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name, expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected

