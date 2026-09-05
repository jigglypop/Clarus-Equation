"""실제 원천 응답의 비선형 가지·정준 제약·유한 진폭 미분을 검산한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_source_response.py"
SPEC = importlib.util.spec_from_file_location("ce_source_response",SOURCE)
checks = importlib.util.module_from_spec(SPEC)
sys.path.insert(0,str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_two_branches_satisfy_actual_geometry_and_sourced_constraints(report):
    for row in report["stationary_branches"]:
        assert row["roots"][0] < 2 < row["roots"][1]
        assert row["direct_regge_gradient_error"] < 1e-8
        assert row["c_residual"] < 1e-12
        assert row["source_F_residual"] < 1e-8
        assert row["source_derivative_relative_error"] < 1e-5
        assert row["susceptibility"][0] < 0 < row["susceptibility"][1]


def test_center_shift_and_remainder_match_exact_regge_taylor_coefficients(report):
    rows = report["stationary_branches"]
    errors = [row["center_coefficient_error"] for row in rows]
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert errors[-1] < 1e-7
    assert all(row["scaled_branch_remainder"] < .013 for row in rows)
    certificate = checks.geometry.exact_certificate()
    assert certificate["critical_derivatives_2_3_4"] == ["0","48*sqrt(3)","720*sqrt(3)"]


def test_response_lifts_constraint_degeneracy_with_fixed_coefficient(report):
    rows = report["stationary_branches"]
    errors = [row["a_squared_ratio_error"] for row in rows]
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert errors[-1] < .003
    for row in rows:
        left,right = row["a_over_sqrt_source"]
        assert left > 0 > right
    assert rows[-1]["a_squared_over_source"] == pytest.approx([math.sqrt(3)]*2,abs=.003)


def test_negative_source_has_no_stationary_point_by_exact_sign_identity():
    import sympy as sp
    edge = sp.Symbol("edge",real=True)
    cosine = (8-3*edge**2)/(2*(16-3*edge**2))
    assert sp.simplify(cosine+sp.Rational(1,2)-3*(4-edge**2)/(16-3*edge**2)) == 0
    points = np.r_[np.linspace(.01,1.999,60),np.linspace(2.001,checks.geometry.limit(1.)-.001,60)]
    assert np.all(checks.geometry.gradient(points,1.) > 0)
    assert checks.geometry.gradient(2.,1.) == pytest.approx(0.,abs=1e-12)
    with pytest.raises(ValueError):
        checks.source_roots(-1e-6)


def test_full_kernel_and_source_derivatives_survive_both_source_signs(report):
    for row in report["kernel_cases"]:
        assert row["quadrature_difference"] < 1e-8
        assert row["first_derivative_error"] < 1e-6
        assert row["second_derivative_error"] < 1e-6
        assert max(row["moment_bound_ratios"]) <= 1+1e-12
        if row["beta"] == 0.:
            assert row["kernel"] == pytest.approx([1.,0.],abs=1e-12)
            assert np.array(row["derivatives"]) == pytest.approx(np.zeros((2,2)),abs=1e-12)


def test_saved_evidence_uses_current_sources():
    saved=json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name,expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest() == expected

