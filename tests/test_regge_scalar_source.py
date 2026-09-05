"""실제 단체 스칼라의 원천 부호·정준 전달·끝점 특이 응답을 독립 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_scalar_source.py"
SPEC=importlib.util.spec_from_file_location("ce_regge_scalar_source",SOURCE)
checks=importlib.util.module_from_spec(SPEC)
sys.path.insert(0,str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_exact_simplex_geometry_and_source_coefficients(report):
    cert=report["certificate"]
    assert cert["determinant_identity"]=="0"
    assert cert["norm_identity"]=="0"
    assert cert["d_derivatives"]==["sqrt(3)/27","-5*sqrt(3)/27","-35*sqrt(3)/54"]
    assert cert["b_derivatives"]==["4*sqrt(3)/3","20*sqrt(3)/3","90*sqrt(3)"]
    for edge in (1.7,1.95,2.,2.04,2.09):
        q=checks.reduction.flat_lengths(1.)
        q[checks.E]=edge
        phi=checks.modes(.4,.07)
        data=checks.scalar_data(q,phi)
        d,b=checks.weights(edge)
        assert data["action"]==pytest.approx(.5*(d*.4**2+b*.07**2),abs=1e-12)
        volume=edge*math.sqrt(3)*math.sqrt(checks.L**2-edge**2)/36
        assert data["volumes"]==pytest.approx([volume]*4,abs=1e-13)


def test_generic_geometry_has_consistent_length_and_field_backreaction(report):
    for row in report["geometry_checks"]:
        assert row["length_gradient_error"]<1e-9
        assert row["mixed_reciprocity_error"]<1e-9
        assert row["action_differential_error"]<1e-8
        assert row["scale_identity_error"]<1e-12
        assert row["field_euler_error"]<1e-12
        assert row["minimum_gram"]>0
        assert row["canonical"]["field_momentum_norm"]>0
        assert row["canonical"]["scalar_geometry_force_norm"]>0


def test_free_connected_scalar_has_only_constant_zero_action_stationary_fields(report):
    for row in report["geometry_checks"]:
        eig=row["eigenvalues"]
        assert abs(eig[0])<1e-12
        assert eig[1]>.1
        assert row["constant_mode_error"]<1e-12
        assert abs(row["shift_charge"])<1e-12
    q=checks.reduction.flat_lengths(1.)
    constant=checks.scalar_data(q,np.full(6,2.7))
    assert abs(constant["action"])<1e-12
    assert np.max(np.abs(constant["field_momentum"]))<1e-12
    assert np.max(np.abs(constant["gradient"]))<1e-12


def test_total_canonical_constraints_and_branch_scaling(report):
    for row in report["branches"]+report["coupling_scaling"]:
        assert row["branches"][0]["edge"]<2<row["branches"][1]["edge"]
        assert row["scaled_remainder"]<.002
        for branch in row["branches"]:
            for key in ("closed_action_error","closed_force_error","composition_residual",
                        "c_residual","F_residual","final_momentum_residual","shift_charge_residual"):
                assert branch[key]<1e-9,(key,branch[key])
            assert branch["independent_bracket_error"]<1e-6
            assert branch["minimum_gram"]>0
    for ratio in (0.,1/12):
        rows=[r for r in report["branches"] if r["ratio"]==ratio]
        errors=[r["square_ratio_error"] for r in rows]
        assert all(a>b for a,b in zip(errors,errors[1:]))
        assert errors[-1]<1e-3
        assert abs(rows[-1]["center_coefficient"]-rows[-1]["expected_center"])<1e-7


def test_zero_coupling_restores_original_regge_boundary_relation():
    q=checks.reduction.flat_lengths(1.)
    phi=np.array([-.2,.3,.1,0.,-.1,.2])
    for beta in (.3,1.,5.):
        row=checks.canonical_check(q,phi,beta,0.)
        assert row["F_residual"]<1e-12
        assert row["c_residual"]<1e-12
        assert row["composition_residual"]<1e-12
        assert row["field_momentum_norm"]==0.
        assert row["scalar_geometry_force_norm"]==0.


def test_source_outside_cone_and_pure_mean_are_negative_controls():
    edge=np.linspace(.001,checks.L-.001,200)
    r=np.sqrt(checks.L**2-edge**2)
    bprime=4*checks.C*checks.L**2/r**3
    assert np.all(checks.geometry.gradient(edge,1.)+.5*.2**2*bprime>0)
    local=np.linspace(1.99,2.01,101)
    for eta in (.03,.01,.003):
        assert np.all(checks.stationary_force(local,eta,.25)>0)
    with pytest.raises(ValueError):
        checks.roots(.01,.25)


def test_cone_boundary_has_distinct_quadratic_splitting(report):
    rows=report["threshold"]
    for row in rows:
        assert row["flat_residual"]==0.
        assert row["other_edge"]<2
    assert rows[-1]["coefficient_error"]<2e-7
    assert rows[-1]["coefficient_error"]<rows[0]["coefficient_error"]


def test_full_kernel_has_nonzero_logarithmic_endpoint_response(report):
    slopes=report["endpoint_log_slopes"]
    assert abs(complex(*report["endpoint_log_coefficient"]))==pytest.approx(math.sqrt(3)/18)
    assert all(a["coefficient_error"]>b["coefficient_error"] for a,b in zip(slopes,slopes[1:]))
    assert slopes[-1]["coefficient_error"]<1e-5
    for row in report["endpoint_cases"]:
        assert row["tolerance_difference"]<2e-9
        assert row["quadrature_error_estimate"]<1e-9
    # beta=0의 양의 진폭 원점을 쓰는 독립 부호·상한 대조.
    delta,error=checks.endpoint_difference(.02,beta=0.,tolerance=1e-10)
    assert abs(1+delta)<=1+1e-9
    assert delta.real<0<delta.imag
    assert error<1e-8


def test_nonpositive_simplex_and_stale_evidence_are_rejected():
    q=checks.reduction.flat_lengths(1.)
    q[checks.E]=checks.L+.1
    with pytest.raises(ValueError):
        checks.scalar_data(q,checks.modes(1.))
    saved=json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["source_sha256"]==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for name,expected in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()==expected

