"""로런츠 응답을 독립 경계 기하·차분·평탄 증인과 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"regge_lorentz_clock.py"
SPEC=importlib.util.spec_from_file_location("ce_lorentz_clock",SOURCE)
checks=importlib.util.module_from_spec(SPEC)
sys.path.insert(0,str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_complex_regge_schlafli_and_generic_scalar_variations(report):
    for row in report["geometry_checks"]:
        assert row["regge_gradient_error"]<1e-7
        assert row["scalar_gradient_error"]<1e-9
        assert row["mixed_derivative_error"]<1e-9
        assert row["homogeneity_error"]<1e-10
        assert row["scalar_scale_error"]<1e-12
        assert row["constant_mode_error"]<1e-12


def test_local_canonical_increments_match_full_complex_action(report):
    for row in report["geometry_checks"]:
        assert row["composition_error"]<1e-10
    for beta,coupling in ((2.,.3),(.7,4.)):
        xi=.01
        T=checks.stationary(xi)
        s=checks.symmetric(T)
        phi=checks.fields(math.sqrt(xi*beta/coupling))
        data=checks.totals(s,phi,beta,coupling)
        assert abs(data["final"]["momentum"][checks.E])<1e-10
        assert abs(data["middle"]["momentum"][checks.Y]-data["first"]["momentum"][checks.Y])<1e-12
        expected=np.r_[beta*checks.regge(s)["gradient"].real+coupling*checks.scalar(s,phi)["gradient"],
                       coupling*checks.scalar(s,phi)["field_momentum"]]
        assert data["final"]["momentum"]==pytest.approx(expected,abs=1e-12)


def test_lorentz_flat_limit_has_an_independent_minkowski_embedding():
    points=np.zeros((6,4))
    points[0,0],points[1,0]=-1/6,1/6
    points[2:,1:]=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/math.sqrt(3)
    metric=np.diag([-1.,1.,1.,1.])
    actual=np.array([(points[a]-points[b])@metric@(points[a]-points[b]) for a,b in checks.EDGES])
    assert actual==pytest.approx(checks.symmetric(1/3),abs=1e-14)
    gravity=checks.regge(actual)
    internal=[row for row in gravity["hinges"] if not row["boundary"]]
    assert len(internal)==4
    assert max(abs(row["deficit"]) for row in internal)<1e-13
    assert abs(gravity["gradient"][checks.E])<1e-13
    assert checks.stationary(0.)==1/3
    matter=checks.scalar(actual,np.full(6,.7))
    assert abs(matter["action"])<1e-14
    assert np.max(np.abs(matter["gradient"]))<1e-14
    phi=checks.fields(.8)
    totals=checks.totals(actual,phi,3.,0.)
    assert totals["final"]["momentum"][:checks.N]==pytest.approx(3*gravity["gradient"].real)
    assert np.max(np.abs(totals["final"]["momentum"][checks.N:]))==0


def test_source_response_coefficient_and_unique_stationary_branch(report):
    rows=report["branches"]
    assert all(a["response_error"]>b["response_error"] for a,b in zip(rows,rows[1:]))
    assert rows[-1]["response_error"]<1e-4
    assert all(row["T"]>1/3 for row in rows)
    for row in rows:
        for name in ("stationary_residual","creation_residual","gravity_force_error","scalar_force_error"):
            assert row[name]<1e-10
    for xi in (.001,.1,1.,10.):
        grid=np.linspace(.05,3.,101)
        force=[checks.gravity_force(T)+xi*checks.matter_force(T) for T in grid]
        assert np.all(np.diff(force)>0)
        assert force[0]<0<force[-1]


def test_boundary_corner_is_retained_and_has_zero_internal_variation(report):
    for row in report["branches"]:
        assert row["corner_error"]<1e-10
        assert row["corner_internal_derivative"]<1e-10
        assert row["corner_imaginary"]==pytest.approx(-8*math.sqrt(3)*math.pi/3)
    # 경계 모양을 바꾸면 허수항이 실제로 달라져야 한다.
    s=checks.symmetric(.6)
    s[checks.Y]+=.004
    assert abs(checks.regge(s)["action"].imag+8*math.sqrt(3)*math.pi/3)>1e-5


def test_clock_and_energy_are_computed_from_boundary_tetrahedra(report):
    rows=report["clock_checks"]+[row["clock"] for row in report["branches"]]
    for row in rows:
        for name in ("clock_norm_error","energy_flux_error","volume_error","field_flux_error",
                     "weighted_energy_force_error","internal_flux_error"):
            assert row[name]<1e-10
        assert row["minimum_boundary_gram"]>0
        assert row["energy"]>0
        assert row["pressure"]==row["rho"]
        assert abs(row["shift_charge"])<1e-12
    T,v,coupling=.6,.7,1.3
    momentum=coupling*checks.scalar(checks.symmetric(T),checks.fields(v))["field_momentum"]
    expected=coupling*math.sqrt(3)/9*math.sqrt(T*T+1/3)*v/T
    assert momentum==pytest.approx([-expected,expected,0,0,0,0],abs=1e-13)


def test_label_permutation_preserves_action_and_gradient():
    s=checks.symmetric(.6)
    s[checks.Y]+=.004
    s[checks.INDEX[0,2]]+=.001
    for permutation in ((1,0,2,3,4,5),(0,1,4,2,5,3)):
        mapping=[checks.INDEX[tuple(sorted((permutation[a],permutation[b])))] for a,b in checks.EDGES]
        permuted=s[mapping]
        original,new=checks.regge(s),checks.regge(permuted)
        assert new["action"]==pytest.approx(original["action"],abs=1e-10)
        assert new["gradient"]==pytest.approx(original["gradient"][mapping],abs=1e-10)


def test_degenerate_signature_and_unsupplied_future_gluing_are_rejected():
    with pytest.raises(ValueError):
        checks.symmetric(.6,b=8/9)
    s=checks.symmetric(.6)
    s[checks.E]=.1
    with pytest.raises(ValueError):
        checks.regge(s)
    with pytest.raises(ValueError):
        checks.clock_audit(.6,preserve_future=False)
    with pytest.raises(ValueError):
        checks.stationary(-1.)
    # 라벨을 뒤집어 붙이면 같은 01 변의 시간값도 서로 다르다.
    t=checks.fields(.6)
    assert abs(t[0]-t[1])==pytest.approx(.6)


def test_saved_report_tracks_inputs():
    path=SOURCE.with_suffix(".json")
    if not path.exists():
        pytest.skip("검산 산출물을 쓰기 전 최초 선별")
    saved=json.loads(path.read_text(encoding="utf-8"))
    for name,digest in saved["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()==digest
