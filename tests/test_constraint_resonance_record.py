"""제약 생성자의 기록 응답과 정의역·준비 대조를 검증한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify/Q-0020/constraint_resonance_record.py"
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location("constraint_resonance_checks",SOURCE)
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_resonance_and_uniform_nonzero_mode_envelope(report):
    for row in report["spectra"]:
        assert abs(row["zero_response"]-1)<1e-12
        assert row["maximum_off_response"]<=row["off_envelope"]+1e-12
        assert row["posterior_off"]<=row["posterior_bound"]+1e-12
    near=[row for row in report["spectra"] if row["size"]==65 and row["beta"]==5]
    tuned=next(row for row in near if abs(row["gamma_over_gap"]-.01)<1e-12)
    shifted=next(row for row in near if abs(row["gamma_over_gap"]-.01037)<1e-12)
    assert shifted["maximum_off_response"]>100*tuned["maximum_off_response"]
    # 유한 절단 밖의 모드도 같은 해석적 상계와 대조한다.
    frequencies=np.arange(1,10001)
    for gamma in (.5,.1,.01):
        values=checks.response(frequencies,gamma)
        assert max(values)<=gamma**2/(1+gamma**2)+1e-12


def test_actual_regge_generator_matches_differential_constraint(report):
    assert max(row["total_eigen_residual"] for row in report["differential"])<1e-6
    assert min(row["minimum_battery"] for row in report["differential"])>0
    assert min(row["system_only_residual"] for row in report["differential"])>1e-3


def test_direct_matrix_detector_preserves_constraint_and_total_energy(report):
    for row in report["matrix"]:
        assert row["direct_exponential_error"]<1e-10
        assert row["constraint_commutator"]<1e-10
        assert row["constraint_distribution_error"]<1e-12
        assert row["pointer_response_error"]<1e-12
        assert row["energy_balance_error"]<1e-12
        assert row["minimum_battery_energy"]>0
        assert abs(row["system_energy_change"])>1e-3
        assert row["norm_error"]<1e-12
    budget=report["branch_budget"]
    assert budget["probability_sum_error"]<1e-12
    assert budget["total_energy_sum_error"]<1e-12
    assert budget["branches"][0]["probability"]>.96
    assert budget["branches"][0]["total_energy_unnormalized"]>2.6
    for branch in budget["branches"]:
        assert abs(branch["conditional_total_energy"]-2.8)<1e-12


def test_autonomous_clock_square_and_factorized_readout(report):
    for row in report["clock"]:
        assert row["constraint_residual"]<1e-6
        assert row["flux_error"]<1e-8
        assert row["click_flux_error"]<1e-8
        assert row["position_flux_error"]<1e-12
        assert row["omitted_square_residual"]>.1
    assert max(row["difference"] for row in report["preparation"]["factorized"])<1e-12


def test_twist_and_no_gap_do_not_certify_an_exact_zero(report):
    rows=report["twist"]["twisted"]
    assert all(row["exact_zero_mode"] is False for row in rows)
    assert rows[-1]["minimum_abs_eigenvalue"]>0
    assert rows[-1]["click"]>.99999
    accumulation=report["twist"]["accumulation"]
    assert accumulation[-1]["lambda_over_gamma"]>0
    assert accumulation[-1]["click"]>1-1e-8


def test_atomic_and_continuous_energy_correlations_have_different_limits(report):
    data=report["preparation"];rows=data["correlated"]
    assert rows[-1]["atomic_position"]>1-1e-5
    assert abs(rows[-1]["continuous_position"]-data["continuous_limit"])<1e-6
    assert rows[-1]["continuous_position"]<.8
    assert max(row["continuous_quad_error"] for row in rows)<1e-12


def test_click_without_zero_prior_can_be_entirely_false():
    model=checks.Detector(ratio=.25)
    state=np.zeros((1,model.n,2),complex);state[0,1,0]=1
    output=model.rotate_fraction(1.,state)
    clicked=output[0,:,1]
    probability=float(np.linalg.norm(clicked)**2)
    assert probability>1e-4
    assert abs(clicked[0])<1e-14
    assert np.sum(abs(clicked[1:])**2)/probability==pytest.approx(1.)


def test_invalid_resolution_and_mode_cutoff():
    for gamma in (0.,-1.,np.inf,np.nan):
        with pytest.raises(ValueError):
            checks.response([0.,1.],gamma)
    for size in (True,2,16):
        with pytest.raises(ValueError):
            checks.Detector(size)


def test_saved_artifact_matches_sources():
    path=SOURCE.with_suffix(".json")
    if not path.exists():
        pytest.skip("최초 선별 후 산출물 작성")
    data=json.loads(path.read_text(encoding="utf-8"))
    for name,sha in data["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()==sha
