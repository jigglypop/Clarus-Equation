"""무시간 분할 제약을 위치 차분·실제 Gaussian 출력·판독 반례로 검증한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify"/"Q-0020"/"relational_split_clock.py"
SPEC=importlib.util.spec_from_file_location("ce_relational_split_clock",SOURCE)
checks=importlib.util.module_from_spec(SPEC)
sys.path.insert(0,str(SOURCE.parent))
try:
    SPEC.loader.exec_module(checks)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_stationary_constraint_flux_and_local_energy_with_reference(report):
    rows=report["two_level"]+[x for row in report["fock"] for x in row["checks"]]
    for row in rows:
        assert row["constraint_relative_residual"]<1e-6
        assert row["flux_error"]<1e-8
        assert row["total_energy_error"]<1e-7
        assert row["conditional_reduction_error"]<1e-10
        assert row["covariant_kinetic_energy"]>0
        assert row["dressed_internal_energy"]>0
    assert any(row["omitted_square_residual"]>100 for row in rows)
    assert max(row["omitted_square_residual"] for row in report["two_level"] if row["x"]<0 or row["x"]>1)<1e-6


def test_actual_split_generator_requires_fock_refinement(report):
    rows=report["fock"]
    # 최초 작은 절단의 실패도 산출물에 남긴다.
    assert rows[2]["gaussian_covariance_error"]>1e-3
    tail=[row for row in rows if row["cutoff"]>=10]
    assert all(a["gaussian_covariance_error"]>b["gaussian_covariance_error"] for a,b in zip(tail,tail[1:]))
    assert rows[-2]["gaussian_covariance_error"]<1e-3
    assert rows[-1]["gaussian_covariance_error"]<1e-3
    assert rows[-1]["boundary_occupation"]<1e-6
    assert all(row["generator_hermiticity_error"]<1e-10 for row in rows)


def test_finite_energy_limit_has_quantitative_operator_bound(report):
    rows=report["finite_energy"]
    assert all(a["coefficient_error"]>b["coefficient_error"] for a,b in zip(rows,rows[1:]))
    assert rows[-1]["coefficient_error"]<1e-4
    for row in rows:
        for item in row["propagator"]:
            assert item["operator_error"]<=item["bound"]+1e-12
    # 제약의 분산식에서 직접 구한 두 채널의 위상차와 대조한다.
    energy,mass=16.,1.7
    h=np.array([1.,3.])
    p=np.sqrt(2*mass*(energy-h))
    v0=math.sqrt(2*energy/mass)
    actual_gap=v0*(p[0]-p[1])
    values=checks.shell(h,energy,mass)["effective"]
    assert actual_gap==pytest.approx(values[1]-values[0],abs=1e-12)


def test_direction_is_not_selected_by_the_stationary_constraint(report):
    samples=report["directions"]["samples"]
    assert samples[0]["density"]>.5
    assert samples[2]["density"]>.5
    assert samples[1]["density"]<1e-25
    assert max(abs(row["flux"]) for row in samples)<1e-12
    model=checks.two_level()
    energy=4.
    a=np.array([1.,0.])
    p=checks.shell(model.levels,energy)["momentum"][0]
    for direction in (-1,1):
        value=model.state(.2,a,energy,direction=direction)
        assert np.vdot(value,direction*p*value).real==pytest.approx(direction,abs=1e-12)


def test_position_and_flux_probabilities_are_different_readouts(report):
    rows=report["readouts"]
    assert rows[0]["probability_total_variation"]>.2
    assert all(a["probability_total_variation"]>b["probability_total_variation"] for a,b in zip(rows,rows[1:]))
    assert rows[-1]["probability_total_variation"]<.003
    for row in rows:
        assert sum(row["position_probability"])==pytest.approx(1.)
        assert sum(row["flux_probability"])==pytest.approx(1.)
        assert row["energy_clock_schrodinger_error"]<1e-12
        assert row["position_clock_trace_distance"]>0


def test_energy_threshold_and_wrong_inputs_are_rejected():
    for energy in (2.,3.,math.nan,math.inf):
        with pytest.raises(ValueError):
            checks.shell([1.,2.,3.],energy)
    with pytest.raises(ValueError):
        checks.shell([-1.],4.)
    with pytest.raises(ValueError):
        checks.two_level().state(math.nan,[1.,0.],4.)
    data=checks.shell([1.,3.],3.+1e-8)
    assert 1/math.sqrt(data["velocity"][1])>80
    assert data["effective"][1]<2*(3.+1e-8)


def test_independent_difference_step_keeps_noncommuting_constraint():
    model=checks.two_level()
    a=np.array([1.,1j])/math.sqrt(2)
    for mass,energy,x in ((.7,4.,.35),(2.3,16.,.65)):
        rows=[checks.local_audit(model,x,a,energy,mass,step=h) for h in (1.2e-3,8e-4)]
        assert max(row["constraint_relative_residual"] for row in rows)<1e-6
        assert max(row["total_energy_error"] for row in rows)<1e-7
        assert min(row["omitted_square_residual"] for row in rows)>.1


def test_report_keeps_preparation_and_time_origin_open(report):
    scope=report["scope"]
    assert not scope["external_time_propagation_used"]
    assert scope["constraint_supplied"]
    assert scope["clock_coordinate_and_energy_sector_supplied"]
    assert scope["incoming_direction_supplied"]
    for name in ("physical_clock_origin_derived","persistent_record_order_derived",
                 "finite_normalizable_preparation_derived","readout_independent_correction",
                 "common_metric_or_einstein_limit_derived","dark_sector_or_hubble_result"):
        assert scope[name] is False


def test_saved_report_tracks_the_same_source_bytes():
    path=SOURCE.with_suffix(".json")
    if not path.exists():
        pytest.skip("산출물 작성 전 첫 선별")
    data=json.loads(path.read_text(encoding="utf-8"))
    for name,digest in data["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()==digest
