"""기록 누설의 파동·행렬·극한과 장치 포함 제약의 보존을 독립 대조한다."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"verify"/"Q-0020"))
import regge_record_constraint as model


@pytest.mark.parametrize("kind",["length","squared"])
@pytest.mark.parametrize("target",["conditional","constraint"])
def test_wave_integrals_and_full_energy_blocks_agree_with_overlap_formula(kind,target):
    for width in (2.,8.):
        row=model.direct_wave_check(kind=kind,target=target,width=width)
        assert row["formula_error"]<1e-11
        assert row["norm_error"]<1e-11
        assert row["leakage"]>1e-4
    finite=model.finite_block_check(kind=kind,target=target)
    assert max(v for k,v in finite.items() if k.endswith("error"))<1e-12
    assert finite["leakage"]>1e-3


@pytest.mark.parametrize("target",["conditional","constraint"])
def test_actual_fibers_converge_with_both_measures_and_nonzero_phase(target):
    for kind in ("length","squared"):
        for y in (.75,1.,1.03):
            for beta in (0.,1.,5.):
                a=model.grid_check(64,y,beta,kind,target)
                b=model.grid_check(128,y,beta,kind,target)
                for x,z in zip(a["rows"],b["rows"]):
                    assert 0<z["leakage"]<=1
                    assert abs(z["leakage"]-x["leakage"])<max(.02*z["leakage"],1e-6)
                    if target=="constraint":
                        assert z["leakage"]<=z["upper_bound"]
                        assert z["continuum_difference"]<x["continuum_difference"]


def test_distinct_infinite_resource_limits_and_adapted_phase_independence():
    bare=model.grid_check(128,1.,0.,"squared","conditional")["rows"][-1]
    assert bare["ideal_overlap"]==pytest.approx(8/9,abs=2e-4)
    assert bare["leakage"]==pytest.approx(16/81,abs=3e-4)
    values=[model.adapted_continuum(w) for w in (8.,32.,128.,512.)]
    assert all(0<r["leakage"]<r["upper_bound"] for r in values)
    errors=[r["asymptotic"]-r["scaled"] for r in values]
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert errors[-1]/values[-1]["asymptotic"]<.002
    a=model.grid_check(64,1.,0.,"length","constraint")
    b=model.grid_check(64,1.,5.,"squared","constraint")
    assert [r["leakage"] for r in a["rows"]]==pytest.approx([r["leakage"] for r in b["rows"]],abs=1e-12)


@pytest.mark.parametrize("kind",["length","squared"])
def test_apparatus_constraint_cancels_actual_regge_derivative_but_system_alone_does_not(kind):
    for y in (.75,1.,1.03):
        for beta in (0.,5.):
            for fraction in (.2,.6,.85):
                row=model.coupled_constraint_check(y,beta,kind,fraction)
                assert max(v for k,v in row.items() if k.endswith("error"))<1e-8
                assert row["system_constraint_residual"]>.1
                assert row["fixed_b_boundary_defect"]>.1
                assert row["minimum_boundary_battery"]>0


def test_coupled_spectrum_and_energy_preserve_zero_mode_record():
    row=model.coupled_spectrum_check()
    assert max(v for k,v in row.items() if k.endswith("error"))<1e-10
    for mode in row["modes"]:
        assert mode["eigen_error"]<1e-10
        assert mode["zero_record_probability"]==pytest.approx(float(mode["mode"]==0),abs=1e-12)
