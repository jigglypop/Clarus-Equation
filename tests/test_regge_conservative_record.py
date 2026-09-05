"""기록 오차의 연속 상계와 양의 유한 배터리의 보존 전달을 대조한다."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"verify"/"Q-0020"))
import regge_conservative_record as model


@pytest.mark.parametrize("width",[2.,16.])
def test_sine_overlap_and_battery_moments_match_independent_integrals(width):
    result=model.overlap_check(width)
    assert max(v for k,v in result.items() if k.endswith("error"))<1e-10
    assert float(model.sine_defect(1e-8,width))>0
    assert float(model.sine_defect(2*width,width))==pytest.approx(1.)


@pytest.mark.parametrize("kind",["length","squared"])
def test_actual_record_noise_converges_and_covers_prepared_inputs(kind):
    coarse=model.mesh_check(32,kind)
    fine=model.mesh_check(64,kind)
    for first,last in zip(coarse["record"],fine["record"]):
        assert abs(last["worst_noise_squared"]/first["worst_noise_squared"]-1)<.02
        assert last["continuum_difference"]<first["continuum_difference"]
        assert last["continuum_noise_squared"]<=last["upper_bound_squared"]
        assert max(last["input_noise_squared"].values())<=last["worst_noise_squared"]+1e-12
        assert last["core_spectrum_error"]<1e-12
        assert last["core_minimum_effect"]>-1e-12
    assert all(b["worst_noise_squared"]<a["worst_noise_squared"] for a,b in zip(fine["record"],fine["record"][1:]))


@pytest.mark.parametrize("kind",["length","squared"])
def test_finite_positive_battery_unitary_preserves_total_energy_and_record_bound(kind):
    result=model.finite_battery_check(4,16,kind)
    assert result["minimum_battery_energy"]>=0
    assert max(v for k,v in result.items() if k.endswith("error"))<1e-10
    assert result["witness_noise_squared"]>=result["way_lower_bound"]-1e-12
    assert abs(result["system_energy_change"])>1e-3
    assert result["system_energy_change"]+result["battery_energy_change"]==pytest.approx(0.,abs=1e-12)


def test_constructive_resources_meet_rms_targets_and_necessary_bounds():
    for row in model.resource_table():
        assert row["constructed_rms"]<=row["target_rms"]
        assert row["battery_std_sufficient"]>row["apparatus_std_necessary"]
        assert row["battery_mean"]>0
    ratio=math.sqrt(math.pi**2/3-2)
    assert 1<ratio<1.14


def test_invalid_width_and_measure_are_rejected():
    for width in (0.,-1.,float("inf")):
        with pytest.raises(ValueError):
            model.sine_defect(1.,width)
    with pytest.raises(ValueError):
        model.core_noise(2.,"unknown")


@pytest.mark.parametrize("kind",["length","squared"])
@pytest.mark.parametrize("size,packet",[(3,8),(4,16)])
def test_sequential_record_preserves_single_record_energy_and_full_inverse(kind,size,packet):
    single=model.finite_battery_check(size,packet,kind)
    for count in (1,2,3,5):
        row=model.repeated_record_check(size,packet,kind,count)
        assert max(v for k,v in row.items() if k.endswith("error"))<1e-12
        assert row["mixed_record_probability_bound"]<1e-12
        for key in ("noise_squared","system_energy_change","battery_energy_change"):
            assert row[key]==pytest.approx(single[key],abs=1e-12)
        assert row["archive_probability"]==pytest.approx(.5,abs=1e-12)
        assert row["archive_recovery_fidelity"]==pytest.approx(.5,abs=1e-12)
        assert abs(row["system_energy_change"])>1e-3
