"""무시간 기록 장치의 에너지·위치 판독·배터리 문턱을 대조한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify/Q-0020/relational_record_clock.py"
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location("relational_record_checks",SOURCE)
checks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_actual_regge_phase_and_battery_overlap(report):
    assert all(row["actual_action_range"]>0 for row in report["resources"])
    assert max(row["level_spacing_error"] for row in report["resources"])<1e-14
    assert max(row["autocorrelation_error"] for row in report["resources"])<1e-12
    assert max(row["nearest_shift_error"] for row in report["resources"])<1e-12
    phase=report["phase_controls"]
    assert abs(phase[0]["q_flux"]-phase[1]["q_flux"])<1e-12
    assert abs(phase[-2]["q_flux"]-phase[-1]["q_flux"])>1e-3


def test_same_bare_energy_and_direct_matrix_control(report):
    row=report["matrix"]
    assert row["matrix_exponential_error"]<1e-11
    assert row["involution_error"]<1e-11
    assert row["hermitian_error"]<1e-11
    assert row["same_bare_energy_commutator"]<1e-11
    assert abs(row["generator_maximum"]-np.pi)<1e-11
    exchange=report["exchange"]
    assert exchange["minimum_prepared_battery_energy"]>0
    assert abs(exchange["system_energy_change"])>1e-3
    assert exchange["energy_balance_error"]<1e-11
    assert exchange["local_energy_balance_error"]<1e-11


def test_independent_constraint_and_record_current_differences(report):
    assert max(row["constraint_relative_residual"] for row in report["local"])<1e-6
    assert max(row["record_current_error"] for row in report["local"])<1e-8
    assert max(row["source_formula_error"] for row in report["local"])<1e-8
    assert max(row["local_pointer_error"] for row in report["local"])<1e-12
    assert max(row["energy_error"] for row in report["local"])<1e-8
    for row in report["local"]:
        assert abs(sum(row["record_currents"])-1)<1e-8
        assert row["record_source"]>=-1e-14
    center=[row for row in report["local"] if row["x"]==.5]
    assert min(row["omitted_square_residual"] for row in center)>.1


def test_packet_and_active_gap_budget(report):
    for row in report["resources"]:
        assert row["q_flux"]<=row["target_packet_bound"]+1e-12
        assert row["target_packet_bound"]<=row["target_asymptotic_bound"]+1e-12
        assert row["q_position"]<=row["position_bound"]+1e-12
        assert row["position_flux_difference"]<=row["total_variation"]+1e-12
        assert row["total_variation"]<=row["sharp_weight_bound"]+1e-12
        assert row["pointer_formula_error"]<1e-12
        assert row["norm_error"]<1e-12
    for n in (4,8):
        rows=[r for r in report["resources"] if r["size"]==n and r["kind"]=="length"]
        assert all(a["q_flux"]>b["q_flux"] for a,b in zip(rows,rows[1:]))
        assert all(a["q_position"]>b["q_position"] for a,b in zip(rows,rows[1:]))


def test_clock_threshold_can_spoil_an_accurate_record(report):
    for row in report["threshold"]:
        if row["scaled_gap"]==1e-16:
            assert abs(row["q_position"]-.75)<1e-4
            assert row["q_position"]>5*row["q_flux"]
    last=report["threshold"][-1]
    assert last["q_flux"]<.02
    assert last["q_position"]>.74


def test_reversal_erases_record_and_archive_prevents_full_return(report):
    row=report["matrix"]
    assert row["reverse_return_error"]<1e-12
    assert row["reverse_pointer_one"]<1e-24
    assert abs(row["archive_fidelity"]-row["archive_fidelity_formula"])<1e-12
    assert row["archive_fidelity"]<.99
    standing=report["standing_wave"]
    assert max(abs(row["current"]) for row in standing)<1e-8
    assert standing[0]["density"]>0
    assert standing[1]["density"]<1e-28


def test_weight_bound_is_sharp_for_a_binary_distribution():
    for ratio in (1.01,2.,16.,1e4):
        probability=np.array([np.sqrt(ratio),1.])/(1+np.sqrt(ratio))
        weights=np.array([1.,ratio])
        tilted=probability*weights/(probability@weights)
        tv=sum(abs(tilted-probability))/2
        assert abs(tv-(np.sqrt(ratio)-1)/(np.sqrt(ratio)+1))<1e-12


def test_nonuniform_threshold_and_strict_gap():
    model=checks.RecordClock(4,16,"squared","conditional")
    row=model.stats(model.u,model.delta*1e-16)
    assert abs(row["q_position"]-(1-7/16))<1e-4
    for gap in (0.,-1.,np.inf,np.nan):
        with pytest.raises(ValueError):
            model.stats(model.u,gap)
    with pytest.raises(ValueError):
        checks.RecordClock(True,8)
    with pytest.raises(ValueError):
        model.prepare(np.ones(4))
    # 문턱 계산은 큰 E에서 뺄셈하지 않고 양의 간격 자체를 유지한다.
    assert model.kinetic(1e-20)[-1]==1e-20


def test_saved_report_tracks_sources():
    path=SOURCE.with_suffix(".json")
    if not path.exists():
        pytest.skip("최초 선별 후 산출물 기록")
    report=json.loads(path.read_text(encoding="utf-8"))
    for name,sha in report["dependencies"].items():
        assert hashlib.sha256(SOURCE.with_name(name).read_bytes()).hexdigest()==sha
