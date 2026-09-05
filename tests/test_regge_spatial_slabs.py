"""두 구간 Lorentz 레게 작용의 합성, 변분과 대칭축약 반례를 검사한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/"verify/Q-0020/regge_spatial_slabs.py"
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location("regge_spatial_slabs_checks",SOURCE)
checks=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_two_slab_action_gradient_and_shift_charge_compose(report):
    row=report["composition"]
    assert row["cell_count"]==40 and row["internal_edge_count"]==40
    assert row["internal_vertex_count"]==5 and row["inertia_margin"]>1e-3
    for key in ("action_error","complex_regge_error","gradient_error","scalar_shift_error"):
        assert row[key]<1e-10
    assert report["gradient"]["edge_samples"]==6
    assert report["gradient"]["field_samples"]==5
    assert report["gradient"]["edge_error"]<1e-7
    assert report["gradient"]["field_error"]<1e-7


def test_symmetric_stationary_point_fails_original_internal_equations(report):
    row=report["symmetric"]
    assert row["success"] and row["max_reduced_residual"]<1e-7
    assert row["max_full_edge_residual"]>1
    assert row["max_full_field_residual"]>.1
    assert row["inertia_margin"]>1e-3


def test_unrestricted_internal_solution_is_lorentz_regular_and_isolated(report):
    row=report["full"]
    assert row["success"] and row["evaluations"]<100
    assert row["max_equation_residual"]<2e-7
    assert row["inertia_margin"]>1e-3
    assert row["hessian_asymmetry"]<1e-5
    assert row["hessian_rank_1e8"]==45
    assert row["hessian_singular_values"][-1]>.1
    assert row["scalar_shift_error"]<1e-10
    assert sum(row["boundary_field_charges"])==pytest.approx(0,abs=1e-10)
    assert min(row["internal_squared"])<0<max(row["internal_squared"])


def test_geometry_and_input_guards():
    assert len(checks.FULL.cells)==40
    assert all(len(checks.FULL.owners[t]) in (1,2,3,4,5,6) for t in checks.FULL.triangles)
    with pytest.raises(ValueError):
        checks.family([1,1],[.1,.1],[0,0,0])
    with pytest.raises(ValueError):
        checks.family([1,1,1],[0,.1],[0,0,0])
    with pytest.raises(ValueError):
        checks.restricted([1,1,1],[.1,.1],[0,0,0],beta=-1)


def test_saved_artifact_source_hash():
    artifact=SOURCE.with_suffix(".json")
    if not artifact.exists():
        pytest.skip("첫 수치 실행 뒤 산출물을 저장한다")
    data=json.loads(artifact.read_text(encoding="utf-8"))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==data["dependencies"][SOURCE.name]
