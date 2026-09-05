"""면적–각 접착, 독립 길이 복원, 계량 부호와 실제 작용을 대조한다."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "verify/Q-0020/simplex_boundary_reconstruction.py"
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("simplex_boundary_checks", SOURCE)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def report():
    return checks.run()


def test_exact_linear_constraints_and_area_local_inverse(report):
    row = report["linear"]
    assert row["petersen_spectrum"] == {"3":1, "1":5, "-2":4}
    assert row["area_squared_jacobian_determinant"] == "3/67108864"
    assert (row["closure_rank"],row["matching_rank"]) == (20,20)
    assert row["combined_rank"] == row["fixed_area_rank"] == 30
    assert row["geometry_tangent_rank"] == 10
    for key in ("constraint_jacobian_error","geometry_tangent_constraint_error",
                "area_squared_jacobian_error"):
        assert row[key] < 1e-7
    assert row["fixed_area_smallest_singular"] > .28
    inverse = report["inverse_area"]
    assert inverse["success"] and inverse["gram_minimum"] > 0
    assert inverse["length_error"] < 1e-10 and inverse["area_error"] < 1e-10


def test_boundary_shapes_reconstruct_each_owner_and_original_action(report):
    for row in report["reconstruction"] + report["signature"]:
        for key in ("constraints_error","restored_edge_error","owner_edge_mismatch",
                    "four_angle_choice_error"):
            assert row[key] < 1e-10
        assert min(row["tetrahedron_volumes"]) > 0
        if row["euclidean_action_error"] is not None:
            assert row["euclidean_action_error"] < 1e-10
            assert row["euclidean_gradient_error"] < 1e-7


def test_matching_boundary_does_not_fix_euclidean_signature_or_time_arrow(report):
    lorentz, degenerate, euclidean, regular = report["signature"]
    assert lorentz["inertia"] == [1,0,3] and lorentz["all_edges_spacelike"]
    assert lorentz["gram_determinant"] == pytest.approx(-1/96, abs=1e-14)
    assert degenerate["inertia"] == [0,1,3]
    assert euclidean["inertia"] == regular["inertia"] == [0,0,4]
    for row in report["signature"]:
        assert row["side_determinant_error"] < 1e-10
        assert row["full_determinant_error"] < 1e-10
        if row["coordinate_reconstruction_error"] is not None:
            assert row["coordinate_reconstruction_error"] < 1e-10
            assert row["time_reflection_edge_error"] < 1e-10
    for r2 in (17/48,3/8):
        x = checks.apex_family(r2)
        if r2 < 3/8:
            with pytest.raises(ValueError):
                checks.SIMPLEX.evaluate(np.sqrt(x))
        with pytest.raises(ValueError):
            checks.boundary_action(checks.intrinsic(x))


def test_equal_face_areas_and_closure_do_not_imply_shape_matching(report):
    for row in report["unglued"]:
        assert row["area_change"] == 0
        assert row["closure_error"] < 1e-10
        assert row["changed_tetrahedron_gram_minimum"] > 0
        assert row["matching_defect"] > .01
        assert row["owner_edge_mismatch"] > .01


def test_original_lorentz_action_and_scalar_cannot_stationarize_this_family(report):
    imaginary = []
    for row in report["lorentz_action"]:
        for key in ("regge_action_error","regge_gradient_error",
                    "scalar_action_error","scalar_gradient_error",
                    "scalar_volume_error","scalar_norm_relative_error"):
            assert row[key] < 1e-10
        assert row["regge_difference_error"] < 1e-7
        assert row["total_derivative"] < 0
        assert row["total_derivative"] == pytest.approx(row["derivative_formula"], abs=1e-10)
        imaginary.append(row["regge_imaginary"])
    assert max(imaginary)-min(imaginary) < 1e-10


def test_invalid_and_degenerate_data_are_rejected():
    for value in (np.ones(9), np.zeros(10), np.full(10,np.nan)):
        with pytest.raises(ValueError):
            checks.intrinsic(value)
    for r2 in (1/3,0,float("nan")):
        with pytest.raises(ValueError):
            checks.apex_family(r2)
    for T in (0,1/np.sqrt(24),1):
        with pytest.raises(ValueError):
            checks.lorentz_action_case(T)
    broken = checks.intrinsic(np.ones(10))
    broken[10] += .1
    with pytest.raises(ValueError):
        checks.reconstructed_edges(broken)


def test_saved_artifact_sources():
    artifact = SOURCE.with_suffix(".json")
    if not artifact.exists():
        pytest.skip("첫 수치 실행 뒤 검산 파일을 저장한다")
    data = json.loads(artifact.read_text(encoding="utf-8"))
    for name,digest in data["dependencies"].items():
        assert hashlib.sha256((SOURCE.parent/name).read_bytes()).hexdigest() == digest
