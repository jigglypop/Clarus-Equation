"""새 모서리의 실제 작용·곡률·제약과 준비를 바꾼 역합성을 대조한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1]/"verify/Q-0020/regge_pachner_creation.py"
SPEC = importlib.util.spec_from_file_location("ce_pachner_creation_checks", SOURCE)
pachner = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SOURCE.parent))
try:
    SPEC.loader.exec_module(pachner)
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def move():
    return pachner.PachnerCreation()


@pytest.fixture(scope="module")
def fibers():
    return pachner.creation_fibers()


def test_real_topology_changes_boundary_hinge_to_internal(move):
    assert (3, 4) not in move.old.edges
    assert set(move.new.edges)-set(move.old.edges) == {(3, 4)}
    assert (len(move.old.edges), len(move.new.edges)) == (14, 15)
    assert (len(move.old.triangles), len(move.new.triangles)) == (16, 19)
    assert np.all(move.old.boundary)
    assert [t for t, boundary in zip(move.new.triangles, move.new.boundary) if not boundary] == [(0, 1, 2)]


def test_unit_local_geometry_has_exact_open_interval(move):
    lengths = move.old.lengths(pachner.reference_points())
    assert lengths[move.local_old_ids] == pytest.approx(np.ones(9))
    low, high = pachner.admissible_interval(lengths[move.local_old_ids])
    assert low == pytest.approx(0, abs=1e-14)
    assert high == pytest.approx(math.sqrt(8/3), abs=1e-14)


def test_general_interval_agrees_with_independent_gram_geometry(move):
    points = pachner.reference_points()
    points[3] += np.array([.12, -.08, .17, .04])
    points[4] += np.array([-.07, .09, .06, -.13])
    old = move.old.lengths(points)
    low, high = pachner.admissible_interval(old[move.local_old_ids])
    assert 0 < low < np.linalg.norm(points[3]-points[4]) < high
    def gram(y):
        distances = np.zeros((5, 5))
        edge_lengths = dict(zip(move.old.edges, old))
        edge_lengths[3, 4] = y
        for (i, j), value in edge_lengths.items():
            if i < 5 and j < 5:
                distances[i, j] = distances[j, i] = value**2
        return (distances[0, 1:, None]+distances[None, 0, 1:]-distances[1:, 1:])/2
    assert np.linalg.eigvalsh(gram((low+high)/2))[0] > 0
    assert abs(np.linalg.det(gram(low))) < 1e-13
    assert abs(np.linalg.det(gram(high))) < 1e-13
    assert np.linalg.eigvalsh(gram(.99*low))[0] < 0
    assert np.linalg.eigvalsh(gram(1.01*high))[0] < 0


@pytest.mark.parametrize("y", [0, pachner.LENGTH_LIMIT, -1, float("nan")])
def test_invalid_new_length_is_rejected(move, y):
    with pytest.raises(ValueError):
        move.lengths_after(move.old.lengths(pachner.reference_points()), y)


@pytest.mark.parametrize("z", [.1, .4, .95])
def test_full_bulk_action_and_gradient_agree_with_local_increment(move, z):
    old = move.old.lengths(pachner.reference_points())
    data = move.evaluate(old, z*pachner.LENGTH_LIMIT, beta=1.7)
    assert abs(data["action_residual"]) < 2e-13
    assert data["gradient_residual"] < 2e-12
    assert data["curvature"] == pytest.approx(math.pi/2-2*math.asin(z), abs=3e-12)
    # 전체 새·이전 작용을 따로 차분하므로 국소 미분 구현을 그대로 검사하지 않는다.
    def full_difference(lengths):
        return move.new.evaluate(lengths, 1.7)["action"]-move.old.evaluate(lengths[move.old_ids], 1.7)["action"]
    for index in range(len(move.new.edges)):
        direction = np.zeros(len(move.new.edges))
        direction[index] = 2e-6
        finite = (full_difference(data["lengths"]+direction)-full_difference(data["lengths"]-direction))/(4e-6)
        assert finite == pytest.approx(data["gradient"][index], abs=4e-7)


def test_creation_and_inverse_require_new_momentum_constraint(move):
    old = move.old.lengths(pachner.reference_points())
    old_momenta = np.linspace(-1, 2, len(old))
    lengths, momenta = move.create(old, old_momenta, .6*pachner.LENGTH_LIMIT)
    recovered_lengths, recovered_momenta = move.undo(lengths, momenta)
    np.testing.assert_allclose(recovered_lengths, old, atol=1e-14)
    np.testing.assert_allclose(recovered_momenta, old_momenta, atol=1e-14)
    momenta[move.new_id] += .01
    with pytest.raises(ValueError, match="운동량 제약"):
        move.undo(lengths, momenta)


def test_normalized_profiles_have_different_actual_curvature_means(fibers):
    expected = {"uniform": 2-math.pi/2, "first": 296/15-49*math.pi/8,
                "second": 1264/35-183*math.pi/16}
    for kind, curvature in expected.items():
        assert fibers["records"][kind]["norm_squared"] == pytest.approx(1, abs=2e-13)
        assert fibers["records"][kind]["curvature_mean"] == pytest.approx(curvature, abs=3e-12)
    assert abs(expected["first"]-expected["second"]) > .25


def test_profiles_do_not_solve_same_exact_quantum_constraint(fibers):
    assert fibers["records"]["uniform"]["constraint_residual_squared"] == 0
    assert fibers["records"]["first"]["constraint_residual_squared"] == pytest.approx(15/4, abs=1e-12)
    assert fibers["records"]["second"]["constraint_residual_squared"] == pytest.approx(21/4, abs=1e-12)
    for kind in ("first", "second"):
        chi, _ = pachner.preparation(kind, [0, pachner.LENGTH_LIMIT])
        assert chi.tolist() == [0, 0]


def test_inverse_is_identity_on_image_but_not_on_arbitrary_fine_state(fibers):
    v1, v2 = fibers["vectors"][:, 1], fibers["vectors"][:, 2]
    j1 = np.kron(np.eye(3), v1[:, None])
    j2 = np.kron(np.eye(3), v2[:, None])
    np.testing.assert_allclose(j1.conj().T @ j1, np.eye(3), atol=2e-13)
    np.testing.assert_allclose(j2.conj().T @ j2, np.eye(3), atol=2e-13)
    np.testing.assert_allclose(j1.conj().T @ j2, math.sqrt(14)/4*np.eye(3), atol=2e-13)
    # 부모와 별도 참조계가 얽힌 입력에서도 같은 역합성이 성립한다.
    joint = np.array([[1, 1j], [.3, -.2j], [1j, .7]])
    joint /= np.linalg.norm(joint)
    np.testing.assert_allclose(j1.conj().T @ (j1 @ joint), joint, atol=2e-13)
    projection = np.outer(v1, v1.conj())
    np.testing.assert_allclose(projection @ projection, projection, atol=2e-13)
    assert np.linalg.norm(projection @ v2)**2 == pytest.approx(7/8, abs=2e-13)
    assert np.linalg.norm(v2-projection @ v2)**2 == pytest.approx(1/8, abs=2e-13)


def test_report_matches_current_sources_and_preserves_scope():
    report = json.loads(SOURCE.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert report["geometry_source_sha256"] == hashlib.sha256(SOURCE.with_name("regge_tent_transfer.py").read_bytes()).hexdigest()
    assert report["quadrature_order_convergence"] < 1e-11
    assert report["curvature_exact_residual"] < 1e-11
    assert "물리 에너지가 아니다" in report["alternatives"]["dirichlet_preparations"]
    assert "자기수반 도메인" in report["alternatives"]["phase_only"]
    assert "공통 계량" in " ".join(report["unfinished"])
