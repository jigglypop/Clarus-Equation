"""정준 길이 축약의 입력 의존성과 실제 Regge 작용의 경계 미분을 검산한다."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "verify" / "Q-0020"
original = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("canonical_lengths_under_test", HERE/"canonical_length_reduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original


@pytest.mark.parametrize("depth", [1, 2])
def test_reduction_preserves_canonical_form_and_constrained_energy(depth):
    data = module.gluing.length_gluing(depth)
    a, c = data["assembly"], data["constraint"]
    w = np.linspace(.6, 1.4, len(a))
    reduced = module.canonical_reduction(a, c, w)
    b = reduced["momentum_lift"]
    np.testing.assert_allclose(a.T @ b, np.eye(a.shape[1]), atol=1e-13)
    np.testing.assert_allclose((c/w) @ b, 0, atol=1e-13)
    np.testing.assert_allclose(reduced["dirac_qp"], a @ b.T, atol=1e-12)
    rng = np.random.default_rng(20260905)
    q, p = rng.normal(size=(2, a.shape[1]))
    stiffness = np.diag(np.linspace(1., 2., len(a)))
    full_energy = ((b @ p/w) @ (b @ p)+(a @ q) @ stiffness @ (a @ q))/2
    reduced_energy = (p @ np.linalg.solve(reduced["mass"], p)+q @ (a.T @ stiffness @ a) @ q)/2
    assert full_energy == pytest.approx(reduced_energy, rel=1e-13)
    row = module.reduction_case(depth)
    for choice in row["choices"].values():
        assert choice["symplectic_residual"] < 1e-12
        assert choice["compatible_flow_residual"] < 1e-12
        assert choice["incompatible_flow_residual"] > .01


def test_full_dirac_constraint_bracket_agrees_with_reduction():
    data = module.gluing.length_gluing(1)
    a, c = data["assembly"], data["constraint"]
    w = np.linspace(.7, 1.3, len(a))
    reduced = module.canonical_reduction(a, c, w)
    rows = reduced["independent_constraint"]
    zero = np.zeros_like(rows)
    constraints = np.block([[rows, zero], [zero, rows/w]])
    omega = module._omega(len(a))
    direct = omega-omega @ constraints.T @ np.linalg.solve(constraints @ omega @ constraints.T, constraints @ omega)
    np.testing.assert_allclose(direct[:len(a), len(a):], reduced["dirac_qp"], atol=1e-12)
    np.testing.assert_allclose(constraints @ direct, 0, atol=1e-12)


@pytest.mark.parametrize("weights", [np.zeros(50), np.full(50, -1.), np.full(50, np.nan), np.ones(49)])
def test_invalid_mass_is_rejected(weights):
    data = module.gluing.length_gluing(1)
    with pytest.raises(ValueError):
        module.canonical_reduction(data["assembly"], data["constraint"], weights)


def test_missing_constraint_and_unowned_global_edge_are_rejected():
    data = module.gluing.length_gluing(1)
    a, c = data["assembly"], data["constraint"]
    with pytest.raises(ValueError):
        module.canonical_reduction(a, np.zeros_like(c), np.ones(len(a)))
    with pytest.raises(ValueError):
        module.canonical_reduction(np.column_stack((a, np.zeros(len(a)))), c, np.ones(len(a)))


def test_same_gluing_has_distinct_pure_vacua_with_different_mass_inputs():
    data = module.gluing.length_gluing(1)
    a, c = data["assembly"], data["constraint"]
    owners = a.sum(axis=0)
    np.testing.assert_equal(np.sort(owners), np.r_[np.full(10, 3.), np.full(5, 4.)])
    q0, p0 = module.oscillator_covariance(module.canonical_reduction(a, c, np.ones(len(a))))
    q1, p1 = module.oscillator_covariance(module.canonical_reduction(a, c, 1/(a @ owners)))
    np.testing.assert_allclose(q0 @ p0, np.eye(len(owners))/4, atol=1e-14)
    np.testing.assert_allclose(q1 @ p1, np.eye(len(owners))/4, atol=1e-14)
    np.testing.assert_allclose(np.diag(q0), 1/(4*owners), atol=1e-14)
    np.testing.assert_allclose(q1, np.eye(len(owners))/4, atol=1e-14)
    np.testing.assert_allclose(c @ a @ q0 @ a.T @ c.T, 0, atol=1e-14)
    np.testing.assert_allclose(c @ a @ q1 @ a.T @ c.T, 0, atol=1e-14)


def test_actual_refinement_distinguishes_mass_conservation_from_copy_counting():
    row = module.refinement_case()
    np.testing.assert_allclose(row["fine_old_owner_counts"], 3*np.array(row["coarse_owner_counts"]))
    np.testing.assert_allclose(row["unit_copy_old_variance_ratio"], 1/3, atol=1e-14)
    np.testing.assert_allclose(row["additive_old_variance_ratio"], 1, atol=1e-14)
    assert row["additive_old_mass_residual"] < 1e-12
    np.testing.assert_allclose(row["new_edge_masses"], 1., atol=1e-14)
    alternative = module.refinement_case(2.)
    assert alternative["additive_old_mass_residual"] < 1e-12
    np.testing.assert_allclose(alternative["new_edge_masses"], 2., atol=1e-14)


@pytest.mark.parametrize("step", [1e-3, 5e-4])
def test_regular_boundary_momentum_matches_independent_closed_form(step):
    row = module.regge_boundary_case(step=step)
    angle = math.pi-math.acos(.25)
    assert row["fine_action"] == pytest.approx(5*math.sqrt(3)*angle, abs=1e-12)
    np.testing.assert_allclose(row["boundary_momentum"], math.sqrt(3/2)*angle, atol=2e-8)
    assert abs(row["action_difference"]) < 1e-12
    assert row["copy_to_global_chain_residual"] < 1e-8
    assert row["internal_gradient_norm"] < 1e-8
    assert row["boundary_gradient_residual"] < 1e-8


def test_flat_irregular_subdivision_with_moved_center_preserves_boundary_gradient():
    row = module.regge_boundary_case(irregular=True, moved_center=True)
    assert abs(row["action_difference"]) < 1e-11
    assert row["copy_to_global_chain_residual"] < 1e-7
    assert row["internal_gradient_norm"] < 1e-7
    assert row["boundary_gradient_residual"] < 1e-7


def test_independent_internal_length_change_breaks_on_shell_identity():
    row = module.regge_boundary_case(relative_internal_shift=.02)
    assert row["copy_to_global_chain_residual"] < 1e-7
    assert abs(row["action_difference"]) > 1e-5
    assert row["internal_gradient_norm"] > 1e-3
    assert row["boundary_gradient_residual"] > 1e-3


def test_recorded_artifact_matches_sources_and_retains_physical_obligations():
    report = json.loads((HERE/"canonical_length_reduction.json").read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    for key in ("unique_vacuum_selected_by_length_gluing", "oscillator_mass_derived_from_regge",
                "finite_energy_preparation_in_original_hilbert_space_constructed",
                "internal_vertex_gauge_removed", "common_metric_selected", "lorentzian_einstein_limit_derived"):
        assert report["scope"][key] is False
