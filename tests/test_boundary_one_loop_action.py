"""Independent checks for the supplied Regge boundary one-loop action."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]/"verify"/"Q-0020"
original = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("boundary_one_loop_under_test", HERE/"boundary_one_loop_action.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original


def irregular_boundary():
    return module.REGULAR*(1+.08*np.random.default_rng(20260905).normal(size=10))


def test_exact_centroid_formula_recovers_regular_curvature_and_shape_tensor():
    row = module.centroid_one_loop(module.REGULAR)
    (p1, p4, p5), _ = module.shape_projectors()
    assert row["curvature"] == pytest.approx(40*math.sqrt(5), rel=1e-13)
    np.testing.assert_allclose(row["gradient"], 0., atol=1e-14)
    np.testing.assert_allclose(row["hessian"], .4*p4+p5, atol=1e-13)
    np.testing.assert_allclose(row["hessian"] @ p1, 0., atol=1e-13)
    for projector, rank in zip((p1, p4, p5), (1, 4, 5)):
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-14)
        assert np.trace(projector) == pytest.approx(rank, abs=1e-14)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("shear", [0., .2])
def test_general_boundary_assembly_matches_previous_independent_geometry(depth, shear):
    r = module.fixed.gluing.reference()
    transform = np.eye(4)
    transform[0, 1] = shear
    points = {v: transform @ p for v, p in r.points_from_squared(module.REGULAR**2).items()}
    b = r.cell_lengths(tuple(range(5)), points)
    row = module.evaluate(b, depth)
    previous = module.fixed.composition(depth, shear=shear)
    assert row["phi"] == pytest.approx(-previous["orthonormal_quotient_log_integral"], abs=1e-10)
    assert row["rank"] == previous["physical_dimension"]


@pytest.mark.parametrize("depth", [1, 2])
def test_irregular_centroid_formula_and_direct_and_sequential_integrals_agree(depth):
    row = module.evaluate(irregular_boundary(), depth)
    assert row["smallest_internal_curvature"] > 0
    assert row["phi"] == pytest.approx(row["phi_sequential"], abs=1e-10)
    assert row["phi"] == pytest.approx(row["centroid_formula_phi"], abs=1e-10)
    for name in ("boundary_schur_residual", "gauge_hessian_residual",
                 "internal_gradient_residual", "classical_action_residual"):
        assert row[name] < 2e-9
    assert row["gamma"] == pytest.approx(row["classical_action"]+row["phi"], abs=1e-12)


def test_exact_centroid_derivatives_satisfy_euler_identity_and_action_differences():
    b = irregular_boundary()
    row = module.centroid_one_loop(b)
    assert b @ row["gradient"] == pytest.approx(0., abs=1e-13)
    np.testing.assert_allclose(row["hessian"] @ b, -row["gradient"], atol=1e-12)
    u = np.random.default_rng(317).normal(size=10)
    u /= np.linalg.norm(u)
    h = 1e-5
    plus, minus = module.centroid_one_loop(b+h*u), module.centroid_one_loop(b-h*u)
    assert (plus["phi"]-minus["phi"])/(2*h) == pytest.approx(row["gradient"] @ u, abs=1e-8)
    np.testing.assert_allclose((plus["gradient"]-minus["gradient"])/(2*h),
                               row["hessian"] @ u, atol=1e-8, rtol=1e-8)
    assert module.evaluate(b, 1)["smallest_internal_curvature"] == pytest.approx(row["curvature"], rel=1e-11)


@pytest.mark.parametrize("depth", [1, 2])
def test_scale_and_vertex_relabeling_preserve_shape_potential(depth):
    b = irregular_boundary()
    row = module.evaluate(b, depth)
    permutation = [3, 0, 4, 1, 2]
    permuted = np.array([b[module.EDGES.index(tuple(sorted((permutation[u], permutation[v]))))]
                         for u, v in module.EDGES])
    assert module.evaluate(permuted, depth)["phi"] == pytest.approx(row["phi"], abs=1e-10)
    scaled = module.evaluate(2*b, depth)
    assert scaled["phi"] == pytest.approx(row["phi"], abs=1e-10)
    assert scaled["classical_action"] == pytest.approx(4*row["classical_action"], abs=1e-10)
    beta_changed = module.evaluate(b, depth, beta=.7)
    assert beta_changed["phi"]-row["phi"] == pytest.approx(row["rank"]*math.log(.7)/2, abs=1e-10)


@pytest.mark.parametrize("depth", [1, 2])
def test_shape_tensor_matches_independent_directions_and_step_refinement(depth):
    coarse = module.regular_shape_case(depth, .004)
    fine = module.regular_shape_case(depth, .002)
    hessian = np.array(fine["phi_hessian"])
    np.testing.assert_allclose(coarse["phi_hessian"], hessian, atol=5e-7, rtol=5e-7)
    assert fine["sequential_curvature_residual"] < 1e-7
    rng = np.random.default_rng(819)
    for u in (rng.normal(size=10), module.REGULAR/np.linalg.norm(module.REGULAR)):
        u /= np.linalg.norm(u)
        gradient, curvature, _ = module.directional_derivatives(module.REGULAR, u, depth, .002)
        assert abs(gradient[0]) < 1e-7
        assert curvature[0] == pytest.approx(u @ hessian @ u, abs=8e-7)
        assert curvature[1] == pytest.approx(curvature[0], abs=1e-7)
    if depth == 1:
        assert fine["exact_one_insertion_hessian_residual"] < 5e-7


def test_shape_dependent_jacobian_cannot_be_omitted():
    u4 = module.shape_projectors()[1][0]
    _, curvature, _ = module.directional_derivatives(module.REGULAR, u4, 2, .004)
    assert abs(curvature[3]) > 9e-4
    assert curvature[0] == pytest.approx(curvature[2]+curvature[3], abs=1e-7)
    assert abs(curvature[0]-curvature[2]) > 9e-4


def test_loop_correction_changes_affine_shape_curvature_without_proving_a_vacuum():
    first = module.regular_shape_case(1)
    second = module.regular_shape_case(2)
    assert max(first["gamma_affine_shape_curvatures_beta_1"]) < 0
    assert min(second["gamma_affine_shape_curvatures_beta_1"]) > .6
    classical = np.array(second["classical_affine_shape_curvatures"])
    loop = np.array(second["phi_shape_curvatures_multiplicity_4_5"])
    assert max(2*classical+loop) < 0
    # The regular point is not stationary in the unconstrained ten-dimensional space.
    coarse_gradient = module.fixed.simplex_derivatives(module.REGULAR, np.full(10, math.pi))[1]
    assert np.linalg.norm(coarse_gradient) > 1.


@pytest.mark.parametrize("bad", [np.zeros(10), np.ones(9), np.full(10, math.nan),
                                 np.r_[10., np.ones(9)]])
def test_invalid_boundary_is_rejected_by_both_routes(bad):
    for function in (module.evaluate, module.centroid_one_loop):
        with pytest.raises(ValueError):
            function(bad)


@pytest.mark.parametrize("kwargs", [{"depth": True}, {"depth": 3}, {"beta": 0.}, {"beta": math.nan}])
def test_invalid_integral_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        module.evaluate(module.REGULAR, **kwargs)


def test_artifact_sources_and_unabsorbed_refinement_correction():
    report = json.loads((HERE/"boundary_one_loop_action.json").read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    assert report["single_beta_fit_on_affine_shape_slice"]["unabsorbed_curvature_norm"] > .1
    rows = report["volume_preserving_controls"]
    assert rows[-1]["refinement_phi_increment"]-rows[0]["refinement_phi_increment"] > .04
    for name in ("full_nonlinear_internal_integral_computed", "boundary_derivative_error_rigorously_bounded",
                 "controlled_loop_expansion_proven_at_formal_threshold", "global_boundary_measure_normalized",
                 "microscopic_measure_derived", "dynamic_common_metric_selection_proven",
                 "continuum_einstein_limit_derived"):
        assert report["scope"][name] is False
