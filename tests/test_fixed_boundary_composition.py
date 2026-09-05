"""Test analytic Regge derivatives and fixed-boundary Gaussian composition."""

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
    spec = importlib.util.spec_from_file_location("fixed_boundary_under_test", HERE/"fixed_boundary_composition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import regge_gluing_environment as previous
finally:
    sys.path[:] = original


def test_regular_analytic_derivatives_match_exact_action_gradient_and_hessian():
    lengths = np.full(10, math.sqrt(2))
    action, gradient, hessian, asymmetry = module.simplex_derivatives(lengths, np.full(10, math.pi))
    angle = math.pi-math.acos(.25)
    assert action == pytest.approx(5*math.sqrt(3)*angle, abs=1e-12)
    np.testing.assert_allclose(gradient, math.sqrt(3/2)*angle, atol=1e-12)
    np.testing.assert_allclose(hessian, previous.exact_coarse_hessian(), atol=1e-12)
    assert asymmetry < 1e-12


def test_irregular_cell_with_unequal_kappas_matches_independent_action_differences():
    r = module.gluing.reference()
    cells, points = module.gluing.geometry(2)
    lengths = r.cell_lengths(cells[3], points)
    kappas = np.linspace(.7, 2.8, 10)
    action, gradient, hessian, asymmetry = module.simplex_derivatives(lengths, kappas)
    assert action == pytest.approx(r.simplex_action(lengths, kappas), abs=1e-12)
    direction = np.random.default_rng(20260905).normal(size=10)
    direction /= np.linalg.norm(direction)
    step = 1e-4*min(lengths)
    def first_difference(h):
        return (r.simplex_action(lengths+h*direction, kappas)-r.simplex_action(lengths-h*direction, kappas))/(2*h)
    numerical = (4*first_difference(step/2)-first_difference(step))/3
    assert gradient @ direction == pytest.approx(numerical, rel=1e-7, abs=1e-7)
    plus = module.simplex_derivatives(lengths+step*direction, kappas)[1]
    minus = module.simplex_derivatives(lengths-step*direction, kappas)[1]
    np.testing.assert_allclose(hessian @ direction, (plus-minus)/(2*step), rtol=3e-6, atol=3e-6)
    assert asymmetry < 1e-10


@pytest.mark.parametrize("bad", [np.zeros(10), np.full(10, math.nan), np.ones(9),
                                 np.r_[10., np.ones(9)]])
def test_invalid_simplex_is_rejected(bad):
    with pytest.raises(ValueError):
        module.simplex_derivatives(bad, np.full(10, math.pi))


@pytest.mark.parametrize("depth, internal, gauge, physical", [(1, 5, 4, 1), (2, 30, 24, 6)])
def test_actual_raw_regge_internal_quotient_and_boundary_composition(depth, internal, gauge, physical):
    result = module.composition(depth)
    assert (result["internal_lengths"], result["gauge_dimension"], result["physical_dimension"]) == (internal, gauge, physical)
    assert min(result["physical_curvatures"]) > 0
    assert result["parent_marginal_curvature"] == pytest.approx(40*math.sqrt(5), rel=1e-11)
    for name in ("gauge_hessian_residual", "mixed_gauge_residual", "internal_gradient_residual",
                 "classical_action_residual", "boundary_hessian_residual", "sequential_hessian_residual"):
        assert result[name] < 2e-9


def test_two_level_children_are_not_assumed_regular():
    result = module.composition(2)
    assert len(result["child_conditional_curvatures"]) == 5
    assert min(result["child_conditional_curvatures"]) > 250.
    assert max(result["child_conditional_curvatures"]) < 270.
    assert min(result["child_conditional_curvatures"]) > 2*40*math.sqrt(5)


def test_quotient_jacobian_is_needed_for_normalization():
    result = module.composition(2)
    assert .97 < result["quotient_jacobian"] < .99
    assert result["omitted_jacobian_relative_error"] > .02
    assert abs(result["log_normalization_residual"]) < 1e-11
    assert result["covariance_transport_residual"] < 1e-12
    assert result["boundary_covariance_composition_residual"] < 1e-12


def test_root_gauge_invariant_marginal_survives_second_refinement():
    first, second = module.composition(1), module.composition(2)
    assert second["root_observable_gauge_residual"] < 1e-12
    expected = 1/(40*math.sqrt(5))
    assert first["root_marginal_variance"] == pytest.approx(expected, rel=1e-11)
    assert second["root_marginal_variance"] == pytest.approx(expected, rel=1e-11)
    assert second["root_variance_from_parent_schur"] == pytest.approx(expected, rel=1e-11)


def test_background_scaling_and_reference_coefficient_preserve_absolute_width():
    standard = module.composition(2)
    scaled = module.composition(2, scale=2.)
    beta_changed = module.composition(2, beta=.7)
    exact = 8*math.pi/(40*math.sqrt(5))
    for row in (standard, scaled, beta_changed):
        assert row["root_collective_variance_in_planck_units"] == pytest.approx(exact, rel=1e-11)
    assert beta_changed["root_marginal_variance"]*.7 == pytest.approx(standard["root_marginal_variance"], rel=1e-11)
    np.testing.assert_allclose(scaled["physical_curvatures"], standard["physical_curvatures"], rtol=1e-11)


def test_shape_change_preserves_composition_but_changes_gaussian_prefactor():
    standard = module.composition(2)
    shaped = module.composition(2, shear=.2)
    assert min(shaped["physical_curvatures"]) > 0
    assert shaped["boundary_hessian_residual"] < 2e-9
    assert abs(shaped["log_normalization_residual"]) < 1e-10
    assert abs(shaped["orthonormal_quotient_log_integral"]-standard["orthonormal_quotient_log_integral"]) > 1e-5


def test_artifact_sources_and_scope_are_current():
    report = json.loads((HERE/"fixed_boundary_composition.json").read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    scope = report["scope"]
    assert scope["raw_regge_hessian_used"]
    assert scope["global_length_gluing_imposed"]
    for name in ("physical_quotient_measure_derived", "computed_internal_width_is_gluing_resolution",
                 "canonical_quantum_vacuum_prepared", "common_metric_selected_dynamically",
                 "continuum_einstein_limit_derived"):
        assert scope[name] is False


def test_insufficient_boundary_prior_does_not_return_a_covariance():
    result = module.composition(1, beta=3.)
    assert not result["conditional_boundary_prior_normalizable"]
    assert result["boundary_covariance_composition_residual"] is None
    assert result["root_collective_variance_in_planck_units"] == pytest.approx(
        8*math.pi/(40*math.sqrt(5)), rel=1e-11
    )