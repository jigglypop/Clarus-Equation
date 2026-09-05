"""Check the off-glue one-loop flow against exact entries and a preserving control."""

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
    spec = importlib.util.spec_from_file_location("one_loop_gluing_flow_under_test", HERE/"one_loop_gluing_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path[:] = original


def test_actual_copy_geometry_and_exclusive_edge_support():
    data = module.geometry()
    a, c = data["assembly"], data["constraint"]
    assert a.shape == (50, 15)
    assert np.linalg.matrix_rank(c) == 35
    np.testing.assert_array_equal(c @ a, 0.)
    _, left, right, exclusive, global_f = module.witness(data)
    row = module.product_action(data["copies"], data)
    h = row["hessian"]
    assert (h @ a)[left, global_f] == 0.
    assert (h @ a)[right, global_f] == pytest.approx(h[right, exclusive], abs=1e-13)
    assert h[right, exclusive] < -9.


def test_regular_exact_mixed_entries_and_kappa_independence():
    data = module.geometry()
    row = module.product_action(data["copies"], data)
    _, _, right, exclusive, _ = module.witness(data)
    assert row["classical_hessian"][right, exclusive] == pytest.approx(-2*math.sqrt(2), abs=1e-12)
    assert row["loop_hessian"][right, exclusive] == pytest.approx(-1449*math.sqrt(10)/722, abs=1e-12)
    lengths = data["copies"][10:20]
    kappas = np.linspace(.1, 4., 10)
    changed = module.loop.fixed.simplex_derivatives(lengths, kappas)[2]
    assert changed[3, 4] == pytest.approx(-2*math.sqrt(2), abs=1e-12)


def test_regular_force_can_match_while_neighboring_glued_configurations_fail():
    result = module.flow_case()
    assert result["glued_force_residual"] < 1e-11
    assert result["linearized_gluing_defect"] > 200.
    witness = result["witness"]
    assert witness["force_difference_derivative"] > 9.
    assert witness["independent_gradient_difference"] == pytest.approx(
        witness["force_difference_derivative"], abs=1e-7)
    # An irregular flat glued geometry already has a nonzero force mismatch.
    shaped = module.flow_case(shear=.2)
    assert shaped["glued_position_residual"] == 0.
    assert shaped["glued_force_residual"] > 1.


@pytest.mark.parametrize("beta", [.01, 1., 100.])
def test_no_positive_beta_cancellation_with_positive_diagonal_masses(beta):
    w = np.exp(np.linspace(-2, 2, 50))
    result = module.flow_case(beta, w)
    witness = result["witness"]
    exact = (2*math.sqrt(2)*beta+1449*math.sqrt(10)/722)/w[witness["right_copy"]]
    assert witness["force_difference_derivative"] == pytest.approx(exact, rel=1e-12)
    assert witness["independent_gradient_difference"] == pytest.approx(exact, rel=1e-7)


def test_penalty_changes_normal_curvature_but_not_the_gluing_defect():
    data = module.geometry()
    a, c = data["assembly"], data["constraint"]
    np.testing.assert_allclose(c.T @ c @ a, 0., atol=1e-13)
    baseline = module.flow_case()
    strong = module.flow_case(penalty=1000.)
    assert strong["witness"]["force_difference_derivative"] == pytest.approx(
        baseline["witness"]["force_difference_derivative"], abs=1e-10)
    assert strong["linearized_gluing_defect"] == pytest.approx(baseline["linearized_gluing_defect"], abs=1e-9)


@pytest.mark.parametrize("shear", [0., .2])
def test_boundary_completion_retains_the_entire_previous_flat_action(shear):
    result = module.full_flat_action_check(shear, beta=.7)
    assert result["full_action_value_residual"] < 1e-10
    assert result["root_and_global_jacobian_completion"] != 0.


def test_arbitrary_boundary_only_completion_cannot_touch_internal_witness():
    data = module.geometry()
    b = data["boundary_readout"]
    _, left, right, _, global_f = module.witness(data)
    random = np.random.default_rng(197).normal(size=(10, 10))
    completion_hessian = b.T @ (random+random.T) @ b
    np.testing.assert_array_equal(completion_hessian[left], 0.)
    np.testing.assert_array_equal(completion_hessian[right], 0.)
    assert (completion_hessian @ data["assembly"])[right, global_f] == 0.


@pytest.mark.parametrize("penalty", [0., 1000.])
def test_frozen_variation_leaves_glue_with_predicted_quadratic_time_term(penalty):
    rows = module.short_time_variation(penalty=penalty)
    for previous, current in zip(rows, rows[1:]):
        assert current["residual"] < .26*previous["residual"]
    assert rows[-1]["mismatch_over_time_squared"] == pytest.approx(
        -(2*math.sqrt(2)+1449*math.sqrt(10)/722)/2, abs=2e-5)


@pytest.mark.parametrize("penalty", [0., 3.])
def test_supplied_projection_control_preserves_same_glued_action_and_flow(penalty):
    data = module.geometry(.1)
    a, c = data["assembly"], data["constraint"]
    w = np.linspace(.7, 1.3, 50)
    product = module.product_action(data["copies"], data)
    control = module.projected_control(data["copies"], data, w, penalty)
    assert control["value"] == pytest.approx(product["value"], abs=1e-12)
    np.testing.assert_allclose(a.T @ control["gradient"], a.T @ product["gradient"], atol=1e-11)
    np.testing.assert_allclose(a.T @ control["hessian"] @ a,
                               a.T @ product["hessian"] @ a, atol=1e-10)
    assert np.linalg.norm((c/w) @ control["gradient"]) < 1e-11
    assert np.linalg.norm((c/w) @ control["hessian"] @ a) < 1e-10
    # Normal data really change, although every glued value is preserved.
    assert np.linalg.norm(control["gradient"]-product["gradient"]) > 1.


def test_projected_control_gradient_and_hessian_match_its_own_off_glue_action():
    data = module.geometry()
    w = np.linspace(.7, 1.3, 50)
    rng = np.random.default_rng(991)
    x = data["copies"]+.001*rng.normal(size=50)
    u = rng.normal(size=50)
    u /= np.linalg.norm(u)
    row = module.projected_control(x, data, w, penalty=2.)
    step = 1e-5
    plus = module.projected_control(x+step*u, data, w, penalty=2.)
    minus = module.projected_control(x-step*u, data, w, penalty=2.)
    assert (plus["value"]-minus["value"])/(2*step) == pytest.approx(row["gradient"] @ u, abs=1e-7)
    np.testing.assert_allclose((plus["gradient"]-minus["gradient"])/(2*step),
                               row["hessian"] @ u, atol=1e-6, rtol=1e-7)


@pytest.mark.parametrize("kwargs", [{"beta": 0.}, {"beta": math.nan}, {"penalty": -1.},
                                   {"copy_masses": np.zeros(50)}, {"copy_masses": np.ones(49)},
                                   {"shear": math.nan}])
def test_invalid_flow_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        module.flow_case(**kwargs)


def test_artifact_dependencies_and_scope_are_current():
    result = json.loads((HERE/"one_loop_gluing_flow.json").read_text(encoding="utf-8"))
    for name, digest in result["source_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    scope = result["scope"]
    assert scope["complete_prior_flat_action_value_retained"]
    assert scope["local_product_off_glue_continuation_supplied"]
    for name in ("all_glued_configurations_preserved_by_tested_flow",
                 "smaller_common_metric_invariant_sectors_excluded",
                 "all_possible_cross_cell_interactions_excluded",
                 "full_nonlinear_trajectory_computed", "quantum_noise_removed",
                 "microscopic_ce_action_derived", "continuum_einstein_limit_derived"):
        assert scope[name] is False
