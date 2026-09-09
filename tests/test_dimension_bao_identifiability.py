"""Fair model profiling and nuisance projection, with independent fit checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0,str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_bao_identifiability",HERE/"dimension_bao_identifiability.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_fair_profiles_match_independent_two_parameter_fit():
    data = model.combined_data()
    chol = np.linalg.cholesky(data["cov"])
    def residual(p):
        return np.linalg.solve(chol,data["y"]-p[0]*model.distance_shape(data["z"],data["kind"],p[1]))
    independent = least_squares(residual,[30.,.3],bounds=([1.,.05],[100.,.6]),
                                ftol=1e-13,xtol=1e-13,gtol=1e-13)
    values = [model.profile_leading_model(data,s) for s in (0.,1.2e-5,.02)]
    for value in values:
        assert value["chi2"] == pytest.approx(independent.fun@independent.fun,abs=1e-10)
        assert value["omega_effective_fitted"] == pytest.approx(independent.x[1],abs=2e-7)
        assert value["fitted_nuisance_count"] == 2
        assert value["s_identifiable_from_leading_BAO"] is False


def test_distance_nuisance_derivative_matches_finite_difference():
    d = model.combined_data()
    omega,h = .315,1e-5
    difference = (model.distance_shape(d["z"],d["kind"],omega+h)
                  -model.distance_shape(d["z"],d["kind"],omega-h))/(2*h)
    assert model.distance_derivative(d["z"],d["kind"],omega) == pytest.approx(difference,rel=2e-8,abs=1e-10)


def test_exact_nuisance_signal_is_removed_and_units_order_preserved():
    c = np.array([[2.,.3,0.],[.3,1.,.1],[0.,.1,3.]])
    j = np.array([[1.,.2],[2.,1.],[.4,3.]])
    delta = j@np.array([.05,-.03])
    result = model.nuisance_orthogonal_signal(delta,c,j)
    assert result["nuisance_orthogonal_whitened_norm"] < 1e-14
    delta = delta+np.array([.01,.02,-.01])
    reference = model.nuisance_orthogonal_signal(delta,c,j)
    scale = np.diag([1e-4,1e5,3.])
    transformed = model.nuisance_orthogonal_signal(scale@delta,scale@c@scale,scale@j)
    assert transformed["nuisance_orthogonal_whitened_norm"] == pytest.approx(reference["nuisance_orthogonal_whitened_norm"],rel=1e-10)
    p = [2,0,1]
    permuted = model.nuisance_orthogonal_signal(delta[p],c[np.ix_(p,p)],j[p])
    assert permuted["nuisance_orthogonal_whitened_norm"] == pytest.approx(reference["nuisance_orthogonal_whitened_norm"],rel=1e-10)


def test_report_does_not_credit_shared_nuisance_gain_to_new_physics():
    r = model.report()
    baseline,candidate,_ = r["equally_profiled_leading_models"]
    assert baseline["whitened_rmse"] < r["fixed_reference"]["whitened_rmse"]
    assert candidate["whitened_rmse"] == pytest.approx(baseline["whitened_rmse"],abs=1e-10)
    assert candidate["BAO_plus_Cassini_independence_assumed_chi2"] > baseline["BAO_plus_Cassini_independence_assumed_chi2"]
    shape = r["shape_eta_one_minus_zero_local_diagnostic"]
    assert 0 < shape["nuisance_orthogonal_whitened_norm"] < shape["raw_whitened_norm"] < 1e-6
    assert r["all_domain_rmse"] is None and not r["scientific_success"]
