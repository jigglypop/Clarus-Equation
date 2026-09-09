"""Mass-density identity, bounded tails and independent oscillatory integrals."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_frequency_quadrature", HERE/"dimension_frequency_quadrature.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


@pytest.mark.parametrize("eta,beta", [(0., 1.), (.1, .5), (.1, 2.), (1., 1.)])
def test_prepared_measure_matches_exact_heat_integral(eta, beta):
    p = model.FrequencyProbe(eta=eta, beta=beta, coupling=0., cells=200, x_max=160.)
    x, w, info = p.positive_quadrature(40, 8)
    assert (w >= 0).all()
    assert abs(w.sum()-1) < 2e-12
    for t in (.1, 1., 10.):
        exact = math.exp(p.kernel(beta+t)["log_heat_trace"]-p.kernel(beta)["log_heat_trace"])
        assert np.sum(w*np.exp(-t*x)) == pytest.approx(exact, abs=2e-12)
    assert np.sum(w/(p.m**2+p.a*x)) == pytest.approx(p.bare_resolvent(0.)[0], abs=2e-12)
    assert info["omitted_probability_upper"] < 1e-20


@pytest.mark.parametrize("phase", [20., 80., 160.])
def test_oscillatory_moment_matches_independent_infinite_fourier_integral(phase):
    p = model.FrequencyProbe(eta=0., coupling=0., cells=200)
    x, w, _ = p.positive_quadrature(32, 8)
    # For eta=0, beta=m=a=1, nu(dx)=x exp(-x) dx on x>=0.
    # Weighted infinite-interval Fourier quadrature does not use our finite cells.
    density = lambda y: 2*y*(y*y-1)*math.exp(1-y*y)
    exact, error = quad(density, 1., math.inf, weight="cos", wvar=phase,
                        epsabs=2e-12, limlst=200)
    calculated = np.sum(w*np.cos(phase*np.sqrt(1+x)))
    assert error < 1e-10
    assert calculated == pytest.approx(exact, abs=2e-10)


def test_old_static_quadrature_can_alias_a_dynamic_moment():
    phase = 80.
    old = model.BoundaryProbe(eta=0., coupling=0.)
    x, w, _ = old.positive_quadrature(24, 40)
    density = lambda y: 2*y*(y*y-1)*math.exp(1-y*y)
    exact = quad(density, 1., math.inf, weight="cos", wvar=phase, epsabs=2e-12)[0]
    aliased = np.sum(w*np.cos(phase*np.sqrt(1+x)))
    assert abs(aliased-exact) > 1e-3
    # Static agreement alone is a weak validation of oscillatory histories.
    assert abs(np.sum(w/(1+x))-old.bare_resolvent(0.)[0]) < 1e-8


def test_numerical_tail_limit_does_not_renormalize_measure():
    p = model.FrequencyProbe(eta=0., coupling=0., cells=100, x_max=3.)
    x, w, info = p.positive_quadrature(32, 8)
    missing = 4*math.exp(-3.)
    assert 1-w.sum() == pytest.approx(missing, abs=1e-12)
    assert info["omitted_probability_upper"] == pytest.approx(missing, abs=1e-12)


def test_homogeneous_distance_refinement_resolves_previous_aliasing():
    coarse = model.run_branch(.1, cells=100)
    fine = model.run_branch(.1, cells=200)
    differences = np.array(coarse["distances"])/np.array(fine["distances"])-1
    assert np.max(np.abs(differences)) < 2e-12
    assert fine["block_diagonal_BAO_delta_chi2"] > 0
