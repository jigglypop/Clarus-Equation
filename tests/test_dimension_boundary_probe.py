"""Independent oscillator spectra audit a continuous collective-mode candidate."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_boundary_probe", HERE/"dimension_boundary_probe.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = original_path


@pytest.mark.parametrize("z", [0., 1., 100., 1e6])
def test_heat_resolvent_matches_independent_positive_measure(z):
    p = model.BoundaryProbe()
    x, w, info = p.positive_quadrature(n_u=24, n_x=48)
    assert abs(info["weight_sum"]-1) < 1e-10
    assert info["u_tail_probability_bound"] < 1e-20
    independent = np.sum(w/(p.m**2+p.a*x+z))
    assert p.bare_resolvent(z)[0] == pytest.approx(independent, rel=2e-7)
    assert p.response(z)["coupled"] <= 1/(p.m**2+z)


@pytest.mark.parametrize("beta", [.5, 1., 2.])
def test_energy_covariance_and_force_match_finite_oscillators(beta):
    p = model.BoundaryProbe(beta=beta)
    exact = p.vacuum()
    finite = p.finite_oscillators(n_u=20, n_x=48)
    assert abs(finite["weight_sum"]-1) < 1e-9
    for quantity in ("relative_energy", "variance_Q", "d_energy_d_a"):
        assert finite[quantity] == pytest.approx(exact[quantity], rel=1e-5, abs=2e-8)
    assert 0 < exact["relative_energy"] <= exact["energy_upper_bound"]
    assert exact["d_energy_d_a"] < 0
    assert finite["minimum_squared_frequency"] >= p.m**2
    assert finite["static_response"] == pytest.approx(p.response()["coupled"], rel=1e-5)


def test_vacuum_derivatives_are_actual_parameter_derivatives():
    h = 2e-4
    base = model.BoundaryProbe().vacuum()
    e_plus = model.BoundaryProbe(coupling=.5+h).vacuum()["relative_energy"]
    e_minus = model.BoundaryProbe(coupling=.5-h).vacuum()["relative_energy"]
    assert (e_plus-e_minus)/(2*h) == pytest.approx(base["variance_Q"]/2, rel=1e-6)
    e_plus = model.BoundaryProbe(a=1+h).vacuum()["relative_energy"]
    e_minus = model.BoundaryProbe(a=1-h).vacuum()["relative_energy"]
    assert (e_plus-e_minus)/(2*h) == pytest.approx(base["d_energy_d_a"], rel=1e-6)


def test_zero_coupling_and_high_frequency_sum_rule():
    p = model.BoundaryProbe(coupling=0)
    result = p.vacuum()
    assert result["relative_energy"] == 0
    assert result["d_energy_d_a"] == 0
    assert p.response()["bare"] == p.response()["coupled"]
    p = model.BoundaryProbe()
    assert 1e8*p.response(1e8)["coupled"] == pytest.approx(1, rel=1e-6)


def test_negative_coupling_is_rejected_not_silently_stabilized():
    with pytest.raises(ValueError):
        model.BoundaryProbe(coupling=-1)


@pytest.mark.parametrize("kwargs", [{"beta": 0}, {"m": 0}, {"a": -1}, {"eta": 2}, {"b": float('nan')}])
def test_invalid_preparation_and_parameters(kwargs):
    with pytest.raises(ValueError):
        model.BoundaryProbe(**kwargs)


def test_reciprocal_dynamics_conserves_energy_and_quantum_commutator():
    result = model.coupled_collective_evolution(n_u=6, n_x=6)
    assert result["max_absolute_energy_drift"] < 2e-7
    assert result["max_canonical_commutator_error"] < 2e-7
    assert max(result["variance_Q"])-min(result["variance_Q"]) > 1e-4
    control = model.coupled_collective_evolution(n_u=6, n_x=6, reciprocal=False)
    assert control["max_absolute_energy_drift"] > 1e-3


def test_zero_coupling_recovers_independent_collective_oscillator():
    result = model.coupled_collective_evolution(n_u=4, n_x=4, kappa=0)
    expected = .6*np.cos(.7*np.array(result["times"]))
    assert result["q"] == pytest.approx(expected, abs=2e-8)
    assert max(result["variance_Q"])-min(result["variance_Q"]) < 1e-8


@pytest.mark.parametrize("coupling", [.01, .5, 2.])
def test_four_dimensional_subtracted_loop_matches_single_mass_formula(coupling):
    # Independent integrated Coleman-Weinberg expression in the declared scheme.
    m = 1.3
    z = m*m
    result = model.local_potential_integral(lambda p2: 1/(z+p2), coupling, m)
    expected = ((z+coupling)**2*np.log1p(coupling/z)-z*coupling-1.5*coupling**2)/(64*np.pi**2)
    assert result["renormalized_relative_potential"] == pytest.approx(expected, rel=2e-7, abs=1e-14)


def test_continuum_local_loop_is_finite_and_derivative_matches():
    p = model.BoundaryProbe()
    result = p.local_potential()
    assert 0 < result["renormalized_relative_potential"] < result["potential_upper_bound"]
    h = 2e-4
    plus = model.BoundaryProbe(coupling=.5+h).local_potential()["renormalized_relative_potential"]
    minus = model.BoundaryProbe(coupling=.5-h).local_potential()["renormalized_relative_potential"]
    assert (plus-minus)/(2*h) == pytest.approx(result["d_potential_d_coupling"], rel=2e-6)
    assert model.BoundaryProbe(coupling=0).local_potential()["renormalized_relative_potential"] == 0
