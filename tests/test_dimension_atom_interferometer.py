"""Finite source, light-pulse phase identity and conservative linear bounds."""
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
    spec = importlib.util.spec_from_file_location("dimension_atom_interferometer", HERE/"dimension_atom_interferometer.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_drilled_sphere_mass_and_center_of_mass():
    r, b, rho = .0095, .0015, 2700.
    points, mass = model.drilled_sphere(r,b,rho,order=12,azimuths=16)
    assert mass.sum() == pytest.approx(4*math.pi*rho/3*(r*r-b*b)**1.5, rel=1e-13)
    assert np.linalg.norm(np.sum(points*mass[:,None],axis=0)) < 1e-19


def test_solid_sphere_matches_newton_shell_theorem():
    points, mass = model.drilled_sphere(bore=0.,order=28,azimuths=48)
    observer = np.array([.017,.008,.021])
    calculated = model.source_acceleration(observer,points,mass)
    exact = -model.G_N*mass.sum()*observer/np.linalg.norm(observer)**3
    assert calculated == pytest.approx(exact, rel=2e-10)


def test_constant_acceleration_phase_has_T_squared_and_sign():
    T,k = .0155,1.4e7
    force = lambda x: np.array([0.,0.,-2e-6])
    trajectory = lambda t: np.array([0.,0.,.02-4.905*t*t])
    result = model.pulse_phase(force,trajectory,T=T,k_eff=k)
    assert result == pytest.approx(k*2e-6*T*T,rel=1e-13)
    reverse = model.pulse_phase(force,trajectory,T=T,k_eff=k,direction=(0.,0.,1.))
    assert reverse == pytest.approx(-result,rel=1e-13)


def test_finite_arm_force_average_equals_independent_cubic_potential_phase():
    T,k,mass = .02,2e7,2e-25
    center = lambda t: np.array([0.,0.,.02+.1*t-4.9*t*t])
    # Phi(z)=-c*z^3/3; a_z=c*z^2. Nonzero recoil probes curvature along arms.
    c = .3
    force = lambda x: np.array([0.,0.,c*x[2]**2])
    actual = model.pulse_phase(force,center,T=T,k_eff=k,atom_mass=mass,direction=(0.,0.,1.))
    vr = model.HBAR*k/mass
    def integrand(t):
        dz = vr*min(t,2*T-t)
        z = center(t)[2]
        delta_potential = -c*((z+dz/2)**3-(z-dz/2)**3)/3
        return -mass/model.HBAR*delta_potential
    expected = quad(integrand,0,T,epsabs=1e-12)[0]+quad(integrand,T,2*T,epsabs=1e-12)[0]
    assert actual == pytest.approx(expected,rel=1e-11)


def test_calibrated_point_force_is_below_general_bound():
    # Single positive mass spectral measure, E[d]=1, is an independent
    # special case of the proof (not the default continuous model).
    class SingleMass:
        m, a, beta = 1.,0.,1.
        def kernel(self,t):
            return {"dimension": 0.}
    s,inverse,source,rcal,rmin = .02,10.,.01,.1,.005
    bound = model.calibrated_linear_signal_bound(source_mass=source,inverse_length=inverse,
                s=s,minimum_distance=rmin,calibration_distance=rcal,probe=SingleMass())
    f = lambda r: (1+inverse*r)*math.exp(-inverse*r)
    for r in [.005,.01,.1,1.]:
        actual = model.G_N*s*source*(f(r)-f(rcal))/((1+s*f(rcal))*r*r)
        assert abs(actual) <= bound["one_source_acceleration_upper_m_s2"]
    zero = model.calibrated_linear_signal_bound(source_mass=source,inverse_length=0.,s=s,minimum_distance=rmin)
    assert zero["near_minus_far_phase_upper_rad"] == 0.


def test_representative_geometry_is_explicitly_not_the_data_likelihood():
    result = model.report()
    assert result["representative"]["arm_to_material_distance_lower_bound_m"] > .003
    assert abs(result["source_refinement_phase_difference_rad"]) < 1e-14
    assert result["linear_bound_over_reported_sigma"] < 1e-50
    assert result["actual_experiment_rmse"] is None and result["scientific_success"] is False
