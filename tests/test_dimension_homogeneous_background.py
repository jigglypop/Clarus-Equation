"""Conservation, frame calibration and independent limits for the background."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_homogeneous_background", HERE/"dimension_homogeneous_background.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_zero_coupling_matches_independent_distance_integrals():
    m = model.HomogeneousBackground(s=0., mass_over_h0=20., n_u=12, n_x=12).calibrate()
    z, kinds = [.38, .698, 1.48], ["DM_over_rs", "DH_over_rs", "DM_over_rs"]
    assert m.observed_distances(z, kinds) == pytest.approx(model.distance_shape(z, kinds, .315), abs=2e-10)
    assert m.diagnostics()["maximum_constraint_error"] < 2e-9


def test_initial_minimum_and_velocity_use_matter_exchange():
    m = model.HomogeneousBackground(s=.02, mass_over_h0=20., n_u=12, n_x=12)
    state = m.initial(.9, 2.1)
    rho, _, _, h, _, log_a_dot = m.quantities(m.n_initial, state, .9, 2.1)
    u, w = state[:m.n], state[m.n:2*m.n]
    assert np.max(np.abs(m.d*u+m.v*rho)) < 1e-12
    rho_dot = (-3*h+log_a_dot)*rho
    assert w == pytest.approx(-m.v*rho_dot/m.d, abs=2e-12)


def test_calibration_conservation_and_observed_redshift_inversion():
    m = model.HomogeneousBackground(s=.02, mass_over_h0=30., n_u=16, n_x=24).calibrate()
    diag = m.diagnostics()
    assert diag["H_J0_over_Href"] == pytest.approx(1., abs=2e-12)
    assert diag["omega_local_reconstructed"] == pytest.approx(.315, abs=2e-12)
    assert diag["maximum_constraint_error"] < 2e-9
    assert diag["maximum_scalar_energy_to_dust"] > 0
    # The observed redshift differs from exp(-N)-1 when A evolves.
    n = -.6
    point = m.at(n)
    assert abs(point["z_J"]-(np.exp(-n)-1)) > 1e-7
    assert m.observed_distances([point["z_J"]], ["DH_over_rs"])[0] == pytest.approx(1/point["H_J_over_Href"], rel=1e-10)
    # Jordan dust conservation provides an independent frame-factor check.
    rho_j_comoving = []
    for n in [-1., -.5, 0.]:
        p = m.at(n)
        rho_j_comoving.append(p["rho_E"]/p["A"]**4*(p["A"]*np.exp(n))**3)
    assert rho_j_comoving == pytest.approx([m.rho_bar]*3, rel=1e-12)


def test_kinematical_distance_derivative_in_jordan_redshift():
    m = model.HomogeneousBackground(s=.02, mass_over_h0=20., n_u=12, n_x=12).calibrate()
    z, dz = .6, 1e-4
    dm = m.observed_distances([z-dz, z+dz], ["DM_over_rs"]*2)
    dh = m.observed_distances([z], ["DH_over_rs"])[0]
    assert (dm[1]-dm[0])/(2*dz) == pytest.approx(dh, rel=2e-7)
    with pytest.raises(ValueError, match="outside"):
        m.observed_distances([10.], ["DM_over_rs"])


def test_ambient_density_hessian_matches_independent_matrix_inverse():
    # eta=0 retains a continuous Gamma mass measure; it is not a single mass.
    m = model.HomogeneousBackground(s=.02, mass_over_h0=20.,
                                    probe=model.BoundaryProbe(eta=0., coupling=0.), n_u=12, n_x=64)
    rho, z = 40000., .4
    hessian = np.diag(m.d+z)+m.epsilon*rho*np.outer(m.v, m.v)
    independent = m.v@np.linalg.solve(hessian, m.v)
    assert m.ambient_resolvent(z, rho) == pytest.approx(independent, rel=1e-10)
    assert 0 < independent < m.ambient_resolvent(z, 0.)
