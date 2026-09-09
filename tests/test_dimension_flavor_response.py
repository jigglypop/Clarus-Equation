"""Derivative checks use independent finite matrix diagonalizations."""
import numpy as np
import pytest
from verify.dimension_flavor_response import mass_response, mixing_response, differential_free_fall, majorana_vacuum_probabilities


def matrices():
    rng = np.random.default_rng(92)
    m = np.diag([1., 2., 4.])+0.2*(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    c = 0.1*(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    return m, c


@pytest.mark.parametrize("majorana", [False, True])
def test_mass_and_projector_derivatives_against_finite_difference(majorana):
    m, c = matrices()
    if majorana:
        m, c = (m+m.T)/2, (c+c.T)/2
    alpha, eps = .03, 1e-5
    result = mass_response(m, c, alpha=alpha, majorana=majorana)
    samples = []
    for q in [-eps, eps]:
        shifted = np.exp(alpha*q)*(m+q*c)
        h = shifted.conj().T@shifted if majorana else shifted@shifted.conj().T
        lam, u = np.linalg.eigh(h)
        samples.append((np.log(np.sqrt(lam)), u))
    assert (samples[1][0]-samples[0][0])/(2*eps) == pytest.approx(result["log_mass_derivatives"], abs=1e-9)
    u, du = result["basis"], result["basis"]@result["basis_generator"]
    for j in range(3):
        derivative = np.outer(du[:,j], u[:,j].conj())+np.outer(u[:,j], du[:,j].conj())
        finite = (np.outer(samples[1][1][:,j], samples[1][1][:,j].conj())
                  -np.outer(samples[0][1][:,j], samples[0][1][:,j].conj()))/(2*eps)
        assert finite == pytest.approx(derivative, abs=1e-9)


def test_complex_pmns_modulus_derivative_and_probability_conservation():
    m, c = matrices()
    n, d = (m+m.T)/2, (c+c.T)/2
    result = mixing_response(mass_response(m,c), mass_response(n,d,majorana=True))
    eps = 1e-5
    values = []
    for q in [-eps, eps]:
        _, ue = np.linalg.eigh((m+q*c)@(m+q*c).conj().T)
        _, un = np.linalg.eigh((n+q*d).conj().T@(n+q*d))
        values.append(abs(ue.conj().T@un)**2)
    assert (values[1]-values[0])/(2*eps) == pytest.approx(result["modulus_squared_derivative"], abs=1e-9)
    assert result["modulus_squared_derivative"].sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-14)
    assert result["modulus_squared_derivative"].sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-14)


def test_universal_alignment_and_off_diagonal_response_are_distinct():
    m = np.diag([1., 2., 4.])
    aligned = mass_response(m, .2*m, alpha=.03)
    assert aligned["log_mass_derivatives"] == pytest.approx([.23]*3)
    assert aligned["log_mass_ratio_derivatives"] == pytest.approx(np.zeros((3,3)))
    assert aligned["basis_generator"] == pytest.approx(np.zeros((3,3)))
    c = np.array([[0,.2,0],[.2,0,0],[0,0,0]])
    response = mass_response(m,c)
    assert response["log_mass_derivatives"] == pytest.approx(np.zeros(3))
    assert np.linalg.norm(response["basis_generator"]) > .1


def test_rejects_degenerate_and_wrong_majorana_domain():
    with pytest.raises(ValueError, match="nondegenerate"):
        mass_response(np.eye(3), np.zeros((3,3)))
    m,c=matrices()
    with pytest.raises(ValueError, match="symmetric"):
        mass_response(m,c,majorana=True)


def test_differential_force_matches_direct_force_and_allows_signed_charges():
    source, a, b, shape = .1, -.2, .3, .8
    fa, fb = 1+2*source*a*shape, 1+2*source*b*shape
    assert differential_free_fall(source,a,b,shape) == pytest.approx(2*(fa-fb)/(fa+fb))
    assert differential_free_fall(source,a,a,shape) == 0


def test_identical_mass_spectra_leave_relative_flavor_orientation_undetermined():
    # Same spectra at every theta, different mixing and mixed spectral invariant.
    up = np.diag([1., 2., 3.])
    down = np.diag(np.sqrt([2., 5., 10.]))
    for theta in (.0, .2, .7):
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        rotated = rotation@down
        a = mass_response(up, np.zeros((3,3)))
        b = mass_response(rotated, np.zeros((3,3)))
        mixing = mixing_response(a,b)["modulus_squared"]
        assert b["masses"] == pytest.approx(np.sqrt([2.,5.,10.]))
        assert mixing[0,1] == pytest.approx(s*s, abs=1e-14)
        for t in (.01, 1., 10.):
            assert np.exp(-t*b["masses"]**2).sum() == pytest.approx(
                np.exp(-t*np.array([2.,5.,10.])).sum(), abs=1e-14)
        mixed = np.trace((up@up.T)@(rotated@rotated.T))
        assert mixed == pytest.approx(112-9*s*s)


def test_trace_orientation_interaction_rotation_energy_and_stable_pairing():
    from itertools import permutations
    a, b = np.array([1.,4.,9.]), np.array([2.,5.,10.])
    energies = {p: a@b[list(p)] for p in permutations(range(3))}
    assert min(energies, key=energies.get) == (2,1,0)
    assert max(energies, key=energies.get) == (0,1,2)
    for p in energies:
        bp = b[list(p)]
        for i,j in [(0,1),(0,2),(1,2)]:
            theta = .137
            r = np.eye(3)
            r[i,i] = r[j,j] = np.cos(theta)
            r[i,j], r[j,i] = -np.sin(theta), np.sin(theta)
            actual = np.trace(np.diag(a)@r@np.diag(bp)@r.T)-energies[p]
            expected = (a[i]-a[j])*(bp[j]-bp[i])*np.sin(theta)**2
            assert actual == pytest.approx(expected, abs=4e-14)


def test_degenerate_majorana_pair_has_no_vacuum_oscillation_despite_maximal_takagi_mixing():
    m = np.array([[0,2.,0],[2.,0,0],[0,0,5.]])
    u2 = np.array([[1j,1],[-1j,1]])/np.sqrt(2)
    assert u2.T@m[:2,:2]@u2 == pytest.approx(2*np.eye(2), abs=1e-14)
    for phase in (0, .3, 10.):
        assert majorana_vacuum_probabilities(m,phase) == pytest.approx(np.eye(3), abs=1e-14)


def test_split_pair_probability_matches_closed_two_level_formula():
    off, eps, a, b = 2., .1, .7, -.2
    m = np.array([[eps*a,off,0],[off,eps*b,0],[0,0,5.]])
    amplitude = 4*off**2/(4*off**2+eps**2*(a-b)**2)
    delta_m2 = abs(2*eps*(a+b)*np.sqrt(off**2+eps**2*(a-b)**2/4))
    for phase in (.01, .5, 3.):
        p = majorana_vacuum_probabilities(m,phase)
        assert p[1,0] == pytest.approx(amplitude*np.sin(delta_m2*phase/2)**2, abs=1e-14)
        assert p.sum(axis=0) == pytest.approx(np.ones(3), abs=1e-14)
    from scipy.linalg import expm
    direct = abs(expm(-.4j*(m.conj().T@m)))**2
    assert majorana_vacuum_probabilities(m,.4) == pytest.approx(direct, abs=1e-14)
