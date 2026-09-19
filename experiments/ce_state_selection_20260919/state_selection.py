"""CE collective-noise state-preparation theorem and no-fit verification.

Conditional microscopic model, not a derivation of PMNS, cosmic abundances,
or the empirical alpha_s matching. All units below are diagnostic.
Run: OPENBLAS_NUM_THREADS=1 python state_selection.py --output results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


def generators() -> list[np.ndarray]:
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a = np.zeros((3, 3), complex)
        a[i, j] = a[j, i] = 0.5
        out.append(a)
        b = np.zeros((3, 3), complex)
        b[i, j], b[j, i] = -0.5j, 0.5j
        out.append(b)
    out.append(np.diag([1., -1., 0.]).astype(complex) / 2)
    out.append(np.diag([1., 1., -2.]).astype(complex) / (2*np.sqrt(3)))
    return out


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a@b-b@a


def cs(a: np.ndarray) -> np.ndarray:
    n = len(a)
    return np.kron(np.eye(n), a)-np.kron(a.T, np.eye(n))


def vec(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, order='F')


def mat(a: np.ndarray) -> np.ndarray:
    return a.reshape((9, 9), order='F')


class Model:
    def __init__(self, gamma: float = 1.):
        if gamma <= 0:
            raise ValueError('gamma must be positive')
        self.gamma = gamma
        self.alpha = 0.11789
        self.p = 4*self.alpha**(4/3)
        self.delta = self.p*(1-self.p)
        self.D = 3+self.delta
        self.q = brentq(lambda q: np.exp(-self.D*(1-q))-q,
                        0., 1/self.D, xtol=5e-16)
        self.T = generators()
        self.I = np.eye(9, dtype=complex)
        self.S = np.zeros((9, 9), complex)
        for i in range(3):
            for j in range(3):
                self.S[3*i+j, 3*j+i] = 1
        self.Pp = (self.I+self.S)/2
        self.Pm = (self.I-self.S)/2
        self.target = (1-self.delta)*self.Pp/6+self.delta*self.Pm/3
        self.masses = np.sqrt([1-(1+np.sqrt(3))*.15,
                               1+(np.sqrt(3)-1)*.15, 1.3])
        h = np.diag(self.masses)
        self.H0 = np.kron(h, np.eye(3))+np.kron(np.eye(3), h)
        self.L = [np.kron(t, np.eye(3))+np.kron(np.eye(3), t)
                  for t in self.T]
        self.diss = sum(-gamma/2*cs(a)@cs(a) for a in self.L)
        self.liouv = self.diss-1j*cs(self.H0)
        self.rho = np.diag([self.p, 1-self.p, 0.]).astype(complex)
        self.initial = np.kron(self.rho, self.rho)
        self.A = np.diag([1., -1., 0.]).astype(complex)
        self.contrasts = np.kron(self.A, self.A)
        self.cov_target = self.delta*np.eye(8)/8

    def evolve(self, x: np.ndarray, t: float) -> np.ndarray:
        if t < 0:
            raise ValueError('t must be nonnegative')
        return mat(expm(t*self.liouv)@vec(x))

    def twirl(self, x: np.ndarray) -> np.ndarray:
        return np.trace(self.Pp@x)*self.Pp/6+np.trace(self.Pm@x)*self.Pm/3

    def covariance(self, t: float) -> np.ndarray:
        b = self.evolve(self.contrasts, t)
        return np.array([[self.delta*np.trace(b@np.kron(a, c)).real
                          for c in self.T] for a in self.T])

    def energy_fraction(self, rho: np.ndarray) -> float:
        return float((np.trace(rho@self.H0@self.Pm)/
                      np.trace(rho@self.H0)).real)

    def step(self, dt: float) -> np.ndarray:
        """Eight fresh |0> qubit collisions, then collective free evolution.
        Each collision is exactly CP/TP and conserves SWAP. The limit is
        self.liouv; finite step sizes are numerical diagnostics, not fits.
        """
        channel = np.eye(81, dtype=complex)
        for a in self.L:
            ev, u = np.linalg.eigh(a)
            c = (u*np.cos(np.sqrt(self.gamma*dt)*ev))@u.conj().T
            s = (u*np.sin(np.sqrt(self.gamma*dt)*ev))@u.conj().T
            local = np.kron(c.conj(), c)+np.kron(s.conj(), s)
            channel = local@channel
        v = expm(-1j*dt*self.H0)
        return np.kron(v.conj(), v)@channel

    def count_pgf(self, z: float, t: float) -> float:
        a = self.gamma*t*(z-1)
        return float((1-self.delta)*np.exp(10*a/3)
                     +self.delta*np.exp(4*a/3))

    def count_direct(self, z: float, t: float) -> float:
        recycle = sum(self.gamma*np.kron(a.conj(), a) for a in self.L)
        out = expm(t*(self.liouv+(z-1)*recycle))@vec(self.initial)
        return float(np.trace(mat(out)).real)


def mub_average(p: float) -> np.ndarray:
    omega = np.exp(2j*np.pi/3)
    bases = [np.eye(3, dtype=complex)]
    bases += [np.array([[omega**(b*k*k+j*k)/np.sqrt(3)
                         for j in range(3)] for k in range(3)])
              for b in range(3)]
    out = np.zeros((9, 9), complex)
    for basis in bases:
        ps = [np.outer(basis[:,j], basis[:,j].conj()) for j in range(3)]
        for j in range(3):
            for k in range(3):
                if j != k:
                    r = p*ps[j]+(1-p)*ps[k]
                    out += np.kron(r,r)/24
    return out


class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = Model()

    def test_01_normalization_and_swap(self):
        m = self.m
        np.testing.assert_allclose(m.S@m.S, m.I, atol=2e-15)
        for a in m.L:
            np.testing.assert_allclose(comm(a,m.S), 0, atol=2e-15)
        np.testing.assert_allclose([[np.trace(a@b) for b in m.T] for a in m.T],
                                   np.eye(8)/2, atol=2e-15)

    def test_02_casimir(self):
        m = self.m
        np.testing.assert_allclose(sum(a@a for a in m.L),
                                   10*m.Pp/3+4*m.Pm/3, atol=3e-15)

    def test_03_dissipator_spectrum(self):
        ev = np.linalg.eigvalsh(self.m.diss)
        expected = np.array([-4.]*27+[-3.]*20+[-1.5]*32+[0.]*2)
        np.testing.assert_allclose(ev, expected, atol=2e-14)

    def test_04_stationary_space(self):
        m = self.m
        self.assertLess(np.linalg.norm(m.liouv@vec(m.target)), 2e-15)
        sv = np.linalg.svd(m.liouv, compute_uv=False)
        self.assertEqual(np.sum(sv < 1e-12), 2)

    def test_05_exact_mub_average(self):
        np.testing.assert_allclose(mub_average(self.m.p), self.m.target,
                                   atol=3e-15)

    def test_06_density_and_conserved_weight(self):
        m = self.m
        for t in (0., .2, 1., 4., 10.):
            r=m.evolve(m.initial,t)
            self.assertLess(abs(np.trace(r)-1), 6e-15)
            self.assertGreater(np.linalg.eigvalsh(r).min(), -2e-15)
            self.assertLess(abs(np.trace(m.Pm@r)-m.delta), 3e-15)

    def test_07_spectral_error_bound(self):
        m = self.m
        for t in (.2,1.,4.,8.):
            err=np.linalg.norm(m.evolve(m.initial,t)-m.target)
            upper=np.exp(-1.5*t)*np.linalg.norm(m.initial-m.target)
            self.assertLessEqual(err,upper+5e-15)

    def test_08_direction_covariance(self):
        m = self.m
        for t in (0.,1.,4.,8.):
            c=m.covariance(t)
            self.assertLess(abs(np.trace(c)-m.delta), 3e-15)
            self.assertGreater(np.linalg.eigvalsh((c+c.T)/2).min(),-3e-15)
        np.testing.assert_allclose(m.covariance(16),m.cov_target,atol=3e-12)

    def test_09_energy_matching_and_cost(self):
        m=self.m
        self.assertAlmostEqual(m.energy_fraction(m.target),m.delta,places=14)
        e_minus=np.trace(m.H0@m.Pm).real/3
        e_plus=np.trace(m.H0@m.Pp).real/6
        self.assertAlmostEqual(e_minus,e_plus,places=14)
        self.assertGreater(abs(np.trace(m.H0@(m.target-m.initial)).real),1e-6)

    def test_10_energy_error_bound(self):
        m=self.m
        emin=np.linalg.eigvalsh(m.H0).min()
        factor=np.linalg.norm(m.initial-m.target)*np.linalg.norm(
            m.H0@(m.Pm-m.delta*m.I))/emin
        for t in (.2,1.,4.,8.):
            self.assertLessEqual(abs(m.energy_fraction(m.evolve(m.initial,t))-m.delta),
                                  factor*np.exp(-1.5*t)+5e-15)

    def test_11_finite_collision_convergence(self):
        m=self.m
        ref=m.evolve(m.initial,1.)
        errors=[]
        for steps in (16,32,64,128):
            r=mat(np.linalg.matrix_power(m.step(1/steps),steps)@vec(m.initial))
            self.assertLess(abs(np.trace(r)-1),2e-12)
            self.assertLess(abs(np.trace(m.Pm@r)-m.delta),1e-12)
            errors.append(np.linalg.norm(r-ref))
        self.assertTrue(all(errors[i+1]<.6*errors[i] for i in range(3)))

    def test_12_counting_from_tilted_generator(self):
        m=self.m
        for z,t in ((0.,.2),(.3,1.),(.7,2.),(1.,1.)):
            self.assertLess(abs(m.count_direct(z,t)-m.count_pgf(z,t)),2e-14)

    def test_13_not_poisson_or_state_weight_selection(self):
        m=self.m
        rate=10/3-2*m.delta
        self.assertGreater(abs(rate-m.D),.1)
        self.assertGreater(abs(m.count_pgf(0,1)-np.exp(-rate)),.005)
        alt=.1*m.Pm/3+.9*m.Pp/6
        self.assertLess(np.linalg.norm(m.liouv@vec(alt)),2e-15)

    def test_14_not_flavour_or_vacuum_prediction(self):
        m=self.m
        marginal=np.trace(m.target.reshape(3,3,3,3),axis1=1,axis2=3)
        np.testing.assert_allclose(marginal,np.eye(3)/3,atol=3e-15)
        c=2.4
        np.testing.assert_allclose(cs(m.H0+c*m.I),cs(m.H0),atol=2e-15)
        self.assertAlmostEqual(np.trace(m.target@(m.H0+c*m.I)).real-
                               np.trace(m.target@m.H0).real,c,places=13)


    def test_15_entropy_maximum_and_monotonicity(self):
        m=self.m
        def entropy(a):
            v=np.linalg.eigvalsh((a+a.conj().T)/2)
            v=v[v>1e-15]
            return float(-np.sum(v*np.log(v)))
        values=[entropy(m.evolve(m.initial,t)) for t in (0.,.2,1.,4.,12.)]
        upper=entropy(m.target)
        self.assertTrue(all(values[i+1]>=values[i]-1e-13 for i in range(4)))
        self.assertTrue(all(v<=upper+1e-13 for v in values))
        explicit=-(1-m.delta)*np.log((1-m.delta)/6)-m.delta*np.log(m.delta/3)
        self.assertAlmostEqual(upper,explicit,places=14)

    def test_16_general_initial_states(self):
        m=self.m
        rng=np.random.default_rng(19092026)
        for _ in range(5):
            a=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9))
            r=a@a.conj().T
            r/=np.trace(r)
            target=m.twirl(r)
            out=m.evolve(r,6.)
            self.assertLessEqual(np.linalg.norm(out-target),
                np.exp(-9)*np.linalg.norm(r-target)+3e-15)

    def test_17_explicit_system_ancilla_unitary(self):
        m=self.m
        sigma_x=np.array([[0,1],[1,0]],complex)
        a=m.L[0]
        angle=.07
        u=expm(-1j*angle*np.kron(a,sigma_x))
        initial=np.kron(m.initial,np.diag([1.,0.]))
        total=u@initial@u.conj().T
        reduced=np.trace(total.reshape(9,2,9,2),axis1=1,axis2=3)
        ev,v=np.linalg.eigh(a)
        c=(v*np.cos(angle*ev))@v.conj().T
        si=(v*np.sin(angle*ev))@v.conj().T
        np.testing.assert_allclose(reduced,c@m.initial@c+si@m.initial@si,atol=3e-15)


    def test_18_conditional_energy_fluctuations(self):
        m=self.m
        e0=np.trace(m.target@m.H0).real
        vh=np.var(m.masses)
        vm=np.trace(m.Pm@m.H0@m.H0).real/3-e0**2
        vp=np.trace(m.Pp@m.H0@m.H0).real/6-e0**2
        self.assertAlmostEqual(vm,vh,places=13)
        self.assertAlmostEqual(vp,2.5*vh,places=13)

    def test_19_macro_ratio_variance(self):
        m=self.m
        energies=[]; labels=[]; probabilities=[]
        for i in range(3):
            for j in range(i,3):
                energies.append(m.masses[i]+m.masses[j]); labels.append(0.)
                probabilities.append((1-m.delta)/6)
        for i in range(3):
            for j in range(i+1,3):
                energies.append(m.masses[i]+m.masses[j]); labels.append(1.)
                probabilities.append(m.delta/3)
        e,y,w=map(np.array,(energies,labels,probabilities))
        e0=np.dot(w,e)
        coefficient=np.dot(w,e**2*(y-m.delta)**2)/e0**2
        expected=m.delta*(1-m.delta)*(1+(1+1.5*m.delta)*np.var(m.masses)/e0**2)
        self.assertAlmostEqual(coefficient,expected,places=14)
        self.assertAlmostEqual(np.dot(w,e*y)/e0,m.delta,places=14)


    def test_20_additive_energy_closed_form(self):
        m=self.m
        e0=np.trace(m.H0@m.target).real
        ei=np.trace(m.H0@m.initial).real
        ai=np.trace(m.H0@m.Pm@m.initial).real
        for t in (0.,.2,1.,4.,8.,12.):
            factor=np.exp(-1.5*t)
            energy=e0+(ei-e0)*factor
            numerator=m.delta*e0+(ai-m.delta*e0)*factor
            r=m.evolve(m.initial,t)
            self.assertAlmostEqual(np.trace(m.H0@r).real,energy,places=13)
            self.assertAlmostEqual(m.energy_fraction(r),numerator/energy,places=14)


def compute() -> dict:
    m=Model()
    eig=np.linalg.eigvalsh(m.diss)
    values, counts=np.unique(np.round(eig,10),return_counts=True)
    out={'scope':'Conditional microscopic preparation, no observational fit; original alpha_s matching, PMNS and abundance maps not derived.',
         'alpha_s_input':m.alpha,'p':m.p,'delta':m.delta,'u_legacy':m.delta/8,
         'D_legacy':m.D,'q_legacy':m.q,
         'dissipator_eigenvalues_per_gamma':dict(zip(map(str,values),map(int,counts))),
         'initial_one_copy_purity':float(np.trace(m.rho@m.rho).real),
         'stationary_one_copy_purity':1/3,
         'stationary_entropy_nats':float(-(1-m.delta)*np.log((1-m.delta)/6)-m.delta*np.log(m.delta/3)),
         'initial_pair_energy':float(np.trace(m.initial@m.H0).real),
         'stationary_pair_energy':float(np.trace(m.target@m.H0).real),
         'preparation_energy_change':float(np.trace((m.target-m.initial)@m.H0).real),
         'stationary_energy_fraction':m.energy_fraction(m.target),
         'stationary_mub_max_abs':float(np.max(np.abs(mub_average(m.p)-m.target))),
         'time_diagnostics':[], 'collision_diagnostics':[]}
    e0=out['stationary_pair_energy']
    var_coeff=m.delta*(1-m.delta)*(1+(1+1.5*m.delta)*np.var(m.masses)/e0**2)
    out['macro_energy_sampling']={
        'independent_cells_assumed':True,
        'variance_times_cell_count_asymptotic':float(var_coeff),
        'standard_error_at_1e6_cells_asymptotic':float(np.sqrt(var_coeff/1e6)),
        'hoeffding_99_percent_absolute_bound_at_1e6_cells':float(
            (m.masses.max()/m.masses.min())*np.sqrt(np.log(200)/(2e6))),
        'not_a_cosmic_power_spectrum':True}

    for t in (0.,1.,2.,4.,8.,12.):
        r=m.evolve(m.initial,t)
        out['time_diagnostics'].append({'gamma_t':t,
            'state_hs_error':float(np.linalg.norm(r-m.target)),
            'state_hs_error_upper':float(np.exp(-1.5*t)*np.linalg.norm(m.initial-m.target)),
            'swap_probability':float(np.trace(r@m.Pm).real),
            'energy_fraction':m.energy_fraction(r),
            'covariance_max_abs_error':float(np.max(np.abs(m.covariance(t)-m.cov_target)))})
    for n in (16,32,64,128,256):
        r=mat(np.linalg.matrix_power(m.step(1/n),n)@vec(m.initial))
        out['collision_diagnostics'].append({'steps':n,'time_gamma':1,
            'state_hs_error_to_lindblad':float(np.linalg.norm(r-m.evolve(m.initial,1)))})
    mean=10/3-2*m.delta
    extra=4*m.delta*(1-m.delta)
    out['physical_record_diagnostic_gamma_t_1']={
        'mean':mean,'variance':mean+extra,'fano':(mean+extra)/mean,
        'zero_count_probability':m.count_pgf(0.,1.),
        'poisson_same_mean_zero_count':float(np.exp(-mean)),
        'legacy_D_not_equal':m.D,
        'tilted_generator_max_abs':max(abs(m.count_direct(z,t)-m.count_pgf(z,t))
            for z,t in ((0.,.2),(.3,1.),(.7,2.),(1.,1.)))}
    out['time_for_absolute_hs_bound_1e_8_per_gamma']=float(np.log(
        np.linalg.norm(m.initial-m.target)/1e-8)/1.5)
    return out


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results.json'))
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Verification))
    if not result.wasSuccessful():
        raise SystemExit(1)
    data=compute()
    data['verification']={'tests_run':result.testsRun,'failures':len(result.failures),
                          'errors':len(result.errors)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(data,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
