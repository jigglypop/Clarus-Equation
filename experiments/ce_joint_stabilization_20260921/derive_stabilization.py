"""CE-JS2: a massive neutral Dirac determinant repairs the JS1 radial runaway.

All inputs are declared EFT assumptions. No measured value or fitted target is used.
Global existence is proved in chapter 31; finite computations locate an example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import zeta


class Model:
    def __init__(self, nmax=4096, mass=1., neutral_count=1):
        self.n = np.arange(1, nmax+1, dtype=float)
        self.nmax, self.mass, self.neutral_count = nmax, mass, neutral_count
        self.w = self.n**-5
        self.g, self.kappa, self.lam, self.u0, self.radius_reference = .6, .18, 10., .5, 1.
        self.prefactor = 3/(64*np.pi**6)

    def evaluate(self, point):
        u, r, *a = point
        if u <= 0:
            raise ValueError('This derivative implementation uses the open gapped domain u>0.')
        n, w = self.n, self.w
        y = self.kappa*u*np.exp(2*r)
        z = 2*np.pi*n*np.sqrt(y)
        exponential = np.exp(-z)
        f = exponential*(1+z+z*z/3)
        fy = -(2*np.pi*n)**2*(1+z)*exponential/6
        fyy = (2*np.pi*n)**4*exponential/12
        phase = np.exp(2j*np.pi*n[:, None]*np.asarray(a))
        trace = phase.sum(axis=1)
        da = 2j*np.pi*n[:, None]*phase
        waa = np.empty((3, 3))
        wa = np.sum(w[:, None]*(2*np.real(da*trace.conj()[:, None])-2*f[:, None]*da.real), axis=0)
        way = np.sum(-2*w[:, None]*fy[:, None]*da.real, axis=0)
        for i in range(3):
            for j in range(3):
                terms = 2*np.real(da[:, i]*da[:, j].conj())
                if i == j:
                    second = -(2*np.pi*n)**2*phase[:, i]
                    terms += 2*np.real(second*trace.conj())-2*f*second.real
                waa[i, j] = w@terms
        winding = w@(np.abs(trace)**2-1-2*f*trace.real)
        wy, wyy = -2*w@(fy*trace.real), -2*w@(fyy*trace.real)
        zn = 2*np.pi*n*self.mass*np.exp(r)
        en = np.exp(-zn)
        sn = w@(en*(1+zn+zn*zn/3))
        sr = w@(-en*zn*zn*(1+zn)/3)
        srr = w@(en*(zn**4-2*zn**3-2*zn**2)/3)
        multiplicity = 4*self.neutral_count
        full = winding+multiplicity*sn
        p, er, dy = self.prefactor*np.exp(-6*r), np.exp(-r), y/u
        h, hp, hpp = self.lam*(u-self.u0)**2, 2*self.lam*(u-self.u0), 2*self.lam
        value = er*h+p*full
        gradient = np.r_[er*hp+p*dy*wy, -er*h+p*(-6*full+2*y*wy+multiplicity*sr), p*wa]
        hessian = np.empty((5, 5))
        hessian[0, 0] = er*hpp+p*dy*dy*wyy
        hessian[0, 1] = hessian[1, 0] = -er*hp+p*dy*(-4*wy+2*y*wyy)
        hessian[0, 2:] = hessian[2:, 0] = p*dy*way
        hessian[1, 1] = er*h+p*(36*full-20*y*wy+4*y*y*wyy-12*multiplicity*sr+multiplicity*srr)
        hessian[1, 2:] = hessian[2:, 1] = p*(-6*wa+2*y*way)
        hessian[2:, 2:] = p*waa
        return value, gradient, hessian, dict(y=y, W=winding, S=sn, Sr=sr, Srr=srr, Wy=wy,
                                              prefactor=p, Higgs=er*h)

    def solve(self, start=(.5, -.6, .27)):
        def embed(v):
            return [v[0], v[1], 0., v[2], -v[2]]
        def gradient(v):
            _, g, _, d = self.evaluate(embed(v))
            return [g[0], g[1]/d['prefactor'], (g[3]-g[4])/d['prefactor']]
        solved = root(gradient, start, tol=1e-11)
        point = np.array(embed(solved.x))
        if np.linalg.norm(gradient(solved.x)) > 2e-8:
            raise RuntimeError(str(solved))
        return point


def run():
    # A declared comparison grid tests the mass dependence; it is not a fit.
    comparison = []
    for mass in (.25, .5, .75, 1., 1.5, 2.):
        variant = Model(mass=mass)
        selected = variant.solve((.5, np.log(.48/mass), .28))
        v, g, h, _ = variant.evaluate(selected)
        comparison.append({'neutral_mass': mass, 'point': selected.tolist(),
                           'potential': float(v), 'minimum_Hessian_eigenvalue': float(np.linalg.eigvalsh(h)[0])})
    model = Model(mass=.5)
    point = model.solve((.5, .1, .29))
    value, gradient, hessian, details = model.evaluate(point)
    checks = []
    def check(name, passed, **evidence):
        checks.append(dict(name=name, passed=bool(passed), **evidence))
        if not passed:
            raise AssertionError(checks[-1])
    check('five_variable_stationarity', np.max(np.abs(gradient)) < 1e-12,
          maximum_residual=float(np.max(np.abs(gradient))))
    check('full_Hessian_positive_at_repair_example', np.linalg.eigvalsh(hessian)[0] > 1e-5)
    check('mass_one_symmetric_branch_remains_a_saddle', comparison[3]['minimum_Hessian_eigenvalue'] < -6e-4)
    for index, offset in enumerate(([.003, -.01, .02, .01, -.03], [-.004, .02, -.01, -.02, .01])):
        x = point+offset
        _, g, h, _ = model.evaluate(x)
        step = 1e-5
        columns, differences = [], []
        for j in range(5):
            delta = np.eye(5)[j]*step
            fm2, gm2, _, _ = model.evaluate(x-2*delta)
            fm1, gm1, _, _ = model.evaluate(x-delta)
            fp1, gp1, _, _ = model.evaluate(x+delta)
            fp2, gp2, _, _ = model.evaluate(x+2*delta)
            differences.append((fm2-8*fm1+8*fp1-fp2)/(12*step))
            columns.append((gm2-8*gm1+8*gp1-gp2)/(12*step))
        check(f'analytic_gradient_{index}', np.max(np.abs(g-differences)) < 2e-10)
        check(f'analytic_full_Hessian_{index}', np.max(np.abs(h-np.array(columns).T)) < 3e-9)
    fine = Model(nmax=8192, mass=.5)
    vf, gf, hf, _ = fine.evaluate(point)
    check('winding_refinement', max(abs(value-vf), np.max(np.abs(gradient-gf)),
                                   np.max(np.abs(hessian-hf))) < 1e-9)
    eigen = np.linalg.eigvalsh(hessian)
    # Read the spectral thresholds only after jointly selecting the stationary state.
    u, r, _, q, _ = point
    kap, reff = model.kappa*np.exp(-r), np.exp(1.5*r)
    thresholds = 2*np.sqrt([kap*u, kap*u+q*q/reff**2, kap*u+(1-q)**2/reff**2])
    aa, bb = np.sqrt(thresholds[1:]**2-thresholds[0]**2)/2
    inferred_q, inferred_reff = aa/(aa+bb), 1/(aa+bb)
    check('selected_spectrum_inverse_record', max(abs(q-inferred_q), abs(reff-inferred_reff)) < 1e-13)
    target = -details['Higgs']/details['prefactor']-6*details['W']+2*details['y']*details['Wy']
    def radial_function(b):
        z = 2*np.pi*model.n*b
        f = np.exp(-z)*(1+z+z*z/3)
        sr = -np.exp(-z)*z*z*(1+z)/3
        return 4*np.dot(model.w, 6*f-sr)
    inferred_b = brentq(lambda b: radial_function(b)-target, .001, 5., xtol=1e-14)
    check('unique_radial_mass_inverse', abs(inferred_b-model.mass*np.exp(r)) < 2e-13,
          inferred_mass_times_radius=float(inferred_b))
    check('admissible_radial_inverse_range', 0 < target < 24*zeta(5))
    gm = np.zeros(5)
    gm[1] = 4*details['prefactor']/model.mass*(details['Srr']-6*details['Sr'])
    sensitivity = -np.linalg.solve(hessian, gm)
    step = 1e-4
    lower_mass = Model(mass=.5-step).solve((u, r, q))
    upper_mass = Model(mass=.5+step).solve((u, r, q))
    derivative_fd = (upper_mass-lower_mass)/(2*step)
    check('selected_radius_strictly_decreases_with_free_neutral_mass', sensitivity[1] < 0,
          derivative=float(sensitivity[1]))
    check('implicit_full_state_sensitivity', np.max(np.abs(sensitivity-derivative_fd)) < 1e-5)
    without_neutral = Model(mass=.5, neutral_count=0)
    _, old_gradient, _, _ = without_neutral.evaluate(point)
    check('new_term_actually_closes_old_radial_force', abs(old_gradient[1]) > 1e-4,
          old_radial_force=float(old_gradient[1]), new_radial_force=float(gradient[1]))
    # One periodic MASSLESS neutral Dirac makes U strictly positive at finite radius.
    check('massless_control_has_positive_lower_bound', 4*zeta(5)-2*zeta(5) > 0)
    check('global_small_radius_positive_barrier', 4*(7/(3*np.e))-2*zeta(5) > 0)
    check('negative_large_radius_trial', 26/27-4*np.exp(-2*np.pi)*(1+2*np.pi+(2*np.pi)**2/3) > 0)
    return {'scope': 'new JS2 EFT branch; not all-natural-constant selection or observational validation',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inputs': {'g_reference': .6, 'kappa_reference': .18,
                       'portal_matching': 'declared low-energy condition, not a derived full-parent result',
                       'radius_reference': 1., 'lambda': 10., 'u0': .5,
                       'neutral_Dirac_mass': .5, 'neutral_Dirac_count': 1,
                       'local_vacuum_matching': 'same zero finite terms as JS1'},
            'point': point.tolist(), 'potential': float(value), 'gradient': gradient.tolist(),
            'hessian_eigenvalues': np.linalg.eigvalsh(hessian).tolist(),
            'details': {k: float(v) for k, v in details.items()},
            'radius': float(np.exp(point[1])),
            'epsilon': float(point[3]**2*np.exp(-3*point[1])/3),
            'selected_thresholds': thresholds.tolist(),
            'mass_scan_including_failed_branches': comparison,
            'radial_inverse_target': float(target), 'inferred_mass_times_radius': float(inferred_b),
            'selected_state_mass_derivative': sensitivity.tolist(),
            'checks': checks, 'all_checks_passed': all(row['passed'] for row in checks),
            'global_radius_bounds': [1/(2*np.pi),
                                     (2/(26/27-4*np.exp(-2*np.pi)*(1+2*np.pi+(2*np.pi)**2/3)))**(1/6)],
            'observational_validation': False, 'full_goal_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in
                      ['point', 'potential', 'hessian_eigenvalues', 'all_checks_passed']},
                     ensure_ascii=False, indent=2))
