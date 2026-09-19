#!/usr/bin/env python3
"""Independent equation checks for the curated CE results; no observational fit.

This is a new compact implementation of equations in the cited research notes,
not a copy or rerun of their unavailable simulation archives. Run from any path.
Dependencies: numpy, scipy, sympy, mpmath. Only --output writes a result file.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
import sympy as sp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

mp.mp.dps = 60
PI = mp.pi
DIAGNOSTICS: dict[str, object] = {}


def spectrum(s, eps, theta):
    s, eps, theta = map(mp.mpf, (s, eps, theta))
    if not s > 2 * abs(eps):
        raise ValueError('Require s > 2*abs(epsilon).')
    return [s + 2 * eps * mp.cos((theta + 2 * PI * j) / 3) for j in range(3)]


def potential(s, eps, theta):
    """Relative complex-scalar potential at mu^2=1; epsilon is dimensional."""
    f = lambda x: x*x*(mp.log(x) - mp.mpf('1.5'))
    return (sum(map(f, spectrum(s, eps, theta)))
            - sum(map(f, spectrum(s, eps, PI)))) / (32 * PI**2)


def determinant(s, eps, theta):
    return s**3 - 3*s*eps**2 + 2*eps**3*mp.cos(theta)


def orbit_coefficient(a, b):
    a, b = map(mp.mpf, (a, b))
    if min(a, b) <= 0:
        raise ValueError('Positive squared masses are required.')
    if a == b:
        return mp.mpf('0')
    return ((a+b)/2 - a*b*mp.log(b/a)/(b-a)) / (16*PI**2)


def reference_gravity(s, eps, B, xi=0):
    alpha = mp.mpf('0.25') - xi
    if alpha <= 0 or B <= s + 2*abs(eps):
        raise ValueError('Require xi < 1/4 and B above the light spectrum.')
    return alpha/(8*PI**2)*sum(
        a*mp.log(a)+B*mp.log(B)-(a+B)*mp.log((a+B)/2)
        for a in spectrum(s, eps, PI))


def phase_kinetic(s, eps, B, theta):
    xs = spectrum(s, eps, theta)
    dx = [-2*eps*mp.sin((theta+2*PI*j)/3)/3 for j in range(3)]
    return sum(d*d/x for d, x in zip(dx, xs))/(96*PI**2) + sum(
        orbit_coefficient(x, B) for x in xs)/2


def singlet_levels(eta: float, cutoff: int = 64):
    """Jacobi basis for h=-y(1-y)d2-(2-3y)d+eta*y, weight 2y dy."""
    if eta < 0 or cutoff < 3:
        raise ValueError('Require eta >= 0 and cutoff >= 3.')
    n = np.arange(cutoff, dtype=float)
    yd = (1 + 1/((2*n+1)*(2*n+3)))/2
    k = n[:-1]
    yo = -np.sqrt((k+1)*(k+2))/(2*(2*k+3))
    vals, vecs = eigh_tridiagonal(n*(n+2)+eta*yd, eta*yo,
                                 select='i', select_range=(0, 1))
    deriv = np.sum(yd[:, None]*vecs**2, axis=0) + 2*np.sum(
        yo[:, None]*vecs[:-1]*vecs[1:], axis=0)
    return vals, 2*eta*deriv-vals


class SelectedAdvances(unittest.TestCase):
    def near(self, a, b, tol='1e-45'):
        self.assertLessEqual(abs(a-b), mp.mpf(tol)*max(1, abs(a), abs(b)))

    def test_01_spectral_invariants(self):
        for s, eps in [(mp.mpf(1), mp.mpf('.15')), (mp.mpf(3), mp.mpf('.7'))]:
            for t in [mp.mpf(0), mp.mpf('.4'), mp.mpf('1.2'), PI]:
                xs = spectrum(s, eps, t)
                self.near(sum(xs), 3*s)
                self.near(sum(x*x for x in xs), 3*s*s+6*eps*eps)
                self.near(mp.fprod(xs), determinant(s, eps, t))
                for x in xs:
                    self.near((x-s)**3-3*eps**2*(x-s), 2*eps**3*mp.cos(t))

    def test_02_positive_resolvent_potential(self):
        for r in [mp.mpf('.15'), mp.mpf('.35')]:
            for t in [mp.mpf(0), mp.mpf('.4'), mp.mpf('2.4'), PI]:
                integrand = lambda u: u*mp.log1p(2*r**3*(1+mp.cos(t))/(
                    (u+1-2*r)*(u+1+r)**2))
                value = mp.quad(integrand, [0, 1, mp.inf])/(16*PI**2)
                self.near(value, potential(1, r, t))
                self.assertGreaterEqual(value, 0)

    def test_03_global_minimum_derivative(self):
        s, eps, t = mp.mpf(1), mp.mpf('.15'), mp.mpf('1.2')
        xs = spectrum(s, eps, t)
        positive = mp.quad(lambda u: u/mp.fprod(u+x for x in xs),
                           [0, 1, mp.inf])/(16*PI**2)
        derivative = mp.diff(lambda z: potential(s, eps, z), t)/(-2*eps**3*mp.sin(t))
        self.near(positive, derivative)
        self.assertGreater(positive, 0)
        self.near(potential(s, eps, PI), 0)
        DIAGNOSTICS['barrier_r015'] = str(potential(s, eps, 0))

    def test_04_projector_metric(self):
        rng = np.random.default_rng(20260919)
        errors = []
        for _ in range(24):
            z = rng.normal(size=2)+1j*rng.normal(size=2)
            dz = rng.normal(size=2)+1j*rng.normal(size=2)
            v = np.r_[1, z]; dv = np.r_[0, dz]
            norm = float(np.vdot(v, v).real)
            u = v/np.sqrt(norm)
            du = dv/np.sqrt(norm)-v*np.vdot(v, dv).real/norm**1.5
            dP = np.outer(du, u.conj())+np.outer(u, du.conj())
            lhs = np.trace(dP@dP).real/2
            rhs = np.vdot(dz, dz).real/norm-abs(np.vdot(z, dz))**2/norm**2
            errors.append(float(abs(lhs-rhs)))
            self.assertAlmostEqual(lhs, rhs, places=12)
        DIAGNOSTICS['projector_metric_max_abs_error'] = max(errors)

    def test_05_positive_orbit_coefficient(self):
        for a, b in [(1, 2), (mp.mpf('.7'), mp.mpf('1.15')), (1, 100)]:
            a, b = mp.mpf(a), mp.mpf(b)
            independent = (a-b)**2/(16*PI**2)*mp.quad(
                lambda u: u*(1-u)/(u*a+(1-u)*b), [0, 1])
            self.near(orbit_coefficient(a, b), independent)
            self.assertGreater(independent, 0)

    def test_06_common_mass_derivatives(self):
        for s, eps, t in [('1', '.15', '1.2'), ('3', '.7', '.4')]:
            s, eps, t = map(mp.mpf, (s, eps, t))
            uss = mp.diff(lambda q: potential(q, eps, t), s, 2)
            usss = mp.diff(lambda q: potential(q, eps, t), s, 3)
            self.near(uss, mp.log(determinant(s, eps, t)/determinant(s, eps, PI))/(16*PI**2))
            inverse = sum(1/x for x in spectrum(s, eps, t))-sum(1/x for x in spectrum(s, eps, PI))
            self.near(usss, inverse/(16*PI**2))

    def test_07_representation_traces(self):
        multiplets = [(6, Fraction(1, 6)), (3, Fraction(-2, 3)),
                      (3, Fraction(1, 3)), (2, Fraction(-1, 2)),
                      (1, Fraction(1)), (1, Fraction(0))]
        ty = sum(d*y*y for d, y in multiplets)
        q2 = 3*(Fraction(2, 3)**2+Fraction(-1, 3)**2)+3*Fraction(-2, 3)**2+3*Fraction(1, 3)**2+1+1
        self.assertEqual(ty, Fraction(10, 3))
        self.assertEqual(q2, Fraction(16, 3))
        self.assertEqual(Fraction(3, 5)*ty, 2)
        alpha, mass, charge = sp.symbols('alpha mass charge')
        self.assertEqual(sp.simplify((2*alpha**2*mass**2*charge/45)/(16*sp.pi**2)
                                    -(alpha/sp.pi)**2*mass**2*charge/360), 0)

    def test_08_matching_sum_rules(self):
        s, eps, B, t = map(mp.mpf, ('1', '.15', '10', '1.2'))
        a, x = spectrum(s, eps, PI), spectrum(s, eps, t)
        c = [(v+B)/2 for v in a]
        bosons, fermions = x+[B]*3+c, a+[B]*3+c
        for n in range(3):
            self.near(sum(v**n for v in bosons), sum(v**n for v in fermions))
        for xi in [mp.mpf(0), mp.mpf(1)/6]:
            xic = mp.mpf(3)/4-2*xi
            for n in range(2):
                total = (mp.mpf(1)/6-xi)*sum(v**n for v in x+[B]*3)
                total += (mp.mpf(1)/6-xic)*sum(v**n for v in c)
                total += sum(v**n for v in fermions)/12
                self.near(total, 0)

    def test_09_positive_induced_gravity(self):
        s, eps, B = mp.mpf(1), mp.mpf('.15'), mp.mpf(10)
        a = spectrum(s, eps, PI)
        integral = sum(mp.quad(lambda u: ((mp.exp(-v*u/2)-mp.exp(-B*u/2))/u)**2,
                               [0, 1, mp.inf]) for v in a)/(32*PI**2)
        self.near(reference_gravity(s, eps, B), integral)
        self.assertGreater(integral, 0)

    def test_10_common_kinetic_gravity_limit(self):
        s, eps = mp.mpf(1), mp.mpf('.15')
        target = 1/(2*mp.log(2))
        errors = []
        for B in map(mp.mpf, ('1000', '100000', '10000000')):
            ratio = phase_kinetic(s, eps, B, PI)/reference_gravity(s, eps, B)
            errors.append(abs(ratio/target-1))
        self.assertTrue(errors[2] < errors[1] < errors[0])
        self.assertLess(errors[-1], mp.mpf('1e-5'))
        DIAGNOSTICS['f_over_Mind_limit_xi0'] = str(mp.sqrt(target))
        DIAGNOSTICS['K_over_F_relative_errors'] = list(map(str, errors))

    def test_11_singlet_minmax_bound(self):
        for eta in [10.01, 20, 100]:
            A = np.array([[2*eta/3, -eta*np.sqrt(2)/6],
                          [-eta*np.sqrt(2)/6, 3+8*eta/15]])
            self.assertGreater(np.linalg.eigvalsh(eta*np.eye(2)-A).min(), 0)
            self.assertAlmostEqual(np.linalg.det(eta*np.eye(2)-A), eta*(eta/10-1), places=10)
        values, _ = singlet_levels(0)
        np.testing.assert_allclose(values, [0, 3], atol=1e-12)

    def test_12_mixed_state_stationary_points(self):
        points = []
        for zeta, expected in [(.01, .3008636651), (.1, .9761212567),
                               (.25, 1.6042340855), (.5, 2.3355316936)]:
            roots = []
            for cutoff in [64, 96]:
                def slope(z):
                    _, d = singlet_levels(z*z/2, cutoff)
                    return ((1-zeta)*d[0]+zeta*d[1])/(z*z)
                roots.append(brentq(slope, .03, np.sqrt(20), xtol=1e-11))
            self.assertLess(abs(roots[0]-roots[1]), 1e-8)
            self.assertLess(abs(roots[1]-expected), 2e-8)
            points.append({'zeta': zeta, 'z_c_star': roots[1],
                           'cutoff_difference': abs(roots[0]-roots[1])})
        DIAGNOSTICS['mixed_state_roots'] = points

    def test_13_state_pressure_identity(self):
        L, F, C = sp.symbols('L F C', positive=True)
        W = sp.Function('W')
        rho = W(C*L**2/F)/L**4
        pressure = -rho-L*sp.diff(rho, L)/3
        self.assertEqual(sp.simplify(L*sp.diff(rho, L)+2*F*sp.diff(rho, F)+4*rho), 0)
        self.assertEqual(sp.simplify(pressure-rho/3-2*F*sp.diff(rho, F)/3), 0)

    def test_14_force_slope_relation(self):
        k = sp.symbols('k', positive=True)
        alpha = -1/(2*sp.sqrt(k)); lam = 2/sp.sqrt(k)
        epsilon_v = lam**2/2
        self.assertEqual(sp.simplify(lam+4*alpha), 0)
        self.assertEqual(sp.simplify(1+2*alpha**2-(1+epsilon_v/4)), 0)

    def test_15_common_relative_force(self):
        transform = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        for k in [.3, 1.75, 4.2]:
            a, b = 1/(2*k), .00719529945218
            matrix = np.array([[a+b, a-b], [a-b, a+b]])
            np.testing.assert_allclose(transform.T@matrix@transform,
                                       np.diag([2*a, 2*b]), atol=2e-15)
        DIAGNOSTICS['common_force_ratio_at_kappa_7_over_4'] = 1+1/(2*1.75)

    def test_16_principal_characteristics(self):
        a, b, c, omega, k = sp.symbols('a b c omega k')
        gamma = sp.Matrix([[a, b], [b, c]])
        self.assertEqual(sp.expand((gamma*(omega**2-k**2)).det()
                                   -gamma.det()*(omega**2-k**2)**2), 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the current run summary as JSON.')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelectedAdvances)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        'schema': 'ce-selected-advances-v1', 'date': '2026-09-19',
        'scope': 'Independent equation checks; not original archive or full cosmology reproduction.',
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'passed': result.wasSuccessful(),
        'observational_fit_performed': False, 'full_joint_rmse': None,
        'original_simulation_archives_rerun': False,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'sympy': sp.__version__, 'mpmath': mp.__version__},
        'diagnostics': DIAGNOSTICS,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
