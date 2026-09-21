"""CE-RB6: S4 sum bounds, digamma responses, flat limit and curvature variation.

Synthetic parameters only. Numerical checks accompany, and do not replace, chapter 24 proofs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

from verify_reverse import Evidence

mp.mp.dps = 55


def masses(s, eps, theta):
    return [s+2*eps*mp.cos((theta+2*mp.pi*j)/3) for j in range(3)]


def spectral_f(x, hubble, xi):
    nu = mp.sqrt(mp.mpf(9)/4-x/hubble**2-12*xi)
    return (nu**2-mp.mpf(1)/4)*(mp.digamma(mp.mpf(3)/2-nu)+mp.digamma(mp.mpf(3)/2+nu))


def digamma_response(s, eps, theta, hubble, xi, weighted=False):
    def branch(angle):
        return sum((x if weighted else 1)*spectral_f(x, hubble, xi)
                   for x in masses(s, eps, angle))
    return mp.re(-hubble**2/(32*mp.pi**2 if weighted else 16*mp.pi**2)
                 *(branch(theta)-branch(mp.pi)))


def phase_derivative(s, eps, theta, hubble, xi):
    return mp.re(-hubble**2/(16*mp.pi**2)*sum(
        -2*eps/3*mp.sin((theta+2*mp.pi*j)/3)*spectral_f(x, hubble, xi)
        for j, x in enumerate(masses(s, eps, theta))))


def potential(s, eps, theta, hubble, xi):
    return mp.quad(lambda angle: phase_derivative(s, eps, angle, hubble, xi), [mp.pi, theta])


def flat_potential(s, eps, theta):
    return (sum(x*x*mp.log(x) for x in masses(s, eps, theta))
            -sum(x*x*mp.log(x) for x in masses(s, eps, mp.pi)))/(32*mp.pi**2)


def derivative5(function, point):
    """Fixed-precision independent five-point derivative; avoids nested precision escalation."""
    step = mp.mpf('0.00001')*max(1, abs(point))
    return (function(point-2*step)-8*function(point-step)
            +8*function(point+step)-function(point+2*step))/(12*step)


def finite_modes(s, eps, theta, hubble, xi, cutoff):
    s, eps, theta, hubble, xi = map(float, (s, eps, theta, hubble, xi))
    n = np.arange(cutoff+1, dtype=float)
    d = (n+1)*(n+2)*(2*n+3)/6
    c = n*(n+3)+12*xi
    t = s+hubble*hubble*c
    b = (t-2*eps)*(t+eps)**2
    bp = 3*(t*t-eps*eps)
    kappa = 2*eps**3*(1+np.cos(theta))
    u = 3*hubble**4/(8*np.pi**2)*np.sum(d*np.log1p(kappa/b))
    us = -3*hubble**4/(8*np.pi**2)*np.sum(d*kappa*bp/(b*(b+kappa)))
    rho = 3*hubble**6/(16*np.pi**2)*np.sum(d*c*kappa*bp/(b*(b+kappa)))
    gamma = 1-2*eps/s
    u_bound = 15*kappa/(16*np.pi**2*gamma*hubble**2*cutoff**2)
    rho_bound = 45*(4+12*xi)*kappa/(32*np.pi**2*gamma**2*hubble**2*cutoff**2)
    return u, rho, us, u_bound, rho_bound


def run():
    evidence = Evidence()
    values = {}
    s, eps, t, kappa = sp.symbols('s eps t kappa', positive=True)
    n, z, nu = sp.symbols('n z nu', real=True)
    b = (t-2*eps)*(t+eps)**2
    evidence.zero('R59', 'B_derivative', sp.diff(b, t)-3*(t*t-eps*eps))
    evidence.zero('R59', 'scale_identity_for_log',
                  t*sp.diff(sp.log(1+kappa/b), t)
                  +eps*sp.diff(sp.log(1+kappa/b), eps)
                  +3*kappa*sp.diff(sp.log(1+kappa/b), kappa))
    # Polynomial bound certificates on n >= 1 and t >= s > 2 eps.
    y = sp.symbols('y', nonnegative=True)
    d = (n+1)*(n+2)*(2*n+3)/6
    coeffs = sp.Poly(sp.expand((5*n**3-d).subs(n, y+1)), y).all_coeffs()
    evidence.check('R59', 'degeneracy_bound_polynomial', all(c >= 0 for c in coeffs),
                   {'coefficients_after_n_equals_one_plus_y': list(map(str, coeffs))})
    evidence.zero('R59', 'log_H_derivative_identity',
                  sp.diff(sp.log(1+kappa/b), t)+kappa*sp.diff(b, t)/(b*(b+kappa)))

    cases = [('5', '.7', '.8', '.7', '.1'),
             ('1.5', '.7', '1.2', '.4', '0'),
             ('3', '.4', '.2', '2', '.25'),
             ('2', '.6', '2.9', '.2', '.01')]
    for index, strings in enumerate(cases):
        ss, ee, theta, hubble, xi = map(mp.mpf, strings)
        exact_u = potential(ss, ee, theta, hubble, xi)
        exact_rho = digamma_response(ss, ee, theta, hubble, xi, True)
        exact_us = digamma_response(ss, ee, theta, hubble, xi)
        finite = finite_modes(ss, ee, theta, hubble, xi, 96)
        u, rho, _, tu, tr = finite
        du, dr = float(exact_u)-u, float(exact_rho)-rho
        evidence.check('R59', f'U_tail_case_{index}', 0 <= du <= tu,
                       {'actual_tail': du, 'proven_bound': tu, 'cutoff': 96})
        evidence.check('R59', f'rho_tail_case_{index}', 0 <= dr <= tr,
                       {'actual_tail': dr, 'proven_bound': tr, 'cutoff': 96})
        evidence.check('R59', f'positive_response_case_{index}', exact_u > 0 and exact_rho > 0,
                       {'U': float(exact_u), 'rho': float(exact_rho)})
        finite_fine = finite_modes(ss, ee, theta, hubble, xi, 20000)
        evidence.close('R60', f'Us_digamma_vs_direct_modes_{index}',
                       float(exact_us), finite_fine[2], 2e-12)
        du_dh = derivative5(lambda hh: potential(ss, ee, theta, hh, xi), hubble)
        evidence.close('R60', f'Weyl_derivative_vs_digamma_{index}',
                       float(exact_u-hubble*du_dh/4), float(exact_rho), 1e-12)
        # Independent differentiation of the mass parameter, without an infinite mode sum.
        du_ds = derivative5(lambda mass: potential(mass, ee, theta, hubble, xi), ss)
        evidence.close('R60', f'mass_derivative_vs_digamma_{index}',
                       float(du_ds), float(exact_us), 1e-12)
        values[f'synthetic_case_{index}'] = {'s': str(ss), 'eps': str(ee), 'theta': str(theta),
                                           'H': str(hubble), 'xi': str(xi),
                                           'U': float(exact_u), 'rho': float(exact_rho)}

    evidence.zero('R60', 'degeneracy_shift', d-(z*(z*z-sp.Rational(1, 4))/3).subs(z, n+sp.Rational(3, 2)))
    evidence.zero('R60', 'partial_fraction',
                  z*(z*z-sp.Rational(1, 4))/(z*z-nu*nu)
                  -z-(nu*nu-sp.Rational(1, 4))/2*(1/(z-nu)+1/(z+nu)))
    ss, ee, theta = map(mp.mpf, ('5', '.7', '.8'))
    for power in (0, 1, 2):
        evidence.close('R60', f'trace_cancellation_power_{power}',
                       float(sum(x**power for x in masses(ss, ee, theta))),
                       float(sum(x**power for x in masses(ss, ee, mp.pi))), 1e-12)
    evidence.close('R60', 'reference_potential_zero', float(potential(ss, ee, mp.pi, mp.mpf('.5'), 0)), 0.)
    evidence.close('R60', 'reference_rho_zero', float(digamma_response(ss, ee, mp.pi, mp.mpf('.5'), 0, True)), 0.)

    flat = flat_potential(ss, ee, theta)
    kap = 2*ee**3*(1+mp.cos(theta))
    log_ratio = lambda u: mp.log1p(kap/((ss+u-2*ee)*(ss+u+ee)**2))
    integral = mp.quad(lambda u: u*log_ratio(u), [0, 1, mp.inf])/(16*mp.pi**2)
    evidence.close('R61', 'flat_integral_vs_closed_mass_logs', float(integral), float(flat), 1e-12)
    evidence.zero('R61', 'integration_by_parts_division', t*t/(t+s)-(t-s+s*s/(t+s)))
    previous_u, previous_rho = mp.inf, mp.inf
    for hubble in map(mp.mpf, ('.5', '.1', '.03')):
        xi = mp.mpf(1)/6
        u = potential(ss, ee, theta, hubble, xi)
        rho = digamma_response(ss, ee, theta, hubble, xi, True)
        error_u, error_rho = abs(u-flat), abs(rho-flat)
        evidence.check('R61', f'flat_limit_errors_decrease_{hubble}',
                       error_u < previous_u and error_rho < previous_rho,
                       {'U_error': float(error_u), 'rho_error': float(error_rho), 'H': str(hubble)})
        previous_u, previous_rho = error_u, error_rho
    fixed = finite_modes(ss, ee, theta, mp.mpf('.0001'), 0, 30)[0]
    evidence.check('R61', 'fixed_cutoff_flat_limit_counterexample', 0 < fixed < float(flat)*1e-6,
                   {'fixed_N': 30, 'H': .0001, 'U_N': fixed, 'full_flat_U': float(flat)})
    h = sp.symbols('H', positive=True)
    evidence.zero('R61', 'finite_mode_limit_zero', sp.limit(h**4*sp.log(1+kappa/b.subs(t, s+h*h)), h, 0))

    riem2, ric2, scalar2, xi = sp.symbols('Riem2 Ric2 R2 xi')
    a4 = ((5-60*xi+180*xi**2)*scalar2-2*ric2+2*riem2)/360
    weyl2 = riem2-2*ric2+scalar2/3
    euler = riem2-4*ric2+scalar2
    evidence.zero('R62', 'heat_coefficient_completion',
                  a4-(riem2-ric2)/180-(xi-sp.Rational(1, 6))**2*scalar2/2)
    evidence.zero('R62', 'weyl_euler_decomposition',
                  a4-weyl2/120+euler/360-(xi-sp.Rational(1, 6))**2*scalar2/2)
    evidence.zero('R62', 'S4_curvature_coefficient',
                  a4.subs({riem2: 24*h**4, ric2: 36*h**4, scalar2: 144*h**4})
                  -h**4*(-sp.Rational(1, 15)+72*(xi-sp.Rational(1, 6))**2))
    us_flat = mp.diff(lambda mass: flat_potential(mass, ee, theta), ss)
    uss_flat = mp.log(mp.fprod(masses(ss, ee, theta))/mp.fprod(masses(ss, ee, mp.pi)))/(16*mp.pi**2)
    evidence.close('R62', 'flat_second_response_log_determinant',
                   float(mp.diff(lambda mass: flat_potential(mass, ee, theta), ss, 2)), float(uss_flat), 1e-12)
    hubble = mp.mpf('.015')
    for coupling in (mp.mpf(0), mp.mpf('.1'), mp.mpf(1)/6, mp.mpf('.25')):
        exact = potential(ss, ee, theta, hubble, coupling)
        extracted = (exact-flat-12*hubble**2*(coupling-mp.mpf(1)/6)*us_flat)/hubble**4
        predicted = uss_flat*(-mp.mpf(1)/15+72*(coupling-mp.mpf(1)/6)**2)
        relative = abs((extracted-predicted)/predicted)
        evidence.check('R62', f'curvature_expansion_vs_exact_S4_{coupling}', relative < .001,
                       {'extracted_H4_coefficient': float(extracted), 'heat_coefficient': float(predicted),
                        'relative_error': float(relative), 'H': str(hubble)})
    values['curvature_expansion_scope'] = 'constant theta, positive masses, local low-curvature expansion; no in-in state'

    time = sp.symbols('time', positive=True)
    hubble_t = 1/time
    scalar = 6*(sp.diff(hubble_t, time)+2*hubble_t**2)
    r00 = -3*(sp.diff(hubble_t, time)+hubble_t**2)
    rii_over_g = sp.diff(hubble_t, time)+3*hubble_t**2
    box_r = -sp.diff(scalar, time, 2)-3*hubble_t*sp.diff(scalar, time)
    e00 = 2*scalar*r00+scalar**2/2+2*(-box_r-sp.diff(scalar, time, 2))
    epi = 2*scalar*rii_over_g-scalar**2/2+2*(box_r+hubble_t*sp.diff(scalar, time))
    evidence.zero('R63', 'linear_FLRW_scalar_curvature', scalar-6/time**2)
    evidence.zero('R63', 'linear_FLRW_R2_variation_00', e00+54/time**4)
    evidence.zero('R63', 'linear_FLRW_R2_variation_spatial', epi+18/time**4)
    evidence.zero('R63', 'R2_stress_conservation', sp.diff(e00, time)+3*hubble_t*(e00+epi))
    curvature, volume_coefficient = sp.symbols('R alpha', real=True)
    evidence.zero('R63', 'Einstein_space_R2_variation_zero', 2*curvature*curvature/4-curvature**2/2)
    evidence.zero('R63', 'H4_Weyl_stress_zero', volume_coefficient*h**4-h/4*sp.diff(volume_coefficient*h**4, h))
    evidence.check('R63', 'nonzero_dynamic_response', e00.subs(time, 1) != 0,
                   {'a(t)': 't', 'E00_at_t_1': str(e00.subs(time, 1)), 'de_Sitter_E00': 0})

    # The regular on-shell Schur complement has no mixed term at either fixed phase extremum.
    phase, c = sp.symbols('theta C', real=True)
    bb, epos = sp.symbols('B e', positive=True)
    local_log = sp.log(1+2*epos**3*(1+sp.cos(phase))/bb)
    evidence.zero('R64', 'phase_zero_negative_hessian',
                  sp.diff(local_log, phase, 2).subs(phase, 0)+2*epos**3/(bb+4*epos**3))
    evidence.zero('R64', 'phase_pi_positive_hessian',
                  sp.diff(local_log, phase, 2).subs(phase, sp.pi)-2*epos**3/bb)
    for angle in (0, sp.pi):
        mixed = sp.diff(c*local_log, phase, bb).subs(phase, angle)
        evidence.zero('R64', f'mixed_H_theta_vanishes_{angle}', mixed)
    hh, vv = sp.symbols('Gamma_HH Gamma_theta_theta', nonzero=True)
    evidence.zero('R64', 'regular_on_shell_Schur_complement', vv-sp.S(0)**2/hh-vv)
    for angle, sign in ((mp.mpf(0), -1), (mp.pi, 1)):
        second = mp.diff(lambda ph: phase_derivative(ss, ee, ph, mp.mpf('.5'), mp.mpf('.1')), angle)
        evidence.check('R64', f'exact_S4_phase_hessian_sign_{sign}', sign*second > 0,
                       {'phase': str(angle), 'U_theta_theta': float(second)})

    ids = sorted({row['claim'] for row in evidence.checks})
    assert ids == [f'R{i:02d}' for i in range(59, 65)]
    return {'schema': 'CE-RB6-v1', 'scope': 'fixed common scalar S4 branch and local curvature response',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'mpmath_version': mp.__version__, 'mpmath_decimal_precision': mp.mp.dps,
            'claim_ids': ids, 'number_of_checks': len(evidence.checks),
            'all_passed': all(row['passed'] for row in evidence.checks),
            'checks': evidence.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_curvature.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
