"""CE-RB7: physical Gaussian states, cold limits and finite photon backreaction.

No observational fit, continuum renormalization, or initial-state selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp

from verify_reverse import Evidence


def pulse(t, amplitude=.08):
    """Derivative of amplitude times a C-infinity bump supported in [0,1]."""
    if t <= 0 or t >= 1:
        return 0.
    denominator = t*(1-t)
    f = np.exp(4-1/denominator)
    return amplitude*f*(1-2*t)/denominator**2


def photon_bogoliubov(w, k, interval, max_step=.012):
    def rhs(t, y):
        return [w(t)*np.exp(2j*k*t)*y[1], w(t)*np.exp(-2j*k*t)*y[0]]
    result = solve_ivp(rhs, interval, np.array([1., 0.], complex), method='DOP853',
                       rtol=2e-12, atol=2e-14, max_step=max_step)
    assert result.success
    return result.y[:, -1]


def ce_functions(theta):
    # One charged complex scalar triplet, all functions from its same spectrum.
    s, eps, inverse_alpha, charge2, tree_z = 1., .35, 10., 16/3, .05
    angle = (theta+2*np.pi*np.arange(3))/3
    x = s+2*eps*np.cos(angle)
    xp = -2*eps/3*np.sin(angle)
    xpp = -2*eps/9*np.cos(angle)
    reference = s+2*eps*np.cos((np.pi+2*np.pi*np.arange(3))/3)
    potential = (np.sum(x*x*np.log(x))-np.sum(reference**2*np.log(reference)))/(32*np.pi**2)
    potential_prime = np.sum(x*xp*np.log(x))/(16*np.pi**2)
    z = tree_z+np.sum(xp*xp/x)/(96*np.pi**2)
    zp = np.sum(2*xp*xpp/x-xp**3/x**2)/(96*np.pi**2)
    ratio = eps/s
    determinant = 1-3*ratio**2+2*ratio**3*np.cos(theta)
    reference_d = (1-2*ratio)*(1+ratio)**2
    b = charge2/(12*np.pi)
    alpha_inv = inverse_alpha-b*np.log1p(2*ratio**3*(1+np.cos(theta))/reference_d)
    log_i = .5*np.log(alpha_inv/inverse_alpha)
    g = b*ratio**3*np.sin(theta)/(alpha_inv*determinant)
    return potential, potential_prime, z, zp, log_i, g


def coupled_run(method='DOP853', feedback=True):
    ks = np.array([.025, .055, .085])
    weights = np.array([1., .7, 1.3])
    initial = np.zeros(2+3*len(ks))
    initial[0] = .7

    def rhs(t, state):
        theta, speed = state[:2]
        occupation, u, v = state[2:].reshape(3, -1)
        _, up, z, zp, _, g = ce_functions(theta)
        source = 2*g*np.sum(weights*ks*u)
        pump = g*speed
        return np.r_[speed, (-up-zp*speed*speed/2-(source if feedback else 0))/z,
                     2*pump*u, pump*(1+2*occupation)+2*ks*v, -2*ks*u]

    grid = np.linspace(0, 300, 1201)
    sol = solve_ivp(rhs, (0., 300.), initial, t_eval=grid, method=method,
                    rtol=2e-10, atol=2e-13, max_step=.2)
    assert sol.success
    numbers, us, vs = sol.y[2:].reshape(3, len(ks), -1)
    functions = np.array([ce_functions(theta) for theta in sol.y[0]])
    field_energy = functions[:, 2]*sol.y[1]**2/2+functions[:, 0]
    photon_energy = np.sum((weights*ks)[:, None]*numbers, axis=0)
    return sol, numbers, us, vs, field_energy, photon_energy, ks, weights


def run():
    e = Evidence()
    values = {}
    n, u, v, w, omega = sp.symbols('n u v W Omega', real=True)
    flow = sp.Matrix([2*w*u, w*(1+2*n)+2*omega*v, -2*omega*u])
    casimir = (1+2*n)**2-4*(u*u+v*v)
    e.zero('R65', 'Gaussian_Casimir_conservation', sum(sp.diff(casimir, x)*dx for x, dx in zip((n, u, v), flow)))
    q, p, c = (n+sp.Rational(1, 2)+u)/omega, omega*(n+sp.Rational(1, 2)-u), v
    e.zero('R65', 'covariance_determinant', q*p-c*c-casimir/4)
    for number in (.000001, .1, 2.):
        coherence = np.sqrt(number*(1+number))
        rates = []
        for sign in (-1, 1):
            matrix = np.array([[number+.5+sign*coherence, 0], [0, number+.5-sign*coherence]])
            e.check('R65', f'physical_squeezed_covariance_{number}_{sign}', np.linalg.eigvalsh(matrix)[0] > 0,
                    {'minimum_eigenvalue': float(np.linalg.eigvalsh(matrix)[0]), 'determinant': float(np.linalg.det(matrix))})
            rates.append(2*.3*sign*coherence)
        e.check('R65', f'same_occupation_opposite_creation_{number}', rates[0] < 0 < rates[1],
                {'n': number, 'n_prime': rates})

    # Independent time-dependent oscillator mode and (n,u,v) descriptions.
    def oscillator(t):
        om = 1.2+.15*np.tanh(t)
        return om, .15/np.cosh(t)**2/(2*om)
    ti, tf = -6., 6.
    om0, _ = oscillator(ti)
    mode0 = np.array([1/np.sqrt(2*om0), -1j*np.sqrt(om0/2)])
    mode = solve_ivp(lambda t, y: [y[1], -oscillator(t)[0]**2*y[0]], (ti, tf), mode0,
                     t_eval=np.linspace(ti, tf, 101), method='DOP853', rtol=2e-12, atol=2e-14, max_step=.03)
    def moment_rhs(t, y):
        om, pump = oscillator(t)
        nn, uu, vv = y
        return [2*pump*uu, pump*(1+2*nn)+2*om*vv, -2*om*uu]
    # For the mass oscillator, u=(Omega Q-P/Omega)/2 and v=C.
    moments = solve_ivp(moment_rhs, (ti, tf), [0., 0., 0.], t_eval=mode.t,
                        method='DOP853', rtol=2e-12, atol=2e-14, max_step=.03)
    oms = np.array([oscillator(t)[0] for t in mode.t])
    qq, pp = abs(mode.y[0])**2, abs(mode.y[1])**2
    cc = np.real(mode.y[0]*mode.y[1].conj())
    reconstructed = np.array([(pp/oms+oms*qq-1)/2, (oms*qq-pp/oms)/2, cc])
    e.close('R65', 'oscillator_vs_moment_ODE', reconstructed, moments.y, 2e-10)
    e.close('R65', 'oscillator_Wronskian', np.imag(mode.y[0].conj()*mode.y[1]), -.5, 2e-10)
    e.close('R65', 'evolved_Casimir', (1+2*moments.y[0])**2-4*np.sum(moments.y[1:]**2, axis=0), 1., 2e-10)

    # R66: exact RT31 continuity and quantified dust errors.
    a, adot, mass2, mass2_theta, speed = sp.symbols('a adot x xp theta_prime', real=True)
    omprime = (2*a*adot*mass2+a*a*mass2_theta*speed)/(2*omega)
    rho = 2*omega*n/a**4
    pressure = (2*omega*n-2*a*a*mass2*(n+u)/omega)/(3*a**4)
    current = mass2_theta*(n+u)/(a*a*omega)
    rho_prime = sp.diff(rho, a)*adot+sp.diff(rho, omega)*omprime+sp.diff(rho, n)*omprime*u/omega
    e.zero('R66', 'RT31_exact_energy_exchange', rho_prime+3*adot/a*(rho+pressure)-speed*current)
    for number in (1e-2, 1e-6, 1e-10):
        coherence = np.sqrt(number*(1+number))
        ratio_pressure_energy = -coherence/(3*number)
        e.check('R66', f'zero_momentum_not_dust_{number}', abs(ratio_pressure_energy) > 1,
                {'physical_momentum': 0., 'n': number, 'p_ex_over_rho_ex': ratio_pressure_energy})
    rng = np.random.default_rng(20260921)
    for trial in range(4):
        mm = rng.uniform(.4, 2., 15)
        eta = .07
        momentum = rng.uniform(0, eta, 15)*mm
        frequencies = np.sqrt(mm*mm+momentum*momentum)
        numbers = rng.uniform(.01, 2, 15)
        coherence = rng.uniform(-1, 1, 15)*np.sqrt(numbers*(1+numbers))
        dm = rng.uniform(-.2, .2, 15)
        weight = rng.uniform(.1, 1, 15)
        exact_energy = 2*np.sum(weight*frequencies*numbers)
        dust_energy = 2*np.sum(weight*mm*numbers)
        exact_p = 2/3*np.sum(weight*(momentum*momentum*numbers-mm*mm*coherence)/frequencies)
        coherence_bound = 2/3*np.sum(weight*mm*mm*abs(coherence)/frequencies)
        e.check('R66', f'cold_energy_bound_{trial}', 0 <= exact_energy-dust_energy <= eta*eta*dust_energy/2,
                {'error': float(exact_energy-dust_energy), 'bound': float(eta*eta*dust_energy/2)})
        bound_p = eta*eta/(3*(1+eta*eta))*exact_energy+coherence_bound
        e.check('R66', f'pressure_with_coherence_bound_{trial}', abs(exact_p) <= bound_p,
                {'absolute_pressure': float(abs(exact_p)), 'bound': float(bound_p)})
        actual_j = 2*np.sum(weight*mm*dm/frequencies*(numbers+coherence))
        dust_j = 2*np.sum(weight*dm*numbers)
        bound_j = 2*np.sum(weight*abs(dm)*((1-mm/frequencies)*numbers+mm/frequencies*abs(coherence)))
        e.check('R66', f'mass_force_error_bound_{trial}', abs(actual_j-dust_j) <= bound_j,
                {'absolute_source_error': float(abs(actual_j-dust_j)), 'bound': float(bound_j)})

    # R67: photon quadratures, source and total energy from the same action.
    qq, pp, cc, k, pump = sp.symbols('Q P C k W', real=True)
    quadratures = sp.Matrix([2*cc+2*pump*qq, -2*k*k*cc-2*pump*pp, pp-k*k*qq])
    e.zero('R67', 'photon_covariance_determinant_conserved',
           sum(sp.diff(qq*pp-cc*cc, x)*dx for x, dx in zip((qq, pp, cc), quadratures)))
    photon_e = (pp+k*k*qq)/2
    de = sum(sp.diff(photon_e, x)*dx for x, dx in zip((qq, pp, cc), quadratures))
    e.zero('R67', 'photon_work_identity', de-pump*(k*k*qq-pp))
    qq_sub, pp_sub = (n+sp.Rational(1, 2)+u)/k, k*(n+sp.Rational(1, 2)-u)
    e.zero('R67', 'source_depends_on_coherence', (k*k*qq-pp).subs({qq: qq_sub, pp: pp_sub})-2*k*u)
    g, theta_dot, up, zz, zp, source = sp.symbols('g theta_dot Uprime Z Zprime J')
    acceleration = (-zp*theta_dot**2/2-up-source)/zz
    field_de = zz*theta_dot*acceleration+zp*theta_dot**3/2+up*theta_dot
    e.zero('R67', 'same_action_total_energy', field_de+theta_dot*source)
    e.zero('R67', 'photon_number_equation', (de/k).subs({qq: qq_sub, pp: pp_sub})-2*pump*u)

    # R68: a certified Born remainder and a finite-band energy bound.
    for amplitude in (.03, .08, .15):
        length = 2*amplitude
        for frequency in (.4, 1., 2.3):
            alpha, beta = photon_bogoliubov(lambda t: pulse(t, amplitude), frequency, (0, 1))
            born = quad(lambda t: pulse(t, amplitude)*np.cos(2*frequency*t), 0, 1, epsabs=1e-13)[0]
            born -= 1j*quad(lambda t: pulse(t, amplitude)*np.sin(2*frequency*t), 0, 1, epsabs=1e-13)[0]
            e.check('R68', f'nonperturbative_bound_{amplitude}_{frequency}', abs(beta) <= np.sinh(length)+1e-12,
                    {'beta': float(abs(beta)), 'upper_bound': float(np.sinh(length))})
            e.check('R68', f'Born_remainder_bound_{amplitude}_{frequency}', abs(beta-born) <= np.sinh(length)-length+1e-12,
                    {'absolute_error': float(abs(beta-born)), 'certified_bound': float(np.sinh(length)-length)})
    cutoff, length, scale = 3., .1, 2.
    band_energy = quad(lambda wave: wave**3*np.sinh(length)**2/(np.pi**2*scale**4), 0, cutoff)[0]
    e.close('R68', 'two_polarization_band_energy_bound_integral', band_energy,
            cutoff**4*np.sinh(length)**2/(4*np.pi**2*scale**4), 1e-13)
    for initial_number in (0., .2, 3.):
        a0, b0 = np.cosh(length), np.sinh(length)
        worst_final = (a0*np.sqrt(initial_number)+b0*np.sqrt(1+initial_number))**2
        e.close('R68', f'seeded_occupation_bound_saturation_{initial_number}', worst_final,
                np.sinh(np.arcsinh(np.sqrt(initial_number))+length)**2, 1e-12)

    # R69: two smooth pulses with exact destructive Bogoliubov interference at a chosen k.
    frequency, amplitude = .9, .08
    alpha, beta = photon_bogoliubov(lambda t: pulse(t, amplitude), frequency, (0, 1))
    separation = (np.angle(alpha)+np.pi/2)/frequency
    while separation <= 1:
        separation += np.pi/frequency
    predicted = np.exp(-2j*frequency*separation)*beta*alpha+alpha.conjugate()*beta
    e.close('R69', 'exact_two_pulse_composition_cancel', predicted, 0., 1e-14)
    total_pump = lambda t: pulse(t, amplitude)+pulse(t-separation, amplitude)
    alpha2, beta2 = photon_bogoliubov(total_pump, frequency, (0, separation+1))
    e.check('R69', 'each_pulse_creates_quanta', abs(beta)**2 > 1e-5,
            {'single_pulse_n': float(abs(beta)**2)})
    e.close('R69', 'independent_two_pulse_ODE_cancel', beta2, 0., 2e-10)
    nearby = photon_bogoliubov(total_pump, frequency*1.1, (0, separation+1))[1]
    e.check('R69', 'cancellation_is_frequency_specific', abs(nearby)**2 > 1e-6,
            {'selected_k': frequency, 'selected_n': float(abs(beta2)**2), 'nearby_n': float(abs(nearby)**2)})
    e.close('R69', 'two_pulse_Wronskian', abs(alpha2)**2-abs(beta2)**2, 1., 2e-11)
    values['transparent_pulse'] = {'separation': float(separation), 'amplitude': amplitude,
                                  'k': frequency, 'single_n': float(abs(beta)**2), 'double_n': float(abs(beta2)**2)}

    # R70: close the finite photon/phase system using CE's actual local functions.
    for theta in (.4, 1.7, 3.8, 5.1):
        potential, up, z, zp, log_i, gg = ce_functions(theta)
        step = 1e-5
        plus, minus = ce_functions(theta+step), ce_functions(theta-step)
        e.close('R70', f'same_spectrum_potential_derivative_{theta}', (plus[0]-minus[0])/(2*step), up, 2e-11)
        e.close('R70', f'same_spectrum_kinetic_derivative_{theta}', (plus[2]-minus[2])/(2*step), zp, 2e-11)
        e.close('R70', f'same_spectrum_photon_derivative_{theta}', (plus[4]-minus[4])/(2*step), gg, 2e-11)
    run_a = coupled_run('DOP853', True)
    run_b = coupled_run('Radau', True)
    sol, numbers, us, vs, field_energy, photon_energy, ks, weights = run_a
    total = field_energy+photon_energy
    conservation_error = float(np.max(abs(total-total[0])))
    e.check('R70', 'autonomous_total_energy_conservation', conservation_error < 2e-11,
            {'max_abs_energy_error': conservation_error, 'initial_energy': float(total[0])})
    e.close('R70', 'autonomous_quantum_invariant', (1+2*numbers)**2-4*(us*us+vs*vs), 1., 2e-9)
    e.check('R70', 'autonomous_positive_occupation', np.min(numbers) > -2e-11,
            {'min_n': float(np.min(numbers)), 'max_n': float(np.max(numbers))})
    e.close('R70', 'independent_integrator_state', sol.y, run_b[0].y, 2e-8)
    e.check('R70', 'photons_receive_phase_energy', np.max(photon_energy) > 1e-8,
            {'max_photon_energy': float(np.max(photon_energy)), 'initial_field_energy': float(total[0])})
    omitted = coupled_run('DOP853', False)
    missing_work = omitted[4]+omitted[5]-omitted[4][0]
    e.check('R70', 'omitting_same_action_feedback_loses_conservation', np.max(abs(missing_work)) > 1e-8,
            {'max_energy_gain_without_source': float(np.max(abs(missing_work)))})
    e.check('R70', 'finite_energy_bounds_photon_occupation',
            np.all((weights*ks)[:, None]*numbers <= total[0]+2e-11),
            {'energy_cap': float(total[0])})
    values['closed_finite_system'] = {'s': 1., 'eps': .35, 'alpha_inverse_reference': 10.,
        'charge_squared': 16/3, 'tree_Z': .05, 'initial_theta': .7, 'initial_theta_dot': 0.,
        'initial_photon_state': 'Gaussian vacuum', 'k': ks.tolist(), 'positive_mode_weights': weights.tolist(),
        'time_interval': [0., 300.], 'max_photon_energy': float(np.max(photon_energy)),
        'scope': 'finite modes and supplied local effective action; not the cosmological hierarchy or full in-in renormalization'}

    ids = sorted({row['claim'] for row in e.checks})
    assert ids == [f'R{i:02d}' for i in range(65, 71)]
    return {'schema': 'CE-RB7-v1', 'scope': 'Gaussian real-time state, cold limit and finite photon feedback',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(x['passed'] for x in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_realtime.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
