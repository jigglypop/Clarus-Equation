"""CE-RB8: spatial stability, thermal curvature and actual Gaussian boundary premises."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.linalg import block_diag, expm

from verify_reverse import Evidence

J2 = np.array([[0., 1.], [-1., 0.]])


def rotation(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def oscillator_map(frequency, duration):
    c, s = np.cos(frequency*duration), np.sin(frequency*duration)
    return np.array([[c, s/frequency], [-frequency*s, c]])


def fixed_covariance(matrix):
    a, b, c, d = matrix.ravel()
    trace_half = (a+d)/2
    assert abs(trace_half) < 1
    return np.sign(b)/(2*np.sqrt(1-trace_half**2))*np.array([[b, (d-a)/2], [(d-a)/2, -c]])


def spectrum(theta, eps=.2):
    angle = (theta+2*np.pi*np.arange(3))/3
    return 1+2*eps*np.cos(angle), -2*eps/3*np.sin(angle), -2*eps/9*np.cos(angle)


def potential(theta, eps=.2):
    x, xp, xpp = spectrum(theta, eps)
    reference = spectrum(np.pi, eps)[0]
    value = (np.sum(x*x*np.log(x))-np.sum(reference**2*np.log(reference)))/(32*np.pi**2)
    first = np.sum(x*xp*np.log(x))/(16*np.pi**2)
    second = np.sum((xp*xp+x*xpp)*np.log(x)+xp*xp)/(16*np.pi**2)
    return value, first, second


def thermal_derivatives(x, temperature):
    def integrand(momentum, order):
        energy = np.sqrt(momentum*momentum+x)
        argument = energy/temperature
        if argument > 700:
            return 0.
        n = 1/np.expm1(argument)
        if order == 1:
            value = n/(2*energy)
        elif order == 2:
            value = -n/(4*energy**3)-n*(1+n)/(4*temperature*energy**2)
        else:
            value = 3*n/(8*energy**5)+3*n*(1+n)/(8*temperature*energy**4)
            value += n*(1+n)*(1+2*n)/(8*temperature**2*energy**3)
        return momentum**2/(2*np.pi**2)*value
    return [quad(lambda momentum: integrand(momentum, order), 0, np.inf,
                 epsabs=1e-16, epsrel=2e-11)[0] for order in (1, 2, 3)]


def ermakov_field(state):
    theta, widths = state[0], state[1:4]
    momenta = state[4:]
    x, xp, xpp = spectrum(theta)
    value, first, second = potential(theta)
    gradient = np.r_[first+np.sum(xp*(widths**2-x**(-.5)))/2, -widths**(-3)+x*widths]
    hessian = np.zeros((4, 4))
    hessian[0, 0] = second+np.sum(xpp*(widths**2-x**(-.5))+.5*xp*xp*x**(-1.5))/2
    hessian[0, 1:] = hessian[1:, 0] = xp*widths
    hessian[1:, 1:] = np.diag(3*widths**(-4)+x)
    linearization = np.block([[np.zeros((4, 4)), np.eye(4)], [-hessian, np.zeros((4, 4))]])
    energy = np.sum(momenta**2)/2+value+np.sum(widths**(-2)+x*widths**2-2*np.sqrt(x))/2
    return np.r_[momenta, -gradient], linearization, energy


def run():
    e = Evidence()
    values = {}
    wave, mass, coupling, sound = sp.symbols('k m C cs', positive=True)
    matrix = sp.Matrix([[wave**2+mass**2, sp.sqrt(coupling)*wave],
                        [sp.sqrt(coupling)*wave, sound**2*wave**2]])
    e.zero('R71', 'sound_speed_stability_determinant',
           matrix.det()-wave**2*(sound**2*(wave**2+mass**2)-coupling))
    w2 = sp.symbols('omega_squared')
    e.zero('R71', 'cold_dispersion_polynomial',
           (matrix.subs(sound, 0)-w2*sp.eye(2)).det()-(w2**2-(wave**2+mass**2)*w2-coupling*wave**2))
    for kk in (.01, .5, 2., 30.):
        cc, mm = .7, 1.3
        a = kk*kk+mm*mm
        gamma2 = 2*cc*kk*kk/(np.sqrt(a*a+4*cc*kk*kk)+a)
        eig = np.linalg.eigvalsh([[a, np.sqrt(cc)*kk], [np.sqrt(cc)*kk, 0]])
        e.close('R71', f'growth_branch_{kk}', gamma2, -eig[0], 1e-12)
        e.check('R71', f'bounded_nonzero_cold_growth_{kk}', 0 < gamma2 < cc,
                {'growth_squared': gamma2, 'upper_bound': cc})
        for damping in (0., 2., 20.):
            ksym = np.array([[a, np.sqrt(cc)*kk], [np.sqrt(cc)*kk, 0.]])
            dynamical = np.block([[np.zeros((2, 2)), np.eye(2)], [-ksym, -np.diag([3*damping, 2*damping])]])
            maximum = float(np.max(np.linalg.eigvals(dynamical).real))
            e.check('R71', f'friction_does_not_remove_frozen_growth_{kk}_{damping}', maximum > 0,
                    {'positive_real_eigenvalue': maximum})
    first, second = 1., 2.
    quarter_product = oscillator_map(second, np.pi/(2*second))@oscillator_map(first, np.pi/(2*first))
    e.close('R71', 'positive_instantaneous_stiffness_parametric_counterexample',
            quarter_product, np.diag([-first/second, -second/first]), 1e-14)
    e.check('R71', 'parametric_spectral_radius_above_one', np.max(abs(np.linalg.eigvals(quarter_product))) > 1.9,
            {'Floquet_multipliers': np.linalg.eigvals(quarter_product).tolist()})

    # R72: strict negative curvature for all finite Gibbs temperatures and the free Bose gas.
    ratio, zz = sp.symbols('q z', positive=True)
    bracket = (3-ratio**2)/2-(ratio+1)*zz/2-sp.exp(-zz)/ratio
    e.zero('R72', 'Gibbs_zero_z_factorization', bracket.subs(zz, 0)+(ratio-1)**2*(ratio+2)/(2*ratio))
    e.zero('R72', 'Gibbs_derivative_upper_bound',
           (-(ratio+1)/2+1/ratio)+(ratio-1)*(ratio+2)/(2*ratio))
    for eps in (.15, .35):
        mh, ml = np.sqrt(1+2*eps), np.sqrt(1-eps)
        for temperature in (.01, .1, 1., 10.):
            qq, z = mh/ml, (mh-ml)/temperature
            p_light = 1/(2+np.exp(-z))
            curvature = p_light*eps/(9*ml)*((3-qq*qq)/2-(qq+1)*z/2-np.exp(-z)/qq)
            original = (np.exp(-z)*(-eps/(9*mh))
                        +2*(eps/(18*ml)-eps**2/(12*ml**3))
                        -eps**2/(6*ml**2*temperature))/(2+np.exp(-z))
            e.close('R72', f'Gibbs_exact_reduction_{eps}_{temperature}', curvature, original, 1e-13)
            e.check('R72', f'Gibbs_negative_all_sampled_T_{eps}_{temperature}', curvature < 0,
                    {'curvature': float(curvature)})
        for temperature in (.1, .5, 2.):
            low = thermal_derivatives(1-eps, temperature)
            high = thermal_derivatives(1+2*eps, temperature)
            curvature = 2*eps/9*(low[0]-high[0])+2*eps*eps/3*low[1]
            e.check('R72', f'Bose_derivative_signs_{eps}_{temperature}', low[0] > 0 and low[1] < 0 and low[2] > 0,
                    {'f_prime': low[0], 'f_second': low[1], 'f_third': low[2]})
            e.check('R72', f'full_momentum_Bose_curvature_negative_{eps}_{temperature}', curvature < 0,
                    {'thermal_curvature': curvature})

    # R73: the actual pure elliptic condition holds; parabolic boundaries are singular.
    rng = np.random.default_rng(20260923)
    for trial in range(5):
        sym = rng.normal(size=(2, 2)); sym = (sym+sym.T)/2
        conjugator = expm(J2@sym)
        angle = rng.uniform(.2, 2.8)
        transfer = conjugator@rotation(angle)@np.linalg.inv(conjugator)
        covariance = fixed_covariance(transfer)
        e.close('R73', f'elliptic_fixed_covariance_{trial}', transfer@covariance@transfer.T, covariance, 2e-11)
        e.close('R73', f'elliptic_purity_{trial}', np.linalg.det(covariance), .25, 2e-11)
        e.close('R73', f'independent_conjugation_covariance_{trial}', covariance, conjugator@conjugator.T/2, 2e-11)
    qv, cv, pv = sp.symbols('Q C P', real=True)
    sig = sp.Matrix([[qv, cv], [cv, pv]])
    shear = sp.Matrix([[1, 1], [0, 1]])
    solution = sp.solve(list(shear*sig*shear.T-sig), [qv, cv, pv], dict=True)
    e.check('R73', 'nontrivial_parabolic_has_no_positive_covariance', solution == [{cv: 0, pv: 0}],
            {'required': {str(k): str(v) for k, v in solution[0].items()}})
    energies = []
    for delta in (.1, .001, .00001):
        transfer = np.array([[1., 1.], [-delta, 1-delta]])
        covariance = fixed_covariance(transfer)
        predicted_q = 1/(2*np.sqrt(delta-delta*delta/4))
        e.close('R73', f'near_parabolic_width_{delta}', covariance[0, 0], predicted_q, 1e-8)
        energies.append(float(np.trace(covariance)/2))
    e.check('R73', 'pure_boundary_energy_unbounded_sequence', energies[0] < energies[1] < energies[2],
            {'reference_oscillator_energies': energies})

    # R74: same literal CE spectrum, same period and energy, different globally pure states.
    eps = .2
    masses, xp, _ = spectrum(np.pi, eps)
    wh, wl = np.sqrt(1+eps), np.sqrt(1-2*eps)
    period = 2*np.pi/(wh+wl)
    transfer = block_diag(oscillator_map(wh, period), oscillator_map(wl, period), oscillator_map(wh, period))
    j6 = block_diag(J2, J2, J2)
    # Input L,B,D; output heavy0,light,heavy2. B,D are bright/dark heavy combinations.
    mix = np.zeros((6, 6))
    mix[0:2, 2:4] = mix[0:2, 4:6] = np.eye(2)/np.sqrt(2)
    mix[2:4, 0:2] = np.eye(2)
    mix[4:6, 2:4] = np.eye(2)/np.sqrt(2)
    mix[4:6, 4:6] = -np.eye(2)/np.sqrt(2)
    scaling = np.diag([1/np.sqrt(wh), np.sqrt(wh), 1/np.sqrt(wl), np.sqrt(wl), 1/np.sqrt(wh), np.sqrt(wh)])
    squeeze = .6
    covariances, energies = [], []
    for phase in (0., .7):
        cov = np.eye(6)/2
        cov[:4, :4] = np.cosh(2*squeeze)*np.eye(4)/2
        cross = np.sinh(2*squeeze)/2*rotation(phase)@np.diag([1., -1.])
        cov[:2, 2:4], cov[2:4, :2] = cross, cross.T
        cov = scaling@mix@cov@mix.T@scaling
        covariances.append(cov)
        energy = .5*np.trace(np.diag([wh*wh, 1, wl*wl, 1, wh*wh, 1])@cov)
        energies.append(float(energy))
        e.close('R74', f'global_purity_{phase}', cov@j6@cov, j6/4, 2e-12)
        e.close('R74', f'CE_periodic_covariance_{phase}', transfer@cov@transfer.T, cov, 2e-12)
        e.close('R74', f'CE_mean_phase_force_zero_{phase}', np.dot(xp, np.diag(cov)[::2]), 0., 2e-12)
        e.check('R74', f'positive_global_covariance_{phase}', np.linalg.eigvalsh(cov)[0] > 0,
                {'minimum_eigenvalue': float(np.linalg.eigvalsh(cov)[0]), 'light_reduced_determinant': float(np.linalg.det(cov[2:4, 2:4]))})
    e.close('R74', 'same_energy_different_correlations', energies[0], energies[1], 1e-12)
    e.check('R74', 'globally_pure_periodic_state_not_unique', np.linalg.norm(covariances[0]-covariances[1]) > .1,
            {'covariance_difference_norm': float(np.linalg.norm(covariances[0]-covariances[1])), 'period': period, 'energy': energies[0]})
    values['multimode_boundary_counterexample'] = {'s': 1., 'eps': eps, 'theta': float(np.pi),
        'period': period, 'global_squeeze': squeeze, 'correlation_phases': [0., .7],
        'same_energy': energies[0], 'modewise_purity_assumed': False}

    # R75: source29's Hamiltonian is bounded and volume preserving, not an attracting reset.
    j8 = np.block([[np.zeros((4, 4)), np.eye(4)], [-np.eye(4), np.zeros((4, 4))]])
    initial_theta = .4
    widths = spectrum(initial_theta)[0]**(-.25)*np.array([1.03, .98, 1.02])
    initial = np.r_[initial_theta, widths, .12, .01, -.015, .005]
    field, linearization, energy = ermakov_field(initial)
    e.close('R75', 'Hamiltonian_flow_zero_divergence', np.trace(linearization), 0., 1e-14)
    e.close('R75', 'Hamiltonian_linearization_symplectic', linearization.T@j8+j8@linearization, 0., 1e-14)
    def rhs(t, state):
        flow, jac, _ = ermakov_field(state[:8])
        return np.r_[flow, (jac@state[8:].reshape(8, 8)).ravel()]
    result = solve_ivp(rhs, (0, 8), np.r_[initial, np.eye(8).ravel()], method='DOP853',
                       rtol=2e-12, atol=2e-14, max_step=.025, t_eval=np.linspace(0, 8, 101))
    assert result.success
    trajectory_energies = np.array([ermakov_field(y)[2] for y in result.y[:8].T])
    e.close('R75', 'joint_Ermakov_energy_conserved', trajectory_energies, energy, 2e-10)
    final_map = result.y[8:, -1].reshape(8, 8)
    e.close('R75', 'joint_variational_map_symplectic', final_map.T@j8@final_map, j8, 2e-10)
    e.close('R75', 'joint_phase_volume_preserved', np.linalg.det(final_map), 1., 2e-10)
    bound = 2*energy+2*np.sqrt(1+2*.2)
    e.check('R75', 'widths_bounded_away_from_singularities',
            np.min(result.y[1:4]) >= 1/np.sqrt(bound) and np.max(result.y[1:4]) <= np.sqrt(bound/(1-2*.2)),
            {'minimum_b': float(np.min(result.y[1:4])), 'maximum_b': float(np.max(result.y[1:4])),
             'proven_lower': 1/np.sqrt(bound), 'proven_upper': np.sqrt(bound/(1-2*.2))})

    # R76: Floquet response at fixed time, then the actual missing terms if time/boundary vary.
    tau = 1.7
    def mode_map(source):
        def flow(t, y):
            k = np.diag([1.2+.2*np.cos(t)+source, 1.])
            return (J2@k@y.reshape(2, 2)).ravel()
        return solve_ivp(flow, (0, tau), np.eye(2).ravel(), method='DOP853',
                         rtol=2e-13, atol=2e-15, max_step=.02, dense_output=True)
    source, step = .15, 1e-5
    trajectory = mode_map(source)
    transfer = trajectory.y[:, -1].reshape(2, 2)
    covariance = fixed_covariance(transfer)
    def response(t):
        s = trajectory.sol(t).reshape(2, 2)
        return (s@covariance@s.T)[0, 0]/2
    rhs_response = quad(response, 0, tau, epsabs=1e-12, epsrel=1e-12)[0]
    angles = [np.arccos(np.trace(mode_map(source+sign*step).y[:, -1].reshape(2, 2))/2) for sign in (-1, 1)]
    lhs_response = (angles[1]-angles[0])/(4*step)
    e.close('R76', 'fixed_domain_Floquet_Hellmann_Feynman', lhs_response, rhs_response, 2e-9)
    source, angle = 1.3, 1.2
    duration, duration_prime = angle/source, -angle/source**2
    fixed_domain_term, moving_time_term = duration/2, source*duration_prime/2
    e.close('R76', 'moving_period_requires_endpoint_energy', fixed_domain_term+moving_time_term, 0., 1e-14)
    e.check('R76', 'omitting_endpoint_gives_wrong_response', fixed_domain_term > .1,
            {'exact_phase_derivative': 0., 'fixed_domain_term_alone': fixed_domain_term, 'endpoint_term': moving_time_term})
    tau, kick = 1., .2
    def kicked(c):
        return np.array([[1., 0.], [c, 1.]])@rotation(tau)
    cov = fixed_covariance(kicked(kick))
    bprime = np.array([[0., 0.], [1., 0.]])
    boundary = np.array([[1., 0.], [kick, 1.]])
    generator = -J2@bprime@np.linalg.inv(boundary)
    boundary_term = np.trace(cov@generator)/2
    fd = (np.arccos(np.trace(kicked(kick+step))/2)-np.arccos(np.trace(kicked(kick-step))/2))/(4*step)
    e.close('R76', 'parameter_dependent_boundary_term', fd, boundary_term, 2e-9)
    e.close('R76', 'boundary_generator_is_symmetric', generator, generator.T, 1e-14)

    ids = sorted({row['claim'] for row in e.checks})
    assert ids == [f'R{i:02d}' for i in range(71, 77)]
    return {'schema': 'CE-RB8-v1', 'scope': 'fixed density, Gibbs/Bose and Gaussian periodic boundary claims',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(x['passed'] for x in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_stability.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
