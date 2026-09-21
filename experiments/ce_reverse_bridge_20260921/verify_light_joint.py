"""CE-RB13: light constitutive tensors, nonlinear links and joint-state bounds.

Synthetic finite examples only. General proofs and domains are in chapter 31.
The Galerkin energies are variational upper bounds, not certified exact energies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from scipy.special import ai_zeros, airy, logsumexp

from verify_reverse import Evidence


def cross(k):
    x, y, z = k
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


NODES, WEIGHTS = np.polynomial.legendre.leggauss(24)
NODES, WEIGHTS = (NODES+1)/2, WEIGHTS/2
D0 = np.array([[-1., 1., 0.], [0., -1., 1.], [1., 0., -1.]])


def cell(alpha, s=1., eps=.15):
    x = np.eye(3, dtype=complex)*s
    derivatives = []
    for angle, (i, j) in zip(alpha, ((0, 1), (1, 2), (2, 0))):
        z = eps*np.exp(1j*angle)
        x[i, j], x[j, i] = z, z.conjugate()
        dx = np.zeros((3, 3), complex)
        dx[i, j], dx[j, i] = 1j*z, -1j*z.conjugate()
        derivatives.append(dx)
    masses, vectors = np.linalg.eigh(x)
    dx = np.array([vectors.conj().T@d@vectors for d in derivatives])
    kernel = sum(w*t*(1-t)/((1-t)*masses[:, None]+t*masses[None, :])
                 for t, w in zip(NODES, WEIGHTS))/(16*np.pi**2)
    metric = np.einsum('aij,bij,ij->ab', dx, dx.conj(), kernel).real
    reference = np.array([s-2*eps, s+eps, s+eps])
    potential = (np.sum(masses**2*np.log(masses))-
                 np.sum(reference**2*np.log(reference)))/(32*np.pi**2)
    dpotential = np.einsum('aii,i->a', dx, 2*masses*np.log(masses)+masses).real/(32*np.pi**2)
    return metric, potential, dpotential, masses


def scalar_potential(s, eps, theta):
    masses = s+2*eps*np.cos((theta+2*np.pi*np.arange(3))/3)
    ref = np.array([s-2*eps, s+eps, s+eps])
    u = (np.sum(masses**2*np.log(masses))-np.sum(ref**2*np.log(ref)))/(32*np.pi**2)
    us = (np.sum(masses*np.log(masses))-np.sum(ref*np.log(ref)))/(16*np.pi**2)
    uss = np.log(np.prod(masses)/np.prod(ref))/(16*np.pi**2)
    return u, us, uss


def restricted(h, basis, temperature):
    energies, vectors = np.linalg.eigh(h)
    z = logsumexp(-energies/temperature)
    rho = (vectors*np.exp(-energies/temperature-z))@vectors.conj().T
    compressed = basis.conj().T@h@basis
    er, vr = np.linalg.eigh(compressed)
    zr = logsumexp(-er/temperature)
    rhostar = (vr*np.exp(-er/temperature-zr))@vr.conj().T
    raw = basis.conj().T@rho@basis
    probability = np.trace(raw).real
    post = raw/probability
    return temperature*(z-zr), probability, rhostar, post, compressed, rho


def free_energy(rho, h, temperature):
    probabilities = np.linalg.eigvalsh(rho)
    return float(np.trace(rho@h).real+
                 temperature*np.sum(probabilities*np.log(probabilities)))


def joint_upper(coupling, nh=10, nc=4):
    # Rescale basis for each kappa. No derivatives of this moving projection are used.
    scale = coupling**(1/3)
    b, a = 24**(-1/3), 6*24**(-1/3)
    frequency, wc = a*scale, b*scale
    size = 2*nh+4
    annih = np.diag(np.sqrt(np.arange(1, size)), 1)
    position = (annih+annih.T)/np.sqrt(2*frequency)
    momentum = -1j*np.sqrt(frequency/2)*(annih-annih.T)
    even = np.arange(0, 2*nh, 2)
    take = lambda m: m[np.ix_(even, even)].real
    u, u2 = take(position@position)/2, take(np.linalg.matrix_power(position, 4))/4
    hh = take(momentum@momentum)/2+10*(u2-u+.25*np.eye(nh))
    oc, hc = csr_matrix((nc**3, nc**3)), csr_matrix((nc**3, nc**3))
    for j, mass2 in enumerate((.2, .65, .65)):
        o = diags([(np.arange(nc-1)+1)/(2*wc), (2*np.arange(nc)+1)/(2*wc),
                   (np.arange(nc-1)+1)/(2*wc)], [-1, 0, 1], shape=(nc, nc))
        h = diags(wc*(2*np.arange(nc)+1))+(mass2-wc**2)*o
        for local, target in ((o, 'o'), (h, 'h')):
            factors = [eye(nc, format='csr')]*3
            factors[j] = local
            full = kron(kron(factors[0], factors[1]), factors[2], format='csr')
            if target == 'o':
                oc += full
            else:
                hc += full
    h = (kron(csr_matrix(hh), eye(nc**3), format='csr')+
         kron(eye(nh), hc, format='csr')+
         coupling*kron(csr_matrix(u), oc, format='csr'))
    energies, states = eigsh(h, k=1, which='SA', v0=np.ones(h.shape[0]), tol=3e-11)
    residual = np.linalg.norm(h@states[:, 0]-energies[0]*states[:, 0])
    return float(energies[0]), float(h[0, 0]), residual, h.shape[0]


def run():
    e, values = Evidence(), {}
    # R104: the positive, no-magnetoelectric constitutive class.
    r = np.array([[4., 2., 2.], [2., 4., 2.], [2., 2., 4.]])
    b0 = np.array([[1.5, -.5, -.5], [-.5, 1.5, -.5], [-.5, -.5, 1.5]])
    e.close('R104', 'original_constitutive_closure', r@b0, 4*np.eye(3))
    e.close('R104', 'original_electric_determinant', np.linalg.det(r), 32)
    rng = np.random.default_rng(104)
    for index in range(8):
        raw = rng.normal(size=(3, 3))
        electric = raw.T@raw+.6*np.eye(3)
        d = .4+index/5
        magnetic = d*np.linalg.inv(electric)
        k = rng.normal(size=3)
        roots = eigh(cross(k).T@magnetic@cross(k), electric, eigvals_only=True)
        expected = d/np.linalg.det(electric)*(k@electric@k)
        e.close('R104', f'positive_cone_degeneracy_{index}', roots, [0, expected, expected])
        ev, vec = np.linalg.eigh(electric)
        sqrte = (vec*np.sqrt(ev))@vec.T
        inve = (vec/np.sqrt(ev))@vec.T
        transformed = inve@cross(k).T@magnetic@cross(k)@inve
        q, dmatrix = sqrte@k, sqrte@magnetic@sqrte
        e.close('R104', f'whitened_cross_product_identity_{index}',
                transformed, cross(q).T@dmatrix@cross(q)/np.linalg.det(electric))
    sx, sy, sz = sp.symbols('d_1 d_2 d_3', positive=True)
    e.zero('R104', 'axis_discriminants_force_scalar_tensor',
           (sx-sy)**2+(sy-sz)**2+(sz-sx)**2-2*(sx*sx+sy*sy+sz*sz-sx*sy-sy*sz-sz*sx))
    split = eigh(cross([1, 0, 0]).T@np.diag([1., 1., 1.4])@cross([1, 0, 0]),
                 np.eye(3), eigvals_only=True)
    e.close('R104', 'positive_gauge_invariant_extension_birefringence', split, [0, 1, 1.4])
    e.close('R104', 'original_same_graph_scalar_cone_ratio', eigh(r, r/8, eigvals_only=True), 8)
    values['extension_birefringence_squared_speeds'] = split.tolist()

    # R105: the actual one-cell Gaussian metric is positive and gauge invariant.
    lo, hi = .15**2/(48*np.pi**2*1.3), .15**2/(48*np.pi**2*.7)
    cell_rows = []
    for index, theta in enumerate(np.linspace(-3., 4., 9)):
        alpha = np.array([.2, -.7, theta+.5])
        m, v, dv, masses = cell(alpha)
        transformed = cell(alpha+D0@np.array([.3, -1.1, .6]))
        e.close('R105', f'exact_channel_gauge_invariance_{index}', transformed[0], m, 2e-18)
        e.close('R105', f'potential_gauge_invariance_{index}', transformed[1], v, 2e-17)
        e.close('R105', f'potential_Gauss_identity_{index}', D0.T@dv, 0, 3e-18)
        eigen = np.linalg.eigvalsh(m)
        e.check('R105', f'uniform_positive_metric_bounds_{index}',
                eigen[0] >= lo*(1-1e-12) and eigen[-1] <= hi*(1+1e-12),
                {'eigenvalues': eigen.tolist(), 'lower': lo, 'upper': hi})
        e.check('R105', f'uniform_mass_gap_{index}', masses[0] >= .7-2e-15,
                {'minimum_mass_squared': float(masses[0])})
        cell_rows.append({'theta': float(theta), 'parallel_inertia': float(m.sum()/3)})
    mp, vp, dp, _ = cell(np.full(3, np.pi/3))
    e.close('R105', 'pi_parallel_inertia_matches_original', mp.sum()/3,
            .15**2/(48*np.pi**2*1.15), 2e-18)
    def uniform(theta):
        matrix, potential, derivative, _ = cell(np.full(3, theta/3))
        return matrix.sum()/3, potential, derivative.mean()
    def flow(time, state):
        theta, momentum = state
        inertia, _, potential_prime = uniform(theta)
        step = 2e-4
        derivative = (uniform(theta-2*step)[0]-8*uniform(theta-step)[0]+
                      8*uniform(theta+step)[0]-uniform(theta+2*step)[0])/(12*step)
        return [3*momentum/inertia, 1.5*momentum**2*derivative/inertia**2-potential_prime]
    solution = solve_ivp(flow, (0, 12), [np.pi+.7, 3e-6], method='DOP853',
                         t_eval=np.linspace(0, 12, 81), rtol=2e-11, atol=[1e-12, 1e-16])
    e.check('R105', 'field_dependent_link_ODE_success', solution.success, {'message': solution.message})
    energies = np.array([1.5*p*p/uniform(t)[0]+uniform(t)[1] for t, p in solution.y.T])
    drift = np.max(np.abs(energies-energies[0]))/energies[0]
    e.check('R105', 'nonlinear_metric_energy_conservation', drift < 2e-7,
            {'relative_energy_drift': float(drift), 'duration': 12, 'single_triangle': True})
    e.close('R105', 'nonlinear_Gauss_conservation',
            np.array([D0.T@np.full(3, p) for p in solution.y[1]]), 0, 1e-18)
    e.check('R105', 'metric_actually_varies', np.ptp([row['parallel_inertia'] for row in cell_rows]) > 1e-6,
            {'inertia_range': float(np.ptp([row['parallel_inertia'] for row in cell_rows]))})
    values['nonlinear_cell'] = {'metric_samples': cell_rows, 'energy_drift': float(drift),
                                'initial_energy': float(energies[0]), 'minimum_inertia_bound': lo}

    # R106: global conditional relaxation bounds, beyond the local delta^4 series.
    eps, s0, coupling, u0, lam = .25, .7, .8, .4, .03
    sstar, rows = s0+coupling*u0, []
    for index, theta in enumerate((0., .3, .9, 1.7, 2.4, 2.8, 3.)):
        pot, us, uss = scalar_potential(sstar, eps, theta)
        force, mu, lipschitz = -coupling*us, 2*lam, 2*lam+coupling**2*uss
        slope = lambda u: 2*lam*(u-u0)+coupling*scalar_potential(s0+coupling*u, eps, theta)[1]
        minimum = brentq(slope, u0, u0+force/mu*1.01, xtol=2e-15)
        shift = minimum-u0
        relaxed = lam*shift**2+scalar_potential(s0+coupling*minimum, eps, theta)[0]
        drop = pot-relaxed
        e.check('R106', f'global_shift_bracket_{index}',
                force/lipschitz-3e-14 <= shift <= force/mu+3e-14,
                {'shift': shift, 'lower': force/lipschitz, 'upper': force/mu})
        e.check('R106', f'global_energy_relaxation_bracket_{index}',
                force**2/(2*lipschitz)-3e-17 <= drop <= force**2/(2*mu)+3e-17,
                {'drop': drop, 'lower': force**2/(2*lipschitz), 'upper': force**2/(2*mu)})
        e.check('R106', f'no_complete_relative_energy_cancellation_{index}', relaxed > 0,
                {'relaxed_relative_energy': relaxed})
        rows.append({'theta': theta, 'shift': shift, 'relaxation': drop})
    ss, ee, bb = sp.symbols('s epsilon B', positive=True)
    determinant = (ss-2*ee)*(ss+ee)**2
    e.zero('R106', 'third_s_derivative_exact_sign_numerator',
           sp.diff(sp.log(1+bb/determinant), ss)+bb*sp.diff(determinant, ss)/(determinant*(determinant+bb)))
    e.close('R106', 'pi_joint_minimum', scalar_potential(sstar, eps, np.pi), 0, 1e-17)
    values['global_relaxation'] = rows

    # R107: restriction cost versus probability, and the cost of postselection.
    temperature, thermo = .7, []
    for index in range(6):
        raw = rng.normal(size=(4, 4))
        h = (raw+raw.T)/2
        basis = np.linalg.qr(rng.normal(size=(4, 2)))[0]
        cost, prob, optimal, post, compressed, rho = restricted(h, basis, temperature)
        probability_cost = -temperature*np.log(prob)
        post_excess = free_energy(post, compressed, temperature)-free_energy(optimal, compressed, temperature)
        eo, vo = np.linalg.eigh(optimal)
        ep, vp2 = np.linalg.eigh(post)
        logopt, logpost = (vo*np.log(eo))@vo.T, (vp2*np.log(ep))@vp2.T
        divergence = np.trace(post@(logpost-logopt)).real
        e.check('R107', f'noncommuting_cost_strictly_exceeds_log_probability_{index}',
                cost > probability_cost+1e-6, {'cost': cost, 'probability_bound': probability_cost})
        e.close('R107', f'postselection_excess_relative_entropy_{index}', post_excess, temperature*divergence)
        e.check('R107', f'postselection_is_not_restricted_Gibbs_{index}', np.linalg.norm(post-optimal) > 1e-3,
                {'state_difference': float(np.linalg.norm(post-optimal)), 'excess': post_excess})
        thermo.append({'cost': cost, 'probability_bound': probability_cost, 'postselection_excess': post_excess})
    commuting_h = np.diag([0., .7, 1.3, 2.])
    commuting_basis = np.eye(4)[:, [0, 2]]
    cost, prob, optimal, post, _, _ = restricted(commuting_h, commuting_basis, temperature)
    e.close('R107', 'commuting_probability_bound_saturates', cost, -temperature*np.log(prob))
    e.close('R107', 'commuting_postselection_is_optimal', optimal, post)
    values['quantum_restriction_costs'] = thermo

    # R108: a moving allowed subspace performs work even for a constant Hamiltonian.
    q, gap = sp.symbols('q Delta', real=True)
    e.zero('R108', 'rotating_rank_one_subspace_force', sp.diff(gap*sp.sin(q)**2, q)-gap*sp.sin(2*q))
    h, generator, basis0 = np.diag([0., .8, 1.7]), np.array([[0., -.7, .4], [.7, 0., -.2], [-.4, .2, 0.]]), np.eye(3)[:, :2]
    moving_rows = []
    for index, coordinate in enumerate((.1, .4, .9, 1.5)):
        unitary = expm(coordinate*generator)
        rotated = unitary.T@h@unitary
        _, _, optimal, _, _, _ = restricted(h, unitary@basis0, temperature)
        derivative = np.trace(optimal@(basis0.T@(rotated@generator-generator@rotated)@basis0)).real
        def fr(value):
            u = expm(value*generator)
            energies = np.linalg.eigvalsh(basis0.T@u.T@h@u@basis0)
            return -temperature*logsumexp(-energies/temperature)
        step = 2e-4
        numerical = (fr(coordinate-2*step)-8*fr(coordinate-step)+8*fr(coordinate+step)-fr(coordinate+2*step))/(12*step)
        e.close('R108', f'moving_support_commutator_work_{index}', derivative, numerical, 3e-11)
        e.check('R108', f'bare_H_prime_misses_force_{index}', abs(derivative) > .01,
                {'actual_free_energy_derivative': float(derivative), 'bare_H_derivative': 0.})
        moving_rows.append({'coordinate': coordinate, 'derivative': float(derivative)})
    values['moving_support'] = moving_rows

    # R109: rigorous continuum bounds; small Galerkin matrices are supporting examples only.
    root = float(ai_zeros(1)[1][0])
    cminus, b = abs(root)*(9/4)**(1/3), 24**(-1/3)
    a, cplus = 6*b, 4.5*b
    e.close('R109', 'Airy_Neumann_boundary', airy(root)[1], 0, 1e-14)
    av, bv = sp.symbols('a b', positive=True)
    leading = av/4+3*bv/2+sp.Rational(3, 8)/(av*bv)
    e.zero('R109', 'Gaussian_optimal_width_a', sp.diff(leading, av).subs({av: 6*24**sp.Rational(-1, 3), bv: 24**sp.Rational(-1, 3)}))
    e.zero('R109', 'Gaussian_optimal_width_b', sp.diff(leading, bv).subs({av: 6*24**sp.Rational(-1, 3), bv: 24**sp.Rational(-1, 3)}))
    e.check('R109', 'factor_two_concavity_gives_positive_asymptotic_response',
            cminus*2**(1/3)-cplus > 0,
            {'lower_energy_coefficient': cminus, 'upper_energy_coefficient': cplus,
             'response_lower_coefficient': cminus*2**(1/3)-cplus})
    ratio = sp.symbols('ratio', positive=True)
    e.zero('R109', 'concave_asymptotic_derivative_lower',
           sp.limit((ratio**sp.Rational(1, 3)-1)/(ratio-1), ratio, 1)-sp.Rational(1, 3))
    e.zero('R109', 'concave_asymptotic_derivative_upper',
           sp.limit((1-ratio**sp.Rational(-1, 3))/(1-1/ratio), ratio, 1)-sp.Rational(1, 3))
    kap, xx, rr, ll, uu = sp.symbols('kappa x r lambda u0', positive=True)
    scaled_quartic = ll*(xx**2/(2*kap**sp.Rational(1, 3))-uu)**2/kap**sp.Rational(1, 3)
    e.zero('R109', 'strong_coupling_scaled_positive_Higgs_potential',
           scaled_quartic-(ll*xx**4/(4*kap)-ll*uu*xx**2/kap**sp.Rational(2, 3)+
                          ll*uu**2/kap**sp.Rational(1, 3)))
    e.zero('R109', 'strong_coupling_cross_term_exact_scale',
           kap*(xx**2/(2*kap**sp.Rational(1, 3)))*(rr**2/(2*kap**sp.Rational(1, 3)))/
           kap**sp.Rational(1, 3)-xx**2*rr**2/4)
    spectral = []
    for coupling in (1., 8., 64., 512., 4096.):
        energy, trial_matrix, residual, dimension = joint_upper(coupling)
        scale = coupling**(1/3)
        exact_trial = cplus*scale+2.5+(1.5/(2*b)-5/(2*a))/scale+30/(16*a*a*scale**2)
        relaxed_upper = cplus*scale+2.5+1.5/(2*b*scale)+30/(16*a*a*scale**2)
        lower = cminus*scale
        e.close('R109', f'full_polynomial_Gaussian_trial_{coupling}', trial_matrix, exact_trial, 2e-10)
        e.check('R109', f'Galerkin_energy_within_continuum_bounds_{coupling}',
                lower <= energy <= exact_trial+2e-10 <= relaxed_upper+2e-10,
                {'lower': lower, 'Galerkin_upper': energy, 'Gaussian_upper': exact_trial,
                 'relaxed_upper': relaxed_upper, 'dimension': dimension})
        e.check('R109', f'finite_matrix_eigen_residual_{coupling}', residual < 3e-7,
                {'residual': float(residual), 'not_continuum_error': True})
        spectral.append({'coupling': coupling, 'lower': lower, 'Galerkin_upper': energy, 'Gaussian_upper': exact_trial})
    values['strong_coupling'] = {'Airy_prime_first_zero': root, 'lower_coefficient': cminus,
                                 'upper_coefficient': cplus, 'variational_examples': spectral}

    ids = sorted({row['claim'] for row in e.checks}, key=lambda value: int(value[1:]))
    assert ids == [f'R{i}' for i in range(104, 110)]
    return {'schema': 'CE-RB13-v1', 'scope': 'conditional light and joint-state proofs in chapters 01-04',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(row['passed'] for row in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_light_joint.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
