"""CE-RB11: actual positive-lattice and Gaussian record/preparation branches.

Written general proofs are in chapter 29; no observational validation is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

from verify_reverse import Evidence


def comm(a, b):
    return a@b-b@a


def norm1(a):
    return np.linalg.svd(a, compute_uv=False).sum()


SHIFT = np.roll(np.eye(3), 1, axis=0)


def lattice(s, weights, delta, eps=.25):
    d = np.array(s)+weights.sum(axis=1)
    t = np.kron(weights, np.eye(3)).astype(complex)
    for j, phase in enumerate(delta):
        block = eps*(np.exp(1j*phase/3)*SHIFT+np.exp(-1j*phase/3)*SHIFT.T)
        t[3*j:3*j+3, 3*j:3*j+3] = block
    dd = np.repeat(d, 3)
    k = np.diag(dd)-t
    r = t/np.sqrt(dd[:, None]*dd[None, :])
    return k, r, d


def logdet(k):
    sign, value = np.linalg.slogdet(k)
    assert abs(sign-1) < 1e-10
    return value


def lattice_hessian(s, weights, eps):
    k, _, d = lattice(s, weights, np.zeros(len(s)), eps)
    ki = np.linalg.inv(k)
    derivatives, second = [], []
    for j in range(len(s)):
        first = np.zeros_like(k)
        sec = np.zeros_like(k)
        sl = slice(3*j, 3*j+3)
        first[sl, sl] = -1j*eps*(SHIFT-SHIFT.T)/3
        sec[sl, sl] = eps*(SHIFT+SHIFT.T)/9
        derivatives.append(first)
        second.append(sec)
    h = np.array([[(np.trace(ki@second[i]) if i == j else 0)-
                   np.trace(ki@derivatives[i]@ki@derivatives[j])
                   for j in range(len(s))] for i in range(len(s))]).real
    return h, 2*eps**3/d**3


def ramp_beta(k, xa=.4, xb=1., tau=.6):
    slope = (xb-xa)/tau
    a = k*k+xa
    def rhs(t, y):
        b = a+slope*t
        phase = (2*t/3)*(a+b+np.sqrt(a*b))/(np.sqrt(a)+np.sqrt(b))
        g = slope/(4*b)
        return [g*np.exp(2j*phase)*y[1], g*np.exp(-2j*phase)*y[0]]
    sol = solve_ivp(rhs, [0, tau], [1+0j, 0j], method='DOP853', rtol=2e-11, atol=2e-13)
    assert sol.success
    return sol.y[:, -1]


def ramp_bound(k, xa=.4, xb=1., tau=.6):
    delta = abs(xb-xa)
    a, b = delta/(4*tau), 7*delta**2/(32*tau)
    return np.exp(delta/(4*k*k))*(a/k**3+b/k**5)


def run():
    e, values = Evidence(), {}
    rng = np.random.default_rng(290921)
    # R90: closed walks, a controlled tail and exact matrix Hessian.
    for n in (1, 2, 4):
        weights = rng.uniform(.02, .13, (n, n))
        weights = (weights+weights.T)/2
        np.fill_diagonal(weights, 0)
        s = rng.uniform(.9, 1.5, n)
        delta = rng.uniform(-2.8, 2.8, n)
        k, r, d = lattice(s, weights, delta)
        k0, r0, _ = lattice(s, weights, np.zeros(n))
        gap = logdet(k)-logdet(k0)
        radius = max(abs(np.linalg.eigvalsh(r0)))
        e.check('R90', f'positive_lattice_{n}', min(np.linalg.eigvalsh(k)) > 0 and radius < 1,
                {'minimum_eigenvalue': float(min(np.linalg.eigvalsh(k))), 'spectral_radius': float(radius)})
        partial = 0.
        rn, rn0 = np.eye(3*n, dtype=complex), np.eye(3*n, dtype=complex)
        nonnegative = True
        for order in range(1, 21):
            rn, rn0 = rn@r, rn0@r0
            term = (np.trace(rn0-rn)/order).real
            nonnegative = nonnegative and term >= -1e-13
            partial += term
            if order in (3, 8, 20):
                bound = 6*n*radius**(order+1)/((order+1)*(1-radius))
                e.check('R90', f'closed_walk_tail_{n}_{order}', -1e-12 <= gap-partial <= bound+1e-12,
                        {'remainder': float(gap-partial), 'upper_bound': float(bound)})
        e.check('R90', f'all_walk_orders_nonnegative_{n}', nonnegative, {'orders': 20})
        local = 2*.25**3*np.sum((1-np.cos(delta))/d**3)
        e.check('R90', f'local_triangle_lower_bound_{n}', gap >= local-1e-12,
                {'logdet_gap': float(gap), 'triangle_bound': float(local)})
        h, lower = lattice_hessian(s, weights, .25)
        e.check('R90', f'full_Hessian_lower_bound_{n}', min(np.linalg.eigvalsh(h-np.diag(lower))) > -1e-12,
                {'Hessian_eigenvalues': np.linalg.eigvalsh(h).tolist(), 'diagonal_lower_bound': lower.tolist()})
        direction = rng.normal(size=n)
        step = 2e-3
        numeric = (logdet(lattice(s, weights, step*direction)[0])+logdet(lattice(s, weights, -step*direction)[0])-2*logdet(k0))/step**2
        e.close('R90', f'independent_Hessian_difference_{n}', numeric, direction@h@direction, 5e-7)
        phases = np.exp(1j*rng.normal(size=3*n))
        e.close('R90', f'equality_under_all_edge_gauge_transform_{n}',
                logdet(phases[:, None]*k0*phases.conj()[None, :]), logdet(k0))
    weights = np.array([[0., .3], [.3, 0.]])
    k, r, d = lattice([1., 1.3], weights, [0., 2*np.pi])
    k0, r0, _ = lattice([1., 1.3], weights, [0., 0.])
    rectangle = 9*.25**2*.3**2/(d[0]*d[1])**2
    fourth = np.trace(np.linalg.matrix_power(r0, 4)-np.linalg.matrix_power(r, 4)).real/4
    e.close('R90', 'rectangle_walk_coefficient', fourth, rectangle)
    e.check('R90', 'same_local_holonomy_different_spatial_transport',
            logdet(k)-logdet(k0) >= rectangle > 0,
            {'logdet_gap': float(logdet(k)-logdet(k0)), 'rectangle_lower_bound': float(rectangle)})

    # R91: integrate a positive discrete Higgs measure before differentiating.
    eps, ss = .35, np.array([1.1, 1.4, 1.8])
    det0 = ss**3-3*ss*eps**2-2*eps**3
    base_weights = np.array([.2, .3, .5])
    probs = base_weights/det0
    probs /= probs.sum()
    coeff = 2*eps**3/det0
    def free(delta):
        v = 2*np.sin(delta/2)**2
        return -np.log1p(-np.dot(probs, coeff*v/(1+coeff*v)))
    hf = float(probs@coeff)
    e.close('R91', 'Higgs_integrated_Hessian', (free(1e-4)+free(-1e-4))/1e-8, hf, 1e-9)
    e.check('R91', 'integrated_Hessian_strict_lower_bound', hf >= np.dot(probs, 2*eps**3/ss**3) > 0,
            {'Hessian': hf, 'lower_bound': float(np.dot(probs, 2*eps**3/ss**3))})
    x = .8
    step = 1e-4
    gammap = coeff*np.sin(x)/(1+coeff*(1-np.cos(x)))
    gammapp = coeff*np.cos(x)/(1+coeff*(1-np.cos(x)))-gammap**2
    tilted = probs/(1+coeff*(1-np.cos(x)))
    tilted /= tilted.sum()
    rhs = tilted@gammapp-(tilted@gammap**2-(tilted@gammap)**2)
    e.close('R91', 'off_minimum_covariance_is_required', (free(x+step)-2*free(x)+free(x-step))/step**2, rhs, 5e-9)
    concentration = []
    for strength in (20., 100., 500., 2500.):
        den = quad(lambda t: np.exp(-strength*free(t)), -np.pi, np.pi, epsabs=1e-12)[0]
        variance = quad(lambda t: t*t*np.exp(-strength*free(t)), -np.pi, np.pi, epsabs=1e-12)[0]/den
        concentration.append(strength*variance)
        e.check('R91', f'finite_strength_has_positive_variance_{strength}', variance > 0,
                {'strength_times_variance': strength*variance, 'inverse_Hessian': 1/hf})
    e.check('R91', 'Laplace_covariance_limit', abs(concentration[-1]*hf-1) < .02,
            {'strength_times_variance': concentration[-1], 'inverse_Hessian': 1/hf})
    den = quad(lambda t: np.exp(-20*free(t)), -np.pi, np.pi)[0]
    tail = 2*quad(lambda t: np.exp(-20*free(t)), .5, np.pi)[0]/den
    e.check('R91', 'finite_weight_is_not_single_outcome', 0 < tail < 1,
            {'probability_outside_half_radian': tail})
    values['positive_measure_concentration'] = concentration

    # R92: normalized record blocks and actual flux, independent finite difference.
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    h = (raw+raw.conj().T)/2
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    rho = raw@raw.conj().T
    rho /= np.trace(rho)
    for t in (.1, .6, 1.3):
        evolve = lambda v: expm(-1j*h*v)@rho@expm(1j*h*v)
        state = evolve(t)
        rates = -1j*comm(h, state)
        trace_rates = []
        for index in (0, 1):
            r, s = slice(2*index, 2*index+2), slice(2*(1-index), 2*(1-index)+2)
            p, ps = np.trace(state[r, r]).real, np.trace(state[s, s]).real
            sigma = state[r, r]/p
            flux = -1j*(h[r, s]@state[s, r]-state[r, s]@h[s, r])
            pdot = np.trace(flux).real
            predicted = -1j*comm(h[r, r], sigma)+(flux-sigma*pdot)/p
            actual = rates[r, r]/p-sigma*np.trace(rates[r, r])/p
            e.close('R92', f'normalized_record_equation_{t}_{index}', predicted, actual)
            dt = 1e-5
            plus, minus = evolve(t+dt)[r, r], evolve(t-dt)[r, r]
            numeric = (plus/np.trace(plus)-minus/np.trace(minus))/(2*dt)
            e.close('R92', f'independent_record_derivative_{t}_{index}', numeric, predicted, 3e-8)
            block_bound = np.sqrt(p*ps)
            e.check('R92', f'positive_block_trace_norm_{t}_{index}', norm1(state[r, s]) <= block_bound+1e-12,
                    {'off_diagonal_norm': float(norm1(state[r, s])), 'bound': float(block_bound)})
            bound = 4*np.linalg.norm(h[r, s], 2)*np.sqrt(ps/p)
            e.check('R92', f'conditional_flux_bound_{t}_{index}',
                    norm1(predicted+1j*comm(h[r, r], sigma)) <= bound+1e-12,
                    {'conditional_flux_norm': float(norm1(predicted+1j*comm(h[r, r], sigma))), 'bound': float(bound)})
            trace_rates.append(pdot)
        e.close('R92', f'total_record_probability_conserved_{t}', sum(trace_rates), 0)
    for p in (.01, .0001, .000001):
        h = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        state = expm(-1j*h*np.sqrt(p))@np.array([np.sqrt(p), 0, np.sqrt(1-p)])
        conditional = abs(state[1])**2/(abs(state[0])**2+abs(state[1])**2)
        formula = (1-p)*np.sin(np.sqrt(p))**2/(p+(1-p)*np.sin(np.sqrt(p))**2)
        e.close('R92', f'rare_record_exact_amplification_{p}', conditional, formula)
        e.check('R92', f'short_time_large_conditional_change_{p}', .49 < conditional < .51,
                {'time': np.sqrt(p), 'conditional_probability': float(conditional)})

    # R93: dephasing can occur without a distinguishable mixed environment marginal.
    ua, ub = np.eye(2), np.diag([1., -1.])
    mixed = np.eye(2)/2
    nu = np.trace(ua@mixed@ub.conj().T)
    e.close('R93', 'zero_coherence_factor_mixed_counterexample', nu, 0)
    e.close('R93', 'identical_conditional_environment_counterexample', ua@mixed@ua.T, ub@mixed@ub.T)
    n = 6
    annih = np.diag(np.sqrt(np.arange(1, n+4)), 1)
    coordinate = (annih+annih.T)/np.sqrt(2)
    obs = (coordinate@coordinate)[:n, :n]/2
    ha, hb = np.diag(np.arange(n)+.5)+.2*obs, np.diag(np.arange(n)+.5)+.8*obs
    density = np.diag(np.exp(-.7*np.arange(n)))
    density /= np.trace(density)
    pure = np.zeros((n, n))
    pure[0, 0] = 1
    for label, initial in [('pure', pure), ('thermal', density), ('maximally_mixed', np.eye(n)/n)]:
        aa, bb = expm(-1j*ha), expm(-1j*hb)
        nu = np.trace(aa@initial@bb.conj().T)
        distance = norm1(aa@initial@aa.conj().T-bb@initial@bb.conj().T)/2
        upper = np.sqrt(max(0, 1-abs(nu)**2))
        e.check('R93', f'accessible_record_bound_{label}', distance <= upper+1e-12,
                {'environment_trace_distance': float(distance), 'coherence_bound': float(upper)})
        if label == 'pure':
            e.close('R93', 'pure_state_equality', distance, upper)
        if label == 'maximally_mixed':
            e.check('R93', 'projected_oscillator_mixed_counterexample', distance < 1e-12 and abs(nu) < .99,
                    {'distance': float(distance), 'coherence_factor': float(abs(nu))})
        dt = 1e-4
        aa, bb = expm(-1j*ha*dt), expm(-1j*hb*dt)
        nu = np.trace(aa@initial@bb.conj().T)
        variance = (np.trace(initial@obs@obs)-np.trace(initial@obs)**2).real
        e.close('R93', f'variance_controls_dephasing_{label}', (1-abs(nu))/dt**2, .5*.6**2*variance, 1e-6)
        distance = norm1(aa@initial@aa.conj().T-bb@initial@bb.conj().T)/2
        e.close('R93', f'commutator_controls_accessible_record_{label}', distance/dt, .3*norm1(comm(obs, initial)), 4e-5)

    # R94: exact return pulse, destructive interference at one mode, UV total.
    xa, xb, duration = .4, 1., .9
    def pulse_n(k):
        wa, wb = np.sqrt(k*k+xa), np.sqrt(k*k+xb)
        return (xb-xa)**2*np.sin(wb*duration)**2/(4*wa**2*wb**2)
    for k in (0., .3, 1., 4., 20., 100.):
        wa, wb = np.sqrt(k*k+xa), np.sqrt(k*k+xb)
        alpha, beta = (np.sqrt(wb/wa)+np.sqrt(wa/wb))/2, (np.sqrt(wb/wa)-np.sqrt(wa/wb))/2
        squeeze = np.array([[alpha, beta], [beta, alpha]])
        inv = np.array([[alpha, -beta], [-beta, alpha]])
        out = inv@np.diag([np.exp(-1j*wb*duration), np.exp(1j*wb*duration)])@squeeze@np.array([1., 0.])
        e.close('R94', f'exact_two_jump_transfer_{k}', abs(out[1])**2, pulse_n(k), 1e-12)
    transparent_k = np.sqrt((np.pi/duration)**2-xb)
    e.close('R94', 'one_transparent_mode_does_not_remove_jumps', pulse_n(transparent_k), 0, 1e-25)
    energies = [quad(lambda k: k*k*np.sqrt(k*k+xa)*pulse_n(k)/np.pi**2, 0, cutoff,
                     epsabs=2e-10, epsrel=2e-9, limit=1200)[0] for cutoff in (64., 128., 256., 512.)]
    slope = (energies[-1]-energies[-2])/np.log(2)
    predicted = 2*(xb-xa)**2/(16*np.pi**2)
    e.check('R94', 'return_pulse_log_divergence_coefficient', abs(slope/predicted-1) < .02,
            {'numerical_log_slope': slope, 'sum_of_jump_squares_coefficient': predicted, 'net_endpoint_mass_change': 0})
    e.check('R94', 'return_pulse_energy_grows_with_cutoff', np.all(np.diff(energies) > 0),
            {'cutoffs': [64, 128, 256, 512], 'energies': energies})
    values['return_pulse_cutoff_energy'] = energies

    # R95: continuous piecewise-linear mass, whose derivative has jump discontinuities.
    for tau in (.3, .8):
        for k in (.3, 1., 2., 4., 8., 16.):
            alpha, beta = ramp_beta(k, tau=tau)
            bound = ramp_bound(k, tau=tau)
            e.close('R95', f'ramp_Bogoliubov_normalization_{tau}_{k}', abs(alpha)**2-abs(beta)**2, 1, 1e-10)
            e.check('R95', f'BV_preparation_bound_{tau}_{k}', abs(beta) <= bound+1e-12,
                    {'beta_abs': float(abs(beta)), 'bound': float(bound)})
        kmin, kmax = 4., 20.
        bands = []
        for count in (32, 64):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            nodes, weights = (nodes+1)*(kmax-kmin)/2+kmin, weights*(kmax-kmin)/2
            bands.append(float(sum(w*k*k*np.sqrt(k*k+1)*abs(ramp_beta(k, tau=tau)[1])**2/np.pi**2
                                   for k, w in zip(nodes, weights))))
        a, b = .6/(4*tau), 7*.6**2/(32*tau)
        tail_bound = 2*np.exp(.6/(2*kmin**2))/np.pi**2*np.sqrt(1+1/kmin**2)*(a*a/(2*kmin**2)+b*b/(6*kmin**6))
        e.close('R95', f'independent_band_quadrature_{tau}', bands[0], bands[1], 2e-9)
        e.check('R95', f'finite_high_k_band_below_full_tail_bound_{tau}', 0 <= bands[-1] <= tail_bound,
                {'computed_band': [kmin, kmax], 'band_energy': bands[-1], 'full_tail_upper_bound': tail_bound})

    # R96: vacuum overlap density in the actual 3D box and the non-atomic HS failure.
    xa, xb = .4, 1.
    def overlap_log(k):
        ratio = .25*np.log1p((xb-xa)/(k*k+xa))
        return np.log1p(2*np.sinh(ratio/2)**2)
    def static_beta2(k):
        wa, wb = np.sqrt(k*k+xa), np.sqrt(k*k+xb)
        return (xb-xa)**2/(4*wa*wb*(wa+wb)**2)
    continuum = quad(lambda k: k*k*overlap_log(k)/(2*np.pi**2), 0, 5., epsabs=1e-12)[0]
    bulk = quad(lambda k: k*k*overlap_log(k)/(2*np.pi**2), 0, np.inf, epsabs=1e-12)[0]
    boxes = []
    for length in (8., 16., 32.):
        maximum = int(np.floor(5*length/(2*np.pi)))
        grid = np.arange(-maximum, maximum+1)*(2*np.pi/length)
        k2 = grid[:, None, None]**2+grid[None, :, None]**2+grid[None, None, :]**2
        density = np.sum(overlap_log(np.sqrt(k2[k2 <= 25])))/length**3
        boxes.append({'length': length, 'cutoff_density': float(density), 'finite_cutoff_overlap': float(np.exp(-density*length**3))})
        e.check('R96', f'finite_box_overlap_is_positive_{length}', 0 < np.exp(-density*length**3) < 1,
                boxes[-1])
    e.check('R96', 'finite_box_density_converges_to_same_cutoff_integral',
            abs(boxes[-1]['cutoff_density']/continuum-1) < .01,
            {'last_density': boxes[-1]['cutoff_density'], 'continuum_same_cutoff': continuum, 'full_bulk_density': bulk})
    e.check('R96', 'extensive_overlap_loss', boxes[-1]['finite_cutoff_overlap'] < boxes[0]['finite_cutoff_overlap'],
            {'first_overlap': boxes[0]['finite_cutoff_overlap'], 'last_overlap': boxes[-1]['finite_cutoff_overlap']})
    number_density = quad(lambda k: k*k*static_beta2(k)/(2*np.pi**2), 0, np.inf, epsabs=1e-12)[0]
    e.check('R96', 'finite_particle_density_not_global_HS_norm', np.isfinite(number_density) and number_density > 0,
            {'one_real_component_particle_density': number_density})
    minimum_beta2 = static_beta2(2.)
    for count in (8, 32, 128):
        partition = np.linspace(1., 2., count+1)
        partial_trace = sum(quad(lambda k: k*k*static_beta2(k), left, right, epsabs=1e-13)[0]/
                            ((right**3-left**3)/3) for left, right in zip(partition[:-1], partition[1:]))
        e.check('R96', f'orthonormal_shells_force_HS_divergence_{count}', partial_trace >= count*minimum_beta2,
                {'orthonormal_vectors': count, 'sum_image_norm_squared': partial_trace, 'lower_bound': count*minimum_beta2})
    for k in (20., 100., 500.):
        e.close('R96', f'static_beta_UV_coefficient_{k}', k**4*static_beta2(k), (xb-xa)**2/16, 1e-4)
    values['box_overlap_examples'] = boxes

    ids = sorted({row['claim'] for row in e.checks})
    assert ids == [f'R{i:02d}' for i in range(90, 97)]
    return {'schema': 'CE-RB11-v1', 'scope': 'preserved chapters 05 and 06 under their actual assumptions',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(row['passed'] for row in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_records.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
