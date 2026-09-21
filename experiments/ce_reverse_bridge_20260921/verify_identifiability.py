"""CE-RB12: calibrated inverse records and the actual curvature-portal branches.

Finite-basis conditioning examples are not new observations or certified continuum
ground-state values. General statements and domains are in chapter 30.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp

from verify_reverse import Evidence


def inverse_thresholds(tl, th):
    return (th*th-tl*tl)/12, (tl*tl+2*th*th)/12


def spectrum(omega, x, coupling=1.):
    if omega <= 0:
        return 0.
    return coupling**2*np.sqrt(np.maximum(0, 1-4*np.array(x)/omega**2)).sum()/(8*np.pi)


def hann_fourier(z, duration):
    x = z*duration/(2*np.pi)
    return duration*(.5*np.sinc(x)+.25*np.sinc(x-1)+.25*np.sinc(x+1))


def joint_record_data(nh=12, nc=5):
    # Same finite-mode Hamiltonian/parameters as chapter 07, independent Galerkin construction.
    size, freq = 2*nh+4, 1.5
    annih = np.diag(np.sqrt(np.arange(1, size)), 1)
    position = (annih+annih.T)/np.sqrt(2*freq)
    momentum = -1j*np.sqrt(freq/2)*(annih-annih.T)
    even = np.arange(0, 2*nh, 2)
    restrict = lambda a: a[np.ix_(even, even)].real
    u, u2 = restrict(position@position)/2, restrict(np.linalg.matrix_power(position, 4))/4
    hh = restrict(momentum@momentum)/2+10*(u2-u+.25*np.eye(nh))
    oc, ec = csr_matrix((nc**3, nc**3)), np.zeros(nc**3)
    for index, mass2 in enumerate((.2, .65, .65)):
        omega = np.sqrt(mass2)
        o = diags([(np.arange(nc-1)+1)/(2*omega), (2*np.arange(nc)+1)/(2*omega),
                   (np.arange(nc-1)+1)/(2*omega)], [-1, 0, 1], shape=(nc, nc))
        factors = [eye(nc, format='csr')]*3
        factors[index] = o
        full = kron(kron(factors[0], factors[1]), factors[2], format='csr')
        oc += full
        shape = [1, 1, 1]
        shape[index] = nc
        ec += np.broadcast_to((omega*(2*np.arange(nc)+1)).reshape(shape), (nc, nc, nc)).ravel()
    h = kron(csr_matrix(hh), eye(nc**3), format='csr')+kron(eye(nh), diags(ec), format='csr')+kron(csr_matrix(u), oc, format='csr')
    energy, state = eigsh(h, k=1, which='SA', v0=np.ones(h.shape[0]), tol=2e-11)
    residual = np.linalg.norm(h@state[:, 0]-energy[0]*state[:, 0])
    ovalues, ovec = np.linalg.eigh(oc.toarray())
    wave = state[:, 0].reshape(nh, nc**3)@ovec
    weights = np.sum(wave**2, axis=0)
    means = [np.sum(wave*(a@wave), axis=0) for a in (u2, u)]
    def row(record, sigma):
        likelihood = np.exp(-(record-ovalues)**2/(2*sigma*sigma))
        norm = weights@likelihood
        return np.array([m@likelihood/norm for m in means])
    a0, b0 = [m.sum() for m in means]
    o0 = weights@ovalues
    ca, cb = [m@ovalues-m.sum()*o0 for m in means]
    return row, a0*cb-b0*ca, residual, h.shape[0]


def quartic_tensor(lh, lc, coupling):
    out = np.zeros((10,)*4)
    for indices in itertools.product(range(10), repeat=4):
        groups = [i < 4 for i in indices]
        count = sum(groups)
        a, b, c, d = indices
        if count in (0, 4):
            strength = lh if count == 4 else lc
            out[indices] = 2*strength*((a == b)*(c == d)+(a == c)*(b == d)+(a == d)*(b == c))
        elif count == 2:
            hi = [i for i in indices if i < 4]
            ci = [i for i in indices if i >= 4]
            out[indices] = coupling*(hi[0] == hi[1])*(ci[0] == ci[1])
    return out


def even_covariant_spectrum(a, cutoff):
    extra = cutoff+2
    modes = np.arange(-extra, extra+1)
    size = len(modes)
    sine = (np.diag(np.ones(size-1), -1)-np.diag(np.ones(size-1), 1))/(2j)
    momentum = np.diag(modes)
    full = momentum@momentum-a*(momentum@sine+sine@momentum)+a*a*sine@sine
    # Project the full square, keeping intermediate modes beyond the cutoff.
    vectors = np.zeros((size, cutoff+1))
    vectors[extra, 0] = 1
    for n in range(1, cutoff+1):
        vectors[extra-n, n] = vectors[extra+n, n] = 1/np.sqrt(2)
    return np.linalg.eigvalsh(vectors.T@full@vectors), float(full[extra, extra].real)


def run():
    e, values = Evidence(), {}
    rng = np.random.default_rng(300921)
    # R97: integrate the actual Kraus factors independently of their channel formula.
    obs = np.array([.1, .5, 1.4, 3.])
    for sigma in (.4, 1., 2.):
        kernel = np.array([[quad(lambda r: np.exp(-((r-a)**2+(r-b)**2)/(4*sigma*sigma))/
                                np.sqrt(2*np.pi*sigma*sigma), -np.inf, np.inf, epsabs=1e-12)[0]
                            for b in obs] for a in obs])
        expected = np.exp(-(obs[:, None]-obs[None, :])**2/(8*sigma*sigma))
        e.close('R97', f'Kraus_channel_direct_integral_{sigma}', kernel, expected, 1e-11)
        e.check('R97', f'Gaussian_dephasing_CP_kernel_{sigma}', np.linalg.eigvalsh(kernel)[0] >= -1e-12,
                {'minimum_kernel_eigenvalue': float(np.linalg.eigvalsh(kernel)[0])})
        prior = np.array([.1, .2, .4, .3])
        def fields(r):
            logparts = np.log(prior)-(r-obs)**2/(2*sigma*sigma)-np.log(np.sqrt(2*np.pi)*sigma)
            lp = logsumexp(logparts)
            post = np.exp(logparts-lp)
            mean = post@obs
            variance = post@(obs-mean)**2
            return np.exp(lp), (mean-r)/(sigma*sigma), variance
        limits = [min(obs)-12*sigma, max(obs)+12*sigma]
        info = quad(lambda r: fields(r)[0]*fields(r)[1]**2, *limits, epsabs=2e-12)[0]
        postvar = quad(lambda r: fields(r)[0]*fields(r)[2], *limits, epsabs=2e-12)[0]
        e.close('R97', f'Fisher_posterior_variance_identity_{sigma}', info, 1/sigma**2-postvar/sigma**4, 2e-10)
        e.check('R97', f'Fisher_resolution_bound_{sigma}', 0 <= info <= 1/sigma**2,
                {'location_Fisher_information': info, 'upper_bound': 1/sigma**2})
    nodes, weights = np.polynomial.hermite.hermgauss(18)
    weights /= np.sqrt(np.pi)
    grid = np.array(np.meshgrid(nodes, nodes, indexing='ij')).reshape(2, -1)
    joint_weights = np.outer(weights, weights).ravel()
    for correlation in (0., .4):
        precision = np.array([[1.3, correlation], [correlation, .9]])
        eig, vec = np.linalg.eigh(precision)
        coords = (vec*eig**(-.5))@vec.T@grid
        q = coords[1]
        dlog = -(precision@coords)[1]
        initial_kinetic = .5*joint_weights@(dlog*dlog)
        mean_o = .5*joint_weights@(q*q)
        for sigma in (.3, .8, 2.):
            # Direct q derivative of M_r(q)*psi(phi,q), integrate pointer and coordinates.
            derivative = dlog[:, None]+q[:, None]*(np.sqrt(2)*nodes[None, :])/(2*sigma)
            measured_kinetic = .5*np.sum(joint_weights[:, None]*weights[None, :]*derivative**2)
            e.close('R97', f'coordinate_measurement_work_{correlation}_{sigma}',
                    measured_kinetic-initial_kinetic, mean_o/(4*sigma*sigma), 2e-12)
    o, p = sp.symbols('o p', real=True)
    # Classical differential representation verifies the canonical double commutator on a test function.
    qv = sp.symbols('q', real=True)
    f = sp.Function('f')(qv)
    O = lambda v: qv*qv*v/2
    H = lambda v: -sp.diff(v, qv, 2)/2+qv**4*v
    once = lambda v: O(H(v))-H(O(v))
    e.zero('R97', 'canonical_double_commutator', O(once(f))-once(O(f))+qv*qv*f)

    # R98: worst-direction deterministic threshold errors, no fitted data.
    s0, eps, coupling = .5, .15, 1.
    sources = [.2, .8]
    true_t = [2*np.sqrt([s0+coupling*u-2*eps, s0+coupling*u+eps]) for u in sources]
    errors = [np.array([.003, .005]), np.array([.004, .002])]
    es = [(2*t[0]*err[0]+err[0]**2+2*(2*t[1]*err[1]+err[1]**2))/12 for t, err in zip(true_t, errors)]
    ek = sum(es)/(sources[1]-sources[0])
    for signs in itertools.product([-1, 1], repeat=4):
        estimates = [inverse_thresholds(*(t+err*np.array(signs[2*i:2*i+2]))) for i, (t, err) in enumerate(zip(true_t, errors))]
        khat = (estimates[1][1]-estimates[0][1])/(sources[1]-sources[0])
        shat = estimates[0][1]-khat*sources[0]
        e.check('R98', 'calibrated_threshold_error_'+''.join('p' if s > 0 else 'm' for s in signs),
                abs(khat-coupling) <= ek+1e-12 and abs(shat-s0) <= es[0]+sources[0]*ek+1e-12,
                {'coupling_error': abs(khat-coupling), 'coupling_bound': ek, 's0_error': abs(shat-s0)})
    omega, threshold, amplitude = sp.symbols('omega T P', positive=True)
    ln_k = sp.log(8*sp.pi*amplitude)/2-sp.log(1-threshold**2/omega**2)/4
    e.zero('R98', 'threshold_amplitude_sensitivity', sp.diff(ln_k, threshold)-threshold/(2*(omega**2-threshold**2)))
    e.zero('R98', 'frequency_amplitude_sensitivity', sp.diff(ln_k, omega)+threshold**2/(2*omega*(omega**2-threshold**2)))
    for scale in (.5, 3.):
        for source in sources:
            old_x = np.array([s0+coupling*source-2*eps, s0+coupling*source+eps, s0+coupling*source+eps])
            new_x = np.array([s0+(coupling/scale)*(scale*source)-2*eps, s0+coupling*source+eps, s0+coupling*source+eps])
            e.close('R98', f'unknown_source_scale_threshold_degeneracy_{scale}_{source}', old_x, new_x)
            e.close('R98', f'unknown_gain_record_degeneracy_{scale}_{source}',
                    spectrum(3., old_x, coupling), scale**2*spectrum(3., new_x, coupling/scale))

    # R99: finite time window produces positive below-gap signal, with explicit bounds.
    masses = [.4, .85, .85]
    tl, maximum = 2*np.sqrt(masses[0]), 3/(8*np.pi)
    window_values = []
    for duration in (5., 10., 40.):
        norm = quad(lambda t: np.cos(np.pi*t/duration)**4, -duration/2, duration/2)[0]
        norm1 = quad(lambda t: (np.pi/duration*np.sin(2*np.pi*t/duration))**2, -duration/2, duration/2)[0]
        norm2 = quad(lambda t: (2*np.pi**2/duration**2*np.cos(2*np.pi*t/duration))**2, -duration/2, duration/2)[0]
        e.close('R99', f'Hann_first_derivative_norm_{duration}', norm1/norm, 4*np.pi**2/(3*duration**2))
        e.close('R99', f'Hann_second_derivative_norm_{duration}', norm2/norm, 16*np.pi**4/(3*duration**4))
        read_frequency = tl-.5
        # Split at actual species thresholds; integrate the bounded UV tail directly.
        integrand = lambda v: spectrum(v, masses)*hann_fourier(read_frequency-v, duration)**2/(2*np.pi*norm)
        th = 2*np.sqrt(masses[1])
        leakage = quad(integrand, tl, th, epsabs=2e-12)[0]+quad(integrand, th, np.inf, epsabs=2e-12, limit=800)[0]
        bound = maximum*norm2/(norm*.5**4)
        e.check('R99', f'below_gap_window_signal_{duration}', 0 < leakage <= bound,
                {'signal': leakage, 'second_derivative_bound': bound, 'below_threshold_by': .5})
        window_values.append({'duration': duration, 'signal': leakage, 'bound': bound})
    values['finite_window_leakage'] = window_values

    # R100: actual coupled finite-mode ground state supplies the record design rows.
    record_row, leading, residual, dimension = joint_record_data()
    e.check('R100', 'finite_Galerkin_ground_state_residual', residual < 2e-8,
            {'dimension': dimension, 'residual': residual, 'scope': 'conditioning witness, no continuum energy certification'})
    e.check('R100', 'weak_record_leading_coefficient_nonzero', abs(leading) > 1e-6,
            {'a_cov_uO_minus_b_cov_u2O': leading})
    weak = []
    for sigma in (20., 40., 80., 160.):
        design = np.array([record_row(.5, sigma), record_row(4., sigma)])
        determinant = np.linalg.det(design)
        weak.append({'sigma': sigma, 'scaled_determinant': float(sigma*sigma*determinant),
                     'scaled_condition': float(np.linalg.cond(design)/(sigma*sigma))})
    target = 3.5*leading
    e.check('R100', 'weak_measurement_determinant_asymptotic', abs(weak[-1]['scaled_determinant']/target-1) < .01,
            {'scaled_determinant': weak[-1]['scaled_determinant'], 'predicted_limit': target})
    e.check('R100', 'condition_number_grows_as_sigma_squared',
            abs(weak[-1]['scaled_condition']/weak[-2]['scaled_condition']-1) < .01,
            {'last_scaled_conditions': [r['scaled_condition'] for r in weak[-2:]]})
    design = np.array([record_row(.5, 1.), record_row(4., 1.)])
    z = np.array([40., -20.])
    qdata = design@z
    smin = np.linalg.svd(design, compute_uv=False)[-1]
    for trial in range(8):
        dx = rng.normal(size=(2, 2))
        dx *= smin/(8*np.linalg.norm(dx, 2))
        dq = rng.normal(size=2)*1e-5
        recovered = np.linalg.solve(design+dx, qdata+dq)
        bound = (np.linalg.norm(dq)+np.linalg.norm(dx, 2)*np.linalg.norm(z))/(smin-np.linalg.norm(dx, 2))
        e.check('R100', f'data_and_design_noise_bound_{trial}', np.linalg.norm(recovered-z) <= bound,
                {'coefficient_error': float(np.linalg.norm(recovered-z)), 'bound': float(bound),
                 'scope': 'linear inverse stress test on physical record rows'})
        da, db = recovered-z
        if abs(da) < abs(z[0]):
            u_bound = (abs(db)+.5*abs(da))/(abs(z[0])-abs(da))
            e.check('R100', f'ratio_parameter_error_bound_{trial}', abs(-recovered[1]/recovered[0]-.5) <= u_bound+1e-14,
                    {'u0_error': float(abs(-recovered[1]/recovered[0]-.5)), 'bound': float(u_bound)})
    values['weak_record_conditioning'] = weak

    # R101: independently contract the original quartic tensor for O(4)+O(6).
    bases = [quartic_tensor(*v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
    lh, lc, kp = sp.symbols('lambda_H lambda_C kappa', real=True)
    parameters = [lh, lc, kp]
    betas = []
    for sector in ('H', 'C', 'portal'):
        expression = 0
        for i, bi in enumerate(bases):
            for j, bj in enumerate(bases):
                if sector == 'H':
                    factor = np.sum(bi[0, 0]*bj[0, 0])/2
                elif sector == 'C':
                    factor = np.sum(bi[4, 4]*bj[4, 4])/2
                else:
                    factor = np.sum(bi[0, 0]*bj[4, 4])+2*np.sum(bi[0, 4]*bj[0, 4])
                expression += sp.Rational(str(factor))*parameters[i]*parameters[j]
        betas.append(sp.expand(expression))
    for label, actual, expected in zip(['H', 'C', 'portal'], betas,
                                       [24*lh**2+3*kp**2, 28*lc**2+2*kp**2, kp*(12*lh+16*lc+4*kp)]):
        e.zero('R101', f'exact_quartic_tensor_{label}', actual-expected)
    def flow(t, y):
        h, c, k = y
        return [24*h*h+3*k*k, 28*c*c+2*k*k, k*(12*h+16*c+4*k)]
    initial = [.1, .08, .2]
    sol = solve_ivp(flow, [0, .1], initial, t_eval=np.linspace(0, .1, 21), rtol=2e-12, atol=1e-14, method='DOP853')
    e.check('R101', 'positive_cone_flow', sol.success and np.all(sol.y > 0),
            {'final_couplings': sol.y[:, -1].tolist()})
    e.check('R101', 'inverse_portal_comparison', np.all(1/sol.y[2] <= 1/initial[2]-4*sol.t+1e-12),
            {'last_inverse_portal': 1/sol.y[2, -1], 'comparison_bound': 1/initial[2]-4*sol.t[-1]})
    for index, rate in [(0, 24), (1, 28)]:
        e.check('R101', f'inverse_self_coupling_comparison_{index}',
                np.all(1/sol.y[index] <= 1/initial[index]-rate*sol.t+1e-12),
                {'last_inverse': 1/sol.y[index, -1], 'comparison_bound': 1/initial[index]-rate*sol.t[-1]})
    e.zero('R101', 'zero_residual_quartic_not_invariant', betas[1].subs(lc, 0)-2*kp**2)

    # R102: actual full squared parent matrix, including its mixing.
    g = .6
    for d in (-3., 3.):
        for zvalue in (.8, 1.2):
            for u in (.02, .4, 2.):
                lam = (d-np.sign(d)*np.sqrt(d*d+2*u))/2
                expected = g*g*(zvalue-lam)**2
                parent = np.array([[d, 0, np.sqrt(u/2)], [0, d, 0], [np.sqrt(u/2), 0, 0.]])-zvalue*np.eye(3)
                eigenvalues = np.linalg.eigvalsh(g*g*parent@parent)
                e.close('R102', f'full_parent_pole_{d}_{zvalue}_{u}', min(abs(eigenvalues-expected)), 0, 1e-12)
                linear = g*g*(zvalue*zvalue+zvalue*u/d)
                bound = g*g*u*u*(abs(zvalue)/(2*abs(d)**3)+1/(4*d*d))
                e.check('R102', f'finite_source_linear_remainder_{d}_{zvalue}_{u}',
                        abs(expected-linear) <= bound+1e-12, {'error': abs(expected-linear), 'bound': bound})
    for zvalue in (.8, 1.2):
        d = -3.
        ustar = 2*zvalue*(zvalue-d)
        lam = (d+np.sqrt(d*d+2*ustar))/2
        e.close('R102', f'actual_gap_closing_{zvalue}', g*g*(zvalue-lam)**2, 0, 1e-24)
    schur_rows = []
    u, zvalue = .7, .8
    for d in (10., 100., 1000.):
        a = g*g*(zvalue*zvalue+u/2)
        b = g*g*(d-2*zvalue)*np.sqrt(u/2)
        heavy = g*g*((d-zvalue)**2+u/2)
        schur_rows.append({'d': d, 'B2_over_D': b*b/heavy, 'schur_mass2': a-b*b/heavy, 'true_portal_slope': g*g*zvalue/d})
    e.close('R102', 'heavy_mixing_does_not_vanish', schur_rows[-1]['B2_over_D'], g*g*u/2, 3e-4)
    e.close('R102', 'heavy_limit_cancels_diagonal_portal', schur_rows[-1]['schur_mass2'], g*g*zvalue*zvalue, 3e-4)
    e.check('R102', 'true_portal_vanishes_in_same_heavy_limit',
            schur_rows[-1]['true_portal_slope'] < g*g/200,
            {'full_slope': schur_rows[-1]['true_portal_slope'], 'discarded_block_slope': g*g/2})
    values['heavy_channel_cancellation'] = schur_rows

    # R103: an allowed odd diagonal background is pure gauge in the full even KK sector.
    kk_rows = []
    for amplitude in (.5, 1.2, 2.):
        lowest = []
        for cutoff in (2, 4, 8, 12):
            eigenvalues, naive = even_covariant_spectrum(amplitude, cutoff)
            lowest.append(float(eigenvalues[0]))
            e.close('R103', f'naive_constant_mode_mass_{amplitude}_{cutoff}', naive, amplitude**2/2)
            e.check('R103', f'full_Galerkin_operator_nonnegative_{amplitude}_{cutoff}',
                    eigenvalues[0] > -2e-12, {'lowest_eigenvalue': float(eigenvalues[0]), 'cutoff': cutoff})
        e.check('R103', f'KK_mixing_restores_actual_zero_mode_{amplitude}', abs(lowest[-1]) < 1e-11,
                {'cutoffs': [2, 4, 8, 12], 'lowest_eigenvalues': lowest, 'constant_projection_mass2': amplitude**2/2})
        kk_rows.append({'amplitude': amplitude, 'lowest_eigenvalues': lowest})
    p5, p6 = np.diag([-1, -1, 1, 1, 1, 1]), np.diag([1, 1, 1, -1, -1, -1])
    diagonal = np.diag([0., 0., 0., .4, .7, -1.1])
    hbase = np.zeros((6, 6), complex)
    hbase[0, 2], hbase[2, 0] = .3, .3
    for y in (.2, 1., 2.7):
        odd = diagonal*np.sin(y)
        e.close('R103', f'allowed_nonconstant_diagonal_parity_{y}', diagonal*np.sin(-y), -p5@odd@p5)
        unitary = expm(1j*g*diagonal*(1-np.cos(y)))
        uprime = 1j*g*odd@unitary
        transformed = unitary.conj().T@(hbase+odd)@unitary+1j/g*unitary.conj().T@uprime
        e.close('R103', f'full_gauge_transform_removes_only_odd_diagonal_{y}', transformed, hbase)
        e.close('R103', f'orbifold_allowed_gauge_transform_{y}', [unitary@p5-p5@unitary, unitary@p6-p6@unitary], 0)
    values['pure_gauge_KK_convergence'] = kk_rows

    ids = sorted({row['claim'] for row in e.checks}, key=lambda s: int(s[1:]))
    assert ids == [f'R{i:02d}' for i in range(97, 104)]
    return {'schema': 'CE-RB12-v1', 'scope': 'actual conditional record and curvature portal contracts in chapters 07-10',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(row['passed'] for row in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_identifiability.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
