"""Chapter 77: closed, homogeneous classical source/Higgs backreaction.

All five complex fields evolve in the original polynomial action. Supplied
classical seeds are not quantum vacuum draws or actual observation records.
Run in an isolated uv environment with numba, numpy, scipy, and sympy.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numba
from numba import njit
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root
import sympy as sy


PREREG = 'e9b621150bfc4b1a0d71e6054a038ecb71eb95ce0dedae8d10588e400ab72133'
M0, KAPPA = 1e-4, .05
MASSES = M0*np.array([5+np.pi/2, 5-np.pi/2])
MREF = float(MASSES[1])
N0 = 2*.1**2*MREF**2/KAPPA
AREF = np.sqrt(N0/(2*MREF))
A = 1/(2*np.sqrt(15))
B, D = 3*A, .5
TIMES = np.linspace(0, 60, 6001)
HEAVY = np.array([0, 1, 3, 5, 6, 8])
LIGHT = np.array([2, 4, 7, 9])


def maximum(x):
    return float(np.max(np.abs(x)))


@njit(cache=False)
def fields(x, mass):
    z, t, s, u, q = x
    f = np.empty(5, np.complex128)
    f[0] = -z+A*z*z+3*A*t*t+A*s*s+3*A*u*u-B*q*q/2
    f[1] = -5*t+6*A*z*t+6*A*s*u-D*q*q/2
    f[2] = (mass+2*A*z)*s+6*A*t*u
    f[3] = (mass-4+6*A*z)*u+6*A*t*s
    f[4] = -(B*z+D*t)*q
    w = np.zeros((5, 5), np.complex128)
    w[0, 0], w[1, 1] = -1+2*A*z, -5+6*A*z
    w[2, 2], w[3, 3], w[4, 4] = mass+2*A*z, mass-4+6*A*z, -B*z-D*t
    w[0, 1], w[0, 2], w[0, 3], w[0, 4] = 6*A*t, 2*A*s, 6*A*u, -B*q
    w[1, 2], w[1, 3], w[1, 4], w[2, 3] = 6*A*u, 6*A*s, -D*q, 6*A*t
    for i in range(5):
        for j in range(i):
            w[i, j] = w[j, i]
    return f, w


@njit(cache=False)
def acceleration(x, mass):
    f, w = fields(x, mass)
    out = np.zeros(5, np.complex128)
    for i in range(5):
        for j in range(5):
            out[i] -= w[j, i].conjugate()*f[j]
    return out


@njit(cache=False)
def rhs(tau, y, mass):
    out = np.empty(10, np.complex128)
    out[:5] = y[5:]
    out[5:] = acceleration(y[:5]*AREF, mass)/(MREF*MREF*AREF)
    return out


def cubic_polynomial():
    z, t, s, u, q, m = sy.symbols('z t s u q m')
    a = 1/(2*sy.sqrt(15))
    variables = [z, t, s, u, q]
    polynomial = (-z*z/2-5*t*t/2+m*s*s/2+(m-4)*u*u/2
                  +a*z**3/3+3*a*z*t*t+a*z*s*s+3*a*z*u*u
                  +6*a*t*s*u-(3*a*z+t/2)*q*q/2)
    f = sy.Matrix([sy.diff(polynomial, v) for v in variables])
    w = f.jacobian(variables)
    cubic = np.array([[[float(sy.diff(w[i, j], v)) for v in variables]
                       for j in range(5)] for i in range(5)])
    return cubic, sy.lambdify([variables, m], f, 'numpy'), sy.lambdify([variables, m], w, 'numpy')


def real_hessian(x, mass, cubic):
    f, w = fields(x, mass)
    aa = w.conj().T@w
    bb = np.einsum('k,kij->ij', f.conj(), cubic)
    return np.block([[(aa+bb).real, -(aa+bb).imag],
                     [(aa-bb).imag, (aa-bb).real]])


def initial_state(mass, eta, theta):
    cubic, _, _ = cubic_polynomial()
    amplitude = np.sqrt(N0/(2*mass))
    x = np.zeros(5, complex)
    target = np.array([amplitude*np.exp(1j*theta), eta*AREF])
    residuals = []
    for fraction in [.25, .5, .75, 1.]:
        x[[2, 4]] = fraction*target
        def unpack(h):
            xx = np.r_[x.real, x.imag]
            xx[HEAVY] = h
            return xx[:5]+1j*xx[5:]
        def fun(h):
            grad = -acceleration(unpack(h), mass)
            return np.r_[grad.real, grad.imag][HEAVY]
        def jac(h):
            return real_hessian(unpack(h), mass, cubic)[np.ix_(HEAVY, HEAVY)]
        guess = np.r_[x.real, x.imag][HEAVY]
        solution = root(fun, guess, jac=jac, tol=1e-11)
        x = unpack(solution.x)
        residuals.append(maximum(fun(solution.x))/(MREF*MREF*AREF))
    hessian = real_hessian(x, mass, cubic)
    hh = hessian[np.ix_(HEAVY, HEAVY)]
    velocity = np.zeros(5, complex)
    velocity[2] = -1j*mass*x[2]
    vr = np.r_[velocity.real, velocity.imag]
    vr[HEAVY] = -np.linalg.solve(hh, hessian[np.ix_(HEAVY, LIGHT)]@vr[LIGHT])
    velocity = vr[:5]+1j*vr[5:]
    tangent_error = maximum((hessian@vr)[HEAVY])/(MREF**3*AREF)
    assert max(residuals+[tangent_error]) < 1e-8, (residuals, tangent_error)
    assert np.linalg.eigvalsh(hh)[0] > 0
    return np.r_[x/AREF, velocity/(MREF*AREF)], {
        'amplitude_over_V': float(amplitude),
        'heavy_stationarity_scaled_residual': max(residuals),
        'heavy_tangent_scaled_residual': tangent_error,
        'heavy_real_Hessian_min_eigenvalue_over_V2': float(np.linalg.eigvalsh(hh)[0]),
        'initial_fields_real_imag': np.c_[x.real, x.imag].tolist(),
        'initial_velocities_real_imag': np.c_[velocity.real, velocity.imag].tolist()}


def algebra_checks():
    from ce_singlet_record_observability import build_model
    from ce_supersymmetric_record import f_terms
    _, _, reps, matrix, cubic, vacuum = build_model(M0)
    _, w0 = f_terms(vacuum, matrix, cubic)
    yred, fsym, wsym = cubic_polynomial()
    errors = {}
    def check(key, value):
        errors[key] = max(errors.get(key, 0.), maximum(value))
    rng = np.random.default_rng(7701)
    for species, mass in enumerate(MASSES):
        embedding = np.zeros((82, 5), complex)
        embedding[11, 0], embedding[10, 1] = 1, 1
        sign = 1 if species == 0 else -1
        embedding[35, 2], embedding[59, 2] = 1/np.sqrt(2), sign/np.sqrt(2)
        embedding[34, 3], embedding[58, 3] = 1/np.sqrt(2), sign/np.sqrt(2)
        embedding[75, 4], embedding[80, 4] = 1/np.sqrt(2), -1/np.sqrt(2)
        check('kinetic_embedding', embedding.conj().T@embedding-np.eye(5))
        projected = np.einsum('ijk,ia,jb,kc->abc', cubic, embedding, embedding, embedding, optimize=True)
        check('independent_cubic_tensor', projected-yred)
        check('projected_gauge_generators', np.einsum('ia,kij,jb->kab', embedding.conj(), reps, embedding, optimize=True))
        for _ in range(3):
            x = .003*(rng.normal(size=5)+1j*rng.normal(size=5))
            velocity = .0001*(rng.normal(size=5)+1j*rng.normal(size=5))
            f, w = fields(x, mass)
            check('symbolic_F', f-np.array(fsym(x, mass)).ravel())
            check('symbolic_Wij', w-np.array(wsym(x, mass)))
            delta = embedding@x
            ff = w0@delta+.5*np.einsum('ijk,j,k->i', cubic, delta, delta, optimize=True)
            ww = w0+np.einsum('ijk,k->ij', cubic, delta, optimize=True)
            check('full_F_closure', (ff-embedding@f)/maximum(f))
            check('full_force_closure', (ww.conj().T@ff+embedding@acceleration(x, mass))/maximum(acceleration(x, mass)))
            state = vacuum+delta
            dv = embedding@velocity
            check('D_zero', np.einsum('i,aij,j->a', state.conj(), reps, state, optimize=True))
            check('gauge_current_zero', np.einsum('i,aij,j->a', state.conj(), reps, dv, optimize=True))
            check('full_energy', (np.vdot(ff, ff).real+np.vdot(dv, dv).real
                                  -np.vdot(f, f).real-np.vdot(velocity, velocity).real)/np.vdot(f, f).real)
            current = 2*np.real(f.conj()[:, None]*w*velocity[None, :]
                                -f.conj()[None, :]*w.T*velocity[:, None])
            check('exchange_antisymmetry', current+current.T)
            edot = 2*np.real(velocity.conj()*acceleration(x, mass)+f.conj()*(w@velocity))
            check('exchange_energy_identity', (edot-current.sum(axis=1))/maximum(current))
            direction = rng.normal(size=10)
            direction /= np.linalg.norm(direction)
            dx = direction[:5]+1j*direction[5:]
            step = 1e-4
            def v(offset):
                ff = np.array(fsym(offset, mass)).ravel()
                return float(np.vdot(ff, ff).real)
            # Central derivative of a quartic; Richardson cancels the cubic error.
            cd = lambda h: (v(x+h*dx)-v(x-h*dx))/(2*h)
            finite = (4*cd(step/2)-cd(step))/3
            expected = -2*np.vdot(acceleration(x, mass), dx).real
            check('independent_potential_gradient', (finite-expected)/maximum(acceleration(x, mass)))
            # Independent finite derivative also checks the real stationary Hessian.
            dg = lambda h: (-acceleration(x+h*dx, mass)+acceleration(x-h*dx, mass))/(2*h)
            df = (4*dg(step/2)-dg(step))/3
            check('real_hessian', np.r_[df.real, df.imag]-real_hessian(x, mass, yred)@direction)
    for mass in MASSES:
        y0, _ = initial_state(mass, 0., np.pi/4)
        check('zero_seed_invariant', rhs(0., y0, mass)[[4, 9]])
    assert max(errors.values()) < 1e-8, errors
    return {'errors': errors, 'max_scaled_independent_error': max(errors.values()),
            'invariant_subspace': 'five canonical complex coordinates, one record mass species',
            'zero_seed_is_not_a_quantum_vacuum_no_click_proof': True}


def integrate(y0, mass, level, interval=(0., 60.), times=TIMES, method='DOP853'):
    # All coordinates and tau derivatives are dimensionless. The fine setting
    # is repeated with an independent RK45 method on three bounded windows.
    rtol, atol = [(2e-8, 2e-11), (2e-10, 2e-13)][level]
    start = time.perf_counter()
    solution = solve_ivp(lambda t, y: rhs(t, y, mass), interval, y0,
                         method=method, t_eval=times, rtol=rtol, atol=atol)
    assert solution.success, solution.message
    return solution.y, {'method': method, 'rtol': rtol, 'atol': atol,
                        'function_evaluations': solution.nfev,
                        'seconds': time.perf_counter()-start}


def diagnostics(trajectory, mass):
    count = trajectory.shape[1]
    energy = np.empty((5, count))
    kinetic = np.empty(count)
    free_residual = np.empty(count)
    for j in range(count):
        x, v = trajectory[:5, j]*AREF, trajectory[5:, j]*MREF*AREF
        f, _ = fields(x, mass)
        energy[:, j] = abs(v)**2+abs(f)**2
        kinetic[j] = np.sum(abs(v)**2)
        free_residual[j] = abs(acceleration(x, mass)[2]+mass*mass*x[2])/(mass*mass*np.sqrt(N0/(2*mass)))
    total = energy.sum(axis=0)
    qpower = abs(trajectory[4])**2
    return energy, total, qpower, kinetic, free_residual


def crossing_times(values, upward):
    mask = (values[:-1] < 1) & (values[1:] >= 1) if upward else (values[:-1] >= 1) & (values[1:] < 1)
    indices = np.flatnonzero(mask)
    return [float(TIMES[i]+(1-values[i])*(TIMES[i+1]-TIMES[i])/(values[i+1]-values[i])) for i in indices]


def run_case(species, eta, theta):
    mass = float(MASSES[species])
    y0, initialization = initial_state(mass, eta, theta)
    rhs(0., y0, mass)  # Compile before reporting elapsed integration time.
    low, coarse_info = integrate(y0, mass, 0)
    high, fine_info = integrate(y0, mass, 1)
    energy, total, power, kinetic, residual = diagnostics(high, mass)
    e_low, _, p_low, _, _ = diagnostics(low, mass)
    comparisons = {
        'fields_over_Aref': maximum(high[:5]-low[:5]),
        'velocities_over_mref_Aref': maximum(high[5:]-low[5:]),
        'qpower_absolute': maximum(power-p_low),
        'Fi_partition_energy_over_initial_total': maximum((energy-e_low)/total[0]),
        'total_energy_relative_drift': maximum(total/total[0]-1)}
    other = []
    if species == 1 and eta == .1 and theta == 0.:
        for start in [0., 30., 59.]:
            indices = np.flatnonzero((TIMES >= start) & (TIMES <= start+1))
            independent, info = integrate(high[:, indices[0]], mass, 1,
                                          (start, start+1), TIMES[indices], 'RK45')
            error = maximum(independent-high[:, indices])
            other.append({'tau_interval': [start, start+1], 'scaled_state_max_difference': error, **info})
    upward, downward = crossing_times(power, True), crossing_times(power, False)
    late = TIMES >= 50
    valid = max(comparisons.values()) < 1e-6 and all(r['scaled_state_max_difference'] < 1e-6 for r in other)
    stride = slice(None, None, 10)
    return {'species': '+' if species == 0 else '-', 'mass_over_V': mass, 'eta': eta, 'theta': theta,
            'initialization': initialization, 'integration': [coarse_info, fine_info],
            'numerical_comparisons': comparisons, 'independent_RK45_windows': other,
            'numerical_gate_passed': valid, 'pointer_judgment_withheld': not valid,
            'initial_energy_density_over_V4': float(total[0]),
            'qpower_peak_sampled': float(max(power)), 'qpower_late_min_max_sampled': [float(min(power[late])), float(max(power[late]))],
            'threshold_upward_tau_linear_interpolation': upward,
            'threshold_downward_tau_linear_interpolation': downward,
            'late_threshold_maintained_at_all_sampled_times': bool(np.all(power[late] >= 1)),
            'Higgs_Fi_partition_fraction_peak': float(max(energy[4]/total)),
            'Higgs_Fi_partition_fraction_late_min_max': [float(min(energy[4, late]/total[late])), float(max(energy[4, late]/total[late]))],
            'record_free_equation_scaled_residual_peak': float(max(residual)),
            'heavy_record_amplitude_over_initial_light_peak': float(max(abs(high[3]))*AREF/np.sqrt(N0/(2*mass))),
            'trace': {'tau': TIMES[stride].tolist(), 'qpower': power[stride].tolist(),
                      'Sigma_Fi_partition_fraction': ((energy[0]+energy[1])/total)[stride].tolist(),
                      'record_Fi_partition_fraction': ((energy[2]+energy[3])/total)[stride].tolist(),
                      'Higgs_Fi_partition_fraction': (energy[4]/total)[stride].tolist(),
                      'pressure_over_energy': (2*kinetic/total-1)[stride].tolist(),
                      'free_record_scaled_residual': residual[stride].tolist()},
            'sampling_scope': '6001 common tau samples; crossings interpolated, not certified all-time extrema'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algebra-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/77_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 77.2')[0].encode()).hexdigest()
    assert digest == PREREG, digest
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__), folder/'ce_singlet_record_observability.py',
               folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']}
    result = {'schema_version': 1, 'candidate': 'CE-UR4-D3', 'scientific_success': False,
              'full_joint_rmse': None, 'new_observational_inputs_or_fits': False,
              'preregistration_sha256': digest, 'source_hashes': hashes,
              'common_preparation': {'n0_over_V3': N0, 'mref_over_V': MREF,
                                     'Aref_over_V': AREF, 'physical_time_end_times_V': 60/MREF},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__, 'numba': numba.__version__},
              'action_checks': algebra_checks()}
    print(json.dumps({'algebra': result['action_checks']}), flush=True)
    if args.algebra_only:
        return
    # Durable completed cases are reusable only for identical code and prereg.
    destination = folder/'ce_record_backreaction.json'
    rows = []
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved.get('source_hashes') == hashes and saved.get('preregistration_sha256') == digest:
            rows = saved.get('cases', [])
    result['cases'] = rows
    result['run_complete'] = False
    def save():
        result['cases'].sort(key=lambda r: (r['mass_over_V'], r['eta'], r['theta']))
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        temporary.replace(destination)
    save()
    cases = [(species, eta, theta) for species in range(2) for eta in [.01, .1] for theta in [0., np.pi/4]]
    done = {(r['species'], r['eta'], r['theta']) for r in rows}
    pending = [c for c in cases if ('+' if c[0] == 0 else '-', c[1], c[2]) not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(run_case, *c): c for c in pending}
        for job in as_completed(jobs):
            row = job.result()
            result['cases'].append(row)
            save()
            print(json.dumps({k: row[k] for k in ['species', 'eta', 'theta', 'numerical_gate_passed',
                             'numerical_comparisons', 'qpower_peak_sampled', 'qpower_late_min_max_sampled',
                             'threshold_upward_tau_linear_interpolation', 'threshold_downward_tau_linear_interpolation']}), flush=True)
    result['run_complete'] = len(rows) == 8
    result['numerical_gate_passed'] = all(r['numerical_gate_passed'] for r in rows)
    result['actual_instrument_derived'] = False
    result['open_gates'] = ['spatial source preparation and outgoing radiation',
                           'quantum fluctuation state and actual asynchronous record',
                           'full gravity and frozen covariance-aware joint RMSE']
    save()
    assert result['run_complete'] and result['numerical_gate_passed'], 'withhold precise pointer judgment; inspect saved errors'
    print('all eight nonlinear cases and independent numerical checks passed', flush=True)


if __name__ == '__main__':
    main()
