"""CE-UR4-D4: spherical five-complex-field Hamiltonian, chapter 78.

The local quartic force uses an exact two-node AVF integral. No fields are
eliminated during evolution. Run --checks before --suite; output is resumable
only with identical implementation and frozen preregistration hashes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import platform

import numpy as np
import numba
import scipy
import sympy
from numba import njit
from scipy.linalg import solve_banded
from scipy.optimize import root
from scipy.integrate import solve_ivp

import ce_record_backreaction as base

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT/'paper/06_QFT_재설계/78_유한_기록원의_방사와_전체_공간_에너지.md'
OUTPUT = Path(__file__).with_suffix('.json')
PREREG = 'd92d6f7545b0754a3641393446799c71de9beb622246ee8a5f2950e32e2a520e'
MREF, AREF = base.MREF, base.AREF
CUBIC = base.cubic_polynomial()[0]
GAUSS = np.array([.5-np.sqrt(3)/6, .5+np.sqrt(3)/6])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    frozen = hashlib.sha256(CHAPTER.read_text(encoding='utf-8').split('## 78.2')[0].encode()).hexdigest()
    assert frozen == PREREG, 'Chapter 78 preregistration changed'
    return {'preregistration_sha256': frozen,
        'source_sha256': {Path(p).name: digest(p) for p in [__file__, base.__file__]}}


def mesh(dx, outer):
    edges = np.arange(round(outer/dx)+1)*dx
    return (edges[:-1]+dx/2, np.diff(edges**3)/3, edges[1:-1]**2/dx)


@njit(cache=True)
def local(y, mass, cubic, free=False):
    """Return half the real gradient of Vhat, its Jacobian, and Fi energies."""
    if free:
        return np.zeros(10), np.zeros((10, 10)), np.zeros(5)
    f, w = base.fields(AREF*(y[:5]+1j*y[5:]), mass)
    g = np.zeros(10)
    hess = np.empty((10, 10))
    for i in range(5):
        z = 0j
        for k in range(5):
            z += w[k, i].conjugate()*f[k]
        g[i], g[i+5] = z.real/(AREF*MREF**2), z.imag/(AREF*MREF**2)
        for j in range(5):
            aa, bb = 0j, 0j
            for k in range(5):
                aa += w[k, i].conjugate()*w[k, j]
                bb += f[k].conjugate()*cubic[k, i, j]
            hess[i, j] = (aa+bb).real/MREF**2
            hess[i, j+5] = -(aa+bb).imag/MREF**2
            hess[i+5, j] = (aa-bb).imag/MREF**2
            hess[i+5, j+5] = (aa-bb).real/MREF**2
    return g, hess, np.abs(f)**2/(AREF*MREF)**2


@njit(cache=True)
def assembly(y0, v0, y1, dt, vol, face, mass, cubic, free):
    n = len(vol)
    ab, residual = np.zeros((21, n*10)), np.empty((n, 10))
    middle = (y0+y1)/2
    for i in range(n):
        avg, jac = np.zeros(10), np.zeros((10, 10))
        for c in GAUSS:
            gg, hh, _ = local(y0[i]+c*(y1[i]-y0[i]), mass, cubic, free)
            avg += gg/2
            jac += c*hh/2
        left = face[i-1]/vol[i] if i > 0 else 0.
        right = face[i]/vol[i] if i < n-1 else 0.
        for a in range(10):
            lap = -(left+right)*middle[i, a]
            if i > 0:
                lap += left*middle[i-1, a]
                ab[20, (i-1)*10+a] = -dt*dt*left/4
            if i < n-1:
                lap += right*middle[i+1, a]
                ab[0, (i+1)*10+a] = -dt*dt*right/4
            residual[i, a] = y1[i, a]-y0[i, a]-dt*v0[i, a]-dt*dt*(lap-avg[a])/2
            for b in range(10):
                ab[10+a-b, i*10+b] = dt*dt*jac[a, b]/2
            ab[10, i*10+a] += 1+dt*dt*(left+right)/4
    return ab, residual


def step(y, v, dt, vol, face, mass, free=False):
    end = y+dt*v
    correction = float('inf')
    for iteration in range(12):
        ab, residual = assembly(y, v, end, dt, vol, face, mass, CUBIC, free)
        delta = solve_banded((10, 10), ab, -residual.ravel(),
                             overwrite_ab=True, overwrite_b=True, check_finite=False).reshape(y.shape)
        end += delta
        correction = float(np.max(np.abs(delta)))
        if correction < 2e-13:
            break
    else:
        raise RuntimeError(f'Newton failed: correction {correction}')
    if not np.isfinite(end).all():
        raise RuntimeError('Nonfinite field')
    return end, 2*(end-y)/dt-v, iteration+1, correction


@njit(cache=True)
def energy(y, v, vol, face, mass, cubic, free=False):
    cells = np.zeros((len(vol), 5))
    for i in range(len(vol)):
        _, _, potential = local(y[i], mass, cubic, free)
        cells[i] = vol[i]*(v[i, :5]**2+v[i, 5:]**2+potential)
    for i in range(len(face)):
        d = y[i+1]-y[i]
        edge = face[i]*(d[:5]**2+d[5:]**2)/2
        cells[i] += edge
        cells[i+1] += edge
    return cells


def initial(mass, radius, theta, x, empty=False):
    y, v = np.zeros((len(x), 10)), np.zeros((len(x), 10))
    residual, tangent, min_eig = 0., 0., float('inf')
    for i in np.flatnonzero(x < radius):
        profile = np.exp(1-1/(1-(x[i]/radius)**2))
        xx = np.zeros(5, complex)
        target = profile*np.array([0 if empty else np.sqrt(base.N0/(2*mass))*np.exp(1j*theta), .1*AREF])
        for fraction in [.25, .5, .75, 1.]:
            xx[[2, 4]] = fraction*target
            def unpack(h):
                real = np.r_[xx.real, xx.imag]
                real[base.HEAVY] = h
                return real[:5]+1j*real[5:]
            def fun(h):
                g = -base.acceleration(unpack(h), mass)
                return np.r_[g.real, g.imag][base.HEAVY]
            def jac(h):
                return base.real_hessian(unpack(h), mass, CUBIC)[np.ix_(base.HEAVY, base.HEAVY)]
            sol = root(fun, np.r_[xx.real, xx.imag][base.HEAVY], jac=jac, tol=1e-11)
            xx = unpack(sol.x)
            residual = max(residual, float(np.max(np.abs(fun(sol.x))))/(MREF**2*AREF))
        hh = base.real_hessian(xx, mass, CUBIC)
        hheavy = hh[np.ix_(base.HEAVY, base.HEAVY)]
        vv = np.zeros(5, complex)
        vv[2] = -1j*mass*xx[2]
        vr = np.r_[vv.real, vv.imag]
        vr[base.HEAVY] = -np.linalg.solve(hheavy, hh[np.ix_(base.HEAVY, base.LIGHT)]@vr[base.LIGHT])
        tangent = max(tangent, float(np.max(np.abs((hh@vr)[base.HEAVY])))/(MREF**3*AREF))
        min_eig = min(min_eig, float(np.linalg.eigvalsh(hheavy)[0]))
        y[i], v[i] = np.r_[xx.real, xx.imag]/AREF, vr/(MREF*AREF)
    assert max(residual, tangent) < 1e-8 and min_eig > 0, (residual, tangent, min_eig)
    return y, v, {'heavy_stationarity_residual': residual, 'heavy_tangent_residual': tangent,
                  'heavy_Hessian_min_eigenvalue': min_eig}


def cases():
    return [(f'{label}_R{r}_p{phase}', float(m), r, phase*np.pi/4, False)
            for label, m in zip(['plus', 'minus'], base.MASSES)
            for r in [10, 30] for phase in [0, 1]]+[
                (f'empty_R{r}', float(base.MASSES[1]), r, 0., True) for r in [10, 30]]


def simulate(case, dx=.5, dt=.04, outer=140, duration=80):
    started = time.monotonic()
    name, mass, radius, theta, empty = case
    x, vol, face = mesh(dx, outer)
    y, v, init = initial(mass, radius, theta, x, empty)
    core, cut = x < radius/4, round(40/dx)
    e0 = float(energy(y, v, vol, face, mass, CUBIC).sum())
    flux, drift, balance, newton, correction = 0., 0., 0., 0, 0.
    outward, inward = 0., 0.
    trace = []
    final_fields = None
    for k in range(round(duration/dt)+1):
        ee = energy(y, v, vol, face, mass, CUBIC)
        total, outside, higgs = float(ee.sum()), float(ee[cut:].sum()), float(ee[cut:, 4].sum())
        q = float(np.sum(vol[core]*(y[core, 4]**2+y[core, 9]**2))*3/(radius/4)**3)
        drift = max(drift, abs(total/e0-1))
        balance = max(balance, abs(outside-flux)/e0)
        trace.append([k*dt, q, outside/e0, higgs/e0, flux/e0,
                      float(ee[x >= outer-5].sum())/e0])
        if k == round(duration/dt):
            final_fields = np.c_[y, v].tolist()
            break
        yy, vv, iterations, corr = step(y, v, dt, vol, face, mass)
        middle, velocity = (yy+y)/2, (yy-y)/dt
        current = -face[cut-1]*np.dot(middle[cut]-middle[cut-1], velocity[cut]+velocity[cut-1])
        flux += dt*current
        outward += dt*max(0., current)
        inward += dt*max(0., -current)
        newton, correction = max(newton, iterations), max(correction, corr)
        y, v = yy, vv
    tr = np.array(trace)
    up = np.flatnonzero((tr[:-1, 1] < 1) & (tr[1:, 1] >= 1))
    down = np.flatnonzero((tr[:-1, 1] >= 1) & (tr[1:, 1] < 1))
    def crossings(indices):
        return [float(tr[i, 0]+dt*(1-tr[i, 1])/(tr[i+1, 1]-tr[i, 1])) for i in indices]
    late = tr[tr[:, 0] >= min(70, duration), 1]
    return {'case': name, 'dx': dx, 'dt': dt, 'outer': outer, 'duration': duration,
            'mass': mass, 'radius': radius, 'phase': theta, 'empty_source': empty,
            'initialization': init, 'initial_energy_dimensionless_no_4pi': e0,
            'energy_relative_residual': drift, 'boundary_balance_relative_residual': balance,
            'max_Newton_iterations': newton, 'max_last_Newton_correction': correction,
            'Q_peak': float(tr[:, 1].max()), 'Q_peak_time': float(tr[tr[:, 1].argmax(), 0]),
            'Q_late_min': float(late.min()), 'Q_late_max': float(late.max()),
            'late_retention': bool(late.min() >= 1), 'up_crossings': crossings(up),
            'down_crossings': crossings(down), 'outer_tail_max_energy_fraction': float(tr[:, 5].max()),
            'outward_integrated_flux_over_E0': outward/e0,
            'inward_integrated_flux_over_E0': inward/e0,
            'trace_columns': ['tau', 'Q_core', 'E_out_over_E0', 'E_Higgs_out_over_E0',
                              'signed_integrated_flux_over_E0', 'outer_5_energy_over_E0'],
            'trace': trace, 'final_fields_real10_velocity10': final_fields,
            'elapsed_seconds': time.monotonic()-started}


def checks():
    rng = np.random.default_rng(7801)
    mass = float(base.MASSES[1])
    a, b = .01*rng.normal(size=(2, 10))
    ga = sum(local(a+c*(b-a), mass, CUBIC)[0]/2 for c in GAUSS)
    va, vb = (local(z, mass, CUBIC)[2].sum() for z in [a, b])
    chain = abs(vb-va-2*np.dot(ga, b-a))/max(1, abs(va), abs(vb))
    h = local(a, mass, CUBIC)[1]
    reference_h = base.real_hessian(AREF*(a[:5]+1j*a[5:]), mass, CUBIC)/MREF**2
    herror = float(np.max(np.abs(h-reference_h)))/max(1, np.max(np.abs(h)))
    # A random Hamiltonian state exercises all complex components and both flux directions.
    _, vol, face = mesh(.5, 4)
    y, v = .001*rng.normal(size=(2, len(vol), 10))
    ee = energy(y, v, vol, face, mass, CUBIC)
    yy, vv, _, _ = step(y, v, .02, vol, face, mass)
    e1 = energy(yy, vv, vol, face, mass, CUBIC)
    j = 4
    current = -face[j-1]*np.dot(((yy+y)/2)[j]-((yy+y)/2)[j-1], ((yy-y)/.02)[j]+((yy-y)/.02)[j-1])
    edrift = abs(e1.sum()-ee.sum())/ee.sum()
    fluxerror = abs(e1[j:].sum()-ee[j:].sum()-.02*current)/ee.sum()
    # Free spherical solution: r*u = [w0(r-t)+w0(r+t)]/2, odd w0=r*exp(-(r/3)^2).
    free_results = []
    for dx, dt in [(.5, .04), (.25, .02), (.125, .01)]:
        x, vol, face = mesh(dx, 30)
        y, v = np.zeros((len(x), 10)), np.zeros((len(x), 10))
        y[:, 4] = np.exp(-(x/3)**2)
        e0 = energy(y, v, vol, face, mass, CUBIC, True).sum()
        for _ in range(round(8/dt)):
            y, v, _, _ = step(y, v, dt, vol, face, mass, True)
        w = lambda r: r*np.exp(-(r/3)**2)
        exact = (w(x-8)+w(x+8))/(2*x)
        free_results.append({'dx': dx, 'dt': dt,
            'weighted_relative_L2_error': float(np.sqrt(np.sum(vol*(y[:, 4]-exact)**2)/np.sum(vol*exact**2))),
            'max_absolute_error': float(np.max(np.abs(y[:, 4]-exact))),
            'energy_relative_residual': float(abs(energy(y, v, vol, face, mass, CUBIC, True).sum()/e0-1))})
    # Uniform full-action limit, same preparation and force as chapter 77; no spatial term.
    homogeneous = []
    for mass in base.MASSES:
        state, _ = base.initial_state(float(mass), .1, 0.)
        times = np.linspace(0, 4, 201)
        ref = solve_ivp(lambda t, z: base.rhs(t, z, mass), (0, 4), state,
                        t_eval=times, method='DOP853', rtol=2e-10, atol=2e-13)
        assert ref.success
        for dt in [.04, .02, .01]:
            y = np.r_[state[:5].real, state[:5].imag][None, :]
            v = np.r_[state[5:].real, state[5:].imag][None, :]
            sampled = [np.r_[y[0], v[0]]]
            for _ in range(round(4/dt)):
                y, v, _, _ = step(y, v, dt, np.ones(1), np.zeros(0), float(mass))
                sampled.append(np.r_[y[0], v[0]])
            result = np.array(sampled)
            tref = np.arange(len(result))*dt
            rq = np.interp(tref, ref.t, np.abs(ref.y[4])**2)
            q = result[:, 4]**2+result[:, 9]**2
            # Interpolate real/imag only at common actual DOP853 evaluation times.
            stride = max(1, round(.02/dt))
            inds = np.rint(tref[::stride]/.02).astype(int)
            rr = np.c_[ref.y[:5].real.T, ref.y[:5].imag.T, ref.y[5:].real.T, ref.y[5:].imag.T][inds]
            dd = result[::stride]-rr
            homogeneous.append({'mass': float(mass), 'dt': dt, 'duration': 4,
                'Q_max_scaled_difference': float(np.max(np.abs(q-rq)/np.maximum(1, np.abs(rq)))),
                'heavy_field_max_absolute_error_over_Aref': float(np.max(np.abs(dd[:, base.HEAVY]))),
                'heavy_velocity_max_absolute_error_over_Mref_Aref': float(np.max(np.abs(dd[:, 10+base.HEAVY]))),
                'all_fields_max_absolute_error_over_Aref': float(np.max(np.abs(dd[:, :10]))),
                'reference_nfev': ref.nfev})
    out = {'AVF_chain_rule_scaled_error': float(chain), 'Hessian_relative_error': herror,
           'random_full_step_energy_relative_error': float(edrift), 'random_boundary_flux_relative_error': float(fluxerror),
           'free_spherical_wave': free_results, 'homogeneous_DOP853': homogeneous}
    assert max(chain, herror, edrift, fluxerror) < 1e-10, out
    assert free_results[2]['weighted_relative_L2_error'] < free_results[1]['weighted_relative_L2_error']/3
    assert max(r['Q_max_scaled_difference'] for r in homogeneous if r['dt'] == .02) < .005
    return out


def compare(coarse, fine):
    c, f = np.array(coarse['trace']), np.array(fine['trace'])
    aligned = np.column_stack([np.interp(c[:, 0], f[:, 0], f[:, j]) for j in range(6)])
    errors = np.max(np.abs(c[:, 1:5]-aligned[:, 1:5])/np.maximum(1, np.abs(aligned[:, 1:5])), axis=0)
    return {'observable_max_scaled_differences': dict(zip(coarse['trace_columns'][1:5], errors.tolist())),
            'energy_and_flux_pass': max(coarse['energy_relative_residual'], fine['energy_relative_residual'],
                coarse['boundary_balance_relative_residual'], fine['boundary_balance_relative_residual']) <= 1e-6,
            'observable_convergence_pass': bool(errors.max() <= .005),
            'retention_agrees': coarse['late_retention'] == fine['late_retention']}


def save(data, path=OUTPUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--case', choices=[c[0] for c in cases()])
    parser.add_argument('--dx', type=float, default=.5)
    parser.add_argument('--dt', type=float, default=.04)
    parser.add_argument('--outer', type=float, default=140)
    parser.add_argument('--duration', type=float, default=80)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = {'schema': 'CE-UR4-D4-v1', **provenance(), 'scientific_success': False,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'numba': numba.__version__, 'sympy': sympy.__version__},
            'full_joint_rmse': None, 'runs': {}, 'comparisons': {}}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding='utf-8'))
        for key, value in provenance().items():
            assert existing[key] == value, f'Cannot resume changed provenance: {key}'
        data = existing
    if args.checks:
        data['checks'] = checks()
        save(data, args.output)
        print(json.dumps(data['checks']), flush=True)
    tasks = []
    if args.suite:
        assert 'checks' in data, 'Run --checks first'
        tasks = [(c, dx, dt, 140, 80) for c in cases() for dx, dt in [(.5, .04), (.25, .02)]]
        tasks.append((next(c for c in cases() if c[0] == 'minus_R30_p0'), .25, .02, 160, 80))
    elif args.case:
        tasks = [(next(c for c in cases() if c[0] == args.case), args.dx, args.dt, args.outer, args.duration)]
    def key(t):
        return f'{t[0][0]}_dx{t[1]:g}_dt{t[2]:g}_L{t[3]:g}_T{t[4]:g}'
    tasks = [t for t in tasks if key(t) not in data['runs']]
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(simulate, *t): key(t) for t in tasks}
            for future in as_completed(futures):
                name = futures[future]
                run = future.result()
                data['runs'][name] = run
                save(data, args.output)
                print(json.dumps({name: {k: run[k] for k in ['Q_peak', 'Q_late_min', 'Q_late_max',
                    'late_retention', 'energy_relative_residual', 'boundary_balance_relative_residual', 'elapsed_seconds']}}), flush=True)
    for c in cases():
        relevant = [r for r in data['runs'].values() if r['case'] == c[0] and r['outer'] == 140 and r['duration'] == 80]
        relevant.sort(key=lambda r: r['dx'], reverse=True)
        if len(relevant) >= 2:
            data['comparisons'][c[0]] = compare(relevant[-2], relevant[-1])
    outer_runs = [r for r in data['runs'].values() if r['case'] == 'minus_R30_p0' and r['dx'] == .25 and r['dt'] == .02 and r['duration'] == 80]
    if len(outer_runs) == 2:
        data['outer_boundary_comparison'] = compare(*sorted(outer_runs, key=lambda r: r['outer']))
    if args.checks or tasks:
        save(data, args.output)
    print(json.dumps(data['comparisons']), flush=True)


if __name__ == '__main__':
    main()
