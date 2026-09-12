"""Exact invariant-coordinate linear solve for the empty-source D4 control.

For real initial z,t,q and s=u=0, the omitted seven REAL coordinates have
identically zero force. Both nonzero heavy fields z,t still evolve by AVF.
Only the Newton linear system is reduced; no heavy minimum is imposed.
The original full-field initialization, flux, and output are reused. The energy
polynomial is evaluated directly on the same invariant coordinates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.linalg import solve_banded
import ce_record_radiation as model
import ce_record_backreaction as base

ACTIVE = np.array([0, 1, 4])
INACTIVE = np.array([2, 3, 5, 6, 7, 8, 9])
A, B, D = base.A, base.B, base.D
AREF, MREF = base.AREF, base.MREF
FULL_STEP = model.step
FULL_ENERGY = model.energy


@njit(cache=True)
def reduced_local(y):
    z, t, q = y*AREF
    f0 = -z+A*z*z+3*A*t*t-B*q*q/2
    f1 = -5*t+6*A*z*t-D*q*q/2
    f2 = -(B*z+D*t)*q
    w00, w11, w22 = -1+2*A*z, -5+6*A*z, -B*z-D*t
    w01, w02, w12 = 6*A*t, -B*q, -D*q
    g = np.array([w00*f0+w01*f1+w02*f2,
                  w01*f0+w11*f1+w12*f2,
                  w02*f0+w12*f1+w22*f2])/(AREF*MREF**2)
    h = np.empty((3, 3))
    h[0, 0] = w00*w00+w01*w01+w02*w02+2*A*f0
    h[1, 1] = w01*w01+w11*w11+w12*w12+6*A*f0
    h[2, 2] = w02*w02+w12*w12+w22*w22-B*f0-D*f1
    h[0, 1] = h[1, 0] = w01*(w00+w11)+w02*w12+6*A*f1
    h[0, 2] = h[2, 0] = w02*(w00+w22)+w01*w12-B*f2
    h[1, 2] = h[2, 1] = w01*w02+w12*(w11+w22)-D*f2
    return g, h/MREF**2


@njit(cache=True)
def energy(y, v, vol, face, mass, cubic, free=False):
    assert not free
    cells = np.zeros((len(vol), 5))
    for i in range(len(vol)):
        for a in INACTIVE:
            assert y[i, a] == 0 and v[i, a] == 0
        z, t, q = AREF*y[i, 0], AREF*y[i, 1], AREF*y[i, 4]
        f0 = -z+A*z*z+3*A*t*t-B*q*q/2
        f1 = -5*t+6*A*z*t-D*q*q/2
        f2 = -(B*z+D*t)*q
        cells[i, 0] = vol[i]*(v[i, 0]**2+(f0/(AREF*MREF))**2)
        cells[i, 1] = vol[i]*(v[i, 1]**2+(f1/(AREF*MREF))**2)
        cells[i, 4] = vol[i]*(v[i, 4]**2+(f2/(AREF*MREF))**2)
    for i in range(len(face)):
        for a in ACTIVE:
            edge = face[i]*(y[i+1, a]-y[i, a])**2/2
            cells[i, a] += edge
            cells[i+1, a] += edge
    return cells


@njit(cache=True)
def assembly(y0, v0, end, dt, vol, face):
    n = len(vol)
    ab, residual = np.zeros((7, 3*n)), np.empty((n, 3))
    mid = (y0+end)/2
    for i in range(n):
        avg, jac = np.zeros(3), np.zeros((3, 3))
        for c in model.GAUSS:
            g, h = reduced_local(y0[i]+c*(end[i]-y0[i]))
            avg += g/2
            jac += c*h/2
        left = face[i-1]/vol[i] if i > 0 else 0.
        right = face[i]/vol[i] if i < n-1 else 0.
        for a in range(3):
            lap = -(left+right)*mid[i, a]
            if i > 0:
                lap += left*mid[i-1, a]
                ab[6, (i-1)*3+a] = -dt*dt*left/4
            if i < n-1:
                lap += right*mid[i+1, a]
                ab[0, (i+1)*3+a] = -dt*dt*right/4
            residual[i, a] = end[i, a]-y0[i, a]-dt*v0[i, a]-dt*dt*(lap-avg[a])/2
            for b in range(3):
                ab[3+a-b, i*3+b] = dt*dt*jac[a, b]/2
            ab[3, i*3+a] += 1+dt*dt*(left+right)/4
    return ab, residual


def step(y, v, dt, vol, face, mass, free=False):
    assert not free
    assert np.count_nonzero(y[:, INACTIVE]) == np.count_nonzero(v[:, INACTIVE]) == 0
    yy, vv = y[:, ACTIVE], v[:, ACTIVE]
    end = yy+dt*vv
    for iteration in range(12):
        ab, residual = assembly(yy, vv, end, dt, vol, face)
        delta = solve_banded((3, 3), ab, -residual.ravel(), overwrite_ab=True,
                             overwrite_b=True, check_finite=False).reshape(end.shape)
        end += delta
        correction = float(np.max(np.abs(delta)))
        if correction < 2e-13:
            break
    else:
        raise RuntimeError(f'Newton failed: {correction}')
    assert np.isfinite(end).all()
    y1, v1 = np.zeros_like(y), np.zeros_like(v)
    y1[:, ACTIVE], v1[:, ACTIVE] = end, 2*(end-yy)/dt-vv
    return y1, v1, iteration+1, correction


def simulate(dx, dt, duration=80):
    case = next(c for c in model.cases() if c[0] == 'empty_R10')
    # A process-local callback substitution reuses the unchanged full-state
    # recorder; restore it immediately, including on an exception.
    assert model.step is FULL_STEP and model.energy is FULL_ENERGY
    model.step, model.energy = step, energy
    try:
        run = model.simulate(case, dx, dt, 140, duration)
    finally:
        model.step, model.energy = FULL_STEP, FULL_ENERGY
    run['exact_empty_control_driver_sha256'] = model.digest(__file__)
    return run


def checks(reference):
    rng = np.random.default_rng(7811)
    force, hessian, zero, step_error, energy_error = 0., 0., 0., 0., 0.
    for _ in range(4):
        y = .01*rng.normal(size=3)
        full = np.zeros(10)
        full[ACTIVE] = y
        g, h, _ = model.local(full, base.MASSES[1], model.CUBIC)
        gr, hr = reduced_local(y)
        force = max(force, float(np.max(abs(g[ACTIVE]-gr)))/max(1, np.max(abs(g))))
        hessian = max(hessian, float(np.max(abs(h[np.ix_(ACTIVE, ACTIVE)]-hr)))/max(1, np.max(abs(h))))
        zero = max(zero, float(np.max(abs(g[INACTIVE]))), float(np.max(abs(h[np.ix_(INACTIVE, ACTIVE)]))))
        _, vol, face = model.mesh(.5, 4)
        yy, vv = np.zeros((len(vol), 10)), np.zeros((len(vol), 10))
        yy[:, ACTIVE], vv[:, ACTIVE] = rng.normal(size=(2, len(vol), 3))*.01
        actual = energy(yy, vv, vol, face, base.MASSES[1], model.CUBIC)
        expected = FULL_ENERGY(yy, vv, vol, face, base.MASSES[1], model.CUBIC)
        energy_error = max(energy_error, float(np.max(abs(actual-expected)))/max(1, np.max(abs(expected))))
    x, vol, face = model.mesh(.5, 140)
    y, v, _ = model.initial(float(base.MASSES[1]), 10, 0., x, True)
    for _ in range(10):
        yr, vr, _, _ = step(y, v, .04, vol, face, float(base.MASSES[1]))
        yf, vf, _, _ = FULL_STEP(y, v, .04, vol, face, float(base.MASSES[1]))
        step_error = max(step_error, float(np.max(abs(yr-yf))), float(np.max(abs(vr-vf))))
        y, v = yr, vr
    comparison_run = simulate(.5, .04)
    old = reference['runs']['empty_R10_dx0.5_dt0.04_L140_T80']
    comparison = model.compare(old, comparison_run)
    trace_error = max(comparison['observable_max_scaled_differences'].values())
    fields_error = float(np.max(abs(np.array(old['final_fields_real10_velocity10'])-
                                        np.array(comparison_run['final_fields_real10_velocity10']))))
    assert max(force, hessian, zero, step_error, trace_error, fields_error, energy_error) < 1e-8
    return {'force_relative_error': force, 'Hessian_relative_error': hessian,
            'inactive_force_and_mixed_Hessian': zero, 'full_step_max_absolute_difference': step_error,
            'full_tau80_trace_max_scaled_difference': trace_error,
            'full_tau80_final_field_velocity_max_absolute_difference': fields_error,
            'independent_full_energy_max_scaled_difference': energy_error,
            'reference_run_sha256': __import__('hashlib').sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dx', type=float, default=.03125)
    parser.add_argument('--dt', type=float, default=.0025)
    parser.add_argument('--admit', action='store_true', help='Merge a checked result after all other writers have ended')
    args = parser.parse_args()
    path = Path(__file__).with_suffix('.json')
    reference = json.loads(model.OUTPUT.read_text(encoding='utf-8'))
    assert all(reference[k] == v for k, v in model.provenance().items())
    provenance = {**model.provenance(), 'empty_control_source_sha256': model.digest(__file__)}
    if args.admit:
        result = json.loads(path.read_text(encoding='utf-8'))
        assert all(result[k] == v for k, v in provenance.items())
        run = result['run']
        key = f'empty_R10_dx{run["dx"]:g}_dt{run["dt"]:g}_L140_T80'
        reference['runs'][key] = run
        relevant = sorted([r for r in reference['runs'].values() if r['case'] == 'empty_R10'], key=lambda r: r['dx'], reverse=True)
        previous = reference['comparisons']['empty_R10']
        reference.setdefault('earlier_comparisons', {}).setdefault('empty_R10', []).append(previous)
        reference['comparisons']['empty_R10'] = model.compare(relevant[-2], relevant[-1])
        model.save(reference)
        print(json.dumps(reference['comparisons']['empty_R10']), flush=True)
        return
    result = {**provenance, 'scientific_success': False, 'full_joint_rmse': None,
              'checks': checks(reference)}
    print(json.dumps(result['checks']), flush=True)
    result['run'] = simulate(args.dx, args.dt)
    model.save(result, path)
    print(json.dumps({k: result['run'][k] for k in ['dx', 'dt', 'Q_peak', 'energy_relative_residual',
        'boundary_balance_relative_residual', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
