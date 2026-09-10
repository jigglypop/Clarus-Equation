"""Chapter 70: supplied N=1 protection and a complete threshold gate.

No soft-breaking mechanism, gravity, autonomous record, or joint empirical
success is claimed. The gauge exclusion is only for the registered hierarchy.
"""
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
import scipy
import sympy as sy

from ce_simple_group_record import basis


R = sy.Rational
N = 82  # Sigma24 + record24 + record24 + H5 + Hbar5.
TOL = 1e-8
FD_TOL = 1e-5


def error(x):
    return float(np.max(np.abs(x)))


def build_model(m0=0.):
    t, y = basis()
    adj = np.array([[[2*np.trace(a@(gen@b-b@gen)) for b in t] for a in t] for gen in t])
    d = np.array([[[2*np.trace(a@(b@c+c@b)).real for c in t] for b in t] for a in t])
    reps = np.zeros((24, N, N), complex)
    for start in [0, 24, 48]:
        reps[:, start:start+24, start:start+24] = adj
    reps[:, 72:77, 72:77] = t
    reps[:, 77:82, 77:82] = -t.conj()
    mass = np.zeros((N, N), complex)
    mass[:24, :24] = np.eye(24)
    k = m0*(5*np.eye(2)+np.pi/2*np.array([[0, 1], [1, 0]]))
    mass[24:72, 24:72] = np.kron(k+6*np.eye(2), np.eye(24))
    mass[72:77, 77:82] = 3*np.eye(5)
    mass[77:82, 72:77] = 3*np.eye(5)
    cubic = np.zeros((N, N, N), complex)
    cubic[:24, :24, :24] = d
    for start in [24, 48]:
        cubic[:24, start:start+24, start:start+24] = d
        cubic[start:start+24, :24, start:start+24] = d
        cubic[start:start+24, start:start+24, :24] = d
    for a in range(24):
        for i in range(5):
            for j in range(5):
                for inds in itertools.permutations([a, 77+i, 72+j]):
                    cubic[inds] = t[a, i, j]
    vacuum = np.zeros(N, complex)
    sigma0 = np.diag([2, 2, 2, -3, -3])
    vacuum[:24] = np.array([2*np.trace(a@sigma0) for a in t])
    assert error(cubic-cubic.transpose(1, 0, 2)) < TOL
    assert error(cubic-cubic.transpose(0, 2, 1)) < TOL
    return t, y, reps, mass, cubic, vacuum


def f_terms(z, mass, cubic):
    wz = mass+np.einsum('ijk,k->ij', cubic, z, optimize=True)
    f = (mass+wz)@z/2
    return f, wz


def component_matrices(z, reps, mass, cubic):
    f, wz = f_terms(z, mass, cubic)
    # D vanishes on the registered real adjoint backgrounds. The implementation
    # retains D and its Hessian for independent generic-gradient verification.
    row = np.einsum('i,aij->aj', z.conj(), reps, optimize=True)
    d = np.einsum('ai,i->a', row, z).real
    dj = np.sqrt(2)*np.concatenate([row.real, -row.imag], axis=1)
    aa = wz.conj().T@wz
    bb = np.einsum('k,kij->ij', f.conj(), cubic, optimize=True)
    hs = np.block([[(aa+bb).real, -aa.imag-bb.imag],
                   [aa.imag-bb.imag, (aa-bb).real]])
    hs += dj.T@dj
    if np.max(np.abs(d)) > 1e-14:
        for da, rep in zip(d, reps):
            hs += da*np.block([[rep.real, -rep.imag], [rep.imag, rep.real]])
    mixing = np.sqrt(2)*row
    mf = np.block([[wz, mixing.T], [mixing, np.zeros((24, 24))]])
    gv = 2*(row.conj()@row.T).real
    return hs, mf, gv, f, d


def potential_and_gradient(z, reps, mass, cubic):
    f, wz = f_terms(z, mass, cubic)
    row = np.einsum('i,aij->aj', z.conj(), reps, optimize=True)
    d = np.einsum('ai,i->a', row, z).real
    potential = np.vdot(f, f).real+np.dot(d, d)/2
    wirtinger = wz.conj().T@f+np.einsum('a,ai->i', d, row.conj())
    grad = np.sqrt(2)*np.concatenate([wirtinger.real, wirtinger.imag])
    return float(potential), grad


def eigen_hist(values, tol=1e-7):
    hist = {}
    for value in values:
        nearest = int(round(float(value)))
        assert abs(value-nearest) < tol
        hist[str(nearest)] = hist.get(str(nearest), 0)+1
    return hist


def action_checks(model):
    t, _, reps, mass, cubic, vacuum = model
    rng = np.random.default_rng(7001)
    errors = []
    for _ in range(3):
        z = .1*(rng.normal(size=N)+1j*rng.normal(size=N))
        sigma = np.einsum('a,aij->ij', z[:24], t)
        xs = [np.einsum('a,aij->ij', z[a:a+24], t) for a in [24, 48]]
        direct = np.trace(sigma@sigma)+(2/3)*np.trace(sigma@sigma@sigma)
        direct += sum(6*np.trace(x@x)+2*np.trace(sigma@x@x) for x in xs)
        direct += z[77:82]@(3*np.eye(5)+sigma)@z[72:77]
        tensor = z@mass@z/2+np.einsum('ijk,i,j,k', cubic, z, z, z, optimize=True)/6
        errors.append(abs(direct-tensor))
        # Finite gauge transformation acts in all three adjoints and 5+bar5.
        u = expm(.23j*(reps[2]+reps[15]+reps[23]))
        zu = u@z
        transformed = zu@mass@zu/2+np.einsum('ijk,i,j,k', cubic, zu, zu, zu, optimize=True)/6
        errors.append(abs(transformed-tensor))
    hs, mf, gv, f, d = component_matrices(vacuum, reps, mass, cubic)
    assert max(errors+[error(f), error(d)]) < TOL
    scalar = eigen_hist(np.linalg.eigvalsh(hs))
    fermion = eigen_hist(np.linalg.eigvalsh(mf.conj().T@mf))
    vector = eigen_hist(np.linalg.eigvalsh(gv))
    assert scalar == {'0': 32, '1': 2, '16': 4, '25': 82, '50': 12, '100': 32}
    assert fermion == {'0': 22, '1': 1, '16': 2, '25': 41, '50': 24, '100': 16}
    assert vector == {'0': 12, '50': 12}
    assert np.linalg.eigvalsh(hs)[0] > -TOL
    return {'superpotential_trace_and_gauge_error': float(max(errors)),
            'vacuum_F_error': error(f), 'vacuum_D_error': error(d),
            'scalar_mass_squared_histogram': scalar, 'weyl_mass_squared_histogram': fermion,
            'vector_mass_squared_histogram': vector,
            'MX_squared': 50, 'canonical_adjoint_kinetic': '2 Tr Phi_dagger Phi',
            'mass_ratios': {'Sigma_octet_and_triplet': '1/sqrt(2)', 'Sigma_singlet': '1/(5 sqrt(2))',
                           'H_color_pair': '1/sqrt(2)', 'record_octet': 'sqrt(2)+m_pm/MX',
                           'record_broken': '1/sqrt(2)+m_pm/MX',
                           'record_singlet': '4/(5 sqrt(2))+m_pm/MX', 'record_triplet': 'm_pm/MX'},
            'decoupled_SM_matter_chiral_pairs': 45,
            'SM_matter_zero_X2_reason': 'H=Hbar=0, D=0, no Sigma or record Yukawa to SM matter'}


def trace_function_derivatives(a0, a1, a2, scale, linear_only=False):
    """Coefficients of Tr[A(x)^2 (log(A(x)/scale^2)-3/2)]."""
    values, u = np.linalg.eigh(a0)
    values[np.abs(values) < 1e-8] = 0
    assert values.min() >= 0
    b1, b2 = u.conj().T@a1@u, u.conj().T@a2@u
    fp = np.zeros_like(values)
    nz = values > 0
    fp[nz] = 2*values[nz]*(np.log(values[nz]/scale**2)-1)
    linear = float(np.dot(fp, np.diag(b1)).real)
    if linear_only:
        return linear, None
    divided = np.zeros((len(values), len(values)))
    for i, vi in enumerate(values):
        for j, vj in enumerate(values):
            if vi == 0 and vj == 0:
                assert abs(b1[i, j]) < TOL
            elif abs(vi-vj) < 1e-8:
                divided[i, j] = 2*np.log(vi/scale**2)
            else:
                divided[i, j] = (fp[i]-fp[j])/(vi-vj)
    quadratic = float(np.dot(fp, np.diag(b2)).real+np.sum(np.abs(b1)**2*divided)/2)
    return linear, quadratic


def exact_trace_coefficients(flat):
    """Independent sparse exact matrices from diagonal commutators and Jordan blocks."""
    sparse = lambda n, m: sy.MutableSparseMatrix(n, m, {})
    w0, w1 = sparse(N, N), sparse(N, N)
    for i, value in enumerate([5]*8+[-5]*3+[-1]+[0]*12):
        w0[i, i] = value
    for start in [24, 48]:
        for i, value in enumerate([10]*8+[0]*3+[4]+[5]*12):
            w0[start+i, start+i] = value
    for i in range(3):
        w0[72+i, 77+i] = w0[77+i, 72+i] = 5
    je = sparse(24, 24)
    je[10, 11] = je[11, 10] = sy.sqrt(R(3, 5))
    for i, value in enumerate([R(1, 2), R(1, 2), -R(1, 2), -R(1, 2)]*3, start=12):
        je[i, i] = value
    amplitudes = [R(1, 2), sy.I/2] if flat else [1/sy.sqrt(2), 0]
    for start, amplitude in zip([24, 48], amplitudes):
        for (i, j), value in je.todok().items():
            w1[i, start+j] = w1[start+j, i] = amplitude*value
    row0, row1 = sparse(24, N), sparse(24, N)
    for a in range(12, 24, 2):
        row0[a, a+1], row0[a+1, a] = 5*sy.I, -5*sy.I
    for start, amplitude in zip([24, 48], amplitudes):
        amp = sy.conjugate(amplitude)
        row1[8, start+9], row1[9, start+8] = sy.I*amp, -sy.I*amp
        for pair, a in enumerate(range(12, 24, 2)):
            weight = -R(1, 2) if pair % 2 == 0 else R(1, 2)
            row1[a, start+a+1] = sy.I*weight*amp
            row1[a+1, start+a] = -sy.I*weight*amp

    def realify(a, holomorphic=False):
        out = sparse(2*N, 2*N)
        for (i, j), value in a.todok().items():
            re, im = sy.re(value), sy.im(value)
            out[i, j], out[i, N+j] = re, -im
            out[N+i, j] = -im if holomorphic else im
            out[N+i, N+j] = -re if holomorphic else re
        return out

    def dj(row):
        out = sparse(24, 2*N)
        for (i, j), value in row.todok().items():
            out[i, j], out[i, N+j] = sy.sqrt(2)*sy.re(value), -sy.sqrt(2)*sy.im(value)
        return out

    j0, j1 = dj(row0), dj(row1)
    h0 = realify(w0.conjugate().T*w0)+j0.T*j0
    h1 = realify(w0.conjugate().T*w1+w1.conjugate().T*w0)+j0.T*j1+j1.T*j0
    h2 = realify(w1.conjugate().T*w1)+j1.T*j1
    if not flat:
        bb = sparse(N, N)
        for start in [0, 24, 48]:
            for i, value in enumerate([4]*8+[-6]*3+[-2]+[-1]*12):
                bb[start+i, start+i] = -R(value, 40)
        for i, value in enumerate([-R(1, 20)]*3+[R(3, 40)]*2):
            bb[72+i, 77+i] = bb[77+i, 72+i] = value
        h2 += realify(bb, holomorphic=True)

    def fermion(w, row):
        out = sparse(N+24, N+24)
        for inds, value in w.todok().items():
            out[inds] = value
        for (a, i), value in row.todok().items():
            out[N+a, i] = out[i, N+a] = sy.sqrt(2)*value
        return out

    mf0, mf1 = fermion(w0, row0), fermion(w1, row1)
    a0 = mf0.conjugate().T*mf0
    a1 = mf0.conjugate().T*mf1+mf1.conjugate().T*mf0
    a2 = mf1.conjugate().T*mf1
    gv0 = 2*(row0.conjugate()*row0.T).applyfunc(sy.re)
    gv2 = 2*(row1.conjugate()*row1.T).applyfunc(sy.re)

    def trprod(a, b):
        return sum(value*b[j, i] for (i, j), value in a.todok().items())

    cs = sy.simplify(trprod(h1, h1)+2*trprod(h0, h2))
    cf = sy.simplify(trprod(a1, a1)+2*trprod(a0, a2))
    cv = sy.simplify(6*trprod(gv0, gv2))
    return cs, cf, cv


def protection_checks(model, flat=False):
    _, _, reps, mass, cubic, vacuum = model
    direction = np.zeros(N, complex)
    if flat:
        direction[34], direction[58] = .5, .5j
    else:
        direction[34] = 1/np.sqrt(2)
    h0, mf0, gv0, _, _ = component_matrices(vacuum, reps, mass, cubic)
    hp, mfp, gvp, _, _ = component_matrices(vacuum+direction, reps, mass, cubic)
    hm, mfm, gvm, _, _ = component_matrices(vacuum-direction, reps, mass, cubic)
    h1, h2 = (hp-hm)/2, (hp+hm)/2-h0
    mf1 = (mfp-mfm)/2
    assert error((mfp+mfm)/2-mf0) < TOL
    gv2 = (gvp+gvm)/2-gv0
    af0 = mf0.conj().T@mf0
    af1 = mf0.conj().T@mf1+mf1.conj().T@mf0
    af2 = mf1.conj().T@mf1
    cs = float(np.trace(h1@h1+2*h0@h2).real)
    cf = float(np.trace(af1@af1+2*af0@af2).real)
    cv = float(6*np.trace(gv0@gv2).real)
    total = cs+cv-2*cf
    if flat:
        assert abs(total) < TOL
        for x in [0., .2, 1.]:
            *_, fx, dx = component_matrices(vacuum+x*direction, reps, mass, cubic)
            assert max(error(fx), error(dx)) < TOL
    # Independent exact sparse block construction avoids rounding to fractions.
    exacts = exact_trace_coefficients(flat)
    assert max(abs(float(ex)-num) for ex, num in zip(exacts, [cs, cf, cv])) < TOL
    if flat:
        assert exacts[0]+exacts[2]-2*exacts[1] == 0
    fd_errors = []
    potential_errors = []
    rng = np.random.default_rng(7002)
    for x in [0., .2]:
        bg = vacuum+x*direction

        def fd_hessian(step):
            out = np.empty((2*N, 2*N))
            for j in range(2*N):
                dz = np.zeros(N, complex)
                dz[j % N] = step/np.sqrt(2)*(1 if j < N else 1j)
                out[:, j] = (potential_and_gradient(bg+dz, reps, mass, cubic)[1]
                              -potential_and_gradient(bg-dz, reps, mass, cubic)[1])/(2*step)
            return out

        fd = (4*fd_hessian(.0125)-fd_hessian(.025))/3
        fd_errors.append(error(fd-(h0+x*h1+x*x*h2)))
        for _ in range(4):
            real_dir = rng.normal(size=2*N)
            real_dir /= np.linalg.norm(real_dir)
            complex_dir = (real_dir[:N]+1j*real_dir[N:])/np.sqrt(2)
            def derivative(step):
                return (potential_and_gradient(bg+step*complex_dir, reps, mass, cubic)[0]
                        -potential_and_gradient(bg-step*complex_dir, reps, mass, cubic)[0])/(2*step)
            numeric = (4*derivative(.025)-derivative(.05))/3
            potential_errors.append(abs(numeric-real_dir@potential_and_gradient(bg, reps, mass, cubic)[1]))
    assert max(fd_errors+potential_errors) < TOL

    def traces(x, finite=False, scale=1.):
        hs, mf, gv, _, _ = component_matrices(vacuum+x*direction, reps, mass, cubic)
        ff = mf.conj().T@mf
        if not finite:
            return float((np.trace(hs@hs)+3*np.trace(gv@gv)-2*np.trace(ff@ff)).real)
        # All spin constants are 3/2 in the supersymmetry-preserving DR scheme.
        # Tiny/negative off-shell eigenvalues enter the real part of the log.
        def f(vals):
            vals = np.asarray(vals)
            good = np.abs(vals) > 1e-14
            return np.sum(vals[good]**2*(np.log(np.abs(vals[good])/scale**2)-1.5))
        return float((f(np.linalg.eigvalsh(hs))+3*f(np.linalg.eigvalsh(gv))
                      -2*f(np.linalg.eigvalsh(ff)))/(64*np.pi**2))

    def quadratic(fn, h):
        return (fn(h)+fn(-h)-2*fn(0))/(2*h*h)
    str_coeff = (4*quadratic(traces, .05)-quadratic(traces, .1))/3
    assert abs(str_coeff-total) < FD_TOL
    # Off-shell F is O(x^2), so x^4 log x need not vanish. Record convergence,
    # rather than asserting that the whole effective potential is zero.
    finite_rows = []
    for scale in [1., np.sqrt(50)]:
        vals = []
        for step in [.08, .04, .02]:
            fn = lambda x: traces(x, finite=True, scale=scale)
            vals.append({'step': step, 'quadratic_estimator': quadratic(fn, step)})
        spectral = (trace_function_derivatives(h0, h1, h2, scale)[1]
                    +3*trace_function_derivatives(gv0, np.zeros_like(gv0), gv2, scale)[1]
                    -2*trace_function_derivatives(af0, af1, af2, scale)[1])/(64*np.pi**2)
        if flat:
            assert abs(spectral) < TOL
            assert max(abs(v['quadratic_estimator']) for v in vals) < FD_TOL
        finite_rows.append({'scale_over_V': scale, 'spectral_quadratic_coefficient': spectral, 'estimates': vals})
    tadpole_rows = []
    if not flat:
        sigma_direction = np.zeros(N, complex)
        sigma_direction[11] = 1
        p = component_matrices(vacuum+.1*sigma_direction, reps, mass, cubic)
        m = component_matrices(vacuum-.1*sigma_direction, reps, mass, cubic)
        dh = (p[0]-m[0])/.2
        df = (p[1].conj().T@p[1]-m[1].conj().T@m[1])/.2
        dg = (p[2]-m[2])/.2
        c = np.sqrt(3/5)/4
        # At quadratic order in x the relaxed path adds c times the tadpole.
        str_tadpole = float((2*np.trace(h0@dh)+6*np.trace(gv0@dg)-4*np.trace(af0@df)).real)
        assert abs(total+c*str_tadpole) < TOL
        for finite_row in finite_rows:
            scale = finite_row['scale_over_V']
            tadpole = (trace_function_derivatives(h0, dh, np.zeros_like(h0), scale, True)[0]
                       +3*trace_function_derivatives(gv0, dg, np.zeros_like(gv0), scale, True)[0]
                       -2*trace_function_derivatives(af0, df, np.zeros_like(af0), scale, True)[0])/(64*np.pi**2)
            relaxed = finite_row['spectral_quadratic_coefficient']+c*tadpole
            assert abs(relaxed) < TOL
            tadpole_rows.append({'scale_over_V': scale, 'one_loop_singlet_tadpole': tadpole,
                                 'fixed_slice_quadratic': finite_row['spectral_quadratic_coefficient'],
                                 'tree_response_c': c, 'relaxed_quadratic': relaxed})
    return {'scalar_Hessian_dimension': 2*N, 'fermion_Weyl_dimension': N+24,
            'scalar_X2_coefficient': str(exacts[0]), 'fermion_unweighted_X2_coefficient': str(exacts[1]),
            'vector_weighted_X2_coefficient': str(exacts[2]),
            'supertrace_X2_exact': str(exacts[0]+exacts[2]-2*exacts[1]),
            'supertrace_X2_numeric_error': abs(total-float(exacts[0]+exacts[2]-2*exacts[1])),
            'supertrace_quartic_difference_error': abs(str_coeff-total),
            'full_Hessian_gradient_difference_error': max(fd_errors),
            'independent_potential_direction_error': max(potential_errors),
            'finite_DR_potential_small_field_check': finite_rows,
            'singlet_tadpole_and_relaxed_path': tadpole_rows,
            'protected_quadratic_term': 'CANCELS_ON_F_D_FLAT_PATH' if flat else 'FIXED_SLICE_NONZERO_RELAXED_PATH_CANCELS',
            'whole_off_shell_potential_vanishes': False, 'soft_broken_protection_proved': False}


def free_record():
    t, y, reps, mass, cubic, vacuum = build_model(m0=1.)
    _, w = f_terms(vacuum, mass, cubic)
    rec = w[24:72, 24:72].real
    evals, vec = np.linalg.eigh(rec)
    assert evals[0] > 0
    projection = np.kron(np.diag([0, 1]), np.eye(24))
    charge = reps[10, 24:72, 24:72]+reps[11, 24:72, 24:72]/np.sqrt(3/5)
    initial = np.zeros(48, complex)
    initial[10] = 1
    kg_map = vec@np.diag(1/np.sqrt(2*evals))@vec.T
    errors = []
    for time in [0., .25, .5, 1., 1.5, 2.]:
        u = expm(-1j*rec*time)
        psi = u@initial
        phi = kg_map@psi
        dot_phi = -1j*rec@phi
        kg = 1j*(np.vdot(phi, dot_phi)-np.vdot(dot_phi, phi))
        errors += [abs(np.vdot(psi, projection@psi).real-np.sin(np.pi*time/2)**2),
                   abs(np.vdot(psi, psi)-1), abs(kg-1), abs(np.vdot(psi, rec@psi)-5),
                   error(charge@psi), error(u.conj().T@projection@u+u.conj().T@(np.eye(48)-projection)@u-np.eye(48))]
    assert max(errors) < TOL
    return {'free_rest_record_error': float(max(errors)), 'mass_minimum': float(evals[0]),
            'autonomous_actual_record': False, 'instrument_CP': 'Kraus P_a U; positive outer-product Choi'}


def holomorphic_running(model):
    _, _, reps, _, cubic, _ = model
    # SM Yukawas are unspecified, but have no Sigma/X leg and do not enter
    # these superfield anomalous dimensions at one loop.
    gamma = .5*np.einsum('ikl,jkl->ij', cubic, cubic.conj(), optimize=True)
    gamma -= 2*np.einsum('aij,ajk->ik', reps, reps, optimize=True)
    gamma_sigma_error = error(gamma[:24, :24]+16*np.eye(24)/5)
    gamma_record_error = error(gamma[24:72, 24:72]+29*np.eye(48)/5)
    cross_error = error(gamma[:24, 24:72])
    assert max(gamma_sigma_error, gamma_record_error, cross_error) < TOL
    mu, lr, ms, ls, gs, gx = sy.symbols('mu_R lambda_R m_S lambda_S gamma_S gamma_X')
    deviation = mu-6*lr*ms/ls
    betas = [2*gx*mu, (gs+2*gx)*lr, 2*gs*ms, 3*gs*ls]
    identity = sy.simplify(sum(sy.diff(deviation, p)*beta for p, beta in zip([mu, lr, ms, ls], betas))
                           -2*gx*deviation)
    assert identity == 0
    return {'gamma_Sigma_numerator_at_g_lambda_1': '-16/5',
            'gamma_X_numerator_at_g_lambda_1': '-29/5', 'normalization': 'divide by 16 pi^2',
            'matrix_error': max(gamma_sigma_error, gamma_record_error, cross_error),
            'deviation': str(deviation), 'beta_deviation_minus_2gammaX_deviation': str(identity),
            'scope': 'perturbative superpotential running in exact SUSY; scalar soft masses not covered',
            'equal_lambda_and_g_preserved_by_RG': False}


def gauge_gate(manifest):
    k = sy.Matrix([1, 1, R(5, 3)])
    b = sy.Matrix([-3, 1, 11])
    w = sy.Matrix([1, -R(12, 7), R(3, 7)])
    assert w.dot(k) == 0 and w.dot(b) == 0
    # Each species is a separate threshold, including each SM generation.
    species = []
    for gen in range(3):
        for name, db in [('Q', [R(1, 3), R(1, 2), R(1, 18)]),
                         ('uc', [R(1, 6), 0, R(4, 9)]), ('dc', [R(1, 6), 0, R(1, 9)]),
                         ('L', [0, R(1, 6), R(1, 6)]), ('ec', [0, 0, R(1, 3)])]:
            species.append((f'{name}_{gen+1}', sy.Matrix(db)))
    species += [('gluino', sy.Matrix([2, 0, 0])), ('wino', sy.Matrix([0, R(4, 3), 0])),
                ('Higgsinos_pair', sy.Matrix([0, R(2, 3), R(2, 3)])),
                ('extra_Higgs', sy.Matrix([0, R(1, 6), R(1, 6)]))]
    # Bino contributes zero; it imposes no threshold in a gauge difference.
    total_soft = sum((s for _, s in species), sy.zeros(3, 1))
    assert total_soft == sy.Matrix([4, R(25, 6), R(25, 6)])
    bsm = sy.Matrix([-7, -R(19, 6), R(41, 6)])
    assert bsm+total_soft == b
    b5 = -3*5+3*(R(3, 2)+R(1, 2))+1+5+2*5
    assert b5 == 7
    sv = sy.Matrix([2, 3, R(25, 3)])
    sigma = sy.Matrix([3, 2, 0])
    hcolor = sy.Matrix([1, 0, R(2, 3)])
    rec_oct = sy.Matrix([3, 0, 0])
    rec_t = sy.Matrix([0, 2, 0])
    rec_b = sv
    assert b5*k-(b+2*rec_t)-(-2*sv+sigma+hcolor+2*rec_oct+2*rec_b) == sy.zeros(3, 1)
    delta = (sigma+hcolor-2*rec_oct+2*rec_b)*sy.log(2)/(4*sy.pi)
    scheme = sy.Matrix([3, 2, 0])/(12*sy.pi)
    # Physical heavy vector: gauge+Goldstone -7/2, fermions +4/3, real scalar +1/6.
    # Sum of absolute beta coefficients is 5 S_V, bounding soft mass splitting.
    abs_coeff = 5*sv+sigma+hcolor+2*rec_oct+2*rec_b+2*rec_t
    relative = R(1, 1000)
    # |log(1+e)|<=relative/(1-relative); tree K mass shifts at most sqrt(2)*eta.
    # sqrt(2)<3/2 and pi>3 yield a fully rational conservative bound.
    remainder_bounds = (abs_coeff*relative/(1-relative)
                        +(2*rec_oct+2*rec_b)*R(3, 2)*relative)/6
    assert all(value < R(1, 20) for value in remainder_bounds)
    eps = R(1, 20)
    hcap = sy.log(100)/(2*sy.pi)
    # Relax the fixed ratio m_plus/m_minus: both t's exceed log(1000)/(2 pi).
    # Dropping its extra positive log produces a still more optimistic upper bound.
    tmin = sy.log(1000)/sy.pi
    soft_max_coeff = sum(max(-w.dot(db), 0) for _, db in species)
    assert soft_max_coeff == R(11, 2)
    eps_upper = eps*sum(abs(v) for v in w)
    constant = sy.simplify(w.dot(delta+scheme))
    obs = manifest['observables']
    val = lambda key, field='value': R(str(obs[key][field]))
    A, s = val('alpha_em_inverse'), val('sin_squared_theta_W')
    Ae, se = 3*val('alpha_em_inverse', 'quoted_error'), 3*val('sin_squared_theta_W', 'quoted_error')
    as_high = val('alpha_s')+3*val('alpha_s', 'quoted_error')
    # This upper bound uses the upper EW corner, because both derivatives are positive.
    baseline = R(12, 7)*(A+Ae)*(s+se)-R(3, 7)*(A+Ae)*(1-s-se)
    upper = baseline+constant+soft_max_coeff*hcap+w.dot(rec_t)*tmin+eps_upper
    margin = 1/as_high-upper
    assert sy.N(margin, 50) > 0
    # Exact rational sign, independent of floating logs: collect log2,log5 terms.
    rest = sy.expand_log(upper-baseline-eps_upper, force=True).expand()
    symbolic = (-R(41, 7)*sy.log(2)-R(67, 14)*sy.log(5)-R(1, 28))/sy.pi
    assert sy.simplify(rest-symbolic) == 0
    # log2>=1/2, log5>=4/5, pi<4 -> negative correction at most this rational.
    rest_upper = (-R(41, 7)*R(1, 2)-R(67, 14)*R(4, 5)-R(1, 28))/4
    rational_margin = 1/as_high-baseline-eps_upper-rest_upper
    # A looser elementary logarithm bound may not separate the boxes; if so use
    # log5=log(5/4)+2log2 >= 1/5+1, still from the same integral inequality.
    if rational_margin <= 0:
        rest_upper = (-R(41, 7)*R(1, 2)-R(67, 14)*R(6, 5)-R(1, 28))/4
        rational_margin = 1/as_high-baseline-eps_upper-rest_upper
    assert rational_margin > 0
    # Independent LP with the same optimistic tmin and otherwise full equations.
    ns = len(species)
    nv = 3+ns+3  # a,l,t_sum,h_s,epsilon_i
    eq = np.zeros((2, nv))
    for row, idx in enumerate([1, 2]):
        eq[row, :3] = [float(k[idx]), float(b[idx]), float(rec_t[idx])]
        eq[row, 3:3+ns] = [-float(db[idx]) for _, db in species]
        eq[row, 3+ns+idx] = 1
    rhs = np.array([float((A+Ae)*(s+se)), float((A+Ae)*(1-s-se))])
    rhs -= np.array(delta+scheme, float).ravel()[1:]
    objective = np.zeros(nv)
    objective[:3] = [1, -3, 0]
    objective[3:3+ns] = [-float(db[0]) for _, db in species]
    objective[3+ns] = 1
    inequality = np.zeros((1, nv))
    inequality[0, 1:3] = [-2, 1]
    # m_minus>=1e5 MZ, with even its positive fixed-ratio correction omitted.
    lp = linprog(-objective, A_ub=inequality, b_ub=[-float(sy.log(10**5)/sy.pi)],
                 A_eq=eq, b_eq=rhs,
                 bounds=[(0, None), (0, None), (float(tmin), None)]
                        +[(0, float(hcap))]*ns+[(-float(eps), float(eps))]*3, method='highs')
    assert lp.success
    lp_max = -lp.fun+float(delta[0]+scheme[0])
    assert abs(lp_max-float(upper)) < TOL
    return {'b_MSSM_order_3_2_Y': list(map(str, b)), 'b_SU5': str(b5),
            'soft_species': [{'name': name, 'db': list(map(str, db)), 'projected': str(w.dot(db))} for name, db in species],
            'bino_db': [0, 0, 0], 'soft_beta_sum': list(map(str, total_soft)),
            'record_triplet_beta_each': list(map(str, rec_t)),
            'matching_scale_beta_identity': True,
            'fixed_heavy_delta': list(map(str, delta)), 'MS_minus_DR_inverse_alpha': list(map(str, scheme)),
            'heavy_remainder_rational_bounds': list(map(str, remainder_bounds)),
            'allowed_epsilon_coordinate': str(eps),
            'soft_projected_max_coefficient': str(soft_max_coeff),
            'inverse_alpha_s_MS_upper': float(upper),
            'alpha_s_MS_lower_if_inverse_positive': float(1/upper),
            'observed_alpha_s_MS_upper': float(as_high),
            'exclusion_margin': str(sy.N(margin, 60)),
            'strict_rational_margin': str(rational_margin),
            'independent_LP_difference': abs(lp_max-float(upper)),
            'result': 'REJECTED_REGISTERED_LOW_SOFT_HIERARCHICAL_RECORD_ONE_LOOP_BRANCH',
            'fits_or_selected_soft_spectrum': False, 'all_SUSY_or_all_orders_rejected': False}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/70_*.md'))
    input_path = folder/'ce_gauge_matching_inputs.json'
    manifest = json.loads(input_path.read_text(encoding='utf-8'))
    model = build_model()
    action = action_checks(model)
    fixed_slice = protection_checks(model)
    protection = protection_checks(model, flat=True)
    report = {'schema_version': 1, 'candidate': 'CE-UR3', 'scientific_success': False,
              'full_joint_rmse': None,
              'preregistration_sha256': hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 70.2')[0].encode()).hexdigest(),
              'followup_preregistration_sha256': hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 70.2')[1].split('## 70.3')[0].encode()).hexdigest(),
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), input_path, folder/'ce_simple_group_record.py']},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'sympy': sy.__version__},
              'action': action, 'fixed_slice': fixed_slice, 'protection': protection,
              'holomorphic_running': holomorphic_running(model), 'record': free_record(),
              'gauge_gate': gauge_gate(manifest), 'observational_manifest': manifest,
              'sources': ['https://arxiv.org/abs/hep-ph/9709356', 'https://arxiv.org/abs/hep-ph/9308222',
                          'https://arxiv.org/abs/hep-ph/0111209',
                          'https://link.springer.com/article/10.1140/epjc/s10052-016-4437-6'],
              'open_gates': ['origin of SUSY and SU5', 'soft breaking action and Higgs vacuum',
                            'autonomous actual record', 'quantum and gravity common derivation',
                            'joint particle/cosmology/gravity predictions and covariance']}
    (folder/'ce_supersymmetric_record.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'protection': protection, 'gauge_gate': report['gauge_gate']['result'],
                      'inverse_alpha_upper': report['gauge_gate']['inverse_alpha_s_MS_upper'],
                      'exclusion_margin': report['gauge_gate']['exclusion_margin']}, indent=2))


if __name__ == '__main__':
    main()
