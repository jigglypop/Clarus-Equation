"""CE-UR4: neutral adjoint records, a threshold envelope, and visible readout.

Only the registered conditional claims in chapter 71 are tested. No empirical
parameter fit, autonomous apparatus, or quantum-gravity success is asserted.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import linprog
import sympy as sy

from ce_supersymmetric_record import (
    N, build_model as ur3_model, component_matrices, eigen_hist, error,
    f_terms, potential_and_gradient,
)


R = sy.Rational
TOL = 1e-8
PREREG = '858befdec3b167395038147f74681ee9d6ee8ef1ecd16b1a732d84dd404240cf'


def build_model(m0=0.):
    t, y, reps, mass, cubic, vacuum = ur3_model(m0)
    mass[24:72, 24:72] -= 4*np.eye(48)
    return t, y, reps, mass, cubic, vacuum


def spectra_and_protection(model):
    t, _, reps, mass, cubic, vacuum = model
    hs, mf, gv, f, d = component_matrices(vacuum, reps, mass, cubic)
    histories = {'scalar': eigen_hist(np.linalg.eigvalsh(hs)),
                 'Weyl': eigen_hist(np.linalg.eigvalsh(mf.conj().T@mf)),
                 'vector': eigen_hist(np.linalg.eigvalsh(gv))}
    assert histories == {
        'scalar': {'0': 24, '1': 50, '16': 12, '25': 34, '36': 32, '50': 12},
        'Weyl': {'0': 18, '1': 25, '16': 6, '25': 17, '36': 16, '50': 24},
        'vector': {'0': 12, '50': 12}}
    assert max(error(f), error(d)) < TOL
    # Independent signed representation masses, away from m0=0 as well.
    _, _, _, shifted, _, _ = build_model(.003)
    _, w0 = f_terms(vacuum, shifted, cubic)
    masses = np.linalg.eigvalsh(.003*(5*np.eye(2)+np.pi/2*np.array([[0, 1], [1, 0]])))
    expected = sorted(value+shift for value in masses
                      for shift in [6]*8+[-4]*3+[0]+[1]*12)
    mass_error = error(np.linalg.eigvalsh(w0[24:72, 24:72])-expected)
    neutral_error = max(error(gen@t[11]-t[11]@gen) for gen in t[:12])
    # These are the same free scalar/Weyl coordinates, now along the singlet.
    direction = np.zeros(N, complex)
    direction[35], direction[59] = .5, .5j
    samples = []
    for x in [0., .07, .3]:
        z = vacuum+x*direction
        hs, mf, gv, f, d = component_matrices(z, reps, mass, cubic)
        es = np.linalg.eigvalsh(hs)
        ef = np.linalg.eigvalsh(mf.conj().T@mf)
        ev = np.linalg.eigvalsh(gv)
        bosons = np.sort(np.r_[es[es > TOL], np.repeat(ev[ev > TOL], 3)])
        fermions = np.sort(np.repeat(ef[ef > TOL], 2))
        assert len(bosons) == len(fermions)
        spectral_error = error(bosons-fermions)
        assert max(error(f), error(d), spectral_error, -float(es.min())) < TOL
        samples.append({'x': x, 'F_error': error(f), 'D_error': error(d),
                        'weighted_positive_spectrum_error': spectral_error,
                        'STr_M4': float(np.sum(es**2)-2*np.sum(ef**2)+3*np.sum(ev**2))})
    # Hessian-vector products from an independent potential gradient, including
    # all real coordinates and a generic direction through the flat background.
    rng = np.random.default_rng(7101)
    z = vacuum+.13*direction
    hs = component_matrices(z, reps, mass, cubic)[0]
    fd_errors = []
    for _ in range(3):
        dr = rng.normal(size=2*N)
        dr /= np.linalg.norm(dr)
        dz = (dr[:N]+1j*dr[N:])/np.sqrt(2)
        def central(h):
            gp = potential_and_gradient(z+h*dz, reps, mass, cubic)[1]
            gm = potential_and_gradient(z-h*dz, reps, mass, cubic)[1]
            return (gp-gm)/(2*h)
        extrapolated = (4*central(.002)-central(.004))/3
        fd_errors.append(error(extrapolated-hs@dr))
    assert max(fd_errors+[mass_error, neutral_error]) < TOL
    # The mass-independent anomalous dimensions are computed from the tensor,
    # with a NEW protected combination (2 rather than UR3's 6).
    gamma = .5*np.einsum('ikl,jkl->ij', cubic, cubic.conj(), optimize=True)
    gamma -= 2*np.einsum('aij,ajk->ik', reps, reps, optimize=True)
    gamma_error = max(error(gamma[:24, :24]+16*np.eye(24)/5),
                      error(gamma[24:72, 24:72]+29*np.eye(48)/5),
                      error(gamma[:24, 24:72]))
    mu, lr, ms, ls, gs, gx = sy.symbols('mu_R lambda_R m_S lambda_S gamma_S gamma_X')
    deviation = mu-2*lr*ms/ls
    betas = [2*gx*mu, (gs+2*gx)*lr, 2*gs*ms, 3*gs*ls]
    identity = sy.simplify(sum(sy.diff(deviation, p)*beta
                              for p, beta in zip([mu, lr, ms, ls], betas))-2*gx*deviation)
    assert identity == 0 and gamma_error < TOL
    return {'vacuum_mass_squared_multiplicities': histories,
            'nonzero_m0_signed_mass_error': mass_error,
            'SM_singlet_commutator_error': neutral_error,
            'flat_backgrounds': samples, 'Hessian_vector_FD_error': max(fd_errors),
            'protected_holomorphic_combination': str(deviation),
            'beta_minus_2gammaX_combination': str(identity), 'gamma_matrix_error': gamma_error,
            'scope': 'exact supplied global SUSY; no protection of unspecified soft terms'}


def free_record(model):
    _, _, reps, _, cubic, vacuum = model
    _, _, _, mass, _, _ = build_model(1.)
    _, w = f_terms(vacuum, mass, cubic)
    inds = [35, 59]
    k = w[np.ix_(inds, inds)].real
    expected = 5*np.eye(2)+np.pi/2*np.array([[0, 1], [1, 0]])
    assert error(k-expected) < TOL
    e, v = np.linalg.eigh(k)
    kg_map = v@np.diag(1/np.sqrt(2*e))@v.T
    p = np.diag([0, 1])
    errors, choi_min = [], 0.
    for time in [0., .25, .5, 1., 1.5, 2.]:
        u = expm(-1j*k*time)
        psi = u[:, 0]
        phi = kg_map@psi
        dot_phi = -1j*k@phi
        kg = 1j*(np.vdot(phi, dot_phi)-np.vdot(dot_phi, phi))
        kraus = [p@u, (np.eye(2)-p)@u]
        choi = sum(np.outer(a.ravel(), a.ravel().conj()) for a in kraus)
        choi_min = min(choi_min, float(np.linalg.eigvalsh(choi).min()))
        errors += [abs(np.vdot(psi, p@psi).real-np.sin(np.pi*time/2)**2),
                   abs(kg-1), abs(np.vdot(psi, k@psi)-5),
                   error(sum(a.conj().T@a for a in kraus)-np.eye(2))]
    neutral = error(reps[:12, :, inds])
    assert max(errors+[neutral, -choi_min]) < TOL
    return {'free_rest_record_error': float(max(errors)), 'neutral_error': neutral,
            'Kraus_Choi_minimum': choi_min, 'mass_pair_at_m0_1': e.tolist(),
            'autonomous_actual_record': False,
            'scope': 'free energy conservation; formal CP projections are not a physical apparatus'}


def soft_species():
    fields = [('Q', [R(1, 3), R(1, 2), R(1, 18)]),
              ('uc', [R(1, 6), 0, R(4, 9)]), ('dc', [R(1, 6), 0, R(1, 9)]),
              ('L', [0, R(1, 6), R(1, 6)]), ('ec', [0, 0, R(1, 3)])]
    rows = [(f'{name}_{gen+1}', sy.Matrix(db)) for gen in range(3) for name, db in fields]
    return rows+[(name, sy.Matrix(db)) for name, db in
                 [('gluino', [2, 0, 0]), ('wino', [0, R(4, 3), 0]),
                  ('Higgsinos_pair', [0, R(2, 3), R(2, 3)]),
                  ('extra_Higgs', [0, R(1, 6), R(1, 6)])]]


def gauge_gate(manifest):
    k, b = sy.Matrix([1, 1, R(5, 3)]), sy.Matrix([-3, 1, 11])
    w = sy.Matrix([1, -R(12, 7), R(3, 7)])
    rows = soft_species()
    soft = sum((db for _, db in rows), sy.zeros(3, 1))
    assert soft == sy.Matrix([4, R(25, 6), R(25, 6)])
    assert sy.Matrix([-7, -R(19, 6), R(41, 6)])+soft == b
    assert w.dot(k) == w.dot(b) == 0
    sv, sigma, hc = sy.Matrix([2, 3, R(25, 3)]), sy.Matrix([3, 2, 0]), sy.Matrix([1, 0, R(2, 3)])
    octet, triplet = sy.Matrix([3, 0, 0]), sy.Matrix([0, 2, 0])
    assert octet+triplet+sv == 5*k
    assert 7*k == b-2*sv+sigma+hc+2*(octet+triplet+sv)
    delta = ((sigma+hc)*sy.log(sy.sqrt(2))
             +2*(octet*sy.log(sy.sqrt(50)/6)+triplet*sy.log(sy.sqrt(50)/4)
                 +sv*sy.log(sy.sqrt(50))))/(2*sy.pi)
    scheme = sy.Matrix([3, 2, 0])/(12*sy.pi)
    constant = sy.expand_log(w.dot(delta+scheme), force=True).expand()
    expected_constant = (R(57, 14)*sy.log(2)-3*sy.log(3)-R(1, 28))/sy.pi
    assert sy.simplify(constant-expected_constant) == 0
    abs_coeff = 5*sv+sigma+hc+2*(octet+triplet+sv)
    assert abs_coeff == sy.Matrix([24, 27, 59])
    relative = R(1, 1000)
    # sqrt(50)<7.1 and q/(1-q)<0.0075 give a rational log-shift bound.
    q = R(71, 10)*relative
    assert R(71, 10)**2 > 50 and q/(1-q) < R(3, 400)
    remainder = (abs_coeff*relative/(1-relative)+10*k*R(3, 400))/6
    assert all(value < R(1, 20) for value in remainder)
    hcap, eps = sy.log(100)/(2*sy.pi), R(1, 20)
    cmin = sum(min(-w.dot(db), 0) for _, db in rows)
    cmax = sum(max(-w.dot(db), 0) for _, db in rows)
    assert (cmin, cmax) == (-R(29, 7), R(11, 2))
    obs = manifest['observables']
    value = lambda key, field='value': R(str(obs[key][field]))
    A, s = value('alpha_em_inverse'), value('sin_squared_theta_W')
    Ae, se = 3*value('alpha_em_inverse', 'quoted_error'), 3*value('sin_squared_theta_W', 'quoted_error')
    emin, emax = (A-Ae, s-se), (A+Ae, s+se)
    baseline = lambda aa, ss: aa*(15*ss-3)/7
    width = eps*sum(abs(x) for x in w)
    bounds_exact = [baseline(*emin)+constant+cmin*hcap-width,
                    baseline(*emax)+constant+cmax*hcap+width]
    # Independent full-equation LP: a=alpha5^-1, l=log(MX/MZ)/(2pi).
    # The hierarchy admits some m0 iff l exceeds this fixed-ratio lower bound.
    ratio = (5+sy.pi/2)/(5-sy.pi/2)
    lmin = sy.log(10**8*ratio)/(2*sy.pi)
    ns, nv = len(rows), 2+len(rows)+3
    eq = np.zeros((2, nv))
    for row, i in enumerate([1, 2]):
        eq[row, :2] = [float(k[i]), float(b[i])]
        eq[row, 2:2+ns] = [-float(db[i]) for _, db in rows]
        eq[row, 2+ns+i] = 1
    objective = np.zeros(nv)
    objective[:2] = [1, -3]
    objective[2:2+ns] = [-float(db[0]) for _, db in rows]
    objective[2+ns] = 1
    lp_errors, lp_edges = [], []
    offsets = np.array(delta+scheme, float).ravel()
    for sign, corner, predicted in zip([1, -1], [emin, emax], bounds_exact):
        aa, ss = corner
        rhs = np.array([float(aa*ss), float(aa*(1-ss))])-offsets[1:]
        lp = linprog(sign*objective, A_eq=eq, b_eq=rhs,
                     bounds=[(0, None), (float(lmin), None)]+[(0, float(hcap))]*ns
                            +[(-float(eps), float(eps))]*3, method='highs')
        assert lp.success
        result = sign*lp.fun+offsets[0]
        lp_errors += [abs(result-float(predicted)), error(eq@lp.x-rhs)]
        lp_edges.append({'inverse_alpha_s': result, 'alpha5_inverse': float(lp.x[0]),
                         'log_MX_over_MZ_div_2pi': float(lp.x[1])})
    assert max(lp_errors) < TOL
    observed = [1/(value('alpha_s')+3*value('alpha_s', 'quoted_error')),
                1/(value('alpha_s')-3*value('alpha_s', 'quoted_error'))]
    assert float(bounds_exact[0]) < float(observed[0]) < float(observed[1]) < float(bounds_exact[1])
    # No observed value enters this implementation check of the omitted K shift.
    sample_errors = []
    for rplus in [.0001, .0005, .001]:
        full = np.array((sigma+hc)*sy.log(sy.sqrt(2))/(2*sy.pi), float).ravel()
        for rr in [rplus, rplus/float(ratio)]:
            for db, mass_ratio in [(octet, 6/np.sqrt(50)+rr),
                                   (triplet, 4/np.sqrt(50)-rr), (sv, 1/np.sqrt(50)+rr)]:
                full += np.array(db, float).ravel()*np.log(1/mass_ratio)/(2*np.pi)
        diff = np.abs(full-np.array(delta, float).ravel())
        assert np.all(diff < np.array(remainder, float).ravel())
        sample_errors.append(diff.tolist())
    return {'result': 'NOT_EXCLUDED_BY_RELAXED_ONE_LOOP_THRESHOLD_ENVELOPE',
            'light_record_beta': [0, 0, 0], 'b_MSSM': list(map(str, b)), 'b_SU5': 7,
            'matching_scale_beta_identity': True, 'projected_fixed_constant': str(constant),
            'fixed_heavy_delta': list(map(str, delta)), 'soft_projected_coefficients': [str(cmin), str(cmax)],
            'heavy_remainder_rational_bounds': list(map(str, remainder)),
            'inverse_alpha_s_envelope': list(map(float, bounds_exact)),
            'alpha_s_envelope': [float(1/bounds_exact[1]), float(1/bounds_exact[0])],
            'observed_inverse_alpha_s_box': list(map(float, observed)),
            'independent_LP_error': max(lp_errors), 'LP_envelope_edges': lp_edges,
            'omitted_K_shift_samples': sample_errors,
            'selected_or_fitted_soft_spectrum': False, 'fitted_parameters': 0,
            'physical_correlated_soft_action_exists': None,
            'scope': 'necessary relaxed family gate; not an exact spectrum witness or joint RMSE'}


def effective_action(model):
    _, _, reps, mass, cubic, vacuum = model
    s1, s2, h1, h2, b1, b2 = sy.symbols('S1 S2 H1 H2 B1 B2')
    lr, lh, ls, v = sy.symbols('lambda_R lambda_H lambda_S V', positive=True)
    Q, P = s1**2+s2**2, b1*h1+b2*h2
    j0 = (lr*Q+3*lh*P)/(2*sy.sqrt(15))
    pauli = [sy.Matrix([[0, 1], [1, 0]]), sy.Matrix([[0, -sy.I], [sy.I, 0]]), sy.diag(1, -1)]
    jtriplet = [lh*(sy.Matrix([[b1, b2]])*a*sy.Matrix([h1, h2]))[0]/2 for a in pauli]
    assert sy.simplify(sum(j**2 for j in jtriplet)-lh**2*P**2/4) == 0
    eliminated = j0**2/(2*ls*v)+sum(j**2 for j in jtriplet)/(10*ls*v)
    proposed = lr**2*Q**2/(120*ls*v)+lr*lh*Q*P/(20*ls*v)+lh**2*P**2/(10*ls*v)
    assert sy.simplify(eliminated-proposed) == 0
    _, w0 = f_terms(vacuum, mass, cubic)
    heavy_mass = w0[:12, :12]
    assert error(heavy_mass-np.diag([5]*8+[-5]*3+[-1])) < TOL
    inds = [35, 59, 75, 76, 80, 81]
    tensor = cubic[:, inds, :][:, :, inds]
    numerical = sy.lambdify([s1, s2, h1, h2, b1, b2], proposed.subs({lr: 1, lh: 1, ls: 1, v: 1}), 'numpy')
    rng, errors, zeros, kmin = np.random.default_rng(7102), [], [], []
    for _ in range(6):
        light = .07*(rng.normal(size=6)+1j*rng.normal(size=6))
        source = .5*np.einsum('aij,i,j->a', tensor, light, light)
        response = -np.linalg.solve(heavy_mass, source[:12])
        quartic = source[:12]@response/2  # holomorphic, no complex conjugation
        errors.append(abs(quartic-numerical(*light)))
        other = np.r_[12:35, 36:59, 60:75, 77:80]
        zeros.append(error(source[other]))
        z = vacuum.copy()
        z[inds] += light
        row = np.einsum('i,aij->aj', z.conj(), reps, optimize=True)
        zeros.append(error(np.einsum('ai,i->a', row[12:], z)))
        # K_eff=K_light+|response(light)|^2 at this chiral order.
        dj = np.einsum('aij,j->ai', tensor[:12], light)
        response_jac = -np.linalg.solve(heavy_mass, dj)
        metric = np.eye(6)+response_jac.conj().T@response_jac
        kmin.append(float(np.linalg.eigvalsh(metric).min()))
        q, p = light[0]**2+light[1]**2, light[4]*light[2]+light[5]*light[3]
        j0_value = (q+3*p)/(2*np.sqrt(15))
        jts = np.array([np.array(a, complex) for a in pauli])
        jt_values = np.einsum('i,aij,j->a', light[4:], jts, light[2:4])/2
        expected_K = abs(j0_value)**2+np.vdot(jt_values, jt_values).real/25
        errors.append(abs(np.vdot(response, response).real-expected_K))
    assert max(errors+zeros) < TOL and min(kmin) >= 1-TOL
    return {'W_quartic_in_Q_P': 'lambda_R^2 Q^2/(120 lambda_S V) + lambda_R lambda_H Q P/(20 lambda_S V) + lambda_H^2 P^2/(10 lambda_S V)',
            'symbolic_elimination_identity': True, 'numeric_cubic_elimination_error': float(max(errors)),
            'other_heavy_and_Goldstone_sources_error': max(zeros),
            'Kahler': '|J_s|^2/(lambda_S V)^2 + sum_a |J_a|^2/(5 lambda_S V)^2',
            'sample_Kahler_metric_minimum': min(kmin),
            'scope': 'leading local chiral elimination about zero visible/light fields; no derived detector'}


def partial_visible(rho, hidden_dim):
    return np.einsum('aiaj->ij', rho.reshape(hidden_dim, 2, hidden_dim, 2))


def trace_distance(a, b):
    return float(np.sum(np.abs(np.linalg.eigvalsh(a-b)))/2)


def observability(model):
    _, _, reps, mass, cubic, vacuum = model
    permutation = np.r_[0:24, 48:72, 24:48, 72:82]
    sym_errors = [error(mass[np.ix_(permutation, permutation)]-mass),
                  error(cubic[np.ix_(permutation, permutation, permutation)]-cubic),
                  error(reps[:, permutation, :][:, :, permutation]-reps),
                  error(vacuum[permutation]-vacuum)]
    # K != 0 also preserves exchange; this is not merely the enlarged m0=0 symmetry.
    shifted = build_model(.003)[3]
    sym_errors.append(error(shifted[np.ix_(permutation, permutation)]-shifted))
    assert max(sym_errors) < TOL
    I, X, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    r0, r1 = np.diag([1, 0]), np.diag([0, 1])
    pp, pm = (I+X)/2, (I-X)/2
    unitary = np.kron(pp, I)+np.kron(pm, X)
    covariance = error(unitary@np.kron(X, I)-np.kron(X, I)@unitary)
    output = lambda state: partial_visible(unitary@np.kron(state, r0)@unitary.conj().T, 2)
    label_distance = trace_distance(output(r0), output(r1))
    mass_distance = trace_distance(output(pp), output(pm))
    assert max(covariance, label_distance, abs(mass_distance-1)) < TOL
    # Independent symbolic characterization of every accessible Hermitian effect.
    a, d, b, c = sy.symbols('a d b c', real=True)
    effect = sy.Matrix([[a, b+sy.I*c], [b-sy.I*c, d]])
    xx = sy.Matrix(X)
    commutator = effect*xx-xx*effect
    solution = sy.solve(list(commutator), [d, c], dict=True)
    assert solution == [{c: 0, d: a}]
    # Operator-norm bound >= max(|a|,|a-1|)>=1/2, attained by E=I/2.
    optimum_error = float(np.linalg.norm(I/2-r1, 2))
    assert optimum_error == .5
    # A relational reference demonstrates exactly which assumption must change.
    same = (np.eye(4)+np.kron(Z, Z))/2
    different = np.eye(4)-same
    uref = np.kron(same, I)+np.kron(different, X)
    global_exchange = np.kron(np.kron(X, X), I)
    reference_covariance = error(uref@global_exchange-global_exchange@uref)
    def pointer(state, reference):
        rho = np.kron(np.kron(state, reference), r0)
        return partial_visible(uref@rho@uref.conj().T, 4)
    asym_distance = trace_distance(pointer(r0, r0), pointer(r1, r0))
    sym_distance = trace_distance(pointer(r0, I/2), pointer(r1, I/2))
    unitary_error = max(error(unitary.conj().T@unitary-np.eye(4)),
                        error(uref.conj().T@uref-np.eye(8)))
    assert max(reference_covariance, sym_distance, abs(asym_distance-1), unitary_error) < TOL
    return {'action_exchange_error': max(sym_errors),
            'accessible_effects': 'a I + b sigma_x, 0 <= a +/- b <= 1',
            'optimal_absolute_label_error_equal_priors': .5,
            'minimum_operator_norm_effect_minus_R': optimum_error,
            'symmetric_apparatus_label_trace_distance': label_distance,
            'mass_basis_positive_control_trace_distance': mass_distance,
            'reference_unitary_exchange_error': reference_covariance,
            'asymmetric_reference_trace_distance': asym_distance,
            'symmetric_reference_trace_distance': sym_distance,
            'control_unitarity_error': unitary_error,
            'result': 'REJECTED_ABSOLUTE_COPY_LABEL_READOUT_WITH_INVARIANT_APPARATUS',
            'relational_control_derived_from_action': False,
            'scope': 'exchange-covariant dynamics and invariant apparatus; asymmetric references excluded'}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/71_*.md'))
    prereg = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 71.2')[0].encode()).hexdigest()
    assert prereg == PREREG
    manifest = json.loads((folder/'ce_gauge_matching_inputs.json').read_text(encoding='utf-8'))
    model = build_model()
    action = spectra_and_protection(model)
    gauge = gauge_gate(manifest)
    report = {'schema_version': 1, 'candidate': 'CE-UR4', 'scientific_success': False,
              'full_joint_rmse': None, 'preregistration_sha256': prereg,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), folder/'ce_supersymmetric_record.py',
                                 folder/'ce_simple_group_record.py', folder/'ce_gauge_matching_inputs.json']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__},
              'action_and_protection': action, 'free_record': free_record(model),
              'gauge_gate': gauge, 'effective_action': effective_action(model),
              'observability': observability(model), 'observational_manifest': manifest,
              'sources': ['https://arxiv.org/abs/0904.0370',
                          'https://arxiv.org/abs/quant-ph/0610030',
                          'https://arxiv.org/abs/hep-ph/9709356',
                          'https://arxiv.org/abs/hep-ph/9308222'],
              'open_gates': ['derived soft spectrum and electroweak vacuum',
                             'relational reference preparation and an actual detector from this action',
                             'asynchronous event ontology with preserved amplitudes',
                             'common quantum and gravity derivation',
                             'joint predictions, covariance, and frozen parameter validation']}
    (folder/'ce_singlet_record_observability.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['action_and_protection', 'gauge_gate',
                                                'effective_action', 'observability']}, indent=2))


if __name__ == '__main__':
    main()
