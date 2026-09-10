"""Chapter 69: adjoint record splitting, feasibility, and radiative closure.

Real/complex field choices are registered competitors. No strong-coupling fit
or mass selection is performed. Feasibility is not a parameter-free prediction.
"""
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, linprog
import scipy
import sympy as sy

from ce_simple_group_record import basis


R = sy.Rational
K_METRIC = sy.Matrix([1, 1, R(5, 3)])
B_SM = sy.Matrix([-7, -R(19, 6), R(41, 6)])
GAMMA = np.pi/2
C_MINUS, C_PLUS = 5-GAMMA, 5+GAMMA
RATIO = C_PLUS/C_MINUS
ETA_CAP = 1e-3
TOL, FD_TOL = 1e-9, 1e-6


def error(a):
    return float(np.max(np.abs(a)))


def traceless(a):
    return a-np.trace(a)*np.eye(5)/5


def jordan(sigma, x):
    return traceless(sigma@x+x@sigma)


def operator_matrix(t, sigma):
    return np.array([[2*np.trace(a@jordan(sigma, b)).real for b in t] for a in t])


def setup():
    t, y = basis()
    sigma = np.diag([2., 2., 2., -3., -3.])
    j = operator_matrix(t, sigma)
    f = j+6*np.eye(24)
    expected_j = np.diag([4]*8+[-6]*3+[-2]+[-1]*12)
    # Direct subgroup adjoint action. No Dynkin-index lookup is used here.
    reps = np.array([[[2*np.trace(a@(gen@b-b@gen)) for b in t] for a in t]
                     for gen in [t[0], t[8], y]])
    indices = {}
    for name, ix in [('octet', list(range(8))), ('triplet', list(range(8, 11))),
                     ('singlet', [11]), ('broken', list(range(12, 24)))]:
        indices[name] = [float(np.trace(a[np.ix_(ix, ix)]@a[np.ix_(ix, ix)]).real)
                         for a in reps]
    expected_indices = {'octet': [3, 0, 0], 'triplet': [0, 2, 0],
                        'singlet': [0, 0, 0], 'broken': [2, 3, 25/3]}
    # Gauge covariance of the full operator, including its trace subtraction.
    u = expm(0.37j*(t[1]+t[16]+t[23]))
    x = t[3]+0.3*t[11]-0.7*t[20]
    cov_error = error(jordan(u@sigma@u.conj().T, u@x@u.conj().T)
                      -u@jordan(sigma, x)@u.conj().T)
    errors = {'jordan_spectrum': error(j-expected_j), 'jordan_self_adjoint': error(j-j.T),
              'covariance': cov_error,
              'subgroup_indices': max(error(np.array(indices[k])-expected_indices[k]) for k in indices)}
    assert max(errors.values()) < TOL
    return t, y, sigma, f, {
        'errors': errors, 'J_eigenvalues_multiplicity': {'4': 8, '-6': 3, '-2': 1, '-1': 12},
        'F_eigenvalues_multiplicity': {'10': 8, '0': 3, '4': 1, '5': 12},
        'adjoint_subgroup_indices': indices,
        'mass_squared_by_record_eigenvalue': {
            'octet': 'm_pm^2 + 4 MX^2', 'triplet': 'm_pm^2',
            'singlet': 'm_pm^2 + 16 MX^2/25', 'broken': 'm_pm^2 + MX^2'},
        'real_fields': {'CE-UR2-R': 48, 'CE-UR2-C': 96},
        'one_particle_amplitudes_complex_for_both': True}


def record_check(t, y, f):
    k = 5*np.eye(2)+GAMMA*np.array([[0., 1.], [1., 0.]])
    m2 = np.kron(k@k, np.eye(24))+np.kron(np.eye(2), f@f)
    evals, evecs = np.linalg.eigh(m2)
    mass = evecs@np.diag(np.sqrt(evals))@evecs.T
    kg_map = evecs@np.diag(1/np.sqrt(2*np.sqrt(evals)))@evecs.T
    projection = np.kron(np.diag([0., 1.]), np.eye(24))
    q_fund = t[10]+y
    q_adj = np.array([[2*np.trace(a@(q_fund@b-b@q_fund)) for b in t] for a in t])
    charge = np.kron(np.eye(2), q_adj)
    initial = np.zeros(48, complex)
    initial[10] = 1
    rows = []
    for time in [0, .125, .5, .75, 1, 1.5, 2]:
        u = expm(-1j*mass*time)
        psi = u@initial
        field = kg_map@psi
        dot_field = -1j*mass@field
        kg = 1j*(np.vdot(field, dot_field)-np.vdot(dot_field, field))
        rows.append({'time_m0': time,
                     'probability_error': abs(float(np.vdot(psi, projection@psi).real)-np.sin(GAMMA*time)**2),
                     'norm_error': float(abs(np.vdot(psi, psi)-1)),
                     'KG_norm_error': float(abs(kg-1)),
                     'energy_error': float(abs(np.vdot(psi, mass@psi)-5)),
                     'charge_error': error(charge@psi),
                     'kraus_completeness_error': error(u.conj().T@projection@u
                         +u.conj().T@(np.eye(48)-projection)@u-np.eye(48))})
    errors = {key: max(row[key] for row in rows) for key in rows[0] if key != 'time_m0'}
    errors['charge_record_commutator'] = error(charge@projection-projection@charge)
    assert max(errors.values()) < TOL and evals[0] > 0
    return {'errors': errors, 'rows': rows, 'positive_frequency_mass_minimum': float(np.sqrt(evals[0])),
            'scope': 'free rest one-particle sector; R and C have the same neutral two-species subspace',
            'CP_reason': 'Kraus operators P_a U; Choi is a sum of positive outer products',
            'autonomous_readout': False}


def base_delta():
    return sy.Matrix([-2, -3, -R(25, 3)])/(12*sy.pi)+sy.Matrix([
        sy.log(5/sy.sqrt(10))/2, sy.log(5/sy.sqrt(40))/3, 0])/(2*sy.pi)


def coefficient_tables(nu):
    # nu=1: two real adjoints; nu=2: two complex adjoints.
    octet = nu*sy.Matrix([R(1, 2), 0, 0])
    triplet = nu*sy.Matrix([0, R(1, 3), 0])
    broken = nu*sy.Matrix([R(1, 3), R(1, 2), R(25, 18)])
    assert octet+triplet+broken == R(5*nu, 6)*K_METRIC
    b5 = -R(55, 3)+4+R(1, 6)+R(5, 6)+R(5*nu, 3)
    # SM includes one light doublet; the old Higgs color triplet and adjoint
    # breaking scalar remain heavy, as do X/Y gauge+Goldstone modes.
    fixed_heavy = sy.Matrix([R(1, 2), R(1, 3), 0])+sy.Matrix([R(1, 6), 0, R(1, 9)])
    vector = -R(7, 2)*sy.Matrix([2, 3, R(25, 3)])
    assert b5*K_METRIC-(B_SM+2*triplet)-2*octet-2*broken-fixed_heavy-vector == sy.zeros(3, 1)
    return octet, triplet, broken, b5


def exact_delta(eta, nu):
    octet, _, broken, _ = coefficient_tables(nu)
    octet, broken = np.array(octet, float).ravel(), np.array(broken, float).ravel()
    delta = np.array(base_delta(), float).ravel()
    for ratio in [eta/RATIO, eta]:
        delta += octet*(-0.5*np.log(4+ratio*ratio))/(2*np.pi)
        delta += broken*(-0.5*np.log1p(ratio*ratio))/(2*np.pi)
    return delta


def empirical_gate(nu, manifest):
    octet, triplet, broken, b5 = coefficient_tables(nu)
    const = base_delta()-2*octet*sy.log(2)/(2*sy.pi)
    # Per-coordinate finite hierarchy correction is in [-nu*1e-6, 0].
    # log(1+z)<=z and pi>3 yield an exact conservative bound.
    bound = nu*R(1, 10**6)
    for a, b in zip(octet, broken):
        assert (2*R(1, 10**6))*(a/16+b/4)/3 <= bound
    b_eff = B_SM+2*triplet
    yy2, yyY = sy.symbols('y2 yY')
    sol = sy.solve([K_METRIC[0]-yy2*K_METRIC[1]-yyY*K_METRIC[2],
                    b_eff[0]-yy2*b_eff[1]-yyY*b_eff[2]], [yy2, yyY])
    y2, yY = sol[yy2], sol[yyY]
    w = sy.Matrix([1, -y2, -yY])
    assert w.dot(K_METRIC) == 0 and w.dot(b_eff) == 0
    deficit_coefficient = -w.dot(triplet)
    assert deficit_coefficient > 0
    cnum = np.array(const, float).ravel()
    # Exact sign of the linear epsilon minimum: epsilon_i lies in [-bound,0].
    eps_min = -bound*sum(max(v, 0) for v in w)
    obs = manifest['observables']
    value = lambda key, field='value': R(str(obs[key][field]))
    A, s = value('alpha_em_inverse'), value('sin_squared_theta_W')
    Ae = 3*value('alpha_em_inverse', 'quoted_error')
    se = 3*value('sin_squared_theta_W', 'quoted_error')
    alpha_low = value('alpha_s')-3*value('alpha_s', 'quoted_error')
    alpha_high = value('alpha_s')+3*value('alpha_s', 'quoted_error')
    assert y2-yY > 0 and yY+(y2-yY)*(s-se) > 0
    lower_x3 = (A-Ae)*(yY+(y2-yY)*(s-se))+w.dot(const)+eps_min
    lower_margin = lower_x3-1/alpha_low
    rational_certificate = None
    if nu == 1:
        # The projected constant is -(25 log2 + 2 log10 + 4)/(132 pi).
        # log2<=1, log10<=9 and pi>3 give a rigorous, deliberately loose bound.
        constant_lower = -R(47, 396)
        simple = -(25*sy.log(2)+2*sy.log(10)+4)/(132*sy.pi)
        assert sy.simplify(sy.expand_log(w.dot(const)-simple, force=True)) == 0
        rational_certificate = (A-Ae)*(yY+(y2-yY)*(s-se))+constant_lower+eps_min-1/alpha_low
        assert rational_certificate > 0
    # Independent linear program over a, l, summed triplet log, epsilon3,2,Y.
    eq = np.zeros((2, 6))
    for row, i in enumerate([1, 2]):
        eq[row, :3] = [float(K_METRIC[i]), float(B_SM[i]), float(triplet[i])]
        eq[row, 3+i] = 1
    rhs = np.array([float((A-Ae)*(s-se)), float((A-Ae)*(1-s+se))])-cnum[1:]
    objective = np.array([1., float(B_SM[0]), 0., 1., 0., 0.])
    lp = linprog(objective, A_ub=[[0, -2, 1, 0, 0, 0]], b_ub=[0], A_eq=eq, b_eq=rhs,
                 bounds=[(0, None)]*3+[(-float(bound), 0)]*3, method='highs')
    assert lp.success
    lp_error = abs(lp.fun+cnum[0]-float(lower_x3))
    assert lp_error < TOL
    k_num, b_num, t_num = (np.array(v, float).ravel() for v in [K_METRIC, B_SM, triplet])
    ew_matrix = np.array([[k_num[1], b_num[1]], [k_num[2], b_num[2]]])
    endpoint_rows = []
    derivative_errors = []
    for label, aa, ss in [('central', A, s)]+[(f'corner_{ia}_{is_}', A+ia*Ae, s+is_*se)
                                            for ia, is_ in itertools.product([-1, 1], repeat=2)]:
        x2, xY = float(aa*ss), float(aa*(1-ss))

        def condition(u):
            eta = np.exp(-u)
            d = exact_delta(eta, nu)
            sum_t = (2*u+np.log(RATIO))/(2*np.pi)
            common, log2pi = np.linalg.solve(ew_matrix, np.array([x2, xY])-d[1:]-t_num[1:]*sum_t)
            L = 2*np.pi*log2pi
            x3 = common+b_num[0]*log2pi+t_num[0]*sum_t+d[0]
            return common, L, x3, L-u-np.log(RATIO), eta, sum_t

        u_min = np.log(1/ETA_CAP)
        u_max = brentq(lambda u: condition(u)[3], u_min, 100., xtol=1e-12)
        # 0<L'(u)<10 nu/109<1: physical-domain endpoint is unique. Along it,
        # common inverse coupling and x3 both decrease and remain positive.
        for u in [u_min, (u_min+u_max)/2, u_max]:
            eta = np.exp(-u)
            ratios = np.array([eta/RATIO, eta])
            dL = 5*nu/109*(2-np.sum(ratios**2/(1+ratios**2)))
            dx3 = -111/10*dL/(2*np.pi)+nu/(4*np.pi)*np.sum(
                ratios**2/(4+ratios**2)-ratios**2/(1+ratios**2))
            assert 0 < dL < 1 and dx3 < 0
            step = 1e-4
            numeric = (np.array(condition(u+step)[:3])-np.array(condition(u-step)[:3]))/(2*step)
            derivative_errors += [abs(numeric[1]-dL), abs(numeric[2]-dx3)]
        # Endpoints depend only on the two EW inputs and predeclared boundaries.
        # The observed strong coupling is never a root-finding target.
        per_input = []
        for boundary, u in [('m_plus_over_MX_cap', u_min), ('m_minus_equals_MZ', u_max)]:
            common, L, x3, log_light, eta, sum_t = condition(u)
            assert common > 0 and L > 0 and x3 > 0 and eta <= ETA_CAP*(1+1e-12)
            assert log_light > -TOL
            pred = np.array([x3, x2, xY])
            direct = k_num*common+b_num*L/(2*np.pi)+t_num*sum_t+exact_delta(eta, nu)
            assert error(pred-direct) < TOL
            per_input.append({'boundary': boundary, 'alpha_s': 1/x3,
                              'log_MX_over_MZ': L, 'inverse_alpha5': common,
                              'eta': eta, 'log_m_minus_over_MZ': log_light,
                              'equation_residual': error(pred-direct)})
        endpoint_rows.append({'input': label, 'endpoints': per_input})
    assert max(derivative_errors) < TOL
    central_values = sorted(row['alpha_s'] for row in endpoint_rows[0]['endpoints'])
    necessary_survival = bool(central_values[0] < float(alpha_low) < float(alpha_high) < central_values[1])
    rejected = bool(sy.N(lower_margin, 50) > 0)
    assert not (rejected and necessary_survival)
    return {'variant': f'CE-UR2-{"R" if nu == 1 else "C"}',
            'b5': str(b5), 'per_mass_eigenvalue_beta': {'octet': list(map(str, octet)),
                'triplet': list(map(str, triplet)), 'broken': list(map(str, broken))},
            'matching_scale_beta_identity': True,
            'hierarchy_delta_coordinate_bound': str(bound),
            'relaxed_bound': {'y2': str(y2), 'yY': str(yY),
                'deficit_coefficient': str(deficit_coefficient),
                'constant_projected_delta': str(sy.expand_log(w.dot(const), force=True).expand()),
                'strict_rational_exclusion_margin': str(rational_certificate) if rational_certificate is not None else None,
                'epsilon_projected_lower': str(eps_min),
                'alpha_s_upper': float(1/lower_x3),
                'inverse_alpha_margin': str(sy.N(lower_margin, 50)), 'independent_LP_error': lp_error},
            'exact_mass_endpoints': endpoint_rows,
            'physical_interval_derivative_verification_error': max(derivative_errors),
            'central_endpoint_bracket': central_values,
            'observed_alpha_s_box': [float(alpha_low), float(alpha_high)],
            'rejected_by_relaxed_bound': rejected,
            'necessary_feasibility_survives': necessary_survival,
            'm0_selected_from_strong_coupling': False, 'prediction_with_frozen_m0': None,
            'scope': 'one-loop feasibility of a family, not a fitted or parameter-free empirical success'}


def radiation(t, sigma, f, n):
    """Full scalar mixing plus vector supertrace, at zero protected light mass."""
    d = 24*(n+1)
    h_sigma = np.diag([10.]*8+[40.]*3+[148.]+[0.]*12)
    e = np.zeros(24)
    e[10] = 1
    je = operator_matrix(t, t[10])
    h0 = np.zeros((d, d))
    h0[:24, :24] = h_sigma
    for k in range(n):
        sl = slice(24*(k+1), 24*(k+2))
        h0[sl, sl] = f@f
    h1 = np.zeros_like(h0)
    h1[:24, 24:48] = je@f
    h1[24:48, :24] = f@je
    h2 = np.zeros_like(h0)
    h2[:24, :24] = je.T@je
    h2[24:, 24:] = np.eye(n*24)
    h2[34, 34] += 2
    scalar_coeff = float(np.trace(h1@h1+2*h0@h2))
    comm_sigma = np.array([a@sigma-sigma@a for a in t])
    comm_e = np.array([a@t[10]-t[10]@a for a in t])
    gv0 = 2*np.einsum('aij,bij->ab', comm_sigma.conj(), comm_sigma).real
    gv2 = 2*np.einsum('aij,bij->ab', comm_e.conj(), comm_e).real
    vector_coeff = float(6*np.trace(gv0@gv2))
    # Independent exact diagonal invariants of J_(T3)^2 in the supplied basis.
    je2_diag = sy.Matrix([0]*10+[R(3, 5), R(3, 5)]+[R(1, 4)]*12)
    f2_diag = sy.Matrix([100]*8+[0]*3+[16]+[25]*12)
    hs_diag = sy.Matrix([10]*8+[40]*3+[148]+[0]*12)
    assert error(je@je-np.diag(np.array(je2_diag, float).ravel())) < TOL
    exact_scalar = 2*(f2_diag+hs_diag).dot(je2_diag)+2*n*sum(f2_diag)
    exact_vector = sy.Integer(450)
    assert abs(scalar_coeff-float(exact_scalar)) < TOL
    assert abs(vector_coeff-float(exact_vector)) < TOL

    def potential_and_gradient(coords):
        s = sigma+np.einsum('a,aij->ij', coords[:24], t)
        xx = np.einsum('ka,aij->kij', coords[24:].reshape(n, 24), t)
        z = np.array([jordan(s, x)+6*x for x in xx])
        norm = sum(np.trace(x@x).real for x in xx)
        v = (-74*np.trace(s@s)+np.trace(s@s)**2+np.trace(s@s@s@s)
             +sum(np.trace(a@a) for a in z)+norm*norm).real
        gs = -148*s+4*np.trace(s@s)*s+4*s@s@s
        for x, a in zip(xx, z):
            gs += 2*(x@a+a@x)
        gx = np.array([2*(jordan(s, a)+6*a)+4*norm*x for x, a in zip(xx, z)])
        grad = np.concatenate([np.einsum('aij,ji->a', t, gs).real,
                               np.einsum('aij,kji->ka', t, gx).real.ravel()])
        return float(v), grad

    hessian_errors, potential_gradient_errors = [], []
    rng = np.random.default_rng(6901)
    for x in [0., .2]:
        bg = np.zeros(d)
        bg[34] = x

        def fd_hessian(step):
            out = np.empty((d, d))
            for i in range(d):
                shift = np.zeros(d)
                shift[i] = step
                out[:, i] = (potential_and_gradient(bg+shift)[1]-potential_and_gradient(bg-shift)[1])/(2*step)
            return out

        fd = (4*fd_hessian(.0125)-fd_hessian(.025))/3
        hessian_errors.append(error(fd-(h0+x*h1+x*x*h2)))
        for _ in range(5):
            direction = rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            def derivative(step):
                return (potential_and_gradient(bg+step*direction)[0]
                        -potential_and_gradient(bg-step*direction)[0])/(2*step)
            num = (4*derivative(.025)-derivative(.05))/3
            potential_gradient_errors.append(abs(num-direction@potential_and_gradient(bg)[1]))
    assert max(hessian_errors+potential_gradient_errors) < FD_TOL
    # Trace polynomial evaluated independently at two nonzero backgrounds.
    def supertrace(x):
        hs = h0+x*h1+x*x*h2
        gv = gv0+x*x*gv2
        return float(np.trace(hs@hs)+3*np.trace(gv@gv))
    def quadratic(step):
        return (supertrace(step)+supertrace(-step)-2*supertrace(0))/(2*step*step)
    numeric_coeff = (4*quadratic(.05)-quadratic(.1))/3
    exact_total = exact_scalar+exact_vector
    assert abs(numeric_coeff-float(exact_total)) < FD_TOL
    assert exact_total > 0
    higgs_a = (sigma+3*np.eye(5))@(sigma+3*np.eye(5))
    higgs_hessian = np.block([[higgs_a.real, -higgs_a.imag], [higgs_a.imag, higgs_a.real]])
    assert error(higgs_hessian-np.diag([25]*3+[0]*2+[25]*3+[0]*2)) < TOL
    gamma_symbol = sy.Symbol('gamma', positive=True)
    mass_shape = (25+gamma_symbol**2)*sy.eye(2)+10*gamma_symbol*sy.Matrix([[0, 1], [1, 0]])
    gram_det = sy.simplify(2*sy.trace(mass_shape*mass_shape)-sy.trace(mass_shape)**2)
    assert gram_det == 400*gamma_symbol**2
    return {'variant_real_copies': n, 'scalar_hessian_dimension': d,
            'Higgs_real_components_independent_of_X': 10,
            'Higgs_hessian_eigenvalues': np.linalg.eigvalsh(higgs_hessian).tolist(),
            'fermion_supertrace_X2': 0,
            'jordan_T3_squared_diagonal_exact': list(map(str, je2_diag)),
            'scalar_supertrace_X2_exact': str(exact_scalar),
            'vector_supertrace_X2_exact': str(exact_vector),
            'total_supertrace_X2_exact': str(exact_total),
            'total_units': 'g5^4 V^2 x^2',
            'full_hessian_finite_difference_error': max(hessian_errors),
            'independent_potential_gradient_error': max(potential_gradient_errors),
            'supertrace_quartic_difference_error': abs(numeric_coeff-float(exact_total)),
            'log_mu_derivative_potential_X2': f'-({exact_total}) g5^4 V^2 / (32 pi^2)',
            'additive_light_mass_beta_required': f'({exact_total}) g5^4 V^2 / (16 pi^2)',
            'record_identity_and_original_mass_shape_gram_determinant': str(gram_det),
            'record_copy_symmetry_at_zero_mass': f'O({n})',
            'protected_zero_mass_relation': 'REJECTED_AT_ONE_LOOP',
            'physical_pole_mass_computed': False,
            'renormalized_EFT_with_independent_counterterm_rejected': False}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/69_*.md'))
    manifest_path = folder/'ce_gauge_matching_inputs.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    t, y, sigma, f, op_report = setup()
    gate_reports = [empirical_gate(nu, manifest) for nu in [1, 2]]
    radiation_reports = {item['variant']: radiation(t, sigma, f, 2*nu)
                         for nu, item in zip([1, 2], gate_reports) if item['necessary_feasibility_survives']}
    report = {'schema_version': 1, 'candidate': 'CE-UR2-R/C',
              'scientific_success': False, 'full_joint_rmse': None,
              'preregistration_sha256': hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 69.2')[0].encode()).hexdigest(),
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), manifest_path, folder/'ce_simple_group_record.py']},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'sympy': sy.__version__},
              'observational_manifest': manifest, 'operator': op_report,
              'record': record_check(t, y, f), 'gauge_feasibility': gate_reports,
              'radiative_closure': radiation_reports,
              'effective_potential_source': {'url': 'https://arxiv.org/abs/hep-ph/0111209',
                    'equations': ['1.1', '1.2', '3.1-3.4'], 'gauge': 'Landau', 'scheme': 'MSbar'},
              'open_gates': ['frozen m0 from same dynamics', 'complete renormalized scalar action',
                  'physical actual stable records', 'joint quantum/muon/cosmology/gravity predictions',
                  'dynamical spacetime and all standard limits', 'full covariance and joint RMSE']}
    (folder/'ce_broken_record_operator.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'variants': [{k: item[k] for k in ['variant', 'central_endpoint_bracket',
                      'rejected_by_relaxed_bound', 'necessary_feasibility_survives']} for item in gate_reports],
                      'radiative_closure': radiation_reports}, indent=2))


if __name__ == '__main__':
    main()
