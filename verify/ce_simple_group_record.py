"""Chapter 68: supplied SU(5), neutral multiplicity record, fixed-mass matching.

This replaces LR2 representations. It is not an embedding of its old scalar18,
an autonomous record construction, or a full joint empirical validation.
"""
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
from scipy.linalg import expm
import scipy
import sympy as sy


RAT = sy.Rational
TOL = 1e-10
FD_TOL = 1e-8


def err(value):
    return float(np.max(np.abs(value)))


def su(n):
    """Hermitian fundamental basis with Tr(Ta Tb) = delta_ab / 2."""
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            a = np.zeros((n, n), complex)
            a[i, j] = a[j, i] = 0.5
            result.append(a)
            a = np.zeros((n, n), complex)
            a[i, j], a[j, i] = -0.5j, 0.5j
            result.append(a)
    for k in range(1, n):
        result.append(np.diag([1]*k + [-k] + [0]*(n-k-1))
                      / np.sqrt(2*k*(k+1)))
    return result


def basis():
    result = []
    for t in su(3):
        a = np.zeros((5, 5), complex)
        a[:3, :3] = t
        result.append(a)
    for t in su(2):
        a = np.zeros((5, 5), complex)
        a[3:, 3:] = t
        result.append(a)
    hypercharge = np.diag([-1/3]*3 + [0.5]*2)
    result.append(np.sqrt(3/5)*hypercharge)
    for i in range(3):
        for j in range(3, 5):
            a = np.zeros((5, 5), complex)
            a[i, j] = a[j, i] = 0.5
            result.append(a)
            a = np.zeros((5, 5), complex)
            a[i, j], a[j, i] = -0.5j, 0.5j
            result.append(a)
    return np.array(result), hypercharge


def algebra_and_anomaly(t, y):
    gram = np.einsum('aij,bji->ab', t, t)
    brackets = np.array([[a@b-b@a for b in t] for a in t])
    structure = -2j*np.einsum('abij,cji->abc', brackets, t)
    closure = brackets-1j*np.einsum('abc,cij->abij', structure, t)
    wedge = np.zeros((25, 10), complex)
    for k, (i, j) in enumerate(itertools.combinations(range(5), 2)):
        wedge[5*i+j, k], wedge[5*j+i, k] = 1/np.sqrt(2), -1/np.sqrt(2)
    ten = np.array([wedge.conj().T@(np.kron(a, np.eye(5))
                      + np.kron(np.eye(5), a))@wedge for a in t])
    bar = -t.conj()

    def anomaly(rep):
        cube = np.einsum('aij,bjk,cki->abc', rep, rep, rep, optimize=True)
        return cube + cube.transpose(0, 2, 1)

    anomaly_ten, anomaly_bar = anomaly(ten), anomaly(bar)
    # Independent exact cubic trace for a generic traceless diagonal generator.
    h = [-5, -2, 0, 3, 4]
    diagonal_anomaly = sum((h[i]+h[j])**3 for i, j in
                           itertools.combinations(range(5), 2))-sum(x**3 for x in h)
    central_image = np.diag([np.exp(-2j*np.pi/3)*np.exp(2j*np.pi/3)]*3
                            + [np.exp(1j*np.pi)*(-1)]*2)
    errors = {
        'fundamental_gram': err(gram-np.eye(24)/2),
        'closure_all_pairs': err(closure),
        'real_structure_constants': err(structure.imag),
        'wedge_isometry': err(wedge.conj().T@wedge-np.eye(10)),
        'ten_index': err(np.einsum('aij,bji->ab', ten, ten)-1.5*np.eye(24)),
        'bar_five_index': err(np.einsum('aij,bji->ab', bar, bar)-0.5*np.eye(24)),
        'cubic_anomaly_all_24_cubed': err(anomaly_ten+anomaly_bar),
        'mixed_gravity_trace': err(np.trace(ten, axis1=1, axis2=2)
                                   + np.trace(bar, axis1=1, axis2=2)),
        'Z6_central_kernel_generator': err(central_image-np.eye(5)),
    }
    assert max(errors.values()) < TOL
    assert diagonal_anomaly == 0
    return {
        'generators': 24, 'one_generation': 'wedge^2(5) + conjugate(5)',
        'weyl_generations_supplied': 3, 'errors': errors,
        'individual_cubic_anomaly_nonzero': err(anomaly_ten),
        'independent_exact_diagonal_anomaly': diagonal_anomaly,
        'hypercharge_five': y.diagonal().tolist(),
        'gauge_inverse_alpha_metric': [1, 1, '5/3'],
        'metric_derivation': '2 Tr5(T_color^2, T_weak^2, Y^2)',
        'metric_numeric': [float(2*np.trace(t[0]@t[0]).real),
                           float(2*np.trace(t[8]@t[8]).real), float(2*np.trace(y@y))],
        'boundary_sin_squared_theta_W': '3/8',
        'old_scalar18_embedded': False,
    }


def spectrum(t):
    sigma = np.diag([2, 2, 2, -3, -3]).astype(complex)
    mass_parameter = 74.0

    def potential(a):
        a2 = a@a
        return float((-mass_parameter*np.trace(a2)+np.trace(a2)**2
                      +np.trace(a2@a2)).real)

    comm = np.array([a@sigma-sigma@a for a in t])
    gauge = 2*np.einsum('aij,bij->ab', comm.conj(), comm).real
    hessian = np.empty((24, 24))
    sigma2 = sigma@sigma
    for i, a in enumerate(t):
        for j, b in enumerate(t):
            hessian[i, j] = (
                -2*mass_parameter*np.trace(a@b)
                +8*np.trace(sigma@a)*np.trace(sigma@b)
                +4*np.trace(sigma2)*np.trace(a@b)
                +4*np.trace(a@(sigma2@b+sigma@b@sigma+b@sigma2))
            ).real

    def finite_hessian(step):
        values = np.empty((24, 24))
        for i, a in enumerate(t):
            for j in range(i, 24):
                b = t[j]
                value = (potential(sigma+step*(a+b))-potential(sigma+step*(a-b))
                         -potential(sigma+step*(-a+b))+potential(sigma-step*(a+b)))
                values[i, j] = values[j, i] = value/(4*step*step)
        return values

    fd = (4*finite_hessian(0.05)-finite_hessian(0.1))/3
    # Coefficients of a one-variable polynomial provide an exact check per sector.
    z = sy.Symbol('z', real=True)
    s_exact = sy.diag(2, 2, 2, -3, -3)
    exact_dirs = {
        'octet': sy.diag(1, -1, 0, 0, 0)/2,
        'triplet': sy.diag(0, 0, 0, 1, -1)/2,
        'singlet': s_exact/sy.sqrt(60),
    }
    broken_exact = sy.zeros(5)
    broken_exact[0, 3] = broken_exact[3, 0] = sy.Rational(1, 2)
    exact_dirs['goldstone'] = broken_exact
    exact_masses = {}
    for name, a in exact_dirs.items():
        mat = s_exact+z*a
        poly = -74*sy.trace(mat**2)+sy.trace(mat**2)**2+sy.trace(mat**4)
        exact_masses[name] = sy.expand(poly).coeff(z, 2)*2
    expected_scalar = np.diag([10]*8+[40]*3+[148]+[0]*12)
    expected_gauge = np.diag([0]*12+[25]*12)
    gradient = np.array([np.trace((-148*sigma+120*sigma+4*sigma@sigma2)@a)
                         for a in t])
    higgs = (sigma+3*np.eye(5))@(sigma+3*np.eye(5))
    errors = {'stationarity': err(gradient),
              'gauge_mass': err(gauge-expected_gauge),
              'analytic_scalar_mass': err(hessian-expected_scalar),
              'independent_quartic_finite_difference': err(hessian-fd),
              'higgs_fundamental_mass': err(higgs-np.diag([25]*3+[0]*2))}
    assert errors['independent_quartic_finite_difference'] < FD_TOL
    assert max(v for k, v in errors.items() if 'finite_difference' not in k) < TOL
    assert exact_masses == {'octet': 10, 'triplet': 40, 'singlet': 148, 'goldstone': 0}
    return {'benchmark': {'g5': 1, 'V': 1, 'lambda1': 1, 'lambda2': 1,
                          'm_sigma_squared': 74, 'vEW_over_V': 0},
            'errors': errors, 'gauge_masses_squared_multiplicity': {'0': 12, '25': 12},
            'scalar_masses_squared_multiplicity': {'0': 12, '10': 8, '40': 3, '148': 1},
            'exact_scalar_hessian': {k: str(v) for k, v in exact_masses.items()},
            'mass_ratios_to_MX': {'Sigma8': 'sqrt(10)/5', 'Sigma3': 'sqrt(40)/5',
                                 'Sigma1': 'sqrt(148)/5', 'H_color_triplet': '1'},
            'local_minimum_modulo_gauge_orbit': True, 'global_minimum_proved': False}


def record(t, y):
    pauli_x = np.array([[0, 1], [1, 0]])
    mass = np.kron(5*np.eye(2)+np.pi*pauli_x/2, np.eye(5))
    projection = np.kron(np.diag([0, 1]), np.eye(5))
    generators = np.array([np.kron(np.eye(2), a) for a in t])
    charge = np.kron(np.eye(2), t[10]+y)
    initial = np.zeros(10, complex)
    initial[4] = 1
    eigenvalues, eigenvectors = np.linalg.eigh(mass)
    kg_to_field = eigenvectors@np.diag(1/np.sqrt(2*eigenvalues))@eigenvectors.conj().T
    times = np.array([0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0])
    rows, choi_minima, cp_errors, kg_errors, energy_errors = [], [], [], [], []
    for time in times:
        u = expm(-1j*time*mass)
        psi = u@initial
        field = kg_to_field@psi
        field_dot = -1j*mass@field
        kg_norm = 1j*(np.vdot(field, field_dot)-np.vdot(field_dot, field))
        kg_errors.append(float(abs(kg_norm-np.vdot(psi, psi))))
        energy_errors.append(float(abs(np.vdot(psi, mass@psi)-np.vdot(initial, mass@initial))))
        prob = float(np.vdot(psi, projection@psi).real)
        rows.append({'m0_time': float(time), 'record_probability': prob,
                     'sine_squared': float(np.sin(np.pi*time/2)**2),
                     'norm_error': float(abs(np.vdot(psi, psi)-1)),
                     'neutral_error': err(charge@psi)})
        kraus = [projection@u, (np.eye(10)-projection)@u]
        # The Choi matrix is a sum of outer products; this is an instrument,
        # not a construction of an actual autonomous physical readout.
        choi = sum(np.outer(k.reshape(-1, order='F'),
                            k.reshape(-1, order='F').conj()) for k in kraus)
        choi_minima.append(float(np.linalg.eigvalsh(choi)[0]))
        cp_errors.append(err(sum(k.conj().T@k for k in kraus)-np.eye(10)))
    errors = {
        'mass_gauge_commutator': max(err(mass@a-a@mass) for a in generators),
        'record_gauge_commutator': max(err(projection@a-a@projection) for a in generators),
        'charge_record_commutator': err(charge@projection-projection@charge),
        'record_probability': max(abs(row['record_probability']-row['sine_squared']) for row in rows),
        'norm': max(row['norm_error'] for row in rows),
        'neutrality': max(row['neutral_error'] for row in rows),
        'kraus_completeness': max(cp_errors),
        'positive_frequency_KG_norm': max(kg_errors),
        'free_rest_energy': max(energy_errors),
    }
    assert max(errors.values()) < TOL and min(choi_minima) > -TOL
    assert np.linalg.eigvalsh(mass)[0] > 0
    return {'free_rest_positive_frequency_limit_only': True,
            'mass_eigenvalues': np.linalg.eigvalsh(mass).tolist(),
            'errors': errors, 'probabilities': rows,
            'choi_minimum_roundoff': min(choi_minima),
            'norm_preserved_without_discarding_amplitudes': True,
            'autonomous_actual_record_constructed': False,
            'instrument_energy_change_operator_norm': float(np.linalg.norm(
                projection@mass@projection+(np.eye(10)-projection)@mass@(np.eye(10)-projection)-mass, 2)),
            'mass_record_commutator_operator_norm': float(np.linalg.norm(mass@projection-projection@mass, 2))}


def matching(t, y, manifest):
    k = sy.Matrix([1, 1, RAT(5, 3)])
    b_sm = sy.Matrix([-7, -RAT(19, 6), RAT(41, 6)])
    db_five = k/6
    db_octet = sy.Matrix([RAT(1, 2), 0, 0])
    db_triplet = sy.Matrix([0, RAT(1, 3), 0])
    db_hcolor = sy.Matrix([RAT(1, 6), 0, RAT(1, 9)])
    # Derive massive-vector indices from the real broken adjoint representation.
    broken = t[12:]
    subgroup = [t[0], t[8], y]
    vec_indices = []
    for a in subgroup:
        action = np.array([[2*np.trace(b@(a@c-c@a)) for c in broken] for b in broken])
        vec_indices.append(float(np.trace(action@action).real))
    sv = sy.Matrix([2, 3, RAT(25, 3)])
    assert err(np.array(vec_indices)-np.array(sv, float).ravel()) < TOL
    physical_scalar_errors = []
    for scalar_basis, expected_db in [(t[:8], db_octet), (t[8:11], db_triplet)]:
        for a, expected in zip(subgroup, expected_db):
            action = np.array([[2*np.trace(b@(a@c-c@a)) for c in scalar_basis]
                               for b in scalar_basis])
            physical_scalar_errors.append(abs(float(np.trace(action@action).real/6)-float(expected)))
    for a, expected in zip(subgroup, db_hcolor):
        physical_scalar_errors.append(abs(float(np.trace(a[:3, :3]@a[:3, :3]).real/3)-float(expected)))
    assert max(physical_scalar_errors) < TOL
    # Direct fundamental scalar index, using canonical realification separately.
    five_indices = [float(np.trace(a@a).real) for a in subgroup]
    scalar_db_errors = []
    for a, expected in zip(subgroup, db_five):
        real_generator = np.block([[-a.imag, -a.real], [a.real, -a.imag]])
        real_index = -np.trace(real_generator@real_generator).real
        scalar_db_errors.append(abs(real_index/6-float(expected)))
    assert max(scalar_db_errors) < TOL
    b5 = -RAT(11, 3)*5+RAT(2, 3)*3*(RAT(3, 2)+RAT(1, 2))
    b5 += RAT(1, 3)*(2*RAT(1, 2)+RAT(1, 2))+RAT(1, 6)*5
    db_vector = -RAT(7, 2)*sv
    threshold_beta_identity = k*b5-(b_sm+2*db_five)-db_vector-db_octet-db_triplet-db_hcolor
    assert b5 == -13 and threshold_beta_identity == sy.zeros(3, 1)
    # Integrate heavy scalars at MX with their fixed tree masses. Vector finite
    # term: source eqs (7)-(8), MSbar, Goldstones excluded from scalar sum.
    delta_vec = -sv/(12*sy.pi)
    delta_8 = db_octet*sy.log(5/sy.sqrt(10))/(2*sy.pi)
    delta_3 = db_triplet*sy.log(5/sy.sqrt(40))/(2*sy.pi)
    delta = delta_vec+delta_8+delta_3
    # Eliminate the common inverse coupling and logarithmic interval exactly.
    y2, yy = sy.symbols('y2 yy')
    solution = sy.solve([k[0]-y2*k[1]-yy*k[2],
                         b_sm[0]-y2*b_sm[1]-yy*b_sm[2]], [y2, yy])
    y2, yy = solution[y2], solution[yy]
    w = sy.Matrix([1, -y2, -yy])
    assert w.dot(k) == 0 and w.dot(b_sm) == 0 and w.dot(db_five) == 0
    projected_delta = sy.expand_log(w.dot(delta), force=True).expand().collect([sy.log(2), sy.log(5)])
    obs = manifest['observables']
    mul = RAT(str(manifest['error_box_multiplier']))

    def decimal(key, field='value'):
        return RAT(str(obs[key][field]))

    A, sw = decimal('alpha_em_inverse'), decimal('sin_squared_theta_W')
    Ae = mul*decimal('alpha_em_inverse', 'quoted_error')
    swe = mul*decimal('sin_squared_theta_W', 'quoted_error')
    observed_low = decimal('alpha_s')-mul*decimal('alpha_s', 'quoted_error')
    observed_high = decimal('alpha_s')+mul*decimal('alpha_s', 'quoted_error')
    matrix_ew = np.array([[float(k[1]), float(b_sm[1])],
                          [float(k[2]), float(b_sm[2])]])
    delta_num = np.array(delta.evalf(60), float).ravel()
    rows = []
    direct_errors = []
    for label, a, s in [('central', A, sw)]+[
            (f'corner_{ia}_{iss}', A+ia*Ae, sw+iss*swe)
            for ia, iss in itertools.product([-1, 1], repeat=2)]:
        x2, xy = a*s, a*(1-s)
        predicted_x3 = y2*x2+yy*xy+projected_delta
        predicted_alpha = 1/predicted_x3
        common, log_interval_over_2pi = np.linalg.solve(
            matrix_ew, [float(x2)-delta_num[1], float(xy)-delta_num[2]])
        direct_x3 = common+float(b_sm[0])*log_interval_over_2pi+delta_num[0]
        direct_errors.append(abs(direct_x3-float(predicted_x3)))
        rows.append({'input': label, 'alpha_em_inverse': float(a), 'sin_squared': float(s),
                     'predicted_alpha_s': float(predicted_alpha),
                     'predicted_inverse_alpha_s': float(predicted_x3),
                     'effective_common_inverse_alpha': float(common),
                     'log_MX_over_MZ': float(2*np.pi*log_interval_over_2pi)})
    assert max(direct_errors) < TOL
    # Analytic extrema: x3 = A*(yY + (y2-yY)*s) + constant; derivatives positive
    # on the entire registered box, so maximal alpha_s occurs at its lower corner.
    assert y2-yy > 0 and yy+(y2-yy)*(sw-swe) > 0
    optimistic_x3 = y2*(A-Ae)*(sw-swe)+yy*(A-Ae)*(1-sw+swe)+projected_delta
    margin = optimistic_x3-1/observed_low
    assert sy.N(margin, 70) > 0
    # A strict exclusion certificate needs no floating logarithm: integral bounds
    # 1/2 <= log(2), log(5) <= 4 imply the numerator below is >= 50 > 0.
    # Discarding this positive fixed correction leaves a positive rational margin.
    projected_simple = (112*sy.log(2)-sy.log(5)-2)/(436*sy.pi)
    assert sy.simplify(sy.expand_log(projected_delta-projected_simple, force=True)) == 0
    positive_numerator_lower = 112*RAT(1, 2)-4-2
    rational_margin = sy.simplify(optimistic_x3-projected_delta-1/observed_low)
    assert positive_numerator_lower > 0 and rational_margin > 0
    # Low and high precision agreement checks decimal evaluation; it does not
    # promote omitted loop orders to an error bound.
    precision_error = abs(float(sy.N(margin, 30))-float(sy.N(margin, 80)))
    # Test matching-scale covariance without using observable values: the RHS
    # derivative is -k*b5 + b_low + sum_heavy(db) = 0 (units 1/(2*pi)).
    scale_derivative = -k*b5+b_sm+2*db_five+db_vector+db_octet+db_triplet+db_hcolor
    assert scale_derivative == sy.zeros(3, 1)
    return {
        'order': 'one-loop running plus one-loop MSbar heavy matching',
        'gauge_order': ['3', '2', 'Y_unnormalized'],
        'b_SM': list(map(str, b_sm)), 'b_SU5': str(b5),
        'scalar_five_indices_numeric': five_indices,
        'scalar_realification_beta_error': max(scalar_db_errors),
        'physical_heavy_scalar_beta_matrix_error': max(physical_scalar_errors),
        'massive_vector_indices_from_matrices': vec_indices,
        'massive_vector_indices_exact': list(map(str, sv)),
        'massive_vector_beta_with_goldstone': list(map(str, db_vector)),
        'scalar_heavy_beta': {'Sigma8': list(map(str, db_octet)),
                              'Sigma3': list(map(str, db_triplet)),
                              'H_color': list(map(str, db_hcolor))},
        'threshold_beta_identity_exact': list(map(str, threshold_beta_identity)),
        'matching_scale_derivative_exact': list(map(str, scale_derivative)),
        'complete_record_five_beta': list(map(str, db_five)),
        'record_five_projected_threshold_exact': str(w.dot(db_five)),
        'y2': str(y2), 'yY': str(yy),
        'finite_vector_delta': [float(v) for v in delta_vec],
        'scalar_log_delta': [float(v) for v in delta_8+delta_3],
        'projected_delta_exact': str(projected_delta),
        'projected_delta_log_numerator': '112 log(2) - log(5) - 2',
        'projected_delta_log_denominator': '436 pi',
        'positive_delta_numerator_lower_bound_exact': str(positive_numerator_lower),
        'projected_delta_numeric': float(projected_delta),
        'conditional_predictions': rows,
        'direct_elimination_error': max(direct_errors),
        'precision_comparison_error': precision_error,
        'box_alpha_s_upper': float(1/optimistic_x3),
        'observed_alpha_s_box': [float(observed_low), float(observed_high)],
        'inverse_alpha_exclusion_margin': str(sy.N(margin, 60)),
        'strict_rational_exclusion_margin_discarding_positive_delta': str(rational_margin),
        'additional_projected_correction_required_at_least': -float(margin),
        'result': 'REJECTED_FIXED_MASS_ONE_LOOP_MATCHING_BRANCH',
        'whole_SU5_or_all_loop_orders_rejected': False,
        'alpha_s_fit_parameters': 0,
        'conditioning_inputs_count': 2,
        'nuisance_note': 'Common inverse coupling and MX are eliminated using two EW inputs; record m0 is not selected.',
        'independent_unknown_thresholds_fit': False,
    }


def main():
    base = Path(__file__).resolve().parent
    chapter = next((base.parent/'paper').glob('06_*/68_*.md'))
    prereg = chapter.read_text(encoding='utf-8').split('## 68.2')[0]
    manifest_path = base/'ce_gauge_matching_inputs.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    t, y = basis()
    report = {
        'schema_version': 1, 'candidate': 'CE-UR1',
        'scientific_success': False, 'full_joint_rmse': None,
        'preregistration_sha256': hashlib.sha256(prereg.encode()).hexdigest(),
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), manifest_path]},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'sympy': sy.__version__},
        'threshold_source': {
            'url': 'https://link.springer.com/article/10.1140/epjc/s10052-019-6878-1',
            'equations': ['7', '8'], 'scheme': 'MSbar',
            'goldstones': 'included with massive vectors, excluded from physical scalar sum'},
        'observational_manifest': manifest,
        'algebra': algebra_and_anomaly(t, y),
        'spectrum': spectrum(t),
        'record': record(t, y),
        'matching': matching(t, y, manifest),
        'open_gates': ['actual asynchronous stable record', 'dynamical spacetime from same axiom',
                       'full common quantum/muon/cosmology/gravity predictions and covariance',
                       'fermion masses and Yukawa relations', 'radiative scalar stability',
                       'two-loop running and higher matching orders', 'proton decay'],
    }
    path = base/'ce_simple_group_record.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': report['candidate'],
                      'max_algebra_error': max(report['algebra']['errors'].values()),
                      'spectrum_errors': report['spectrum']['errors'],
                      'central_alpha_s': report['matching']['conditional_predictions'][0]['predicted_alpha_s'],
                      'box_alpha_s_upper': report['matching']['box_alpha_s_upper'],
                      'margin': report['matching']['inverse_alpha_exclusion_margin'],
                      'result': report['matching']['result']}, indent=2))


if __name__ == '__main__':
    main()
