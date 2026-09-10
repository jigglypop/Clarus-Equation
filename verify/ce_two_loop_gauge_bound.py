"""Chapter 67: exact box bounds for a specified two-loop gauge truncation.

Adds a supplied scalar Z2 and a bounded perturbative domain. No new scalar
Yukawas, no finite matching, no fit, and no all-orders exclusion claim.
"""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import sympy as sy

from ce_matter_representation_anomalies import matrices
from ce_record_weak_symmetry import PAULI, old_model, extended_model
from ce_gauge_matching_gate import CANDIDATES, exact_certificate


R = sy.Rational
CA = sy.Matrix([3, 2, 0])
DG = (8, 3, 1)
ALPHA_CAP = F(1, 5)
YUKAWA_CAP = F(3)
TOL = 1e-9


def error(x):
    return float(np.max(np.abs(x)))


def fraction(x):
    x = sy.Rational(x)
    return F(int(x.p), int(x.q))


def scalar_increment(casimir, dimension):
    index = sy.Matrix([casimir[i]*dimension/DG[i] for i in range(3)])
    b = index/3
    B = sy.Matrix(3, 3, lambda i, j: 4*index[i]*casimir[j])
    for i in range(3):
        B[i, i] += R(2, 3)*CA[i]*index[i]
    return b, B


def coefficient_tables():
    # One generation: dimension, C2(color), C2(weak), Y^2.
    fermions = [(6, R(4, 3), R(3, 4), R(1, 36)),
                (3, R(4, 3), 0, R(4, 9)), (3, R(4, 3), 0, R(1, 9)),
                (2, 0, R(3, 4), R(1, 4)), (1, 0, 0, 1)]
    b = -R(11, 3)*CA
    B = sy.diag(*[-R(34, 3)*x*x for x in CA])
    for dimension, *values in fermions:
        cas = sy.Matrix(values)
        index = sy.Matrix([cas[i]*dimension/DG[i] for i in range(3)])
        b += 3*R(2, 3)*index
        B += 3*sy.Matrix(3, 3, lambda i, j: 2*index[i]*cas[j])
        for i in range(3):
            B[i, i] += 3*R(10, 3)*CA[i]*index[i]
    hb, hB = scalar_increment(sy.Matrix([0, R(3, 4), R(1, 4)]), 2)
    b, B = b+hb, B+hB
    assert b == sy.Matrix([-7, -R(19, 6), R(41, 6)])
    assert B == sy.Matrix([[-26, R(9, 2), R(11, 6)],
                           [12, R(35, 6), R(3, 2)],
                           [R(44, 3), R(9, 2), R(199, 18)]])
    scalar_reps = [(3, [R(4, 3), 0, 0]), (3, [R(4, 3), 0, 0]),
                   (8, [3, 0, 0]), (2, [0, R(3, 4), R(1, 4)]),
                   (2, [0, R(3, 4), R(1, 4)])]
    increments = [scalar_increment(sy.Matrix(c), d) for d, c in scalar_reps]
    return b, B, increments, fermions


def casimirs(gens):
    return [sum((g@g for g in part), np.zeros_like(gens[0], dtype=complex))
            for part in (gens[:8], gens[8:11], gens[11:])]


def scalar_matrix_increment(cas, projector):
    index = np.array([np.trace(projector@c).real/d for c, d in zip(cas, DG)])
    B = np.array([[4*np.trace(projector@cas[i]@cas[j]).real/DG[i] for j in range(3)] for i in range(3)])
    for i in range(3):
        B[i, i] += 2/3*float(CA[i])*index[i]
    return index/3, B


def matrix_checks(b, B, increments, fermions):
    raw = matrices((1, -4, 2, -3, 6))
    gen_f = raw[4:]+raw[1:4]+[raw[0]/6]
    cf = casimirs(gen_f)
    index_f = np.array([3*np.trace(c).real/d for c, d in zip(cf, DG)])
    direct_b = -11/3*np.array(CA, float).ravel()+2/3*index_f
    direct_B = np.diag(-34/3*np.array(CA, float).ravel()**2)
    direct_B += np.array([[6*np.trace(cf[i]@cf[j]).real/DG[i] for j in range(3)] for i in range(3)])
    for i in range(3):
        direct_B[i, i] += 10/3*float(CA[i])*index_f[i]
    gen_h = [np.zeros((2, 2), complex) for _ in range(8)]+[p/2 for p in PAULI]+[np.eye(2)/2]
    ch = casimirs(gen_h)
    hb, hB = scalar_matrix_increment(ch, np.eye(2))
    direct_b, direct_B = direct_b+hb, direct_B+hB
    errors = dict(SM_one_loop=error(direct_b-np.array(b, float).ravel()),
                  SM_two_loop=error(direct_B-np.array(B, float)))

    initial, singlets, outside, mass, record, color = old_model()
    _, gen_s = extended_model(initial, singlets, outside, mass, record, color,
                              np.ones(2, complex)/np.sqrt(2))
    cs = casimirs(gen_s)
    projectors = []
    for start, end in ((0, 3), (3, 6), (6, 14)):
        p = np.zeros((18, 18), complex)
        p[start:end, start:end] = np.eye(end-start)
        projectors.append(p)
    for sign in (-1, 1):
        p = np.zeros((18, 18), complex)
        p[14:, 14:] = (np.eye(4)+sign*np.kron(PAULI[0], np.eye(2)))/2
        projectors.append(p)
    assert error(sum(projectors)-np.eye(18)) < TOL
    scalar_error = 0.
    for p, (db, dB) in zip(projectors, increments):
        actual_b, actual_B = scalar_matrix_increment(cs, p)
        scalar_error = max(scalar_error, error(actual_b-np.array(db, float).ravel()),
                           error(actual_B-np.array(dB, float)))
    errors['all_scalar_increments'] = scalar_error
    # Realification directly verifies the factor of two in a real scalar trace.
    real_error = 0.
    for g in gen_s+gen_h:
        theta = -1j*np.block([[-g.imag, -g.real], [g.real, -g.imag]])
        real_error = max(real_error, abs(np.trace(theta@theta).real-2*np.trace(g@g).real))
    errors['complex_vs_real_trace'] = float(real_error)

    exact_cas = [sy.diag(*[value for d, *c in fermions for value in [c[i]]*d]) for i in range(3)]
    y_matrices = []
    for sector in ('u', 'd', 'e'):
        ys = [sy.zeros(15) for _ in range(4)]

        def link(component, a, z, value):
            ys[component][a, z] += value/sy.sqrt(2)
            ys[component][z, a] += value/sy.sqrt(2)

        if sector == 'u':
            for color_index in range(3):
                q, u = 2*color_index, 6+color_index
                for component, qi, value in ((2, q, 1), (3, q, sy.I),
                                              (0, q+1, -1), (1, q+1, -sy.I)):
                    link(component, qi, u, value)
        elif sector == 'd':
            for color_index in range(3):
                q, d = 2*color_index, 9+color_index
                for component, qi, value in ((0, q, 1), (1, q, -sy.I),
                                              (2, q+1, 1), (3, q+1, -sy.I)):
                    link(component, qi, d, value)
        else:
            for component, ell, value in ((0, 12, 1), (1, 12, -sy.I),
                                         (2, 13, 1), (3, 13, -sy.I)):
                link(component, ell, 14, value)
        y_matrices.append(ys)
    columns, numeric_columns = [], []
    for ys in y_matrices:
        square = sum((y*y.conjugate().T for y in ys), sy.zeros(15))
        columns.append(sy.Matrix([sy.trace(c*square)/d for c, d in zip(exact_cas, DG)]))
        numeric = np.array(square, complex)
        numeric_columns.append([np.trace(c@numeric).real/d for c, d in zip(cf, DG)])
    D = sy.Matrix.hstack(*columns)
    assert D == sy.Matrix([[2, 2, 0], [R(3, 2), R(3, 2), R(1, 2)],
                           [R(17, 6), R(5, 6), R(5, 2)]])
    errors['Yukawa_Casimir_trace'] = error(np.array(numeric_columns).T-np.array(D, float))
    assert max(errors.values()) < TOL
    return D, errors


def pi_enclosure():
    def atan_interval(denominator):
        z, value = F(1, denominator), F(0)
        for n in range(20):
            value += (-1)**n*z**(2*n+1)/(2*n+1)
        next_term = z**41/41  # next sign is positive
        return value, value+next_term
    low5, high5 = atan_interval(5)
    low239, high239 = atan_interval(239)
    t = F(1, 5)
    double = 2*t/(1-t*t)
    quadruple = 2*double/(1-double*double)
    assert (quadruple-F(1, 239))/(1+quadruple/F(239)) == 1
    return 16*low5-4*high239, 16*high5-4*low239


def rational_rate_interval(c1, c2, c3, pi):
    lower = upper = F(0)
    for coefficient, denominator, power in ((c1, 2, 1), (c2, 8, 2), (c3, 32, 3)):
        lo = coefficient/(denominator*pi[1]**power)
        hi = coefficient/(denominator*pi[0]**power)
        lower += min(lo, hi)
        upper += max(lo, hi)
    return lower, upper


def extrema(projection, include_one_loop, b, B, D, increments, pi):
    exact_lowers, numerical_values = [], []
    for flags in itertools.product((0, 1), repeat=5):
        active_b = b+sum((f*db for f, (db, _) in zip(flags, increments)), sy.zeros(3, 1))
        active_B = B+sum((f*dB for f, (_, dB) in zip(flags, increments)), sy.zeros(3))
        first = fraction((projection.T*active_b)[0]) if include_one_loop else F(0)
        gauge = list(projection.T*active_B)
        yukawa = list(projection.T*D)
        c2 = ALPHA_CAP*sum((min(F(0), fraction(c)) for c in gauge), F(0))
        c3 = -YUKAWA_CAP*sum((max(F(0), fraction(c)) for c in yukawa), F(0))
        interval = rational_rate_interval(first, c2, c3, pi)
        exact_lowers.append((interval[0], flags, [first, c2, c3]))
        # Independently enumerate all alpha and Yukawa vertices in floating point.
        for av in itertools.product((0., float(ALPHA_CAP)), repeat=3):
            for tv in itertools.product((0., float(YUKAWA_CAP)), repeat=3):
                value = float(first)/(2*np.pi)
                value += np.dot(np.array(gauge, float), av)/(8*np.pi**2)
                value -= np.dot(np.array(yukawa, float), tv)/(32*np.pi**3)
                numerical_values.append(float(value))
    best = min(exact_lowers, key=lambda x: x[0])
    numeric_minimum = min(numerical_values)
    assert abs(float(best[0])-numeric_minimum) < TOL
    return best[0], dict(certified_lower_fraction=str(best[0]), lower_approximation=float(best[0]),
                         minimizer_threshold_flags=list(best[1]),
                         rate_numerators_exact=list(map(str, best[2])),
                         independently_enumerated_vertices=len(numerical_values),
                         independent_minimum=numeric_minimum,
                         independent_error=abs(float(best[0])-numeric_minimum))


def exclusion_bounds(name, k, inputs, b, B, D, increments, pi):
    _, certificate = exact_certificate(k)
    y2, yy = certificate['y2'], certificate['yY']
    q_projection = sy.Matrix([0, -k[2], 1])
    delta_projection = sy.Matrix([1, -y2, -yy])
    q_lower, q_report = extrema(q_projection, True, b, B, D, increments, pi)
    rate_lower, rate_report = extrema(delta_projection, False, b, B, D, increments, pi)
    assert q_lower > 0 and rate_lower < 0
    obs = inputs['observables']
    factor = F(inputs['error_box_multiplier'])
    def value_bounds(key):
        center, err = F(str(obs[key]['value'])), F(str(obs[key]['quoted_error']))
        return center-factor*err, center+factor*err
    A_bounds, s_bounds = value_bounds('alpha_em_inverse'), value_bounds('sin_squared_theta_W')
    rhs_values, one_loop_values = [], []
    for A, s in itertools.product(A_bounds, s_bounds):
        x2, xy = A*s, A*(1-s)
        rhs_values.append(xy-fraction(k[2])*x2)
        one_loop_values.append(fraction(y2)*x2+fraction(yy)*xy)
    length_upper = max(rhs_values)/q_lower
    delta_lower = rate_lower*length_upper
    inverse_lower = min(one_loop_values)+delta_lower
    strong_lower, _ = value_bounds('alpha_s')
    observed_inverse_upper = 1/strong_lower
    margin = inverse_lower-observed_inverse_upper
    assert inverse_lower > 0
    return dict(candidate=name+'-2L-Z2', scope='all trajectories inside registered boxes; finite matching zero',
                positive_length_rate=q_report, projected_two_loop_rate=rate_report,
                maximum_log_matching_length=float(length_upper),
                projected_two_loop_correction_lower=float(delta_lower),
                one_loop_inverse_alpha_lower=float(min(one_loop_values)),
                corrected_inverse_alpha_lower=float(inverse_lower),
                observed_inverse_alpha_upper=float(observed_inverse_upper),
                corrected_alpha_s_upper=float(1/inverse_lower), observed_alpha_s_lower=float(strong_lower),
                positive_exclusion_margin=float(margin),
                exact_certificate=dict(length_upper=str(length_upper), delta_lower=str(delta_lower),
                                       inverse_lower=str(inverse_lower), margin=str(margin)),
                status='REJECTED_IN_REGISTERED_DOMAIN' if margin > 0 else 'BOUND_INCONCLUSIVE',
                all_orders_status='NOT_EVALUATED')


def run():
    root = Path(__file__).resolve().parents[1]
    inp = root/'verify/ce_gauge_matching_inputs.json'
    inputs = json.loads(inp.read_text(encoding='utf-8'))
    b, B, increments, fermions = coefficient_tables()
    D, checks = matrix_checks(b, B, increments, fermions)
    pi = pi_enclosure()
    assert pi[0] < pi[1] and pi[1]-pi[0] < F(1, 10**28)
    results = [exclusion_bounds(name, k, inputs, b, B, D, increments, pi) for name, k in CANDIDATES]
    chapter = next((root/'paper').glob('06_*/67_*.md'))
    prereg = chapter.read_text(encoding='utf-8').split('## 67.2')[0]
    names = ['ce_gauge_matching_gate.py', 'ce_matter_representation_anomalies.py',
             'ce_common_isotropic_frame.py', 'ce_record_weak_symmetry.py',
             'ce_color_covariant_record.py', 'ce_isometric_color_frame.py']
    sources = [Path(__file__), inp]+[root/'verify'/name for name in names]
    return dict(candidate='CE-LR3-2L-Z2 box extension', assumptions=dict(alpha_cap=str(ALPHA_CAP),
                each_SM_Yukawa_trace_cap=str(YUKAWA_CAP), new_scalar_Z2_odd=True,
                new_scalar_Yukawas_zero=True, finite_matching_zero=True),
                SM_b_exact=list(map(str, b)), SM_B_exact=[[str(x) for x in row] for row in B.tolist()],
                scalar_B_increments_exact=[[[str(x) for x in row] for row in dB.tolist()] for _, dB in increments],
                Yukawa_D_exact=[[str(x) for x in row] for row in D.tolist()], matrix_checks=checks,
                pi_interval_exact=list(map(str, pi)), cases=results,
                fitted_parameters=0, full_joint_rmse=None, scientific_success=False, narrow_checks_passed=True,
                limitations=['specified two-loop truncation only; no omitted-order remainder bound',
                             'all trajectories are assumed to stay within the registered domain',
                             'new scalar Z2 is supplied; no new scalar Yukawas included',
                             'no finite matching or gravity loops',
                             'no full Yukawa/scalar evolution, stable record or quantum gravity construction',
                             'error box uses marginal summaries and is not a confidence region'],
                preregistration_section_sha256=hashlib.sha256(prereg.encode('utf-8')).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                                 sympy=sy.__version__, platform=platform.platform()))


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(checks=result['matrix_checks'], cases=[{k: row[k] for k in (
        'candidate', 'maximum_log_matching_length', 'projected_two_loop_correction_lower',
        'corrected_alpha_s_upper', 'observed_alpha_s_lower', 'positive_exclusion_margin', 'status')}
        for row in result['cases']], scientific_success=False), indent=2))
