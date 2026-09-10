"""CE-GC2: compatible block connection, scalar-loop kinetics and excess centers.

The full U(6)/SU(6) connection, scalar action and flat metric are supplied.
This audits those candidates; it does not identify a derived physical group.
"""

import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from ce_isometric_color_frame import comm, exact_rank
from ce_projector_one_loop import coefficient, parameter


TOL = 1e-12
DIMS = np.array([1, 2, 3])
MASSES2 = np.array([1., 2., 5.])
R = [np.diag([1., 0., 0., 0., 0., 0.]),
     np.diag([0., 1., 1., 0., 0., 0.]),
     np.diag([0., 0., 0., 1., 1., 1.])]
X = sum(c * p for c, p in zip(MASSES2, R))


def error(a):
    return float(np.max(np.abs(a)))


def basis(n):
    result = [np.diag(np.eye(n)[i]).astype(complex) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.zeros((n, n), complex), np.zeros((n, n), complex)
            a[i, j] = a[j, i] = 1
            b[i, j], b[j, i] = -1j, 1j
            result.extend((a, b))
    return result


def center_dimension(generators):
    maps = [np.concatenate([comm(a, b) for b in generators], axis=0) for a in generators]
    return len(generators) - exact_rank(maps)


def algebra_audit():
    generators = basis(6)
    rank = exact_rank([comm(X, b) for b in generators])
    retained = [b for b in generators if error(comm(X, b)) == 0]
    trace_constrained_rank = exact_rank([np.r_[comm(X, b).ravel(), np.trace(b)] for b in generators])
    traceless = [b - generators[5] for b in generators[:5]] + retained[6:]
    assert len(retained) == 36 - rank == 14
    assert len(traceless) == 36 - trace_constrained_rank == 13
    assert center_dimension(retained) == 3
    assert center_dimension(traceless) == 2
    pairs = []
    mass_form = np.zeros((36, 36))
    for i in range(3):
        for j in range(i + 1, 3):
            c = coefficient(MASSES2[i], MASSES2[j])
            _, independent_c, _ = parameter(MASSES2[i], MASSES2[j], 0., 96)
            assert abs(c - independent_c) < TOL and c > 0
            projected = [R[i] @ b @ R[j] for b in generators]
            mass_form += c * np.array([[np.vdot(a, b).real for b in projected] for a in projected])
            pairs.append({'blocks': [i + 1, j + 1], 'finite_C_UV_limit': c,
                          'real_off_block_directions': int(2 * DIMS[i] * DIMS[j])})
    eigenvalues = np.linalg.eigvalsh(mass_form)
    assert sum(eigenvalues > TOL) == 22
    assert sum(abs(eigenvalues) < TOL) == 14
    return {'U6_commutant_dimension': len(retained), 'U6_center_dimension': 3,
            'SU6_commutant_dimension': len(traceless), 'SU6_center_dimension': 2,
            'desired_algebra_dimension': 12, 'desired_center_dimension': 1,
            'off_block_normal_form_rank': 22, 'off_block_normal_form_kernel': 14,
            'off_block_coefficients': pairs}


def flag_audit(t):
    g = np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1)
    values, vectors = np.linalg.eigh(g)
    u = (vectors * np.exp(1j * t * values)) @ vectors.conj().T
    projectors = [u @ p @ u.conj().T for p in R]
    derivatives = [1j * comm(g, p) for p in projectors]
    minimal = sum(d @ p for d, p in zip(derivatives, projectors))
    transformed_blocks = sum(p @ (1j * g) @ p for p in projectors)
    restored = minimal + transformed_blocks
    errors = {'sum_projectors': error(sum(projectors) - np.eye(6)),
              'antihermitian_minimal': error(minimal + minimal.conj().T),
              'compatible_flags': max(error(d - comm(minimal, p)) for d, p in zip(derivatives, projectors)),
              'half_commutator_formula': error(minimal - sum(comm(d, p) for d, p in zip(derivatives, projectors)) / 2),
              'full_gauge_connection_restored': error(restored - 1j * g)}
    assert max(errors.values()) < TOL
    return {'t': t, 'errors': errors}


def e1_series(z):
    term = 1.
    terms = []
    for k in range(1, 100):
        term *= -z / k
        terms.append(term / k)
    return -0.5772156649015328606 - math.log(z) - math.fsum(terms)


def e1_integral(z, nodes):
    # E1(z)=integral_0^infty exp(-z exp(u)) du. Stop at z exp(u)=40.
    stop = math.log(40 / z)
    x, w = np.polynomial.legendre.leggauss(nodes)
    u = (x + 1) * stop / 2
    value = float(np.dot(w, np.exp(-z * np.exp(u))) * stop / 2)
    return value, math.exp(-40) / 40


def kinetic_audit(cutoff2):
    integrals = []
    for mass2 in MASSES2:
        z = mass2 / cutoff2
        expected = e1_series(z)
        independent = [e1_integral(z, n) for n in (64, 128)]
        err = max(abs(v - expected) for v, _ in independent)
        assert err < 1e-9 and expected > 0
        integrals.append({'mass_squared': mass2, 'E1': expected, 'absolute_integral_error': err,
                          'quadrature_values': [v for v, _ in independent],
                          'omitted_positive_tail_bound': independent[0][1]})
    c = np.array([row['E1'] for row in integrals]) / (192 * np.pi**2)
    # b=(-2u-3v,u,v) is the entire traceless central subspace.
    embedding = np.array([[-2., -3.], [1., 0.], [0., 1.]])
    assert error(DIMS @ embedding) == 0.
    u6_center = np.diag(DIMS * c)
    su6_center = embedding.T @ u6_center @ embedding
    assert min(np.linalg.eigvalsh(u6_center)) > 0.
    assert min(np.linalg.eigvalsh(su6_center)) > 0.
    return {'cutoff_squared': cutoff2, 'proper_time_integrals': integrals,
            'trace_F_squared_coefficients': c.tolist(),
            'U6_center_kinetic_eigenvalues': np.linalg.eigvalsh(u6_center).tolist(),
            'SU6_center_kinetic_matrix': su6_center.tolist(),
            'SU6_center_kinetic_eigenvalues': np.linalg.eigvalsh(su6_center).tolist(),
            'linearized_Abelian_polarizations_U6': 6,
            'linearized_Abelian_polarizations_SU6': 4}


def maxwell_audit():
    f = np.zeros((4, 4), int)
    f[0, 3], f[3, 0] = 1, -1
    f[1, 2], f[2, 1] = 1, -1
    pfaffian = int(f[0, 1] * f[2, 3] - f[0, 2] * f[1, 3] + f[0, 3] * f[1, 2])
    traceless_center = 2 * R[0] - R[1]
    assert np.trace(traceless_center) == 0.
    assert all(error(comm(traceless_center, p)) == 0 for p in R)
    assert np.linalg.matrix_rank(f) == 4 and pfaffian == 1
    return {'F_matrix': f.tolist(), 'spacetime_rank': 4, 'Pfaffian': pfaffian,
            'wedge_square_volume_coefficient': 2 * pfaffian,
            'U6_generator': 'R1', 'SU6_generator': '2 R1 - R2',
            'constant_field_Maxwell_divergence': 0}


def landau_audit():
    # Exact one-plane scalar Landau heat factor: (s B)/sinh(s B).
    # Delta Gamma / B^2 = integral exp(-x) [1-y/sinh(y)]/y^2 du /(16 pi^2),
    # with x=exp(u)/100, y=B*x for the fixed mass squared 1.
    leading = e1_series(.01) / (96 * np.pi**2)
    rows = []
    for field in (.1, .05, .025):
        values = []
        for nodes in (64, 128):
            x, w = np.polynomial.legendre.leggauss(nodes)
            stop = math.log(4000.)
            xx = .01 * np.exp((x + 1) * stop / 2)
            y = field * xx
            response = np.empty_like(y)
            small = y < .01
            z = y[small]**2
            response[small] = 1/6 - 7*z/360 + 31*z*z/15120 - 127*z*z*z/604800
            response[~small] = (1 - y[~small]/np.sinh(y[~small])) / y[~small]**2
            values.append(float(np.dot(w, np.exp(-xx) * response) * stop / (32 * np.pi**2)))
        assert abs(values[0] - values[1]) < 1e-9
        relative = (leading - values[-1]) / leading
        assert 0 < relative < .001
        rows.append({'B': field, 'exact_Delta_Gamma_over_B_squared': values[-1],
                     'leading_F_squared_prediction': leading,
                     'relative_weak_field_remainder': relative,
                     'quadrature_difference': abs(values[0] - values[1])})
    ratios = [rows[i+1]['relative_weak_field_remainder']/rows[i]['relative_weak_field_remainder'] for i in (0, 1)]
    assert all(.24 < r < .26 for r in ratios)
    return {'mass_squared': 1., 'cutoff_squared': 100., 'rows': rows,
            'successive_remainder_ratios': ratios,
            'omitted_positive_tail_bound_after_B_squared_division': math.exp(-40)/(40*16*np.pi**2)}


def main():
    algebra = algebra_audit()
    flag = [flag_audit(t) for t in (0., .25, .5, 1.)]
    kinetic = [kinetic_audit(c) for c in (10., 100., 1000.)]
    maxwell = maxwell_audit()
    landau = landau_audit()
    here = Path(__file__).resolve()
    out = {'candidate': 'CE-GC2', 'matrix_tolerance': TOL, 'integral_tolerance': 1e-9,
           'inputs': {'block_dimensions': DIMS.tolist(), 'mass_squared': MASSES2.tolist(),
                      'complex_scalar_multiplicity': 1, 'fit_parameters': 0},
           'algebra': algebra, 'flag_connection': flag, 'kinetic': kinetic, 'Maxwell_witness': maxwell,
           'independent_Landau_heat_kernel': landau,
           'maximum_flag_identity_error': max(max(r['errors'].values()) for r in flag),
           'maximum_integral_error': max(r['absolute_integral_error'] for k in kinetic for r in k['proper_time_integrals']),
           'environment': {'python': platform.python_version(), 'numpy': np.__version__},
           'source_sha256': {name: hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                             for name in [here.name, 'ce_isometric_color_frame.py', 'ce_projector_one_loop.py']},
           'limits': ['Independent full connection, chosen group, finite active scalar and flat metric are inputs.',
                      'F squared is the local weak-field derivative term, not the complete determinant.',
                      'Finite C is its UV-removed normal response; no vector mass is inferred by mixing regulators.',
                      'Extra centers are a conditional failure of the U6/SU6 candidates, not a general no-go.',
                      'No observed couplings, hidden infinite sector, chiral dynamics or Einstein limit are obtained.']}
    here.with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('candidate', 'algebra', 'maximum_flag_identity_error',
                                         'maximum_integral_error', 'independent_Landau_heat_kernel')}, indent=2))


if __name__ == '__main__':
    main()
