"""CE-GR5: Urbantke signature and the self-metric Yang-Mills obstruction.

Exact determinant audits, Hodge identities, an actual connection family and a
separate pure-connection action comparison. None establishes quantum gravity.
"""

from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np


PAIRS = ((0, 1), (0, 2), (0, 3), (2, 3), (3, 1), (1, 2))
TOL = 1e-10


def epsilon(n):
    result = np.zeros((n,) * n, dtype=int)
    for p in itertools.permutations(range(n)):
        inversions = sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
        result[p] = (-1)**inversions
    return result


EPS3, EPS4 = epsilon(3), epsilon(4)


def forms(six):
    result = np.zeros((3, 4, 4), dtype=np.asarray(six).dtype)
    for i in range(3):
        for value, (a, b) in zip(six[i], PAIRS):
            result[i, a, b], result[i, b, a] = value, -value
    return result


def density_numerator(b):
    return np.einsum('ijk,abcd,ima,jbc,kdn->mn', EPS3, EPS4, b, b, b, optimize=True)


def wedge_numerator(b, c):
    return np.einsum('abcd,iab,jcd->ij', EPS4, b, c, optimize=True)


def determinant_exact(matrix):
    a = [[Fraction(x) for x in row] for row in matrix]
    determinant = Fraction(1)
    for i in range(len(a)):
        pivot = next((r for r in range(i, len(a)) if a[r][i]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            determinant = -determinant
        divisor = a[i][i]
        determinant *= divisor
        for r in range(i + 1, len(a)):
            ratio = a[r][i] / divisor
            for j in range(i + 1, len(a)):
                a[r][j] -= ratio * a[i][j]
    return determinant


def hodge(b, metric):
    inverse = np.linalg.inv(metric)
    raised = np.einsum('ra,sb,iab->irs', inverse, inverse, b)
    return np.sqrt(abs(np.linalg.det(metric))) * np.einsum('mnrs,irs->imn', EPS4, raised) / 2


def scalar_wedge_sum(b, c):
    return np.trace(wedge_numerator(b, c)) / 4


def real_audit(six, label):
    b = forms(np.array(six, dtype=int))
    gn, mn = density_numerator(b), wedge_numerator(b, b)
    g_exact = [[Fraction(int(value), 12) for value in row] for row in gn]
    m_exact = [[Fraction(int(value), 8) for value in row] for row in mn]
    detg, detm = determinant_exact(g_exact), determinant_exact(m_exact)
    assert detg == detm**2
    g, m = gn / 12., mn / 8.
    assert np.max(abs(g - g.T)) < TOL
    eigenvalues = np.linalg.eigvalsh(g)
    nondegenerate = bool(detm)
    star_error = None
    if nondegenerate:
        relative_scale = max(1., float(np.max(abs(b))))
        star_error = float(np.max(abs(hodge(b, g) - np.sign(float(detm)) * b)) / relative_scale)
        assert star_error < TOL, (label, star_error)
        negative = int(sum(eigenvalues < 0))
        assert negative in (0, 2, 4)
    else:
        negative = None
    return {'label': label, 'six_components': np.asarray(six).tolist(),
            'det_metric_exact': str(detg), 'det_half_wedge_exact': str(detm),
            'metric_eigenvalues': eigenvalues.tolist(), 'nondegenerate': nondegenerate,
            'negative_eigenvalues_if_nondegenerate': negative, 'relative_Hodge_error': star_error}


def lorentz_audit():
    sigma = forms(np.concatenate([1j*np.eye(3), -np.eye(3)], axis=1))
    eta = np.diag([-1., 1., 1., 1.])
    density = density_numerator(sigma) / 12
    gram = wedge_numerator(sigma, sigma) / 8
    mixed = wedge_numerator(sigma, sigma.conj()) / 4
    errors = {'density_i_eta': float(np.max(abs(density - 1j * eta))),
              'half_wedge_minus_i_identity': float(np.max(abs(gram + 1j*np.eye(3)))),
              'Lorentzian_self_dual': float(np.max(abs(hodge(sigma, eta) - 1j*sigma))),
              'mixed_wedge_reality': float(np.max(abs(mixed)))}
    assert max(errors.values()) < TOL, errors
    return {'errors': errors, 'metric_after_dividing_density_by_i': eta.tolist(),
            'negative_metric_eigenvalues': 1}


def connection_audit(parameter):
    integrals = []
    maximum_error = 0.
    for nodes in (16, 32):
        x, weights = np.polynomial.legendre.leggauss(nodes)
        t, weights = (x+1)/2, weights/2
        composite, fixed = [], []
        for value in t:
            f = 1 + value + parameter*value*(1-value)
            df = 1 + parameter*(1-2*value)
            b = forms(np.concatenate([df*np.eye(3), f*f*np.eye(3)], axis=1))
            g = density_numerator(b) / 12
            expected_g = np.diag([df**3, df*f**4, df*f**4, df*f**4])
            maximum_error = max(maximum_error, float(np.max(abs(g-expected_g))))
            composed_value = float(scalar_wedge_sum(b, hodge(b, g)))
            topological_value = float(scalar_wedge_sum(b, b))
            fixed_value = float(scalar_wedge_sum(b, hodge(b, np.eye(4))))
            maximum_error = max(maximum_error, abs(composed_value-topological_value),
                                abs(composed_value-6*df*f*f), abs(fixed_value-3*(df*df+f**4)))
            composite.append(composed_value)
            fixed.append(fixed_value)
        integrals.append({'nodes': nodes, 'composite_metric_action': float(np.dot(weights, composite)),
                          'fixed_metric_action': float(np.dot(weights, fixed))})
    assert max(abs(r['composite_metric_action']-14.) for r in integrals) < TOL
    assert maximum_error < TOL
    return {'epsilon': parameter, 'boundary_Pontryagin_value': 14., 'integrals': integrals,
            'maximum_pointwise_error': maximum_error}


def action_hessian_audit():
    basis = [np.diag([1., -1., 0.]), np.diag([1., 1., -2.])]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        b = np.zeros((3, 3));b[i, j] = b[j, i] = 1.
        basis.append(b)
    expected = np.array([[-np.trace(a@b)/2 for b in basis] for a in basis])
    def function(x):
        return float(np.sum(np.sqrt(np.linalg.eigvalsh(x)))**2 / 3)
    rows = []
    for step in (.01, .005, .0025):
        hessian = np.zeros((5, 5))
        for i, a in enumerate(basis):
            for j, b in enumerate(basis):
                if i == j:
                    hessian[i, i] = (function(np.eye(3)+step*a)-6+function(np.eye(3)-step*a))/step**2
                else:
                    hessian[i, j] = (function(np.eye(3)+step*(a+b))-function(np.eye(3)+step*(a-b))
                                     -function(np.eye(3)+step*(-a+b))+function(np.eye(3)-step*(a+b)))/(4*step**2)
        err = float(np.max(abs(hessian-expected)))
        trace_value = (function((1+step)*np.eye(3))-6+function((1-step)*np.eye(3)))/step**2
        assert abs(trace_value) < 1e-8
        rows.append({'step': step, 'trace_free_hessian': hessian.tolist(),
                     'maximum_error': err, 'trace_direction_second_difference': trace_value})
    ratios = [rows[i+1]['maximum_error']/rows[i]['maximum_error'] for i in (0, 1)]
    assert all(.24 < r < .26 for r in ratios)
    return {'topological_trace_Hessian_rank_exact': 0, 'normalized_GR_comparison_Hessian': expected.tolist(),
            'comparison_trace_free_Hessian_rank': int(np.linalg.matrix_rank(expected)),
            'finite_differences': rows, 'successive_error_ratios': ratios}


def main():
    real = []
    for label, magnetic in [('Euclidean', [1,1,1]), ('split', [1,1,-1])]:
        real.append(real_audit(np.concatenate([np.eye(3, dtype=int), np.diag(magnetic)], axis=1), label))
    real.append(real_audit(np.concatenate([np.zeros((3,3),int), np.eye(3,dtype=int)], axis=1), 'spatial_only'))
    rng = np.random.default_rng(20260910)
    for i in range(24):
        real.append(real_audit(rng.integers(-2, 3, (3, 6)), f'integer_{i}'))
    lorentz = lorentz_audit()
    connection = [connection_audit(p) for p in (-.5, 0., .5)]
    hessian = action_hessian_audit()
    here = Path(__file__).resolve()
    out = {'candidate': 'CE-GR5', 'tolerance': TOL, 'seed': 20260910,
           'real_triples': real, 'complex_Lorentzian_witness': lorentz,
           'actual_connection_families': connection, 'pure_connection_action_comparison': hessian,
           'maximum_Hodge_relative_error': max(r['relative_Hodge_error'] or 0 for r in real),
           'environment': {'python': platform.python_version(), 'numpy': np.__version__},
           'source_sha256': {here.name: hashlib.sha256(here.read_bytes()).hexdigest()},
           'limits': ['Direct curvature-to-metric identification is tested, not all gravity constructions.',
                      'The self-metric F squared term is topological; higher terms and other fields are not evaluated.',
                      'The GR square-root action is a literature comparison, not derived from the scalar loop.',
                      'Metric signature witnesses and matrix Hessians do not prove Einstein dynamics or quantum unitarity.']}
    here.with_suffix('.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': out['candidate'], 'exact_determinant_cases': len(real),
                      'max_Hodge_error': out['maximum_Hodge_relative_error'],
                      'Lorentzian': lorentz, 'connection': connection,
                      'Hessian_error_ratios': hessian['successive_error_ratios']}, indent=2))


if __name__ == '__main__':
    main()
