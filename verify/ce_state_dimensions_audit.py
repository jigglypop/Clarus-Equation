"""Exact independent fixture paths for CE-DIM1 and CE-DIM-F1.

This verifies implementations against analytic examples, not physical truth.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import sympy as s

import ce_state_dimensions as dim
import ce_dimension_correspondence as corr

ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)


def equal(a, b):
    if isinstance(a, s.MatrixBase):
        return a.shape == b.shape and all(dim.is_zero(v) for v in a-b)
    return dim.is_zero(a-b)


def rejects(name, fn):
    try:
        fn()
    except ValueError:
        check(name, True)
    else:
        raise AssertionError(f"invalid input accepted: {name}")


def run():
    CHECKS.clear()
    paper13 = ROOT/'paper/01_측정과_접힘/13_확률중간_경계와_비관측_자유도의_차원.md'
    paper14 = ROOT/'paper/01_측정과_접힘/14_관계차원과_시공간_네힘_대응의_조건.md'
    prereg = []
    for path, marker, expected in [
        (paper13, '## 13.2', '20daa00114817ec7711dcc4df6d414876512d1806aab0e397284163e9d9c3b65'),
        (paper14, '## 14.2', '93a1aacfe46810cd0efb08ba33e493c2c38226e574f82109926caab0a8004ba9'),
    ]:
        prefix = path.read_text(encoding='utf-8').split(marker)[0].rstrip()+'\n'
        sha = hashlib.sha256(prefix.encode('utf-8')).hexdigest()
        check(f'preregistration {path.name}', sha == expected)
        prereg.append({'path': str(path.relative_to(ROOT)), 'sha256': sha})

    x, y, z = s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)
    for d in [1, 2, 3, 4]:
        for r in range(1, d+1):
            rho = s.diag(*([s.Rational(1, r)]*r+[0]*(d-r)))
            tangent = dim.rank_stratum_tangent(rho)
            check(f'd{d} r{r} independent tangent count', len(tangent) == 2*d*r-r*r-1)
            q = s.diag(*([0]*r+[1]*(d-r)))
            check(f'd{d} r{r} trace and kernel tangent constraints',
                  all(equal(s.trace(b), 0) and equal(q*b*q, s.zeros(d)) for b in tangent))
    rotate = s.Matrix([[3, -4, 0], [4, 3, 0], [0, 0, 5]])/5
    mixed_rotated = rotate*s.diag(1, 0, 0)*rotate.H
    check('rotated rank-one tangent, non-coordinate kernel', len(dim.rank_stratum_tangent(mixed_rotated)) == 4)
    result = dim.analyse_state(s.eye(2)/2, [z, 2*z+s.eye(2), s.eye(2)], [0, 1, 1])
    check('duplicate and affine observables cost one rank', result['certified_local_fibre_dimension'] == 2)
    critical = dim.analyse_state(s.diag(1, 0), [z], [1])
    check('extremal pure Z fibre is not falsely certified as two-dimensional',
          critical['linearized_nullity'] == 2 and critical['certified_local_fibre_dimension'] is None)
    isolated = dim.analyse_state(s.diag(1, 0), [x, y, z])
    check('pure tomography injective differential certifies isolated fibre', isolated['certified_local_fibre_dimension'] == 0)
    check('pure balanced circle regular dimension', dim.analyse_state((s.eye(2)+x)/2, [z])['certified_local_fibre_dimension'] == 1)
    rejects('inconsistent state witness', lambda: dim.analyse_state(s.eye(2)/2, [z], [1]))
    rejects('negative eigenvalue state', lambda: dim.density_matrix(s.diag(2, -1)))
    rejects('non-Hermitian state', lambda: dim.density_matrix([[1, 1], [0, 0]]))
    rejects('float matrix', lambda: dim.density_matrix([[0.5, 0], [0, 0.5]]))
    for bad in [0.5, True, 'nan', '__import__("os")', '1/0', {'re': 1}, {'re': 1, 'im': 0, 'x': 2}]:
        rejects(f'exact JSON rejects {bad!r}', lambda b=bad: dim.scalar_from_json(b))
    check('exact complex JSON', dim.scalar_from_json({'re': '1/3', 'im': '-2/7'}) == s.Rational(1, 3)-2*s.I/7)

    check('qubit midpoint disk and circle', dim.balanced_dimensions(z)['mixed_dimension'] == 2
          and dim.balanced_dimensions(z)['pure_regular_locus_dimension'] == 1)
    check('semidefinite kernel face', dim.balanced_dimensions(s.diag(1, 0, 0))['mixed_dimension'] == 3)
    check('definite empty is not zero-dimensional', dim.balanced_dimensions(s.eye(2))['mixed_dimension'] is None)
    check('zero operator whole space', dim.balanced_dimensions(s.zeros(3))['mixed_dimension'] == 8)
    check('singular pure boundary critical locus', dim.balanced_dimensions(s.diag(-1, 0, 1))['pure_critical_locus_dimension'] == 0)
    check('unpaired spectrum no unitary sign toggle', not dim.balanced_dimensions(s.diag(-1, 2))['exact_unitary_toggle_exists'])
    for u, mixed, pure in [(x, 1, 0), (s.eye(3), 8, 4), (s.diag(1, 1, -1, -1), 7, 2), (s.diag(1, s.I, -1), 2, 0)]:
        f = dim.fixed_state_dimensions(u)
        check(f'spectral vs commutator fixed dimensions {u.rows},{mixed}', f['mixed_fixed_dimension'] == mixed and f['pure_fixed_dimension'] == pure)
    rejects('nonunitary toggle', lambda: dim.fixed_state_dimensions(s.diag(1, 2)))

    fibres = []
    for d, p, r, w, expected, pure in [
        (2, 1, 1, '1/2', 2, 1), (4, 2, 2, '1/2', 11, None),
        (4, 2, 1, '1/2', 7, 3), (4, 2, 0, 0, 3, 2),
        (4, 2, 1, 1, 0, 0), (4, 2, 2, 1, 0, None), (2, 0, 0, 0, 3, 2), (1, 1, 1, 1, 0, 0),
    ]:
        f = dim.observation_fibre(d, p, r, w)
        check(f'raw fibre d{d}p{p}r{r}w{w}', f['raw_fibre_real_dimension'] == expected and f['pure_fibre_real_dimension'] == pure)
        fibres.append(f)
        if dim.scalar_from_json(w) < 1:
            # PSD first removes the kernel of the fixed R. On its remaining
            # support, use direct independent linear equations, not D4's formula.
            q = d-p
            basis = dim.traceless_basis(r+q)
            columns = [dim.hermitian_coordinates(b[:r, :r]) for b in basis] if r else []
            rank = s.Matrix.hstack(*columns).rank() if columns else 0
            check(f'independent reduced-support constraints d{d}p{p}r{r}', len(basis)-rank == expected)
    for args in [(2, 1, 0, '1/2'), (2, 2, 1, '1/2'), (2, 1, 2, '1/2'), (2, 1, 1, 0), (2, 1, 1, '3/2')]:
        rejects(f'invalid split {args}', lambda a=args: dim.observation_fibre(*a))
    # Pure visible rank 1: differential of a sphere phase chart, then psi psi^dagger.
    for q in [1, 2, 3]:
        psi = s.zeros(q+1, 1)
        psi[0] = psi[1] = s.sqrt(2)/2
        variations = [s.I*s.eye(q+1)[:, 1]]
        for j in range(2, q+1):
            variations.extend([s.eye(q+1)[:, j], s.I*s.eye(q+1)[:, j]])
        jac = s.Matrix.hstack(*(dim.hermitian_coordinates(v*psi.H+psi*v.H) for v in variations))
        check(f'pure hidden phase sphere differential q{q}', jac.rank() == 2*q-1)

    histories = []
    for h, expected in [(s.zeros(2), 1), (x/2, 2), ((x+z)/2, 3)]:
        span = dim.observable_history_basis(h, [z])
        check(f'Pauli history span {expected}', len(span) == expected)
        histories.append({'hamiltonian': str(h), 'visible_dimension': len(span), 'hidden_dimension': 3-len(span)})
    t = s.symbols('t', real=True)
    u = s.cos(t/2)*s.eye(2)-s.I*s.sin(t/2)*x
    check('independent analytic Pauli history formula', equal(u.H*z*u, z*s.cos(t)+y*s.sin(t)))
    check('third history repeated commutator gives independent X-Z', equal(s.I*((x+z)/2*y-y*(x+z)/2), x-z))

    flag = corr.observation_flag(s.eye(2)/2, [z, z, x, y])
    check('nested qubit visible and hidden ranks', [v['visible_rank'] for v in flag] == [0, 1, 1, 2, 3]
          and [v['local_hidden_dimension'] for v in flag] == [3, 2, 2, 1, 0])
    check('fully identified mixed state retains binary outcome uncertainty',
          flag[-1]['local_hidden_dimension'] == 0 and equal(s.trace(s.eye(2)/2*(s.eye(2)+z)/2), s.Rational(1, 2)))
    flag3 = corr.observation_flag(s.eye(3)/3, dim.traceless_basis(3)[:4])
    check('four records need not specify entire state', [v['local_hidden_dimension'] for v in flag3] == [8, 7, 6, 5, 4])
    check('qubit commutant restriction', [len(corr.commutant_basis(2, a, False)) for a in [[], [z], [z, x]]] == [4, 2, 1])
    check('qubit traceless commutant restriction', [len(corr.commutant_basis(2, a)) for a in [[], [z], [z, x]]] == [3, 1, 0])
    block = s.diag(0, 1, 1, 2, 2, 2)
    check('1+2+3 block centers persist in U6', len(corr.commutant_basis(6, [block], False)) == 14)
    check('1+2+3 block centers persist in SU6', len(corr.commutant_basis(6, [block])) == 13)
    # Global product-state calculation reduces to these exact one-factor identities.
    spinor = s.Matrix([s.cos(t/2), s.sin(t/2)])
    derivative = spinor.diff(t)
    check('real spinor norm identity', equal((spinor.H*spinor)[0], 1))
    check('real spinor horizontal derivative identity', equal((spinor.H*derivative)[0], 0))
    check('real spinor derivative norm identity', equal((derivative.H*derivative)[0], s.Rational(1, 4)))
    product_frame = s.eye(16)[:, :1]
    product_derivatives = [s.eye(16)[:, 2**j:2**j+1]/2 for j in range(4)]
    for k in range(1, 5):
        g = corr.frame_geometry(product_frame, product_derivatives[:k])
        check(f'product stage {k} full metric rank and zero Berry curvature',
              equal(g['metric'], s.eye(k)/4) and all(equal(a, s.zeros(1)) for a in g['curvature'].values()))
    frame = s.eye(5)[:, :1]
    ds = [s.eye(5)[:, j:j+1] for j in range(1, 5)]
    real = corr.frame_geometry(frame, ds)
    complex_g = corr.frame_geometry(frame, [ds[0], s.I*ds[0], ds[1], ds[2]])
    check('same rank-four metric has different curvature', equal(real['metric'], complex_g['metric'])
          and real['metric_rank'] == 4 and equal(complex_g['curvature'][0, 1], s.Matrix([[-2]]))
          and all(equal(v, s.zeros(1)) for v in real['curvature'].values()))
    # Separate projector differential formula for every entry of this metric.
    pdots = [b*frame.H+frame*b.H for b in [ds[0], s.I*ds[0], ds[1], ds[2]]]
    projector_metric = s.Matrix(4, 4, lambda i, j: s.trace(pdots[i]*pdots[j])/2)
    check('Grassmann projector metric independent path', equal(projector_metric, complex_g['metric']))
    for m in [2, 3]:
        g = corr.frame_geometry(s.kronecker_product(frame, s.eye(m)),
                                [s.kronecker_product(a, s.eye(m)) for a in ds[:2]])
        check(f'internal replication m{m} leaves spatial rank two', g['metric_rank'] == 2 and equal(g['metric'], m*s.eye(2)))
    gauge_vertical = corr.frame_geometry(s.Matrix([[1], [0]]), [s.Matrix([[s.I], [0]])])
    check('pure phase derivative contributes no space direction', gauge_vertical['metric_rank'] == 0)
    rejects('invalid Stiefel frame', lambda: corr.frame_geometry([[2], [0]], []))
    rejects('non-tangent frame derivative', lambda: corr.frame_geometry([[1], [0]], [[[1], [0]]]))
    # CP and normalization of the fixed PVM example, plus loss of prior certainty.
    p0, p1 = (s.eye(2)+z)/2, (s.eye(2)-z)/2
    vectors = [s.Matrix(list(p)) for p in [p0, p1]]
    choi = sum((v*v.H for v in vectors), s.zeros(4))
    check('PVM Choi PSD and completeness', choi.is_positive_semidefinite is True and equal(p0.H*p0+p1.H*p1, s.eye(2)))
    plus = (s.eye(2)+x)/2
    posterior = plus*p0*plus/s.trace(plus*p0*plus)
    check('later X record loses prior Z certainty', equal(s.trace(z*posterior), 0))

    with tempfile.TemporaryDirectory(prefix='ce-dim-audit-') as tmp:
        path = Path(tmp)/'input.json'
        path.write_text(json.dumps({'rho': [['1/2', '1/2'], ['1/2', '1/2']], 'observables': [[[1, 0], [0, -1]]],
                                    'values': [0], 'hamiltonian': [[0, '1/2'], ['1/2', 0]],
                                    'unitary': [[0, 1], [1, 0]]}), encoding='utf-8')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT/'verify/ce_state_dimensions.py'), '--input', str(path)],
                                capture_output=True, text=True, encoding='utf-8', check=True)
        output = json.loads(result.stdout)
        check('CLI exact input and conservative science metadata', output['state_measurement']['certified_local_fibre_dimension'] == 1
              and output['history']['observable_span_dimension'] == 2 and output['scientific_success'] is False and output['full_joint_rmse'] is None)
        bad = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT/'verify/ce_state_dimensions.py'), '--levels', '0'],
                             capture_output=True, text=True, encoding='utf-8')
        check('CLI invalid dimension rejects with usage error', bad.returncode == 2 and 'Traceback' not in bad.stderr)

    sources = [Path(__file__).resolve(), ROOT/'verify/ce_state_dimensions.py', ROOT/'verify/ce_dimension_correspondence.py', paper13, paper14]
    receipt = {
        'schema_version': 1, 'checks_passed': len(CHECKS), 'checks': CHECKS,
        'arithmetic': 'exact SymPy, rational and symbolic identities; no rank tolerance',
        'floating_point_error_budget': None, 'fitted_parameters': 0,
        'preregistrations': prereg,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'environment': {'python': platform.python_version(), 'sympy': s.__version__},
        'observation_fibres': fibres, 'history_examples': histories, 'qubit_flag': flag,
        'counterexample_status': {'dimension_alone_selects_four_forces': 'rejected_as_necessary_implication',
                                  'relation_geometry_candidate': 'conditional_local_construction_only'},
        'independence': 'alternate analytic or matrix paths by same author, not external peer review',
        'scientific_success': False, 'full_joint_rmse': None,
        'unproved': ['physical dimension selection', 'Lorentz signature emergence', 'four-force unification',
                     'actual measurement dynamics from common action', 'joint observational validation'],
    }
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(CHECKS), 'receipt': str(out), 'scientific_success': False}, ensure_ascii=False))


if __name__ == '__main__':
    run()
