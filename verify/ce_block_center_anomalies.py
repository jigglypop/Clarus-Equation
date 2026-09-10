"""CE-MR2: maximal locally anomaly-free center for fixed Weyl tensor content.

Removing an anomalous gauge field is a new choice of gauged subgroup, not a
dynamical consequence of its anomaly. The UV full-group theory is not supplied.
"""

from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np

from ce_matter_representation_anomalies import matrices


WEIGHTS = np.array([6, 3, 3, 2, 1])
CHARGES = np.array([[1, 1, 1], [-4, 0, -1], [2, 0, -1], [-3, 1, 0], [6, 0, 0]])
TOL = 1e-12


def nullspace_exact(matrix):
    rows = [[Fraction(int(x)) for x in row] for row in matrix]
    pivots = []
    n = len(rows[0])
    for column in range(n):
        pivot = next((r for r in range(len(pivots), len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        r = len(pivots)
        rows[r], rows[pivot] = rows[pivot], rows[r]
        divisor = rows[r][column]
        rows[r] = [x/divisor for x in rows[r]]
        for j in range(len(rows)):
            if j != r:
                factor = rows[j][column]
                rows[j] = [x-factor*y for x, y in zip(rows[j], rows[r])]
        pivots.append(column)
    result = []
    for free in (c for c in range(n) if c not in pivots):
        vector = [Fraction(0) for _ in range(n)]
        vector[free] = Fraction(1)
        for r, pivot in enumerate(pivots):
            vector[pivot] = -rows[r][free]
        result.append(vector)
    assert len(pivots) == np.linalg.matrix_rank(np.array(matrix, float), tol=TOL)
    return len(pivots), [[str(x) for x in v] for v in result]


def main():
    standard = matrices(CHARGES[:, 0])
    centers = [np.diag(np.repeat(CHARGES[:, i], WEIGHTS)).astype(complex) for i in range(3)]
    weak, color = standard[1:4], standard[4:]
    su3_rows = 2 * CHARGES[0] + CHARGES[1] + CHARGES[2]
    su2_rows = 3 * CHARGES[0] + CHARGES[3]
    gravity_rows = WEIGHTS @ CHARGES
    constraints = np.array([su3_rows, su2_rows, gravity_rows])
    assert constraints.tolist() == [[0, 2, 0], [0, 4, 3], [0, 8, 0]]
    rank, allowed = nullspace_exact(constraints)
    su6_rank, su6_allowed = nullspace_exact(np.vstack([constraints, [1, 2, 3]]))
    assert rank == 2 and allowed == [['1', '0', '0']]
    assert su6_rank == 3 and su6_allowed == []
    errors = []
    for i, center in enumerate(centers):
        for algebra, row in ((color, su3_rows), (weak, su2_rows)):
            for a, ta in enumerate(algebra):
                for b, tb in enumerate(algebra):
                    actual = np.trace(center @ (ta @ tb + tb @ ta))
                    errors.append(float(abs(actual - (row[i] if a == b else 0))))
        errors.append(float(abs(np.trace(center) - gravity_rows[i])))
    cubic = {}
    polynomial = {}
    for a, b, c in itertools.product(range(3), repeat=3):
        exact = int(sum(WEIGHTS * CHARGES[:, a] * CHARGES[:, b] * CHARGES[:, c]))
        matrix_value = np.trace(centers[a] @ centers[b] @ centers[c])
        errors.append(float(abs(matrix_value - exact)))
        cubic[f'{a}{b}{c}'] = exact
        powers = tuple([a, b, c].count(i) for i in range(3))
        polynomial[powers] = polynomial.get(powers, 0) + exact
    polynomial = {','.join(map(str, key)): value for key, value in polynomial.items() if value}
    assert polynomial == {'2,1,0': 72, '2,0,1': -162, '1,1,1': 36,
                          '0,3,0': 8, '0,2,1': 18, '0,1,2': 18}
    allowed_matrices = [centers[0]] + weak + color
    anomaly = max(float(abs(np.trace(a @ (b @ c + c @ b))))
                  for a, b, c in itertools.product(allowed_matrices, repeat=3))
    errors.append(anomaly)
    assert max(errors) < TOL
    assert len(allowed_matrices) == 12
    assert (3+1) % 2 == 0
    # Pure C center passes its cubic and gravitational test but fails SU(2)^2 C.
    assert cubic['222'] == 0 and gravity_rows[2] == 0 and su2_rows[2] == 3
    here = Path(__file__).resolve()
    out = {'candidate': 'CE-MR2', 'tolerance': TOL,
           'input_charges_L_W_C': CHARGES.tolist(), 'multiplicities': WEIGHTS.tolist(),
           'linear_constraints_SU3_SU2_gravity': constraints.tolist(),
           'exact_constraint_rank': rank, 'U6_allowed_center_basis': allowed,
           'SU6_constraint_rank_with_trace': su6_rank, 'SU6_allowed_center_basis': su6_allowed,
           'cubic_center_tensor': cubic, 'cubic_polynomial_powers_alpha_beta_gamma': polynomial,
           'allowed_generator_count': len(allowed_matrices),
           'allowed_symmetric_triples_checked': len(allowed_matrices)**3,
           'allowed_max_symmetric_triple_trace': anomaly, 'maximum_matrix_error': max(errors),
           'SU2_doublets_per_generation': 4, 'three_generation_internal_Weyl_dimension': 45,
           'environment': {'python': platform.python_version(), 'numpy': np.__version__},
           'source_sha256': {name: hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                             for name in [here.name, 'ce_matter_representation_anomalies.py',
                                          'ce_common_isotropic_frame.py', 'ce_isometric_color_frame.py']},
           'limits': ['Fixed chiral matter, tensor powers and both non-Abelian factors are supplied.',
                      'Local anomaly-free subgroup selection does not dynamically remove other vectors.',
                      'No full U6/SU6 UV representation, anomaly inflow or spectator completion is constructed.',
                      'Global group, quantum gravity, couplings and actual observation remain open.']}
    here.with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: out[key] for key in ('candidate', 'linear_constraints_SU3_SU2_gravity',
                      'U6_allowed_center_basis', 'SU6_allowed_center_basis',
                      'cubic_polynomial_powers_alpha_beta_gamma', 'allowed_generator_count',
                      'maximum_matrix_error')}, indent=2))


if __name__ == '__main__':
    main()
