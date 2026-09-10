"""CE-GR6: embedding variation counterexamples and a free flat witness.

Einstein-Hilbert is supplied. This is a classical variational audit, not a
derivation of gravity from quantum states or a quantum equivalence proof.
"""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


ETA = np.diag([-1., 1., 1., 1.])
ETA5 = np.diag([-1., 1., 1., 1., 1.])
ETA24 = np.diag([-1.] + [1.]*23)
TOL = 1e-10
PAIRS = [(i, j) for i in range(4) for j in range(i+1, 4)]
COMPONENTS = [(i, i) for i in range(4)] + PAIRS
VECTORS = np.array([np.eye(4)[i] for i in range(4)] +
                   [np.eye(4)[i]+np.eye(4)[j] for i, j in PAIRS])
DYADS = np.einsum('ai,aj->aij', VECTORS, VECTORS)


def error(value):
    return float(np.max(abs(value)))


def exact_rank(matrix):
    """Rational row reduction, independent of floating point singular values."""
    rows = [[Fraction(int(x)) for x in row] for row in matrix]
    pivot = 0
    for col in range(len(rows[0])):
        found = next((i for i in range(pivot, len(rows)) if rows[i][col]), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        scale = rows[pivot][col]
        rows[pivot] = [x/scale for x in rows[pivot]]
        for i in range(len(rows)):
            if i == pivot:
                continue
            scale = rows[i][col]
            rows[i] = [x-scale*y for x, y in zip(rows[i], rows[pivot])]
        pivot += 1
        if pivot == len(rows):
            break
    return pivot


def gauss_curvature(second_forms):
    # All normals in these one-time ambient examples are spacelike.
    riemann = (np.einsum('aik,ajl->ijkl', second_forms, second_forms)
               - np.einsum('ail,ajk->ijkl', second_forms, second_forms))
    ricci = np.einsum('ik,ijkl->jl', ETA, riemann)
    scalar = float(np.einsum('ij,ij', ETA, ricci))
    return riemann, ricci, scalar, ricci-.5*scalar*ETA


def frw_geometry(a, adot, addot, cosmological):
    f = np.sqrt(1+adot**2)
    tangent = np.zeros((5, 4))
    tangent[:, 0] = [f, adot, 0., 0., 0.]
    tangent[2:, 1:] = np.eye(3)
    normal = np.array([adot, f, 0., 0., 0.])
    # Raw second derivatives in locally orthonormal spatial coordinates.
    raw = np.zeros((5, 4, 4))
    raw[:, 0, 0] = [adot*addot/f, addot, 0., 0., 0.]
    for j in range(1, 4):
        raw[1, j, j] = -1/a
        raw[j+1, 0, j] = raw[j+1, j, 0] = adot/a
    projected_b = np.einsum('a,ab,bij->ij', normal, ETA5, raw)
    b = np.diag([addot/f] + [-f/a]*3)
    riemann, ricci, scalar, einstein = gauss_curvature(b[None, :, :])
    rho = 3*f**2/a**2
    pressure = -(2*addot/a+f**2/a**2)
    direct_einstein = np.diag([rho]+[pressure]*3)
    direct_scalar = 6*(addot/a+f**2/a**2)
    residual = einstein+cosmological*ETA
    raised = ETA@residual@ETA
    rt = float(np.einsum('ij,ij', raised, b))
    rt_formula = 3*f/a**3*(3*a*addot+f**2) - cosmological*(addot/f+3*f/a)
    errors = {
        'induced_metric': error(tangent.T@ETA5@tangent-ETA),
        'normal_norm': float(abs(normal@ETA5@normal-1)),
        'normal_tangent': error(normal@ETA5@tangent),
        'second_form_from_embedding': error(projected_b-b),
        'Einstein_Gauss_vs_FRW': error(einstein-direct_einstein),
        'scalar_Gauss_vs_FRW': abs(scalar-direct_scalar),
        'RT_contraction_vs_formula': abs(rt-rt_formula),
        'RT_equation': abs(rt),
    }
    assert max(errors.values()) < TOL
    assert error(residual) > 1
    return {'a': float(a), 'adot': float(adot), 'addot': float(addot),
            'cosmological_constant': cosmological, 'scalar_curvature': scalar,
            'second_form_diagonal': np.diag(b).tolist(),
            'Einstein_residual_diagonal': np.diag(residual).tolist(),
            'RT_residual': rt, 'Einstein_residual_max': error(residual),
            'normal_rank': int(np.linalg.matrix_rank(b.reshape(1, 16))),
            'errors': errors}


def cosmological_counterexample(chi):
    s, c = np.sin(chi), np.cos(chi)
    a, adot, addot = s**3, c/s, -1/(3*s**5)
    out = frw_geometry(a, adot, addot, 0.)
    lapse, lapse_prime = 3*s**3, 9*s**2*c
    first = np.array([3*s**2, 3*s**2*c])
    second = np.array([6*s*c, 6*s*c**2-3*s**3])
    eta2 = np.diag([-1., 1.])
    proper_second = (second-lapse_prime/lapse*first)/lapse**2
    normal2 = np.array([adot, 1/s])
    rho, pressure = 3/s**8, -1/(3*s**8)
    rho_dot = -8*c/s**12
    hubble = adot/a
    out['errors'].update({
        'proper_lapse': float(abs(first@eta2@first+lapse**2)),
        'proper_velocity': error(first/lapse-np.array([1/s, adot])),
        'proper_second_form': float(abs(normal2@eta2@proper_second-addot*s)),
        'RT_reduced_ODE': float(abs(3*a*addot+adot**2+1)),
        'first_integral': float(abs((1+adot**2)*a**(2/3)-1)),
        'scalar_closed_form': abs(out['scalar_curvature']-4/s**8),
        'residual_closed_form': error(np.array(out['Einstein_residual_diagonal'])
                                      -np.array([rho]+[pressure]*3)),
        'residual_conservation': float(abs(rho_dot+3*hubble*(rho+pressure))),
    })
    assert max(out['errors'].values()) < TOL
    out.update({'chi': float(chi), 'proper_time': float(2-3*c+c**3),
                'ambient_time': float(1.5*(chi-s*c)), 'lapse': float(lapse),
                'effective_residual_w': float(pressure/rho)})
    return out


def free_embedding(radius, x):
    h = ETA-radius**2*np.sum(DYADS, axis=0)
    values, vectors = np.linalg.eigh(h)
    assert sum(values < 0) == 1 and sum(values > 0) == 3
    linear = np.diag(np.sqrt(abs(values)))@vectors.T
    y = np.zeros(24)
    y[:4] = linear@x
    tangent = np.zeros((24, 4))
    tangent[:4] = linear
    second = np.zeros((24, 4, 4))
    normals = np.zeros((24, 10))
    normal_derivatives = np.zeros((24, 10, 4))
    for a, v in enumerate(VECTORS):
        phase = v@x
        radial = np.array([np.cos(phase), np.sin(phase)])
        angular = np.array([-np.sin(phase), np.cos(phase)])
        rows = slice(4+2*a, 6+2*a)
        y[rows] = radius*radial
        tangent[rows] = radius*np.outer(angular, v)
        normals[rows, a] = radial
        normal_derivatives[rows, a] = np.outer(angular, v)
        second[rows] = -radius*radial[:, None, None]*DYADS[a]
    induced = tangent.T@ETA24@tangent
    metric_derivative = (np.einsum('Akm,AB,Bn->kmn', second, ETA24, tangent)
                         + np.einsum('Am,AB,Bkn->kmn', tangent, ETA24, second))
    normal_forms = np.einsum('Aa,AB,Bij->aij', normals, ETA24, second)
    curvature = gauss_curvature(normal_forms)[0]
    integer_matrix = np.array([[form[i, j] for form in DYADS]
                               for i, j in COMPONENTS], dtype=int)
    rank_fraction = exact_rank(integer_matrix)
    rank_numpy = int(np.linalg.matrix_rank(normal_forms.reshape(10, 16)))
    assert rank_fraction == rank_numpy == 10
    errors = {
        'linear_signature_factor': error(linear.T@ETA@linear-h),
        'induced_Minkowski_metric': error(induced-ETA),
        'constant_metric_derivative': error(metric_derivative),
        'normal_orthonormality': error(normals.T@ETA24@normals-np.eye(10)),
        'normal_tangent_orthogonality': error(normals.T@ETA24@tangent),
        'second_derivatives_are_normal': error(np.einsum('Ai,AB,Bjk->ijk', tangent, ETA24, second)),
        'second_form_formula': error(normal_forms+radius*DYADS),
        'Gauss_flat_curvature': error(curvature),
    }
    reconstruction = []
    for i, j in COMPONENTS:
        target = np.zeros((4, 4))
        target[i, j] = target[j, i] = 1
        phi = np.array([(target[k, k]-sum(target[k, l] for l in range(4) if l != k))
                        /(2*radius) for k in range(4)]
                       + [target[k, l]/(2*radius) for k, l in PAIRS])
        algebraic = -2*np.einsum('a,aij->ij', phi, normal_forms)
        # Different route: differentiate Y+epsilon*sum(phi_a*N_a).
        delta_tangent = np.einsum('a,Aai->Ai', phi, normal_derivatives)
        geometric = delta_tangent.T@ETA24@tangent+tangent.T@ETA24@delta_tangent
        reconstruction.append({'component': [i, j], 'normal_coefficients': phi.tolist(),
                               'algebraic_error': error(algebraic-target),
                               'differentiated_embedding_error': error(geometric-target)})
    errors['all_metric_components_algebraic'] = max(row['algebraic_error'] for row in reconstruction)
    errors['all_metric_components_geometric'] = max(row['differentiated_embedding_error'] for row in reconstruction)
    # RT contractions detect every symmetric E^{ij}; off-diagonal entries count twice.
    contraction_matrix = integer_matrix.T@np.diag([1]*4+[2]*6)
    assert exact_rank(contraction_matrix) == int(np.linalg.matrix_rank(contraction_matrix)) == 10
    assert max(errors.values()) < TOL
    return {'x': x.tolist(), 'ambient_dimension': 24, 'normal_dimension': 20,
            'radius': radius, 'H_eigenvalues': values.tolist(),
            'second_form_rank_fraction': rank_fraction, 'second_form_rank_numpy': rank_numpy,
            'normal_metric_kernel_dimension': 10,
            'Einstein_residual_detection_rank': 10,
            'dyad_component_matrix': integer_matrix.tolist(),
            'metric_reconstruction': reconstruction, 'errors': errors}


def main():
    static = frw_geometry(1., 0., 0., 1.)
    assert error(np.array(static['Einstein_residual_diagonal'])-np.array([2., 0., 0., 0.])) < TOL
    evolving = [cosmological_counterexample(chi) for chi in [np.pi/4, np.pi/3, np.pi/2]]
    free = [free_embedding(.1, x) for x in [np.zeros(4), np.array([.1, -.2, .3, -.1])]]
    maximum = max(max(row['errors'].values()) for row in [static]+evolving+free)
    here = Path(__file__).resolve()
    out = {
        'candidate': 'CE-GR6', 'status': 'conditional_classical_variation_gate',
        'inputs': {'vacuum': True, 'Planck_coefficient': 1,
                   'action': 'supplied Einstein-Hilbert; all independent embedding variations',
                   'ambient_signature': 'one negative time direction', 'tolerance': TOL,
                   'fit_parameters': 0},
        'static_non_Einstein_RT_solution': static,
        'zero_Lambda_non_Einstein_RT_solutions': evolving,
        'free_flat_embedding': free,
        'maximum_algebraic_error': maximum,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'source_sha256': {here.name: hashlib.sha256(here.read_bytes()).hexdigest()},
        'limits': [
            'Embedding coordinates and the Einstein-Hilbert action are inputs, not derived quantum data.',
            'RT equations are weaker than Einstein equations on rank-deficient embeddings.',
            'Rank ten is a pointwise sufficient condition, not a necessary dimension theorem for all formulations.',
            'The 24-dimensional witness is flat with full normal rank; no global embedding theorem is proved.',
            'Classical equivalence requires rank retention and dependence on Y only through the metric.',
            'Metric-null embedding variations are not assigned a quantum measure or automatically declared gauge.',
            'No graviton quantization, common gauge action, hidden-state coupling or joint RMSE is established.',
        ],
    }
    here.with_suffix('.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': out['candidate'], 'maximum_algebraic_error': maximum,
                      'static_Einstein_residual': static['Einstein_residual_diagonal'],
                      'zero_Lambda': [{'chi': row['chi'], 'R': row['scalar_curvature'],
                                       'RT': row['RT_residual'], 'Einstein_max': row['Einstein_residual_max']}
                                      for row in evolving],
                      'free_normal_ranks': [row['second_form_rank_fraction'] for row in free],
                      'metric_components_reconstructed_per_point': 10}, indent=2))


if __name__ == '__main__':
    main()
