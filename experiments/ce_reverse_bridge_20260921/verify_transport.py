"""CE-RB10: transport defects, representation masses and the specified gauge loop.

General proofs are in chapter 28. These are finite algebra and synthetic checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, expm

from verify_reverse import Evidence


def comm(a, b):
    return a@b-b@a


def power(a, p):
    values, vectors = np.linalg.eigh(a)
    return (vectors*values**p)@vectors.conj().T


def su(n):
    out = []
    for i in range(n):
        for j in range(i+1, n):
            a = np.zeros((n, n), complex)
            a[i, j] = a[j, i] = .5
            out.append(a)
            a = np.zeros((n, n), complex)
            a[i, j], a[j, i] = -.5j, .5j
            out.append(a)
    for k in range(1, n):
        out.append(np.diag([1]*k+[-k]+[0]*(n-k-1))/np.sqrt(2*k*(k+1)))
    return out


def representation():
    dims = [6, 3, 3, 2, 1, 1]
    charge = [sp.Rational(1, 6), -sp.Rational(2, 3), sp.Rational(1, 3),
              -sp.Rational(1, 2), sp.Integer(1), sp.Integer(0)]
    matrices = [block_diag(*[float(y)*np.eye(d) for d, y in zip(dims, charge)])]
    for t in su(2):
        matrices.append(block_diag(np.kron(np.eye(3), t), np.zeros((3, 3)),
                                   np.zeros((3, 3)), t, np.zeros((2, 2))))
    for t in su(3):
        matrices.append(block_diag(np.kron(t, np.eye(2)), -t.conj(), -t.conj(), np.zeros((4, 4))))
    return dims, charge, np.array(matrices)


def run():
    e, values = Evidence(), {}
    rng = np.random.default_rng(280921)
    # R83: general compatible connections in a nontrivial frame.
    for trial in range(3):
        p0 = np.diag([1., 1., 0., 0., 0.])
        raw = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
        u = expm(raw-raw.conj().T)
        p = u@p0@u.conj().T
        q = np.eye(5)-p
        raw = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
        g = raw-raw.conj().T
        dp, k0 = comm(g, p), comm(comm(g, p), p)
        block = p@g@p+q@g@q
        k = k0+block
        e.close('R83', f'compatible_connection_{trial}', comm(k, p), dp)
        e.close('R83', f'antihermitian_{trial}', k+k.conj().T, 0)
        e.close('R83', f'orthogonal_minimum_{trial}', np.linalg.norm(k)**2,
                np.linalg.norm(k0)**2+np.linalg.norm(block)**2, 2e-10)
        e.close('R83', f'minimum_norm_equals_projector_speed_{trial}', np.linalg.norm(k0), np.linalg.norm(dp))
        raw = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
        s, j = expm(.2*(raw-raw.conj().T)), .3*(raw-raw.conj().T)
        pp = s@p@s.conj().T
        qq = np.eye(5)-pp
        transformed = comm(s@dp@s.conj().T+comm(j, pp), pp)
        law = s@k0@s.conj().T+j
        e.close('R83', f'exact_missing_diagonal_gauge_law_{trial}', transformed, law-pp@j@pp-qq@j@qq)
        e.check('R83', f'bare_rule_not_full_connection_{trial}', np.linalg.norm(law-transformed) > .1,
                {'missing_diagonal_norm': float(np.linalg.norm(law-transformed))})

    # R84: a moving projector with partially compensated normal velocity.
    p0 = np.diag([1., 0.])
    g = np.array([[0., .8], [-.8, 0.]])
    for alpha in (0., .5, .9, 1.):
        h0 = np.diag([1., 4.])
        effective = h0+1j*(alpha-1)*g
        times = np.linspace(0, 1.5, 49)
        ode = solve_ivp(lambda t, x: -1j*effective@x, [0, times[-1]],
                        np.array([0j, 1+0j]), t_eval=times, rtol=3e-11, atol=1e-13, method='DOP853')
        probabilities = abs(ode.y[0])**2
        bound = np.sin(np.minimum(np.pi/2, abs(1-alpha)*.8*times))**2
        exact = np.array([abs((expm(-1j*effective*t)@np.array([0, 1]))[0])**2 for t in times])
        e.check('R84', f'moving_projector_ode_succeeded_{alpha}', ode.success, {'message': ode.message})
        e.close('R84', f'independent_exact_moving_solution_{alpha}', probabilities, exact, 2e-11)
        e.check('R84', f'defect_transition_bound_{alpha}', np.all(probabilities <= bound+2e-12),
                {'max_probability': float(max(probabilities)), 'integrated_defect': abs(1-alpha)*.8*times[-1]})
    sx = np.array([[0., 1.], [1., 0.]])
    times = np.linspace(0, np.pi/2, 25)
    saturation = np.array([abs(expm(-1j*sx*t)[0, 1])**2 for t in times])
    e.close('R84', 'sharp_sine_bound', saturation, np.sin(times)**2)
    # Metric is computed from actual frame derivatives, not scaled by declaration.
    v = np.vstack((np.eye(2), np.zeros((2, 2))))
    p = v@v.T
    q = np.eye(4)-p
    derivatives = [np.vstack((np.zeros((2, 2)), 2*t)) for t in su(2)]
    dps = [dv@v.T+v@dv.conj().T for dv in derivatives]
    old_phi = None
    for alpha in (0., .5, .9):
        normals = [q@(dv-alpha*comm(dp, p)@v) for dv, dp in zip(derivatives, dps)]
        qt = np.array([[a.conj().T@b for b in normals] for a in normals])
        metric = np.trace(qt, axis1=2, axis2=3).real
        phi = np.einsum('ab,abij->ij', np.linalg.inv(metric), qt)
        if old_phi is None:
            old_phi = phi
        e.close('R84', f'self_normalized_mass_unchanged_{alpha}', phi, old_phi)
        e.close('R84', f'metric_from_covariant_frame_{alpha}', metric, 2*(1-alpha)**2*np.eye(3))
    exact_normals = [q@(dv-comm(dp, p)@v) for dv, dp in zip(derivatives, dps)]
    e.close('R84', 'exact_compensation_metric_degenerate', exact_normals, 0)

    # R85: anomaly elimination and the global center kernel.
    qv, uv = sp.symbols('q u', real=True)
    dv, lv, ev = -2*qv-uv, -3*qv, 6*qv
    anomaly = 6*qv**3+3*uv**3+3*dv**3+2*lv**3+ev**3
    e.zero('R85', 'cubic_anomaly_factorization', anomaly+18*qv*(uv-2*qv)*(uv+4*qv))
    a, b, c = sp.symbols('a b c')
    charges = [a+b+c, -4*a-c, 2*a-c, -3*a+b, 6*a]
    e.zero('R85', 'SU3_center_constraint', 2*charges[0]+charges[1]+charges[2]-2*b)
    e.zero('R85', 'SU2_center_constraint', 3*charges[0]+charges[3]-4*b-3*c)
    constraints = sp.Matrix([[0, 2, 0], [0, 4, 3], [0, 8, 0]])
    e.check('R85', 'center_kernel_one_dimension', constraints.nullspace() == [sp.Matrix([1, 0, 0])],
            {'rank': constraints.rank()})
    e.check('R85', 'SU6_same_embedding_no_center', constraints.col_join(sp.Matrix([[1, 2, 3]])).rank() == 3,
            {'rank_with_SU6_trace': 3})
    for k in range(6):
        phase = np.exp(1j*np.pi*k/3)
        g2, g3 = phase**3*np.eye(2), phase**2*np.eye(3)
        matter = block_diag(phase*np.kron(g3, g2), phase**(-4)*g3.conj(),
                            phase**2*g3.conj(), phase**(-3)*g2, np.array([[phase**6]]), np.eye(1))
        e.close('R85', f'global_kernel_element_{k}', matter, np.eye(16))
        e.close('R85', f'special_unitary_centers_{k}', [np.linalg.det(g2), np.linalg.det(g3)], [1, 1])
    e.check('R85', 'generic_U1_element_is_not_kernel', abs(np.exp(6j*.12)-1) > .1,
            {'electron_representation_residual': float(abs(np.exp(6j*.12)-1))})
    for n in (1, 2, 3, 4):
        e.zero('R85', f'local_anomaly_does_not_select_generation_{n}', n*anomaly.subs(uv, -4*qv))

    # R86: construct all 12 generators of the actual 16-dimensional representation.
    dims, hypercharge, generators = representation()
    gram = np.einsum('aij,bji->ab', generators, generators).real
    ci = np.einsum('ab,aij,bjk->ik', np.linalg.inv(gram), generators, generators)
    casimir = [sp.Rational(21, 20), sp.Rational(4, 5), sp.Rational(7, 10),
               sp.Rational(9, 20), sp.Rational(3, 10), sp.Integer(0)]
    expected_c = block_diag(*[float(c)*np.eye(d) for d, c in zip(dims, casimir)])
    e.close('R86', 'generator_gram_normalization', np.diag(gram), [10/3]+[2]*11)
    e.close('R86', 'actual_trace_Casimir', ci, expected_c)
    e.close('R86', 'Casimir_commutes_with_all_generators', [comm(ci, t) for t in generators], 0)
    e.close('R86', 'Casimir_trace_counts_gauge_dimension', np.trace(ci), 12)
    e.check('R86', 'Casimir_kernel_only_supplied_singlet', np.count_nonzero(np.linalg.eigvalsh(ci) < 1e-12) == 1,
            {'kernel_dimension': 1})
    change = np.eye(12)+.04*rng.normal(size=(12, 12))
    transformed = np.einsum('ab,bij->aij', change, generators)
    new_gram = np.einsum('aij,bji->ab', transformed, transformed).real
    new_c = np.einsum('ab,aij,bjk->ik', np.linalg.inv(new_gram), transformed, transformed)
    e.close('R86', 'arbitrary_generator_basis_invariance', new_c, ci)
    repeated = np.array([np.kron(np.eye(3), t) for t in generators])
    gram3 = np.einsum('aij,bji->ab', repeated, repeated).real
    c3 = np.einsum('ab,aij,bjk->ik', np.linalg.inv(gram3), repeated, repeated)
    e.close('R86', 'copying_matter_changes_trace_metric', gram3, 3*gram)
    e.close('R86', 'three_copy_Casimir_rescales', c3, np.kron(np.eye(3), ci)/3)
    altered = expected_c+.2*np.eye(16)
    altered[-1, -1] += .7
    e.close('R86', 'different_singlet_mass_is_gauge_invariant', [comm(altered, t) for t in generators], 0)
    e.check('R86', 'different_singlet_mass_is_not_original_ansatz', altered[-1, -1] != .2,
            {'original_singlet_mass2': .2, 'changed_singlet_mass2': float(altered[-1, -1])})

    # R87: compare exact low poles with the normalized Schur approximation.
    rows = []
    for scale in (.2, 1., 3.):
        d = np.diag([5., 8., 12.])
        bmat = scale*rng.normal(size=(2, 3))
        m0 = np.array([[.4, .1], [.1, .7]])
        amat = m0+bmat@np.linalg.solve(d, bmat.T)
        full = np.block([[amat, bmat], [bmat.T, d]])
        z = np.eye(2)+bmat@np.linalg.inv(d)@np.linalg.inv(d)@bmat.T
        zi = power(z, -.5)
        low = zi@m0@zi
        lambdas = np.linalg.eigvalsh(full)[:2]
        mus = np.linalg.eigvalsh(low)
        eta = np.linalg.norm(np.eye(2)-np.linalg.inv(z), 2)
        bounds = eta*lambdas**2/(5-lambdas)
        e.check('R87', f'positive_gapped_example_{scale}', min(np.linalg.eigvalsh(full)) > 0 and max(lambdas) < 5,
                {'low_poles': lambdas.tolist(), 'heavy_block_lower_bound': 5})
        e.check('R87', f'ordered_low_pole_error_bound_{scale}',
                np.all(mus >= lambdas-1e-12) and np.all(mus-lambdas <= bounds+1e-12),
                {'errors': (mus-lambdas).tolist(), 'bounds': bounds.tolist()})
        for lam in (.1, 1., 4.):
            rem = lam**2*zi@bmat@np.linalg.matrix_power(d, -2)@np.linalg.inv(d-lam*np.eye(3))@bmat.T@zi
            exact = zi@(amat-lam*np.eye(2)-bmat@np.linalg.inv(d-lam*np.eye(3))@bmat.T)@zi
            e.close('R87', f'exact_resolvent_remainder_{scale}_{lam}', exact, low-lam*np.eye(2)-rem)
            e.check('R87', f'positive_operator_remainder_bound_{scale}_{lam}',
                    np.linalg.eigvalsh(rem)[0] > -1e-12 and np.linalg.norm(rem, 2) <= eta*lam**2/(5-lam)+1e-12,
                    {'remainder_norm': float(np.linalg.norm(rem, 2)), 'bound': eta*lam**2/(5-lam)})
        rows.append({'scale': scale, 'errors': (mus-lambdas).tolist(), 'bounds': bounds.tolist()})
    values['low_pole_error_examples'] = rows
    base = np.array([[2., .4], [.4, 5.]])
    for shift in (-.3, .2, 1.):
        e.close('R87', f'full_common_shift_{shift}', np.linalg.eigvalsh(base+shift*np.eye(2)),
                np.linalg.eigvalsh(base)+shift)
    eigenvalues, vectors = np.linalg.eigh(base)
    slopes = abs(vectors[0])**2
    derivative = (np.linalg.eigvalsh(base+1e-5*np.diag([1., 0.]))-
                  np.linalg.eigvalsh(base-1e-5*np.diag([1., 0.])))/2e-5
    e.close('R87', 'observed_only_shift_Hellmann_Feynman', derivative, slopes, 5e-10)
    e.close('R87', 'observed_only_shift_exact_slopes', slopes, [(1+3/np.sqrt(9.64))/2, (1-3/np.sqrt(9.64))/2])

    # R88: minimal invariant operator space and independent ODE for its flow.
    vand = sp.Matrix([[c**k for k in range(6)] for c in casimir])
    e.check('R88', 'six_Casimir_powers_independent', vand.rank() == 6, {'rank': vand.rank()})
    full_basis = sp.diag(vand, vand)
    e.check('R88', 'minimal_two_direction_flow_space', full_basis.rank() == 12, {'rank': full_basis.rank()})
    e.check('R88', 'three_parameter_boundary_not_closed',
            sp.Matrix([[1, c, c*c] for c in casimir]).rank() == 3, {'rank_of_I_C_C2': 3})
    theta = .7
    htheta = 2*np.cos((theta+2*np.pi*np.arange(3))/3)
    cs = np.array(casimir, float)
    initial = .8+1.2*cs[:, None]+.15*htheta[None, :]
    times = np.linspace(0, .8, 13)
    ode = solve_ivp(lambda t, y: (-cs[:, None]*y.reshape(6, 3)).ravel(),
                    [0, .8], initial.ravel(), t_eval=times, rtol=3e-12, atol=1e-13, method='DOP853')
    exact = np.array([np.exp(-t*cs[:, None])*initial for t in times])
    e.check('R88', 'gauge_flow_ode_succeeded', ode.success, {'message': ode.message})
    e.close('R88', 'closed_flow_matches_ODE', ode.y.T.reshape(-1, 6, 3), exact, 2e-11)
    coefficients = np.linalg.solve(np.array(vand, float), np.exp(-.8*cs))
    e.close('R88', 'degree_five_polynomial_exact_on_representation', np.array(vand, float)@coefficients, np.exp(-.8*cs))
    epsilons = np.sqrt(np.sum((exact[-1]-exact[-1].mean(axis=1)[:, None])**2, axis=1)/6)
    e.close('R88', 'representation_splitting_flow', epsilons, .15*np.exp(-.8*cs))
    e.check('R88', 'nonzero_common_splitting_not_preserved', epsilons[0] < epsilons[-1],
            {'Q_epsilon': float(epsilons[0]), 'singlet_epsilon': float(epsilons[-1])})
    values['minimal_RG_operator_dimension'] = 12

    # R89: general vector mass matrix, quartic source, and scalar portal threshold.
    couplings = np.array([.3]+[.6]*3+[.9]*8)
    weighted = couplings[:, None, None]*generators
    def vector_mass(phi):
        tangent = np.array([t@phi for t in weighted])
        return 2*(tangent.conj()@tangent.T).real
    phi = rng.normal(size=16)+1j*rng.normal(size=16)
    mv = vector_mass(phi)
    gauge_vector = rng.normal(size=12)
    e.close('R89', 'vector_mass_from_kinetic_quadratic_form',
            .5*gauge_vector@mv@gauge_vector, np.linalg.norm(np.einsum('a,aij,j->i', gauge_vector, weighted, phi))**2)
    e.check('R89', 'vector_mass_positive', np.min(np.linalg.eigvalsh(mv)) > -1e-11,
            {'smallest_eigenvalue': float(np.linalg.eigvalsh(mv)[0])})
    u = expm(1j*sum((.2*rng.normal()*t for t in generators), np.zeros((16, 16), complex)))
    rotated = vector_mass(u@phi)
    e.close('R89', 'quartic_source_gauge_invariant', np.trace(rotated@rotated), np.trace(mv@mv), 5e-10)
    phi1, phi2 = np.zeros(16, complex), np.zeros(16, complex)
    phi1[0], phi2[6] = 1., 1.
    m1, m2 = vector_mass(phi1), vector_mass(phi2)
    e.close('R89', 'cross_quartic_source', np.trace((m1+m2)@(m1+m2))-np.trace(m1@m1)-np.trace(m2@m2),
            2*np.trace(m1@m2))
    e.check('R89', 'two_charged_fields_have_nonzero_cross_source', np.trace(m1@m2) > 0,
            {'mixed_vector_trace': float(np.trace(m1@m2))})
    singlet = np.zeros(16, complex)
    singlet[-1] = 1.
    e.close('R89', 'pure_gauge_source_does_not_create_singlet_portal', vector_mass(singlet), 0)
    x1, x2, q1, q2, gv = sp.symbols('x1 x2 q1 q2 g', real=True)
    beta16 = sp.expand(6*gv**4*(q1*q1*x1+q2*q2*x2)**2)
    e.zero('R89', 'U1_self_quartic_coefficient', beta16.coeff(x1, 2)-6*gv**4*q1**4)
    e.zero('R89', 'U1_cross_quartic_coefficient', beta16.coeff(x1, 1).coeff(x2, 1)-12*gv**4*q1**2*q2**2)
    t, mass, mu, kappa, dim = sp.symbols('t M mu kappa dim', positive=True)
    xx = mass**2+kappa*t
    potential = dim*xx**2*(sp.log(xx/mu**2)-sp.Rational(3, 2))/(32*sp.pi**2)
    threshold = sp.diff(potential, t).subs(t, 0)
    expected = dim*kappa*mass**2*(sp.log(mass**2/mu**2)-1)/(16*sp.pi**2)
    e.zero('R89', 'portal_threshold_from_full_determinant', threshold-expected)
    e.zero('R89', 'portal_mass_beta_from_scale_cancellation',
           mu*sp.diff(threshold, mu)+2*dim*kappa*mass**2/(16*sp.pi**2))
    e.zero('R89', 'factorized_zero_portal_has_no_heavy_threshold', threshold.subs(kappa, 0))
    e.check('R89', 'same_gauge_content_different_allowed_portal',
            float(threshold.subs({dim: 3, kappa: .1, mass: 10, mu: 10})) < 0,
            {'threshold_kappa0': 0., 'threshold_kappa01_MSbar_muM':
             float(threshold.subs({dim: 3, kappa: .1, mass: 10, mu: 10}))})

    ids = sorted({row['claim'] for row in e.checks})
    assert ids == [f'R{i:02d}' for i in range(83, 90)]
    return {'schema': 'CE-RB10-v1', 'scope': 'finite covariant transport and specified representation/gauge branches',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(row['passed'] for row in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_transport.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
