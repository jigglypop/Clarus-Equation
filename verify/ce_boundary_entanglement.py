"""CE-BE1: boundary-action obstruction and conditional entanglement equilibrium.

No common microscopic action, new Einstein derivation, or observational fit.
Independent routes: constant-curvature quadrature, rational rank, entropy
eigenvalues, and the matrix-log resolvent. Inputs were registered in chapter 57.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from ce_color_covariant_record import HF, setup


SIGMA = np.diag([.2, .3, .5])
X = np.array([[.10, .04, 0.], [.04, -.06, .03], [0., .03, -.04]])


def boundary_action_audit():
    rows = []
    for omega in (1., 4.):
        for n in (1, 2):
            numerical, uncertainty = quad(
                lambda t: (n*np.pi*np.cos(n*np.pi*t))**2
                - omega**2*np.sin(n*np.pi*t)**2, 0., 1., epsabs=1e-12, epsrel=1e-13)
            analytic = ((n*np.pi)**2-omega**2)/2
            assert abs(numerical-analytic) < 1e-12
            rows.append(dict(omega=omega, mode=n, second_variation=numerical,
                             analytic=analytic, quadrature_error=uncertainty))
    assert rows[2]['second_variation'] < 0 < rows[3]['second_variation']
    return dict(same_zero_endpoints=True, rows=rows,
                omega_4_stationary_zero_path_is_saddle=True,
                unique_action_from_endpoints=False)


def geometry_audit():
    with mp.workdps(60):
        pi = mp.pi
        volume0 = 4*pi/3

        def radial(k, r):
            return mp.sin(mp.sqrt(k)*r)/mp.sqrt(k) if k > 0 else mp.sinh(mp.sqrt(-k)*r)/mp.sqrt(-k)

        def volume(k, radius):
            return 4*pi*mp.quad(lambda r: radial(k, r)**2, [0, radius])

        rows, slopes = [], []
        expected = -4*pi/5  # d(A at fixed V)/d kappa, R^(3)=6*kappa, ell=1
        for step in (mp.mpf('1e-4'), mp.mpf('1e-5')):
            areas = []
            for k in (-step, step):
                radius = mp.findroot(lambda r: volume(k, r)-volume0, (mp.mpf('.99'), mp.mpf('1.01')))
                area = 4*pi*radial(k, radius)**2
                areas.append(area)
                rows.append(dict(kappa=float(k), radius=float(radius), area=float(area),
                                 volume_error=float(abs(volume(k, radius)-volume0))))
            derivative = (areas[1]-areas[0])/(2*step)
            relative_error = abs(derivative/expected-1)
            assert relative_error < mp.mpf('1e-8')
            slopes.append(dict(step=float(step), derivative=float(derivative),
                               relative_error=float(relative_error)))
        modular_weight = mp.quad(lambda r: 4*pi*r*r*(1-r*r)/2, [0, 1])
        expected_weight = 4*pi/15
        assert abs(modular_weight-expected_weight) < mp.mpf('1e-50')
        # eta*dA + delta<K> = 0, with G00 = 3*kappa.
        # eta is an INPUT. This is the coefficient G=1/(4*hbar*eta).
        inferred_G00_per_T00_at_eta_hbar_1 = -3*(2*pi*modular_weight)/expected
        assert abs(inferred_G00_per_T00_at_eta_hbar_1-2*pi) < mp.mpf('1e-50')
        return dict(constant_curvature_cases=rows, centered_area_slopes=slopes,
                    expected_fixed_volume_slope=float(expected),
                    fixed_radius_slope=float(-4*pi/3),
                    modular_radial_integral=float(modular_weight),
                    modular_integral_error=float(abs(modular_weight-expected_weight)),
                    G00_over_T00_at_eta_hbar_1=float(inferred_G00_per_T00_at_eta_hbar_1),
                    newton_constant_status='G=1/(4*hbar*eta); eta supplied, not predicted')


def entropy(rho):
    values = np.linalg.eigvalsh(rho)
    assert values.min() > -1e-12
    positive = values[values > 1e-14]
    return float(-np.sum(positive*np.log(positive)))


def entropy_audit():
    p = np.diag(SIGMA)
    modular = -np.diag(np.log(p))
    first = float(np.trace(X@modular))
    divided = np.array([[1/p[i] if i == j else
                         (np.log(p[i])-np.log(p[j]))/(p[i]-p[j])
                         for j in range(3)] for i in range(3)])
    quadratic = float(np.sum(abs(X)**2*divided))
    resolvent, uncertainty = quad(lambda t: float(np.sum(abs(X)**2/
        np.outer(p+t, p+t))), 0., np.inf, epsabs=1e-12, epsrel=1e-13)
    assert abs(quadratic-resolvent) < 1e-12
    rows = []
    for epsilon in (-.01, -.001, .001, .01):
        rho = SIGMA+epsilon*X
        assert np.linalg.eigvalsh(rho).min() > 0
        assert abs(np.trace(rho)-1) < 1e-12
        vals, vecs = np.linalg.eigh(rho)
        log_rho = (vecs*np.log(vals))@vecs.T
        relative = float(np.trace(rho@(log_rho+modular)))
        ds = entropy(rho)-entropy(SIGMA)
        dk = float(np.trace((rho-SIGMA)@modular))
        identity_error = abs(relative-(dk-ds))
        assert relative > 0 and identity_error < 1e-12
        rows.append(dict(epsilon=epsilon, minimum_eigenvalue=float(vals.min()),
                         entropy_change=ds, modular_change=dk, relative_entropy=relative,
                         identity_error=identity_error, linear_area_balance_entropy_change=ds-dk))
    h = .001
    derivative = (entropy(SIGMA+h*X)-entropy(SIGMA-h*X))/(2*h)
    curvature = (entropy(SIGMA+h*X)+entropy(SIGMA-h*X)-2*entropy(SIGMA))/(h*h)
    assert abs(derivative-first) < 1e-7
    assert abs(curvature+quadratic) < 1e-6
    return dict(sigma=SIGMA.tolist(), perturbation=X.tolist(), cases=rows,
                analytic_entropy_derivative=first, numerical_entropy_derivative=derivative,
                second_entropy_derivative=curvature, positive_relative_entropy_hessian=quadratic,
                log_resolvent_hessian=resolvent, resolvent_quadrature_error=uncertainty,
                scope='maximum of S-<K> at fixed sigma; not a nonlinear gravity stability proof')


def record_audit():
    p = np.diag(SIGMA)
    pure = np.sqrt(p)
    isometry = np.zeros((9, 3))
    for i in range(3):
        isometry[3*i+i, i] = 1
    recorded = isometry@pure
    joint = np.outer(recorded, recorded)
    reduced = np.trace(joint.reshape(3, 3, 3, 3), axis1=1, axis2=3)
    kraus = [np.diag(np.eye(3)[i]) for i in range(3)]
    choi = sum(np.outer(k.reshape(-1), k.reshape(-1)) for k in kraus)
    dephased_joint = sum(np.kron(k, np.eye(3))@joint@np.kron(k, np.eye(3)) for k in kraus)
    remote_before = np.trace(joint.reshape(3, 3, 3, 3), axis1=0, axis2=2)
    remote_after = np.trace(dephased_joint.reshape(3, 3, 3, 3), axis1=0, axis2=2)
    errors = dict(isometry=float(np.max(abs(isometry.T@isometry-np.eye(3)))),
                  completeness=float(np.max(abs(sum(k.T@k for k in kraus)-np.eye(3)))),
                  reduced_state=float(np.max(abs(reduced-SIGMA))),
                  remote_marginal=float(np.max(abs(remote_after-remote_before))))
    assert max(errors.values()) < 1e-12
    assert np.linalg.eigvalsh(choi).min() >= -1e-12
    return dict(errors=errors, choi_eigenvalues=np.linalg.eigvalsh(choi).tolist(),
                whole_recorded_state_entropy=entropy(joint),
                restricted_system_entropy=entropy(reduced),
                dephased_joint_entropy=entropy(dephased_joint),
                no_signalling_scope='finite tensor factors and nonselective channel only',
                actual_outcome_selection_derived=False)


def exact_rank(rows):
    a = [[F(x) for x in row] for row in rows]
    rank = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x/scale for x in a[rank]]
        for i in range(rank+1, len(a)):
            multiplier = a[i][col]
            a[i] = [x-multiplier*y for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def observer_audit():
    observers = [[F(1), F(0), F(0), F(0)]]
    for axis in (1, 2, 3):
        for sign in (-1, 1):
            u = [F(5, 4), F(0), F(0), F(0)]
            u[axis] = sign*F(3, 4)
            observers.append(u)
    for i, j in ((1, 2), (1, 3), (2, 3)):
        u = [F(3), F(0), F(0), F(0)]
        u[i] = u[j] = F(2)
        observers.append(u)
    for u in observers:
        assert -u[0]**2+sum(v*v for v in u[1:]) == -1
    components = [(i, j) for i in range(4) for j in range(i, 4)]
    matrix = [[u[i]*u[j]*(1 if i == j else 2) for i, j in components] for u in observers]
    assert exact_rank(matrix) == 10
    assert np.linalg.matrix_rank(np.array(matrix, float)) == 10
    return dict(observers=[[str(v) for v in u] for u in observers], rank=exact_rank(matrix),
                one_rest_observer_rank=exact_rank(matrix[:1]),
                counterexample_tensor_diagonal=[0, 1, -1, 0],
                rest_contraction=0, first_axis_boost_contraction=float(F(9, 16)))


def am4_modular_bridge_audit():
    z, phi, _ = setup()
    embedding = np.column_stack([z, phi])
    energy = embedding.conj().T@HF@embedding
    sigma = np.diag([.2, .8])
    tangent = np.diag([1., -1.])
    modular = -np.diag(np.log(np.diag(sigma)))
    energy_derivative = float(np.trace(tangent@energy).real)
    entropy_derivative = float(np.trace(tangent@modular))
    assert np.max(abs(energy-5*np.eye(2))) < 1e-12
    assert abs(energy_derivative) < 1e-12
    assert abs(entropy_derivative-np.log(4)) < 1e-12
    h = .001
    numeric = (entropy(sigma+h*tangent)-entropy(sigma-h*tangent))/(2*h)
    # This finite difference has an O(h^2) remainder bounded here by 5e-6.
    assert abs(numeric-entropy_derivative) < 5e-6
    return dict(support_dimension=2, full_AM4_dimension=16,
                compressed_free_energy=energy.real.tolist(),
                entropy_derivative=entropy_derivative, numerical_entropy_derivative=numeric,
                free_energy_derivative=energy_derivative,
                best_identity_plus_energy_operator_norm_error=float(np.log(4)/2),
                symmetric_positions_have_equal_ball_modular_weight=True,
                disposition='direct_prepared_AM4_to_CFT_vacuum_modular_bridge_rejected',
                limits='The prepared finite support state is not a relativistic local vacuum.')


def record_charge_audit():
    record = np.diag([0., 1.])
    transition = np.array([[0., 1.], [1., 0.]])
    hamiltonian = 5*np.eye(2)+(np.pi/2)*transition
    modular = -np.diag(np.log([.2, .8]))
    commutator = hamiltonian@modular-modular@hamiltonian
    norm = float(np.linalg.norm(commutator, 2))
    exact_norm = np.pi*np.log(4)/2
    assert abs(norm-exact_norm) < 1e-12 and norm > 1
    vals, vecs = np.linalg.eigh(hamiltonian)
    weights = np.exp(-vals)
    gibbs = (vecs*(weights/weights.sum()))@vecs.T
    gibbs_record = float(np.trace(gibbs@record))
    conservation_error = float(np.max(abs(hamiltonian@gibbs-gibbs@hamiltonian)))
    assert abs(gibbs_record-.5) < 1e-12 and conservation_error < 1e-12
    return dict(prepared_record_probability=.8,
                commutator_H_K_operator_norm=norm,
                independent_norm_formula=exact_norm,
                record_as_independent_conserved_modular_charge=False,
                fixed_beta_1_gibbs_state=gibbs.tolist(), gibbs_record_probability=gibbs_record,
                gibbs_conservation_error=conservation_error,
                gibbs_does_not_replace_prepared_state=True)


def run():
    return dict(candidate='CE-BE1', status='conditional_boundary_equilibrium_gate',
                boundary_action=boundary_action_audit(), geometry=geometry_audit(),
                entropy=entropy_audit(), record=record_audit(), observers=observer_audit(),
                AM4_bridge=am4_modular_bridge_audit(),
                record_charge=record_charge_audit(),
                fitted_parameters=0, full_joint_rmse=None, scientific_success=False,
                limits=['Local Lorentz metric, local CFT vacuum and universal area density are assumed.',
                        'No common microscopic action or quantum metric state is constructed.',
                        'No autonomous outcome selection or matter/gauge representation is derived.',
                        'Massive-field modular terms and record-conditioned gravity remain open.',
                        'No observational input, holdout assignment, or joint score.'])


if __name__ == '__main__':
    result = run()
    here = Path(__file__).resolve()
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__)
    result['script_sha256'] = hashlib.sha256(here.read_bytes()).hexdigest()
    result['dependency_sha256'] = {name: hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                                   for name in ('ce_color_covariant_record.py', 'ce_isometric_color_frame.py')}
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(candidate=result['candidate'], status=result['status'],
                         saddle=result['boundary_action']['omega_4_stationary_zero_path_is_saddle'],
                         area_slopes=result['geometry']['centered_area_slopes'],
                         observer_rank=result['observers']['rank'],
                         AM4_bridge=result['AM4_bridge'],
                         record=result['record']), indent=2))
