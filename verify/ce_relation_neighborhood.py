"""Finite-neighborhood and hierarchy checks of the shared relation frame.

No new fit, state, counterterm or observational success is introduced.
The curvature commutator tests parallel mass transport, not gauge covariance
of an endomorphism-valued potential, which holds irrespective of this test.
"""
import json
from pathlib import Path
import numpy as np
import sympy as sy
from scipy.linalg import block_diag

from ce_relation_mass_force import make_frame


def factors(theta=.7, t=1., v=.3):
    S = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    Z = np.diag(np.exp(2j*np.pi*np.arange(3)/3))
    U = [np.eye(3), Z, S@Z@Z]
    B1 = [np.array([[1.], [0.]], complex)/2,
          np.array([[1j], [0.]])/2, np.array([[0.], [1.]])/2]
    B2 = [np.array([[0., 1.], [1., 0.]], complex),
          np.array([[0., -1j], [1j, 0.]]), np.diag([1., -1.])]
    Bg = [block_diag(*blocks) for blocks in zip(B1, B2, U)]
    C = t*np.eye(3)+v*np.exp(1j*theta/3)*S
    Bf = []
    for a in range(3):
        b = np.zeros((9, 3), complex)
        b[3*a:3*a+3] = C
        Bf.append(b)
    return Bg, Bf, C, S


def graph_tensor(B, q):
    Y = sum(qa*b for qa, b in zip(q, B))
    w, u = np.linalg.eigh(np.eye(Y.shape[1])+Y.conj().T@Y)
    root = (u/np.sqrt(w))@u.conj().T
    R = np.linalg.inv(np.eye(Y.shape[0])+Y@Y.conj().T)
    return np.array([[root@a.conj().T@R@b@root for b in B] for a in B])


def tensor_geometry(q, theta=.7):
    Bg, Bf, C, S = factors(theta)
    Qg, Qf = graph_tensor(Bg, q), graph_tensor(Bf, q)
    h = 3*np.trace(Qg, axis1=2, axis2=3).real+6*np.trace(Qf, axis1=2, axis2=3).real
    inv = np.linalg.inv(h)
    Pg = np.einsum('ab,abij->ij', inv, Qg)
    Pf = np.einsum('ab,abij->ij', inv, Qf)
    Fg = 1j*(Qg-Qg.swapaxes(0, 1))
    Ff = 1j*(Qf-Qf.swapaxes(0, 1))
    # Analytic flavor tensor: M commutes with I+|q|^2 M.
    M = C.conj().T@C
    T = M@np.linalg.inv(np.eye(3)+np.dot(q, q)*M)
    expected = np.array([[float(a == b)*T-q[a]*q[b]*T@T for b in range(3)] for a in range(3)])
    assert np.max(abs(expected-Qf)) < 3e-14
    assert np.max(abs(Ff)) < 3e-14
    assert abs(3*np.trace(Pg)+6*np.trace(Pf)-3) < 3e-14
    assert np.linalg.eigvalsh(h)[0] > 0
    return h, Pg, Pf, Fg, Qg, Qf


def finite_frame_check(q, step):
    Bg, Bf, _, _ = factors()
    def frame(x): return np.kron(make_frame(Bg, x), make_frame(Bf, x))
    V = frame(q)
    derivatives = [(frame(q+step*np.eye(3)[a])-frame(q-step*np.eye(3)[a]))/(2*step) for a in range(3)]
    horizontal = [d-V@(V.conj().T@d) for d in derivatives]
    direct = np.array([[a.conj().T@b for b in horizontal] for a in horizontal])
    _, _, _, _, Qg, Qf = tensor_geometry(q)
    expected = np.array([[np.kron(Qg[a,b], np.eye(3))+np.kron(np.eye(6), Qf[a,b]) for b in range(3)] for a in range(3)])
    return float(np.max(abs(direct-expected)))


def run():
    direction = np.array([.4, -.3, .5]); direction /= np.linalg.norm(direction)
    rows = []
    for distance in [0., .01, .02, .05, .1, .2, .5]:
        q = distance*direction
        h, Pg, Pf, Fg, _, _ = tensor_geometry(q)
        comm = max(np.linalg.norm(Pg@F-F@Pg, 'fro') for F in Fg.reshape(-1, 6, 6))
        centered = Pf-np.trace(Pf)*np.eye(3)/3
        eps = float(np.sqrt(np.trace(centered@centered).real/6))
        cos_phase = float(np.trace(centered@centered@centered).real/(6*eps**3))
        color = Pg[3:, 3:]
        color_spread = float(np.ptp(np.linalg.eigvalsh(color)))
        rows.append(dict(distance=distance, relation_point=q.tolist(),
                         metric_eigenvalues=np.linalg.eigvalsh(h).tolist(),
                         parallel_mass_integrability_commutator=float(comm),
                         color_geometric_potential_eigenvalue_spread=color_spread,
                         cyclic_epsilon_invariant=eps, cyclic_cos_phase_invariant=cos_phase))
    assert rows[0]['parallel_mass_integrability_commutator'] < 1e-14
    assert rows[3]['parallel_mass_integrability_commutator'] > 1e-6
    scaling = rows[2]['parallel_mass_integrability_commutator']/rows[1]['parallel_mass_integrability_commutator']
    assert 3.9 < scaling < 4.1
    errors = [finite_frame_check(.2*direction, step) for step in [2e-4, 1e-4]]
    assert errors[1] < .27*errors[0]
    # Gauge conjugation cannot remove a nonzero Frobenius commutator norm.
    _, Pg, _, Fg, _, _ = tensor_geometry(.2*direction)
    rng = np.random.default_rng(36)
    U, _ = np.linalg.qr(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))
    Pnew = U.conj().T@Pg@U
    before = [np.linalg.norm(Pg@F-F@Pg, 'fro') for F in Fg.reshape(-1,6,6)]
    after = []
    for F in Fg.reshape(-1,6,6):
        new = U.conj().T@F@U
        after.append(np.linalg.norm(Pnew@new-new@Pnew, 'fro'))
    gauge_error = float(np.max(abs(np.array(after)-before)))
    assert gauge_error < 1e-14
    # No choice of overall length can create the required diagonal hierarchy.
    r = sy.symbols('r', positive=True)
    a_min = r/(2-4*r)
    upper = sy.factor((1+a_min)/(sy.Rational(1,4)+a_min))
    assert sy.simplify(upper-(4-6*r)) == 0
    hierarchy = []
    for rd, mass in [(.15,.027615), (.35,.014414)]:
        ratio = 4-6*rd
        maximum = np.sqrt(ratio)*mass
        hierarchy.append(dict(r_D=rd, neutral_mass_scale_eV=mass,
                              maximum_sH_over_sD=ratio,
                              maximum_charged_mass_scale_eV=float(maximum),
                              required_sH_over_sD=(1e12/mass)**2))
    old = json.loads(Path(__file__).with_name('ce_symmetric_bao_ruler.json').read_text())
    return dict(rows=rows, small_radius_quadratic_ratio=scaling,
                independent_frame_difference_errors=errors,
                gauge_conjugation_commutator_norm_error=gauge_error,
                hierarchy_bound=str(upper), hierarchy=hierarchy,
                previous_partial_rmse_range=[min(x['partial_rmse_14'] for x in old['rows']),
                                            max(x['partial_rmse_14'] for x in old['rows'])],
                new_joint_rmse=None, fitted_parameters=[],
                verdict='This fixed graph family is a local example, not a completion of the physical common-mass spectrum. No new observational prediction is promoted.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
