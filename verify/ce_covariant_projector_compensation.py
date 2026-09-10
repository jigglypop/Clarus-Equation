"""CE-GC1: a supplied compatible connection removes normal transport.

Exact finite-dimensional identities and two independent time-evolution forms.
This is not a derivation of physical gauge symmetry or an observation process.
"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from ce_isometric_color_frame import KS, Q0, V0, comm, frame


TOL = 1e-12
TIMES = (0., .25, .5, 1.)
ID = np.eye(4, dtype=complex)
P0 = ID - Q0


def error(matrix):
    return float(np.max(np.abs(matrix)))


def unitary(h, t):
    values, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-1j * t * values)) @ vectors.conj().T


def spatial_audit(t):
    v = frame((t, 0., 0.))
    p = v @ v.conj().T
    q = ID - p
    dv = [1j * g @ v for g in KS]
    dp = [1j * comm(g, p) for g in KS]
    connection = [comm(d, p) for d in dp]
    cov_dv = [d - k @ v for d, k in zip(dv, connection)]
    original_a = [1j * v.conj().T @ d for d in dv]
    cov_a = [1j * v.conj().T @ d for d in cov_dv]
    original_qgt = np.array([[a.conj().T @ q @ b for b in dv] for a in dv])
    cov_qgt = np.array([[a.conj().T @ q @ b for b in cov_dv] for a in cov_dv])
    h = np.trace(original_qgt, axis1=2, axis2=3).real
    cov_h = np.trace(cov_qgt, axis1=2, axis2=3).real
    phi = sum(original_qgt[a, a] for a in range(3))
    cov_phi = sum(cov_qgt[a, a] for a in range(3))
    errors = {
        'normalization': error(v.conj().T @ v - np.eye(3)),
        'antihermitian_connection': max(error(k + k.conj().T) for k in connection),
        'covariant_projector': max(error(d - comm(k, p)) for d, k in zip(dp, connection)),
        'covariant_normal_derivative': max(error(q @ d) for d in cov_dv),
        'projected_connection_preserved': max(error(a - b) for a, b in zip(original_a, cov_a)),
        'original_metric_identity': error(h - np.eye(3)),
        'original_normal_mass_identity': error(phi - np.eye(3)),
        'covariant_normal_metric_zero': error(cov_h),
        'covariant_normal_mass_zero': error(cov_phi),
    }
    curvature_norms = []
    curvature_errors = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        # Differentiate K_b=[d_b P,P] before using projector identities.
        dd_ab = -comm(KS[a], comm(KS[b], p))
        dd_ba = -comm(KS[b], comm(KS[a], p))
        d_a_k_b = comm(dd_ab, p) + comm(dp[b], dp[a])
        d_b_k_a = comm(dd_ba, p) + comm(dp[a], dp[b])
        curvature = -d_a_k_b + d_b_k_a + comm(connection[a], connection[b])
        expected = comm(dp[a], dp[b])
        # Here the compressed A_a are constant because all G_a commute.
        light_curvature = -1j * comm(original_a[a], original_a[b])
        curvature_errors.extend((error(curvature - expected),
                                 error(v.conj().T @ curvature @ v + 1j * light_curvature)))
        curvature_norms.append(float(np.linalg.norm(light_curvature)))
    errors['curvature_and_projection'] = max(curvature_errors)
    assert max(errors.values()) < TOL, errors
    assert min(curvature_norms) > 1.
    assert np.linalg.matrix_rank(cov_h, tol=TOL) == 0
    return {'q': [t, 0., 0.], 'errors': errors,
            'original_normal_mass_trace': float(np.trace(phi).real),
            'covariant_normal_mass_trace': float(np.trace(cov_phi).real),
            'covariant_normal_metric_rank_at_tolerance': 0,
            'retained_curvature_frobenius_norms': curvature_norms}


def time_audit(t):
    g = KS[0]
    u = np.cos(t) * ID + 1j * np.sin(t) * g
    p = u @ P0 @ u.conj().T
    q = ID - p
    h0 = P0 + 5 * Q0
    h = p + 5 * q
    dp = 1j * comm(g, p)
    k = comm(dp, p)
    g_diag = P0 @ g @ P0 + Q0 @ g @ Q0
    g_off = g - g_diag
    ordinary = u @ unitary(h0 + g, t)
    compensated = u @ unitary(h0 + g_diag, t)
    ordinary_dot = 1j * g @ ordinary - 1j * u @ (h0 + g) @ unitary(h0 + g, t)
    compensated_dot = (1j * g @ compensated
                       - 1j * u @ (h0 + g_diag) @ unitary(h0 + g_diag, t))
    psi0 = ID[:, 3]
    psi = ordinary @ psi0
    chi = compensated @ psi0
    ordinary_p = float(np.vdot(psi, p @ psi).real)
    compensated_p = float(np.vdot(chi, p @ chi).real)
    exact_p = float(np.sin(np.sqrt(5.) * t)**2 / 5.)
    errors = {
        'ordinary_unitarity': error(ordinary.conj().T @ ordinary - ID),
        'compensated_unitarity': error(compensated.conj().T @ compensated - ID),
        'ordinary_equation': error(1j * ordinary_dot - h @ ordinary),
        'compensated_equation': error(1j * compensated_dot - (h + 1j * k) @ compensated),
        'supplied_drive': error(1j * k + u @ g_off @ u.conj().T),
        'all_state_intertwining': error(compensated.conj().T @ p @ compensated - P0),
        'instantaneous_conservation_operator': error(dp + 1j * comm(h + 1j * k, p)),
        'ordinary_probability_closed_form': abs(ordinary_p - exact_p),
        'compensated_probability_zero': abs(compensated_p),
        'ordinary_instantaneous_energy': abs(float(np.vdot(psi, h @ psi).real) - (5 - 4 * exact_p)),
    }
    assert max(errors.values()) < TOL, errors
    if t > 0:
        assert ordinary_p > .01
    return {'t': t, 'ordinary_P_probability': ordinary_p,
            'closed_form_P_probability': exact_p,
            'compensated_P_probability': compensated_p,
            'drive_operator_norm': float(np.linalg.norm(1j * k, ord=2)),
            'ordinary_P_conservation_operator_error': error(ordinary.conj().T @ p @ ordinary - P0),
            'errors': errors}


def gauge_law_audit():
    # P'=S P0 S^dagger=P0 for S=exp(i t P0), but dS S^dagger=i P0.
    recomputed_k = np.zeros((4, 4), complex)
    required_k = 1j * P0
    mismatch = float(np.linalg.norm(recomputed_k - required_k))
    assert abs(mismatch - np.sqrt(3.)) < TOL
    assert error(comm(required_k, P0)) == 0.
    return {'fixed_projector_gauge_law_mismatch_frobenius': mismatch,
            'both_connections_preserve_P': True,
            'off_diagonal_rule_is_full_connection_gauge_law': False}


def main():
    spatial = [spatial_audit(t) for t in TIMES]
    temporal = [time_audit(t) for t in TIMES]
    gauge = gauge_law_audit()
    maximum_error = max(max(row['errors'].values()) for row in spatial + temporal)
    here = Path(__file__).resolve()
    output = {
        'candidate': 'CE-GC1', 'tolerance': TOL,
        'inputs': {'times': TIMES, 'energy_a': 1., 'energy_b': 5., 'ell': 1.,
                   'hbar': 1., 'initial_state': '|11>', 'fit_parameters': 0},
        'spatial': spatial, 'temporal': temporal, 'gauge_law': gauge,
        'maximum_identity_error': maximum_error,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'source_sha256': {name: hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                          for name in [here.name, 'ce_isometric_color_frame.py']},
        'limits': ['A new supplied connection changes the original action.',
                   'Normal mass uses the fixed old metric; the new normal metric is singular.',
                   'Berry curvature survives; no complete effective action was evaluated.',
                   'P/Q nonmixing holds for compatible free dynamics, not arbitrary interactions.',
                   'No physical gauge completion, autonomous record or Einstein limit is derived.'],
    }
    here.with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'candidate': output['candidate'], 'maximum_identity_error': maximum_error,
                      'temporal': temporal, 'gauge_law': gauge}, indent=2))


if __name__ == '__main__':
    main()
