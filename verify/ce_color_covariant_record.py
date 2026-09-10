"""CE-AM4: supplied energy-conserving singlet-to-pair record coupling.

The 16-dimensional dilation is reversible. Pointer conditioning is an input,
and neither physical QCD nor a unique observed outcome is derived here.
"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from ce_isometric_color_frame import KS, Q0, comm


TOL = 1e-12
I4 = np.eye(4, dtype=complex)
I16 = np.eye(16, dtype=complex)
P0 = I4 - Q0
HS = P0 + 5 * Q0
REC = np.diag([0., 1., 1., 1.])
HA = 4 * REC
HF = np.kron(HS, I4) + np.kron(I4, HA)
GAMMA = np.pi / 2


def error(a):
    return float(np.max(np.abs(a)))


def unitary(h, t):
    values, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-1j * t * values)) @ vectors.conj().T


def generators():
    result = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        x = np.zeros((3, 3), complex)
        x[a, b] = x[b, a] = .5
        y = np.zeros((3, 3), complex)
        y[a, b], y[b, a] = -.5j, .5j
        result.extend((x, y))
    result.extend((np.diag([1., -1., 0.]) / 2,
                   np.diag([1., 1., -2.]) / (2 * np.sqrt(3.))))
    return result


def setup():
    initial = np.zeros(16, complex)
    initial[12] = 1.
    phi = np.zeros(16, complex)
    for i in range(3):
        phi[4 * i + i + 1] = 1 / np.sqrt(3.)
    transition = np.outer(phi, initial.conj()) + np.outer(initial, phi.conj())
    assert error(comm(HF, transition)) < TOL
    assert error(HF @ initial - 5 * initial) < TOL
    assert error(HF @ phi - 5 * phi) < TOL
    return initial, phi, transition


def symmetry_audit(transition):
    errors = []
    system_generators = []
    for g in generators():
        gs, ga = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        gs[:3, :3] = g
        ga[1:, 1:] = -g.conj()
        total = np.kron(gs, I4) + np.kron(I4, ga)
        errors.extend((error(comm(total, transition)), error(comm(total, HF)),
                       error(comm(total, np.kron(I4, REC)))))
        system_generators.append(gs)
    casimir = sum(g @ g for g in generators())
    errors.append(error(casimir - 4 * np.eye(3) / 3))
    assert max(errors) < TOL
    return system_generators, {'generators_checked': 8,
                               'maximum_symmetry_error': max(errors),
                               'fundamental_casimir_eigenvalues': np.linalg.eigvalsh(casimir).tolist(),
                               'singlet_apparatus_invariant_transition_dimension': 0}


def instrument_audit(t, transition, system_generators):
    theta = GAMMA * t
    evolution = unitary(transition, theta)
    direct = I16 + (np.cos(theta) - 1) * (transition @ transition) - 1j * np.sin(theta) * transition
    kraus = [evolution.reshape(4, 4, 4, 4)[:, outcome, :, 0] for outcome in range(4)]
    expected = [P0 + np.cos(theta) * Q0]
    for i in range(3):
        k = np.zeros((4, 4), complex)
        k[i, 3] = -1j * np.sin(theta) / np.sqrt(3.)
        expected.append(k)
    choi = sum(np.outer(k.reshape(-1, order='F'), k.reshape(-1, order='F').conj()) for k in kraus)
    eigenvalues = np.linalg.eigvalsh(choi)
    errors = {'exponential_vs_closed_form': error(evolution - direct),
              'kraus_vs_formula': max(error(k - e) for k, e in zip(kraus, expected)),
              'kraus_completeness': error(sum(k.conj().T @ k for k in kraus) - I4),
              'choi_trace': abs(float(np.trace(choi).real) - 4.)}
    ward_errors = []
    for subset in (kraus[:1], kraus[1:]):
        def channel(rho):
            return sum(k @ rho @ k.conj().T for k in subset)
        for g in system_generators:
            for a in range(4):
                for b in range(4):
                    basis = np.zeros((4, 4), complex)
                    basis[a, b] = 1.
                    ward_errors.append(error(channel(comm(g, basis)) - comm(g, channel(basis))))
    errors['all_state_branch_covariance'] = max(ward_errors)
    assert max(errors.values()) < TOL, errors
    assert min(eigenvalues) > -TOL
    return {'t': t, 'theta': theta, 'errors': errors,
            'minimum_choi_eigenvalue': float(min(eigenvalues)),
            'choi_rank_at_tolerance': int(np.linalg.matrix_rank(choi, tol=TOL)),
            'branch_covariance_basis_checks': len(ward_errors)}


def moving_audit(t, initial, transition):
    g = KS[0]
    u = np.cos(t) * I4 + 1j * np.sin(t) * g
    p = u @ P0 @ u.conj().T
    dp = 1j * comm(g, p)
    k = comm(dp, p)
    g_diag = P0 @ g @ P0 + Q0 @ g @ Q0
    w = u @ unitary(g_diag, t)
    w_dot = 1j * g @ w - 1j * u @ g_diag @ unitary(g_diag, t)
    wt, kt = np.kron(w, I4), np.kron(k, I4)
    moving_hf = wt @ HF @ wt.conj().T
    moving_int = wt @ (GAMMA * transition) @ wt.conj().T
    rotated_evolution = unitary(HF + GAMMA * transition, t)
    evolution = wt @ rotated_evolution
    evolution_dot = np.kron(w_dot, I4) @ rotated_evolution - 1j * wt @ (HF + GAMMA * transition) @ rotated_evolution
    physical_h = moving_hf + moving_int + 1j * kt
    hf_dot = -4 * np.kron(dp, I4)
    psi = evolution @ initial
    record_prob = float(np.vdot(psi, np.kron(I4, REC) @ psi).real)
    light_prob = float(np.vdot(psi, np.kron(p, I4) @ psi).real)
    system_energy = float(np.vdot(psi, np.kron(p + 5 * (I4 - p), I4) @ psi).real)
    apparatus_energy = float(np.vdot(psi, np.kron(I4, HA) @ psi).real)
    expected_prob = float(np.sin(GAMMA * t)**2)
    errors = {
        'kato_equation': error(w_dot - k @ w),
        'evolution_equation': error(1j * evolution_dot - physical_h @ evolution),
        'unitarity': error(evolution.conj().T @ evolution - I16),
        'all_state_free_energy_conservation': error(hf_dot + 1j * comm(physical_h, moving_hf)),
        'all_state_free_energy_intertwining': error(evolution.conj().T @ moving_hf @ evolution - HF),
        'record_probability': abs(record_prob - expected_prob),
        'light_probability': abs(light_prob - expected_prob),
        'system_energy': abs(system_energy - (5 - 4 * expected_prob)),
        'apparatus_energy': abs(apparatus_energy - 4 * expected_prob),
        'energy_sum': abs(system_energy + apparatus_energy - 5.),
    }
    assert max(errors.values()) < TOL, errors
    return {'t': t, 'record_probability': record_prob, 'light_probability': light_prob,
            'system_energy': system_energy, 'apparatus_energy': apparatus_energy,
            'total_free_energy': system_energy + apparatus_energy, 'errors': errors}


def main():
    initial, phi, transition = setup()
    system_generators, symmetry = symmetry_audit(transition)
    instruments = [instrument_audit(t, transition, system_generators) for t in (0., .25, .5, 1.)]
    moving = [moving_audit(t, initial, transition) for t in (0., .25, .5, 1.)]
    maximum_error = max(symmetry['maximum_symmetry_error'],
                        *(max(row['errors'].values()) for row in instruments + moving))
    here = Path(__file__).resolve()
    output = {'candidate': 'CE-AM4', 'tolerance': TOL,
              'inputs': {'system_energies': [1., 5.], 'apparatus_record_energy': 4.,
                         'gamma': GAMMA, 'dimensions': [4, 4], 'fit_parameters': 0},
              'symmetry': symmetry, 'instruments': instruments, 'moving_record': moving,
              'maximum_identity_error': maximum_error,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__},
              'source_sha256': {name: hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                                for name in [here.name, 'ce_isometric_color_frame.py']},
              'limits': ['Supplied SU(3) representations and finite interaction, not QCD dynamics.',
                         'Coarse record is invariant; individual color labels are not observables here.',
                         'The full dilation is reversible and the pulse/control is prescribed.',
                         'No unique outcome, spacetime locality, asynchronous event algebra or gravity is derived.']}
    here.with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'candidate': output['candidate'], 'maximum_identity_error': maximum_error,
                      'symmetry': symmetry, 'moving_record': moving}, indent=2))


if __name__ == '__main__':
    main()
