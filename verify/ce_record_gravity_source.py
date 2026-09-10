"""CE-QG1: complete record energy as a Newton source and a mean-field gate.

The Newton interaction, pinned paths and record controls are supplied inputs.
This finite effective model is not a covariant or quantized gravity theory.
"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from ce_color_covariant_record import HF, HS, HA, REC, setup, unitary, generators


I2, I4, I16 = np.eye(2), np.eye(4), np.eye(16)
SX = np.array([[0., 1.], [1., 0.]])
SY = np.array([[0., -1j], [1j, 0.]])
PLUS = np.ones(2)/np.sqrt(2)
PATHS = [np.diag([1., 0.]), np.diag([0., 1.])]
Y = np.array([-3., 3.])
WS, WA = 1/abs(Y+1), 1/abs(Y-1)
HS16, HA16 = np.kron(HS, I4), np.kron(I4, HA)
R = np.kron(I4, REC)
Z, PHI, TRANSITION = setup()
TOL = 1e-10
GAMMA = np.pi/2


def error(x):
    return float(np.max(abs(x)))


def comm(a, b):
    return a@b-b@a


def expect(state, operator):
    return float(np.vdot(state, operator@state).real)


def density(state):
    return np.outer(state, state.conj())


def probe_trace(rho):
    return np.einsum('aiaj->ij', rho.reshape(16, 2, 16, 2))


def negativity(rho):
    partial_transpose = rho.reshape(16, 2, 16, 2).transpose(0, 3, 2, 1).reshape(32, 32)
    return float(np.sum(np.maximum(-np.linalg.eigvalsh(partial_transpose), 0)))


def potentials(lam):
    local = [-lam*(ws*HS16+wa*HA16) for ws, wa in zip(WS, WA)]
    full = sum(np.kron(v, path) for v, path in zip(local, PATHS))
    branch_z = np.array([expect(Z, v) for v in local])
    branch_phi = np.array([expect(PHI, v) for v in local])
    return local, full, branch_z, branch_phi


def source_audit(lam):
    local, full, vz, vf = potentials(lam)
    errors = {'total_energy_on_z': error(HF@Z-5*Z),
              'total_energy_on_phi': error(HF@PHI-5*PHI),
              'branch_z_potentials': error(vz-lam*np.array([-2.5, -1.25])),
              'branch_phi_potentials': error(vf-lam*np.array([-1.5, -2.25]))}
    for gen in generators():
        gs, ga = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        gs[:3, :3], ga[1:, 1:] = gen, -gen.conj()
        total = np.kron(gs, I4)+np.kron(I4, ga)
        errors['color_invariance'] = max(errors.get('color_invariance', 0),
                                          error(comm(full, np.kron(total, I2))))
    colocated = sum(np.kron(-lam*ws*HF, path) for ws, path in zip(WS, PATHS))
    errors['colocated_record_commutator'] = error(comm(colocated, np.kron(TRANSITION, I2)))
    errors['colocated_branch_identity'] = max(error((-lam*ws*HF)@PHI+5*lam*ws*PHI)
                                                for ws in WS)
    rows = []
    for theta in [np.pi/6, np.pi/4, np.pi/3]:
        state = np.cos(theta)*Z-1j*np.sin(theta)*PHI
        p = np.sin(theta)**2
        es, ea = expect(state, HS16), expect(state, HA16)
        errors['source_energy_bookkeeping'] = max(errors.get('source_energy_bookkeeping', 0),
                                                   abs(es-(5-4*p)), abs(ea-4*p), abs(es+ea-5))
        rows.append({'record_probability': float(p), 'system_only_monopole': es,
                     'apparatus_monopole': ea, 'complete_monopole': es+ea})
    assert max(errors.values()) < TOL
    return {'lambda': lam, 'w_system': WS.tolist(), 'w_apparatus': WA.tolist(),
            'unrecorded_potential': vz.tolist(), 'recorded_potential': vf.tolist(),
            'monopole_comparison': rows, 'errors': errors}


def readout_audit(theta, lam, tau):
    p, t = np.sin(theta)**2, tau/lam
    source = np.cos(theta)*Z-1j*np.sin(theta)*PHI
    initial = np.kron(source, PLUS)
    _, v, vz, vf = potentials(lam)
    h = np.kron(HF, I2)+v
    u = unitary(h, t)
    state = u@initial
    branch_probes = [np.exp(-1j*vb*t)*PLUS for vb in [vz, vf]]
    analytic = np.exp(-5j*t)*(np.cos(theta)*np.kron(Z, branch_probes[0])
                              -1j*np.sin(theta)*np.kron(PHI, branch_probes[1]))
    rho = density(state)
    reduced = probe_trace(rho)
    branch_mixture = (1-p)*density(branch_probes[0])+p*density(branch_probes[1])
    mean_potential = (1-p)*vz+p*vf
    mf_probe = np.exp(-1j*mean_potential*t)*PLUS
    mean_field = density(mf_probe)
    purity = float(np.trace(reduced@reduced).real)
    neg = negativity(rho)
    covariance = expect(state, np.kron(R, SY))-expect(state, np.kron(R, I2))*expect(state, np.kron(I16, SY))
    ys = [expect(q, SY) for q in branch_probes]
    full_record = np.kron(R, I2)
    no_record = np.eye(32)-full_record
    dephased = full_record@rho@full_record+no_record@rho@no_record
    dephased_neg = negativity(dephased)
    kraus = [np.sqrt(1-p)*np.diag(np.exp(-1j*vz*t)), np.sqrt(p)*np.diag(np.exp(-1j*vf*t))]
    choi = sum(density(k.reshape(4, order='F')) for k in kraus)
    channel = sum(k@density(PLUS)@k.conj().T for k in kraus)
    errors = {
        'unitarity': error(u.conj().T@u-np.eye(32)),
        'analytic_state': error(state-analytic),
        'reversibility': error(u.conj().T@state-initial),
        'probe_branch_mixture': error(reduced-branch_mixture),
        'record_probability': abs(expect(state, full_record)-p),
        'complete_energy': abs(expect(state, np.kron(HF, I2))-5),
        'total_H_conservation': abs(expect(state, h)-expect(initial, h)),
        'probe_purity_formula': abs(purity-(1-2*p*(1-p)*np.sin(tau)**2)),
        'negativity_formula': abs(neg-np.sqrt(p*(1-p))*abs(np.sin(tau))),
        'record_probe_covariance_formula': abs(covariance-p*(1-p)*(ys[1]-ys[0])),
        'dephasing_preserves_probe': error(probe_trace(dephased)-reduced),
        'dephasing_preserves_record_joint_X': abs(np.trace((rho-dephased)@np.kron(R, SX))),
        'dephasing_preserves_record_joint_Y': abs(np.trace((rho-dephased)@np.kron(R, SY))),
        'dephased_negativity_zero': dephased_neg,
        'Kraus_completeness': error(sum(k.conj().T@k for k in kraus)-I2),
        'Choi_negative_eigenvalue': float(max(0, -min(np.linalg.eigvalsh(choi)))),
        'reduced_channel': error(channel-reduced),
        'mean_field_purity': abs(np.trace(mean_field@mean_field).real-1),
    }
    for j, projector in enumerate([no_record, full_record]):
        weight = [1-p, p][j]
        conditional = probe_trace(projector@rho@projector)/weight
        errors[f'conditional_probe_{j}'] = error(conditional-density(branch_probes[j]))
    td = float(np.sum(abs(np.linalg.eigvalsh(reduced-mean_field)))/2)
    errors['mean_field_variance_bound'] = float(max(0, td-p*(1-p)*tau**2))
    if tau > 0:
        assert td > 1e-3 and 1-purity > 1e-3
    assert max(errors.values()) < TOL
    return {'theta': float(theta), 'record_probability': float(p), 'lambda': lam,
            'tau': tau, 'time': float(t), 'probe_purity': purity,
            'mean_field_probe_purity': float(np.trace(mean_field@mean_field).real),
            'probe_mean_field_trace_distance': td,
            'mean_field_trace_distance_upper_bound': float(p*(1-p)*tau**2),
            'record_probe_Y_covariance': covariance, 'mean_field_record_probe_Y_covariance': 0.,
            'source_probe_negativity': neg, 'dephased_source_probe_negativity': dephased_neg,
            'branch_probe_Y': ys,
            'variance_of_probe_phase_rate': float(4*p*(1-p)*lam**2),
            'errors': {key: float(value) for key, value in errors.items()}}


def simultaneous_audit(lam, t):
    local, v, vz, vf = potentials(lam)
    record_h = GAMMA*np.kron(TRANSITION, I2)
    free_h = np.kron(HF, I2)
    h = free_h+record_h+v
    initial = np.kron(Z, PLUS)
    u = unitary(h, t)
    state = u@initial
    current = 1j*GAMMA*comm(TRANSITION, HA16)
    current_full = np.kron(current, I2)
    gravity_flow = sum(np.kron(-lam*(wa-ws)*current, path)
                       for ws, wa, path in zip(WS, WA, PATHS))
    errors = {
        'total_H_operator_conservation': error(u.conj().T@h@u-h),
        'free_energy_operator_conservation': error(u.conj().T@free_h@u-free_h),
        'system_current': error(1j*comm(h, np.kron(HS16, I2))+current_full),
        'apparatus_current': error(1j*comm(h, np.kron(HA16, I2))-current_full),
        'gravity_energy_current': error(1j*comm(h, v)-gravity_flow),
        'record_interaction_energy_current': error(1j*comm(h, record_h)+gravity_flow),
    }
    analytic = np.zeros(32, complex)
    conditional = []
    for j, path in enumerate(PATHS):
        delta = vf[j]-vz[j]
        omega = np.sqrt(GAMMA**2+delta**2/4)
        common = np.exp(-1j*(5+(vz[j]+vf[j])/2)*t)
        amp_z = np.cos(omega*t)+1j*delta/(2*omega)*np.sin(omega*t)
        amp_phi = -1j*GAMMA/omega*np.sin(omega*t)
        q = common*(amp_z*Z+amp_phi*PHI)
        analytic += np.kron(q, I2[:, j])/np.sqrt(2)
        q_matrix = state.reshape(16, 2)[:, j]*np.sqrt(2)
        pr = expect(q_matrix, R)
        theory = GAMMA**2/omega**2*np.sin(omega*t)**2
        energy_v, energy_record = expect(q_matrix, local[j]), expect(q_matrix, GAMMA*TRANSITION)
        errors[f'path_{j}_probability'] = abs(np.vdot(q_matrix, q_matrix)-1)
        errors[f'path_{j}_record_formula'] = abs(pr-theory)
        errors[f'path_{j}_gravity_energy'] = abs(energy_v-vz[j]-delta*pr)
        errors[f'path_{j}_record_interaction_energy'] = abs(energy_record+delta*pr)
        errors[f'path_{j}_total_energy'] = abs(expect(q_matrix, HF)+energy_v+energy_record-5-vz[j])
        conditional.append({'path': j, 'detuning': float(delta), 'record_probability': pr,
                            'gravity_energy_change': float(energy_v-vz[j]),
                            'record_interaction_energy': energy_record})
    errors['detuned_state_formula'] = error(state-analytic)
    assert max(errors.values()) < TOL
    return {'lambda': lam, 'time': t, 'record_probability': expect(state, np.kron(R, I2)),
            'no_gravity_record_probability': float(np.sin(GAMMA*t)**2),
            'conditional_path_energy': conditional,
            'gravity_record_commutator_norm': float(np.linalg.norm(comm(v, np.kron(TRANSITION, I2)), 2)),
            'errors': {key: float(value) for key, value in errors.items()}}


def main():
    sources = [source_audit(lam) for lam in [.01, .1]]
    readouts = [readout_audit(theta, lam, tau)
                for theta in [np.pi/6, np.pi/4, np.pi/3]
                for lam in [.01, .1] for tau in [0., .4, 1., np.pi/2]]
    simultaneous = [simultaneous_audit(lam, t)
                    for lam in [0., .01, .1] for t in [0., .25, .5, 1.]]
    maximum = max(max(row['errors'].values()) for row in sources+readouts+simultaneous)
    here = Path(__file__).resolve()
    deps = [here, here.with_name('ce_color_covariant_record.py'), here.with_name('ce_isometric_color_frame.py')]
    out = {'candidate': 'CE-QG1', 'status': 'finite_Newton_source_and_mean_field_gate',
           'inputs': {'Newton_potential_supplied': True, 'system_position': -1., 'apparatus_position': 1.,
                      'probe_positions': Y.tolist(), 'pinned_probe_paths': True, 'gamma': GAMMA,
                      'lambda_definition': 'G*m/(c^2*reference_length)',
                      'units': 'energy E_star; time hbar/E_star; positions reference_length',
                      'tolerance': TOL, 'fit_parameters': 0},
           'source_audits': sources, 'readout_audits': readouts, 'simultaneous_audits': simultaneous,
           'maximum_algebraic_error': maximum,
           'environment': {'python': platform.python_version(), 'numpy': np.__version__},
           'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
           'limits': ['The Newton interaction is assumed, not derived from hidden states or the Einstein action.',
                      'The probe and supports are frozen; this is not a local covariant stress tensor.',
                      'A single deterministic expectation-valued source loses conditional correlations here.',
                      'Dephased source mixtures reproduce the tested record correlations without entanglement.',
                      'Predicted negativity is internal to the quantum potential model, not an experimental gravity witness.',
                      'Energy conservation applies during fixed-H intervals, not unaccounted switching of controls.',
                      'No retarded field, autonomous definite outcome, quantum gravity or joint RMSE is established.']}
    here.with_suffix('.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    example = next(row for row in readouts if row['theta'] == np.pi/4 and row['lambda'] == .1 and row['tau'] == np.pi/2)
    print(json.dumps({'candidate': out['candidate'], 'maximum_algebraic_error': maximum,
                      'readout_cases': len(readouts), 'simultaneous_cases': len(simultaneous),
                      'balanced_record_example': {key: example[key] for key in
                       ['probe_purity', 'mean_field_probe_purity', 'probe_mean_field_trace_distance',
                        'record_probe_Y_covariance', 'source_probe_negativity', 'dephased_source_probe_negativity']},
                      'simultaneous_final_record_probability': simultaneous[-1]['record_probability'],
                      'simultaneous_final_conditional_energies': simultaneous[-1]['conditional_path_energy']}, indent=2))


if __name__ == '__main__':
    main()
