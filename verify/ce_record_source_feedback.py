"""Feedback consistency gate in the exact CE-QG1 record/probe sector.

All observations below are model calculations. No experimental Bell test or
covariant gravity model is constructed. Bell histories are integrated exactly
on monotone two-state rotation intervals, not resampled at a measurement.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import expm

from ce_record_gravity_source import HF, Z as RECORD_ZERO, PHI, potentials, unitary


I = np.eye(2)
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
N = np.diag([0., 1.])
NA, NB = np.kron(N, I), np.kron(I, N)
INITIAL = np.ones(4)/2
TOL = 1e-10
SETTINGS = {'z_b0': (0., np.pi/4), 'z_b1': (0., 3*np.pi/4),
            'x_b0': (np.pi/2, np.pi/4), 'x_b1': (np.pi/2, 3*np.pi/4),
            'xz': (np.pi/2, 0.), 'zx': (0., np.pi/2)}


def error(value):
    return float(np.max(abs(value)))


def density(psi):
    return np.outer(psi, psi.conj())


def mean(operator, psi):
    return float(np.vdot(psi, operator@psi).real)


def rotation(theta):
    return np.array([[np.cos(theta/2), np.sin(theta/2)],
                     [-np.sin(theta/2), np.cos(theta/2)]])


def monotone_transport(p_start, p_end):
    """Integrated minimal-current transition matrix; columns are input labels."""
    if p_end >= p_start:
        fraction = (p_end-p_start)/(1-p_start) if p_start < 1 else 0.
        matrix = np.array([[1., fraction], [0., 1-fraction]])
    else:
        fraction = (p_start-p_end)/p_start
        matrix = np.array([[1-fraction, 0.], [fraction, 1.]])
    assert matrix.min() >= -1e-13 and error(matrix.sum(axis=0)-1) < 1e-13
    return matrix


def bell_rotation(psi, distribution, site, angle):
    """Exact local Bell transport, retaining a possibly non-Born input Q law.

For a fixed other bit, p0(s)=w/2+A cos(s)+B sin(s). Its derivative
vanishes at atan2(B,A)+k*pi. Segmenting there retains current reversals.
"""
    result = distribution.copy()
    pairs = [(0, 2), (1, 3)] if site == 0 else [(0, 1), (2, 3)]
    for indices in pairs:
        index = list(indices)
        pair = psi[index]
        weight = float(np.vdot(pair, pair).real)
        if weight == 0.:
            assert error(distribution[index]) < 1e-13
            continue
        a = (abs(pair[0])**2-abs(pair[1])**2)/2
        b = float((pair[0].conj()*pair[1]).real)
        stationary = np.arctan2(b, a)
        cuts = [0.]+sorted(s for k in range(-2, 3)
                           if 0. < (s := stationary+k*np.pi) < angle)+[angle]
        mass = distribution[index].copy()
        for left, right in zip(cuts[:-1], cuts[1:]):
            first, last = rotation(left)@pair, rotation(right)@pair
            p_start = float(abs(first[0])**2/np.vdot(first, first).real)
            p_end = float(abs(last[0])**2/np.vdot(last, last).real)
            mass = monotone_transport(p_start, p_end)@mass
        result[index] = mass
    change = np.kron(rotation(angle), I) if site == 0 else np.kron(I, rotation(angle))
    assert abs(result.sum()-distribution.sum()) < TOL and result.min() > -TOL
    return change@psi, result


def readout(psi, distribution, angles):
    state, law = bell_rotation(psi, distribution, 0, angles[0])
    state, law = bell_rotation(state, law, 1, angles[1])
    correlation = float(np.dot([1., -1., -1., 1.], law))
    return correlation, law, state


def correlations(rho):
    result = {}
    for name, (a, b) in SETTINGS.items():
        obs_a, obs_b = np.cos(a)*Z+np.sin(a)*X, np.cos(b)*Z+np.sin(b)*X
        result[name] = float(np.trace(rho@np.kron(obs_a, obs_b)).real)
    return result


def chsh(values):
    return values['z_b0']+values['z_b1']+values['x_b0']-values['x_b1']


def marginal_audit(joints):
    pairs = [('z_b0', 'z_b1', 1), ('x_b0', 'x_b1', 1),
             ('z_b0', 'x_b0', 0), ('z_b1', 'x_b1', 0)]
    mismatch = max(error(joints[a].sum(axis=axis)-joints[b].sum(axis=axis)) for a, b, axis in pairs)
    assert mismatch < TOL
    return dict(post_interaction_local_setting_marginal_error=mismatch,
                scope='Independent read rotations after the interaction is off; not relativistic propagation.')


def quantum_joints(rho):
    result = {}
    for name, (a, b) in SETTINGS.items():
        u = np.kron(rotation(a), rotation(b))
        result[name] = np.diag(u@rho@u.conj().T).real.reshape(2, 2)
    return result


def choi_audit(kraus):
    choi = sum(density(k.reshape(-1, order='F')) for k in kraus)
    eigen = np.linalg.eigvalsh(choi)
    completeness = error(sum(k.conj().T@k for k in kraus)-np.eye(4))
    assert completeness < TOL and eigen.min() > -TOL
    return dict(completeness_error=completeness, minimum_choi_eigenvalue=float(eigen.min()),
                choi_trace=float(np.trace(choi).real))


def negativity(rho):
    pt = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    return float(np.sum(np.maximum(-np.linalg.eigvalsh(pt), 0.)))


def operator_source(lam, time):
    _, potential, _, _ = potentials(lam)
    embed = np.kron(np.column_stack((RECORD_ZERO, PHI)), I)
    full_h = np.kron(HF, I)+potential
    effective = embed.conj().T@full_h@embed
    local = (5-2.5*lam)*np.eye(4)+lam*NA+1.25*lam*NB
    g = -2*lam
    target_h = local+g*NA@NB
    source_errors = dict(isometry=error(embed.conj().T@embed-np.eye(4)),
                         invariant_sector=error(full_h@embed-embed@effective),
                         exact_potential_restriction=error(effective-target_h))
    full_u = unitary(full_h, time)
    effective_u = expm(-1j*time*effective)
    source_errors['independent_32_vs_4_state'] = error(full_u@embed@INITIAL-embed@effective_u@INITIAL)
    source_errors['full_H_operator_conservation'] = error(full_u.conj().T@full_h@full_u-full_h)
    compensated = expm(1j*time*local)@embed.conj().T@full_u@embed@INITIAL
    interaction_u = expm(-1j*time*g*NA@NB)
    state = interaction_u@INITIAL
    graph = np.array([1., 1., 1., -1.])/2
    source_errors['local_phase_compensation'] = error(compensated-state)
    source_errors['graph_state'] = error(state-graph)
    source_errors['interaction_energy'] = abs(mean(g*NA@NB, state)-g/4)
    assert max(source_errors.values()) < TOL
    observed = correlations(density(state))
    bell_errors = []
    for angles in SETTINGS.values():
        _, law, after = readout(state, abs(state)**2, angles)
        bell_errors.append(error(law-abs(after)**2))
    assert max(bell_errors) < TOL and abs(chsh(observed)-2*np.sqrt(2)) < TOL
    return state, dict(status='same_QG1_operator_source_invariant_sector', coupling=g,
                       interaction_time=time, source_errors=source_errors,
                       correlations=observed, chsh=chsh(observed), negativity=negativity(density(state)),
                       quantum_Bell_transport_max_error=max(bell_errors),
                       readout_marginals=marginal_audit(quantum_joints(density(state))),
                       channel=choi_audit([interaction_u]))


def actual_bit_feedback(g, time, operator_state):
    branches = []
    averaged_wave_density = np.zeros((4, 4), complex)
    joints = {name: np.zeros((2, 2)) for name in SETTINGS}
    for a in (0, 1):
        for b in (0, 1):
            h = g*(b*NA+a*NB-a*b*np.eye(4))
            psi = expm(-1j*time*h)@INITIAL
            actual = np.zeros(4)
            actual[2*a+b] = 1.
            expected = (-1)**(a*b)*np.kron([1., (-1)**b], [1., (-1)**a])/2
            energy_error = abs(mean(h, psi)-mean(h, INITIAL))
            conditional_tv = float(np.sum(abs(actual-abs(psi)**2))/2)
            assert error(psi-expected) < TOL and energy_error < TOL
            assert abs(conditional_tv-.75) < TOL
            branch_observed = {}
            for name, angles in SETTINGS.items():
                correlation, law, _ = readout(psi, actual, angles)
                branch_observed[name] = correlation
                joints[name] += law.reshape(2, 2)/4
            averaged_wave_density += density(psi)/4
            branches.append(dict(actual_bits=[a, b], energy_before=mean(h, INITIAL),
                                 energy_error=energy_error, norm_error=abs(np.vdot(psi, psi).real-1),
                                 conditional_Born_total_variation=conditional_tv,
                                 correlations=branch_observed))
    observed = {name: sum(branch['correlations'][name] for branch in branches)/4 for name in SETTINGS}
    oracle = dict(z_b0=1/np.sqrt(2), z_b1=1/np.sqrt(2),
                  x_b0=1-1/np.sqrt(2), x_b1=0., xz=1., zx=1.)
    oracle_error = max(abs(observed[name]-oracle[name]) for name in SETTINGS)
    assert oracle_error < TOL
    assert error(averaged_wave_density-np.eye(4)/4) < TOL
    k1, k2 = np.kron(X, Z), np.kron(Z, X)
    forced_density = (np.eye(4)+k1)@(np.eye(4)+k2)/4
    assert error(forced_density-density(operator_state)) < TOL
    forced_observed = correlations(forced_density)
    mismatch = max(abs(observed[name]-forced_observed[name]) for name in SETTINGS)
    assert mismatch > .1 and abs(chsh(observed)-(1+1/np.sqrt(2))) < TOL
    return dict(candidate='CE-FB1', status='rejected_conditional_Born_and_quantum_observation_gate',
                branches=branches, correlations=observed, chsh=chsh(observed),
                closed_form_correlation_error=oracle_error,
                positive_density_representation_max_mismatch=mismatch,
                averaged_wave_density_is_maximally_mixed_error=error(averaged_wave_density-np.eye(4)/4),
                averaged_branch_energy=sum(branch['energy_before'] for branch in branches)/4,
                readout_marginals=marginal_audit(joints),
                reason='XZ=ZX=1 forces one quantum state, but another fixed setting disagrees with that state.',
                sampled_trajectories=0, method='exact_piecewise_monotone_Bell_transport_no_Born_resampling')


def physical_measurement_feedback(g, time):
    probe_u = expm(-1j*g*time*N)
    kraus = [np.kron(I-N, I), np.kron(N, probe_u)]
    rho = sum(k@density(INITIAL)@k.conj().T for k in kraus)
    copy_gate = np.kron(np.kron(I-N, I), I)+np.kron(np.kron(N, I), X)
    feedback_h = g*np.kron(np.kron(I, N), N)
    dilation = expm(-1j*time*feedback_h)@copy_gate
    iso = dilation@np.kron(np.eye(4), np.array([[1.], [0.]]))
    expanded = iso@INITIAL
    reduced = expanded.reshape(4, 2)@expanded.reshape(4, 2).conj().T
    channel_error = max(error(iso.reshape(4, 2, 4)[:, a, :]-kraus[a]) for a in (0, 1))
    assert error(reduced-rho) < TOL and channel_error < TOL
    observed = correlations(rho)
    assert abs(chsh(observed)-np.sqrt(2)) < TOL and negativity(rho) < TOL
    return dict(candidate='CE-FB2', status='CPTP_measurement_feedback_with_explicit_dephasing',
                channel=choi_audit(kraus), full_dilation_dimension=8,
                dilation_channel_max_error=channel_error, reduced_density_error=error(reduced-rho),
                full_state_purity=float(np.trace(density(expanded)@density(expanded)).real),
                reduced_state_purity=float(np.trace(rho@rho).real),
                correlations=observed, chsh=chsh(observed), negativity=negativity(rho),
                readout_marginals=marginal_audit(quantum_joints(rho)),
                limitations=['A physical measurement record is added, rather than hidden-bit substitution.',
                             'Discarded interference is retained in the full quantum memory dilation.',
                             'This is a finite channel, not a covariant classical-gravity dynamics.'])


if __name__ == '__main__':
    lam = .1
    time = np.pi/(2*lam)
    state, operator = operator_source(lam, time)
    naive = actual_bit_feedback(-2*lam, time, state)
    alternative = (physical_measurement_feedback(-2*lam, time)
                   if naive['status'].startswith('rejected') else None)
    result = dict(test='CE_record_source_feedback', source_model='CE-QG1_exact_logical_sector',
                  source_lambda=lam, operator_source=operator, actual_bit_feedback=naive,
                  physical_measurement_feedback=alternative, fitted_parameters=0,
                  full_joint_rmse=None, scientific_success=False,
                  limitations=['The configuration basis, coherent preparations and local analysis rotations are inputs.',
                               'The QG1 Newton kernel and pinned positions are supplied; no retarded metric or stress is derived.',
                               'No unification with the LM3 Hamiltonian or the three gauge sectors is claimed.',
                               'CHSH values are model calculations, not experimental observations.',
                               'No universal no-go for all classical, stochastic or hybrid gravitational theories.'])
    here = Path(__file__).resolve()
    sources = {here}
    for name, module in list(sys.modules.items()):
        if name.startswith('ce_') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if path.parent == here.parent:
                sources.add(path)
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(operator=operator, naive_feedback=naive,
                          measurement_feedback=alternative), indent=2))
