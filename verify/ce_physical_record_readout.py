"""CE-EV2: physical reader memories, Bell records, and explicit pulse work.

The LM3-G model is unchanged. The extra reader schedule is externally supplied.
Only the exact two-state calibration samples paths; the full model tests currents.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from ce_autonomous_record_instrument import build


def mean(operator, psi):
    return float(np.vdot(psi, operator @ psi).real)


def joint(psi):
    return np.sum(abs(psi.reshape(-1, 2, 2)) ** 2, axis=0)


def current_audit(hamiltonian, psi):
    matrix = hamiltonian.tocoo()
    select = matrix.row != matrix.col
    row, col = matrix.row[select], matrix.col[select]
    value = matrix.data[select]
    p = abs(psi) ** 2
    current = 2 * np.imag(psi[row].conj() * value * psi[col])
    flux = np.maximum(current, 0.)
    assert np.all(flux[p[col] == 0.] == 0.)
    rates = np.divide(flux, p[col], out=np.zeros_like(flux), where=p[col] > 0.)
    master = (np.bincount(row, weights=rates*p[col], minlength=len(psi))
              - np.bincount(col, weights=rates*p[col], minlength=len(psi)))
    schrodinger = 2 * np.real(psi.conj() * (-1j * (hamiltonian @ psi)))
    error = float(np.max(abs(master - schrodinger)))
    bound = float(np.max(np.bincount(col, weights=abs(value), minlength=len(psi))))
    assert error < 1e-10 and abs(p.sum()-1) < 1e-9
    assert float(flux.sum()) <= bound + 1e-9
    return dict(equivariance_max_error=error, expected_jump_intensity=float(flux.sum()),
                expected_intensity_bound=bound,
                memory_changing_directed_edges=int(np.sum(row % 4 != col % 4)))


def reader_kraus(hamiltonian, pointer_one, psi, tau):
    """Independent X-memory block construction, without a four-memory-space H."""
    plus = expm_multiply(-1j*tau*hamiltonian, psi)
    minus = expm_multiply(-1j*(tau*hamiltonian + np.pi*pointer_one), psi)
    branches = ((plus+minus)/2, (plus-minus)/2)
    assert abs(sum(np.vdot(v, v).real for v in branches) - np.vdot(psi, psi).real) < 1e-9
    return branches


def independent_two_reads(hamiltonian, pointer_one, prepared_at_five, tau):
    first = reader_kraus(hamiltonian, pointer_one, prepared_at_five, tau)
    final = np.empty((len(prepared_at_five), 2, 2), complex)
    for a, branch in enumerate(first):
        branch = expm_multiply(-1j*(5.-tau)*hamiltonian, branch)
        for b, output in enumerate(reader_kraus(hamiltonian, pointer_one, branch, tau)):
            final[:, a, b] = output
    return final.reshape(-1)


def ideal_control(hamiltonian, labels, prepared_at_five):
    dimension = len(prepared_at_five)
    first = np.zeros((dimension, 2), complex)
    for a in (0, 1):
        first[labels == a, a] = prepared_at_five[labels == a]
    work_one = sum(mean(hamiltonian, first[:, a]) for a in (0, 1)) - mean(hamiltonian, prepared_at_five)
    later = expm_multiply(-5j*hamiltonian, first)
    final = np.zeros((dimension, 2, 2), complex)
    for b in (0, 1):
        final[labels == b, :, b] = later[labels == b, :]
    work_two = (sum(mean(hamiltonian, final[:, a, b]) for a in (0, 1) for b in (0, 1))
                - sum(mean(hamiltonian, later[:, a]) for a in (0, 1)))
    probabilities = joint(final.reshape(-1))
    receipt = json.loads(Path(__file__).with_name('ce_autonomous_record_instrument.json').read_text(encoding='utf-8'))
    earlier = next(case for case in receipt['cases'] if case['apparatus_gap'] == 1.)
    old_joint = next(row for row in earlier['rows'] if row['time'] == 10.)['sequential_joint_probabilities']
    error = float(np.max(abs(probabilities-np.array(old_joint))))
    assert error < 1e-9 and abs(work_one-earlier['first_read_energy']['work']) < 1e-9
    return dict(joint_probabilities=probabilities.tolist(), previous_instrument_error=error,
                first_read_work=work_one, second_read_work=work_two,
                status='instantaneous_CNOT_control_not_a_finite_pulse')


def full_case(hamiltonian, pointer_one, prepared_at_five, initial_energy, tau, ideal):
    free = sparse.kron(hamiltonian, sparse.eye(4), format='csr')
    psi = np.kron(prepared_at_five, [1., 0., 0., 0.])
    flip = np.array([[0., 1.], [1., 0.]])
    memory_x = (np.kron(flip, np.eye(2)), np.kron(np.eye(2), flip))
    total_work = 0.
    pulses = []
    for register in (0, 1):
        if register:
            psi = expm_multiply(-1j*(5.-tau)*free, psi)
        coupling = np.pi/(2*tau)*sparse.kron(pointer_one, np.eye(4)-memory_x[register], format='csr')
        active = free+coupling
        before_free = mean(free, psi)
        work_on = mean(coupling, psi)
        active_energy = mean(active, psi)
        midpoint = expm_multiply(-.5j*tau*active, psi)
        audit = current_audit(active, midpoint)
        psi = expm_multiply(-.5j*tau*active, midpoint)
        active_error = abs(mean(active, psi)-active_energy)
        work_off = -mean(coupling, psi)
        work = work_on+work_off
        ledger_error = abs(mean(free, psi)-before_free-work)
        assert active_error < 1e-9 and ledger_error < 1e-9
        total_work += work
        pulses.append(dict(register=register, start=5.+5*register, duration=tau,
                           work_on=work_on, work_off=work_off, work=work,
                           active_energy_error=active_error, work_ledger_error=ledger_error,
                           midpoint_current=audit))
    end = psi.copy()
    reference = independent_two_reads(hamiltonian, pointer_one, prepared_at_five, tau)
    independent_error = float(np.max(abs(end-reference)))
    assert independent_error < 1e-9
    end_joint = joint(end)
    psi = expm_multiply(-1j*(10.-tau)*free, end)
    stored_joint = joint(psi)
    storage_error = float(np.max(abs(stored_joint-end_joint)))
    ledger_error = abs(mean(free, psi)-initial_energy-total_work)
    audit = current_audit(free, psi)
    assert storage_error < 1e-9 and ledger_error < 1e-9
    assert audit['memory_changing_directed_edges'] == 0
    coherence = end.reshape(-1, 4).conj().T @ end.reshape(-1, 4)
    offdiag = coherence-np.diag(np.diag(coherence))
    return dict(pulse_duration=tau, coupling=np.pi/(2*tau), pulses=pulses,
                joint_probabilities=end_joint.tolist(), stored_joint_probabilities_at_twenty=stored_joint.tolist(),
                record_disagreement_probability=float(end_joint[0, 1]+end_joint[1, 0]),
                distance_from_instantaneous_joint=float(np.max(abs(end_joint-np.array(ideal['joint_probabilities'])))),
                independent_X_block_amplitude_error=independent_error,
                storage_probability_error=storage_error, total_external_work=total_work,
                base_energy_at_twenty=mean(free, psi), total_work_ledger_error=ledger_error,
                reduced_memory_offdiagonal_max=float(np.max(abs(offdiag))),
                storage_current=audit)


def exact_path_control():
    """Exact minimal Bell jump samples up to pi/2; no discretized waiting steps."""
    t_one, t_two = np.pi/6, np.pi/3
    rng = np.random.default_rng(9110)
    count = 65536
    first_jump = np.arcsin(np.sqrt(rng.random(count)))
    a, b = (first_jump <= t_one).astype(int), (first_jump <= t_two).astype(int)
    empirical = np.bincount(2*a+b, minlength=4).reshape(2, 2)/count
    hidden = np.array([[np.cos(t_two)**2, np.sin(t_two)**2-np.sin(t_one)**2],
                       [0., np.sin(t_one)**2]])
    flip = np.array([[0., 1.], [1., 0.]])
    psi = expm(-1j*t_one*flip) @ np.array([1., 0.])
    first_memory = np.zeros((2, 2), complex)
    first_memory[0, 0], first_memory[1, 1] = psi
    later = expm(-1j*(t_two-t_one)*flip) @ first_memory
    all_records = np.zeros((2, 2, 2), complex)
    all_records[0, :, 0], all_records[1, :, 1] = later[0, :], later[1, :]
    physical = np.sum(abs(all_records)**2, axis=0)
    p_one = np.array([np.cos(t_one)**2, np.sin(t_one)**2])
    transition = np.array([[np.cos(t_two-t_one)**2, np.sin(t_two-t_one)**2],
                           [np.sin(t_two-t_one)**2, np.cos(t_two-t_one)**2]])
    projected = p_one[:, None]*transition
    error = float(np.max(abs(empirical-hidden)))
    assert error < .01 and np.max(abs(physical-projected)) < 1e-12
    assert abs(hidden[0, 1]+hidden[1, 0]-.5) < 1e-12
    assert abs(projected[0, 1]+projected[1, 0]-.25) < 1e-12
    return dict(status='exact_two_state_method_and_history_counterexample_only',
                times=[t_one, t_two], seed=9110, sampled_trajectories=count,
                hidden_path_joint=hidden.tolist(), sampled_hidden_joint=empirical.tolist(),
                maximum_sampling_error=error, preregistered_sampling_tolerance=.01,
                sequential_projective_joint=projected.tolist(), physical_memory_joint=physical.tolist(),
                physical_projective_max_error=float(np.max(abs(physical-projected))),
                hidden_history_flip_probability=.5, physically_read_flip_probability=.25)


if __name__ == '__main__':
    h, _, _, _, _, _, _, ground, packet, _, _ = build(1.)
    initial = np.kron(np.kron(ground, [1., 0.]), packet)
    prepared = expm_multiply(-5j*h, initial)
    labels = (np.arange(len(initial))//17) % 2
    pointer = sparse.diags(labels, format='csr', dtype=float)
    ideal = ideal_control(h, labels, prepared)
    cases = []
    for tau in (1., .5, .25):
        result = full_case(h, pointer, prepared, mean(h, initial), tau, ideal)
        cases.append(result)
        print(json.dumps(dict(tau=tau, independent_error=result['independent_X_block_amplitude_error'],
                              work=result['total_external_work'], joint=result['joint_probabilities'])), flush=True)
    here = Path(__file__).resolve()
    result = dict(candidate='CE-EV2', status='conditional_records_with_external_reader_pulses',
                  base_model='CE-LM3-G', configuration_count=4*len(initial),
                  base_initial_energy=mean(h, initial), cases=cases, ideal_control=ideal,
                  exact_two_state_path_control=exact_path_control(),
                  full_model_sampled_trajectories=0, fitted_parameters=0,
                  full_joint_rmse=None, scientific_success=False,
                  assumptions=['The EV1 configuration and initial Born axioms are retained.',
                               'Two initially pure degenerate memories and prescribed read pulses are supplied.',
                               'Memory couplings vanish exactly after their read pulses.'],
                  limitations=['No autonomous read scheduler or preparation/reset cost is derived.',
                               'Stable stored bits do not correct LM3 initial-occupation detector errors.',
                               'Hidden trajectories without readers are not sequential measurement data.',
                               'No gravitational backreaction, relativistic locality, gauge, or empirical fit.'])
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (here, here.with_name('ce_autonomous_record_instrument.py'), here.with_name('ce_local_vacuum_modular.py'))}
    receipt = here.with_name('ce_autonomous_record_instrument.json')
    result['input_sha256'] = {receipt.name: hashlib.sha256(receipt.read_bytes()).hexdigest()}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['exact_two_state_path_control'], indent=2))
