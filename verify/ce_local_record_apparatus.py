"""CE-LM2: explicit local apparatus, fixed-H energy currents and switching work.

The coupled ground reference is computed in the preregistered N_particle=4
sector. It does not replace the prepared state or certify an actual outcome.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from ce_local_vacuum_modular import annihilators, second_quantize, reduced_density, max_error, comm, entropy


N = 8
D = 1 << N
TOL = 1e-10
PAULI_X = np.array([[0., 1.], [1., 0.]])
PAULI_Y = np.array([[0., -1j], [1j, 0.]])
RECORD = np.diag([0., 1.])


def expectation(psi, operator):
    return float(np.vdot(psi, operator@psi).real)


def gaussian_density(correlation):
    vals, modes = np.linalg.eigh(correlation)
    assert vals.min() > 0 and vals.max() < 1
    one = (modes*np.log((1-vals)/vals))@modes.conj().T
    h = second_quantize(one, annihilators(len(correlation))).toarray()
    hv, hu = np.linalg.eigh(h)
    weights = np.exp(-(hv-hv.min()))
    return (hu*(weights/weights.sum()))@hu.conj().T


def coupled_reference(full_h, number_operators, c, gamma, gap, prepared):
    sector = [i for i in range(2*D) if (i >> 1).bit_count() == 4]
    restricted = full_h[np.ix_(sector, sector)]
    vals, modes = np.linalg.eigh(restricted)
    psi = np.zeros(2*D, complex)
    psi[sector] = modes[:, 0]
    rho = np.outer(psi, psi.conj())
    block = reduced_density(rho, [2, 3, 8], total=9)
    rv, ru = np.linalg.eigh(block)
    assert rv.min() > -TOL
    positive = rv > 1e-12
    support_log = (ru[:, positive]*(-np.log(rv[positive])))@ru[:, positive].conj().T
    prepared_region = reduced_density(np.outer(prepared, prepared.conj()), [2, 3, 8], total=9)
    outside_support = ru[:, ~positive]@ru[:, ~positive].conj().T
    leakage = float(np.trace(prepared_region@outside_support).real)
    relative = None if leakage > TOL else float(np.trace(prepared_region@support_log).real-entropy(prepared_region))
    if relative is not None:
        assert relative > -TOL
    reconstructed = (ru[:, positive]*np.exp(np.log(rv[positive])))@ru[:, positive].conj().T
    errors = dict(sector_eigenstate=max_error(full_h@psi-vals[0]*psi),
                  block_trace=abs(float(np.trace(block).real)-1),
                  support_reconstruction=max_error(block-reconstructed))
    assert max(errors.values()) < TOL
    free_control = None
    if gap == 0:
        one = -np.diag(np.ones(N-1), 1)-np.diag(np.ones(N-1), -1)
        one[3, 3] -= gamma
        independent_energy = float(np.linalg.eigvalsh(one)[:4].sum())
        pointer_x = expectation(psi, sparse.kron(sparse.eye(D), PAULI_X))
        energy_error = abs(float(vals[0])-independent_energy)
        assert energy_error < TOL and abs(pointer_x+1) < TOL
        free_control = dict(conditional_free_energy=independent_energy,
                            energy_error=energy_error, pointer_X=pointer_x)
    region = [2, 3]
    correlator = np.array([[expectation(psi, sparse.kron(c[i].T@c[j], sparse.eye(2)))
                            for j in region] for i in region])
    actual = expectation(psi, sparse.kron(number_operators[2]@number_operators[3], sparse.eye(2)))
    wick = actual-correlator[0, 0]*correlator[1, 1]+abs(correlator[0, 1])**2
    matter_block = reduced_density(rho, region, total=9)
    gaussian = gaussian_density(correlator)
    return dict(particle_number_sector=4, sector_dimension=len(sector),
                ground_energy=float(vals[0]), sector_gap=float(vals[1]-vals[0]),
                region_matter_plus_apparatus=[2, 3, 8], region_eigenvalues=rv.tolist(),
                support_rank=int(positive.sum()), rank_threshold=1e-12,
                finite_log_defined_only_on_support=True,
                prepared_weight_outside_reference_support=leakage,
                prepared_relative_entropy=relative,
                relative_entropy_status='infinite_support_mismatch' if relative is None else 'finite',
                support_modular_norm=float(np.linalg.norm(support_log, 2)),
                errors=errors, matter_wick_connected=float(wick),
                zero_gap_independent_control=free_control,
                matter_gaussian_density_error=max_error(matter_block-gaussian),
                matter_region_entropy=entropy(matter_block),
                prepared_state_replaced=False)


def case(gamma, gap, hs, c, ground):
    number = [a.T@a for a in c]
    n = number[3]
    ident = sparse.eye(D)
    hs_full = sparse.kron(hs, sparse.eye(2), format='csr')
    ha_full = sparse.kron(ident, gap*RECORD, format='csr')
    interaction = sparse.kron(gamma*n, PAULI_X, format='csr')
    total = hs_full+ha_full+interaction
    dense = total.toarray()
    values, vectors = np.linalg.eigh(dense)
    incoming = np.kron(ground, [1., 0.])
    particle_current = 1j*comm(hs, n)
    flow_s = sparse.kron(-gamma*particle_current, PAULI_X, format='csr')
    flow_a = sparse.kron(-gamma*gap*n, PAULI_Y, format='csr')
    flow_i = 1j*comm(total, interaction)
    number_full = sparse.kron(sum(number), sparse.eye(2))
    errors = dict(matter_current=max_error(1j*comm(total, hs_full)-flow_s),
                  apparatus_current=max_error(1j*comm(total, ha_full)-flow_a),
                  total_current=max_error(flow_s+flow_a+flow_i),
                  particle_number=max_error(comm(total, number_full)),
                  distant_density_coupling=max_error(comm(interaction, sparse.kron(number[0], sparse.eye(2)))))
    assert max(errors.values()) < TOL
    tstar = np.pi/(2*gamma)
    base_s = expectation(incoming, hs_full)
    work_on = expectation(incoming, interaction)
    spectrum_initial = vectors.conj().T@incoming
    rows = []
    for ratio in (0., .25, .5, 1., 2.):
        t = ratio*tstar
        psi = vectors@(np.exp(-1j*t*values)*spectrum_initial)
        independent = expm_multiply(-1j*t*total, incoming)
        es, ea, ei = [expectation(psi, op) for op in (hs_full, ha_full, interaction)]
        norm_error = abs(float(np.vdot(psi, psi).real)-1)
        energy_error = abs(es+ea+ei-base_s-work_on)
        switching_error = abs((es-base_s)+ea-(work_on-ei))
        state_error = max_error(psi-independent)
        assert max(norm_error, energy_error, switching_error, state_error) < TOL
        rows.append(dict(time_over_tstar=ratio, time=float(t),
                         matter_energy_change=es-base_s, apparatus_energy=ea,
                         interaction_energy=ei, switch_on_work=work_on, switch_off_work=-ei,
                         record_probability=expectation(psi, sparse.kron(ident, RECORD)),
                         matter_energy_current=expectation(psi, flow_s),
                         apparatus_energy_current=expectation(psi, flow_a),
                         norm_error=norm_error, total_energy_error=energy_error,
                         switching_balance_error=switching_error, independent_state_error=state_error))
    unitary = (vectors*np.exp(-1j*tstar*values))@vectors.conj().T
    kraus = [unitary[a::2, 0::2] for a in (0, 1)]
    completeness = max_error(sum(k.conj().T@k for k in kraus)-np.eye(D))
    effect = kraus[1].conj().T@kraus[1]
    effect_error = float(np.max(abs(np.linalg.eigvalsh(effect-n.toarray()))))
    operator_energy_error = max_error(unitary.conj().T@dense@unitary-dense)
    conditional = []
    for outcome, projector in ((0, ident-n), (1, n)):
        state = projector@ground
        state /= np.linalg.norm(state)
        p1 = float(np.vdot(kraus[1]@state, kraus[1]@state).real)
        conditional.append(dict(initial_local_occupation=outcome, pointer_one_probability=p1))
    reverse_error = max_error(unitary.conj().T@(unitary@incoming)-incoming)
    assert max(completeness, operator_energy_error, reverse_error) < TOL
    # Independent no-hopping, zero-gap control has exact K0=1-n and K1=-i*n.
    ideal = (ident-n).toarray()+np.cos(gamma*tstar)*n.toarray()
    ideal_one = -1j*np.sin(gamma*tstar)*n.toarray()
    ideal_error = max(max_error(ideal-(ident-n).toarray()), max_error(ideal_one+1j*n.toarray()))
    return dict(gamma=gamma, apparatus_gap=gap, tstar=float(tstar), current_errors=errors,
                evolution=rows, instrument=dict(kraus_completeness_error=completeness,
                total_energy_operator_error=operator_energy_error, reverse_error=reverse_error,
                effect_operator_distance_from_ideal_occupation=effect_error,
                conditional_probabilities=conditional,
                equal_prior_classification_error=(conditional[0]['pointer_one_probability']
                    +1-conditional[1]['pointer_one_probability'])/2,
                no_hopping_zero_gap_ideal_control_error=ideal_error,
                channel_CP_reason='Kraus form from a fixed pure apparatus input',
                actual_outcome_selection_derived=False),
                coupled_reference=coupled_reference(dense, number, c, gamma, gap, incoming))


def run():
    c = annihilators(N)
    h = -np.diag(np.ones(N-1), 1)-np.diag(np.ones(N-1), -1)
    hs = second_quantize(h, c)
    _, vectors = np.linalg.eigh(hs.toarray())
    ground = vectors[:, 0]
    cases = [case(gamma, gap, hs, c, ground) for gamma in (.5, 2., 8.) for gap in (0., 1.)]
    return dict(candidate='CE-LM2', status='local_apparatus_energy_and_reference_gate',
                inputs=dict(matter_sites=N, local_site=3, hopping=1, mass=0,
                            gamma=[.5, 2., 8.], gap=[0., 1.], fit_parameters=0),
                cases=cases, full_joint_rmse=None, scientific_success=False,
                limits=['Finite one-dimensional local matter-apparatus action is supplied.',
                        'Total energy is conserved only with interaction energy included during fixed-H intervals.',
                        'Switching work is computed but an autonomous controller/energy reservoir is not constructed.',
                        'Finite-time readout is not an instantaneous ideal projector or stable unique outcome.',
                        'The fixed-particle-number coupled ground reference does not replace the prepared state.',
                        'No covariant stress, gauge unification, quantum metric or joint observation fit.'])


if __name__ == '__main__':
    result = run()
    here = Path(__file__).resolve()
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (here, here.with_name('ce_local_vacuum_modular.py'))}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(gamma=r['gamma'], gap=r['apparatus_gap'],
        readout=r['instrument']['equal_prior_classification_error'],
        effect_distance=r['instrument']['effect_operator_distance_from_ideal_occupation'],
        work_off=r['evolution'][3]['switch_off_work'],
        reference_rank=r['coupled_reference']['support_rank'],
        wick=r['coupled_reference']['matter_wick_connected']) for r in result['cases']], indent=2))
