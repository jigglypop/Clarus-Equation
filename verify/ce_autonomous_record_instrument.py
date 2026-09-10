"""CE-LM3: full sector instrument, repeated reads, and autonomous reference.

No outcome is selected autonomously. The intermediate ideal read has explicit
energy backaction. The gap-one model is a separate preregistered branch.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh, expm_multiply

from ce_local_vacuum_modular import annihilators, second_quantize, max_error, entropy


LENGTH = 17
TIMES = (5., 10., 20., 40.)
X = np.array([[0., 1.], [1., 0.]])
R = np.diag([0., 1.])


def build(gap):
    c = annihilators(8)
    one = -np.diag(np.ones(7), 1)-np.diag(np.ones(7), -1)
    sector = [i for i in range(256) if i.bit_count() == 4]
    hs = second_quantize(one, c)[sector][:, sector].real.tocsr()
    n = (c[3].T@c[3])[sector][:, sector].tocsr()
    count = len(sector)
    clock = -sparse.diags([np.ones(16), np.ones(16)], [-1, 1], shape=(17, 17), format='csr')
    contact = sparse.csr_matrix(([1.], ([8], [8])), shape=(17, 17))
    interaction = 2*sparse.kron(sparse.kron(n, X), contact, format='csr')
    total = (sparse.kron(hs, sparse.eye(34))+sparse.kron(sparse.eye(2*count), clock)
             +sparse.kron(sparse.eye(count), sparse.kron(gap*R, sparse.eye(17)))+interaction).tocsr()
    energies, modes = np.linalg.eigh(hs.toarray())
    ground = modes[:, 0]
    sites = np.arange(LENGTH)
    packet = np.exp(-(sites-4)**2/4)*np.exp(1j*np.pi*sites/2)
    packet /= np.linalg.norm(packet)
    return total, interaction, hs, n, clock, contact, sector, ground, packet, energies, modes


def local_amplitudes(psi, sector):
    """Exact embedding of the one-controller-particle sector across a local cut."""
    values = psi.reshape(len(sector), 2, LENGTH)
    result = np.zeros((16, 64*17), complex)
    for i, mask in enumerate(sector):
        bits = [(mask >> (7-j)) & 1 for j in range(8)]
        local = 2*bits[2]+bits[3]
        outer = 0
        for j in (0, 1, 4, 5, 6, 7):
            outer = 2*outer+bits[j]
        for pointer in (0, 1):
            for position in range(LENGTH):
                at_contact = int(position == 8)
                outer_clock = 0 if at_contact else position+1 if position < 8 else position
                result[4*local+2*pointer+at_contact, 17*outer+outer_clock] = values[i, pointer, position]
    assert abs(np.linalg.norm(result)-np.linalg.norm(psi)) < 1e-12
    return result


def reference_audit(total, interaction, hs, n, clock, contact, sector, prepared, gap):
    rng = np.random.default_rng(7301)
    vals, modes = eigsh(total, k=2, which='SA', tol=1e-12, maxiter=20000,
                       v0=rng.normal(size=total.shape[0]))
    order = np.argsort(vals)
    vals, modes = vals[order], modes[:, order]
    psi = modes[:, 0].astype(complex)
    residual = max_error(total@psi-vals[0]*psi)
    assert residual < 1e-9
    a = local_amplitudes(psi, sector)
    rho = a@a.conj().T
    b = local_amplitudes(prepared, sector)
    prepared_rho = b@b.conj().T
    rv, ru = np.linalg.eigh(rho)
    assert rv.min() > -1e-10
    numerical_support = rv > 1e-12
    leakage = float(np.trace(prepared_rho@(ru[:, ~numerical_support]@ru[:, ~numerical_support].conj().T)).real)
    control = None
    if gap == 0:
        # Exact conserved X blocks give an independent ground-energy calculation.
        base = sparse.kron(hs, sparse.eye(17))+sparse.kron(sparse.eye(len(sector)), clock)
        branches = []
        for sign in (-1, 1):
            h = (base+2*sign*sparse.kron(n, contact)).tocsr()
            e, v = eigsh(h, k=1, which='SA', tol=1e-12,
                         v0=rng.normal(size=h.shape[0]))
            branches.append(float(e[0]))
        pointer_x = sparse.kron(sparse.eye(len(sector)), sparse.kron(X, sparse.eye(17)))
        px = float(np.vdot(psi, pointer_x@psi).real)
        assert abs(vals[0]-min(branches)) < 1e-9 and abs(px+1) < 1e-9
        assert abs(leakage-.5) < 1e-9
        control = dict(X_block_energies=branches, energy_error=abs(float(vals[0])-min(branches)), pointer_X=px)
        relative, status = None, 'infinite_exact_X_support_mismatch'
    elif rv.min() > 1e-10:
        modular = -(ru*np.log(rv))@ru.conj().T
        relative = float(np.trace(prepared_rho@modular).real-entropy(prepared_rho))
        assert relative > -1e-9
        status = 'finite_positive_reference'
    else:
        relative, status = None, 'unresolved_small_eigenvalues_no_clipping'
    return dict(sector_ground_energy=float(vals[0]), sector_gap=float(vals[1]-vals[0]),
                eigenstate_residual=residual, local_dimension=16,
                local_reference_eigenvalues=rv.tolist(), numerical_support_rank=int(numerical_support.sum()),
                numerical_support_threshold=1e-12, prepared_weight_outside_numerical_support=leakage,
                relative_entropy=relative, relative_entropy_status=status,
                local_reference_entropy=entropy(rho), local_prepared_entropy=entropy(prepared_rho),
                independent_X_block_control=control)


def run_case(gap):
    total, interaction, hs, n, clock, contact, sector, ground, packet, energies, modes = build(gap)
    count = len(sector)
    environment = np.kron([1., 0.], packet)
    initial_map = np.kron(np.eye(count), environment[:, None])
    prepared = initial_map@ground
    independent_gapped = None
    if gap:
        solution = solve_ivp(lambda t, psi: -1j*(total@psi), (0., TIMES[-1]), prepared,
                             t_eval=TIMES, method='DOP853', rtol=1e-10, atol=1e-12)
        assert solution.success, solution.message
        independent_gapped = solution.y.T
    free_arrival = (modes*np.exp(-2j*energies))@modes.T
    targets = {'initial_occupation': n.toarray(),
               'free_arrival_occupation': free_arrival.conj().T@n@free_arrival}
    incoming = []
    for p in (sparse.eye(count)-n, n):
        state = p@ground
        incoming.append(state/np.linalg.norm(state))
    evolved = initial_map.astype(complex)
    read_branches = None
    first_read_energy = None
    previous = 0.
    rows = []
    for time_index, t in enumerate(TIMES):
        elapsed = t-previous
        evolved = expm_multiply(-1j*elapsed*total, evolved)
        if read_branches is not None:
            read_branches = expm_multiply(-1j*elapsed*total, read_branches)
        values = evolved.reshape(count, 2, LENGTH, count)
        one = values[:, 1, :, :].reshape(count*LENGTH, count)
        effect = one.conj().T@one
        completeness = max_error(evolved.conj().T@evolved-np.eye(count))
        energy_matrix_error = max_error(evolved.conj().T@(total@evolved)-hs.toarray())
        ke = np.linalg.eigvalsh(effect)
        kraus_rows = values.transpose(1, 2, 0, 3).reshape(34, count*count)
        choi_gram = kraus_rows.conj()@kraus_rows.T
        gram_eigen = np.linalg.eigvalsh(choi_gram)
        assert completeness < 1e-9 and ke.min() > -1e-9 and ke.max() < 1+1e-9
        assert energy_matrix_error < 1e-8
        assert gram_eigen.min() > -1e-9 and abs(np.trace(choi_gram).real-count) < 1e-8
        distances = {name: float(np.max(abs(np.linalg.eigvalsh(effect-target)))) for name, target in targets.items()}
        p1 = [float(np.vdot(psi, effect@psi).real) for psi in incoming]
        psi = evolved@ground
        if gap == 0:
            base = sparse.kron(hs, sparse.eye(17))+sparse.kron(sparse.eye(count), clock)
            initial = np.kron(ground, packet)
            plus = expm_multiply(-1j*t*(base+2*sparse.kron(n, contact)), initial)
            minus = expm_multiply(-1j*t*(base-2*sparse.kron(n, contact)), initial)
            independent = np.stack([(plus+minus).reshape(count, LENGTH)/2,
                                    (plus-minus).reshape(count, LENGTH)/2], axis=1).reshape(-1)
            independent_error = max_error(psi-independent)
            assert independent_error < 1e-9
        else:
            independent_error = max_error(psi-independent_gapped[time_index])
            assert independent_error < 1e-8
        if read_branches is None:
            read_branches = np.zeros((len(psi), 2), complex)
            tensor = psi.reshape(count, 2, LENGTH)
            for a in (0, 1):
                branch = np.zeros_like(tensor)
                branch[:, a, :] = tensor[:, a, :]
                read_branches[:, a] = branch.reshape(-1)
            before = float(np.vdot(psi, total@psi).real)
            after = float(np.trace(read_branches.conj().T@(total@read_branches)).real)
            prediction = -float(np.vdot(psi, interaction@psi).real)
            assert abs(after-before-prediction) < 1e-9
            first_read_energy = dict(time=t, work=after-before, negative_interaction=prediction)
        tensor = read_branches.reshape(count, 2, LENGTH, 2)
        joint = np.array([[np.sum(abs(tensor[:, b, :, a])**2) for b in (0, 1)] for a in (0, 1)])
        assert abs(joint.sum()-1) < 1e-9
        rows.append(dict(time=t, completeness_error=completeness,
                         energy_matrix_error=energy_matrix_error,
                         effect_eigenvalue_range=[float(ke.min()), float(ke.max())],
                         choi_gram_minimum=float(gram_eigen.min()), effect_distances=distances,
                         conditional_pointer_one=p1, equal_prior_error=(p1[0]+1-p1[1])/2,
                         vacuum_pointer_one=float(np.vdot(ground, effect@ground).real),
                         sequential_joint_probabilities=joint.tolist(),
                         sequential_flip_probability=float(joint[0, 1]+joint[1, 0]),
                         first_read_changes_late_pointer_one=float(joint[:, 1].sum()-np.vdot(ground, effect@ground).real),
                         independent_state_error=independent_error,
                         independent_method='X_block_exponentials' if gap == 0 else 'DOP853'))
        previous = t
        print(f'gap={gap:g} t={t:g}: instrument checked', flush=True)
    reference = reference_audit(total, interaction, hs, n, clock, contact, sector, prepared, gap)
    return dict(apparatus_gap=gap, rows=rows, first_read_energy=first_read_energy, reference=reference,
                instrument_scope='all 70 matter inputs in fixed particle-number-four sector',
                readout_rule='pointer equals initial occupation label; no relabeling or time refitting',
                actual_outcome_selection_derived=False)


if __name__ == '__main__':
    zero = run_case(0.)
    cases = [zero]
    if zero['reference']['relative_entropy_status'] == 'infinite_exact_X_support_mismatch':
        cases.append(run_case(1.))
    result = dict(candidate='CE-LM3', status='autonomous_instrument_and_reference_gate',
                  cases=cases, fitted_parameters=0, full_joint_rmse=None, scientific_success=False,
                  limitations=['Finite one-dimensional regulator and prepared clock; no spacetime or gauge derivation.',
                               'Intermediate projective read is supplied and has energy backaction.',
                               'Finite sampled stability checks are not permanent or unique-outcome proofs.',
                               'Faithful local reference is not a CFT modular-stress identity or Einstein limit.',
                               'No observations or independent holdout are evaluated.'])
    here = Path(__file__).resolve()
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (here, here.with_name('ce_local_vacuum_modular.py'))}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(gap=r['apparatus_gap'], first_read=r['first_read_energy'],
                          reference=r['reference']) for r in cases], indent=2))
