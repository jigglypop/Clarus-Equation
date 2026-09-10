"""CE-LM2-C: finite-energy autonomous control track for the same local record.

Sparse exponential propagation is crosschecked by DOP853. No external switch
is applied during evolution; preparation of the controller is still supplied.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from ce_local_vacuum_modular import annihilators, second_quantize, max_error, comm


def expect(psi, op):
    return float(np.vdot(psi, op@psi).real)


def case(clock_hopping):
    c = annihilators(8)
    one = -np.diag(np.ones(7), 1)-np.diag(np.ones(7), -1)
    full = second_quantize(one, c)
    sector = [i for i in range(256) if i.bit_count() == 4]
    hs = full[sector][:, sector].tocsr()
    n = (c[3].T@c[3])[sector][:, sector].tocsr()
    vals, vecs = np.linalg.eigh(hs.toarray())
    ground = vecs[:, 0]
    count = len(sector)
    length = 17
    clock = -clock_hopping*sparse.diags([np.ones(16), np.ones(16)], [-1, 1], shape=(17, 17), format='csr')
    contact = sparse.csr_matrix(([1.], ([8], [8])), shape=(17, 17))
    sigma_x = np.array([[0., 1.], [1., 0.]])
    record = np.diag([0., 1.])
    matter_h = sparse.kron(hs, sparse.eye(2*length), format='csr')
    clock_h = sparse.kron(sparse.eye(2*count), clock, format='csr')
    interaction = 2*sparse.kron(sparse.kron(n, sigma_x), contact, format='csr')
    total = matter_h+clock_h+interaction
    site = np.arange(length)
    clock_initial = np.exp(-(site-4)**2/4)*np.exp(1j*np.pi*site/2)
    clock_initial /= np.linalg.norm(clock_initial)
    initial = np.kron(np.kron(ground, [1., 0.]), clock_initial)
    initial_matter = expect(initial, matter_h)
    initial_clock = expect(initial, clock_h)
    initial_interaction = expect(initial, interaction)
    # Remove only a global phase in BOTH independent propagators.
    generator = total-expect(initial, total)*sparse.eye(total.shape[0], format='csr')
    times = np.arange(6)/clock_hopping
    direct = expm_multiply(-1j*generator, initial, start=0., stop=times[-1], num=6)
    independent = solve_ivp(lambda t, psi: -1j*(generator@psi), (0., times[-1]), initial,
                            method='DOP853', rtol=1e-10, atol=1e-12, t_eval=times)
    assert independent.success, independent.message
    clock_minimum = -2*clock_hopping*np.cos(np.pi/(length+1))
    eigen_minimum = np.linalg.eigvalsh(clock.toarray())[0]
    assert abs(eigen_minimum-clock_minimum) < 1e-12
    budget = initial_clock-clock_minimum
    pr = sparse.kron(sparse.eye(count), sparse.kron(record, sparse.eye(length)))
    pc = sparse.kron(sparse.eye(2*count), contact)
    flow_s = 1j*comm(total, matter_h)
    flow_c = 1j*comm(total, clock_h)
    flow_i = 1j*comm(total, interaction)
    assert max_error(flow_s+flow_c+flow_i) < 1e-12
    rows = []
    for j, t in enumerate(times):
        psi = direct[j]
        es, ec, ei = [expect(psi, op) for op in (matter_h, clock_h, interaction)]
        crosscheck = max_error(psi-independent.y[:, j])
        energy_error = abs(es+ec+ei-initial_matter-initial_clock-initial_interaction)
        norm_error = abs(float(np.vdot(psi, psi).real)-1)
        assert crosscheck < 1e-8 and energy_error < 1e-9 and norm_error < 1e-9
        assert es-initial_matter <= budget-ei+initial_interaction+1e-9
        rows.append(dict(time=float(t), scaled_time=float(t*clock_hopping),
                         matter_energy_change=es-initial_matter,
                         clock_energy_change=ec-initial_clock, interaction_energy=ei,
                         record_probability=expect(psi, pr), contact_probability=expect(psi, pc),
                         matter_energy_current=expect(psi, flow_s), clock_energy_current=expect(psi, flow_c),
                         independent_state_error=crosscheck, total_energy_error=energy_error, norm_error=norm_error))
    # Same vacuum's ideal occupation record: destroyed incident bond energy.
    hvals, hmodes = np.linalg.eigh(one)
    occupied = hmodes[:, hvals < 0]
    correlator = occupied@occupied.T
    ideal_cost = float(2*(correlator[2, 3]+correlator[3, 4]))
    return dict(clock_hopping=clock_hopping, clock_sites=length,
                fixed_particle_sector=4, state_dimension=len(initial),
                gamma=2, apparatus_gap=0, controller_initial_position=4, contact_position=8,
                initial_clock_energy=initial_clock, clock_minimum_energy=float(clock_minimum),
                fully_decoupled_free_energy_budget=float(budget), ideal_occupation_record_cost=ideal_cost,
                ideal_decoupled_record_excluded_by_energy=bool(ideal_cost > budget+1e-10),
                initial_interaction_energy=initial_interaction, explicit_external_switching_work=0,
                particle_number_conserved_by_sector_and_density_coupling=True,
                current_sum_operator_error=max_error(flow_s+flow_c+flow_i),
                integrator_evaluations=int(independent.nfev), evolution=rows,
                limitations=['The prepared controller packet supplies finite energy; its preparation is not derived.',
                             'Finite track reflection and recurring contact preclude assuming a permanent switch-off.',
                             'Pointer probability alone is not a measurement-fidelity or unique-outcome certificate.',
                             'No covariant control field, spacetime derivation or joint observational score.'])


if __name__ == '__main__':
    result = dict(candidate='CE-LM2-C', status='finite_autonomous_control_energy_gate',
                  cases=[case(j) for j in (.25, 1.)], fitted_parameters=0,
                  full_joint_rmse=None, scientific_success=False)
    here = Path(__file__).resolve()
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (here, here.with_name('ce_local_vacuum_modular.py'))}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(clock_hopping=r['clock_hopping'], budget=r['fully_decoupled_free_energy_budget'],
                          ideal_cost=r['ideal_occupation_record_cost'],
                          ideal_record_excluded=r['ideal_decoupled_record_excluded_by_energy'],
                          last=r['evolution'][-1]) for r in result['cases']], indent=2))
