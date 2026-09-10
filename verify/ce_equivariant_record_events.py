"""CE-EV1: extra Bell-type configuration law with uncollapsed global state.

This tests finite probability currents, not sampled paths, covariant dynamics,
or an empirical demonstration of the added configuration ontology.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from scipy.sparse.linalg import expm_multiply

from ce_autonomous_record_instrument import build


def edge_types(rows, cols, sector):
    counts = dict(matter_hop=0, controller_hop=0, pointer_flip=0)
    for target, source in zip(rows, cols):
        tm, rest = divmod(int(target), 34)
        ta, tx = divmod(rest, 17)
        sm, rest = divmod(int(source), 34)
        sa, sx = divmod(rest, 17)
        if tm != sm:
            difference = sector[tm] ^ sector[sm]
            bits = [i for i in range(8) if difference & (1 << i)]
            assert ta == sa and tx == sx and len(bits) == 2 and bits[1]-bits[0] == 1
            counts['matter_hop'] += 1
        elif tx != sx:
            assert ta == sa and abs(tx-sx) == 1
            counts['controller_hop'] += 1
        else:
            assert ta != sa and tx == sx == 8 and (sector[tm] & (1 << 4))
            counts['pointer_flip'] += 1
    return counts


def run():
    total, _, _, _, _, _, sector, ground, packet, _, _ = build(1.)
    initial = np.kron(np.kron(ground, [1., 0.]), packet)
    matrix = total.tocoo()
    mask = matrix.row != matrix.col
    row, col, coefficient = matrix.row[mask], matrix.col[mask], matrix.data[mask]
    size = len(initial)
    edge_counts = edge_types(row, col, sector)
    bound = float(np.max(np.bincount(col, weights=abs(coefficient), minlength=size)))
    diagonal = total.diagonal()
    initial_energy = float(np.vdot(initial, total@initial).real)
    cases = []
    for t in (0., .5, 1., 2., 5., 10.):
        psi = expm_multiply(-1j*t*total, initial)
        p = abs(psi)**2
        current = 2*np.imag(psi[row].conj()*coefficient*psi[col])
        positive = np.maximum(current, 0.)
        assert np.all(positive[p[col] == 0] == 0)
        rates = np.divide(positive, p[col], out=np.zeros_like(positive), where=p[col] > 0)
        assert np.isfinite(rates).all()
        gain = np.bincount(row, weights=rates*p[col], minlength=size)
        loss = np.bincount(col, weights=rates*p[col], minlength=size)
        schrodinger = 2*np.real(psi.conj()*(-1j*(total@psi)))
        error = float(np.max(abs(gain-loss-schrodinger)))
        traffic = .1*abs(coefficient)*p[row]*p[col]
        alternative = positive+traffic
        alternative_rhs = (np.bincount(row, weights=alternative, minlength=size)
                           -np.bincount(col, weights=alternative, minlength=size))
        alternate_error = float(np.max(abs(alternative_rhs-schrodinger)))
        activity = float(positive.sum())
        alternative_activity = float(alternative.sum())
        assert error < 1e-10 and alternate_error < 1e-10
        assert activity <= bound+1e-10 and alternative_activity > activity
        total_energy = float(np.vdot(psi, total@psi).real)
        diagonal_energy = float(np.dot(p, diagonal))
        diagonal_derivative = float(np.dot(gain-loss, diagonal))
        quantum_diagonal_derivative = float(np.vdot(psi, 1j*(total@(diagonal*psi)-diagonal*(total@psi))).real)
        assert abs(total_energy-initial_energy) < 1e-10
        assert abs(diagonal_derivative-quantum_diagonal_derivative) < 1e-10
        cases.append(dict(time=t, equivariance_max_error=error,
                          alternative_equivariance_max_error=alternate_error,
                          probability_norm_error=abs(float(p.sum())-1),
                          expected_jump_intensity=activity, alternative_jump_intensity=alternative_activity,
                          largest_conditional_rate=float(rates.max()), exact_zero_probability_states=int(np.sum(p == 0)),
                          total_energy=total_energy, diagonal_configuration_energy_mean=diagonal_energy,
                          offdiagonal_energy_mean=total_energy-diagonal_energy,
                          diagonal_energy_derivative=diagonal_derivative))
    return dict(candidate='CE-EV1', status='conditional_extra_configuration_law',
                same_H_as='CE-LM3-G', configuration_count=size, directed_edge_counts=edge_counts,
                expected_activity_uniform_bound=bound, cases=cases,
                fitted_parameters=0, full_joint_rmse=None, scientific_success=False,
                sampled_trajectories=0,
                axioms=['Actual configuration Q in addition to the global wavefunction.',
                        'Initial configuration distribution equals Born probabilities.',
                        'Minimal positive-current jump rates; not uniquely forced by equivariance.'],
                limits=['The global wavefunction is not collapsed or conditioned on Q.',
                        'Expected quantum energy conservation is not trajectory-level diagonal-energy conservation.',
                        'Local graph edges do not prove relativistic locality of state-dependent rates.',
                        'No Q-dependent gravitational backreaction or covariant stress is constructed.',
                        'LM3 record errors/instability are not fixed by adding Q.',
                        'No new observation or physical validation of the added ontology.'])


if __name__ == '__main__':
    result = run()
    here = Path(__file__).resolve()
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (here, here.with_name('ce_autonomous_record_instrument.py'), here.with_name('ce_local_vacuum_modular.py'))}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
