"""CE-LR1: a new local scalar-multiplet interpretation of the AM4 record.

The mass matrix is supplied from AM4. The result is a co-located internal
record model, not separated pair creation, an observed particle model or UV GR.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import expm, sqrtm

from ce_color_covariant_record import HF, GAMMA, REC, generators, setup, unitary


SIGNS = np.array([-1., 1., 1., 1.])
ETA = np.diag(SIGNS)
TOL = 1e-10


def error(value):
    return float(np.max(abs(value)))


def spectral(values, vectors, function):
    return (vectors*function(values))@vectors.conj().T


def field_jet(momenta, internal, mass_values, modes, event):
    phi = np.zeros(16, complex)
    first = np.zeros((4, 16), complex)
    second = np.zeros((4, 4, 16), complex)
    time, position = event[0], np.array(event[1:])
    for momentum, chi in zip(momenta, internal):
        omega_values = np.sqrt(mass_values**2+momentum@momentum)
        coefficients = (modes.conj().T@chi)/np.sqrt(2*omega_values*len(momenta))
        mode = coefficients*np.exp(-1j*time*omega_values+1j*momentum@position)
        derivatives = [-1j*omega_values]+[1j*p*np.ones(16) for p in momentum]
        phi += modes@mode
        for a in range(4):
            first[a] += modes@(derivatives[a]*mode)
            for b in range(4):
                second[a, b] += modes@(derivatives[a]*derivatives[b]*mode)
    return phi, first, second


def stress_audit(mass_squared, jet):
    phi, first, second = jet
    inner_lagrangian = sum(SIGNS[a]*np.vdot(first[a], first[a]).real for a in range(4))
    inner_lagrangian += np.vdot(phi, mass_squared@phi).real
    tensor = np.array([[2*np.vdot(first[m], first[n]).real-ETA[m, n]*inner_lagrangian
                        for n in range(4)] for m in range(4)])
    differentiated = np.zeros((4, 4, 4))
    for a in range(4):
        dl = 2*(sum(SIGNS[b]*np.vdot(second[a, b], first[b]) for b in range(4))
                +np.vdot(phi, mass_squared@first[a])).real
        for m in range(4):
            for n in range(4):
                differentiated[a, m, n] = (2*(np.vdot(second[a, m], first[n])
                                              +np.vdot(first[m], second[a, n])).real
                                           -ETA[m, n]*dl)
    divergence = np.array([sum(SIGNS[m]*SIGNS[n]*differentiated[m, m, n] for m in range(4))
                           for n in range(4)])
    kg = sum(SIGNS[a]*second[a, a] for a in range(4))-mass_squared@phi
    identity = np.array([2*SIGNS[n]*np.vdot(kg, first[n]).real for n in range(4)])
    positive_energy = sum(np.vdot(v, v).real for v in first)+np.vdot(phi, mass_squared@phi).real
    assert error(kg) < TOL and error(divergence) < TOL and error(divergence-identity) < TOL
    assert tensor[0, 0] > 0 and abs(tensor[0, 0]-positive_energy) < TOL
    return dict(stress_lower=tensor.tolist(), KG_residual=error(kg), direct_divergence=divergence.tolist(),
                divergence_max_error=error(divergence), Noether_identity_error=error(divergence-identity),
                positive_energy_error=abs(tensor[0, 0]-positive_energy))


def run():
    initial, final, transition = setup()
    mass = HF+GAMMA*transition
    mass_squared = mass@mass
    values, modes = np.linalg.eigh(mass)
    assert values.min() > 0
    square_root_error = error(sqrtm(mass_squared)-mass)
    symmetry_error = 0.
    for generator in generators():
        gs, ga = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        gs[:3, :3], ga[1:, 1:] = generator, -generator.conj()
        total = np.kron(gs, np.eye(4))+np.kron(np.eye(4), ga)
        symmetry_error = max(symmetry_error, error(mass@total-total@mass),
                             error(mass_squared@total-total@mass_squared))
    assert square_root_error < TOL and symmetry_error < TOL
    record = np.kron(np.eye(4), REC)
    cases = []
    for momentum in (np.zeros(3), np.array([0., .3, 0.])):
        p_squared = momentum@momentum
        omega_values = np.sqrt(values**2+p_squared)
        omega = spectral(values, modes, lambda m: np.sqrt(m*m+p_squared))
        normalizer = spectral(values, modes, lambda m: (2*np.sqrt(m*m+p_squared))**-.5)
        inverse_normalizer = spectral(values, modes, lambda m: (2*np.sqrt(m*m+p_squared))**.5)
        matrix_root_error = error(omega-sqrtm(mass_squared+p_squared*np.eye(16)))
        initial_field = normalizer@initial
        kg_generator = np.block([[np.zeros((16, 16)), np.eye(16)],
                                 [-mass_squared-p_squared*np.eye(16), np.zeros((16, 16))]])
        initial_data = np.concatenate((initial_field, -1j*omega@initial_field))
        rows = []
        for time in (0., .25, .5, .75, 1.):
            chi = (modes*np.exp(-1j*time*omega_values))@modes.conj().T@initial
            kg_data = expm(time*kg_generator)@initial_data
            field, velocity = kg_data[:16], kg_data[16:]
            independent_error = max(error(field-normalizer@chi), error(velocity+1j*omega@field))
            charge = float((1j*(np.vdot(field, velocity)-np.vdot(velocity, field))).real)
            energy = float((np.vdot(velocity, velocity)+np.vdot(field, (mass_squared+p_squared*np.eye(16))@field)).real)
            expected_energy = float(np.vdot(chi, omega@chi).real)
            record_probability = float(np.vdot(chi, record@chi).real)
            frequency = (np.sqrt(p_squared+(5+GAMMA)**2)-np.sqrt(p_squared+(5-GAMMA)**2))/2
            record_error = abs(record_probability-np.sin(frequency*time)**2)
            rest_error = error(chi-unitary(mass, time)@initial) if p_squared == 0 else None
            assert independent_error < TOL and abs(charge-1) < TOL and abs(energy-expected_energy) < TOL
            assert record_error < TOL and error(inverse_normalizer@field-chi) < TOL
            if rest_error is not None:
                assert rest_error < TOL
            rows.append(dict(time=time, independent_KG_state_error=independent_error,
                             KG_charge=charge, field_energy=energy, Hilbert_energy=expected_energy,
                             record_probability=record_probability, closed_form_record_error=record_error,
                             original_AM4_rest_state_error=rest_error))
        cases.append(dict(momentum=momentum.tolist(), matrix_square_root_error=matrix_root_error, rows=rows))
    momenta = [np.array([0., .3, 0.]), np.array([.2, 0., .1])]
    spacetime = []
    for event in ((0., 0., 0., 0.), (.3, .2, -.1, .4), (.7, -.2, .5, .1)):
        row = stress_audit(mass_squared, field_jet(momenta, [initial, final], values, modes, event))
        row['event'] = event
        spacetime.append(row)
    return dict(candidate='CE-LR1', status='conditional_local_scalar_multiplet_with_internal_record',
                field_components=16, mass_eigenvalues=values.tolist(),
                mass_square_root_error=square_root_error, color_invariance_error=symmetry_error,
                plane_wave_cases=cases, spacetime_stress_cases=spacetime,
                fitted_parameters=0, full_joint_rmse=None, scientific_success=False,
                assumptions=['The AM4 Hilbert space is now the internal space of one co-located scalar multiplet.',
                             'The positive AM4 Hamiltonian is supplied as the relativistic mass matrix.',
                             'The local KG action and canonical free-field quantization are extra axioms.'],
                limitations=['Separated source/apparatus pair creation and transport are not represented.',
                             'Plane waves are normalization-density checks, not localized finite-energy packets.',
                             'An internal projector is not yet a complete local measurement apparatus.',
                             'No emergent EH term, gauge-field generation, nonlinear quantum backreaction or joint RMSE.'])


if __name__ == '__main__':
    result = run()
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
    print(json.dumps(result, indent=2))
