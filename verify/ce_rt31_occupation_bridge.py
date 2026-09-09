"""Diagnose cold closures using the saved RT31 state, without refitting it."""
import hashlib
import json
from pathlib import Path

import numpy as np


def run():
    path = Path('_workspace/ce_rt31_checkpoints/t25_K48_n1536.npz')
    with np.load(path, allow_pickle=False) as saved:
        state = saved['state']
        k, weights = saved['k'], saved['weights']
        eps = float(saved['parameters'][2])
        time = float(saved['time'])
    theta, velocity, loga = state[:3]
    a = np.exp(loga)
    n, u, v = state[3:].reshape(3, 3, len(k))
    numbers = 2 * (n @ weights)

    def spectrum(phase):
        angles = (phase + 2*np.pi*np.arange(3))/3
        return 1+2*eps*np.cos(angles), -2*eps*np.sin(angles)/3

    def cold(phase, populations):
        x, derivative = spectrum(phase)
        return float(populations @ np.sqrt(x)/a**3), float(populations @ (derivative/(2*np.sqrt(x)))/a**3)

    x, derivative = spectrum(theta)
    omega = np.sqrt(k[None, :]**2 + a*a*x[:, None])
    rho = float(np.sum(2*omega*n*weights)/a**4)
    jn = float(np.sum(derivative[:, None]*n/omega*weights)/a**2)
    ju = float(np.sum(derivative[:, None]*u/omega*weights)/a**2)
    weighted_rho, weighted_j = cold(theta, numbers)
    equal_rho, equal_j = cold(theta, np.full(3, numbers.sum()/3))
    h = 1e-4
    derivative_check = (cold(theta-2*h, numbers)[0]-8*cold(theta-h, numbers)[0]
                        +8*cold(theta+h, numbers)[0]-cold(theta+2*h, numbers)[0])/(12*h)
    assert abs(derivative_check-weighted_j) < 1e-11
    # A 2pi phase shift permutes species. The state labels must follow it.
    shifted = cold(theta+2*np.pi, np.roll(numbers, -1))
    assert np.allclose(shifted, (weighted_rho, weighted_j), rtol=1e-11, atol=1e-14)
    return dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                time=time, theta=float(theta), occupation_fractions=(numbers/numbers.sum()).tolist(),
                full_mode=dict(rho=rho, particle_force=jn, coherence_force=ju, total_force=jn+ju),
                conserved_unequal_cold=dict(rho=weighted_rho, force=weighted_j),
                equal_cold_same_total_number=dict(rho=equal_rho, force=equal_j),
                unequal_cold_energy_relative_error=weighted_rho/rho-1,
                equal_cold_energy_relative_error=equal_rho/rho-1,
                unequal_cold_particle_force_relative_error=weighted_j/jn-1,
                equal_cold_particle_force_relative_error=equal_j/jn-1,
                mass_derivative_absolute_error=abs(derivative_check-weighted_j),
                species_permutation_covariance_verified=True,
                assumptions='Cold closures freeze measured N_j and omit coherence; neither is the full evolved state.',
                fitted_parameters=[], observational_rmse=None)


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
