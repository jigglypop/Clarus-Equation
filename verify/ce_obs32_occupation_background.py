"""OBS32 cold background with explicit conserved species populations.

Unequal populations require an explicit initial velocity: the original regular
mode assumes a stationary mass function at theta=0. No RT31 state transplant.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from ce_obs32_profiled_matter import ce


class OccupationSpectrum(ce.Spectrum):
    def __init__(self, parameters, populations):
        super().__init__(parameters)
        self.populations = np.asarray(populations, dtype=float)
        if self.populations.shape != (3,) or not np.all(np.isfinite(self.populations)):
            raise ValueError('Three finite species populations required')
        if np.any(self.populations < 0) or self.populations.sum() <= 0:
            raise ValueError('Nonnegative populations with positive total required')
        self.populations = self.populations/self.populations.sum()
        self.mnorm = float(self.populations @ np.sqrt(self.eigs(0.)[0]))

    def evaluate(self, theta):
        result = list(super().evaluate(theta))
        x, x1, x2 = self.eigs(theta)
        m = np.sqrt(x)
        result[3:6] = [float(self.populations @ term/self.mnorm) for term in
                       (m, x1/(2*m), x2/(2*m)-x1*x1/(4*m**3))]
        return tuple(result)

    def initial_velocity(self):
        if not np.allclose(self.populations, np.ones(3)/3, rtol=0, atol=1e-14):
            raise ValueError('Unequal populations require explicitly matched initial velocity')
        return super().initial_velocity()


class OccupationBackground(ce.Background):
    def __init__(self, parameters, populations, initial_velocity=None, method='DOP853'):
        self.p = parameters
        self.sp = OccupationSpectrum(parameters, populations)
        self.Ni = np.log(parameters.ai)
        velocity = self.sp.initial_velocity() if initial_velocity is None else float(initial_velocity)
        if not np.isfinite(velocity):
            raise ValueError('Finite initial dtheta/dln(a) required')
        self.sol = solve_ivp(self.rhs, (self.Ni, 0.), [parameters.theta_initial, velocity],
                             method=method, rtol=2e-10, atol=2e-12, dense_output=True, max_step=.08)
        if not self.sol.success:
            raise RuntimeError(self.sol.message)
        self.H0 = self.quantities(0.)['H']


def matter_stationary_modes(sp):
    """Leading cold matter-era modes; fixed species labels have period 6pi.

    delta'' + 3/2 delta' + K delta = F a^3, primes d/dln(a).
    A finite past limit requires m'(theta_star)=0. Homogeneous amplitudes
    remain state inputs; finding a root does not select the cosmological state.
    """
    grid = np.linspace(0., 6*np.pi, 721)
    roots = []
    for left, right in zip(grid[:-1], grid[1:]):
        if sp.evaluate(left)[4]*sp.evaluate(right)[4] < 0:
            root = brentq(lambda t: sp.evaluate(t)[4], left, right, xtol=1e-13)
            if not roots or abs(root-roots[-1]) > 1e-9:
                roots.append(root)
    rows = []
    for root in roots:
        U, U1, U2, m, m1, m2, Z, Z1 = sp.evaluate(root)
        D = 1-sp.p.fc+sp.p.fc*m
        K = 3*sp.p.fc*m2/(Z*D)
        exponents = np.roots([1., 1.5, K])
        force = -3*U1/(Z*sp.p.Rm*D)
        coefficient = force/(13.5+K)
        h = 1e-4
        fd = (sp.evaluate(root+h)[4]-sp.evaluate(root-h)[4])/(2*h)
        assert abs(fd-m2) < 1e-9
        assert abs(m1) < 1e-12
        assert max(abs(exponents**2+1.5*exponents+K)) < 1e-12
        rows.append(dict(theta=root, mass=m, mass_curvature=m2, K=K,
                         homogeneous_exponents=[dict(real=float(q.real), imag=float(q.imag)) for q in exponents],
                         forced_a3_coefficient=coefficient,
                         freely_specifiable_past_regular_modes=int(sum(q.real>0 for q in exponents))))
    return rows


def validate():
    rows = []
    for r in [.15, .35]:
        for theta in [.1, .3, .5]:
            p = ce.Parameters(r, theta)
            original, generalized = ce.Background(p), OccupationBackground(p, [1, 1, 1])
            error = max(abs(original.distances(z)['F_AP']-generalized.distances(z)['F_AP'])
                        for z in [.3, .7, 1.5, 2.3])
            assert error < 1e-9
            rows.append(dict(r=r, theta=theta, equal_population_AP_absolute_difference=error))
    fractions = json.loads(Path('verify/ce_rt31_occupation_bridge.json').read_text())['occupation_fractions']
    sp = OccupationSpectrum(ce.Parameters(.35, .1), fractions)
    try:
        sp.initial_velocity()
    except ValueError:
        pass
    else:
        raise AssertionError('Original regular initial mode cannot be silently inherited')
    # Explicit zero velocity is a diagnostic input, not an inferred preparation.
    bg = OccupationBackground(ce.Parameters(.35, .1), fractions, initial_velocity=0.)
    checks = bg.source_checks()
    assert checks['relative_total_continuity'] < 1e-11
    assert checks['H_derivative_finite_difference_abs'] < 1e-7
    return dict(equal_population_regression=rows, unequal_population_mprime_at_zero=sp.evaluate(0.)[4],
                cold_matter_stationary_modes=matter_stationary_modes(sp),
                explicit_velocity_diagnostic_checks=checks, fitted_parameters=[],
                observational_rmse=None, state_matching_complete=False)


if __name__ == '__main__':
    result = validate()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
