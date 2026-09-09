"""Source chapter 19 locked branch: fixed inputs, no parameter optimization.

Compare with LCDM at identical densities, rather than changing densities while
changing theories. Reconstruct the loop curvature with high precision and check
distance quadrature against an independently integrated distance ODE.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp

from ce_obs32_profiled_matter import ce, SOURCE_SHA


def run():
    mp.mp.dps = 60
    r = mp.mpf('.495')  # Chapter 19 example, not a derived constant.

    def vacuum(theta):
        masses = [1 + 2*r*mp.cos((theta+2*mp.pi*j)/6) for j in range(6)]
        return -sum(d**4*(mp.log(d*d)-mp.mpf('1.5')) for d in masses)/(32*mp.pi**2)

    u0 = vacuum(0)-vacuum(mp.pi)
    u2 = mp.diff(vacuum, 0, 2)
    critical_ratio = float(-u2/(u0*r/(18*(1-2*r))))
    *_, z, observed, cov = ce.load_data()
    z = np.asarray(z)
    chol = np.linalg.cholesky(cov)
    cases = []
    # Retain both published inputs; never select the smaller score as a fit.
    for name, om, oc in [('chapter19', .315, .120/.674**2),
                         ('obs32_density_control', .3, .84*.3)]:
        ratio = oc/(1-om)
        astar = (ratio/critical_ratio)**(1/3)

        def expansion(zz):
            # Exact theta=0, theta_dot=0, Lambda_R=0 solution of chapter 19.
            return np.sqrt(om*(1+zz)**3 + 1-om)

        prediction = np.array([expansion(zz)*quad(
            lambda x: 1/expansion(x), 0, zz, epsabs=1e-12, epsrel=1e-12)[0]
            for zz in z])
        ode = solve_ivp(lambda zz, y: [1/expansion(zz)], (0, float(max(z))), [0.],
                        method='DOP853', rtol=2e-12, atol=2e-13, dense_output=True)
        assert ode.success
        check = expansion(z)*ode.sol(z)[0]
        difference = float(np.max(abs(prediction-check)))
        assert difference < 1e-9
        residual = np.linalg.solve(chol, prediction-observed)
        chi2 = float(residual@residual)
        assert abs(chi2-float((prediction-observed) @
                             np.linalg.solve(cov, prediction-observed))) < 1e-10
        assert ratio > critical_ratio  # Past z>=0 has still greater density.
        cases.append(dict(name=name, omega_m=om, omega_c=oc,
                          current_density_ratio=ratio, critical_scale_factor=astar,
                          stable_over_data_redshifts=True,
                          prediction=prediction.tolist(), chi2=chi2,
                          ap_rmse=float(np.sqrt(chi2/len(z))),
                          independent_distance_max_abs=difference,
                          matched_LCDM_delta_rmse=0.0))
    return dict(source_chapter=19, data_loader_sha256=SOURCE_SHA,
                fixed_r=float(r), u0=float(u0), u_second=float(u2),
                critical_density_ratio=critical_ratio,
                initial_phase=0, initial_phase_velocity=0, Lambda_R=0,
                fitted_parameters=[], cases=cases,
                joint_quantum_macro_rmse=None,
                conclusion='Locked branch preserves matched LCDM distances exactly; no RMSE reduction.',
                limitations=['Observed density normalization and selected r remain inputs.',
                             'Radiation neglected as in original late-time branch.',
                             'No charged portal in this action; no direct new one-loop muon term.',
                             'Early universe, primordial state, and four-force completion not computed.'])


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
