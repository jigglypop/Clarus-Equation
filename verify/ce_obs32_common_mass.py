"""Fixed-input AP comparison after matching the supplied dark mass in Z.

No fit. f, abundance and initial phase remain inputs; not a unified model.
"""
import json
from pathlib import Path

import numpy as np
import mpmath as mp

from ce_obs32_profiled_matter import ce, SOURCE_SHA
from common_spectrum_vacuum_budget import shape_integral


def run():
    _, _, _, _, zs, obs, cov = ce.load_data()
    chol = np.linalg.cholesky(cov)
    def predict(p, method='DOP853'):
        bg = ce.Background(p, method=method)
        return np.array([bg.distances(z)['F_AP'] for z in zs]), bg
    def score(pred):
        w = np.linalg.solve(chol, pred-obs)
        return float(np.sqrt(w@w/len(obs)))
    baseline, _ = predict(ce.Parameters(.35, 0.))
    rows = []
    for r, mass_ev in [(.15, .027615), (.35, .014414)]:
        physical_s = (mass_ev*1e-9/2.435e18)**2
        with mp.workdps(50):
            potential_scale = float((mp.mpf(str(mass_ev))*mp.mpf('1e-9'))**4
                                    *shape_integral(mp.mpf(str(r)), mp.mpf(0)))
        for theta in [.1, .3, .5]:
            old, _ = predict(ce.Parameters(r, theta))
            p = ce.Parameters(r, theta, s_over_Mp2=physical_s)
            new, bg = predict(p)
            independent, _ = predict(p, 'Radau')
            discrepancy = float(np.max(abs(new-independent)))
            assert discrepancy < 1e-8
            end = bg.quantities(0.)
            # H_code^2 = rho_code/3; physical rho = U_top * rho_code.
            h_gev = bg.H0*np.sqrt(potential_scale)/2.435e18
            rho_gev4 = potential_scale*sum(end[k] for k in ['rho_b','rho_c','U','kinetic'])
            assert np.isclose(h_gev**2, rho_gev4/(3*(2.435e18)**2), rtol=1e-13, atol=0)
            h_km_s_mpc = h_gev/6.582119569e-25*3.0856775814913673e19
            rows.append(dict(r=r, theta_initial=theta, mass_eV=mass_ev,
                potential_scale_GeV4=potential_scale,
                conditional_H0_km_s_Mpc=float(h_km_s_mpc),
                scalar_density_GeV4=float(potential_scale*(end['U']+end['kinetic'])),
                s_over_Mp2=physical_s, old_rmse=score(old), common_mass_rmse=score(new),
                rmse_change=score(new)-score(old),
                max_AP_change=float(np.max(abs(new-old))),
                independent_solver_max_AP_difference=discrepancy,
                relative_continuity=bg.source_checks()['relative_total_continuity']))
    return dict(source_sha256=SOURCE_SHA, baseline_rmse=score(baseline), cases=rows,
        fitted_parameters=0, independent_holdout=False, joint_rmse=None,
        absolute_scale_scope='H0 conditional on supplied dark mass, abundance, f and initial state; no recombination or sound horizon; not a Hubble tension resolution',
        all_candidates_worse_than_baseline=all(x['common_mass_rmse']>score(baseline) for x in rows),
        interpretation='Matching the supplied dark mass in the loop kinetic term does not resolve AP error; the supplied f=1 dominates Z.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
