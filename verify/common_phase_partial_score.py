"""Conditional AP + EM muon subset + clock score at a shared evolved phase.

Eight residuals, NOT the full goal's joint RMSE or a completed common action.
No fitting or selection of the best case; block independence assumed.
"""
import json
from pathlib import Path

import numpy as np

from common_spectrum_muon import exact_em
from common_spectrum_pairing import dirac_em
from ce_obs32_profiled_matter import ce


def read(name):
    return json.loads(Path(__file__).with_name(name).read_text())


def run():
    macro = read('ce_obs32_common_mass.json')
    clock = read('common_phase_clock.json')
    muon = read('common_spectrum_muon.json')['frozen_summary']
    base = np.array([macro['baseline_rmse'], muon['residual']/muon['combined_sigma'],
                     clock['baseline_zero_drift_residual']])
    counts = np.array([6, 1, 1])
    _, _, _, _, zs, ap_observed, ap_covariance = ce.load_data()
    full_covariance = np.zeros((8,8))
    full_covariance[:6,:6] = ap_covariance
    full_covariance[6,6] = muon['combined_sigma']**2
    full_covariance[7,7] = clock['sigma_per_year']**2
    full_cholesky = np.linalg.cholesky(full_covariance)
    def score(groups):
        return float(np.sqrt(np.sum(counts*groups**2)/8))
    rows = []
    for m, c in zip(macro['cases'], clock['cases'], strict=True):
        assert (m['r'],m['theta_initial']) == (c['r'],c['initial_theta'])
        theta = c['theta_today']
        scalar = exact_em(1000., .025, theta)
        fermion = dirac_em(1000., .025, theta)
        correction = 2*scalar+fermion
        muon_residual = abs(muon['residual']-correction)/muon['combined_sigma']
        groups = np.array([m['common_mass_rmse'], muon_residual, c['standardized_absolute_residual']])
        bg = ce.Background(ce.Parameters(m['r'],m['theta_initial'],s_over_Mp2=m['s_over_Mp2']))
        raw_residual = np.array([bg.distances(z)['F_AP']-y for z,y in zip(zs,ap_observed)]
            +[correction-muon['residual'], c['conditional_alpha_fractional_drift_per_year']-clock['observation_per_year']])
        whitened = np.linalg.solve(full_cholesky, raw_residual)
        independent_score = float(np.linalg.norm(whitened)/np.sqrt(8))
        assert abs(independent_score-score(groups)) < 1e-11
        changes = counts*(groups**2-base**2)
        rows.append(dict(r=m['r'],initial_theta=m['theta_initial'],shared_theta_today=theta,
            paired_muon_EM_subset=correction, group_rmse=groups.tolist(),
            delta_chi_squared_by_group=changes.tolist(),
            partial_rmse=score(groups), partial_rmse_change=score(groups)-score(base),
            full_covariance_crosscheck_absolute_error=abs(independent_score-score(groups)),
            all_three_groups_improved=bool(np.all(groups<base))))
    return dict(scope='conditional partial score only', groups=['AP','muon_EM_subset','clock'],
        counts=counts.tolist(), covariance_assumption='independent blocks; no theory uncertainty for missing CE terms',
        baseline_group_rmse=base.tolist(), baseline_partial_rmse=score(base), cases=rows,
        fitted_parameters=0, independent_holdout=False, joint_rmse=None,
        all_partial_scores_worsened=all(row['partial_rmse_change']>0 for row in rows),
        missing=['complete paired interactions and backreaction', 'dark matter growth',
                 'early universe and sound horizon', 'four-force absolute unification',
                 'derived mass scales, kinetic term and initial state'])


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
