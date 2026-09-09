"""Conditional clock test of an unscreened common cosmological phase.

Neutral CE-OBS32 background plus exactly paired charged test sector.
Missing full interactions, local environmental response and higher-loop backreaction.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from ce_obs32_profiled_matter import ce, SOURCE_SHA
from common_spectrum_muon import ALPHA, Q2


def run():
    reference = json.loads(Path(__file__).with_name('ce_obs32_common_mass.json').read_text())
    assert reference['source_sha256'] == SOURCE_SHA
    observed, sigma = 1e-18, 1.1e-18  # Lange et al., PRL 126, 011102 (2021).
    charged_r = .025
    rows = []
    for item in reference['cases']:
        bg = ce.Background(ce.Parameters(item['r'], item['theta_initial'], s_over_Mp2=item['s_over_Mp2']))
        end = bg.quantities(0)
        theta, velocity = end['theta'], end['v']
        determinant = 1-3*charged_r**2+2*charged_r**3*np.cos(theta)
        # Paired charged sector = two complex scalars + one Dirac, threshold factor 6.
        slope = 6*Q2/(12*np.pi)*2*charged_r**3*np.sin(theta)/determinant
        with mp.workdps(50):
            r = mp.mpf(str(charged_r))
            def inverse_alpha_shift(t):
                roots = [1+2*r*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
                return -6*mp.mpf(str(Q2))/(12*mp.pi)*mp.log(mp.fprod(roots)/((1-2*r)*(1+r)**2))
            independent = float(mp.diff(inverse_alpha_shift, mp.mpf(str(theta))))
        assert np.isclose(slope, independent, rtol=1e-12, atol=0)
        h_per_year = item['conditional_H0_km_s_Mpc']/3.0856775814913673e19*31557600
        drift = -ALPHA*slope*velocity*h_per_year
        rows.append(dict(r=item['r'], initial_theta=item['theta_initial'], theta_today=theta,
            dtheta_dN_today=velocity, d_inverse_alpha_dtheta=float(slope),
            conditional_alpha_fractional_drift_per_year=float(drift),
            standardized_absolute_residual=float(abs(drift-observed)/sigma)))
    return dict(source='https://arxiv.org/abs/2010.06620', observation_per_year=observed,
        sigma_per_year=sigma, baseline_zero_drift_residual=abs(observed)/sigma,
        charged_r=charged_r, paired_threshold_factor=6, fitted_parameters=0,
        assumptions=['same phase and cosmic time dependence locally', 'no environmental screening',
            'constant reference electromagnetic coupling; its present value is input',
            'exact charged pairing at one loop; no complete supersymmetric completion'],
        cases=rows, joint_rmse=None, independent_holdout=False,
        conclusion='Conditional cross-sector comparison; not a completed unified likelihood.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
