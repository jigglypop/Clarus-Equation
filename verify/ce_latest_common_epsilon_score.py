"""Seven-row conditional score under the actual shared-epsilon premise.

Six AP ratios plus one fixed muon summary; no full-theory or RT31 history
claim. Explicitly replace the old kinetic mass by the supplied physical dark
mass, and derive charged splitting from the SAME dimensionful epsilon.
"""
import json
from pathlib import Path
import numpy as np
from ce_obs32_profiled_matter import ce
from common_spectrum_muon import exact_em


def run():
    *_,zs,observed,cov=ce.load_data()
    chol=np.linalg.cholesky(cov)
    gap=385e-12;sigma=np.hypot(145,620)*1e-12
    mu_base=gap/sigma
    baseline_bg=ce.Background(ce.Parameters(.35,0))
    base=np.array([baseline_bg.distances(z)['F_AP'] for z in zs])
    resid=np.linalg.solve(chol,base-observed)
    baseline=float(np.sqrt((resid@resid+mu_base**2)/7))
    joint_cov=np.zeros((7,7));joint_cov[:6,:6]=cov;joint_cov[6,6]=sigma*sigma
    rows=[]
    for r,mass in [(.15,.027615),(.35,.014414)]:
        sD=(mass*1e-9)**2;eps=r*sD;charged_r=eps/1000**2
        assert 1000**2>2*eps
        for theta0 in [.1,.3,.5]:
            p=ce.Parameters(r,theta0,s_over_Mp2=sD/(2.435e18)**2)
            bg=ce.Background(p)
            pred=np.array([bg.distances(z)['F_AP'] for z in zs])
            theta=bg.quantities(0)['theta']
            correction=exact_em(1000.,charged_r,theta)
            independent=exact_em(1000.,charged_r,theta,representation='spectral')
            assert abs(correction/independent-1)<1e-8
            whitened=np.linalg.solve(chol,pred-observed)
            chiAP=float(whitened@whitened)
            mu=(gap-correction)/sigma
            score=float(np.sqrt((chiAP+mu*mu)/7))
            residual=np.r_[pred-observed,correction-gap]
            raw=float(np.sqrt(residual@np.linalg.solve(joint_cov,residual)/7))
            assert abs(raw-score)<1e-12
            rows.append(dict(r_dark=r,initial_theta=theta0,final_theta=float(theta),
                             dark_mass_eV=mass,shared_epsilon_GeV2=eps,charged_r=charged_r,
                             AP_rmse=float(np.sqrt(chiAP/6)),muon_correction=correction,
                             muon_standardized_residual=float(mu),partial_common_rmse=score,
                             delta_partial_rmse=score-baseline,
                             continuity=bg.source_checks()['relative_total_continuity']))
    return dict(assumption='common dimensionful epsilon and phase; distinct diagonal masses',
                physical_dark_mass_replaces_OBS32_kinetic_mass=True,
                baseline_partial_rmse=baseline,cases=rows,fitted_parameters=[],
                score_dimension=7,cross_block_covariance_assumption='zero AP-muon cross covariance',
                full_joint_rmse=None,independent_holdout=False,
                limitations=['Physical dark masses, f/Mp=1, charged mass 1TeV and initial populations are supplied.',
                             'RT31 rotating generated state is not identified with this late-time state.',
                             'Muon term is scalar electromagnetic subset; heavy relative dark potential is neglected at the previously verified tiny ratio.',
                             'No early-universe H0, clustering or four-force observable likelihood.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
