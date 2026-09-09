"""Post-comparison diagnostics only; no parameter optimization is applied."""
from pathlib import Path
import json
import numpy as np
from observable_bridge import load_data
ROOT=Path(__file__).resolve().parent

def main():
    d=json.loads((ROOT/'results/observable_results.json').read_text())
    z,obs,k,C,zs,rat,Cr=load_data(); Wi=np.linalg.inv(Cr)
    base=np.array(d['null']['F_AP']); e=base-rat
    directions=[]
    full_directions=[]
    Ci=np.linalg.inv(C)
    A0=d['null']['profiled_ruler_A']
    b0=np.array(d['null']['full13_model_with_calibrated_A'])/A0
    r0=A0*b0-obs
    for r in [.15,.35]:
        small=next(x for x in d['cases'] if x['r']==r and x['initial_theta']==.1)
        v=(np.array(small['F_AP'])-base)/.1**2
        db=(np.array(small['full13_model_with_calibrated_A'])/small['profiled_ruler_A']-b0)/.1**2
        full_directions.append(dict(r=r,first_order_profiled_chi2_derivative_wrt_theta_squared=float(2*A0*db@Ci@r0),scope='approximate local direction from fixed .1 perturbation, with one global ruler amplitude profiled'))
        A=float(v@Wi@v); B=float(2*e@Wi@v)
        directions.append(dict(r=r,direction_estimated_at_theta=.1,linear_term_B=B,quadratic_term_A=A,
            unconstrained_delta_squared_stationary_point=-B/(2*A),physical_domain='delta_squared>=0',
            conclusion='Local approximate direction only; not a global model rejection or fit'))
    profiles=[]
    for case in [dict(d['null'],r=None,initial_theta=0)]+d['cases']:
        chi=0.
        for zz,F in zip(zs,case['F_AP']):
            iM=np.flatnonzero((z==zz)&(k=='DM_over_rs'))[0]
            iH=np.flatnonzero((z==zz)&(k=='DH_over_rs'))[0]
            chi+=(obs[iM]-F*obs[iH])**2/(C[iM,iM]+F*F*C[iH,iH]-2*F*C[iM,iH])
        profiles.append(dict(r=case['r'],initial_theta=case['initial_theta'],
            exact_pair_amplitude_profile_chi2=chi,
            first_order_ratio_chi2=case['AP_chi2_first_order']))
    # The supplied covariance has no cross-redshift blocks, verified here.
    cross=np.array([[C[i,j] if z[i]!=z[j] else 0 for j in range(13)] for i in range(13)])
    assert np.max(np.abs(cross))==0
    result=dict(residual_directions=directions,full13_profiled_directions=full_directions,ratio_profile_crosscheck=profiles,
        warning='Ratio profile eliminates six measurement amplitudes for a geometry-only robustness check. These are not six parameters fitted in the CE action. The separate full13 result profiles only one global ruler amplitude. No combined p-value, Bayes factor, or model selection is claimed.')
    (ROOT/'results/postcomparison_diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
