"""New exploratory comparison with equal A and matter-coefficient freedom.

Original CE-OBS32 source and contract stay unchanged. This fit uses development
data; it is not an independent holdout or a joint quantum/cosmology result.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'_workspace/ce_obs32_reproduction/CE-OBS32/observable_bridge.py'
SOURCE_SHA='d6abc29bf1d4b157281a541daea84d088f0161010edb5c3357a01e8d74cf9ae8'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
spec=importlib.util.spec_from_file_location('ce_obs32_original',SOURCE)
ce=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=ce
spec.loader.exec_module(ce)


def run():
    z,y,kind,cov,*_=ce.load_data()
    chol=np.linalg.cholesky(cov);wy=np.linalg.solve(chol,y)
    keys={'DM_over_rs':'DM_H0_over_c','DH_over_rs':'DH_H0_over_c','DV_over_rs':'DV_H0_over_c'}
    def evaluate(r,theta,rm,method='DOP853'):
        bg=ce.Background(ce.Parameters(r,theta,Rm=rm),method=method)
        distances={zz:bg.distances(zz) for zz in set(z)}
        b=np.array([distances[zz][keys[k]] for zz,k in zip(z,kind)])
        wb=np.linalg.solve(chol,b);A=float(wb@wy/(wb@wb))
        pred=A*b;res=np.linalg.solve(chol,pred-y);chi=float(res@res)
        end=bg.quantities(0)
        return dict(r=r,initial_theta=theta,Rm=rm,A=A,chi2=chi,rmse=math.sqrt(chi/13),
                    omega_m_0=1-end['omega_phi'],w_phi_0=end['w_phi'],
                    prediction=pred.tolist(),continuity=bg.source_checks()['relative_total_continuity'])
    cases=[]
    for r,theta in [(.35,0)]+[(r,t) for r in [.15,.35] for t in [.1,.3,.5]]:
        cache={}
        def obj(rm):
            rm=float(rm)
            if rm not in cache:cache[rm]=evaluate(r,theta,rm)
            return cache[rm]['chi2']
        grid=np.linspace(.1,1.2,9)
        values=[obj(x) for x in grid]
        intervals=[(grid[i-1],grid[i+1]) for i in range(1,len(grid)-1)
                   if values[i]<=values[i-1] and values[i]<=values[i+1]]
        for lo,hi in intervals:
            fit=minimize_scalar(obj,bounds=(lo,hi),method='bounded',options={'xatol':1e-8})
            assert fit.success
        best=min(cache.values(),key=lambda x:x['chi2'])
        assert .1<best['Rm']<1.2,'Optimum at search boundary; expand before claiming a profile'
        h=1e-4
        curvature=(obj(best['Rm']+h)+obj(best['Rm']-h)-2*best['chi2'])/h**2
        gradient=(obj(best['Rm']+h)-obj(best['Rm']-h))/(2*h)
        assert curvature>0 and abs(gradient)<.02
        best.update(grid_Rm=grid.tolist(),grid_chi2=values,profile_curvature=curvature,
                    profile_gradient=gradient,evaluations=len(cache),fitted_parameters=['Rm','A'])
        cases.append(best)
        print(json.dumps({k:best[k] for k in ['r','initial_theta','Rm','chi2','rmse']},ensure_ascii=False),flush=True)
    null=cases[0]
    for case in cases:
        case['delta_chi2_from_fitted_null']=case['chi2']-null['chi2']
        case['delta_rmse_from_fitted_null']=case['rmse']-null['rmse']
    best=min(cases[1:],key=lambda x:x['chi2'])
    independent=evaluate(best['r'],best['initial_theta'],best['Rm'],method='Radau')
    prediction_error=float(np.max(abs(np.array(best['prediction'])-independent['prediction'])))
    assert prediction_error<1e-7 and abs(independent['chi2']-best['chi2'])<1e-7
    assert all(x['continuity']<1e-11 for x in cases)
    return dict(role='exploratory_equal_parameter_profile',source_sha256=SOURCE_SHA,
        original_contract_unchanged=True,original_fixed_Rm=3/7,new_Rm_search_range=[.1,1.2],
        common_fitted_parameters=['physical matter coefficient Rm','one global ruler amplitude A'],
        fixed_candidate_grid='original six r/theta points; selected retrospectively for comparison',
        data_rows=13,cases=cases,independent_Radau_prediction_max_abs=prediction_error,
        independent_Radau_chi2_difference=independent['chi2']-best['chi2'],
        independent_holdout=False,joint_quantum_macro_rmse=None,scientific_success=False,
        limitations=['profile over supplied Rm and calibrated A, not a prediction of their origin',
                     'bounded grid/refinement search, not proof of global optimality',
                     'neutral cold late-time model only; no added charged-sector matching',
                     'no CMB, absolute H0, initial generation or full joint likelihood',
                     'no correction for selecting among candidate hypotheses'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
