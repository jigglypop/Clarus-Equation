"""Retrospective leave-one-redshift-block-out diagnostic, not independent data.

The candidate was selected using the full dataset in the previous exploration.
This calculation therefore cannot certify unbiased model selection or holdout
success. It checks whether the small in-sample gain survives block omission.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from ce_obs32_profiled_matter import ce, SOURCE_SHA


def run():
    z,y,kind,cov,*_=ce.load_data()
    cross=cov.copy()
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i]==z[j]:cross[i,j]=0
    assert np.max(abs(cross))==0,'Need conditional covariance for correlated blocks'
    keys={'DM_over_rs':'DM_H0_over_c','DH_over_rs':'DH_H0_over_c','DV_over_rs':'DV_H0_over_c'}
    output=[]
    for theta in [0.,.5]:
        r=.35
        shape_cache={}
        def physical(rm):
            rm=float(rm)
            if rm not in shape_cache:
                bg=ce.Background(ce.Parameters(r,theta,Rm=rm))
                distances={zz:bg.distances(zz) for zz in set(z)}
                b=np.array([distances[zz][keys[k]] for zz,k in zip(z,kind)])
                shape_cache[rm]=b
            return shape_cache[rm]
        folds=[]; all_test_predictions=np.zeros(len(y))
        for held_z in sorted(set(z)):
            test=np.flatnonzero(z==held_z);train=np.flatnonzero(z!=held_z)
            assert not set(test)&set(train) and len(test)+len(train)==13
            lc=np.linalg.cholesky(cov[np.ix_(train,train)])
            wy=np.linalg.solve(lc,y[train])
            evaluations={}
            def score(rm):
                rm=float(rm)
                b=physical(rm);wb=np.linalg.solve(lc,b[train])
                A=float(wb@wy/(wb@wb));res=A*wb-wy
                chi=float(res@res)
                evaluations[rm]=(chi,A)
                return chi
            grid=np.linspace(.1,1.2,9); values=[score(t) for t in grid]
            for i in range(1,len(grid)-1):
                if values[i]<=values[i-1] and values[i]<=values[i+1]:
                    fit=minimize_scalar(score,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-8})
                    assert fit.success
            rm=min(evaluations,key=lambda t:evaluations[t][0])
            assert .1<rm<1.2,'Profile minimum at boundary'
            chi,A=evaluations[rm];pred=A*physical(rm)
            residual=pred[test]-y[test];ct=cov[np.ix_(test,test)]
            wr=np.linalg.solve(np.linalg.cholesky(ct),residual)
            test_chi=float(wr@wr)
            assert abs(test_chi-float(residual@np.linalg.solve(ct,residual)))<1e-10
            all_test_predictions[test]=pred[test]
            bg=ce.Background(ce.Parameters(r,theta,Rm=rm))
            conservation=bg.source_checks()['relative_total_continuity']
            assert conservation<1e-11
            folds.append(dict(redshift=float(held_z),train_rows=train.tolist(),test_rows=test.tolist(),
                              Rm=rm,A=A,training_chi2=chi,held_block_chi2=test_chi,
                              test_prediction=pred[test].tolist(),continuity=conservation))
        total=sum(f['held_block_chi2'] for f in folds)
        residual=all_test_predictions-y
        assert abs(total-float(residual@np.linalg.solve(cov,residual)))<1e-9
        output.append(dict(r=r,initial_theta=theta,folds=folds,
                           held_blocks_chi2_sum=total,held_blocks_rmse=math.sqrt(total/13),
                           predictions_from_different_training_fits=all_test_predictions.tolist()))
        print(json.dumps({k:output[-1][k] for k in ['initial_theta','held_blocks_chi2_sum','held_blocks_rmse']}),flush=True)
    deltas=[dict(redshift=a['redshift'],candidate_minus_baseline_chi2=b['held_block_chi2']-a['held_block_chi2'])
            for a,b in zip(output[0]['folds'],output[1]['folds'])]
    return dict(role='retrospective_block_omission_diagnostic',source_sha256=SOURCE_SHA,
                common_fitted_parameters=['Rm','A'],cases=output,block_differences=deltas,
                delta_held_blocks_rmse=output[1]['held_blocks_rmse']-output[0]['held_blocks_rmse'],
                every_block_improved=all(x['candidate_minus_baseline_chi2']<0 for x in deltas),
                independent_holdout=False,model_selection_used_all_data=True,
                joint_quantum_macro_rmse=None,scientific_success=False,
                limitations=['same development dataset; candidate selection not nested',
                             'fold scores use different fitted parameters, not a single frozen universe',
                             'not a full quantum, CMB, Hubble or force-unification comparison'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
