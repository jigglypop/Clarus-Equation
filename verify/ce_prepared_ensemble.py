"""Integrate QP27 independent exponential mode energies, without selecting samples."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import roots_laguerre, roots_genlaguerre, roots_jacobi, roots_legendre, logsumexp, gamma
from ce_obs32_profiled_matter import ce
from ce_prepared_radiation import RadiationBackground,MP


def quadrature(order,radial):
    if not radial:
        nodes,weights=roots_laguerre(order)
        return [(nodes[list(i)],float(np.prod(weights[list(i)]))) for i in itertools.product(range(order),repeat=3)]
    nr,na=order
    total,wt=roots_genlaguerre(nr,2);wt=wt/2
    first,wf=roots_jacobi(na,1,0);first=(first+1)/2;wf=wf/2
    second,ws=roots_legendre(na);second=(second+1)/2;ws=ws/2
    rows=[(t*np.array([a,(1-a)*b,(1-a)*(1-b)]),float(w1*w2*w3))
        for t,w1 in zip(total,wt) for a,w2 in zip(first,wf) for b,w3 in zip(second,ws)]
    probabilities=np.array([p for _,p in rows]); factors=np.array([x for x,_ in rows])
    assert abs(probabilities.sum()-1)<1e-13
    assert np.max(abs(probabilities@factors-1))<1e-13
    assert np.max(abs(probabilities@(factors*factors)-2))<1e-12
    return rows


def run(radial=False):
    scales=json.loads(Path(__file__).with_name('ce_obs32_common_mass.json').read_text())
    _,_,_,_,zs,obs,cov=ce.load_data(); chol=np.linalg.cholesky(cov)
    rows=[]
    for r,mass_ev in [(.15,.027615),(.35,.014414)]:
        scale=next(x['potential_scale_GeV4'] for x in scales['cases'] if x['r']==r)
        mass=mass_ev*1e-9;s=mass*mass
        x=s*(1+2*r*np.cos((.5+2*np.pi*np.arange(3))/3))
        hi=1e7;duration=1e11;hr=8.5471961e-45
        variance=3*hi**4/(8*np.pi**2*x)*(-np.expm1(-2*x*duration/(3*hi**2)))
        mean_number=4*gamma(1.25)**2/np.pi*hr**1.5*2*variance/x**.25
        p=ce.Parameters(r,.5,ai=1e-8,s_over_Mp2=(mass/MP)**2)
        orders=[]
        for order in ([(8,2),(16,2),(16,3)] if radial else [3,5]):
            records=[]
            for random_factors,probability in quadrature(order,radial):
                number=mean_number*random_factors
                rho_c=float(number@np.sqrt(x))
                bg=RadiationBackground(p,number/number.sum(),scale,rho_c)
                prediction=np.array([bg.distances(z)['F_AP'] for z in zs])
                residual=np.linalg.solve(chol,prediction-obs)
                records.append((probability,float(residual@residual),prediction))
            probabilities=np.array([a[0] for a in records]);chi=np.array([a[1] for a in records])
            predictions=np.array([a[2] for a in records]);mean=probabilities@predictions
            mean_residual=np.linalg.solve(chol,mean-obs)
            expected_chi=float(probabilities@chi)
            spread=predictions-mean
            whitened=np.linalg.solve(chol,spread.T).T
            variance_term=float(probabilities@np.sum(whitened**2,axis=1))
            assert abs(expected_chi-(mean_residual@mean_residual+variance_term))<1e-10
            row=dict(order=order,background_count=len(records),
                expected_chi_squared=expected_chi,
                root_mean_expected_standardized_squared_error=float(np.sqrt(expected_chi/6)),
                mean_prediction_rmse=float(np.linalg.norm(mean_residual)/np.sqrt(6)),
                normalized_predictive_minus2logL=float(-2*logsumexp(np.log(probabilities)-chi/2)),
                mean_AP_prediction=mean.tolist(),preparation_variance_chi_squared=variance_term)
            orders.append(row)
            print(json.dumps(dict(r=r,**row)),flush=True)
        rows.append(dict(r=r,orders=orders,
            expected_chi_relative_order_change=abs(orders[-1]['expected_chi_squared']/orders[-2]['expected_chi_squared']-1),
            predictive_log_score_order_change=orders[-1]['normalized_predictive_minus2logL']-orders[-2]['normalized_predictive_minus2logL']))
    return dict(scope='radial Gamma / Dirichlet preparation quadrature' if radial else 'tensor Gauss-Laguerre preparation prior predictive diagnostic',N_pre=1e11,
        assumptions='three independent exponential mode energies; homogeneous separate patches with same supplied radiation and phase',
        cases=rows,fitted_parameters=0,joint_rmse=None,
        caveat='Low quadrature orders are diagnostics; order comparison must be examined before claiming converged marginal likelihood. Not a CMB spatial ensemble.')


if __name__=='__main__':
    radial='--radial' in sys.argv
    result=run(radial)
    output=Path(__file__).with_name('ce_prepared_ensemble_radial.json') if radial else Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
