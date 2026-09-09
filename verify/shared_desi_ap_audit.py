"""Audit rounded AP predictions printed in the original shared conversation.

This does not rerun the missing CE background solver. The six AP ratios use
first-order covariance propagation; they are not the original 13-distance
likelihood or a Hubble-constant determination.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'benchmarks/cosmology/desi_dr2'


def run():
    manifest=json.loads((DATA/'manifest.json').read_text())
    for asset in manifest['assets'].values():
        assert hashlib.sha256((DATA/asset['filename']).read_bytes()).hexdigest()==asset['sha256']
    rows=np.loadtxt(DATA/manifest['assets']['mean']['filename'],dtype=str,skiprows=1)
    cov=np.loadtxt(DATA/manifest['assets']['covariance']['filename'])
    assert cov.shape==(13,13) and np.isfinite(cov).all() and np.allclose(cov,cov.T)
    np.linalg.cholesky(cov)
    redshifts=np.array([.510,.706,.934,1.321,1.484,2.330])
    supplied=np.array([.59457,.87885,1.25350,1.99859,2.35273,4.55019])
    displayed_observed=np.array([.62149,.89182,1.22301,1.94701,2.38058,4.51703])
    displayed_sd=np.array([.01711,.02078,.01916,.04514,.13588,.09694])
    jac=np.zeros((6,13)); observed=[]
    for i,z in enumerate(redshifts):
        match=np.isclose(rows[:,0].astype(float),z,atol=1e-9,rtol=0)
        im=np.flatnonzero(match&(rows[:,2]=='DM_over_rs')).item()
        ih=np.flatnonzero(match&(rows[:,2]=='DH_over_rs')).item()
        dm,dh=float(rows[im,1]),float(rows[ih,1])
        observed.append(dm/dh)
        jac[i,im]=1/dh; jac[i,ih]=-dm/(dh*dh)
    observed=np.array(observed); cap=jac@cov@jac.T
    sd=np.sqrt(np.diag(cap)); chol=np.linalg.cholesky(cap)
    assert np.max(abs(observed-displayed_observed))<5.1e-6
    assert np.max(abs(sd-displayed_sd))<5.1e-6
    def score(pred):
        r=pred-observed; white=np.linalg.solve(chol,r)
        chi=float(white@white)
        independent=float(r@np.linalg.solve(cap,r))
        assert abs(chi-independent)<1e-10
        return dict(chi2=chi,standardized_rmse=float(np.sqrt(chi/6)))
    def lcdm(omega):
        def ez(z):return np.sqrt(omega*(1+z)**3+1-omega)
        return np.array([ez(z)*quad(lambda t:1/ez(t),0,z,epsabs=1e-12,epsrel=1e-12)[0] for z in redshifts])
    fit=minimize_scalar(lambda o:score(lcdm(o))['chi2'],bounds=(.05,.6),method='bounded',options={'xatol':1e-12})
    assert fit.success
    ce=score(supplied); fixed=score(lcdm(.3)); fitted=score(lcdm(fit.x))
    return dict(role='rounded_shared_table_observational_audit',
        shared_source='https://chatgpt.com/share/6aa0e93e-2dec-83ee-b20a-d1ee57fa611d',
        data=manifest,n_distance_inputs=13,n_ap_ratios=6,
        source_solver_rerun=False,ratio_covariance_approximation='first-order J C J^T',
        points=[dict(z=float(z),observed=float(y),sigma=float(err),shared_rounded_prediction=float(p)) for z,y,err,p in zip(redshifts,observed,sd,supplied)],
        ratio_covariance=cap.tolist(),shared_prediction_score=ce,
        flat_lcdm_fixed_omega_03=fixed,
        flat_lcdm_fitted=dict(omega_m=float(fit.x),fitted_parameters=1,**fitted),
        shared_rmse_minus_fixed=ce['standardized_rmse']-fixed['standardized_rmse'],
        shared_rmse_minus_fitted=ce['standardized_rmse']-fitted['standardized_rmse'],
        shared_rmse_rounding_bound=float(5e-6/np.sqrt(np.linalg.eigvalsh(cap).min())),
        excluded_from_ratio_diagnostic='BGS DV datum has no DM/DH pair; retained in source, not scored here',
        limitations=['rounded CE predictions, unavailable background code and parameters',
                     'not the full 13-distance likelihood', 'retrospective comparison, no independent holdout',
                     'no theory covariance or parameter-complexity assessment',
                     'no absolute H0 or sound-horizon prediction'],
        joint_rmse=None,scientific_success=False)


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['shared_prediction_score','flat_lcdm_fixed_omega_03','flat_lcdm_fitted','shared_rmse_minus_fixed','shared_rmse_minus_fitted','shared_rmse_rounding_bound']},indent=2))
