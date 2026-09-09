"""Full five-phase late-time benchmark with supplied high initial occupation.

No cold-dust replacement of the gear, no parameter fit. Same initial energy
coefficients as ce_collective_ap; the baryon component is pressureless.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import root, brentq
from ce_obs32_profiled_matter import ce

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'_workspace/ce_collective_original'))
from collective_phase_check import Loop, difference


def run(method='DOP853',rtol=2e-12):
    loop=Loop();N=4;q=3;f=.05;R=128.;ai=.01;Rm=3/7;fc=.84
    D=difference(N,q);ell=q**np.arange(N,-1,-1,dtype=float)
    B=null_space(ell[None,:]);vals,vec=np.linalg.eigh(B.T@D.T@D@B)
    mode=B@vec[:,0];phi0=np.pi-.6
    def raw(theta):
        W,W1,_=loop.values(theta[-1]);link=D@theta
        grad=R*D.T@np.sin(link);grad[-1]+=W1
        return 2*R*np.sum(np.sin(link/2)**2)+W,grad
    relaxed=root(lambda eta:B.T@raw(ell*phi0+B@eta)[1],np.zeros(N),tol=1e-10)
    assert np.linalg.norm(relaxed.fun)<1e-8
    theta0=ell*phi0+B@relaxed.x
    norm=raw(theta0)[0]
    kinetic0=fc*Rm/ai**3
    velocity0=np.sqrt(2*kinetic0)/f*mode
    H0=np.sqrt((Rm/ai**3+1)/3)
    def rhs(t,y):
        theta,velocity,n,H=y[:5],y[5:10],y[10],y[11]
        V,grad=raw(theta);V/=norm;grad/=norm
        baryons=(1-fc)*Rm*np.exp(-3*n)
        return np.r_[velocity,-3*H*velocity-grad/f**2,H,
                     -.5*(f*f*(velocity@velocity)+baryons),np.exp(-n)]
    def stop(t,y):return y[10]
    stop.terminal=True;stop.direction=1
    sol=solve_ivp(rhs,(0,30),np.r_[theta0,velocity0,np.log(ai),H0,0.],
                  method=method,rtol=rtol,atol=rtol*.01,dense_output=True,events=stop)
    assert sol.success and len(sol.t_events[0])==1,sol.message
    endtime=sol.t_events[0][0];end=sol.sol(endtime)
    *_,z,obs,cov=ce.load_data()
    predictions=[]
    for zz in z:
        n=-np.log1p(zz)
        t=brentq(lambda t:sol.sol(t)[10]-n,0,endtime,xtol=1e-12)
        y=sol.sol(t)
        predictions.append(y[11]*(end[12]-y[12]))
    predictions=np.array(predictions)
    residual=np.linalg.solve(np.linalg.cholesky(cov),predictions-obs)
    constraint=[]
    for t in np.linspace(0,endtime,2001):
        y=sol.sol(t);V,_=raw(y[:5]);rho=.5*f*f*(y[5:10]@y[5:10])+V/norm+(1-fc)*Rm*np.exp(-3*y[10])
        constraint.append(abs(3*y[11]**2-rho)/rho)
    assert max(constraint)<2e-5
    collective_phi=float(ell@end[:5]/(ell@ell))
    collective_v=float(ell@end[5:10]/(ell@ell))
    transverse_v=end[5:10]-ell*collective_v
    link_potential=2*R*np.sum(np.sin(D@end[:5]/2)**2)/norm
    fast_diag=.5*f*f*(transverse_v@transverse_v)+link_potential
    return dict(method=method,rtol=rtol,inputs=dict(N=N,q=q,f=f,R_H=R,ai=ai,Rm=Rm,fc=fc,
                initial_phi=phi0,initial_potential=1,initial_fast_kinetic=kinetic0),
                source='full cosine chain; lowest massive eigenmode initial velocity',
                ap_prediction=predictions.tolist(),ap_rmse=float(np.sqrt(residual@residual/len(z))),
                final_phase=collective_phi,final_H=float(end[11]),
                final_fast_energy_diagnostic=float(fast_diag),cold_extrapolation_energy=fc*Rm,
                friedmann_max_relative=float(max(constraint)),nfev=sol.nfev,
                fitted_parameters=[],joint_rmse=None,
                limitations=['Initial occupation, current epoch and baryons supplied.',
                             'Fast energy decomposition at final instant is diagnostic, not time averaged.',
                             'No radiation, quantum production, spatial perturbations or muon likelihood.'])


if __name__=='__main__':
    result=run()
    tighter=run(rtol=2e-13)
    difference=float(np.max(abs(np.array(result['ap_prediction'])-tighter['ap_prediction'])))
    assert difference<2e-6
    result['tighter_check']=tighter
    result['tolerance_AP_max_difference']=difference
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
