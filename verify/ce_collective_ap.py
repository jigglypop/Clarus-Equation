"""Fixed-input AP test of the leading collective action, not a full EFT fit.

The six-channel W and derived F are from chapter 13. Cold gear occupation is
represented by its leading conserved adiabatic energy. Present time and initial
abundance are supplied for this new benchmark, not predicted by chapter 13.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp, quad
from ce_obs32_profiled_matter import ce

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'_workspace/ce_collective_original'))
from collective_phase_check import Loop


def run():
    loop=Loop()
    phi0=np.pi-.6
    W0=loop.values(phi0)[0]
    F2=.05**2*sum(3.**(2*k) for k in range(5))
    Rm=3/7
    ai=.01
    *_,z,obs,cov=ce.load_data()
    chol=np.linalg.cholesky(cov)
    corrected=False
    c=.12485181260408908
    mass_corrected=False
    chain_N=4
    lam=10-6*np.cos(np.pi/(chain_N+1))
    overlap=18*np.sin(np.pi/(chain_N+1))**2/((chain_N+1)*lam)

    def matter_terms(n,phi):
        total=Rm*np.exp(-3*n)
        if not mass_corrected:return total,0.
        _,_,W2=loop.values(phi)
        sine,cosine=np.sin(phi),np.cos(phi)
        b=loop.z;den=1+b*(1-cosine)
        W3=float(loop.weight@(-b*sine/den-3*b*b*sine*cosine/den**2
                              +2*b**3*sine**3/den**3))/loop.curv
        x=1+overlap*W2/(128*lam)
        x0=1+overlap*loop.values(phi0)[2]/(128*lam)
        assert min(x,x0)>0
        m=np.sqrt(x/x0)
        charge=overlap*W3/(256*lam*x)
        rho_c=.84*total*m
        return .16*total+rho_c,rho_c*charge

    def potential(phi):
        W,W1,W2=loop.values(phi)
        if corrected:
            _,initial_W1,_=loop.values(phi0)
            norm=W0-c*initial_W1**2/(2*128)
            return (W-c*W1**2/(2*128))/norm,(W1-c*W1*W2/128)/norm
        return W/W0,W1/W0

    def rhs(n,y):
        phi,v=y
        U,U1=potential(phi)
        matter,force=matter_terms(n,phi)
        H2=(matter+U)/(3-.5*F2*v*v)
        hprime=-matter/(2*H2)-.5*F2*v*v
        return [v,-(3+hprime)*v-(U1+force)/(F2*H2)]

    def evaluate(method):
        sol=solve_ivp(rhs,(np.log(ai),0),[phi0,0.],method=method,
                      rtol=2e-10,atol=2e-12,dense_output=True)
        assert sol.success
        def H2(n):
            phi,v=sol.sol(n)
            return (matter_terms(n,phi)[0]+potential(phi)[0])/(3-.5*F2*v*v)
        def E(redshift):
            return np.sqrt(H2(-np.log1p(redshift))/H2(0))
        pred=np.array([E(zz)*quad(lambda x:1/E(x),0,zz,epsabs=1e-11)[0] for zz in z])
        residual=np.linalg.solve(chol,pred-obs)
        phi,v=sol.sol(0)
        kinetic=.5*F2*H2(0)*v*v
        U=potential(phi)[0]
        # Differential energy conservation, computed from the independent terms.
        defects=[]
        for n in np.linspace(np.log(ai),0,80):
            ph,vel=sol.sol(n);acc=rhs(n,[ph,vel])[1]
            u,u1=potential(ph)
            matter,force=matter_terms(n,ph);hh=H2(n)
            hp=-matter/(2*hh)-.5*F2*vel*vel
            derivative=F2*hh*(hp*vel*vel+vel*acc)+u1*vel-3*matter+force*vel
            defects.append(abs(derivative+3*(F2*hh*vel*vel+matter))/(3*hh))
        assert max(defects)<1e-12
        return dict(prediction=pred.tolist(),ap_rmse=float(np.sqrt(residual@residual/len(z))),
                    omega_m_today=float(matter_terms(0,phi)[0]/(3*H2(0))),w_slow_today=float((kinetic-U)/(kinetic+U)),
                    phase_today=float(phi),continuity_max_relative=max(defects))

    result=evaluate('DOP853')
    independent=evaluate('Radau')
    diff=max(abs(np.array(result['prediction'])-independent['prediction']))
    assert diff<1e-8
    corrected=True
    correction_result=evaluate('DOP853')
    mass_corrected=True
    mass_results=[]
    for chain_N in [4,12,20,28,36]:
        lam=10-6*np.cos(np.pi/(chain_N+1))
        overlap=18*np.sin(np.pi/(chain_N+1))**2/((chain_N+1)*lam)
        D=np.eye(chain_N,chain_N+1)-3*np.eye(chain_N,chain_N+1,k=1)
        endpoint=np.zeros(chain_N+1);endpoint[-1]=1
        bvec=np.linalg.solve(D@D.T,D@endpoint)
        c=float(bvec@bvec)
        row=evaluate('DOP853')
        check=evaluate('Radau')
        error=float(np.max(abs(np.array(row['prediction'])-check['prediction'])))
        assert error<1e-8
        f2=F2/sum(3.**(2*k) for k in range(chain_N+1))
        norm=W0-c*loop.values(phi0)[1]**2/(256)
        m2=(128*lam+overlap*loop.values(phi0)[2])/(f2*norm)
        initial_H2=(Rm/ai**3+1)/3
        left_mode=np.sqrt(2/(chain_N+1))*np.sin(np.pi*np.arange(1,chain_N+1)/(chain_N+1))
        # Turning-point link excursion implied by the harmonic occupation.
        # Large excursions invalidate replacing the cosine gear by cold dust,
        # even if its small-amplitude frequency is much larger than H.
        excursion=float(np.sqrt(2*.84*Rm/ai**3*norm/128)*np.max(abs(left_mode)))
        row.update(N=chain_N,initial_gear_mass_over_H=float(np.sqrt(m2/initial_H2)),
                   initial_harmonic_link_excursion=excursion,
                   small_amplitude_condition_met=bool(excursion<.1),
                   independent_AP_difference=error)
        mass_results.append(row)
    def base_E(zz):return np.sqrt(.3*(1+zz)**3+.7)
    base=np.array([base_E(zz)*quad(lambda x:1/base_E(x),0,zz)[0] for zz in z])
    residual=np.linalg.solve(chol,base-obs)
    baseline=float(np.sqrt(residual@residual/len(z)))
    return dict(inputs=dict(initial_scale_factor=ai,initial_phase=phi0,initial_phase_velocity=0,
                           initial_U_normalization=1,matter_coefficient=Rm,leading_F_squared=F2),
                approximation='Leading collective W plus conserved cold gear energy; baryon fraction not separately constrained by AP.',
                baseline_rmse=baseline,candidate=result,delta_rmse=result['ap_rmse']-baseline,
                first_potential_correction=correction_result,
                corrected_delta_rmse=correction_result['ap_rmse']-baseline,
                mass_exchange_cases=mass_results,
                independent_method_max_AP_difference=diff,fitted_parameters=[],joint_rmse=None,
                limitations=['Abundance and reference present epoch supplied, not selected by the common principle.',
                             'First vacuum-Hessian gear mass correction and energy exchange included in additional cases; higher corrections, radiation and recombination omitted.',
                             'Cold adiabatic treatment requires gear mass/H much greater than one; inspect the initial ratio for each N.',
                             'All tested initial harmonic occupations have link excursions above one radian: the dust approximation is not a controlled full-chain prediction.',
                             'No quantum or clustering likelihood; not the original synthetic initial state.'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
