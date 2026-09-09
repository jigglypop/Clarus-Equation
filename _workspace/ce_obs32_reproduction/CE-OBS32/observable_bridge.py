"""CE spectral potential -> background -> measured BAO geometry.

Exploratory check of existing chapter-22 benchmarks, NOT an absolute cosmology
prediction, independent holdout, or validated BAO-template likelihood for CE.
The physics solver has no data argument. Report every fixed benchmark.
Run: OPENBLAS_NUM_THREADS=1 python -W error observable_bridge.py
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, platform, csv
import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from scipy.special import hyp2f1
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent

@dataclass(frozen=True)
class Parameters:
    r: float
    theta_initial: float
    f: float = 1.0
    s_over_Mp2: float = 1e-12
    Rm: float = 3/7
    fc: float = .84
    ai: float = .01

class Spectrum:
    def __init__(self,p:Parameters):
        if not (0<p.r<.5 and p.f>0 and 0<p.ai<1):
            raise ValueError('Positive-mass, positive-kinetic branch required')
        self.p=p
        self.ref=self.raw(np.pi)
        self.top=self.raw(0.)-self.ref
        self.mnorm=np.sqrt(self.eigs(0.)[0]).sum()
        if not self.top>0: raise ValueError('Nonpositive potential normalization')
    def eigs(self,theta):
        a=(theta+2*np.pi*np.arange(3))/3
        return (1+2*self.p.r*np.cos(a),-2*self.p.r*np.sin(a)/3,
                -2*self.p.r*np.cos(a)/9)
    def raw(self,theta):
        x,_,_=self.eigs(theta)
        return np.sum(x*x*(np.log(x)-1.5))/(32*np.pi**2)
    def evaluate(self,theta):
        x,x1,x2=self.eigs(theta); m=np.sqrt(x)
        U=(self.raw(theta)-self.ref)/self.top
        U1=np.sum(2*x*(np.log(x)-1)*x1)/(32*np.pi**2*self.top)
        U2=np.sum(2*np.log(x)*x1*x1+2*x*(np.log(x)-1)*x2)/(32*np.pi**2*self.top)
        mm=m.sum()/self.mnorm
        mm1=np.sum(x1/(2*m))/self.mnorm
        mm2=np.sum(x2/(2*m)-x1*x1/(4*m**3))/self.mnorm
        zfac=self.p.s_over_Mp2/(96*np.pi**2)
        Z=self.p.f**2+zfac*np.sum(x1*x1/x)
        Z1=zfac*np.sum(2*x1*x2/x-x1**3/x**2)
        return float(U),float(U1),float(U2),float(mm),float(mm1),float(mm2),float(Z),float(Z1)
    def initial_velocity(self):
        # Chapter-22 regular linear mode, scaled to theta(ai)=theta_initial.
        _,_,u2,_,_,m2,Z,_=self.evaluate(0.)
        A=-4*u2/(3*Z);B=-4*self.p.fc*m2/(3*Z)
        K=np.sqrt(1+A);nu=(np.sqrt(1+4*B)-1)/4
        aa=nu+(1-K)/2;bb=nu+(1+K)/2;cc=2*nu+1.5
        X=self.p.ai**3/self.p.Rm
        F=hyp2f1(aa,bb,cc,-X)
        slope=3*nu-3*X*aa*bb/cc*hyp2f1(aa+1,bb+1,cc+1,-X)/F
        return float(self.p.theta_initial*slope)

class Background:
    def __init__(self,p:Parameters,method='DOP853',rtol=2e-10,atol=2e-12):
        self.p=p;self.sp=Spectrum(p)
        self.Ni=np.log(p.ai)
        y0=[p.theta_initial,self.sp.initial_velocity()]
        self.sol=solve_ivp(self.rhs,(self.Ni,0.),y0,method=method,rtol=rtol,atol=atol,dense_output=True,max_step=.08)
        if not self.sol.success: raise RuntimeError(self.sol.message)
        self.H0=self.quantities(0.)['H']
    def at(self,N,theta,v):
        U,U1,U2,m,m1,m2,Z,Z1=self.sp.evaluate(theta)
        rho_b=(1-self.p.fc)*self.p.Rm*np.exp(-3*N)
        rho_c=self.p.fc*self.p.Rm*np.exp(-3*N)*m
        den=3-.5*Z*v*v
        if den<=0: raise RuntimeError('Friedmann denominator is nonpositive')
        H2=(rho_b+rho_c+U)/den
        if H2<=0: raise RuntimeError('Nonpositive H squared')
        kinetic=.5*Z*H2*v*v
        hN=-(rho_b+rho_c+2*kinetic)/(2*H2)
        return dict(U=U,U1=U1,U2=U2,m=m,m1=m1,m2=m2,Z=Z,Z1=Z1,
                    rho_b=rho_b,rho_c=rho_c,H=np.sqrt(H2),H2=H2,
                    kinetic=kinetic,hN=hN,q=-1-hN,
                    w_phi=(kinetic-U)/(kinetic+U),
                    omega_phi=(kinetic+U)/(3*H2),theta=theta,v=v)
    def rhs(self,N,y):
        theta,v=y;a=self.at(N,theta,v)
        acc=-(3+a['hN'])*v-.5*a['Z1']/a['Z']*v*v-(a['U1']+a['rho_c']*a['m1']/a['m'])/(a['Z']*a['H2'])
        return [v,acc]
    def quantities(self,N):
        if N<self.Ni-1e-12 or N>1e-12: raise ValueError('Outside calculated history')
        theta,v=self.sol.sol(N)
        return self.at(N,float(theta),float(v))
    def E(self,z):
        return self.quantities(-np.log1p(z))['H']/self.H0
    def distances(self,z):
        dm=quad(lambda x:1/self.E(x),0,float(z),epsabs=2e-10,epsrel=2e-10)[0]
        dh=1/self.E(z)
        return {'DM_H0_over_c':dm,'DH_H0_over_c':dh,'DV_H0_over_c':(z*dm*dm*dh)**(1/3), 'F_AP':dm/dh}
    def source_checks(self):
        max_cons=0.;max_dH=0.
        for N in np.linspace(self.Ni+.02,-.02,120):
            a=self.quantities(N);v=a['v'];dv=self.rhs(N,[a['theta'],v])[1]
            # Analytic total energy derivative includes the mass-change source.
            rbN=-3*a['rho_b'];rcN=-3*a['rho_c']+a['rho_c']*a['m1']/a['m']*v
            KN=.5*a['Z1']*v*a['H2']*v*v+a['Z']*a['H2']*(a['hN']*v*v+v*dv)
            dR=rbN+rcN+KN+a['U1']*v
            wanted=-3*(a['rho_b']+a['rho_c']+2*a['kinetic'])
            max_cons=max(max_cons,abs(dR-wanted)/max(abs(wanted),1e-30))
            h=1e-4
            fd=(np.log(self.quantities(N-2*h)['H'])-8*np.log(self.quantities(N-h)['H'])+8*np.log(self.quantities(N+h)['H'])-np.log(self.quantities(N+2*h)['H']))/(12*h)
            max_dH=max(max_dH,abs(fd-a['hN']))
        return {'relative_total_continuity':max_cons,'H_derivative_finite_difference_abs':max_dH}

def loop_checks(sp:Spectrum):
    errors=[];derivative=[]
    for t in [.0,.4,1.2,2.4,3.0]:
        # Independent positive integral over common mass source, chapter22.
        r=sp.p.r;K=2*r**3*(1+np.cos(t))
        def fun(y):
            B=(y-2*r)*(y+r)**2
            return (y-1)*np.log1p(K/B)/(16*np.pi**2)
        integ=quad(fun,1,np.inf,epsabs=1e-14,epsrel=2e-12)[0]
        raw=(sp.raw(t)-sp.ref)
        errors.append(abs(raw-integ)/max(abs(integ),1e-20))
        h=2e-4
        fd=(sp.evaluate(t-2*h)[0]-8*sp.evaluate(t-h)[0]+8*sp.evaluate(t+h)[0]-sp.evaluate(t+2*h)[0])/(12*h)
        derivative.append(abs(fd-sp.evaluate(t)[1]))
    return {'loop_integral_relative_max':max(errors),'Uprime_finite_difference_abs_max':max(derivative)}

def cosmic_check(bg:Background):
    p=bg.p;sp=bg.sp;a0=bg.quantities(bg.Ni);v0=a0['v']*a0['H']
    def rhs(t,y):
        a,th,v,H=y;U,U1,_,m,m1,_,Z,Z1=sp.evaluate(th)
        rb=(1-p.fc)*p.Rm/a**3;rc=p.fc*p.Rm*m/a**3
        return [a*H,v,-3*H*v-.5*Z1/Z*v*v-(U1+rc*m1/m)/Z,-.5*(rb+rc+Z*v*v)]
    def event(t,y):return y[0]-1
    event.terminal=True;event.direction=1
    sol=solve_ivp(rhs,(0,10),[p.ai,p.theta_initial,v0,a0['H']],events=event,rtol=2e-12,atol=2e-14,method='DOP853',max_step=.002)
    if not sol.success or not len(sol.t_events[0]):raise RuntimeError('Cosmic-time check failed')
    y=sol.y_events[0][0];end=bg.quantities(0.)
    target=np.array([end['theta'],end['v']*end['H'],end['H']])
    return {'theta_velocity_H_relative_max':float(np.max(np.abs(y[1:]-target)/np.maximum(np.abs(target),1e-12))),'cosmic_final_H':float(y[3])}

def load_data():
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    for item in manifest:
        data=(ROOT/'data'/item['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Data checksum mismatch')
    raw=np.loadtxt(ROOT/'data/desi_gaussian_bao_ALL_GCcomb_mean.txt',dtype=str,skiprows=1)
    z=raw[:,0].astype(float);obs=raw[:,1].astype(float);kind=raw[:,2]
    C=np.loadtxt(ROOT/'data/desi_gaussian_bao_ALL_GCcomb_cov.txt')
    if C.shape!=(13,13) or not np.allclose(C,C.T):raise ValueError('Covariance structure')
    np.linalg.cholesky(C)
    zs=sorted(set(z[kind=='DM_over_rs']))
    J=np.zeros((len(zs),len(obs)));rat=[]
    for j,zz in enumerate(zs):
        iM=np.flatnonzero((z==zz)&(kind=='DM_over_rs'))[0]
        iH=np.flatnonzero((z==zz)&(kind=='DH_over_rs'))[0]
        rat.append(obs[iM]/obs[iH]);J[j,iM]=1/obs[iH];J[j,iH]=-obs[iM]/obs[iH]**2
    Cr=J@C@J.T
    return z,obs,kind,C,np.array(zs),np.array(rat),Cr

def main():
    contract=json.loads((ROOT/'calculation_contract.json').read_text())
    z,obs,kind,C,zs,rat,Cr=load_data()
    Ci=np.linalg.inv(C);Cri=np.linalg.inv(Cr)
    null=Background(Parameters(.35,0.))
    null_dist={zz:null.distances(zz) for zz in sorted(set(z))}
    null_ap=np.array([null_dist[zz]['F_AP'] for zz in zs])
    def stat(bg):
        dd={zz:bg.distances(zz) for zz in sorted(set(z))}
        pred=np.array([dd[zz]['F_AP'] for zz in zs])
        resid=pred-rat
        # One global A=c/(H0*rd) is a CALIBRATION parameter, not predicted H0.
        b=np.array([dd[zz][{'DM_over_rs':'DM_H0_over_c','DH_over_rs':'DH_H0_over_c','DV_over_rs':'DV_H0_over_c'}[k]] for zz,k in zip(z,kind)])
        A=float(b@Ci@obs/(b@Ci@b));res=A*b-obs
        return {'F_AP':pred.tolist(),'AP_chi2_first_order':float(resid@Cri@resid),
                'distance_from_fixed_null_in_AP_covariance':float(np.sqrt((pred-null_ap)@Cri@(pred-null_ap))),
                'profiled_ruler_A':A,'A_sigma_fixed_shape':float(1/np.sqrt(b@Ci@b)),
                'profiled_full13_chi2':float(res@Ci@res),'full13_model_with_calibrated_A':(A*b).tolist()}
    results={'contract_sha256':hashlib.sha256((ROOT/'calculation_contract.json').read_bytes()).hexdigest(),
             'data_rows':len(obs),'AP_ratios':len(rat),'observed_AP':[dict(z=float(zz),F=float(v),sigma=float(s)) for zz,v,s in zip(zs,rat,np.sqrt(np.diag(Cr)))],
             'null':stat(null),'cases':[], 'interpretation':contract['scope_limits'],
             'environment':dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)}
    validations={'covariance_min_eigenvalue':float(np.linalg.eigvalsh(C)[0]),'AP_covariance_min_eigenvalue':float(np.linalg.eigvalsh(Cr)[0])}
    maxnull=0.
    for zz in zs:
        E=lambda z:np.sqrt(.3*(1+z)**3+.7)
        expected=E(zz)*quad(lambda z:1/E(z),0,zz,epsabs=1e-12)[0]
        maxnull=max(maxnull,abs(null_dist[zz]['F_AP']-expected))
    validations['null_AP_LCDM_max_abs']=maxnull
    for case in contract['fixed_cases']:
        p=Parameters(case['r'],case['initial_theta']);bg=Background(p);end=bg.quantities(0.)
        out=dict(case,**stat(bg),w_phi_0=end['w_phi'],q_dec_0=end['q'],omega_phi_0=end['omega_phi'],theta_0=end['theta'],theta_N0=end['v'],H0_internal=end['H'])
        out['checks']=dict(**bg.source_checks(),**loop_checks(bg.sp))
        Ns=np.linspace(bg.Ni,0,2001)
        history=np.array([[np.exp(N),*[(q:=bg.quantities(N))[k] for k in ['theta','v','H','w_phi','q','rho_b','rho_c','U']]] for N in Ns])
        fn=f'history_r{p.r}_theta{p.theta_initial}.csv'
        np.savetxt(ROOT/'results'/fn,history,delimiter=',',header='a,theta,theta_N,H_internal,w_phi,q_dec,rho_b,rho_c,U',comments='')
        out['history_file']=fn
        if p.r==.35 and p.theta_initial==.5:
            rad=Background(p,method='Radau',rtol=5e-11,atol=5e-13)
            dop=np.array([bg.distances(zz)['F_AP'] for zz in zs]);rap=np.array([rad.distances(zz)['F_AP'] for zz in zs])
            validations['AP_DOP853_Radau_max_abs']=float(np.max(np.abs(dop-rap)))
            validations['independent_cosmic_time']=cosmic_check(bg)
        results['cases'].append(out)
        print(f'r={p.r:.2f} d={p.theta_initial:.1f} w={end["w_phi"]:.10f} q={end["q"]:.8f} APchi2={out["AP_chi2_first_order"]:.5f} full13chi2={out["profiled_full13_chi2"]:.5f} nullsep={out["distance_from_fixed_null_in_AP_covariance"]:.5f}')
    results['validation']=validations
    # Assert only numerical/structural checks, never an observational success.
    for out in results['cases']:
        assert out['checks']['relative_total_continuity']<1e-11
        assert out['checks']['loop_integral_relative_max']<1e-8
        assert out['checks']['Uprime_finite_difference_abs_max']<1e-7
        assert out['checks']['H_derivative_finite_difference_abs']<1e-6
    assert maxnull<1e-9
    assert validations['AP_DOP853_Radau_max_abs']<1e-8
    assert validations['independent_cosmic_time']['theta_velocity_H_relative_max']<2e-6
    (ROOT/'results/observable_results.json').write_text(json.dumps(results,indent=2))
    (ROOT/'results/numerical_validation.json').write_text(json.dumps(validations,indent=2))
    print('NULL',results['null']);print('VALIDATION',validations)

if __name__=='__main__':main()
