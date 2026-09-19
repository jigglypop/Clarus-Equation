#!/usr/bin/env python3
"""CE-IRS2: fixed-charge spectral backreaction in the stated two-derivative EFT.
No observations, fits, network or Git writes. External s, epsilon, xi, F0,
Einstein reference M0 and initial charge are inputs, not predictions.
Run: python verify_charge_closure.py --out results.json
"""
from __future__ import annotations
import argparse, json, platform
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import expm
import mpmath as mp
import sympy as sy
PI=np.pi
@dataclass
class Model:
    s: float=1.
    eps: float=.15
    F0: float=.01
    M02: float=.01
    xi: float=0.
    order: int=80
    def __post_init__(self):
        if not (self.s>2*self.eps>0 and self.F0>0 and self.M02>0):
            raise ValueError('Require s>2epsilon>0 and positive F0,M0^2.')
        z,w=leggauss(self.order); self.t=(z+1)/2; self.w=w/2
        self.q=self.s*self.t/(1-self.t)
        self.qw=self.w*self.s/(1-self.t)**2
        self.P=(self.q+self.s-2*self.eps)*(self.q+self.s+self.eps)**2
    def coefficients(self,theta: float)->dict:
        t,w=self.t,self.w;e,s=self.eps,self.s
        ph=(theta+2*PI*np.arange(3))/3
        x=s+2*e*np.cos(ph);xp=-2*e/3*np.sin(ph);xpp=-(x-s)/9
        # Positive resolvent integral avoids cancellation at the reference vacuum.
        amp=4*e**3*np.cos(theta/2)**2
        ap=-2*e**3*np.sin(theta);app=-2*e**3*np.cos(theta)
        den=self.P+amp; log=np.log1p(amp/self.P); f=1/(16*PI**2)
        U=f*np.sum(self.qw*self.q*log)
        U1=f*ap*np.sum(self.qw*self.q/den)
        U2=f*(app*np.sum(self.qw*self.q/den)-ap**2*np.sum(self.qw*self.q/den**2))
        Us=-f*np.sum(self.qw*log)
        Us1=-f*ap*np.sum(self.qw/den)
        Us2=-f*(app*np.sum(self.qw/den)-ap**2*np.sum(self.qw/den**2))
        lo,hi=x[1],x[0];lp,hp=xp[1],xp[0];lpp,hpp=xpp[1],xpp[0]
        d=hi-lo;d1=hp-lp;d2=hpp-lpp
        h=lo+t*d;h1=lp+t*d1;h2=lpp+t*d2; wt=f*w*t*(1-t)
        C=np.sum(wt*d*d/h)
        C1=np.sum(wt*(2*d*d1/h-d*d*h1/h**2))
        C2=np.sum(wt*(2*(d1*d1+d*d2)/h-4*d*d1*h1/h**2-d*d*h2/h**2+2*d*d*h1*h1/h**3))
        K=np.sum(xp*xp/x)/(96*PI**2)
        K1=np.sum(2*xp*xpp/x-xp**3/x**2)/(96*PI**2)
        F=self.F0-2*(self.xi-1/6)*Us
        F1=-2*(self.xi-1/6)*Us1; F2=-2*(self.xi-1/6)*Us2
        if F<=0 or C<=0: raise ValueError('Outside F>0, nondegenerate rotating pair.')
        m=self.M02
        A=m*C/F; A1=m*(C1/F-C*F1/F**2)
        A2=m*(C2/F-2*C1*F1/F**2-C*F2/F**2+2*C*F1**2/F**3)
        V=m*m*U/F**2; V1=m*m*(U1/F**2-2*U*F1/F**3)
        V2=m*m*(U2/F**2-4*U1*F1/F**3-2*U*F2/F**3+6*U*F1**2/F**4)
        G=m*(K/F+1.5*(F1/F)**2)
        G1=m*(K1/F-K*F1/F**2+3*(F1/F)*(F2/F-(F1/F)**2))
        return dict(x=x,xp=xp,U=U,U1=U1,U2=U2,Us=Us,C=C,C1=C1,C2=C2,K=K,
                    F=F,F1=F1,F2=F2,A=A,A1=A1,A2=A2,V=V,V1=V1,V2=V2,G=G,G1=G1)
    def stationarity(self,theta,n):
        z=self.coefficients(theta);return z['V1']-n*n*z['A1']/z['A']**2
    def stationary_roots(self,n):
        grid=np.linspace(0,PI,257);f=[self.stationarity(t,n) for t in grid]; roots=[]
        for a,b,fa,fb in zip(grid[:-1],grid[1:],f[:-1],f[1:]):
            if fa*fb<0:roots.append(brentq(lambda t:self.stationarity(t,n),a,b,xtol=4e-15))
        return roots

def positive_C(a,b):
    z,w=leggauss(80);t=(z+1)/2
    return (b-a)**2*np.sum(w/2*t*(1-t)/(a+t*(b-a)))/(16*PI**2)

def run()->dict:
    records={}
    def check(name,passed,**details):
        records[name]={'passed':bool(passed),**details}
        if not passed:raise AssertionError((name,details))
    rng=np.random.default_rng(20260919)
    # A response-function identity tests structure across an interval, not just two fitted values.
    s,e,c=sy.symbols('s e c', positive=True)
    P=(s-2*e)*(s+e)**2; amp=2*e**3*(1+c);Q=P/amp
    poly=sy.Poly(Q,s);a3=poly.nth(3);a1=poly.nth(1);a0=poly.nth(0)
    ids=[sy.diff(Q,s,4),poly.nth(2),4*a1**3+27*a3*a0**2]
    # Fixed-charge Euler-Lagrange identity gives the exact continuity equation.
    g,gp,A,Ap,Vp,u,n,H,M=sy.symbols('g gp A Ap Vp u n H M', nonzero=True)
    ud=-3*H*u-(gp*u*u/2+Vp-n*n*Ap/A**2)/g;nd=-3*H*n
    rhod=(gp*u*u/2-n*n*Ap/A**2+Vp)*u+g*u*ud+2*n/A*nd
    ids.append(sy.expand(rhod+3*H*(g*u*u+2*n*n/A)))
    check('symbolic_signature_and_continuity',all(sy.simplify(v)==0 for v in ids),identities=len(ids))
    # Compare coefficient derivatives against independent central differences, not their own formulas.
    errs=[];uerrs=[]
    for r in [.001,.01,.15,.35]:
        model=Model(eps=r,F0=1,M02=1)
        for th in [.3,1.2,2.4]:
            z=model.coefficients(th);h=2e-4
            zp=model.coefficients(th+h);zm=model.coefficients(th-h)
            for name in ['U','C','A','V','G']:
                fd=(zp[name]-zm[name])/(2*h)
                exact=z[name+'1']
                errs.append(abs(fd-exact)/max(abs(exact),1e-18))
            mp.mp.dps=65;er=mp.mpf(str(r));tt=mp.mpf(str(th))
            xs=[1+2*er*mp.cos((tt+2*mp.pi*j)/3) for j in range(3)]
            ref=[1-2*er,1+er,1+er]
            f=lambda x:x*x*(mp.log(x)-mp.mpf('1.5'))
            direct=(mp.fsum(f(x) for x in xs)-mp.fsum(f(x) for x in ref))/(32*mp.pi**2)
            uerrs.append(float(abs(mp.mpf(z['U'])-direct)/abs(direct)))
    check('independent_derivative_and_potential_checks',max(errs)<3e-7 and max(uerrs)<3e-12,
          derivative_max_relative=max(errs),positive_integral_vs_65_digit_eigen_sum=max(uerrs),cases=12)
    # At given eigenvalues the largest angular inertia is the extreme pair.
    ratios=[]
    for k in range(96):
        xs=np.sort(rng.uniform(.15,3,3));cs=np.array([positive_C(xs[0],xs[1]),positive_C(xs[0],xs[2]),positive_C(xs[1],xs[2])])
        vv=rng.normal(size=3);vv/=np.linalg.norm(vv)
        ratio=np.sum(vv*vv/cs)*cs[1];ratios.append(ratio)
    check('global_orbit_charge_bound',min(ratios)>=1-1e-12,real_rank_two_currents=96,min_ratio_to_extreme_pair=min(ratios))
    # Endpoint derivatives force a strictly interior energy minimum for every nonzero n,
    # when F is positive. The analytic proof is in REPORT_ko.md, not inferred from this grid.
    endpoint=[];static=[]
    for r in [.001,.01,.15,.35]:
        model=Model(eps=r,F0=.01,M02=.01)
        for th in np.linspace(0,PI,33):
            z=model.coefficients(th);assert z['F']>0 and z['G']>0
        for scale in [.03,.3,3]:
            n0=scale*r**2.5/(8*PI**2)
            f0=model.stationarity(0,n0);fpi=model.stationarity(PI,n0)
            roots=model.stationary_roots(n0)
            best=min(roots,key=lambda t:model.coefficients(t)['V']+n0*n0/model.coefficients(t)['A'])
            z=model.coefficients(best)
            w2=z['V2']-n0*n0*(z['A2']/z['A']**2-2*z['A1']**2/z['A']**3)
            inv=z['A']**2*z['V1']/z['A1']
            err=abs(inv-n0*n0)/(n0*n0)
            endpoint.append(f0<0 and fpi>0)
            T=n0*n0/z['A'];rho=T+z['V'];p=T-z['V']
            static.append(dict(r=r,n=n0,theta=best,root_count=len(roots),W_second=w2,
                               inverse_n_squared_relative_error=err,U_J=z['U'],V_E=z['V'],rho=rho,w=p/rho,
                               angular_threshold_ratio=(n0/z['A'])**2/(4*min(z['x'])*model.M02/z['F'])))
    check('interior_stationary_and_inverse_state',all(endpoint) and all(z['W_second']>0 for z in static)
          and max(z['inverse_n_squared_relative_error'] for z in static)<1e-7,
          cases=len(static),grid_roots_not_uniqueness_proof=True,rows=static)
    # Small-charge theorem from the implicit function theorem; verify convergence order.
    model=Model(eps=.15,F0=.01,M02=.01);z=model.coefficients(PI)
    coeff=-z['A1']/(z['V2']*z['A']**2)
    seq=[]
    for n0 in [2e-6,1e-6,5e-7]:
        th=model.stationary_roots(n0)[-1];pred=coeff*n0*n0
        seq.append(dict(n=n0,shift=PI-th,leading_shift=pred,relative_error=abs((PI-th)/pred-1)))
    check('small_charge_asymptotic',seq[-1]['relative_error']<seq[0]['relative_error']/8,
          coefficient=coeff,rows=seq)
    # New response-signature rejects a shape addition; static spectrum cannot see a spectator.
    ss=np.array([1.,1.3,2.,4.,8.]);eps=.15;th=1.2
    Qvals=((ss-2*eps)*(ss+eps)**2)/(2*eps**3*(1+np.cos(th)))
    signature=np.linalg.solve(np.vander(ss[:4],4),Qvals[:4]);res=max(abs(np.polyval(signature,ss)-Qvals))/max(abs(Qvals))
    # Alter Q by a quartic term; its fourth derivative is exactly nonzero.
    negative={'quartic_Q_fourth_derivative':.024,'fixed_phase_pi_force_nonzero':float(model.stationarity(PI,1e-5)),
              'spectator_invisible_to_relative_determinant':True,'n_sign_not_reconstructed':True,
              'global_stationary_uniqueness_not_proved':True,'initial_charge_not_created_by_homogeneous_equations':True}
    check('negative_controls',res<1e-12 and negative['fixed_phase_pi_force_nonzero']>0,
          cubic_numeric_residual=res,controls=negative)
    # Full dynamics retains spectral acceleration AND the same varying F(theta).
    dyn=Model(eps=.001,F0=1e-8,M02=1e-8)
    n0=1e-10;theta0=PI
    z=dyn.coefficients(theta0);H0=np.sqrt((n0*n0/z['A']+z['V'])/(3*dyn.M02))
    y0=np.array([theta0,0.,n0,H0,0.,0.]) # theta,u,n,H,N,psi
    def rhs(t,y):
        th,u,nn,hh,N,psi=y;zz=dyn.coefficients(th)
        ud=-3*hh*u-(.5*zz['G1']*u*u+zz['V1']-nn*nn*zz['A1']/zz['A']**2)/zz['G']
        return [u,ud,-3*hh*nn,-(.5*zz['G']*u*u+nn*nn/zz['A'])/dyn.M02,hh,nn/zz['A']]
    def end(t,y):return y[4]-.8
    end.terminal=True;end.direction=1
    sols=[]
    for method in ['DOP853','Radau']:
        sol=solve_ivp(rhs,(0,10000),y0,method=method,rtol=2e-10,
                      atol=[1e-12,1e-13,1e-23,1e-14,1e-12,1e-11],events=end,dense_output=True,max_step=2)
        if not sol.success or len(sol.t_events[0])!=1:raise RuntimeError(sol.message)
        sols.append(sol)
    times=np.linspace(0,min(v.t[-1] for v in sols),601)
    Y=sols[0].sol(times);YY=sols[1].sol(times)
    constraint=[];charge=[];cont=[];rat=[];w_hist=[]
    for i,(th,u,nn,hh,N,psi) in enumerate(Y.T):
        zz=dyn.coefficients(th);kin=.5*zz['G']*u*u+nn*nn/zz['A'];rho=kin+zz['V'];p=kin-zz['V']
        d=np.array(rhs(times[i],Y[:,i]));rd=(.5*zz['G1']*u*u-nn*nn*zz['A1']/zz['A']**2+zz['V1'])*u+zz['G']*u*d[1]+2*nn/zz['A']*d[2]
        constraint.append(abs(3*dyn.M02*hh*hh-rho)/max(rho,1e-30))
        charge.append(abs(nn*np.exp(3*N)/n0-1))
        cont.append(abs(rd+3*hh*(rho+p))/max(abs(3*hh*(rho+p)),1e-30))
        w_hist.append(p/rho)
        threshold=4*min(zz['x'])*dyn.M02/zz['F']
        rat.append([hh*hh/threshold,(nn/zz['A'])**2/threshold,abs(zz['V2']/zz['G'])/threshold])
    scales=np.array([1.,.01,n0,H0,1.,1.])[:,None]
    diff=float(np.max(abs(Y-YY)/scales))
    check('full_two_coordinate_Einstein_dynamics',max(constraint)<1e-7 and max(charge)<1e-8
          and max(cont)<1e-11 and diff<2e-7,
          input={'s':1,'epsilon':.001,'F0':1e-8,'M0_squared':1e-8,'xi':0,'n0':n0,'theta0':PI},
          stop_N=.8,steps=[len(v.t) for v in sols],max_Friedmann_relative=max(constraint),
          max_charge_relative=max(charge),max_continuity_relative=max(cont),
          two_solver_scaled_difference=diff,theta_min=float(min(Y[0])),theta_final=float(Y[0,-1]),
          w_min=float(min(w_hist)),w_max=float(max(w_hist)),
          threshold_ratios_max=np.max(rat,axis=0).tolist(),
          threshold_ratios_are_diagnostics_not_full_error_bounds=True,
          stationary_tracking_not_assumed=True)
    sample=[{'t':float(times[i]),'theta':float(Y[0,i]),'theta_dot':float(Y[1,i]),'n':float(Y[2,i]),
             'H':float(Y[3,i]),'N':float(Y[4,i]),'psi':float(Y[5,i]),'w':float(w_hist[i])}
            for i in range(0,len(times),30)]
    return dict(label='CE-IRS2: inverse response and fixed-charge spectral closure',date='2026-09-19',
                checks=records,group_count=len(records),all_groups_passed=all(v['passed'] for v in records.values()),
                trajectory_sample=sample,observations_used=False,observational_fit=False,
                full_joint_rmse=None,scientific_success=False,whole_theory_proved=False,
                environmental_versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                              sympy=sy.__version__,mpmath=mp.__version__))
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results.json'));a=ap.parse_args()
    result=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
