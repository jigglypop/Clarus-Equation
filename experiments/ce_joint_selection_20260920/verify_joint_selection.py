#!/usr/bin/env python3
"""CE-JS1: same HS1 winding function, now test joint Higgs/radius selection.

New diagnostic assumptions are explicit: a product-circle Einstein reduction,
fixed g5, a bulk Higgs quartic normalized at R_ref, no omitted spectator or
local vacuum terms. This is not the full SU(6) parent or a cosmological fit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import zeta
import sympy as sp


class Checks:
    def __init__(self): self.rows = []
    def test(self, name, passed, **data):
        row = dict(name=name, passed=bool(passed), **data)
        self.rows.append(row)
        print(('PASS ' if passed else 'FAIL ') + name, flush=True)
        if not passed: raise AssertionError(row)
    def near(self, name, error, tolerance):
        self.test(name, np.isfinite(error) and error <= tolerance,
                  error=float(error), tolerance=float(tolerance))


class JointModel:
    """Einstein-frame variables u=phi^2/2, r=log(R/R_ref), a=(0,q,-q)."""
    def __init__(self, nmax=2048, gref=.6, Rref=1., lam=10., u0=.5):
        if nmax < 16 or min(gref, Rref, lam) <= 0 or u0 < 0:
            raise ValueError('invalid fixed diagnostic inputs')
        self.n = np.arange(1, nmax + 1, dtype=float)
        self.w = self.n**-5
        self.gref, self.Rref, self.lam, self.u0 = gref, Rref, lam, u0
        self.kref = gref*gref/2
        self.C = 3/(64*np.pi**6*Rref**4)
        self.Kq0 = 4/(gref*gref*Rref**2)
        self.Kr = 1.5  # M_Pl=1 ONLY for the finite mechanical conservation test.
        self.nmax = nmax
    def winding(self, q, y):
        if y <= 0: raise ValueError('positive gapped y required')
        n = self.n; z = 2*np.pi*n*np.sqrt(y)
        ez = np.exp(-z); f = ez*(1+z+z*z/3)
        fy = -(2*np.pi*n)**2*(1+z)*ez/6
        fyy = (2*np.pi*n)**4*ez/12
        t = 2*np.pi*n*q
        T = 1+2*np.cos(t); D = -4*np.pi*n*np.sin(t)
        DD = -8*np.pi**2*n*n*np.cos(t)
        return dict(w=float(np.dot(self.w,T*T-1-2*f*T)),
                    wq=float(2*np.dot(self.w,(T-f)*D)),
                    wqq=float(2*np.dot(self.w,D*D+(T-f)*DD)),
                    wy=float(-2*np.dot(self.w,fy*T)),
                    wqy=float(-2*np.dot(self.w,fy*D)),
                    wyy=float(-2*np.dot(self.w,fyy*T)))
    def selected(self, y):
        q=brentq(lambda q:self.winding(q,y)['wq'],.249,1/3+1e-9,xtol=5e-15)
        a=self.winding(q,y)
        a.update(q=q,y=y,qy=-a['wqy']/a['wqq'],
                 relaxed_wyy=a['wyy']-a['wqy']**2/a['wqq'])
        return a
    def potential(self,u,r,q,neutral=0.):
        y=self.kref*u*self.Rref**2*np.exp(2*r)
        w=self.winding(q,y)
        pref=self.C*np.exp(-6*r)
        H=self.lam*(u-self.u0)**2; Hp=2*self.lam*(u-self.u0)
        pot=np.exp(-r)*H+pref*(w['w']+neutral)
        Uu=np.exp(-r)*Hp+pref*y/u*w['wy']
        Ur=-np.exp(-r)*H+pref*(-6*(w['w']+neutral)+2*y*w['wy'])
        Uq=pref*w['wq']
        return dict(U=pot,Uu=Uu,Ur=Ur,Uq=Uq,y=y,**w)
    def masses(self,u,r,q,n):
        a=np.array([0.,q,-q])
        return self.kref*np.exp(-r)*u+np.exp(-3*r)*(np.asarray(n)[...,None]+a)**2/self.Rref**2
    def thresholds(self,u,r,q):
        m2=self.kref*np.exp(-r)*u
        d=np.exp(-3*r)/self.Rref**2
        return 2*np.sqrt([m2,m2+d*q*q,m2+d*(1-q)**2])
    def noise(self,omega,u,r,q):
        Reff=self.Rref*np.exp(1.5*r)
        n=np.arange(-int(np.ceil(Reff*omega/2))-2,int(np.ceil(Reff*omega/2))+3)
        x=self.masses(u,r,q,n)
        S=np.sqrt(np.maximum(0.,1-4*x/omega**2)).sum()
        return (self.kref*np.exp(-r))**2*S/(8*np.pi)
    def infer(self,thresholds,noise,omega):
        L,H,N=np.asarray(thresholds,dtype=float)
        if not (0<L<H<N):raise ValueError('ordered positive thresholds required')
        A=np.sqrt(H*H-L*L)/2;B=np.sqrt(N*N-L*L)/2
        q=A/(A+B);Reff=1/(A+B);y=(L*Reff/2)**2
        if not H<omega<N:raise ValueError('use the three-mode open interval')
        S=np.sqrt(1-(L/omega)**2)+2*np.sqrt(1-(H/omega)**2)
        kap=np.sqrt(8*np.pi*noise/S)
        r=(2/3)*np.log(Reff/self.Rref)
        u=(L/2)**2/kap
        n=self.n; z=2*np.pi*n*np.sqrt(y)
        f=np.exp(-z)*(1+z+z*z/3)
        T=1+2*np.cos(2*np.pi*n*q);D=-4*np.pi*n*np.sin(2*np.pi*n*q)
        denominator=np.dot(self.w,T*D)
        if abs(denominator)<1e-10:raise ValueError('stationarity inversion is ill-conditioned')
        nadj=(3+np.dot(self.w,f*D)/denominator)/4
        return dict(q=q,Reff=Reff,y=y,kappa=kap,r=r,u=u,g4=np.sqrt(2*kap),
                    nadj=float(nadj),stationarity=self.winding(q,y)['wq'],
                    inverse_denominator=float(denominator))


def fivepoint(f,x,h=1e-4):
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)


def run():
    C=Checks();m=JointModel();rows=[]
    # Independent copy of the same winding kernel, not an observational fit.
    inherited=Path(__file__).resolve().parents[1]/'ce_holonomy_selection_20260920'/'verify_holonomy.py'
    if not inherited.exists():
        raise FileNotFoundError('publish with the inherited HS1 script in the sibling directory')
    import importlib.util
    spec=importlib.util.spec_from_file_location('ce_hs1',inherited)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    hs=mod.Model(nmax=2048)
    matches=[]
    for u in [.2,.5,.8]:
        s=m.selected(m.kref*u); old=hs.selected(u)
        matches.append(max(abs(s['q']-old['q']),abs(s['w']-old['V'])))
        P=m.potential(u,0.,s['q'])
        T=m.thresholds(u,0.,s['q']);om=(T[1]+T[2])/2
        inv=m.infer(T,m.noise(om,u,0.,s['q']),om)
        rows.append(dict(u=u,q=s['q'],y=s['y'],w=s['w'],wy=s['wy'],
                         loop_higgs_source=-m.C*m.kref*s['wy'],
                         loop_radion_gradient=m.C*(-6*s['w']+2*s['y']*s['wy']),
                         total_higgs_gradient=P['Uu'],total_radion_gradient=P['Ur'],
                         thresholds=T.tolist(),readout=inv))
    C.near('independent winding kernel reproduces HS1 at fixed radius',max(matches),5e-13)
    # Direct derivatives at general radii test all conformal and gauge factors.
    errs=[]
    for u,r,q in [(.37,-.12,.276),(.61,.21,.296),(.5,0.,rows[1]['q'])]:
        a=m.potential(u,r,q)
        for key,x,f in [('Uu',u,lambda t:m.potential(t,r,q)['U']),
                        ('Ur',r,lambda t:m.potential(u,t,q)['U']),
                        ('Uq',q,lambda t:m.potential(u,r,t)['U'])]:
            errs.append(abs(a[key]-fivepoint(f,x)))
    C.near('full Einstein potential gradients against independent five-point differences',max(errs),2e-10)
    C.near('g5 is unchanged while g4 and radius vary',max(abs(m.gref**2*np.exp(-r)*2*np.pi*m.Rref*np.exp(r)-m.gref**2*2*np.pi*m.Rref) for r in [-.6,0.,.4]),2e-15)
    C.near('Einstein masses equal Jordan masses after Weyl conversion',max(np.max(abs(m.masses(.4,r,.28,np.arange(-2,3))-np.exp(-r)*(m.kref*.4+(np.arange(-2,3)[:,None]+[0,.28,-.28])**2/(m.Rref**2*np.exp(2*r))))) for r in [-.2,.3]),2e-15)
    C.test('old phase minima are not stationary in the loop radion direction',all(r['loop_radion_gradient']>0 for r in rows),values=[r['loop_radion_gradient'] for r in rows])
    C.test('old middle Higgs input is not an exact coupled Higgs stationary point',rows[1]['total_higgs_gradient']>0,value=rows[1]['total_higgs_gradient'])
    # Solve u at fixed radius, then verify the radius force is still nonzero.
    cond=[]
    for r in [-.1,0.,.1]:
        def ug(u):return m.potential(u,r,m.selected(m.kref*u*np.exp(2*r))['q'])['Uu']
        u=brentq(ug,.4,.6,xtol=5e-15);s=m.selected(m.kref*u*np.exp(2*r));a=m.potential(u,r,s['q'])
        cond.append(dict(r=r,u=u,q=s['q'],Uu=a['Uu'],Ur=a['Ur']))
    C.near('conditional Higgs minimization solves the Higgs equation',max(abs(a['Uu']) for a in cond),2e-13)
    C.test('conditional Higgs minimization does not solve the radius equation',all(a['Ur']>0 for a in cond),rows=cond)
    # Monotonicity proof support: w_y is decreasing in q on (0,1/2).
    # Integral representation of the sine series provides the analytic sign.
    integ=[]
    for y,q in [(.036,.25),(.09,.28145),(.144,1/3)]:
        c=2*np.pi*np.sqrt(y);ang=2*np.pi*q
        lhs=np.sum(np.exp(-c*m.n)*(1+c*m.n)*np.sin(m.n*ang)/m.n**2)
        def f(t):
            a=np.exp(-(c+t));return (t+c)*a*np.sin(ang)/(1-2*a*np.cos(ang)+a*a)
        rhs=quad(f,0,np.inf,epsabs=1e-13)[0]
        integ.append(abs(lhs-rhs))
    C.near('positive integral representation of the derivative sine series',max(integ),3e-13)
    monotone=[]
    for y in np.logspace(-4,1,16):
        qq=np.linspace(.25,1/3,41)
        vals=np.array([m.winding(q,y)['wy'] for q in qq])
        monotone.append(bool(np.all(vals>0) and np.all(np.diff(vals)<=1e-12)))
    C.test('positive mass-source derivative throughout the tested stable branch',all(monotone))
    a=1/3;exact=-26*zeta(5,1)/27
    C.test('center-symmetric competitor proves a negative global phase minimum',all(m.winding(a,y)['w']<exact+1e-12 for y in [.001,.036,.09,.144,1.,5.]),upper_bound=float(exact))
    # At any interior joint stationary point on this branch u/u0<1/5.
    # The variation r->r+t, u->u exp(-2t) holds y fixed and proves a saddle.
    t=sp.symbols('t',real=True);u,u0,lam,H,L=sp.symbols('u u0 lam H L',positive=True)
    V=lam*(u*u*sp.exp(-5*t)-2*u*u0*sp.exp(-3*t)+u0*u0*sp.exp(-t))+L*sp.exp(-6*t)
    Lcrit=sp.solve(sp.diff(V,t).subs(t,0),L)[0]
    dd=sp.simplify(sp.diff(V,t,2).subs(t,0).subs(L,Lcrit))
    C.test('symbolic joint stationary scale-direction curvature identity',sp.simplify(dd-lam*(-5*u*u+18*u*u0-5*u0*u0))==0)
    ratio=[]
    for y in np.logspace(-5,0,30):
        s=m.selected(y);A=y*s['wy'];ratio.append(A/(-12*s['w']+5*A))
    C.test('joint stationary necessary Higgs ratio lies below one fifth',all(0<x<.2 for x in ratio),largest_sample_ratio=float(max(ratio)))
    C.test('negative scale direction on the entire necessary-ratio interval',sp.simplify((-5*t*t+18*t-5).subs(t,sp.Rational(1,5)))<0)
    C.test('scale-invariant quartic has a negative stationary dilation curvature',sp.simplify(dd.subs(u0,0)+5*lam*u*u)==0)
    # The u=0 boundary cannot rescue a stable finite radius in this potential.
    Hb=sp.symbols('Hb',positive=True); Lb=sp.symbols('Lb',real=True)
    Vb=Hb*sp.exp(-t)+Lb*sp.exp(-6*t)
    Lbc=sp.solve(sp.diff(Vb,t).subs(t,0),Lb)[0]
    C.test('zero-Higgs boundary has negative radial curvature at any stationary radius',
           sp.simplify(sp.diff(Vb,t,2).subs(t,0).subs(Lb,Lbc)+5*Hb)==0)
    # Only a finite parameter scan is asserted here; no global root search claim.
    scan=[]
    for y in np.logspace(-6,.3,55):
        s=m.selected(y);A=y*s['wy'];D=-6*s['w']+2*A
        uj=m.u0*A/(2*D+A);r=.5*np.log(y/(m.kref*uj))
        mismatch=np.log(m.C*D/(m.lam*(uj-m.u0)**2))-5*r
        scan.append(mismatch)
    C.test('fixed diagnostic inputs fail the remaining radial equation on the tested grid',max(scan)<0,largest_log_mismatch=float(max(scan)))
    # Full q-response Hessian is the Schur complement, not the frozen q value.
    s=m.selected(.09);du=1e-4
    eff=lambda y:m.selected(y)['w']
    fd=(eff(.09+du)-2*eff(.09)+eff(.09-du))/(du*du)
    C.near('relaxed source curvature includes the holonomy Schur correction',abs(fd-s['relaxed_wyy']),5e-4)
    C.test('freezing holonomy omits a strictly negative relaxation term',s['wyy']-s['relaxed_wyy']>0,missing=s['wyy']-s['relaxed_wyy'])
    # A spectator invisible to mediator records changes the radial source.
    q=rows[1]['q'];u=.5;r=0.
    P0=m.potential(u,r,q);Pf=m.potential(u,r,q,4*zeta(5,1));Pb=m.potential(u,r,q,-zeta(5,1))
    C.near('neutral Casimir changes neither phase nor direct Higgs gradients',max(abs(Pf['Uq']-P0['Uq']),abs(Pf['Uu']-P0['Uu'])),1e-18)
    C.test('one neutral massless Dirac reverses the tested radial gradient',P0['Ur']>0 and Pf['Ur']<0,without=P0['Ur'],with_dirac=Pf['Ur'],with_real_scalar=Pb['Ur'])
    C.near('spectator shift in radius source equals its Casimir derivative',abs((Pf['Ur']-P0['Ur'])+24*m.C*zeta(5,1)),1e-18)
    # Algebraic inversion of THREE thresholds and ONE normalized noise value.
    inversions=[]
    for u,r in [(.2,0.),(.5,0.),(.8,0.),(.43,-.2),(.65,.17)]:
        y=m.kref*u*np.exp(2*r);q=m.selected(y)['q'];T=m.thresholds(u,r,q);om=(T[1]+T[2])/2
        v=m.infer(T,m.noise(om,u,r,q),om)
        inversions.append(dict(u=u,r=r,q=q,thresholds=T.tolist(),**{'inverse':v}))
    C.near('three thresholds recover selected phase and Einstein spectral radius',max(max(abs(v['inverse']['q']-v['q']),abs(v['inverse']['r']-v['r'])) for v in inversions),2e-14)
    C.near('absolute noise recovers portal and Higgs source in the fixed readout scheme',max(max(abs(v['inverse']['u']-v['u']),abs(v['inverse']['kappa']-m.kref*np.exp(-v['r']))) for v in inversions),2e-14)
    C.near('threshold-ratio stationarity is a parameter-free conditional null test',max(abs(v['inverse']['stationarity']) for v in inversions),3e-11)
    C.near('same synthetic records recover the fixed adjoint multiplicity without optimization',max(abs(v['inverse']['nadj']-1) for v in inversions),2e-12)
    C.near('record gauge-radius scaling follows fixed g5 rather than fixed g4',max(abs(v['inverse']['g4']**2*v['inverse']['Reff']**(2/3)-m.gref**2*m.Rref**(2/3)) for v in inversions),2e-14)
    v=inversions[1];T=np.array(v['thresholds']);T[1]*=1.02;om=(T[1]+T[2])/2
    bad=m.infer(T,m.noise(om,.5,0.,v['q']),om)
    C.test('independently perturbed threshold fails fixed-content stationarity',abs(bad['stationarity'])>.01 and abs(bad['nadj']-1)>.01,residual=bad['stationarity'],inferred_count=bad['nadj'])
    # Complete KK spectrum in Einstein units, independently use transformed HS1 sum.
    er=[]
    for u,r in [(.5,-.2),(.5,0.),(.5,.2)]:
        y=m.kref*u*np.exp(2*r);q=m.selected(y)['q'];a=np.array([0,q,-q]);b=np.sqrt(y)
        closed=np.exp(3*r)*m.Rref**2*np.sum(np.pi/b*np.sinh(2*np.pi*b)/(np.cosh(2*np.pi*b)-np.cos(2*np.pi*a)))
        N=100;ns=np.arange(-N,N+1)
        direct=np.sum(1/m.masses(u,r,q,ns));tail=0.
        for aj in a:
            for j in range(7):tail+=(-y)**j*(zeta(2*j+2,N+1+aj)+zeta(2*j+2,N+1-aj))
        direct+=np.exp(3*r)*m.Rref**2*tail
        er.append(abs(direct-closed))
    C.near('all KK inverse masses retain the correct Einstein radius factors',max(er),2e-13)
    # Six-dimensional Hamiltonian phase space: a LOCAL finite mechanics check.
    # No Friedmann solution, no initial-state prediction, and no decoherence claim.
    dyn=JointModel(nmax=256)
    qi=dyn.selected(dyn.kref*.5)['q'];z0=np.array([1.,qi,0.,0.,0.,0.])
    def rhs(t,z):
        phi,q,r,p,pi,pr=z;Kq=dyn.Kq0*np.exp(-2*r)
        v=dyn.potential(phi*phi/2,r,q)
        return [p,pi/Kq,pr/dyn.Kr,-phi*v['Uu'],-v['Uq'],-pi*pi/Kq-v['Ur']]
    times=np.linspace(0,8,65)
    a=solve_ivp(rhs,[0,8],z0,t_eval=times,method='DOP853',rtol=2e-11,atol=2e-13)
    b=solve_ivp(rhs,[0,8],z0,t_eval=times,method='Radau',rtol=2e-10,atol=2e-12)
    C.test('both full coupled mechanical integrations complete',a.success and b.success)
    C.near('independent coupled integrators agree',np.max(abs(a.y-b.y)),3e-9)
    energies=[];dynrows=[];recon=[]
    for j,t in enumerate(times):
        phi,q,r,p,pi,pr=a.y[:,j];u=phi*phi/2;Kq=dyn.Kq0*np.exp(-2*r)
        energies.append(.5*p*p+pi*pi/(2*Kq)+pr*pr/(2*dyn.Kr)+dyn.potential(u,r,q)['U'])
        if j in [0,16,32,64]:
            T=dyn.thresholds(u,r,q);om=(T[1]+T[2])/2
            inv=dyn.infer(T,dyn.noise(om,u,r,q),om)
            recon.append(max(abs(inv['u']-u),abs(inv['r']-r),abs(inv['q']-q)))
            dynrows.append(dict(t=t,u=u,q=q,r=r,thresholds=T.tolist(),stationarity_residual=inv['stationarity']))
    C.near('coupled Higgs-holonomy-radius mechanics conserves total energy',np.ptp(energies),3e-12)
    C.test('radius leaves the previously fixed phase point',a.y[2,-1]<-1e-4,final_r=float(a.y[2,-1]))
    C.near('instantaneous synthetic-spectrum inversion tracks coupled mechanical coordinates',max(recon),2e-13)
    C.test('nonstationary trajectories must not be forced onto static selection formula',abs(dynrows[-1]['stationarity_residual'])>1e-3,residual=float(dynrows[-1]['stationarity_residual']))
    # Local mechanical evolution is not a controlled real-time QFT prediction.
    # The retained Higgs curvature is above the first mediator pair threshold.
    q0=rows[1]['q']
    mh2=fivepoint(lambda ph:ph*m.potential(ph*ph/2,0.,q0)['Uu'],1.,1e-4)
    pair2=float(m.thresholds(.5,0.,q0)[0]**2)
    C.test('local mechanical Higgs frequency exceeds the mediator derivative-expansion gate',
           mh2>pair2,higgs_curvature=float(mh2),first_pair_threshold_squared=pair2)
    # Analytic two-loop diagnostic only; no unknown coefficient assigned a value.
    x,A,B,G=sp.symbols('x A B G',positive=True)
    V2=-A/x**6+B*G/x**7;xs=7*B*G/(6*A)
    C.test('two-loop balancing criterion is algebraically stationary',sp.simplify(sp.diff(V2,x).subs(x,xs))==0)
    C.test('two-loop balancing curvature requires a positive coefficient',sp.simplify(sp.diff(V2,x,2).subs(x,xs)/(6*A/xs**8))==1)
    C.test('two-loop size at such a root is six-sevenths of the one-loop term',sp.simplify((B*G/(A*x)).subs(x,xs))==sp.Rational(6,7))
    return dict(experiment='CE-JS1',date='2026-09-20',check_count=len(C.rows),checks=C.rows,
                inherited_source_sha256=hashlib.sha256(inherited.read_bytes()).hexdigest(),
                inputs=dict(gref=.6,Rref=1.,lambda_ref=10.,u0=.5,nadj=1,nmax=2048),
                fixed_radius_rows=rows,conditional_higgs=cond,inversions=inversions,
                coupled_diagnostic=dynrows,energy_drift=float(np.ptp(energies)),
                negative_controls=dict(neutral_dirac_radial_gradient=Pf['Ur'],base_radial_gradient=P0['Ur'],perturbed_record=bad),
                scope=dict(main_modified_by_script=False,observations_used=False,full_parent=False,
                           complete_radion_potential=False,cosmological_evolution=False,
                           real_time_qft_prediction=False,dynamic_records_are_instantaneous_synthetic=True,
                           all_constants_selected=False,prior_holonomy_results_preserved=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('results.json'));args=ap.parse_args()
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"{result['check_count']} checks passed; {args.output}")
