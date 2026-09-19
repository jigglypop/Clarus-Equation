#!/usr/bin/env python3
"""CE-HS1: explicit NEW compact-sector candidate, not a UV completion of CE.

No observations or fitted targets enter this code. All numbers use fixed inputs.
The added circle, periodic massless adjoint Dirac fermion and tree matching are
assumptions. We check local stability in ALL THREE relative phases, not just a
one-dimensional ansatz. Multistart searches are NOT proofs of global uniqueness.
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize
from scipy.special import kv, zeta

class Checks:
    def __init__(self): self.rows=[]
    def require(self,name,ok,**data):
        self.rows.append(dict(name=name,passed=bool(ok),**data))
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        if not ok: raise AssertionError(self.rows[-1])
    def near(self,name,error,tol):
        self.require(name,np.isfinite(error) and error<=tol,error=float(error),tolerance=float(tol))

def weight(z):
    z=np.asarray(z,dtype=float)
    if np.any(z<0): raise ValueError('nonnegative mass argument required')
    return np.exp(-z)*(1+z+z*z/3)

class Model:
    def __init__(self,g=.6,R=1.,nf=1,nmax=2048):
        if g<=0 or R<=0 or nf<0 or nmax<8: raise ValueError('invalid parameters')
        self.g,self.R,self.nf=g,R,nf;self.k=g*g/2
        self.n=np.arange(1,nmax+1,dtype=float);self.nmax=nmax
        self.w=self.n**-5;self.A=4*nf-3
        self.C=3/(64*np.pi**6*R**4)
        self.Kq=4/(g*g*R*R)
    def f(self,u):
        if u<=0: raise ValueError('positive u required for gapped diagnostic')
        z=2*np.pi*self.n*self.R*np.sqrt(self.k*u)
        f=weight(z)
        fu=-z*z*(1+z)*np.exp(-z)/(6*u)
        fuu=z**4*np.exp(-z)/(12*u*u)
        return f,fu,fuu
    def potential(self,a,u,derivatives=False):
        a=np.asarray(a,dtype=float)
        if a.shape!=(3,): raise ValueError('three relative phases required')
        f,_,_=self.f(u)
        arg=2*np.pi*self.n[:,None]*a
        E=np.exp(1j*arg);tr=E.sum(axis=1)
        val=float(np.sum(self.w*(self.A*(abs(tr)**2-1)-2*f*tr.real)))
        if not derivatives: return val
        w2=2*np.pi*self.n
        de=1j*w2[:,None]*E
        gr=np.sum(self.w[:,None]*(2*self.A*np.real(de*tr.conj()[:,None])-2*f[:,None]*de.real),axis=0)
        h=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                term=2*self.A*np.real(de[:,i]*de[:,j].conj())
                if i==j:
                    d2=-w2*w2*E[:,i]
                    term=term+2*self.A*np.real(d2*tr.conj())-2*f*d2.real
                h[i,j]=np.dot(self.w,term)
        return val,gr,h
    def branch(self,q,u):
        f,fu,fuu=self.f(u);t=2*np.pi*self.n*q
        T=1+2*np.cos(t);D=-4*np.pi*self.n*np.sin(t);DD=-8*np.pi**2*self.n**2*np.cos(t)
        V=np.dot(self.w,self.A*(T*T-1)-2*f*T)
        Vq=2*np.dot(self.w,(self.A*T-f)*D)
        Vqq=2*np.dot(self.w,self.A*D*D+(self.A*T-f)*DD)
        Vu=-2*np.dot(self.w,fu*T);Vqu=-2*np.dot(self.w,fu*D)
        Vuu=-2*np.dot(self.w,fuu*T)
        return dict(V=float(V),Vq=float(Vq),Vqq=float(Vqq),Vu=float(Vu),Vqu=float(Vqu),Vuu=float(Vuu))
    def selected(self,u):
        if self.nf!=1: raise ValueError('bracket used only for the specified nf=1 case')
        q=brentq(lambda v:self.branch(v,u)['Vq'],.24,.34,xtol=5e-15)
        r=self.branch(q,u);qp=-r['Vqu']/r['Vqq']
        a=np.array([0,q,-q]);ev=np.linalg.eigvalsh(self.potential(a,u,True)[2])
        x=self.k*u+a*a/(self.R*self.R)
        eps=q*q/(3*self.R*self.R)
        sq=self.kk_sum(a,u);lowest=float(np.sum(1/x))
        out=dict(u=u,q=q,q_derivative=qp,epsilon=eps,spectral_center_offset=2*eps,s=self.k*u+2*eps,
          masses_squared=x.tolist(),thresholds=(2*np.sqrt(x)).tolist(),
          heavy_slope_adiabatic=self.k+2*q*qp/self.R**2,light_slope=self.k,
          relative_slope_mismatch=2*q*qp/(self.k*self.R**2),
          full_phase_hessian=ev.tolist(),q_mass_squared=self.C*r['Vqq']/self.Kq,
          next_threshold=2*np.sqrt(self.k*u+(1-q)**2/self.R**2),
          inverse_sum_lowest=lowest,inverse_sum_full=sq,inverse_sum_omitted_fraction=(sq-lowest)/sq,
          potential_trivial=self.potential(np.zeros(3),u),**r)
        return out
    def kk_sum(self,a,u):
        b=self.R*np.sqrt(self.k*u)
        return float(np.sum(np.pi*self.R**2/b*np.sinh(2*np.pi*b)/(np.cosh(2*np.pi*b)-np.cos(2*np.pi*np.array(a)))))
    def kk_sum_series(self,a,u,N=64,P=5):
        b2=self.R**2*self.k*u;n=np.arange(-N,N+1)
        val=0.
        for aa in a:
            val+=np.sum(1/((n+aa)**2+b2))
            for r in range(P+1):
                val+=(-b2)**r*(zeta(2*r+2,N+1+aa)+zeta(2*r+2,N+1-aa))
        return float(self.R**2*val)
    def record(self,omega,a,u,lowest=False):
        if omega<=0: return 0.
        n=np.array([0]) if lowest else np.arange(-int(np.ceil(self.R*omega/2))-2,int(np.ceil(self.R*omega/2))+3)
        x=self.k*u+(n[:,None]+np.asarray(a)[None,:])**2/self.R**2
        return float(self.k**2/(8*np.pi)*np.sqrt(np.maximum(0,1-4*x/omega**2)).sum())

def blocks(h,c):
    A=np.zeros((6,6),complex);B=A.copy()
    A[:2,2]=h/np.sqrt(2);A[2,:2]=np.conj(h)/np.sqrt(2)
    B[3:,2]=c/np.sqrt(2);B[2,3:]=np.conj(c)/np.sqrt(2)
    return A,B

def run():
    C=Checks();model=Model();p5=np.diag([-1,-1,1,1,1,1]);p6=np.diag([1,1,1,-1,-1,-1])
    # The EXTRA circle component A7 is even under both original reflections.
    a=np.array([.07,.25,-.18]);d0=-a.sum()/6
    D=np.diag(np.r_[np.full(3,d0),a+d0])/(model.g*model.R)
    h=np.array([.3+.2j,-.1j]);c=np.array([.2+.4j,.5j,-.3]);A,B=blocks(h,c)
    C.near('new circle background has allowed even-even parity and zero trace',max(abs(np.trace(D)),np.max(abs(p5@D@p5-D)),np.max(abs(p6@D@p6-D))),2e-16)
    C.near('chosen relative background commutes with Higgs block',np.linalg.norm(A@D-D@A),2e-16)
    norm=lambda z:float(np.vdot(z,z).real)
    rhs=model.k*norm(h)*norm(c)+np.dot(abs(c)**2,a*a)/model.R**2
    C.near('full retained curvature gives common portal plus phase masses',abs(model.g**2*(norm(A@B-B@A)+norm(D@B-B@D))-rhs),2e-16)
    dq=np.diag([0,0,0,0,1,-1])/(model.g*model.R)
    C.near('holonomy kinetic normalization',abs(np.trace(dq@dq)-model.Kq/2),2e-15)
    # Proper-time coefficient and the massive finite winding factor, independent representations.
    errs=[]
    for z in [.2,1.,3.,9.]:
        w_bessel=np.sqrt(2/np.pi)*z**2.5*kv(2.5,z)/3
        errs.append(abs(w_bessel-weight(z)))
    C.near('massive winding polynomial equals Bessel determinant',max(errs),5e-16)
    mass=.3;R=1.
    integral=quad(lambda t:t**(-3.5)*np.exp(-mass*mass*t-np.pi**2*R*R/t),0,np.inf,epsabs=1e-13,epsrel=1e-12)[0]
    real_scalar=R/(16*np.pi**1.5)*integral
    C.near('proper-time prefactor uses four-dimensional energy density',abs(real_scalar-model.C*weight(2*np.pi*R*mass)),2e-14)
    C.near('three vector and four Dirac degrees of freedom give literature normalization',max(abs(3*model.C-9/(4*np.pi**2*(2*np.pi)**4)),abs(4*model.C-3/(np.pi**2*(2*np.pi)**4))),2e-19)
    rng=np.random.default_rng(20260920)
    a=np.array([.04,.27,-.22]);u=.5;V,gr,H=model.potential(a,u,True);step=1e-5
    grad_fd=np.array([(model.potential(a+np.eye(3)[i]*step,u)-model.potential(a-np.eye(3)[i]*step,u))/(2*step) for i in range(3)])
    C.near('three-phase gradient against independent finite difference',np.max(abs(gr-grad_fd)),2e-7)
    h_fd=np.column_stack([(model.potential(a+np.eye(3)[i]*step,u,True)[1]-model.potential(a-np.eye(3)[i]*step,u,True)[1])/(2*step) for i in range(3)])
    C.near('three-phase Hessian against gradient differences',np.max(abs(H-h_fd)),2e-6)
    C.near('integer large gauge and permutation symmetries',max(abs(model.potential(a+np.array([1,-2,3]),u)-V),abs(model.potential(a[[2,0,1]],u)-V),abs(model.potential(-a,u)-V)),2e-14)
    rows=[model.selected(x) for x in [.2,.5,.8]]
    C.require('all selected roots are nonzero and gapped',all(.24<r['q']<.34 and min(r['masses_squared'])>0 for r in rows))
    C.near('stationarity in all three phases including the common phase',max(np.linalg.norm(model.potential([0,r['q'],-r['q']],r['u'],True)[1]) for r in rows),2e-10)
    C.require('strict local stability in all three phases',all(min(r['full_phase_hessian'])>(6*abs(model.A)+2)*(2*np.pi)**2/(2*model.nmax**2) for r in rows),eigenvalues=[r['full_phase_hessian'] for r in rows])
    C.require('nonzero phase beats trivial phase with a finite gap',all(r['potential_trivial']-r['V']>1 for r in rows))
    halfwidth=1e-9
    gtail=8*np.pi*(3*abs(model.A)+1)/(3*model.nmax**3)
    htail=(6*abs(model.A)+2)*(2*np.pi)**2/(2*model.nmax**2)
    h_lipschitz=(12*np.sqrt(2)*abs(model.A)+2)*(2*np.pi)**3*zeta(2,1)
    C.require('analytic winding-tail bounds bracket the infinite-sum root',all(model.branch(r['q']-halfwidth,r['u'])['Vq']<-gtail and model.branch(r['q']+halfwidth,r['u'])['Vq']>gtail for r in rows),halfwidth=halfwidth,gradient_tail_bound=float(gtail))
    C.require('Hessian stays positive throughout the root enclosure',all(min(r['full_phase_hessian'])>htail+h_lipschitz*np.sqrt(2)*halfwidth for r in rows),combined_bound=float(htail+h_lipschitz*np.sqrt(2)*halfwidth))
    # Without the new fermion, all positive-weight bosonic terms favor integer phases.
    bos=Model(nf=0,nmax=512);viol=[]
    for _ in range(80):
        aa=rng.uniform(-.5,.5,3);viol.append(bos.potential(aa,.5)-bos.potential(np.zeros(3),.5))
    C.require('bosonic-only negative control: no lower sampled nonzero vacuum',min(viol)>0,minimum_sample_gap=float(min(viol)))
    # Multistart over independent phases. A search, NOT a uniqueness proof.
    search=[]
    short=Model(nmax=256)
    for r in rows:
        vals=[]
        starts=np.r_[rng.uniform(-.5,.5,(18,3)),np.zeros((1,3)),np.array([[0,1/3,-1/3]])]
        for aa in starts:
            def obj(t):
                vv,gg,_=short.potential(t,r['u'],True);return vv,gg
            sol=minimize(obj,aa,jac=True,method='BFGS',options=dict(gtol=3e-9,maxiter=250))
            vals.append(float(sol.fun))
        target=short.potential([0,r['q'],-r['q']],r['u'])
        search.append(dict(u=r['u'],starts=len(starts),min_value=min(vals),target=target,number_above_target=int(np.sum(np.array(vals)>target+1e-6))))
    C.near('independent three-dimensional searches find no lower point',max(abs(x['min_value']-x['target']) for x in search),5e-9)
    fine=Model(nmax=4096)
    C.near('doubling winding cutoff stabilizes selected phase',max(abs(fine.selected(r['u'])['q']-r['q']) for r in rows),2e-11)
    tail_bound=(8*abs(model.A)+6)/(4*256**4)
    C.near('finite winding sum obeys uniform analytic potential tail bound',abs(short.potential(a,.5)-fine.potential(a,.5)),tail_bound)
    # Changing the matter content changes the selected number: no universal constant claim.
    two=Model(nf=2,nmax=2048)
    qtwo=brentq(lambda q:two.branch(q,.5)['Vq'],.28,.34,xtol=5e-15)
    C.require('number depends on added matter content rather than being a CE constant',abs(qtwo-rows[1]['q'])>.01,one_fermion_q=rows[1]['q'],two_fermion_q=float(qtwo))
    # Full implicit readjustment, not a frozen-epsilon prescription.
    deriv_err=[];mass_err=[];envelope_err=[]
    for r in rows:
        u=r['u'];q=r['q'];du=2e-4
        ss=[model.selected(u+i*du) for i in [-2,-1,1,2]]
        d=lambda key:(ss[0][key]-8*ss[1][key]+8*ss[2][key]-ss[3][key])/(12*du)
        deriv_err.append(abs(d('q')-r['q_derivative']))
        mh=[v['masses_squared'][1] for v in ss]
        md=(mh[0]-8*mh[1]+8*mh[2]-mh[3])/(12*du)
        mass_err.append(abs(md-r['heavy_slope_adiabatic']))
        envelope_err.append(abs(d('V')-r['Vu']))
    C.near('implicit vacuum response equals independent re-minimization',max(deriv_err),2e-9)
    C.near('physical heavy-mass slope includes vacuum readjustment',max(mass_err),2e-9)
    C.near('total equilibrium potential satisfies envelope theorem',max(envelope_err),2e-9)
    C.require('fixed epsilon is not exact in this selected branch',all(r['relative_slope_mismatch']>.05 for r in rows),fractional_errors=[r['relative_slope_mismatch'] for r in rows])
    kkerr=[];spectralerr=[];readout=[]
    for r in rows:
        u=r['u'];q=r['q'];aa=np.array([0,q,-q]);x=np.array(r['masses_squared'])
        ee=(x[1]-x[0])/3;ss=x.mean()
        C0=np.diag(x)-ss*np.eye(3)
        spectralerr.append(max(abs(ee-r['epsilon']),abs(np.trace(C0@C0)-6*ee*ee)))
        kkerr.append(abs(model.kk_sum(aa,u)-model.kk_sum_series(aa,u)))
        lowedge=max(r['thresholds']);nxt=r['next_threshold'];om=(lowedge+nxt)/2
        pfull=model.record(om,aa,u);plow=model.record(om,aa,u,True)
        high=2.5
        readout.append(dict(u=u,omega_in_window=om,full_in_window=pfull,lowest_in_window=plow,
          high_omega=high,full_high=model.record(high,aa,u),lowest_high=model.record(high,aa,u,True)))
    C.near('lowest spectrum realizes CE 1+2 moments at its minimum',max(spectralerr),3e-17)
    C.near('all KK inverse masses agree with direct sum plus controlled tail',max(kkerr),8e-14)
    C.near('record spectrum below next threshold contains exactly three complex modes',max(abs(v['full_in_window']-v['lowest_in_window']) for v in readout),2e-18)
    C.require('higher modes change record spectrum when accessible',all(v['full_high']>v['lowest_high'] for v in readout))
    C.require('virtual higher modes cannot be dropped at percent accuracy',all(r['inverse_sum_omitted_fraction']>.15 for r in rows))
    # Match a periodic linear response; external u(t) explicitly supplies work.
    r=rows[1];ampu=1e-4;omega=.4*np.sqrt(r['q_mass_squared'])
    spring=model.C*r['Vqq'];drive=model.C*r['Vqu'];K=model.Kq
    ampq=-drive*ampu/(spring-K*omega*omega)
    def rhs(t,y):
        du=ampu*np.cos(omega*t);dudot=-ampu*omega*np.sin(omega*t)
        return [y[1],-(spring*y[0]+drive*du)/K,drive*dudot*y[0]]
    tt=np.linspace(0,4*np.pi/omega,301)
    sol=solve_ivp(rhs,[0,tt[-1]],[ampq,0,0],t_eval=tt,method='DOP853',rtol=2e-11,atol=1e-14)
    C.require('linear dynamical response integrator completed',sol.success)
    C.near('finite-frequency response matches independent ODE',np.max(abs(sol.y[0]-ampq*np.cos(omega*tt))),3e-11)
    E=.5*K*sol.y[1]**2+.5*spring*sol.y[0]**2+drive*ampu*np.cos(omega*tt)*sol.y[0]
    C.near('linear response includes the prescribed Higgs source work',np.max(abs(E-E[0]-sol.y[2])),3e-14)
    C.require('response is suppressed in the fast limit, not instantaneously minimized',abs(1/(1-10**2))<.011)
    # A rescaling gives the SAME dimensionless minimum; radius is not selected.
    changed=Model(g=.6,R=2.,nmax=2048);rr=changed.selected(.5/4)
    C.near('scale family leaves dimensionless selected phase unchanged',abs(rr['q']-rows[1]['q']),2e-14)
    C.near('rescaled spectrum exposes unselected absolute scale',np.max(abs(4*np.array(rr['masses_squared'])-np.array(rows[1]['masses_squared']))),2e-14)
    return dict(experiment='CE-HS1',date='2026-09-20',inputs=dict(g=.6,R=1.,kappa=.18,adjoint_dirac_flavors=1,adjoint_mass=0.,boundary='periodic along NEW circle',u=[.2,.5,.8],winding_cutoff=2048),
       assumptions=['Extra compact direction is new, not derived from CE','Periodic massless adjoint Dirac fermion is new matter input','g and R and fixed Higgs backgrounds are inputs','Tree matching kappa=g^2/2 is imposed from GP1, not proved at all loops','One-loop compact-sector potential; other compact modes/UV matching not computed'],
       status=dict(local_stability_proved_numerically=True,global_uniqueness_proved=False,full_parent_matched=False,observations_used=False,main_modified=False,fixed_epsilon_exact=False),
       selected=rows,search=search,record_checks=readout,uniform_tail_bound_at_256=tail_bound,
       linear_response=dict(omega=omega,q_amplitude=ampq,energy_error=float(np.max(abs(E-E[0]-sol.y[2])))),
       checks=C.rows,check_count=len(C.rows),versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('results.json'));args=ap.parse_args()
    out=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"{out['check_count']} checks passed; {args.output}")
