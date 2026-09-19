#!/usr/bin/env python3
"""CE-RV1: a finite-resolution physical record -> conditional value -> evolution.

The inherited model has one signed Higgs-proxy coordinate and three complex
oscillators. This is NOT a full SM Higgs doublet or a renormalized 4D QFT.
All Hamiltonian parameters are inherited inputs, not fitted or predicted.
A Gaussian instrument measuring the EXISTING operator O is explicitly added;
its resolution sigma is an apparatus input. Born conditioning is assumed.

Run: OPENBLAS_NUM_THREADS=1 python verify_record_values.py --section all
No network, repository modifications, or observational datasets are used.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from scipy.integrate import quad, quad_vec
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply

Array = NDArray[np.float64]

@dataclass(frozen=True)
class Parameters:
    s0: float = .5
    epsilon: float = .15
    kappa: float = 1.
    lam: float = 10.
    u0: float = .5
    theta: float = float(np.pi)
    def validate(self):
        if not (self.s0 > 2*abs(self.epsilon) and self.epsilon >= 0
                and self.kappa >= 0 and self.lam > 0 and self.u0 >= 0):
            raise ValueError('Positive gap, nonnegative portal, confining Higgs proxy required')

@dataclass
class Model:
    H: Any
    u: Array
    u2: Array
    pH2: Array
    nH: int
    nC: int
    params: Parameters
    local_O: list
    local_H: list
    evalO: list
    vecO: list
    energy0: float
    ground: Array
    eigen_residual: float
    wH: float

class Checks:
    def __init__(self): self.rows = []
    def require(self, name: str, ok: bool, **details):
        self.rows.append(dict(name=name, passed=bool(ok), **details))
        if not ok: raise AssertionError(f'{name}: {details}')
        print('PASS '+name,flush=True)
    def near(self, name: str, error: float, tolerance: float):
        self.require(name, np.isfinite(error) and error <= tolerance,
                     error=float(error), tolerance=float(tolerance))

def kron(seq):
    result = sp.csr_matrix([[1.]])
    for a in seq: result = sp.kron(result, a, format='csr')
    return result

def transform(a, matrices):
    """Apply three local transformations, retaining the leading Higgs axis."""
    for axis, matrix in enumerate(matrices, 1):
        a = np.moveaxis(np.tensordot(matrix, a, axes=(1, axis)), 0, axis)
    return a

def build_ground(nH: int, nC: int, params=Parameters(), wH=3.) -> Model:
    params.validate()
    if nH < 3 or nC < 2 or wH <= 0: raise ValueError('Invalid basis')
    # Polynomial projection must include intermediate states outside the basis.
    nf = 2*nH+6
    a = np.diag(np.sqrt(np.arange(1,nf)),1)
    q = (a+a.T)/np.sqrt(2*wH)
    p = 1j*np.sqrt(wH/2)*(a.T-a)
    q2=q@q; q4=q2@q2; p2=(p@p).real
    idx=np.arange(0,2*nH,2); ix=np.ix_(idx,idx)
    u=q2[ix]/2; u2=q4[ix]/4; pH2=p2[ix]
    HH=sp.csr_matrix(pH2/2+params.lam*(u2-2*params.u0*u+params.u0**2*np.eye(nH)))
    x0=params.s0+2*params.epsilon*np.cos((params.theta+2*np.pi*np.arange(3))/3)
    wr=np.sqrt(x0+.5) # numerical coordinates; unchanged when kappa changes
    n=np.arange(nC,dtype=float)
    Os=[]; Hs=[]; es=[]; Us=[]
    for x,w in zip(x0,wr):
        O=np.diag((2*n+1)/(2*w))+np.diag(n[1:]/(2*w),1)+np.diag(n[1:]/(2*w),-1)
        HC=np.diag(w*(2*n+1))+(x-w*w)*O
        vals,V=eigh(O)
        Os.append(O); Hs.append(HC); es.append(vals); Us.append(V)
    IC=sp.eye(nC,format='csr'); IH=sp.eye(nH,format='csr'); IB=kron([IC]*3)
    HC=sp.csr_matrix(IB.shape); OC=sp.csr_matrix(IB.shape)
    for j in range(3):
        fac=[IC]*3; fac[j]=sp.csr_matrix(Hs[j]); HC+=kron(fac)
        fac=[IC]*3; fac[j]=sp.csr_matrix(Os[j]); OC+=kron(fac)
    H=sp.kron(HH,IB,format='csr')+sp.kron(IH,HC,format='csr')+params.kappa*sp.kron(sp.csr_matrix(u),OC,format='csr')
    H=H.tocsr()
    vals,V=eigsh(H,k=1,which='SA',tol=2e-12,ncv=36,maxiter=20000,
                 v0=np.random.default_rng(123).normal(size=H.shape[0]))
    v=V[:,0]; v/=np.linalg.norm(v)
    return Model(H,u,u2,pH2,nH,nC,params,Os,Hs,es,Us,float(vals[0]),v,
                 float(np.linalg.norm(H@v-vals[0]*v)),wH)

class RecordFamily:
    """All records of the Gaussian instrument M_r(O), applied to one joint state."""
    def __init__(self, model: Model, sigma: float=1., state=None):
        if not (np.isfinite(sigma) and sigma > 0): raise ValueError('sigma must be positive')
        self.m=model; self.sigma=float(sigma)
        v=model.ground if state is None else state/np.linalg.norm(state)
        self.v=v
        nH,nC=model.nH,model.nC
        A=transform(v.reshape(nH,nC,nC,nC),[U.T for U in model.vecO])
        self.A=A; self.a=A.reshape(nH,-1)
        e=model.evalO
        self.o=(e[0][:,None,None]+e[1][None,:,None]+e[2][None,None,:]).ravel()
        a=self.a
        col=lambda op: np.sum(a.conj()*(op@a),axis=0).real
        self.weights=np.sum(abs(a)**2,axis=0)
        self.column=np.array([self.weights,col(model.u),col(model.u2),col(model.pH2),
                              self.weights*self.o,col(model.u)*self.o])
    def raw_moments(self,r):
        g=np.exp(-.5*((r-self.o)/self.sigma)**2)/(np.sqrt(2*np.pi)*self.sigma)
        return self.column@g
    def summary(self,r,with_state=False):
        a=self.a*np.exp(-.25*((r-self.o)/self.sigma)**2)/(2*np.pi*self.sigma**2)**.25
        pdf=float(np.sum(abs(a)**2))
        if pdf < np.finfo(float).tiny: raise ValueError('Record density underflow')
        a/=np.sqrt(pdf)
        vals=self.raw_moments(r)/pdf
        _,u,u2,pH2,O,uO=vals
        cov=float(uO-u*O)
        vr=transform(a.reshape(self.m.nH,self.m.nC,self.m.nC,self.m.nC),self.m.vecO).ravel()
        Hv=self.m.H@vr
        uv=(self.m.u@vr.reshape(self.m.nH,-1)).ravel()
        Hu=(self.m.u@Hv.reshape(self.m.nH,-1)).ravel()
        exact_acc=float(2*np.vdot(Hv,Hu).real-2*np.vdot(uv,self.m.H@Hv).real)
        p=self.m.params
        moment_acc=float(pH2-4*p.lam*(u2-p.u0*u)-2*p.kappa*uO)
        out=dict(record=float(r),probability_density=pdf,mean_u=float(u),mean_u2=float(u2),
                 mean_pH2=float(pH2),mean_O=float(O),mean_uO=float(uO),cov_uO=cov,
                 variance_u=float(u2-u*u),std_u=float(np.sqrt(max(0.,u2-u*u))),
                 du_drecord=cov/self.sigma**2,
                 dlogp_drecord=float((O-r)/self.sigma**2),
                 energy=float(np.vdot(vr,Hv).real),
                 acceleration_exact=exact_acc,acceleration_moment=moment_acc,
                 initial_norm_error=float(abs(np.vdot(vr,vr)-1)))
        return (out,vr) if with_state else out
    def unconditional_energy_change(self):
        """Exact finite-matrix dephasing map; not the untruncated commutator identity."""
        delta=0.
        for j in range(3):
            a=np.moveaxis(self.A,j+1,0).reshape(self.m.nC,-1)
            rho=a@a.conj().T
            U=self.m.vecO[j]; e=self.m.evalO[j]
            h=U.T@self.m.local_H[j]@U
            damp=np.exp(-(e[:,None]-e[None,:])**2/(8*self.sigma**2))
            delta+=np.sum(rho.T*h*(damp-1)).real
        return float(delta)
    def basic(self):
        w=self.column.sum(axis=1)
        return dict(nH=self.m.nH,nC=self.m.nC,dimension=self.m.H.shape[0],
                    energy=self.m.energy0,mean_u=float(w[1]),mean_O=float(w[4]),
                    mean_uO=float(w[5]),cov_uO=float(w[5]-w[1]*w[4]),
                    eigen_residual=self.m.eigen_residual,
                    energy_change_finite=self.unconditional_energy_change(),
                    energy_change_untruncated_formula=float(w[4]/(4*self.sigma**2)))

def propagate(model: Model, states, t: float):
    if t == 0: return np.asarray(states,dtype=complex).copy()
    return expm_multiply((-1j*t)*model.H,states,
                         traceA=(-1j*t)*float(model.H.diagonal().sum()))

def static_checks(check: Checks):
    records=(.5,2.,4.)
    families=[]; summaries=[]
    for nC in (9,13,17,21):
        f=RecordFamily(build_ground(24,nC))
        rows=[f.summary(r) for r in records]
        summaries.append(dict(base=f.basic(),records=rows))
        print(f'static nH=24 nC={nC}: '+', '.join(f"u({x['record']})={x['mean_u']:.10f}" for x in rows),flush=True)
        # Only retain the largest family; earlier numerical rows remain in results.
        families=[f]
    f=families[0]; m=f.m; high=summaries[-1]; low=summaries[-2]
    check.near('01 inherited ground-state value reproduced',abs(f.basic()['mean_u']-.2403996848091),2e-9)
    check.near('02 ground eigen-equation residual',m.eigen_residual,3e-10)
    check.near('03 Hamiltonian is Hermitian',float(np.max(abs((m.H-m.H.T).data),initial=0)),1e-12)
    u_error=max(abs(a['mean_u']-b['mean_u']) for a,b in zip(low['records'],high['records']))
    check.near('04 conditional u basis convergence 17->21',u_error,1e-8)
    acc_error=max(abs(a['acceleration_exact']-b['acceleration_exact']) for a,b in zip(low['records'],high['records']))
    check.near('05 conditional acceleration basis convergence',acc_error,2e-8)
    e_error=max(abs(a['energy']-b['energy']) for a,b in zip(low['records'],high['records']))
    check.near('06 conditional energy basis convergence',e_error,1.2e-5)
    check.near('07 exact commutator vs physical moment acceleration',
               max(abs(x['acceleration_exact']-x['acceleration_moment']) for x in high['records']),3e-9)
    # Numerical normalization and law of total expectation, using unnormalized columns.
    integrated,quaderr=quad_vec(f.raw_moments,-np.inf,np.inf,epsabs=3e-11,epsrel=3e-11)
    expected=f.column.sum(axis=1)
    check.near('08 record density normalizes',abs(integrated[0]-1),3e-10)
    check.near('09 average conditioned moments recover original joint moments',float(np.max(abs(integrated-expected))),5e-10)
    slope_errors=[]; score_errors=[]; source_errors=[]
    for row in high['records']:
        r=row['record']; h=2e-4
        # Differentiate ratios and log densities without using analytic covariances.
        a=f.raw_moments(r-h); b=f.raw_moments(r+h)
        slope=(b[1]/b[0]-a[1]/a[0])/(2*h)
        score=(np.log(b[0])-np.log(a[0]))/(2*h)
        slope_errors.append(abs(slope-row['du_drecord']))
        score_errors.append(abs(score-row['dlogp_drecord']))
        reconstructed=row['mean_u']*(r+score*f.sigma**2)+f.sigma**2*slope
        source_errors.append(abs(reconstructed-row['mean_uO']))
    check.near('10 record-to-value derivative equals conditional covariance',max(slope_errors),3e-10)
    check.near('11 density score reconstructs conditional O',max(score_errors),3e-9)
    check.near('12 record derivatives reconstruct full correlated source',max(source_errors),2e-9)
    check.require('13 anticorrelation in the tested records, not a universal sign theorem',
                  all(x['cov_uO']<0 for x in high['records']))
    check.require('14 dropping u-O correlation changes acceleration',
                  all(abs(2*x['cov_uO'])>0.014 for x in high['records']))
    heat_res=[abs(x['base']['energy_change_finite']-x['base']['energy_change_untruncated_formula']) for x in summaries]
    check.require('15 heating commutator identity approached with increasing cutoff',
                  all(a>b for a,b in zip(heat_res,heat_res[1:])),residuals=heat_res)
    check.near('16 finite record has finite positive mean energy cost',heat_res[-1],5e-7)
    # A separate small matrix calculation implements the dephasing channel directly.
    sm=build_ground(5,3); sf=RecordFamily(sm)
    B=sf.a; o=sf.o
    full=B.reshape(-1,1)@B.reshape(1,-1).conj()
    df=np.exp(-(o[:,None]-o[None,:])**2/8)
    channel=full.reshape(sm.nH,len(o),sm.nH,len(o))*df[None,:,None,:]
    rhoH=np.einsum('iaja->ij',channel)
    check.near('17 a nonselective local record cannot instantly change Higgs state',
               float(np.linalg.norm(rhoH-B@B.conj().T)),2e-13)
    # Independent quadrature over finite-matrix conditional energies.
    def e_integrand(r):
        a=sf.a*np.exp(-.25*(r-sf.o)**2)/(2*np.pi)**.25
        v=transform(a.reshape(sm.nH,sm.nC,sm.nC,sm.nC),sm.vecO).ravel()
        return float(np.vdot(v,sm.H@v).real)
    e_avg,e_err=quad(e_integrand,-np.inf,np.inf,epsabs=2e-10,epsrel=2e-10)
    check.near('18 instrument-energy quadrature equals exact finite channel',
               abs(e_avg-sm.energy0-sf.unconditional_energy_change()),2e-9)
    # When kappa=0, a product ground state has no record-conditioned Higgs shift.
    off=RecordFamily(build_ground(18,11,Parameters(kappa=0.)))
    offvals=[off.summary(r)['mean_u'] for r in records]
    check.near('19 zero-portal control has no conditional Higgs selection',float(np.ptp(offvals)),2e-10)
    weak=RecordFamily(m,sigma=1e4)
    vals=weak.raw_moments(2.)
    check.near('20 uninformative detector limit recovers original mean_u',abs(vals[1]/vals[0]-expected[1]),2e-9)
    # Record p(r) alone does not determine the Higgs state absent a joint-state law.
    c=np.zeros(sm.nC**3); c[0]=1
    h0=np.eye(sm.nH)[:,0];h1=np.eye(sm.nH)[:,1]
    ff0=RecordFamily(sm,state=np.kron(h0,c));ff1=RecordFamily(sm,state=np.kron(h1,c))
    ps0=[ff0.raw_moments(r)[0] for r in records];ps1=[ff1.raw_moments(r)[0] for r in records]
    check.near('21 same environmental record density can hide different Higgs states',float(np.max(abs(np.array(ps0)-ps1))),2e-14)
    check.require('22 joint-state assumption is necessary for record-value inference',abs(sm.u[0,0]-sm.u[1,1])>.1)
    return dict(convergence=summaries,quadrature=integrated.tolist(),quadrature_estimate=float(quaderr),
                slope_errors=slope_errors,score_errors=score_errors,source_errors=source_errors,
                heating_residuals=heat_res,zero_portal_values=offvals,
                scope='finite jointly quantized proxy; given ground-state preparation and Gaussian instrument')

def spectral_checks(check: Checks):
    # Distinct frozen-background Gaussian limit, NOT applied to the entangled joint state.
    rng=np.random.default_rng(290619)
    S=np.array([[0,1,0],[0,0,1],[1,0,0]],complex)
    errs=[]; inverrs=[]; looperrs=[]
    examples=[]
    for _ in range(24):
        s=float(rng.uniform(.7,1.3));eps=.15;theta=float(rng.uniform(-np.pi,np.pi));k=.7
        X=s*np.eye(3)+eps*(np.exp(1j*theta/3)*S+np.exp(-1j*theta/3)*S.conj().T)
        x,V=eigh(X)
        C=(V*(1/(2*np.sqrt(k*k+x))))@V.conj().T
        ci=np.linalg.inv(C); rec=.25*ci@ci-k*k*np.eye(3)
        sr=float(np.trace(rec).real/3);Y=rec-sr*np.eye(3)
        er=float(np.sqrt(np.trace(Y@Y).real/6))
        cr=float(np.trace(Y@Y@Y).real/(6*er**3))
        loop=Y[0,1]*Y[1,2]*Y[2,0]/er**3
        errs.append(float(np.linalg.norm(rec-X)))
        inverrs.append(max(abs(sr-s),abs(er-eps),abs(cr-np.cos(theta))))
        looperrs.append(float(abs(loop-np.exp(1j*theta))))
    check.near('23 Gaussian covariance reconstructs the original CE mass matrix',max(errs),3e-13)
    check.near('24 centered invariants recover s epsilon cos(theta)',max(inverrs),3e-12)
    check.near('25 oriented cycle needs channel-frame correlations',max(looperrs),3e-12)
    # Same field covariance in a different occupied state: vacuum inverse is conditional.
    x1=.7;x2=1.4;occupation=.5*(np.sqrt(x2/x1)-1)
    C1=.5/np.sqrt(x1);C2=(occupation+.5)/np.sqrt(x2)
    check.near('26 vacuum reconstruction cannot be applied to an unknown occupied state',abs(C1-C2),2e-15)
    # Resolved complex oscillator intensity r has density 2w exp(-2wr).
    x=np.array([.7,1.15,1.15]);kappa=1.;k=.7
    w=np.sqrt(k*k+x)
    fisher=[]; fidelity=[]
    for wi in w:
        density=lambda r:2*wi*np.exp(-2*wi*r)
        score=lambda r:kappa/(2*wi*wi)-kappa*r/wi
        n=quad(density,0,np.inf,epsabs=1e-12)[0]
        fi=quad(lambda r:density(r)*score(r)**2,0,np.inf,epsabs=1e-12)[0]
        fisher.append(fi)
        wj=wi+.2
        b=quad(lambda r:2*np.sqrt(wi*wj)*np.exp(-(wi+wj)*r),0,np.inf,epsabs=1e-12)[0]
        fidelity.append(abs(b-2*np.sqrt(wi*wj)/(wi+wj)))
        if abs(n-1)>2e-12: raise AssertionError('Intensity density normalization')
    check.near('27 resolved intensity records attain the fixed-vacuum local QFI',
               float(np.max(abs(np.array(fisher)-kappa**2/(4*w**4)))),2e-12)
    check.near('28 intensity-record classical affinity equals vacuum amplitude overlap',max(fidelity),2e-12)
    fden=quad(lambda k:sum(k*k*kappa*kappa/(4*(k*k+xj)**2) for xj in x)/(2*np.pi*np.pi),0,np.inf,epsabs=1e-12)[0]
    target=kappa*kappa/(32*np.pi)*sum(1/np.sqrt(x))
    check.near('29 continuum record Fisher density matches CE-CR1',abs(fden-target),2e-12)
    # Independent high-precision derivatives of the inherited one-loop function.
    import mpmath as mp
    mp.mp.dps=55
    ee=mp.mpf('.15');tt=mp.mpf('1.2');ss=mp.mpf('1')
    def xs(s,t):return [s+2*ee*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
    def fun(s):
        f=lambda x:x*x*(mp.log(x)-mp.mpf('1.5'))
        return (sum(f(x) for x in xs(s,tt))-sum(f(x) for x in xs(s,mp.pi)))/(32*mp.pi**2)
    d2=mp.diff(fun,ss,2);d3=mp.diff(fun,ss,3)
    det=lambda t:ss**3-3*ss*ee**2+2*ee**3*mp.cos(t)
    r2=mp.log(det(tt)/det(mp.pi))/(16*mp.pi**2)
    r3=(sum(1/x for x in xs(ss,tt))-sum(1/x for x in xs(ss,mp.pi)))/(16*mp.pi**2)
    check.near('30 same spectrum recovers the inherited unified response derivatives',float(max(abs(d2-r2),abs(d3-r3))),1e-45)
    return dict(max_covariance_reconstruction_error=max(errs),max_invariant_error=max(inverrs),
                occupied_state_counterexample=dict(x1=x1,x2=x2,occupation=occupation,C1=C1,C2=C2),
                fisher_density=fden,source_Uss=float(d2),source_Usss=float(d3),
                scope='Separate fixed-background Gaussian vacuum limit. Not the joint entangled ground state.')

def time_checks(check: Checks):
    records=(.5,2.,4.);grids=[]; final=None
    checkpoint=Path(__file__).with_name('time_checkpoint.json')
    if os.environ.get('CE_RV_RESUME')=='1' and checkpoint.exists():
        grids=json.loads(checkpoint.read_text())
        print('Resuming completed time grids from local checkpoint',flush=True)
    times=np.linspace(0,1,5)
    for nC in (9,13,17):
        if any(g['nC']==nC for g in grids):continue
        m=build_ground(18,nC,wH=8.); f=RecordFamily(m)
        info_states=[f.summary(r,with_state=True) for r in records]
        B=np.column_stack([v for _,v in info_states]+[m.ground]).astype(complex)
        tick=time.monotonic()
        trajectory=expm_multiply(-1j*m.H,B,start=0.,stop=1.,num=5,endpoint=True,
                                 traceA=-1j*float(m.H.diagonal().sum()))
        rows=[];maxnorm=0.;maxenergy=0.
        E0=[row['energy'] for row,v in info_states]+[m.energy0]
        for t,bt in zip(times,trajectory):
            us=[];es=[]
            for j in range(4):
                v=bt[:,j];a=v.reshape(m.nH,-1)
                us.append(float(np.sum(a.conj()*(m.u@a)).real))
                es.append(float(np.vdot(v,m.H@v).real))
                maxnorm=max(maxnorm,float(abs(np.vdot(v,v)-1)))
                maxenergy=max(maxenergy,abs(es[-1]-E0[j]))
            rows.append(dict(time=float(t),conditional_u=us[:3],unmeasured_u=us[3],energies=es))
        grids.append(dict(nH=18,nC=nC,numerical_wH=8.,dimension=m.H.shape[0],initial=[r for r,v in info_states],
                          trajectory=rows,max_norm_error=maxnorm,max_energy_drift=maxenergy))
        Path(__file__).with_name('time_checkpoint.json').write_text(json.dumps(grids,indent=2)+'\n')
        print(f'time nH=18 nC={nC}: u(t=1)={rows[-1]["conditional_u"]}; {time.monotonic()-tick:.2f}s',flush=True)
        final=(m,f,B,trajectory)
    check.near('31 conditional full-quantum trajectories preserve norms',max(g['max_norm_error'] for g in grids),3e-11)
    check.near('32 conditional trajectories preserve post-record energies',max(g['max_energy_drift'] for g in grids),2e-10)
    check.near('33 no-readout ground-state trajectory remains stationary',
               max(abs(row['unmeasured_u']-.2403996848091) for g in grids for row in g['trajectory']),2e-9)
    convergence={}
    for a,b,label in [(grids[0],grids[1],'9_to_13'),(grids[1],grids[2],'13_to_17')]:
        errs=[float(np.max(abs(np.array(x['conditional_u'])-y['conditional_u']))) for x,y in zip(a['trajectory'],b['trajectory'])]
        convergence[label]=errs
    check.require('34 increasing environment basis improves time-trajectory convergence',
                  max(convergence['13_to_17'])<max(convergence['9_to_13']),errors=convergence)
    # A declared absolute numerical target, not an observational accuracy claim.
    check.near('35 resolved conditional trajectories through t=1',max(convergence['13_to_17']),5e-5)
    # Increase the Higgs basis and change its frequency independently at nC=9.
    mm=build_ground(24,9);ff=RecordFamily(mm)
    BB=np.column_stack([ff.summary(r,with_state=True)[1] for r in records])
    bt=propagate(mm,BB,.5)
    uh=[]
    for j in range(3):
        a=bt[:,j].reshape(mm.nH,-1);uh.append(float(np.sum(a.conj()*(mm.u@a)).real))
    hdiff=float(np.max(abs(np.array(uh)-grids[0]['trajectory'][2]['conditional_u'])))
    check.near('36 independent Higgs-basis dimension and frequency convergence at t=.5',hdiff,2e-7)
    # Direct short-time evolution tests the conditional second-derivative identity.
    m=build_ground(18,9,wH=8.);f=RecordFamily(m);B=np.column_stack([f.summary(r,with_state=True)[1] for r in records]);h=1e-3
    bb1=propagate(m,B[:,:3],h);bb2=propagate(m,B[:,:3],2*h)
    fderr=[]
    for j,r in enumerate(records):
        fun=lambda v:float(np.sum(v.reshape(m.nH,-1).conj()*(m.u@v.reshape(m.nH,-1))).real)
        # Real H and real initial psi make <u>(-t)=<u>(t).
        second=(-fun(bb2[:,j])+16*fun(bb1[:,j])-15*fun(B[:,j]))/(6*h*h)
        fderr.append(abs(second-grids[0]['initial'][j]['acceleration_exact']))
    check.near('37 direct unitary second derivative agrees with correlated source',max(fderr),2e-7)
    # Independent dense spectral vs sparse exponential calculation in a small model.
    sm=build_ground(6,4);sf=RecordFamily(sm)
    _,v=sf.summary(.5,with_state=True)
    e,V=eigh(sm.H.toarray());direct=V@(np.exp(-.5j*e)*(V.T@v))
    sparse=propagate(sm,v,.5)
    check.near('38 dense spectral and sparse unitary evolution agree',float(np.linalg.norm(direct-sparse)),5e-11)
    shifted=expm_multiply(-.5j*(sm.H+1.7*sp.eye(sm.H.shape[0],format='csr')),v,
                          traceA=-.5j*(float(sm.H.diagonal().sum())+1.7*sm.H.shape[0]))
    check.near('39 additive energy constant leaves record-conditioned physics unchanged in fixed spacetime',
               float(np.linalg.norm(shifted-np.exp(-.5j*1.7)*sparse)),5e-11)
    # A record affects future values through an unequal-time SYMMETRIZED covariance.
    km=build_ground(18,9,wH=8.);kf=RecordFamily(km);rr=2.;dh=2e-4;tt=.5
    row,vc=kf.summary(rr,with_state=True)
    vm=kf.summary(rr-dh,with_state=True)[1];vp=kf.summary(rr+dh,with_state=True)[1]
    ac=transform(vc.reshape(km.nH,km.nC,km.nC,km.nC),[U.T for U in km.vecO])
    ov=transform((ac.reshape(km.nH,-1)*kf.o).reshape(ac.shape),km.vecO).ravel()
    vv=propagate(km,np.column_stack([vm,vc,vp,ov]),tt)
    u_of=lambda v:float(np.vdot(v,(km.u@v.reshape(km.nH,-1)).ravel()).real)
    fd=(u_of(vv[:,2])-u_of(vv[:,0]))/(2*dh)
    corr=float(np.vdot(vv[:,1],(km.u@vv[:,3].reshape(km.nH,-1)).ravel()).real)
    analytic=(corr-u_of(vv[:,1])*row['mean_O'])/kf.sigma**2
    check.near('40 future record-to-value slope equals unequal-time symmetrized covariance',abs(fd-analytic),3e-9)
    future_kernel=dict(record=rr,time=tt,finite_difference=fd,correlation_prediction=analytic,error=abs(fd-analytic))
    return dict(convergence=grids,trajectory_cutoff_errors=convergence,Higgs_basis_error=hdiff,
                short_time_derivative_errors=fderr,future_record_kernel=future_kernel,
                scope='Exact unitary evolution of each finite projected joint Hamiltonian; convergence checks are numerical, not interval certificates.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--section',choices=['all','static','time'],default='all')
    ap.add_argument('--output',type=Path,default=Path('results.json'))
    args=ap.parse_args();check=Checks();start=time.monotonic()
    out=dict(experiment='CE-RV1',section=args.section,physical_inputs=asdict(Parameters()),
             apparatus=dict(operator='O=sum_j chi_j^dagger chi_j',sigma=1.,records=[.5,2.,4.],
                            record_labels_are_density_arguments_not_event_probabilities=True),
             interpretation=dict(observational_fit=False,observational_prediction=False,
                predicts_fundamental_constants=False,derives_Born_rule=False,
                supplied_ground_state_boundary_condition=True,full_SM=False,continuum_QFT=False,
                conditional_values_are_outputs=True,external_Higgs_time_history=False,
                detector_instrument_assumed=True,measurement_energy_included=True),
             versions={x:importlib.metadata.version(x) for x in ['numpy','scipy','mpmath']})
    try:
        if args.section in ('all','static'):
            out['static']=static_checks(check);out['gaussian_limit']=spectral_checks(check)
        if args.section in ('all','time'):out['time']=time_checks(check)
        out['passed']=True
    except Exception as exc:
        out['passed']=False;out['error']=repr(exc)
        raise
    finally:
        out['checks']=check.rows;out['check_count']=len(check.rows)
        out['elapsed_seconds']=time.monotonic()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{len(check.rows)} checks passed; saved {args.output}',flush=True)

if __name__=='__main__':main()
