#!/usr/bin/env python3
"""CE-RI1: spectral records -> response and identifiable parameters.

New diagnostics, no observations and no fitting. The field calculation is a
fixed-source free complex-scalar bath in 3+1D; source influence is second order.
The separate joint-state check is the inherited finite-mode Higgs proxy.
The two approximations are never equated. No natural-constant selection claim.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import quad
from scipy.linalg import eigh, expm

ROOT=Path(__file__).resolve().parent

class Checks:
    def __init__(self): self.rows=[]
    def require(self,name,ok,**detail):
        row=dict(name=name,passed=bool(ok),**detail); self.rows.append(row)
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
        if not ok: raise AssertionError(row)
    def near(self,name,error,tol):
        self.require(name,np.isfinite(error) and error<=tol,error=float(error),tolerance=float(tol))

@dataclass(frozen=True)
class Params:
    s0: float=.5
    epsilon: float=.15
    kappa: float=1.
    def masses2(self,u,theta=np.pi):
        x=self.s0+self.kappa*u+2*self.epsilon*np.cos((theta+2*np.pi*np.arange(3))/3)
        if np.min(x)<=0: raise ValueError('Non-positive squared mass')
        return np.sort(x)

def shape(omega,x):
    om=np.asarray(omega,dtype=float)
    return np.sqrt(np.maximum(0.,1.-4*x/np.maximum(om*om,1e-300)))*(om>2*np.sqrt(x))

def noise_plus(omega,x,kappa=1.):
    if omega<=0:return 0.
    return float(kappa*kappa*sum(shape(omega,float(v)) for v in x)/(8*np.pi))

def phase_space_smeared(omega,x,width):
    # Start from d^3p/(2pi)^3 * (2pi)/(4 E^2) delta(omega-2E),
    # independently integrating the radial momentum with a normalized mollifier.
    lo=max(2*np.sqrt(x),omega-10*width); hi=omega+10*width
    if hi<=lo:return 0.
    p0=np.sqrt(max(0.,lo*lo/4-x)); p1=np.sqrt(hi*hi/4-x)
    def f(p):
        E=np.sqrt(p*p+x)
        d=np.exp(-.5*((omega-2*E)/width)**2)/(np.sqrt(2*np.pi)*width)
        return p*p/E**2*d/(4*np.pi)
    return quad(f,p0,p1,epsabs=1e-13,epsrel=1e-12,limit=100)[0]

def bubble_sub_feynman(Q2,x):
    return -quad(lambda a:np.log1p(a*(1-a)*Q2/x),0,1,epsabs=1e-13,epsrel=1e-13)[0]/(16*np.pi**2)

def bubble_sub_spectral(Q2,x):
    # t=4x/(1-v^2) in integral rho(t) [1/(t+Q2)-1/t]dt.
    return -Q2/(8*np.pi**2)*quad(lambda v:v*v/(4*x+Q2*(1-v*v)),0,1,epsabs=1e-13,epsrel=1e-13)[0]

def relative_Uss(s,e,theta):
    D=lambda t:s**3-3*s*e*e+2*e**3*np.cos(t)
    return np.log(D(theta)/D(np.pi))/(16*np.pi**2)

def record_sum_rule(s,e,theta,T=1e8):
    # Integrate the measured threshold spectrum analytically up to a common
    # invariant-mass-squared cutoff. High precision avoids subtractive loss.
    with mp.workdps(65):
        sm,em,tm=mp.mpf(str(s)),mp.mpf(str(e)),mp.mpf(str(theta))
        def primitive(x):
            v=mp.sqrt(1-4*x/mp.mpf(T))
            return (mp.log((1+v)/(1-v))-2*v)/(16*mp.pi**2)
        xs=lambda t:[sm+2*em*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
        return float(-sum(primitive(x) for x in xs(tm))+sum(primitive(x) for x in xs(mp.pi)))

def field_checks(C):
    p=Params();ua,ub=.2,.8
    xa,xb=p.masses2(ua),p.masses2(ub)
    S=np.roll(np.eye(3),1,axis=0)
    th=1.2;s=p.s0+p.kappa*.5
    X=s*np.eye(3)+p.epsilon*(np.exp(1j*th/3)*S+np.exp(-1j*th/3)*S.T)
    C.near('original circulant spectrum',np.max(abs(np.linalg.eigvalsh(X)-p.masses2(.5,th))),2e-15)
    phase=[]
    for x,om in [(.4,1.7),(.85,2.4),(1.45,3.0)]:
        exact=float(shape(om,x)/(8*np.pi))
        a=phase_space_smeared(om,x,.008);b=phase_space_smeared(om,x,.002)
        phase.append(dict(x=x,omega=om,exact=exact,coarse=a,fine=b,error=abs(b-exact)))
    C.near('two-body normalization via independent momentum integral',max(r['error'] for r in phase),2e-7)
    C.require('delta-sequence phase-space convergence',all(abs(r['fine']-r['exact'])<abs(r['coarse']-r['exact'])/10 for r in phase))
    C.near('strict vacuum threshold',noise_plus(2*np.sqrt(xa[0])*.999,xa),0.)
    C.require('spectral positivity above threshold',all(noise_plus(om,xa)>0 for om in [1.4,2.,3.]))
    Ta=2*np.sqrt(xa[[0,1]]);Tb=2*np.sqrt(xb[[0,1]])
    def inv(T):return ((T[0]**2+2*T[1]**2)/12,(T[1]**2-T[0]**2)/12)
    sa,ea=inv(Ta);sb,eb=inv(Tb);ki=(sb-sa)/(ub-ua);s0i=sa-ki*ua
    C.near('algebraic two-background parameter identification',max(abs(ki-p.kappa),abs(s0i-p.s0),abs(ea-p.epsilon),abs(eb-p.epsilon)),2e-15)
    om=(Ta[0]+Ta[1])/2
    amp=np.sqrt(8*np.pi*noise_plus(om,xa)/float(shape(om,xa[0])))
    C.near('independent force-noise amplitude portal check',abs(amp-ki),2e-15)
    omhi=1.1*Ta[1]
    mult=(8*np.pi*noise_plus(omhi,xa)-ki*ki*shape(omhi,xa[0]))/(ki*ki*shape(omhi,xa[1]))
    C.near('one-plus-two multiplicity from second onset',abs(mult-2),2e-14)
    slopes=(Tb**2-Ta**2)/(ub-ua)
    C.near('common portal slopes and epsilon protection test',np.max(abs(slopes-4*ki)),3e-15)
    # Flat space dilation: records without an independently calibrated clock cannot select a mass unit.
    a=2.3
    C.near('absolute-scale ambiguity without clock calibration',abs(noise_plus(a*2.6,xa*a*a)-noise_plus(2.6,xa)),2e-17)
    C.require('epsilon-zero recording counterexample retained',noise_plus(3.,Params(epsilon=0).masses2(ua))>0)
    vals=[]
    for x in [.4,.85,1.45]:
        for q in [.01,.7,5.,50.]:
            f=bubble_sub_feynman(q,x);b=bubble_sub_spectral(q,x)
            vals.append(dict(x=x,Q2=q,feynman=f,spectral=b,error=abs(f-b)))
    C.near('subtracted dispersion reproduces Feynman parameter loop',max(r['error'] for r in vals),2e-14)
    # direct spectral moment, with t=4x/(1-v^2)
    dm=[]
    for x in [.4,.85,1.45]:
        integ=quad(lambda v:2*v*v/(4*x),0,1,epsabs=1e-13)[0]/(16*np.pi**2)
        dm.append(abs(integ-1/(96*np.pi**2*x)))
    C.near('finite first derivative sum rule',max(dm),2e-17)
    us=[]
    for st,et,tht in [(1.,.15,0.),(1.,.15,1.2),(.6,.15,2.2),(1.4,.2,.6)]:
        direct=relative_Uss(st,et,tht);spec=record_sum_rule(st,et,tht)
        us.append(dict(s=st,epsilon=et,theta=tht,direct=direct,from_noise=spec,error=abs(direct-spec)))
    C.near('CE U_ss reconstructed from relative noise sum rule',max(r['error'] for r in us),2e-16)
    # Noise and response conventions: chi_R=-i theta(t)<[F(t),F(0)]>.
    omegas=[-3.,-1.5,1.,1.5,3.]
    fdterr=[]
    for om in omegas:
        sp=noise_plus(om,xa);sm=noise_plus(-om,xa)
        N=(sp+sm)/2;im=-(sp-sm)/2
        fdterr.append(abs(N+np.sign(om)*im))
    C.near('zero-temperature fluctuation-dissipation convention',max(fdterr),0.)
    # KMS reconstruction is conditional on a stationary zero-mean Gaussian Gibbs family.
    therm=[];beta=2.
    for om in [1.5,2.,3.]:
        n=1/np.expm1(beta*om/2);vac=noise_plus(om,xa)
        plus=vac*(1+n)**2;minus=vac*n*n
        beta_rec=np.log(plus/minus)/om
        therm.append(dict(omega=om,positive=plus,negative=minus,beta_from_ratio=beta_rec))
    C.near('thermal beta identifiable from detailed-balance records',max(abs(r['beta_from_ratio']-beta) for r in therm),2e-15)
    # A nonthermal stationary occupation function fails frequency-independent beta.
    betas=[]
    for om in [1.5,2.,3.]:
        n=.3/(1+om*om)
        betas.append(2*np.log((1+n)/n)/om)
    C.require('nonthermal negative control against automatic Gibbs assumption',np.ptp(betas)>.3,inferred_betas=betas)
    # Packet with Fourier support below the pair threshold has no on-shell bath excitation at O(kappa^2).
    C.near('subthreshold vacuum has no linear-response absorption',sum(noise_plus(om,xa) for om in [.1,.5,1.0]),0.)
    return dict(phase_space=phase,record_fixture=dict(ua=ua,ub=ub,masses2_a=xa.tolist(),masses2_b=xb.tolist(),thresholds_a=Ta.tolist(),thresholds_b=Tb.tolist(),recovered_s0=s0i,recovered_epsilon=ea,recovered_kappa=ki,amplitude_kappa=amp,second_multiplicity=float(mult)),dispersion=vals,relative_sum_rules=us,thermal=therm)

def renormalization_checks(C):
    # Hard Euclidean 4-momentum cutoff: dV/dx = [L2-x log(1+L2/x)]/(16pi2).
    # Unequal radial backgrounds share sum x but not sum x^2: a logarithmic term survives.
    with mp.workdps(65):
        s=mp.mpf('1');e1=mp.mpf('.1');e2=mp.mpf('.2')
        def V(x,L):
            A=L*L;y=A+x
            primitive=lambda yy: (yy*yy/2-x*yy)*mp.log(yy)-yy*yy/4+x*yy
            return (primitive(A+x)-primitive(x))/(16*mp.pi**2)
        xs=lambda e:[s-2*e,s+e,s+e]
        def delta(L):return sum(V(x,mp.mpf(L)) for x in xs(e2))-sum(V(x,mp.mpf(L)) for x in xs(e1))
        val=[float(delta(L)) for L in [100,1000,10000]]
        target=float(-6*(e2*e2-e1*e1)/(16*mp.pi**2))
        slope=(val[2]-val[1])/np.log(10)
    C.near('radial variation reintroduces the predicted logarithmic divergence',abs(slope-target),2e-9)
    # Same bath records at prescribed u, different allowed local potentials.
    p=Params();u=.5;om=2.4;x=p.masses2(u)
    alternatives=[]
    for lam,u0 in [(10.,.5),(13.,.3)]:
        alternatives.append(dict(lam=lam,u0=u0,bath_noise=noise_plus(om,x),local_force=-2*lam*(u-u0)))
    C.near('fixed-source bath records do not fix Higgs local potential',abs(alternatives[0]['bath_noise']-alternatives[1]['bath_noise']),0.)
    C.require('same bath records permit different conservative source',abs(alternatives[0]['local_force']-alternatives[1]['local_force'])>1.)
    # With theta fixed at pi, relative U is zero for every s and e; phase selection cannot fix radius.
    C.near('relative phase minimum leaves radial parameters free',max(abs(relative_Uss(s,e,np.pi)) for s,e in [(1.,.1),(1.,.2),(.8,.3)]),0.)
    return dict(cutoffs=[100,1000,10000],radial_difference=val,log_slope=slope,predicted_log_slope=target,local_counterexamples=alternatives)

def import_inherited():
    path=ROOT/'inherited'/'ce_rv1.py'
    spec=importlib.util.spec_from_file_location('ce_rv1',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def joint_identification_checks(C):
    rv=import_inherited();runs=[]
    for nH,nC,wH in [(20,7,3.),(24,9,3.)]:
        m=rv.build_ground(nH,nC,wH=wH);f=rv.RecordFamily(m)
        states=[f.summary(r,with_state=True) for r in [.5,4.]]
        rows=[r for r,v in states]
        # Acceleration is evaluated by the full double commutator, NOT by the moment identity being inverted.
        mat=np.array([[r['mean_u2'],r['mean_u']] for r in rows])
        rhs=np.array([r['mean_pH2']-2*m.params.kappa*r['mean_uO']-r['acceleration_exact'] for r in rows])
        A,B=np.linalg.solve(mat,rhs);lam=A/4;u0=-B/A
        # Independent held-out record.
        r2,v2=f.summary(2.,with_state=True)
        acc_pred=r2['mean_pH2']-4*lam*(r2['mean_u2']-u0*r2['mean_u'])-2*m.params.kappa*r2['mean_uO']
        runs.append(dict(nH=nH,nC=nC,dimension=m.H.shape[0],ground_energy=m.energy0,eigen_residual=m.eigen_residual,matrix_condition=float(np.linalg.cond(mat)),recovered_lambda=float(lam),recovered_u0=float(u0),held_out_acceleration=float(acc_pred),held_out_exact=r2['acceleration_exact'],records=rows))
        if nH==24:
            row,v=states[0];h=.001
            def mean(t):
                psi=rv.propagate(m,v,t)
                return float(np.vdot(psi.reshape(m.nH,-1),m.u@psi.reshape(m.nH,-1)).real)
            f0=row['mean_u'];f1=mean(h);f2=mean(2*h)
            finite=(-2*f2+32*f1-30*f0)/(12*h*h)
            C.near('independent time evolution verifies acceleration used for identification',abs(finite-row['acceleration_exact']),2e-7)
            # Adding a multiple of identity leaves all probabilities and conditional means unchanged.
            def diag_add(t,c):return np.exp(-1j*c*t)*rv.propagate(m,v,t)
            psi=rv.propagate(m,v,.03);psic=diag_add(.03,7.)
            C.near('absolute vacuum constant is invisible to flat-space normalized records',abs(np.vdot(psi,m.H@psi)-np.vdot(psic,m.H@psic)),2e-12)
    C.near('local Higgs coefficients algebraically recovered from two records',max(max(abs(r['recovered_lambda']-10),abs(r['recovered_u0']-.5)) for r in runs),2e-7)
    C.near('third-record force prediction without refitting',max(abs(r['held_out_acceleration']-r['held_out_exact']) for r in runs),2e-9)
    C.near('coefficient identification under basis extension',abs(runs[0]['recovered_lambda']-runs[1]['recovered_lambda']),2e-7)
    return dict(runs=runs,finite_time_acceleration=finite)

def parity_checks(C):
    # A separate finite proxy example, not a re-quantization of the 4D field.
    n=7;nf=n+4;a=np.diag(np.sqrt(np.arange(1,nf)),1)
    q=(a+a.T)/np.sqrt(2);mom=1j*(a.T-a)/np.sqrt(2)
    q2=(q@q)[:n,:n];q4=(q@q@q@q)[:n,:n];p2=(mom@mom)[:n,:n]
    u=q2/2;O=q2/2;I=np.eye(n)
    Hh=p2/2+2*(q4/4-.7*q2+.49*I)
    Hc=p2/2+.8*q2/2
    H=np.kron(Hh,I)+np.kron(I,Hc)+.9*np.kron(u,O)
    P=np.kron(np.diag((-1.)**np.arange(n)),I);OO=np.kron(I,O)
    C.near('parity is conserved by the interacting diagnostic',np.linalg.norm(H@P-P@H),0.)
    C.require('parity example has genuine dynamical bath coupling',np.linalg.norm(H@OO-OO@H)>1.)
    ev,V=eigh(OO)
    def M(r):return (V*np.exp(-.25*(r-ev)**2)/(2*np.pi)**.25)@V.conj().T
    e=np.zeros(n*n);o=e.copy();e[0]=1;o[n]=1
    plus=(e+o)/np.sqrt(2);minus=(e-o)/np.sqrt(2)
    C.near('indistinguishable input states are orthogonal',abs(np.vdot(plus,minus)),2e-16)
    seqs=[[(.2,.5)],[(.1,1.2),(.3,.7)],[(.15,.4),(.2,1.7),(.1,2.1)]]
    diffs=[];probs=[]
    for seq in seqs:
        vp=plus.copy();vm=minus.copy()
        for t,r in seq:
            U=expm(-1j*t*H);T=M(r)@U;vp=T@vp;vm=T@vm
        pp=float(np.vdot(vp,vp).real);pm=float(np.vdot(vm,vm).real)
        diffs.append(abs(pp-pm));probs.append(dict(sequence=seq,probability_density_plus=pp,probability_density_minus=pm))
    C.near('entire even-record histories fail to select initial parity coherence',max(diffs),2e-15)
    return dict(history_tests=probs,max_difference=max(diffs),scope='finite signed scalar proxy, not an assertion that gauge-equivalent SM Higgs signs are distinct worlds')

def selection_route_checks(C):
    # Scalar-only UV completion diagnostic in d=4. h has 4 real components,
    # three complex chi's have 6. Gauge/Yukawa interactions are deliberately
    # excluded; this is not a beta function for the complete CE/SM theory.
    nh,nc=4,6;N=nh+nc
    def tensor(lh,lc,k):
        T=np.zeros((N,N,N,N))
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    for d in range(N):
                        h=sum(i<nh for i in [a,b,c,d])
                        pairs=int(a==b and c==d)+int(a==c and b==d)+int(a==d and b==c)
                        if h==4:T[a,b,c,d]=2*lh*pairs
                        elif h==0:T[a,b,c,d]=2*lc*pairs
                        elif h==2:
                            hs=[i for i in [a,b,c,d] if i<nh];cs=[i for i in [a,b,c,d] if i>=nh]
                            T[a,b,c,d]=k*int(hs[0]==hs[1] and cs[0]==cs[1])
        return T
    rows=[]
    for lh,lc,k in [(.1,0.,.2),(.3,.4,.5),(-.2,.1,-.3)]:
        T=tensor(lh,lc,k)
        # 16 pi^2 beta_abcd = T_abef T_efcd + two permutations.
        beta=lambda a,b,c,d: (np.sum(T[a,b]*T[:,:,c,d])+np.sum(T[a,c]*T[:,:,b,d])+np.sum(T[a,d]*T[:,:,b,c]))/(16*np.pi**2)
        via=np.array([beta(0,0,0,0)/6,beta(nh,nh,nh,nh)/6,beta(0,0,nh,nh)])
        formula=np.array([2*(nh+8)*lh*lh+nc*k*k/2,2*(nc+8)*lc*lc+nh*k*k/2,k*(2*(nh+2)*lh+2*(nc+2)*lc+4*k)])/(16*np.pi**2)
        rows.append(dict(lamH=lh,lamC=lc,kappa=k,beta_tensor=via.tolist(),beta_formula=formula.tolist(),error=float(np.max(abs(via-formula)))))
    C.near('one-loop scalar beta functions by explicit tensor contractions',max(r['error'] for r in rows),2e-17)
    generated=nh/(2*16*np.pi**2)
    C.require('nonzero portal radiatively generates residual quartic',generated>0,at_lambdaC_zero_kappa_one=generated)
    # Sum of squares is an analytic no-nonzero-fixed-point statement at this order;
    # test its coefficients rather than pretending a grid search proves it.
    C.require('four-dimensional scalar fixed-point sum-of-squares gate',2*(nh+8)>0 and nc/2>0 and 2*(nc+8)>0)
    # A positive Schur relaxation term is subtracted. If its off-diagonal
    # coupling alone is proportional to a Higgs amplitude, the induced portal
    # has the opposite sign to the positive portal used in the inherited model.
    rng=np.random.default_rng(27);L=rng.normal(size=(3,2));Q=rng.normal(size=(2,2));D=Q.T@Q+np.eye(2)
    induced=-L@np.linalg.solve(D,L.T);ev=np.linalg.eigvalsh(induced)
    C.require('naive Higgs-linear Schur elimination yields nonpositive portal',float(ev[-1])<1e-13 and float(ev[0])<0,eigenvalues=ev.tolist())
    C.near('Schur portal is an exact polynomial coefficient',np.max(abs((-((.7*L)@np.linalg.solve(D,.7*L.T)))/(.7*.7)-induced)),2e-15)
    return dict(scalar_beta_checks=rows,generated_residual_quartic_beta=generated,scope='one-loop, d=4, scalar-only, no gauge or Yukawa; nonperturbative/full-CE selection remains untested',schur_portal_eigenvalues=ev.tolist(),schur_assumptions='A and C independent of Higgs; B proportional to Higgs; positive C')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',default=str(ROOT/'results.json'));args=ap.parse_args()
    C=Checks();start=time.monotonic()
    out=dict(experiment='CE-RI1',date='2026-09-20',remote_base='8721ab787a3848bddc6cfaf8f8c3622479d62639',inputs=asdict(Params()),interpretation=dict(observational_data=False,observational_fitting=False,synthetic_records=True,fundamental_constants_selected=False,full_initial_state_selected=False,field_bath_fixed_source_Gaussian=True,joint_proxy_separate=True,main_modified=False))
    try:
        out['field']=field_checks(C);out['renormalization']=renormalization_checks(C)
        out['joint']=joint_identification_checks(C);out['state']=parity_checks(C);out['selection_routes']=selection_route_checks(C);out['passed']=True
    except Exception as e:out['passed']=False;out['error']=repr(e);raise
    finally:
        out['checks']=C.rows;out['check_count']=len(C.rows);out['elapsed_seconds']=time.monotonic()-start
        Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{len(C.rows)} checks passed; {args.output}',flush=True)
if __name__=='__main__':main()
