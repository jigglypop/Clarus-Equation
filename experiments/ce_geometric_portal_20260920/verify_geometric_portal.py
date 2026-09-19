#!/usr/bin/env python3
"""CE-GP1: accuracy gate for a curvature-generated portal.

This is a NEW conditional parent-model diagnostic, not a derivation of the
parent geometry from CE. 36 inherited checks are rerun separately. Here we
verify canonical normalization, the danger of a non-invariant truncation,
an explicit reflection projection, and its failure to select nonzero CE Y.
No fitting, experimental inputs, remote writes, or claim of full KK matching.
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import expm
import sympy as sp

class Checks:
    def __init__(self): self.rows=[]
    def require(self,name,passed,**detail):
        row={"name":name,"passed":bool(passed),**detail};self.rows.append(row)
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
        if not passed: raise AssertionError(row)
    def near(self,name,error,tol):
        self.require(name,np.isfinite(error) and error<=tol,error=float(error),tolerance=float(tol))

def norm2(a): return float(np.vdot(a,a).real)
def comm(a,b): return a@b-b@a

def blocks(h,c):
    h=np.asarray(h,dtype=complex);c=np.asarray(c,dtype=complex)
    if h.shape!=(2,) or c.shape!=(3,): raise ValueError('H has 2 and chi 3 complex components')
    A=np.zeros((6,6),complex);B=A.copy()
    A[:2,2]=h/np.sqrt(2);A[2,:2]=h.conj()/np.sqrt(2)
    B[3:,2]=c/np.sqrt(2);B[2,3:]=c.conj()/np.sqrt(2)
    return A,B

def V(A,B,g):
    if g<0: raise ValueError('g is a nonnegative gauge-coupling magnitude')
    return g*g*norm2(comm(A,B))

def sun(n):
    out=[]
    for i in range(n):
        for j in range(i+1,n):
            t=np.zeros((n,n),complex);t[i,j]=t[j,i]=.5;out.append(t)
            t=np.zeros((n,n),complex);t[i,j]=-.5j;t[j,i]=.5j;out.append(t)
    for k in range(1,n):
        d=np.zeros(n);d[:k]=1.;d[k]=-k
        out.append(np.diag(d)/np.sqrt(2*k*(k+1)))
    return out

def projector(t,p,q,s5,s6):
    # A_mu (+,+), A5 (-,+), A6 (+,-) including vector-index parity.
    return (t+s5*p@t@p+s6*q@t@q+s5*s6*p@q@t@q@p)/4

def near_zero_eigenvalue(d,u):
    if d==0 or u<0:raise ValueError('d nonzero and u nonnegative required')
    return -u/(d+np.sign(d)*np.sqrt(d*d+2*u))

def sym_checks(C):
    r=sp.symbols('h1r h1i h2r h2i c1r c1i c2r c2i c3r c3i',real=True)
    h=sp.Matrix([r[0]+sp.I*r[1],r[2]+sp.I*r[3]])
    c=sp.Matrix([r[4]+sp.I*r[5],r[6]+sp.I*r[7],r[8]+sp.I*r[9]])
    A=sp.zeros(6);B=sp.zeros(6)
    for i in range(2): A[i,2]=h[i]/sp.sqrt(2);A[2,i]=sp.conjugate(h[i])/sp.sqrt(2)
    for j in range(3): B[3+j,2]=c[j]/sp.sqrt(2);B[2,3+j]=sp.conjugate(c[j])/sp.sqrt(2)
    u=(h.conjugate().T*h)[0];v=(c.conjugate().T*c)[0]
    R=A*B-B*A
    C.require('symbolic complex canonical normalization',sp.simplify(sp.trace(A*A)-u)==0 and sp.simplify(sp.trace(B*B)-v)==0)
    C.require('symbolic curvature portal coefficient',sp.simplify(sp.trace(R.conjugate().T*R)-u*v/2)==0)
    C.require('no hidden odd-field or orientation term',sp.Poly(sp.expand(sp.trace(R.conjugate().T*R)-u*v/2),*r).is_zero)
    g,a,b,uu,vv=sp.symbols('g a b u v',positive=True)
    C.require('raw-coordinate rescaling does not change canonical kappa',sp.simplify(2*g*g*a*a*b*b*uu*vv/((2*a*a*uu)*(2*b*b*vv))-g*g/2)==0)

def run():
    C=Checks();sym_checks(C);rng=np.random.default_rng(20260920)
    trials=[]
    for _ in range(64):
        h=rng.normal(size=2)+1j*rng.normal(size=2);c=rng.normal(size=3)+1j*rng.normal(size=3)
        g=float(rng.uniform(.15,1.1));A,B=blocks(h,c)
        actual=V(A,B,g);target=g*g*norm2(h)*norm2(c)/2
        trials.append(abs(actual-target)/(1+abs(target)))
    C.near('64 full-matrix portal checks',max(trials),3e-15)
    h=np.array([.2+.3j,-.4+.1j]);c=np.array([.3+.5j,-.2j,.7-.1j]);g=.6
    A,B=blocks(h,c)
    C.near('scalar kinetic normalization checked directly',max(abs(np.trace(A@A).real-norm2(h)),abs(np.trace(B@B).real-norm2(c))),1e-15)
    # A possible hypercharge generator; NOT an SM embedding claim.
    Yq=np.diag([1/3,1/3,-1/6,-1/6,-1/6,-1/6])
    C.near('H hypercharge plus one half and chi neutrality',max(np.max(abs(comm(Yq,A)[:2,2]-.5*A[:2,2])),np.max(abs(comm(Yq,B)))),1e-15)
    C.near('hypercharge normalization retained rather than silently reset',abs(np.trace(Yq@Yq).real-1/3),1e-15)
    mats=sun(6)
    C.near('all 35 parent generators canonically normalized',max(abs(np.trace(a@b)-(.5 if i==j else 0)) for i,a in enumerate(mats) for j,b in enumerate(mats)),5e-16)
    UH=expm(.3j*np.array([[1,.2j],[-.2j,-1]],complex))
    z=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));z=(z+z.conj().T)/2;z-=np.trace(z)/3*np.eye(3);UC=expm(.2j*z)
    At,Bt=blocks(UH@h,UC@c)
    C.near('SU2 by SU3 block covariance of potential',abs(V(At,Bt,g)-V(A,B,g)),1e-15)
    # Full parent versus simply deleting the heavy B_{H,C} channels.
    full_rows=[];eig_errors=[];projected_errors=[];slope_errors=[];mixing=[]
    zvals=np.array([.8,1.2,1.2]);u_probe=.2;step=1e-4
    for d in [-3.,3.]:
        Hk=np.array([[d,0,np.sqrt(u_probe/2)],[0,d,0],[np.sqrt(u_probe/2),0,0]],float)
        lam=np.linalg.eigvalsh(Hk)
        for zz in zvals:
            K=g*g*(Hk-zz*np.eye(3))@(Hk-zz*np.eye(3))
            eig_errors.append(np.max(abs(np.sort(np.linalg.eigvalsh(K))-np.sort(g*g*(lam-zz)**2))))
            projected_errors.append(abs(K[2,2]-g*g*(zz*zz+u_probe/2)))
            mixing.append(abs(K[0,2]))
            def branch(uu): return g*g*(zz-near_zero_eigenvalue(d,uu))**2
            # A five-point one-sided derivative (physical u>=0) independent of the claimed coefficient.
            fs=[branch(k*step) for k in range(5)]
            deriv=(-25*fs[0]+48*fs[1]-36*fs[2]+16*fs[3]-3*fs[4])/(12*step)
            target=g*g*zz/d;slope_errors.append(abs(deriv-target))
            full_rows.append(dict(d=d,z=zz,u=u_probe,projected_mass2=float(K[2,2]),full_connected_mass2=branch(u_probe),projected_slope=g*g/2,full_slope=target,fd_slope=deriv))
    C.near('full parent mass eigenvalues from independent diagonalization',max(eig_errors),8e-15)
    C.near('restricted diagonal really has positive universal portal',max(projected_errors),5e-16)
    C.require('discarded parent channels actually mix',min(mixing)>1e-3,min_mixing=float(min(mixing)))
    C.near('full mass slopes include the discarded channels',max(slope_errors),2e-11)
    C.require('full-parent negative-slope counterexample',all(r['full_slope']<0 for r in full_rows if r['d']<0))
    C.require('positive-slope full-parent case is not channel universal',np.ptp([r['full_slope'] for r in full_rows if r['d']>0])>.02)
    # d=2z switches off the offending mixing only for equal z across channels.
    d,zz=2.,1.;uu=.2;Hk=np.array([[d,0,np.sqrt(uu/2)],[0,d,0],[np.sqrt(uu/2),0,0.]])
    C.near('mixing-free special case requires common z',abs((g*g*(Hk-zz*np.eye(3))@(Hk-zz*np.eye(3)))[0,2]),1e-15)
    # Reflection automorphisms of a rectangle compactification.
    p=np.diag([-1,-1,1,1,1,1]);q=np.diag([1,1,1,-1,-1,-1])
    C.near('commuting involutive reflection automorphisms',max(np.max(abs(p@p-np.eye(6))),np.max(abs(q@q-np.eye(6))),np.max(abs(comm(p,q)))),0.)
    counts={}
    for name,s5,s6 in [('vector',1,1),('A5',-1,1),('A6',1,-1),('mixed_curvature',-1,-1)]:
        counts[name]=sum(norm2(projector(t,p,q,s5,s6))>.1 for t in mats)
    C.require('zero-mode field content explicitly counted',counts==dict(vector=13,A5=4,A6=6,mixed_curvature=12),counts=counts)
    C.near('H and chi matrices survive their correct reflections',max(np.max(abs(projector(A,p,q,-1,1)-A)),np.max(abs(projector(B,p,q,1,-1)-B))),0.)
    D=np.diag([-.7,-.7,0.,.8,.9,-.3]);D-=np.trace(D)/6*np.eye(6)
    C.near('reflection projection removes diagonal mass background',max(np.max(abs(projector(D,p,q,-1,1))),np.max(abs(projector(D,p,q,1,-1)))),0.)
    bridge=np.zeros((6,6),complex);bridge[0,3]=bridge[3,0]=1/np.sqrt(2)
    C.near('problematic A6 bridge has no constant zero mode',np.max(abs(projector(bridge,p,q,1,-1))),0.)
    C.near('curvature parity is compatible with the full action',np.max(abs(projector(comm(A,B),p,q,-1,-1)-comm(A,B))),0.)
    # Gauge-invariant constant masses in an unbroken SU3 fundamental are proportional to I.
    b3=[np.eye(3)/np.sqrt(3)]+[np.sqrt(2)*t for t in sun(3)]
    responses=[]
    for e in b3:
        ss=np.concatenate([comm(e,t).ravel() for t in sun(3)])
        responses.append(np.r_[ss.real,ss.imag])
    singular=np.linalg.svd(np.array(responses).T,compute_uv=False)
    C.require('unbroken SU3 commutant is one-dimensional',int(np.sum(singular>1e-10))==8,rank=int(np.sum(singular>1e-10)))
    Yce=np.diag([-.3,.15,.15])
    C.require('nonzero CE splitting breaks the surviving hidden SU3',max(norm2(comm(Yce,t)) for t in sun(3))>0.)
    C.near('singlet mass does respect surviving SU3',max(norm2(comm(.5*np.eye(3),t)) for t in sun(3)),0.)
    C.near('tree radial flat direction retained',max(V(a*A,np.zeros_like(B),g) for a in [0.,.3,1.,4.]),0.)
    # Conditional record-level diagnostic of the projected leading theory: epsilon=0, no added mass.
    kappa=g*g/2;ua,ub=.2,.8
    Ta=2*np.sqrt(kappa*ua);Tb=2*np.sqrt(kappa*ub)
    C.near('projected zero-mode threshold slope equals twice g squared',abs((Tb*Tb-Ta*Ta)/(ub-ua)-2*g*g),1e-15)
    om=1.4;shape=np.sqrt(1-4*kappa*ua/om**2)
    noise=kappa*kappa*3*shape/(8*np.pi);fromg=g**4*3*shape/(32*np.pi)
    C.near('record amplitude shares the same tree gauge coefficient',abs(noise-fromg),1e-18)
    # Any positive CE matrix can be encoded by a square root -- not selected.
    S=np.roll(np.eye(3),1,axis=0);s0=.5;e=.15;th=1.2
    X=s0*np.eye(3)+e*(np.exp(1j*th/3)*S+np.exp(-1j*th/3)*S.T)
    w,U=np.linalg.eigh(X);Z=(U*np.sqrt(w)/g)@U.conj().T
    C.near('square-root background encoding is not a selection',np.max(abs(g*g*Z@Z-X)),2e-15)
    # If independently chosen kinetic/curvature weights are allowed the coefficient is no longer fixed.
    ZH,ZC,ZV=1.2,.8,1.1
    arbitrary=kappa*ZV/(ZH*ZC)
    C.require('four-dimensional gauge invariance alone does not fix normalization',abs(arbitrary-kappa)>.02,kappa_canonical=kappa,kappa_with_independent_weights=arbitrary)
    # These are already present scalar-subset loops, NOT the full parent beta function.
    generated=2*kappa*kappa/(16*np.pi**2)
    C.require('scalar-subset quartic regeneration cannot be discarded',generated>0,scalar_only_beta_lambdaC_at_zero=generated)
    return dict(experiment='CE-GP1',research_date='2026-09-20',remote_main_modified=False,
        status='conditional geometric portal; no CE spectral-radius or complete SM matching',
        additional_assumptions=['SU(6) parent algebra','two compact directions and flat constant profiles for the matching normalization','two stated reflection automorphisms','no boundary kinetic terms in the tree matching'],
        checks=C.rows,check_count=len(C.rows),passed=all(x['passed'] for x in C.rows),
        full_parent_diagnostics=full_rows,zero_mode_counts=counts,
        projected_record_diagnostic=dict(g=g,kappa=kappa,epsilon=0.,s0=0.,ua=ua,ub=ub,threshold_a=Ta,threshold_b=Tb,threshold_squared_slope=2*g*g,omega=om,noise=noise,leading_free_mediator_approximation=True),
        normalization=dict(hypercharge_trace2=float(np.trace(Yq@Yq)),gY_over_g=np.sqrt(1.5),tree_sin2_if_only_this_U1_is_used=.6,extra_U1_retained=True),
        environments=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sympy=sp.__version__))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results.json'));a=p.parse_args()
    r=run();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(f"{r['check_count']} checks passed; {a.output}")
if __name__=='__main__':main()
