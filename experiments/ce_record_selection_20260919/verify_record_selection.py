#!/usr/bin/env python3
"""CE-RS1: finite-regulator phase selection and record-conditioned sources.
No observational fitting. All test inputs are explicit diagnostic inputs.
Run: OPENBLAS_NUM_THREADS=1 python verify_record_selection.py --output results.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag, eigh, expm

SEED = 20260919
EPS = .15


def Y(theta: float, epsilon: float = EPS, order: int = 0) -> NDArray:
    S = np.roll(np.eye(3), 1, axis=1)
    return epsilon * ((1j/3)**order*np.exp(1j*theta/3)*S
                      +(-1j/3)**order*np.exp(-1j*theta/3)*S.T)


def laplacian(w: NDArray) -> NDArray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not np.allclose(w, w.T):
        raise ValueError('weights must be a symmetric square matrix')
    if np.min(w) < 0 or not np.allclose(np.diag(w), 0):
        raise ValueError('off-diagonal weights must be nonnegative, diagonal zero')
    return np.diag(w.sum(axis=1))-w


def chain(n: int, weight: float = .2) -> NDArray:
    w = np.zeros((n,n))
    for i in range(n-1):
        w[i,i+1]=w[i+1,i]=weight
    return laplacian(w)


def kernel(L: NDArray, s: NDArray, theta: NDArray, epsilon: float = EPS) -> NDArray:
    s=np.atleast_1d(s).astype(float); theta=np.atleast_1d(theta).astype(float)
    if len(s)!=len(theta) or L.shape!=(len(s),len(s)) or epsilon<=0 or np.min(s)<=2*epsilon:
        raise ValueError('need positive epsilon, s>2 epsilon, and matching arrays')
    return np.kron(L,np.eye(3))+block_diag(*[a*np.eye(3)+Y(t,epsilon) for a,t in zip(s,theta)])


def ld(K: NDArray) -> float:
    vals=np.linalg.eigvalsh(K)
    if vals[0]<=0:
        raise ValueError('positive definite kernel required')
    return float(np.log(vals).sum())


def delta_gamma(L: NDArray,s: NDArray,theta: NDArray,epsilon: float=EPS) -> float:
    return ld(kernel(L,s,theta,epsilon))-ld(kernel(L,s,np.full(len(s),np.pi),epsilon))


def embed(n: int, i: int, M: NDArray) -> NDArray:
    out=np.zeros((3*n,3*n),complex); out[3*i:3*i+3,3*i:3*i+3]=M
    return out


def phase_hessian(L: NDArray,s: NDArray,theta: NDArray,epsilon:float=EPS) -> NDArray:
    n=len(s); G=np.linalg.inv(kernel(L,s,theta,epsilon))
    A=[embed(n,i,Y(theta[i],epsilon,1)) for i in range(n)]
    B=[embed(n,i,Y(theta[i],epsilon,2)) for i in range(n)]
    h=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            h[i,j]=((np.trace(G@B[i]) if i==j else 0)-np.trace(G@A[i]@G@A[j])).real
    return h


def loop_difference(K: NDArray, terms: int=160) -> dict:
    d=np.diag(K).real
    T=np.diag(d)-K
    R=T/np.sqrt(d[:,None]*d[None,:]); R0=np.abs(R)
    ev=np.linalg.eigvalsh(R); ev0=np.linalg.eigvalsh(R0)
    radius=float(np.max(np.abs(ev0)))
    if radius>=1:
        raise ValueError('loop series condition failed')
    coeff=np.array([float((np.sum(ev0**m)-np.sum(ev**m))/m) for m in range(1,terms+1)])
    tail=2*len(d)*radius**(terms+1)/((terms+1)*(1-radius))
    K0=np.diag(d)-np.abs(T)
    return dict(value=float(coeff.sum()),exact=ld(K)-ld(K0),tail_bound=float(tail),
                radius=radius,min_coefficient=float(coeff.min()))


# Conditional portal histories: three complex oscillators, six real quadratures.
# u_a,u_b are prescribed slowly-varying/frozen Higgs-background alternatives,
# not predicted electroweak values. No extra measurement coupling is inserted.

def oscillator_operators(n: int,w: float) -> tuple[NDArray,NDArray]:
    # Form the polynomials before restricting, preventing a boundary squaring error.
    a=np.diag(np.sqrt(np.arange(1,n+4)),1)
    q=(a+a.T)/np.sqrt(2*w); p=-1j*np.sqrt(w/2)*(a-a.T)
    return (q@q)[:n,:n],(p@p)[:n,:n]


def portal(t: float, n: int=32, kappa:float=1., ua:float=.2,ub:float=.8,
           epsilon:float=EPS,s0:float=.5,uref:float=.5) -> dict:
    x0=np.array([s0-2*epsilon,s0+epsilon,s0+epsilon]); xr=x0+kappa*uref
    if min(xr.min(),(x0+kappa*ua).min(),(x0+kappa*ub).min())<=0:
        raise ValueError('positive diagnostic frequencies required')
    nu=1.+0j; nu_abs_analytic=1.; obs=np.zeros(2); obs_an=np.zeros(2)
    norm_error=0.; energy_change=0.
    for x,w in zip(x0,np.sqrt(xr)):
        q2,p2=oscillator_operators(n,float(w)); sts=[]; widths=[]
        for j,u in enumerate([ua,ub]):
            omega=np.sqrt(x+kappa*u)
            h=(p2+(x+kappa*u)*q2)/2
            e,v=eigh(h); c=v[0,:].conj()
            st=v@(c*np.exp(-1j*e*t)); sts.append(st)
            obs[j]+=float(np.vdot(st,q2@st).real)  # both real modes, factor 1/2 each
            a=(w*np.cos(omega*t)+1j*omega*np.sin(omega*t))/(np.cos(omega*t)+1j*w/omega*np.sin(omega*t))
            widths.append(a); obs_an[j]+=1/(2*a.real)
            norm_error=max(norm_error,float(abs(np.vdot(st,st)-1)))
            energy_change=max(energy_change,float(abs(np.vdot(st,h@st).real-h[0,0].real)))
        single=np.vdot(sts[1],sts[0]); nu*=single**2
        a,b=widths
        # Squared overlap for one real oscillator = overlap magnitude of a complex pair.
        nu_abs_analytic*=2*np.sqrt(a.real*b.real)/abs(a+np.conj(b))
    nu_abs=float(abs(nu))
    return dict(t=float(t),overlap_real=float(nu.real),overlap_imag=float(nu.imag),
                visibility=nu_abs,analytic_visibility=float(nu_abs_analytic),
                distinguishability_equal_prior=float(np.sqrt(max(0,1-nu_abs**2))),
                conditional_O=obs.tolist(),analytic_O=obs_an.tolist(),
                norm_error=norm_error,conditional_energy_error=energy_change,
                u_alternatives=[ua,ub],kappa=kappa,n=n)


def pure(v: NDArray) -> NDArray:
    return np.outer(v,v.conj())


def ptrace_last(rho: NDArray, dA: int,dR: int=2) -> NDArray:
    return np.trace(rho.reshape(dA,dR,dA,dR),axis1=1,axis2=3)


def recorded_state(p:float, alternate:int=1) -> tuple[NDArray,NDArray]:
    # Algebraic record test, not a physical truncation of the oscillator spectrum.
    z=np.array([1.,0.]); o=np.array([0.,1.]); plus=(z+o)/np.sqrt(2)
    ca=plus; cb=o if alternate else z
    psi=np.sqrt(p)*np.kron(np.kron(z,ca),z)+np.sqrt(1-p)*np.kron(np.kron(o,cb),o)
    phi=np.diag([1.,2.]); O=np.diag([1.,3.]); force=-np.kron(np.kron(phi,O),np.eye(2))
    return pure(psi),force


def record_source(rho:NDArray,force:NDArray,r:int) -> tuple[float,float]:
    P=np.kron(np.eye(4),np.diag([1.,0.]) if r==0 else np.diag([0.,1.]))
    b=P@rho@P; p=float(np.trace(b).real)
    if p<=0: raise ValueError('nonzero record probability required')
    return p,float(np.trace(b@force).real/p)


class Checks(unittest.TestCase):
    def test_01_spectrum(self):
        for t in [-2.,0.,1.2,np.pi,4.]:
            exact=1+2*EPS*np.cos((t+2*np.pi*np.arange(3))/3)
            np.testing.assert_allclose(np.linalg.eigvalsh(np.eye(3)+Y(t)),np.sort(exact),atol=1e-14)

    def test_02_moments(self):
        for t in np.linspace(-5,5,15):
            a=Y(t)
            self.assertLess(abs(np.trace(a)),1e-14)
            self.assertAlmostEqual(np.trace(a@a).real,6*EPS**2,places=14)

    def test_03_positive_nonuniform_kernel(self):
        s=np.array([.5,.7,.6,.9]); K=kernel(chain(4),s,np.array([0.,1.,2.,3.]))
        self.assertGreaterEqual(np.linalg.eigvalsh(K).min(),s.min()-2*EPS-1e-13)

    def test_04_inherited_constant_phase_product(self):
        L=chain(5); s=np.array([.6,.8,.5,.9,.7]); t=1.2
        h=np.linalg.eigvalsh(L+np.diag(s))
        expected=np.sum(np.log((h**3-3*h*EPS**2+2*EPS**3*np.cos(t))))
        self.assertAlmostEqual(ld(kernel(L,s,np.full(5,t))),expected,places=12)

    def test_05_nonuniform_lattice_minimum(self):
        rng=np.random.default_rng(SEED)
        for n in [1,2,4,8]:
            for _ in range(32):
                w=rng.uniform(0,.15,(n,n)); w=(w+w.T)/2; np.fill_diagonal(w,0)
                s=.4+rng.uniform(0,1,n); t=rng.uniform(-4,4,n)
                self.assertGreaterEqual(delta_gamma(laplacian(w),s,t),-1e-12)

    def test_06_closed_walk_domination(self):
        a=loop_difference(kernel(chain(5),np.array([.6,.7,.8,.9,1.]),np.array([.1,1.2,-1.,3.,.7])))
        self.assertGreaterEqual(a['min_coefficient'],-1e-14)

    def test_07_series_tail_bound(self):
        K=kernel(chain(5),np.array([.6,.7,.8,.9,1.]),np.array([.1,1.2,-1.,3.,.7]))
        a=loop_difference(K,40)
        self.assertLessEqual(abs(a['value']-a['exact']),a['tail_bound']+1e-12)

    def test_08_positive_phase_hessian(self):
        h=phase_hessian(chain(6),np.linspace(.5,1,6),np.full(6,np.pi))
        self.assertGreater(np.linalg.eigvalsh(h).min(),0)

    def test_09_phase_gradient(self):
        L=chain(4); s=np.array([.5,.7,.9,.6]); t=np.array([.2,1.2,-.4,2.])
        G=np.linalg.inv(kernel(L,s,t)); dt=1e-5
        for i in range(4):
            v=np.zeros(4);v[i]=dt
            direct=(ld(kernel(L,s,t+v))-ld(kernel(L,s,t-v)))/(2*dt)
            exact=np.trace(G@embed(4,i,Y(t[i],order=1))).real
            self.assertAlmostEqual(direct,exact,places=8)

    def test_10_local_higgs_source(self):
        L=chain(4);s=np.array([.5,.7,.9,.6]);t=np.array([.2,1.2,-.4,2.]);ds=1e-5
        G=np.linalg.inv(kernel(L,s,t));G0=np.linalg.inv(kernel(L,s,np.full(4,np.pi)))
        for i in range(4):
            v=np.zeros(4);v[i]=ds
            direct=(delta_gamma(L,s+v,t)-delta_gamma(L,s-v,t))/(2*ds)
            exact=np.trace((G-G0)@embed(4,i,np.eye(3))).real
            self.assertAlmostEqual(direct,exact,places=8)

    def test_11_mixed_derivative_reciprocity(self):
        L=chain(3);s=np.array([.5,.8,.7]);t=np.array([.2,1.2,2.]);G=np.linalg.inv(kernel(L,s,t))
        A=embed(3,0,np.eye(3));B=embed(3,1,Y(t[1],order=1))
        self.assertLess(abs(np.trace(G@A@G@B)-np.trace(G@B@G@A)),1e-13)

    def test_12_gauge_invariance(self):
        K=kernel(chain(4),np.full(4,.7),np.full(4,np.pi))
        D=np.diag(np.tile(np.exp(2j*np.pi*np.arange(3)/3),4))
        transformed=D.conj().T@K@D
        stoquastic=np.diag(np.diag(K))-np.abs(K-np.diag(np.diag(K)))
        np.testing.assert_allclose(transformed,stoquastic,atol=1e-14)
        self.assertAlmostEqual(ld(K),ld(stoquastic),places=12)

    def test_13_extra_phase_potential_can_overturn(self):
        difference=delta_gamma(chain(1),np.array([.7]),np.array([0.]))
        # An explicitly excluded independent local term -cos(theta).
        self.assertLess(difference-2.,0)

    def test_14_positive_higgs_marginal(self):
        # Finite positive quadrature diagnostic of the pointwise theorem.
        ssets=[np.array([.5,.6]),np.array([.8,1.]),np.array([1.2,.6])]
        weights=np.array([.2,.3,.5]); L=chain(2); t=np.array([0.,1.2])
        z=sum(w*np.exp(-ld(kernel(L,s,t))) for s,w in zip(ssets,weights))
        z0=sum(w*np.exp(-ld(kernel(L,s,np.full(2,np.pi)))) for s,w in zip(ssets,weights))
        self.assertLess(z/z0,1)

    def test_15_finite_phase_measure_has_full_support(self):
        angles=np.linspace(-np.pi,np.pi,33,endpoint=False)
        weights=np.exp([-delta_gamma(chain(1),np.array([.7]),np.array([t])) for t in angles])
        p=weights/weights.sum()
        self.assertTrue(np.all(p>0));self.assertLess(p.max(),.05)

    def test_16_portal_fock_analytic_overlap(self):
        for t in [.1,1,2,4,8,12]:
            d=portal(t,32)
            self.assertLess(abs(d['visibility']-d['analytic_visibility']),1e-11)
            np.testing.assert_allclose(d['conditional_O'],d['analytic_O'],atol=1e-10)

    def test_17_portal_cutoff_convergence(self):
        for t in [1.,2.,4.,12.]:
            a,b=portal(t,24),portal(t,40)
            self.assertLess(abs(a['visibility']-b['visibility']),2e-11)
            np.testing.assert_allclose(a['conditional_O'],b['conditional_O'],atol=2e-10)

    def test_18_short_time_variance_coefficient(self):
        xr=np.array([.7,1.15,1.15]);varO=float(np.sum(1/(4*xr)))
        expected=.5*.6**2*varO
        t=1e-3;d=portal(t)
        self.assertLess(abs((1-d['visibility'])/t**2-expected),2e-7)

    def test_19_portal_cannot_distinguish_sign_of_phi(self):
        # u(phi)=phi^2/2; opposing signs give identical conditional Hamiltonians.
        a=portal(3.,ua=.7**2/2,ub=(-.7)**2/2)
        self.assertAlmostEqual(a['visibility'],1,places=12)

    def test_20_zero_portal_no_record(self):
        self.assertAlmostEqual(portal(3.,kappa=0)['visibility'],1,places=12)

    def test_21_finite_environment_not_perfect_record(self):
        a=portal(2.)
        self.assertGreater(a['visibility'],.8);self.assertLess(a['distinguishability_equal_prior'],.5)

    def test_22_finite_environment_recoherence(self):
        self.assertGreater(portal(4.)['visibility'],portal(2.)['visibility'])

    def test_23_unitarity_and_conditional_energy(self):
        for t in [0.,1.,4.,12.]:
            a=portal(t)
            self.assertLess(a['norm_error'],1e-12)
            self.assertLess(a['conditional_energy_error'],1e-11)

    def test_24_populations_not_selected(self):
        v=portal(2.);nu=v['overlap_real']+1j*v['overlap_imag']
        for p in [.2,.5,.8]:
            r=np.array([[p,np.sqrt(p*(1-p))*nu],[np.sqrt(p*(1-p))*nu.conjugate(),1-p]])
            self.assertGreaterEqual(np.linalg.eigvalsh(r).min(),-1e-13)
            self.assertAlmostEqual(r[0,0].real,p)

    def test_25_weighted_record_sources(self):
        for p in [.2,.5,.8]:
            rho,F=recorded_state(p)
            pieces=[record_source(rho,F,r) for r in [0,1]]
            self.assertAlmostEqual(sum(a*b for a,b in pieces),np.trace(rho@F).real,places=13)

    def test_26_discarded_branch_does_not_add_force(self):
        for p in [.1,.3,.9]:
            for alt in [0,1]:
                rho,F=recorded_state(p,alt)
                self.assertAlmostEqual(record_source(rho,F,0)[1],-2.,places=13)

    def test_27_remote_record_unitary_no_signal(self):
        rho,F=recorded_state(.3)
        u=expm(-.7j*np.array([[.2,1],[1,-.2]]))
        U=np.kron(np.eye(4),u)
        np.testing.assert_allclose(ptrace_last(U@rho@U.conj().T,4),ptrace_last(rho,4),atol=1e-14)

    def test_28_remote_record_channel_no_signal(self):
        rho,_=recorded_state(.3);r=.4
        a=np.array([[1,0],[0,np.sqrt(1-r)]],complex);b=np.array([[0,np.sqrt(r)],[0,0]],complex)
        transformed=sum(np.kron(np.eye(4),v)@rho@np.kron(np.eye(4),v.conj().T) for v in [a,b])
        np.testing.assert_allclose(ptrace_last(transformed,4),ptrace_last(rho,4),atol=1e-14)

    def test_29_conditioning_not_unconditional_force(self):
        rho,F=recorded_state(.3)
        self.assertGreater(abs(record_source(rho,F,0)[1]-np.trace(rho@F).real),1)

    def test_30_residual_exists_inside_selected_branch(self):
        rho,F=recorded_state(.3)
        P=np.kron(np.eye(4),np.diag([1.,0.]));b=P@rho@P/.3
        O=np.kron(np.kron(np.eye(2),np.diag([1.,3.])),np.eye(2))
        mean=np.trace(b@O).real;variance=np.trace(b@O@O).real-mean**2
        self.assertAlmostEqual(variance,1.,places=13)

    def test_31_gauge_invariant_portal_orientation_blind(self):
        rng=np.random.default_rng(SEED);q,_=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
        K=.8*np.eye(3)+Y(1.2);Kt=q@K@q.conj().T
        self.assertAlmostEqual(ld(Kt),ld(K),places=13)
        self.assertLess(abs(np.trace(np.linalg.inv(Kt))-np.trace(np.linalg.inv(K))),1e-13)

    def test_32_absolute_scale_not_selected(self):
        L=chain(3);s=np.array([.5,.7,.8]);theta=np.array([.2,1.2,2.])
        original=delta_gamma(L,s,theta)
        for scale in [.3,2.,7.]:
            res=delta_gamma(L*scale**2,s*scale**2,theta,EPS*scale**2)
            self.assertAlmostEqual(res,original,places=12)

    def test_33_nonzero_portal_does_not_fix_kappa(self):
        for k in [.5,1.,2.]:
            d=portal(1.,kappa=k)
            self.assertLess(d['visibility'],1-1e-4)
            self.assertGreater(d['visibility'],0)

    def test_34_unitary_cannot_erase_initial_distinctions(self):
        h=np.array([[.5,.15],[.15,.9]])
        a=np.diag([.2,.8]);b=np.diag([.8,.2]);U=expm(-2j*h)
        before=np.linalg.svd(a-b,compute_uv=False).sum()/2
        after=np.linalg.svd(U@(a-b)@U.conj().T,compute_uv=False).sum()/2
        self.assertAlmostEqual(before,after,places=14)


    def test_35_free_epsilon_has_no_interior_minimum(self):
        L=chain(4);s=np.array([.5,.7,.8,.6]);h=np.linalg.eigvalsh(L+np.diag(s))
        for epsilon in [.02,.08,.15,.22]:
            exact=float(np.sum(-6*epsilon/((h-2*epsilon)*(h+epsilon))))
            step=1e-6
            numerical=(ld(kernel(L,s,np.full(4,np.pi),epsilon+step))-ld(kernel(L,s,np.full(4,np.pi),epsilon-step)))/(2*step)
            self.assertLess(exact,0)
            self.assertAlmostEqual(exact,numerical,places=7)

    def test_36_free_kappa_prefers_decoupling_without_its_own_cost(self):
        L=chain(2);u_profiles=[np.array([.2,.8]),np.array([.5,.5]),np.array([.1,1.])]
        weights=[.2,.5,.3];kappas=[0.,.5,1.,2.]
        fs=[]
        for k in kappas:
            z=sum(w*np.exp(-ld(kernel(L,.5+k*u,np.full(2,np.pi)))) for w,u in zip(weights,u_profiles))
            fs.append(-np.log(z))
        self.assertTrue(np.all(np.diff(fs)>0))
        for u in u_profiles:
            K=kernel(L,.5+u,np.full(2,np.pi))
            self.assertGreater(np.trace(np.linalg.inv(K)@np.kron(np.diag(u),np.eye(3))).real,0)


    def test_37_cross_record_coupling_is_a_real_interaction(self):
        # Positive control: an explicit inter-sector Hamiltonian invalidates stable-record isolation.
        P=np.diag([1.,0.]);h=np.array([[0.,.15],[.15,1.]])
        self.assertGreater(np.linalg.norm(P@h-h@P),0)
        state=expm(-2j*h)@np.array([1.,0.])
        self.assertGreater(abs(state[1])**2,1e-3)
        # Its virtual correction is fixed by coupling and gap, not an independently added branch probability.
        e=np.linalg.eigvalsh(h)[0]
        self.assertAlmostEqual(e,-.15**2/(1-e),places=14)


def diagnostics() -> dict:
    rng=np.random.default_rng(SEED)
    diffs=[];mineig=[]
    for n in [1,2,4,8]:
        for _ in range(32):
            w=rng.uniform(0,.15,(n,n));w=(w+w.T)/2;np.fill_diagonal(w,0)
            s=.4+rng.uniform(0,1,n);theta=rng.uniform(-4,4,n);L=laplacian(w)
            diffs.append(delta_gamma(L,s,theta));mineig.append(np.linalg.eigvalsh(kernel(L,s,theta)).min())
    s=np.linspace(.5,1,6);L=chain(6);theta=np.full(6,np.pi)
    hess=phase_hessian(L,s,theta)
    loop=loop_difference(kernel(chain(5),np.array([.6,.7,.8,.9,1.]),np.array([.1,1.2,-1.,3.,.7])),80)
    phases=np.linspace(-np.pi,np.pi,33,endpoint=False)
    w=np.exp([-delta_gamma(chain(1),np.array([.7]),np.array([t])) for t in phases]);prob=w/w.sum()
    times=[.1,1.,2.,4.,8.,12.]
    portals=[portal(t,40) for t in times]
    records=[]
    for p in [.2,.5,.8]:
        rho,F=recorded_state(p)
        records.append(dict(p=p,selected_A=record_source(rho,F,0)[1],selected_B=record_source(rho,F,1)[1],
                            unconditioned=float(np.trace(rho@F).real)))
    return {
      'study':'CE-RS1','date':'2026-09-19','seed':SEED,
      'scope':'finite positive bosonic regulator; cyclic phase family, prescribed record/initial diagnostics',
      'observational_fitting':False,'full_joint_rmse':None,'unique_outcome_derived':False,
      'all_parameters_derived':False,'git_modified':False,
      'lattice':{'samples':len(diffs),'delta_gamma_min':float(min(diffs)),'delta_gamma_max':float(max(diffs)),
                 'kernel_eigenvalue_min':float(min(mineig)),'phase_hessian_eigenvalues':np.linalg.eigvalsh(hess).tolist(),
                 'loop_series':loop},
      'finite_phase_distribution':{'points':33,'probability_min':float(prob.min()),'probability_max':float(prob.max()),
                                   'note':'discrete uniform base measure, inverse diagnostic scale 1; not a derived cosmological state'},
      'portal':{'s0':.5,'epsilon':EPS,'kappa':1.,'uref':.5,'u_alternatives':[.2,.8],
                'variance_O_initial':float(np.sum(1/(4*np.array([.7,1.15,1.15])))),
                'small_time_decoherence_coefficient':float(.5*.6**2*np.sum(1/(4*np.array([.7,1.15,1.15])))),
                'results':portals,'max_analytic_visibility_error':max(abs(d['visibility']-d['analytic_visibility']) for d in portals),
                'max_analytic_O_error':max(float(np.max(np.abs(np.array(d['conditional_O'])-d['analytic_O']))) for d in portals)},
      'record_sources':records,
      'naive_parameter_selection':{
        'epsilon_scan':[{'epsilon':e,'derivative_logdet':float(np.sum(-6*e/((np.linalg.eigvalsh(chain(4)+np.diag([.5,.7,.8,.6]))-2*e)*(np.linalg.eigvalsh(chain(4)+np.diag([.5,.7,.8,.6]))+e))))} for e in [.02,.08,.15,.22]],
        'kappa_free_energy_monotone':'positive under the same finite measure and no parameter-dependent cost; minimum at kappa=0',
        'scope':'countertest of bare minimization over inputs, not a claim that measured constants minimize this finite free energy'
      },
      'indeterminate_portal_family':[{'kappa':k,'visibility_t1':portal(1.,kappa=k)['visibility']} for k in [.5,1.,2.]],
      'claims':{
        'nonuniform_phase_minimum':'proved within finite diagonal-dominant scalar graph and fixed edge magnitudes',
        'portal_generates_record_information':'conditional histories entangle with oscillator environment; partial and recurrent',
        'conditioned_source':'residual degrees within the selected record, not the probability-weighted source of a discarded record',
        'higgs_equals_selected_world':'not a consequence of this model',
        'clarus_equals_all_unselected_worlds':'not a consequence of this model'
      }
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():return 1
    out=diagnostics();out['verification']={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Results: {args.output}')
    return 0

if __name__=='__main__':
    sys.exit(main())
