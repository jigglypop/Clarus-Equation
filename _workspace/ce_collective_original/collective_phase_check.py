"""Conditional CE follow-up: exact collective symmetry and numerical FLRW check.
No network access, observational fits, branch edits or repository writes.
python collective_phase_check.py; requires numpy scipy sympy.
"""
from __future__ import annotations
import json, math, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from numpy.polynomial.legendre import leggauss
import sympy as sp

def difference(N: int, q: int) -> np.ndarray:
    D=np.zeros((N,N+1))
    for j in range(N):D[j,j]=1.;D[j,j+1]=-q
    return D

class Loop:
    """six-channel complex-scalar determinant; all masses in a supplied unit."""
    def __init__(self, order=160):
        self.a=1.;self.b=1.5;self.kap=.6;self.p=.5;self.r=.4
        x,w=leggauss(order); t=(x+1)/2
        self.s=t/(1-t);self.weight=w/2*t/(1-t)**3/(16*np.pi**2)
        A=(self.s+self.a)*(self.s+self.b)-self.kap**2*self.p
        h=self.kap**2*self.p*self.r
        self.z=2*h**3/((A-2*h)*(A+h)**2)
        self.curv=float(self.weight@self.z)
    def values(self,phi):
        om=1-np.cos(phi); sn=np.sin(phi);co=np.cos(phi)
        den=1+self.z*om
        v=float(self.weight@np.log1p(self.z*om))/self.curv
        d=float(self.weight@(self.z*sn/den))/self.curv
        dd=float(self.weight@(self.z*co/den-self.z**2*sn**2/den**2))/self.curv
        return v,d,dd
    def matrix(self,phi):
        K=self.kap**2*self.p
        t0=math.sqrt((K+math.sqrt(K*K-4*(K*self.r)**2))/2);t1=K*self.r/t0
        S=np.array([[0,1,0],[0,0,1],[np.exp(1j*phi),0,0]])
        B=t0*np.eye(3)+t1*S
        return np.block([[self.a*np.eye(3),B],[B.conj().T,self.b*np.eye(3)]])
    def spectral_v(self,phi):
        def f(t):
            s=np.linalg.eigvalsh(self.matrix(t))
            return np.sum(s*s*(np.log(s)-1.5))/(32*np.pi**2)
        return f(phi)-f(0.)
    def deltaZ(self,phi):
        mat=self.matrix(phi);e,U=np.linalg.eigh(mat)
        K=self.kap**2*self.p;t0=math.sqrt((K+math.sqrt(K*K-4*(K*self.r)**2))/2);t1=K*self.r/t0
        dm=np.zeros((6,6),complex);dm[2,3]=1j*t1*np.exp(1j*phi);dm[3,2]=dm[2,3].conjugate()
        rot=U.conj().T@dm@U
        x,w=leggauss(80);x=(x+1)/2;w=w/2
        out=0.
        for i in range(6):
            for j in range(6):out+=abs(rot[i,j])**2*np.sum(w*x*(1-x)/(x*e[i]+(1-x)*e[j]))
        return float(out/(16*np.pi**2))

def algebra():
    ans=[]
    rng=np.random.default_rng(20260908)
    for N in [1,2,4,6]:
        for q in [1,2,3]:
            D=difference(N,q);l=np.array([q**(N-j) for j in range(N+1)],float);norm=np.linalg.norm(l);u=l/norm
            assert np.linalg.norm(D@l)<1e-10
            # Nonzero eigenvalues of D^T D equal DD^T; numerically stable null handling.
            expected=np.array([1+q*q-2*q*np.cos(k*np.pi/(N+1)) for k in range(1,N+1)])
            error=np.max(abs(np.linalg.eigvalsh(D@D.T)-expected))
            R=rng.normal(size=(N,N));G=R.T@R+np.eye(N)
            weighted=D.T@G@D
            invariant=np.linalg.norm(weighted@u)
            assert error<1e-10 and invariant<1e-9
            ans.append(dict(N=N,q=q,F_over_f=norm,nonzero_spectrum_error=float(error),weighted_null_error=float(invariant)))
    # A gauged simple ring has no lower-degree phase-dependent scalar monomial.
    rings=[]
    for L in [3,4,5,6,8]:
        incidence=sp.zeros(L,L)
        for j in range(L):incidence[j,j]=1;incidence[(j+1)%L,j]=-1
        kernel=incidence.nullspace();assert len(kernel)==1 and kernel[0]==sp.ones(L,1)
        f=np.exp(rng.normal(size=L));speed=1/np.sum(1/f**2)
        rates=speed/f**2
        assert abs(rates.sum()-1)<1e-12
        assert abs(np.sum(f*f*rates*rates)-speed)<1e-12
        rings.append(dict(L=L,first_phase_operator_dimension=L,renormalizable_phase_operator_allowed=L<=4,
                          equal_link_F_over_f=1/math.sqrt(L)))
    # Shift-symmetric nonlinear fermion mass: constant magnitude, not the linear truncation.
    th,m=sp.symbols('th m', real=True)
    modsq=sp.trigsimp(m*m*(sp.cos(th)**2+sp.sin(th)**2))
    assert sp.diff(modsq,th,2)==0
    trunc=m*m*(1+th*th)
    return dict(clockwork=ans,ring= rings,phase_fermion_mass_second_derivative=str(sp.diff(modsq,th,2)),
                linear_only_mass_second_derivative=str(sp.diff(trunc,th,2)),
                primitive_clockwork_operator_dimension_for_q3=4,
                same_term_if_each_phase_is_six_link_holonomy_dimension=24)

def evolve(loop: Loop, N=4, q=3, f=.05, lock=8., full=True, rtol=1e-9):
    D=difference(N,q);ell=np.array([q**(N-j) for j in range(N+1)],float)
    norm2=float(ell@ell);Feff=f*math.sqrt(norm2);phi0=np.pi-.6
    V0=loop.values(phi0)[0];h0=math.sqrt(V0/3);tend=3/h0
    tgrid=np.linspace(0,tend,1201)
    if full:
        n=N+1;theta0=ell*phi0
        y0=np.r_[theta0,np.zeros(n),0.,h0,0.,0.]
        def rhs(t,y):
            th=y[:n];vel=y[n:2*n];H=y[2*n+1];x=D@th
            vl,dv,_=loop.values(th[-1]);VH=2*lock*np.sum(np.sin(x/2)**2);v=VH+vl
            grad=lock*D.T@np.sin(x);grad[-1]+=dv
            kin=.5*f*f*(vel@vel)
            return np.r_[vel,-3*H*vel-grad/(f*f),H,-kin,kin-v,kin+v]
        s=solve_ivp(rhs,(0,tend),y0,t_eval=tgrid,method='DOP853',rtol=rtol,atol=rtol*.01)
        assert s.success,s.message
        theta=s.y[:n];vel=s.y[n:2*n];ln_a=s.y[2*n];H=s.y[2*n+1];x=D@theta
        values=np.array([loop.values(z)[0] for z in theta[-1]])+2*lock*np.sum(np.sin(x/2)**2,axis=0)
        kinetic=.5*f*f*np.sum(vel*vel,axis=0)
        phi=(ell@theta)/norm2
        forcing=np.max(abs(x));scale_mass=np.sqrt(np.linalg.eigvalsh(D@D.T)[0]*lock)/f
        extra=dict(max_link_phase_displacement=float(forcing),min_gear_frequency_over_initial_H=scale_mass/h0)
        Pint=s.y[-2,-1];Rint=s.y[-1,-1]
    else:
        y0=np.array([phi0,0,0,h0,0,0])
        def rhs(t,y):
            ph,v,ln,H,_,_=y;V,dV,_=loop.values(ph);kin=.5*Feff**2*v*v
            return [v,-3*H*v-dV/Feff**2,H,-kin,kin-V,kin+V]
        s=solve_ivp(rhs,(0,tend),y0,t_eval=tgrid,method='DOP853',rtol=rtol,atol=rtol*.01)
        assert s.success
        phi=s.y[0];values=np.array([loop.values(z)[0] for z in phi]);kinetic=.5*Feff**2*s.y[1]**2
        H=s.y[3];ln_a=s.y[2];Pint=s.y[4,-1];Rint=s.y[5,-1];extra={}
    rho=kinetic+values;p=kinetic-values
    normerr=np.max(abs(3*H*H-rho))/max(rho[0],1e-20)
    out=dict(N=N,q=q,local_f_over_Mpl=f,F_eff_over_Mpl=Feff,lock_over_loop_curvature=lock,
             full=full,t_end_initial_H_units=3.,initial_phase=phi0,final_phase=float(phi[-1]),
             initial_w=float(p[0]/rho[0]),final_w=float(p[-1]/rho[-1]),time_integrated_w=float(Pint/Rint),
             max_w=float(np.max(p/rho)),accelerating_all_sampled=bool(np.all(rho+3*p<0)),
             final_H=float(H[-1]),final_ln_a=float(ln_a[-1]),friedmann_relative_error=float(normerr),nfev=s.nfev,**extra)
    assert normerr<2e-6, out
    return out,dict(t=tgrid,phi=phi,H=H)

def main():
    start=time.time();loop=Loop();alg=algebra()
    # Independent spectral finite difference vs convergent Euclidean integral.
    errors=[]
    for phi in [.3,1.,2.,np.pi]:
        errors.append(abs(loop.values(phi)[0]*loop.curv-loop.spectral_v(phi)))
    assert max(errors)<1e-13
    l2=Loop(320);qerr=max(abs(loop.values(t)[0]-l2.values(t)[0]) for t in [.3,1.,2.,np.pi])
    assert qerr<1e-10
    D=difference(4,3);ell=3.**np.arange(4,-1,-1);norm2=float(ell@ell);f=.05
    mass_checks=[]
    for locking in [2,8,32,128]:
        H=locking*D.T@D;H[-1,-1]+=1
        exact=np.linalg.eigvalsh(H)[0]/f**2;approx=1/(f*f*norm2)
        mass_checks.append(dict(lock=locking,m_light2_exact=float(exact),adiabatic_prediction=approx,
                                relative_error=abs(exact-approx)/exact))
    reduced,ref=evolve(loop,full=False,rtol=2e-11)
    full_results=[]
    for locking in [2,8,32]:
        result,tr=evolve(loop,lock=locking,full=True,rtol=2e-10)
        result['max_relative_H_difference_from_reduced']=float(np.max(abs(tr['H']-ref['H'])/ref['H']))
        result['max_collective_phase_difference_from_reduced']=float(np.max(abs(tr['phi']-ref['phi'])))
        full_results.append(result)
    tighter,tight=evolve(loop,lock=8,full=True,rtol=2e-11)
    regular,reg=evolve(loop,lock=8,full=True,rtol=2e-10)
    conv=max(abs(tight['H']-reg['H']))/max(ref['H']);assert conv<1e-7
    # Without collective enhancement at the same local coefficient and potential.
    plain,pt=evolve(loop,N=0,full=False,rtol=1e-10)
    # Bad explicitly breaking harmonic on fast endpoint: sensitivity must not be hidden.
    eps=1e-3;shift=eps*ell[0]**2;harmonic_bound=1/ell[0]**2
    A=32*D.T@D;A[-1,-1]+=1;A[0,0]+=eps
    harmful=float(np.linalg.eigvalsh(A)[0]/f**2)
    # Full theory may contain higher local-scalar operators: not automatically protected from QG.
    zmax=max(loop.deltaZ(ph) for ph in [0.,1.,np.pi])
    # light pseudoscalar Yukawa integral, same chiral-phase portal not free scalar Yukawa.
    mus=[]
    for r in [0.,.1,1.,10.]:
        integ=quad(lambda x:x**3/(x*x+(1-x)*r*r) if x else 0.,0,1,epsabs=1e-13)[0]
        mus.append(dict(mediator_over_muon_mass=r,delta_a_over_y_squared=-integ/(8*np.pi**2)))
    return dict(date='2026-09-08',candidate='CE-CW-01',base_commit='e41d9d04810378698081c320af5b43d9e966614c',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sympy=sp.__version__),
        algebra=alg,loop=dict(a=loop.a,b=loop.b,kappa=loop.kap,p=loop.p,r=loop.r,curvature_in_heavy_mass4=loop.curv,
                           spectral_integral_absolute_error=max(errors),quadrature_order_difference=qerr,
                           max_deltaZ_in_heavy_mass2=zmax,kinetic_fraction_at_heavy_mass_1e_minus3_Mpl=zmax*1e-6/f**2),
        finite_lock_mass=mass_checks,reduced=reduced,full=full_results,unamplified=plain,
        numerical_convergence_relative_H=float(conv),
        dangerous_harmonic=dict(epsilon_relative_to_loop_curvature=eps,mass_squared_ratio_in_constrained_mode=1+shift,
          minimum_bound_epsilon_much_less_than=harmonic_bound,full_light_mass_squared=harmful),
        muon_phase_portal=mus,
        limitations=['Local f, integer charge q, number N, locking scale and initial state are supplied.',
         'Clockwork is known; not a CE derivation of internal dimensions or 3 forces.',
         'Residual global symmetry is perturbative protection, not proof against quantum-gravity breaking.',
         'Combining gauge-holonomy loop protection with this scalar-chain renormalizable UV model is not completed.',
         'Projected pi and six-channel minimum zero are different candidate graphs.',
         'No empirical H0, dark matter abundance, muon anomaly, or unified gauge coupling fit.',
         'Covariant phase EFT on a supplied metric is not quantized gravity.'],runtime_seconds=time.time()-start)

if __name__=='__main__':
    result=main();p=Path(__file__).with_name('collective_phase_results.json');p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
