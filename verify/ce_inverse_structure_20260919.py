#!/usr/bin/env python3
"""CE-IRS1: reproducible algebra/numerics for a conditional inverse-structure proof.

No experimental data, parameter fitting, network, credentials, or Git writes.
The mathematical class is the positive, equal-multiplicity, three-complex-scalar
Gaussian loop, with common dimensionful epsilon and a pi reference. It does not
include the later nine-scalar/nine-Majorana completion or arbitrary local terms.

Run: python verify_inverse_structure.py --out results.json
Dependencies: numpy, scipy, sympy, mpmath. See environment in the output.
"""
from __future__ import annotations
import argparse
import json
import math
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import expm
import sympy as sy
import mpmath as mp

mp.mp.dps = 80
SEED = 20260919
rng = np.random.default_rng(SEED)
checks: dict[str, dict] = {}


def record(name: str, passed: bool, **details: object) -> None:
    checks[name] = {"passed": bool(passed), **details}
    if not passed:
        raise AssertionError(f"Verification failed: {name}: {details}")


def spectral(s: mp.mpf, eps: mp.mpf, theta: mp.mpf) -> list[mp.mpf]:
    return [s + 2*eps*mp.cos((theta + 2*mp.pi*j)/3) for j in range(3)]


def direct_loop(s: mp.mpf, eps: mp.mpf, theta: mp.mpf) -> mp.mpf:
    f = lambda x: x*x*(mp.log(x) - mp.mpf(3)/2)
    return mp.fsum(f(x) for x in spectral(s, eps, theta))/(32*mp.pi**2) - \
           mp.fsum(f(x) for x in spectral(s, eps, mp.pi))/(32*mp.pi**2)


def inverse(s: mp.mpf, L: mp.mpf, M: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Exact-class inverse. M = partial_s L, NOT the total observed muon g-2."""
    if s <= 0 or L <= 0 or M >= 0:
        raise ValueError("Requires s>0, L>0, M<0; zero relative response is singular.")
    eta = -s*M/(-mp.expm1(-L))
    if eta <= 3:
        raise ValueError("No positive epsilon branch: eta must exceed 3.")
    v = 1-3/eta
    r = 2*v/(mp.sqrt(v*v+8*v)+v)
    cosine = mp.expm1(L)*(1-2*r)*(1+r)**2/(2*r**3)-1
    if cosine < -1-mp.mpf('1e-55') or cosine > 1+mp.mpf('1e-55'):
        raise ValueError("No real phase in the declared model class.")
    return r, cosine, eta


def kernel(a: float, b: float) -> float:
    """Positive divided-difference metric, integral form avoids degeneracy loss."""
    return quad(lambda t: t*(1-t)/((1-t)*a+t*b), 0, 1,
                epsabs=1e-13, epsrel=1e-13)[0]/(16*np.pi**2)


def Ccoef(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("Positive mass squares required.")
    return (b-a)**2 * kernel(a, b)


def matrix_metric(X: np.ndarray, A: np.ndarray) -> float:
    xs, V = np.linalg.eigh(X)
    if min(xs) <= 0:
        raise ValueError("X must be positive definite.")
    Ahat = V.conj().T @ A @ V
    return float(sum(kernel(float(a), float(b))*abs(Ahat[i,j])**2
                     for i,a in enumerate(xs) for j,b in enumerate(xs)))


def unitary(n: int) -> np.ndarray:
    Z = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    Q, _ = np.linalg.qr(Z)
    return Q


def hermitian(n: int) -> np.ndarray:
    Z = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    return (Z+Z.conj().T)/2


def run() -> dict:
    s,e,c,t = sy.symbols('s e c t', real=True)
    P=(s-2*e)*(s+e)**2
    D=s**3-3*s*e**2+2*e**3*c
    A=2*e**3*(1+c)
    L=sy.log(D/P)
    M=sy.diff(L,s)
    sym = {
        'determinant_difference': sy.simplify(D-P-A)==0,
        'log_response_derivative': sy.simplify(M-3*(s*s-e*e)*(1/D-1/P))==0,
        'inverse_eta': sy.simplify(-s*M/(1-P/D)-3*s*(s-e)/((s-2*e)*(s+e)))==0,
    }
    r,k = sy.symbols('r k', real=True)
    eta=3*(1-r)/((1-2*r)*(1+r))
    sym['inverse_quadratic'] = sy.simplify(2*eta*r*r+(eta-3)*r-(eta-3))==0
    sym['eta_monotonicity'] = sy.simplify(sy.diff(eta,r)-6*r*(2-r)/((1-2*r)*(1+r))**2)==0
    x,y,z,h = sy.symbols('x y z h', real=True)
    q = sy.symbols('q', nonzero=True)
    Y=sy.Matrix([[0,x,z/q],[x,0,y],[z*q,y,0]])
    tri=(h*sy.eye(3)+Y).det()
    target=h**3-h*(x*x+y*y+z*z)+x*y*z*(q+1/q)
    sym['general_triangle_determinant']=sy.simplify(tri-target)==0
    X=h*sy.eye(3)+Y
    schur=X[0,0]-(X[:1,1:]*X[1:,1:].inv()*X[1:,:1])[0,0]
    sym['schur_complement']=sy.simplify(schur-target/(h*h-y*y))==0
    sym['schur_determinant_accounting']=sy.simplify(X[1:,1:].det()*schur-tri)==0
    # Shifted characteristic polynomial and real-root discriminant.
    lam=sy.symbols('lam',real=True)
    sym['cubic_discriminant']=sy.factor(sy.discriminant(lam**3-3*e**2*lam-2*e**3*c,lam)-108*e**6*(1-c*c))==0
    record('symbolic_identities',all(sym.values()),count=len(sym),identities=sym)

    # Independent eigenvalue trace calculations, not closed-form determinant input.
    max_r=mp.mpf(0); max_c=mp.mpf(0); max_L=mp.mpf(0)
    n_hp=0
    for sv in [mp.mpf('.1'),mp.mpf(1),mp.mpf(13)]:
      for rv in [mp.mpf('.0001'),mp.mpf('.03'),mp.mpf('.15'),mp.mpf('.35'),mp.mpf('.49')]:
       for th in [mp.mpf('.12'),mp.mpf('1.2'),mp.mpf('2.8'),mp.pi-mp.mpf('.0001')]:
        ev=sv*rv
        xs=spectral(sv,ev,th); xr=spectral(sv,ev,mp.pi)
        LL=mp.fsum(mp.log(xx) for xx in xs)-mp.fsum(mp.log(xx) for xx in xr)
        MM=mp.fsum(1/xx for xx in xs)-mp.fsum(1/xx for xx in xr)
        rr,cc,_=inverse(sv,LL,MM)
        max_r=max(max_r,abs(rr-rv));max_c=max(max_c,abs(cc-mp.cos(th)))
        pp=(sv-2*ev)*(sv+ev)**2
        ll=mp.log1p(2*ev**3*(1+mp.cos(th))/pp)
        max_L=max(max_L,abs(LL-ll));n_hp+=1
    record('high_precision_inverse',max_r<mp.mpf('1e-50') and max_c<mp.mpf('1e-50'),
           cases=n_hp,precision_decimal_digits=mp.mp.dps,
           max_abs_r_error=mp.nstr(max_r,10),max_abs_cosine_error=mp.nstr(max_c,10),
           max_abs_trace_vs_determinant_L=mp.nstr(max_L,10))

    # Random physical channel phases + unrelated unitary bases; source of L is slogdet.
    max_double_r=0.;max_double_c=0.;max_schur=0.;max_rephase=0.
    for _ in range(192):
        sv=float(np.exp(rng.uniform(-1,1)));rv=float(rng.uniform(.07,.43));ev=sv*rv
        phases=rng.uniform(-2,2,3); theta=float(sum(phases))
        if 1+np.cos(theta)<.05:
            phases=np.array([.2,.3,.7]);theta=1.2
        a,b,cc=ev*np.exp(1j*phases)
        YY=np.array([[0,a,np.conj(cc)],[np.conj(a),0,b],[cc,np.conj(b),0]])
        XX=sv*np.eye(3)+YY
        # Reference has the same centered moments, phase pi.
        S=np.roll(np.eye(3),1,axis=1)
        XXr=sv*np.eye(3)+ev*(np.exp(1j*np.pi/3)*S+np.exp(-1j*np.pi/3)*S.T)
        V=unitary(3); XX=V@XX@V.conj().T; XXr=V@XXr@V.conj().T
        LL=float(np.linalg.slogdet(XX)[1]-np.linalg.slogdet(XXr)[1])
        MM=float(np.trace(np.linalg.inv(XX)-np.linalg.inv(XXr)).real)
        rr,cosv,_=inverse(mp.mpf(sv),mp.mpf(LL),mp.mpf(MM))
        max_double_r=max(max_double_r,abs(float(rr)-rv))
        max_double_c=max(max_double_c,abs(float(cosv)-np.cos(theta)))
        Xphys=sv*np.eye(3)+YY
        K_eff=Xphys[0,0]-(Xphys[:1,1:]@np.linalg.solve(Xphys[1:,1:],Xphys[1:,:1]))[0,0]
        expected=(sv**3-3*sv*ev**2+2*ev**3*np.cos(theta))/(sv**2-ev**2)
        max_schur=max(max_schur,abs(K_eff-expected)/sv)
        chi=rng.normal(size=3);Q=np.diag(np.exp(1j*chi));Yg=Q@YY@Q.conj().T
        max_rephase=max(max_rephase,abs(Yg[0,1]*Yg[1,2]*Yg[2,0]-a*b*cc)/ev**3)
    record('random_matrix_inverse',max_double_r<1e-8 and max_double_c<1e-7,
           cases=192,max_abs_r_error=max_double_r,max_abs_cosine_error=max_double_c)
    record('projection_and_rephasing',max_schur<1e-12 and max_rephase<1e-12,
           cases=192,max_scaled_schur_error=max_schur,max_scaled_cycle_product_error=max_rephase)

    # Reconstruct U and its first mass derivative by integrating ONLY L.
    max_U=mp.mpf(0);max_Us=mp.mpf(0)
    for rv in [mp.mpf('.03'),mp.mpf('.15'),mp.mpf('.35'),mp.mpf('.49')]:
      for theta in [mp.mpf('.3'),mp.mpf('1.2')]:
        sv=mp.mpf('1.3');ev=sv*rv
        amp=2*ev**3*(1+mp.cos(theta))
        def ll(u): return mp.log1p(amp/((u-2*ev)*(u+ev)**2))
        Ui=mp.quad(lambda v:v*ll(sv+v),[0,sv,mp.inf])/(16*mp.pi**2)
        Usi=-mp.quad(lambda v:ll(sv+v),[0,sv,mp.inf])/(16*mp.pi**2)
        Ud=direct_loop(sv,ev,theta)
        Usd=mp.diff(lambda ss:direct_loop(ss,ev,theta),sv)
        max_U=max(max_U,abs((Ui-Ud)/Ud));max_Us=max(max_Us,abs((Usi-Usd)/Usd))
    record('inverse_quadrature',max_U<mp.mpf('1e-55') and max_Us<mp.mpf('1e-55'),
           cases=8,max_relative_U_error=mp.nstr(max_U,10),max_relative_Us_error=mp.nstr(max_Us,10))

    # The exact mass derivatives of the original Coleman-Weinberg expression.
    md2=mp.mpf(0);md3=mp.mpf(0)
    for sv in [mp.mpf('.7'),mp.mpf('3')]:
      for rv in [mp.mpf('.04'),mp.mpf('.15'),mp.mpf('.4')]:
       for theta in [mp.mpf(0),mp.mpf('1.2'),mp.mpf('2.7')]:
        ev=sv*rv;xx=spectral(sv,ev,theta);ref=spectral(sv,ev,mp.pi)
        Ld=mp.fsum(mp.log(x) for x in xx)-mp.fsum(mp.log(x) for x in ref)
        Md=mp.fsum(1/x for x in xx)-mp.fsum(1/x for x in ref)
        d2=16*mp.pi**2*mp.diff(lambda z:direct_loop(z,ev,theta),sv,2)
        d3=16*mp.pi**2*mp.diff(lambda z:direct_loop(z,ev,theta),sv,3)
        md2=max(md2,abs(d2-Ld));md3=max(md3,abs(d3-Md))
    record('original_generator_mass_derivatives',max(md2,md3)<mp.mpf('1e-65'),cases=18,
           max_absolute_L_error=mp.nstr(md2,10),max_absolute_M_error=mp.nstr(md3,10),
           fixed_quantity='dimensionful epsilon, not epsilon/s')

    # Shared epsilon and phase across different mass shifts. Positive scalar weights
    # preserve the same minimum; this is not a fermion-supertrace completion.
    worst_derivative=mp.inf
    eps=mp.mpf('.1');ss=[mp.mpf('.7'),mp.mpf('1.3'),mp.mpf('3.1')];dd=[1,2,6]
    for th in [mp.mpf('.2'),mp.mpf('1.1'),mp.mpf('2.4'),mp.pi]:
        val=mp.fsum(d*direct_loop(si,eps,th) for si,d in zip(ss,dd))
        if th!=mp.pi:worst_derivative=min(worst_derivative,val)
    record('shared_multisector_minimum_samples',worst_derivative>0,cases=4,
           masses_squared=[str(x) for x in ss],epsilon=str(eps),positive_multiplicities=dd,
           minimum_reference='theta=pi',smallest_nonreference_potential=mp.nstr(worst_derivative,12),
           note='Global common minimum is an analytical consequence of positivity, not a grid claim.')

    # Signs: finite checks supplement the integral proof, not replace it.
    sign_values=[]
    for rv in [mp.mpf('.03'),mp.mpf('.15'),mp.mpf('.4')]:
      for theta in [mp.mpf('.4'),mp.mpf('2.5')]:
        for n in range(7):
          value=(-1)**n*mp.diff(lambda ss:direct_loop(ss,rv,theta),mp.mpf(1),n)
          sign_values.append(value)
    record('complete_monotonicity_samples',all(v>0 for v in sign_values),
           derivative_orders=list(range(7)),cases=len(sign_values),
           note='All-orders result is proved analytically in the report.')

    # General one-loop metric, invariant under constant unitary changes of basis.
    max_metric=0.;min_metric=float('inf');max_C=0.;max_CP=0.
    for _ in range(24):
        eigen=np.exp(rng.uniform(-2,2,3));V=unitary(3)
        XX=V@np.diag(eigen)@V.conj().T;AA=hermitian(3);W=unitary(3)
        gg=matrix_metric(XX,AA);gg2=matrix_metric(W@XX@W.conj().T,W@AA@W.conj().T)
        min_metric=min(min_metric,gg)
        max_metric=max(max_metric,abs(gg-gg2)/gg)
        aa,bb=sorted(np.exp(rng.uniform(-2,2,2)))
        closed=((aa+bb)/2-aa*bb/(bb-aa)*np.log(bb/aa))/(16*np.pi**2)
        Cval=Ccoef(aa,bb)
        max_C=max(max_C,abs(Cval-closed)/max(Cval,1e-30))
        # At a rank-one orbit, metric = C tr(dPi^2) = 2C g_FS.
        w=rng.normal(size=2)+1j*rng.normal(size=2)
        Pi=np.diag([1.,0.,0.]);dPi=np.zeros((3,3),complex)
        dPi[1:,0]=w;dPi[0,1:]=w.conj()
        Xo=bb*np.eye(3)+(aa-bb)*Pi;Ao=(aa-bb)*dPi
        gp=matrix_metric(Xo,Ao);wanted=Cval*np.trace(dPi@dPi).real
        max_CP=max(max_CP,abs(gp-wanted)/wanted)
    record('positive_induced_metric',min_metric>0 and max_metric<1e-11,
           cases=24,min_sample_metric=min_metric,max_relative_unitary_invariance_error=max_metric)
    record('divided_difference_and_projective_metric',max_C<1e-8 and max_CP<1e-11,
           cases=24,max_relative_C_integral_vs_log_error=max_C,max_relative_CP2_metric_error=max_CP)

    # Direct bubble derivative check: d/dp^2 of logarithmic Feynman integral.
    bubble_errors=[]
    for av,bv in [(mp.mpf('.3'),mp.mpf('1.4')),(mp.mpf(1),mp.mpf(1)),(mp.mpf(1),mp.mpf(100))]:
        J=lambda pp:mp.quad(lambda u:mp.log((1-u)*av+u*bv+u*(1-u)*pp),[0,1])/(16*mp.pi**2)
        derivative=mp.diff(J,mp.mpf(0))
        integr=mp.quad(lambda u:u*(1-u)/((1-u)*av+u*bv),[0,1])/(16*mp.pi**2)
        bubble_errors.append(abs(derivative-integr))
    record('bubble_derivative_metric',max(bubble_errors)<mp.mpf('1e-65'),cases=3,
           max_absolute_error=mp.nstr(max(bubble_errors),10))

    # Equality at the spectral endpoint implies equal edges in a fixed,
    # zero-diagonal, phase-only graph. Counterexample at a nonendpoint retained.
    saturation_gap=[]
    for _ in range(100):
        w=np.exp(rng.normal(size=3));w*=np.sqrt(3/np.dot(w,w))
        saturation_gap.append(1-np.prod(w))
    record('triangle_saturation_samples',min(saturation_gap)>=-1e-14,cases=100,
           min_gap=float(min(saturation_gap)),equal_edge_gap=0.,
           analytic_condition='a^2+b^2+c^2=3eps^2; |abc|<=eps^3, equality iff a=b=c=eps.')

    # Exact homogeneous CP2 vacuum-orbit transport from the same induced C.
    # Spectral theta is fixed at pi; this is eigenvector motion, not theta evolution.
    ctotal=Ccoef(.7,1.15)+2*Ccoef(1.3,1.75)
    H=.17;max_J=0.;max_eom=0.;max_rho=0.;max_fd=0.;max_charge_shape=0.
    for _ in range(16):
        PP=np.diag([1.,0.,0.]).astype(complex)
        ww=.2*(rng.normal(size=2)+1j*rng.normal(size=2))
        dPP=np.zeros((3,3),complex);dPP[1:,0]=ww;dPP[0,1:]=ww.conj()
        JJ=1j*ctotal*(PP@dPP-dPP@PP)
        jp=ctotal*np.linalg.norm(ww)
        eigJ=np.linalg.eigvalsh(JJ)
        max_charge_shape=max(max_charge_shape,float(max(abs(eigJ-np.array([-jp,0,jp])))))
        def projector(tt):
            tau=-np.expm1(-3*H*tt)/(3*H*ctotal)
            VV=expm(1j*JJ*tau)
            return VV@PP@VV.conj().T
        tt=float(rng.uniform(.1,2));aa=np.exp(H*tt);Pt=projector(tt)
        vel=1j*(JJ@Pt-Pt@JJ)/(aa**3*ctotal)
        acc=-3*H*vel+1j*(JJ@vel-vel@JJ)/(aa**3*ctotal)
        current=1j*aa**3*ctotal*(Pt@vel-vel@Pt)
        eom=Pt@(acc+3*H*vel)-(acc+3*H*vel)@Pt
        rho=ctotal*np.trace(vel@vel).real/2
        rho0=ctotal*np.trace(dPP@dPP).real/2
        step=1e-5;fd=(projector(tt+step)-projector(tt-step))/(2*step)
        max_J=max(max_J,float(np.linalg.norm(current-JJ)/np.linalg.norm(JJ)))
        max_eom=max(max_eom,float(np.linalg.norm(eom)))
        max_rho=max(max_rho,float(abs(rho*aa**6/rho0-1)))
        max_fd=max(max_fd,float(np.linalg.norm(fd-vel)/np.linalg.norm(vel)))
    record('vacuum_orbit_noether_transport',max_J<1e-12 and max_eom<1e-12 and max_rho<1e-12 and max_fd<1e-8,
           cases=16,max_relative_current_error=max_J,max_absolute_eom_residual=max_eom,
           max_relative_a6_energy_invariant_error=max_rho,max_relative_finite_difference_velocity_error=max_fd,
           max_absolute_charge_eigenvalue_error=max_charge_shape,
           scope='Fixed spectral theta=pi, constant induced C, prescribed FLRW metric; initial charge is input.')

    # Negative controls establish limitations of the inference.
    ev=.15;sv=1.;cos_target=.3
    w=ev*np.sqrt(np.array([1.5,1.,.5]))
    ph=np.arccos(cos_target/(np.prod(w)/ev**3))
    Yuneq=np.array([[0,w[0],w[2]*np.exp(-1j*ph)],
                   [w[0],0,w[1]],[w[2]*np.exp(1j*ph),w[1],0]])
    spec_uneq=np.linalg.eigvalsh(sv*np.eye(3)+Yuneq)
    spec_target=np.sort([sv+2*ev*np.cos((np.arccos(cos_target)+2*np.pi*j)/3) for j in range(3)])
    # Two paths with identical eigenvalue derivatives; one also rotates eigenvectors.
    th=1.2;xs=np.array([sv+2*ev*np.cos((th+2*np.pi*j)/3) for j in range(3)])
    dx=np.array([-2*ev/3*np.sin((th+2*np.pi*j)/3) for j in range(3)])
    Om=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,0.]])
    Xdiag=np.diag(xs);A0=np.diag(dx);A1=A0+Om@Xdiag-Xdiag@Om
    g0=matrix_metric(Xdiag,A0);g1=matrix_metric(Xdiag,A1)
    expected_increase=2*Ccoef(xs[0],xs[1])
    # spectator cancellation at determinant level
    xr=np.array(spectral(mp.mpf(1),mp.mpf('.15'),mp.pi),dtype=float)
    LL0=np.sum(np.log(xs))-np.sum(np.log(xr))
    LLextra=np.sum(np.log(np.r_[xs,[7.,11.]]))-np.sum(np.log(np.r_[xr,[7.,11.]]))
    neg={
      'unequal_edges_single_state_same_spectrum_error':float(max(abs(spec_uneq-spec_target))),
      'equal_edges_not_implied_by_single_nonendpoint_state':bool(max(abs(spec_uneq-spec_target))<1e-12),
      'same_spectrum_different_kinetic_metric':{'nonrotating':g0,'rotating':g1,
         'positive_difference':g1-g0,'expected_difference':expected_increase},
      'spectator_cancellation_absolute_error':float(abs(LLextra-LL0)),
      'zero_response_inverse_undefined':True,
      'absolute_vacuum_constant_invisible_to_relative_responses':True,
      'phase_orientation_theta_vs_minus_theta_not_identified':True,
      'total_g2_not_equal_to_relative_leading_new_sector_term':True,
    }
    rejected=False
    try:inverse(mp.mpf(1),mp.mpf(0),mp.mpf(0))
    except ValueError:rejected=True
    record('negative_controls',rejected and neg['equal_edges_not_implied_by_single_nonendpoint_state']
           and g1>g0 and abs(g1-g0-expected_increase)<1e-12 and abs(LLextra-LL0)<1e-12,
           **neg)

    sv=mp.mpf(1);ev=mp.mpf('.15');th=mp.mpf('1.2')
    xs=spectral(sv,ev,th);xr=spectral(sv,ev,mp.pi)
    LL=mp.fsum(mp.log(xx) for xx in xs)-mp.fsum(mp.log(xx) for xx in xr)
    MM=mp.fsum(1/xx for xx in xs)-mp.fsum(1/xx for xx in xr)
    rr,cosv,eta=inverse(sv,LL,MM)
    example={k:mp.nstr(v,32) for k,v in dict(s=sv,epsilon=ev,theta=th,L=LL,M=MM,
          eta_inverse=eta,recovered_r=rr,recovered_cos_theta=cosv,U=direct_loop(sv,ev,th),
          Us=mp.diff(lambda z:direct_loop(z,ev,th),sv)).items()}
    return {'label':'CE-IRS1: conditional inverse structure and common response audit',
        'date':'2026-09-19','seed':SEED,'scope':'Mathematical model and implementation verification only.',
        'new_observational_data_used':False,'parameter_fit_performed':False,
        'full_joint_rmse':None,'observational_improvement_established':False,
        'whole_theory_proved':False,'remote_write':False,
        'conditional_inverse_and_realization_proofs':'See REPORT_ko.md for hypotheses and derivations.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                       'sympy':sy.__version__,'mpmath':mp.__version__},
        'verification_groups':checks,'group_count':len(checks),
        'all_groups_passed':all(x['passed'] for x in checks.values()),'example':example}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results.json',type=Path)
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
