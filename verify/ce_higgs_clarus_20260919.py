#!/usr/bin/env python3
"""CE-HC1: conditional Higgs/Clarus portal and unexpanded Gaussian pole.
No observations, fit, network, credentials or Git writes. This is NOT the
9-scalar/9-Majorana completion. All pole statements concern the fixed-background
Gaussian matter determinant with no independent collective quadratic terms.
Run: python ce_higgs_clarus_20260919.py --out results.json
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import numpy as np
import mpmath as mp
import sympy as sp
mp.mp.dps=80
RNG=np.random.default_rng(20260919)
PI=mp.pi

def eigs(s,e,th):
    return [s+2*e*mp.cos((th+2*mp.pi*j)/3) for j in range(3)]

def U(s,e,th):
    f=lambda x:x*x*(mp.log(x)-mp.mpf(3)/2)
    return (mp.fsum(f(x) for x in eigs(s,e,th))-mp.fsum(f(x) for x in eigs(s,e,mp.pi)))/(32*PI**2)

def kval(s,e): return e*e/(144*PI**2*(s+e))
def muval(s,e):
    a,b=s-2*e,s+e
    return e/(72*PI**2)*(3*e-a*mp.log(b/a))

def beta(s,e):
    return 1-(s-2*e)/(3*e)*mp.log((s+e)/(s-2*e))

def jfun(z):
    if z==0:return mp.mpf(0)
    if not 0<z<4:raise ValueError('Requires 0<z<4.')
    return 2-2*mp.sqrt((4-z)/z)*mp.atan(mp.sqrt(z/(4-z)))

def jprime(z):
    return mp.quad(lambda t:t*(1-t)/(1-z*t*(1-t)),[0,1])

def bisect(f,lo,hi,steps=220):
    lo,hi=mp.mpf(lo),mp.mpf(hi)
    if not f(lo)<=0<=f(hi):raise ValueError('Root not bracketed.')
    for _ in range(steps):
        mid=(lo+hi)/2
        if f(mid)<0:lo=mid
        else:hi=mid
    return (lo+hi)/2,lo,hi

def pole(s,e):
    if not s>2*e>0:raise ValueError('Requires s>2epsilon>0.')
    B=beta(s,e)
    z,lo,hi=bisect(lambda z:jfun(z)-B,mp.mpf('1e-70'),mp.mpf('3.99999999999999999999'))
    return (s+e)*z,z,lo,hi

def from_pole(b,m2):
    z=m2/b; B=jfun(z)
    if not 0<B<1:raise ValueError('Pole is outside declared class.')
    def bt(t):return 1+(1-t)/t*mp.log1p(-t)
    t,_,_=bisect(lambda t:bt(t)-B,mp.mpf('1e-65'),1-mp.mpf('1e-65'))
    e=t*b/3
    return b-e,e

def run():
    result={}
    def test(name,ok,**details):
        result[name]={'passed':bool(ok),**details}
        if not ok:raise AssertionError((name,details))
    # A finite exact polarization set forces any Hermitian portal to be scalar.
    a,b,c,d,e,f,g,h,i=sp.symbols('a b c d e f g h i',real=True)
    B=sp.Matrix([[a,d+sp.I*e,f+sp.I*g],[d-sp.I*e,b,h+sp.I*i],[f-sp.I*g,h-sp.I*i,c]])
    basis=[sp.eye(3)[:,k] for k in range(3)]; states=list(basis)
    for p,q in [(0,1),(0,2),(1,2)]:
        states += [(basis[p]+basis[q])/sp.sqrt(2),(basis[p]+sp.I*basis[q])/sp.sqrt(2)]
    equations=[sp.expand((v.conjugate().T*B*v)[0]-a) for v in states[1:]]
    sol=sp.linsolve(equations,(a,b,c,d,e,f,g,h,i))
    test('portal_polarization_necessity',len(sol)==1 and list(sol)[0][0]==list(sol)[0][1]==list(sol)[0][2] and all(x==0 for x in list(sol)[0][3:]),
         independent_equation_rank=sp.Matrix(equations).jacobian([a,b,c,d,e,f,g,h,i]).rank(),solution=str(sol))
    # Direct trace-square comparison including a violating anisotropic portal.
    max_res=0.;max_identity=0.;anis=[]
    for _ in range(64):
        eps=.13;u=.27;s=2.
        Z=RNG.normal(size=(3,3))+1j*RNG.normal(size=(3,3));M=(Z+Z.conj().T)/8
        v=RNG.normal(size=3)+1j*RNG.normal(size=3);v/=np.linalg.norm(v)
        P=np.outer(v,v.conj());P0=np.diag([1.,0.,0.]);Y=eps*np.eye(3)-3*eps*P;Y0=eps*np.eye(3)-3*eps*P0
        X=s*np.eye(3)+Y+u*M;X0=s*np.eye(3)+Y0+u*M
        lhs=np.trace(X@X-X0@X0).real;rhs=-6*u*eps*np.trace(M@(P-P0)).real
        max_res=max(max_res,abs(lhs-rhs));anis.append(abs(lhs))
        Xi=(s+.2*u)*np.eye(3)+Y;Xi0=(s+.2*u)*np.eye(3)+Y0
        max_identity=max(max_identity,abs(np.trace(Xi@Xi-Xi0@Xi0)))
    test('relative_UV_trace_identity',max_res<1e-13 and max_identity<1e-13 and max(anis)>.01,
         cases=64,max_trace_identity_error=max_res,max_isotropic_residual=float(max_identity),max_anisotropic_violation=max(anis))
    # Moment reconstruction under arbitrary constant unitary changes of basis.
    errs=[];gram_errors=[];gram_min=1.;cross=[]
    for _ in range(64):
        s0=1.;kap=.2;rad=float(RNG.uniform(.2,1.5));u=rad*rad/2;sv=s0+kap*u;ep=.1;th=float(RNG.uniform(.15,2.9))
        x=sv+2*ep*np.cos((th+2*np.pi*np.arange(3))/3);xp=-2*ep/3*np.sin((th+2*np.pi*np.arange(3))/3)
        Z=RNG.normal(size=(3,3))+1j*RNG.normal(size=(3,3));V,_=np.linalg.qr(Z);X=V@np.diag(x)@V.conj().T
        sr=np.trace(X).real/3;Y=X-sr*np.eye(3);er=np.sqrt(np.trace(Y@Y).real/6);cr=np.trace(Y@Y@Y).real/(6*er**3)
        errs.append(max(abs(sr-sv),abs(er-ep),abs(cr-np.cos(th))))
        w=1/x;tangent=np.array([np.full(3,kap*rad),xp]);gg=(tangent*w)@tangent.T/(96*np.pi**2)
        det_expected=(kap*rad)**2*sum(w[p]*w[q]*(xp[p]-xp[q])**2 for p,q in [(0,1),(0,2),(1,2)])/(96*np.pi**2)**2
        gram_errors.append(abs(np.linalg.det(gg)-det_expected)/det_expected);gram_min=min(gram_min,float(np.linalg.eigvalsh(gg)[0]))
        closed=-kap*rad*ep**3*np.sin(th)/(48*np.pi**2*np.prod(x));cross.append(abs(gg[0,1]-closed))
    test('source_shape_inverse_and_metric',max(errs)<1e-10 and max(gram_errors)<1e-12 and gram_min>0 and max(cross)<1e-17,
         cases=64,max_moment_inverse_error=max(errs),max_metric_determinant_relative=max(gram_errors),min_metric_eigenvalue=gram_min,max_cross_metric_error=max(cross))
    # All mixed source responses use epsilon fixed. Compare original eigenvalue sums.
    mixed=[];vac=[]
    for sv,ep,th in [(mp.mpf(1),mp.mpf('.15'),mp.mpf('1.2')),(mp.mpf(2),mp.mpf('.02'),mp.mpf('.7'))]:
        kap=mp.mpf('.3');ur=mp.mpf('.4');s0=sv-kap*ur
        lhs=mp.diff(lambda u:mp.diff(lambda t:U(s0+kap*u,ep,t),th),ur)
        rhs=kap*mp.diff(lambda ss:mp.diff(lambda t:U(ss,ep,t),th),sv)
        mixed.append(abs(lhs-rhs));vac.append(abs(mp.diff(lambda ss:mp.diff(lambda t:U(ss,ep,t),mp.pi),sv)))
    test('mixed_generator_and_vacuum_decoupling',max(mixed+vac)<mp.mpf('1e-65'),cases=2,max_error=mp.nstr(max(mixed+vac),12))
    # Vacuum Hessian and kinetic coefficient: direct eigenvalue derivatives vs closed forms.
    curvature=[];kin=[];bubble=[]
    for rr in ['.0001','.001','.01','.15','.35','.49']:
        sv=mp.mpf(1);ep=mp.mpf(rr)
        mm=mp.diff(lambda th:U(sv,ep,th),mp.pi,2);k=mp.fsum(mp.diff(lambda th:eigs(sv,ep,th)[j],mp.pi)**2/eigs(sv,ep,mp.pi)[j] for j in range(3))/(96*PI**2)
        curvature.append(abs(mm/muval(sv,ep)-1));kin.append(abs(k/kval(sv,ep)-1))
        xhi=sv+ep
        dp=mp.diff(lambda p:ep**2/(24*PI**2)*mp.quad(lambda t:mp.log1p(t*(1-t)*p/xhi),[0,1]),mp.mpf(0))
        bubble.append(abs(dp/k-1))
    test('vacuum_curvature_and_unexpanded_kernel',max(curvature+kin+bubble)<mp.mpf('1e-60'),cases=6,max_relative_error=mp.nstr(max(curvature+kin+bubble),12))
    rows=[];pole_errors=[];quad_errors=[];inverse_errors=[]
    for rr in ['.0001','.001','.01','.05','.15','.35','.49']:
        sv=mp.mpf(1);ep=mp.mpf(rr);m2,z,lo,hi=pole(sv,ep);BB=beta(sv,ep)
        integ=-mp.quad(lambda t:mp.log1p(-z*t*(1-t)),[0,.5,1]);quad_errors.append(abs(integ-jfun(z)))
        pole_errors.append(abs(jfun(z)-BB));sr,er=from_pole(sv+ep,m2);inverse_errors.append(max(abs(sr-sv),abs(er-ep)))
        m2low=muval(sv,ep)/kval(sv,ep);residue_slope=ep**2/(24*PI**2*(sv+ep))*jprime(z)
        if not(0<lo<z<hi<4 and m2<m2low and residue_slope>0):raise AssertionError('pole positivity')
        row={key:mp.nstr(val,22) for key,val in dict(r=ep,exact_gaussian_mass_squared=m2,two_derivative_mass_squared=m2low,
            two_derivative_relative_overestimate=m2low/m2-1,z=z,beta=BB,bracket_width=hi-lo,positive_pole_slope=residue_slope).items()}
        row['closed_form_target_bracketed']=bool(jfun(lo)<=BB<=jfun(hi));rows.append(row)
    test('unique_positive_residue_pole_and_inverse',max(pole_errors+quad_errors+inverse_errors)<mp.mpf('1e-55'),
         cases=7,max_pole_residual=mp.nstr(max(pole_errors),12),max_independent_integral_error=mp.nstr(max(quad_errors),12),
         max_inverse_spectrum_error=mp.nstr(max(inverse_errors),12),rows=rows)
    # Exact unexpanded pole response vs actual finite changes of the same Higgs source.
    sens=[];sens_err=[]
    for rr in ['.001','.01','.05','.15']:
        sv=mp.mpf(1);ep=mp.mpf(rr);kap=mp.mpf('.2');m2,z,_,_=pole(sv,ep);bb=sv+ep
        Bs=1/bb-mp.log(bb/(sv-2*ep))/(3*ep)
        deriv=kap*(z+bb*Bs/jprime(z))
        step=mp.mpf('1e-8')
        fun=lambda u:pole(sv+kap*u,ep)[0]
        fd=(-fun(2*step)+8*fun(step)-8*fun(-step)+fun(-2*step))/(12*step)
        sens_err.append(abs((fd-deriv)/deriv))
        m0=muval(sv,ep)/kval(sv,ep);pot=kap*mp.diff(lambda s:muval(s,ep),sv)/kval(sv,ep)
        kinetic=kap*mp.diff(lambda s:kval(s,ep),sv)/kval(sv,ep)
        low=kap*mp.diff(lambda s:muval(s,ep)/kval(s,ep),sv)
        if abs(low-(pot-m0*kinetic))>mp.mpf('1e-60'):raise AssertionError('normalization')
        sens.append({key:mp.nstr(val,22) for key,val in dict(r=ep,kappa=kap,potential_only_coefficient=pot,
                    derivative_kinetic_coefficient=kinetic,two_derivative_mass_response=low,full_gaussian_mass_response=deriv).items()})
    test('Higgs_source_pole_sensitivity',max(sens_err)<mp.mpf('1e-20'),cases=4,max_five_point_relative_error=mp.nstr(max(sens_err),12),rows=sens)
    # Symbolic low-r expansion: leading finite-momentum correction matters for source sensitivity.
    r=sp.symbols('r');B=1-(1-2*r)/(3*r)*sp.log((1+r)/(1-2*r))
    zser=9*r-sp.Rational(81,10)*r**2+sp.Rational(3033,350)*r**3
    Jser=sum(zser**n*sp.factorial(n)**2/(n*sp.factorial(2*n+1)) for n in range(1,4))
    resid=sp.series(Jser-B,r,0,4).removeO().expand()
    mass_ser=sp.series((1+r)*zser,r,0,4).removeO().expand()
    expected=9*r+sp.Rational(9,10)*r**2+sp.Rational(99,175)*r**3
    test('pole_asymptotic_coefficients',resid==0 and sp.expand(mass_ser-expected)==0,
         exact_gaussian_m2_over_s=str(mass_ser),remainder='O(r^4)',two_derivative_m2_over_s='9*r + 9*r^2 + 9*r^3/2 + O(r^4)')
    # Coupled Higgs/Clarus stationary equations after eliminating the same conserved charge.
    Vh,Vt,Ah,At,AA,nn=sp.symbols('Vh Vt Ah At AA nn',nonzero=True)
    gh=Vh-nn**2*Ah/AA**2;gt=Vt-nn**2*At/AA**2
    cond=sp.expand(At*gh-Ah*gt-(At*Vh-Ah*Vt))
    test('common_charge_stationary_elimination',cond==0,equations=['n^2=A^2 V_theta/A_theta>=0','V_h A_theta - V_theta A_h=0'],
         scope='Necessary and sufficient for both stationary gradients when A>0 and A_theta!=0; not existence or stability.')
    # Representation and mode multiplicity algebra: a singlet has no SU(2) doublet orbit.
    pauli=[sp.Matrix([[0,1],[1,0]]),sp.Matrix([[0,-sp.I],[sp.I,0]]),sp.diag(1,-1)]
    cas=sp.zeros(2)
    for S in pauli:cas+=(S/2)**2
    hb=[sp.diag(1,0,0),sp.diag(0,1,0),sp.diag(0,0,1)]
    for p,q in [(0,1),(0,2),(1,2)]:
        T=sp.zeros(3);T[p,q]=T[q,p]=1;hb.append(T)
        T=sp.zeros(3);T[p,q]=sp.I;T[q,p]=-sp.I;hb.append(T)
    Y0=sp.diag(-2,1,1)
    constraints=sp.Matrix([[sp.trace(T) for T in hb],[sp.trace(Y0*T) for T in hb]])
    tangent_dim=9-constraints.rank()
    columns=[]
    for T in hb:
        Q=sp.I*(T*Y0-Y0*T)
        columns.append(sp.Matrix(list(Q.applyfunc(sp.re))+list(Q.applyfunc(sp.im))))
    orbit_dim=sp.Matrix.hstack(*columns).rank()
    test('gauge_representation_and_internal_modes',cas==sp.Rational(3,4)*sp.eye(2) and tangent_dim==7 and orbit_dim==4,
         Higgs_SU2_Casimir='3/4',SM_singlet_Casimir=0,constrained_matrix_real_tangent=tangent_dim,
         vacuum_orbit_real_dimensions=orbit_dim,normal_traceless_2x2_real_dimensions=tangent_dim-orbit_dim,
         note='Counts and representation algebra, not observed particle counts. Four orbit modes are global-symmetry modes in this candidate.')
    # Demonstrate why a fixed channel path is insufficient for the portal uniqueness theorem.
    ep=.1;sv=2.;M=np.diag([1.,2.,3.]);cyc=np.roll(np.eye(3),1,axis=1)
    Ys=lambda th:ep*(np.exp(1j*th/3)*cyc+np.exp(-1j*th/3)*cyc.T)
    narrow=abs(np.trace(M@(Ys(.4)-Ys(np.pi))))
    p=np.diag([1.,0.,0.]);p1=np.diag([0.,1.,0.]);broad=abs(np.trace(M@(-3*ep*(p1-p))))
    test('negative_controls',narrow<1e-14 and broad>.1,
         anisotropic_portal_passes_only_fixed_channel_path=float(narrow),same_portal_fails_full_vacuum_orbit=float(broad),
         untested=['Higgs decay width or finite-momentum 3-point vertex','SM parameter prediction','collective or gravity loops','observational likelihood'],
         free_inputs=['portal kappa','Higgs potential and Yukawa couplings','s0 and epsilon','local vacuum/gravity/kinetic matching','state and conserved charge'])
    return {'label':'CE-HC1: Higgs/Clarus distinction, portal selection and Gaussian pole','date':'2026-09-19',
            'base_commit':'5ede900537a20da8a30edad9b0709b13a34f9151','groups':result,'group_count':len(result),'all_passed':all(v['passed'] for v in result.values()),
            'observations_used':False,'parameter_fit':False,'full_joint_rmse':None,'scientific_success':False,
            'exactness_scope':'Gaussian matter determinant two-point kernel on constant flat background; not all-loop exact theory.',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'mpmath':mp.__version__,'sympy':sp.__version__}}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results.json'));args=ap.parse_args()
    ans=run();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(ans,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'group_count':ans['group_count'],'all_passed':ans['all_passed'],'out':str(args.out)},ensure_ascii=False))
