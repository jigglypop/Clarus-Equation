"""CE-IR1: quartic correction to the Gaussian common-response identity.

The coefficient linear in the U(3)-invariant quartic is calculated in MSbar
with an external Abelian field. This is not the complete dynamical gauge theory
at two loops. The analytic derivation and its premises are in chapter 34.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from mpmath import iv

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
mp.mp.dps=70


def spectrum(s,e,c):
    theta=mp.acos(c)
    return [s+2*e*mp.cos((theta+2*mp.pi*j)/3) for j in range(3)]


def tadpole(x,mu2):
    return x*(mp.log(x/mu2)-1)/(16*mp.pi**2)


def vacuum0(xs,mu2):
    return sum(x*x*(mp.log(x/mu2)-mp.mpf('1.5')) for x in xs)/(32*mp.pi**2)


def vacuum1(xs,mu2,lam):
    aa=[tadpole(x,mu2) for x in xs]
    return lam*(sum(aa)**2+sum(a*a for a in aa))/2


def gauge0(xs,mu2,q):
    return -q*q*sum(mp.log(x/mu2) for x in xs)/(48*mp.pi**2)


def gauge1(xs,mu2,q,lam):
    aa=[tadpole(x,mu2) for x in xs]
    app=[1/(16*mp.pi**2*x) for x in xs]
    return -lam*q*q*(sum(aa)*sum(app)+sum(a*b for a,b in zip(aa,app)))/3


def C(xs,mu2):
    logs=[mp.log(x/mu2) for x in xs]
    return sum(logs)**2+sum(l*l for l in logs)


def magnetic_tadpole_change(x,B,q):
    # Finite background difference, independent of a vacuum subtraction.
    def integrand(t):
        z=q*B*t
        if abs(z)<mp.mpf('1e-10'):
            # Remainder O(z^8); divided by t^2 it is integrable at zero.
            change=-z*z/6+7*z**4/360-31*z**6/15120
        else:
            change=z/mp.sinh(z)-1
        return mp.exp(-x*t)*change/(t*t)
    return mp.quad(integrand,[0,1/x,mp.inf])/(16*mp.pi**2)


def finite_field_gauge(xs,mu2,q,lam,B):
    aa=[tadpole(x,mu2) for x in xs]
    dd=[magnetic_tadpole_change(x,B,q) for x in xs]
    # Expand the difference algebraically to avoid subtracting close squares.
    dv=lam*(2*sum(aa)*sum(dd)+sum(dd)**2+
            sum(2*a*d+d*d for a,d in zip(aa,dd)))/2
    return 2*dv/(B*B)


def main():
    checks=[]
    def check(name,condition):
        assert bool(condition),name
        checks.append(name)
    def near(name,a,b,tol=mp.mpf('1e-55')):
        check(name,abs(a-b)<tol)

    A=sp.symbols('A0:3'); Ap=sp.symbols('Ap0:3'); App=sp.symbols('App0:3')
    P=sp.symbols('P0:3')
    lam,q,h=sp.symbols('lam q h')
    form=lambda values:lam*(sum(values)**2+sum(v*v for v in values))/2
    V=form(A)
    D=lambda expr:sum(sp.diff(expr,A[i])*Ap[i]+sp.diff(expr,Ap[i])*App[i] for i in range(3))
    k=-lam*q*q*(sum(A)*sum(App)+sum(A[i]*App[i] for i in range(3)))/3
    residual=sp.expand(k+q*q*D(D(V))/3)
    expected=lam*q*q*(sum(Ap)**2+sum(v*v for v in Ap))/3
    check('exact_common_response_correction',sp.expand(residual-expected)==0)
    # Wick normalization: Phi=(Q1+i Q2)/sqrt(2).
    variances=[a for a in A for _ in range(2)]
    wick_O2=(sum(variances)**2+2*sum(v*v for v in variances))/4
    check('six_real_coordinate_Wick_factor',
          sp.expand(wick_O2-(sum(A)**2+sum(a*a for a in A)))==0)
    # Forest subtraction of tadpole subdivergences; the vacuum pole is local.
    G=[P[i]+A[i] for i in range(3)]
    counter=sum(sp.diff(form(P),P[i])*G[i] for i in range(3))
    check('quadratic_subdivergence_subtraction',
          sp.expand(form(G)-counter+form(P)-form(A))==0)
    field=[A[i]-q*q*h*App[i]/6 for i in range(3)]
    check('magnetic_coefficient_factor_two',
          sp.expand(2*sp.diff(form(field),h).subs(h,0)-k)==0)
    alpha=sp.symbols('alpha')
    curved=[A[i]-alpha*h*Ap[i] for i in range(3)]
    check('linear_curvature_relation_survives',
          sp.expand(sp.diff(form(curved),h).subs(h,0)+alpha*D(V))==0)
    a,L,C0=sp.symbols('a L C0')
    # N=3 logs: C -> C-8 a L+12 a^2; differences remove the constant.
    logs=sp.symbols('L0:3')
    check('common_log_shift',
          sp.expand((sum(l-a for l in logs))**2+sum((l-a)**2 for l in logs)
                    -(sum(logs)**2+sum(l*l for l in logs)-8*a*sum(logs)+12*a*a))==0)

    numerical=[]
    for rtxt in ['0.1','0.25','0.49']:
        s,e,mu2=mp.mpf(1),mp.mpf(rtxt),mp.mpf(1)
        coupling,charge=mp.mpf('.1'),mp.mpf('.7')
        c=mp.mpf('.2')
        xs=spectrum(s,e,c)
        vss=mp.diff(lambda ss:vacuum1(spectrum(ss,e,c),mu2,coupling),s,2)
        vs=mp.diff(lambda ss:vacuum1(spectrum(ss,e,c),mu2,coupling),s)
        aa_curve=[tadpole(x,mu2) for x in xs]
        ap_curve=[mp.log(x/mu2)/(16*mp.pi**2) for x in xs]
        alpha_curve=mp.mpf(1)/6-mp.mpf('.1')
        curvature_coefficient=-alpha_curve*coupling*(
            sum(aa_curve)*sum(ap_curve)+sum(a*b for a,b in zip(aa_curve,ap_curve)))
        near('independent_curvature_coefficient_'+rtxt,curvature_coefficient,-alpha_curve*vs)
        correction=coupling*charge**2*C(xs,mu2)/(3*(16*mp.pi**2)**2)
        near('direct_mass_derivative_'+rtxt,gauge1(xs,mu2,charge,coupling)+charge**2*vss/3,correction)
        # Independent mass-insertion calculation of the same photon coefficient.
        aa=[tadpole(x,mu2) for x in xs]
        inserted=sum(coupling*(sum(aa)+aa[j])*(-charge**2/(48*mp.pi**2*xs[j]))
                     for j in range(3))
        near('independent_mass_insertion_'+rtxt,inserted,gauge1(xs,mu2,charge,coupling))
        B=min(xs)*mp.mpf('1e-4')
        kb=finite_field_gauge(xs,mu2,charge,coupling,B)
        kh=finite_field_gauge(xs,mu2,charge,coupling,B/2)
        richardson=(4*kh-kb)/3
        near('independent_finite_magnetic_field_'+rtxt,richardson,inserted,mp.mpf('1e-20'))
        numerical.append({'epsilon_over_s':rtxt,'finite_field_error':str(abs(richardson-inserted)),
                          'absolute_common_response_residual':str(correction)})

    # Outward interval certificate for a scale-shift obstruction and invertibility.
    iv.dps=60
    one=iv.mpf(1); half=one/2
    ref=[half,5*one/4,5*one/4]
    xs0=[one-iv.sqrt(3)/4,one,one+iv.sqrt(3)/4]
    xs1=[3*one/4,3*one/4,3*one/2]
    L_iv=lambda xs:sum(iv.ln(x) for x in xs)
    C_iv=lambda xs:L_iv(xs)**2+sum(iv.ln(x)**2 for x in xs)
    dc0,dc1=C_iv(xs0)-C_iv(ref),C_iv(xs1)-C_iv(ref)
    dl0,dl1=L_iv(xs0)-L_iv(ref),L_iv(xs1)-L_iv(ref)
    J=dc1*dl0-dc0*dl1
    matrix_det=(3*dc1/16-3*dc0/8)/(16*iv.pi**2)**2
    lower=lambda x:float(np.nextafter(float(x.a),-np.inf))
    upper=lambda x:float(np.nextafter(float(x.b),np.inf))
    check('log_shift_obstruction_interval_positive',lower(J)>0)
    check('two_response_inverse_interval_nonsingular',lower(matrix_det)>0)
    interval={'precision':iv.dps,'J':str(J),'J_outward_bounds':[lower(J),upper(J)],
              'response_determinant':str(matrix_det),
              'response_determinant_outward_bounds':[lower(matrix_det),upper(matrix_det)]}

    s,e,mu2=mp.mpf(1),mp.mpf('.25'),mp.mpf(1)
    charge=mp.mpf(1)
    beta_true,lambda_true=mp.mpf('-.0001'),mp.mpf('.1')
    cc=[mp.mpf(0),mp.mpf(1)]
    aa=[12*e**3*(c+1)/s**3 for c in cc]
    bb=[(C(spectrum(s,e,c),mu2)-C(spectrum(s,e,-1),mu2))/(16*mp.pi**2)**2 for c in cc]
    recorded=[]
    for c in cc:
        def vtotal(ss,ct):
            xs=spectrum(ss,e,ct)
            return (vacuum0(xs,mu2)+vacuum1(xs,mu2,lambda_true)
                    +6*beta_true*e**3*ct/ss)
        vss=mp.diff(lambda ss:vtotal(ss,c)-vtotal(ss,-1),s,2)
        def ktotal(ct):
            xs=spectrum(s,e,ct)
            return gauge0(xs,mu2,charge)+gauge1(xs,mu2,charge,lambda_true)
        recorded.append(vss+3*(ktotal(c)-ktotal(-1))/charge**2)
    det=aa[0]*bb[1]-aa[1]*bb[0]
    beta=(bb[1]*recorded[0]-bb[0]*recorded[1])/det
    coupling=(aa[0]*recorded[1]-aa[1]*recorded[0])/det
    near('two_response_beta_recovery',beta,beta_true)
    near('two_response_lambda_recovery',coupling,lambda_true)
    for shift in [mp.mpf('-2'),mp.mpf('1.5')]:
        shifted=mu2*mp.exp(shift)
        ds=[C(spectrum(s,e,c),shifted)-C(spectrum(s,e,-1),shifted) for c in cc]
        dl=[mp.log(mp.fprod(spectrum(s,e,c))/mp.fprod(spectrum(s,e,-1))) for c in cc]
        original=bb[1]*(16*mp.pi**2)**2*dl[0]-bb[0]*(16*mp.pi**2)**2*dl[1]
        near('log_shift_invariant_'+str(shift),ds[1]*dl[0]-ds[0]*dl[1],original)
    inverse={'s':str(s),'epsilon':str(e),'mu_squared':str(mu2),
             'response_matrix':[[str(aa[i]),str(bb[i])] for i in range(2)],
             'synthetic_responses':[str(r) for r in recorded],
             'beta_recovered':str(beta),'lambda_recovered':str(coupling),
             'false_beta_if_lambda_omitted_at_c_one':str(bb[1]*lambda_true/aa[1]),
             'beta_error_factor':str((abs(bb[0])+abs(bb[1]))/abs(det)),
             'lambda_error_factor':str((abs(aa[0])+abs(aa[1]))/abs(det))}

    # A different inverse uses curvature and gauge responses at one shape.
    # Curvature is calculated from the background tadpole expansion, not by
    # substituting the total potential derivative whose identity is being tested.
    alpha_value=mp.mpf(1)/6-mp.mpf('.1')
    def curvature_total(ct):
        xs=spectrum(s,e,ct)
        av=[tadpole(x,mu2) for x in xs]
        ap=[mp.log(x/mu2)/(16*mp.pi**2) for x in xs]
        return -alpha_value*(sum(av)+lambda_true*(
            sum(av)*sum(ap)+sum(a*b for a,b in zip(av,ap))))
    shape=mp.mpf(1)
    def potential(ss,ct):
        xs=spectrum(ss,e,ct)
        return (vacuum0(xs,mu2)+vacuum1(xs,mu2,lambda_true)
                +6*beta_true*e**3*ct/ss)
    energy_s=mp.diff(lambda ss:potential(ss,shape)-potential(ss,-1),s)
    curvature_residual=curvature_total(shape)-curvature_total(-1)+alpha_value*energy_s
    d=-6*alpha_value*e**3*(shape+1)/s**2
    beta_curvature=curvature_residual/d
    lambda_joint=(recorded[1]-aa[1]*beta_curvature)/bb[1]
    near('curvature_isolates_finite_coefficient',beta_curvature,beta_true)
    near('one_shape_curvature_gauge_recovers_quartic',lambda_joint,lambda_true)
    check('one_shape_inverse_nonsingular',d*bb[1]!=0)
    joint_inverse={'alpha':str(alpha_value),'curvature_response_coefficient':str(d),
                   'curvature_residual':str(curvature_residual),
                   'beta_recovered':str(beta_curvature),'lambda_recovered':str(lambda_joint),
                   'determinant':str(d*bb[1]),
                   'beta_error_per_curvature_error':str(1/abs(d)),
                   'lambda_error_per_gauge_error':str(1/abs(bb[1])),
                   'lambda_error_per_curvature_error':str(abs(aa[1]/(d*bb[1])))}

    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    parent=ROOT/'paper'
    sources=[next((parent/'06_QFT_재설계').glob('82_*.md')),
             next((parent/'후속연구_기록과_상태선택').glob('28_*.md')),
             next((parent/'후속연구_기록과_상태선택').glob('33_*.md'))]
    result={'scope':'coefficient linear in the U(3)-invariant quartic; external Abelian field; MSbar',
            'all_checks_passed':True,'checks':checks,
            'source_sha256':sha(Path(__file__)),
            'premise_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'numerical_checks':numerical,'interval_certificate':interval,
            'two_response_inverse':inverse,'curvature_gauge_inverse':joint_inverse,
            'full_goal_complete':False,
            'not_claimed':['complete two-loop dynamical gauge theory','all orders in lambda',
                           'scheme-independent mass-source definition','RG-invariant full observable',
                           'natural parameter selection','measured responses','UV completion']}
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'interval_certificate':interval,
                      'two_response_inverse':inverse},indent=2))


if __name__=='__main__':
    main()
