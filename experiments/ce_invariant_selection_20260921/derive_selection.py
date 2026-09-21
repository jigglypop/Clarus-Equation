"""CE-IS1: invariant classification, exact phase boundary, and reverse response.

This audits the missing finite-matching premise in the original Gaussian branch.
It does not refute that branch with its no-added-potential premise retained.
Analytic proofs, including statements for every 0 < epsilon/s < 1/2, are in
chapter 33. Numerical checks below compare independent expressions, not a scan
offered in place of those proofs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
mp.mp.dps = 70


def spectrum(s, e, c):
    theta = mp.acos(c)
    return [s + 2*e*mp.cos((theta+2*mp.pi*j)/3) for j in range(3)]


def gaussian(s, e, c):
    f = lambda x: x*x*(mp.log(x)-mp.mpf(3)/2)
    return (sum(map(f, spectrum(s,e,c))) -
            sum(map(f, spectrum(s,e,mp.mpf(-1)))))/(32*mp.pi**2)


def determinant(t, s, e, c):
    return (t+s)**3 - 3*e**2*(t+s) + 2*e**3*c


def relative_integral(s, e, c):
    return mp.quad(lambda t: t*mp.log1p(
        2*e**3*(c+1)/determinant(t,s,e,-1)), [0,s,mp.inf])/(16*mp.pi**2)


def derivative_c(s, e, c):
    return e**3/(8*mp.pi**2)*mp.quad(
        lambda t: t/determinant(t,s,e,c), [0,s,mp.inf])


def second_c(s, e, c):
    return -e**6/(4*mp.pi**2)*mp.quad(
        lambda t: t/determinant(t,s,e,c)**2, [0,s,mp.inf])


def relative_total(s, e, c, beta):
    return gaussian(s,e,c) + 6*beta*e**3*(c+1)/s


def total_theta(s, e, theta, beta):
    xs=[s+2*e*mp.cos((theta+2*mp.pi*j)/3) for j in range(3)]
    raw=sum(x*x*(mp.log(x)-mp.mpf(3)/2) for x in xs)/(32*mp.pi**2)
    return raw+6*beta*e**3*mp.cos(theta)/s


def main():
    checks = []
    def check(name, condition):
        assert bool(condition), name
        checks.append(name)
    def near(name, a, b, tol=mp.mpf('1e-55')):
        check(name, abs(a-b) < tol)

    # Exact trace algebra; an arbitrary traceless diagonal covers conjugacy.
    a,b,z = sp.symbols('a b z')
    ys = [a,b,-a-b]
    p = {n: sp.expand(sum(y**n for y in ys)) for n in range(1,9)}
    check('Cayley_Hamilton_for_each_eigenvalue', all(
        sp.expand(y**3-p[2]*y/2-p[3]/3)==0 for y in ys))
    check('fourth_trace_has_no_new_shape',
          sp.expand(p[4]-p[2]**2/2)==0)
    check('fifth_trace_reduces',sp.expand(p[5]-sp.Rational(5,6)*p[2]*p[3])==0)
    check('sixth_trace_reduces',
          sp.expand(p[6]-p[2]**3/4-p[3]**2/3)==0)
    # The discriminant is exactly the Hermitian shape interval constraint.
    p2,p3 = sp.symbols('p2 p3')
    disc=sp.discriminant(z**3-p2*z/2-p3/3,z)
    check('discriminant_identity',sp.expand(disc-(p2**3/2-3*p3**2))==0)

    # Three independent shape directions normal to the CP2 minimum orbit.
    e,t,z1,z2,z3=sp.symbols('e t z1 z2 z3',positive=True,real=True)
    ym=sp.diag(-2*e,e,e)
    normal=sp.Matrix([[0,0,0],[0,z1,z2-sp.I*z3],[0,z2+sp.I*z3,-z1]])
    q=sp.trace(normal*normal)
    raw=sp.trace((ym+t*normal)**3)
    curve=raw*(1+t*t*q/(6*e*e))**(-sp.Rational(3,2))/(6*e**3)
    check('full_normal_Hessian_coefficient',
          sp.simplify(sp.diff(curve,t,2).subs(t,0)-3*q/(2*e**2))==0)

    # The logarithm expansion proves positivity and monotonicity term by term.
    r=sp.symbols('r',positive=True)
    f=lambda x:x*x*sp.log(x)
    numerator=f(1+2*r)+2*f(1-r)-f(1-2*r)-2*f(1+r)
    expansion=sp.series(numerator/(4*r**3),r,0,10).removeO().expand()
    for k in range(5):
        n=2*k+3
        coefficient=sp.Rational(2**n-2,n*(n-1)*(n-2))
        check('positive_series_coefficient_'+str(k),
              expansion.coeff(r,2*k)==coefficient and coefficient>0)
    # L = log(2^16/3^9) < 3/2 follows without floating-point logarithms.
    check('uniform_flip_exact_bound',sp.Rational(29,8)>sp.Rational(2**16,3**9))

    rng=np.random.default_rng(20260921)
    matrix=np.diag([-.5,.25,.25])
    v=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    unitary=np.linalg.qr(v)[0]
    rotated=unitary@matrix@unitary.conj().T
    check('conjugation_preserves_cubic',
          abs(np.trace(rotated@rotated@rotated)-np.trace(matrix@matrix@matrix))<1e-14)

    beta_flip=-1/(64*mp.pi**2)
    cases=[]
    for r_text in ['0.01','0.1','0.25','0.49']:
        s=mp.mpf(1); e=mp.mpf(r_text)
        delta=gaussian(s,e,1)
        critical=-s*delta/(12*e**3)
        lower=-s*derivative_c(s,e,-1)/(6*e**3)
        upper=-s*derivative_c(s,e,1)/(6*e**3)
        check('coexistence_between_spinodals_'+r_text,lower<critical<upper<0)
        check('fixed_beta_flips_order_'+r_text,beta_flip<critical<0)
        for c_text in ['-1','-0.3','0.2','1']:
            c=mp.mpf(c_text)
            near('independent_log_integral_'+r_text+'_'+c_text,
                 gaussian(s,e,c),relative_integral(s,e,c),mp.mpf('1e-52'))
            check('strict_concavity_'+r_text+'_'+c_text,second_c(s,e,c)<0)
        # Independent derivative of the eigenvalue/log expression.
        near('slope_from_log_'+r_text,
             mp.diff(lambda cc:gaussian(s,e,cc),mp.mpf('.2')),derivative_c(s,e,mp.mpf('.2')))
        near('curvature_from_log_'+r_text,
             mp.diff(lambda cc:gaussian(s,e,cc),mp.mpf('.2'),2),second_c(s,e,mp.mpf('.2')))
        Q=6*e**3/s
        mu_minus=derivative_c(s,e,-1)
        mu_plus=-(derivative_c(s,e,1)+beta_flip*Q)
        check('both_example_vacua_normal_stable_'+r_text,mu_minus>0 and mu_plus>0)
        near('direct_theta_curvature_minus_'+r_text,
             mp.diff(lambda theta:total_theta(s,e,theta,0),mp.pi,2),mu_minus)
        near('direct_theta_curvature_plus_'+r_text,
             mp.diff(lambda theta:total_theta(s,e,theta,beta_flip),mp.mpf(0),2),mu_plus)
        check('flipped_global_endpoint_gap_'+r_text,
              relative_total(s,e,1,beta_flip)<relative_total(s,e,-1,beta_flip))
        near('critical_degeneracy_'+r_text,relative_total(s,e,1,critical),mp.mpf(0))
        # All intermediate c are above equal endpoints by strict concavity.
        check('coexistence_barrier_'+r_text,relative_total(s,e,mp.mpf(0),critical)>0)
        recovered_from_curvature=s*(-mu_plus-derivative_c(s,e,1))/(6*e**3)
        near('curvature_reverse_map_'+r_text,recovered_from_curvature,beta_flip)
        cases.append({
            'epsilon_over_s':r_text,'beta_critical':str(critical),
            'beta_minus_spinodal':str(lower),'beta_plus_spinodal':str(upper),
            'beta_zero_gap_plus_minus':str(delta),
            'beta_flip_gap_plus_minus':str(relative_total(s,e,1,beta_flip)),
            'beta_zero_minus_curvature':str(mu_minus),'beta_flip_plus_curvature':str(mu_plus),
            'coexistence_barrier_c_zero':str(relative_total(s,e,0,critical))
        })

    # Independent s differentiation of the full potential, including matching.
    s,e,c=mp.mpf(1),mp.mpf('.25'),mp.mpf('.2')
    log_ratio=mp.log(determinant(0,s,e,c)/determinant(0,s,e,-1))
    relative_gauge=-log_ratio/(96*mp.pi**2) # independent threshold formula, T=1/2
    near('Gaussian_common_response_before_deformation',
         mp.diff(lambda ss:gaussian(ss,e,c),s,2),log_ratio/(16*mp.pi**2))
    total_ss=mp.diff(lambda ss:relative_total(ss,e,c,beta_flip),s,2)
    response_residual=total_ss+6*relative_gauge
    expected=12*beta_flip*e**3*(c+1)/s**3
    near('independent_common_response_residual',response_residual,expected)
    near('response_reverse_map',s**3*response_residual/(12*e**3*(c+1)),beta_flip)
    near('reference_state_is_blind',relative_total(s,e,-1,beta_flip),mp.mpf(0))
    check('same_moments_opposite_order',all(abs(a-b)<mp.mpf('1e-60') for a,b in [
        (sum(spectrum(s,e,-1)),sum(spectrum(s,e,1))),
        (sum(x*x for x in spectrum(s,e,-1)),sum(x*x for x in spectrum(s,e,1)))]))

    sources=[next((ROOT/'paper/06_QFT_재설계').glob(prefix+'*.md'))
             for prefix in ['81_','86_','88_']]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    output={
        'scope':'CE-IS1 original three-channel Gaussian shape sector; finite matching freedom',
        'all_checks_passed':True,'checks':checks,'source_sha256':sha(Path(__file__)),
        'premise_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        'beta_flip_exact':'-1/(64*pi^2)','beta_flip_decimal':str(beta_flip),
        'critical_limits':{
            'small_split':str(-1/(96*mp.pi**2)),
            'gap_closing_limit':str(-(16*mp.log(2)-9*mp.log(3))/(96*mp.pi**2))},
        'leading_series':str(expansion),'cases':cases,
        'example_spectrum_minus':[str(x) for x in sorted(spectrum(s,e,-1))],
        'example_spectrum_plus':[str(x) for x in sorted(spectrum(s,e,1))],
        'response_residual_example':str(response_residual),
        'full_goal_complete':False,
        'not_claimed':['counterexample to the original no-added-potential Gaussian branch',
                       'choice of beta from nature','radial stabilization',
                       'isolated minimum along conjugation orbits','UV completion',
                       'full quantum effective action','observational validation']
    }
    (HERE/'results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['all_checks_passed','beta_flip_decimal',
                                         'critical_limits','example_spectrum_minus',
                                         'example_spectrum_plus']},indent=2))


if __name__=='__main__':
    main()
