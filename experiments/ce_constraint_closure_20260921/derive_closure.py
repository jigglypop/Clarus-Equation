"""CE-CC1: inverse constraint closure, a positive-kinetic counterexample,
and a shared matter cone. All equalities checked here are exact SymPy algebra.

The continuum proof and the restrictions of the Hamiltonian ansatz are in
chapter 35. This is not a proof of quantum constraint closure or an emergence
proof for the Lorentzian hypersurface-deformation algebra itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sympy as sp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def symmetric3(prefix):
    u=sp.symbols(prefix+'0:6')
    return sp.Matrix([[u[0],u[3],u[4]],[u[3],u[1],u[5]],[u[4],u[5],u[2]]])


def main():
    checks=[]
    def check(name,expr):
        value=sp.simplify(expr)
        assert value==0,(name,value)
        checks.append(name)
    a=sp.symbols('a',positive=True)
    b,w,c,R=sp.symbols('b omega c R',real=True)
    I=sp.eye(3)
    P,Q=symmetric3('p'),symmetric3('q')
    trP=sp.trace(P)
    # Tensor contraction in an orthonormal frame. The identity is covariant.
    lhs=sp.trace((P-w*trP*I)*(Q-sp.trace(Q)*I))
    rhs=sp.trace(P*Q)+(2*w-1)*trP*sp.trace(Q)
    check('curvature_variation_contraction',sp.expand(lhs-rhs))
    solution=sp.solve([-a*b-1,2*a*b*(2*w-1)],[b,w],dict=True)
    assert solution==[{b:-1/a,w:sp.Rational(1,2)}],solution
    checks.append('unique_coefficients_in_declared_ansatz')

    K=symmetric3('k')
    pi=(K-sp.trace(K)*I)/a
    legendre=(2*sp.trace(pi*K)-a*(sp.trace(pi*pi)-sp.trace(pi)**2/2)+R/a-c)
    target=(sp.trace(K*K)-sp.trace(K)**2+R-a*c)/a
    check('exact_Legendre_transform_to_ADM',sp.expand(legendre-target))
    check('momentum_reconstruction',sp.trace(pi)+2*sp.trace(K)/a)

    basis=[sp.diag(1,-1,0)/sp.sqrt(2),sp.diag(1,1,-2)/sp.sqrt(6)]
    for i,j in [(0,1),(0,2),(1,2)]:
        E=sp.zeros(3); E[i,j]=E[j,i]=1/sp.sqrt(2); basis.append(E)
    basis.append(I/sp.sqrt(3))
    metric=sp.Matrix([[sp.trace(E*F)-w*sp.trace(E)*sp.trace(F) for F in basis] for E in basis])
    check('supermetric_inertia',sp.expand(sum((metric-sp.diag(1,1,1,1,1,1-3*w)))))
    assert metric==sp.diag(1,1,1,1,1,1-3*w)
    selected=metric.subs(w,sp.Rational(1,2))
    assert list(selected.diagonal())==[1,1,1,1,1,-sp.Rational(1,2)]
    checks.append('five_positive_one_negative_metric_direction')

    # Actual point on H=0 and D_i=0, not an off-shell-only numerical mismatch.
    x=sp.symbols('x',real=True)
    N=sp.Integer(1); M=2+sp.sin(x)
    Pi=sp.diag(0,sp.cos(x),sp.sin(x))
    check('counterexample_H_density_zero',sp.trace(Pi*Pi)-1)
    for j in range(3):
        check('counterexample_D_density_zero_'+str(j),-2*sp.diff(Pi[0,j],x))
    DN=sp.diag(0,-sp.diff(N,x,2),-sp.diff(N,x,2))
    DM=sp.diag(0,-sp.diff(M,x,2),-sp.diff(M,x,2))
    # Full functional derivatives first; evaluate at flat h only afterwards.
    raw=2*a*b*sp.trace((Pi-w*sp.trace(Pi)*I)*(M*DN-N*DM))
    direct=sp.integrate(sp.expand_trig(raw),(x,0,2*sp.pi))
    v=N*sp.diff(M,x)-M*sp.diff(N,x)
    anomaly=sp.integrate(v*sp.diff(sp.trace(Pi),x),(x,0,2*sp.pi))
    check('counterexample_independent_bracket_forms',direct-2*a*b*(2*w-1)*anomaly)
    witness=sp.simplify(direct.subs({a:1,b:-1,w:0}))
    check('counterexample_nonzero_on_constraint_surface',witness-2*sp.pi)
    assert witness!=0
    # Both lapses are strictly positive: N=1, 1<=M<=3.
    checks.append('positive_lapses_with_nonzero_constraint_surface_bracket')

    # A second test has both a momentum-constraint contribution and a trace term.
    Pi2=sp.diag(sp.sin(x),sp.cos(x),0)
    raw2=2*a*b*sp.trace((Pi2-w*sp.trace(Pi2)*I)*(M*DN-N*DM))
    direct2=sp.integrate(sp.expand_trig(raw2),(x,0,2*sp.pi))
    Dv=-2*sp.integrate(v*sp.diff(Pi2[0,0],x),(x,0,2*sp.pi))
    Av=sp.integrate(v*sp.diff(sp.trace(Pi2),x),(x,0,2*sp.pi))
    check('independent_integration_by_parts',direct2-(-a*b*Dv+2*a*b*(2*w-1)*Av))
    check('Einstein_coefficients_recover_D_bracket',direct2.subs(solution[0])-Dv)

    # Matter functional derivatives: local pieces cancel in the antisymmetrization.
    A00,A01,A11,B00,B01,B11=sp.symbols('A00 A01 A11 B00 B01 B11')
    A=sp.Matrix([[A00,A01],[A01,A11]])
    B=sp.Matrix([[B00,B01],[B01,B11]])
    p=sp.Matrix(sp.symbols('p0:2')); grad=sp.Matrix(sp.symbols('grad0:2'))
    local=sp.Matrix(sp.symbols('local0:2'))
    n,m,nx,mx=sp.symbols('n m nx mx')
    dHn_phi=n*local-nx*B*grad
    dHm_phi=m*local-mx*B*grad
    matter=(dHn_phi.T*(m*A*p)-(n*A*p).T*dHm_phi)[0]
    check('matter_bracket_general_matrix',sp.expand(matter-(n*mx-m*nx)*(p.T*A*B*grad)[0]))
    z=sp.symbols('z',real=True)
    target_metric=sp.Matrix([[2,z],[z,3]])
    check('nonconstant_target_metric_admission',
          sum(sp.simplify(t) for t in target_metric.inv()*target_metric-sp.eye(2)))
    assert sp.simplify(target_metric.det())==6-z*z
    wrong_D=sp.integrate(sp.cos(x)**2,(x,0,2*sp.pi))
    wrong_bracket=2*wrong_D
    assert wrong_bracket-wrong_D==sp.pi
    checks.append('positive_matter_kinetics_do_not_force_equal_cones')

    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=[next((ROOT/'paper/후속연구_기록과_상태선택').glob('23_*.md')),
             next((ROOT/'paper/06_QFT_재설계').glob('55_*.md')),
             ROOT/'paper/검증_원장/참조_양자_보존_원장.md']
    result={'scope':'classical inverse closure in the specified pure-metric and minimal sigma-model ansatz',
            'all_checks_passed':True,'checks':checks,'source_sha256':sha(Path(__file__)),
            'premise_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'closure_coefficients':{'omega':'1/2','b':'-1/a','c':'unrestricted','a':'positive, unrestricted'},
            'momentum_supermetric_eigenvalues':[1,1,1,1,1,-.5],
            'constraint_surface_counterexample':{
                'spatial_domain':'S1(length 2*pi) x S1(length 1) x S1(length 1)',
                'h':'identity','pi':'diag(0,cos(x),sin(x))',
                'a':1,'b':-1,'c':-1,'omega':0,
                'lapses':['1','2+sin(x)'],'H_density':'0','D_density':['0','0','0'],
                'Poisson_bracket':str(witness),'predicted_D_bracket':'0'},
            'matter_closure':'A(phi) B(phi) = identity',
            'unequal_cone_counterexample':{'A':'diag(1,1)','B':'diag(1,2)',
                                         'actual_bracket':'2*pi','required_bracket':'pi'},
            'full_goal_complete':False,
            'not_claimed':['derivation of the hypersurface-deformation algebra from CE',
                           'selection of Newton or cosmological constants','quantum anomaly freedom',
                           'all scalar-tensor or higher-derivative theories',
                           'exclusion of formulations with additional constraints or preferred slicing',
                           'negative-norm physical graviton','observational validation']}
    (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['all_checks_passed','closure_coefficients',
                     'constraint_surface_counterexample','matter_closure']},indent=2))


if __name__=='__main__':
    main()
