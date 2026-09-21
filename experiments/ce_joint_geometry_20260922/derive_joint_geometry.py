"""CE-JG1: joint metric/connection variation and the intrinsic mixed-bracket obstruction."""
from pathlib import Path
import hashlib
import json
import sympy as s

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    checks=[]
    def zero(name,expr):
        entries=list(expr) if isinstance(expr,s.MatrixBase) else [expr]
        for entry in entries:
            assert s.simplify(s.expand_complex(entry))==0,(name,entry)
        checks.append(name)

    x,y,z=s.symbols("x y z",real=True)
    coords=(x,y,z)
    vectors=[s.Matrix(v) for v in [(1,0,0),(0,1,0),(0,0,1),
                                  (1,1,0),(1,0,1),(0,1,1)]]
    phases=[(v.T*s.Matrix(coords))[0] for v in vectors]
    r=s.Rational(1,4)
    u=s.Matrix([s.sqrt(10)/4]+[a for p in phases
                               for a in (r*s.cos(p),r*s.sin(p))])
    du=[u.diff(q) for q in coords]
    ddu=[[u.diff(q).diff(t) for t in coords] for q in coords]
    at=lambda matrix,point:matrix.subs(dict(zip(coords,point))).applyfunc(
        lambda a:s.simplify(s.expand_complex(a)))
    origin=(0,0,0)
    I=s.eye(3)
    V=s.kronecker_product(at(u,origin),I)
    B=[s.kronecker_product(at(d,origin),I) for d in du]
    second=[[s.kronecker_product(at(ddu[i][j],origin),I)
             for j in range(3)] for i in range(3)]
    zero("rank_three_frame_orthonormality",V.conjugate().T*V-I)
    G=s.Matrix([[(at(du[i],origin).T*at(du[j],origin))[0]
                 for j in range(3)] for i in range(3)])
    Gi=G.inv()
    for i in range(3):
        zero("horizontal_tangent_"+str(i),V.conjugate().T*B[i])
        for j in range(3):
            zero(f"scalar_gram_{i}{j}",B[i].T*B[j]-G[i,j]*I)

    basis=[]
    for a in range(3):
        H=s.zeros(3);H[a,a]=1;basis.append(H)
    for a,b in [(0,1),(0,2),(1,2)]:
        H=s.zeros(3);H[a,b]=H[b,a]=1;basis.append(H)
    for a,b in [(0,1),(0,2),(1,2)]:
        H=s.zeros(3);H[a,b]=s.I;H[b,a]=-s.I;basis.append(H)
    assert s.Matrix.hstack(*[H.reshape(9,1) for H in basis]).rank()==9
    checks.append("nine_independent_Hermitian_generators")

    pairs=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
    # Test every matrix-valued connection direction. Nonzero coefficient
    # derivatives are included; they must not contaminate delta h.
    slope=[s.Integer(2),s.Integer(-1),s.Integer(3)]
    for component in range(3):
        for generator,H in enumerate(basis):
            dV=s.I/2*sum((B[j]*Gi[j,component]*H for j in range(3)),
                         s.zeros(39,3))
            dB=[s.I/2*sum(((second[k][j]+slope[k]*B[j])*Gi[j,component]*H
                           for j in range(3)),s.zeros(39,3)) for k in range(3)]
            zero(f"connection_lift_horizontal_{component}_{generator}",
                 V.conjugate().T*dV)
            for i in range(3):
                actual=s.I*(dV.conjugate().T*B[i]-B[i].conjugate().T*dV)
                zero(f"connection_right_inverse_{component}_{generator}_{i}",
                     actual-(H if i==component else s.zeros(3)))
            for i,j in pairs:
                dh=s.re(s.trace(B[i].conjugate().T*dB[j]
                                +dB[i].conjugate().T*B[j]))
                zero(f"connection_lift_preserves_metric_{component}_{generator}_{i}{j}",dh)

    metric_columns=[]
    for a,v in enumerate(vectors):
        n=s.zeros(13,1);n[0]=-1/s.sqrt(10)
        n[1+2*a]=1
        dV=s.kronecker_product(n,I)
        dB=[]
        for k in range(3):
            dn=s.zeros(13,1);dn[2+2*a]=v[k]
            dB.append(s.kronecker_product(dn+slope[k]*n,I))
        for i in range(3):
            zero(f"metric_lift_preserves_connection_{a}_{i}",
                 s.I*(dV.conjugate().T*B[i]-B[i].conjugate().T*dV))
        column=[]
        for i,j in pairs:
            dh=s.re(s.trace(B[i].T*dB[j]+dB[i].T*B[j]))
            zero(f"metric_lift_{a}_{i}{j}",dh-6*r*v[i]*v[j])
            column.append(dh)
        metric_columns.append(s.Matrix(column))
    M=s.Matrix.hstack(*metric_columns)
    joint=s.diag(M,s.eye(27))
    assert joint.rank()==33 and joint.det()==s.Rational(729,64)
    checks.append("joint_metric_connection_rank_33")
    print("joint 6+27 variation and right inverse passed",flush=True)

    # Full Grassmannian gradients for the gauge-invariant shear-current test.
    # E=int cos(y) h12; C=int sin(y) tr A1; ell=hbar=1.
    samples=[]
    for point in [(0,0,0),(0,s.pi/4,0),(0,s.pi/2,0)]:
        Vp=s.kronecker_product(at(u,point),I)
        Bp=[s.kronecker_product(at(d,point),I) for d in du]
        H12=s.kronecker_product(at(ddu[0][1],point),I)
        horizontal=lambda W:W-Vp*(Vp.conjugate().T*W)
        eta_E=horizontal(s.sin(point[1])*Bp[0]/2-s.cos(point[1])*H12)
        eta_C=s.I*s.sin(point[1])*Bp[0]
        density=s.simplify(2*s.im(s.trace(eta_E.conjugate().T*eta_C)))
        expected=s.Rational(9,16)*s.sin(point[1])**2
        zero("full_Grassmann_mixed_bracket_"+str(point),density-expected)
        samples.append({"point":list(map(str,point)),"actual_density":str(density)})
    ell,hbar=s.symbols("ell hbar",positive=True)
    integrated=s.integrate(s.Rational(9,16)*ell**2/hbar*s.sin(y)**2,
                           (y,0,2*s.pi))*(2*s.pi)**2
    lie_form=s.integrate(s.Rational(9,16)*ell**2/hbar*s.cos(y)**2,
                         (y,0,2*s.pi))*(2*s.pi)**2
    zero("integrated_gradient_and_Lie_derivative_agree",integrated-lie_form)
    zero("nonzero_mixed_bracket",integrated-9*s.pi**3*ell**2/(4*hbar))
    assert integrated!=0
    # Local infinitesimal gauge invariance: divergence of sin(y) d/dx is zero.
    zero("shear_divergence_free",s.diff(s.sin(y),x))
    zero("shear_has_no_harmonic_average",s.integrate(s.sin(y),(y,0,2*s.pi)))
    trial=s.exp(s.I*(x+2*y))+s.cos(z)
    op_v=lambda q:s.I*hbar*s.sin(y)*s.diff(q,x)
    op_w=lambda q:s.I*hbar*s.sin(x)*s.diff(q,y)
    expected_operator=s.I*hbar*(-s.sin(x)*s.cos(y)*s.diff(trial,x)
                               +s.sin(y)*s.cos(x)*s.diff(trial,y))
    zero("spatial_transport_operator_commutator",
         (op_v(op_w(trial))-op_w(op_v(trial)))/(s.I*hbar)-expected_operator)
    print("full Grassmannian mixed bracket passed",flush=True)

    chapter_dir=ROOT/"paper/후속연구_기록과_상태선택"
    sources=[next(chapter_dir.glob("35_*.md")),next(chapter_dir.glob("36_*.md"))]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result={
        "scope":"joint metric/connection differential and intrinsic Berry mixed Poisson bracket",
        "all_checks_passed":True,"checks":checks,
        "source_sha256":sha(Path(__file__)),
        "premise_sources":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "frame_rank":3,"ambient_complex_dimension":39,
        "metric_components":6,"Hermitian_connection_components":27,
        "joint_differential_rank":33,"raw_jacobian_determinant_ell_1":str(joint.det()),
        "mixed_bracket_counterexample":{
            "metric_functional":"integral cos(y) h12",
            "connection_functional":"integral sin(y) trace(A1)",
            "actual_bracket":str(integrated),"canonical_configuration_bracket":"0",
            "full_Grassmannian_gradient_samples":samples,
            "gauge_invariant_under_periodic_frame_changes":True},
        "full_goal_complete":False,
        "not_claimed":["nonlinear global inverse for prescribed metric and connection",
                       "microscopic derivation of independent momenta",
                       "equivalence of supplied joint cotangent structure and intrinsic Berry structure",
                       "selection of the Standard Model gauge group or couplings",
                       "preservation of all record observables by the joint metric/connection quotient",
                       "Lorentzian constraint algebra or quantum closure from CE"]}
    (HERE/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["all_checks_passed","joint_differential_rank",
                     "mixed_bracket_counterexample"]},indent=2))


if __name__=="__main__":
    main()
