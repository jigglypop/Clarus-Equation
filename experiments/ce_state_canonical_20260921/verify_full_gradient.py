"""Independent exact full-projective gradient evaluation, including entangled normals."""
from pathlib import Path
import hashlib
import json
import sympy as s

HERE = Path(__file__).resolve().parent
x,y,z=s.symbols("x y z",real=True)
coords=(x,y,z)
vectors=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1)]
u=s.Matrix([s.sqrt(10)/4]+[term for a,b,c in vectors
             for term in (s.cos(a*x+b*y+c*z)/4,s.sin(a*x+b*y+c*z)/4)])
v=s.Matrix([s.sqrt(3)/2,s.exp(s.I*(x+y))/2])
psi=s.kronecker_product(u,v)
first=[psi.diff(x),psi.diff(y)]
second=[psi.diff(x,2),psi.diff(y,2)]
records=[]
for point in [(0,0,0),(s.pi/4,0,0),(s.pi/2,s.pi/2,0)]:
    sub=dict(zip(coords,point))
    at=lambda mat:mat.subs(sub).applyfunc(lambda a:s.simplify(s.expand_complex(a)))
    state=at(psi)
    d=[at(a) for a in first]
    dd=[at(a) for a in second]
    assert s.simplify((state.conjugate().T*state)[0])==1
    Q=s.eye(26)-state*state.conjugate().T
    horizontal=[Q*a for a in d]
    connection=[(state.conjugate().T*a)[0] for a in d]
    tension=[Q*(dd[i]-2*connection[i]*d[i]) for i in range(2)]
    f=s.sin(point[0]); fx=s.cos(point[0]); g=s.cos(point[0])
    grad_f=-(f*tension[0]+fx*horizontal[0])
    grad_g=-g*tension[1]
    # ell=hbar=1. No restriction to product-state tangent vectors here.
    actual=s.simplify(s.expand_complex(2*s.im((grad_f.conjugate().T*grad_g)[0])))
    expected=3*s.cos(point[0])**2/16
    assert s.simplify(actual-expected)==0,(point,actual,expected)
    records.append({"point":list(map(str,point)),"actual_density":str(actual),
                    "expected_density":str(expected)})
    print("full 26-component gradient passed at",point,flush=True)
result={"scope":"independent full-projective gradient samples, exact arithmetic, ell=hbar=1",
        "all_checks_passed":True,"records":records,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "continuum_integral_proof":"chapter 36; samples alone do not prove the functional identity",
        "full_goal_complete":False}
(HERE/"full_gradient_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
