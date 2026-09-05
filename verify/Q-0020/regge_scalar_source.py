"""동일 레게 단체의 스칼라 원천·정준 반작용·끝점 응답을 검산한다.

양의 유클리드 선형 보간 작용과 경계 장 준비를 공급한다. 이 작용값이나
스칼라 이동 전하를 물리 에너지로 해석하지 않으며 로런츠 시간은 미유도다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import sici

import regge_coalescing_kernel as geometry
import regge_constraint_reduction as reduction
import regge_postconstraint_projection as projected
import regge_pachner_constraints as moves

L = geometry.limit(1.)
N = len(moves.FINAL.edges)
E, Y = moves.E_ID, moves.Y_ID
C = math.sqrt(3)/9
LOCAL_PAIRS = tuple((i,j) for i in range(5) for j in range(i+1,5))


def scalar_data(lengths, fields, cells=None):
    """각 단체의 그램·부피에서 작용, 길이 힘, 장 운동량을 직접 조립한다."""
    q, phi = np.asarray(lengths,dtype=float), np.asarray(fields,dtype=float)
    if q.shape != (N,) or phi.shape != (6,) or not np.all(np.isfinite(phi)):
        raise ValueError("길이 15개와 유한한 꼭짓점 장 6개가 필요하다")
    if not np.all(np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("길이는 유한한 양수여야 한다")
    cells = moves.FINAL.cells if cells is None else tuple(cells)
    stiffness, gradient = np.zeros((6,6)), np.zeros(N)
    minimum, volumes = math.inf, []
    difference = np.column_stack((-np.ones(4),np.eye(4)))
    for cell in cells:
        ids = moves.FINAL.indices(tuple((cell[i],cell[j]) for i,j in LOCAL_PAIRS))
        squared = np.zeros((5,5))
        for (i,j), index in zip(LOCAL_PAIRS,ids):
            squared[i,j] = squared[j,i] = q[index]**2
        gram = (squared[0,1:,None]+squared[None,0,1:]-squared[1:,1:])/2
        smallest = float(np.linalg.eigvalsh(gram)[0])
        minimum = min(minimum,smallest)
        if smallest <= 0:
            raise ValueError("모든 단체의 그램 행렬이 양의 정부호여야 한다")
        inverse = np.linalg.inv(gram)
        volume = math.sqrt(float(np.linalg.det(gram)))/24
        volumes.append(volume)
        local_phi = phi[list(cell)]
        delta = difference @ local_phi
        w = inverse @ delta
        norm = float(delta @ w)
        stiffness[np.ix_(cell,cell)] += volume*difference.T @ inverse @ difference
        for (i,j), index in zip(LOCAL_PAIRS,ids):
            ds = np.zeros((5,5))
            ds[i,j] = ds[j,i] = 2*q[index]
            dg = (ds[0,1:,None]+ds[None,0,1:]-ds[1:,1:])/2
            gradient[index] += .5*volume*(.5*np.trace(inverse@dg)*norm-w@dg@w)
    return {"action":float(.5*phi@stiffness@phi), "gradient":gradient,
            "field_momentum":stiffness@phi, "stiffness":stiffness,
            "minimum_gram":minimum, "volumes":volumes}


def modes(v, mean=0.):
    return np.array([mean-v/2,mean+v/2,0.,0.,0.,0.])


def weights(edge):
    edge=np.asarray(edge,dtype=float)
    if not np.all(np.isfinite(edge)) or np.any((edge<=0)|(edge>=L)):
        raise ValueError("열린 그램 구간 안의 유한 길이가 필요하다")
    r=np.sqrt((L-edge)*(L+edge))
    return C*r/edge, 4*C*edge/r


def weight_force(edge, ratio=0.):
    """d'+ratio² b'의 임계 상쇄를 인수분해해 계산한다."""
    edge=np.asarray(edge,dtype=float)
    weights(edge)
    r=np.sqrt((L-edge)*(L+edge))
    if abs(ratio) == 1/6:
        numerator=(10/9)*(edge-2)*(edge+2)
    else:
        numerator=(1+4*ratio**2)*edge**2-L**2
    return C*L**2*numerator/(r**3*edge**2)


def stationary_force(edge, eta, ratio):
    return geometry.gradient(edge,1.)+.5*eta**2*weight_force(edge,ratio)


def roots(eta, ratio=0.):
    if not np.isfinite(eta) or eta<=0 or not np.isfinite(ratio) or abs(ratio)>=1/6:
        raise ValueError("양의 eta와 열린 원천 원뿔의 고정 비율이 필요하다")
    scale=math.sqrt(5*(1-36*ratio**2))/36*eta
    return np.array([brentq(lambda e:stationary_force(e,eta,ratio),2-3*scale,2,xtol=5e-15),
                     brentq(lambda e:stationary_force(e,eta,ratio),2,2+3*scale,xtol=5e-15)])


def derivative(function, q, index, step=1e-4):
    offset=np.eye(len(q))[index]
    coarse=(function(q+step*offset)-function(q-step*offset))/(2*step)
    fine=(function(q+step*offset/2)-function(q-step*offset/2))/step
    return (4*fine-coarse)/3


def canonical_check(q, phi, beta=1., kappa=1.):
    """중력·물질의 각 증분을 먼저 조립하고 전체 경계 운동량과 대조한다."""
    grav=moves.actions(q)
    cells={"old":moves.MOVE.old.cells,"middle":moves.MOVE.new.cells,
           "final":moves.FINAL.cells,"first":((0,1,2,3,4),),
           "second":(moves.FUTURE_CELL,)}
    matter={key:scalar_data(q,phi,value) for key,value in cells.items()}
    totals={}
    for key in cells:
        gg=grav[key]["gradient"]
        if key=="old":
            full=np.zeros(N)
            full[moves.OLD_IDS]=gg
            gg=full
        totals[key]={"action":beta*grav[key]["action"]+kappa*matter[key]["action"],
                     "momentum":np.r_[beta*gg+kappa*matter[key]["gradient"],
                                       kappa*matter[key]["field_momentum"]]}
    middle=totals["old"]["momentum"]+totals["first"]["momentum"]
    final=middle+totals["second"]["momentum"]
    c=middle[Y]-totals["first"]["momentum"][Y]
    constraint=final[E]
    errors=[]
    for before,after,change in (("old","middle","first"),("middle","final","second")):
        errors.extend([abs(totals[after]["action"]-totals[before]["action"]-totals[change]["action"]),
                       float(np.max(np.abs(totals[after]["momentum"]-totals[before]["momentum"]-totals[change]["momentum"])))])
    return {"composition_residual":max(errors),"c_residual":float(abs(c)),
            "F_residual":float(abs(constraint)),
            "final_momentum_residual":float(np.max(np.abs(final-totals["final"]["momentum"]))),
            "shift_charge_residual":float(abs(sum(final[N:]))),
            "scalar_action":matter["final"]["action"],
            "field_momentum_norm":float(np.linalg.norm(final[N:])),
            "scalar_geometry_force_norm":float(np.linalg.norm(kappa*matter["final"]["gradient"]))}


def branch_check(eta, ratio=0., beta=1., kappa=1.):
    v=eta*math.sqrt(beta/kappa)
    phi=modes(v,ratio*v)
    edges=roots(eta,ratio)
    expected=5*(1-36*ratio**2)/18
    center=-5/1728-85*ratio**2/144
    out=[]
    for edge in edges:
        q=reduction.flat_lengths(1.)
        q[E]=edge
        matter=scalar_data(q,phi)
        d,b=weights(edge)
        added=moves.MOVE.new.cells[-1:]+(moves.FUTURE_CELL,)
        mixed=derivative(lambda x:scalar_data(x,phi,added)["gradient"][Y],q,E)
        bracket=-beta*float(projected.symmetric_g_e(edge,1.))-kappa*mixed
        direct=-derivative(lambda x:beta*(moves.actions(x)["first"]["gradient"][Y]+
                                            moves.actions(x)["second"]["gradient"][Y])+
                                       kappa*scalar_data(x,phi,added)["gradient"][Y],q,E)
        row=canonical_check(q,phi,beta,kappa)
        row.update({"edge":float(edge),"a_total":bracket,
                    "a_squared_ratio":bracket**2/(beta*kappa*v*v),
                    "independent_bracket_error":abs(bracket-direct),
                    "closed_action_error":abs(matter["action"]-.5*(d*v*v+b*(ratio*v)**2)),
                    "closed_force_error":abs(matter["gradient"][E]-.5*v*v*weight_force(edge,ratio)),
                    "minimum_gram":matter["minimum_gram"]})
        out.append(row)
    prediction=2+np.array([-1,1])*math.sqrt(5*(1-36*ratio**2))/36*eta+center*eta**2
    return {"eta":eta,"ratio":ratio,"beta":beta,"kappa":kappa,"expected_square_ratio":expected,
            "branches":out,"square_ratio_error":max(abs(r["a_squared_ratio"]-expected) for r in out),
            "branch_remainder":float(np.max(np.abs(edges-prediction))),
            "scaled_remainder":float(np.max(np.abs(edges-prediction))/eta**3),
            "center_coefficient":float((edges.mean()-2)/eta**2),"expected_center":center}


def threshold_check(eta):
    other=brentq(lambda e:stationary_force(e,eta,1/6)/(e-2),
                  2-.08*eta**2,2-.01*eta**2,xtol=5e-15)
    return {"eta":eta,"flat_residual":float(stationary_force(2.,eta,1/6)),
            "other_edge":other,"scaled_other_shift":(other-2)/eta**2,
            "coefficient_error":abs((other-2)/eta**2+25/648)}


def geometry_check(offset):
    q=reduction.flat_lengths(1.)
    q[E]+=offset[0]
    q[Y]+=offset[1]
    q[2]+=offset[2]
    phi=np.array([-.31,.47,.12,-.08,.16,-.05])
    data=scalar_data(q,phi)
    fd=np.array([derivative(lambda x:scalar_data(x,phi)["action"],q,i,2e-4) for i in range(N)])
    mixed=[]
    for i in (E,Y,2):
        left=derivative(lambda x:scalar_data(x,phi)["field_momentum"],q,i,2e-4)
        right=np.array([derivative(lambda f:scalar_data(q,f)["gradient"][i],phi,k,1e-4) for k in range(6)])
        mixed.append(float(np.max(np.abs(left-right))))
    direction=np.linspace(-.2,.3,N+6)
    joined=np.r_[q,phi]
    total_gradient=np.r_[data["gradient"],data["field_momentum"]]
    step=1e-4
    plus,minus=joined+step*direction,joined-step*direction
    work=(scalar_data(plus[:N],plus[N:])["action"]-scalar_data(minus[:N],minus[N:])["action"])/(2*step)
    eig=np.linalg.eigvalsh(data["stiffness"])
    return {"offset":offset,"length_gradient_error":float(np.max(np.abs(fd-data["gradient"]))),
            "mixed_reciprocity_error":max(mixed),"action_differential_error":abs(work-total_gradient@direction),
            "scale_identity_error":abs(q@data["gradient"]-2*data["action"]),
            "field_euler_error":abs(phi@data["field_momentum"]-2*data["action"]),
            "constant_mode_error":float(np.max(np.abs(data["stiffness"]@np.ones(6)))),
            "eigenvalues":eig.tolist(),"shift_charge":float(sum(data["field_momentum"])),
            "canonical":canonical_check(q,phi),"minimum_gram":data["minimum_gram"]}


def endpoint_difference(kappa, beta=1., v=1., tolerance=1e-10):
    """순수 차이 모드의 전체 진폭 차이에서 c/e 특이항을 해석적으로 뺀다."""
    if not np.isfinite(kappa) or kappa<=0:
        raise ValueError("끝점 로그 검사는 양의 유한 결합에서 수행한다")
    c=math.sqrt(3)*L*v*v/18
    a=kappa*c
    x=a/L
    si,ci=sici(x)
    singular=L*np.expm1(1j*x)+a*(-(math.pi/2-si)-1j*ci)
    r0=float(geometry.action(0.,1.))
    phase0=np.exp(1j*beta*r0)/L
    def regular(angle):
        edge=L*math.sin(angle)
        leading=c/edge
        correction=-(c/L)*math.tan(angle/2)
        change=beta*(float(geometry.action(edge,1.))-r0)
        phase=np.exp(1j*change)
        value=(np.expm1(1j*change)*np.expm1(1j*kappa*leading)
               +phase*np.exp(1j*kappa*leading)*np.expm1(1j*kappa*correction))
        return phase0*value*L*math.cos(angle)
    real,er=quad(lambda t:regular(t).real,0,math.pi/2,epsabs=tolerance,epsrel=tolerance,limit=300)
    imag,ei=quad(lambda t:regular(t).imag,0,math.pi/2,epsabs=tolerance,epsrel=tolerance,limit=300)
    return phase0*singular+complex(real,imag),er+ei


def exact_certificate():
    import sympy as sp
    e=sp.Symbol("e",positive=True)
    m,v=sp.symbols("m v",real=True)
    gram=sp.Matrix([[e**2,e**2/2,e**2/2,e**2/2],
                    [e**2/2,2,sp.Rational(2,3),sp.Rational(2,3)],
                    [e**2/2,sp.Rational(2,3),2,sp.Rational(2,3)],
                    [e**2/2,sp.Rational(2,3),sp.Rational(2,3),2]])
    delta=sp.Matrix([v,-m+v/2,-m+v/2,-m+v/2])
    r2=sp.Rational(40,9)-e**2
    norm=sp.factor((delta.T*gram.inv()*delta)[0])
    d=sp.sqrt(3)*sp.sqrt(r2)/(9*e)
    b=4*sp.sqrt(3)*e/(9*sp.sqrt(r2))
    return {"gram_determinant":str(sp.factor(gram.det())),
            "determinant_identity":str(sp.simplify(gram.det()-sp.Rational(4,3)*e**2*r2)),
            "norm_identity":str(sp.simplify(norm-v*v/e**2-4*m*m/r2)),
            "d_derivatives":[str(sp.simplify(sp.diff(d,e,n).subs(e,2))) for n in range(3)],
            "b_derivatives":[str(sp.simplify(sp.diff(b,e,n).subs(e,2))) for n in range(3)]}


def run():
    branches=[branch_check(eta,ratio) for ratio in (0.,1/12) for eta in (.1,.03,.01,.003,.001)]
    scaling=[branch_check(.01,0.,beta,kappa) for beta,kappa in ((1.,.2),(5.,1.),(2.,3.))]
    geometries=[geometry_check(offset) for offset in ((0.,0.,0.),(-.02,.01,.003),(.02,-.01,-.002))]
    endpoints=[]
    for kappa in (1e-2,1e-3,1e-4,1e-5,1e-6):
        value,error=endpoint_difference(kappa,tolerance=2e-11)
        other,_=endpoint_difference(kappa,tolerance=1e-9)
        endpoints.append({"kappa":kappa,"difference":[value.real,value.imag],
                          "quadrature_error_estimate":error,"tolerance_difference":abs(value-other)})
    coefficient=1j*math.sqrt(3)/18*np.exp(1j*float(geometry.action(0.,1.)))
    slopes=[]
    for large,small in zip(endpoints,endpoints[1:]):
        vl=complex(*large["difference"])/large["kappa"]
        vs=complex(*small["difference"])/small["kappa"]
        slope=(vs-vl)/math.log(large["kappa"]/small["kappa"])
        slopes.append({"kappa":small["kappa"],"slope":[slope.real,slope.imag],
                       "coefficient_error":abs(slope-coefficient)})
    dependencies=("regge_coalescing_kernel.py","regge_constraint_reduction.py",
                  "regge_postconstraint_projection.py","regge_pachner_constraints.py",
                  "regge_pachner_creation.py","regge_pachner_transport.py","regge_tent_transfer.py",
                  "regge_internal_marginal.py","regge_quantum_constraint_transfer.py")
    return {"status":"[산출]","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in dependencies},
            "scope":"공급 스칼라 FEM 작용의 실제 레게 원천·정준 경계 반작용·전체 진폭 끝점",
            "certificate":exact_certificate(),"branches":branches,"coupling_scaling":scaling,
            "geometry_checks":geometries,"threshold":[threshold_check(x) for x in (.03,.01,.003)],
            "endpoint_cases":endpoints,"endpoint_log_slopes":slopes,
            "endpoint_log_coefficient":[coefficient.real,coefficient.imag],
            "negative_controls":{"pure_mean":"R_e>=0, b'>0: 전체 열린 구간 무근",
                "outside_cone":"|m/v|>1/6: e=2에 접근하는 국소 정상 가지 없음",
                "oscillator":"a=0, beta*g*P/M!=0이면 원래 F_j 보존 불가",
                "auxiliary_action":"별도 Qdot² 및 R(e) 퍼텐셜은 원래 정준 모형의 유도가 아님"},
            "unfinished":["경계 스칼라 준비와 계량을 공급했으므로 0D 원천·공통 계량 선택은 미유도",
                "유클리드 작용·이동 전하를 에너지로 승격할 수 없음",
                "로런츠 물리 시간·인과 응답·총에너지·환류의 같은 작용 유도",
                "3+1 Plebanski/Einstein·다른 힘·암흑부문·허블 텐션"]}


if __name__=="__main__":
    result=run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"branch_errors":[[r["eta"],r["ratio"],r["square_ratio_error"],r["scaled_remainder"]] for r in result["branches"]],
                      "threshold":result["threshold"],"endpoint_slopes":result["endpoint_log_slopes"],
                      "geometry_checks":result["geometry_checks"]},ensure_ascii=False))

