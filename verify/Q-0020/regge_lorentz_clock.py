"""로런츠 네 단체의 스칼라 시계·경계 에너지·기하 응답을 검산한다.

부호 있는 길이제곱, 미래를 보존하는 접합과 경계 장 준비는 입력이다.
복소 레게 경계 모서리 항을 보존하며 고전 생성함수에는 실수부를 쓴다.
이 유한 경계 문제는 지연 전파자나 미시적 공통 계량 선택을 유도하지 않는다.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
import cmath
import hashlib
import json
import math

import numpy as np
from scipy.optimize import brentq

import regge_pachner_constraints as moves

EDGES = moves.FINAL.edges
N = len(EDGES)
INDEX = {edge:i for i,edge in enumerate(EDGES)}
E, Y = INDEX[0,1], INDEX[3,4]
OLD, MIDDLE, FINAL = moves.MOVE.old.cells, moves.MOVE.new.cells, moves.FINAL.cells
C = math.sqrt(3)/9
B = 35/36
D = 1/3
T0 = 1/3
DIFFERENCE = np.column_stack((-np.ones(4),np.eye(4)))


def gram(squared, vertices):
    vertices = tuple(vertices)
    matrix = np.zeros((len(vertices),len(vertices)))
    for i,j in combinations(range(len(vertices)),2):
        matrix[i,j] = matrix[j,i] = squared[INDEX[tuple(sorted((vertices[i],vertices[j])))]]
    return (matrix[0,1:,None]+matrix[None,0,1:]-matrix[1:,1:])/2


def validate(squared, cells):
    s = np.asarray(squared,dtype=float)
    if s.shape != (N,) or not np.all(np.isfinite(s)):
        raise ValueError("유한한 부호 있는 길이제곱 15개가 필요하다")
    for cell in cells:
        eigenvalues = np.linalg.eigvalsh(gram(s,cell))
        if not (eigenvalues[0]<-1e-12 and eigenvalues[1]>1e-12):
            raise ValueError("각 단체에 비퇴화 로런츠 부호 (-+++)가 필요하다")
    return s


def boundary_data(cells):
    facets = Counter(tuple(sorted(f)) for cell in cells for f in combinations(cell,4))
    boundary = {f for f,count in facets.items() if count==1}
    triangles = sorted({tuple(sorted(h)) for cell in cells for h in combinations(cell,3)})
    kappa = {h:1 if any(set(h).issubset(f) for f in boundary) else 2 for h in triangles}
    return boundary,kappa


def triangle(squared, hinge):
    ids = [INDEX[edge] for edge in combinations(hinge,2)]
    sides = squared[ids]
    area2 = (2*(sides[0]*sides[1]+sides[0]*sides[2]+sides[1]*sides[2])
             -sides@sides)/16
    if abs(area2)<1e-12:
        raise ValueError("영면적 힌지는 이 검산 가지에서 제외한다")
    area = cmath.sqrt(complex(area2))
    gradient = np.zeros(N,dtype=complex)
    gradient[ids] = (sum(sides)-2*sides)/(16*area)
    return area,gradient


def angle(squared, cell, hinge):
    """힌지의 직교 평면을 슈어 보완으로 구하고 지정한 복소 가지를 쓴다."""
    rest = tuple(v for v in cell if v not in hinge)
    g = gram(squared,hinge+rest)
    plane = g[2:,2:]-g[2:,:2]@np.linalg.solve(g[:2,:2],g[:2,2:])
    aa,bb,ab = float(plane[0,0]),float(plane[1,1]),float(plane[0,1])
    if min(abs(aa),abs(bb))<1e-12:
        raise ValueError("영 법선 면은 이 검산 가지에서 제외한다")
    value = (ab-1j*cmath.sqrt(complex(aa*bb-ab*ab)))/(cmath.sqrt(complex(aa))*cmath.sqrt(complex(bb)))
    if value.real<0 and abs(value.imag)<1e-12*abs(value.real):
        logarithm = complex(math.log(-value.real),-math.pi)
    else:
        logarithm = cmath.log(value)
    return -1j*logarithm


def regge(squared, cells=FINAL):
    """경계와 내부 힌지를 모두 합산하고 슈레플리 항등식으로 미분한다."""
    cells = tuple(cells)
    s = validate(squared,cells)
    boundary,kappa = boundary_data(cells)
    action = 0j
    gradient = np.zeros(N,dtype=complex)
    hinges = []
    for hinge,coefficient in kappa.items():
        area,da = triangle(s,hinge)
        deficit = coefficient*math.pi+sum(angle(s,cell,hinge) for cell in cells if set(hinge).issubset(cell))
        action -= 1j*area*deficit
        gradient -= 1j*da*deficit
        hinges.append({"hinge":hinge,"boundary":coefficient==1,"area":area,"deficit":deficit})
    return {"action":action,"gradient":gradient,"hinges":hinges,"boundary":boundary}


def scalar(squared, fields, cells=FINAL):
    """로런츠 선형 스칼라 작용과 두 종류의 변분을 그램에서 조립한다."""
    cells = tuple(cells)
    s = validate(squared,cells)
    phi = np.asarray(fields,dtype=float)
    if phi.shape!=(6,) or not np.all(np.isfinite(phi)):
        raise ValueError("유한한 꼭짓점 장 6개가 필요하다")
    stiffness,gradient = np.zeros((6,6)),np.zeros(N)
    volumes,norms = [],[]
    for cell in cells:
        g = gram(s,cell)
        inv = np.linalg.inv(g)
        volume = math.sqrt(-float(np.linalg.det(g)))/24
        delta = DIFFERENCE@phi[list(cell)]
        w = inv@delta
        norm = float(delta@w)
        stiffness[np.ix_(cell,cell)] -= volume*DIFFERENCE.T@inv@DIFFERENCE
        for i,j in combinations(range(5),2):
            ds = np.zeros((5,5))
            ds[i,j] = ds[j,i] = 1
            dg = (ds[0,1:,None]+ds[None,0,1:]-ds[1:,1:])/2
            index = INDEX[tuple(sorted((cell[i],cell[j])))]
            gradient[index] -= .5*volume*(.5*np.trace(inv@dg)*norm-w@dg@w)
        volumes.append(volume)
        norms.append(norm)
    return {"action":float(.5*phi@stiffness@phi),"gradient":gradient,
            "field_momentum":stiffness@phi,"stiffness":stiffness,
            "volumes":volumes,"norms":norms}


def local_increment(squared, fields, before, after, cell, beta, coupling):
    """전체 작용의 차를 쓰지 않고 새 단체와 경계각 상수 변화로 조립한다."""
    local = regge(squared,(cell,))
    _,before_k = boundary_data(before)
    _,after_k = boundary_data(after)
    action,gradient = local["action"],local["gradient"].copy()
    for hinge in boundary_data((cell,))[1]:
        coefficient = after_k[hinge]-before_k.get(hinge,0)-1
        area,da = triangle(squared,hinge)
        action -= 1j*math.pi*coefficient*area
        gradient -= 1j*math.pi*coefficient*da
    matter = scalar(squared,fields,(cell,))
    return {"complex_action":beta*action+coupling*matter["action"],
            "momentum":np.r_[beta*gradient.real+coupling*matter["gradient"],
                             coupling*matter["field_momentum"]]}


def totals(squared, fields, beta=1., coupling=1.):
    out = {}
    for label,cells in (("old",OLD),("middle",MIDDLE),("final",FINAL)):
        gravity,matter = regge(squared,cells),scalar(squared,fields,cells)
        out[label] = {"complex_action":beta*gravity["action"]+coupling*matter["action"],
                      "momentum":np.r_[beta*gravity["gradient"].real+coupling*matter["gradient"],
                                      coupling*matter["field_momentum"]]}
    out["first"] = local_increment(squared,fields,OLD,MIDDLE,(0,1,2,3,4),beta,coupling)
    out["second"] = local_increment(squared,fields,MIDDLE,FINAL,moves.FUTURE_CELL,beta,coupling)
    return out


def symmetric(T, b=B):
    if not np.isfinite(T) or T<=0 or not np.isfinite(b) or b<=8/9:
        raise ValueError("T>0와 spacelike 경계 b>8/9가 필요하다")
    s = np.empty(N)
    for i,(a,c) in enumerate(EDGES):
        s[i] = -T*T if (a,c)==(0,1) else b if a<2 else 8/3
    return s


def fields(v):
    return np.array([-v/2,v/2,0.,0.,0.,0.])


def radius(T):
    return math.sqrt(D+T*T)


def gravity_force(T):
    z = B+T*T/4
    deficit = 2*math.pi-3*math.acos((3*z-4)/(6*z-4))
    return 2*(2*B+T*T)/math.sqrt(4*B+T*T)*deficit


def matter_force(T):
    return -C*D/(2*radius(T)*T*T)


def stationary(xi):
    if not np.isfinite(xi) or xi<0:
        raise ValueError("유한한 음이 아닌 원천 세기가 필요하다")
    if xi==0:
        return T0
    upper = 1.
    while gravity_force(upper)+xi*matter_force(upper)<0:
        upper *= 2
    return brentq(lambda T:gravity_force(T)+xi*matter_force(T),T0,upper,xtol=1e-14)


def clock_audit(T, v=1., coupling=1., preserve_future=True):
    """시간 함수의 단위 법선과 실제 경계 사면체의 에너지 플럭스를 검사한다."""
    if not preserve_future:
        raise ValueError("미래 방향을 보존하는 접합을 따로 공급해야 한다")
    s = symmetric(T)
    time = fields(T)
    rho = coupling*v*v/(2*T*T)
    boundary,_ = boundary_data(FINAL)
    flux = {"past":0.,"future":0.}
    scalar_flux = np.zeros(6)
    norm_errors,spatial,volumes,internal_flux = [],[],[],[]
    for cell in FINAL:
        g = gram(s,cell)
        inv = np.linalg.inv(g)
        dt = DIFFERENCE@time[list(cell)]
        norm_errors.append(abs(float(dt@inv@dt)+1))
        volumes.append(math.sqrt(-float(np.linalg.det(g)))/24)
        for omitted in range(5):
            face = tuple(sorted(vx for i,vx in enumerate(cell) if i!=omitted))
            normal = DIFFERENCE[:,omitted]
            if face not in boundary:
                internal_flux.append(abs(float(normal@inv@dt)))
                continue
            face_gram = gram(s,face)
            spatial.append(float(np.linalg.eigvalsh(face_gram)[0]))
            volume = math.sqrt(float(np.linalg.det(face_gram)))/6
            lapse = abs(float(normal@inv@dt))/math.sqrt(-float(normal@inv@normal))
            side = "past" if 0 in face else "future"
            flux[side] += rho*lapse*volume
            # 기저 함수의 면 평균은 1/4, 양쪽 면의 부호를 포함한다.
            charge = coupling*v/T*lapse*volume*(-1 if side=="past" else 1)
            scalar_flux[list(face)] += charge/4
    r = radius(T)
    V3 = 4*C*r
    energy = rho*V3
    matter = scalar(s,fields(v))
    return {"T":T,"rho":rho,"pressure":rho,"V3":V3,"V4":sum(volumes),
            "clock_norm_error":max(norm_errors),"minimum_boundary_gram":min(spatial),
            "internal_flux_error":max(internal_flux),
            "past_energy":flux["past"],"future_energy":flux["future"],"energy":energy,
            "energy_flux_error":max(abs(flux[k]-energy) for k in flux),
            "volume_error":abs(sum(volumes)-T*V3/4),
            "field_flux_error":float(np.max(np.abs(scalar_flux-coupling*matter["field_momentum"]))),
            "weighted_energy_force_error":abs(-coupling*matter["gradient"][E]*(-2*T)-energy*D/(4*r*r)),
            "shift_charge":float(sum(coupling*matter["field_momentum"]))}


def derivative(function, point, index, step=1e-4):
    direction = np.eye(len(point))[index]
    coarse = (function(point+step*direction)-function(point-step*direction))/(2*step)
    fine = (function(point+step*direction/2)-function(point-step*direction/2))/step
    return (4*fine-coarse)/3


def geometry_audit(T, offset=0.):
    s = symmetric(T)
    s[Y] += offset
    s[INDEX[0,2]] += .3*offset
    phi = np.array([-.31,.47,.12,-.08,.16,-.05])
    gravity,matter = regge(s),scalar(s,phi)
    gd = np.array([derivative(lambda q:regge(q)["action"],s,i) for i in range(N)])
    md = np.array([derivative(lambda q:scalar(q,phi)["action"],s,i) for i in range(N)])
    joined = totals(s,phi,2.,.7)
    errors = []
    for before,after,inc in (("old","middle","first"),("middle","final","second")):
        errors.extend([abs(joined[after]["complex_action"]-joined[before]["complex_action"]-joined[inc]["complex_action"]),
                       np.max(np.abs(joined[after]["momentum"]-joined[before]["momentum"]-joined[inc]["momentum"]))])
    mixed = []
    for i in (E,Y,INDEX[0,2]):
        left = derivative(lambda q:scalar(q,phi)["field_momentum"],s,i)
        right = np.array([derivative(lambda f:scalar(s,f)["gradient"][i],phi,j) for j in range(6)])
        mixed.append(np.max(np.abs(left-right)))
    return {"T":T,"offset":offset,"regge_gradient_error":float(np.max(np.abs(gd-gravity["gradient"]))),
            "scalar_gradient_error":float(np.max(np.abs(md-matter["gradient"]))),
            "mixed_derivative_error":float(max(mixed)),"composition_error":float(max(errors)),
            "homogeneity_error":float(abs(s@gravity["gradient"]-gravity["action"])),
            "scalar_scale_error":float(abs(s@matter["gradient"]-matter["action"])),
            "constant_mode_error":float(np.max(np.abs(matter["stiffness"]@np.ones(6))))}


def branch_audit(xi):
    T = stationary(xi)
    s,phi = symmetric(T),fields(math.sqrt(xi))
    data = totals(s,phi)
    g,m = regge(s),scalar(s,phi)
    c = data["middle"]["momentum"][Y]-data["first"]["momentum"][Y]
    return {"xi":xi,"T":T,"response_coefficient":(T-T0)/xi,
            "response_error":abs((T-T0)/xi-9/74),
            "stationary_residual":abs(data["final"]["momentum"][E]),
            "creation_residual":abs(c),
            "gravity_force_error":abs(-2*T*g["gradient"][E].real-gravity_force(T)),
            "scalar_force_error":abs(-2*T*m["gradient"][E]-xi*matter_force(T)),
            "corner_imaginary":g["action"].imag,
            "corner_error":abs(g["action"].imag+8*math.sqrt(3)*math.pi/3),
            "corner_internal_derivative":abs(g["gradient"][E].imag),
            "clock":clock_audit(T,math.sqrt(xi))}


def run():
    dependencies = {Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("regge_pachner_constraints.py","regge_pachner_creation.py","regge_tent_transfer.py"):
        dependencies[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return {"model":"조건부 로런츠 스칼라 시계와 고전 레게 응답",
            "source":"https://arxiv.org/html/2407.13601v2#S2.SS1",
            "dependencies":dependencies,"boundary_b":B,"D":D,"vacuum_T":T0,
            "predicted_response":9/74,"branches":[branch_audit(xi) for xi in (1.,.1,.01,.001,.0001,.00001)],
            "geometry_checks":[geometry_audit(.4,0.),geometry_audit(.6,.004),geometry_audit(.9,-.003)],
            "clock_checks":[clock_audit(T,.7,1.3) for T in (.2,T0,.6,1.5)],
            "limits":["로런츠 부호·공유 계량·미래 접합·경계 원천은 공급 조건",
                      "경계 에너지 플럭스는 보충·저장·재충전의 발생 증명이 아님",
                      "질량 없는 순수 시계는 p=rho이며 암흑에너지나 차가운 암흑물질 자체가 아님",
                      "지연 그린 함수·연속 극한·Plebanski·다른 힘·우주론 관측은 미완성"]}


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=True,indent=2,allow_nan=False))
