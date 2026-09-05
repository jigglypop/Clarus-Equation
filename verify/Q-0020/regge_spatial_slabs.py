"""실제 두 구간 Lorentz 레게 복합체의 공간·시간·장 변분을 비교한다.

공간 S3의 조합, 차원4, 길이·장 준비와 작용은 입력이다.
대칭 축약 정상점을 원래 내부 모서리·장 방정식의 해로 자동 승격하지 않는다.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import cmath
import hashlib
import json
import math

import numpy as np
from scipy.optimize import least_squares, root

HERE = Path(__file__).resolve().parent
BASE_TETRA = tuple(combinations(range(5),4))
DIFFERENCE = np.column_stack((-np.ones(4),np.eye(4)))


def staircase(slabs=2):
    cells = []
    for n in range(slabs):
        for tetra in BASE_TETRA:
            for cut in range(4):
                cells.append(tuple(5*n+v for v in tetra[:cut+1])+
                             tuple(5*(n+1)+v for v in tetra[cut:]))
    return tuple(cells)


class LorentzComplex:
    """경계항과 기호 있는 변제곱 미분을 포함하는 임의 유한 복합체."""
    def __init__(self,cells):
        self.cells = tuple(tuple(c) for c in cells)
        if not self.cells or any(len(c)!=5 or len(set(c))!=5 for c in self.cells):
            raise ValueError("다섯 꼭짓점의 비어 있지 않은 4단체 목록이 필요하다")
        facets = Counter(tuple(sorted(f)) for c in self.cells for f in combinations(c,4))
        if max(facets.values())>2:
            raise ValueError("사면체 면의 소유자는 두 개 이하여야 한다")
        self.boundary = tuple(f for f,count in facets.items() if count==1)
        self.edges = tuple(sorted({tuple(sorted(e)) for c in self.cells for e in combinations(c,2)}))
        self.index = {e:i for i,e in enumerate(self.edges)}
        self.triangles = tuple(sorted({tuple(sorted(t)) for c in self.cells for t in combinations(c,3)}))
        self.hinge_index = {t:i for i,t in enumerate(self.triangles)}
        self.owners = {t:tuple(c for c in self.cells if set(t).issubset(c)) for t in self.triangles}
        self.coefficients = np.array([1 if any(set(t).issubset(f) for f in self.boundary) else 2
                                      for t in self.triangles])
        boundary_edges = {tuple(sorted(e)) for f in self.boundary for e in combinations(f,2)}
        self.internal_edges = tuple(i for i,e in enumerate(self.edges) if e not in boundary_edges)
        boundary_vertices = {v for f in self.boundary for v in f}
        self.vertices = tuple(sorted({v for c in self.cells for v in c}))
        self.vertex_count = max(self.vertices)+1
        self.internal_vertices = tuple(v for v in self.vertices if v not in boundary_vertices)
        self.cell_edge_ids = {c:[self.index[tuple(sorted((c[i],c[j])))] for i,j in combinations(range(5),2)]
                              for c in self.cells}
        self.gram_derivatives = []
        for i,j in combinations(range(5),2):
            ds = np.zeros((5,5));ds[i,j]=ds[j,i]=1
            self.gram_derivatives.append((ds[0,1:,None]+ds[None,0,1:]-ds[1:,1:])/2)

    def gram(self,squared,vertices):
        vertices = tuple(vertices)
        d = np.zeros((len(vertices),len(vertices)))
        for i,j in combinations(range(len(vertices)),2):
            d[i,j]=d[j,i]=squared[self.index[tuple(sorted((vertices[i],vertices[j])))]]
        return (d[0,1:,None]+d[None,0,1:]-d[1:,1:])/2

    def angle(self,squared,cell,hinge):
        rest = tuple(v for v in cell if v not in hinge)
        g = self.gram(squared,hinge+rest)
        plane = g[2:,2:]-g[2:,:2]@np.linalg.solve(g[:2,:2],g[:2,2:])
        aa,bb,ab = float(plane[0,0]),float(plane[1,1]),float(plane[0,1])
        if min(abs(aa),abs(bb))<1e-12:
            raise ValueError("영 법선 사면체는 현재 가지에서 제외한다")
        value = ((ab-1j*cmath.sqrt(complex(aa*bb-ab*ab)))/
                 (cmath.sqrt(complex(aa))*cmath.sqrt(complex(bb))))
        if value.real<0 and abs(value.imag)<1e-12*abs(value.real):
            logarithm = complex(math.log(-value.real),-math.pi)
        else:
            logarithm = cmath.log(value)
        return -1j*logarithm

    def evaluate(self,squared,fields):
        s = np.asarray(squared,dtype=float)
        phi = np.asarray(fields,dtype=float)
        if s.shape!=(len(self.edges),) or not np.all(np.isfinite(s)):
            raise ValueError("모서리별 유한한 변제곱이 필요하다")
        if phi.shape!=(self.vertex_count,) or not np.all(np.isfinite(phi)):
            raise ValueError("꼭짓점별 유한한 스칼라장이 필요하다")
        inverses,volumes,norms = {},[],[]
        inertia_margin = math.inf
        for cell in self.cells:
            g = self.gram(s,cell)
            ev = np.linalg.eigvalsh(g)
            if not (ev[0]<-1e-12 and ev[1]>1e-12):
                raise ValueError("비퇴화 Lorentz 단체가 필요하다")
            inertia_margin = min(inertia_margin,-ev[0],ev[1])
            inverses[cell] = np.linalg.inv(g)
            volumes.append(math.sqrt(-float(np.linalg.det(g)))/24)
        regge,regge_gradient = 0j,np.zeros(len(s),dtype=complex)
        deficit_list = []
        for t,coefficient in zip(self.triangles,self.coefficients):
            ids = [self.index[e] for e in combinations(t,2)]
            sides = s[ids]
            area2 = (sides.sum()**2-2*sides@sides)/16
            if abs(area2)<1e-12:
                raise ValueError("영 면적 삼각형은 현재 가지에서 제외한다")
            area = cmath.sqrt(complex(area2))
            da = (sides.sum()-2*sides)/(16*area)
            deficit = coefficient*math.pi+sum(self.angle(s,c,t) for c in self.owners[t])
            regge -= 1j*area*deficit
            regge_gradient[ids] -= 1j*da*deficit
            deficit_list.append(deficit)
        scalar_action = 0.
        scalar_gradient = np.zeros(len(s))
        field_gradient = np.zeros(len(phi))
        for cell,volume in zip(self.cells,volumes):
            inverse = inverses[cell]
            delta = DIFFERENCE@phi[list(cell)]
            w = inverse@delta
            norm = float(delta@w);norms.append(norm)
            scalar_action -= .5*volume*norm
            field_gradient[list(cell)] -= volume*DIFFERENCE.T@w
            for index,dg in zip(self.cell_edge_ids[cell],self.gram_derivatives):
                scalar_gradient[index] -= .5*volume*(.5*np.trace(inverse@dg)*norm-w@dg@w)
        return {"regge":regge,"regge_gradient":regge_gradient,
                "scalar":float(scalar_action),"scalar_gradient":scalar_gradient,
                "field_gradient":field_gradient,"volumes":volumes,"norms":norms,
                "inertia_margin":float(inertia_margin),"deficits":np.array(deficit_list)}


FULL = LorentzComplex(staircase(2))
SLABS = tuple(LorentzComplex(tuple(c for c in FULL.cells if min(c)//5==n)) for n in range(2))


def family(scales,struts,fields,complex_=FULL):
    a,m,p = np.asarray(scales,float),np.asarray(struts,float),np.asarray(fields,float)
    if a.shape!=(3,) or m.shape!=(2,) or p.shape!=(3,):
        raise ValueError("공간 크기3·연결 길이2·장3이 필요하다")
    if not all(np.all(np.isfinite(x)) for x in (a,m,p)) or np.any(a<=0) or np.any(m<=0):
        raise ValueError("양의 공간 크기·시간꼴 연결 길이와 유한한 장이 필요하다")
    squared=np.zeros(len(complex_.edges))
    jacobian=np.zeros((len(squared),5))
    for k,(i,j) in enumerate(complex_.edges):
        n,v=divmod(i,5);q,w=divmod(j,5)
        if n==q:
            squared[k]=a[n]**2
            jacobian[k,n]=2*a[n]
        elif q==n+1:
            squared[k]=-m[n]**2
            jacobian[k,3+n]=-2*m[n]
            if v!=w:
                squared[k]+=a[n]*a[q]
                jacobian[k,n]=a[q];jacobian[k,q]=a[n]
        else:
            raise ValueError("이웃하지 않는 시간 층은 연결하지 않는다")
    phi=np.zeros(complex_.vertex_count)
    for vertex in complex_.vertices:
        phi[vertex]=p[vertex//5]
    return squared,phi,jacobian


def restricted(scales,struts,fields,beta=1.,coupling=1.,complex_=FULL):
    if not math.isfinite(beta) or not math.isfinite(coupling) or beta<=0 or coupling<=0:
        raise ValueError("양의 무차원 결합이 필요하다")
    s,phi,jac=family(scales,struts,fields,complex_)
    native=complex_.evaluate(s,phi)
    edge_gradient=beta*native["regge_gradient"].real+coupling*native["scalar_gradient"]
    field_gradient=coupling*native["field_gradient"]
    gradient=np.r_[jac.T@edge_gradient,
                    [sum(field_gradient[v] for v in complex_.vertices if v//5==n) for n in range(3)]]
    native.update({"action":float(beta*native["regge"].real+coupling*native["scalar"]),
                   "gradient":gradient,"edge_gradient":edge_gradient,
                   "total_field_gradient":field_gradient,
                   "squared":s,"fields":phi,"family_jacobian":jac})
    return native


def centered(function,point,step=1e-5):
    point=np.asarray(point,float)
    columns=[]
    for d in np.eye(len(point)):
        coarse=(np.asarray(function(point+step*d))-np.asarray(function(point-step*d)))/(2*step)
        fine=(np.asarray(function(point+step*d/2))-np.asarray(function(point-step*d/2)))/step
        columns.append((4*fine-coarse)/3)
    return np.array(columns).T


def composition_audit(scales=(1.,1.08,1.02),struts=(.15,.18),fields=(-1.,.1,1.)):
    full=restricted(scales,struts,fields)
    parts=[restricted(scales,struts,fields,complex_=c) for c in SLABS]
    return {"action_error":abs(full["action"]-sum(p["action"] for p in parts)),
            "complex_regge_error":abs(full["regge"]-sum(p["regge"] for p in parts)),
            "gradient_error":float(np.max(abs(full["gradient"]-sum(p["gradient"] for p in parts)))),
            "scalar_shift_error":abs(sum(full["total_field_gradient"])),
            "cell_count":len(FULL.cells),"edge_count":len(FULL.edges),
            "internal_edge_count":len(FULL.internal_edges),
            "internal_vertex_count":len(FULL.internal_vertices),
            "inertia_margin":full["inertia_margin"]}


def gradient_audit(scales=(1.,1.08,1.02),struts=(.15,.18),fields=(-1.,.1,1.)):
    """대표 독립 내부 변과 장에 대해 작용 차분을 직접 대조한다."""
    row=restricted(scales,struts,fields)
    edge_ids=np.array(FULL.internal_edges[::7],dtype=int)
    field_ids=np.array(FULL.internal_vertices,dtype=int)
    edge_errors=[]
    for index in edge_ids:
        def action(point):
            squared=row["squared"].copy();squared[index]=point[0]
            native=FULL.evaluate(squared,row["fields"])
            return np.array([native["regge"].real+native["scalar"]])
        direct=centered(action,[row["squared"][index]],step=1e-5)[0,0]
        edge_errors.append(abs(direct-row["edge_gradient"][index]))
    field_errors=[]
    for vertex in field_ids:
        def action(point):
            phi=row["fields"].copy();phi[vertex]=point[0]
            native=FULL.evaluate(row["squared"],phi)
            return np.array([native["regge"].real+native["scalar"]])
        direct=centered(action,[row["fields"][vertex]],step=1e-5)[0,0]
        field_errors.append(abs(direct-row["total_field_gradient"][vertex]))
    return {"edge_samples":len(edge_ids),"field_samples":len(field_ids),
            "edge_error":float(max(edge_errors)),"field_error":float(max(field_errors))}


def solve_symmetric(v=1.,guess=(1.03,.1)):
    """대칭 경계 a0=a2=1, phi0=-v, phi2=v에서 두 식만 선별한다."""
    def residual(z):
        a1,m=np.exp(z)
        try:
            row=restricted([1.,a1,1.],[m,m],[-v,0.,v])
            return row["gradient"][[1,3]]
        except ValueError:
            return np.array([1e6,1e6])
    solved=root(residual,np.log(guess),tol=1e-10)
    a1,m=np.exp(solved.x)
    row=restricted([1.,a1,1.],[m,m],[-v,0.,v])
    free=row["gradient"][[1,3,4,6]]
    edge=row["edge_gradient"][list(FULL.internal_edges)]
    fields=row["total_field_gradient"][list(FULL.internal_vertices)]
    return {"v":v,"success":bool(solved.success),"a1":float(a1),"m":float(m),
            "reduced_gradient":free.tolist(),"full_internal_edge_gradient":edge.tolist(),
            "full_internal_field_gradient":fields.tolist(),
            "max_reduced_residual":float(np.max(abs(free))),
            "max_full_edge_residual":float(np.max(abs(edge))),
            "max_full_field_residual":float(np.max(abs(fields))),
            "action":row["action"],"regge_real":float(row["regge"].real),
            "regge_imaginary":float(row["regge"].imag),"scalar":row["scalar"],
            "inertia_margin":row["inertia_margin"]}


def solve_full(v=1., max_nfev=300):
    """대칭 경계만 고정하고 내부 변40개와 장5개를 모두 독립 변분한다."""
    symmetric = solve_symmetric(v)
    start = restricted([1.,symmetric["a1"],1.],
                       [symmetric["m"],symmetric["m"]],[-v,0.,v])
    edges = np.array(FULL.internal_edges,dtype=int)
    vertices = np.array(FULL.internal_vertices,dtype=int)
    initial = np.r_[start["squared"][edges],start["fields"][vertices]]

    def unpack(point):
        squared=start["squared"].copy();fields=start["fields"].copy()
        squared[edges]=point[:len(edges)]
        fields[vertices]=point[len(edges):]
        return squared,fields

    def residual(point):
        try:
            row=FULL.evaluate(*unpack(point))
            return np.r_[row["regge_gradient"].real[edges]+row["scalar_gradient"][edges],
                         row["field_gradient"][vertices]]
        except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):
            return np.full(len(point),1e3)

    solved=least_squares(residual,initial,x_scale="jac",max_nfev=max_nfev,
                         xtol=1e-12,ftol=1e-12,gtol=1e-12,diff_step=1e-6)
    squared,fields=unpack(solved.x)
    row=FULL.evaluate(squared,fields)
    equation=residual(solved.x)
    hessian=centered(residual,solved.x,step=2e-5)
    singular=np.linalg.svd(hessian,compute_uv=False)
    edge_values=squared[edges]
    field_values=fields[vertices]
    return {"success":bool(solved.success),"status":int(solved.status),
            "evaluations":int(solved.nfev),"cost":float(solved.cost),
            "max_equation_residual":float(np.max(abs(equation))),
            "solution_move":float(np.linalg.norm(solved.x-initial)),
            "inertia_margin":row["inertia_margin"],
            "internal_squared":edge_values.tolist(),
            "internal_fields":field_values.tolist(),
            "edge_minimum":float(edge_values.min()),"edge_maximum":float(edge_values.max()),
            "field_minimum":float(field_values.min()),"field_maximum":float(field_values.max()),
            "hessian_asymmetry":float(np.max(abs(hessian-hessian.T))),
            "hessian_singular_values":singular.tolist(),
            "hessian_rank_1e8":int(np.sum(singular>1e-8*singular[0])),
            "scalar_shift_error":abs(sum(row["field_gradient"])),
            "boundary_field_charges":[float(sum(row["field_gradient"][5*n:5*n+5]))
                                      for n in (0,2)],
            "regge_imaginary":float(row["regge"].imag),
            "action":float(row["regge"].real+row["scalar"]),
            "regge_real":float(row["regge"].real),"scalar":row["scalar"]}


def run():
    report={"status":"두 구간 실제 내부 정상점; 시간 방향과 0D 발생은 공급",
            "composition":composition_audit(),"gradient":gradient_audit(),
            "symmetric":solve_symmetric(),"full":solve_full(),
            "sources":["https://arxiv.org/abs/2312.11639",
                       "https://arxiv.org/abs/1501.07614"]}
    report["dependencies"]={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                            for name in ("regge_spatial_slabs.py",)}
    return report


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2))
