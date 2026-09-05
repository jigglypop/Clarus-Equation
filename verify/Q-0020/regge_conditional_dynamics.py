"""실제 두 경계의 정준 관계와 조건부 읽기의 보존·위상 예산을 검산한다.

공유 기하, 무차원 작용, 두 공동 측도와 조건부 읽기는 공급 전제다.
남은 진폭의 노름을 보존하되 에너지 장부나 유일한 물리 양자화로 해석하지 않는다.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import regge_postconstraint_projection as previous
from regge_tent_transfer import LOCAL_EDGES, COMPLEMENTS

moves, full = previous.moves, previous.full
E, Y = moves.E_ID, moves.Y_ID
REST = np.array([i for i in range(len(moves.FINAL.edges)) if i not in (E,Y)])
Y0 = math.sqrt(8/3)


def y_limit(edge,h=1.):
    edge=np.asarray(edge,dtype=float)
    upper=full.limit(h)
    if np.any((edge<0)|(edge>upper)) or not np.all(np.isfinite(edge)):
        raise ValueError("길이는 닫힌 실제 e 구간 안에 있어야 한다")
    return np.sqrt(4*Y0**2-Y0**4/(h*h+1-edge**2/4))


def e_limit(y,h=1.):
    y=np.asarray(y,dtype=float)
    top=float(y_limit(0.,h))
    if np.any((y<0)|(y>top)) or not np.all(np.isfinite(y)):
        raise ValueError("길이는 닫힌 실제 y 구간 안에 있어야 한다")
    curved=2*np.sqrt(np.maximum(h*h+1-Y0**4/(4*Y0**2-y*y),0.))
    return np.minimum(full.limit(h),curved)


def batch_action(complex_,lengths):
    """기존 단체 그램·법선·슐레플리 작용을 독립 배열 축으로 일괄 계산한다."""
    lengths=np.asarray(lengths,dtype=float)
    if lengths.ndim!=2 or lengths.shape[1]!=len(complex_.edges) or np.any(lengths<=0):
        raise ValueError("양의 모서리 길이 행렬이 필요하다")
    count=len(lengths)
    deficits=np.broadcast_to(np.where(complex_.boundary,math.pi,2*math.pi),
                             (count,len(complex_.triangles))).copy()
    minimum=np.full(count,np.inf)
    ci,cj=np.array(COMPLEMENTS).T
    for edges,triangles in zip(complex_.cell_edges,complex_.cell_triangles):
        distances=np.zeros((count,5,5))
        for (i,j),index in zip(LOCAL_EDGES,edges):
            distances[:,i,j]=distances[:,j,i]=lengths[:,index]**2
        base=distances[:,0,1:]
        gram=(base[:,:,None]+base[:,None,:]-distances[:,1:,1:])/2
        minimum=np.minimum(minimum,np.linalg.eigvalsh(gram)[:,0])
        if np.any(minimum<=0):
            raise ValueError("비퇴화 실제 그램 영역 밖이다")
        inverse=np.linalg.inv(gram)
        normals=np.empty((count,5,5))
        normals[:,1:,1:]=inverse
        normals[:,0,1:]=normals[:,1:,0]=-inverse.sum(axis=1)
        normals[:,0,0]=inverse.sum(axis=(1,2))
        cosine=-normals[:,ci,cj]/np.sqrt(normals[:,ci,ci]*normals[:,cj,cj])
        deficits[:,triangles]-=np.arccos(np.clip(cosine,-1,1))
    squared=lengths[:,np.array(complex_.triangle_edges)]**2
    area_squared=(2*(squared[:,:,0]*squared[:,:,1]+squared[:,:,0]*squared[:,:,2]+
                     squared[:,:,1]*squared[:,:,2])-(squared*squared).sum(axis=2))/16
    if np.any(area_squared<=0):
        raise ValueError("삼각형 면적이 양수가 아니다")
    return np.sum(np.sqrt(area_squared)*deficits,axis=1),minimum


def geometry_check(h=1.,beta=5.):
    q0=previous.transfer.geometry_lengths(h)
    qs=[]
    outside=0
    for fraction in (.2,.6,.9):
        edge=full.limit(h)*fraction
        for y_fraction in (.25,.75,.99):
            q=q0.copy();q[E]=edge;q[Y]=float(y_limit(edge,h))*y_fraction
            qs.append(q)
        q=q0.copy();q[E]=edge;q[Y]=float(y_limit(edge,h))*1.001
        try:
            moves.FINAL.evaluate(q)
        except ValueError:
            outside+=1
    values,minima=batch_action(moves.FINAL,np.array(qs))
    errors=[];canonical=[];support=[]
    for q,value in zip(qs,values):
        data=moves.actions(q)
        gradient=data["first"]["gradient"]+data["second"]["gradient"]
        incoming=np.zeros(len(q));incoming[REST]=np.linspace(-.4,.6,len(REST))
        incoming[E]=-beta*gradient[E]
        _,middle=moves.MOVE.create(q[moves.OLD_IDS],incoming[moves.OLD_IDS],q[Y],beta=beta)
        outgoing=middle+beta*data["second"]["gradient"]
        expected=incoming+beta*gradient
        canonical.append(float(np.max(np.abs(outgoing-expected))))
        canonical.append(float(abs(outgoing[E])))
        errors.append(abs(value-data["final"]["action"]))
        bounds=previous.transfer.fine_interval(q)
        support.append(abs(bounds[0]))
        support.append(abs(bounds[1]-float(e_limit(q[Y],h))))
    return {"h":h,"action_error":float(max(errors)),"canonical_error":max(canonical),
            "support_error":float(max(support)),"minimum_gram":float(min(minima)),
            "outside_rejections":outside}


def canonical_relation_check(h=1.,beta=5.,step=2e-5):
    """임계점에서도 전체 28차원 경계 관계가 남는지 독립 미분으로 대조한다."""
    q=previous.transfer.geometry_lengths(h)
    count=len(q)
    def gradient(lengths):
        data=moves.actions(lengths)
        return data["first"]["gradient"]+data["second"]["gradient"]
    def central(delta):
        return np.column_stack([(gradient(q+np.eye(count)[i]*delta)-
                                 gradient(q-np.eye(count)[i]*delta))/(2*delta)
                                for i in range(count)])
    hessian=(4*central(step/2)-central(step))/3
    incoming=np.zeros((28,28));outgoing=np.zeros((28,28))
    incoming[0,E]=1;outgoing[0,Y]=1
    for j,index in enumerate(REST):
        incoming[1+j,index]=outgoing[1+j,index]=1
        incoming[15+j,:count]=-beta*hessian[index]
        incoming[15+j,count+j]=outgoing[15+j,count+j]=1
    incoming[14,:count]=-beta*hessian[E]
    outgoing[14,:count]=beta*hessian[Y]
    omega=np.block([[np.zeros((14,14)),np.eye(14)],[-np.eye(14),np.zeros((14,14))]])
    pull_in=incoming.T@omega@incoming
    pull_out=outgoing.T@omega@outgoing
    tolerance=1e-6
    rank=lambda matrix:int(np.sum(np.linalg.svd(matrix,compute_uv=False)>tolerance))
    return {"h":h,"mixed_derivative":float(hessian[E,Y]),
            "relation_rank":rank(np.vstack((incoming,outgoing))),
            "input_projection_rank":rank(incoming),"output_projection_rank":rank(outgoing),
            "pullback_rank":rank(pull_out),
            "lagrangian_error":float(np.max(np.abs(pull_out-pull_in))),
            "hessian_skew":float(np.max(np.abs(hessian-hessian.T))),
            "rank_tolerance":tolerance}


def joint_mesh(size=64,h=1.,kind="length"):
    """곡선 그램 경계로 잘린 각 직사각형의 질량·중심을 적분한다."""
    if not isinstance(size,int) or size<8:
        raise ValueError("분할 수는 8 이상 정수여야 한다")
    if kind not in ("length","squared"):
        raise ValueError("공동 길이 또는 제곱길이 측도만 지원한다")
    es=np.linspace(0,full.limit(h),size+1)
    ys=np.linspace(0,float(y_limit(0,h)),size+1)
    elo,ylo=np.meshgrid(es[:-1],ys[:-1])
    ehi,yhi=np.meshgrid(es[1:],ys[1:])
    cut_hi=e_limit(yhi,h);cut_lo=e_limit(ylo,h)
    nodes,weights=np.polynomial.legendre.leggauss(4)
    totals=[np.zeros_like(elo) for _ in range(3)]
    for left,right,curved in ((elo,np.minimum(ehi,cut_hi),False),
                              (np.maximum(elo,cut_hi),np.minimum(ehi,cut_lo),True)):
        width=np.maximum(right-left,0.)
        edge=left[:,:,None]+width[:,:,None]*(nodes+1)/2
        weight=width[:,:,None]*weights/2
        top=y_limit(edge,h) if curved else yhi[:,:,None]+np.zeros_like(edge)
        low=ylo[:,:,None]
        if kind=="length":
            density=np.maximum(top-low,0.)
            y_moment=np.maximum(top*top-low*low,0.)/2
        else:
            density=2*edge*np.maximum(top*top-low*low,0.)
            y_moment=4*edge*np.maximum(top**3-low**3,0.)/3
        totals[0]+=np.sum(weight*density,axis=2)
        totals[1]+=np.sum(weight*density*edge,axis=2)
        totals[2]+=np.sum(weight*y_moment,axis=2)
    mass,emoment,ymoment=totals
    active=mass>0
    ec,yc=emoment[active]/mass[active],ymoment[active]/mass[active]
    lengths=np.tile(previous.transfer.geometry_lengths(h),(len(ec),1))
    lengths[:,E]=ec;lengths[:,Y]=yc
    final,minima=batch_action(moves.FINAL,lengths)
    old,_=batch_action(moves.MOVE.old,lengths[:,moves.OLD_IDS])
    action=np.zeros_like(mass);action[active]=final-old
    final_grid=np.zeros_like(mass);final_grid[active]=final
    volume=float(mass.sum())
    def volume_integrand(e):
        top=float(y_limit(e,h))
        return top if kind=="length" else 2*e*top*top
    independent=quad(volume_integrand,0,full.limit(h),epsabs=1e-11,epsrel=1e-11)[0]
    return {"joint":mass/volume,"action":action,"final_action":final_grid,
            "volume":volume,"volume_error":abs(volume-independent),
            "minimum_gram":float(min(minima)),"size":size,"h":h,"kind":kind}


def interaction_budget(joint,action):
    """입력·출력에 따로 붙는 위상을 제거한 최소 가중 제곱 잔차."""
    pe,py=joint.sum(axis=0),joint.sum(axis=1)
    count=len(pe)
    normal=np.block([[np.diag(pe),joint.T],[joint,np.diag(py)]])
    rhs=np.r_[(joint*action).sum(axis=0),(joint*action).sum(axis=1)]
    gauge=np.r_[pe,np.zeros(len(py))]
    augmented=np.block([[normal,gauge[:,None]],[gauge[None,:],np.zeros((1,1))]])
    coefficients=np.linalg.solve(augmented,np.r_[rhs,0.])[:-1]
    a,b=coefficients[:count],coefficients[count:]
    residual=action-a[None,:]-b[:,None]
    error=max(np.max(np.abs((joint*residual).sum(axis=0))),
              np.max(np.abs((joint*residual).sum(axis=1))))
    return {"coefficient":float(np.sum(joint*residual**2)),
            "normal_equation_error":float(error)}


def transfer_check(joint,action,beta):
    pe,py=joint.sum(axis=0),joint.sum(axis=1)
    phase=np.exp(1j*beta*action)
    matrix=joint*phase/np.sqrt(py[:,None]*pe[None,:])
    _,singular,vh=np.linalg.svd(matrix,full_matrices=False)
    source=vh[0].conj()/np.sqrt(pe)
    target=np.sum(joint*phase*source[None,:],axis=1)/py
    remainder=phase*source[None,:]-target[:,None]
    kept=float(np.sum(py*abs(target)**2));folded=float(np.sum(joint*abs(remainder)**2))
    return {"beta":beta,"norm_squared":float(singular[0]**2),
            "second_singular":float(singular[1]),"optimal_remainder":folded,
            "budget_error":abs(kept+folded-1),
            "spectral_budget_error":abs(folded-(1-singular[0]**2)),
            "weak_phase_ratio":None if beta==0 else float((1-singular[0]**2)/beta**2),
            "hilbert_schmidt_squared":float(np.sum(abs(matrix)**2))}


def energy_cross_matrix(joint,action,beta,levels):
    """공급한 길이 에너지의 교차항을 입력 내적의 에르미트 행렬로 표현한다."""
    levels=np.asarray(levels,dtype=float)
    if levels.shape!=(joint.shape[1],) or not np.all(np.isfinite(levels)) or np.any(levels<=0):
        raise ValueError("검사 에너지는 입력 길이마다 유한한 양수여야 한다")
    pe,py=joint.sum(axis=0),joint.sum(axis=1)
    matrix=joint*np.exp(1j*beta*action)/np.sqrt(py[:,None]*pe[None,:])
    average=joint@levels/py
    gram=matrix.conj().T@matrix
    cross=levels[:,None]*gram+gram*levels[None,:]-2*matrix.conj().T@(average[:,None]*matrix)
    return cross,average


def energy_state_budget(joint,action,beta,levels,source):
    """주변 내적에서 주어진 입력의 대각·교차 에너지를 직접 공동 합으로 계산한다."""
    pe,py=joint.sum(axis=0),joint.sum(axis=1)
    source=np.asarray(source,dtype=complex)
    norm=float(np.sum(pe*abs(source)**2))
    if not np.isfinite(norm) or norm<=0:
        raise ValueError("영이 아닌 유한 노름의 검사 입력이 필요하다")
    source=source/np.sqrt(norm)
    state=np.exp(1j*beta*action)*source[None,:]
    target=np.sum(joint*state,axis=1)/py
    remainder=state-target[:,None]
    total=float(np.sum(pe*levels*abs(source)**2))
    kept=float(np.sum(joint*levels[None,:]*abs(target[:,None])**2))
    folded=float(np.sum(joint*levels[None,:]*abs(remainder)**2))
    cross=float(2*np.real(np.sum(joint*levels[None,:]*target[:,None].conj()*remainder)))
    norm_error=abs(np.sum(py*abs(target)**2)+np.sum(joint*abs(remainder)**2)-1)
    return {"total":total,"kept":kept,"remainder":folded,"cross":cross,
            "budget_error":abs(total-kept-folded-cross),"norm_error":float(norm_error),
            "diagonal_state_energy_change":kept+folded-total}


def energy_check(joint,action,beta,levels):
    """교차항의 극값·위상독립 대각항과 직접 에너지 장부를 대조한다."""
    pe,py=joint.sum(axis=0),joint.sum(axis=1)
    cross,average=energy_cross_matrix(joint,action,beta,levels)
    hermitian_error=float(np.max(abs(cross-cross.conj().T)))
    values,vectors=np.linalg.eigh((cross+cross.conj().T)/2)
    states=[energy_state_budget(joint,action,beta,levels,vectors[:,i]/np.sqrt(pe)) for i in (0,-1)]
    diagonal=2*np.sum(joint**2*(levels[None,:]-average[:,None])/py[:,None],axis=0)/pe
    witness=energy_state_budget(joint,action,beta,levels,1+levels)
    variance=joint@(levels**2)/py-average**2
    expected=float(2*np.sum(py*(1+average)*variance)/np.sum(pe*(1+levels)**2))
    return {"beta":beta,"minimum":float(values[0]),"maximum":float(values[-1]),
            "hermitian_error":hermitian_error,
            "diagonal_phase_error":float(np.max(abs(np.diag(cross)-diagonal))),
            "spectral_error":max(abs(states[j]["cross"]-values[i]) for j,i in enumerate((0,-1))),
            "states":states,"positive_input":witness,
            "zero_phase_identity_error":None if beta!=0 else abs(witness["cross"]-expected)}


def mesh_check(size=64,h=1.,kind="length"):
    mesh=joint_mesh(size,h,kind)
    joint,action=mesh["joint"],mesh["action"]
    result=interaction_budget(joint,action)
    final_result=interaction_budget(joint,mesh["final_action"])
    row=np.linspace(-.4,.7,size);column=np.linspace(.3,-.2,size)**2
    shifted=interaction_budget(joint,action+row[:,None]+column[None,:])
    separable=row[:,None]+column[None,:]
    separated=transfer_check(joint,separable,5.)
    levels=1+(np.arange(size)+.5)/size
    energy=[energy_check(joint,action,beta,levels) for beta in (0.,1.,5.)]
    original,_=energy_cross_matrix(joint,action,5.,levels)
    shifted_energy,_=energy_cross_matrix(joint,action+row[:,None]+column[None,:],5.,levels)
    phase=np.exp(5j*column)
    constant,_=energy_cross_matrix(joint,action,5.,np.full(size,2.))
    return {"size":size,"h":h,"kind":kind,"volume":mesh["volume"],
            "volume_error":mesh["volume_error"],"minimum_gram":mesh["minimum_gram"],
            **result,"full_action_coefficient":final_result["coefficient"],
            "boundary_phase_coefficient_error":abs(shifted["coefficient"]-result["coefficient"]),
            "separable_coefficient":interaction_budget(joint,separable)["coefficient"],
            "separable_norm_squared":separated["norm_squared"],
            "transfer":[transfer_check(joint,action,beta) for beta in (0.,.025,.05,.1,.2,1.,5.)],
            "energy_probe":"epsilon=1+e/L; 각 입력 셀 중점의 검사 에너지이며 물리 해밀토니안은 미유도",
            "energy":energy,"constant_energy_cross_error":float(np.max(abs(constant))),
            "boundary_phase_energy_error":float(np.max(abs(shifted_energy-phase.conj()[:,None]*original*phase[None,:])))}


def run():
    geometry=[geometry_check(h) for h in (.5,.9,1.,1.1)]
    relations=[canonical_relation_check(h) for h in (.9,1.,1.1)]
    grids=[mesh_check(size,kind=kind) for kind in ("length","squared") for size in (32,64,128)]
    parent=json.loads(Path(previous.__file__).with_suffix(".json").read_text(encoding="utf-8"))
    dependencies={Path(previous.__file__).name:hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
                  **parent["dependencies"]}
    for name,sha in dependencies.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha:
            raise ValueError("선행 소스 해시 불일치: "+name)
    return {"status":"[산출]","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies":dependencies,"geometry":geometry,"relations":relations,"grids":grids,
            "scope":"공급 공동 측도에서 실제 두 경계 작용의 조건부 읽기·잔여 진폭 보존과 검사 에너지의 교차항",
            "unfinished":["실제 축약의 양자화·물리 측도·초기 준비·에너지 장부",
                          "미시 접힘·공통 계량·GR·암흑부문·허블 텐션"]}


if __name__=="__main__":
    report=run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"geometry":report["geometry"],"relations":report["relations"],
                      "grids":[{"size":g["size"],"kind":g["kind"],"coefficient":g["coefficient"],
                                "volume_error":g["volume_error"],
                                "weak_ratio":g["transfer"][1]["weak_phase_ratio"],
                                "norm_beta5":g["transfer"][-1]["norm_squared"],
                                "budget_error":max(r["budget_error"] for r in g["transfer"]),
                                "energy_beta5":{k:g["energy"][-1][k] for k in ("minimum","maximum","spectral_error")}}
                               for g in report["grids"]]},ensure_ascii=False))
