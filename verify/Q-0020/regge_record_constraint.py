"""양의 보존 기록이 실제 사전 제약의 공급 영모드를 보존하는지 검사한다.

한 주기 사전 제약의 검사이며 전체 중력 물리사영이나 미시 에너지를 도출하지 않는다.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import regge_conservative_record as previous

geometry=previous.previous
constraint=geometry.previous.transfer


def fiber(size=64,y_ratio=1.,beta=5.,kind="length",target="conditional"):
    if size<2 or target not in ("conditional","constraint") or kind not in ("length","squared"):
        raise ValueError("섬유 격자·측도·기록 표적을 확인해야 한다")
    y=y_ratio*geometry.Y0
    length=float(geometry.e_limit(y,1.))
    upper=geometry.full.limit(1.)
    edge=(np.arange(size)+.5)*length/size
    lengths=np.tile(constraint.geometry_lengths(1.),(size,1))
    lengths[:,geometry.E]=edge;lengths[:,geometry.Y]=y
    final,_=geometry.batch_action(geometry.moves.FINAL,lengths)
    ids=geometry.moves.FINAL.indices(geometry.moves.MOVE.new.edges)
    middle,_=geometry.batch_action(geometry.moves.MOVE.new,lengths[:,ids])
    action=final-middle
    j=np.exp(-1j*beta*action)/np.sqrt(size)
    probability=np.ones(size)/size if kind=="length" else (2*np.arange(size)+1)/size**2
    a=np.sqrt(probability).astype(complex) if target=="conditional" else j.copy()
    return {"j":j,"target":a,"levels":1+edge/upper,"action":action,
            "length_ratio":length/upper,"lengths":lengths}


def autocorrelation_budget(j,target,gap_step,defect):
    """파동 이동의 자기상관으로 계산하되 1에 가까운 수의 상쇄를 피한다."""
    v=np.conj(target)*j
    coefficients=np.correlate(np.conj(v),np.conj(v),mode="full")
    gaps=np.arange(-len(j)+1,len(j))*gap_step
    p=float(abs(np.vdot(target,j))**2)
    first=float(np.real(np.dot(coefficients,defect(gaps))))
    second=float(np.real(np.vdot(coefficients,defect(gaps[:,None]-gaps[None,:])@coefficients)))
    q=p-first
    k2=p*p-second
    leakage=2*(p-p*p-first+second)
    return {"ideal_overlap":p,"ideal_leakage":2*p*(1-p),"record_probability":q,
            "retained_first_branch_norm":k2,"leakage":leakage}


def adapted_continuum(width,alpha=1.):
    """균등 길이의 두 에너지 차이와 네 에너지 합성 분포를 독립 적분한다."""
    first=2*quad(lambda x:(1-x)*float(previous.sine_defect(alpha*x,width)),0,1,epsabs=1e-13)[0]
    def density(x):
        return 2/3-x*x+x**3/2 if x<=1 else (2-x)**3/6
    second=2*sum(quad(lambda x:density(x)*float(previous.sine_defect(alpha*x,width)),lo,hi,epsabs=1e-13)[0] for lo,hi in ((0,1),(1,2)))
    return {"leakage":2*(second-first),"upper_bound":math.pi**2*alpha**2/(6*width**2),
            "scaled":2*width**2*(second-first),"asymptotic":math.pi**2*alpha**2/6}


def grid_check(size=64,y_ratio=1.,beta=5.,kind="length",target="conditional"):
    data=fiber(size,y_ratio,beta,kind,target)
    alpha=data["length_ratio"]
    rows=[]
    for width in (2.,8.,32.,128.):
        result=autocorrelation_budget(data["j"],data["target"],alpha/size,
                                     lambda d:previous.sine_defect(d,width))
        result.update({"width":width})
        if target=="constraint":
            continuous=adapted_continuum(width,alpha)
            result.update({"continuum_leakage":continuous["leakage"],
                           "continuum_difference":abs(result["leakage"]-continuous["leakage"]),
                           "upper_bound":continuous["upper_bound"]})
        rows.append(result)
    return {"size":size,"y_ratio":y_ratio,"beta":beta,"kind":kind,"target":target,
            "length_ratio":alpha,"rows":rows}


def direct_wave_check(size=12,y_ratio=1.,beta=5.,kind="squared",target="conditional",width=8.):
    data=fiber(size,y_ratio,beta,kind,target)
    j,a,levels=data["j"],data["target"],data["levels"]
    gaps=levels[:,None]-levels[None,:]
    # 전체 e별 배터리 파동을 적분한다. 자기상관 공식을 사용하지 않는다.
    boundaries=np.unique(np.r_[0.,width+2.,(1-gaps).ravel(),(1+width-gaps).ravel()])
    boundaries=boundaries[(boundaries>=0)&(boundaries<=width+2)]
    nodes,weights=np.polynomial.legendre.leggauss(12)
    b=((boundaries[1:]+boundaries[:-1])[:,None]+np.diff(boundaries)[:,None]*nodes)/2
    measure=(np.diff(boundaries)[:,None]*weights/2).ravel()
    b=b.ravel()
    translated=previous.sine_wave(b[:,None,None]+gaps[None,:,:],width)
    first=a[None,:]*np.einsum("bij,j->bi",translated,np.conj(a)*j)
    second=previous.sine_wave(b,width)[:,None]*j-first
    projected_first=first@np.conj(j)
    projected_second=second@np.conj(j)
    leakage=float(measure@(np.sum(abs(first-projected_first[:,None]*j)**2,axis=1)+
                           np.sum(abs(second-projected_second[:,None]*j)**2,axis=1)))
    norm=float(measure@(np.sum(abs(first)**2+abs(second)**2,axis=1)))
    formula=autocorrelation_budget(j,a,data["length_ratio"]/size,
                                  lambda d:previous.sine_defect(d,width))
    return {"target":target,"kind":kind,"width":width,"leakage":leakage,
            "formula_error":abs(leakage-formula["leakage"]),"norm_error":abs(norm-1)}


def finite_block_check(size=4,packet_size=16,beta=5.,kind="squared",target="conditional"):
    data=fiber(size,1.,beta,kind,target)
    j,a,levels=data["j"],data["target"],data["levels"]
    nb=packet_size+3*size+1
    projection=np.eye(size*nb,dtype=complex)
    local=np.outer(a,np.conj(a))
    for total in range(size,nb):
        ids=np.arange(size)*nb+total-np.arange(size)
        projection[np.ix_(ids,ids)]=local
    complement=np.eye(size*nb)-projection
    packet=np.zeros(nb)
    packet[size+1:size+packet_size+1]=np.sqrt(2/(packet_size+1))*np.sin(math.pi*np.arange(1,packet_size+1)/(packet_size+1))
    z=np.kron(j,packet)
    branches=[(projection@z).reshape(size,nb),(complement@z).reshape(size,nb)]
    leakage=sum(float(np.linalg.norm(branch-j[:,None]*(np.conj(j)@branch)[None,:])**2) for branch in branches)
    corr=np.array([np.dot(packet[:nb-d],packet[d:]) for d in range(2*size-1)])
    formula=autocorrelation_budget(j,a,1/size,lambda gap:1-corr[np.rint(abs(gap)*size).astype(int)])
    hs=np.repeat(levels,nb);hb=np.tile(np.arange(nb)/size,size);ht=hs+hb
    output_density=sum(abs(branch.ravel())**2 for branch in branches)
    source_density=abs(z)**2
    return {"kind":kind,"target":target,"beta":beta,"leakage":leakage,
            "projector_error":float(np.max(abs(projection@projection-projection))),
            "energy_commutator_error":float(np.max(abs(projection*ht[None,:]-ht[:,None]*projection))),
            "norm_error":abs(float(output_density.sum())-1),
            "formula_error":abs(leakage-formula["leakage"]),
            "system_energy_change":float((output_density-source_density)@hs),
            "battery_energy_change":float((output_density-source_density)@hb),
            "energy_budget_error":abs(float((output_density-source_density)@ht))}


def coupled_constraint_check(y_ratio=1.,beta=5.,kind="length",fraction=.43,step=2e-5):
    """실제 작용 파동의 계 미분과 배터리 제약 기여를 독립 차분한다."""
    y=y_ratio*geometry.Y0
    length=float(geometry.e_limit(y,1.));upper=geometry.full.limit(1.)
    edge=fraction*length;t0=2.8
    battery=t0-1-edge/upper
    template=constraint.geometry_lengths(1.);template[geometry.Y]=y
    width=1.6;lower=2.2
    def chi(t):
        return math.sqrt(2/width)*math.sin(math.pi*(t-lower)/width) if lower<t<lower+width else 0.
    def wave(e,b):
        current=template.copy();current[geometry.E]=e
        action=geometry.moves.actions(current)["second"]["action"]
        rho,_=constraint.measure(e,(0.,length),kind)
        return np.exp(-1j*beta*action)/np.sqrt(length*rho)*chi(1+e/upper+b)
    def derivative(axis):
        def central(delta):
            return ((wave(edge+delta,battery)-wave(edge-delta,battery))/(2*delta) if axis==0 else
                    (wave(edge,battery+delta)-wave(edge,battery-delta))/(2*delta))
        return (4*central(step/2)-central(step))/3
    q=template.copy();q[geometry.E]=edge
    data=geometry.moves.actions(q)["second"]
    rho,score=constraint.measure(edge,(0.,length),kind)
    value=wave(edge,battery)
    system=-1j*(derivative(0)+.5*score*value)+beta*data["gradient"][geometry.E]*value
    compensation=1j*derivative(1)/upper
    j=np.exp(-1j*beta*data["action"])/np.sqrt(length*rho)
    expected=-1j*j*math.sqrt(2/width)*math.pi/width*math.cos(math.pi*(t0-lower)/width)/upper
    # 경계식은 위상·밀도를 제거한 함수에 적용한다.
    alpha=length/upper
    shifted_boundary_error=abs(chi(1+alpha+(t0-1-alpha))-chi(t0))
    wrong_fixed_b_boundary=abs(chi(t0+alpha)-chi(t0))
    return {"y_ratio":y_ratio,"beta":beta,"kind":kind,"fraction":fraction,
            "domain_note":"계 잔차는 내부 미분식이다. 이 상관 준비는 원래 고정 b 주기 F_s의 도메인 밖이며 고정 t의 F_tot 도메인을 사용한다.",
            "system_constraint_residual":float(abs(system)),
            "system_formula_error":float(abs(system-expected)),
            "total_constraint_error":float(abs(system+compensation)),
            "shifted_boundary_error":shifted_boundary_error,
            "fixed_b_boundary_defect":wrong_fixed_b_boundary,
            "minimum_boundary_battery":t0-1-alpha}


def coupled_spectrum_check(size=7,beta=5.):
    """고정 총에너지 표현의 제약 스펙트럼과 영모드 기록을 대조한다."""
    data=fiber(size,1.,beta,"length","constraint")
    length=geometry.full.limit(1.)
    frequencies=np.fft.fftfreq(size)*size
    unit=(np.arange(size)+.5)/size
    fourier=np.exp(2j*math.pi*unit[:,None]*frequencies[None,:])/np.sqrt(size)
    dressing=np.exp(-1j*beta*data["action"])
    modes=dressing[:,None]*fourier
    eigenvalues=2*math.pi*frequencies/length
    f=(modes*eigenvalues)@modes.conj().T
    a=np.outer(data["j"],np.conj(data["j"]))
    t=2.2+np.arange(9)/8
    total_f=np.kron(f,np.eye(len(t)))
    total_h=np.kron(np.eye(size),np.diag(t))
    total_a=np.kron(a,np.eye(len(t)))
    values=[]
    for index in (0,1,size-1):
        mode=modes[:,index]
        values.append({"mode":int(round(frequencies[index])),
                       "eigen_error":float(np.linalg.norm(f@mode-eigenvalues[index]*mode)),
                       "zero_record_probability":float(np.linalg.norm(a@mode)**2)})
    return {"hermitian_error":float(np.max(abs(f-f.conj().T))),
            "constraint_energy_commutator_error":float(np.max(abs(total_f@total_h-total_h@total_f))),
            "record_constraint_commutator_error":float(np.max(abs(total_a@total_f-total_f@total_a))),
            "record_energy_commutator_error":float(np.max(abs(total_a@total_h-total_h@total_a))),
            "modes":values}


def run():
    parent=json.loads(Path(previous.__file__).with_suffix(".json").read_text(encoding="utf-8"))
    dependencies={Path(previous.__file__).name:hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),**parent["dependencies"]}
    if dependencies[Path(previous.__file__).name]!=parent["source_sha256"]:
        raise ValueError("선행 산출물의 소스 해시 불일치")
    for name,digest in dependencies.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError("선행 소스 해시 불일치: "+name)
    geometry_errors=[]
    for ratio in (.75,1.,1.03):
        data=fiber(12,ratio)
        geometry_errors.extend(abs(geometry.moves.actions(data["lengths"][i])["second"]["action"]-data["action"][i]) for i in (1,5,10))
    return {"status":"[산출]","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies":dependencies,"geometry_error":float(max(geometry_errors)),
            "grids":[grid_check(n,y,beta,kind,target) for target in ("conditional","constraint")
                     for kind in ("length","squared") for y in (.75,1.,1.03)
                     for beta in (0.,1.,5.) for n in (32,64,128)],
            "direct_wave":[direct_wave_check(target=target,kind=kind,width=w)
                           for target in ("conditional","constraint") for kind in ("length","squared") for w in (2.,8.)],
            "finite_blocks":[finite_block_check(kind=kind,target=target,beta=beta)
                             for target in ("conditional","constraint") for kind in ("length","squared") for beta in (0.,5.)],
            "adapted_continuum":[{"width":w,**adapted_continuum(w)} for w in (2.,8.,32.,128.,512.)],
            "exact_conditional_beta0_squared_limit":16/81,
            "coupled_constraint":[coupled_constraint_check(y,beta,kind,fraction) for y in (.75,1.,1.03)
                                  for beta in (0.,5.) for kind in ("length","squared") for fraction in (.2,.6,.85)],
            "coupled_spectrum":coupled_spectrum_check(),
            "scope":"한 사전 제약의 공급 주기 영모드. 전체 물리사영·원래 두 제약의 공동 축약과 구별한다.",
            "unfinished":["장치 포함 제약 또는 실제 두 제약 축약에서 에너지·기록의 양립",
                          "같은 미시작용의 원천·0D 보충·공통 계량·GR·모든 힘·암흑부문·허블 텐션"]}


if __name__=="__main__":
    result=run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"geometry_error":result["geometry_error"],"finite_blocks":result["finite_blocks"],
                      "adapted_continuum":result["adapted_continuum"]},ensure_ascii=False))
