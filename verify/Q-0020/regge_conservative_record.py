"""공급한 조건부 사영의 기록 오차와 양의 배터리 에너지를 검산한다.

물리 해밀토니안·국소 시간·배터리 준비를 유도하는 모듈은 아니다.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

import regge_conditional_dynamics as previous

BATTERY_VARIANCE = 1/12-1/(2*math.pi**2)


def sine_defect(gap,width):
    """사인 배터리의 1-자기상관. 작은 에너지 차이의 뺄셈 소실을 피한다."""
    if not np.isfinite(width) or width<=0:
        raise ValueError("배터리 폭은 유한한 양수여야 한다")
    r=np.minimum(np.abs(np.asarray(gap,dtype=float))/width,1.)
    direct=1-(1-r)*np.cos(math.pi*r)-np.sin(math.pi*r)/math.pi
    series=math.pi**2*r**2/2-math.pi**2*r**3/3-math.pi**4*r**4/24+math.pi**4*r**5/30
    return np.where(r<1e-3,series,direct)


def sine_wave(b,width):
    b=np.asarray(b,dtype=float)
    return np.where((b>1)&(b<1+width),np.sqrt(2/width)*np.sin(math.pi*(b-1)/width),0.)


def overlap_check(width):
    errors=[]
    for gap in (0.,.2,.7,1.,width,1.1*width):
        right=1+width-gap
        direct=0. if right<=1 else quad(lambda b:float(sine_wave(b,width)*sine_wave(b+gap,width)),1,right,epsabs=1e-12)[0]
        errors.append(abs(direct-(1-float(sine_defect(gap,width)))))
    moments=[quad(lambda b:b**order*float(sine_wave(b,width))**2,1,1+width,epsabs=1e-12)[0] for order in (0,1,2)]
    return {"width":width,"overlap_error":max(errors),"normalization_error":abs(moments[0]-1),
            "mean_error":abs(moments[1]-(1+width/2)),
            "variance_error":abs(moments[2]-moments[1]**2-BATTERY_VARIANCE*width**2)}


def core_noise(width,kind):
    if kind=="length":
        density=lambda gap:2*(1-gap)
    elif kind=="squared":
        density=lambda gap:4/3*(1-gap)**2*(2+gap)
    else:
        raise ValueError("정의되지 않은 공동측도")
    return float(quad(lambda gap:float(sine_defect(gap,width))*density(gap),0,1,epsabs=1e-13)[0])


def fiber_operators(probability,levels,width):
    u=np.sqrt(probability)
    projection=np.outer(u,u)
    effect=projection*(1-sine_defect(levels[:,None]-levels[None,:],width))
    fu=effect@u
    noise=effect+projection-np.outer(fu,u)-np.outer(u,fu)
    gamma=float(np.sum(probability[:,None]*probability[None,:]*sine_defect(levels[:,None]-levels[None,:],width)))
    return projection,effect,noise,gamma


def mesh_check(size=32,kind="length"):
    mesh=previous.joint_mesh(size,h=1.,kind=kind)
    joint,action=mesh["joint"],mesh["action"]
    py=joint.sum(axis=1)
    levels=1+(np.arange(size)+.5)/size
    rows=[]
    for width in (2.,4.,8.,16.):
        maximum=0.; noise_inputs={beta:0. for beta in (0.,1.,5.)}
        spectrum_error=0.; minimum_effect=1.
        for j in range(size):
            active=joint[j]>0
            probability=joint[j,active]/py[j]
            energy=levels[active]
            p,f,b,gamma=fiber_operators(probability,energy,width)
            maximum=max(maximum,gamma)
            for beta in noise_inputs:
                state=np.sqrt(probability)*np.exp(1j*beta*action[j,active])
                noise_inputs[beta]+=py[j]*float(np.real(np.vdot(state,b@state)))
            if j==0:
                spectrum_error=abs(np.linalg.eigvalsh(b)[-1]-gamma)
                minimum_effect=float(np.linalg.eigvalsh(f)[0])
        exact=core_noise(width,kind)
        sigma2=1/12 if kind=="length" else 1/18
        rows.append({"width":width,"worst_noise_squared":maximum,"continuum_noise_squared":exact,
                     "continuum_difference":abs(maximum-exact),"upper_bound_squared":math.pi**2*sigma2/width**2,
                     "input_noise_squared":noise_inputs,"core_spectrum_error":spectrum_error,
                     "core_minimum_effect":minimum_effect})
    return {"size":size,"kind":kind,"record":rows}


def resource_table():
    rows=[]
    for kind,sigma2,variance in (("length",1/12,7/60),("squared",1/18,9/100)):
        for target in (.1,.01,.001):
            width=math.pi*math.sqrt(sigma2)/target
            rows.append({"kind":kind,"target_rms":target,"apparatus_std_necessary":math.sqrt(max(0.,sigma2/(4*target**2)-variance)),
                         "battery_width_sufficient":width,"battery_std_sufficient":math.sqrt(BATTERY_VARIANCE)*width,
                         "battery_mean":1+width/2,"constructed_rms":math.sqrt(core_noise(width,kind)),
                         "reference_trace_distance_bound":math.sqrt(2)*target})
    return rows


def finite_battery_check(size=4,packet_size=16,kind="length"):
    """양의 유한 배터리에서 전체 에너지별 단위 전달을 직접 구성한다."""
    levels=1+(np.arange(size)+.5)/size
    probability=np.ones(size) if kind=="length" else 2*np.arange(size)+1.
    probability=probability/probability.sum()
    u=np.sqrt(probability);p=np.outer(u,u);q=np.eye(size)-p
    swap=np.array([[0.,1.],[1.,0.]])
    ideal=np.kron(p,np.eye(2))+np.kron(q,swap)
    battery_count=packet_size+3*size+1
    dimension=2*size*battery_count
    unitary=np.eye(dimension)
    for total in range(size,battery_count):
        ids=[(i*2+r)*battery_count+total-i for i in range(size) for r in range(2)]
        unitary[np.ix_(ids,ids)]=ideal
    packet=np.zeros(battery_count)
    packet[size+1:size+packet_size+1]=np.sqrt(2/(packet_size+1))*np.sin(math.pi*np.arange(1,packet_size+1)/(packet_size+1))
    insertion=np.zeros((dimension,size))
    ideal_insertion=np.zeros_like(insertion)
    for i in range(size):
        insertion[i*2*battery_count:(i*2+1)*battery_count,i]=packet
        for j in range(size):
            for r in range(2):
                ideal_insertion[(i*2+r)*battery_count:(i*2+r+1)*battery_count,j]=ideal[i*2+r,j*2]*packet
    output=unitary@insertion
    pointer=np.tile(np.repeat([1.,0.],battery_count),size)
    system_energy=np.repeat(levels,2*battery_count)
    battery_energy=np.tile(np.arange(battery_count)/size,2*size)
    total_energy=system_energy+battery_energy
    correlations=np.array([np.dot(packet[:battery_count-d],packet[d:]) for d in range(size)])
    effect=p*correlations[np.abs(np.arange(size)[:,None]-np.arange(size)[None,:])]
    noise=effect+p-effect@p-p@effect
    actual_noise=pointer[:,None]*output-output@p
    difference=output-ideal_insertion
    mean=float(probability@levels);sigma=float(np.sqrt(probability@(levels-mean)**2))
    witness=(u+1j*(levels-mean)*u/sigma)/np.sqrt(2)
    exchange_input=(u+(levels-mean)*u/sigma)/np.sqrt(2)
    state=output@exchange_input
    input_mean=float(np.sum(abs(exchange_input)**2*levels))
    battery_mean=float(np.sum(packet**2*np.arange(battery_count)/size))
    system_change=float(np.sum(abs(state)**2*system_energy)-input_mean)
    battery_change=float(np.sum(abs(state)**2*battery_energy)-battery_mean)
    witness_mean=float(np.sum(abs(witness)**2*levels))
    system_variance=float(np.sum(abs(witness)**2*(levels-witness_mean)**2))
    battery_variance=float(np.sum(packet**2*(np.arange(battery_count)/size-battery_mean)**2))
    observed=float(np.real(np.vdot(witness,noise@witness)))
    return {"size":size,"packet_size":packet_size,"kind":kind,"minimum_battery_energy":float(min(battery_energy)),
            "unitary_error":float(np.max(abs(unitary.T@unitary-np.eye(dimension)))),
            "conservation_error":float(np.max(abs(unitary*total_energy[None,:]-total_energy[:,None]*unitary))),
            "effect_error":float(np.max(abs(output.T@(pointer[:,None]*output)-effect))),
            "noise_error":float(np.max(abs(actual_noise.T@actual_noise-noise))),
            "isometry_difference_error":float(np.max(abs(difference.T@difference-2*noise))),
            "noise_squared":float(np.linalg.eigvalsh(noise)[-1]),"witness_noise_squared":observed,
            "way_lower_bound":sigma**2/(4*(system_variance+battery_variance)),
            "exchange_input":"평균 성분과 중심 에너지 성분의 실수 중첩; 교환자 최악 상태와 구별",
            "system_energy_change":system_change,"battery_energy_change":battery_change,
            "energy_budget_error":abs(system_change+battery_change)}


def repeated_record_check(size=4,packet_size=16,kind="length",count=3):
    """같은 계·배터리의 순차 기록과 외부 기록 보존 뒤의 역전을 검사한다."""
    if size<2 or packet_size<1 or count<1 or count>8:
        raise ValueError("검산 크기 또는 기록 수가 허용 범위 밖이다")
    if kind not in ("length","squared"):
        raise ValueError("정의되지 않은 공동측도")
    levels=1+(np.arange(size)+.5)/size
    probability=np.ones(size) if kind=="length" else 2*np.arange(size)+1.
    probability=probability/probability.sum()
    u=np.sqrt(probability);p=np.outer(u,u);q=np.eye(size)-p
    nb=packet_size+3*size+1; nr=2**count
    packet=np.zeros(nb)
    packet[size+1:size+packet_size+1]=np.sqrt(2/(packet_size+1))*np.sin(math.pi*np.arange(1,packet_size+1)/(packet_size+1))
    initial=np.zeros((size,nb,nr,size),complex)
    for i in range(size):
        initial[i,:,0,i]=packet

    def gate(state,bit):
        output=state.copy()
        flipped=np.arange(nr)^(1<<bit)
        for total in range(size,nb):
            b=total-np.arange(size)
            block=state[np.arange(size),b]
            output[np.arange(size),b]=np.einsum("ij,jra->ira",p,block)+np.einsum("ij,jra->ira",q,block[:,flipped])
        return output

    first=gate(initial,0)
    output=initial.copy()
    for bit in range(count):
        output=gate(output,bit)
    a0=output[:,:,0,:].reshape(size*nb,size)
    a1=output[:,:,-1,:].reshape(size*nb,size)
    first0=first[:,:,0,:].reshape(size*nb,size)
    first1=first[:,:,1,:].reshape(size*nb,size)
    flat=output.reshape(-1,size)
    hsystem=np.repeat(levels,nb)
    hbattery=np.tile(np.arange(nb)/size,size)
    meanb=float(packet**2@np.arange(nb)/size)
    es=a0.conj().T@(hsystem[:,None]*a0)+a1.conj().T@(hsystem[:,None]*a1)-np.diag(levels)
    eb=a0.conj().T@(hbattery[:,None]*a0)+a1.conj().T@(hbattery[:,None]*a1)-meanb*np.eye(size)
    # 오차는 두 기록 가지 모두의 M V - V P 성분을 포함한다.
    noise0=a0-a0@p; noise1=-a1@p
    gram=noise0.conj().T@noise0+noise1.conj().T@noise1
    reference0=np.einsum("ij,b->ibj",p,packet).reshape(size*nb,size)
    reference1=np.einsum("ij,b->ibj",q,packet).reshape(size*nb,size)
    difference0=a0-reference0; difference1=a1-reference1
    iso_difference=difference0.conj().T@difference0+difference1.conj().T@difference1
    inverse=output.copy()
    for bit in reversed(range(count)):
        inverse=gate(inverse,bit)
    effect=a0.conj().T@a0
    eigenvalues,vectors=np.linalg.eigh(effect)
    weight=(.5-eigenvalues[0])/(eigenvalues[-1]-eigenvalues[0])
    if not 0<weight<1:
        raise ValueError("균형 기록의 시험 상태가 없다")
    balanced=np.sqrt(1-weight)*vectors[:,0]+1j*np.sqrt(weight)*vectors[:,-1]
    z=np.einsum("ibra,a->ibr",initial,balanced)
    copied=[]
    for label in (0,nr-1):
        branch=np.zeros_like(output)
        branch[:,:,label,:]=output[:,:,label,:]
        for bit in reversed(range(count)):
            branch=gate(branch,bit)
        copied.append(branch)
    fidelity=sum(abs(np.vdot(z,np.einsum("ibra,a->ibr",branch,balanced)))**2 for branch in copied)
    pzero=float(np.real(np.vdot(balanced,effect@balanced)))
    archive_energy=np.zeros((size,size),complex)
    total=np.repeat(hsystem+hbattery,nr)
    for branch in copied:
        bflat=branch.reshape(-1,size)
        archive_energy+=bflat.conj().T@(total[:,None]*bflat)
    reset_error=max(float(np.max(abs(branch[:,:,1:,:]))) for branch in copied)
    archive_sb_error=max(float(np.max(abs(copied[0][:,:,0,:].reshape(size*nb,size)-a0))),
                         float(np.max(abs(copied[1][:,:,0,:].reshape(size*nb,size)-a1))))
    mean=float(probability@levels);sigma=float(np.sqrt(probability@(levels-mean)**2))
    exchange=(u+(levels-mean)*u/sigma)/np.sqrt(2)
    system_change=float(np.real(np.vdot(exchange,es@exchange)))
    battery_change=float(np.real(np.vdot(exchange,eb@exchange)))
    mixed=[r for r in range(nr) if r not in (0,nr-1)]
    mixed_norm=0. if not mixed else float(np.linalg.norm(output[:,:,mixed,:].reshape(-1,size),ord=2)**2)
    return {"size":size,"packet_size":packet_size,"kind":kind,"count":count,
            "isometry_error":float(np.max(abs(flat.conj().T@flat-np.eye(size)))),
            "mixed_record_probability_bound":mixed_norm,
            "same_as_single_error":max(float(np.max(abs(a0-first0))),float(np.max(abs(a1-first1)))),
            "energy_operator_error":float(np.max(abs(es+eb))),
            "noise_squared":float(np.linalg.eigvalsh(gram)[-1]),
            "reference_gram_error":float(np.max(abs(iso_difference-2*gram))),
            "system_energy_change":system_change,"battery_energy_change":battery_change,
            "full_inverse_error":float(np.max(abs(inverse-initial))),
            "archive_probability":pzero,"archive_recovery_fidelity":float(fidelity),
            "archive_formula_error":abs(float(fidelity)-(pzero**2+(1-pzero)**2)),
            "archive_record_reset_error":reset_error,"archive_sb_state_error":archive_sb_error,
            "archive_energy_operator_error":float(np.max(abs(archive_energy-np.diag(levels)-meanb*np.eye(size))))}


def run():
    parent=json.loads(Path(previous.__file__).with_suffix(".json").read_text(encoding="utf-8"))
    dependencies={Path(previous.__file__).name:hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),**parent["dependencies"]}
    if dependencies[Path(previous.__file__).name]!=parent["source_sha256"]:
        raise ValueError("선행 산출물의 소스 해시 불일치")
    for name,digest in dependencies.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError("선행 소스 해시 불일치: "+name)
    return {"status":"[산출]","source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"dependencies":dependencies,
            "overlap":[overlap_check(width) for width in (2.,4.,8.,16.)],
            "grids":[mesh_check(size,kind) for kind in ("length","squared") for size in (16,32,64)],
            "finite_battery":[finite_battery_check(size,packet,kind) for kind in ("length","squared") for size,packet in ((3,8),(4,16))],
            "repeated_record":[repeated_record_check(size,packet,kind,count) for kind in ("length","squared") for size,packet in ((3,8),(4,16)) for count in (1,2,3,5)],
            "resources":resource_table(),"asymptotic_resource_ratio":math.sqrt(math.pi**2/3-2),
            "scope":"공급된 길이 에너지와 공동 사영의 단회·동일 계 반복 기록. 양의 배터리, 기록 상관과 외부 복사 뒤의 환류",
            "unfinished":["실제 미시 작용·물리 시간·배터리 준비·국소 기록 결합·원래 정준 제약과의 양립",
                          "독립 새 계의 투입·외부 기록 초기화·충전, 0D 보충·공통 계량·GR·모든 힘·암흑부문·허블 텐션"]}


if __name__=="__main__":
    result=run()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"finite_battery":result["finite_battery"],"resources":result["resources"],
                      "grids":[{"size":g["size"],"kind":g["kind"],"rms_width16":math.sqrt(g["record"][-1]["worst_noise_squared"])} for g in result["grids"]]},ensure_ascii=False))
