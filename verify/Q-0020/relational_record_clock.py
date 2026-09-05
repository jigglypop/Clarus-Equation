"""실제 레게 섬유의 보존 기록을 무시간 위치시계 결합으로 옮긴다.

시계·입사 방향·기록 비트·배터리 준비와 검사 에너지는 공급값이다.
원래 레게 사전 제약의 보존이나 0D 시간 발생을 유도하지 않는다.
모든 에너지는 E_star 단위이며 간격 g=E-lambda_max를 직접 저장한다.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import regge_record_constraint as record
import relational_split_clock as clock

HERE=Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def source(size,kind,target,beta):
    return record.fiber(size=size,y_ratio=1.,beta=beta,kind=kind,target=target)


class RecordClock:
    def __init__(self,size=4,packet_size=16,kind="length",target="constraint",beta=5.):
        if isinstance(size,bool) or not isinstance(size,int) or size<2:
            raise ValueError("계의 준위 수는 2 이상의 정수여야 한다")
        if isinstance(packet_size,bool) or not isinstance(packet_size,int) or packet_size<2:
            raise ValueError("배터리의 점유 준위 수는 2 이상의 정수여야 한다")
        if not np.isfinite(beta):
            raise ValueError("작용 계수는 유한해야 한다")
        data=source(size,kind,target,float(beta))
        self.n,self.m=size,packet_size
        self.kind,self.target,self.beta=kind,target,float(beta)
        self.u=np.asarray(data["target"],dtype=complex)
        self.levels=np.asarray(data["levels"],dtype=float)
        self.delta=float(data["length_ratio"]/size)
        self.totals=np.arange(size+1,2*size+packet_size)
        self.lambdas=self.levels[0]+self.delta*self.totals
        self.offset=self.delta*(self.totals[-1]-self.totals)
        self.battery_indices=self.totals[:,None]-np.arange(size)[None,:]
        index=self.battery_indices-size
        self.packet=np.where((index>=1)&(index<=packet_size),
                             np.sqrt(2/(packet_size+1))*np.sin(math.pi*index/(packet_size+1)),0.)
        self.action=np.asarray(data["action"])
        self.shape=(len(self.totals),size,2)

    def prepare(self,psi):
        psi=np.asarray(psi,dtype=complex)
        if psi.shape!=(self.n,) or not np.all(np.isfinite(psi)) or abs(np.vdot(psi,psi)-1)>1e-10:
            raise ValueError("정규화된 유한 계 진폭이 필요하다")
        state=np.zeros(self.shape,dtype=complex)
        state[:,:,0]=self.packet*psi[None,:]
        return state

    def q(self,block):
        coefficient=np.einsum("i,ti->t",self.u.conj(),block)
        return block-coefficient[:,None]*self.u[None,:]

    def generator(self,state):
        component=math.pi/2*self.q(state[:,:,0]-state[:,:,1])
        return np.stack((component,-component),axis=-1)

    def rotate(self,x,state):
        if not np.isfinite(x):
            raise ValueError("시계 위치는 유한해야 한다")
        fraction=float(clock.autonomous.profile(x)[0])
        return state+np.expm1(-1j*math.pi*fraction)/math.pi*self.generator(state)

    def kinetic(self,gap):
        if not np.isfinite(gap) or gap<=0:
            raise ValueError("활성 총에너지의 문턱 간격은 유한한 양수여야 한다")
        return gap+self.offset

    def state(self,x,psi,gap,mass=1.):
        if not np.isfinite(mass) or mass<=0:
            raise ValueError("시계 질량은 유한한 양수여야 한다")
        momentum=np.sqrt(2*mass*self.kinetic(gap))
        free=self.prepare(psi)*(np.exp(1j*momentum*x)/np.sqrt(momentum/mass))[:,None,None]
        return self.rotate(x,free)

    def stats(self,psi,gap):
        kinetic=self.kinetic(gap)
        initial=self.prepare(psi)
        a=initial[:,:,0]
        probability=np.sum(abs(a)**2,axis=1)
        rejected=np.sum(abs(self.q(a))**2,axis=1)
        qflux=float(sum(rejected))
        # 공통 sqrt(g)를 곱해 문턱 근처의 큰 가중치를 만들지 않는다.
        weights=np.sqrt(gap/kinetic)
        normalization=float(weights@probability)
        qposition=float(weights@rejected/normalization)
        tilted=weights*probability/normalization
        ratio=math.sqrt(kinetic[0]/gap)
        output=self.rotate(1.,initial)
        system=self.levels[None,:,None]
        battery=self.delta*self.battery_indices[:,:,None]
        change=abs(output)**2-abs(initial)**2
        delta_system=float(np.sum(change*system))
        delta_battery=float(np.sum(change*battery))
        local_change=change*weights[:,None,None]/normalization
        pointer_probability=np.sum(abs(output[:,:,1])**2,axis=1)
        probabilities=abs(self.u)**2
        mean=float(probabilities@np.arange(self.n))
        variance=float(probabilities@(np.arange(self.n)-mean)**2)
        return {
            "gap":gap,"lambda_min":float(self.lambdas[0]),"lambda_max":float(self.lambdas[-1]),
            "q_flux":qflux,"q_position":qposition,"position_flux_difference":abs(qposition-qflux),
            "weight_condition":ratio,"position_bound":min(1.,ratio*qflux),
            "total_variation":float(sum(abs(tilted-probability))/2),
            "sharp_weight_bound":math.tanh(math.log(ratio)/4),
            "target_packet_bound":4*variance*math.sin(math.pi/(2*(self.m+1)))**2,
            "target_asymptotic_bound":math.pi**2*variance/(self.m+1)**2,
            "threshold_limit_if_target":float(1-abs(self.u[-1])**2),
            "maximum_fiber_weight":float(probability[-1]),
            "minimum_prepared_battery_energy":float((self.n+1)*self.delta),
            "prepared_battery_mean":float(self.delta*(self.n+(self.m+1)/2)),
            "pointer_formula_error":float(np.max(abs(pointer_probability-rejected))),
            "norm_error":abs(float(np.linalg.norm(output)**2)-1),
            "system_energy_change":delta_system,"battery_energy_change":delta_battery,
            "energy_balance_error":abs(delta_system+delta_battery),
            "local_energy_balance_error":abs(float(np.sum(local_change*(system+battery)))),
        }


def local_audit(model,psi,gap,x,mass=1.,step=8e-4):
    function=lambda z:model.state(z,psi,gap,mass)
    value,first,second=clock.differences(function,x,step)
    _,first2,second2=clock.differences(function,x,step/2)
    first=(16*first2-first)/15
    second=(16*second2-second)/15
    fraction,slope=clock.autonomous.profile(x)
    curvature=60*x*(1-x)*(1-2*x) if 0<x<1 else 0.
    kval=model.generator(value)
    square=model.generator(kval)
    kinetic_square=-second-2j*slope*model.generator(first)-1j*curvature*kval+slope*slope*square
    residual=kinetic_square/(2*mass)-model.kinetic(gap)[:,None,None]*value
    momentum=-1j*first+slope*kval
    currents=[float(np.vdot(value[:,:,r],momentum[:,:,r]).real/mass) for r in (0,1)]
    density=float(np.vdot(value,value).real)
    kinetic_expectation=float(np.vdot(momentum,momentum).real/(2*mass*density))
    expected_kinetic=float(np.sum(abs(value)**2*model.kinetic(gap)[:,None,None])/density)
    stats=model.stats(psi,gap)
    transfer=math.sin(math.pi*float(fraction)/2)**2
    expected=[1-transfer*stats["q_flux"],transfer*stats["q_flux"]]
    source=math.pi/2*float(slope)*math.sin(math.pi*float(fraction))*stats["q_flux"]
    # 기록 전류의 공간 변화도 별도의 다섯점 차분과 비교한다.
    def current_at(z):
        state=function(z)
        p=np.sqrt(2*mass*model.kinetic(gap))
        covariant=model.rotate(z,model.prepare(psi)*
            (np.exp(1j*p*z)*p/np.sqrt(p/mass))[:,None,None])
        return np.array([np.vdot(state[:,:,r],covariant[:,:,r]).real/mass for r in (0,1)])
    _,current_derivative,_=clock.differences(current_at,x,step)
    position=float(np.sum(abs(value[:,:,1])**2)/density)
    return {
        "x":x,"gap":gap,"mass":mass,
        "constraint_relative_residual":float(np.linalg.norm(residual)/np.linalg.norm(value)),
        "record_currents":currents,"record_current_error":float(np.max(abs(np.array(currents)-expected))),
        "source_formula_error":abs(float(current_derivative[1])-source),
        "record_source":source,"local_pointer_error":abs(position-transfer*stats["q_position"]),
        "energy_error":abs(kinetic_expectation-expected_kinetic),
        "omitted_square_residual":float(np.linalg.norm(residual-slope*slope*square/(2*mass))/np.linalg.norm(value)),
    }


def matrix_control(model):
    p=np.outer(model.u,model.u.conj());q=np.eye(model.n)-p
    flip=np.array([[0.,1.],[1.,0.]])
    gate=np.kron(p,np.eye(2))+np.kron(q,flip)
    k=math.pi/2*(np.eye(2*model.n)-gate)
    psi=(model.u+np.exp(.71j)*np.roll(model.u,1))
    psi/=np.linalg.norm(psi)
    a=model.prepare(psi)
    fractions=(.0,.17,.5,.89,1.)
    rotation_error=0.
    for f in fractions:
        direct=np.einsum("ij,tj->ti",expm(-1j*f*k),a.reshape(len(model.totals),-1))
        formula=a+np.expm1(-1j*math.pi*f)/math.pi*model.generator(a)
        rotation_error=max(rotation_error,float(np.linalg.norm(direct-formula.reshape(direct.shape))))
    output=model.rotate(1.,a);back=model.rotate(1.,output)
    weights=[float(np.linalg.norm(output[:,:,r])**2) for r in (0,1)]
    fidelity=0.
    for r in (0,1):
        branch=np.zeros_like(output);branch[:,:,r]=output[:,:,r]
        returned=model.rotate(1.,branch)
        fidelity+=abs(np.vdot(a,returned))**2
    full_generator=np.kron(np.eye(len(model.totals)),k)
    full_energy=np.repeat(model.lambdas,2*model.n)
    commutator=full_generator*(full_energy[None,:]-full_energy[:,None])
    return {
        "matrix_exponential_error":rotation_error,
        "involution_error":float(np.linalg.norm(gate@gate-np.eye(len(gate)))),
        "hermitian_error":float(np.linalg.norm(gate-gate.conj().T)),
        "generator_maximum":float(np.linalg.eigvalsh(k)[-1]),
        "same_bare_energy_commutator":float(np.max(abs(commutator))),
        "same_bare_energy_reason":"각 완전 섬유에서 H_I=lambda_T I; 서로 다른 섬유 사이 결합 없음",
        "reverse_return_error":float(np.linalg.norm(back-a)),
        "reverse_pointer_one":float(np.linalg.norm(back[:,:,1])**2),
        "archive_fidelity":float(fidelity),"archive_fidelity_formula":sum(w*w for w in weights),
    }


def overlap_control(model):
    packet=np.sqrt(2/(model.m+1))*np.sin(math.pi*np.arange(1,model.m+1)/(model.m+1))
    probabilities=abs(model.u)**2
    correlations=np.array([1. if d==0 else
        (float(packet[:-d]@packet[d:]) if d<model.m else 0.) for d in range(model.n)])
    predicted=1-float(np.sum(probabilities[:,None]*probabilities[None,:]*
                           correlations[abs(np.arange(model.n)[:,None]-np.arange(model.n)[None,:])]))
    stats=model.stats(model.u,.5)
    return {"autocorrelation_error":abs(stats["q_flux"]-predicted),
            "nearest_shift_error":abs(correlations[1]-math.cos(math.pi/(model.m+1))),
            "actual_action_range":float(np.ptp(model.action)),
            "level_spacing_error":float(np.max(abs(np.diff(model.levels)-model.delta)))}


def standing_wave_control(model):
    fiber=len(model.totals)//2
    a=model.prepare(model.u)
    a[:fiber]=0.;a[fiber+1:]=0.;a/=np.linalg.norm(a)
    gap=.5;mass=1.
    p=math.sqrt(2*mass*float(model.kinetic(gap)[fiber]));v=p/mass
    def wave(x):
        return model.rotate(x,math.sqrt(2/v)*math.cos(p*x)*a)
    rows=[]
    for x in (0.,math.pi/(2*p),math.pi/p):
        value,first,_=clock.differences(wave,x,2e-4)
        slope=float(clock.autonomous.profile(x)[1])
        pi=-1j*first+slope*model.generator(value)
        rows.append({"x":x,"density":float(np.linalg.norm(value)**2),
                     "current":float(np.vdot(value,pi).real/mass)})
    return rows


def run():
    model=RecordClock()
    centered=(model.levels-float(abs(model.u)**2@model.levels))*model.u
    exchange=(model.u+centered/np.linalg.norm(centered))/math.sqrt(2)
    rows=[]
    for n in (4,8):
        for m in (8,16,32,64):
            for kind,target in (("length","constraint"),("squared","conditional")):
                item=RecordClock(n,m,kind,target)
                fixed=item.stats(item.u,float(item.offset[0])/2)
                rows.append({"size":n,"packet_size":m,"kind":kind,"target":target,
                             **fixed,**overlap_control(item)})
    threshold=[]
    for m in (8,16,32):
        item=RecordClock(4,m)
        for scaled_gap in (1e-2,1e-6,1e-10,1e-16):
            threshold.append({"packet_size":m,"scaled_gap":scaled_gap,
                              **item.stats(item.u,scaled_gap*item.delta)})
    phase=[]
    for beta in (0.,1.,5.):
        for target in ("conditional","constraint"):
            item=RecordClock(8,16,target=target,beta=beta)
            prepared=source(8,"length","constraint",beta)["j"]
            phase.append({"beta":beta,"target":target,**item.stats(prepared,.2)})
    paths=[Path(__file__),HERE/"regge_record_constraint.py",HERE/"regge_conservative_record.py",
           HERE/"regge_conditional_dynamics.py",HERE/"relational_split_clock.py",HERE/"autonomous_split_clock.py"]
    return {
        "model":"실제 레게 표적의 자율 보존 기록과 문턱 간격",
        "dependencies":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "local":[local_audit(model,exchange,gap,x,mass) for gap,mass in ((.2,.7),(3.,2.))
                 for x in (-.2,.2,.5,.8,1.2)],
        "resources":rows,"threshold":threshold,"phase_controls":phase,
        "matrix":matrix_control(model),"exchange":model.stats(exchange,.2),
        "standing_wave":standing_wave_control(model),
        "scope":{
            "fixed_energy_constraint":True,"same_bare_energy_in_and_out":True,
            "positive_battery_and_total_exchange":True,"actual_regge_target_used":True,
            "clock_profile_and_direction_supplied":True,"degenerate_record_memory_supplied":True,
            "product_preparation_in_flux_amplitudes":True,
            "normalizable_laboratory_wave_packet_preparation_derived":False,
            "original_regge_constraint_exactly_preserved":False,
            "zero_dimensional_time_or_direction_derived":False,
            "recharge_and_permanent_record_cycle_closed":False,
            "common_metric_or_forces_or_cosmology_derived":False,
        },
    }


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=True,indent=2,allow_nan=False))
