"""기존 자율 분할 작용의 고정 에너지 해와 관계적 시계를 대조한다.

외부 시간 진화를 호출하지 않는다. 위치·질량·분할 생성자·프로파일·에너지
껍질과 방향은 공급 조건이다. 고정 에너지 해는 위치 L2 상태가 아니며,
플럭스 정규화·위치 조건화·공변 에너지 시계의 판독을 구별한다.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

import autonomous_split_clock as autonomous

HERE=Path(__file__).resolve().parent


def shell(levels, energy, mass=1.):
    h=np.asarray(levels,dtype=float)
    if h.ndim!=1 or len(h)==0 or not np.all(np.isfinite(h)) or np.any(h<0):
        raise ValueError("음이 아닌 유한 내부 에너지 목록이 필요하다")
    if not np.isfinite(energy) or energy<=float(max(h)):
        raise ValueError("전체 에너지는 입력 띠의 최대 에너지보다 커야 한다")
    if not np.isfinite(mass) or mass<=0:
        raise ValueError("시계 질량은 양의 유한수여야 한다")
    root=np.sqrt(1-h/energy)
    momentum=np.sqrt(2*mass*(energy-h))
    effective=2*h/(1+root)
    correction=h*h/(energy*(1+root)**2)
    return {"momentum":momentum,"velocity":momentum/mass,"effective":effective,
            "correction":correction,"reference_momentum":math.sqrt(2*mass*energy),
            "reference_velocity":math.sqrt(2*energy/mass)}


def multiply(diagonal, vector):
    return diagonal*vector if vector.ndim==1 else diagonal[:,None]*vector


class ClockModel:
    def __init__(self, levels, generator, band=None):
        self.levels=np.asarray(levels,dtype=float)
        self.generator=np.asarray(generator,dtype=complex)
        n=len(self.levels)
        if self.levels.ndim!=1 or self.generator.shape!=(n,n):
            raise ValueError("내부 스펙트럼과 생성자 크기가 맞아야 한다")
        if not np.all(np.isfinite(self.levels)) or np.any(self.levels<0):
            raise ValueError("전체 내부 해밀토니안은 음이 아니어야 한다")
        if not np.all(np.isfinite(self.generator)) or not np.allclose(self.generator,self.generator.conj().T,atol=1e-12,rtol=0):
            raise ValueError("유한 에르미트 생성자가 필요하다")
        self.band=np.arange(n) if band is None else np.asarray(band,dtype=int)
        if len(set(self.band))!=len(self.band) or np.any(self.band<0) or np.any(self.band>=n):
            raise ValueError("입력 에너지 띠의 인덱스가 잘못되었다")
        self.gvalues,self.gvectors=np.linalg.eigh(self.generator)

    def rotate(self,x,vector,inverse=False):
        fraction=float(autonomous.profile(float(x))[0])
        sign=1 if inverse else -1
        return self.gvectors@multiply(np.exp(sign*1j*fraction*self.gvalues),self.gvectors.conj().T@vector)

    def embed(self,values):
        out=np.zeros((len(self.levels),)+values.shape[1:],dtype=complex)
        out[self.band]=values
        return out

    def state(self,x,amplitudes,energy,mass=1.,direction=1):
        a=np.asarray(amplitudes,dtype=complex)
        if not np.isfinite(x):
            raise ValueError("시계 좌표는 유한해야 한다")
        if a.ndim not in (1,2) or a.shape[0]!=len(self.band) or not np.all(np.isfinite(a)):
            raise ValueError("입력 띠와 크기가 맞는 유한 진폭이 필요하다")
        if direction not in (-1,1):
            raise ValueError("방향은 +1 또는 -1이어야 한다")
        data=shell(self.levels[self.band],energy,mass)
        free=multiply(np.exp(1j*direction*data["momentum"]*x)/np.sqrt(data["velocity"]),a)
        return self.rotate(x,self.embed(free))


class FockClockModel(ClockModel):
    def __init__(self,levels,generator,band):
        self.levels=np.asarray(levels,dtype=float)
        self.generator=generator.tocsr()
        self.band=np.asarray(band,dtype=int)

    def rotate(self,x,vector,inverse=False):
        fraction=float(autonomous.profile(float(x))[0])
        coefficient=(1 if inverse else -1)*1j*fraction
        if coefficient==0:
            return np.array(vector,copy=True)
        return expm_multiply(coefficient*self.generator,vector,
                             traceA=coefficient*self.generator.diagonal().sum())


def two_level():
    return ClockModel([.5,1.5],[[1.4,.6],[.6,.8]])


def fock_model(cutoff):
    """실제 k=2 이차 분할 생성자를 유한 Fock 공간에 직접 올린다."""
    if not isinstance(cutoff,int) or cutoff<3:
        raise ValueError("Fock 절단은 3 이상의 정수여야 한다")
    lowering=sparse.diags(np.sqrt(np.arange(1,cutoff)),1,shape=(cutoff,cutoff),format="csr")
    position=(lowering+lowering.T)/math.sqrt(2)
    momentum=(lowering-lowering.T)/(1j*math.sqrt(2))
    identity=sparse.eye(cutoff,format="csr")
    operators=[sparse.kron(position,identity,format="csr"),sparse.kron(momentum,identity,format="csr"),
               sparse.kron(identity,position,format="csr"),sparse.kron(identity,momentum,format="csr")]
    certificate=autonomous.generator.witness(2)
    metric=certificate["generator"]
    generator=sum(.5*metric[i,j]*(operators[i]@operators[j]) for i in range(4) for j in range(4))
    levels=np.array([i+j+1 for i in range(cutoff) for j in range(cutoff)],dtype=float)
    model=FockClockModel(levels,generator,[0,cutoff,cutoff+1])
    return model,operators,certificate


def differences(function,x,step):
    fm2,fm1,f0,fp1,fp2=[function(x+j*step) for j in (-2,-1,0,1,2)]
    first=(fm2-8*fm1+8*fp1-fp2)/(12*step)
    second=(-fm2+16*fm1-30*f0+16*fp1-fp2)/(12*step*step)
    return f0,first,second


def local_audit(model,x,amplitudes,energy,mass=1.,step=8e-4):
    """독립 위치 차분으로 완전제곱·전달 에너지와 current를 검사한다."""
    function=lambda z:model.state(z,amplitudes,energy,mass)
    value,first,second=differences(function,x,step)
    _,fine_first,fine_second=differences(function,x,step/2)
    first=(16*fine_first-first)/15
    second=(16*fine_second-second)/15
    _,slope=autonomous.profile(x)
    curvature=60*x*(1-x)*(1-2*x) if 0<x<1 else 0.
    g=model.generator
    connection=float(slope)*g
    pi_value=-1j*first+connection@value
    kinetic=(-second-2j*connection@first-1j*curvature*g@value+connection@connection@value)/(2*mass)
    free=model.rotate(x,value,inverse=True)
    potential=model.rotate(x,multiply(model.levels,free))
    residual=kinetic+potential-energy*value
    norm2=float(np.vdot(value,value).real)
    current=float(np.vdot(value,pi_value).real/mass)
    kinetic_form=float(np.vdot(pi_value,pi_value).real/(2*mass*norm2))
    potential_mean=float(np.vdot(value,potential).real/norm2)
    data=shell(model.levels[model.band],energy,mass)
    reduced=multiply(np.sqrt(data["velocity"]),free[model.band])*np.exp(-1j*data["reference_momentum"]*x)
    tau=x/data["reference_velocity"]
    expected=multiply(np.exp(-1j*data["effective"]*tau),np.asarray(amplitudes))
    missing=(connection@connection@value)/(2*mass)
    return {"x":float(x),"energy":energy,"constraint_relative_residual":float(np.linalg.norm(residual)/math.sqrt(norm2)),
            "flux":current,"flux_error":abs(current-float(np.vdot(amplitudes,amplitudes).real)),
            "total_energy_error":abs(kinetic_form+potential_mean-energy),
            "covariant_kinetic_energy":kinetic_form,"dressed_internal_energy":potential_mean,
            "conditional_reduction_error":float(np.linalg.norm(reduced-expected)),
            "omitted_square_residual":float(np.linalg.norm(residual-missing)/math.sqrt(norm2))}


def finite_energy_audit(energy,levels=(1.,2.,3.),times=(.25,1.,3.)):
    data=shell(levels,energy)
    rows=[]
    for time in times:
        difference=float(max(abs(np.exp(-1j*time*data["effective"])-np.exp(-1j*time*np.array(levels)))))
        rows.append({"time":time,"operator_error":difference,
                     "bound":min(2.,abs(time)*float(max(data["correction"])))})
    hmax=max(levels)
    return {"energy":energy,"maximum_correction":float(max(data["correction"])),
            "scaled_coefficient":float(energy*max(data["correction"])/(hmax*hmax)),
            "coefficient_error":abs(float(energy*max(data["correction"])/(hmax*hmax))-.25),
            "propagator":rows}


def fock_audit(cutoff):
    model,operators,certificate=fock_model(cutoff)
    vacuum=model.state(1.,[1.,0.,0.],16.)
    vacuum/=np.linalg.norm(vacuum)
    images=[op@vacuum for op in operators]
    means=np.array([np.vdot(vacuum,w).real for w in images])
    covariance=np.array([[np.vdot(a,b).real for b in images] for a in images])-np.outer(means,means)
    expected=.5*certificate["target"]@certificate["target"].T
    boundary=sum(abs(vacuum[i*cutoff+j])**2 for i in range(cutoff) for j in range(cutoff) if i==cutoff-1 or j==cutoff-1)
    amplitudes=np.array([1.,1j,.5])/1.5
    return {"cutoff":cutoff,"dimension":len(model.levels),
            "gaussian_covariance_error":float(np.max(np.abs(covariance-expected))),
            "boundary_occupation":float(boundary),
            "generator_hermiticity_error":float(sparse.linalg.norm(model.generator-model.generator.conj().T)),
            "checks":[local_audit(model,x,amplitudes,16.) for x in ((.2,.5,.8) if cutoff<=14 else (.5,))]}


def direction_control():
    model=two_level()
    energy,mass=4.,1.
    a=np.array([1.,0.])
    data=shell(model.levels,energy,mass)
    p=float(data["momentum"][0])
    nodes=[]
    for x in (0.,math.pi/(2*p),math.pi/p):
        plus=model.state(x,a,energy,mass,1)
        minus=model.state(x,a,energy,mass,-1)
        standing=(plus+minus)/math.sqrt(2)
        pi=(p*plus-p*minus)/math.sqrt(2)
        nodes.append({"x":x,"density":float(np.vdot(standing,standing).real),
                      "flux":float(np.vdot(standing,pi).real/mass)})
    return {"energy":energy,"samples":nodes,
            "interpretation":"방향 선택 전 정상파는 위치밀도를 갖지만 영 플럭스와 판독 노드를 갖는다"}


def readout_control(energy):
    levels=np.array([1.,2.,3.])
    amplitudes=np.array([1.,1j,.5])/1.5
    data=shell(levels,energy)
    flux_probability=abs(amplitudes)**2
    position_probability=flux_probability/data["velocity"]
    position_probability/=sum(position_probability)
    time=1.
    energy_readout=np.exp(-1j*energy*time)*np.exp(1j*(energy-levels)*time)*amplitudes
    ordinary=np.exp(-1j*levels*time)*amplitudes
    position_reduced=np.exp(-1j*data["effective"]*time)*amplitudes
    return {"energy":energy,"flux_probability":flux_probability.tolist(),
            "position_probability":position_probability.tolist(),
            "probability_total_variation":float(sum(abs(position_probability-flux_probability))/2),
            "energy_clock_schrodinger_error":float(np.linalg.norm(energy_readout-ordinary)),
            "position_clock_state_difference":float(np.linalg.norm(position_reduced-ordinary)),
            "position_clock_trace_distance":math.sqrt(max(0.,1-abs(np.vdot(position_reduced,ordinary))**2)),
            "scope":"에너지 공변 시계와 위치 시계는 다른 판독이다. dressed 에너지 시계는 일반적으로 공동 관측량이다."}


def run():
    bell=np.eye(2,dtype=complex)/math.sqrt(2)
    model=two_level()
    paths=[Path(__file__),HERE/"autonomous_split_clock.py",HERE/"split_quadratic_generator.py",
           HERE/"split_quantum_source.py"]
    return {"model":"기존 양의 분할 작용의 무시간 제약과 두 시계 판독",
            "dependencies":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "source":"https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.587083/full",
            "two_level":[local_audit(model,x,bell,8.) for x in (-.3,.2,.5,.8,1.3)],
            "finite_energy":[finite_energy_audit(e) for e in (4.,16.,64.,256.,1024.,16384.)],
            "fock":[fock_audit(n) for n in (6,10,14,20,28,40,56,64)],
            "directions":direction_control(),"readouts":[readout_control(e) for e in (3.1,4.,16.,64.)],
            "scope":{"external_time_propagation_used":False,"constraint_supplied":True,
                     "clock_coordinate_and_energy_sector_supplied":True,"incoming_direction_supplied":True,
                     "physical_clock_origin_derived":False,"persistent_record_order_derived":False,
                     "finite_normalizable_preparation_derived":False,"readout_independent_correction":False,
                     "common_metric_or_einstein_limit_derived":False,"dark_sector_or_hubble_result":False}}


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=True,indent=2,allow_nan=False))
